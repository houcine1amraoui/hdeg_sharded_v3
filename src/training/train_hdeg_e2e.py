from __future__ import annotations

"""
End-to-end HDEG training over paired CU window shards.

This runner is deliberately separate from the persisted DBRL/BSE/BIL/EBRL
artifact runners. Those runners are inference/artifact-generation paths and
persist detached scientific representations. This file instead executes the
live trainable graph:

    X_t -> DBRL -> BSE -> BIL -> EBRL -> R_t -> HBF -> R_hat_{t+1}
    Y_t -> DBRL -> BSE -> BIL -> EBRL -> R_{t+1}
                                      \
                                       -> MO -> L_HDEG -> backward()

Only one paired window shard is loaded at a time, and only one mini-batch from
that shard is placed on the execution device at a time.

The implementation deliberately does not detach the target hierarchy. V1.0
specifies joint optimization of representation learning and forecasting but
specifies no stop-gradient target operation; therefore the live observed
future representation remains part of the autograd graph.
"""

from dataclasses import dataclass
import argparse
import gc
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import time

import numpy as np
import torch
from torch import Tensor, nn
import yaml

from src.models.hdeg.dbrl import DBRL
from src.models.hdeg.bse import BehavioralStateEstimator
from src.models.hdeg.bil import BehavioralInteractionLearner
from src.models.hdeg.ebrl import EcosystemBehavioralRepresentationLearner
from src.models.hdeg.hbf import HierarchicalBehavioralForecaster
from src.models.hdeg.mo import ModelOptimization
from src.common.graph.semantics import load_behavioral_state_config
from src.utils.device import get_device
from src.utils.get_folders_utils import get_processed_folder
from src.utils.experiment_protection import (
    fresh_run_directory, require_new_file, write_json_new, sha256_file,
)
from src.utils.seed import set_seed

# Reuse the authoritative paired-window loading/manifest utilities from the
# existing DBRL sharded implementation. Do not duplicate their contracts.
from run_dbrl_sharded import (
    load_manifest,
    load_window_shard,
    resolve_shard_path,
    verify_window_target_alignment,
)


SPLITS = ("train", "val", "actor2_test", "actor1_test")
HIERARCHY_LEVELS = ("Z", "S", "S_tilde", "g")


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    L_Z: float
    L_S: float
    L_S_tilde: float
    L_G: float
    samples: int
    shards: int
    batches: int


class HDEGEndToEndModel(nn.Module):
    """Live trainable DBRL -> BSE -> BIL -> EBRL -> HBF -> MO graph."""

    def __init__(
        self,
        *,
        dbrl: DBRL,
        bse: BehavioralStateEstimator,
        bil: BehavioralInteractionLearner,
        ebrl: EcosystemBehavioralRepresentationLearner,
        hbf: HierarchicalBehavioralForecaster,
        mo: ModelOptimization,
        compatibility_mask: Tensor,
    ) -> None:
        super().__init__()
        self.dbrl = dbrl
        self.bse = bse
        self.bil = bil
        self.ebrl = ebrl
        self.hbf = hbf
        self.mo = mo
        self.register_buffer("compatibility_mask", compatibility_mask, persistent=True)

    def encode_window(self, x: Tensor) -> Dict[str, Tensor]:
        """Run one raw observation mini-batch through DBRL/BSE/BIL/EBRL."""
        Z = self.dbrl(x)
        S = self.bse(Z, self.compatibility_mask)
        S_tilde = self.bil(S)
        g = self.ebrl(S_tilde)
        return {"Z": Z, "S": S, "S_tilde": S_tilde, "g": g}

    def forecast(self, hierarchy_t: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.hbf(
            hierarchy_t["Z"],
            hierarchy_t["S"],
            hierarchy_t["S_tilde"],
            hierarchy_t["g"],
        )

    def optimize_pair(
        self,
        x_t: Tensor,
        x_t1: Tensor,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], object]:
        """
        Execute both sides of one paired window mini-batch.

        No branch is wrapped in no_grad and no representation is detached.
        """
        observed_t = self.encode_window(x_t)
        predicted_t1 = self.forecast(observed_t)
        observed_t1 = self.encode_window(x_t1)
        objectives = self.mo(observed_t1, predicted_t1)
        return observed_t, observed_t1, objectives


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_live_model(
    *,
    config: dict,
    behavioral_config,
    compatibility_mask: Tensor,
    device: torch.device,
) -> HDEGEndToEndModel:
    num_devices = int(behavioral_config.num_devices)
    num_states = int(behavioral_config.num_states)

    dbrl_cfg = config["hdeg"]["dbrl"]
    bse_cfg = config["hdeg"]["bse"]
    hbf_cfg = config["hdeg"]["hbf"]

    embedding_dim = int(dbrl_cfg["embedding_dim"])

    if int(dbrl_cfg["graph_top_k"]) > (
        num_devices if bool(dbrl_cfg["graph_self_loops"]) else num_devices - 1
    ):
        raise ValueError("DBRL graph_top_k exceeds the allowed device-neighbor count.")

    dbrl = DBRL(
        hidden_dim=int(dbrl_cfg["hidden_dim"]),
        embedding_dim=embedding_dim,
        gru_layers=int(dbrl_cfg["gru_layers"]),
        dropout=0.0,
        graph_top_k=int(dbrl_cfg["graph_top_k"]),
        graph_self_loops=bool(dbrl_cfg["graph_self_loops"]),
        graph_symmetric=bool(dbrl_cfg["graph_symmetric"]),
        graph_heads=int(dbrl_cfg["graph_heads"]),
        graph_dropout=0.0,
        # Constructor argument is not a DBRL architectural parameter; it is
        # retained by the frozen driver for explicit validation.
    )

    bse = BehavioralStateEstimator(
        num_states=num_states,
        embedding_dim=embedding_dim,
        num_heads=int(bse_cfg["num_heads"]),
    )

    # BIL's frozen CU topology is internal to the current implementation.
    if num_states != 9:
        raise ValueError(f"Frozen CU BIL expects K=9; received K={num_states}.")
    bil = BehavioralInteractionLearner(
        num_states=num_states,
        embedding_dim=embedding_dim,
    )

    ebrl = EcosystemBehavioralRepresentationLearner(
        num_states=num_states,
        embedding_dim=embedding_dim,
    )

    hbf = HierarchicalBehavioralForecaster(
        num_devices=num_devices,
        num_states=num_states,
        embedding_dim=embedding_dim,
        dynamics_hidden_dim=int(hbf_cfg["dynamics_hidden_dim"]),
    )

    mo_cfg = config.get("hdeg", {}).get("mo", {})
    mo = ModelOptimization(
        lambda_z=float(mo_cfg.get("lambda_z", 1.0)),
        lambda_s=float(mo_cfg.get("lambda_s", 1.0)),
        lambda_s_tilde=float(mo_cfg.get("lambda_s_tilde", 1.0)),
        lambda_g=float(mo_cfg.get("lambda_g", 1.0)),
    )

    model = HDEGEndToEndModel(
        dbrl=dbrl,
        bse=bse,
        bil=bil,
        ebrl=ebrl,
        hbf=hbf,
        mo=mo,
        compatibility_mask=compatibility_mask,
    ).to(device)

    return model


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: Path,
    *,
    model: HDEGEndToEndModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: EpochMetrics,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics.__dict__,
        "config": config,
    }
    # Exclusive creation: never truncate an existing checkpoint, including on resume.
    require_new_file(path)
    with path.open("xb") as stream:
        torch.save(payload, stream)


def load_checkpoint(
    path: Path,
    *,
    model: HDEGEndToEndModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint {path} must contain a dictionary payload.")
    for key in ("epoch", "model_state_dict", "optimizer_state_dict"):
        if key not in payload:
            raise KeyError(f"Checkpoint {path} is missing '{key}'.")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload["epoch"])


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------


def _set_train_mode(model: HDEGEndToEndModel) -> None:
    model.train()


def _set_eval_mode(model: HDEGEndToEndModel) -> None:
    model.eval()


def _move_batch(array: np.ndarray, start: int, end: int, device: torch.device) -> Tensor:
    batch = torch.from_numpy(array[start:end])
    if batch.dtype != torch.float32:
        raise TypeError(f"Window batch must be float32; received {batch.dtype}.")
    return batch.to(device, non_blocking=False)


def _validate_live_hierarchy(
    hierarchy: Dict[str, Tensor],
    *,
    batch_size: int,
    num_devices: int,
    num_states: int,
    embedding_dim: int,
    label: str,
) -> None:
    expected = {
        "Z": (batch_size, num_devices, embedding_dim),
        "S": (batch_size, num_states, embedding_dim),
        "S_tilde": (batch_size, num_states, embedding_dim),
        "g": (batch_size, embedding_dim),
    }
    for level in HIERARCHY_LEVELS:
        tensor = hierarchy[level]
        if tuple(tensor.shape) != expected[level]:
            raise RuntimeError(
                f"{label} {level} shape mismatch: expected {expected[level]}, "
                f"received {tuple(tensor.shape)}."
            )
        if tensor.dtype != torch.float32:
            raise RuntimeError(f"{label} {level} must be float32.")
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{label} {level} contains NaN/Inf.")


def _accumulate_metrics(
    totals: Dict[str, float],
    objectives,
    batch_size: int,
) -> None:
    totals["L_HDEG"] += float(objectives.L_HDEG.detach().cpu()) * batch_size
    totals["L_Z"] += float(objectives.L_Z.detach().cpu()) * batch_size
    totals["L_S"] += float(objectives.L_S.detach().cpu()) * batch_size
    totals["L_S_tilde"] += float(objectives.L_S_tilde.detach().cpu()) * batch_size
    totals["L_G"] += float(objectives.L_G.detach().cpu()) * batch_size


def run_epoch(
    model: HDEGEndToEndModel,
    optimizer: Optional[torch.optim.Optimizer],
    *,
    shard_paths: Iterable[Path],
    split: str,
    window_size: int,
    num_devices: int,
    num_states: int,
    embedding_dim: int,
    batch_size: int,
    device: torch.device,
    train: bool,
    max_batches_per_shard: Optional[int] = None,
) -> EpochMetrics:
    if train and optimizer is None:
        raise ValueError("optimizer is required for training mode.")

    if train:
        _set_train_mode(model)
    else:
        _set_eval_mode(model)

    totals = {"L_HDEG": 0.0, "L_Z": 0.0, "L_S": 0.0, "L_S_tilde": 0.0, "L_G": 0.0}
    total_samples = 0
    total_batches = 0
    total_shards = 0
    start_time = time.time()
    
    grad_context = torch.enable_grad() if train else torch.no_grad()

    with grad_context:
        for shard_path in shard_paths:
            artifact = load_window_shard(
                shard_path=shard_path,
                expected_split=split,
                expected_window_size=window_size,
                expected_num_devices=num_devices,
            )
            verify_window_target_alignment(artifact)

            X = artifact["X"]
            Y = artifact["Y"]
            shard_samples = int(X.shape[0])
            shard_index = int(artifact["shard_index"].item())

            print(
                f"  shard {shard_index:06d}: "
                f"[{int(artifact['start_index'].item())}, "
                f"{int(artifact['end_index'].item())}) "
                f"samples={shard_samples}"
            )

            shard_batches = 0
            for start in range(0, shard_samples, batch_size):
                if max_batches_per_shard is not None and shard_batches >= max_batches_per_shard:
                    break
                end = min(start + batch_size, shard_samples)
                current_batch_size = end - start

                x_t = _move_batch(X, start, end, device)
                y_t1 = _move_batch(Y, start, end, device)

                if train:
                    optimizer.zero_grad(set_to_none=True)

                _, observed_t1, objectives = model.optimize_pair(x_t, y_t1)

                # Verify that the target representation is a live autograd
                # tensor rather than a persisted/detached artifact.
                if train:
                    if not observed_t1["Z"].requires_grad:
                        raise RuntimeError("Target Z is detached from the live autograd graph.")
                    if not objectives.L_HDEG.requires_grad:
                        raise RuntimeError("MO loss is detached from the live autograd graph.")
                    objectives.L_HDEG.backward()
                    optimizer.step()

                _accumulate_metrics(totals, objectives, current_batch_size)
                total_samples += current_batch_size
                total_batches += 1
                shard_batches += 1

                if train and total_batches % 100 == 0:
                    elapsed = time.time() - start_time
                    eta = elapsed / total_batches * (
                        len(list(shard_paths)) * (
                            (shard_samples + batch_size - 1) // batch_size
                        ) - total_batches
                    )
                    print(
                        f"  Batch {total_batches} | "
                        f"Elapsed: {elapsed/60:.1f} min | "
                        f"ETA: {max(0, eta)/60:.1f} min",
                        flush=True,
                    )
                
                del x_t, y_t1, observed_t1, objectives

            total_shards += 1
            del X, Y, artifact
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if total_samples == 0:
        raise RuntimeError("No paired window samples were processed.")

    return EpochMetrics(
        loss=totals["L_HDEG"] / total_samples,
        L_Z=totals["L_Z"] / total_samples,
        L_S=totals["L_S"] / total_samples,
        L_S_tilde=totals["L_S_tilde"] / total_samples,
        L_G=totals["L_G"] / total_samples,
        samples=total_samples,
        shards=total_shards,
        batches=total_batches,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HDEG end-to-end over paired CU window shards."
    )
    parser.add_argument("--project_root_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--split", type=str, choices=("train",), default="train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--max_batches_per_shard", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=("cpu", "cuda"))
    parser.add_argument("--no_save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError("config.yaml must contain a mapping.")

    if args.project_root_dir:
        config["project_root_dir"] = args.project_root_dir

    set_seed(int(config["seed"]))

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device) if args.device else get_device()

    preprocessing = config["preprocessing"]
    window_size = int(preprocessing["window_size"])

    dbrl_cfg = config["hdeg"]["dbrl"]
    batch_size = int(args.batch_size if args.batch_size is not None else dbrl_cfg["batch_size"])
    epochs = int(args.epochs if args.epochs is not None else config.get("hdeg", {}).get("e2e_training", {}).get("epochs", 1))
    lr = float(args.lr if args.lr is not None else config.get("hdeg", {}).get("e2e_training", {}).get("lr", 1e-4))

    if batch_size <= 0 or epochs <= 0 or lr <= 0.0:
        raise ValueError("batch_size, epochs, and lr must be positive.")
    if args.max_shards is not None and args.max_shards <= 0:
        raise ValueError("max_shards must be positive when provided.")
    if args.max_batches_per_shard is not None and args.max_batches_per_shard <= 0:
        raise ValueError("max_batches_per_shard must be positive when provided.")

    processed_data_folder = Path(get_processed_folder(config))
    windows_dir = processed_data_folder / "windows"
    split_dir = windows_dir / args.split

    manifest = load_manifest(windows_dir=windows_dir, split=args.split)
    num_devices = int(manifest["num_devices"])
    manifest_window_size = int(manifest["window_size"])
    if manifest_window_size != window_size:
        raise ValueError(
            f"Config/manifest window-size mismatch: config={window_size}, manifest={manifest_window_size}."
        )

    shard_entries = list(manifest["shards"])
    if args.max_shards is not None:
        shard_entries = shard_entries[:args.max_shards]
    shard_paths = [resolve_shard_path(split_dir, entry) for entry in shard_entries]

    behavioral_config_path = (
        Path(config["project_root_dir"]) / "configs" / "hdeg" / "behavioral_states.yaml"
    )
    devices_path = processed_data_folder / "devices.json"
    behavioral_config = load_behavioral_state_config(
        config_path=behavioral_config_path,
        devices_path=devices_path,
    )

    if behavioral_config.num_devices != num_devices:
        raise ValueError(
            f"Behavioral-state device count {behavioral_config.num_devices} "
            f"does not match window manifest N={num_devices}."
        )

    num_states = int(behavioral_config.num_states)
    embedding_dim = int(dbrl_cfg["embedding_dim"])
    compatibility_mask = behavioral_config.torch_mask(
        dtype=torch.float32,
        device=device,
        clone=True,
    )

    model = build_live_model(
        config=config,
        behavioral_config=behavioral_config,
        compatibility_mask=compatibility_mask,
        device=device,
    )

    # MO has no trainable parameters; all optimizer parameters come from the
    # live representation-learning and forecasting modules.
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable HDEG parameters were found.")

    optimizer = torch.optim.Adam(trainable, lr=lr)

    # Validate output paths before loading shards or performing optimization.
    if args.checkpoint and not args.resume and not args.no_save:
        require_new_file(Path(args.checkpoint))
    checkpoint_dir = None
    if not args.no_save:
        checkpoint_dir = fresh_run_directory(
            processed_data_folder / "hdeg_checkpoints" / "runs",
            explicit=args.checkpoint_dir,
        )
        write_json_new(checkpoint_dir / "run_configuration.json", {
            "status": "RESERVED", "configuration": config,
            "epochs_requested": epochs, "batch_size": batch_size, "learning_rate": lr,
            "resume_source": str(Path(args.checkpoint).resolve()) if args.resume and args.checkpoint else None,
            "resume_source_sha256": sha256_file(args.checkpoint) if args.resume and args.checkpoint else None,
            "max_shards": args.max_shards, "max_batches_per_shard": args.max_batches_per_shard,
            "trainer_sha256": sha256_file(__file__),
        })
        print(f"Protected training output directory: {checkpoint_dir}", flush=True)

    start_epoch = 1
    if args.resume:
        if not args.checkpoint:
            raise ValueError("--resume requires --checkpoint.")
        start_epoch = load_checkpoint(
            Path(args.checkpoint),
            model=model,
            optimizer=optimizer,
            device=device,
        ) + 1

    if start_epoch > epochs:
        raise ValueError("Resume epoch is already at or beyond --epochs; no training or checkpoint write performed.")

    print("=" * 78)
    print("HDEG — END-TO-END TRAINING")
    print("=" * 78)
    print(f"Split                 : {args.split}")
    print(f"Window directory     : {split_dir}")
    print(f"Shards                : {len(shard_paths)}")
    print(f"Samples in manifest   : {manifest['num_samples']}")
    print(f"N / K / D             : {num_devices} / {num_states} / {embedding_dim}")
    print(f"Batch size             : {batch_size}")
    print(f"Epochs                 : {epochs}")
    print(f"Learning rate          : {lr}")
    print(f"Device                 : {device}")
    print(f"Trainable parameters   : {trainable_parameter_count(model):,}")
    print("Target representation  : LIVE AUTOGRAD (no detach)")
    print()


    history = []
    for epoch in range(start_epoch, epochs + 1):
        print("-" * 78)
        print(f"Epoch {epoch}/{epochs}")
        print("-" * 78)

        metrics = run_epoch(
            model,
            optimizer,
            shard_paths=shard_paths,
            split=args.split,
            window_size=window_size,
            num_devices=num_devices,
            num_states=num_states,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            device=device,
            train=True,
            max_batches_per_shard=args.max_batches_per_shard,
        )
        history.append(metrics.__dict__)

        print(
            f"Epoch {epoch}: "
            f"L_HDEG={metrics.loss:.8f} | "
            f"L_Z={metrics.L_Z:.8f} | "
            f"L_S={metrics.L_S:.8f} | "
            f"L_S_tilde={metrics.L_S_tilde:.8f} | "
            f"L_G={metrics.L_G:.8f} | "
            f"samples={metrics.samples} | batches={metrics.batches}"
        )

        if not args.no_save:
            epoch_checkpoint = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(
                epoch_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )
            print(f"Checkpoint saved: {epoch_checkpoint}")

    if args.checkpoint and not args.resume and not args.no_save:
        final_metrics = metrics
        save_checkpoint(
            Path(args.checkpoint),
            model=model,
            optimizer=optimizer,
            epoch=epochs,
            metrics=final_metrics,
            config=config,
        )
        print(f"Final checkpoint saved: {args.checkpoint}")

    print("=" * 78)
    if not args.no_save:
        write_json_new(checkpoint_dir / "completed.json", {
            "status": "COMPLETE", "start_epoch": start_epoch, "final_epoch": epochs,
            "checkpoints": [
                {"path": str(p), "sha256": sha256_file(p)}
                for p in sorted(checkpoint_dir.glob("epoch_*.pt"))
            ],
        })
    print("HDEG end-to-end training completed")
    print("=" * 78)


if __name__ == "__main__":
    main()
