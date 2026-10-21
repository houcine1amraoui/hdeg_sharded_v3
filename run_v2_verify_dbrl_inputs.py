"""Bounded fresh-initialization DBRL compatibility check; no optimizer or checkpoint output."""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch_geometric
import yaml

from src.data_preparation import input_state_windows as windows
from src.models.hdeg import dbrl as dbrl_module
from src.utils.get_folders_utils import get_processed_folder

require = windows.require


def rejected(call, message):
    try:
        call()
    except (ValueError, TypeError, RuntimeError) as exc:
        require(message in str(exc), f"Unexpected rejection: {exc}")
        return str(exc)
    raise ValueError("An incompatible input/checkpoint was accepted")


def verify(config, processed, args, report):
    settings = config.get("input_state_dbrl_verification", {})
    count = args.windows if args.windows is not None else settings.get("windows", 32)
    batch_size = args.batch_size if args.batch_size is not None else settings.get("batch_size", 8)
    device_name = args.device or settings.get("device", "cpu")
    seed = config.get("seed", 42)
    require(type(count) is int and 2 <= count <= 128, "windows must be 2..128")
    require(type(batch_size) is int and 1 <= batch_size <= 16, "batch_size must be 1..16")
    require(type(seed) is int and 0 <= seed < 2**32, "seed must be an integer in [0, 2**32)")
    require(device_name in ("cpu", "cuda"), "device must be cpu or cuda")
    require(device_name != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    dataset = windows.InputStateWindowDataset(args.input_dir, config["preprocessing"]["window_size"])
    require(count <= len(dataset), "Not enough training windows for requested budget")
    require(config["preprocessing"] == dataset.preparation["configuration"]["preprocessing"],
            "Preprocessing configuration differs from preparation")
    proof_path = Path(args.window_report)
    proof = json.loads(proof_path.read_text())
    require(proof["status"] == "PASS" and proof["protocol"] == "v2_input_state_window_verification_v1"
            and proof["row_integrity_scan_complete"] is True, "Expected successful window verification")
    require(proof["manifest_sha256"] == dataset.manifest_sha256, "Window report manifest differs")
    require(proof["window_size"] == dataset.window_size and proof["target_shift_rows"] == 1,
            "Window report window/shift differs")
    require(proof["configuration"]["preprocessing"] == config["preprocessing"], "Window report preprocessing differs")
    dependencies = {"config": Path("configs/config.yaml"), "window_report": proof_path,
                    "runner": Path(__file__), "dbrl": Path(dbrl_module.__file__),
                    "window_loader": Path(windows.__file__),
                    "devices": processed / "devices.json", "timestamps": processed / "timestamps.npz",
                    "registry": Path(config["preprocessing"].get("input_state_registry", "configs/CU_input_state_registry.json")),
                    "selected_features": Path("configs/CU_selected_features.txt")}
    # Include the unchanged graph implementation in the execution provenance.
    import inspect
    dependencies["graph_builder"] = Path(inspect.getfile(dbrl_module.GraphBuilder))
    dependencies["graph_reasoner"] = Path(inspect.getfile(dbrl_module.GraphReasoner))
    hashes = {k: windows.sha256(v) for k, v in dependencies.items()}
    require(hashes["window_loader"] == proof["source_hashes"]["window_loader"], "Verified loader changed")
    for key in ("devices", "timestamps", "registry", "selected_features"):
        require(hashes[key] == dataset.metadata["source_hashes"][key] == proof["source_hashes"][key],
                f"Source dependency differs: {key}")
    # Fixed endpoints and uniform seeded training indices, sampled without replacement.
    middle = np.random.default_rng(seed).choice(len(dataset)-2, count-2, replace=False) + 1
    indices = np.sort(np.r_[0, middle, len(dataset)-1]).astype(np.int64)
    p = config["hdeg"]["dbrl"]
    kwargs = {k: p.get(k, default) for k, default in {
        "hidden_dim": 64, "embedding_dim": 64, "gru_layers": 1, "dropout": 0.0,
        "graph_top_k": 15, "graph_self_loops": False, "graph_symmetric": False,
        "graph_heads": 1, "graph_dropout": 0.0}.items()}
    require(p.get("input_dim", 6) == 6, "Explicit hdeg.dbrl.input_dim conflicts with this six-feature check")
    model = dbrl_module.DBRL(**kwargs, input_dim=6).to(device).eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    report.update(source_hashes=hashes, manifest_sha256=dataset.manifest_sha256,
                  selected_window_indices=indices.tolist(), device=str(device), seed=seed,
                  model_parameters={**kwargs, "input_dim": 6}, window_size=dataset.window_size,
                  input_feature_order=["value"]+list(windows.STATUS_CODES),
                  budget={"windows": count, "batch_size": batch_size},
                  framework_versions={"torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
                                      "numpy": np.__version__})
    gradient_norms = {name: 0.0 for name, _ in model.named_parameters()}
    feature_gradient = np.zeros(6)
    status_counts = np.zeros(5, dtype=np.int64)
    batches = []
    for left in range(0, count, batch_size):
        batch = dataset.get_batch(indices[left:left+batch_size])
        x = torch.from_numpy(batch["X"]).to(device).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        z = model(x)
        require(tuple(z.shape) == (len(x), 23, kwargs["embedding_dim"]), "Output shape differs")
        require(torch.isfinite(z).all().item(), "Nonfinite output")
        # A deterministic nonuniform linear probe avoids a zero-loss-at-origin test.
        probe = torch.linspace(0.5, 1.5, z.numel(), device=device).reshape_as(z)
        loss = (z * probe).mean()
        loss.backward()
        require(x.grad is not None and torch.isfinite(x.grad).all().item(), "Input gradient missing/nonfinite")
        require(x.grad.abs().sum().item() > 0, "Input disconnected from output")
        feature_gradient += x.grad.detach().abs().sum(dim=(0, 1, 2)).cpu().numpy()
        for name, parameter in model.named_parameters():
            require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(),
                    f"Missing/nonfinite parameter gradient: {name}")
            gradient_norms[name] += float(parameter.grad.detach().norm())
        status_counts += np.bincount(batch["X_status"].ravel(), minlength=5)
        batches.append({"size": len(x), "output_shape": list(z.shape), "probe": float(loss.detach()),
                        "output_min": float(z.detach().min()), "output_max": float(z.detach().max())})
    for group in ("temporal_encoder.", "embedding_projection.", "graph_reasoner."):
        require(sum(v for k, v in gradient_norms.items() if k.startswith(group)) > 0,
                f"No nonzero gradient in {group}")
    require(all(torch.equal(before[k], v) for k, v in model.state_dict().items()), "Parameters/buffers changed")
    # Same-mode state reload, explicit scalar/six-feature mismatch (also strict=False),
    # and recursive loading through a parent module.
    clone = dbrl_module.DBRL(**kwargs, input_dim=6).to(device).eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    scalar = dbrl_module.DBRL(**kwargs).to(device).eval()
    mismatch = []
    for strict in (True, False):
        mismatch.append(rejected(lambda: clone.load_state_dict(scalar.state_dict(), strict=strict),
                                 "Incompatible DBRL checkpoint"))
        rejected(lambda: scalar.load_state_dict(model.state_dict(), strict=strict), "Incompatible DBRL checkpoint")
    parent = torch.nn.ModuleDict({"dbrl": clone})
    legacy_parent = torch.nn.ModuleDict({"dbrl": scalar})
    rejected(lambda: parent.load_state_dict(legacy_parent.state_dict(), strict=False), "Incompatible DBRL checkpoint")
    sample = torch.from_numpy(dataset.get_batch(indices[:1])["X"]).to(device)
    with torch.no_grad():
        reference = model(sample)
        require(torch.equal(reference, clone(sample)), "Same-mode checkpoint roundtrip differs")
        scalar_z = scalar(sample[..., 0])
        require(tuple(scalar_z.shape) == (1, 23, kwargs["embedding_dim"]) and torch.isfinite(scalar_z).all().item(),
                "Scalar mode forward failed")
    rejected(lambda: model(sample[..., 0]), "expects")
    rejected(lambda: model(sample[..., :5]), "expects")
    rejected(lambda: scalar(sample), "expects")
    rejected(lambda: model(sample.long()), "floating-point")
    bad = sample.clone(); bad[0, 0, 0, 0] = float("nan")
    rejected(lambda: model(bad), "NaN")
    dataset.verify_unchanged()
    require(all(windows.sha256(v) == hashes[k] for k, v in dependencies.items()), "Dependency changed during check")
    report.update(status="PASS", batches=batches, parameter_gradient_norm_sums=gradient_norms,
                  input_feature_abs_gradient_sums=feature_gradient.tolist(),
                  sampled_X_status_counts=dict(zip(windows.STATUS_CODES, map(int, status_counts))),
                  parameters_unchanged=True, scalar_forward_passed=True, same_mode_roundtrip_passed=True,
                  incompatible_checkpoint_rejections=mismatch, recursive_checkpoint_rejection_passed=True,
                  malformed_input_rejections_passed=True, successful_forward_calls=len(batches)+3, rejected_forward_calls=5,
                  backward_calls=len(batches), checkpoint_loaded="synthetic in-memory compatibility probes only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root_dir")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--window_report", required=True)
    parser.add_argument("--windows", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    if args.project_root_dir:
        config["project_root_dir"] = args.project_root_dir
    processed = Path(get_processed_folder(config))
    report = {"protocol": "v2_dbrl_six_feature_compatibility_v1", "status": "BLOCKED",
              "configuration": config, "optimizer_steps": 0, "scientific_claim_decision": "NOT_EVALUATED",
              "limitations": ["Fresh initialization only; no trained checkpoint or downstream HDEG integration tested.",
                              "A nonuniform probe tests gradient connectivity, not a training objective or convergence.",
                              "Hard top-k neighbor selection is discrete; this does not establish differentiability of graph selection.",
                              "Absent status categories do not demonstrate learned handling of those categories.",
                              "Inputs mix values and availability status; no physical-behavior-only claim.",
                              "Exact numerical reproducibility across devices/framework versions is not asserted."]}
    try:
        verify(config, processed, args, report)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, IndexError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = processed / "v2_validation" / "dbrl_input_compatibility" / run_id
    output.mkdir(parents=True, exist_ok=False)
    path = output / "dbrl_input_compatibility_report.json"
    with path.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f"DBRL input compatibility: {report['status']}\nReport: {path}")
    if "error" in report:
        print(report["error"])
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
