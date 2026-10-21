"""Bounded window checks plus one full row-shard integrity pass; no model execution."""
from __future__ import annotations

import argparse
import json
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.data_preparation import input_state_windows as windows
from src.utils.get_folders_utils import get_processed_folder


def choose_indices(dataset, random_windows, seed, max_windows):
    # Every pair crossing each row-shard boundary, plus the neighboring pairs.
    mandatory = {0, len(dataset) - 1}
    for boundary in dataset.starts[1:]:
        mandatory.update(range(max(0, boundary-dataset.window_size-1),
                               min(len(dataset), boundary+1)))
    windows.require(len(mandatory) <= max_windows,
                    "Mandatory boundary checks exceed max_windows; increase the explicit budget")
    rng = np.random.default_rng(seed)
    # Budget must not silently truncate requested random probes.
    windows.require(len(mandatory) + min(random_windows, len(dataset)) <= max_windows,
                    "Requested verification probes exceed max_windows")
    random = rng.choice(len(dataset), size=min(random_windows, len(dataset)), replace=False)
    return sorted(mandatory | set(map(int, random))), sorted(mandatory)


class ReferenceRows:
    """Independent reference assembly: overlap all manifest intervals directly."""
    def __init__(self, dataset):
        self.directory = dataset.directory
        self.entries = dataset.shards
        self.cache = OrderedDict()

    def slice(self, start, stop):
        parts = {k: [] for k in ["values", "status", "timestamps"]}
        for e in self.entries:
            left, right = max(start, e["start"]), min(stop, e["stop"])
            if left >= right:
                continue
            name = e["file"]
            if name in self.cache:
                a = self.cache.pop(name)
            else:
                with np.load(self.directory / name, allow_pickle=False) as data:
                    a = {k: data[k] for k in parts}
            self.cache[name] = a
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
            for k in parts:
                parts[k].append(a[k][left-e["start"]:right-e["start"]])
        return {k: np.concatenate(v) for k, v in parts.items()}


def verify(config, processed, args, report):
    settings = config.get("input_state_window_verification", {})
    batch_size = args.batch_size if args.batch_size is not None else settings.get("batch_size", 32)
    random_count = args.random_windows if args.random_windows is not None else settings.get("random_windows", 256)
    maximum = args.max_windows if args.max_windows is not None else settings.get("max_windows", 2048)
    windows.require(type(batch_size) is int and 1 <= batch_size <= 256, "batch_size must be 1..256")
    windows.require(type(random_count) is int and 0 <= random_count <= 4096, "random_windows must be 0..4096")
    windows.require(type(maximum) is int and 1 <= maximum <= 8192, "max_windows must be 1..8192")
    window_size = config["preprocessing"]["window_size"]
    dataset = windows.InputStateWindowDataset(args.input_dir, window_size)
    report["manifest_sha256"] = dataset.manifest_sha256
    report["input_dir"] = str(dataset.directory)
    report["policy_version"] = dataset.manifest["policy_version"]
    report["num_training_rows"] = dataset.num_rows
    report["num_training_pairs"] = len(dataset)
    report["window_size"] = window_size
    report["target_shift_rows"] = 1
    windows.require(config["preprocessing"] == dataset.preparation["configuration"]["preprocessing"],
                    "Preprocessing settings differ from prepared inputs")
    registry_path = Path(config["preprocessing"].get("input_state_registry", "configs/CU_input_state_registry.json"))
    dependencies = {"config": Path("configs/config.yaml"), "devices": processed / "devices.json",
                    "timestamps": processed / "timestamps.npz", "registry": registry_path,
                    "selected_features": Path("configs/CU_selected_features.txt"),
                    "verification_runner": Path(__file__), "window_loader": Path(windows.__file__)}
    hashes = {k: windows.sha256(p) for k, p in dependencies.items()}
    report["source_hashes"] = hashes
    for key in ["devices", "timestamps", "registry", "selected_features"]:
        windows.require(hashes[key] == dataset.metadata["source_hashes"][key], f"Dependency differs: {key}")
    windows.require(json.loads(dependencies["devices"].read_text()) == dataset.devices, "Current device order differs")
    with np.load(dependencies["timestamps"], allow_pickle=False) as a:
        expected_times = a["train"].astype("datetime64[ns]")
    windows.require(len(expected_times) == dataset.num_rows, "Saved training timestamp count differs")
    indices, mandatory = choose_indices(dataset, random_count, int(config.get("seed", 42)), maximum)
    report["verification_budget"] = {"batch_size": batch_size, "random_windows_requested": random_count,
                                     "max_windows": maximum, "seed": int(config.get("seed", 42)),
                                     "mandatory_windows": len(mandatory), "unique_windows": len(indices)}
    report["checked_window_indices"] = indices
    counts = np.zeros((23, 5), dtype=np.int64)
    binary = np.zeros((23, 2), dtype=np.int64)
    minima, maxima = np.full(23, np.inf), np.full(23, -np.inf)
    for index, e in enumerate(dataset.shards):
        a = dataset.load_shard(index)
        windows.require(np.array_equal(a["timestamps"], expected_times[e["start"]:e["stop"]]),
                        "Shard timestamps differ from original saved training timestamps")
        for j in range(23):
            counts[j] += np.bincount(a["status"][:, j], minlength=5)
            valid = a["values"][a["status"][:, j] == 0, j]
            if len(valid):
                minima[j] = min(minima[j], float(valid.min()))
                maxima[j] = max(maxima[j], float(valid.max()))
                if dataset.binary[j]:
                    binary[j] += [(valid == 0).sum(), (valid == 1).sum()]
        print(f"Verified row shard {index+1}/{len(dataset.shards)}", flush=True)
    for j, name in enumerate(dataset.devices):
        windows.require(counts[j].tolist() == [dataset.preparation["counts"][name][k] for k in windows.STATUS_CODES],
                        f"Prepared status counts differ: {name}")
        if dataset.binary[j]:
            windows.require(binary[j].tolist() == [dataset.preparation["binary_counts"][name][k] for k in ["zero", "one"]],
                            f"Binary value counts differ: {name}")
        else:
            transform = dataset.metadata["transforms"][name]
            expected_max = 0 if transform["constant_unit_denominator"] else 1
            windows.require(minima[j] == 0 and maxima[j] == expected_max, f"Normalized range differs: {name}")
    report["all_row_shards_verified"] = len(dataset.verified_shards)
    report["status_totals"] = dict(zip(windows.STATUS_CODES, map(int, counts.sum(axis=0))))
    reference = ReferenceRows(dataset)
    checked = 0
    def check_batch(batch):
        nonlocal checked
        b = len(batch["indices"])
        for prefix in ["X", "Y"]:
            features, status, values = batch[prefix], batch[prefix+"_status"], batch[prefix+"_values"]
            windows.require(features.shape == (b, window_size, 23, 6) and features.dtype == np.float32,
                            "Feature tensor shape/dtype differs")
            windows.require(np.array_equal(features[..., 0], values), "Feature value slot differs")
            windows.require(np.all(features[..., 1:].sum(axis=-1) == 1), "Status encoding is not one-hot")
            for code in range(5):
                windows.require(np.array_equal(features[..., code+1], status == code), "Status feature order differs")
        windows.require(np.array_equal(batch["X"][:, 1:], batch["Y"][:, :-1]), "Window overlap differs")
        for row, start in enumerate(batch["indices"]):
            src = reference.slice(int(start), int(start) + window_size + 1)
            for prefix, select in [("X", slice(None, -1)), ("Y", slice(1, None))]:
                for key in ["values", "status", "timestamps"]:
                    windows.require(np.array_equal(batch[prefix+"_"+key][row], src[key][select]),
                                    f"Independent reference mismatch: window {start}, {prefix}/{key}")
            checked += 1
    for left in range(0, len(indices), batch_size):
        requested = np.array(indices[left:left+batch_size], dtype=np.int64)
        batch = dataset.get_batch(requested)
        windows.require(np.array_equal(batch["indices"], requested), "Batch order changed")
        check_batch(batch)
    # Exercise nonmonotonic/repeated indices and iterator final partial batches.
    probe = np.array([len(dataset)-1, 0, len(dataset)-1], dtype=np.int64)
    batch = dataset.get_batch(probe)
    windows.require(np.array_equal(batch["indices"], probe), "Repeated/reordered batch indices changed")
    check_batch(batch)
    stop = min(len(dataset), batch_size + 1)
    sequential = []
    for batch in dataset.iter_batches(batch_size, stop=stop):
        sequential.extend(map(int, batch["indices"]))
        check_batch(batch)
    windows.require(sequential == list(range(stop)), "Sequential iteration duplicated/dropped samples")
    for bad in [-1, len(dataset), 1.5, True]:
        try:
            dataset[bad]
        except ValueError:
            continue
        raise ValueError(f"Invalid sample index accepted: {bad}")
    dataset.verify_unchanged()
    windows.require(all(windows.sha256(p) == hashes[k] for k, p in dependencies.items()),
                    "A verification dependency changed during execution")
    report.update(status="PASS", windows_checked_including_interface_probes=checked,
                  boundary_count=len(dataset.starts)-1, row_integrity_scan_complete=True,
                  output_feature_order=["value"]+list(windows.STATUS_CODES),
                  batch_shape=[batch_size, window_size, 23, 6])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root_dir", default=None)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--random_windows", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    args = parser.parse_args()
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    if args.project_root_dir:
        config["project_root_dir"] = args.project_root_dir
    processed = Path(get_processed_folder(config))
    report = {"protocol": "v2_input_state_window_verification_v1", "status": "BLOCKED",
              "configuration": config, "optimizer_steps": 0, "model_forward_calls": 0,
              "scientific_claim_decision": "NOT_EVALUATED",
              "scope": "Full row-artifact integrity scan; bounded window/interface verification",
              "limitations": ["Window content checks are bounded, not all training pairs.",
                              "No raw-source re-audit or checkpoint/model compatibility assertion.",
                              "Output tensors include availability status, not physical behavior alone."]}
    try:
        verify(config, processed, args, report)
    except (ValueError, TypeError, KeyError, OSError, IndexError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = processed / "v2_validation" / "input_state_windows" / run_id
    output.mkdir(parents=True, exist_ok=False)
    path = output / "window_loader_verification_report.json"
    with path.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f"Window verification: {report['status']}\nReport: {path}")
    if "error" in report:
        print(report["error"])
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
