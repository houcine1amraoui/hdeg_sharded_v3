"""Lazy training window pairs over immutable, verified input-state row shards."""
from __future__ import annotations

import hashlib
import io
import json
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import numpy as np

SOURCE_POLICY = "cu_source_input_state_v1"
LEGACY_POLICY = "cu_input_state_v1"
FEATURE_ORDER = ["value", "valid", "unknown", "unavailable", "missing", "invalid"]

STATUS_CODES = {"valid": 0, "unknown": 1, "unavailable": 2, "missing": 3, "invalid": 4}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stamp(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def safe_file(directory, name):
    require(isinstance(name, str) and name and Path(name).name == name
            and name not in [".", ".."], "Artifact filenames must be local basenames")
    path = directory / name
    require(path.resolve().parent == directory.resolve(), "Artifact escapes input directory")
    return path


EXPECTED_SOURCE_POLICY = {'policy_version': 'cu_source_input_state_v1', 'missing_tokens': ['', 'NULL'], 'status_codes': {'valid': 0, 'unknown': 1, 'unavailable': 2, 'missing': 3, 'invalid': 4}, 'feature_order': ['value', 'valid', 'unknown', 'unavailable', 'missing', 'invalid'], 'time_field': 'last_updated', 'timezone_conversion': None, 'resampling': 'causal_backward_asof', 'backward_fill': False, 'skip_nonvalid_records': False, 'expiry_seconds': None, 'duplicate_conflicts': 'block', 'nonvalid_placeholder': 0, 'binary_literals': {'on': 1, 'off': 0}, 'binary_numeric_values': [0, 1], 'continuous_values': 'finite numeric literals', 'whitespace_normalization': False, 'numeric_status_sentinel_interpretation': False, 'continuous_scaling': 'valid-training-grid minmax; constant denominator 1; no clipping'}


def validate_source_metadata(manifest, meta, prep, policy, metadata_hashes):
    require(policy == EXPECTED_SOURCE_POLICY, "Source policy semantics differ from supported contract")
    require(manifest["source_policy_sha256"] == metadata_hashes["source_policy.json"],
            "Source policy hash mismatch")
    require(manifest["feature_order"] == meta["feature_order"] == FEATURE_ORDER,
            "Six-feature order differs")

    def source_hash(name):
        values = [v for p, v in meta["source_hashes"].items() if Path(p).name == name]
        require(len(values) == 1, "Missing or ambiguous source provenance: " + name)
        value = values[0]
        require(isinstance(value, str) and len(value) == 64
                and all(c in "0123456789abcdef" for c in value), "Invalid source digest")
        return value

    require(manifest["registry_sha256"] == source_hash("CU_input_state_registry.json"),
            "Registry provenance mismatch")
    require(manifest["reference_input_manifest_sha256"] == source_hash("manifest.json"),
            "Reference input provenance mismatch")
    require(manifest["coverage_report_sha256"] == source_hash("training_fill_coverage_report.json"),
            "Coverage audit provenance mismatch")
    require(prep["source_size_mtime_recheck_passed"] is True, "Preparation source recheck failed")
    require(prep["model_forwards"] == prep["optimizer_steps"] == 0,
            "Unexpected model execution in preparation")
    require(prep["training_bounds"] == [meta["training"]["first"], meta["training"]["last"]],
            "Preparation training bounds differ")
    for channel in meta["registry"]["channels"]:
        name, kind = channel["name"], channel["type"]
        d = meta["channel_diagnostics"][name]
        counts = d["status_counts"]
        require(set(counts) == set(STATUS_CODES) and
                all(type(v) is int and v >= 0 for v in counts.values()) and
                sum(counts.values()) == manifest["num_rows"], "Invalid channel status counts")
        require(d["no_prior_record_rows"] >= 0 and d["explicit_missing_rows"] >= 0 and
                counts["missing"] == d["no_prior_record_rows"] + d["explicit_missing_rows"],
                "Missing-state decomposition differs")
        scale = meta["scaling"][name]
        require(scale["type"] == kind and scale["eligible_grid_rows"] == counts["valid"],
                "Scaler eligibility differs")
        if kind == "binary":
            require(scale["transform"] == "identity", "Binary transform differs")
        else:
            lo, hi, den = scale["minimum"], scale["maximum"], scale["denominator"]
            require(counts["valid"] > 0 and np.isfinite([lo, hi, den]).all() and hi >= lo,
                    "Invalid training extrema")
            require(scale["transform"] == "minmax" and scale["clip"] is False and
                    scale["constant"] == (hi == lo) and den == (hi-lo if hi > lo else 1.),
                    "Continuous scaling contract differs")


class InputStateWindowDataset:
    """NumPy map-style dataset; no model, fitting, or materialized window files.

    Window i uses rows [i:i+W] for X and [i+1:i+W+1] for Y.
    __getitem__ returns one pair; get_batch returns batched pairs.
    X/Y contain [value, valid, unknown, unavailable, missing, invalid].
    A two-shard LRU bounds persistent cache memory. Returned arrays are copies.
    """
    def __init__(self, input_dir, window_size, cache_shards=2):
        require(type(window_size) is int and window_size > 0, "window_size must be positive")
        require(type(cache_shards) is int and cache_shards > 0, "cache_shards must be positive")
        self.directory = Path(input_dir)
        self.window_size = window_size
        self.cache_limit = cache_shards
        self.cache = OrderedDict()
        self.verified_shards = set()
        self.manifest_path = self.directory / "manifest.json"
        blob = self.manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(blob).hexdigest()
        self.manifest = m = json.loads(blob)
        require(m["status"] == "COMPLETE" and m["format"] == "hdeg_input_state_rows_v1"
                and m["split"] == "train" and m["policy_version"] in [LEGACY_POLICY, SOURCE_POLICY],
                "Expected complete training row artifacts with a supported policy")
        self.policy_version = m["policy_version"]
        require(m["windows_constructed"] is False, "Expected row-level artifacts")
        self.devices = m["devices"]
        require(m["num_channels"] == len(self.devices) == len(set(self.devices)) == 23,
                "Expected 23 unique ordered channels")
        require(m["status_codes"] == STATUS_CODES, "Unexpected status-code mapping")
        self.num_rows = m["num_rows"]
        require(type(self.num_rows) is int and self.num_rows > window_size,
                "At least W+1 rows are needed for a training pair")
        self.metadata_hashes = {}
        loaded = {}
        metadata_names = ["transformation_metadata.json", "preparation_report.json"]
        if self.policy_version == SOURCE_POLICY:
            metadata_names.append("source_policy.json")
        for name in metadata_names:
            entry = m["metadata"][name]
            require(entry["file"] == name, "Metadata filename mismatch")
            path = safe_file(self.directory, name)
            b = path.read_bytes()
            require(hashlib.sha256(b).hexdigest() == entry["sha256"], f"Metadata hash mismatch: {name}")
            loaded[name] = json.loads(b)
            self.metadata_hashes[name] = entry["sha256"]
        self.metadata = meta = loaded["transformation_metadata.json"]
        self.preparation = prep = loaded["preparation_report.json"]
        require(meta["devices"] == self.devices and meta["status_codes"] == STATUS_CODES
                and meta["nonvalid_placeholder"] == 0, "Transformation metadata contract differs")
        require(prep["status"] == "COMPLETE" and prep["rows_written"] == self.num_rows,
                "Preparation report is incomplete or has different row count")
        require(all(d["policy_version"] == m["policy_version"] and d["split"] == "train"
                    for d in [meta, prep]), "Metadata split/policy mismatch")
        require(meta["source_hashes"] == prep["source_hashes"], "Metadata provenance mismatch")
        if self.policy_version == LEGACY_POLICY:
            require(m["preflight_report_sha256"] == meta["source_hashes"]["preflight_report"]
                    and m["registry_sha256"] == meta["source_hashes"]["registry"], "Manifest provenance mismatch")
        else:
            validate_source_metadata(m, meta, prep, loaded["source_policy.json"], self.metadata_hashes)
        channels = meta["registry"]["channels"]
        require([c["name"] for c in channels] == self.devices, "Registry device order mismatch")
        require(meta["registry"]["status_order"] == list(STATUS_CODES), "Registry status order mismatch")
        require(all(c["type"] in ["binary", "continuous"] for c in channels), "Invalid registry types")
        self.binary = np.array([c["type"] == "binary" for c in channels])
        require(int(self.binary.sum()) == 18, "Expected 18 binary and five continuous channels")
        self.first = np.datetime64(meta["training"]["first"], "ns")
        require(not np.isnat(self.first), "Invalid first timestamp")
        require(meta["training"]["rows_matched"] == meta["training"]["rows_expected"] == self.num_rows,
                "Training row count mismatch")
        require(np.datetime64(meta["training"]["last"], "ns") ==
                self.first + np.timedelta64(self.num_rows - 1, "s"), "Training interval is not continuous at 1s")
        self.shards = m["shards"]
        require(len(self.shards) == prep["shards_written"] and len(self.shards) > 0, "Shard count mismatch")
        cursor = 0
        filenames = set()
        self.paths = []
        for i, e in enumerate(self.shards):
            require(e["index"] == i and type(e["rows"]) is int and e["rows"] > 0
                    and e["start"] == cursor and e["stop"] == cursor + e["rows"],
                    "Shard ranges must partition the rows exactly")
            require(e["file"] not in filenames, "Repeated shard filename")
            filenames.add(e["file"])
            self.paths.append(safe_file(self.directory, e["file"]))
            require(np.datetime64(e["first_timestamp"], "ns") == self.first + np.timedelta64(cursor, "s")
                    and np.datetime64(e["last_timestamp"], "ns") ==
                    self.first + np.timedelta64(e["stop"] - 1, "s"), "Shard time boundary mismatch")
            cursor = e["stop"]
        require(cursor == self.num_rows, "Shard ranges do not cover the training rows")
        self.starts = [e["start"] for e in self.shards]

    def __len__(self):
        return self.num_rows - self.window_size

    def load_shard(self, index):
        """Load, hash, and validate a shard; cached arrays are read-only."""
        require(isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_))
                and 0 <= index < len(self.shards), "Invalid shard index")
        index = int(index)
        path = self.paths[index]
        if index in self.cache:
            arrays, previous_stamp = self.cache.pop(index)
            require(stamp(path) == previous_stamp, "Cached shard changed on disk")
            self.cache[index] = arrays, previous_stamp
            return arrays
        e = self.shards[index]
        before = stamp(path)
        blob = path.read_bytes()
        require(stamp(path) == before, "Shard changed during read")
        require(len(blob) == e["bytes"] and hashlib.sha256(blob).hexdigest() == e["sha256"],
                f"Shard hash/size mismatch: {path.name}")
        with np.load(io.BytesIO(blob), allow_pickle=False) as a:
            require(set(a.files) == {"values", "status", "timestamps"}, "Unexpected shard array keys")
            v, s, t = a["values"], a["status"], a["timestamps"]
        require(v.dtype == np.float32 and s.dtype == np.uint8 and t.dtype == np.dtype("datetime64[ns]"),
                "Unexpected shard dtypes")
        require(v.shape == s.shape == (e["rows"], 23) and t.shape == (e["rows"],), "Shard shape mismatch")
        require(np.isfinite(v).all() and np.isin(s, [0, 1, 2, 3, 4] if self.policy_version == SOURCE_POLICY else [0, 1, 2, 3]).all(), "Invalid shard values/status codes")
        require(np.all(v[s != 0] == 0), "Non-valid placeholder must be zero")
        require(np.isin(v[:, self.binary][s[:, self.binary] == 0], [0, 1]).all(), "Invalid binary value")
        continuous = v[:, ~self.binary][s[:, ~self.binary] == 0]
        require(np.all((continuous >= 0) & (continuous <= 1)), "Training continuous value outside [0,1]")
        expected = self.first + np.arange(e["start"], e["stop"]).astype("timedelta64[s]")
        require(np.array_equal(t, expected), "Shard timestamp gap or misalignment")
        arrays = {"values": v, "status": s, "timestamps": t}
        for a in arrays.values():
            a.setflags(write=False)
        self.cache[index] = arrays, before
        while len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
        self.verified_shards.add(index)
        return arrays

    def read_rows(self, start, stop):
        require(isinstance(start, (int, np.integer)) and isinstance(stop, (int, np.integer))
                and not isinstance(start, (bool, np.bool_)) and not isinstance(stop, (bool, np.bool_))
                and 0 <= start < stop <= self.num_rows, "Invalid row interval")
        result = {"values": [], "status": [], "timestamps": []}
        cursor = int(start)
        while cursor < stop:
            index = bisect_right(self.starts, cursor) - 1
            e = self.shards[index]
            a = self.load_shard(index)
            end = min(int(stop), e["stop"])
            for key in result:
                result[key].append(a[key][cursor-e["start"]:end-e["start"]])
            cursor = end
        return {key: np.concatenate(parts, axis=0) for key, parts in result.items()}

    def get_batch(self, indices):
        indices = np.asarray(indices)
        require(indices.ndim == 1 and len(indices) > 0 and indices.dtype.kind in "iu",
                "Batch indices must be a nonempty integer vector")
        require(np.all(indices < len(self)) and np.all(indices >= 0), "Window index out of bounds")
        indices = indices.astype(np.int64)
        blocks = [self.read_rows(int(i), int(i) + self.window_size + 1) for i in indices]
        values = np.stack([b["values"] for b in blocks])
        status = np.stack([b["status"] for b in blocks])
        times = np.stack([b["timestamps"] for b in blocks])
        features = np.concatenate([values[..., None], np.eye(5, dtype=np.float32)[status]], axis=-1)
        return {"indices": indices.copy(), "X": features[:, :-1].copy(), "Y": features[:, 1:].copy(),
                "X_values": values[:, :-1].copy(), "Y_values": values[:, 1:].copy(),
                "X_status": status[:, :-1].copy(), "Y_status": status[:, 1:].copy(),
                "X_timestamps": times[:, :-1].copy(), "Y_timestamps": times[:, 1:].copy()}

    def __getitem__(self, index):
        require(isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_)),
                "Window index must be an integer")
        return {k: v[0] for k, v in self.get_batch([index]).items()}

    def iter_batches(self, batch_size, start=0, stop=None):
        stop = len(self) if stop is None else stop
        require(type(batch_size) is int and batch_size > 0, "batch_size must be positive")
        require(type(start) is int and type(stop) is int and 0 <= start <= stop <= len(self),
                "Invalid window iteration range")
        for left in range(start, stop, batch_size):
            yield self.get_batch(np.arange(left, min(left + batch_size, stop), dtype=np.int64))

    def verify_unchanged(self):
        require(sha256(self.manifest_path) == self.manifest_sha256, "Manifest changed during use")
        for name, expected in self.metadata_hashes.items():
            require(sha256(self.directory / name) == expected, "Metadata changed during use")
        for path, entry in zip(self.paths, self.shards):
            require(sha256(path) == entry["sha256"], "Shard changed during use")
