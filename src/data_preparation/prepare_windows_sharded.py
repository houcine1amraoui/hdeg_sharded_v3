from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from src.utils.get_folders_utils import get_processed_folder


SPLITS = (
    "train",
    "val",
    "actor2_test",
    "actor1_test",
)

DEFAULT_SHARD_SIZE = 8192
OUTPUT_DIR_NAME = "windows"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct sharded window-to-window forecasting pairs from "
            "the normalized preprocessing outputs."
        )
    )

    parser.add_argument(
        "--processed_dir",
        type=str,
        default=None,
        help=(
            "Directory containing arrays.npz, timestamps.npz, and "
            "devices.json. If omitted, the directory is resolved from config."
        ),
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help=(
            "Observation window length W. If omitted, uses "
            "config['preprocessing']['window_size']."
        ),
    )

    parser.add_argument(
        "--shard_size",
        type=int,
        default=None,
        help=(
            "Maximum number of window-to-window samples per shard. "
            "If omitted, uses config['preprocessing']['window_shard_size'] "
            f"when available, otherwise {DEFAULT_SHARD_SIZE}."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory in which sharded window files are saved. "
            "Defaults to <processed_dir>/windows."
        ),
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Splits to process. By default all splits are processed.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shard files and split manifests.",
    )

    return parser.parse_args()


def validate_window_size(window_size: int) -> None:
    if window_size <= 0:
        raise ValueError(
            f"window_size must be greater than zero. Received {window_size}."
        )


def validate_shard_size(shard_size: int) -> None:
    if shard_size <= 0:
        raise ValueError(
            f"shard_size must be greater than zero. Received {shard_size}."
        )


def load_devices(processed_dir: Path) -> list:
    devices_path = processed_dir / "devices.json"

    if not devices_path.is_file():
        raise FileNotFoundError(
            f"Device metadata was not found:\n{devices_path}"
        )

    with devices_path.open("r", encoding="utf-8") as file:
        devices = json.load(file)

    if not isinstance(devices, list):
        raise ValueError("devices.json must contain a JSON list.")

    if not devices:
        raise ValueError("devices.json contains no devices.")

    return devices


def validate_data_array(data: np.ndarray, split: str) -> None:
    if data.ndim != 2:
        raise ValueError(
            f"Split '{split}' must have shape (T, N), "
            f"but received {data.shape}."
        )

    num_timesteps, num_devices = data.shape

    if num_timesteps <= 0:
        raise ValueError(
            f"Split '{split}' contains no observations."
        )

    if num_devices <= 0:
        raise ValueError(
            f"Split '{split}' contains no devices."
        )

    if not np.issubdtype(data.dtype, np.floating):
        raise TypeError(
            f"Split '{split}' must contain floating-point values. "
            f"Received dtype {data.dtype}."
        )

    if not np.isfinite(data).all():
        raise ValueError(
            f"Split '{split}' contains NaN or infinite values."
        )


def validate_timestamps(
    timestamps: np.ndarray,
    data: np.ndarray,
    split: str,
) -> None:
    if timestamps.ndim != 1:
        raise ValueError(
            f"Timestamps for split '{split}' must be one-dimensional. "
            f"Received shape {timestamps.shape}."
        )

    if len(timestamps) != len(data):
        raise ValueError(
            f"Timestamp/data length mismatch for split '{split}': "
            f"{len(timestamps)} timestamps vs {len(data)} observations."
        )


def validate_devices(
    devices: Sequence[str],
    num_devices: int,
) -> None:
    if len(devices) != num_devices:
        raise ValueError(
            "Device-count mismatch: "
            f"devices.json contains {len(devices)} devices, "
            f"but the data contains {num_devices} columns."
        )


def construct_window_to_window_shard(
    data: np.ndarray,
    timestamps: np.ndarray,
    start: int,
    end: int,
    window_size: int,
) -> Dict[str, np.ndarray]:
    """
    Construct one contiguous shard of window-to-window samples.

    For each starting index t:

        X_t = data[t : t + W]
        Y_t = data[t + 1 : t + W + 1]

    Therefore both X and Y have shape:

        (shard_samples, W, N)

    The source slice has exactly shard_samples + W observations.
    """
    shard_samples = end - start

    if shard_samples <= 0:
        raise ValueError(
            f"Invalid shard range [{start}, {end})."
        )

    source_end = end + window_size

    data_chunk = data[start:source_end]
    timestamp_chunk = timestamps[start:source_end]

    expected_chunk_length = shard_samples + window_size

    if len(data_chunk) != expected_chunk_length:
        raise RuntimeError(
            "Unexpected source chunk length. "
            f"Expected {expected_chunk_length}, "
            f"received {len(data_chunk)}."
        )

    if len(timestamp_chunk) != expected_chunk_length:
        raise RuntimeError(
            "Unexpected timestamp chunk length. "
            f"Expected {expected_chunk_length}, "
            f"received {len(timestamp_chunk)}."
        )

    # sliding_window_view(..., axis=0) returns:
    #   (num_windows, N, W)
    #
    # Transpose to the HDEG convention:
    #   (num_windows, W, N)
    views = np.lib.stride_tricks.sliding_window_view(
        data_chunk,
        window_shape=window_size,
        axis=0,
    )
    views = np.transpose(views, (0, 2, 1))

    # There are shard_samples + 1 possible windows in the source
    # chunk. The first shard_samples are X_t and the next shard_samples
    # are Y_t = X_{t+1}.
    X = np.ascontiguousarray(
        views[:shard_samples],
        dtype=np.float32,
    )
    Y = np.ascontiguousarray(
        views[1 : shard_samples + 1],
        dtype=np.float32,
    )

    # Timestamp semantics:
    #
    # window_start_timestamps[t] = timestamp[start + t]
    # target_timestamps[t]       = timestamps[start+t+1 : start+t+W+1]
    #
    # Thus target_timestamps has shape (S, W), matching Y.
    timestamp_views = np.lib.stride_tricks.sliding_window_view(
        timestamp_chunk,
        window_shape=window_size,
    )

    window_start_timestamps = np.asarray(
        timestamp_chunk[:shard_samples]
    ).copy()

    target_timestamps = np.asarray(
        timestamp_views[1 : shard_samples + 1]
    ).copy()

    num_devices = data.shape[1]

    expected_x_shape = (
        shard_samples,
        window_size,
        num_devices,
    )
    expected_y_shape = expected_x_shape
    expected_start_shape = (shard_samples,)
    expected_target_timestamp_shape = (
        shard_samples,
        window_size,
    )

    if X.shape != expected_x_shape:
        raise RuntimeError(
            "Unexpected X shard shape. "
            f"Expected {expected_x_shape}, received {X.shape}."
        )

    if Y.shape != expected_y_shape:
        raise RuntimeError(
            "Unexpected Y shard shape. "
            f"Expected {expected_y_shape}, received {Y.shape}."
        )

    if window_start_timestamps.shape != expected_start_shape:
        raise RuntimeError(
            "Unexpected window-start timestamp shape. "
            f"Expected {expected_start_shape}, "
            f"received {window_start_timestamps.shape}."
        )

    if target_timestamps.shape != expected_target_timestamp_shape:
        raise RuntimeError(
            "Unexpected target timestamp shape. "
            f"Expected {expected_target_timestamp_shape}, "
            f"received {target_timestamps.shape}."
        )

    return {
        "X": X,
        "Y": Y,
        "window_start_timestamps": window_start_timestamps,
        "target_timestamps": target_timestamps,
    }


def verify_shard_alignment(
    data: np.ndarray,
    windows: Dict[str, np.ndarray],
    start: int,
    window_size: int,
) -> None:
    """
    Verify first, middle, and last samples of a shard.

    This directly checks the window-to-window contract:

        X_t = data[t:t+W]
        Y_t = data[t+1:t+W+1]
    """
    X = windows["X"]
    Y = windows["Y"]

    num_samples = X.shape[0]

    if num_samples == 0:
        raise RuntimeError("Cannot verify an empty shard.")

    indices = {0, num_samples - 1}

    if num_samples >= 3:
        indices.add(num_samples // 2)

    for local_idx in sorted(indices):
        global_idx = start + local_idx

        expected_x = data[
            global_idx : global_idx + window_size
        ].astype(np.float32, copy=False)

        expected_y = data[
            global_idx + 1 : global_idx + window_size + 1
        ].astype(np.float32, copy=False)

        if not np.array_equal(X[local_idx], expected_x):
            raise RuntimeError(
                "Window alignment failure: "
                f"X at global sample {global_idx} is incorrect."
            )

        if not np.array_equal(Y[local_idx], expected_y):
            raise RuntimeError(
                "Target-window alignment failure: "
                f"Y at global sample {global_idx} is incorrect."
            )

    # Timestamp verification for the same samples.
    timestamps = None


def verify_shard_timestamp_alignment(
    timestamps: np.ndarray,
    windows: Dict[str, np.ndarray],
    start: int,
    window_size: int,
) -> None:
    window_starts = windows["window_start_timestamps"]
    target_timestamps = windows["target_timestamps"]

    num_samples = window_starts.shape[0]

    indices = {0, num_samples - 1}

    if num_samples >= 3:
        indices.add(num_samples // 2)

    for local_idx in sorted(indices):
        global_idx = start + local_idx

        expected_start = timestamps[global_idx]
        expected_target = timestamps[
            global_idx + 1 : global_idx + window_size + 1
        ]

        if not np.array_equal(
            window_starts[local_idx],
            expected_start,
        ):
            raise RuntimeError(
                "Window-start timestamp alignment failure at "
                f"global sample {global_idx}."
            )

        if not np.array_equal(
            target_timestamps[local_idx],
            expected_target,
        ):
            raise RuntimeError(
                "Target timestamp alignment failure at "
                f"global sample {global_idx}."
            )


def save_window_shard(
    output_path: Path,
    split: str,
    shard_index: int,
    start: int,
    end: int,
    window_size: int,
    num_devices: int,
    windows: Dict[str, np.ndarray],
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output shard already exists:\n{output_path}\n\n"
            "Use --overwrite if you want to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X=windows["X"],
        Y=windows["Y"],
        window_start_timestamps=windows["window_start_timestamps"],
        target_timestamps=windows["target_timestamps"],
        split=np.asarray(split),
        shard_index=np.asarray(shard_index, dtype=np.int64),
        start_index=np.asarray(start, dtype=np.int64),
        end_index=np.asarray(end, dtype=np.int64),
        window_size=np.asarray(window_size, dtype=np.int64),
        num_devices=np.asarray(num_devices, dtype=np.int64),
    )


def build_split_manifest(
    *,
    split: str,
    data: np.ndarray,
    timestamps: np.ndarray,
    window_size: int,
    shard_size: int,
    shard_paths: list[str],
) -> dict:
    num_timesteps, num_devices = data.shape
    num_samples = num_timesteps - window_size

    return {
        "split": split,
        "num_timesteps": int(num_timesteps),
        "num_devices": int(num_devices),
        "window_size": int(window_size),
        "num_samples": int(num_samples),
        "shard_size": int(shard_size),
        "num_shards": len(shard_paths),
        "dtype": "float32",
        "input_shape_per_sample": [
            int(window_size),
            int(num_devices),
        ],
        "target_shape_per_sample": [
            int(window_size),
            int(num_devices),
        ],
        "input_semantics": "X_t = data[t:t+W]",
        "target_semantics": "Y_t = data[t+1:t+W+1]",
        "shards": shard_paths,
    }


def process_split(
    *,
    split: str,
    data: np.ndarray,
    timestamps: np.ndarray,
    devices: Sequence[str],
    output_root: Path,
    window_size: int,
    shard_size: int,
    overwrite: bool,
) -> dict:
    print()
    print("=" * 70)
    print(f"Processing split: {split}")
    print("=" * 70)

    validate_data_array(data, split)
    validate_timestamps(timestamps, data, split)

    num_timesteps, num_devices = data.shape
    validate_devices(devices, num_devices)

    num_samples = num_timesteps - window_size

    if num_samples <= 0:
        raise ValueError(
            f"Split '{split}' does not contain enough observations "
            f"for window_size={window_size}. "
            f"Received T={num_timesteps}."
        )

    print(f"Input shape          : {data.shape}")
    print(f"Input dtype          : {data.dtype}")
    print(f"Number of devices    : {num_devices}")
    print(f"Window size          : {window_size}")
    print(f"Window-to-window samples: {num_samples}")
    print(f"Shard size           : {shard_size}")

    split_dir = output_root / split

    if overwrite and split_dir.exists():
        for path in split_dir.glob("shard_*.npz"):
            path.unlink()

    split_dir.mkdir(parents=True, exist_ok=True)

    shard_paths: list[str] = []
    shard_index = 0

    for start in range(0, num_samples, shard_size):
        end = min(start + shard_size, num_samples)

        print(
            f"  Shard {shard_index:06d}: "
            f"samples [{start}, {end}) "
            f"({end - start} samples)"
        )

        windows = construct_window_to_window_shard(
            data=data,
            timestamps=timestamps,
            start=start,
            end=end,
            window_size=window_size,
        )

        verify_shard_alignment(
            data=data,
            windows=windows,
            start=start,
            window_size=window_size,
        )

        verify_shard_timestamp_alignment(
            timestamps=timestamps,
            windows=windows,
            start=start,
            window_size=window_size,
        )

        if not np.isfinite(windows["X"]).all():
            raise RuntimeError(
                f"Generated X for shard {shard_index} "
                "contains NaN or infinite values."
            )

        if not np.isfinite(windows["Y"]).all():
            raise RuntimeError(
                f"Generated Y for shard {shard_index} "
                "contains NaN or infinite values."
            )

        if windows["X"].dtype != np.float32:
            raise RuntimeError(
                f"Generated X for shard {shard_index} is not float32."
            )

        if windows["Y"].dtype != np.float32:
            raise RuntimeError(
                f"Generated Y for shard {shard_index} is not float32."
            )

        shard_name = f"shard_{shard_index:06d}.npz"
        shard_path = split_dir / shard_name

        save_window_shard(
            output_path=shard_path,
            split=split,
            shard_index=shard_index,
            start=start,
            end=end,
            window_size=window_size,
            num_devices=num_devices,
            windows=windows,
            overwrite=overwrite,
        )

        # Re-open the saved artifact and verify its persisted contents.
        with np.load(
            shard_path,
            allow_pickle=False,
        ) as archive:
            required = (
                "X",
                "Y",
                "window_start_timestamps",
                "target_timestamps",
                "split",
                "shard_index",
                "start_index",
                "end_index",
                "window_size",
                "num_devices",
            )

            missing = [
                key for key in required
                if key not in archive.files
            ]

            if missing:
                raise RuntimeError(
                    f"Saved shard {shard_path} is missing entries: "
                    f"{missing}"
                )

            saved_X = archive["X"]
            saved_Y = archive["Y"]

            if saved_X.shape != windows["X"].shape:
                raise RuntimeError(
                    f"Persisted X shape mismatch in {shard_path}."
                )

            if saved_Y.shape != windows["Y"].shape:
                raise RuntimeError(
                    f"Persisted Y shape mismatch in {shard_path}."
                )

            if saved_X.dtype != np.float32:
                raise RuntimeError(
                    f"Persisted X dtype mismatch in {shard_path}: "
                    f"{saved_X.dtype}"
                )

            if saved_Y.dtype != np.float32:
                raise RuntimeError(
                    f"Persisted Y dtype mismatch in {shard_path}: "
                    f"{saved_Y.dtype}"
                )

            if int(archive["start_index"]) != start:
                raise RuntimeError(
                    f"Persisted start_index mismatch in {shard_path}."
                )

            if int(archive["end_index"]) != end:
                raise RuntimeError(
                    f"Persisted end_index mismatch in {shard_path}."
                )

        shard_paths.append(str(Path(split) / shard_name))
        shard_index += 1

    manifest = build_split_manifest(
        split=split,
        data=data,
        timestamps=timestamps,
        window_size=window_size,
        shard_size=shard_size,
        shard_paths=shard_paths,
    )

    manifest_path = split_dir / "manifest.json"

    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest already exists:\n{manifest_path}\n\n"
            "Use --overwrite if you want to replace it."
        )

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    # Global shard-count verification.
    total_shard_samples = 0

    for shard_path_relative in shard_paths:
        shard_path = output_root / shard_path_relative

        with np.load(
            shard_path,
            allow_pickle=False,
        ) as archive:
            total_shard_samples += int(
                archive["X"].shape[0]
            )

    if total_shard_samples != num_samples:
        raise RuntimeError(
            f"Shard coverage mismatch for split '{split}': "
            f"expected {num_samples} samples, "
            f"but shards contain {total_shard_samples}."
        )

    expected_num_shards = (
        num_samples + shard_size - 1
    ) // shard_size

    if len(shard_paths) != expected_num_shards:
        raise RuntimeError(
            f"Unexpected number of shards for split '{split}': "
            f"expected {expected_num_shards}, "
            f"received {len(shard_paths)}."
        )

    print()
    print(f"[PASS] Split '{split}' verified.")
    print(f"       Samples : {num_samples}")
    print(f"       Shards  : {len(shard_paths)}")
    print(f"       Format  : X_t -> Y_t = X_(t+1)")
    print(f"       Dtype   : float32")

    return manifest


def resolve_settings(config: dict, args: argparse.Namespace):
    preprocessing = config["preprocessing"]

    window_size = (
        args.window_size
        if args.window_size is not None
        else preprocessing["window_size"]
    )

    shard_size = (
        args.shard_size
        if args.shard_size is not None
        else preprocessing.get(
            "window_shard_size",
            DEFAULT_SHARD_SIZE,
        )
    )

    validate_window_size(window_size)
    validate_shard_size(shard_size)

    return window_size, shard_size


def main_prepare_windows(config) -> None:
    """
    Construct sharded HDEG window-to-window artifacts.

    This function preserves the existing main_preprocess.py integration:

        main_prepare_windows(config)

    and therefore requires no change to main_preprocess.py.
    """
    processed_data_folder = Path(
        get_processed_folder(config)
    )

    arrays_path = processed_data_folder / "arrays.npz"
    timestamps_path = processed_data_folder / "timestamps.npz"

    if not arrays_path.is_file():
        raise FileNotFoundError(
            f"Processed arrays were not found:\n{arrays_path}"
        )

    if not timestamps_path.is_file():
        raise FileNotFoundError(
            f"Processed timestamps were not found:\n{timestamps_path}"
        )

    devices = load_devices(processed_data_folder)

    window_size = config["preprocessing"]["window_size"]
    shard_size = config["preprocessing"].get(
        "window_shard_size",
        DEFAULT_SHARD_SIZE,
    )

    validate_window_size(window_size)
    validate_shard_size(shard_size)

    output_root = processed_data_folder / OUTPUT_DIR_NAME
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HDEG SHARDED WINDOW PREPARATION")
    print("=" * 70)
    print(f"Processed directory : {processed_data_folder}")
    print(f"Window size         : {window_size}")
    print(f"Shard size          : {shard_size}")
    print(f"Number of devices   : {len(devices)}")
    print(f"Output directory    : {output_root}")
    print()
    print("Forecasting contract:")
    print("  X_t = data[t:t+W]")
    print("  Y_t = data[t+1:t+W+1]")
    print()

    manifests = {}

    # NPZ archives are kept open while each requested split is processed.
    # Accessing arrays[split] materializes only the current split.
    with np.load(
        arrays_path,
        allow_pickle=False,
    ) as arrays, np.load(
        timestamps_path,
        allow_pickle=False,
    ) as timestamps_archive:

        for split in SPLITS:
            if split not in arrays.files:
                raise KeyError(
                    f"Split '{split}' was not found in {arrays_path}. "
                    f"Available entries: {arrays.files}"
                )

            if split not in timestamps_archive.files:
                raise KeyError(
                    f"Split '{split}' was not found in {timestamps_path}. "
                    f"Available entries: {timestamps_archive.files}"
                )

            # Only this split is materialized at a time.
            data = arrays[split]
            split_timestamps = timestamps_archive[split]

            manifests[split] = process_split(
                split=split,
                data=data,
                timestamps=split_timestamps,
                devices=devices,
                output_root=output_root,
                window_size=window_size,
                shard_size=shard_size,
                overwrite=True,
            )

            del data
            del split_timestamps

    global_manifest = {
        "format": "HDEG sharded window-to-window dataset",
        "window_size": int(window_size),
        "shard_size": int(shard_size),
        "num_devices": len(devices),
        "dtype": "float32",
        "input_semantics": "X_t = data[t:t+W]",
        "target_semantics": "Y_t = data[t+1:t+W+1]",
        "splits": manifests,
    }

    global_manifest_path = output_root / "manifest.json"

    with global_manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(global_manifest, file, indent=2)

    print()
    print("=" * 70)
    print("Window preparation completed successfully.")
    print("=" * 70)
    print(f"Window artifacts saved under:\n{output_root}")
    print(f"Global manifest:\n{global_manifest_path}")


def main() -> None:
    # This standalone entry point is optional; main_preprocess.py continues
    # to call main_prepare_windows(config) directly.
    import yaml

    parser = argparse.ArgumentParser(
        description="Prepare sharded HDEG window-to-window artifacts."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if args.window_size is not None:
        config["preprocessing"]["window_size"] = args.window_size

    if args.shard_size is not None:
        config["preprocessing"]["window_shard_size"] = args.shard_size

    # main_prepare_windows intentionally processes the four HDEG splits
    # defined by the preprocessing contract. The CLI remains available for
    # explicit configuration overrides, while the integrated path remains
    # unchanged.
    main_prepare_windows(config)


if __name__ == "__main__":
    main()