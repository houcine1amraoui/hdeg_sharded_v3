from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json

import numpy as np
import torch
import yaml


class BehavioralStateConfigError(ValueError):
    """Raised when the BSE behavioral-state configuration is invalid."""


@dataclass(frozen=True)
class BehavioralDevice:
    """A selected device feature with its fixed input-column index."""

    index: int
    device_id: str


@dataclass(frozen=True)
class BehavioralState:
    """A validated behavioral state and its compatible device columns."""

    id: str
    index: int
    name: str
    compatible_device_indices: tuple[int, ...]
    rationale: str
    provenance_type: str
    confidence: str


@dataclass(frozen=True)
class BehavioralStateConfiguration:
    """
    Fully validated BSE semantic configuration.

    This object is configuration infrastructure. It does not implement
    semantic attention or any neural computation.
    """

    schema_version: str
    artifact_type: str
    status: str
    dataset_name: str

    devices: tuple[BehavioralDevice, ...]
    states: tuple[BehavioralState, ...]
    compatibility_matrix: np.ndarray

    compatibility_relation: str
    contextual_edges: str

    @property
    def num_devices(self) -> int:
        return len(self.devices)

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return tuple(self.compatibility_matrix.shape)

    def torch_mask(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        clone: bool = True,
    ) -> torch.Tensor:
        """
        Return the validated compatibility matrix as a PyTorch tensor.

        Shape:
            (K, N)
        """
        tensor = torch.from_numpy(
            self.compatibility_matrix
        ).to(
            dtype=dtype,
            device=device,
        )

        return tensor.clone() if clone else tensor

    def device_ids(self) -> tuple[str, ...]:
        return tuple(
            device.device_id
            for device in self.devices
        )

    def state_ids(self) -> tuple[str, ...]:
        return tuple(
            state.id
            for state in self.states
        )

    def state_names(self) -> tuple[str, ...]:
        return tuple(
            state.name
            for state in self.states
        )


def load_behavioral_state_config(
    config_path: str | Path,
    devices_path: str | Path,
) -> BehavioralStateConfiguration:
    """
    Load and fully validate the behavioral-state configuration.

    Parameters
    ----------
    config_path:
        Path to behavioral_states.yaml.

    devices_path:
        Path to the authoritative devices.json.

    Returns
    -------
    BehavioralStateConfiguration
        Validated configuration suitable for BSE.
    """
    config_path = Path(config_path)
    devices_path = Path(devices_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Behavioral-state configuration not found: {config_path}"
        )

    if not devices_path.is_file():
        raise FileNotFoundError(
            f"Device-order file not found: {devices_path}"
        )

    config = _load_yaml(config_path)
    device_ids = _load_devices_json(devices_path)

    return _parse_configuration(
        config=config,
        device_ids=device_ids,
    )


def validate_behavioral_state_config(
    config_path: str | Path,
    devices_path: str | Path,
) -> BehavioralStateConfiguration:
    """Explicit validation alias for load_behavioral_state_config()."""
    return load_behavioral_state_config(
        config_path=config_path,
        devices_path=devices_path,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise BehavioralStateConfigError(
            f"Invalid YAML configuration: {path}"
        ) from exc

    if not isinstance(data, dict):
        raise BehavioralStateConfigError(
            "Behavioral-state configuration must be a mapping."
        )

    return data


def _load_devices_json(
    path: Path,
) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise BehavioralStateConfigError(
            f"Invalid devices.json: {path}"
        ) from exc

    if not isinstance(data, list):
        raise BehavioralStateConfigError(
            "devices.json must contain a list."
        )

    if not data:
        raise BehavioralStateConfigError(
            "devices.json must not be empty."
        )

    if not all(
        isinstance(item, str) and item.strip()
        for item in data
    ):
        raise BehavioralStateConfigError(
            "Every devices.json entry must be a non-empty string."
        )

    if len(set(data)) != len(data):
        raise BehavioralStateConfigError(
            "devices.json contains duplicate device identifiers."
        )

    return tuple(data)


def _parse_configuration(
    *,
    config: dict[str, Any],
    device_ids: tuple[str, ...],
) -> BehavioralStateConfiguration:

    schema_version = _require_string(
        config,
        "schema_version",
        "top-level",
    )

    artifact_type = _require_string(
        config,
        "artifact_type",
        "top-level",
    )

    status = _require_string(
        config,
        "status",
        "top-level",
    )

    if artifact_type != (
        "hdeg_behavioral_state_configuration"
    ):
        raise BehavioralStateConfigError(
            f"Unexpected artifact_type: {artifact_type!r}."
        )

    dataset = _require_mapping(
        config,
        "dataset",
        "top-level",
    )

    dataset_name = _require_string(
        dataset,
        "name",
        "dataset",
    )

    selected_device_count = _require_int(
        dataset,
        "selected_device_count",
        "dataset",
    )

    if selected_device_count != len(device_ids):
        raise BehavioralStateConfigError(
            "Device-count mismatch: "
            f"configuration={selected_device_count}, "
            f"devices.json={len(device_ids)}."
        )

    order_source = _require_string(
        dataset,
        "device_order_source",
        "dataset",
    )

    if order_source != "devices.json":
        raise BehavioralStateConfigError(
            "device_order_source must be 'devices.json'."
        )

    ordering_rule = _require_string(
        dataset,
        "device_ordering_rule",
        "dataset",
    )

    if ordering_rule != "exact_file_order":
        raise BehavioralStateConfigError(
            "device_ordering_rule must be "
            "'exact_file_order'."
        )

    semantic_policy = _require_mapping(
        config,
        "semantic_policy",
        "top-level",
    )

    compatibility_relation = _require_string(
        semantic_policy,
        "compatibility_relation",
        "semantic_policy",
    )

    contextual_edges = _require_string(
        semantic_policy,
        "contextual_edges",
        "semantic_policy",
    )

    if compatibility_relation != "DIRECT":
        raise BehavioralStateConfigError(
            "Version 1.0 requires compatibility_relation='DIRECT'."
        )

    if contextual_edges != "none":
        raise BehavioralStateConfigError(
            "Version 1.0 requires contextual_edges='none'."
        )

    states_raw = config.get("behavioral_states")

    if not isinstance(states_raw, list) or not states_raw:
        raise BehavioralStateConfigError(
            "behavioral_states must be a non-empty list."
        )

    states = _parse_states(
        states_raw=states_raw,
        device_ids=device_ids,
    )

    matrix_section = _require_mapping(
        config,
        "matrix",
        "top-level",
    )

    matrix = _parse_matrix(
        matrix_section=matrix_section,
        num_states=len(states),
        num_devices=len(device_ids),
    )

    devices = tuple(
        BehavioralDevice(
            index=index,
            device_id=device_id,
        )
        for index, device_id in enumerate(device_ids)
    )

    configuration = BehavioralStateConfiguration(
        schema_version=schema_version,
        artifact_type=artifact_type,
        status=status,
        dataset_name=dataset_name,
        devices=devices,
        states=tuple(states),
        compatibility_matrix=matrix,
        compatibility_relation=compatibility_relation,
        contextual_edges=contextual_edges,
    )

    _validate_configuration(
        configuration
    )

    return configuration


def _parse_states(
    *,
    states_raw: list[Any],
    device_ids: tuple[str, ...],
) -> list[BehavioralState]:

    states: list[BehavioralState] = []
    state_ids: set[str] = set()

    for expected_index, raw_state in enumerate(
        states_raw
    ):
        if not isinstance(raw_state, dict):
            raise BehavioralStateConfigError(
                f"behavioral_states[{expected_index}] "
                "must be a mapping."
            )

        context = (
            f"behavioral_states[{expected_index}]"
        )

        state_id = _require_string(
            raw_state,
            "id",
            context,
        )

        if state_id in state_ids:
            raise BehavioralStateConfigError(
                f"Duplicate behavioral-state id: {state_id!r}."
            )

        state_ids.add(state_id)

        index = _require_int(
            raw_state,
            "index",
            context,
        )

        if index != expected_index:
            raise BehavioralStateConfigError(
                f"{context}.index={index}; expected "
                f"{expected_index}."
            )

        name = _require_string(
            raw_state,
            "name",
            context,
        )

        rationale = _require_string(
            raw_state,
            "rationale",
            context,
        )

        provenance_type = _require_string(
            raw_state,
            "provenance_type",
            context,
        )

        confidence = _require_string(
            raw_state,
            "confidence",
            context,
        )

        if confidence not in {
            "high",
            "moderate",
            "low",
        }:
            raise BehavioralStateConfigError(
                f"Invalid confidence {confidence!r} "
                f"for state {state_id!r}."
            )

        compatible_raw = raw_state.get(
            "compatible_devices"
        )

        if not isinstance(
            compatible_raw,
            list,
        ) or not compatible_raw:
            raise BehavioralStateConfigError(
                f"{context}.compatible_devices must be "
                "a non-empty list."
            )

        compatible_indices: list[int] = []

        for device_entry_index, raw_device in enumerate(
            compatible_raw
        ):
            device_context = (
                f"{context}.compatible_devices"
                f"[{device_entry_index}]"
            )

            if not isinstance(
                raw_device,
                dict,
            ):
                raise BehavioralStateConfigError(
                    f"{device_context} must be a mapping."
                )

            device_index = _require_int(
                raw_device,
                "index",
                device_context,
            )

            if not (
                0 <= device_index < len(device_ids)
            ):
                raise BehavioralStateConfigError(
                    f"{device_context}.index={device_index} "
                    f"is outside [0, {len(device_ids) - 1}]."
                )

            declared_device_id = _require_string(
                raw_device,
                "id",
                device_context,
            )

            expected_device_id = device_ids[
                device_index
            ]

            if declared_device_id != expected_device_id:
                raise BehavioralStateConfigError(
                    f"{device_context}.id does not match "
                    "devices.json ordering: "
                    f"declared={declared_device_id!r}, "
                    f"expected={expected_device_id!r}."
                )

            relation = _require_string(
                raw_device,
                "relation",
                device_context,
            )

            if relation != "DIRECT":
                raise BehavioralStateConfigError(
                    f"{device_context}.relation must be "
                    "'DIRECT' in Version 1.0."
                )

            compatible_indices.append(
                device_index
            )

        if len(
            set(compatible_indices)
        ) != len(compatible_indices):
            raise BehavioralStateConfigError(
                f"State {state_id!r} contains duplicate "
                "compatible-device indices."
            )

        states.append(
            BehavioralState(
                id=state_id,
                index=index,
                name=name,
                compatible_device_indices=tuple(
                    compatible_indices
                ),
                rationale=rationale,
                provenance_type=provenance_type,
                confidence=confidence,
            )
        )

    return states


def _parse_matrix(
    *,
    matrix_section: dict[str, Any],
    num_states: int,
    num_devices: int,
) -> np.ndarray:

    shape = matrix_section.get("shape")

    if shape != [
        num_states,
        num_devices,
    ]:
        raise BehavioralStateConfigError(
            "Declared matrix shape does not match "
            f"expected [{num_states}, {num_devices}]."
        )

    if matrix_section.get("dtype") != "uint8":
        raise BehavioralStateConfigError(
            "Matrix dtype must be 'uint8'."
        )

    rows = matrix_section.get("rows")

    if not isinstance(rows, list):
        raise BehavioralStateConfigError(
            "matrix.rows must be a list."
        )

    if len(rows) != num_states:
        raise BehavioralStateConfigError(
            "Matrix row count does not match "
            "the number of behavioral states."
        )

    for row_index, row in enumerate(rows):
        if not isinstance(row, list):
            raise BehavioralStateConfigError(
                f"Matrix row {row_index} must be a list."
            )

        if len(row) != num_devices:
            raise BehavioralStateConfigError(
                f"Matrix row {row_index} must contain "
                f"{num_devices} entries."
            )

    matrix = np.asarray(
        rows,
        dtype=np.uint8,
    )

    if not np.all(
        np.isin(matrix, [0, 1])
    ):
        raise BehavioralStateConfigError(
            "Compatibility matrix must contain only "
            "0 and 1."
        )

    return matrix


def _validate_configuration(
    configuration: BehavioralStateConfiguration,
) -> None:

    K = configuration.num_states
    N = configuration.num_devices
    matrix = configuration.compatibility_matrix

    if matrix.shape != (K, N):
        raise BehavioralStateConfigError(
            f"Matrix shape {matrix.shape} does not match "
            f"(K={K}, N={N})."
        )

    # ---------------------------------------------------------------
    # Semantic declarations must exactly reproduce the matrix.
    # ---------------------------------------------------------------

    for state in configuration.states:

        declared = set(
            state.compatible_device_indices
        )

        actual = set(
            np.flatnonzero(
                matrix[state.index]
            ).tolist()
        )

        if declared != actual:
            raise BehavioralStateConfigError(
                f"State {state.id!r} declaration does not "
                "match its matrix row: "
                f"declared={sorted(declared)}, "
                f"matrix={sorted(actual)}."
            )

    # ---------------------------------------------------------------
    # Every behavioral state must have at least one compatible device.
    # ---------------------------------------------------------------

    row_counts = matrix.sum(axis=1)

    if not np.all(row_counts > 0):
        invalid = np.flatnonzero(
            row_counts == 0
        ).tolist()

        raise BehavioralStateConfigError(
            "States without compatible devices: "
            f"{invalid}."
        )

    # ---------------------------------------------------------------
    # Version 1.0 semantic partition:
    # every selected feature has exactly one primary state.
    # ---------------------------------------------------------------

    column_counts = matrix.sum(axis=0)

    if not np.all(column_counts == 1):
        invalid = np.flatnonzero(
            column_counts != 1
        ).tolist()

        raise BehavioralStateConfigError(
            "Every selected device must belong to exactly "
            "one primary behavioral state. Invalid columns: "
            f"{invalid}."
        )

    # ---------------------------------------------------------------
    # Final binary invariant.
    # ---------------------------------------------------------------

    if not np.all(
        np.isin(matrix, [0, 1])
    ):
        raise BehavioralStateConfigError(
            "Compatibility matrix is not binary."
        )


def _require_mapping(
    mapping: dict[str, Any],
    key: str,
    context: str,
) -> dict[str, Any]:

    value = mapping.get(key)

    if not isinstance(value, dict):
        raise BehavioralStateConfigError(
            f"{context}.{key} must be a mapping."
        )

    return value


def _require_string(
    mapping: dict[str, Any],
    key: str,
    context: str,
) -> str:

    value = mapping.get(key)

    if not isinstance(value, str):
        raise BehavioralStateConfigError(
            f"{context}.{key} must be a string."
        )

    value = value.strip()

    if not value:
        raise BehavioralStateConfigError(
            f"{context}.{key} must not be empty."
        )

    return value


def _require_int(
    mapping: dict[str, Any],
    key: str,
    context: str,
) -> int:

    value = mapping.get(key)

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise BehavioralStateConfigError(
            f"{context}.{key} must be an integer."
        )

    return value


__all__ = [
    "BehavioralDevice",
    "BehavioralState",
    "BehavioralStateConfiguration",
    "BehavioralStateConfigError",
    "load_behavioral_state_config",
    "validate_behavioral_state_config",
]