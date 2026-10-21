from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn


class HierarchicalBehavioralForecaster(nn.Module):
    """
    HDEG Hierarchical Behavioral Forecasting (HBF).

    Scientific contract
    -------------------
    Input hierarchical representation set at time t:

        R_t = {Z_t, S_t, S_tilde, g_t}

    with shapes:
        Z_t       : (B, N, D)
        S_t       : (B, K, D)
        S_tilde_t : (B, K, D)
        g_t       : (B, D)

    Output predicted hierarchical representation set:

        R_hat_{t+1} = {Z_hat, S_hat, S_tilde_hat, g_hat}

    Current V1.0 implementation realization
    -----------------------------------------
    The implementation specification intentionally leaves the temporal
    forecasting architecture independent. This implementation realizes the
    required components as follows:

        1. Hierarchical Representation Encoder
           Concatenate all four hierarchy levels and project them into a
           shared latent representation of dimension D.

        2. Shared Temporal Dynamics Model
           Apply a shared residual MLP to the encoded hierarchy. Its output
           is the single shared behavioral-dynamics representation used by
           every prediction head.

        3. Hierarchical Prediction Generator
           Four representation-specific linear heads decode the shared
           dynamics representation into Z, S, S_tilde, and g predictions.

        4. Hierarchical Prediction Constructor
           Reshape the decoded tensors and return the predicted hierarchy.

    The scientific specification does not prescribe these particular neural
    layers; they are an implementation realization of the V1.0 contract.
    """

    def __init__(
        self,
        *,
        num_devices: int,
        num_states: int = 9,
        embedding_dim: int = 64,
        dynamics_hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self._validate_constructor_parameters(
            num_devices=num_devices,
            num_states=num_states,
            embedding_dim=embedding_dim,
            dynamics_hidden_dim=dynamics_hidden_dim,
        )

        self.num_devices = num_devices
        self.num_states = num_states
        self.embedding_dim = embedding_dim
        self.dynamics_hidden_dim = dynamics_hidden_dim

        # The complete hierarchy is encoded jointly. This preserves the
        # information from every semantic level before temporal dynamics are
        # modeled.
        encoder_input_dim = (
            num_devices * embedding_dim
            + num_states * embedding_dim
            + num_states * embedding_dim
            + embedding_dim
        )

        self.hierarchical_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, dynamics_hidden_dim),
            nn.GELU(),
            nn.Linear(dynamics_hidden_dim, embedding_dim),
        )

        # Shared behavioral dynamics representation.
        self.dynamics_model = nn.Sequential(
            nn.Linear(embedding_dim, dynamics_hidden_dim),
            nn.GELU(),
            nn.Linear(dynamics_hidden_dim, embedding_dim),
        )

        # Residual dynamics realization: predict the change in the shared
        # latent dynamics state rather than relearning the complete state.
        self.prediction_residual = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # Representation-specific prediction heads. All heads consume the
        # same shared behavioral-dynamics representation.
        self.device_prediction_head = nn.Linear(
            embedding_dim,
            num_devices * embedding_dim,
        )
        self.state_prediction_head = nn.Linear(
            embedding_dim,
            num_states * embedding_dim,
        )
        self.context_prediction_head = nn.Linear(
            embedding_dim,
            num_states * embedding_dim,
        )
        self.global_prediction_head = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

    @staticmethod
    def _validate_constructor_parameters(
        *,
        num_devices: int,
        num_states: int,
        embedding_dim: int,
        dynamics_hidden_dim: int,
    ) -> None:
        values = {
            "num_devices": num_devices,
            "num_states": num_states,
            "embedding_dim": embedding_dim,
            "dynamics_hidden_dim": dynamics_hidden_dim,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")

    def forward(
        self,
        Z_t: Tensor,
        S_t: Tensor,
        S_tilde_t: Tensor,
        g_t: Tensor,
    ) -> Dict[str, Tensor]:
        self._validate_inputs(Z_t, S_t, S_tilde_t, g_t)

        batch_size = Z_t.shape[0]

        # -------------------------------------------------------------
        # 1. Hierarchical Representation Encoding
        # -------------------------------------------------------------
        hierarchy = torch.cat(
            (
                Z_t.reshape(batch_size, -1),
                S_t.reshape(batch_size, -1),
                S_tilde_t.reshape(batch_size, -1),
                g_t,
            ),
            dim=1,
        )

        encoded_hierarchy = self.hierarchical_encoder(hierarchy)

        # -------------------------------------------------------------
        # 2. Shared Behavioral Dynamics Modeling
        # -------------------------------------------------------------
        dynamics_delta = self.dynamics_model(encoded_hierarchy)
        shared_dynamics = encoded_hierarchy + self.prediction_residual(
            dynamics_delta
        )

        # -------------------------------------------------------------
        # 3. Hierarchical Representation Prediction
        # -------------------------------------------------------------
        Z_hat = self.device_prediction_head(shared_dynamics).reshape(
            batch_size,
            self.num_devices,
            self.embedding_dim,
        )
        S_hat = self.state_prediction_head(shared_dynamics).reshape(
            batch_size,
            self.num_states,
            self.embedding_dim,
        )
        S_tilde_hat = self.context_prediction_head(shared_dynamics).reshape(
            batch_size,
            self.num_states,
            self.embedding_dim,
        )
        g_hat = self.global_prediction_head(shared_dynamics)

        outputs = {
            "Z": Z_hat,
            "S": S_hat,
            "S_tilde": S_tilde_hat,
            "g": g_hat,
        }

        for name, tensor in outputs.items():
            if not torch.isfinite(tensor).all():
                raise RuntimeError(
                    f"HBF produced NaN or infinite values in '{name}'."
                )

        return outputs

    def _validate_inputs(
        self,
        Z_t: Tensor,
        S_t: Tensor,
        S_tilde_t: Tensor,
        g_t: Tensor,
    ) -> None:
        tensors = {
            "Z_t": Z_t,
            "S_t": S_t,
            "S_tilde_t": S_tilde_t,
            "g_t": g_t,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must be a floating-point tensor.")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN or infinite values.")

            if tensor.device != self.hierarchical_encoder[0].weight.device:
                raise ValueError(
                    f"{name} and HBF parameters must be on the same device."
                )
            if tensor.dtype != self.hierarchical_encoder[0].weight.dtype:
                raise TypeError(
                    f"{name} dtype must match HBF parameter dtype."
                )

        if Z_t.ndim != 3:
            raise ValueError("Z_t must have shape (B, N, D).")
        if S_t.ndim != 3:
            raise ValueError("S_t must have shape (B, K, D).")
        if S_tilde_t.ndim != 3:
            raise ValueError("S_tilde_t must have shape (B, K, D).")
        if g_t.ndim != 2:
            raise ValueError("g_t must have shape (B, D).")

        batch_size = Z_t.shape[0]
        if batch_size <= 0:
            raise ValueError("batch size must be greater than zero.")

        if Z_t.shape != (
            batch_size,
            self.num_devices,
            self.embedding_dim,
        ):
            raise ValueError(
                "Z_t shape mismatch: expected "
                f"({batch_size}, {self.num_devices}, {self.embedding_dim}), "
                f"received {tuple(Z_t.shape)}."
            )

        expected_state_shape = (
            batch_size,
            self.num_states,
            self.embedding_dim,
        )
        if tuple(S_t.shape) != expected_state_shape:
            raise ValueError(
                "S_t shape mismatch: expected "
                f"{expected_state_shape}, received {tuple(S_t.shape)}."
            )
        if tuple(S_tilde_t.shape) != expected_state_shape:
            raise ValueError(
                "S_tilde_t shape mismatch: expected "
                f"{expected_state_shape}, received {tuple(S_tilde_t.shape)}."
            )

        expected_global_shape = (batch_size, self.embedding_dim)
        if tuple(g_t.shape) != expected_global_shape:
            raise ValueError(
                "g_t shape mismatch: expected "
                f"{expected_global_shape}, received {tuple(g_t.shape)}."
            )


def hierarchical_representation_shapes(
    outputs: Dict[str, Tensor],
) -> Tuple[Tuple[int, ...], ...]:
    """Return the four predicted hierarchy shapes in canonical order."""
    return (
        tuple(outputs["Z"].shape),
        tuple(outputs["S"].shape),
        tuple(outputs["S_tilde"].shape),
        tuple(outputs["g"].shape),
    )
