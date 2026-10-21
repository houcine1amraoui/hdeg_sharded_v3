from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import torch
from torch import Tensor, nn


HIERARCHY_LEVELS: Tuple[str, ...] = ("Z", "S", "S_tilde", "g")


@dataclass(frozen=True)
class HierarchicalForecastingObjectives:
    """Four semantic-level forecasting objectives and their integrated value."""

    L_Z: Tensor
    L_S: Tensor
    L_S_tilde: Tensor
    L_G: Tensor
    L_HDEG: Tensor

    def as_dict(self) -> Dict[str, Tensor]:
        return {
            "L_Z": self.L_Z,
            "L_S": self.L_S,
            "L_S_tilde": self.L_S_tilde,
            "L_G": self.L_G,
            "L_HDEG": self.L_HDEG,
        }


class ModelOptimization(nn.Module):
    """
    HDEG Model Optimization (MO).

    Scientific contract
    -------------------
    Inputs:
        observed hierarchical representation set R_{t+1}:
            Z, S, S_tilde, g
        forecasted hierarchical representation set R_hat_{t+1}:
            Z_hat, S_hat, S_tilde_hat, g_hat

    Outputs:
        L_HDEG and the four level-specific forecasting objectives:
            L_Z, L_S, L_S_tilde, L_G

    V1.0 specifies the semantic operation as

        (R_{t+1}, R_hat_{t+1})
            -> {L_Z, L_S, L_S_tilde, L_G}
            -> L_HDEG

    and defines

        L_HDEG = lambda_Z L_Z + lambda_S L_S
                  + lambda_S_tilde L_S_tilde + lambda_G L_G.

    The current V1.0 documents ``ell(.)`` as a differentiable forecasting
    loss and ``d(.)`` as a representation discrepancy, but does not prescribe
    their concrete numerical form. The executable realization therefore uses
    elementwise mean-squared discrepancy/loss. This is an implementation
    choice, not a scientific architecture change, and is isolated here so it
    can be replaced without changing the four-level supervisory contract.

    Important autograd property
    ----------------------------
    This module never detaches, converts tensors to Python scalars, or enters
    no_grad mode. L_HDEG therefore remains connected to forecast tensors and
    can backpropagate through HBF and, when the complete model is executed
    in-memory, through the preceding trainable representation modules.

    Scientific artifacts
    --------------------
    The returned four level losses and L_HDEG are the supervisory artifacts.
    No optimizer is owned by this module; optimizer selection/update belongs
    to the training/orchestration layer.
    """

    def __init__(
        self,
        *,
        lambda_z: float = 1.0,
        lambda_s: float = 1.0,
        lambda_s_tilde: float = 1.0,
        lambda_g: float = 1.0,
    ) -> None:
        super().__init__()
        self._validate_weights(
            lambda_z=lambda_z,
            lambda_s=lambda_s,
            lambda_s_tilde=lambda_s_tilde,
            lambda_g=lambda_g,
        )

        # Registered as buffers so they follow .to(device)/.to(dtype) while
        # remaining non-learnable, exactly as required by the MO contract.
        self.register_buffer("lambda_Z", torch.tensor(float(lambda_z)))
        self.register_buffer("lambda_S", torch.tensor(float(lambda_s)))
        self.register_buffer(
            "lambda_S_tilde", torch.tensor(float(lambda_s_tilde))
        )
        self.register_buffer("lambda_G", torch.tensor(float(lambda_g)))

    @staticmethod
    def _validate_weights(**weights: float) -> None:
        for name, value in weights.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real-valued scalar.")
            if not torch.isfinite(torch.tensor(float(value))):
                raise ValueError(f"{name} must be finite.")
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative.")

        if all(float(value) == 0.0 for value in weights.values()):
            raise ValueError("At least one hierarchical loss weight must be positive.")

    @staticmethod
    def _validate_hierarchy(
        observed: Mapping[str, Tensor],
        predicted: Mapping[str, Tensor],
    ) -> None:
        if not isinstance(observed, Mapping):
            raise TypeError("observed must be a mapping containing Z, S, S_tilde, and g.")
        if not isinstance(predicted, Mapping):
            raise TypeError("predicted must be a mapping containing Z, S, S_tilde, and g.")

        missing_observed = [k for k in HIERARCHY_LEVELS if k not in observed]
        missing_predicted = [k for k in HIERARCHY_LEVELS if k not in predicted]
        if missing_observed:
            raise KeyError(f"observed hierarchy is missing: {missing_observed}")
        if missing_predicted:
            raise KeyError(f"predicted hierarchy is missing: {missing_predicted}")

        batch_size = None
        reference_device = None
        reference_dtype = None

        for level in HIERARCHY_LEVELS:
            target = observed[level]
            forecast = predicted[level]

            if not isinstance(target, Tensor):
                raise TypeError(f"observed['{level}'] must be a torch.Tensor.")
            if not isinstance(forecast, Tensor):
                raise TypeError(f"predicted['{level}'] must be a torch.Tensor.")

            if not target.is_floating_point() or not forecast.is_floating_point():
                raise TypeError(f"'{level}' tensors must be floating-point tensors.")
            if target.dtype != forecast.dtype:
                raise TypeError(
                    f"'{level}' dtype mismatch: observed={target.dtype}, predicted={forecast.dtype}."
                )
            if target.device != forecast.device:
                raise ValueError(
                    f"'{level}' device mismatch: observed={target.device}, predicted={forecast.device}."
                )
            if target.shape != forecast.shape:
                raise ValueError(
                    f"'{level}' shape mismatch: observed={tuple(target.shape)}, "
                    f"predicted={tuple(forecast.shape)}."
                )
            if target.ndim not in (2, 3):
                raise ValueError(
                    f"'{level}' must have rank 2 or 3; got rank {target.ndim}."
                )
            if target.shape[0] <= 0:
                raise ValueError(f"'{level}' must have a positive batch dimension.")
            if not torch.isfinite(target).all():
                raise ValueError(f"observed['{level}'] contains NaN or Inf.")
            if not torch.isfinite(forecast).all():
                raise ValueError(f"predicted['{level}'] contains NaN or Inf.")

            if batch_size is None:
                batch_size = target.shape[0]
                reference_device = target.device
                reference_dtype = target.dtype
            elif target.shape[0] != batch_size:
                raise ValueError("All hierarchy levels must have the same batch size.")
            elif target.device != reference_device or target.dtype != reference_dtype:
                raise ValueError("All hierarchy levels must share device and dtype.")

    @staticmethod
    def _mse(observed: Tensor, predicted: Tensor) -> Tensor:
        """Differentiable mean squared representation discrepancy."""
        residual = observed - predicted
        return torch.mean(residual * residual)

    def forward(
        self,
        observed: Mapping[str, Tensor],
        predicted: Mapping[str, Tensor],
    ) -> HierarchicalForecastingObjectives:
        self._validate_hierarchy(observed, predicted)

        # Each loss is computed only from its semantically corresponding pair.
        L_Z = self._mse(observed["Z"], predicted["Z"])
        L_S = self._mse(observed["S"], predicted["S"])
        L_S_tilde = self._mse(observed["S_tilde"], predicted["S_tilde"])
        L_G = self._mse(observed["g"], predicted["g"])

        # Keep this as tensor arithmetic. Do not detach or convert to float:
        # L_HDEG must retain the complete autograd graph.
        L_HDEG = (
            self.lambda_Z * L_Z
            + self.lambda_S * L_S
            + self.lambda_S_tilde * L_S_tilde
            + self.lambda_G * L_G
        )

        if not torch.isfinite(L_HDEG):
            raise RuntimeError("MO produced a NaN or infinite hierarchical loss.")

        return HierarchicalForecastingObjectives(
            L_Z=L_Z,
            L_S=L_S,
            L_S_tilde=L_S_tilde,
            L_G=L_G,
            L_HDEG=L_HDEG,
        )

    def loss_dict(
        self,
        observed: Mapping[str, Tensor],
        predicted: Mapping[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Return the scientific loss artifacts as a dictionary."""
        return self(observed, predicted).as_dict()
