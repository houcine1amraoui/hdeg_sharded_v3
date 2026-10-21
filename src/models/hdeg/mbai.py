from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
from torch import Tensor, nn


HIERARCHY_LEVELS: Tuple[str, str, str, str] = (
    "Z",
    "S",
    "S_tilde",
    "g",
)


class MultiScaleBehavioralAnomalyInference(nn.Module):
    """
    HDEG Multi-scale Behavioral Anomaly Inference (MBAI), V1.0.

    Scientific contract
    -------------------
    MBAI receives two semantically aligned hierarchical behavioral
    representation sets:

        R_(t+1)     = {Z, S, S_tilde, g}
        R_hat_(t+1) = {Z_hat, S_hat, S_tilde_hat, g_hat}

    and produces hierarchical behavioral anomaly evidence together with the
    integrated behavioral anomaly assessment A_(t+1).

    Current V1.0 implementation realization
    -----------------------------------------
    The scientific specification intentionally leaves the discrepancy,
    evidence interpretation, and fusion functions architecture-independent.
    This implementation therefore uses the minimal deterministic realization:

        1. Behavioral Discrepancy Constructor
           e^l = (R^l - R_hat^l)^2

        2. Behavioral Evidence Interpreter
           E^l = mean(e^l) over all non-batch representation dimensions

        3. Hierarchical Evidence Fusion
           A = sum_l w_l E^l / sum_l w_l

    The four evidence values are therefore independent scalar anomaly
    evidence measures, one for each semantic abstraction level.

    The fixed fusion weights are configuration values, not learnable model
    parameters. Equal weighting is the default because V1.0 does not specify
    scientifically justified learned fusion weights.

    IMPORTANT V1.0 LIMITATION
    --------------------------
    The implementation specification describes MBAI as containing learnable
    discrepancy/evidence/fusion/assessment parameters and states that they
    are jointly optimized through L_HDEG. However, the specified
    L_HDEG objective is defined only from the four hierarchical forecasting
    losses and does not provide a supervisory path for MBAI parameters.

    Consequently, this implementation intentionally contains NO learnable
    MBAI parameters and NO MBAI training objective. The module is an
    inference-only deterministic realization until the scientific
    specification explicitly resolves that optimization interface.

    Scientific artifacts exposed by ``forward``
    --------------------------------------------
        E_Z, E_S, E_S_tilde, E_G : scalar anomaly evidence, shape (B,)
        A                         : integrated anomaly assessment, shape (B,)

    The discrepancy tensors remain internal computational objects.
    """

    def __init__(
        self,
        *,
        fusion_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()

        weights = self._validate_and_prepare_weights(fusion_weights)

        # Registered as a buffer rather than a Parameter so that the module
        # remains device-aware while remaining strictly non-trainable.
        self.register_buffer("fusion_weights", weights, persistent=True)

    @staticmethod
    def _validate_and_prepare_weights(
        fusion_weights: Mapping[str, float] | None,
    ) -> Tensor:
        if fusion_weights is None:
            fusion_weights = {
                "Z": 1.0,
                "S": 1.0,
                "S_tilde": 1.0,
                "g": 1.0,
            }

        missing = [level for level in HIERARCHY_LEVELS if level not in fusion_weights]
        extra = [level for level in fusion_weights if level not in HIERARCHY_LEVELS]
        if missing:
            raise ValueError(f"Missing MBAI fusion weights: {missing}")
        if extra:
            raise ValueError(f"Unknown MBAI fusion-weight keys: {extra}")

        values = []
        for level in HIERARCHY_LEVELS:
            value = fusion_weights[level]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"MBAI fusion weight '{level}' must be numeric.")
            if not torch.isfinite(torch.tensor(float(value))):
                raise ValueError(f"MBAI fusion weight '{level}' must be finite.")
            if value < 0:
                raise ValueError(f"MBAI fusion weight '{level}' must be non-negative.")
            values.append(float(value))

        weights = torch.tensor(values, dtype=torch.float32)
        if float(weights.sum()) <= 0.0:
            raise ValueError("At least one MBAI fusion weight must be positive.")
        return weights

    @property
    def has_learnable_parameters(self) -> bool:
        """Explicitly expose the current V1.0 inference-only status."""
        return any(parameter.requires_grad for parameter in self.parameters())

    def forward(
        self,
        observed: Mapping[str, Tensor],
        predicted: Mapping[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Infer multi-scale anomaly evidence from observed/predicted hierarchies.

        Parameters
        ----------
        observed:
            Mapping with keys ``Z``, ``S``, ``S_tilde``, and ``g``.
        predicted:
            Mapping with semantically corresponding keys ``Z``, ``S``,
            ``S_tilde``, and ``g``.

        Returns
        -------
        dict
            ``E_Z``, ``E_S``, ``E_S_tilde``, ``E_G``, and ``A``; all have
            shape ``(B,)``.
        """
        self._validate_hierarchies(observed, predicted)

        # -------------------------------------------------------------
        # 1. Hierarchical Behavioral Discrepancy Construction
        # -------------------------------------------------------------
        discrepancies = {
            level: (observed[level] - predicted[level]).pow(2)
            for level in HIERARCHY_LEVELS
        }

        # -------------------------------------------------------------
        # 2. Multi-scale Behavioral Evidence Interpretation
        # -------------------------------------------------------------
        evidence = {
            level: self._aggregate_discrepancy(discrepancies[level])
            for level in HIERARCHY_LEVELS
        }

        # -------------------------------------------------------------
        # 3. Hierarchical Evidence Integration
        # -------------------------------------------------------------
        weights = self.fusion_weights.to(
            device=evidence["Z"].device,
            dtype=evidence["Z"].dtype,
        )
        stacked_evidence = torch.stack(
            [evidence[level] for level in HIERARCHY_LEVELS],
            dim=1,
        )
        integrated_evidence = (
            stacked_evidence * weights.unsqueeze(0)
        ).sum(dim=1) / weights.sum()

        # -------------------------------------------------------------
        # 4. Behavioral Anomaly Assessment
        # -------------------------------------------------------------
        assessment = integrated_evidence

        outputs = {
            "E_Z": evidence["Z"],
            "E_S": evidence["S"],
            "E_S_tilde": evidence["S_tilde"],
            "E_G": evidence["g"],
            "A": assessment,
        }

        for name, tensor in outputs.items():
            if not torch.isfinite(tensor).all():
                raise RuntimeError(
                    f"MBAI produced NaN or infinite values in '{name}'."
                )

        return outputs

    @staticmethod
    def _aggregate_discrepancy(discrepancy: Tensor) -> Tensor:
        """Convert a residual representation into scalar evidence per sample."""
        if discrepancy.ndim < 2:
            raise ValueError("MBAI discrepancy must have at least two dimensions.")
        return discrepancy.reshape(discrepancy.shape[0], -1).mean(dim=1)

    def _validate_hierarchies(
        self,
        observed: Mapping[str, Tensor],
        predicted: Mapping[str, Tensor],
    ) -> None:
        if not isinstance(observed, Mapping):
            raise TypeError("observed must be a mapping of hierarchy tensors.")
        if not isinstance(predicted, Mapping):
            raise TypeError("predicted must be a mapping of hierarchy tensors.")

        observed_keys = set(observed.keys())
        predicted_keys = set(predicted.keys())
        expected_keys = set(HIERARCHY_LEVELS)

        if observed_keys != expected_keys:
            raise ValueError(
                "Observed hierarchy keys must be exactly "
                f"{HIERARCHY_LEVELS}; received {tuple(observed.keys())}."
            )
        if predicted_keys != expected_keys:
            raise ValueError(
                "Predicted hierarchy keys must be exactly "
                f"{HIERARCHY_LEVELS}; received {tuple(predicted.keys())}."
            )

        for level in HIERARCHY_LEVELS:
            obs = observed[level]
            pred = predicted[level]

            if not isinstance(obs, Tensor):
                raise TypeError(f"observed['{level}'] must be a torch.Tensor.")
            if not isinstance(pred, Tensor):
                raise TypeError(f"predicted['{level}'] must be a torch.Tensor.")

            if not obs.is_floating_point() or not pred.is_floating_point():
                raise TypeError(
                    f"MBAI hierarchy level '{level}' must use floating-point tensors."
                )
            if obs.dtype != pred.dtype:
                raise TypeError(
                    f"Observed/predicted dtype mismatch at '{level}': "
                    f"{obs.dtype} != {pred.dtype}."
                )
            if obs.device != pred.device:
                raise ValueError(
                    f"Observed/predicted device mismatch at '{level}': "
                    f"{obs.device} != {pred.device}."
                )
            if not torch.isfinite(obs).all():
                raise ValueError(
                    f"observed['{level}'] contains NaN or infinite values."
                )
            if not torch.isfinite(pred).all():
                raise ValueError(
                    f"predicted['{level}'] contains NaN or infinite values."
                )
            if obs.ndim not in (2, 3):
                raise ValueError(
                    f"Hierarchy level '{level}' must have shape (B, D) or (B, N, D)."
                )
            if obs.shape != pred.shape:
                raise ValueError(
                    f"Observed/predicted shape mismatch at '{level}': "
                    f"{tuple(obs.shape)} != {tuple(pred.shape)}."
                )
            if obs.shape[0] <= 0:
                raise ValueError("MBAI batch size must be greater than zero.")

        batch_sizes = {observed[level].shape[0] for level in HIERARCHY_LEVELS}
        if len(batch_sizes) != 1:
            raise ValueError(
                "All observed hierarchy levels must have the same batch size."
            )


def hierarchical_anomaly_evidence_shapes(
    outputs: Mapping[str, Tensor],
) -> Tuple[Tuple[int, ...], ...]:
    """Return the four evidence shapes in canonical hierarchy order."""
    return (
        tuple(outputs["E_Z"].shape),
        tuple(outputs["E_S"].shape),
        tuple(outputs["E_S_tilde"].shape),
        tuple(outputs["E_G"].shape),
    )
