from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


class EcosystemBehavioralRepresentationLearner(nn.Module):
    """
    HDEG Ecosystem Behavioral Representation Learning (EBRL).

    Scientific contract
    -------------------
    Input:
        contextualized behavioral representations
            S_tilde in R^(B x K x d)

    Output:
        ecosystem behavioral representation
            G_t in R^(B x d)

    Current V1.0 implementation mapping
    ------------------------------------
    The implementation specification maps EBRL to an attention-based
    ecosystem representation learner:

        1. Behavioral Representation Projection
        2. Behavioral Importance Estimation
        3. Holistic Behavioral Integration
        4. Global Representation Refinement / final artifact construction

    The research formulation specifies:

        u_k = v^T tanh(W_g s_tilde_k)

        gamma_k = softmax_k(u)

        g_t = sum_k gamma_k s_tilde_k

    Therefore W_g is used for importance estimation, while the
    attention-weighted aggregation is performed over the contextualized
    behavioral representations themselves. No additional post-aggregation
    learnable transformation is introduced because the V1.0 implementation
    mapping does not specify one.

    Scientific artifact exposure
    -----------------------------
    ``forward`` returns only the ecosystem representation G_t. Attention
    scores, coefficients, projected representations, and weighted states
    remain internal computational objects.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        num_states: int = 9,
    ) -> None:
        super().__init__()

        if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int):
            raise TypeError("embedding_dim must be an integer.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero.")

        if isinstance(num_states, bool) or not isinstance(num_states, int):
            raise TypeError("num_states must be an integer.")
        if num_states <= 0:
            raise ValueError("num_states must be greater than zero.")

        self.embedding_dim = embedding_dim
        self.num_states = num_states

        # W_g in the research formulation. Bias is intentionally omitted:
        # the specification gives a matrix multiplication W_g s_tilde.
        self.behavior_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )

        # v in u_k = v^T tanh(W_g s_tilde_k).
        self.attention_vector = nn.Parameter(
            torch.empty(embedding_dim)
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Use standard PyTorch initialization; initialization is not
        specified scientifically by V1.0 and is therefore an engineering
        choice only.
        """
        nn.init.xavier_uniform_(self.behavior_projection.weight)
        nn.init.xavier_uniform_(self.attention_vector.unsqueeze(0))

    def forward(self, contextualized_states: Tensor) -> Tensor:
        """
        Construct the ecosystem behavioral representation.

        Parameters
        ----------
        contextualized_states:
            Tensor with shape ``(B, K, d)`` and floating-point dtype.

        Returns
        -------
        Tensor
            Ecosystem representation with shape ``(B, d)``.
        """
        self._validate_input(contextualized_states)

        # ---------------------------------------------------------------
        # 1. Behavioral Representation Projection
        # ---------------------------------------------------------------
        # projected[b, k, :] = W_g s_tilde[b, k, :]
        projected = self.behavior_projection(contextualized_states)

        # ---------------------------------------------------------------
        # 2. Behavioral Importance Estimation
        # ---------------------------------------------------------------
        # u[b, k] = v^T tanh(W_g s_tilde[b, k])
        importance_scores = torch.tanh(projected).matmul(
            self.attention_vector
        )

        # gamma[b, k] = softmax_k(u[b, :])
        importance = torch.softmax(
            importance_scores,
            dim=1,
        )

        # ---------------------------------------------------------------
        # 3. Holistic Behavioral Integration
        # ---------------------------------------------------------------
        # g[b, :] = sum_k gamma[b,k] * s_tilde[b,k,:]
        integrated = torch.sum(
            importance.unsqueeze(-1) * contextualized_states,
            dim=1,
        )

        # ---------------------------------------------------------------
        # 4. Global Representation Refinement / artifact construction
        # ---------------------------------------------------------------
        # The V1.0 implementation mapping defines this stage as the
        # construction of the final ecosystem representation. No additional
        # learnable refinement transform is specified in the current mapping.
        ecosystem_representation = integrated

        if not torch.isfinite(ecosystem_representation).all():
            raise RuntimeError(
                "EBRL produced NaN or infinite ecosystem representations."
            )

        return ecosystem_representation

    def _validate_input(self, contextualized_states: Tensor) -> None:
        if not isinstance(contextualized_states, Tensor):
            raise TypeError(
                "contextualized_states must be a torch.Tensor."
            )

        if contextualized_states.ndim != 3:
            raise ValueError(
                "contextualized_states must have shape "
                "(batch_size, num_states, embedding_dim). "
                f"Received {tuple(contextualized_states.shape)}."
            )

        batch_size, num_states, embedding_dim = contextualized_states.shape

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if num_states != self.num_states:
            raise ValueError(
                "Unexpected behavioral-state count. "
                f"Expected K={self.num_states}, received K={num_states}."
            )

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                "Unexpected embedding dimension. "
                f"Expected d={self.embedding_dim}, received d={embedding_dim}."
            )

        if not contextualized_states.is_floating_point():
            raise TypeError(
                "contextualized_states must be a floating-point tensor."
            )

        if not torch.isfinite(contextualized_states).all():
            raise ValueError(
                "contextualized_states contains NaN or infinite values."
            )

        parameter_device = self.behavior_projection.weight.device
        parameter_dtype = self.behavior_projection.weight.dtype

        if contextualized_states.device != parameter_device:
            raise ValueError(
                "contextualized_states and EBRL parameters must reside on "
                "the same device. "
                f"Input device={contextualized_states.device}, "
                f"parameter device={parameter_device}."
            )

        if contextualized_states.dtype != parameter_dtype:
            raise TypeError(
                "contextualized_states dtype must match EBRL parameter dtype. "
                f"Input dtype={contextualized_states.dtype}, "
                f"parameter dtype={parameter_dtype}."
            )
