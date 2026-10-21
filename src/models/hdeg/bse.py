from __future__ import annotations

from typing import Tuple

import math

import torch
from torch import Tensor, nn


class BehavioralStateEstimator(nn.Module):
    """
    Behavioral State Estimation (BSE) module.

    Scientific role
    ---------------
    Transform device-level behavioral representations

        Z_t ∈ R^(B × N × d)

    into one latent representation for each predefined behavioral state

        S_t ∈ R^(B × K × d).

    The module realizes the implementation mapping defined in the BSE
    implementation specification:

        1. Trainable behavioral query embeddings
        2. Multi-head query-key semantic matching
        3. Compatibility-constrained attention
        4. Value projection
        5. Attention-weighted behavioral evidence aggregation

    The implementation deliberately does not include an output projection
    after head concatenation. Thus, the externally exposed computation
    corresponds directly to the specified value-weighted aggregation while
    still providing a multi-head implementation realization.

    Parameters
    ----------
    num_states:
        Number K of predefined behavioral states.

    embedding_dim:
        Shared representation dimension d of the input and output spaces.

    num_heads:
        Number of attention heads. ``embedding_dim`` must be divisible by
        ``num_heads``.

    Notes
    -----
    The behavioral compatibility matrix is a non-learnable semantic prior
    supplied to ``forward``. It is expected to have shape ``(K, N)`` and
    contain binary values, where ``M[k, i] = 1`` means that device i may
    contribute evidence to behavioral state k.

    The compatibility matrix is not registered as a parameter or buffer
    because it is part of the semantic input specification and may depend
    on the device configuration.
    """

    def __init__(
        self,
        num_states: int,
        embedding_dim: int,
        num_heads: int = 1,
    ) -> None:
        super().__init__()

        if not isinstance(num_states, int):
            raise TypeError("num_states must be an integer.")
        if num_states <= 0:
            raise ValueError("num_states must be greater than zero.")

        if not isinstance(embedding_dim, int):
            raise TypeError("embedding_dim must be an integer.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero.")

        if not isinstance(num_heads, int):
            raise TypeError("num_heads must be an integer.")
        if num_heads <= 0:
            raise ValueError("num_heads must be greater than zero.")

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads. "
                f"Received embedding_dim={embedding_dim}, "
                f"num_heads={num_heads}."
            )

        self.num_states = num_states
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Behavioral Query Parameters.
        #
        # One learnable semantic query for each predefined behavioral state.
        self.behavioral_queries = nn.Parameter(
            torch.empty(num_states, embedding_dim)
        )

        # Semantic Matching Parameters.
        #
        # Biases are deliberately omitted so that these parameters correspond
        # directly to the projection matrices W_Q and W_K in the specification.
        self.query_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )
        self.key_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )

        # Behavioral Evidence Aggregation Parameters.
        self.value_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize trainable parameters using PyTorch's standard scheme."""
        nn.init.xavier_uniform_(self.behavioral_queries)

        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)

    def forward(
        self,
        device_embeddings: Tensor,
        compatibility_mask: Tensor,
    ) -> Tensor:
        """
        Estimate behavioral state representations.

        Parameters
        ----------
        device_embeddings:
            Device behavioral representations Z_t with shape
            ``(B, N, d)``.

        compatibility_mask:
            Non-learnable behavioral compatibility matrix M with shape
            ``(K, N)``. Non-zero entries indicate permitted device-state
            associations.

        Returns
        -------
        Tensor
            Behavioral state representations S_t with shape
            ``(B, K, d)``.

        Raises
        ------
        TypeError
            If either input has an invalid type or the embeddings are not
            floating-point.

        ValueError
            If tensor ranks, dimensions, values, or compatibility constraints
            violate the BSE computational contract.
        """
        self._validate_inputs(
            device_embeddings,
            compatibility_mask,
        )

        # -------------------------------------------------------------
        # Step 1 — Behavioral Query Construction
        # -------------------------------------------------------------
        #
        # Query order is fixed by behavioral state index:
        # q_0, q_1, ..., q_(K-1).
        #
        # Expand only the view used for computation; no duplicated
        # trainable parameters are created.
        batch_size = device_embeddings.size(0)

        queries = self.behavioral_queries.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

        # -------------------------------------------------------------
        # Step 2 — Semantic Evidence Retrieval
        # -------------------------------------------------------------
        #
        # Q: (B, K, d)
        # K: (B, N, d)
        # V: (B, N, d)
        #
        # After splitting heads:
        # Q: (B, H, K, d_h)
        # K: (B, H, N, d_h)
        # V: (B, H, N, d_h)
        q = self.query_projection(queries)
        k = self.key_projection(device_embeddings)
        v = self.value_projection(device_embeddings)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Multi-head semantic relevance:
        #
        # scores[b, h, k, i]
        #     = <Q[b,h,k], K[b,h,i]> / sqrt(d_h)
        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / math.sqrt(self.head_dim)

        # -------------------------------------------------------------
        # Step 3 — Compatibility-Constrained Semantic Attention
        # -------------------------------------------------------------
        #
        # M has shape (K, N).
        # Broadcast it across batch and head dimensions:
        # (1, 1, K, N).
        mask = compatibility_mask.to(
            device=scores.device,
            dtype=torch.bool,
        ).unsqueeze(0).unsqueeze(0)

        masked_scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        attention_weights = torch.softmax(
            masked_scores,
            dim=-1,
        )

        # -------------------------------------------------------------
        # Step 4 — Behavioral Evidence Aggregation
        # -------------------------------------------------------------
        #
        # state_evidence[b,h,k,:]
        #     = sum_i alpha[b,h,k,i] * V[b,h,i,:]
        state_evidence = torch.matmul(
            attention_weights,
            v,
        )

        # -------------------------------------------------------------
        # Step 5 — Behavioral State Formation
        # -------------------------------------------------------------
        #
        # Concatenate the heads:
        # (B, H, K, d_h) -> (B, K, d)
        behavioral_states = self._merge_heads(
            state_evidence
        )

        return behavioral_states

    def _validate_inputs(
        self,
        device_embeddings: Tensor,
        compatibility_mask: Tensor,
    ) -> None:
        """
        Validate the tensor-level input contract.

        Expected:
            device_embeddings: (B, N, d)
            compatibility_mask: (K, N)
        """
        if not isinstance(device_embeddings, Tensor):
            raise TypeError(
                "device_embeddings must be a torch.Tensor."
            )

        if device_embeddings.ndim != 3:
            raise ValueError(
                "device_embeddings must have shape "
                "(batch_size, num_devices, embedding_dim). "
                f"Received {tuple(device_embeddings.shape)}."
            )

        batch_size, num_devices, embedding_dim = device_embeddings.shape

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if num_devices <= 0:
            raise ValueError(
                "num_devices must be greater than zero."
            )

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                "Unexpected embedding dimension. "
                f"Expected d={self.embedding_dim}, "
                f"received d={embedding_dim}."
            )

        if not device_embeddings.is_floating_point():
            raise TypeError(
                "device_embeddings must be a floating-point tensor."
            )

        if not torch.isfinite(device_embeddings).all():
            raise ValueError(
                "device_embeddings contains NaN or infinite values."
            )

        if not isinstance(compatibility_mask, Tensor):
            raise TypeError(
                "compatibility_mask must be a torch.Tensor."
            )

        if compatibility_mask.ndim != 2:
            raise ValueError(
                "compatibility_mask must have shape (K, N). "
                f"Received {tuple(compatibility_mask.shape)}."
            )

        expected_mask_shape = (
            self.num_states,
            num_devices,
        )

        if tuple(compatibility_mask.shape) != expected_mask_shape:
            raise ValueError(
                "Unexpected compatibility_mask shape. "
                f"Expected {expected_mask_shape}, "
                f"received {tuple(compatibility_mask.shape)}."
            )

        if compatibility_mask.device != device_embeddings.device:
            raise ValueError(
                "compatibility_mask and device_embeddings must be "
                "on the same device."
            )

        # The semantic prior is binary. Bool masks are naturally valid;
        # integer/float masks are accepted only when their values are 0/1.
        if compatibility_mask.dtype == torch.bool:
            binary_mask = compatibility_mask
        else:
            if compatibility_mask.is_floating_point():
                if not torch.isfinite(compatibility_mask).all():
                    raise ValueError(
                        "compatibility_mask contains NaN or infinite values."
                    )

            binary_mask = compatibility_mask != 0

            if not torch.all(
                (compatibility_mask == 0)
                | (compatibility_mask == 1)
            ):
                raise ValueError(
                    "compatibility_mask must contain only binary "
                    "values 0 and 1."
                )

        # The mathematical attention definition has a denominator for
        # every behavioral state. Therefore every state must have at least
        # one compatible device.
        compatible_counts = binary_mask.sum(dim=-1)

        if torch.any(compatible_counts == 0):
            invalid_states = torch.nonzero(
                compatible_counts == 0,
                as_tuple=False,
            ).flatten().tolist()

            raise ValueError(
                "Every behavioral state must have at least one "
                "compatible device. "
                f"States without compatible devices: {invalid_states}."
            )

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        Split the last dimension into attention heads.

        Input:
            (B, L, d)

        Output:
            (B, H, L, d_h)
        """
        batch_size, sequence_length, _ = x.shape

        return x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """
        Merge attention heads without an additional output projection.

        Input:
            (B, H, K, d_h)

        Output:
            (B, K, d)
        """
        batch_size, _, num_states, _ = x.shape

        return (
            x.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                num_states,
                self.embedding_dim,
            )
        )