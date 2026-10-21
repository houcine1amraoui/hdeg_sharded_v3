from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Frozen HDEG Behavioral Interaction Graph (BIG)
# ---------------------------------------------------------------------------
#
# Semantic edge convention:
#
#     (source, target) means:
#         information from the source behavioral state may contribute
#         contextual information to the target behavioral state.
#
# The seven edges were frozen by the HDEG V1.0 implementation discussion:
#
#     0 -> 1
#     1 -> 4
#     1 -> 5
#     1 -> 7
#     4 -> 3
#     7 -> 3
#     8 -> 6
#
# State indices:
#     0 Access / Entry
#     1 Occupancy / Presence
#     2 Physical Disturbance
#     3 Thermal Environment
#     4 Switch-Controlled Appliance Activity
#     5 Lighting Activity
#     6 Air Quality
#     7 Water Heating
#     8 Humidification
#
# This constant is an implementation representation of the already-frozen
# semantic topology. It is not learned.
FROZEN_BIG_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 4),
    (1, 5),
    (1, 7),
    (4, 3),
    (7, 3),
    (8, 6),
)


def build_frozen_big_edge_index(
    *,
    num_states: int = 9,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Build the frozen HDEG Behavioral Interaction Graph as a PyTorch
    Geometric-style edge_index tensor.

    Convention
    ----------
    edge_index[0] = source node
    edge_index[1] = target node

    Thus an edge ``(i, j)`` means:

        behavioral state i -> behavioral state j

    and state j is permitted to receive contextual information from state i.

    Returns
    -------
    Tensor
        Long tensor with shape ``(2, 7)``.
    """
    if isinstance(num_states, bool) or not isinstance(num_states, int):
        raise TypeError("num_states must be an integer.")

    if num_states != 9:
        raise ValueError(
            "The frozen CU BIG contains exactly 9 behavioral states. "
            f"Received num_states={num_states}."
        )

    edge_index = torch.tensor(
        FROZEN_BIG_EDGES,
        dtype=torch.long,
        device=device,
    ).t().contiguous()

    return edge_index


class BehavioralInteractionLearner(nn.Module):
    """
    HDEG Behavioral Interaction Learning (BIL).

    Scientific role
    ---------------
    Transform independently estimated behavioral-state representations

        S_t in R^(B x K x d)

    into contextualized behavioral representations

        S_tilde in R^(B x K x d)

    by propagating contextual information exclusively along the
    predefined Behavioral Interaction Graph (BIG).

    Implementation mapping
    ----------------------
    1. Behavioral State Projection:
         Linear projection W.

    2. Interaction-Aware Information Propagation:
         Single-head Graph Attention Network (GAT)-style attention over
         the fixed directed BIG.

    3. Contextual Evidence Aggregation:
         Attention-weighted aggregation of source-state messages.

    4. Behavioral Representation Refinement:
         Residual integration of the aggregated contextual evidence with
         the original behavioral-state representations.

    Important semantic convention
    ------------------------------
    For an edge

        source -> target

    information flows from source to target.

    Therefore, for the frozen edge

        1 -> 5

    Occupancy contributes contextual information to Lighting Activity;
    the reverse direction is not created.

    Self-loops
    ----------
    No semantic self-loops are used. The residual path preserves each
    state's intrinsic representation instead of representing self-
    preservation as a BIG edge.

    Notes
    -----
    The research manuscript defines the GAT interaction score as

        e_{k,j} =
            LeakyReLU(
                a^T [W s_k || W s_j]
            )

    and normalizes over the neighbors contributing to target k.

    The implementation specification additionally requires residual
    integration during behavioral representation refinement.

    The graph topology is fixed; only projection, attention, and
    refinement parameters are learned.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        num_states: int = 9,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_index: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        self._validate_constructor_parameters(
            embedding_dim=embedding_dim,
            num_states=num_states,
            negative_slope=negative_slope,
            dropout=dropout,
        )

        self.embedding_dim = embedding_dim
        self.num_states = num_states
        self.negative_slope = float(negative_slope)
        self.dropout = float(dropout)

        # ---------------------------------------------------------------
        # Behavioral State Projection
        # ---------------------------------------------------------------
        #
        # Bias is omitted so that W corresponds directly to the
        # projection matrix in the manuscript equation.
        self.state_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )

        # ---------------------------------------------------------------
        # Interaction Modeling
        # ---------------------------------------------------------------
        #
        # One attention vector a implements:
        #
        #   a^T [h_target || h_source]
        #
        # where h = W s.
        self.attention_vector = nn.Parameter(
            torch.empty(2 * embedding_dim)
        )

        # ---------------------------------------------------------------
        # Representation Refinement
        # ---------------------------------------------------------------
        #
        # Context is transformed before being added residually to the
        # original behavioral state representation:
        #
        #   S_tilde = S + R(C)
        #
        # This keeps the output in the original d-dimensional semantic
        # representation space and explicitly preserves intrinsic state
        # information.
        self.context_refinement = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=True,
        )

        # BIG is a fixed semantic structure, not a trainable parameter.
        if edge_index is None:
            edge_index = build_frozen_big_edge_index(
                num_states=num_states
            )

        validated_edge_index = self._validate_edge_index(
            edge_index,
            num_states=num_states,
        )

        self.register_buffer(
            "edge_index",
            validated_edge_index,
            persistent=True,
        )

        # A dense target/source mask is useful for the fixed K=9 graph and
        # makes the attention normalization explicit and easy to verify.
        incoming_mask = torch.zeros(
            num_states,
            num_states,
            dtype=torch.bool,
        )

        source_nodes = validated_edge_index[0]
        target_nodes = validated_edge_index[1]

        incoming_mask[target_nodes, source_nodes] = True

        self.register_buffer(
            "incoming_mask",
            incoming_mask,
            persistent=True,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize all trainable parameters deterministically by seed."""
        nn.init.xavier_uniform_(self.state_projection.weight)
        nn.init.xavier_uniform_(self.attention_vector.unsqueeze(0))
        nn.init.xavier_uniform_(self.context_refinement.weight)
        nn.init.zeros_(self.context_refinement.bias)

    @staticmethod
    def _validate_constructor_parameters(
        *,
        embedding_dim: int,
        num_states: int,
        negative_slope: float,
        dropout: float,
    ) -> None:
        if isinstance(embedding_dim, bool) or not isinstance(
            embedding_dim, int
        ):
            raise TypeError("embedding_dim must be an integer.")

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        if isinstance(num_states, bool) or not isinstance(
            num_states, int
        ):
            raise TypeError("num_states must be an integer.")

        if num_states != 9:
            raise ValueError(
                "The frozen HDEG CU BIG contains exactly 9 "
                f"behavioral states. Received {num_states}."
            )

        if not isinstance(negative_slope, (float, int)) or isinstance(
            negative_slope, bool
        ):
            raise TypeError(
                "negative_slope must be a floating-point value."
            )

        if float(negative_slope) < 0.0:
            raise ValueError(
                "negative_slope must be non-negative."
            )

        if not isinstance(dropout, (float, int)) or isinstance(
            dropout, bool
        ):
            raise TypeError(
                "dropout must be a floating-point value."
            )

        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(
                "dropout must satisfy 0.0 <= dropout < 1.0."
            )

    @staticmethod
    def _validate_edge_index(
        edge_index: Tensor,
        *,
        num_states: int,
    ) -> Tensor:
        """
        Validate the semantic BIG topology.

        The frozen implementation accepts the exact seven-edge topology
        only. This prevents accidental use of a reversed, symmetrized,
        self-looped, or otherwise modified graph.
        """
        if not isinstance(edge_index, Tensor):
            raise TypeError(
                "edge_index must be a torch.Tensor."
            )

        if edge_index.ndim != 2 or tuple(edge_index.shape) != (
            2,
            len(FROZEN_BIG_EDGES),
        ):
            raise ValueError(
                "BIG edge_index must have shape "
                f"(2, {len(FROZEN_BIG_EDGES)}). "
                f"Received {tuple(edge_index.shape)}."
            )

        if edge_index.dtype != torch.long:
            raise TypeError(
                "BIG edge_index must have dtype torch.long."
            )

        if edge_index.device.type == "meta":
            raise ValueError(
                "BIG edge_index must reside on a real device."
            )

        expected = torch.tensor(
            FROZEN_BIG_EDGES,
            dtype=torch.long,
            device=edge_index.device,
        ).t().contiguous()

        if not torch.equal(edge_index, expected):
            raise ValueError(
                "edge_index does not match the frozen HDEG BIG topology. "
                "The semantic BIG must not be reversed, symmetrized, "
                "self-looped, or otherwise modified."
            )

        if torch.any(edge_index < 0) or torch.any(
            edge_index >= num_states
        ):
            raise ValueError(
                "BIG edge_index contains a node outside the valid "
                f"range [0, {num_states - 1}]."
            )

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        if torch.any(source_nodes == target_nodes):
            raise ValueError(
                "BIG must not contain semantic self-loops."
            )

        return edge_index.clone()

    def forward(
        self,
        behavioral_states: Tensor,
    ) -> Tensor:
        """
        Contextualize behavioral-state representations.

        Parameters
        ----------
        behavioral_states:
            Behavioral state representation tensor

                S_t: (B, K, d)

            where K=9 for the frozen CU configuration.

        Returns
        -------
        Tensor
            Contextualized behavioral representation tensor

                S_tilde: (B, K, d)

        Raises
        ------
        TypeError
            If the input type or dtype violates the interface.

        ValueError
            If tensor rank, dimensions, device placement, or values
            violate the BIL computational contract.
        """
        self._validate_input(behavioral_states)

        # ---------------------------------------------------------------
        # Step 1 — Behavioral State Projection
        # ---------------------------------------------------------------
        #
        # S: (B, K, d)
        # H: (B, K, d)
        projected = self.state_projection(
            behavioral_states
        )

        # ---------------------------------------------------------------
        # Step 2 — Interaction-Aware Information Propagation
        # ---------------------------------------------------------------
        #
        # For every semantic edge source -> target:
        #
        #   e[target, source]
        #       = LeakyReLU(
        #           a^T [H_target || H_source]
        #         )
        #
        source_nodes = self.edge_index[0]
        target_nodes = self.edge_index[1]

        source_repr = projected[:, source_nodes, :]
        target_repr = projected[:, target_nodes, :]

        edge_features = torch.cat(
            [
                target_repr,
                source_repr,
            ],
            dim=-1,
        )

        edge_scores = F.leaky_relu(
            torch.sum(
                edge_features
                * self.attention_vector.view(1, 1, -1),
                dim=-1,
            ),
            negative_slope=self.negative_slope,
        )

        # ---------------------------------------------------------------
        # Step 3 — Contextual Evidence Aggregation
        # ---------------------------------------------------------------
        #
        # Build A_score[b, target, source].
        #
        # Only semantic BIG edges receive finite scores. All other
        # source-target pairs remain masked.
        batch_size = behavioral_states.shape[0]

        score_matrix = torch.full(
            (
                batch_size,
                self.num_states,
                self.num_states,
            ),
            float("-inf"),
            dtype=projected.dtype,
            device=projected.device,
        )

        score_matrix[
            :,
            target_nodes,
            source_nodes,
        ] = edge_scores

        valid_targets = self.incoming_mask.any(dim=-1)

        # Softmax over source nodes for each target node.
        attention = torch.softmax(
            score_matrix,
            dim=-1,
        )

        # Nodes with no incoming BIG edges have no contextual evidence.
        # PyTorch's softmax over an all -inf row yields NaN; replace those
        # rows explicitly with zero attention.
        attention = torch.where(
            valid_targets.view(1, -1, 1),
            attention,
            torch.zeros_like(attention),
        )

        if self.training and self.dropout > 0.0:
            attention = F.dropout(
                attention,
                p=self.dropout,
                training=True,
            )

        # Context[b, target, d] =
        #     sum_source attention[b,target,source] * H[b,source,d]
        contextual_information = torch.matmul(
            attention,
            projected,
        )

        # ---------------------------------------------------------------
        # Step 4 — Behavioral Representation Refinement
        # ---------------------------------------------------------------
        #
        # Residual integration:
        #
        #   S_tilde = S + R(C)
        #
        # This preserves the original state identity and guarantees that
        # states without incoming BIG edges still retain their intrinsic
        # representation.
        refined_context = self.context_refinement(
            contextual_information
        )

        contextualized_states = (
            behavioral_states + refined_context
        )

        if not torch.isfinite(
            contextualized_states
        ).all():
            raise RuntimeError(
                "BIL produced NaN or infinite contextualized "
                "behavioral representations."
            )

        return contextualized_states

    def _validate_input(
        self,
        behavioral_states: Tensor,
    ) -> None:
        """Validate the BIL tensor-level input contract."""
        if not isinstance(behavioral_states, Tensor):
            raise TypeError(
                "behavioral_states must be a torch.Tensor."
            )

        if behavioral_states.ndim != 3:
            raise ValueError(
                "behavioral_states must have shape "
                "(batch_size, num_states, embedding_dim). "
                f"Received {tuple(behavioral_states.shape)}."
            )

        batch_size, num_states, embedding_dim = (
            behavioral_states.shape
        )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if num_states != self.num_states:
            raise ValueError(
                "Unexpected behavioral-state count. "
                f"Expected K={self.num_states}, "
                f"received K={num_states}."
            )

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                "Unexpected embedding dimension. "
                f"Expected d={self.embedding_dim}, "
                f"received d={embedding_dim}."
            )

        if not behavioral_states.is_floating_point():
            raise TypeError(
                "behavioral_states must be a floating-point tensor."
            )

        if not torch.isfinite(
            behavioral_states
        ).all():
            raise ValueError(
                "behavioral_states contains NaN or infinite values."
            )

        if behavioral_states.device != self.edge_index.device:
            raise ValueError(
                "behavioral_states and BIG edge_index must be "
                "on the same device."
            )

    @torch.no_grad()
    def incoming_neighbors(
        self,
    ) -> Tuple[Tuple[int, ...], ...]:
        """
        Return the frozen incoming-neighbor set for every target state.

        The result is indexed by target behavioral-state index.
        """
        source_nodes = self.edge_index[0].tolist()
        target_nodes = self.edge_index[1].tolist()

        neighbors = [[] for _ in range(self.num_states)]

        for source, target in zip(
            source_nodes,
            target_nodes,
        ):
            neighbors[target].append(source)

        return tuple(
            tuple(state_neighbors)
            for state_neighbors in neighbors
        )


__all__ = [
    "FROZEN_BIG_EDGES",
    "BehavioralInteractionLearner",
    "build_frozen_big_edge_index",
]