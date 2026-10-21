from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data


class GraphBuilder:
    """
    Build dynamic interaction graphs from node embeddings.

    The builder is a model-independent graph-construction utility. It
    receives a batch of node embeddings and constructs one directed
    k-nearest-neighbor graph for each sample in the batch.

    The graph topology is computed from cosine similarity between node
    embeddings. The resulting graphs are represented using native
    PyTorch Geometric ``Data`` and ``Batch`` objects.

    This implementation realizes the adaptive graph construction stage
    of HDEG. The builder contains no learnable parameters and therefore
    does not inherit from ``torch.nn.Module``.

    Parameters
    ----------
    top_k:
        Number of neighbors selected for every node.

    self_loops:
        If ``False``, a node cannot select itself as a neighbor.
        If ``True``, self-connections are allowed.

    symmetric:
        If ``False``, the graph preserves the directed k-nearest-neighbor
        topology. If ``True``, every selected edge is accompanied by its
        reverse edge. Symmetrization may therefore increase the number of
        outgoing edges beyond ``top_k``.

    Notes
    -----
    Input embeddings have shape:

        ``(batch_size, num_nodes, embedding_dim)``

    The returned PyG ``Batch`` contains flattened node embeddings,
    batched edge connectivity, cosine-similarity edge attributes, and
    a batch assignment vector.
    """

    def __init__(
        self,
        top_k: int,
        self_loops: bool = False,
        symmetric: bool = False,
    ) -> None:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError(
                "top_k must be an integer."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if not isinstance(self_loops, bool):
            raise TypeError(
                "self_loops must be a boolean."
            )

        if not isinstance(symmetric, bool):
            raise TypeError(
                "symmetric must be a boolean."
            )

        self.top_k = top_k
        self.self_loops = self_loops
        self.symmetric = symmetric

    def build(
        self,
        embeddings: Tensor,
    ) -> Batch:
        """
        Construct a batched dynamic graph from node embeddings.

        Parameters
        ----------
        embeddings:
            Node embeddings with shape
            ``(batch_size, num_nodes, embedding_dim)``.

        Returns
        -------
        torch_geometric.data.Batch
            Batched PyTorch Geometric graph containing one graph for
            every sample in the input batch.
        """

        self._validate_inputs(embeddings)

        normalized_embeddings = (
            self._normalize_embeddings(
                embeddings
            )
        )

        similarity = self._compute_similarity(
            normalized_embeddings
        )

        neighbor_indices, neighbor_weights = (
            self._select_neighbors(
                similarity
            )
        )

        graphs = []

        for batch_index in range(
            embeddings.size(0)
        ):
            graphs.append(
                self._build_graph(
                    node_embeddings=embeddings[
                        batch_index
                    ],
                    neighbor_indices=neighbor_indices[
                        batch_index
                    ],
                    neighbor_weights=neighbor_weights[
                        batch_index
                    ],
                )
            )

        return Batch.from_data_list(
            graphs
        )

    def _validate_inputs(
        self,
        embeddings: Tensor,
    ) -> None:
        """
        Validate the tensor-level graph-builder input contract.
        """

        if not isinstance(embeddings, Tensor):
            raise TypeError(
                "embeddings must be a torch.Tensor."
            )

        if embeddings.ndim != 3:
            raise ValueError(
                "Expected embeddings with shape "
                "(batch_size, num_nodes, embedding_dim), "
                f"but received shape "
                f"{tuple(embeddings.shape)}."
            )

        batch_size, num_nodes, embedding_dim = (
            embeddings.shape
        )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if num_nodes <= 0:
            raise ValueError(
                "num_nodes must be greater than zero."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        if not embeddings.is_floating_point():
            raise TypeError(
                "embeddings must be a floating-point tensor."
            )

        if not torch.isfinite(
            embeddings
        ).all():
            raise ValueError(
                "embeddings contains NaN or infinite values."
            )

        max_neighbors = (
            num_nodes
            if self.self_loops
            else num_nodes - 1
        )

        if self.top_k > max_neighbors:
            requirement = (
                "top_k <= num_nodes"
                if self.self_loops
                else "top_k < num_nodes"
            )

            raise ValueError(
                f"Invalid top_k={self.top_k} for "
                f"num_nodes={num_nodes}. "
                f"Expected {requirement}."
            )

    @staticmethod
    def _normalize_embeddings(
        embeddings: Tensor,
    ) -> Tensor:
        """
        L2-normalize every node embedding.
        """

        return F.normalize(
            embeddings,
            p=2,
            dim=-1,
        )

    @staticmethod
    def _compute_similarity(
        embeddings: Tensor,
    ) -> Tensor:
        """
        Compute pairwise cosine similarity.

        Returns a tensor with shape:

            ``(batch_size, num_nodes, num_nodes)``
        """

        return torch.matmul(
            embeddings,
            embeddings.transpose(
                -1,
                -2,
            ),
        )

    def _select_neighbors(
        self,
        similarity: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Select the top-k neighbors for every node.

        Returns
        -------
        neighbor_indices:
            Shape ``(B, N, top_k)``.

        neighbor_weights:
            Shape ``(B, N, top_k)``.
        """

        if similarity.ndim != 3:
            raise ValueError(
                "similarity must have shape "
                "(batch_size, num_nodes, num_nodes)."
            )

        _, num_nodes, _ = similarity.shape

        scores = similarity.clone()

        if not self.self_loops:
            diagonal = torch.eye(
                num_nodes,
                dtype=torch.bool,
                device=scores.device,
            )

            scores = scores.masked_fill(
                diagonal.unsqueeze(0),
                float("-inf"),
            )

        neighbor_weights, neighbor_indices = (
            torch.topk(
                scores,
                k=self.top_k,
                dim=-1,
                largest=True,
                sorted=True,
            )
        )

        return (
            neighbor_indices,
            neighbor_weights,
        )

    def _build_graph(
        self,
        node_embeddings: Tensor,
        neighbor_indices: Tensor,
        neighbor_weights: Tensor,
    ) -> Data:
        """
        Construct a PyG ``Data`` object for one ecosystem.
        """

        num_nodes = node_embeddings.size(0)

        if self.symmetric:
            (
                source_nodes,
                target_nodes,
                edge_weights,
            ) = self._build_symmetric_edges(
                neighbor_indices,
                neighbor_weights,
            )

        else:
            source_nodes = (
                torch.arange(
                    num_nodes,
                    device=node_embeddings.device,
                )
                .view(-1, 1)
                .expand(
                    -1,
                    self.top_k,
                )
                .reshape(-1)
            )

            target_nodes = (
                neighbor_indices.reshape(-1)
            )

            edge_weights = (
                neighbor_weights.reshape(-1)
            )

        edge_index = torch.stack(
            [
                source_nodes,
                target_nodes,
            ],
            dim=0,
        ).long()

        edge_attr = edge_weights.unsqueeze(-1)

        return Data(
            x=node_embeddings,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
        )

    @staticmethod
    def _build_symmetric_edges(
        neighbor_indices: Tensor,
        neighbor_weights: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Build a duplicate-free symmetric edge list.

        The forward k-nearest-neighbor edges are retained and every edge
        is accompanied by its reverse edge. If an edge already exists in
        both directions, each directed edge is stored only once.
        """

        num_nodes, top_k = (
            neighbor_indices.shape
        )

        source_nodes = (
            torch.arange(
                num_nodes,
                device=neighbor_indices.device,
            )
            .view(-1, 1)
            .expand(
                -1,
                top_k,
            )
            .reshape(-1)
        )

        target_nodes = (
            neighbor_indices.reshape(-1)
        )

        edge_weights = (
            neighbor_weights.reshape(-1)
        )

        reverse_source = target_nodes
        reverse_target = source_nodes
        reverse_weights = edge_weights

        all_source = torch.cat(
            [
                source_nodes,
                reverse_source,
            ],
            dim=0,
        )

        all_target = torch.cat(
            [
                target_nodes,
                reverse_target,
            ],
            dim=0,
        )

        all_weights = torch.cat(
            [
                edge_weights,
                reverse_weights,
            ],
            dim=0,
        )

        edges = torch.stack(
            [
                all_source,
                all_target,
            ],
            dim=1,
        )

        unique_edges, inverse = (
            torch.unique(
                edges,
                dim=0,
                return_inverse=True,
            )
        )

        unique_weights = torch.full(
            (
                unique_edges.size(0),
            ),
            float("-inf"),
            dtype=all_weights.dtype,
            device=all_weights.device,
        )

        unique_weights.scatter_reduce_(
            0,
            inverse,
            all_weights,
            reduce="amax",
            include_self=True,
        )

        return (
            unique_edges[:, 0],
            unique_edges[:, 1],
            unique_weights,
        )