from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv


class GraphReasoner(nn.Module):
    """
    Graph-based relational reasoning component of HDEG.

    The GraphReasoner receives the adaptive interaction graph produced
    by GraphBuilder and performs learned graph-attention message
    passing over that graph.

    The component realizes the Graph Relational Reasoning stage of the
    Device Behavioral Representation Learning (DBRL) module.

    Parameters
    ----------
    embedding_dim:
        Dimensionality of the input temporal device embeddings and the
        resulting contextualized device representations.

    heads:
        Number of GAT attention heads.

        The default and Version-1 HDEG realization uses one attention
        head.

    dropout:
        Dropout probability used inside the GAT attention mechanism.

        The default is zero because deterministic inference and the
        minimal Version-1 realization do not require attention dropout.

    Notes
    -----
    The scientific representation interface is preserved:

        Input:
            PyG Batch containing node features of shape
            ``(B * N, embedding_dim)``.

        Output:
            Tensor of shape ``(B, N, embedding_dim)``.

    The interaction graph itself and the intermediate graph messages
    remain internal computational objects and are not exposed.
    """

    def __init__(
        self,
        embedding_dim: int,
        heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not isinstance(embedding_dim, int):
            raise TypeError(
                "embedding_dim must be an integer."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than zero."
            )

        if not isinstance(heads, int):
            raise TypeError(
                "heads must be an integer."
            )

        if heads <= 0:
            raise ValueError(
                "heads must be greater than zero."
            )

        if not isinstance(dropout, (float, int)):
            raise TypeError(
                "dropout must be a floating-point value."
            )

        dropout = float(dropout)

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0.0 <= dropout < 1.0."
            )

        self.embedding_dim = embedding_dim
        self.heads = heads
        self.dropout = dropout

        self.gat = GATConv(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
        )

    def forward(
        self,
        graph: Batch,
    ) -> Tensor:
        """
        Perform graph-attention relational reasoning.

        Parameters
        ----------
        graph:
            PyTorch Geometric Batch produced by GraphBuilder.

        Returns
        -------
        Tensor
            Contextualized device behavioral representations with shape

                ``(batch_size, num_nodes, embedding_dim)``

        Raises
        ------
        TypeError
            If the supplied object is not a PyG Batch.

        ValueError
            If the graph structure or node representation tensor does
            not satisfy the expected interface.
        """

        self._validate_graph(graph)

        x = graph.x
        edge_index = graph.edge_index

        contextualized = self.gat(
            x,
            edge_index,
        )

        if not torch.isfinite(
            contextualized
        ).all():
            raise ValueError(
                "GraphReasoner produced NaN or infinite values."
            )

        return self._restore_batch_structure(
            contextualized,
            graph,
        )

    def _validate_graph(
        self,
        graph: Batch,
    ) -> None:
        """
        Validate the graph input contract.
        """

        if not isinstance(graph, Batch):
            raise TypeError(
                "graph must be a torch_geometric.data.Batch."
            )

        if not hasattr(graph, "x") or graph.x is None:
            raise ValueError(
                "The graph must contain node features in graph.x."
            )

        if graph.x.ndim != 2:
            raise ValueError(
                "graph.x must have shape "
                "(num_nodes, embedding_dim)."
            )

        if graph.x.size(-1) != self.embedding_dim:
            raise ValueError(
                "Graph node feature dimensionality does not match "
                f"embedding_dim={self.embedding_dim}. "
                f"Received {graph.x.size(-1)}."
            )

        if not graph.x.is_floating_point():
            raise TypeError(
                "graph.x must be a floating-point tensor."
            )

        if not torch.isfinite(graph.x).all():
            raise ValueError(
                "graph.x contains NaN or infinite values."
            )

        if not hasattr(graph, "edge_index"):
            raise ValueError(
                "The graph must contain edge_index."
            )

        if graph.edge_index.ndim != 2:
            raise ValueError(
                "graph.edge_index must have shape "
                "(2, num_edges)."
            )

        if graph.edge_index.size(0) != 2:
            raise ValueError(
                "graph.edge_index must have shape "
                "(2, num_edges)."
            )

        if graph.edge_index.dtype != torch.long:
            raise TypeError(
                "graph.edge_index must have dtype torch.long."
            )

        if graph.num_nodes <= 0:
            raise ValueError(
                "The graph must contain at least one node."
            )

        if graph.num_graphs <= 0:
            raise ValueError(
                "The graph batch must contain at least one graph."
            )

        if not hasattr(graph, "batch") or graph.batch is None:
            raise ValueError(
                "The graph Batch must contain a batch assignment "
                "vector."
            )

        if graph.batch.ndim != 1:
            raise ValueError(
                "graph.batch must be a one-dimensional tensor."
            )

        if graph.batch.numel() != graph.num_nodes:
            raise ValueError(
                "graph.batch must contain one assignment for every "
                "node."
            )

        self._validate_equal_graph_sizes(graph)

        self._validate_edge_indices(graph)

    @staticmethod
    def _validate_edge_indices(
        graph: Batch,
    ) -> None:
        """
        Verify that all edge indices reference valid nodes.
        """

        if graph.edge_index.numel() == 0:
            return

        minimum_index = graph.edge_index.min().item()
        maximum_index = graph.edge_index.max().item()

        if minimum_index < 0:
            raise ValueError(
                "graph.edge_index contains a negative node index."
            )

        if maximum_index >= graph.num_nodes:
            raise ValueError(
                "graph.edge_index contains an out-of-range node index."
            )

    @staticmethod
    def _validate_equal_graph_sizes(
        graph: Batch,
    ) -> None:
        """
        Verify that every ecosystem graph contains the same number
        of device nodes.

        HDEG represents all ecosystems using a fixed device dimension,
        so the graph batch must preserve that structural assumption.
        """

        counts = torch.bincount(
            graph.batch,
            minlength=graph.num_graphs,
        )

        if counts.numel() != graph.num_graphs:
            raise ValueError(
                "Invalid graph batch assignment."
            )

        if not torch.all(
            counts == counts[0]
        ):
            raise ValueError(
                "All graphs in the batch must contain the same "
                "number of nodes."
            )

    @staticmethod
    def _restore_batch_structure(
        node_embeddings: Tensor,
        graph: Batch,
    ) -> Tensor:
        """
        Restore the flattened PyG node representation to
        ``(B, N, d)`` layout.
        """

        batch_size = graph.num_graphs

        node_counts = torch.bincount(
            graph.batch,
            minlength=batch_size,
        )

        if node_counts.numel() != batch_size:
            raise ValueError(
                "Unable to determine the number of nodes per graph."
            )

        if not torch.all(
            node_counts == node_counts[0]
        ):
            raise ValueError(
                "Cannot restore batch structure because graph sizes "
                "are not identical."
            )

        num_nodes = int(
            node_counts[0].item()
        )

        embedding_dim = node_embeddings.size(-1)

        return node_embeddings.reshape(
            batch_size,
            num_nodes,
            embedding_dim,
        )