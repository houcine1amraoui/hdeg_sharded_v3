import pytest
import torch
from torch_geometric.data import Batch

from src.common.graph.graph_builder import GraphBuilder


def test_constructor_accepts_valid_configuration():
    builder = GraphBuilder(
        top_k=2,
        self_loops=False,
        symmetric=False,
    )

    assert builder.top_k == 2
    assert builder.self_loops is False
    assert builder.symmetric is False


def test_constructor_rejects_non_positive_top_k():
    with pytest.raises(ValueError):
        GraphBuilder(top_k=0)

    with pytest.raises(ValueError):
        GraphBuilder(top_k=-1)


def test_constructor_rejects_invalid_top_k_type():
    with pytest.raises(TypeError):
        GraphBuilder(top_k=2.5)


def test_build_rejects_non_tensor_input():
    builder = GraphBuilder(top_k=2)

    with pytest.raises(TypeError):
        builder.build([[1.0, 2.0]])


def test_build_rejects_wrong_tensor_rank():
    builder = GraphBuilder(top_k=2)

    embeddings = torch.randn(4, 8)

    with pytest.raises(ValueError):
        builder.build(embeddings)


def test_build_rejects_integer_embeddings():
    builder = GraphBuilder(top_k=2)

    embeddings = torch.randint(
        low=0,
        high=10,
        size=(2, 4, 8),
    )

    with pytest.raises(TypeError):
        builder.build(embeddings)


def test_build_rejects_non_finite_embeddings():
    builder = GraphBuilder(top_k=2)

    embeddings = torch.randn(2, 4, 8)
    embeddings[0, 0, 0] = float("nan")

    with pytest.raises(ValueError):
        builder.build(embeddings)


def test_build_rejects_invalid_top_k_for_number_of_nodes():
    builder = GraphBuilder(top_k=4)

    embeddings = torch.randn(2, 4, 8)

    # self_loops=False means at most N-1 neighbors.
    with pytest.raises(ValueError):
        builder.build(embeddings)


def test_build_returns_pyg_batch():
    builder = GraphBuilder(top_k=2)

    embeddings = torch.randn(3, 5, 8)

    graph = builder.build(embeddings)

    assert isinstance(graph, Batch)


def test_build_preserves_all_nodes():
    batch_size = 3
    num_nodes = 5
    embedding_dim = 8

    embeddings = torch.randn(
        batch_size,
        num_nodes,
        embedding_dim,
    )

    builder = GraphBuilder(top_k=2)

    graph = builder.build(embeddings)

    assert graph.num_nodes == batch_size * num_nodes
    assert graph.x.shape == (
        batch_size * num_nodes,
        embedding_dim,
    )


def test_build_produces_expected_number_of_edges():
    batch_size = 3
    num_nodes = 5
    top_k = 2

    embeddings = torch.randn(
        batch_size,
        num_nodes,
        8,
    )

    builder = GraphBuilder(top_k=top_k)

    graph = builder.build(embeddings)

    expected_edges = (
        batch_size * num_nodes * top_k
    )

    assert graph.edge_index.shape == (
        2,
        expected_edges,
    )

    assert graph.edge_attr.shape == (
        expected_edges,
        1,
    )


def test_each_node_has_exactly_top_k_outgoing_edges():
    batch_size = 2
    num_nodes = 6
    top_k = 3

    embeddings = torch.randn(
        batch_size,
        num_nodes,
        8,
    )

    builder = GraphBuilder(top_k=top_k)

    graph = builder.build(embeddings)

    source_nodes = graph.edge_index[0]

    counts = torch.bincount(
        source_nodes,
        minlength=batch_size * num_nodes,
    )

    assert torch.all(
        counts == top_k
    )


def test_no_self_loops_when_disabled():
    batch_size = 2
    num_nodes = 6

    embeddings = torch.randn(
        batch_size,
        num_nodes,
        8,
    )

    builder = GraphBuilder(
        top_k=3,
        self_loops=False,
    )

    graph = builder.build(embeddings)

    source = graph.edge_index[0]
    target = graph.edge_index[1]

    assert not torch.any(
        source == target
    )


def test_self_loops_are_allowed_when_enabled():
    batch_size = 1
    num_nodes = 4

    # Identical embeddings make the diagonal competitive with all
    # other similarities. We only verify that self-loops are legal.
    embeddings = torch.ones(
        batch_size,
        num_nodes,
        8,
    )

    builder = GraphBuilder(
        top_k=4,
        self_loops=True,
    )

    graph = builder.build(embeddings)

    source = graph.edge_index[0]
    target = graph.edge_index[1]

    assert torch.any(
        source == target
    )


def test_graphs_are_isolated_across_batch_samples():
    batch_size = 3
    num_nodes = 5

    embeddings = torch.randn(
        batch_size,
        num_nodes,
        8,
    )

    builder = GraphBuilder(top_k=2)

    graph = builder.build(embeddings)

    source = graph.edge_index[0]
    target = graph.edge_index[1]

    batch_assignment = graph.batch

    assert torch.all(
        batch_assignment[source]
        == batch_assignment[target]
    )


def test_edge_weights_match_cosine_similarity():
    embeddings = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ]
    )

    builder = GraphBuilder(top_k=1)

    graph = builder.build(embeddings)

    source = graph.edge_index[0]
    target = graph.edge_index[1]
    weights = graph.edge_attr.squeeze(-1)

    normalized = torch.nn.functional.normalize(
        embeddings,
        p=2,
        dim=-1,
    )

    expected = (
        normalized[0, source]
        * normalized[0, target]
    ).sum(dim=-1)

    assert torch.allclose(
        weights,
        expected,
        atol=1e-6,
    )


def test_graph_construction_is_deterministic():
    torch.manual_seed(42)

    embeddings = torch.randn(
        4,
        7,
        16,
    )

    builder = GraphBuilder(top_k=3)

    graph_1 = builder.build(embeddings)
    graph_2 = builder.build(embeddings)

    assert torch.equal(
        graph_1.edge_index,
        graph_2.edge_index,
    )

    assert torch.allclose(
        graph_1.edge_attr,
        graph_2.edge_attr,
    )


def test_graph_builder_does_not_modify_input():
    embeddings = torch.randn(
        2,
        5,
        8,
    )

    original = embeddings.clone()

    builder = GraphBuilder(top_k=2)

    builder.build(embeddings)

    assert torch.equal(
        embeddings,
        original,
    )


def test_cosine_similarity_is_scale_invariant():
    embeddings = torch.tensor(
        [
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
            ]
        ]
    )

    scaled_embeddings = embeddings * 10.0

    builder = GraphBuilder(top_k=1)

    graph_1 = builder.build(embeddings)
    graph_2 = builder.build(scaled_embeddings)

    assert torch.equal(
        graph_1.edge_index,
        graph_2.edge_index,
    )

    assert torch.allclose(
        graph_1.edge_attr,
        graph_2.edge_attr,
        atol=1e-6,
    )


def test_symmetric_graph_contains_reverse_edges():
    embeddings = torch.randn(
        1,
        6,
        8,
    )

    builder = GraphBuilder(
        top_k=2,
        symmetric=True,
    )

    graph = builder.build(embeddings)

    edges = {
        (
            int(source),
            int(target),
        )
        for source, target in graph.edge_index.t().tolist()
    }

    for source, target in edges:
        assert (
            target,
            source,
        ) in edges