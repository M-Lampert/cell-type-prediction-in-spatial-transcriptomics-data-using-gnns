"""Tests for the graph construction module."""

# pylint: disable=missing-function-docstring, invalid-name

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch_geometric as pyg

from ctgnn.data.graph_construction import construct_graph, get_Delaunay_graph

rng = np.random.default_rng(42)

mock_adj_list_list = [
    [0, 1],
    [0, 2],
    [4, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 5],
    [5, 4],
]
mock_adj_list_tuple = [
    (0, 1),
    (0, 2),
    (4, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 5),
    (5, 4),
]
mock_adj_np = np.array(mock_adj_list_list)
mock_adj_coo = sp.coo_matrix(
    (np.ones(mock_adj_np.shape[0]), (mock_adj_np[:, 0], mock_adj_np[:, 1]))
)
mock_positions = torch.tensor([[0, 0], [1, 0.5], [2, 0], [3, 2], [4, 1.5], [5.0, 2.0]])
mock_num_nodes = 6
mock_num_edges = 9
mock_features = rng.random((mock_num_nodes, 2))
mock_edge_features = rng.random((mock_num_edges, 2))
mock_bidirectional_adj_list = np.concatenate(
    (
        mock_adj_np[:-3],
        mock_adj_np[
            :-3, ::-1
        ],  # Repeat all edges that are not self-loops and exist only once
        mock_adj_np[-2:-1],  # Add the self-loop edge
        mock_adj_np[[6, 8]],  # Add the the edges that are bidirectional
    ),
    axis=0,
)
mock_bidirectional_edge_features = np.concatenate(
    (
        mock_edge_features[:-3],
        mock_edge_features[
            :-3
        ],  # Repeat all features of edges that are not self-loops and exist only once
        mock_edge_features[-2:-1],  # Add the feature of the self-loop
        np.repeat(
            np.mean(mock_edge_features[[6, 8]], axis=0, keepdims=True), 2, axis=0
        ),  # Add the mean of the features of the two edges that are bidirectional
    ),
    axis=0,
)
mock_labels = rng.integers(0, 2, size=(mock_num_nodes, 1))
mock_parameter_string = "adj_list, positions, features, labels, edge_features, directed"
mock_parameter_list = [
    (mock_adj_list_list, None, None, None, None, False),
    (mock_adj_list_tuple, None, None, None, None, False),
    (mock_adj_np, None, None, None, None, False),
    (mock_adj_coo, None, None, None, None, False),
    (mock_adj_np, mock_positions, None, None, None, False),
    (mock_adj_np, None, mock_features, None, None, False),
    (mock_adj_np, None, None, mock_labels, None, False),
    (mock_adj_np, None, None, None, mock_edge_features, False),
    (mock_adj_np, None, None, None, None, True),
    (mock_adj_np, None, None, None, mock_edge_features, True),
]
mock_parameter_list_graphs = [
    (mock_positions, 2, False),
    (mock_positions, 2, False),
    (mock_positions, 1, False),
    (mock_positions, 2, True),
]


def test_build_radius_delaunay_graph(positions):
    radius = 0.01
    n_radius_edges = construct_graph(
        algorithm="radius", positions=positions, param=radius
    ).num_edges
    n_delaunay_edges = construct_graph(
        algorithm="delaunay", positions=positions
    ).num_edges
    graph = construct_graph(
        algorithm="radius_delaunay", positions=positions, param=radius
    )
    assert graph.num_edges <= n_radius_edges
    assert graph.num_edges <= n_delaunay_edges

    graph = construct_graph(
        algorithm="radius_delaunay", positions=positions, param=radius, self_loops=True
    )
    assert pyg.utils.contains_self_loops(graph.edge_index)

    graph = construct_graph(
        algorithm="radius_delaunay", positions=positions, param=radius, self_loops=False
    )
    assert not pyg.utils.contains_self_loops(graph.edge_index)


def test_empty_graph(features, labels):
    n_nodes = len(features)

    with pytest.raises(AssertionError):
        construct_graph(
            algorithm="empty", features=features, labels=labels, self_loops=False
        )

    graph = construct_graph(
        algorithm="empty", features=features, labels=labels, self_loops=True
    )
    assert isinstance(graph, pyg.data.Data)
    assert graph.num_nodes == n_nodes
    assert graph.has_self_loops()


@pytest.mark.parametrize(
    "positions, k, self_loops",
    mock_parameter_list_graphs,
)
def test_knn_graph(positions, k, self_loops):
    graph = construct_graph(
        algorithm="knn",
        positions=positions,
        param=k,
        self_loops=self_loops,
    )

    assert isinstance(graph, pyg.data.Data)
    assert graph.num_nodes == mock_num_nodes
    assert graph.num_edges >= mock_num_nodes

    if self_loops:
        assert graph.has_self_loops()
    else:
        assert not graph.has_self_loops()


@pytest.mark.parametrize(
    "positions, radius, self_loops",
    mock_parameter_list_graphs,
)
def test_radius_graph(positions, radius, self_loops):
    graph = construct_graph(
        algorithm="radius",
        positions=positions,
        param=radius,
        self_loops=self_loops,
    )

    assert isinstance(graph, pyg.data.Data)
    assert graph.num_nodes == mock_num_nodes
    if radius == 2:
        assert graph.num_edges > 0
    else:
        assert graph.num_edges == 0

    if self_loops:
        assert graph.has_self_loops()
    else:
        assert not graph.has_self_loops()


@pytest.mark.parametrize(
    "positions, _, self_loops",
    mock_parameter_list_graphs + [(mock_positions, 2, True)],
)
def test_delaunay_graph(positions, _, self_loops):
    graph = construct_graph(
        algorithm="delaunay",
        positions=positions,
        self_loops=self_loops,
    )

    assert isinstance(graph, pyg.data.Data)
    assert graph.num_nodes == mock_num_nodes
    assert graph.num_edges > 0

    if self_loops:
        assert graph.has_self_loops()
    else:
        assert not graph.has_self_loops()


def test_construct_graph_fails():
    with pytest.raises(ValueError):
        construct_graph(algorithm="unknown")


def test_get_Delaunay_graph():
    positions = torch.tensor(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
        ]
    )
    emtpy_graph = pyg.data.Data(pos=positions)
    graph = get_Delaunay_graph(emtpy_graph, self_loops=False)
    assert graph.num_nodes == 5
    edge_index = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            [1, 2, 3, 4, 0, 2, 4, 0, 1, 3, 0, 2, 4, 0, 1, 3],
        ]
    )
    assert torch.all(torch.eq(graph.edge_index, edge_index))

    positions = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
    )
    emtpy_graph = pyg.data.Data(pos=positions)
    graph = get_Delaunay_graph(emtpy_graph, self_loops=False)
    assert graph.num_nodes == 7
    print(graph.edge_index)
    edge_index = torch.tensor(
        [
            [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                4,
                5,
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                6,
            ],
            [
                1,
                2,
                3,
                4,
                5,
                6,
                0,
                2,
                4,
                5,
                6,
                0,
                1,
                3,
                5,
                6,
                0,
                2,
                4,
                5,
                6,
                0,
                1,
                3,
                5,
                6,
                0,
                1,
                2,
                3,
                4,
                0,
                1,
                2,
                3,
                4,
            ],
        ]
    )
    assert torch.all(torch.eq(graph.edge_index, edge_index))

    with pytest.raises(ValueError):
        positions = torch.tensor(
            [
                [0, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 0, 1],
                [-1, 0, 0, 1],
                [0, -1, 0, 1],
                [0, 0, 1, 1],
                [0, 0, -1, 1],
            ]
        )
        emtpy_graph = pyg.data.Data(pos=positions)
        graph = get_Delaunay_graph(emtpy_graph, self_loops=False)
