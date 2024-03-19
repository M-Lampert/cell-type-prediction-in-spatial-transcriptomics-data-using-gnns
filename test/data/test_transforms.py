"""
Tests for the graph construction module.
"""
# pylint: disable=missing-function-docstring, invalid-name

import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, degree

from ctgnn.data.transforms import ShuffleEdges


def test_ShuffleEdges():
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4], [3, 4, 5, 0, 1, 2, 2, 3, 4, 6, 7, 8]],
        dtype=torch.long,
    )
    edge_index = coalesce(edge_index=edge_index, num_nodes=9, sort_by_row=False)
    graph = Data(edge_index=edge_index, num_nodes=9)
    torch.manual_seed(12)
    graph = ShuffleEdges()(graph)
    assert not torch.allclose(graph.edge_index, edge_index)
    assert torch.allclose(
        degree(graph.edge_index[0], graph.num_nodes),
        degree(edge_index[0], graph.num_nodes),
    )
    assert torch.allclose(
        degree(graph.edge_index[1], graph.num_nodes),
        degree(edge_index[1], graph.num_nodes),
    )
