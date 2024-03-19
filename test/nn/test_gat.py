"""
Tests the GAT and GATv2 model
"""
# pylint: disable=missing-function-docstring, duplicate-code

import pytest
from torch_geometric.data import Data

from ctgnn.nn.gat import GAT, GATConv_


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_gat_forward(graph_pyg, num_layers):
    model = GAT(
        in_channels=5, hidden_channels=32, out_channels=6, num_layers=num_layers
    )
    output = model(graph_pyg)
    assert output.shape == (5, 6)

    graph_pyg.batch_size = 5
    graph_pyg.num_sampled_nodes = 5
    graph_pyg.num_sampled_edges = 10
    output = model(graph_pyg)


def test_gatconv_forward(graph_pyg):
    model = GATConv_(in_channels=5, out_channels=32, heads=2)
    output = model(graph_pyg)
    assert isinstance(output, Data)
    assert output.x.shape == (5, 32 * 2)
