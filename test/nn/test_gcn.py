"""
Tests the GCN model
"""
# pylint: disable=missing-function-docstring, duplicate-code

import pytest

from ctgnn.nn.gcn import GCN, GCNConv_


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_gcn_forward(graph_pyg, num_layers):
    gcn = GCN(in_channels=5, hidden_channels=12, out_channels=8, num_layers=num_layers)
    output = gcn.forward(graph_pyg)
    assert output.shape == (5, 8)

    graph_pyg.batch_size = 5
    graph_pyg.num_sampled_nodes = 5
    graph_pyg.num_sampled_edges = 10
    output = gcn.forward(graph_pyg)
    assert output.shape == (5, 8)


def test_gcnconv_forward(graph_pyg):
    conv = GCNConv_(in_channels=5, out_channels=8)
    output = conv.forward(graph_pyg)
    assert output.x.shape == (5, 8)
