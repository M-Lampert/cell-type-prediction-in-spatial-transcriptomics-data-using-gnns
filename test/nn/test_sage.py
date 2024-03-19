"""
Tests the GraphSAGE model
"""
# pylint: disable=missing-function-docstring, duplicate-code

import pytest

from ctgnn.nn.sage import GraphSAGE, SAGEConv_


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_sage_forward(graph_pyg, num_layers):
    sage = GraphSAGE(
        in_channels=5,
        hidden_channels=12,
        out_channels=6,
        num_layers=num_layers,
        dropout=0.1,
        activation="leaky_relu",
    )
    output = sage.forward(graph_pyg)
    assert output.shape == (5, 6)

    graph_pyg.batch_size = 5
    graph_pyg.num_sampled_nodes = 5
    graph_pyg.num_sampled_edges = 10
    output = sage.forward(graph_pyg)


def test_sageconv_forward(graph_pyg):
    conv = SAGEConv_(in_channels=5, out_channels=6, pos_dim=2)
    output = conv.forward(graph_pyg)
    assert output.x.shape == (5, 6)
