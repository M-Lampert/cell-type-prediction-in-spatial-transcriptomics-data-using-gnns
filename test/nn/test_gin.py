"""
Tests the GIN model
"""
# pylint: disable=missing-function-docstring, duplicate-code

import pytest

from ctgnn.nn.gin import GIN, GINConv_


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_gin_forward(graph_pyg, num_layers):
    gin = GIN(in_channels=5, hidden_channels=12, out_channels=8, num_layers=num_layers)
    output = gin.forward(graph_pyg)
    assert output.shape == (5, 8)

    graph_pyg.batch_size = 5
    graph_pyg.num_sampled_nodes = 5
    graph_pyg.num_sampled_edges = 10
    output = gin.forward(graph_pyg)


def test_ginconv_forward(graph_pyg):
    conv = GINConv_(in_channels=5, out_channels=8)
    output = conv.forward(graph_pyg)
    assert output.x.shape == (5, 8)
