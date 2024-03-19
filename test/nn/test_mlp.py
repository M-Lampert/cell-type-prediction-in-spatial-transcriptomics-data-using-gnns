"""
Tests the MLP model
"""
# pylint: disable=missing-function-docstring, duplicate-code

import pytest

from ctgnn.nn.mlp import MLP


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_mlp_forward(graph_pyg, num_layers):
    mlp = MLP(
        in_channels=5,
        hidden_channels=12,
        out_channels=6,
        num_layers=num_layers,
        dropout=0.1,
        activation="leaky_relu",
    )
    output = mlp.forward(graph_pyg)
    assert output.shape == (5, 6)
