"""
Test model setup function.
"""
# pylint: disable=missing-function-docstring, protected-access, redefined-outer-name

import pytest
from omegaconf import OmegaConf

from ctgnn.nn import GAT, GCN, MLP, GraphSAGE
from ctgnn.training.models import setup_model


@pytest.fixture
def dataset_sizes():
    return (10, 5)


@pytest.fixture
def config():
    return OmegaConf.create(
        {
            "model_name": "GCN",
            "hidden_dim": 16,
            "num_layers": 2,
            "activation": "relu",
            "dropout": 0.5,
        }
    )


@pytest.mark.parametrize(
    "model_name, model_class",
    [
        ("GCN", GCN),
        ("GraphSAGE", GraphSAGE),
        ("SAGE", GraphSAGE),
        ("MLP", MLP),
        ("GAT", GAT),
    ],
)
def test_setup_gnn(config, dataset_sizes, model_name, model_class):
    config.model_name = model_name
    if model_name == "GAT":
        config.heads = 8
    model = setup_model(config, dataset_sizes)

    assert isinstance(model, model_class)
    if model_name == "MLP":
        assert len(model.layers) == 4
    else:
        assert len(model.layers) == 2

    hidden_dim = config.hidden_dim

    if model_name in ["GCN", "GAT", "SAGE", "GraphSAGE"]:
        assert model.layers[0].pyg_conv.in_channels == dataset_sizes[0]
        assert model.layers[0].pyg_conv.out_channels == hidden_dim
        assert (
            model.layers[-1].pyg_conv.in_channels == hidden_dim
            if model_name != "GAT"
            else hidden_dim * config.heads
        )
        assert model.layers[-1].pyg_conv.out_channels == hidden_dim
        assert (
            model.final_mlp[0].in_features == hidden_dim
            if model_name != "GAT"
            else hidden_dim * config.heads
        )
        assert model.final_mlp[0].out_features == hidden_dim
        assert model.final_mlp[-1].in_features == hidden_dim
        assert model.final_mlp[-1].out_features == dataset_sizes[1]
    elif model_name == "MLP":
        assert model.layers[0].in_features == dataset_sizes[0]
        assert model.layers[0].out_features == hidden_dim
        assert model.layers[-1].in_features == hidden_dim
        assert model.layers[-1].out_features == dataset_sizes[1]
