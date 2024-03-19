"""
Test configuration file for pytest.
"""
# pylint: disable=missing-function-docstring, unused-argument, redefined-outer-name

import os
from typing import Optional

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explanation
from torch_geometric.explain.config import MaskType
from torch_geometric.testing import get_random_edge_index

rng = np.random.default_rng(42)


@pytest.fixture()
def check_explanation():
    def _check_explanation(
        explanation: Explanation,
        node_mask_type: Optional[MaskType],
        edge_mask_type: Optional[MaskType],
    ):
        if node_mask_type == MaskType.attributes:
            assert explanation.node_mask.size() == explanation.x.size()
            assert explanation.node_mask.min() >= 0
            assert explanation.node_mask.max() <= 1
        elif node_mask_type == MaskType.object:
            assert explanation.node_mask.size() == (explanation.num_nodes, 1)
            assert explanation.node_mask.min() >= 0
            assert explanation.node_mask.max() <= 1
        elif node_mask_type == MaskType.common_attributes:
            assert explanation.node_mask.size() == (1, explanation.x.size(-1))
            assert explanation.node_mask.min() >= 0
            assert explanation.node_mask.max() <= 1
        elif node_mask_type is None:
            assert "node_mask" not in explanation

        if edge_mask_type == MaskType.object:
            assert explanation.edge_mask.size() == (explanation.num_edges,)
            assert explanation.edge_mask.min() >= 0
            assert explanation.edge_mask.max() <= 1
        elif edge_mask_type is None:
            assert "edge_mask" not in explanation

    return _check_explanation


@pytest.fixture()
def data():
    torch.manual_seed(42)
    return Data(
        x=torch.randn(4, 3, requires_grad=True),
        edge_index=get_random_edge_index(4, 4, num_edges=6),
        edge_attr=torch.randn(6, 3),
    )


@pytest.fixture()
def dataset_pyg(tmp_path, scope="session"):
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield Planetoid(root="Planetoid", name="Cora")
    os.chdir(old_cwd)


@pytest.fixture()
def graph_pyg():
    graph = Data(
        x=torch.tensor(rng.random((5, 5)), dtype=torch.float32),
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 2, 0, 1, 3, 4], [1, 2, 3, 4, 0, 0, 2, 3, 1, 2]],
            dtype=torch.long,
        ),
        y=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
    )
    return graph


@pytest.fixture()
def positions():
    return rng.random((50, 2))


@pytest.fixture()
def features():
    return rng.random((50, 5))


@pytest.fixture()
def labels():
    return rng.integers(10, size=50)


@pytest.fixture()
def edges():
    return rng.integers(50, size=(100, 2))


@pytest.fixture
def example_config():
    return {
        "seed": 43,
        "dataset": "intestine",
        "construction_method": "knn",
        "param": [1, 2, 3],
        "features": "raw",
        "self_loops": False,
        "batch_size": [32, 64],
        "lr": [0.01, 0.001],
        "model_name": "GCN",
        "num_layers": 2,
        "hidden_dim": 32,
        "activation": "relu",
    }


@pytest.fixture
def config_dict(tmp_path):
    return OmegaConf.create(
        {
            "seed": 42,
            "experiment_name": "test",
            "dataset": "intestine",
            "construction_method": "delaunay",
            "self_loops": False,
            "features": "raw",
            "transform": None,
            "global_transform": None,
            "num_neighbors": -1,
            "num_workers": 0,
            "splits": [0.8, 0.1, 0.1],
            "batch_size": 10000,
            "max_epochs": 300,
            "patience": 5,
            "lr": 0.001,
            "model_name": "GCN",
            "num_layers": 2,
            "hidden_dim": 64,
            "activation": "relu",
            "dropout": 0.2,
            "output_dir": str(tmp_path),
        }
    )
