"""
Test data setup function.
"""
# pylint: disable=missing-function-docstring, protected-access, redefined-outer-name

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
from torch_geometric.loader import NeighborLoader

from ctgnn.training.data import get_graph, setup_data


def test_setup_data(config_dict):
    train_loader, eval_loader, test_loader, (num_features, num_classes) = setup_data(
        config_dict
    )

    # Check that the loaders are instances of NeighborLoader
    assert isinstance(train_loader, NeighborLoader)
    assert isinstance(eval_loader, NeighborLoader)
    assert isinstance(test_loader, NeighborLoader)

    # Check that the loaders have the correct batch size
    assert train_loader.batch_size == config_dict.batch_size
    assert eval_loader.batch_size == config_dict.batch_size
    assert test_loader.batch_size == config_dict.batch_size

    # Check that the loaders have the correct number of workers
    assert train_loader.num_workers == config_dict.num_workers
    assert eval_loader.num_workers == config_dict.num_workers
    assert test_loader.num_workers == config_dict.num_workers

    # Check that the loaders have the correct number of features and classes
    assert num_features == 241
    assert num_classes == 19


def test_setup_data_features(config_dict):
    for feat in ["raw", "random"]:
        config_dict.features = feat
        setup_data(config_dict)

    with pytest.raises(ValueError):
        config_dict.features = "wrong"
        setup_data(config_dict)


def test_setup_wrong_config(config_dict):
    # check if it fails with a wrong config file
    config_dict.dataset = "wrong"
    with pytest.raises(AssertionError):
        setup_data(config_dict)

    with pytest.raises(ConfigAttributeError):
        config = OmegaConf.create({"dataset": "intestine"})
        config_dict.construction_method = "knn"
        config_dict.param = 3
        setup_data(config)


def test_get_graph(config_dict):
    for method, param in [
        ("knn", 5),
        ("radius", 1e-4),
        ("delaunay", None),
        ("radius_delaunay", 1e-4),
    ]:
        config_dict.construction_method = method
        if param is not None:
            config_dict.param = param
        else:
            config_dict.pop("param", None)
        g = get_graph(config_dict)
        assert (g.train_mask | g.val_mask | g.test_mask).all()
