"""
Test trainer and evaluator setup function.
"""
# pylint: disable=missing-function-docstring, protected-access, redefined-outer-name

import pytest
import torch
from ignite.engine import Engine
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch_geometric.loader import NeighborLoader

from ctgnn.nn import GCN
from ctgnn.training.trainers import setup_evaluator, setup_trainer


@pytest.fixture
def model():
    return GCN(5, 5, 5, 1)


@pytest.fixture
def optimizer(model):
    return SGD(model.parameters(), lr=0.01)


@pytest.fixture
def loss_fn():
    return CrossEntropyLoss()


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def loader(graph_pyg):
    return NeighborLoader(
        graph_pyg,
        num_neighbors=[-1],
        batch_size=5,
    )


def test_setup_trainer(model, optimizer, loss_fn, device, loader):
    # Call the setup_trainer function
    trainer = setup_trainer(model, optimizer, loss_fn, device)

    # Check that the trainer is an instance of Engine
    assert isinstance(trainer, Engine)

    # Check that the trainer function updates the model parameters
    old_params = next(model.parameters()).clone()
    trainer.run(loader)
    assert old_params.ne(next(model.parameters())).any()

    # Check that the trainer function returns the correct output
    outputs, targets = trainer.run(loader).output
    assert outputs.shape == (5, 5)
    assert targets.shape == (5,)


def test_setup_evaluator(model, device, loader):
    # Call the setup_evaluator function
    evaluator = setup_evaluator(model, device)

    # Check that the evaluator is an instance of Engine
    assert isinstance(evaluator, Engine)

    # Check that the evaluator function does not update the model parameters
    old_params = next(model.parameters()).clone()
    evaluator.run(loader)
    assert old_params.eq(next(model.parameters())).all()

    # Check that the evaluator function returns the correct output
    outputs, targets = evaluator.run(loader).output
    assert outputs.shape == (5, 5)
    assert targets.shape == (5,)
