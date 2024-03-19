"""Trainer and evaluator for the model."""

import torch
from ignite.engine import Engine
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.data import Data


def setup_trainer(
    model: Module,
    optimizer: Optimizer,
    loss_fn: Module,
    device: str | torch.device,
) -> Engine:
    """Setup trainer engine.

    Args:
        model: The model.
        optimizer: The optimizer.
        loss_fn: The loss function.
        device: The device to use for training.

    Returns:
        trainer: The trainer engine.
    """

    def train_function(
        engine: Engine, batch: Data  # pylint: disable=unused-argument
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Train function. This is called every iteration with a batch of data."""

        model.train()
        batch_size = batch.batch_size
        batch = batch.to(device, non_blocking=True)
        y = batch.y.to(device, non_blocking=True)

        # Context manager that allows you to perform mixed-precision computations.
        # Speeds up training and reduces memory usage.
        with autocast():
            outputs = model(batch)
            # Only use the first batch_size elements of the output to calculate the loss
            # This is necessary because `NeighborLoader` returns the neighbors
            loss = loss_fn(outputs[:batch_size], y[:batch_size])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return outputs[:batch_size], y[:batch_size]

    return Engine(train_function)


def setup_evaluator(
    model: Module,
    device: str | torch.device,
) -> Engine:
    """Setup evaluator engine.

    Args:
        model: The model.
        device: The device to use for training.

    Returns:
        evaluator: The evaluator engine.
    """

    @torch.no_grad()
    def eval_function(
        engine: Engine, batch: Data  # pylint: disable=unused-argument
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        batch_size = batch.batch_size
        batch = batch.to(device, non_blocking=True)
        y = batch.y.to(device, non_blocking=True)

        with autocast():
            outputs = model(batch)

        return outputs[:batch_size], y[:batch_size]

    return Engine(eval_function)
