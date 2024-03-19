"""Utility functions for neural networks."""

from torch import nn


def get_activation(name: str) -> nn.Module:
    """Get the activation function from a string.

    Args:
        name: name of the activation function

    Returns:
        the activation function
    """
    res: nn.Module
    match name:
        case "relu":
            res = nn.ReLU()
        case "leaky_relu":
            res = nn.LeakyReLU()
        case _:
            raise ValueError(f"Activation function {name} not supported.")

    return res
