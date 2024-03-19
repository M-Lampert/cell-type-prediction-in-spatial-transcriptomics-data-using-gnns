"""
Test the utils module in the nn package.
"""
# pylint: disable=missing-function-docstring

import pytest
from torch import nn

from ctgnn.nn.utils import get_activation


def test_get_activation():
    activation = get_activation("relu")
    assert isinstance(activation, nn.ReLU)

    activation = get_activation("leaky_relu")
    assert isinstance(activation, nn.LeakyReLU)

    with pytest.raises(ValueError):
        activation = get_activation("invalid")
