"""
Test the torch utility functions.
"""
# pylint: disable=missing-function-docstring

import torch

from ctgnn.utils.torch import to_tensor


def test_to_tensor():
    x = torch.tensor([1, 2, 3])
    y = to_tensor(x)
    assert torch.all(torch.eq(x, y))
    assert x.dtype == y.dtype

    x = torch.tensor([1, 2, 3], dtype=torch.float)
    y = to_tensor(x, dtype=torch.float)
    assert torch.all(torch.eq(x, y))

    x = [1, 2, 3]
    y = to_tensor(x)
    assert torch.all(torch.eq(torch.tensor(x), y))
    assert y.dtype == torch.int64

    x = [1, 2, 3]
    y = to_tensor(x, dtype=torch.float)
    assert torch.all(torch.eq(torch.tensor(x, dtype=torch.float), y))
    assert y.dtype == torch.float
