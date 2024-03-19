"""Helper functions for torch"""

from typing import Any

import torch


def to_tensor(x: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Helper function to copy tensors the `right` way i.e. suppress the warning
    that would come if tensors are initialized using a tensor.

    Args:
        x: Array or tensor that should be converted to a tensor
        dtype: The datatype. Defaults to torch.float.

    Returns:
        The tensor
    """
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            return x.clone().detach().to(dtype)
        return x.clone().detach()
    return torch.tensor(x, dtype=dtype)
