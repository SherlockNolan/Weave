"""Version boundaries for the published references and Isaac Lab state arrays."""

from importlib.metadata import version

import torch

ISAACLAB3 = int(version("isaaclab").split(".", 1)[0]) >= 3


def as_torch(value):
    """Return a zero-copy tensor for Lab 3 ProxyArray, or the existing Lab 2 tensor."""
    return value if isinstance(value, torch.Tensor) else value.torch


def reference_quaternion(value):
    """NPZ files use wxyz; Lab 3 physics and quaternion math use xyzw."""
    return value[..., [1, 2, 3, 0]] if ISAACLAB3 else value
