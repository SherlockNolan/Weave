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


def normal_contact_force_matrix(sensor, *, history: bool = False) -> torch.Tensor:
    """Return filtered normal forces in N, preserving the Lab 2 contact definition."""
    name = "normal_force_matrix_w" if ISAACLAB3 else "force_matrix_w"
    if history:
        name += "_history"
    return as_torch(getattr(sensor.data, name))


def net_normal_contact_forces(sensor, *, history: bool = False) -> torch.Tensor:
    """Return net normal forces in N; Lab 3's net_forces_w also includes friction."""
    name = "net_normal_forces_w" if ISAACLAB3 else "net_forces_w"
    if history:
        name += "_history"
    return as_torch(getattr(sensor.data, name))
