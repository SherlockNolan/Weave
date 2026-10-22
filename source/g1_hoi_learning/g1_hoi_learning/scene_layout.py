"""Keep Isaac Lab clone layouts aligned with Weave's reference assignments."""

from collections import Counter
from math import isqrt
from pathlib import Path

import numpy as np


def round_robin_clone_strategy(weights: np.ndarray, num_clones: int) -> np.ndarray:
    """Cycle equally weighted world prototypes, as the original asset spawner did."""
    if not len(weights) or not np.allclose(weights, weights[0]):
        raise ValueError("Weave's reference assignment requires equally weighted object variants.")
    return np.arange(num_clones, dtype=np.int32) % len(weights)


def paired_round_robin_clone_strategy(weights: np.ndarray, num_clones: int) -> np.ndarray:
    """Pair object i with support-table variant i instead of sampling their Cartesian product."""
    num_objects = isqrt(len(weights))
    if not num_objects or num_objects**2 != len(weights):
        raise ValueError("Paired object/table cloning requires two equally sized variant sets.")
    objects = round_robin_clone_strategy(weights, num_clones) % num_objects
    return objects * (num_objects + 1)


def validate_object_assignments(stage, expected_objects: list[str], object_usd_paths: dict[str, str]) -> dict[str, int]:
    """Check the actual USD source of every object's prim before training can start."""
    sources = {name: str(Path(path).resolve()) for name, path in object_usd_paths.items()}
    mismatches = []
    for index, expected in enumerate(expected_objects):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{index}/Object")
        layers = {spec.layer.realPath for spec in prim.GetPrimStack()} if prim.IsValid() else set()
        if sources[expected] not in layers:
            actual = [name for name, source in sources.items() if source in layers]
            mismatches.append(f"env {index}: reference={expected}, spawned={actual or 'unresolved'}")
    if mismatches:
        raise RuntimeError(
            f"Object/reference mismatch in {len(mismatches)}/{len(expected_objects)} environments: "
            + "; ".join(mismatches[:8])
        )
    return dict(Counter(expected_objects))
