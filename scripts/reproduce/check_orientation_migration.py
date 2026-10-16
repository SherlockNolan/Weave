#!/usr/bin/env python3
"""Compare policy rotation features using the actual Lab 2 and Lab 3 math implementations."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def load_math(name, root):
    path = root / "source/isaaclab/isaaclab/utils/math.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def features(math, q, obj, pos, obj_pos, velocity):
    relative_pos, relative_q = math.subtract_frame_transforms(pos, q, obj_pos, obj)
    return {
        "rotation_matrix": math.matrix_from_quat(q),
        "local_velocity": math.quat_apply_inverse(q, velocity),
        "relative_position": relative_pos,
        "relative_rotation6d": math.matrix_from_quat(relative_q)[..., :2].reshape(len(q), -1),
    }


def main():
    old = load_math("weave_lab2_math", REPO.parent / "IsaacLab")
    new = load_math("weave_lab3_math", REPO.parent / "IsaacLab-3.0-latest")
    maxima = {}
    total = 0
    for path in sorted((REPO / "data/pickups").glob("*_pickups.npz")):
        with np.load(path, allow_pickle=True) as data:
            q = torch.tensor(data["body_quat_w"][:, 0])
            obj = torch.tensor(data["object_quat_w"])
            pos = torch.tensor(data["body_pos_w"][:, 0])
            obj_pos = torch.tensor(data["object_pos_w"])
            velocity = torch.tensor(data["object_lin_vel_w"])
        expected = features(old, q, obj, pos, obj_pos, velocity)
        actual = features(new, q[:, [1, 2, 3, 0]], obj[:, [1, 2, 3, 0]], pos, obj_pos, velocity)
        for key in expected:
            error = float((expected[key] - actual[key]).abs().max())
            maxima[key] = max(maxima.get(key, 0), error)
            torch.testing.assert_close(actual[key], expected[key], atol=5e-6, rtol=5e-6)
        total += len(q)
    if not total:
        raise RuntimeError("Pickup reference files are missing")
    report = {"frames": total, "max_absolute_error": maxima, "passed": True}
    target = REPO / "logs/preparation/isaac62-orientation-check.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
