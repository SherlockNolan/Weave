#!/usr/bin/env python3
"""Validate motion metadata and array headers without moving the dataset to the GPU."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIELDS = {
    "joint_pos": (53,), "joint_vel": (53,),
    "body_pos_w": (54, 3), "body_quat_w": (54, 4),
    "body_lin_vel_w": (54, 3), "body_ang_vel_w": (54, 3),
    "object_pos_w": (3,), "object_quat_w": (4,),
    "object_lin_vel_w": (3,), "object_ang_vel_w": (3,), "contact_label": (54,),
}


def main():
    report = {}
    for split, expected_clips in (("train", 7869), ("test", 1605)):
        files = sorted((REPO / "data" / split).glob("*.npz"))
        if len(files) != 9:
            raise RuntimeError(f"Expected nine {split} archives, found {len(files)}")
        results = []
        for path in files:
            with np.load(path, allow_pickle=False) as archive:
                lengths = archive["motion_lengths"]
                objects = archive["object_names"]
                names = archive["motion_names"]
                assert lengths.ndim == 1 and np.all(lengths > 0), path
                assert len(lengths) == len(objects) == len(names), path
                assert int(archive["fps"].reshape(-1)[0]) == 50, path
                assert len(set(objects.tolist())) == 1, path
                frames = int(lengths.sum())
            raw_bytes = 0
            with zipfile.ZipFile(path) as zipped:
                for field, tail in FIELDS.items():
                    with zipped.open(field + ".npy") as stream:
                        version = np.lib.format.read_magic(stream)
                        if version == (1, 0):
                            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                        elif version == (2, 0):
                            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                        else:
                            raise RuntimeError(f"Unsupported NPY header: {path}:{field}:{version}")
                    assert shape == (frames, *tail), (path, field, shape)
                    assert dtype == np.dtype("float32"), (path, field, dtype)
                    raw_bytes += int(np.prod(shape)) * dtype.itemsize
            object_name = str(objects[0])
            asset_dir = REPO / "source/g1_hoi_learning/g1_hoi_learning/objects" / object_name
            assert np.load(asset_dir / "surface.npy", allow_pickle=False).shape == (512, 3), asset_dir
            with np.load(asset_dir / "sdf_128.npz", allow_pickle=False) as sdf:
                assert sdf["sdf"].shape == (128, 128, 128), asset_dir
            results.append({"file": str(path.relative_to(REPO)), "object": object_name, "clips": len(lengths), "frames": frames, "float_buffer_bytes": raw_bytes})
            print(f"Validated {path.name}: {len(lengths)} clips, {frames} frames, {raw_bytes/1024**3:.2f} GiB float buffers", flush=True)
        clips = sum(item["clips"] for item in results)
        assert clips == expected_clips, (split, clips, expected_clips)
        report[split] = {"clips": clips, "float_buffer_bytes": sum(item["float_buffer_bytes"] for item in results), "files": results}
    basis = np.load(REPO / "source/g1_hoi_learning/g1_hoi_learning/objects/geometry/bps_128.npy", allow_pickle=False)
    assert basis.shape == (128, 3), basis.shape
    (REPO / "logs/preparation/dataset.validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Reference splits, object geometry, and motion array headers passed validation.", flush=True)


if __name__ == "__main__":
    main()
