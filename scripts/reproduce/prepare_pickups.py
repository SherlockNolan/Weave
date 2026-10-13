#!/usr/bin/env python3
"""Extract the requested robot-reference clips without altering upstream archives."""
from pathlib import Path
import json

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
REQUESTS = [
    ("sub1_tripod_075", "train", "tripod"),
    ("sub16_clothesstand_012", "test", "clothesstand"),
    ("sub9_clothesstand_039", "train", "clothesstand"),
    ("sub7_whitechair_072", "train", "whitechair"),
]
FRAME_FIELDS = [
    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w",
    "body_ang_vel_w", "object_pos_w", "object_quat_w", "object_lin_vel_w",
    "object_ang_vel_w", "contact_label",
]


def main():
    out = REPO / "data/pickups"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"requested_sequences": [r[0] for r in REQUESTS],
                "missing_sequences": ["sub6_woodchair_066"], "clips": [], "files": []}
    for obj in dict.fromkeys(r[2] for r in REQUESTS):
        chunks = {key: [] for key in FRAME_FIELDS}
        names, objects, lengths = [], [], []
        for sequence, split, object_name in REQUESTS:
            if object_name != obj:
                continue
            source = next((REPO / "data" / split).glob(obj + "*.npz"))
            with np.load(source, allow_pickle=False) as archive:
                all_names = archive["motion_names"].astype(str)
                all_lengths = archive["motion_lengths"]
                starts = np.concatenate(([0], np.cumsum(all_lengths)))
                indices = [i for i, name in enumerate(all_names)
                           if name == sequence or name.startswith(sequence + "_v")]
                if not indices:
                    raise RuntimeError(f"Missing reference: {sequence}")
                assert int(archive["fps"][0]) == 50
                for key in FRAME_FIELDS:
                    array = archive[key]
                    chunks[key].extend(array[starts[i]:starts[i + 1]].copy() for i in indices)
                for i in indices:
                    assert str(archive["object_names"][i]) == obj
                    names.append(str(all_names[i])); objects.append(obj); lengths.append(int(all_lengths[i]))
                    manifest["clips"].append({"sequence": sequence, "motion_name": str(all_names[i]),
                                              "source": str(source.relative_to(REPO)), "source_split": split,
                                              "frames": int(all_lengths[i]), "object": obj})
        arrays = {key: np.concatenate(value) for key, value in chunks.items()}
        frames = sum(lengths)
        assert all(a.shape[0] == frames and a.dtype == np.float32 for a in arrays.values())
        arrays.update(motion_lengths=np.asarray(lengths, dtype=np.int64),
                      motion_names=np.asarray(names), object_names=np.asarray(objects),
                      fps=np.asarray([50], dtype=np.int64))
        path = out / f"{obj}_pickups.npz"
        np.savez_compressed(path, **arrays)
        manifest["files"].append({"path": str(path.relative_to(REPO)), "object": obj,
                                  "clips": len(names), "frames": frames})
        print(f"Prepared {path.name}: {len(names)} clips, {frames} frames", flush=True)
        if obj == "whitechair":
            vertices = np.asarray(trimesh.load(
                REPO / "source/g1_hoi_learning/g1_hoi_learning/objects/whitechair/whitechair.obj",
                force="mesh", process=False).vertices)
            q, pos = arrays["object_quat_w"][0], arrays["object_pos_w"][0]
            world = vertices + 2 * np.cross(q[1:], np.cross(q[1:], vertices) + q[0] * vertices) + pos
            bottom = float(world[:, 2].min())
            supports = world[world[:, 2] <= bottom + 0.01]
            xy_min, xy_max = supports[:, :2].min(0) - 0.04, supports[:, :2].max(0) + 0.04
            top, thickness = bottom - 0.002, 0.04
            assert top > thickness
            manifest["support_table"] = {
                "object": "whitechair", "sequence": "sub7_whitechair_072",
                "size": [float(xy_max[0] - xy_min[0]), float(xy_max[1] - xy_min[1]), thickness],
                "position": [float((xy_min[0] + xy_max[0]) / 2),
                             float((xy_min[1] + xy_max[1]) / 2), top - thickness / 2],
                "top_height": top,
                "note": "Fixed support surface fitted to the published clip, not recovered OMOMO furniture.",
                "starts_with_hand_contact": bool((arrays["contact_label"][0, 30:] > 0).any()),
            }
    assert len(manifest["clips"]) == 12
    manifest["total_clips"] = len(manifest["clips"])
    manifest["paper_test_sequence_used_for_specialist_only"] = "sub16_clothesstand_012"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["support_table"], indent=2), flush=True)


if __name__ == "__main__":
    main()
