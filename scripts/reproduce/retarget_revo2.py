#!/usr/bin/env python3
"""CPU-only Inspire -> Revo2 reference pilot using the official dex-retargeting optimizer.

Outputs carry explicit names and full URDF frames. They are review artifacts,
not drop-in Isaac training archives: contact and PhysX layout must be validated
before any training integration or full-dataset promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from prepare_revo2 import ASSEMBLED_URDF, BODY_URDF, REPO, TARGET
from retarget_dataset_io import get_source_layout, load_clip
from revo2_model import UrdfModel
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

TOOL_ROOT = Path("/mnt/data/ruofei/zhangyu/code/LocoManipulation/repo/hand_retargeting")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
REQUESTS = {
    "sub1_tripod_075": ("tripod", "train"),
    "sub16_clothesstand_012": ("clothesstand", "test"),
    "sub9_clothesstand_039": ("clothesstand", "train"),
    "sub7_whitechair_072": ("whitechair", "train"),
}


def poses(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    transforms = np.tile(np.eye(4), (len(position), 1, 1))
    transforms[:, :3, 3] = position
    transforms[:, :3, :3] = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix()
    return transforms


def tip_names(source: bool) -> list[str]:
    if source:
        return [f"{side}_{finger}_tip" for side in ("L", "R") for finger in FINGERS]
    return [f"{side}_{finger}_tip_link" for side in ("left", "right") for finger in FINGERS]


def optimizer(method: str, metadata: dict):
    from dex_retargeting.retargeting_config import RetargetingConfig

    config = {
        "type": method,
        "urdf_path": str(ASSEMBLED_URDF),
        "target_joint_names": metadata["hand_drive_joints"],
        "normal_delta": 0.0,
        "low_pass_alpha": -1,
        "has_joint_limits": True,
        "ignore_mimic_joint": False,
    }
    if method == "position":
        config.update(target_link_names=tip_names(False), target_link_human_indices=np.arange(10))
    else:
        config.update(
            target_origin_link_names=[f"{side}_hand_base_link" for side in ("left", "right") for _ in FINGERS],
            target_task_link_names=tip_names(False),
            target_link_human_indices=np.tile(np.arange(10), (2, 1)),
        )
    retarget = RetargetingConfig.from_dict(config).build()
    retarget.optimizer.opt.set_maxeval(80)
    retarget.optimizer.opt.set_ftol_abs(1e-9)
    limits = retarget.joint_limits.copy()
    for index, name in enumerate(retarget.optimizer.target_joint_names):
        for mimic_name, spec in metadata["mimic"].items():
            if spec["driver"] == name:
                bound = metadata["hand_joint_limits"][mimic_name]
                child = np.array([bound["lower"], bound["upper"]])
                inverse = np.sort((child - spec["offset"]) / spec["multiplier"])
                limits[index, 0] = max(limits[index, 0], inverse[0])
                limits[index, 1] = min(limits[index, 1], inverse[1])
    retarget.joint_limits = limits
    retarget.optimizer.set_joint_limit(limits, epsilon=0)
    return retarget


def baseline_controls(source: dict, metadata: dict, source_model: UrdfModel) -> dict:
    values = {}
    for side, short in (("left", "L"), ("right", "R")):
        mappings = [
            (f"{side}_thumb_metacarpal_joint", f"{short}_thumb_proximal_yaw_joint"),
            (f"{side}_thumb_proximal_joint", f"{short}_thumb_proximal_pitch_joint"),
        ]
        mappings += [(f"{side}_{finger}_proximal_joint", f"{short}_{finger}_proximal_joint") for finger in FINGERS[1:]]
        for target_name, old_name in mappings:
            old_limit = source_model.joints[old_name].find("limit").attrib
            fraction = np.clip(
                (source[old_name] - float(old_limit["lower"]))
                / (float(old_limit["upper"]) - float(old_limit["lower"])),
                0,
                1,
            )
            limit = metadata["hand_joint_limits"][target_name]
            values[target_name] = float(limit["lower"] + fraction * (limit["upper"] - limit["lower"]))
    return values


def rate_project(values: np.ndarray, names: list[str], metadata: dict, fps: int) -> np.ndarray:
    """Keep clip timing; project onto active and dependent joint speed bounds."""
    result = values.copy()
    limits = metadata["hand_joint_limits"]
    if len(result) >= 9:
        result = savgol_filter(result, 9, 2, axis=0, mode="interp")
    rates = []
    for index, name in enumerate(names):
        bound = limits[name]
        result[:, index] = np.clip(result[:, index], bound["lower"], bound["upper"])
        rate = bound["velocity"]
        for mimic_name, spec in metadata["mimic"].items():
            if spec["driver"] == name:
                rate = min(rate, limits[mimic_name]["velocity"] / abs(spec["multiplier"]))
        rates.append(rate)
    step = np.asarray(rates) / fps
    for _ in range(2):
        for frame in range(1, len(result)):
            result[frame] = np.clip(result[frame], result[frame - 1] - step, result[frame - 1] + step)
        for frame in range(len(result) - 2, -1, -1):
            result[frame] = np.clip(result[frame], result[frame + 1] - step, result[frame + 1] + step)
    return result


def contact_mapping(target: UrdfModel, layout: dict) -> dict[str, list[str]]:
    source_bodies = set(layout["body_names"])
    mapped = {}
    for name in target.links:
        if name in source_bodies:
            mapped[name] = [name]
            continue
        if name in layout["fixed_link_to_body"]:
            mapped[name] = [layout["fixed_link_to_body"][name]]
            continue
        for side, short in (("left", "L"), ("right", "R")):
            if not name.startswith(side + "_"):
                continue
            if name in (f"{side}_hand_base_link", f"{side}_base2_link"):
                mapped[name] = [f"{side}_wrist_yaw_link"]
            for finger in FINGERS:
                if name.startswith(f"{side}_{finger}_"):
                    if finger == "thumb":
                        suffix = name.removeprefix(f"{side}_thumb_")
                        if suffix == "metacarpal_link":
                            mapped[name] = [f"{short}_thumb_proximal_base"]
                        elif suffix == "proximal_link":
                            mapped[name] = [f"{short}_thumb_proximal"]
                        else:
                            mapped[name] = [f"{short}_thumb_intermediate", f"{short}_thumb_distal"]
                    else:
                        part = "proximal" if "proximal" in name else "intermediate"
                        mapped[name] = [f"{short}_{finger}_{part}"]
    return mapped


def convert_clip(args, sequence: str) -> dict:
    object_name, split = REQUESTS[sequence]
    source_path = REPO / f"data/pickups/{object_name}_pickups.npz"
    source, info = load_clip(source_path, sequence, args.variant)
    info["source_split"] = split
    layout = get_source_layout()
    metadata = json.loads((TARGET / "assembly.json").read_text())
    old_model, new_model = UrdfModel(BODY_URDF), UrdfModel(ASSEMBLED_URDF)
    retarget = optimizer(args.method, metadata)
    target_joint_names = [name for name, joint in new_model.joints.items() if joint.attrib["type"] != "fixed"]
    body_names = list(new_model.links)
    source_joint_names = layout["joint_names"]
    pelvis = layout["body_names"].index("pelvis")
    root_poses = poses(source["body_pos_w"][:, pelvis], source["body_quat_w"][:, pelvis])
    frame_count, fps = info["frames"], info["fps"]
    if args.max_frames:
        frame_count = min(frame_count, args.max_frames)
        source = {
            key: value[:frame_count]
            if key
            in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "object_pos_w",
                "object_quat_w",
                "object_lin_vel_w",
                "object_ang_vel_w",
                "contact_label",
            )
            else value
            for key, value in source.items()
        }
        source["motion_lengths"] = np.array([frame_count], dtype=np.int64)
        root_poses = root_poses[:frame_count]
    active = list(retarget.optimizer.target_joint_names)
    raw = np.empty((frame_count, len(active)))
    baseline = np.empty_like(raw)
    reference_tips = np.empty((frame_count, 10, 3))
    statuses = []
    started = time.perf_counter()
    for frame in range(frame_count):
        old_values = dict(zip(source_joint_names, source["joint_pos"][frame]))
        old_fk = old_model.fk(old_values)
        reference_tips[frame] = np.stack([old_fk[name][:3, 3] for name in tip_names(True)])
        base = baseline_controls(old_values, metadata, old_model)
        baseline[frame] = [base[name] for name in active]
        if frame == 0:
            retarget.last_qpos = np.clip(baseline[frame], retarget.joint_limits[:, 0], retarget.joint_limits[:, 1])
        fixed = np.array([old_values[name] for name in retarget.optimizer.fixed_joint_names])
        desired = reference_tips[frame]
        if args.method == "vector":
            palm = np.stack([old_fk[f"{side}_hand_base_link"][:3, 3] for side in ("L", "R") for _ in FINGERS])
            desired = desired - palm
        qpos = retarget.retarget(desired, fixed_qpos=fixed)
        robot_names = retarget.joint_names
        lookup = dict(zip(robot_names, qpos))
        raw[frame] = [lookup[name] for name in active]
        statuses.append(int(retarget.optimizer.opt.last_optimize_result()))
        if (frame + 1) % 100 == 0 or frame == frame_count - 1:
            print(
                f"{info['motion_name']}: {frame + 1}/{frame_count} frames, {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    controls = rate_project(raw, active, metadata, fps)
    joint_pos = np.empty((frame_count, len(target_joint_names)), dtype=np.float32)
    body_pos = np.empty((frame_count, len(body_names), 3), dtype=np.float32)
    body_quat = np.empty((frame_count, len(body_names), 4), dtype=np.float32)
    tip_error, baseline_error, raw_tip_error = [], [], []
    for frame in range(frame_count):
        old_values = dict(zip(source_joint_names, source["joint_pos"][frame]))
        values = {name: old_values[name] for name in metadata["body_drive_joints"]}
        values.update(dict(zip(active, controls[frame])))
        all_values = new_model.configuration(values)
        joint_pos[frame] = [all_values[name] for name in target_joint_names]
        frames = new_model.fk(values)
        current_tips = np.stack([frames[name][:3, 3] for name in tip_names(False)])
        tip_error.append(np.linalg.norm(current_tips - reference_tips[frame], axis=1))
        baseline_values = dict(values)
        baseline_values.update(dict(zip(active, baseline[frame])))
        baseline_fk = new_model.fk(baseline_values)
        baseline_tips = np.stack([baseline_fk[name][:3, 3] for name in tip_names(False)])
        baseline_error.append(np.linalg.norm(baseline_tips - reference_tips[frame], axis=1))
        raw_values = dict(values)
        raw_values.update(dict(zip(active, raw[frame])))
        raw_fk = new_model.fk(raw_values)
        raw_tips = np.stack([raw_fk[name][:3, 3] for name in tip_names(False)])
        raw_tip_error.append(np.linalg.norm(raw_tips - reference_tips[frame], axis=1))
        world = np.stack([root_poses[frame] @ frames[name] for name in body_names])
        body_pos[frame] = world[:, :3, 3]
        body_quat[frame] = Rotation.from_matrix(world[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
    mapping = contact_mapping(new_model, layout)
    labels = np.zeros((frame_count, len(body_names)), dtype=np.float32)
    for index, name in enumerate(body_names):
        if name in mapping:
            original = source["contact_label"][:, [layout["body_names"].index(old) for old in mapping[name]]]
            labels[:, index] = np.where((original > 0).any(axis=1), 1, np.where((original < 0).all(axis=1), -1, 0))
    joint_vel = np.gradient(joint_pos, 1 / fps, axis=0).astype(np.float32)
    # The source body angles can wrap by 2*pi. Preserve their provided velocity
    # channels instead of introducing derivative spikes at those discontinuities.
    for name in metadata["body_drive_joints"]:
        joint_vel[:, target_joint_names.index(name)] = source["joint_vel"][:, source_joint_names.index(name)]
    body_linear = np.gradient(body_pos, 1 / fps, axis=0).astype(np.float32)
    body_angular = np.zeros_like(body_pos)
    for body in range(len(body_names)):
        rotations = Rotation.from_quat(body_quat[:, body, [1, 2, 3, 0]])
        rate = (rotations[1:] * rotations[:-1].inv()).as_rotvec() * fps
        body_angular[0, body], body_angular[-1, body] = rate[0], rate[-1]
        body_angular[1:-1, body] = (rate[:-1] + rate[1:]) / 2
    arrays = {
        key: value.copy()
        for key, value in source.items()
        if key.startswith("object_") or key in ("fps", "motion_lengths", "motion_names", "object_names")
    }
    arrays.update(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_linear,
        body_ang_vel_w=body_angular,
        contact_label=labels,
        joint_names=np.asarray(target_joint_names),
        body_names=np.asarray(body_names),
        source_joint_names=np.asarray(source_joint_names),
        source_body_names=np.asarray(layout["body_names"]),
        preview_only=np.asarray(True),
        quaternion_order=np.asarray("wxyz"),
        schema_version=np.asarray("weave-revo2-kinematic-pilot-v1"),
        hand_control_names=np.asarray(active),
        raw_hand_control_pos=raw.astype(np.float32),
        baseline_hand_control_pos=baseline.astype(np.float32),
        optimizer_status=np.asarray(statuses, dtype=np.int16),
    )
    for key in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "contact_label",
    ):
        arrays["source_" + key] = source[key]
    output = args.output / f"{info['motion_name']}.npz"
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    error, base_error = np.asarray(tip_error), np.asarray(baseline_error)
    report = {
        **info,
        "frames_processed": frame_count,
        "full_clip_retained": frame_count == info["frames"],
        "processed_duration_seconds": frame_count / fps,
        "method": args.method,
        "output": str(output),
        "status": "pilot_requires_contact_and_physics_review",
        "train_ready": False,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "target_urdf_sha256": hashlib.sha256(ASSEMBLED_URDF.read_bytes()).hexdigest(),
        "source_layout": layout,
        "target_joint_names": target_joint_names,
        "target_body_names": body_names,
        "contact_label_mapping": mapping,
        "mean_tip_error_m": float(error.mean()),
        "p95_tip_error_m": float(np.quantile(error, 0.95)),
        "baseline_mean_tip_error_m": float(base_error.mean()),
        "baseline_p95_tip_error_m": float(np.quantile(base_error, 0.95)),
        "raw_before_speed_projection_mean_tip_error_m": float(np.asarray(raw_tip_error).mean()),
        "optimizer_negative_status_frames": sum(status < 0 for status in statuses),
        "optimizer_status_counts": {str(status): statuses.count(status) for status in sorted(set(statuses))},
        "processing_seconds": time.perf_counter() - started,
        "timing": "50Hz clip timing unchanged; all frames retained; new hand active and mimic speed limits applied",
        "body_trajectory": "29 original body joint positions copied by audited joint names",
        "body_joint_velocities": "Original body velocity channels copied; no finite difference across angle wraps",
        "contact_labels": "Semantic mapping only; actual geometric contacts have not been relabeled as success",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "data/revo2/pilot_position")
    parser.add_argument("--sequences", nargs="+", choices=list(REQUESTS), default=list(REQUESTS))
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument("--method", choices=("position", "vector"), default="position")
    parser.add_argument("--max-frames", type=int, default=0, help="Debug only; default retains full clips.")
    args = parser.parse_args()
    if args.max_frames < 0 or args.max_frames == 1:
        parser.error("--max-frames must be zero (full clip) or at least two")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("Set CUDA_VISIBLE_DEVICES='' for this CPU-only preparation job.")
    import torch

    torch.set_num_threads(1)
    import dex_retargeting
    import nlopt
    import pinocchio
    import scipy

    runtime = {
        "python": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "pinocchio": pinocchio.__version__,
        "nlopt": nlopt.__version__,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "dex_retargeting_module": str(Path(dex_retargeting.__file__).resolve()),
    }
    tool_commit = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={TOOL_ROOT / 'dex-retargeting'}",
            "-C",
            str(TOOL_ROOT / "dex-retargeting"),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    reports = [convert_clip(args, sequence) for sequence in args.sequences]
    summary = {
        "tool_repository": "https://github.com/dexsuite/dex-retargeting",
        "tool_commit": tool_commit,
        "method": args.method,
        "runtime": runtime,
        "clips": reports,
        "total_frames": sum(report["frames_processed"] for report in reports),
        "full_dataset_promoted": False,
        "reason": "Pilot contacts, collision, and dynamics must pass before full conversion.",
    }
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "clips": len(reports),
                "mean_tip_error_m": {r["motion_name"]: r["mean_tip_error_m"] for r in reports},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
