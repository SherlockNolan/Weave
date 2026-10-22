#!/usr/bin/env python3
"""Independently review Revo2 pilot references on CPU, without running a policy.

Robot collision shapes use URDF convex hulls. Object collision uses the original
triangle mesh (FCL BVH), never a convex hull of the entire nonconvex object. These
are offline geometric diagnostics, not PhysX contact forces or task success.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from prepare_revo2 import ASSEMBLED_URDF, BODY_URDF, REPO
from revo2_model import UrdfModel, origin
from scipy.spatial.transform import Rotation


def names(data, key):
    return [str(value) for value in data[key]]


def object_mesh_path(object_name):
    return REPO / "source/g1_hoi_learning/g1_hoi_learning/objects" / object_name / f"{object_name}.obj"


def hand_link(name):
    return name.startswith(("L_", "R_")) or (
        name.startswith(("left_", "right_"))
        and any(word in name for word in ("hand", "thumb", "index", "middle", "ring", "pinky"))
    )


def hand_contact_link(name):
    # Source PhysX merges each fixed palm into the wrist-yaw body.
    return hand_link(name) or name in ("left_wrist_yaw_link", "right_wrist_yaw_link")


def hand_side(name):
    if name.startswith(("L_", "left_")):
        return "left"
    if name.startswith(("R_", "right_")):
        return "right"
    return None


def comparable_collision_group(name):
    """Compare corresponding fingers despite different phalanx counts and names."""
    if not hand_link(name):
        return name
    side = hand_side(name)
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        if finger in name.lower():
            return f"{side}_{finger}"
    return f"{side}_palm"


def pose_matrix(position, quaternion):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = position
    return result


def frame_poses(model, data, frame, prefix=""):
    """Use stored world frames, and direct measured-joint FK for missing fixed links.

    Deliberately do not overwrite measured source joints with URDF mimic values.
    Source reference trajectories can have a nonzero mimic residual.
    """
    body_names = names(data, prefix + "body_names")
    positions = data[prefix + "body_pos_w"][frame]
    quaternions = data[prefix + "body_quat_w"][frame]
    poses = {name: pose_matrix(position, quat) for name, position, quat in zip(body_names, positions, quaternions)}
    joints = dict(zip(names(data, prefix + "joint_names"), data[prefix + "joint_pos"][frame]))
    if model.root_link not in poses:
        raise ValueError(f"Missing root body {model.root_link} in {prefix}body_names")
    for joint in model.ordered_joints:
        child = joint.find("child").attrib["link"]
        if child in poses:
            continue
        parent = joint.find("parent").attrib["link"]
        motion = np.eye(4)
        if joint.attrib["type"] != "fixed":
            axis = np.fromstring(joint.find("axis").attrib.get("xyz", "1 0 0"), sep=" ")
            axis /= np.linalg.norm(axis)
            value = joints[joint.attrib["name"]]
            if joint.attrib["type"] in ("revolute", "continuous"):
                motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
            elif joint.attrib["type"] == "prismatic":
                motion[:3, 3] = axis * value
            else:
                raise ValueError(f"Unsupported joint {joint.attrib}")
        poses[child] = poses[parent] @ origin(joint.find("origin")) @ motion
    return poses


def array_review(model, data, prefix=""):
    joint_names = names(data, prefix + "joint_names")
    body_names = names(data, prefix + "body_names")
    positions = data[prefix + "joint_pos"]
    velocities = data[prefix + "joint_vel"]
    frames = len(positions)
    if len(set(joint_names)) != len(joint_names) or len(set(body_names)) != len(body_names):
        raise ValueError(f"Duplicate joint/body names: {prefix}")
    assert positions.shape == velocities.shape == (frames, len(joint_names))
    assert data[prefix + "body_pos_w"].shape == (frames, len(body_names), 3)
    assert data[prefix + "body_quat_w"].shape == (frames, len(body_names), 4)
    expected = {name for name, joint in model.joints.items() if joint.attrib["type"] != "fixed"}
    if set(joint_names) != expected:
        raise ValueError(
            f"{prefix}joint_names mismatch: missing={expected - set(joint_names)}, extra={set(joint_names) - expected}"
        )
    fields = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
    nonfinite = {field: int(np.count_nonzero(~np.isfinite(data[prefix + field]))) for field in fields}
    if any(nonfinite.values()):
        raise ValueError(f"Non-finite reference arrays: {prefix}{nonfinite}")
    limits = []
    mimic_errors = []
    velocity_limits = []
    index = {name: i for i, name in enumerate(joint_names)}
    for name, i in index.items():
        joint = model.joints[name]
        limit = joint.find("limit")
        if limit is not None and joint.attrib["type"] != "continuous":
            error = np.maximum(
                float(limit.attrib["lower"]) - positions[:, i], positions[:, i] - float(limit.attrib["upper"])
            )
            limits.append(
                {
                    "joint": name,
                    "max_violation_rad": max(0.0, float(error.max())),
                    "frames_over_1mrad": int(np.count_nonzero(error > 0.001)),
                }
            )
        if limit is not None and "velocity" in limit.attrib:
            velocity_limit = float(limit.attrib["velocity"])
            excess = np.abs(velocities[:, i]) - velocity_limit
            velocity_limits.append(
                {
                    "joint": name,
                    "limit_rad_s": velocity_limit,
                    "maximum_abs_velocity_rad_s": float(np.max(np.abs(velocities[:, i]))),
                    "maximum_excess_rad_s": max(0.0, float(excess.max())),
                    "frames_over_limit_plus_1mrad_s": int(np.count_nonzero(excess > 0.001)),
                }
            )
        mimic = joint.find("mimic")
        if mimic is not None:
            expected_position = positions[:, index[mimic.attrib["joint"]]] * float(
                mimic.attrib.get("multiplier", 1)
            ) + float(mimic.attrib.get("offset", 0))
            mimic_errors.append(float(np.max(np.abs(positions[:, i] - expected_position))))
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    derivative = np.gradient(positions.astype(np.float64), 1 / fps, axis=0)
    quat_norm_error = np.abs(np.linalg.norm(data[prefix + "body_quat_w"], axis=-1) - 1)
    contact_review = None
    if prefix + "contact_label" in data:
        labels = data[prefix + "contact_label"]
        if labels.shape != (frames, len(body_names)) or not np.isfinite(labels).all():
            raise ValueError(f"Invalid {prefix}contact_label shape or values")
        contact_review = {
            "unique_values": np.unique(labels).tolist(),
            "desired_contact_entries": int(np.count_nonzero(labels > 0)),
            "avoid_contact_entries": int(np.count_nonzero(labels < 0)),
            "semantics": "+1 desired contact, -1 avoid contact, 0 unconstrained; labels are not measured force.",
        }
    return {
        "frames": frames,
        "joints": len(joint_names),
        "bodies": len(body_names),
        "nonfinite_values": nonfinite,
        "maximum_quaternion_norm_error": float(quat_norm_error.max()),
        "joint_limits": sorted(limits, key=lambda row: -row["max_violation_rad"]),
        "joint_velocity_limits": sorted(velocity_limits, key=lambda row: -row["maximum_excess_rad_s"]),
        "maximum_mimic_residual_rad": max(mimic_errors, default=0.0),
        "maximum_abs_joint_velocity_rad_s": float(np.max(np.abs(velocities))),
        "p99_abs_joint_velocity_rad_s": float(np.quantile(np.abs(velocities), 0.99)),
        "velocity_vs_centered_finite_difference_max_rad_s": float(np.max(np.abs(velocities - derivative))),
        "velocity_vs_centered_finite_difference_p99_rad_s": float(np.quantile(np.abs(velocities - derivative), 0.99)),
        "contact_labels": contact_review,
    }


def body_preservation_review(data):
    source_joints = {name: index for index, name in enumerate(names(data, "source_joint_names"))}
    target_joints = {name: index for index, name in enumerate(names(data, "joint_names"))}
    common_joints = sorted(name for name in source_joints.keys() & target_joints.keys() if not hand_link(name))
    source_indices = [source_joints[name] for name in common_joints]
    target_indices = [target_joints[name] for name in common_joints]
    return {
        "common_body_joint_names": common_joints,
        "common_body_joint_count": len(common_joints),
        "maximum_body_joint_position_change_rad": float(
            np.max(np.abs(data["joint_pos"][:, target_indices] - data["source_joint_pos"][:, source_indices]))
        ),
        "maximum_body_joint_velocity_change_rad_s": float(
            np.max(np.abs(data["joint_vel"][:, target_indices] - data["source_joint_vel"][:, source_indices]))
        ),
    }


class CollisionReview:
    def __init__(self, model, object_mesh):
        import fcl

        self.model = model
        self.fcl = fcl
        self.groups, self.adjacent = model.rigid_groups()
        self.shapes = []
        for name, local, mesh in model.geometry("collision"):
            hull = mesh.convex_hull
            faces = np.column_stack((np.full(len(hull.faces), 3), hull.faces)).reshape(-1)
            shape = fcl.Convex(hull.vertices, len(hull.faces), faces)
            self.shapes.append((name, local, hull.vertices, shape))
        self.object_vertices = np.asarray(object_mesh.vertices, dtype=np.float64)
        self.object_shape = fcl.BVHModel()
        self.object_shape.beginModel(len(object_mesh.vertices), len(object_mesh.faces))
        self.object_shape.addSubModel(self.object_vertices, np.asarray(object_mesh.faces, dtype=np.int32))
        self.object_shape.endModel()

    def collision(self, obj_a, obj_b):
        result = self.fcl.CollisionResult()
        self.fcl.collide(obj_a, obj_b, self.fcl.CollisionRequest(num_max_contacts=128, enable_contact=True), result)
        depth = max((max(0.0, float(contact.penetration_depth)) for contact in result.contacts), default=0.0)
        return bool(result.is_collision), depth

    def frame(self, poses, object_pose):
        objects = []
        for name, local, vertices, shape in self.shapes:
            transform = poses[name] @ local
            obj = self.fcl.CollisionObject(shape, self.fcl.Transform(transform[:3, :3], transform[:3, 3]))
            points = vertices @ transform[:3, :3].T + transform[:3, 3]
            objects.append((name, obj, np.array([points.min(axis=0), points.max(axis=0)])))
        contacts = []
        for index, (name_a, obj_a, bounds_a) in enumerate(objects):
            for name_b, obj_b, bounds_b in objects[index + 1 :]:
                if (
                    self.groups[name_a] == self.groups[name_b]
                    or frozenset((self.groups[name_a], self.groups[name_b])) in self.adjacent
                ):
                    continue
                if np.any(bounds_a[1] < bounds_b[0]) or np.any(bounds_b[1] < bounds_a[0]):
                    continue
                collision, depth = self.collision(obj_a, obj_b)
                if collision:
                    contacts.append(
                        {
                            "links": [name_a, name_b],
                            "depth_m": depth,
                            "involves_hand": hand_link(name_a) or hand_link(name_b),
                        }
                    )
        obj = self.fcl.CollisionObject(self.object_shape, self.fcl.Transform(object_pose[:3, :3], object_pose[:3, 3]))
        hand_object = []
        for name, hand_obj, _ in objects:
            if not hand_link(name):
                continue
            # Convex-vs-triangle FCL penetration depths can be nonsensical. Keep
            # only the intersection Boolean and an unsigned noncontact distance.
            result = self.fcl.CollisionResult()
            self.fcl.collide(hand_obj, obj, self.fcl.CollisionRequest(enable_contact=False), result)
            collision = bool(result.is_collision)
            distance = 0.0
            if not collision:
                result = self.fcl.DistanceResult()
                distance = max(0.0, float(self.fcl.distance(hand_obj, obj, self.fcl.DistanceRequest(), result)))
            hand_object.append({"link": name, "collision": collision, "surface_distance_m": distance})
        hand_object_by_side = {}
        for side in ("left", "right"):
            rows = [row for row in hand_object if hand_side(row["link"]) == side]
            hand_object_by_side[side] = {
                "colliding_collision_shapes": sum(row["collision"] for row in rows),
                "minimum_surface_distance_m": min((row["surface_distance_m"] for row in rows), default=float("inf")),
            }
        return {
            "self_contacts": sorted(contacts, key=lambda row: -row["depth_m"]),
            "hand_object": hand_object,
            "hand_object_by_side": hand_object_by_side,
            "maximum_self_depth_m": max((row["depth_m"] for row in contacts), default=0.0),
            "maximum_hand_self_depth_m": max((row["depth_m"] for row in contacts if row["involves_hand"]), default=0.0),
            "minimum_hand_object_surface_distance_m": min(
                (row["surface_distance_m"] for row in hand_object), default=float("inf")
            ),
        }


def summarize_collision(frames):
    result = {"sampled_frames": len(frames)}
    for field in ("maximum_self_depth_m", "maximum_hand_self_depth_m"):
        depths = np.asarray([frame[field] for frame in frames])
        result[field] = float(depths.max())
        result[field.replace("maximum_", "frames_over_2mm_")] = int(np.count_nonzero(depths > 0.002))
        result[field.replace("maximum_", "frames_over_5mm_")] = int(np.count_nonzero(depths > 0.005))
    result["minimum_hand_object_surface_distance_m"] = min(
        frame["minimum_hand_object_surface_distance_m"] for frame in frames
    )
    result["hand_object_collision_frames"] = sum(
        any(row["collision"] for row in frame["hand_object"]) for frame in frames
    )
    return result


def promotion_review(report):
    """Reject obvious regressions; passing these diagnostics never means train-ready."""
    source = report["geometry"]["source"]["frames"]
    target = report["geometry"]["target"]["frames"]
    regressions = []
    desired_contact_gaps = []
    for before, after in zip(source, target):
        baseline = {}
        for contact in before["self_contacts"]:
            if contact["involves_hand"]:
                group = tuple(sorted(comparable_collision_group(name) for name in contact["links"]))
                baseline[group] = max(baseline.get(group, 0.0), contact["depth_m"])
        for contact in after["self_contacts"]:
            if not contact["involves_hand"]:
                continue
            group = tuple(sorted(comparable_collision_group(name) for name in contact["links"]))
            increase = contact["depth_m"] - baseline.get(group, 0.0)
            if increase > 0.005:
                regressions.append(
                    {
                        "frame": after["frame"],
                        "time_s": after["time_s"],
                        "links": contact["links"],
                        "comparable_groups": group,
                        "source_depth_m": baseline.get(group, 0.0),
                        "target_depth_m": contact["depth_m"],
                        "increase_m": increase,
                    }
                )
        desired_sides = {hand_side(name) for name in before.get("source_desired_contact_links", [])}
        for side in desired_sides:
            before_distance = before["hand_object_by_side"][side]["minimum_surface_distance_m"]
            after_distance = after["hand_object_by_side"][side]["minimum_surface_distance_m"]
            if before_distance <= 0.010 and after_distance > 0.020:
                desired_contact_gaps.append(
                    {
                        "frame": after["frame"],
                        "time_s": after["time_s"],
                        "side": side,
                        "source_clearance_m": before_distance,
                        "target_clearance_m": after_distance,
                    }
                )
    target_limits = report["full_frame_arrays"]["target"]["joint_limits"]
    hand_limits = [row for row in target_limits if hand_link(row["joint"]) and row["max_violation_rad"] > 0.001]
    hand_velocity_limits = [
        row
        for row in report["full_frame_arrays"]["target"]["joint_velocity_limits"]
        if hand_link(row["joint"]) and row["maximum_excess_rad_s"] > 0.001
    ]
    body_limits = [row for row in target_limits if not hand_link(row["joint"]) and row["max_violation_rad"] > 0.001]
    return {
        "status": (
            "needs_revision"
            if regressions or hand_limits or hand_velocity_limits or desired_contact_gaps
            else "offline_checks_only_needs_physics_validation"
        ),
        "automatic_full_dataset_promotion": False,
        "new_hand_self_overlap_increase_over_5mm": regressions,
        "target_hand_joint_limit_violations_over_1mrad": hand_limits,
        "target_hand_velocity_limit_violations_over_1mrad_s": hand_velocity_limits,
        "desired_contact_clearance_regressions": desired_contact_gaps,
        "inherited_body_joint_limit_violations_over_1mrad": body_limits,
        "criteria": (
            "Compare per-sampled-frame hand-involved self overlap against corresponding source "
            "finger/palm/body groups (different phalanx counts are grouped by finger); "
            "an increase over 5mm requires revision. New hand joint and velocity limits must hold. "
            "For a desired-contact hand, target clearance over 20mm when source was within 10mm is flagged. "
            "Even if these checks pass, grasp/contact review and physical initialization "
            "validation are required before dataset promotion."
        ),
    }


def plot_report(report, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for prefix, color in (("source", "#64748b"), ("target", "#2563eb")):
        frames = report["geometry"][prefix]["frames"]
        x = [frame["time_s"] for frame in frames]
        axes[0].plot(
            x, [1000 * frame["maximum_hand_self_depth_m"] for frame in frames], "o-", label=prefix, color=color
        )
        axes[1].plot(
            x,
            [1000 * frame["minimum_hand_object_surface_distance_m"] for frame in frames],
            "o-",
            label=prefix,
            color=color,
        )
    axes[0].set_ylabel("Hand self overlap (mm)")
    axes[1].set_ylabel("Hand/object clearance (mm)")
    axes[1].set_xlabel("Reference time (s)")
    axes[0].set_title(report["motion_name"] + " — offline geometry, not policy evaluation")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def snapshot_report(data, models, destination):
    """Render static CPU collision geometry at three trajectory times for review."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    object_name = names(data, "object_names")[0]
    object_mesh = trimesh.load(str(object_mesh_path(object_name)), force="mesh", process=False)
    object_faces = object_mesh.faces[:: max(1, len(object_mesh.faces) // 3000)]
    frame_indices = np.linspace(0, len(data["joint_pos"]) - 1, 3, dtype=int)
    fig = plt.figure(figsize=(14, 8))
    for row, (label, model) in enumerate(models.items()):
        shapes = [
            (name, local, mesh.convex_hull) for name, local, mesh in model.geometry("collision") if hand_link(name)
        ]
        for column, index in enumerate(frame_indices):
            axis = fig.add_subplot(2, 3, row * 3 + column + 1, projection="3d")
            poses = frame_poses(model, data, int(index), "source_" if label == "source" else "")
            points = []
            for name, local, mesh in shapes:
                transform = poses[name] @ local
                vertices = mesh.vertices @ transform[:3, :3].T + transform[:3, 3]
                points.append(vertices)
                axis.add_collection3d(
                    Poly3DCollection(
                        vertices[mesh.faces],
                        facecolors="#c98742" if label == "source" else "#4089c2",
                        edgecolors="#354152",
                        linewidths=0.15,
                        alpha=0.8,
                    )
                )
            object_pose = pose_matrix(data["object_pos_w"][index], data["object_quat_w"][index])
            vertices = object_mesh.vertices @ object_pose[:3, :3].T + object_pose[:3, 3]
            axis.add_collection3d(
                Poly3DCollection(vertices[object_faces], facecolors="#91a96b", edgecolors="none", alpha=0.55)
            )
            hand_points = np.concatenate(points)
            center = (hand_points.min(axis=0) + hand_points.max(axis=0)) / 2
            half_span = max(0.20, float(np.ptp(hand_points, axis=0).max()) * 0.65)
            axis.set_xlim(center[0] - half_span, center[0] + half_span)
            axis.set_ylim(center[1] - half_span, center[1] + half_span)
            axis.set_zlim(center[2] - half_span, center[2] + half_span)
            axis.set_box_aspect((1, 1, 1))
            axis.view_init(elev=25, azim=-55)
            axis.set_title(f"{label}: frame {index}, t={index / float(data['fps'][0]):.2f}s")
            axis.set_xlabel("x (m)")
            axis.set_ylabel("y (m)")
            axis.set_zlabel("z (m)")
    fig.suptitle(
        names(data, "motion_names")[0] + " — collision hull preview, reference only; object display cropped",
        fontsize=12,
    )
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--no-collisions", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--snapshots", type=Path, help="CPU static hand/object collision-mesh previews.")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    with np.load(args.motion, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    frames = len(data["joint_pos"])
    if frames < 2:
        raise ValueError("A review requires at least two frames")
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be finite and positive")
    if len(data["source_joint_pos"]) != frames or int(np.sum(data["motion_lengths"])) != frames:
        raise ValueError("Source, target and motion_lengths frame counts must agree")
    for field in ("object_pos_w", "object_quat_w"):
        if not np.isfinite(data[field]).all() or len(data[field]) != frames:
            raise ValueError(f"Invalid {field}")
    models = {"source": UrdfModel(BODY_URDF), "target": UrdfModel(ASSEMBLED_URDF)}
    report = {
        "motion_file": str(args.motion.resolve()),
        "input_npz_sha256": hashlib.sha256(args.motion.read_bytes()).hexdigest(),
        "review_runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "packages": {
                package: importlib.metadata.version(package)
                for package in ("numpy", "scipy", "trimesh", "python-fcl")
                if package != "python-fcl" or not args.no_collisions
            },
        },
        "motion_name": names(data, "motion_names")[0],
        "purpose": "Offline reference preview; no policy, success rate, forces, or PhysX dynamics evaluated.",
        "full_frame_arrays": {
            name: array_review(model, data, "source_" if name == "source" else "") for name, model in models.items()
        },
        "body_preservation": body_preservation_review(data),
        "object_quaternion_max_norm_error": float(np.max(np.abs(np.linalg.norm(data["object_quat_w"], axis=-1) - 1))),
    }
    if not args.no_collisions:
        object_name = names(data, "object_names")[0]
        mesh_path = object_mesh_path(object_name)
        object_mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        report["geometry"] = {
            "method": (
                "FCL; individual robot URDF collision convex hulls, original object triangle BVH; "
                "fixed-connected and directly adjacent robot bodies excluded."
            ),
            "object_mesh": str(mesh_path),
            "limitations": [
                (
                    "Not PhysX: actual importer filtering, convex decomposition, solver margins, "
                    "ground/support contact and dynamic feasibility are not reproduced."
                ),
                (
                    "FCL convex/triangle contact depths are unreliable and intentionally not reported; "
                    "object intersection Booleans and unsigned noncontact clearances are used instead."
                ),
                (
                    "Self penetration thresholds refer to the selected convex collision geometry; "
                    "convex hulls may overestimate the visible shape."
                ),
                (
                    "Only uniformly sampled frames undergo collision checks; "
                    "full-frame validation applies to arrays and limits."
                ),
                "Geometric proximity or overlap is not successful or stable grasping.",
            ],
        }
        sample_indices = np.unique(np.linspace(0, frames - 1, min(args.samples, frames), dtype=int))
        for label, model in models.items():
            review = CollisionReview(model, object_mesh)
            records = []
            for index in sample_indices:
                record = review.frame(
                    frame_poses(model, data, int(index), "source_" if label == "source" else ""),
                    pose_matrix(data["object_pos_w"][index], data["object_quat_w"][index]),
                )
                record.update(frame=int(index), time_s=float(index / fps))
                if "source_contact_label" in data:
                    labels = data["source_contact_label"][index]
                    contact_names = names(data, "source_body_names")
                    record["source_desired_contact_links"] = [
                        name for name, value in zip(contact_names, labels) if value > 0 and hand_contact_link(name)
                    ]
                    record["source_avoid_contact_links"] = [
                        name for name, value in zip(contact_names, labels) if value < 0 and hand_contact_link(name)
                    ]
                records.append(record)
            report["geometry"][label] = {"summary": summarize_collision(records), "frames": records}
        report["promotion_review"] = promotion_review(report)
    destination = args.output or args.motion.with_suffix(".review.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    if args.plot:
        if args.no_collisions:
            parser.error("--plot requires collision checks")
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_report(report, args.plot)
    if args.snapshots:
        snapshot_report(data, models, args.snapshots)
    print(
        json.dumps(
            {
                "output": str(destination),
                "motion": report["motion_name"],
                "geometry": {label: report["geometry"][label]["summary"] for label in models}
                if "geometry" in report
                else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
