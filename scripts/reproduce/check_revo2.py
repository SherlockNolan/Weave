#!/usr/bin/env python3
"""Check G1/Revo2 assembly, kinematics, limits, and optional convex contacts on CPU."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from prepare_revo2 import ASSEMBLED_URDF, BODY_URDF, REFERENCE_URDF, TARGET
from revo2_model import UrdfModel, origin, standing_pose


def canonical(element: ET.Element, mesh_dir: Path) -> bytes:
    element = copy.deepcopy(element)
    for mesh in element.findall(".//mesh"):
        mesh.set("filename", str((mesh_dir / mesh.attrib["filename"]).resolve()))
    for item in element.iter():
        item.text = None
        item.tail = None
    return ET.tostring(element)


def check_structure(model: UrdfModel) -> dict:
    metadata = json.loads((TARGET / "assembly.json").read_text())
    assert hashlib.sha256(BODY_URDF.read_bytes()).hexdigest() == metadata["body_source_sha256"]
    assert hashlib.sha256(model.path.read_bytes()).hexdigest() == metadata["urdf_sha256"]
    assert (
        hashlib.sha256((TARGET / metadata["software_mimic_urdf"]).read_bytes()).hexdigest()
        == metadata["software_mimic_urdf_sha256"]
    )
    sources = json.loads((TARGET / "sources.json").read_text())
    for entry in sources["files"]:
        assert hashlib.sha256((TARGET / entry["path"]).read_bytes()).hexdigest() == entry["sha256"], entry["path"]
    original = UrdfModel(BODY_URDF)
    body_links = {name: link for name, link in original.links.items() if not name.startswith(("L_", "R_"))}
    body_joints = {
        name: joint for name, joint in original.joints.items() if joint.find("child").attrib["link"] in body_links
    }
    for name, link in body_links.items():
        assert canonical(link, BODY_URDF.parent) == canonical(model.links[name], model.path.parent), name
    for name, joint in body_joints.items():
        assert canonical(joint, BODY_URDF.parent) == canonical(model.joints[name], model.path.parent), name
    assert len(metadata["body_drive_joints"]) == 29
    assert len(metadata["hand_drive_joints"]) == 12
    assert len(metadata["mimic"]) == 10
    assert len(model.configuration()) == 51
    assert not any(name.startswith(("L_", "R_")) for name in model.links)
    for link in model.links.values():
        for mesh in link.findall(".//mesh"):
            assert (model.path.parent / mesh.attrib["filename"]).is_file(), mesh.attrib["filename"]
        inertial = link.find("inertial")
        if inertial is not None:
            assert float(inertial.find("mass").attrib["value"]) > 0
            inertia = inertial.find("inertia").attrib
            matrix = np.array(
                [
                    [float(inertia["ixx"]), float(inertia["ixy"]), float(inertia["ixz"])],
                    [float(inertia["ixy"]), float(inertia["iyy"]), float(inertia["iyz"])],
                    [float(inertia["ixz"]), float(inertia["iyz"]), float(inertia["izz"])],
                ]
            )
            assert np.linalg.eigvalsh(matrix).min() > 0, link.attrib["name"]
    software = UrdfModel(TARGET / metadata["software_mimic_urdf"])
    assert not software.root.findall(".//mimic")
    restored = copy.deepcopy(software.root)
    for joint in restored.findall("joint"):
        spec = model.joints[joint.attrib["name"]].find("mimic")
        if spec is not None:
            joint.append(copy.deepcopy(spec))
    assert canonical(restored, TARGET) == canonical(model.root, TARGET)
    reference = UrdfModel(TARGET / "vendor/unitree" / REFERENCE_URDF)
    new_fk, ref_fk = model.fk(), reference.fk()
    ref_names = {name.lower(): name for name in ref_fk}
    mounts = {}
    for side in ("left", "right"):
        mount = model.joints[f"{side}_revo2_mount_joint"]
        ref_mount = reference.joints[f"{side}_base_joint"]
        assert np.allclose(origin(mount.find("origin")), origin(ref_mount.find("origin")), atol=1e-12)
        for suffix in ("roll", "pitch", "yaw"):
            joint_name = f"{side}_wrist_{suffix}_joint"
            assert np.allclose(
                origin(model.joints[joint_name].find("origin")),
                origin(reference.joints[joint_name].find("origin")),
                atol=1e-12,
            ), joint_name
        # Palm frame equivalence checked using independent proximal finger anchors.
        errors = {}
        new_palm = np.linalg.inv(new_fk[f"{side}_hand_base_link"])
        ref_palm = np.linalg.inv(ref_fk[f"{side}_base_link"])
        for finger in ("thumb_metacarpal", "index_proximal", "middle_proximal", "ring_proximal", "pinky_proximal"):
            new_frame = new_palm @ new_fk[f"{side}_{finger}_link"]
            old_frame = ref_palm @ ref_fk[ref_names[f"{side}_{finger}_link"]]
            error = float(np.linalg.norm(new_frame[:3, 3] - old_frame[:3, 3]))
            assert error < 1e-6, (side, finger, error)
            errors[finger] = error
        mounts[side] = {"finger_anchor_position_error_m": errors, **metadata["mounts"][side]}
    return {
        "source_files_verified": len(sources["files"]),
        "body_links_unchanged": len(body_links),
        "body_joints_unchanged": len(body_joints),
        "links": len(model.links),
        "joints": len(model.joints),
        "movable_joints": 51,
        "independent_controls": 41,
        "mimic_joints": 10,
        "mounts": mounts,
    }


def configurations(model: UrdfModel):
    metadata = json.loads((TARGET / "assembly.json").read_text())
    yield "zero_open", {}
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        positions = standing_pose(model)
        for name in metadata["hand_drive_joints"]:
            limits = metadata["hand_joint_limits"][name]
            positions[name] = limits["lower"] + fraction * (limits["upper"] - limits["lower"])
        yield f"standing_hand_{fraction:g}", positions


def check_contacts(model: UrdfModel) -> dict:
    import fcl

    groups, adjacent = model.rigid_groups()
    shapes = []
    for name, local, mesh in model.geometry("collision"):
        hull = mesh.convex_hull
        faces = np.column_stack((np.full(len(hull.faces), 3), hull.faces)).reshape(-1)
        shape = fcl.Convex(hull.vertices, len(hull.faces), faces)
        shapes.append((name, local, hull, shape))
    results = {}
    for label, positions in configurations(model):
        frames = model.fk(positions)
        objects = []
        for name, local, hull, shape in shapes:
            transform = frames[name] @ local
            obj = fcl.CollisionObject(shape, fcl.Transform(transform[:3, :3], transform[:3, 3]))
            vertices = hull.vertices @ transform[:3, :3].T + transform[:3, 3]
            objects.append((name, obj, np.array([vertices.min(axis=0), vertices.max(axis=0)])))
        contacts = []
        for index, (name_a, obj_a, bounds_a) in enumerate(objects):
            for name_b, obj_b, bounds_b in objects[index + 1 :]:
                if groups[name_a] == groups[name_b] or frozenset((groups[name_a], groups[name_b])) in adjacent:
                    continue
                if np.any(bounds_a[1] < bounds_b[0]) or np.any(bounds_b[1] < bounds_a[0]):
                    continue
                result = fcl.CollisionResult()
                fcl.collide(obj_a, obj_b, fcl.CollisionRequest(num_max_contacts=1, enable_contact=True), result)
                if result.is_collision:
                    depth = max((float(contact.penetration_depth) for contact in result.contacts), default=0)
                    if depth > 1e-5:
                        contacts.append({"links": [name_a, name_b], "penetration_m": depth})
        results[label] = sorted(contacts, key=lambda item: -item["penetration_m"])
    return {
        "method": "FCL convex hulls; fixed-connected and adjacent movable bodies excluded; CPU only",
        "is_physx_validation": False,
        "collision_shapes": len(shapes),
        "poses": results,
        "default_pose": "standing_hand_0.25",
        "default_pose_has_penetration": bool(results["standing_hand_0.25"]),
    }


def check_independent_parser(model: UrdfModel) -> dict:
    import yourdfpy

    independent = yourdfpy.URDF.load(
        str(model.path),
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=True,
        load_collision_meshes=True,
    )
    assert len(independent.actuated_joint_names) == 41
    cases = list(configurations(model))
    generator = np.random.default_rng(42)
    for sample in range(4):
        values = {}
        for name in independent.actuated_joint_names:
            limit = model.joints[name].find("limit").attrib
            values[name] = generator.uniform(float(limit["lower"]), float(limit["upper"]))
        cases.append((f"random_{sample}", values))
    maximum_error = 0.0
    for _, values in cases:
        independent.update_cfg({name: model.configuration(values)[name] for name in independent.actuated_joint_names})
        for name, transform in model.fk(values).items():
            error = float(np.max(np.abs(transform - independent.get_transform(name, model.root_link))))
            maximum_error = max(maximum_error, error)
            assert error < 1e-10, (name, error)
    return {
        "parser": "yourdfpy",
        "version": yourdfpy.__version__,
        "poses": len(cases),
        "frames_per_pose": len(model.links),
        "maximum_fk_matrix_error": maximum_error,
        "visual_geometries": len(independent.scene.geometry),
        "collision_geometries": len(independent.collision_scene.geometry),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contacts", action="store_true", help="Requires optional python-fcl.")
    parser.add_argument("--independent-parser", action="store_true", help="Cross-check FK using optional yourdfpy.")
    parser.add_argument("--output", type=Path, default=TARGET / "validation.json")
    args = parser.parse_args()
    model = UrdfModel(ASSEMBLED_URDF)
    report = {"structure": check_structure(model)}
    for label, positions in configurations(model):
        values = model.configuration(positions)
        for name, value in values.items():
            limits = model.joints[name].find("limit")
            if limits is not None:
                assert float(limits.attrib["lower"]) - 1e-6 <= value <= float(limits.attrib["upper"]) + 1e-6, (
                    label,
                    name,
                    value,
                )
        frames = model.fk(positions)
        assert all(
            np.isfinite(frame).all() and np.isclose(np.linalg.det(frame[:3, :3]), 1) for frame in frames.values()
        )
    report["kinematic_limit_poses_checked"] = 6
    if args.contacts:
        report["contacts"] = check_contacts(model)
    if args.independent_parser:
        report["independent_parser"] = check_independent_parser(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
