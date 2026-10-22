"""Small CPU-only URDF reader for assembly checks and Viser asset inspection.

This is a kinematic mesh viewer, not a dynamics simulator or a policy evaluator.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def origin(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = np.fromstring(element.attrib.get("xyz", "0 0 0"), sep=" ")
        transform[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(element.attrib.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return transform


class UrdfModel:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.root = ET.parse(self.path).getroot()
        self.links = {link.attrib["name"]: link for link in self.root.findall("link")}
        self.joints = {joint.attrib["name"]: joint for joint in self.root.findall("joint")}
        if len(self.links) != len(self.root.findall("link")) or len(self.joints) != len(self.root.findall("joint")):
            raise ValueError("Duplicate link or joint name")
        children = [joint.find("child").attrib["link"] for joint in self.joints.values()]
        if len(set(children)) != len(children):
            raise ValueError("A link has more than one parent joint")
        roots = set(self.links) - set(children)
        if len(roots) != 1:
            raise ValueError(f"Expected one root, got {roots}")
        self.root_link = roots.pop()
        pending = list(self.joints.values())
        visited = {self.root_link}
        self.ordered_joints = []
        while pending:
            ready = [joint for joint in pending if joint.find("parent").attrib["link"] in visited]
            if not ready:
                raise ValueError("Disconnected or cyclic URDF")
            for joint in ready:
                child = joint.find("child").attrib["link"]
                if child not in self.links or child in visited:
                    raise ValueError(f"Missing child or cycle: {child}")
                visited.add(child)
                self.ordered_joints.append(joint)
                pending.remove(joint)
        if visited != set(self.links):
            raise ValueError("Unreachable links")

    def configuration(self, positions: dict[str, float] | None = None) -> dict[str, float]:
        values = {name: 0.0 for name, joint in self.joints.items() if joint.attrib["type"] != "fixed"}
        if positions:
            unknown = set(positions) - set(values)
            if unknown:
                raise ValueError(f"Unknown movable joints: {unknown}")
            values.update(positions)
        for joint in self.ordered_joints:
            mimic = joint.find("mimic")
            if mimic is not None:
                values[joint.attrib["name"]] = values[mimic.attrib["joint"]] * float(
                    mimic.attrib.get("multiplier", 1)
                ) + float(mimic.attrib.get("offset", 0))
        return values

    def fk(self, positions: dict[str, float] | None = None) -> dict[str, np.ndarray]:
        values = self.configuration(positions)
        frames = {self.root_link: np.eye(4)}
        for joint in self.ordered_joints:
            motion = np.eye(4)
            if joint.attrib["type"] != "fixed":
                axis = np.fromstring(joint.find("axis").attrib.get("xyz", "1 0 0"), sep=" ")
                axis = axis / np.linalg.norm(axis)
                if joint.attrib["type"] in ("revolute", "continuous"):
                    motion[:3, :3] = Rotation.from_rotvec(axis * values[joint.attrib["name"]]).as_matrix()
                elif joint.attrib["type"] == "prismatic":
                    motion[:3, 3] = axis * values[joint.attrib["name"]]
                else:
                    raise ValueError(f"Unsupported joint type: {joint.attrib['type']}")
            frames[joint.find("child").attrib["link"]] = (
                frames[joint.find("parent").attrib["link"]] @ origin(joint.find("origin")) @ motion
            )
        return frames

    def geometry(self, kind: str = "visual"):
        """Return (link name, local transform, mesh) for visual or collision shapes."""
        for name, link in self.links.items():
            for element in link.findall(kind):
                geometry = element.find("geometry")
                mesh = geometry.find("mesh")
                if mesh is not None:
                    path = (self.path.parent / mesh.attrib["filename"]).resolve()
                    loaded = trimesh.load(str(path), force="mesh", process=False)
                    loaded.apply_scale(np.fromstring(mesh.attrib.get("scale", "1 1 1"), sep=" "))
                elif geometry.find("box") is not None:
                    loaded = trimesh.creation.box(np.fromstring(geometry.find("box").attrib["size"], sep=" "))
                elif geometry.find("sphere") is not None:
                    loaded = trimesh.creation.icosphere(
                        subdivisions=2, radius=float(geometry.find("sphere").attrib["radius"])
                    )
                elif geometry.find("cylinder") is not None:
                    cylinder = geometry.find("cylinder")
                    loaded = trimesh.creation.cylinder(
                        radius=float(cylinder.attrib["radius"]), height=float(cylinder.attrib["length"])
                    )
                else:
                    raise ValueError(f"Unsupported geometry in {name}")
                yield name, origin(element.find("origin")), loaded

    def rigid_groups(self):
        """Group fixed-connected links; report adjacent movable-body pairs."""
        groups = {self.root_link: self.root_link}
        adjacent = set()
        for joint in self.ordered_joints:
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            groups[child] = groups[parent] if joint.attrib["type"] == "fixed" else child
            if joint.attrib["type"] != "fixed":
                adjacent.add(frozenset((groups[parent], groups[child])))
        return groups, adjacent


def standing_pose(model: UrdfModel) -> dict[str, float]:
    values = {}
    for name in model.joints:
        for suffix, value in (
            ("_hip_pitch_joint", -0.312),
            ("_knee_joint", 0.669),
            ("_ankle_pitch_joint", -0.363),
            ("_elbow_joint", 0.6),
        ):
            if name.endswith(suffix):
                values[name] = value
    values.update(
        {
            "left_shoulder_roll_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_pitch_joint": 0.2,
        }
    )
    return values
