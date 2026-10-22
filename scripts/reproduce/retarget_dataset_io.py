"""CPU-only packed-motion reader and audited G1/Inspire reference layout.

NPZs contain no joint/body names. Their layout is the breadth-first articulation
order after fixed-joint merging, not the URDF XML declaration order. The pilot
schema audit verifies this against both the converted USD joint graph and saved
Isaac Sim 5.1 runtime names, then verifies saved poses with URDF FK.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SOURCE_URDF = REPO / (
    "source/g1_hoi_learning/g1_hoi_learning/assets/unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
)
SOURCE_USD = REPO / ("data/isaac62/g1_inspire_usd/g1_29dof_rev_1_0_with_inspire_hand_DFQ.usd")
FRAME_FIELDS = (
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
CLIP_FIELDS = ("motion_names", "motion_lengths", "object_names")


def get_source_layout() -> dict:
    """Return canonical 53-joint/54-body names and all fixed-link merge groups.

    The original USD preserves movable joint names/parent-child relationships.
    Its breadth-first traversal agrees exactly with the 2026-10-19 Sim 5.1
    runtime snapshot and the original NPZ forward-kinematics check. See
    outputs/revo2-retargeting/schema/source_layout.json for that recorded proof.
    No Isaac modules, simulator, GPU, or USD Python bindings are required here.
    """
    robot = ET.parse(SOURCE_URDF).getroot()
    joints = robot.findall("joint")
    children = {joint.find("child").attrib["link"] for joint in joints}
    roots = {link.attrib["name"] for link in robot.findall("link")} - children
    if len(roots) != 1:
        raise ValueError(f"Expected one URDF root, got {roots}")
    root = roots.pop()
    outgoing = defaultdict(list)
    for joint in joints:
        outgoing[joint.find("parent").attrib["link"]].append(joint)

    # Collapse fixed links before breadth-first traversal; fixed palm and tip
    # frames are not separate PhysX bodies in the source reference.
    merged = {root: root}
    queue = deque([root])
    while queue:
        parent = queue.popleft()
        for joint in outgoing[parent]:
            child = joint.find("child").attrib["link"]
            merged[child] = merged[parent] if joint.attrib["type"] == "fixed" else child
            queue.append(child)
    body_children = defaultdict(list)
    for joint in joints:
        if joint.attrib["type"] != "fixed":
            parent = merged[joint.find("parent").attrib["link"]]
            body_children[parent].append(joint)
    # The converted USD orders sibling joints lexically (not XML order),
    # notably index/middle/pinky/ring/thumb below each fixed-merged palm.
    for group in body_children.values():
        group.sort(key=lambda joint: joint.attrib["name"])
    names = []
    bodies = [root]
    queue = deque([root])
    while queue:
        parent = queue.popleft()
        for joint in body_children[parent]:
            child = joint.find("child").attrib["link"]
            names.append(joint.attrib["name"])
            bodies.append(child)
            queue.append(child)
    if len(names) != 53 or len(bodies) != 54:
        raise ValueError(f"Source layout changed: {len(names)} joints, {len(bodies)} bodies")
    return {
        "joint_names": names,
        "body_names": bodies,
        "fixed_link_to_body": merged,
        "quaternion_order": "wxyz",
        "source_urdf": str(SOURCE_URDF),
        "source_urdf_sha256": hashlib.sha256(SOURCE_URDF.read_bytes()).hexdigest(),
        "source_usd": str(SOURCE_USD),
        "ordering": "breadth_first_after_fixed_joint_merging_with_lexical_siblings",
        "layout_proof": str(REPO / "outputs/revo2-retargeting/schema/source_layout.json"),
    }


def load_clip(npz_path: str | Path, sequence_name: str, variant: int = 0) -> tuple[dict, dict]:
    """Extract exactly one packed clip without changing any source values.

    ``sequence_name`` accepts a full motion name or a base sequence. Base names
    select ``_vXX`` when present; the sole unnumbered whitechair clip is accepted
    only for variant zero. Returns original NPZ keys with one-clip metadata and
    the selected per-frame arrays, plus human-readable source metadata.
    """
    if variant < 0:
        raise ValueError("variant must be non-negative")
    path = Path(npz_path).resolve()
    with np.load(path, allow_pickle=False) as archive:
        names = [str(name) for name in archive["motion_names"]]
        requested = f"{sequence_name}_v{variant:02d}"
        if requested in names:
            index = names.index(requested)
        elif sequence_name in names and ("_v" in sequence_name or variant == 0):
            index = names.index(sequence_name)
        else:
            available = [name for name in names if name.startswith(sequence_name)]
            raise ValueError(f"Motion {requested!r} absent from {path}; available: {available}")
        lengths = np.asarray(archive["motion_lengths"], dtype=np.int64)
        starts = np.r_[0, np.cumsum(lengths)]
        start, stop = int(starts[index]), int(starts[index + 1])
        arrays = {}
        for key in archive.files:
            value = archive[key]
            if key in FRAME_FIELDS:
                if value.shape[0] != starts[-1]:
                    raise ValueError(f"{key} has {value.shape[0]} frames, expected {starts[-1]}")
                arrays[key] = value[start:stop].copy()
            elif key in CLIP_FIELDS:
                if value.shape[0] != len(names):
                    raise ValueError(f"{key} length disagrees with motion_names")
                arrays[key] = value[index : index + 1].copy()
            else:
                arrays[key] = value.copy()
    fps = int(np.asarray(arrays["fps"]).reshape(-1)[0])
    metadata = {
        "source_path": str(path),
        "source_clip_index": index,
        "source_frame_start": start,
        "source_frame_stop": stop,
        "motion_name": names[index],
        "object_name": str(arrays["object_names"][0]),
        "frames": stop - start,
        "fps": fps,
        "duration_seconds": (stop - start) / fps,
        "quaternion_order": "wxyz",
        "contact_label_semantics": {"-1": "avoid contact", "0": "unconstrained", "1": "desired contact"},
    }
    # The existing pickup subset combines train and test clips for inspection.
    # Carry its extraction provenance so a downstream pilot cannot silently
    # relabel held-out sub16_clothesstand_012 as training data.
    manifest = path.parent / "manifest.json"
    if manifest.is_file():
        for clip in json.loads(manifest.read_text()).get("clips", []):
            if clip.get("motion_name") == names[index]:
                metadata["original_source_path"] = clip.get("source")
                metadata["source_split"] = clip.get("source_split")
                break
    return arrays, metadata
