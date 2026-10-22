#!/usr/bin/env python3
"""Download pinned official Revo2 assets and assemble them on Weave's G1 body (CPU only)."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import posixpath
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "source/g1_hoi_learning/g1_hoi_learning/assets"
TARGET = ASSETS / "unitree_g1_revo2"
BRAINCO_REPO = "BrainCoTech/brainco-description"
BRAINCO_COMMIT = "6db7e2ed1d864ee0a3ab6ff62d43a303787137f6"
UNITREE_REPO = "unitreerobotics/unitree_ros"
UNITREE_COMMIT = "5994d4faef0a9cadd3287f8de0199a67eeb2a259"
UNITREE_PATH = "robots/g1_with_brainco_hand"
REFERENCE_URDF = f"{UNITREE_PATH}/g1_29dof_mode_15_brainco_hand.urdf"
BODY_URDF = ASSETS / "unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
ASSEMBLED_URDF = TARGET / "g1_29dof_rev_1_0_with_revo2.urdf"


def fetch(url: str, *, attempts: int = 2) -> bytes:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Weave-Revo2-integration"})
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.read()
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError(url)


def download_assets() -> None:
    records = []
    for repo, commit, vendor in [
        (BRAINCO_REPO, BRAINCO_COMMIT, "brainco"),
        (UNITREE_REPO, UNITREE_COMMIT, "unitree"),
    ]:
        tree = json.loads(fetch(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"))
        if tree.get("truncated"):
            raise RuntimeError(f"Incomplete upstream tree: {repo}")
        entries = {item["path"]: item for item in tree["tree"] if item["type"] == "blob"}
        vendor_dir = TARGET / "vendor" / vendor

        def download(path: str) -> Path:
            entry = entries[path]
            destination = vendor_dir / path
            content = None
            if destination.is_file():
                content = destination.read_bytes()
                digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
                if digest != entry["sha"]:
                    content = None
            if content is None:
                # The branch URL also works behind GitHub caches that cannot serve
                # commit raw URLs. The pinned tree's Git blob hash remains authoritative.
                branch = "main" if vendor == "brainco" else "master"
                for url in (
                    f"https://cdn.jsdelivr.net/gh/{repo}@{commit}/{path}",
                    f"https://github.com/{repo}/raw/refs/heads/{branch}/{path}",
                ):
                    try:
                        content = fetch(url, attempts=1)
                        break
                    except OSError:
                        continue
                if content is None:
                    blob = json.loads(fetch(f"https://api.github.com/repos/{repo}/git/blobs/{entry['sha']}"))
                    content = base64.b64decode(blob["content"])
            digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if digest != entry["sha"]:
                raise RuntimeError(f"Upstream Git blob hash mismatch: {repo}/{path}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(destination)
            return destination

        if vendor == "brainco":
            paths = [f"revo2_system/urdf/revo2_{side}.urdf" for side in ("left", "right")]
            mesh_paths = set()
            for path in paths:
                root = ET.parse(download(path)).getroot()
                for mesh in root.findall(".//mesh"):
                    mesh_paths.add(posixpath.normpath(posixpath.join(posixpath.dirname(path), mesh.attrib["filename"])))
            paths += sorted(mesh_paths) + ["README.md", "RELEASE_NOTES.md", "revo2_system/README.md"]
        else:
            download(REFERENCE_URDF)
            paths = [REFERENCE_URDF] + [f"{UNITREE_PATH}/meshes/{side}_base2_link.STL" for side in ("left", "right")]
            paths += [name for name in ("LICENSE", "LICENSE.md") if name in entries]
        print(f"Downloading {len(paths)} pinned files from {repo} ...", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            files = list(pool.map(download, paths))
        for path, file in zip(paths, files):
            records.append(
                {
                    "repository": repo,
                    "commit": commit,
                    "upstream_path": path,
                    "path": str(file.relative_to(TARGET)),
                    "git_blob": entries[path]["sha"],
                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    "bytes": file.stat().st_size,
                }
            )
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "sources.json").write_text(json.dumps({"files": records}, indent=2) + "\n")
    print(f"Verified {len(records)} upstream files ({sum(r['bytes'] for r in records):,} bytes)", flush=True)


def rewrite_meshes(element: ET.Element, source_dir: Path) -> None:
    for mesh in element.findall(".//mesh"):
        source = (source_dir / mesh.attrib["filename"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        mesh.set("filename", os.path.relpath(source, TARGET))


def assemble() -> None:
    """Keep G1's body unchanged; replace both complete Inspire subtrees."""
    body = ET.parse(BODY_URDF).getroot()
    removed = {"L_hand_base_link", "R_hand_base_link"}
    while True:
        descendants = {
            joint.find("child").attrib["link"]
            for joint in body.findall("joint")
            if joint.find("parent").attrib["link"] in removed
        }
        if descendants <= removed:
            break
        removed |= descendants
    for element in list(body):
        if (
            element.tag == "link"
            and element.attrib["name"] in removed
            or element.tag == "joint"
            and element.find("child").attrib["link"] in removed
        ):
            body.remove(element)
    rewrite_meshes(body, BODY_URDF.parent)
    body.set("name", "g1_29dof_revo2")
    reference_path = TARGET / "vendor/unitree" / REFERENCE_URDF
    reference = ET.parse(reference_path).getroot()
    mounts = {}
    for side in ("left", "right"):
        # The adapter is a separate fixed sibling of the palm in Unitree's model.
        for tag, name in [("link", f"{side}_base2_link"), ("joint", f"{side}_base2_joint")]:
            element = copy.deepcopy(reference.find(f"{tag}[@name='{name}']"))
            rewrite_meshes(element, reference_path.parent)
            body.append(element)
        hand_path = TARGET / f"vendor/brainco/revo2_system/urdf/revo2_{side}.urdf"
        hand = ET.parse(hand_path).getroot()
        for element in hand:
            if element.tag == "link" and element.attrib["name"] == "world":
                continue
            if element.tag == "joint" and element.find("parent").attrib["link"] == "world":
                continue
            element = copy.deepcopy(element)
            rewrite_meshes(element, hand_path.parent)
            for index, material in enumerate(element.findall(".//material")):
                if not material.attrib.get("name"):
                    material.set("name", f"revo2_{element.attrib['name']}_{index}")
            body.append(element)
        mount = copy.deepcopy(reference.find(f"joint[@name='{side}_base_joint']"))
        mount.set("name", f"{side}_revo2_mount_joint")
        mount.find("child").set("link", f"{side}_hand_base_link")
        body.append(mount)
        mounts[side] = {
            "parent": mount.find("parent").attrib["link"],
            "child": mount.find("child").attrib["link"],
            **mount.find("origin").attrib,
        }
    ET.indent(body, space="  ")
    ET.ElementTree(body).write(ASSEMBLED_URDF, encoding="utf-8", xml_declaration=True)
    # Importer 3 ignores convert_mimic_joints_to_normal_joints. Supply a distinct
    # drive URDF without mimic tags for software-coupled control on both versions.
    software_body = copy.deepcopy(body)
    for joint in software_body.findall("joint"):
        if joint.find("mimic") is not None:
            joint.remove(joint.find("mimic"))
    software_path = ASSEMBLED_URDF.with_name(ASSEMBLED_URDF.stem + "_software_mimic.urdf")
    ET.ElementTree(software_body).write(software_path, encoding="utf-8", xml_declaration=True)
    joints = body.findall("joint")
    mimic = {
        joint.attrib["name"]: {
            "driver": joint.find("mimic").attrib["joint"],
            "multiplier": float(joint.find("mimic").attrib.get("multiplier", 1)),
            "offset": float(joint.find("mimic").attrib.get("offset", 0)),
        }
        for joint in joints
        if joint.find("mimic") is not None
    }
    hand_joint_names = {
        joint.attrib["name"]
        for joint in joints
        if joint.attrib["type"] != "fixed"
        and any(f"_{finger}_" in joint.attrib["name"] for finger in ("thumb", "index", "middle", "ring", "pinky"))
    }
    metadata = {
        "urdf": ASSEMBLED_URDF.name,
        "software_mimic_urdf": software_path.name,
        "body_source": str(BODY_URDF.relative_to(REPO)),
        "body_source_sha256": hashlib.sha256(BODY_URDF.read_bytes()).hexdigest(),
        "urdf_sha256": hashlib.sha256(ASSEMBLED_URDF.read_bytes()).hexdigest(),
        "software_mimic_urdf_sha256": hashlib.sha256(software_path.read_bytes()).hexdigest(),
        "mounts": mounts,
        "mimic": mimic,
        "hand_drive_joints": sorted(hand_joint_names - mimic.keys()),
        "body_drive_joints": [
            joint.attrib["name"]
            for joint in joints
            if joint.attrib["type"] != "fixed" and joint.attrib["name"] not in hand_joint_names
        ],
        "hand_joint_limits": {
            joint.attrib["name"]: {key: float(value) for key, value in joint.find("limit").attrib.items()}
            for joint in joints
            if joint.attrib["name"] in hand_joint_names
        },
        "self_collisions": True,
        "controller_status": "Candidate gains only; no dynamic or policy validation has been performed.",
    }
    # A 25% flexion pose is free of non-adjacent convex-hull overlaps in the CPU
    # check. The exact all-zero pose has a small palm/thumb hull overlap.
    metadata["initial_hand_joint_positions"] = {
        name: metadata["hand_joint_limits"][name]["lower"]
        + 0.25 * (metadata["hand_joint_limits"][name]["upper"] - metadata["hand_joint_limits"][name]["lower"])
        for name in metadata["hand_drive_joints"]
    }
    (TARGET / "assembly.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Assembled {ASSEMBLED_URDF}", flush=True)
    print(
        f"Movable joints: {len(metadata['body_drive_joints'])} body + {len(hand_joint_names)} hand; "
        f"{len(metadata['hand_drive_joints'])} hand drivers, {len(mimic)} mimic joints",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument(
        "--offline", action="store_true", help="Assemble already downloaded assets without network access."
    )
    args = parser.parse_args()
    if not args.offline:
        download_assets()
    if not args.download_only:
        assemble()


if __name__ == "__main__":
    main()
