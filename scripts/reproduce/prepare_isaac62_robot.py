#!/usr/bin/env python3
"""Reuse and verify the robot conversion that produced the existing checkpoints."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Directory containing the original Isaac Sim 5.1 robot USD layers.")
    args = parser.parse_args()
    manifest = json.loads((repo / "environments/isaac62/robot-assets.json").read_text())
    target = repo / "data/isaac62/g1_inspire_usd"
    source = args.source or repo / manifest["source"]
    for entry in manifest["files"]:
        destination = target / entry["path"]
        if not destination.is_file():
            original = source / entry["path"]
            if not original.is_file():
                parser.error(
                    f"Missing original robot conversion: {original}. Pass --source with the saved USD directory."
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise RuntimeError(f"Original robot USD checksum mismatch: {destination}")
    print(f"Verified {len(manifest['files'])} original robot files in {target}")


if __name__ == "__main__":
    main()
