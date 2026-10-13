#!/usr/bin/env python3
"""Download the author's reference motions at a fixed revision, with resume and SHA256 checks."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.parse

REPO = Path(__file__).resolve().parents[2]
DATASET = "appolyn/Weave"
REVISION = "db4a90f705f06667bd6b200053237e69644f0064"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json(url: str):
    return json.loads(subprocess.check_output([
        "curl", "--http1.1", "-fsSL", "--retry", "5", "--retry-all-errors", "--connect-timeout", "30", "--max-time", "120", url,
    ], text=True))


def download(entry: dict) -> dict:
    remote = entry["path"]
    relative = remote.removeprefix("reference/")
    target = REPO / "data" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = entry.get("lfs", {}).get("oid")
    if not target.exists():
        part = target.with_name(target.name + ".part")
        print(f"Downloading {remote} ({entry['size'] / 1e6:.1f} MB)", flush=True)
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{urllib.parse.quote(remote)}?download=true"
        # Reinvoke curl after network failures so -C recomputes the resume offset each time.
        for attempt in range(1, 9):
            result = subprocess.run([
                "curl", "--http1.1", "-fLsS", "--connect-timeout", "30",
                "--speed-limit", "1024", "--speed-time", "180", "-C", "-", "-o", str(part), url,
            ])
            if result.returncode == 0:
                break
            print(f"Retry {attempt}/8: {remote}; curl exit={result.returncode}", flush=True)
            if attempt == 8:
                raise subprocess.CalledProcessError(result.returncode, result.args)
            time.sleep(5)
        if part.stat().st_size != entry["size"]:
            raise RuntimeError(f"Incomplete download: {part}")
        actual_hash = sha256(part)
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(f"SHA256 mismatch (file retained for inspection): {part}")
        part.rename(target)
    else:
        if target.stat().st_size != entry["size"]:
            raise RuntimeError(f"Existing file has unexpected size: {target}")
        actual_hash = sha256(target)
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(f"Existing file has unexpected SHA256: {target}")
    print(f"Verified {relative}: {actual_hash}", flush=True)
    return {"remote": remote, "local": str(target.relative_to(REPO)), "size": entry["size"], "sha256": actual_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--include-rollouts", action="store_true", help="Also fetch the optional executed trajectories (~11 GB).")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    entries = []
    # Query each directory separately to stay below the API pagination limit.
    directories = ["reference/test", "reference/train", "objects"]
    if args.include_rollouts:
        directories += ["rollout/test", "rollout/train"]
    for directory in directories:
        entries += [entry for entry in fetch_json(
            f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/{directory}?recursive=true&limit=1000"
        ) if entry["type"] == "file"]
    print(f"Dataset: {DATASET}@{REVISION}; {len(entries)} files; {sum(e['size'] for e in entries)/1e9:.2f} GB", flush=True)
    log_dir = REPO / "logs/preparation"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "dataset.remote.json").write_text(json.dumps({"dataset": DATASET, "revision": REVISION, "files": entries}, indent=2) + "\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(download, entries))
    (log_dir / "dataset.verified.json").write_text(json.dumps({"dataset": DATASET, "revision": REVISION, "files": files}, indent=2) + "\n")
    print("All dataset files downloaded and verified.", flush=True)


if __name__ == "__main__":
    main()
