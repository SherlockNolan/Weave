#!/usr/bin/env python3
"""Prepare the pinned large wheels used by the isolated uv project."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

from download_wheel import download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    project = repo / "environments/isaac62"
    cache = args.cache or Path(os.environ.get("WEAVE_WHEEL_CACHE", repo.parents[1] / "cache/weave-wheels"))
    entries = json.loads((project / "large-wheels.json").read_text())
    wheelhouse = project / "wheels"
    wheelhouse.mkdir(exist_ok=True)

    def prepare(entry):
        path = download(
            entry["url"],
            cache / entry["filename"],
            entry["sha256"],
            workers=entry.get("workers", 16),
            chunk_mib=entry.get("chunk_mib", 4),
        ).resolve()
        link = wheelhouse / entry["filename"]
        if link.is_symlink():
            if link.resolve() == path:
                return
            link.unlink()
        if not link.exists():
            link.symlink_to(path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(prepare, entry) for entry in entries]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    print(f"Verified {len(entries)} large wheels in {cache}")


if __name__ == "__main__":
    main()
