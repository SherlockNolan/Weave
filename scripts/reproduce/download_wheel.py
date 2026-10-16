#!/usr/bin/env python3
"""Download a large public wheel in parallel HTTP ranges and verify its SHA-256."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path


def download(url, target, expected_hash, workers=32, chunk_mib=8):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == expected_hash:
                return target
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206:
            raise RuntimeError("The download server does not support byte ranges")
        size = int(response.headers["Content-Range"].split("/")[-1])
    part = target.with_suffix(target.suffix + ".part")
    progress_path = part.with_suffix(part.suffix + ".json")
    fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)
    chunk_size = chunk_mib * 1024 * 1024
    ranges = [(start, min(start + chunk_size, size) - 1) for start in range(0, size, chunk_size)]
    completed = set()
    if progress_path.is_file():
        saved = json.loads(progress_path.read_text())
        if (
            saved.get("sha256") == expected_hash
            and saved.get("size") == size
            and saved.get("chunk_size", 8 * 1024 * 1024) == chunk_size
        ):
            completed = set(saved["completed"])
    started = time.monotonic()

    def fetch(bounds):
        start, stop = bounds
        for attempt in range(4):
            try:
                deadline = time.monotonic() + 150
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{stop}"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206 or not response.headers["Content-Range"].startswith(f"bytes {start}-"):
                        raise RuntimeError("Server returned the wrong byte range")
                    offset = start
                    while offset <= stop:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Range transfer stalled; retrying with a fresh connection")
                        data = response.read(min(1024 * 1024, stop + 1 - offset))
                        if not data:
                            raise RuntimeError("Incomplete range response")
                        view = memoryview(data)
                        while view:
                            written = os.pwrite(fd, view, offset)
                            offset += written
                            view = view[written:]
                return stop - start + 1
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
        raise RuntimeError("All range download attempts failed")

    try:
        done = sum(stop - start + 1 for i, (start, stop) in enumerate(ranges) if i in completed)
        print(f"Downloading {target.name}: {size / 1e6:.1f} MB, {workers} connections", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fetch, bounds): i for i, bounds in enumerate(ranges) if i not in completed}
            for future in concurrent.futures.as_completed(futures):
                done += future.result()
                completed.add(futures[future])
                temp = progress_path.with_suffix(".tmp")
                temp.write_text(
                    json.dumps(
                        {
                            "sha256": expected_hash,
                            "size": size,
                            "chunk_size": chunk_size,
                            "completed": sorted(completed),
                        }
                    )
                )
                temp.replace(progress_path)
                elapsed = time.monotonic() - started
                print(f"{done / size:.1%} ({done / 1e6:.1f} MB), {done / max(elapsed, 1) / 1e6:.2f} MB/s", flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
    with part.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_hash:
        progress_path.unlink(missing_ok=True)
        raise RuntimeError(f"Wheel checksum mismatch: expected {expected_hash}, received {actual}")
    part.replace(target)
    progress_path.unlink(missing_ok=True)
    print(f"SHA-256 verified: {target}", flush=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("target", type=Path)
    parser.add_argument("sha256")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()
    download(args.url, args.target, args.sha256, args.workers, args.chunk_mib)
