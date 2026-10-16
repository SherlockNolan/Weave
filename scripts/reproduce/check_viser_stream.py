#!/usr/bin/env python3
"""Verify a real Viser HTTP/WebSocket session receives moving scene transforms."""

import argparse
import asyncio
import hashlib
import json
import time
import urllib.request
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import msgspec
import websockets
import zstandard


async def check(url, seconds):
    with urllib.request.urlopen(url, timeout=10) as response:
        assert response.status == 200 and b"<html" in response.read().lower()
    counts = Counter()
    transforms = defaultdict(set)
    camera = dict(
        type="ViewerCameraMessage",
        wxyz=[1.0, 0.0, 0.0, 0.0],
        position=[4.0, -4.0, 3.0],
        fov=1.0,
        near=0.01,
        far=1000.0,
        image_height=540,
        image_width=960,
        look_at=[0.5, -0.7, 0.85],
        up_direction=[0.0, 0.0, 1.0],
    )
    deadline = time.monotonic() + seconds
    frames = 0
    async with websockets.connect(
        url.replace("http", "ws", 1), subprotocols=["viser-v" + version("viser")], max_size=None, open_timeout=10
    ) as socket:
        await socket.send(msgspec.msgpack.encode(camera))
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(socket.recv(), timeout=5)
            except asyncio.TimeoutError:
                continue
            except websockets.ConnectionClosed:
                break
            if not isinstance(raw, bytes):
                continue
            compressed_size = int.from_bytes(raw[8:16], "little")
            payload = msgspec.msgpack.decode(zstandard.ZstdDecompressor().decompress(raw[16 : 16 + compressed_size]))
            binary_start = (16 + compressed_size + 7) // 8 * 8
            buffers = []
            for length in payload["binaryBufferLengths"]:
                buffers.append(raw[binary_start : binary_start + length])
                binary_start += (length + 7) // 8 * 8
            for message in payload["messages"]:
                kind = message["type"]
                counts[kind] += 1
                if kind == "GuiButtonMessage" and message.get("props", {}).get("label") == "Start policy":
                    await socket.send(
                        msgspec.msgpack.encode(
                            {"type": "GuiUpdateMessage", "uuid": message["uuid"], "updates": {"value": True}}
                        )
                    )
                if kind not in ("SetPositionMessage", "SetOrientationMessage", "SceneNodeUpdateMessage"):
                    continue
                # Hash this scene update and only its referenced binary arrays.
                digest = hashlib.sha256(msgspec.msgpack.encode(message))

                def add_arrays(value):
                    if isinstance(value, dict):
                        if "__binary_index" in value:
                            digest.update(buffers[value["__binary_index"]])
                        else:
                            for item in value.values():
                                add_arrays(item)
                    elif isinstance(value, (list, tuple)):
                        for item in value:
                            add_arrays(item)

                add_arrays(message)
                transforms[(kind, message.get("name", ""))].add(digest.hexdigest())
            frames += 1
    moving = [name for (_, name), states in transforms.items() if len(states) > 1]
    report = {
        "url": url,
        "http_status": 200,
        "websocket_frames": frames,
        "message_counts": dict(counts),
        "moving_scene_nodes": sorted(set(moving)),
        "passed": bool(moving),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8088")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--output", type=Path, default=Path("logs/preparation/isaac62-viser-stream-check.json"))
    args = parser.parse_args()
    report = asyncio.run(check(args.url, args.seconds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("No moving scene nodes were received")


if __name__ == "__main__":
    main()
