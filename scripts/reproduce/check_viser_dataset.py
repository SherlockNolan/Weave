#!/usr/bin/env python3
"""Check dataset replay using the real browser WebSocket protocol, on CPU."""

from __future__ import annotations

import argparse
import asyncio
import json
import resource
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import msgspec
import numpy as np
import websockets
import zstandard
from viser_dataset_catalog import DatasetCatalog


class Browser:
    """Retain controls and transform state; discard the large mesh payloads."""

    def __init__(self, socket):
        self.socket = socket
        self.gui = {}
        self.nodes = {}
        self.messages = Counter()
        self.removed = set()
        self.frames = 0
        self.deadline = time.monotonic() + 360

    def control(self, label):
        return next((m for m in self.gui.values() if m.get("props", {}).get("label") == label), None)

    def markdown(self):
        return [m["props"]["_markdown"] for m in self.gui.values() if m["type"] == "GuiMarkdownMessage"]

    def status(self, name):
        return next((text for text in self.markdown() if text.startswith(f"**当前：{name}**")), "")

    def caption(self, frame, length):
        return any(text.startswith(f"帧 **{frame + 1} / {length}**") for text in self.markdown())

    async def update(self, label, value):
        control = self.control(label)
        assert control is not None, f"Missing control: {label}"
        control["value"] = value  # Mirror the browser's local input state.
        await self.socket.send(
            msgspec.msgpack.encode(
                {
                    "type": "GuiUpdateMessage",
                    "uuid": control["uuid"],
                    "updates": {"value": value},
                }
            )
        )

    async def until(self, condition, timeout=75):
        deadline = min(self.deadline, time.monotonic() + timeout)
        while not condition():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Browser condition timed out; status: {self.markdown()}")
            try:
                raw = await asyncio.wait_for(self.socket.recv(), timeout=min(2, max(0.01, deadline - time.monotonic())))
            except asyncio.TimeoutError:
                continue
            if not isinstance(raw, bytes):
                continue
            compressed = int.from_bytes(raw[8:16], "little")
            payload = msgspec.msgpack.decode(zstandard.ZstdDecompressor().decompress(raw[16 : 16 + compressed]))
            self.frames += 1
            for message in payload["messages"]:
                kind = message["type"]
                self.messages[kind] += 1
                if kind == "GuiRemoveMessage":
                    self.removed.add(message["uuid"])
                    self.gui.pop(message["uuid"], None)
                elif kind == "GuiUpdateMessage" and message["uuid"] in self.gui:
                    target = self.gui[message["uuid"]]
                    for key, value in message["updates"].items():
                        (target if key in target else target["props"])[key] = value
                elif kind.startswith("Gui") and "props" in message:
                    self.gui[message["uuid"]] = message
                elif "name" in message:
                    name = message["name"]
                    if kind == "RemoveSceneNodeMessage":
                        self.nodes.pop(name, None)
                        continue
                    node = self.nodes.setdefault(name, {})
                    if kind == "SceneNodeUpdateMessage":
                        node.update(message["updates"])
                    for key in ("position", "wxyz", "visible"):
                        if key in message:
                            node[key] = message[key]

    async def settle(self, seconds=0.2):
        deadline = time.monotonic() + seconds
        await self.until(lambda: time.monotonic() >= deadline, timeout=seconds + 3)

    def position_matches(self, path, expected):
        actual = self.nodes.get(path, {}).get("position")
        return actual is not None and np.allclose(actual, expected, atol=2e-6, rtol=0)

    def quaternion_matches(self, path, expected):
        if path not in self.nodes:
            return False
        actual = np.asarray(self.nodes[path].get("wxyz", [1, 0, 0, 0]))
        expected = np.asarray(expected)
        # q and -q encode the same physical orientation.
        return abs(np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected))) > 1 - 2e-6

    def visible(self, path):
        return self.nodes.get(path, {}).get("visible", True)


async def check(args):
    started = time.monotonic()
    catalog = DatasetCatalog()
    pilots = ["sub1_tripod_075_v00", "sub16_clothesstand_012_v00", "sub9_clothesstand_039_v00", "sub7_whitechair_072"]
    visited = []
    async with websockets.connect(
        args.url.replace("http", "ws", 1),
        subprotocols=["viser-v" + version("viser")],
        max_size=128 * 1024**2,
    ) as socket:
        await socket.send(
            msgspec.msgpack.encode(
                {
                    "type": "ViewerCameraMessage",
                    "wxyz": [1.0, 0.0, 0.0, 0.0],
                    "position": [4.0, -4.0, 3.0],
                    "fov": 1.0,
                    "near": 0.01,
                    "far": 1000.0,
                    "image_height": 540,
                    "image_width": 960,
                    "look_at": [0.0, 0.0, 0.85],
                    "up_direction": [0.0, 0.0, 1.0],
                }
            )
        )
        browser = Browser(socket)
        labels = [
            "手型 / Hand",
            "数据范围",
            "重定向方法",
            "物体",
            "搜索轨迹",
            "轨迹 / Motion",
            "加载 / 重播",
            "播放",
            "循环",
            "速度",
            "帧",
            "上一帧",
            "下一帧",
            "叠加两种手",
            "视觉网格",
            "碰撞网格",
            "重置视角",
        ]
        await browser.until(lambda: all(browser.control(label) for label in labels))
        controls = {label: browser.control(label)["uuid"] for label in labels}
        layouts = [
            "GuiSetPanelPositionMessage",
            "GuiSetPanelWidthMessage",
            "GuiSetPanelHeightMessage",
            "GuiSetPanelCollapsedMessage",
        ]
        initial_layout = {kind: browser.messages[kind] for kind in layouts}

        def stable():
            assert not (set(controls.values()) & browser.removed), "Controls removed during selection"
            assert all(browser.control(label)["uuid"] == uuid for label, uuid in controls.items()), "Controls recreated"
            assert {kind: browser.messages[kind] for kind in layouts} == initial_layout, "Panel layout reset"

        async def select(entry, mode):
            hand = {"both": "两种手对照", "revo2": "Revo2 重定向手", "inspire": "Inspire 原始手"}[mode]
            await browser.update("搜索轨迹", "")
            await browser.update("手型 / Hand", hand)
            await browser.update("数据范围", "训练集 train" if entry.split == "train" else "测试集 test")
            await browser.update("重定向方法", "Vector" if entry.method == "vector" else "Position")
            await browser.update("物体", entry.object)
            await browser.update("搜索轨迹", entry.name)
            await browser.until(lambda: entry.label in browser.control("轨迹 / Motion")["props"]["options"])
            await browser.update("轨迹 / Motion", entry.label)
            data = catalog.load(entry)
            await browser.until(lambda: bool(browser.status(entry.name)) and entry.source in browser.status(entry.name))
            await browser.until(lambda: browser.caption(0, entry.frames))
            assert entry.split in browser.status(entry.name)
            assert browser.control("帧")["props"]["max"] == entry.frames - 1
            stable()
            visited.append(
                {"name": entry.name, "hand": mode, "split": entry.split, "method": entry.method, "frames": entry.frames}
            )
            print(f"Checked selection: {mode}/{entry.method}/{entry.split}/{entry.name}", flush=True)
            return data

        async def seek(entry, data, frame, mode):
            await browser.update("帧", frame)
            active = ("inspire", "revo2") if mode == "both" else (mode,)

            def rendered():
                if not browser.caption(frame, entry.frames):
                    return False
                for hand in active:
                    prefix = "source_" if hand == "inspire" and entry.hand == "revo2" else ""
                    names = list(data[prefix + "body_names"])
                    for index, name in enumerate(names):
                        path = f"/{hand}/robot/{name}"
                        if not browser.position_matches(path, data[prefix + "body_pos_w"][frame, index]):
                            return False
                        if not browser.quaternion_matches(path, data[prefix + "body_quat_w"][frame, index]):
                            return False
                    if not browser.position_matches(f"/{hand}/objects/{entry.object}", data["object_pos_w"][frame]):
                        return False
                    if not browser.quaternion_matches(f"/{hand}/objects/{entry.object}", data["object_quat_w"][frame]):
                        return False
                return True

            await browser.until(rendered)
            assert not browser.control("播放")["value"], "Seeking must pause playback"

        # Compare all four full converted clips, preserving the held-out split.
        for name in pilots:
            entry = next(e for e in catalog.entries if e.name == name and e.hand == "revo2" and e.method == "position")
            data = await select(entry, "both")
            await seek(entry, data, entry.frames - 1, "both")
            await seek(entry, data, 0, "both")
            assert browser.visible("/inspire") and browser.visible("/revo2")
            assert browser.visible(f"/inspire/objects/{entry.object}") and browser.visible(
                f"/revo2/objects/{entry.object}"
            )

        group_positions = {hand: browser.nodes[f"/{hand}"]["position"] for hand in ("inspire", "revo2")}
        await browser.update("叠加两种手", True)
        await browser.until(
            lambda: (
                all(browser.position_matches(f"/{hand}", [0, 0, 0]) for hand in group_positions)
                and not browser.visible(f"/inspire/objects/{entry.object}")
            )
        )
        assert browser.visible(f"/revo2/objects/{entry.object}")
        await browser.update("叠加两种手", False)
        await browser.until(
            lambda: (
                all(browser.position_matches(f"/{hand}", position) for hand, position in group_positions.items())
                and browser.visible(f"/inspire/objects/{entry.object}")
            )
        )

        # A method switch must show the actual Vector archive, not stale Position data.
        for name in pilots:
            entry = next(e for e in catalog.entries if e.name == name and e.hand == "revo2" and e.method == "vector")
            data = await select(entry, "revo2")
            await seek(entry, data, entry.frames - 1, "revo2")
            assert not browser.visible("/inspire") and browser.visible("/revo2")

        # Full original train/test references, including one outside the converted pilot.
        original = next(
            e for e in catalog.entries if e.hand == "inspire" and e.split == "train" and e.name not in pilots
        )
        test = next(
            e
            for e in catalog.entries
            if e.hand == "inspire" and e.name == "sub16_clothesstand_012_v00" and e.split == "test"
        )
        for entry in (original, test):
            data = await select(entry, "inspire")
            await seek(entry, data, entry.frames - 1, "inspire")
            await seek(entry, data, 0, "inspire")
            assert browser.control("重定向方法")["props"]["disabled"]
            assert browser.visible("/inspire") and not browser.visible("/revo2")
            assert not any(browser.visible(path) for path in browser.nodes if path.startswith("/revo2/objects/"))

        # Playback must advance transforms, pause exactly, stop at the final frame and loop.
        initial = np.asarray(browser.nodes["/inspire/robot/pelvis"]["position"])
        await browser.update("播放", True)
        await browser.until(lambda: int(browser.control("帧")["value"]) >= 15)
        await browser.until(lambda: not browser.position_matches("/inspire/robot/pelvis", initial))
        await browser.update("播放", False)
        await browser.settle()
        paused = int(browser.control("帧")["value"])
        await browser.settle(0.25)
        assert int(browser.control("帧")["value"]) == paused
        await browser.update("循环", False)
        await browser.update("速度", 2.0)
        await seek(entry, data, entry.frames - 3, "inspire")
        await browser.update("播放", True)
        await browser.until(
            lambda: browser.caption(entry.frames - 1, entry.frames) and not browser.control("播放")["value"]
        )
        await browser.settle(0.2)
        assert int(browser.control("帧")["value"]) == entry.frames - 1
        await browser.update("循环", True)
        await seek(entry, data, entry.frames - 3, "inspire")
        await browser.update("播放", True)
        await browser.until(lambda: 0 < int(browser.control("帧")["value"]) < 25)
        assert browser.control("播放")["value"]
        await browser.update("播放", False)

        await browser.update("视觉网格", False)
        visual_nodes = [path for path in browser.nodes if path.startswith("/inspire/robot/") and "/visual_" in path]
        await browser.until(lambda: visual_nodes and all(not browser.visible(path) for path in visual_nodes))
        await browser.update("碰撞网格", True)
        collision_nodes = [
            path for path in browser.nodes if path.startswith("/inspire/robot/") and "/collision_" in path
        ]
        await browser.until(lambda: collision_nodes and all(browser.visible(path) for path in collision_nodes))
        await browser.update("视觉网格", True)
        await browser.update("碰撞网格", False)

        # Empty search changes only available selections and retains the current replay.
        current_status = browser.status(entry.name)
        await browser.update("搜索轨迹", "no-such-motion-zzzz")
        await browser.until(lambda: browser.control("轨迹 / Motion")["props"]["disabled"])
        assert browser.status(entry.name) == current_status
        stable()

        return {
            "passed": True,
            "url": args.url,
            "visited": visited,
            "same_websocket": True,
            "all_hand_modes": True,
            "train_and_test": True,
            "position_and_vector": True,
            "nonpilot_original_clip": original.name,
            "final_frame_and_seek_back": True,
            "rendered_positions_match_stored_data": True,
            "playback_changes_transforms": True,
            "all_recorded_body_orientations_match": True,
            "overlay_and_side_by_side": True,
            "pause_stops_frames": True,
            "nonloop_endpoint_retained": True,
            "loop_wraps": True,
            "model_and_object_visibility": True,
            "visual_and_collision_visibility": True,
            "empty_search_retains_replay": True,
            "controls_preserved": True,
            "panel_layout_preserved": True,
            "websocket_frames": browser.frames,
            "scene_transform_messages": browser.messages["SetPositionMessage"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "checker_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
            "simulator_or_policy_loaded": False,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18083")
    parser.add_argument("--output", type=Path, default=Path("outputs/viser/dataset-replay-check.json"))
    args = parser.parse_args()
    try:
        report = asyncio.run(check(args))
    except Exception as error:
        report = {"passed": False, "error": f"{type(error).__name__}: {error}"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
