#!/usr/bin/env python3
"""Exercise task search, replay and object changes through a real Viser WebSocket."""

import argparse
import asyncio
import json
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import msgspec
import websockets
import zstandard


class Browser:
    def __init__(self, socket):
        self.socket = socket
        self.gui = {}
        self.frames = 0
        self.scene_updates = 0
        self.messages = Counter()
        self.removed = set()

    def control(self, label):
        return next((m for m in self.gui.values() if m.get("props", {}).get("label") == label), None)

    def status(self, name):
        return next(
            (
                m["props"]["_markdown"]
                for m in self.gui.values()
                if m["type"] == "GuiMarkdownMessage" and m["props"]["_markdown"].startswith(f"**当前：{name}**")
            ),
            "",
        )

    async def update(self, control, value):
        # The actual browser updates local input state before sending its event.
        control["value"] = value
        await self.socket.send(
            msgspec.msgpack.encode({"type": "GuiUpdateMessage", "uuid": control["uuid"], "updates": {"value": value}})
        )

    async def until(self, condition, timeout=90):
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for browser state; markdown: "
                    f"{[m['props']['_markdown'] for m in self.gui.values() if m['type'] == 'GuiMarkdownMessage']}"
                )
            try:
                raw = await asyncio.wait_for(self.socket.recv(), timeout=5)
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
                elif kind == "SceneNodeUpdateMessage":
                    self.scene_updates += 1

    async def queue(self, query, name):
        await self.update(self.control("搜索全部任务"), query)
        await self.until(lambda: any(name in option for option in self.control("任务 / Task")["props"]["options"]))
        dropdown = self.control("任务 / Task")
        label = next(option for option in dropdown["props"]["options"] if name in option)
        await self.update(dropdown, label)

    async def select(self, query, name, finish=True):
        await self.queue(query, name)
        await self.until(lambda: bool(self.status(name)))
        if finish:
            await self.until(lambda: "保留终止姿态" in self.status(name))
        print(f"Selected {name}; same WebSocket remains connected.", flush=True)


async def check(args):
    async with websockets.connect(
        args.url.replace("http", "ws", 1), subprotocols=["viser-v" + version("viser")], max_size=None
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
                    "look_at": [0.5, -0.7, 0.85],
                    "up_direction": [0.0, 0.0, 1.0],
                }
            )
        )
        browser = Browser(socket)
        await browser.until(
            lambda: browser.control("搜索全部任务") is not None and browser.control("任务 / Task") is not None
        )
        labels = ["搜索全部任务", "任务 / Task", "数据范围", "物体", "开始 / 重播", "暂停"]
        controls = {label: browser.control(label)["uuid"] for label in labels}
        layout_types = [
            "GuiSetPanelPositionMessage",
            "GuiSetPanelWidthMessage",
            "GuiSetPanelHeightMessage",
            "GuiSetPanelCollapsedMessage",
        ]
        initial_layout = {kind: browser.messages[kind] for kind in layout_types}
        original_choice = browser.control("任务 / Task")["value"]
        await browser.update(browser.control("搜索全部任务"), "no-such-task-zzzz")
        await browser.until(lambda: browser.control("任务 / Task")["props"]["disabled"])
        assert browser.control("任务 / Task")["value"] == original_choice
        await browser.select("三脚架 sub1_tripod_075_v01", "sub1_tripod_075_v01", finish=not args.ui_only)
        if args.ui_only:
            await browser.until(lambda: "正在执行" in browser.status("sub1_tripod_075_v01"))
        if not args.ui_only:
            await browser.update(browser.control("开始 / 重播"), True)
            await browser.until(lambda: "正在执行" in browser.status("sub1_tripod_075_v01"))
            await browser.update(browser.control("暂停"), True)
            await browser.until(lambda: browser.control("继续") is not None)
            await browser.update(browser.control("继续"), True)
            await browser.until(lambda: browser.control("暂停") is not None)
            await browser.until(lambda: "保留终止姿态" in browser.status("sub1_tripod_075_v01"))
        visited = ["sub1_tripod_075_v01"]
        targets = ["sub7_whitechair_072", "sub16_clothesstand_012_v00", "sub9_clothesstand_039_v00"]
        if args.ui_only:
            targets = [
                "sub7_whitechair_072",
                "sub17_smallbox_018_v03",
                "sub7_whitechair_072",
                "sub16_clothesstand_012_v00",
                "sub9_clothesstand_039_v00",
            ]
        for name in targets:
            await browser.select(name, name, finish=not args.ui_only)
            if args.ui_only:
                await browser.until(
                    lambda: "正在执行" in browser.status(name) or "保留终止姿态" in browser.status(name)
                )
            for label, uuid in controls.items():
                if label != "暂停":
                    assert browser.control(label)["uuid"] == uuid, f"Recreated control: {label}"
            assert not (set(controls.values()) & browser.removed), "Task controls were removed during switching"
            assert name in browser.control("任务 / Task")["value"]
            assert browser.control("搜索全部任务")["value"] == name
            visited.append(name)
        if args.ui_only:
            # Queue a second object while the first object's scene is loading.
            first, last = "sub17_smallbox_018_v03", "sub16_clothesstand_012_v00"
            await browser.update(browser.control("数据范围"), "测试集 test")
            await browser.queue(first, first)
            await browser.until(
                lambda: any(f"正在加载 **{first}**" in m.get("props", {}).get("_markdown", "")
                            for m in browser.gui.values())
            )
            await browser.queue(last, last)
            await browser.until(lambda: bool(browser.status(last)))
            assert last in browser.control("任务 / Task")["value"]
            assert browser.control("搜索全部任务")["value"] == last
            assert browser.control("数据范围")["value"] == "测试集 test"
            for label, uuid in controls.items():
                if label != "暂停":
                    assert browser.control(label)["uuid"] == uuid
            visited.extend([first, last])
        if args.all_objects:
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rsl_rl"))
            from viser_catalog import TaskCatalog

            catalog = TaskCatalog()
            for obj in ["floorlamp", "largebox", "largetable", "smallbox", "smalltable", "woodchair"]:
                task = next(t for t in catalog.tasks if t.object == obj and t.split == "train")
                await browser.select(task.name, task.name, finish=False)
                await browser.until(
                    lambda: "正在执行" in browser.status(task.name) or "保留终止姿态" in browser.status(task.name)
                )
                visited.append(task.name)
        assert {kind: browser.messages[kind] for kind in layout_types} == initial_layout, "Panel layout was reset"
        return {
            "passed": True,
            "visited": visited,
            "empty_search": True,
            "replay": not args.ui_only,
            "pause_resume": not args.ui_only,
            "same_websocket": True,
            "controls_preserved": True,
            "selection_preserved": True,
            "filters_preserved": True,
            "panel_layout_preserved": True,
            "latest_choice_during_load": args.ui_only,
            "websocket_frames": browser.frames,
            "scene_updates": browser.scene_updates,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--all-objects", action="store_true")
    parser.add_argument(
        "--ui-only", action="store_true", help="Check control/selection persistence without full rollouts."
    )
    parser.add_argument("--output", type=Path, default=Path("logs/preparation/viser-task-switch-check.json"))
    args = parser.parse_args()
    report = asyncio.run(check(args))
    assert report["scene_updates"] > 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
