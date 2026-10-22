#!/usr/bin/env python3
"""Browse and replay recorded Inspire/Revo2 train/test trajectories in Viser."""

from __future__ import annotations

import argparse
import threading
import time
from collections import Counter

import numpy as np
import trimesh
import viser
from check_retargeted import frame_poses, hand_link, object_mesh_path
from prepare_revo2 import ASSEMBLED_URDF, BODY_URDF
from revo2_model import UrdfModel
from scipy.spatial.transform import Rotation
from viser_dataset_catalog import DatasetCatalog

HAND_OPTIONS = {"Inspire 原始手": "inspire", "Revo2 重定向手": "revo2", "两种手对照": "both"}
SPLIT_OPTIONS = {"全部": "all", "训练集 train": "train", "测试集 test": "test"}
METHOD_OPTIONS = {"Position": "position", "Vector": "vector"}
PLACEHOLDER = "— 请选择轨迹 —"


class DatasetViewer:
    """Keep one browser connection while loading clips on the playback thread."""

    def __init__(self, catalog, server, args):
        self.catalog, self.server, self.args = catalog, server, args
        self.lock = threading.RLock()
        self.pending_task = None
        self.pending_frame = None
        self.pending_camera = False
        self.dirty = False
        self.data = self.current = None
        self.current_mode = args.hand
        self.frame = 0
        self.cursor = 0.0
        self.last_tick = time.monotonic()
        self.last_caption = 0.0
        self.robots = {}
        self.objects = {}
        self.selected = None
        server.gui.set_panel_label("Weave · Dataset replay")
        server.scene.set_up_direction("+z")
        server.scene.add_grid("/ground", width=20, height=20, plane="xy")

        with server.gui.add_folder("数据集轨迹"):
            counts = Counter((entry.hand, entry.split, entry.method) for entry in catalog.entries)
            source_train = counts[("inspire", "train", "original")]
            source_test = counts[("inspire", "test", "original")]
            retarget_counts = "; ".join(
                f"{method}: train {counts[('revo2', 'train', method)]} / test {counts[('revo2', 'test', method)]}"
                for method in ("position", "vector")
            )
            server.gui.add_markdown(
                f"**录制轨迹回放**\n\nInspire：train **{source_train}** / test **{source_test}**。"
                f"\n\nRevo2：{retarget_counts}。\n\n选择片段后完整回放，可暂停并逐帧查看。"
            )
            self.hand = server.gui.add_dropdown(
                "手型 / Hand", tuple(HAND_OPTIONS), initial_value=self.option(HAND_OPTIONS, args.hand)
            )
            self.split = server.gui.add_dropdown(
                "数据范围", tuple(SPLIT_OPTIONS), initial_value=self.option(SPLIT_OPTIONS, args.split)
            )
            self.method = server.gui.add_dropdown(
                "重定向方法", tuple(METHOD_OPTIONS), initial_value=self.option(METHOD_OPTIONS, args.method)
            )
            objects = sorted({entry.object for entry in catalog.entries})
            self.object = server.gui.add_dropdown("物体", ("all", *objects), initial_value="all")
            self.query = server.gui.add_text("搜索轨迹", initial_value="", hint="序列 ID、物体名；空格分隔关键词。")
            self.matches = server.gui.add_markdown("")
            self.choice = server.gui.add_dropdown("轨迹 / Motion", (PLACEHOLDER,))
            self.load_button = server.gui.add_button("加载 / 重播", color="green")
            self.status = server.gui.add_markdown("正在建立轨迹目录…")

        with server.gui.add_folder("播放控制"):
            self.playing = server.gui.add_checkbox("播放", initial_value=False)
            self.loop = server.gui.add_checkbox("循环", initial_value=True)
            self.speed = server.gui.add_slider("速度", min=0.1, max=2.0, step=0.1, initial_value=1.0)
            self.frame_slider = server.gui.add_slider("帧", min=0, max=1, step=1, initial_value=0, disabled=True)
            self.previous = server.gui.add_button("上一帧")
            self.next = server.gui.add_button("下一帧")
            self.caption = server.gui.add_markdown("")

        with server.gui.add_folder("显示", expand_by_default=False):
            self.overlay = server.gui.add_checkbox("叠加两种手", initial_value=False)
            self.visual = server.gui.add_checkbox("视觉网格", initial_value=True)
            self.collision = server.gui.add_checkbox("碰撞网格", initial_value=False)
            self.camera_button = server.gui.add_button("重置视角")

        for control in (self.hand, self.split, self.method, self.object, self.query):
            control.on_update(self.filters_changed)

        @self.choice.on_update
        def select(event):
            if event.client is not None:
                with self.lock:
                    self.selected = self.catalog.by_label.get(self.choice.value)
                    self.load_button.disabled = self.selected is None
                    self.queue_selected()

        @self.load_button.on_click
        def load(_event):
            with self.lock:
                self.queue_selected()

        @self.frame_slider.on_update
        def scrub(event):
            if event.client is not None:
                with self.lock:
                    self.playing.value = False
                    self.pending_frame = int(self.frame_slider.value)

        @self.previous.on_click
        def previous(_event):
            self.step_frame(-1)

        @self.next.on_click
        def next_frame(_event):
            self.step_frame(1)

        @self.playing.on_update
        def play(_event):
            with self.lock:
                self.last_tick = time.monotonic()
                if self.data is not None and self.playing.value and self.frame == len(self.data["joint_pos"]) - 1:
                    self.pending_frame = 0

        for control in (self.overlay, self.visual, self.collision):
            control.on_update(self.mark_dirty)

        @self.camera_button.on_click
        def camera(_event):
            with self.lock:
                self.pending_camera = True

        @server.on_client_connect
        def connect(client):
            with self.lock:
                if self.data is not None:
                    self.set_camera(client)

        self.refresh_options()
        initial = self.catalog.search(
            query=args.sequence or "",
            split=args.split,
            hand=args.hand,
            method="all" if args.hand == "inspire" else args.method,
        )
        if args.sequence and not initial:
            raise ValueError(f"No trajectory matches --sequence {args.sequence!r} in the selected hand/split/method")
        if initial:
            self.selected = next((entry for entry in initial if entry.name == args.sequence), initial[0])
            self.query.value = args.sequence or ""
            self.refresh_options()
            self.queue_selected()

    @staticmethod
    def option(mapping, value):
        return next(label for label, key in mapping.items() if key == value)

    def filters_changed(self, event):
        if event.client is not None:
            with self.lock:
                self.refresh_options()

    def refresh_options(self):
        hand = HAND_OPTIONS[self.hand.value]
        self.method.disabled = hand == "inspire"
        self.overlay.disabled = hand != "both"
        matches = self.catalog.search(
            query=self.query.value,
            split=SPLIT_OPTIONS[self.split.value],
            hand=hand,
            method="all" if hand == "inspire" else METHOD_OPTIONS[self.method.value],
            object=self.object.value,
        )
        # Keep an explicitly selected clip reachable, including beyond the first page.
        if self.selected in matches:
            shown = [self.selected, *(entry for entry in matches if entry != self.selected)][:200]
        else:
            self.selected = None
            shown = matches[:200]
        options = (PLACEHOLDER, *(entry.label for entry in shown))
        self.choice.options = options
        self.choice.value = self.selected.label if self.selected else PLACEHOLDER
        self.choice.disabled = not matches
        self.load_button.disabled = self.selected is None
        extra = "；显示前 200 项，请输入关键词缩小范围" if len(matches) > 200 else ""
        self.matches.content = f"匹配 **{len(matches)}** 段{extra}。"

    def queue_selected(self):
        if self.selected is None:
            return
        self.pending_task = (self.selected.key, HAND_OPTIONS[self.hand.value])
        self.status.content = f"正在加载 **{self.selected.name}**…"

    def mark_dirty(self, _event=None):
        with self.lock:
            self.dirty = True

    def step_frame(self, delta):
        with self.lock:
            if self.data is not None:
                self.playing.value = False
                self.pending_frame = max(0, min(len(self.data["joint_pos"]) - 1, self.frame + delta))

    def ensure_robot(self, hand):
        if hand in self.robots:
            return
        model = UrdfModel(BODY_URDF if hand == "inspire" else ASSEMBLED_URDF)
        group = self.server.scene.add_frame(f"/{hand}", show_axes=False, visible=False)
        links = {name: self.server.scene.add_frame(f"/{hand}/robot/{name}", show_axes=False) for name in model.links}
        geometry = {"visual": [], "collision": []}
        for kind in geometry:
            for index, (name, local, mesh) in enumerate(model.geometry(kind)):
                color = (245, 160, 80) if hand == "inspire" else (70, 165, 235)
                handle = self.server.scene.add_mesh_simple(
                    f"/{hand}/robot/{name}/{kind}_{index}",
                    mesh.vertices.astype(np.float32),
                    mesh.faces.astype(np.uint32),
                    color=color if hand_link(name) else (175, 185, 195),
                    opacity=0.35 if kind == "collision" else 0.95,
                    wireframe=kind == "collision",
                    visible=kind == "visual",
                    position=tuple(local[:3, 3]),
                    wxyz=tuple(Rotation.from_matrix(local[:3, :3]).as_quat()[[3, 0, 1, 2]]),
                )
                geometry[kind].append(handle)
        label = self.server.scene.add_label(f"/{hand}/label", "Original Inspire" if hand == "inspire" else "Revo2")
        self.robots[hand] = (model, group, links, geometry, label)

    def ensure_object(self, hand, name):
        key = (hand, name)
        if key not in self.objects:
            mesh = trimesh.load(str(object_mesh_path(name)), force="mesh", process=False)
            self.objects[key] = self.server.scene.add_mesh_simple(
                f"/{hand}/objects/{name}",
                mesh.vertices.astype(np.float32),
                mesh.faces.astype(np.uint32),
                color=(155, 190, 110),
                visible=False,
            )

    def load(self, key, mode):
        entry = self.catalog.by_key[key]
        data = self.catalog.load(entry)
        active = ("inspire", "revo2") if mode == "both" else (mode,)
        for hand in active:
            self.ensure_robot(hand)
            self.ensure_object(hand, entry.object)
        with self.lock:
            self.data, self.current, self.current_mode = data, entry, mode
            self.frame = 0
            self.cursor = 0.0
            self.pending_frame = None
            self.playing.value = False
            self.frame_slider.max = max(1, len(data["joint_pos"]) - 1)
            self.frame_slider.value = 0
            self.frame_slider.disabled = False
            self.last_tick = time.monotonic()
            self.dirty = self.pending_camera = True
            kind = "Inspire 原始轨迹" if entry.hand == "inspire" else f"Revo2 / {entry.method}"
            frames = len(data["joint_pos"])
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            review = "需修正接触或碰撞" if entry.review_status == "needs_revision" else "待物理仿真验证"
            self.status.content = (
                f"**当前：{entry.name}**  \n{entry.split} · {entry.object} · {kind}  \n"
                f"**{frames} 帧 / {fps:g} FPS / {frames / fps:.2f} 秒**\n\n"
                f"来源：`{entry.source}`\n\n点击「播放」；拖动帧滑条可检查完整轨迹。"
                + (f"\n\nRevo2 重定向预览：**{review}**。" if entry.hand == "revo2" else "")
            )
        print(f"[DATASET] {entry.split}/{entry.name} · {kind} · {frames} frames", flush=True)

    def render(self):
        data, entry = self.data, self.current
        both = self.current_mode == "both"
        active = ("inspire", "revo2") if both else (self.current_mode,)
        with self.server.atomic():
            for hand, (model, group, links, geometry, label) in self.robots.items():
                group.visible = hand in active
                if hand not in active:
                    continue
                offset = (
                    0 if not both or self.overlay.value else (-1 if hand == "inspire" else 1) * self.args.separation / 2
                )
                group.position = (0.0, offset, 0.0)
                prefix = "source_" if entry.hand == "revo2" and hand == "inspire" else ""
                poses = frame_poses(model, data, self.frame, prefix)
                for name, transform in poses.items():
                    links[name].position = tuple(transform[:3, 3])
                    links[name].wxyz = tuple(Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]])
                for kind, handles in geometry.items():
                    visible = self.visual.value if kind == "visual" else self.collision.value
                    for handle in handles:
                        handle.visible = visible
                label.position = tuple(poses[model.root_link][:3, 3] + [0, 0, 0.8])
                label.visible = not (both and self.overlay.value)
            for (hand, name), handle in self.objects.items():
                handle.visible = (
                    hand in active and name == entry.object and not (both and self.overlay.value and hand == "inspire")
                )
                if handle.visible:
                    handle.position = tuple(data["object_pos_w"][self.frame])
                    handle.wxyz = tuple(data["object_quat_w"][self.frame])
            self.frame_slider.value = self.frame

    def set_camera(self, client):
        positions = self.data["body_pos_w"][:, 0]
        points = np.concatenate((positions, self.data["object_pos_w"]))
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        center[2] = max(0.5, center[2])
        distance = max(3.2, np.linalg.norm(np.ptp(points[:, :2], axis=0)) * 1.3)
        client.camera.look_at = tuple(center)
        client.camera.position = tuple(center + np.array([0.65, -0.9, 0.5]) * distance)

    def tick(self):
        with self.lock:
            request, self.pending_task = self.pending_task, None
        if request:
            try:
                self.load(*request)
            except Exception as error:
                self.status.content = f"加载失败：`{type(error).__name__}: {error}`。可选择其他轨迹重试。"
                print(f"[DATASET] Load failed: {error!r}", flush=True)
        with self.lock:
            now = time.monotonic()
            elapsed, self.last_tick = now - self.last_tick, now
            if self.data is None:
                return
            length = len(self.data["joint_pos"])
            fps = float(np.asarray(self.data["fps"]).reshape(-1)[0])
            if self.pending_frame is not None:
                self.cursor = float(self.pending_frame)
                self.pending_frame = None
            elif self.playing.value:
                self.cursor += elapsed * fps * self.speed.value
                if self.cursor >= length:
                    if self.loop.value:
                        self.cursor %= length
                    else:
                        self.cursor = float(length - 1)
                        self.playing.value = False
            frame = min(length - 1, int(self.cursor))
            if frame != self.frame or self.dirty:
                self.frame = frame
                self.render()
                self.dirty = False
            if now - self.last_caption > 0.1:
                self.caption.content = f"帧 **{self.frame + 1} / {length}** · **{self.frame / fps:.2f} s**"
                self.last_caption = now
            if self.pending_camera:
                for client in self.server.get_clients().values():
                    self.set_camera(client)
                self.pending_camera = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--hand", choices=("inspire", "revo2", "both"), default="both")
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--method", choices=("position", "vector"), default="position")
    parser.add_argument("--sequence", help="Initial full motion name or search query.")
    parser.add_argument("--separation", type=float, default=1.6)
    args = parser.parse_args()
    catalog = DatasetCatalog()
    server = viser.ViserServer(host=args.host, port=args.port)
    try:
        viewer = DatasetViewer(catalog, server, args)
        print(f"Dataset replay: http://{args.host}:{server.get_port()} (CPU)", flush=True)
        while True:
            viewer.tick()
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
