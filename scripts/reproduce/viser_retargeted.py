#!/usr/bin/env python3
"""View source Inspire and retargeted Revo2 reference trajectories on CPU.

This starts only when explicitly invoked. It does not load a policy or Isaac Sim.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from check_retargeted import frame_poses, hand_link, names, object_mesh_path
from prepare_revo2 import ASSEMBLED_URDF, BODY_URDF
from revo2_model import UrdfModel
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--separation", type=float, default=1.6)
    args = parser.parse_args()
    with np.load(args.motion, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    length = len(data["joint_pos"])
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=8, height=8, plane="xy")
    server.gui.add_markdown(
        "**Inspire → Revo2 参考动作对照**\n\n"
        f"`{names(data, 'motion_names')[0]}` · {length} 帧 · {fps:g} FPS\n\n"
        "左：原始 Inspire；右：Revo2。仅运动学回放，不代表 policy 执行或任务成功。"
    )
    frame_slider = server.gui.add_slider("帧", min=0, max=length - 1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("播放", False)
    speed = server.gui.add_slider("速度", min=0.1, max=2.0, step=0.1, initial_value=1.0)
    overlay = server.gui.add_checkbox("叠加显示", False)
    show_visual = server.gui.add_checkbox("视觉网格", True)
    show_collision = server.gui.add_checkbox("碰撞网格", False)
    models = {"source": UrdfModel(BODY_URDF), "target": UrdfModel(ASSEMBLED_URDF)}
    handles = {}
    mesh_handles = {"visual": [], "collision": []}
    for label, model in models.items():
        handles[label] = {name: server.scene.add_frame(f"/{label}/{name}", show_axes=False) for name in model.links}
        for kind in mesh_handles:
            for index, (name, local, mesh) in enumerate(model.geometry(kind)):
                handle = server.scene.add_mesh_simple(
                    f"/{label}/{name}/{kind}_{index}",
                    mesh.vertices.astype(np.float32),
                    mesh.faces.astype(np.uint32),
                    color=((245, 160, 80) if label == "source" else (70, 165, 235))
                    if hand_link(name)
                    else ((175, 165, 150) if label == "source" else (160, 180, 195)),
                    opacity=0.35 if kind == "collision" else 0.85,
                    wireframe=kind == "collision",
                    visible=kind == "visual",
                    position=tuple(local[:3, 3]),
                    wxyz=tuple(Rotation.from_matrix(local[:3, :3]).as_quat()[[3, 0, 1, 2]]),
                )
                mesh_handles[kind].append(handle)
    object_name = names(data, "object_names")[0]
    mesh = trimesh.load(str(object_mesh_path(object_name)), force="mesh", process=False)
    objects = {
        label: server.scene.add_mesh_simple(
            f"/{label}/object", mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32), color=(155, 190, 110)
        )
        for label in models
    }
    labels = {
        label: server.scene.add_label(
            f"/{label}/label", "Original Inspire" if label == "source" else "Retargeted Revo2"
        )
        for label in models
    }

    def update(_=None):
        index = int(frame_slider.value)
        with server.atomic():
            for label, model in models.items():
                offset = np.array(
                    [0, 0 if overlay.value else (-args.separation / 2 if label == "source" else args.separation / 2), 0]
                )
                poses = frame_poses(model, data, index, "source_" if label == "source" else "")
                for name, transform in poses.items():
                    handles[label][name].position = tuple(transform[:3, 3] + offset)
                    handles[label][name].wxyz = tuple(Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]])
                objects[label].position = tuple(data["object_pos_w"][index] + offset)
                objects[label].wxyz = tuple(data["object_quat_w"][index])
                objects[label].visible = not (overlay.value and label == "source")
                root_position = poses[model.root_link][:3, 3] + offset
                labels[label].position = tuple(root_position + [0, 0, 0.8])
                labels[label].visible = not overlay.value
            for handle in mesh_handles["visual"]:
                handle.visible = show_visual.value
            for handle in mesh_handles["collision"]:
                handle.visible = show_collision.value

    for control in (frame_slider, overlay, show_visual, show_collision):
        control.on_update(update)

    @server.on_client_connect
    def _(client):
        pelvis = data["body_pos_w"][0, names(data, "body_names").index("pelvis")]
        client.camera.position = tuple(pelvis + [2.5, -3.5, 1.3])
        client.camera.look_at = tuple(pelvis + [0, 0, 0.1])

    update()
    print(f"CPU reference viewer: http://{args.host}:{server.get_port()}", flush=True)
    next_frame = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if playing.value and now >= next_frame:
                frame_slider.value = (int(frame_slider.value) + 1) % length
                update()
                next_frame = now + 1 / (fps * speed.value)
            time.sleep(0.005)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
