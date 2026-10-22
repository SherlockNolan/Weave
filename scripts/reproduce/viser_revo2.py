#!/usr/bin/env python3
"""Inspect the assembled G1/Revo2 URDF in Viser on CPU; no Isaac Sim or policy."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import viser
from prepare_revo2 import ASSEMBLED_URDF, TARGET
from revo2_model import UrdfModel, standing_pose
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    model = UrdfModel(ASSEMBLED_URDF)
    metadata = json.loads((TARGET / "assembly.json").read_text())
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=3.0, height=3.0, plane="xy")
    frames = {name: server.scene.add_frame(f"/robot/{name}", show_axes=False) for name in model.links}
    geometries = {"visual": [], "collision": []}
    for kind in geometries:
        for index, (name, local, mesh) in enumerate(model.geometry(kind)):
            handle = server.scene.add_mesh_simple(
                f"/robot/{name}/{kind}_{index}",
                mesh.vertices.astype(np.float32),
                mesh.faces.astype(np.uint32),
                color=(105, 170, 230)
                if "hand" in name or any(f"_{f}_" in name for f in ("thumb", "index", "middle", "ring", "pinky"))
                else (185, 190, 195),
                opacity=0.4 if kind == "collision" else 1.0,
                wireframe=kind == "collision",
                visible=kind == "visual",
                position=tuple(local[:3, 3]),
                wxyz=tuple(Rotation.from_matrix(local[:3, :3]).as_quat()[[3, 0, 1, 2]]),
            )
            geometries[kind].append(handle)
    server.gui.add_markdown("**G1 + Revo2 模型检查**\n\n仅运动学可视化；不执行 policy，不计算接触力。")
    pose = server.gui.add_dropdown("身体姿态", ("standing", "zero"), initial_value="standing")
    collision = server.gui.add_checkbox("显示碰撞网格", False)
    axes = server.gui.add_checkbox("显示连杆坐标轴", False)
    sliders = {}
    for side in ("left", "right"):
        with server.gui.add_folder("左手" if side == "left" else "右手"):
            for name in metadata["hand_drive_joints"]:
                if name.startswith(side + "_"):
                    limit = metadata["hand_joint_limits"][name]
                    sliders[name] = server.gui.add_slider(
                        name.removeprefix(side + "_").removesuffix("_joint"),
                        min=limit["lower"],
                        max=limit["upper"],
                        step=0.005,
                        initial_value=metadata["initial_hand_joint_positions"][name],
                    )

    def update(_=None):
        positions = standing_pose(model) if pose.value == "standing" else {}
        positions.update({name: slider.value for name, slider in sliders.items()})
        transforms = model.fk(positions)
        with server.atomic():
            for name, frame in frames.items():
                transform = transforms[name]
                frame.position = tuple(transform[:3, 3] + np.array([0, 0, 0.76]))
                frame.wxyz = tuple(Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]])
                frame.show_axes = axes.value
            for handle in geometries["visual"]:
                handle.visible = not collision.value
            for handle in geometries["collision"]:
                handle.visible = collision.value

    pose.on_update(update)
    collision.on_update(update)
    axes.on_update(update)
    for slider in sliders.values():
        slider.on_update(update)

    @server.on_client_connect
    def _(client):
        client.camera.position = (1.7, -2.0, 1.5)
        client.camera.look_at = (0.0, 0.0, 0.75)

    update()
    print(f"CPU asset viewer: http://{args.host}:{server.get_port()}", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
