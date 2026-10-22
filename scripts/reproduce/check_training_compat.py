#!/usr/bin/env python3
"""Check actual object assignments and contact interfaces in a small simulation."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/rsl_rl"))
import cli_args

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="G1-Inspire-HOI-v0")
parser.add_argument("--num-envs", type=int, default=19)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--output", type=Path, required=True)
cli_args.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
if args.num_envs < 1 or args.steps < 1:
    parser.error("--num-envs and --steps must be positive")
sys.argv = [sys.argv[0]] + hydra_args
launcher = AppLauncher(args)
simulation_app = launcher.app

import json
from importlib.metadata import PackageNotFoundError, version

import g1_hoi_learning.tasks  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from g1_hoi_learning.isaac_compat import (
    ISAACLAB3,
    as_torch,
    net_normal_contact_forces,
    normal_contact_force_matrix,
)
from g1_hoi_learning.objects.object_cfg import OBJECT_CFG_BY_NAME
from g1_hoi_learning.scene_layout import validate_object_assignments
from hydra_compat import hydra_task_config
from viser_options import run_and_close

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401


def first_clip_files(files, destination):
    """Keep one complete real reference per file without changing the source datasets."""
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            lengths = np.asarray(data["motion_lengths"])
            total, first = int(lengths.sum()), int(lengths[0])
            selected = {}
            for key in data.files:
                value = np.asarray(data[key])
                if value.ndim and value.shape[0] == total:
                    value = value[:first]
                elif value.ndim and value.shape[0] == len(lengths):
                    value = value[:1]
                selected[key] = value
            output = destination / Path(path).name
            np.savez(output, **selected)
            outputs.append(str(output.resolve()))
    return outputs


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(cfg, _agent):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = 42
    cfg.commands.motion.motion_files = first_clip_files(
        cfg.commands.motion.motion_files, output.parent / (output.stem + "-motions")
    )
    cfg.commands.motion.eval_mode = False
    cfg.commands.motion.rsi = False
    if not ISAACLAB3:
        # Reuse the same robot conversion and a cached ground in the old standalone runtime.
        cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
            usd_path=str(REPO / "data/isaac62/g1_inspire_usd/g1_29dof_rev_1_0_with_inspire_hand_DFQ.usd"),
            activate_contact_sensors=True,
            rigid_props=cfg.scene.robot.spawn.rigid_props,
            articulation_props=cfg.scene.robot.spawn.articulation_props,
        )
        cached_ground = Path(
            "/tmp/https/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/"
            "IsaacLab/Environments/Grid/default_ground_plane_checker_v1/default_ground_plane.usda"
        )
        if cached_ground.is_file():
            cfg.scene.ground.spawn.usd_path = str(cached_ground)
    env = gym.make(args.task, cfg=cfg).unwrapped
    if not ISAACLAB3:
        env.sim._disable_app_control_on_stop_handle = True
    try:
        obs, _ = env.reset()
        for _ in range(args.steps):
            obs, rewards, _, _, _ = env.step(
                torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
            )
            assert torch.isfinite(rewards).all()
            assert all(torch.isfinite(as_torch(value)).all() for value in obs.values())
        command = env.command_manager.get_term("motion")
        names = command.motion.object_names
        expected = [names[i] for i in command.env_object.cpu().tolist()]
        counts = validate_object_assignments(
            sim_utils.get_current_stage(), expected,
            {name: OBJECT_CFG_BY_NAME[name].spawn.usd_path for name in names},
        )
        contact = env.scene["contact_sensor"]
        feet = env.scene["feet_contact_sensor"]
        matrix = normal_contact_force_matrix(contact)
        history = net_normal_contact_forces(feet, history=True)
        if ISAACLAB3:
            assert matrix.data_ptr() == as_torch(contact.data.normal_force_matrix_w).data_ptr()
            assert history.data_ptr() == as_torch(feet.data.net_normal_forces_w_history).data_ptr()
        physics = cfg.sim.physics if ISAACLAB3 else cfg.sim.physx
        assert physics.enable_external_forces_every_iteration is False
        try:
            sim_version = version("isaacsim")
        except PackageNotFoundError:
            sim_version = "standalone"
        result = {
            "passed": True, "task": args.task, "isaaclab": version("isaaclab"), "isaacsim": sim_version,
            "num_envs": env.num_envs, "steps": args.steps, "object_counts": counts,
            "all_actual_objects_match_references": True,
            "support_tables_verified_by_factory": hasattr(cfg.scene, "support_table"),
            "normal_contact_matrix_shape": list(matrix.shape), "normal_feet_history_shape": list(history.shape),
            "normal_contact_api_verified": True, "enable_external_forces_every_iteration": False,
            "observations_and_rewards_finite": True,
        }
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    run_and_close(main, simulation_app)
