# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv

from g1_hoi_learning.isaac_compat import ISAACLAB3
from g1_hoi_learning.objects.object_cfg import OBJECT_CFG_BY_NAME
from g1_hoi_learning.scene_layout import (
    paired_round_robin_clone_strategy,
    round_robin_clone_strategy,
    validate_object_assignments,
)

from . import agents


def _make_env(cfg, **kwargs):
    """Resolve scene.object from motion_files' object_names.

    Assign env i to object (i % O), matching MotionCommand.env_object. Lab 3 requires an
    explicit clone strategy; random_choice=False alone no longer controls the assignment.
    """
    from .mdp.commands import MotionLoader

    motion_files = cfg.commands.motion.motion_files

    # Unique objects across all clips, in the SAME order MotionLoader uses, so the object spawned for
    # env i (assets_cfg[i % O]) matches that env's fixed object (env_object = arange % O).
    object_names = MotionLoader.object_names_of(motion_files)
    assets_spawn_cfg: list[sim_utils.SpawnerCfg] = []
    for name in object_names:
        if name not in OBJECT_CFG_BY_NAME:
            raise KeyError(f"Unknown object_name '{name}'; not in OBJECT_CFG_BY_NAME")
        assets_spawn_cfg.append(OBJECT_CFG_BY_NAME[name].spawn)

    cfg.scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=assets_spawn_cfg,
            random_choice=False,
            activate_contact_sensors=True,
        ),
    )
    cfg.scene.replicate_physics = False
    if ISAACLAB3:
        table = getattr(cfg.scene, "support_table", None)
        paired = table is not None and isinstance(table.spawn, sim_utils.MultiAssetSpawnerCfg)
        if paired and len(table.spawn.assets_cfg) != len(object_names):
            raise ValueError("Support-table variants must match the reference object's variant order.")
        cfg.scene.clone_cfg.clone_strategy = (
            paired_round_robin_clone_strategy if paired else round_robin_clone_strategy
        )

    print(f"[INFO]: Resolved scene.object across {len(object_names)} object(s): {object_names}")
    env = ManagerBasedRLEnv(cfg=cfg, **kwargs)
    try:
        command = env.command_manager.get_term("motion")
        expected = [command.motion.object_names[i] for i in command.env_object.cpu().tolist()]
        counts = validate_object_assignments(
            sim_utils.get_current_stage(), expected,
            {name: OBJECT_CFG_BY_NAME[name].spawn.usd_path for name in object_names},
        )
    except Exception:
        env.close()
        raise
    print(f"[WEAVE] Verified all {env.num_envs} object/reference assignments: {counts}", flush=True)
    return env


def _make_pickup_env(cfg, **kwargs):
    from .mdp.commands import MotionLoader
    from .pickup_cfg import configure_pickup_table

    object_names = MotionLoader.object_names_of(cfg.commands.motion.motion_files)
    configure_pickup_table(cfg, object_names)
    env = _make_env(cfg, **kwargs)
    from pxr import Usd, UsdPhysics

    stage = sim_utils.get_current_stage()
    command = env.command_manager.get_term("motion")
    for index, object_id in enumerate(command.env_object.cpu().tolist()):
        obj = command.motion.object_names[object_id]
        root = stage.GetPrimAtPath(f"/World/envs/env_{index}/SupportTable")
        collision_enabled = any(
            prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            for prim in Usd.PrimRange(root)
        )
        if bool(collision_enabled) != (obj == "whitechair"):
            raise RuntimeError(f"Incorrect support-table collision assignment in env {index}: {obj}")
    print("[PICKUPS] Verified table collisions for whitechair and no table collisions for floor tasks.")
    return env


gym.register(
    id="G1-Inspire-HOI-v0",
    entry_point=_make_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1HoiLearningEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="G1-Inspire-HOI-Pickup-v0",
    entry_point=_make_pickup_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_cfg:G1HoiPickupEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
