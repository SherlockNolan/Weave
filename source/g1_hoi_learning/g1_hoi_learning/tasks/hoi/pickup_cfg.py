"""Pickup-only scene with a fixed support surface for the white-chair clip."""
import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from .env_cfg import G1HoiLearningEnvCfg, G1HoiLearningSceneCfg

REPO = Path(__file__).resolve().parents[5]


@configclass
class PickupSceneCfg(G1HoiLearningSceneCfg):
    support_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SupportTable",
        spawn=sim_utils.CuboidCfg(size=(0.001, 0.001, 0.001), visible=False),
    )


@configclass
class G1HoiPickupEnvCfg(G1HoiLearningEnvCfg):
    scene: PickupSceneCfg = PickupSceneCfg(num_envs=4096, env_spacing=4.0)


def configure_pickup_table(cfg, object_names):
    manifest = json.loads((REPO / "data/pickups/manifest.json").read_text())
    table = manifest["support_table"]
    configs = []
    for obj in object_names:
        if obj == table["object"]:
            configs.append(sim_utils.CuboidCfg(
                size=tuple(table["size"]),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.28, 0.12)),
            ))
        else:
            # Invisible, non-colliding placeholder keeps the round-robin asset
            # assignment aligned with the target object in every environment.
            configs.append(sim_utils.CuboidCfg(size=(0.001, 0.001, 0.001), visible=False))
    cfg.scene.support_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SupportTable",
        spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=configs, random_choice=False),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(table["position"])),
    )
    print(f"[PICKUPS] Support tabletop only for whitechair: {table}", flush=True)
