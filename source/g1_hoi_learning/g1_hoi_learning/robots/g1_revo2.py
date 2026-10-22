"""Opt-in G1 + BrainCo Revo2 asset. Existing Inspire tasks are unchanged.

The hand gains are initial simulation candidates, not identified hardware gains.
This asset alone does not retarget Inspire references or adapt an existing policy.
"""

import copy
import json
from pathlib import Path

from g1_hoi_learning.assets import G1_REVO2_URDF_PATH
from g1_hoi_learning.robots.g1_inspire import G1_INSPIRE_CFG

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

_metadata_path = Path(G1_REVO2_URDF_PATH).with_name("assembly.json")
if not _metadata_path.is_file():
    raise FileNotFoundError("Run python scripts/reproduce/prepare_revo2.py before loading G1_REVO2_CFG.")
REVO2_METADATA = json.loads(_metadata_path.read_text())
REVO2_MIMIC = {
    name: (spec["driver"], spec["multiplier"], spec["offset"]) for name, spec in REVO2_METADATA["mimic"].items()
}
REVO2_ACTIVE_JOINTS = REVO2_METADATA["body_drive_joints"] + REVO2_METADATA["hand_drive_joints"]

G1_REVO2_CFG = copy.deepcopy(G1_INSPIRE_CFG)
_initial_hand = REVO2_METADATA["initial_hand_joint_positions"]
G1_REVO2_CFG.init_state.joint_pos.update(_initial_hand)
G1_REVO2_CFG.init_state.joint_pos.update(
    {name: _initial_hand[driver] * multiplier + offset for name, (driver, multiplier, offset) in REVO2_MIMIC.items()}
)
_articulation_props = copy.deepcopy(G1_INSPIRE_CFG.spawn.articulation_props)
_articulation_props.enabled_self_collisions = True
G1_REVO2_CFG.spawn = sim_utils.UrdfFileCfg(
    asset_path=str(Path(G1_REVO2_URDF_PATH).with_name(REVO2_METADATA["software_mimic_urdf"])),
    fix_base=False,
    merge_fixed_joints=True,
    replace_cylinders_with_capsules=True,
    # Software MimicJointPositionAction must own the coupled joint targets;
    # do not also create PhysX mimic constraints for the same joints.
    convert_mimic_joints_to_normal_joints=False,
    self_collision=True,
    force_usd_conversion=False,
    activate_contact_sensors=True,
    rigid_props=copy.deepcopy(G1_INSPIRE_CFG.spawn.rigid_props),
    articulation_props=_articulation_props,
    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    ),
)
_limits = REVO2_METADATA["hand_joint_limits"]
G1_REVO2_CFG.actuators["fingers"] = ImplicitActuatorCfg(
    joint_names_expr=list(_limits),
    effort_limit_sim={name: limits["effort"] for name, limits in _limits.items()},
    velocity_limit_sim={name: limits["velocity"] for name, limits in _limits.items()},
    stiffness=5.0,
    damping=0.5,
    armature=0.01,
)
