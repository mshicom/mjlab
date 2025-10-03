"""NuBot Z1 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  rpm_to_rad,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import ActuatorCfg, CollisionCfg

##
# MJCF and assets.
##

Z1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "nubot_z1" / "xmls" / "z1.xml"
)
assert Z1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, Z1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(Z1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Motor specs (from ENCOS). not verified.
ARMATURE_4310_P2_36 = 31.256e-6
ACTUATOR_4310_P2_36 = ElectricActuator(
  reflected_inertia=ARMATURE_4310_P2_36,
  velocity_limit=rpm_to_rad(87),
  effort_limit=36.0,
)

ARMATURE_6408_P2_25 = 51.457e-6
ACTUATOR_6408_P2_25 = ElectricActuator(
  reflected_inertia=ARMATURE_6408_P2_25,
  velocity_limit=rpm_to_rad(135),
  effort_limit=60.0,
)

ARMATURE_8112_18 = 96.86e-6
ACTUATOR_8112_18 = ElectricActuator(
  reflected_inertia=ARMATURE_8112_18,
  velocity_limit=rpm_to_rad(157),
  effort_limit=90.0,
)

ARMATURE_10020_12 = 465.415e-6
ACTUATOR_10020_12 = ElectricActuator(
  reflected_inertia=ARMATURE_10020_12,
  velocity_limit=rpm_to_rad(140),
  effort_limit=150.0,
)

ARMATURE_10020_P2_24 = 487.314e-6
ACTUATOR_10020_P2_24 = ElectricActuator(
  reflected_inertia=ARMATURE_10020_P2_24,
  velocity_limit=rpm_to_rad(126),
  effort_limit=330,
)

ARMATURE_13715_12 = 1203.149e-6
ACTUATOR_13715_12 = ElectricActuator(
  reflected_inertia=ARMATURE_13715_12,
  velocity_limit=rpm_to_rad(135),
  effort_limit=320,
)

ARMATURE_13720_11 = 1469.242e-6
ACTUATOR_13720_11 = ElectricActuator(
  reflected_inertia=ARMATURE_13720_11,
  velocity_limit=rpm_to_rad(126),
  effort_limit=400,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_4310_P2_36 = ARMATURE_4310_P2_36 * NATURAL_FREQ**2
STIFFNESS_6408_P2_25 = ARMATURE_6408_P2_25 * NATURAL_FREQ**2
STIFFNESS_8112_18 = ARMATURE_8112_18 * NATURAL_FREQ**2
STIFFNESS_10020_12 = ARMATURE_10020_12 * NATURAL_FREQ**2
STIFFNESS_10020_P2_24 = ARMATURE_10020_P2_24 * NATURAL_FREQ**2
STIFFNESS_13715_12 = ARMATURE_13715_12 * NATURAL_FREQ**2
STIFFNESS_13720_11 = ARMATURE_13720_11 * NATURAL_FREQ**2

DAMPING_4310_P2_36 = 2.0 * DAMPING_RATIO * ARMATURE_4310_P2_36 * NATURAL_FREQ
DAMPING_6408_P2_25 = 2.0 * DAMPING_RATIO * ARMATURE_6408_P2_25 * NATURAL_FREQ
DAMPING_8112_18 = 2.0 * DAMPING_RATIO * ARMATURE_8112_18 * NATURAL_FREQ
DAMPING_10020_12 = 2.0 * DAMPING_RATIO * ARMATURE_10020_12 * NATURAL_FREQ
DAMPING_10020_P2_24 = 2.0 * DAMPING_RATIO * ARMATURE_10020_P2_24 * NATURAL_FREQ
DAMPING_13715_12 = 2.0 * DAMPING_RATIO * ARMATURE_13715_12 * NATURAL_FREQ
DAMPING_13720_11 = 2.0 * DAMPING_RATIO * ARMATURE_13720_11 * NATURAL_FREQ


Z1_ACTUATOR_13715_12 = ActuatorCfg(
  joint_names_expr=[".*_hip_pitch_joint"],
  effort_limit=ACTUATOR_13715_12.effort_limit,
  armature=ACTUATOR_13715_12.reflected_inertia,
  stiffness=STIFFNESS_13715_12,
  damping=DAMPING_13715_12,
)
Z1_ACTUATOR_10020_12 = ActuatorCfg(
  joint_names_expr=[".*_hip_yaw_joint"],
  effort_limit=ACTUATOR_10020_12.effort_limit,
  armature=ACTUATOR_10020_12.reflected_inertia,
  stiffness=STIFFNESS_10020_12,
  damping=DAMPING_10020_12,
)
Z1_ACTUATOR_10020_P2_24 = ActuatorCfg(
  joint_names_expr=[".*_hip_roll_joint"],
  effort_limit=ACTUATOR_10020_P2_24.effort_limit,
  armature=ACTUATOR_10020_P2_24.reflected_inertia,
  stiffness=STIFFNESS_10020_P2_24,
  damping=DAMPING_10020_P2_24,
)
Z1_ACTUATOR_13720_11 = ActuatorCfg(
  joint_names_expr=[".*_knee_joint"],
  effort_limit=ACTUATOR_13720_11.effort_limit,
  armature=ACTUATOR_13720_11.reflected_inertia,
  stiffness=STIFFNESS_13720_11,
  damping=DAMPING_13720_11,
)
Z1_ACTUATOR_8112_18 = ActuatorCfg(
  joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"],
  effort_limit=ACTUATOR_8112_18.effort_limit,
  armature=ACTUATOR_8112_18.reflected_inertia,
  stiffness=STIFFNESS_8112_18,
  damping=DAMPING_8112_18,
)
Z1_ACTUATOR_6408_P2_25 = ActuatorCfg(
  joint_names_expr=[".*_shoulder_yaw_joint", ".*_elbow_joint"],
  effort_limit=ACTUATOR_6408_P2_25.effort_limit,
  armature=ACTUATOR_6408_P2_25.reflected_inertia,
  stiffness=STIFFNESS_6408_P2_25,
  damping=DAMPING_6408_P2_25,
)
Z1_ACTUATOR_4310_P2_36 = ActuatorCfg(
  joint_names_expr=[".*_wrist_roll_joint",],
  effort_limit=ACTUATOR_4310_P2_36.effort_limit,
  armature=ACTUATOR_4310_P2_36.reflected_inertia,
  stiffness=STIFFNESS_4310_P2_36,
  damping=DAMPING_4310_P2_36,
)
# Ankle pitch/roll and shoulders are 4-bar linkages with 2 8112 actuators.
# Due to the parallel linkage, the effective armature at the ankle and waist joints
# is configuration dependent. Since the exact geometry of the linkage is unknown, we
# assume a nominal 1:1 gear ratio. Under this assumption, the joint armature in the
# nominal configuration is approximated as the sum of the 2 actuators' armatures.
Z1_ACTUATOR_ANKLE = ActuatorCfg(
  joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
  effort_limit=ACTUATOR_8112_18.effort_limit * 2,
  armature=ACTUATOR_8112_18.reflected_inertia * 2,
  stiffness=STIFFNESS_8112_18 * 2,
  damping=DAMPING_8112_18 * 2,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.07),
  joint_pos={
    ".*_hip_pitch_joint": 0,
    ".*_knee_joint": 0,
    ".*_ankle_pitch_joint": 0,
    ".*_shoulder_pitch_joint": 0,
    ".*_elbow_joint": 0,
    "left_shoulder_roll_joint": 0,
    "right_shoulder_roll_joint": 0,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.0342),
  joint_pos={
    "left_hip_pitch_joint": 0.2792526803190927,
    "right_hip_pitch_joint": -0.2792526803190927,
    ".*_knee_joint": 0.5585053606381855,
    "left_ankle_pitch_joint": 0.2792526803190927,
    "right_ankle_pitch_joint": -0.2792526803190927,
    ".*_elbow_joint": 0.6,
    ".*_shoulder_pitch_joint": 0,
    "left_shoulder_roll_joint": 0.33161255787892263,
    "right_shoulder_roll_joint": -0.33161255787892263,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3 and custom friction and solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=[".*_collision"],
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=[".*_collision"],
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=[r"^(left|right)_foot[1-7]_collision$"],
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

Z1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    Z1_ACTUATOR_13715_12,
    Z1_ACTUATOR_10020_12,
    Z1_ACTUATOR_10020_P2_24,
    Z1_ACTUATOR_13720_11,
    Z1_ACTUATOR_8112_18,
    Z1_ACTUATOR_6408_P2_25,
    Z1_ACTUATOR_4310_P2_36,
    Z1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)

Z1_ROBOT_CFG = EntityCfg(
  init_state=KNEES_BENT_KEYFRAME,
  collisions=(FULL_COLLISION,),
  spec_fn=get_spec,
  articulation=Z1_ARTICULATION,
)

Z1_ACTION_SCALE: dict[str, float] = {}
for a in Z1_ARTICULATION.actuators:
  e = a.effort_limit
  s = a.stiffness
  names = a.joint_names_expr
  if not isinstance(e, dict):
    e = {n: e for n in names}
  if not isinstance(s, dict):
    s = {n: s for n in names}
  for n in names:
    if n in e and n in s and s[n]:
      Z1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(Z1_ROBOT_CFG)

  viewer.launch(robot.spec.compile())
