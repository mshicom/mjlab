"""Tests for entity module."""

from pathlib import Path
import os

import mujoco
import numpy as np
import pytest
import torch

from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg, EntityType
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.spec_config import ActuatorCfg
from mjlab.scene.scene import RecordCfg


def get_test_device() -> str:
  """Get device for testing, preferring CUDA if available."""
  if torch.cuda.is_available():
    return "cuda"
  return "cpu"


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture
def fixed_base_xml():
  """XML for a simple fixed-base entity."""
  return """
    <mujoco>
      <worldbody>
        <body name="object" pos="0 0 0.5">
          <geom name="object_geom" type="box" size="0.1 0.1 0.1" rgba="0.8 0.3 0.3 1"/>
        </body>
      </worldbody>
    </mujoco>
    """


@pytest.fixture
def floating_base_xml():
  """XML for a floating-base entity with freejoint."""
  return """
    <mujoco>
      <worldbody>
        <body name="object" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="object_geom" type="box" size="0.1 0.1 0.1" rgba="0.3 0.3 0.8 1" mass="0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """


@pytest.fixture
def articulated_xml():
  """XML for an articulated entity with joints."""
  return """
    <mujoco>
      <worldbody>
        <body name="base" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
          <body name="link1" pos="0 0 0">
            <joint name="joint1" type="hinge" axis="0 0 1" range="0 1.57"/>
            <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
            <site name="site1" pos="0 0 0"/>
          </body>
          <body name="link2" pos="0 0 0">
            <joint name="joint2" type="hinge" axis="0 0 1" range="0 1.57"/>
            <geom name="link2_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
          </body>
        </body>
      </worldbody>
      <sensor>
        <jointpos name="joint1_pos" joint="joint1"/>
      </sensor>
    </mujoco>
    """


@pytest.fixture
def fixed_articulated_xml():
  """XML for a fixed-base articulated entity (e.g., robot arm bolted to ground)."""
  return """
    <mujoco>
      <worldbody>
        <body name="base" pos="0 0 0.5">
          <geom name="base_geom" type="cylinder" size="0.1 0.05" mass="5.0"/>
          <body name="link1" pos="0 0 0.1">
            <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
            <geom name="link1_geom" type="box" size="0.05 0.05 0.2" mass="1.0"/>
            <body name="link2" pos="0 0 0.4">
              <joint name="joint2" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
              <geom name="link2_geom" type="box" size="0.05 0.05 0.15" mass="0.5"/>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """


@pytest.fixture
def fixed_articulated_entity(fixed_articulated_xml, actuator_cfg):
  """Create a fixed-base articulated entity."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(fixed_articulated_xml),
    articulation=actuator_cfg,
  )
  return Entity(cfg)


@pytest.fixture
def actuator_cfg():
  """Standard actuator configuration."""
  return EntityArticulationInfoCfg(
    actuators=(
      ActuatorCfg(
        joint_names_expr=["joint1", "joint2"],
        effort_limit=1.0,
        stiffness=1.0,
        damping=1.0,
      ),
    )
  )


@pytest.fixture
def fixed_base_entity(fixed_base_xml):
  """Create a fixed-base entity."""
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(fixed_base_xml))
  return Entity(cfg)


@pytest.fixture
def floating_base_entity(floating_base_xml):
  """Create a floating-base entity."""
  cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(floating_base_xml))
  return Entity(cfg)


@pytest.fixture
def articulated_entity(articulated_xml, actuator_cfg):
  """Create an articulated entity with actuators."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(articulated_xml),
    articulation=actuator_cfg,
  )
  return Entity(cfg)


@pytest.fixture
def initialized_floating_entity(floating_base_entity, device):
  """Create an initialized floating-base entity with simulation."""

  entity = floating_base_entity
  model = entity.compile()

  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  entity.initialize(model, sim.model, sim.data, device)

  return entity, sim


@pytest.fixture
def initialized_articulated_entity(articulated_entity, device):
  """Create an initialized articulated entity with simulation."""

  entity = articulated_entity
  model = entity.compile()

  sim_cfg = SimulationCfg()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  entity.initialize(model, sim.model, sim.data, device)
  return entity, sim


class TestEntityProperties:
  """Test entity property detection and element counts."""

  @pytest.mark.parametrize(
    "entity_fixture,expected",
    [
      (
        "fixed_base_entity",
        {
          "is_fixed_base": True,
          "is_articulated": False,
          "is_actuated": False,
          "num_bodies": 1,
          "num_joints": 0,
          "num_actuators": 0,
        },
      ),
      (
        "floating_base_entity",
        {
          "is_fixed_base": False,
          "is_articulated": False,
          "is_actuated": False,
          "num_bodies": 1,
          "num_joints": 0,
          "num_actuators": 0,
        },
      ),
      (
        "articulated_entity",
        {
          "is_fixed_base": False,
          "is_articulated": True,
          "is_actuated": True,
          "num_bodies": 3,
          "num_joints": 2,
          "num_actuators": 2,
        },
      ),
      (
        "fixed_articulated_entity",
        {
          "is_fixed_base": True,
          "is_articulated": True,
          "is_actuated": True,
          "num_bodies": 3,
          "num_joints": 2,
          "num_actuators": 2,
        },
      ),
    ],
  )
  def test_entity_properties(self, entity_fixture, expected, request):
    """Test entity type properties and element counts."""
    entity = request.getfixturevalue(entity_fixture)

    for prop, value in expected.items():
      assert getattr(entity, prop) == value


class TestFindMethods:
  """Test entity element finding methods."""

  def test_find_methods(self, articulated_entity):
    """Test find methods with exact and regex matches."""
    # Test exact matches.
    assert articulated_entity.find_bodies("base")[1] == ["base"]
    assert articulated_entity.find_joints("joint1")[1] == ["joint1"]
    assert articulated_entity.find_sites("site1")[1] == ["site1"]

    # Test regex matches.
    assert articulated_entity.find_bodies("link.*")[1] == ["link1", "link2"]
    assert articulated_entity.find_joints("joint.*")[1] == ["joint1", "joint2"]

    # Test subset filtering.
    _, names = articulated_entity.find_joints(
      "joint1", joint_subset=["joint1", "joint2"]
    )
    assert names == ["joint1"]

    # Test error on invalid subset.
    with pytest.raises(ValueError, match="Not all regular expressions are matched"):
      articulated_entity.find_joints("joint1", joint_subset=["joint2"])


class TestStateManagement:
  """Test reading and writing entity states."""

  def test_root_state_floating_base(self, initialized_floating_entity, device):
    """Test root state operations affect simulation correctly."""
    entity, sim = initialized_floating_entity

    # Set entity with specific state.
    # fmt: off
    root_state = torch.tensor([
        1.0, 2.0, 3.0,           # position
        1.0, 0.0, 0.0, 0.0,      # quaternion (identity)
        0.5, 0.0, 0.0,           # linear velocity in X
        0.0, 0.0, 0.2            # angular velocity around Z
    ], device=device).unsqueeze(0)
    # fmt: on

    entity.write_root_state_to_sim(root_state)

    # Verify the state was actually written.
    q_slice = entity.data.indexing.free_joint_q_adr
    v_slice = entity.data.indexing.free_joint_v_adr

    assert torch.allclose(sim.data.qpos[:, q_slice], root_state[:, :7])
    assert torch.allclose(sim.data.qvel[:, v_slice], root_state[:, 7:])

    # Step once and verify physics is working (gravity should affect Z velocity).
    initial_z_vel = sim.data.qvel[0, v_slice[2]].item()
    sim.step()
    final_z_vel = sim.data.qvel[0, v_slice[2]].item()

    # Z velocity should decrease (become more negative) due to gravity.
    assert final_z_vel < initial_z_vel, "Gravity should affect Z velocity"


class TestExternalForces:
  """Test external force and torque application."""

  def test_force_and_torque_basic(self, initialized_floating_entity):
    """Test forces translate, torques rotate, and forces can be cleared."""
    entity, sim = initialized_floating_entity

    # Apply force in X, torque around Z.
    entity.write_external_wrench_to_sim(
      forces=torch.tensor([[5.0, 0.0, 0.0]], device=sim.device),
      torques=torch.tensor([[0.0, 0.0, 3.0]], device=sim.device),
    )

    initial_pos = sim.data.qpos[0, :3].clone()
    initial_quat = sim.data.qpos[0, 3:7].clone()

    for _ in range(10):
      sim.step()

    # Verify X translation and rotation occurred.
    assert sim.data.qpos[0, 0] > initial_pos[0], "Force should cause X translation"
    assert not torch.allclose(sim.data.qpos[0, 3:7], initial_quat), (
      "Torque should cause rotation"
    )

    # Verify angular velocity is primarily around Z (relative comparison).
    angular_vel = sim.data.qvel[0, 3:6]
    z_rotation = abs(angular_vel[2])
    xy_rotation = abs(angular_vel[0]) + abs(angular_vel[1])
    assert z_rotation > xy_rotation * 5, "Rotation should be primarily around Z axis"

    # Test force clearing.
    entity.write_external_wrench_to_sim(
      forces=torch.zeros((1, 3), device=sim.device),
      torques=torch.zeros((1, 3), device=sim.device),
    )

    # Verify forces are cleared.
    body_id = entity.indexing.body_ids[0]
    assert torch.allclose(
      sim.data.xfrc_applied[:, body_id, :], torch.zeros(6, device=sim.device)
    )

    # Verify gravity still works after clearing.
    initial_z = sim.data.qpos[0, 2].clone()
    sim.step()
    assert sim.data.qpos[0, 2] < initial_z, "Should fall due to gravity"

  def test_force_on_specific_body(self, initialized_articulated_entity):
    """Test applying force to specific body in articulated system."""
    entity, sim = initialized_articulated_entity

    # Apply force only to link1.
    body_ids = entity.find_bodies("link1")[0]
    entity.write_external_wrench_to_sim(
      forces=torch.tensor([[3.0, 0.0, 0.0]], device=sim.device),
      torques=torch.zeros((1, 3), device=sim.device),
      body_ids=body_ids,
    )

    # Verify force applied only to link1.
    link1_id = sim.mj_model.body("link1").id
    base_id = sim.mj_model.body("base").id
    assert torch.allclose(
      sim.data.xfrc_applied[0, link1_id, :3],
      torch.tensor([3.0, 0.0, 0.0], device=sim.device),
    )
    assert torch.allclose(
      sim.data.xfrc_applied[0, base_id, :3], torch.zeros(3, device=sim.device)
    )

    # Verify motion occurs.
    initial_pos = sim.data.xpos[0, link1_id, :].clone()
    for _ in range(10):
      sim.step()
    assert not torch.allclose(sim.data.xpos[0, link1_id, :], initial_pos)

  def test_large_force_stability(self, initialized_floating_entity):
    """Test system handles large forces without numerical issues."""
    entity, sim = initialized_floating_entity

    entity.write_external_wrench_to_sim(
      forces=torch.tensor([[1e6, 0.0, 0.0]], device=sim.device),
      torques=torch.zeros((1, 3), device=sim.device),
    )

    sim.step()
    assert not torch.any(torch.isnan(sim.data.qpos)), "Should not produce NaN"


# ============================================================================
# New tests for record qvel reconstruction, resampling and recording
# ============================================================================

def test_qvel_reconstruction_from_qpos_only(tmp_path: Path, articulated_entity: Entity, device: str):
  """Generate synthetic qpos-only npz and verify qvel is reconstructed correctly."""
  # Build a simple articulated entity as target
  target_ent = articulated_entity
  mj_model = target_ent.compile()

  # Sim wrapper for indexing creation
  from mjlab.sim.sim import Simulation, SimulationCfg
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=mj_model, device=device)
  target_ent.initialize(mj_model, sim.model, sim.data, device)

  # Create synthetic qpos (free + 2 joints) with known linear progression
  T = 10
  freq = 50.0  # Hz
  dt = 1.0 / freq
  pos = np.stack([np.linspace(0, 0.09, T), np.zeros(T), np.zeros(T)], axis=1)
  quat = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (T, 1))
  j1 = np.linspace(0, 0.9, T)[:, None]
  j2 = np.linspace(0, -0.45, T)[:, None]
  qpos = np.concatenate([pos, quat, j1, j2], axis=1).astype(np.float32)

  arrays = {
    "qpos": qpos,
    "_info": {
      "joint_names": target_ent.joint_names_with_free(),
      "jnt_type": np.array([int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_HINGE)], dtype=np.int32),
      "body_names": target_ent.body_names,
      "site_names": target_ent.site_names,
      "frequency": freq,
    },
  }
  npz_path = tmp_path / "qpos_only.npz"
  # Save minimal npz with qpos only
  np.savez(str(npz_path), **{"qpos": arrays["qpos"], "_info": np.array(arrays["_info"], dtype=object)})

  # Create a record entity aliasing the target spec and load
  rec_ent = Entity(EntityCfg(), entity_type=EntityType.REC)
  rec_ent.bind_target_entity(target_ent)
  rec_ent.initialize(mj_model, sim.model, sim.data, device)
  rec_ent.load_record(
    npz_path=npz_path,
    source_xml=None,
    mj_model=mj_model,
    record_frequency=freq,
  )

  # Check qvel reconstructed matches expected finite differences
  # Bind frames and compare local slices
  count = 0
  for idx in rec_ent.frames(0, T):
    qvel = rec_ent.data.data.qvel[0].detach().cpu().numpy()
    # Expected: first frame zeros; later frames approx diff * freq
    if idx == 0:
      assert np.allclose(qvel, qvel * 0.0)
    count += 1
  assert count == T


def test_record_roundtrip_smoke(tmp_path: Path, articulated_entity: Entity, device: str):
  """Roundtrip: start_record -> simulate few steps -> add_frame -> save_record(qpos only) -> reload -> verify buffer."""
  ent = articulated_entity
  mj_model = ent.compile()
  from mjlab.sim.sim import Simulation, SimulationCfg
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=mj_model, device=device)
  ent.initialize(mj_model, sim.model, sim.data, device)

  # Record a few frames during simulation
  ent.start_record(frequency_hz=100.0)
  for _ in range(5):
    sim.step()
    ent.add_frame()
  out_path = tmp_path / "roundtrip_qpos_only.npz"
  ent.save_record(out_path, qpos_only=True)
  assert out_path.exists()

  # Reload into record entity and validate buffer
  rec_ent = Entity(EntityCfg(), entity_type=EntityType.REC)
  rec_ent.bind_target_entity(ent)
  rec_ent.initialize(mj_model, sim.model, sim.data, device)
  rec_ent.load_record(
    npz_path=out_path,
    source_xml=None,
    mj_model=mj_model,
    record_frequency=100.0,
  )
  # Iterate frames to ensure binding works and shapes align
  for _ in rec_ent.frames(0, 5):
    assert rec_ent.data.data.qpos.shape[0] == 1  # nworld==1 offline
    assert rec_ent.data.data.qpos.shape[1] == sim.data.qpos.shape[1]


@pytest.mark.skipif(
  not (os.path.exists("/workspaces/ws_rl/data/loco-mujoco-datasets/DefaultDatasets/mocap/UnitreeG1/stepinplace1.npz")
       and os.path.exists("/workspaces/ws_rl/src/loco-mujoco/loco_mujoco/models/unitree_g1/g1_23dof.xml")),
  reason="External dataset or XML not available in this environment.",
)
def test_load_real_npz_and_env_integration(device: str):
  """Load a real Unitree G1 npz with provided source_xml via an Env (smoke test)."""
  npz_path = Path("/workspaces/ws_rl/data/loco-mujoco-datasets/DefaultDatasets/mocap/UnitreeG1/stepinplace1.npz")
  source_xml = Path("/workspaces/ws_rl/src/loco-mujoco/loco_mujoco/models/unitree_g1/g1_23dof.xml")

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.tasks.velocity.config.g1.flat_env_cfg import UnitreeG1FlatEnvCfg
  cfg = UnitreeG1FlatEnvCfg()
  cfg.scene.records.append(
    RecordCfg(
      path=npz_path,
      name="rec",
      source_xml=source_xml,
    )
  )
  env = ManagerBasedRlEnv(cfg, device="cpu")
  rec:Entity = env.scene["rec"]
  count = 0
  for _ in rec.frames(0, min(5, getattr(rec, "_rec_len", 5))):
    assert rec.data.data.qpos.shape[0] == 1
    count += 1
  assert count > 0


@pytest.mark.parametrize("f_new", [25.0, 100.0])
def test_interpolate_record_resamples_and_recomputes_kinematics(tmp_path: Path, articulated_entity: Entity, device: str, f_new: float):
  """Create a synthetic qpos-only record at 50 Hz, load it, then interpolate to new frequency and verify arrays."""
  target_ent = articulated_entity
  mj_model = target_ent.compile()

  from mjlab.sim.sim import Simulation, SimulationCfg
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=mj_model, device=device)
  target_ent.initialize(mj_model, sim.model, sim.data, device)

  # Synthetic qpos with linear ramps (free pos x ramps, joints ramp linearly)
  T_old = 6
  f_old = 50.0
  pos = np.stack([np.linspace(0.0, 0.05, T_old), np.zeros(T_old), np.zeros(T_old)], axis=1).astype(np.float32)
  quat = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (T_old, 1))
  j1 = np.linspace(0.0, 0.5, T_old, dtype=np.float32)[:, None]
  j2 = np.linspace(0.0, -0.25, T_old, dtype=np.float32)[:, None]
  qpos = np.concatenate([pos, quat, j1, j2], axis=1).astype(np.float32)

  info = {
    "joint_names": target_ent.joint_names_with_free(),
    "jnt_type": np.array([int(mujoco.mjtJoint.mjJNT_FREE)] + [int(mujoco.mjtJoint.mjJNT_HINGE)] * target_ent.num_joints, dtype=np.int32),
    "body_names": target_ent.body_names,
    "site_names": target_ent.site_names,
    "frequency": f_old,
  }
  npz_path = tmp_path / f"synthetic_qpos_only_for_interp_{f_new}.npz"
  np.savez(str(npz_path), **{"qpos": qpos, "_info": np.array(info, dtype=object)})

  # Load into REC entity
  rec_ent = Entity(EntityCfg(), entity_type=EntityType.REC, alias_spec=target_ent.spec)
  rec_ent.bind_target_entity(target_ent)
  rec_ent.initialize(mj_model, sim.model, sim.data, device)
  rec_ent.load_record(
    npz_path=npz_path,
    source_xml=None,
    mj_model=mj_model,
    record_frequency=f_old,
  )

  # Capture first/last qpos before interpolation
  qpos_first = rec_ent._rec_arrays["qpos"][0].copy()
  qpos_last = rec_ent._rec_arrays["qpos"][-1].copy()

  # Interpolate to new frequency
  rec_ent.interpolate_record(f_new, recompute_kinematics=True)

  # Expected new length: round((T_old-1)/f_old * f_new)+1
  dur = (T_old - 1) / f_old
  T_new = int(round(dur * f_new)) + 1
  assert rec_ent._rec_arrays["qpos"].shape[0] == T_new
  assert rec_ent._rec_arrays["qvel"].shape[0] == T_new
  assert rec_ent._rec_arrays["split_points"].tolist() == [0, T_new]
  assert pytest.approx(rec_ent._rec_arrays["_info"]["frequency"]) == f_new

  # First/last samples preserved
  assert np.allclose(rec_ent._rec_arrays["qpos"][0], qpos_first, atol=1e-6)
  assert np.allclose(rec_ent._rec_arrays["qpos"][-1], qpos_last, atol=1e-6)

  # Kinematics recomputed and have correct shape
  for k in ("xpos", "xquat", "cvel", "subtree_com", "site_xpos", "site_xmat"):
    assert k in rec_ent._rec_arrays
    assert rec_ent._rec_arrays[k].shape[0] == T_new

  # qvel should be non-zero after the first frame for the ramped DOFs
  assert np.any(np.abs(rec_ent._rec_arrays["qvel"][1:]) > 0.0)



if __name__ == "__main__":
  pytest.main([__file__, "-v"])
