from dataclasses import replace
import mujoco
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from typing import List

from mjlab.utils.dataset.traj_class import Trajectory, TrajectoryData, interpolate_trajectories


@struct.dataclass
class EmptyState:
    pass


class StatefulObject:

    _instances: List["StatefulObject"] = []

    def __init__(self, n_visual_geoms: int =0):
        self.n_visual_geoms = n_visual_geoms
        self.visual_geoms_idx = None
        self._instances.append(self)

    def reset_state(self, carry, backend):
        return carry

    def init_state(self, env, key, model, data, backend):
        return EmptyState()

    @classmethod
    def get_all_instances(cls) -> List["MjvGeom"]:
        """Returns a list of all instances of this class."""
        return cls._instances


@struct.dataclass
class TrajState:
    traj_no: int
    subtraj_step_no: int
    subtraj_step_no_init: int


@struct.dataclass
class TrajCarry:
    key: jax.Array
    traj_state: TrajState


def LenTrajectory(traj_data, traj_ind):
    return traj_data.split_points[traj_ind + 1] - traj_data.split_points[traj_ind]

class TrajectoryHandler(StatefulObject):
    """
    General class to handle Trajectories. It filters and extends the trajectory data to match
    the current model's joints, bodies and sites. The key idea is to ensure that TrajectoryData has the same
    dimensionality and order for all its attributes as in the Mujoco data structure. So TrajectoryData is a
    simplified version of the Mujoco data structure with fewer attributes. This class also automatically
    interpolates the trajectory to the desired control frequency.

    """
    traj: Trajectory

    def __init__(self, model, traj_path=None, traj: Trajectory = None, control_dt=0.01, random_start=True,
                 fixed_start_conf=None, clip_trajectory_to_joint_ranges=False, warn=True, start_range=None):
        """
        Constructor.

        Args:
            model (mjModel): Current model.
            traj_path (string): path with the trajectory for the model to follow. Should be a numpy zipped file (.npz)
                with a 'traj_data' array and possibly a 'split_points' array inside. The 'traj_data'
                should be in the shape (joints x observations). If traj_files is specified, this should be None.
            traj (Trajectory): Datastructure containing all trajectory files. If traj_path is specified, this
                should be None.
            control_dt (float): Model control frequency used to interpolate the trajectory.
            clip_trajectory_to_joint_ranges (bool): If True, the joint positions in the trajectory are clipped
                between the low and high values in the trajectory. todo
            warn (bool): If True, a warning will be raised, if some trajectory ranges are violated. todo

        """

        assert (traj_path is not None) != (traj is not None), ("Please specify either traj_path or "
                                                               "trajectory, but not both.")

        # load data
        if traj_path is not None:
            traj = Trajectory.load(traj_path)

        # filter/extend the trajectory based on the model/data
        traj_data, traj_info = self.filter_and_extend(traj.data, traj.info, model)

        # todo: implement this in observation types in init_from_traj!
        #self.check_if_trajectory_is_in_range(low, high, keys, joint_pos_idx, warn, clip_trajectory_to_joint_ranges)

        assert (fixed_start_conf is not None) != random_start, "Please specify either fixed_start_conf or random_start."
        # assert not random_start or start_range, "If random_start is True, start_range must not be None."
        self.random_start = random_start
        self.fixed_start_conf = fixed_start_conf
        self.use_fixed_start = True if fixed_start_conf is not None else False
        self.start_range = start_range

        self.traj_dt = 1 / traj_info.frequency
        self.control_dt = control_dt

        if self.traj_dt != self.control_dt:
            traj_data, traj_info = interpolate_trajectories(traj_data, traj_info, 1.0 / self.control_dt)

        self._is_numpy = True if isinstance(traj_data.qpos, np.ndarray) else False
        self.traj = replace(traj, data=traj_data, info=traj_info)

    def len_trajectory(self, traj_ind):
        return self.traj.data.split_points[traj_ind + 1] - self.traj.data.split_points[traj_ind]    

    @property
    def n_trajectories(self):
        return len(self.traj.data.split_points) - 1

    @staticmethod
    def filter_and_extend(traj_data, traj_info, model):
        """
        To ensure that the data structure of the current model and the trajectory data have the same dimensionality
        and order for all supported attributes, this function filters the elements present in the trajectory but not
        the current model and extends the trajectory data's joints, bodies and sites with elements present in
        the current model but not the trajectory. It is doing so by adding dummy joints, bodies and sites to the
        trajectory data if they are not present in the trajectory data but in the model. It also reorders the
        joints, bodies and sites based on the model.

        Args:
            traj_data (TrajectoryData): Trajectory data to be filtered and extended.
            traj_info (TrajectoryInfo): Trajectory info to be filtered and extended.
            model (mjModel): Current model.

        Returns:
            TrajectoryData, TrajectoryInfo: Filtered and extended trajectory data and trajectory info.

        """

        # --- filter the trajectory based on the model and data ---
        # get the joint names from current model
        joint_names = []
        joint_ids = []
        joint_name2id_qpos = dict()
        joint_name2id_qvel = dict()
        j_qpos, j_qvel = 0, 0
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            j_type = model.jnt_type[i]
            joint_names.append(name)

            if j_type == mujoco.mjtJoint.mjJNT_FREE:
                joint_name2id_qpos[name] = jnp.arange(j_qpos, j_qpos + 7)
                joint_name2id_qvel[name] = jnp.arange(j_qvel, j_qvel + 6)
                j_qpos += 7
                j_qvel += 6
            elif j_type == mujoco.mjtJoint.mjJNT_SLIDE or j_type == mujoco.mjtJoint.mjJNT_HINGE:
                joint_name2id_qpos[name] = jnp.array([j_qpos])
                joint_name2id_qvel[name] = jnp.array([j_qvel])
                j_qpos += 1
                j_qvel += 1

            joint_ids.append(i)

        # get the body names from current model
        body_names = set()
        body_name2id = dict()
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            body_names.add(name)
            body_name2id[name] = i

        # get the site names from current model
        site_names = set()
        site_name2id = dict()
        for i in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            site_names.add(name)
            site_name2id[name] = i

        joint_to_be_removed_qpos = dict()
        joint_to_be_removed_qvel = dict()
        for i, j_name in enumerate(traj_info.joint_names):
            if j_name not in joint_names:
                joint_to_be_removed_qpos[j_name] = traj_info.joint_name2ind_qpos[j_name]
                joint_to_be_removed_qvel[j_name] = traj_info.joint_name2ind_qvel[j_name]

        bodies_to_be_removed = dict()
        if traj_info.body_names is not None:
            for i, b_name in enumerate(traj_info.body_names):
                if b_name not in body_names:
                    bodies_to_be_removed[b_name] = i

        site_to_be_removed = dict()
        if traj_info.site_names is not None:
            for i, s_name in enumerate(traj_info.site_names):
                if s_name not in site_names:
                    site_to_be_removed[s_name] = i

        # create new traj_data and traj_info with removed joints, bodies and sites
        if joint_to_be_removed_qpos:
            qpos_ind = jnp.concatenate(list(joint_to_be_removed_qpos.values()))
            qvel_ind = jnp.concatenate(list(joint_to_be_removed_qvel.values()))
            traj_data = traj_data.remove_joints(qpos_ind, qvel_ind)
            traj_info = traj_info.remove_joints(list(joint_to_be_removed_qpos.keys()))
        if bodies_to_be_removed:   # 'left_wrist_roll_rubber_hand', 'right_wrist_roll_rubber_hand'
            traj_data = traj_data.remove_bodies(jnp.array(list(bodies_to_be_removed.values())))
            traj_info = traj_info.remove_bodies(list(bodies_to_be_removed.keys()))
        if site_to_be_removed:
            traj_data = traj_data.remove_sites(jnp.array(list(site_to_be_removed.values())))
            traj_info = traj_info.remove_sites(list(site_to_be_removed.keys()))

        # --- extend the trajectory data's joints, bodies and sites using the current model and data ---
        for j_name, j_id in zip(joint_names, joint_ids):
            j_type = model.jnt_type[j_id]
            if j_name not in traj_info.joint_names:   # 'waist_roll_joint', 'waist_pitch_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
                traj_info = traj_info.add_joint(j_name, j_type)
                traj_data = traj_data.add_joint()

        if traj_info.body_names is not None:
            for b_name in body_names:
                if b_name not in traj_info.body_names:   # 'left_wrist_yaw_link', 'right_wrist_roll_link', 'left_wrist_pitch_link', 'waist_yaw_link', 'right_wrist_pitch_link', 'waist_roll_link', 'debug_axis', 'left_wrist_roll_link', 'right_wrist_yaw_link'
                    b_id = body_name2id[b_name]
                    traj_info = traj_info.add_body(b_name, model.body_rootid[b_id], model.body_weldid[b_id],
                                                   model.body_mocapid[b_id], model.body_pos[b_id],
                                                   model.body_quat[b_id], model.body_ipos[b_id],
                                                   model.body_iquat[b_id])
                    traj_data = traj_data.add_body()

        if traj_info.site_names is not None:
            for s_name in site_names:
                if s_name not in traj_info.site_names:
                    s_id = site_name2id[s_name]
                    traj_info = traj_info.add_site(s_name, model.site_pos[s_id], model.site_quat[s_id],
                                                   model.site_bodyid[s_id])
                    traj_data = traj_data.add_site()

        # --- reorder the joints and bodies based on the model ---
        new_joint_order_names = []
        new_joint_order_ids_qpos = []
        new_joint_order_ids_qvel = []
        for j_name in joint_names:
            new_joint_order_names.append(traj_info.joint_names.index(j_name))
            new_joint_order_ids_qpos.append(traj_info.joint_name2ind_qpos[j_name])
            new_joint_order_ids_qvel.append(traj_info.joint_name2ind_qvel[j_name])

        if traj_info.body_names is not None:
            new_body_order = []
            for b_name in body_name2id.keys():
                new_body_order.append(traj_info.body_names.index(b_name))

        if traj_info.site_names is not None:
            new_site_order = []
            for s_name in site_name2id.keys():
                new_site_order.append(traj_info.site_names.index(s_name))

        traj_info = traj_info.reorder_joints(new_joint_order_names)

        if traj_info.body_names is not None:
            new_body_order_array = TrajectoryHandler._ensure_integer_index_array(new_body_order)
            traj_info = traj_info.reorder_bodies(new_body_order_array)
        else:
            new_body_order_array = None

        if traj_info.site_names is not None:
            new_site_order_array = TrajectoryHandler._ensure_integer_index_array(new_site_order)
            traj_info = traj_info.reorder_sites(new_site_order_array)
        else:
            new_site_order_array = None

        traj_data = traj_data.reorder_joints(jnp.concatenate(new_joint_order_ids_qpos),
                                             jnp.concatenate(new_joint_order_ids_qvel))
        if new_body_order_array is not None:
            traj_data = traj_data.reorder_bodies(new_body_order_array)
        if new_site_order_array is not None:
            traj_data = traj_data.reorder_sites(new_site_order_array)

        return traj_data, traj_info

    @staticmethod
    def _ensure_integer_index_array(new_order):
        if new_order is None:
            return None

        if isinstance(new_order, (list, tuple)):
            return jnp.asarray(new_order, dtype=jnp.int32)

        if isinstance(new_order, np.ndarray):
            return jnp.asarray(new_order.astype(np.int32))

        # jax arrays or other types exposing astype
        if hasattr(new_order, "astype"):
            return new_order.astype(jnp.int32)

        return jnp.asarray(new_order, dtype=jnp.int32)

    def init_state(self, env=None, key=None, model=None, data=None, backend=None):
        return TrajState(0, 0, 0)

    def reset_state(self, carry, backend=jnp):

        key = carry.key

        if self.random_start:
            if backend == jnp:
                key, _k1, _k2 = jax.random.split(key, 3)
                traj_idx = jax.random.randint(_k1, shape=(1,), minval=0, maxval=self.n_trajectories)
                subtraj_step_idx = jax.random.randint(_k2, shape=(1,), minval=0, maxval=self.len_trajectory(traj_idx) - 1) # -1 to avoid only one step
                idx = [traj_idx[0], subtraj_step_idx[0]]
            else:
                traj_idx = np.random.randint(0, self.n_trajectories)
                subtraj_step_idx = np.random.randint(0, self.len_trajectory(traj_idx))
                idx = [traj_idx, subtraj_step_idx]
        elif self.use_fixed_start:
            idx = self.fixed_start_conf
        else:
            idx = [0, 0]

        new_traj_no, new_subtraj_step_no = idx
        new_subtraj_step_no_init = new_subtraj_step_no

        return carry.replace(key=key, traj_state=TrajState(new_traj_no, new_subtraj_step_no, new_subtraj_step_no_init))

    def reset_state_with_trajectory(self, trajectory_data: TrajectoryData, carry, backend=jnp):

        key = carry.key
        n_trajectories = len(trajectory_data.split_points) - 1

        if self.random_start:
            if backend == jnp:
                key, _k1, _k2 = jax.random.split(key, 3)
                traj_idx = jax.random.randint(_k1, shape=(1,), minval=0, maxval=n_trajectories)
                subtraj_step_idx = jax.random.randint(_k2, shape=(1,), minval=0, maxval=LenTrajectory(trajectory_data, traj_idx) - 1) # -1 to avoid only one step
                idx = [traj_idx[0], subtraj_step_idx[0]]
            else:
                traj_idx = np.random.randint(0, n_trajectories)
                subtraj_step_idx = np.random.randint(0, LenTrajectory(trajectory_data, traj_idx))
                idx = [traj_idx, subtraj_step_idx]
        elif self.use_fixed_start:
            idx = self.fixed_start_conf
        else:
            idx = [0, 0]

        new_traj_no, new_subtraj_step_no = idx
        new_subtraj_step_no_init = new_subtraj_step_no

        return carry.replace(key=key, traj_state=TrajState(new_traj_no, new_subtraj_step_no, new_subtraj_step_no_init))

    def reset_state_simple(self, rng):
        if self.random_start:
            idx = jax.random.randint(rng, shape=(), minval=self.start_range[0], maxval=self.start_range[1])
        elif self.use_fixed_start:
            idx = self.fixed_start_conf[1]
        else:
            idx = 0

        return idx

    def update_state(self, carry, backend=jnp):

        traj_state = carry.traj_state
        traj_no = traj_state.traj_no
        subtraj_step_no = traj_state.subtraj_step_no
        subtraj_step_no_init = traj_state.subtraj_step_no_init

        length_trajectory = self.len_trajectory(traj_no)

        subtraj_step_no += 1

        # set to zero once exceeded
        next_subtraj_step_no = backend.mod(subtraj_step_no, length_trajectory)

        if backend == jnp:
            # check whether to go to the next trajectory
            next_traj_no = jax.lax.cond(next_subtraj_step_no == 0, lambda t, nt: jnp.mod(t+1, nt),
                                        lambda t, nt: t, traj_no, self.n_trajectories)
            next_subtraj_step_no_init = jax.lax.cond(next_traj_no != traj_no, lambda: 0,
                                                     lambda: subtraj_step_no_init)
        else:
            next_traj_no = traj_no if next_subtraj_step_no != 0 else (traj_no + 1) % self.n_trajectories
            next_subtraj_step_no_init = 0 if traj_no != next_traj_no else subtraj_step_no_init

        traj_state = traj_state.replace(traj_no=next_traj_no, subtraj_step_no=next_subtraj_step_no,
                                        subtraj_step_no_init=next_subtraj_step_no_init)

        return carry.replace(traj_state=traj_state)

    def update_state_with_trajectory(self, trajectory_data: TrajectoryData, carry, backend=jnp):

        n_trajectories = len(trajectory_data.split_points) - 1

        traj_state = carry.traj_state
        traj_no = traj_state.traj_no
        subtraj_step_no = traj_state.subtraj_step_no
        subtraj_step_no_init = traj_state.subtraj_step_no_init

        length_trajectory = LenTrajectory(trajectory_data, traj_no)

        subtraj_step_no += 1

        # set to zero once exceeded
        next_subtraj_step_no = backend.mod(subtraj_step_no, length_trajectory)

        if backend == jnp:
            # check whether to go to the next trajectory
            next_traj_no = jax.lax.cond(next_subtraj_step_no == 0, lambda t, nt: jnp.mod(t+1, nt),
                                        lambda t, nt: t, traj_no, n_trajectories)
            next_subtraj_step_no_init = jax.lax.cond(next_traj_no != traj_no, lambda: 0,
                                                     lambda: subtraj_step_no_init)
        else:
            next_traj_no = traj_no if next_subtraj_step_no != 0 else (traj_no + 1) % n_trajectories
            next_subtraj_step_no_init = 0 if traj_no != next_traj_no else subtraj_step_no_init

        traj_state = traj_state.replace(traj_no=next_traj_no, subtraj_step_no=next_subtraj_step_no,
                                        subtraj_step_no_init=next_subtraj_step_no_init)

        return carry.replace(traj_state=traj_state)

    def get_current_traj_data(self, carry, backend=jnp):
        traj_no = carry.traj_state.traj_no
        subtraj_step_no = carry.traj_state.subtraj_step_no
        return self.traj.data.get(traj_no, subtraj_step_no, backend)

    def get_current_traj_data_with_trajectory(self, trajectory_data: TrajectoryData, carry, backend=jnp):
        traj_no = carry.traj_state.traj_no
        subtraj_step_no = carry.traj_state.subtraj_step_no
        return trajectory_data.get(traj_no, subtraj_step_no, backend)

    def get_current_traj_data_fast(self, carry, backend=jnp):
        traj_no = carry.traj_state.traj_no
        subtraj_step_no = carry.traj_state.subtraj_step_no
        return self.traj.data.get_fast(traj_no, subtraj_step_no, backend)

    def get_current_traj_data_simple(self, traj_step, backend=jnp):
        return self.traj.data.get(0, traj_step, backend)

    def get_init_traj_data(self, carry, backend=jnp):
        traj_no = carry.traj_state.traj_no
        subtraj_step_no_init = carry.traj_state.subtraj_step_no_init
        return self.traj.data.get(traj_no, subtraj_step_no_init, backend)

    def to_numpy(self):
        if not self._is_numpy:
            traj_model = self.traj.info.model.to_numpy()
            traj_info = replace(self.traj.info, model=traj_model)
            self.traj = replace(self.traj, data=self.traj.data.to_numpy(), info=traj_info)
            self._is_numpy = True

    def to_jax(self):
        if self._is_numpy:
            traj_model = self.traj.info.model.to_numpy()
            traj_info = replace(self.traj.info, model=traj_model)
            self.traj = replace(self.traj, data=self.traj.data.to_jax(), info=traj_info)
            self._is_numpy = False

    @property
    def is_numpy(self):
        return self._is_numpy

    # def check_if_trajectory_is_in_range(self, low, high, keys, j_idx, warn, clip_trajectory_to_joint_ranges):
    #
    #     if warn or clip_trajectory_to_joint_ranges:
    #
    #         # get q_pos indices
    #         j_idx = j_idx[2:]   # exclude x and y
    #         highs = dict(zip(keys[2:], high))
    #         lows = dict(zip(keys[2:], low))
    #
    #         # check if they are in range
    #         for i, item in enumerate(self._trajectory_files.items()):
    #             k, d = item
    #             if i in j_idx and k in keys:
    #                 if warn:
    #                     clip_message = "Clipping the trajectory into range!" if clip_trajectory_to_joint_ranges else ""
    #                     if np.max(d) > highs[k]:
    #                         warnings.warn("Trajectory violates joint range in %s. Maximum in trajectory is %f "
    #                                       "and maximum range is %f. %s"
    #                                       % (k, np.max(d), highs[k], clip_message), RuntimeWarning)
    #                     elif np.min(d) < lows[k]:
    #                         warnings.warn("Trajectory violates joint range in %s. Minimum in trajectory is %f "
    #                                       "and minimum range is %f. %s"
    #                                       % (k, np.min(d), lows[k], clip_message), RuntimeWarning)
    #
    #                 # clip trajectory to min & max
    #                 if clip_trajectory_to_joint_ranges:
    #                     self._trajectory_files[k] = np.clip(self._trajectory_files[k], lows[k], highs[k])
