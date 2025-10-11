from __future__ import annotations

from mjlab.amp.config import AmpCfg, AmpDatasetCfg, AmpFeatureSetCfg, FeatureTermCfg, JointSelectionCfg

AMP_CFG_EXAMPLE = AmpCfg(
    reward_coef=0.5,
    discr_hidden_dims=(1024, 512, 256),
    task_reward_lerp=0.0,
    feature_set=AmpFeatureSetCfg(
        terms=[
            # RMS of joint speed/accel/jerk on informative joints
            FeatureTermCfg(
                name="joint_speed_rms",
                source="qvel",
                channels=["scalar"],  # interpret as scalar per selected DOF
                window_size=30,
                pre_diff="none",
                aggregators=["rms"],
                select=JointSelectionCfg(joints=["left_knee", "right_knee", "left_ankle", "right_ankle", "left_wrist", "right_wrist", "left_elbow", "right_elbow"]),
            ),
            FeatureTermCfg(
                name="joint_accel_rms",
                source="qvel",
                channels=["scalar"],
                window_size=30,
                pre_diff="acceleration",
                aggregators=["rms"],
                select=JointSelectionCfg(joints=["left_knee", "right_knee", "left_ankle", "right_ankle", "left_wrist", "right_wrist", "left_elbow", "right_elbow"]),
            ),
            FeatureTermCfg(
                name="joint_jerk_rms",
                source="qvel",
                channels=["scalar"],
                window_size=30,
                pre_diff="jerk",
                aggregators=["rms"],
                select=JointSelectionCfg(joints=["left_knee", "right_knee", "left_ankle", "right_ankle", "left_wrist", "right_wrist", "left_elbow", "right_elbow"]),
            ),
            # Global energy proxy: mean speed^2 of selected bodies (or full robot)
            FeatureTermCfg(
                name="energy_proxy",
                source="cvel",  # body 6D velocities; use linear components
                channels=["lin.x", "lin.y", "lin.z"],  # assume selector will pick linear parts
                window_size=50,
                aggregators=["mean"],  # mean of squared speed can be implemented in term or via pre_diff mode
                select=JointSelectionCfg(bodies=["pelvis", "torso"]),
            ),
            # Foot-ground duty (L/R) and transition rate
            FeatureTermCfg(
                name="duty_left",
                source="site_xpos",  # or derive from contacts; see preprocessing
                channels=["z<eps->contact"],  # custom channel routine in selector
                window_size=50,
                aggregators=["mean"],  # mean(contact) ~= duty factor
                select=JointSelectionCfg(sites=["left_foot"]),
            ),
            FeatureTermCfg(
                name="duty_right",
                source="site_xpos",
                channels=["z<eps->contact"],
                window_size=50,
                aggregators=["mean"],
                select=JointSelectionCfg(sites=["right_foot"]),
            ),
            FeatureTermCfg(
                name="contact_transition_rate",
                source="site_xpos",
                channels=["z<eps->contact"],
                window_size=50,
                aggregators=["rms"],  # rms of diff(contact) ~ transition rate
                pre_diff="velocity",
                select=JointSelectionCfg(sites=["left_foot", "right_foot"]),
            ),
            # Left-right symmetry: ankle & wrist speed correlations
            FeatureTermCfg(
                name="lr_symmetry_ankle_wrist",
                source="cvel",
                channels=["lin.speed"],  # add "speed" support in selector
                window_size=50,
                aggregators=["mean"],  # implement correlation in term: replace "mean" with dedicated "corr" agg in your code
                select=JointSelectionCfg(bodies=["left_ankle", "right_ankle", "left_wrist", "right_wrist"]),
            ),
            # Upper-lower limb coordination
            FeatureTermCfg(
                name="upper_lower_coord",
                source="cvel",
                channels=["lin.speed"],
                window_size=50,
                aggregators=["mean"],  # implement cross-correlation or phase-lock value as a new agg if desired
                select=JointSelectionCfg(bodies=["left_wrist", "right_wrist", "left_ankle", "right_ankle"]),
            ),
            # Yaw-rate RMS (from shoulders/hips or pelvis angular z)
            FeatureTermCfg(
                name="yaw_rate_rms",
                source="cvel",
                channels=["ang.z"],
                window_size=30,
                aggregators=["rms"],
                select=JointSelectionCfg(bodies=["pelvis"]),
            ),
            # Dominant frequency + spectral entropy of COM vertical velocity
            FeatureTermCfg(
                name="com_vert_freq",
                source="subtree_com",
                channels=["lin.z_vel"],  # derive from diff(x_com.z)/dt in selector or preprocessing
                window_size=64,
                aggregators=["dominant_freq", "spectral_entropy"],
                select=JointSelectionCfg(bodies=["pelvis"]),  # or a virtual COM index
            ),
        ],
    ),
    dataset=AmpDatasetCfg(
        files=["datasets/mocap_humanoid/subject01_walk.npz", "datasets/mocap_humanoid/subject02_run.npz"],
        time_between_frames=0.05,
        preload_transitions=200_000,
        symmetry=dict(enabled=True),
    ),
)