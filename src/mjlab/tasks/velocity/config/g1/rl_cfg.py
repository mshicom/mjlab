from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
  RslRlPpoAmpCfg
)


@dataclass
class UnitreeG1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
  policy: RslRlPpoActorCriticCfg = field(
    default_factory=lambda: RslRlPpoActorCriticCfg(
      init_noise_std=1.0,
      actor_obs_normalization=False,
      critic_obs_normalization=False,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    )
  )
  algorithm: RslRlPpoAlgorithmCfg = field(
    default_factory=lambda: RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      amp_cfg=RslRlPpoAmpCfg(
        enabled=False,
        obs_key="amp_state",
        hidden_dims=[256, 256],
        activation="elu",
        init_output_scale=0.0,
        learning_rate=1.0e-4,
        weight_decay=0.0,
        logit_reg=0.0,
        grad_penalty=0.0,
        disc_weight_decay=0.0,
        reward_scale=1.0,
        reward_coef=1.0,
        eval_batch_size=0,
        norm_until=None,
        # demo_provider=resolved automatically from env.sample_amp_demos if present
        demo_batch_ratio=1.0
      )
    )
  )
  experiment_name: str = "g1_velocity"
  save_interval: int = 50
  num_steps_per_env: int = 24
  max_iterations: int = 30_000
