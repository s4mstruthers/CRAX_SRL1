"""CPO training (Achiam et al., 2017).

Thin wrapper around the base PPO trainer, like the other constrained algorithms. Two hooks
are used:
  * policy_update_fn: the CPO trust-region policy step on the full batch (cpo/losses.py),
    run once per policy update, before the minibatch SGD;
  * loss_fn: the minibatch SGD then only fits the reward and cost value functions.
Rollouts, normalisation, the per-update safety metrics, desynced robots and the final
evaluation are therefore exactly the same as for the other algorithms.
"""

import functools
from typing import Any, Callable, Dict, Optional, Tuple

import jax.numpy as jnp

from crax import base
from crax import envs
from training import types
from training.agents.cpo import losses as cpo_losses
from training.agents.ppo import networks as ppo_networks
from training.agents.ppo import train as ppo_train

Metrics = types.Metrics
TrainingState = ppo_train.TrainingState


def train(
        environment: envs.Env,
        num_timesteps: int,
        episode_length: int,
        max_devices_per_host: Optional[int] = None,
        wrap_env: bool = True,
        num_envs: int = 1,
        action_repeat: int = 1,
        wrap_env_fn: Optional[Callable[[Any], Any]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        learning_rate: float = 1e-4,
        discounting: float = 0.99,
        unroll_length: int = 10,
        batch_size: int = 32,
        num_minibatches: int = 16,
        num_updates_per_batch: int = 2,
        num_resets_per_eval: int = 0,
        normalize_observations: bool = False,
        reward_scaling: float = 1.0,
        gae_lambda: float = 0.95,
        max_grad_norm: Optional[float] = None,
        network_factory: types.NetworkFactory[ppo_networks.PPONetworks] = None,
        seed: int = 0,
        # constraint
        safety_bound: float = 0.0,
        # CPO-specific
        cpo_target_kl: float = 0.01,
        cpo_cg_iters: int = 10,
        cpo_cg_damping: float = 0.1,
        cpo_backtrack_coeff: float = 0.8,
        cpo_backtrack_iters: int = 10,
        cpo_fvp_subsample: int = 1,
        # eval
        num_evals: int = 0,
        eval_env: Optional[envs.Env] = None,
        num_eval_envs: int = 128,
        deterministic_eval: bool = False,
        buffer_size: int = 1000,
        log_training_metrics: bool = True,
        training_metrics_steps: Optional[int] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
        policy_params_fn: Callable[..., None] = lambda *args: None,
        save_checkpoint_path: Optional[str] = None,
        restore_checkpoint_path: Optional[str] = None,
        restore_params: Optional[Any] = None,
        restore_value_fn: bool = True,
        pretrained_params: Optional[Any] = None,
):
    """CPO training.

    Args (beyond the shared PPO ones):
      safety_bound: per-episode cost budget d (converted to d / episode_length per step).
      cpo_target_kl: trust-region size delta, the maximum mean KL(pi_old || pi_new) per update.
      cpo_cg_iters: conjugate-gradient iterations for H^-1 g and H^-1 b.
      cpo_cg_damping: damping added to the Fisher-vector product (H + damping * I).
      cpo_backtrack_coeff, cpo_backtrack_iters: line search step sizes coeff^0 ... coeff^(iters-1).
      cpo_fvp_subsample: use every k-th unroll segment for the Fisher-vector products (1 = all).

    The entropy bonus and PPO clipping are not used by CPO. learning_rate,
    num_minibatches and num_updates_per_batch apply to the value-function fitting only.
    """
    per_step_safety_bound = safety_bound / episode_length if episode_length else safety_bound

    if network_factory is None:
        def network_factory(obs_size, action_size, **kwargs):
            return ppo_networks.make_ppo_networks(
                obs_size, action_size, cost_value_hidden_layer_sizes=(256,) * 5, **kwargs)

    policy_update_fn = functools.partial(
        cpo_losses.cpo_policy_update,
        per_step_safety_bound=per_step_safety_bound,
        target_kl=cpo_target_kl,
        cg_iters=cpo_cg_iters,
        cg_damping=cpo_cg_damping,
        backtrack_coeff=cpo_backtrack_coeff,
        backtrack_iters=cpo_backtrack_iters,
        fvp_subsample=cpo_fvp_subsample,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
    )

    effective_restore_params = pretrained_params if pretrained_params is not None else restore_params

    return ppo_train.train(
        environment=environment, num_timesteps=num_timesteps, max_devices_per_host=max_devices_per_host,
        wrap_env=wrap_env, num_envs=num_envs, episode_length=episode_length, action_repeat=action_repeat,
        wrap_env_fn=wrap_env_fn, randomization_fn=randomization_fn, learning_rate=learning_rate,
        entropy_cost=0.0, discounting=discounting, unroll_length=unroll_length,
        batch_size=batch_size, num_minibatches=num_minibatches, num_updates_per_batch=num_updates_per_batch,
        num_resets_per_eval=num_resets_per_eval, normalize_observations=normalize_observations,
        reward_scaling=reward_scaling, gae_lambda=gae_lambda, max_grad_norm=max_grad_norm,
        network_factory=network_factory, seed=seed, num_evals=num_evals, eval_env=eval_env,
        num_eval_envs=num_eval_envs, deterministic_eval=deterministic_eval, buffer_size=buffer_size,
        log_training_metrics=log_training_metrics, training_metrics_steps=training_metrics_steps,
        progress_fn=progress_fn, policy_params_fn=policy_params_fn,
        save_checkpoint_path=save_checkpoint_path, restore_checkpoint_path=restore_checkpoint_path,
        restore_params=effective_restore_params, restore_value_fn=restore_value_fn,
        loss_fn=cpo_losses.compute_cpo_value_loss,
        extra_fields=('truncation', 'episode_metrics', 'episode_done', 'cost'),
        policy_update_fn=policy_update_fn,
    )
