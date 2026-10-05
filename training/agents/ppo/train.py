# Copyright 2024 The Brax Authors.
# Modifications Copyright 2026 CRAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is derived from Brax 0.12.3 (training/agents/ppo/train.py) and has been modified
# by the CRAX Authors: adds the constrained-RL extension hooks used by every safe algorithm in CRAX:
# pluggable `loss_fn`, `post_step_fn` and `init_aux_state_fn`, an `aux_state` field on
# `TrainingState` for Lagrange multipliers and PID state, cost-metric plumbing, and
# shared vision-encoder support.

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import os
import time
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging

from crax import base
from crax import envs
from training import acting
from training import gradients
from training import logger as metric_logger
from training import pmap
from training import types
from training.acme import running_statistics
from training.acme import specs
from training.agents.ppo import checkpoint
from training.agents.ppo import losses as ppo_losses
from training.agents.ppo import networks as ppo_networks
from training.types import PRNGKey
from training.types import Params

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'

# Type alias for the post-step hook function
PostStepFn = Callable[['TrainingState', Metrics], Tuple['TrainingState', Metrics]]


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: ppo_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: types.UInt64
    aux_state: Optional[Any] = None  # For Lagrange multipliers, PID state, etc.


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    # brax user code is sometimes ambiguous about weak_type.  in order to
    # avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def _maybe_wrap_env(
        env: envs.Env,
        wrap_env: bool,
        num_envs: int,
        episode_length: int,
        action_repeat: int,
        device_count: int,
        key_env: PRNGKey,
        wrap_env_fn: Optional[Callable[[Any], Any]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        vision_kwargs: Optional[Dict[str, Any]] = None,
):
    """Wraps the environment for training/eval if wrap_env is True.

    vision_kwargs, if given, applies GpuPixelObservationWrapper (MJWarp) LAST
    — after episode/vmap/autoreset — since MJWarp's render context needs a
    static batch size and must see the already-fully-batched env. `num_envs`
    (this function's batch size for whichever of env/eval_env is being
    wrapped) is threaded in automatically; do not pass it in vision_kwargs.
    """
    if not wrap_env:
        return env
    if episode_length is None:
        raise ValueError('episode_length must be specified')
    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // device_count
        # all devices gets the same randomization rng
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(
            randomization_fn, rng=randomization_rng
        )
    if wrap_env_fn is not None:
        wrap_for_training = wrap_env_fn
    else:
        wrap_for_training = envs.training.wrap
    env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )  # pytype: disable=wrong-keyword-args
    if vision_kwargs is not None:
        from crax.envs.wrappers.pixel_observation_gpu import GpuPixelObservationWrapper
        env = GpuPixelObservationWrapper(env, num_envs=num_envs, **vision_kwargs)
    return env


def _random_translate_pixels(
        obs: Mapping[str, jax.Array], key: PRNGKey
) -> Mapping[str, jax.Array]:
    """Apply random translations to B x T x ... pixel observations.

    The same shift is applied across the unroll_length (T) dimension.

    Args:
      obs: a dictionary of observations
      key: a PRNGKey

    Returns:
      A dictionary of observations with translated pixels
    """

    @jax.vmap
    def rt_all_views(
            ub_obs: Mapping[str, jax.Array], key: PRNGKey
    ) -> Mapping[str, jax.Array]:
        # Expects dictionary of unbatched observations.
        def rt_view(
                img: jax.Array, padding: int, key: PRNGKey
        ) -> jax.Array:  # TxHxWxC
            # Randomly translates a set of pixel inputs.
            # Adapted from
            # https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/drq/augmentations.py
            crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
            zero = jnp.zeros((1,), dtype=jnp.int32)
            crop_from = jnp.concatenate([zero, crop_from, zero])
            padded_img = jnp.pad(
                img,
                ((0, 0), (padding, padding), (padding, padding), (0, 0)),
                mode='edge',
            )
            return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

        out = {}
        for k_view, v_view in ub_obs.items():
            if k_view.startswith('pixels/'):
                key, key_shift = jax.random.split(key)
                out[k_view] = rt_view(v_view, 4, key_shift)
        return {**ub_obs, **out}

    bdim = next(iter(obs.items()), None)[1].shape[0]
    keys = jax.random.split(key, bdim)
    obs = rt_all_views(obs, keys)
    return obs


def _remove_pixels(
        obs: Union[jnp.ndarray, Mapping[str, jax.Array]],
) -> Union[jnp.ndarray, Mapping[str, jax.Array]]:
    """Removes pixel observations from the observation dict."""
    if not isinstance(obs, Mapping):
        return obs
    return {k: v for k, v in obs.items() if not k.startswith('pixels/')}

# Thresholds on the normalised cost deviation D_norm = (c - d) / d at which the empirical
# CDF is logged (Spoor et al., 2026, Eq. 7). -1 means zero cost, 0 means exactly at the budget.
CDF_THRESHOLDS = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def _threshold_name(k: float) -> str:
    """Key-safe name for a threshold (wandb treats '.' in keys as nesting): -0.5 -> 'm0p5'."""
    return f'{k:g}'.replace('-', 'm').replace('.', 'p')


def _summarise_update_cost(cost: jnp.ndarray, episode_length: int) -> Dict[str, jnp.ndarray]:
    """Summarise the safety of the policy that collected one batch (runs on the GPU).

    Args:
      cost: per-step cost with shape (num_rollouts, unroll_length, num_envs),
        taken before the batch is reshaped for PPO. Every step was produced by
        the same policy pi_k, so these numbers describe pi_k alone.
      episode_length: steps per episode, used to scale slice costs to an
        episode-equivalent that can be compared with the budget d.

    Returns:
      Dict of scalar metrics, prefixed 'safety/'.
    """
    steps_per_env = cost.shape[0] * cost.shape[1]            # 16 * 8 = 128 by default
    env_cost = jnp.sum(cost, axis=(0, 1))                    # each robot's total cost this update
    # Approximation: scale a 128-step slice up to a full episode so it is on the same scale as d.
    env_cost_ep = env_cost * (episode_length / steps_per_env)
    quantiles = jnp.quantile(env_cost_ep, jnp.array([0.5, 0.9, 0.99]))
    n_worst = max(1, env_cost_ep.shape[0] // 20)             # worst 5% of robots
    worst = jnp.sort(env_cost_ep)[-n_worst:]
    return {
        'safety/frac_unsafe_steps': jnp.mean(cost > 0),      # how often any cost occurs
        'safety/frac_envs_with_cost': jnp.mean(env_cost > 0),
        'safety/env_cost_mean': jnp.mean(env_cost_ep),
        'safety/env_cost_p50': quantiles[0],
        'safety/env_cost_p90': quantiles[1],
        'safety/env_cost_p99': quantiles[2],
        'safety/env_cost_max': jnp.max(env_cost_ep),
        'safety/env_cost_cvar95': jnp.mean(worst),           # average of the worst 5%
    }

def _summarise_completed_episodes(
        episode_cost: jnp.ndarray,
        episode_length_so_far: jnp.ndarray,
        episode_done: jnp.ndarray,
        truncation: jnp.ndarray,
        episode_length: int,
        budget: float,
) -> Dict[str, jnp.ndarray]:
    """Episode-level safety: the distribution of cost over episodes completed in this batch.

    Unlike _summarise_update_cost, nothing is extrapolated: each value is the true
    total cost of one finished episode. This gives an honest tail (p90/p99, share
    of episodes over budget). The trade-off is that a finished episode spans the
    last ~episode_length steps, so it mixes the few most recent policies.

    Args:
      episode_cost: running episode cost per step, shape (num_rollouts, unroll_length, num_envs).
        At steps where episode_done == 1 it holds the finished episode's total cost.
      episode_length_so_far: running episode length, same shape.
      episode_done: 1.0 at the step an episode finished, else 0.0.
      truncation: 1.0 if the episode ended because of the time limit (not the env).
      episode_length: the time limit.
      budget: the per-episode cost budget d, used for the "share over budget" metrics.

    Returns:
      Dict of scalar metrics, prefixed 'safety_ep/'. When no (valid) episode finished in
      this batch all values are NaN, which the metrics logger skips.
    """
    done = episode_done > 0
    # Exclude episodes that were cut short by the desync start offset: they hit the
    # time limit (truncation == 1) with fewer than ~episode_length steps. Episodes the
    # environment itself terminated early (truncation == 0) are kept. The running length
    # reads episode_length - 1 for a full episode (the wrapper zeroes the first step), so
    # a 1% tolerance is used.
    full_length = episode_length_so_far >= 0.99 * episode_length
    valid = done & ((truncation < 0.5) | full_length)
    costs = jnp.where(valid, episode_cost, jnp.nan).reshape(-1)
    count = jnp.sum(valid)

    quantiles = jnp.nanquantile(costs, jnp.array([0.5, 0.9, 0.95, 0.99]))
    worst5 = jnp.where(costs >= quantiles[2], costs, jnp.nan)   # worst 5% of episodes
    over = jnp.where(valid.reshape(-1), (costs > budget).astype(jnp.float32), jnp.nan)
    over2 = jnp.where(valid.reshape(-1), (costs > 2 * budget).astype(jnp.float32), jnp.nan)
    nan = jnp.array(jnp.nan)
    has = count > 0
    pick = lambda v: jnp.where(has, v, nan)

    # Metrics of Spoor et al. (2026), so our numbers can be read against theirs.
    # D_norm: signed distance of the mean cost from the budget, in units of d (> 0 = unsafe).
    # D_norm+: mean overshoot of the violating episodes only, in units of d (0 if none violate).
    flat_valid = valid.reshape(-1)
    excess = jnp.where(flat_valid, jnp.maximum(costs - budget, 0.0), jnp.nan)
    n_over = jnp.nansum(over)
    d_norm_plus = jnp.where(n_over > 0, jnp.nansum(excess) / jnp.maximum(n_over, 1.0) / budget, 0.0)
    zero_cost = jnp.where(flat_valid, (costs <= 0.0).astype(jnp.float32), jnp.nan)
    # Empirical CDF of D_norm at fixed thresholds. Pooling these over updates, weighted by
    # safety_ep/count, gives the training-time CDF of Spoor et al. (Eq. 7).
    d_norm_ep = (costs - budget) / budget
    cdf = {
        f'safety_ep/cdf_dnorm_le_{_threshold_name(k)}': pick(jnp.nanmean(
            jnp.where(flat_valid, (d_norm_ep <= k).astype(jnp.float32), jnp.nan)))
        for k in CDF_THRESHOLDS
    }
    return {
        **cdf,
        'safety_ep/d_norm': pick((jnp.nanmean(costs) - budget) / budget),
        'safety_ep/d_norm_plus': pick(d_norm_plus),
        'safety_ep/frac_zero_cost': pick(jnp.nanmean(zero_cost)),   # episodes with no cost at all
        'safety_ep/count': count.astype(jnp.float32),
        'safety_ep/cost_mean': pick(jnp.nanmean(costs)),
        'safety_ep/cost_p50': pick(quantiles[0]),
        'safety_ep/cost_p90': pick(quantiles[1]),
        'safety_ep/cost_p99': pick(quantiles[3]),
        'safety_ep/cost_max': pick(jnp.nanmax(costs)),
        'safety_ep/cost_cvar95': pick(jnp.nanmean(worst5)),
        'safety_ep/frac_over_budget': pick(jnp.nanmean(over)),      # share of episodes with cost > d
        'safety_ep/frac_over_2x_budget': pick(jnp.nanmean(over2)),  # share with cost > 2d
    }


def _violation_rate_upper_bound(n_over: int, n: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on the true violation rate.

    With 0 violations in n episodes this is about 3/n ("rule of three"), so 1000 episodes
    are needed to claim V < 0.3% with 95% confidence.
    """
    if n == 0:
        return float('nan')
    if n_over >= n:
        return 1.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(confidence, n_over + 1, n - n_over))
    except ImportError:  # exact for n_over == 0, which is the case that matters most
        return float(1.0 - (1.0 - confidence) ** (1.0 / n)) if n_over == 0 else float('nan')


def _final_safety_metrics(costs, rewards, budget: float, prefix: str) -> Dict[str, Any]:
    """Safety of one frozen policy over N complete, fresh evaluation episodes.

    Unlike the training metrics, every episode here is played by a single policy, so these
    numbers estimate J_c(pi_final) directly. Metric names follow Spoor et al. (2026).

    Args:
      costs, rewards: total cost / reward of each evaluation episode, shape (N,).
      budget: the per-episode cost budget d.
      prefix: key prefix, e.g. 'final_eval/greedy'.
    """
    costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    n = costs.size
    over = costs > budget
    n_over = int(over.sum())
    q50, q90, q95, q99 = np.quantile(costs, [0.5, 0.9, 0.95, 0.99])
    d_norm = (costs - budget) / budget
    m = {
        'num_episodes': n,
        'reward_mean': rewards.mean(),
        'cost_mean': costs.mean(),
        'violation_rate': over.mean(),                                    # V
        'violation_rate_upper95': _violation_rate_upper_bound(n_over, n),
        'd_norm': (costs.mean() - budget) / budget,                       # D_norm
        'd_norm_plus': float((costs[over] - budget).mean() / budget) if n_over else 0.0,  # D_norm+
        'frac_over_2x_budget': (costs > 2 * budget).mean(),
        'frac_zero_cost': (costs <= 0.0).mean(),
        'cost_p50': q50, 'cost_p90': q90, 'cost_p99': q99,
        'cost_cvar95': costs[costs >= q95].mean(),                        # mean of the worst 5%
        'cost_max': costs.max(),
        **{f'cdf_dnorm_le_{_threshold_name(k)}': (d_norm <= k).mean() for k in CDF_THRESHOLDS},
    }
    out = {f'{prefix}/{k}': float(v) for k, v in m.items()}
    # Raw per-episode values, so the full CDF can be drawn later (train_env.py stores these
    # in the wandb run summary as lists instead of averaging them).
    out[f'{prefix}/episode_costs'] = costs.astype(np.float32)
    out[f'{prefix}/episode_rewards'] = rewards.astype(np.float32)
    return out


def train(
        environment: envs.Env,
        num_timesteps: int,
        episode_length: int,
        max_devices_per_host: Optional[int] = None,
        # high-level control flow
        wrap_env: bool = True,
        augment_pixels: bool = False,
        vision_kwargs: Optional[Dict[str, Any]] = None,
        # environment wrapper
        num_envs: int = 1,
        action_repeat: int = 1,
        wrap_env_fn: Optional[Callable[[Any], Any]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        # ppo params
        learning_rate: float = 1e-4,
        entropy_cost: float = 1e-4,
        discounting: float = 0.9,
        unroll_length: int = 10,
        batch_size: int = 32,
        num_minibatches: int = 16,
        num_updates_per_batch: int = 2,
        num_resets_per_eval: int = 0,
        normalize_observations: bool = False,
        reward_scaling: float = 1.0,
        clipping_epsilon: float = 0.3,
        gae_lambda: float = 0.95,
        max_grad_norm: Optional[float] = None,
        normalize_advantage: bool = True,
        network_factory: types.NetworkFactory[
            ppo_networks.PPONetworks
        ] = ppo_networks.make_ppo_networks,
        seed: int = 0,
        # eval
        num_evals: int = 0,
        eval_env: Optional[envs.Env] = None,
        num_eval_envs: int = 128,
        deterministic_eval: bool = False,
        # training metrics
        buffer_size: int = 1000,
        log_training_metrics: bool = True,
        training_metrics_steps: Optional[int] = None,
        # callbacks
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
        policy_params_fn: Callable[..., None] = lambda *args: None,
        # checkpointing
        save_checkpoint_path: Optional[str] = None,
        restore_checkpoint_path: Optional[str] = None,
        restore_params: Optional[Any] = None,
        restore_value_fn: bool = True,
        # customization hooks for constrained RL variants
        loss_fn: Optional[Callable] = None,
        post_step_fn: Optional[PostStepFn] = None,
        extra_fields: Tuple[str, ...] = ('truncation', 'episode_metrics', 'episode_done'),
        init_aux_state_fn: Optional[Callable[[], Any]] = None,
        # Optional full-batch policy update run before the minibatch SGD (used by CPO):
        # fn(params, normalizer_params, data, key, *, ppo_network, pmap_axis_name)
        #   -> (new_policy_params, metrics). The SGD then starts from the updated policy.
        policy_update_fn: Optional[Callable] = None,
):
    """PPO training.

    Args:
      environment: the environment to train
      num_timesteps: the total number of environment steps to use during training
      max_devices_per_host: maximum number of chips to use per host process
      wrap_env: If True, wrap the environment for training. Otherwise use the
        environment as is.
      augment_pixels: whether to add image augmentation to pixel inputs
      vision_kwargs: if given, adds MJWarp-rendered pixel observations to
        `environment`/`eval_env` (see GpuPixelObservationWrapper). Applied
        after env wrapping/vmapping; `num_envs`/`num_eval_envs` are threaded
        in automatically, do not include them here.
      num_envs: the number of parallel environments to use for rollouts
        NOTE: `num_envs` must be divisible by the total number of chips since each
          chip gets `num_envs // total_number_of_chips` environments to roll out
        NOTE: `batch_size * num_minibatches` must be divisible by `num_envs` since
          data generated by `num_envs` parallel envs gets used for gradient
          updates over `num_minibatches` of data, where each minibatch has a
          leading dimension of `batch_size`
      episode_length: the length of an environment episode
      action_repeat: the number of timesteps to repeat an action
      wrap_env_fn: a custom function that wraps the environment for training. If
        not specified, the environment is wrapped with the default training
        wrapper.
      randomization_fn: a user-defined callback function that generates randomized
        environments
      learning_rate: learning rate for ppo loss
      entropy_cost: entropy reward for ppo loss, higher values increase entropy of
        the policy
      discounting: discounting rate
      unroll_length: the number of timesteps to unroll in each environment. The
        PPO loss is computed over `unroll_length` timesteps
      batch_size: the batch size for each minibatch SGD step
      num_minibatches: the number of times to run the SGD step, each with a
        different minibatch with leading dimension of `batch_size`
      num_updates_per_batch: the number of times to run the gradient update over
        all minibatches before doing a new environment rollout
      num_resets_per_eval: the number of environment resets to run between each
        eval. The environment resets occur on the host
      normalize_observations: whether to normalize observations
      reward_scaling: float scaling for reward
      clipping_epsilon: clipping epsilon for PPO loss
      gae_lambda: General advantage estimation lambda
      max_grad_norm: gradient clipping norm value. If None, no clipping is done
      normalize_advantage: whether to normalize advantage estimate
      network_factory: function that generates networks for policy and value
        functions
      seed: random seed
      num_evals: the number of evals to run during the entire training run.
        Increasing the number of evals increases total training time
      eval_env: an optional environment for eval only, defaults to `environment`
      num_eval_envs: the number of envs to use for evluation. Each env will run 1
        episode, and all envs run in parallel during eval.
      deterministic_eval: whether to run the eval with a deterministic policy
      log_training_metrics: whether to log training metrics and callback to
        progress_fn
      training_metrics_steps: the number of environment steps between logging
        training metrics
      progress_fn: a user-defined callback function for reporting/plotting metrics
      policy_params_fn: a user-defined callback function that can be used for
        saving custom policy checkpoints or creating policy rollouts and videos
      save_checkpoint_path: the path used to save checkpoints. If None, no
        checkpoints are saved.
      restore_checkpoint_path: the path used to restore previous model params
      restore_params: raw network parameters to restore the TrainingState from.
        These override `restore_checkpoint_path`. These paramaters can be obtained
        from the return values of ppo.train().
      restore_value_fn: whether to restore the value function from the checkpoint
        or use a random initialization
      loss_fn: Optional custom loss function. If None, uses compute_ppo_loss.
        For constrained RL variants, pass compute_ppo_lagrange_loss.
      post_step_fn: Optional function called after each training step.
        Signature: (TrainingState, Metrics) -> (TrainingState, Metrics).
        Used for Lagrange multiplier updates in constrained RL.
      extra_fields: Extra fields to collect from env state during rollout.
        Default is ('truncation', 'episode_metrics', 'episode_done').
        For constrained RL, add 'cost'.
      init_aux_state_fn: Optional function to initialize aux_state in TrainingState.
        Returns initial aux_state value. Used for Lagrange multipliers, PID state, etc.

    Returns:
      Tuple of (make_policy function, network params, metrics)
    """
    import sys as _sys  # debug
    def _dbg(msg):
        print(f"[DEBUG ppo/train] {msg}")
        _sys.stdout.flush()

    _dbg(f"train() called: num_envs={num_envs}, num_timesteps={num_timesteps}, episode_length={episode_length}, augment_pixels={augment_pixels}")

    assert batch_size * num_minibatches % num_envs == 0

    xt = time.time()

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        'Device count: %d, process count: %d (id %d), local device count: %d, '
        'devices to be used count: %d',
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count

    # The number of environment steps executed for every training step.
    env_step_per_training_step = (
            batch_size * unroll_length * num_minibatches * action_repeat
    )
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of training_step calls per training_epoch call.
    # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
    #                                 num_resets_per_eval))
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (
                num_evals_after_init
                * env_step_per_training_step
                * max(num_resets_per_eval, 1)
        )
    ).astype(int)

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    key_policy, key_value, key_cost_value, key_encoder = jax.random.split(global_key, 4)
    del global_key

    assert num_envs % device_count == 0

    _dbg("Wrapping environment...")
    env = _maybe_wrap_env(
        environment,
        wrap_env,
        num_envs,
        episode_length,
        action_repeat,
        device_count,
        key_env,
        wrap_env_fn,
        randomization_fn,
        vision_kwargs,
    )
    _dbg(f"Environment wrapped. obs_size={env.observation_size}, action_size={env.action_size}")
    # Optional: desynchronise the parallel training envs (research experiment on
    # lock-step bias in per-update safety metrics). Training env only; the eval
    # env is wrapped separately and is unaffected. Enable with
    # CRAX_DESYNC_EPISODES=1.
    if wrap_env and os.environ.get('CRAX_DESYNC_EPISODES', '0') == '1':
        env = envs.training.RandomStartStepWrapper(env, episode_length)
        print('[ppo/train] CRAX_DESYNC_EPISODES=1: training envs start at random episode steps')
    use_pmap = local_devices_to_use > 1
    if use_pmap:
        reset_fn = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
    else:
        reset_fn = jax.jit(jax.vmap(env.reset))
    pmap_axis_name = _PMAP_AXIS_NAME if use_pmap else None
    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(
        key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
    )
    _dbg(f"Calling reset_fn (use_pmap={use_pmap}, key_envs.shape={key_envs.shape})... This triggers JIT + pixel rendering.")
    _t0 = time.time()
    env_state = reset_fn(key_envs)
    _dbg(f"reset_fn completed in {time.time() - _t0:.1f}s")
    # Discard the batch axes over devices and envs.
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)
    _dbg(f"obs_shape after reset: {obs_shape}")

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize
    _dbg(f"Creating PPO network (network_factory={network_factory.__name__ if hasattr(network_factory, '__name__') else type(network_factory)})...")
    ppo_network = network_factory(
        obs_shape, env.action_size, preprocess_observations_fn=normalize
    )
    _dbg(f"PPO network created. cost_value_network={'yes' if ppo_network.cost_value_network else 'no'}")
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    def _policy_params_tuple(state: 'TrainingState') -> Tuple[Any, ...]:
        """(normalizer, policy, value) params, plus a 4th encoder slot iff
        `ppo_network` has a shared vision encoder. Keeping the encoder slot
        conditional (rather than always-present-but-None) preserves the
        existing 3-tuple shape for state-based / non-shared-encoder networks,
        so it doesn't break callers that unpack that tuple positionally.
        """
        base = (state.normalizer_params, state.params.policy, state.params.value)
        if ppo_network.encoder_network is not None:
            return base + (state.params.encoder,)
        return base

    optimizer = optax.adam(learning_rate=learning_rate)
    if max_grad_norm is not None:
        # TODO: Move gradient clipping to `training/gradients.py`.
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )

    # Use custom loss function if provided, otherwise default to standard PPO loss
    use_aux_in_loss = loss_fn is not None
    if loss_fn is None:
        loss_fn_to_use = functools.partial(
            ppo_losses.compute_ppo_loss,
            ppo_network=ppo_network,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clipping_epsilon=clipping_epsilon,
            normalize_advantage=normalize_advantage,
        )
    else:
        # Custom loss functions may need aux_state (e.g., Lagrange multiplier)
        loss_fn_to_use = functools.partial(
            loss_fn,
            ppo_network=ppo_network,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clipping_epsilon=clipping_epsilon,
            normalize_advantage=normalize_advantage,
        )

    # Create gradient update function
    # For standard PPO, we use the standard gradient_update_fn
    # For constrained RL with aux_state, we use a custom gradient update
    if not use_aux_in_loss:
        gradient_update_fn = gradients.gradient_update_fn(
            loss_fn_to_use, optimizer, pmap_axis_name=pmap_axis_name, has_aux=True
        )
    else:
        # Custom gradient update for loss functions that need aux_state
        def gradient_update_fn(params, normalizer_params, data, rng, optimizer_state, aux_state=None):
            def loss_wrapper(params):
                return loss_fn_to_use(params, normalizer_params, data, rng, aux_state=aux_state)
            
            grad_fn = jax.value_and_grad(loss_wrapper, has_aux=True)
            (loss, metrics), grads = grad_fn(params)
            if pmap_axis_name:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis_name)
            updates, new_optimizer_state = optimizer.update(grads, optimizer_state, params)
            new_params = optax.apply_updates(params, updates)
            return (loss, metrics), new_params, new_optimizer_state

    if policy_update_fn is not None:
        if ppo_network.encoder_network is not None:
            raise NotImplementedError('policy_update_fn (CPO) does not support a shared vision encoder.')
        policy_update_fn = functools.partial(
            policy_update_fn, ppo_network=ppo_network, pmap_axis_name=pmap_axis_name)

    metrics_aggregator = metric_logger.MetricsLogger(
        buffer_size=buffer_size,
        steps_between_logging=training_metrics_steps,
        progress_fn=progress_fn,
    )

    def minibatch_step(
            carry,
            data: types.Transition,
            normalizer_params: running_statistics.RunningStatisticsState,
            aux_state: Optional[Any] = None,
    ):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        
        if use_aux_in_loss:
            # Custom loss function with aux_state support
            (_, metrics), params, optimizer_state = gradient_update_fn(
                params,
                normalizer_params,
                data,
                key_loss,
                optimizer_state,
                aux_state,
            )
        else:
            # Standard PPO loss
            (_, metrics), params, optimizer_state = gradient_update_fn(
                params,
                normalizer_params,
                data,
                key_loss,
                optimizer_state=optimizer_state,
            )

        return (optimizer_state, params, key), metrics

    def sgd_step(
            carry,
            unused_t,
            data: types.Transition,
            normalizer_params: running_statistics.RunningStatisticsState,
            aux_state: Optional[Any] = None,
    ):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        if augment_pixels:
            key, key_rt = jax.random.split(key)
            r_translate = functools.partial(_random_translate_pixels, key=key_rt)
            data = types.Transition(
                observation=r_translate(data.observation),
                action=data.action,
                reward=data.reward,
                discount=data.discount,
                next_observation=r_translate(data.next_observation),
                extras=data.extras,
            )

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params, aux_state=aux_state),
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, key), metrics

    def training_step(
            carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
    ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        policy = make_policy(_policy_params_tuple(training_state))

        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env,
                current_state,
                policy,
                current_key,
                unroll_length,
                extra_fields=extra_fields,
            )
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            f,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )

        # Summarise pi_k's safety now, before learning turns it into pi_{k+1}.
        # 'cost' is only collected by the safe algorithms (it is in their extra_fields).
        update_safety_metrics = {}
        if 'cost' in data.extras['state_extras']:
            update_safety_metrics = _summarise_update_cost(
                data.extras['state_extras']['cost'], episode_length)
        # Episode-level view: true cost of the episodes that finished in this batch.
        ep_extras = data.extras['state_extras']
        if 'cost' in ep_extras.get('episode_metrics', {}):
            update_safety_metrics = {**update_safety_metrics, **_summarise_completed_episodes(
                ep_extras['episode_metrics']['cost'],
                ep_extras['episode_metrics']['length'],
                ep_extras['episode_done'],
                ep_extras['truncation'],
                episode_length,
                # Budget for the "share over budget" metrics; set by training/train_env.py
                # from --safety_bound (default 25).
                float(os.environ.get('CRAX_SAFETY_BOUND', '25')),
            )}
        
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        assert data.discount.shape[1:] == (unroll_length,)

        jax.debug.callback(
            metrics_aggregator.update_env_metrics,
            data.extras['state_extras']['episode_metrics'],
            data.extras['state_extras']['episode_done'],
            training_state.env_steps + env_step_per_training_step,
        )

        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            _remove_pixels(data.observation),
            pmap_axis_name=pmap_axis_name,
        )

        # Optional full-batch policy update (CPO's trust-region step), on pi_k's batch and
        # before the minibatch SGD; the SGD then fits the value functions from there.
        sgd_start_params = training_state.params
        policy_update_metrics = {}
        if policy_update_fn is not None:
            key_sgd, key_policy_update = jax.random.split(key_sgd)
            new_policy_params, policy_update_metrics = policy_update_fn(
                training_state.params, normalizer_params, data, key_policy_update)
            sgd_start_params = training_state.params.replace(policy=new_policy_params)

        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step, data=data, normalizer_params=normalizer_params, aux_state=training_state.aux_state
            ),
            (training_state.optimizer_state, sgd_start_params, key_sgd),
            (),
            length=num_updates_per_batch,
        )
        metrics = {**metrics, **policy_update_metrics}

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
            aux_state=training_state.aux_state,
        )

        # The safety summary of pi_k is merged first, so that post_step_fn can drive the
        # multiplier from it (e.g. PPO-Lag with --lagrangian_signal violation_rate / cvar).
        metrics = {**metrics, **update_safety_metrics}

        # Apply post-step hook if provided (for Lagrange multiplier updates, etc.)
        if post_step_fn is not None:
            new_training_state, extra_metrics = post_step_fn(new_training_state, metrics)
            metrics = {**metrics, **extra_metrics}
        
        if log_training_metrics:
            jax.debug.callback(
                metrics_aggregator.update_train_metrics,
                metrics,
                new_training_state.env_steps,
            )

        return (new_training_state, state, new_key), metrics

    def training_epoch(
            training_state: TrainingState, state: envs.State, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        (training_state, state, _), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        return training_state, state, loss_metrics

    if use_pmap:
        training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)
    else:
        # Single device: vmap over the leading device dim (size 1), then jit.
        # This mirrors pmap's behavior of mapping over the first axis.
        training_epoch = jax.jit(jax.vmap(training_epoch))

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
            training_state: TrainingState, env_state: envs.State, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state))
        result = training_epoch(training_state, env_state, key)
        training_state, env_state, metrics = _strip_weak_type(result)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
                      num_training_steps_per_epoch
                      * env_step_per_training_step
                      * max(num_resets_per_eval, 1)
              ) / epoch_training_time
        metrics = {
            'training/sps': sps,
            'training/walltime': training_walltime,
            **{f'training/{name}': value for name, value in metrics.items()},
        }
        return training_state, env_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade

    # Initialize model params and training state.
    # Handle optional cost_value network for constrained RL variants
    cost_value_params = None
    if ppo_network.cost_value_network is not None:
        cost_value_params = ppo_network.cost_value_network.init(key_value)

    encoder_params = None
    if ppo_network.encoder_network is not None:
        encoder_params = ppo_network.encoder_network.init(key_encoder)

    _dbg("Initializing network params...")
    init_params = ppo_losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
        cost_value=cost_value_params,
        encoder=encoder_params,
    )
    _dbg(f"Network params initialized. policy keys: {list(init_params.policy['params'].keys()) if isinstance(init_params.policy, dict) and 'params' in init_params.policy else 'N/A'}")

    # Initialize aux_state if init function provided (for Lagrange multipliers, PID state, etc.)
    initial_aux_state = None
    if init_aux_state_fn is not None:
        initial_aux_state = init_aux_state_fn()

    _dbg("Building obs_spec and TrainingState...")
    obs_spec = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
    )
    _dbg(f"obs_spec: {obs_spec}")
    _dbg(f"obs_spec after _remove_pixels: {_remove_pixels(obs_spec)}")
    training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        optimizer_state=optimizer.init(init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
        params=init_params,
        normalizer_params=running_statistics.init_state(
            _remove_pixels(obs_spec)
        ),
        env_steps=types.UInt64(hi=0, lo=0),
        aux_state=initial_aux_state,
    )
    _dbg("TrainingState created.")

    def _check_normalizer_shape_compatible(loaded_normalizer, current_normalizer):
        """Check if loaded normalizer has compatible shape with current env."""
        loaded_mean = loaded_normalizer.mean
        current_mean = current_normalizer.mean
        # Handle both array and nested dict observations
        if isinstance(loaded_mean, dict) and isinstance(current_mean, dict):
            for key in current_mean:
                if key not in loaded_mean:
                    return False
                if loaded_mean[key].shape != current_mean[key].shape:
                    return False
            return True
        elif isinstance(loaded_mean, jnp.ndarray) and isinstance(current_mean, jnp.ndarray):
            return loaded_mean.shape == current_mean.shape
        return False

    if restore_checkpoint_path is not None:
        params = checkpoint.load(restore_checkpoint_path)
        value_params = params[2] if restore_value_fn else init_params.value
        # Old (pre shared-encoder) checkpoints only have 3 elements; fall
        # back to a freshly-initialized encoder for those.
        encoder_params = params[3] if len(params) > 3 else init_params.encoder
        # Check if normalizer shapes are compatible
        if _check_normalizer_shape_compatible(params[0], training_state.normalizer_params):
            normalizer_to_use = params[0]
        else:
            logging.warning(
                'Checkpoint normalizer shape does not match current observation shape. '
                'Using freshly initialized normalizer. This may happen when transferring '
                'between environments with different observation sizes.'
            )
            normalizer_to_use = training_state.normalizer_params
        training_state = training_state.replace(
            normalizer_params=normalizer_to_use,
            params=training_state.params.replace(
                policy=params[1], value=value_params, encoder=encoder_params
            ),
        )

    if restore_params is not None:
        logging.info('Restoring TrainingState from `restore_params`.')
        value_params = restore_params[2] if restore_value_fn else init_params.value
        encoder_params = (
            restore_params[3] if len(restore_params) > 3 else init_params.encoder
        )
        # Check if normalizer shapes are compatible
        if _check_normalizer_shape_compatible(restore_params[0], training_state.normalizer_params):
            normalizer_to_use = restore_params[0]
        else:
            logging.warning(
                'Restored params normalizer shape does not match current observation shape. '
                'Using freshly initialized normalizer. This may happen when transferring '
                'between environments with different observation sizes.'
            )
            normalizer_to_use = training_state.normalizer_params
        training_state = training_state.replace(
            normalizer_params=normalizer_to_use,
            params=training_state.params.replace(
                policy=restore_params[1], value=value_params, encoder=encoder_params
            ),
        )

    if num_timesteps == 0:
        return (
            make_policy,
            _policy_params_tuple(training_state),
            {},
        )

    _dbg("Replicating training state to devices...")
    # Add a leading device dimension for vmap/pmap.
    # This replaces jax.device_put_replicated (removed in newer JAX).
    training_state = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (local_devices_to_use,) + x.shape),
        training_state,
    )
    _dbg("Training state replicated.")

    # Unwrapped eval env, kept for the final-policy safety evaluation below.
    raw_eval_env = eval_env or environment

    # Only create evaluator if evaluation is enabled
    evaluator = None
    if num_evals > 0:
        eval_env = _maybe_wrap_env(
            eval_env or environment,
            wrap_env,
            num_eval_envs,
            episode_length,
            action_repeat,
            device_count=1,  # eval on the host only
            key_env=eval_key,
            wrap_env_fn=wrap_env_fn,
            randomization_fn=randomization_fn,
            vision_kwargs=vision_kwargs,
        )
        _dbg(f"Creating Evaluator (num_eval_envs={num_eval_envs})...")
        evaluator = acting.Evaluator(
            eval_env,
            functools.partial(make_policy, deterministic=deterministic_eval),
            num_eval_envs=num_eval_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
            key=eval_key,
        )
        _dbg("Evaluator created.")

    def _for_progress(m):
        """Metrics to pass to progress_fn.

        With log_training_metrics on, every update is already logged individually by the
        metrics logger. training_metrics holds the per-update arrays of the whole epoch;
        passing them to progress_fn logs their *average* at the same wandb step as the
        epoch's last update and overwrites that update's values. So only the epoch-level
        timing entries (sps, walltime) and evaluation metrics are passed through.
        """
        if not log_training_metrics:
            return m
        return {k: v for k, v in m.items()
                if not k.startswith('training/') or k in ('training/sps', 'training/walltime')}

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1 and evaluator is not None:
        _dbg("Running initial evaluation...")
        _t0 = time.time()
        metrics = evaluator.run_evaluation(
            _unpmap(_policy_params_tuple(training_state)),
            training_metrics={},
        )
        _dbg(f"Initial evaluation completed in {time.time() - _t0:.1f}s")
        logging.info(metrics)
        progress_fn(0, metrics)

    training_metrics = {}
    training_walltime = 0
    current_step = 0

    _dbg(f"Entering main training loop: {num_evals_after_init} iterations, {max(num_resets_per_eval, 1)} resets/eval, {num_training_steps_per_epoch} steps/epoch")
    for it in range(num_evals_after_init):
        logging.info('starting iteration %s %s', it, time.time() - xt)

        for _ in range(max(num_resets_per_eval, 1)):
            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            _dbg(f"Starting training_epoch_with_timing (iter {it})... (includes JIT compile on first call)")
            _t0 = time.time()
            (training_state, env_state, training_metrics) = (
                training_epoch_with_timing(training_state, env_state, epoch_keys)
            )
            _dbg(f"training_epoch_with_timing completed in {time.time() - _t0:.1f}s (iter {it})")
            current_step = int(_unpmap(training_state.env_steps))
            progress_fn(current_step, _for_progress(training_metrics))

            key_envs = jax.vmap(
                lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
            )(key_envs, key_envs.shape[1])
            # TODO: move extra reset logic to the AutoResetWrapper.
            env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

        if process_id != 0:
            continue

        # Process id == 0.
        params = _unpmap(_policy_params_tuple(training_state))

        policy_params_fn(current_step, make_policy, params)

        if save_checkpoint_path is not None:
            ckpt_config = checkpoint.network_config(
                observation_size=obs_shape,
                action_size=env.action_size,
                normalize_observations=normalize_observations,
                network_factory=network_factory,
            )
            checkpoint.save(
                save_checkpoint_path, current_step, params, ckpt_config
            )

        # Only run evaluation if enabled
        if num_evals > 0 and evaluator is not None:
            metrics = evaluator.run_evaluation(
                params,
                training_metrics,
            )
            logging.info(metrics)
            progress_fn(current_step, _for_progress(metrics))

    total_steps = current_step
    if not total_steps >= num_timesteps:
        raise AssertionError(
            f'Total steps {total_steps} is less than `num_timesteps`='
            f' {num_timesteps}.'
        )

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    params = _unpmap(_policy_params_tuple(training_state))

    # If no evaluation was run, create basic final metrics
    if not metrics:
        metrics = {'training/final_step': total_steps}
        if training_metrics:
            metrics.update(training_metrics)

    # Final-policy safety evaluation (Spoor et al., 2026): N fresh, complete episodes of the
    # frozen final policy, once with exploration noise (stochastic) and once greedy
    # (deterministic). Set by training/train_env.py from --final_eval_episodes (0 = off).
    final_eval_episodes = int(os.environ.get('CRAX_FINAL_EVAL_EPISODES', '0'))
    if process_id == 0 and final_eval_episodes > 0:
        budget = float(os.environ.get('CRAX_SAFETY_BOUND', '25'))
        final_key = jax.random.PRNGKey(seed + 1_000_003)   # separate stream from training/eval
        wrap_key, final_key = jax.random.split(final_key)
        final_env = _maybe_wrap_env(
            raw_eval_env, wrap_env, final_eval_episodes, episode_length, action_repeat,
            device_count=1, key_env=wrap_key, wrap_env_fn=wrap_env_fn,
            randomization_fn=randomization_fn, vision_kwargs=vision_kwargs,
        )
        for mode, deterministic in (('stochastic', False), ('greedy', True)):
            final_key, mode_key = jax.random.split(final_key)
            final_evaluator = acting.Evaluator(
                final_env,
                functools.partial(make_policy, deterministic=deterministic),
                num_eval_envs=final_eval_episodes,
                episode_length=episode_length,
                action_repeat=action_repeat,
                key=mode_key,
            )
            raw = final_evaluator.run_evaluation(params, training_metrics={}, aggregate_episodes=False)
            if 'eval/episode_cost' not in raw:
                print('[ppo/train] final evaluation skipped: the environment reports no cost')
                break
            final_metrics = _final_safety_metrics(
                raw['eval/episode_cost'], raw['eval/episode_reward'], budget, f'final_eval/{mode}')
            metrics = {**metrics, **final_metrics}
            print(f"[ppo/train] final eval ({mode}, {final_eval_episodes} episodes): "
                  f"cost {final_metrics[f'final_eval/{mode}/cost_mean']:.2f}, "
                  f"V {final_metrics[f'final_eval/{mode}/violation_rate']:.3f} "
                  f"(95% upper {final_metrics[f'final_eval/{mode}/violation_rate_upper95']:.3f}), "
                  f"D_norm+ {final_metrics[f'final_eval/{mode}/d_norm_plus']:.3f}")

    logging.info('total steps: %s', total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics, eval_env)
