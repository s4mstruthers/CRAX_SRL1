"""Tests for the episode-level safety metrics (_summarise_completed_episodes).

Run from the repo root with the crax env active:
    python -m pytest tests/test_episode_safety_metrics.py -q
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

from training.agents.ppo.train import _summarise_completed_episodes

R, T, E, L, D = 16, 8, 64, 1000, 25.0   # rollouts, unroll length, envs, episode length, budget
summarise = jax.jit(_summarise_completed_episodes, static_argnums=(4, 5))


def _batch():
    """Synthetic batch with three kinds of finished episodes plus unfinished-step noise."""
    rng = np.random.default_rng(0)
    cost, length, done, trunc = (np.zeros((R, T, E), np.float32) for _ in range(4))
    full = rng.gamma(2.0, 15.0, 40).astype(np.float32)          # normal full-length episodes
    for k, c in enumerate(full):
        cost[k % R, k % T, k], length[k % R, k % T, k], done[k % R, k % T, k], trunc[k % R, k % T, k] = c, 999, 1, 1
    for k in range(10):                                           # desync-shortened: must be ignored
        cost[k, 0, 40 + k], length[k, 0, 40 + k], done[k, 0, 40 + k], trunc[k, 0, 40 + k] = 500, 300, 1, 1
    terminated = np.array([5, 10, 60, 80, 200], np.float32)       # ended early by the env: kept
    for k, c in enumerate(terminated):
        cost[k, 1, 50 + k], length[k, 1, 50 + k], done[k, 1, 50 + k], trunc[k, 1, 50 + k] = c, 250, 1, 0
    noise = done == 0                                             # running sums of unfinished episodes
    noise[:, :4, :] = False
    cost[noise] += 999.0
    length[noise] = 500
    expected = np.concatenate([full, terminated])
    return (jnp.array(cost), jnp.array(length), jnp.array(done), jnp.array(trunc)), expected


def test_matches_numpy_on_valid_episodes():
    (cost, length, done, trunc), ref = _batch()
    out = {k: float(v) for k, v in summarise(cost, length, done, trunc, L, D).items()}
    q = np.quantile(ref, [0.5, 0.9, 0.95, 0.99])
    assert out["safety_ep/count"] == len(ref)
    assert math.isclose(out["safety_ep/cost_mean"], ref.mean(), rel_tol=1e-4)
    assert math.isclose(out["safety_ep/cost_p50"], q[0], rel_tol=1e-4)
    assert math.isclose(out["safety_ep/cost_p90"], q[1], rel_tol=1e-4)
    assert math.isclose(out["safety_ep/cost_p99"], q[3], rel_tol=1e-4)
    assert math.isclose(out["safety_ep/cost_max"], ref.max(), rel_tol=1e-4)
    assert math.isclose(out["safety_ep/cost_cvar95"], ref[ref >= q[2]].mean(), rel_tol=1e-4)
    assert math.isclose(out["safety_ep/frac_over_budget"], (ref > D).mean(), rel_tol=1e-4)
    assert math.isclose(out["safety_ep/frac_over_2x_budget"], (ref > 2 * D).mean(), rel_tol=1e-4)


def test_no_finished_episodes_gives_nan():
    (cost, length, done, trunc), _ = _batch()
    out = {k: float(v) for k, v in summarise(cost, length, jnp.zeros_like(done), trunc, L, D).items()}
    assert out["safety_ep/count"] == 0
    assert all(math.isnan(v) for k, v in out.items() if k != "safety_ep/count")
