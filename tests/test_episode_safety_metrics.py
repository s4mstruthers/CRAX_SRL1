"""Tests for the episode-level safety metrics (_summarise_completed_episodes).

Run from the repo root with the crax env active:
    python -m pytest tests/test_episode_safety_metrics.py -q
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

from training.agents.ppo.train import (
    CDF_THRESHOLDS, _final_safety_metrics, _summarise_completed_episodes, _threshold_name,
    _violation_rate_upper_bound)

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
    # Spoor et al. (2026) metrics
    assert math.isclose(out["safety_ep/d_norm"], (ref.mean() - D) / D, rel_tol=1e-4)
    over = ref[ref > D]
    assert math.isclose(out["safety_ep/d_norm_plus"], (over - D).mean() / D, rel_tol=1e-4)
    assert math.isclose(out["safety_ep/frac_zero_cost"], (ref <= 0).mean(), abs_tol=1e-6)
    for k in CDF_THRESHOLDS:
        key = f"safety_ep/cdf_dnorm_le_{_threshold_name(k)}"
        assert math.isclose(out[key], ((ref - D) / D <= k).mean(), abs_tol=1e-6), key


def test_no_finished_episodes_gives_nan():
    (cost, length, done, trunc), _ = _batch()
    out = {k: float(v) for k, v in summarise(cost, length, jnp.zeros_like(done), trunc, L, D).items()}
    assert out["safety_ep/count"] == 0
    assert all(math.isnan(v) for k, v in out.items() if k != "safety_ep/count")


def test_threshold_names_are_key_safe():
    assert _threshold_name(-0.5) == "m0p5"
    assert _threshold_name(0.0) == "0"
    assert _threshold_name(2.0) == "2"
    assert all("." not in _threshold_name(k) and "-" not in _threshold_name(k) for k in CDF_THRESHOLDS)


def test_d_norm_plus_is_zero_when_no_episode_violates():
    (cost, length, done, trunc), _ = _batch()
    out = {k: float(v) for k, v in summarise(cost, length, done, trunc, L, 1e6).items()}
    assert out["safety_ep/frac_over_budget"] == 0.0
    assert out["safety_ep/d_norm_plus"] == 0.0


def test_final_safety_metrics_against_numpy():
    rng = np.random.default_rng(1)
    costs = rng.gamma(2.0, 12.0, 1000)
    rewards = rng.normal(50.0, 5.0, 1000)
    m = _final_safety_metrics(costs, rewards, D, "final_eval/greedy")
    over = costs > D
    assert m["final_eval/greedy/num_episodes"] == 1000
    assert math.isclose(m["final_eval/greedy/violation_rate"], over.mean())
    assert math.isclose(m["final_eval/greedy/d_norm"], (costs.mean() - D) / D)
    assert math.isclose(m["final_eval/greedy/d_norm_plus"], (costs[over] - D).mean() / D)
    assert math.isclose(m["final_eval/greedy/cost_cvar95"], costs[costs >= np.quantile(costs, 0.95)].mean())
    assert m["final_eval/greedy/violation_rate_upper95"] > m["final_eval/greedy/violation_rate"]
    assert m["final_eval/greedy/episode_costs"].shape == (1000,)


def test_zero_violations_upper_bound_is_rule_of_three():
    # 0 violations in n episodes: the 95% upper bound is about 3/n.
    for n in (100, 1000):
        ub = _violation_rate_upper_bound(0, n)
        assert 2.9 / n < ub < 3.1 / n
    m = _final_safety_metrics(np.zeros(100), np.zeros(100), D, "x")
    assert m["x/violation_rate"] == 0.0 and m["x/d_norm_plus"] == 0.0 and m["x/frac_zero_cost"] == 1.0
