"""Tests for the CPO implementation (training/agents/cpo/losses.py).

The closed-form CPO step is checked against a generic constrained optimiser (SciPy SLSQP)
on random problems, in every optimisation case. The full policy update is checked on a
synthetic batch with a real CRAX policy network.

Run from the repo root with the crax env active:
    python -m pytest tests/test_cpo.py -q
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from training.agents.cpo import losses as L

jax.config.update("jax_enable_x64", False)
DELTA = 0.01


# ---------------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------------

def _spd(rng, n, cond=20.0):
    """Random symmetric positive-definite matrix with a moderate condition number."""
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    eig = np.exp(rng.uniform(0, math.log(cond), n))
    return (q * eig) @ q.T


def test_conjugate_gradient_solves_spd_system():
    rng = np.random.default_rng(0)
    H = _spd(rng, 30)
    b = rng.normal(size=30)
    x = L.conjugate_gradient(lambda v: jnp.asarray(H, jnp.float32) @ v, jnp.asarray(b, jnp.float32), iters=60)
    np.testing.assert_allclose(np.asarray(x), np.linalg.solve(H, b), rtol=1e-3, atol=1e-3)


def test_diag_gaussian_kl():
    loc = jnp.array([[0.3, -1.0]])
    scale = jnp.array([[0.5, 2.0]])
    assert float(L.diag_gaussian_kl(loc, scale, loc, scale)[0]) == pytest.approx(0.0, abs=1e-7)
    # 1-D closed form: KL(N(0,1) || N(1,2^2)) = log 2 + (1 + 1)/(2*4) - 1/2
    kl = L.diag_gaussian_kl(jnp.array([[0.0]]), jnp.array([[1.0]]), jnp.array([[1.0]]), jnp.array([[2.0]]))
    assert float(kl[0]) == pytest.approx(math.log(2.0) + 2.0 / 8.0 - 0.5, rel=1e-5)


# ---------------------------------------------------------------------------------------
# The CPO step against a generic solver
# ---------------------------------------------------------------------------------------

def _cpo(H, g, b, c):
    """Run cpo_direction with exact H^-1 products."""
    v, w = np.linalg.solve(H, g), np.linalg.solve(H, b)
    q, r, s = g @ v, g @ w, b @ w
    out = L.cpo_direction(*(jnp.asarray(z, jnp.float32) for z in (v, w, q, r, s, c)),
                          DELTA, jnp.asarray(b @ b <= 1e-8))
    return np.asarray(out.direction, np.float64), int(out.optim_case)


def _slsqp(H, g, b, c):
    """Reference: maximise g.x s.t. c + b.x <= 0 and 0.5 x.H.x <= delta (from several starts)."""
    cons = [{"type": "ineq", "fun": lambda x: -(c + b @ x), "jac": lambda x: -b},
            {"type": "ineq", "fun": lambda x: DELTA - 0.5 * x @ H @ x, "jac": lambda x: -H @ x}]
    best = None
    for x0 in (np.zeros_like(g), -0.01 * np.linalg.solve(H, b), 0.01 * np.linalg.solve(H, g)):
        res = minimize(lambda x: -g @ x, x0, jac=lambda x: -g, constraints=cons, method="SLSQP",
                       options={"ftol": 1e-12, "maxiter": 500})
        feasible = (c + b @ res.x <= 1e-6) and (0.5 * res.x @ H @ res.x <= DELTA * (1 + 1e-4))
        if feasible and (best is None or g @ res.x > g @ best):
            best = res.x
    return best


def _random_problem(rng, n, case):
    H = _spd(rng, n)
    g = rng.normal(size=n)
    b = rng.normal(size=n)
    if case == "aligned":            # cost rises along the reward direction: constraint matters
        b = 0.7 * g + 0.3 * b
    s = b @ np.linalg.solve(H, b)
    radius = math.sqrt(2 * DELTA * s)   # |c| below this: the constraint plane meets the trust region
    c = {"tr_feasible": -3.0 * radius, "constrained": -0.5 * radius, "aligned": -0.3 * radius,
         "feasible_recovery": 0.5 * radius, "infeasible": 3.0 * radius}[case if case != "aligned" else "aligned"]
    return H, g, b, c


@pytest.mark.parametrize("case,expected", [
    ("tr_feasible", L.CASE_TR_FEASIBLE),
    ("constrained", L.CASE_CONSTRAINED),
    ("aligned", L.CASE_CONSTRAINED),
    ("feasible_recovery", L.CASE_FEASIBLE_RECOVERY),
])
def test_cpo_step_matches_generic_solver(case, expected):
    rng = np.random.default_rng(hash(case) % 2**32)
    for _ in range(20):
        H, g, b, c = _random_problem(rng, 6, case)
        x, optim_case = _cpo(H, g, b, c)
        assert optim_case == expected
        ref = _slsqp(H, g, b, c)
        assert ref is not None
        # Feasible (small float32 slack) and as good as the generic solver.
        assert c + b @ x <= 1e-4 * (1 + abs(c))
        assert 0.5 * x @ H @ x <= DELTA * 1.01
        assert g @ x >= g @ ref - 1e-3 * (1 + abs(g @ ref))


def test_infeasible_case_takes_the_steepest_cost_decrease():
    rng = np.random.default_rng(3)
    for _ in range(20):
        H, g, b, c = _random_problem(rng, 6, "infeasible")
        x, optim_case = _cpo(H, g, b, c)
        assert optim_case == L.CASE_INFEASIBLE_RECOVERY
        # Minimiser of b.x on the trust region is -sqrt(2 delta / s) H^-1 b.
        w = np.linalg.solve(H, b)
        np.testing.assert_allclose(x, -math.sqrt(2 * DELTA / (b @ w)) * w, rtol=1e-3, atol=1e-6)
        assert 0.5 * x @ H @ x == pytest.approx(DELTA, rel=1e-3)


def test_zero_cost_gradient_gives_trpo_step():
    rng = np.random.default_rng(4)
    H, g = _spd(rng, 6), rng.normal(size=6)
    x, optim_case = _cpo(H, g, np.zeros(6), -1.0)
    assert optim_case == L.CASE_NO_COST_GRADIENT
    v = np.linalg.solve(H, g)
    np.testing.assert_allclose(x, math.sqrt(2 * DELTA / (g @ v)) * v, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------------------
# Line search
# ---------------------------------------------------------------------------------------

def test_line_search_picks_largest_passing_step():
    # KL grows quadratically with the step; only steps with f^2 * 0.05 <= 0.01, i.e. f <= 0.447, pass.
    evaluate = lambda x: (0.05 * jnp.sum(x ** 2), jnp.sum(x), jnp.array(0.0))
    new, info = L.backtracking_line_search(
        jnp.zeros(1), jnp.ones(1), evaluate, jnp.array(L.CASE_TR_FEASIBLE), jnp.array(-1.0),
        DELTA, jnp.array(0.0), jnp.array(0.0), 0.8, 10)
    assert float(info["step_accepted"]) == 1.0
    assert float(info["step_fraction"]) == pytest.approx(0.8 ** 4)   # 0.41 is the first <= 0.447
    assert float(new[0]) == pytest.approx(0.8 ** 4)


def test_line_search_rejects_cost_increase_when_over_budget():
    evaluate = lambda x: (jnp.array(0.0), jnp.sum(x), jnp.sum(x))   # cost surrogate rises with the step
    new, info = L.backtracking_line_search(
        jnp.zeros(1), jnp.ones(1), evaluate, jnp.array(L.CASE_FEASIBLE_RECOVERY), jnp.array(0.5),
        DELTA, jnp.array(0.0), jnp.array(0.0), 0.8, 10)
    assert float(info["step_accepted"]) == 0.0
    assert float(new[0]) == 0.0


# ---------------------------------------------------------------------------------------
# Full policy update with a real CRAX policy network on a synthetic batch
# ---------------------------------------------------------------------------------------

def _synthetic_batch(key, ppo_network, policy_params, normalizer, B=64, T=8, obs_dim=10):
    from training import types
    k1, k2, k3, k4 = jax.random.split(key, 4)
    obs = jax.random.normal(k1, (B, T, obs_dim))
    logits = ppo_network.policy_network.apply(normalizer, policy_params, obs)
    dist = ppo_network.parametric_action_distribution
    raw = dist.sample_no_postprocessing(logits, k2)
    # Cost is high when the first action dimension is positive: the cost gradient is clear.
    cost = (raw[..., 0] > 0).astype(jnp.float32)
    reward = raw[..., 1] + 0.1 * jax.random.normal(k3, (B, T))
    zeros = jnp.zeros((B, T))
    return types.Transition(
        observation=obs, action=jnp.tanh(raw), reward=reward, discount=jnp.ones((B, T)),
        next_observation=obs + 0.01 * jax.random.normal(k4, obs.shape),
        extras={"policy_extras": {"raw_action": raw, "log_prob": dist.log_prob(logits, raw)},
                "state_extras": {"cost": cost, "truncation": zeros}})


@pytest.mark.parametrize("budget,expect_cost_down", [(1.0, False), (0.0, True)])
def test_full_policy_update(budget, expect_cost_down):
    from training.agents.ppo import losses as ppo_losses
    from training.agents.ppo import networks as ppo_networks
    from training.acme import running_statistics, specs

    obs_dim, act = 10, 2
    net = ppo_networks.make_ppo_networks(obs_dim, act, cost_value_hidden_layer_sizes=(32, 32),
                                         value_hidden_layer_sizes=(32, 32))
    key = jax.random.PRNGKey(0)
    kp, kv, kc, kd = jax.random.split(key, 4)
    params = ppo_losses.PPONetworkParams(policy=net.policy_network.init(kp), value=net.value_network.init(kv),
                                         cost_value=net.cost_value_network.init(kc))
    normalizer = running_statistics.init_state(specs.Array((obs_dim,), jnp.dtype("float32")))
    data = _synthetic_batch(kd, net, params.policy, normalizer, obs_dim=obs_dim)

    update = jax.jit(lambda p, d: L.cpo_policy_update(
        p, normalizer, d, key, ppo_network=net, pmap_axis_name=None,
        per_step_safety_bound=budget, target_kl=DELTA, cg_iters=10, cg_damping=0.1,
        backtrack_coeff=0.8, backtrack_iters=10, fvp_subsample=1,
        discounting=0.99, reward_scaling=1.0, gae_lambda=0.95))
    new_policy, m = update(params, data)
    m = {k: float(v) for k, v in m.items()}

    # Mean cost is ~0.5 per step. Budget 1.0 -> feasible with slack; budget 0 -> over budget.
    assert (m["cpo/c"] < 0) == (budget == 1.0)
    assert m["cpo/step_accepted"] == 1.0
    assert 0.0 < m["cpo/kl"] <= DELTA * 1.0001
    if expect_cost_down:
        assert m["cpo/optim_case"] in (L.CASE_INFEASIBLE_RECOVERY, L.CASE_FEASIBLE_RECOVERY)
        assert m["cpo/surr_cost_change"] < 0          # recovery reduces the cost surrogate
    else:
        assert m["cpo/surr_reward_change"] > 0        # feasible: the reward surrogate improves
    changed = jax.tree_util.tree_reduce(
        lambda a, x: a or bool(jnp.any(x)), jax.tree_util.tree_map(lambda a, b: a != b, new_policy, params.policy), False)
    assert changed
