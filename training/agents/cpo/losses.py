"""Constrained Policy Optimization (CPO; Achiam, Held, Tamar & Abbeel, 2017).

CPO changes how the *policy* is updated. Each update solves, on the full batch,

    maximise_x   g^T x                         (improve the reward surrogate)
    subject to   c + b^T x <= 0                (linearised cost constraint)
                 0.5 x^T H x <= delta          (trust region: KL(pi_old || pi_new) <= delta)

where x is the change of the flattened policy parameters, g / b are the gradients of the
reward / cost surrogates, H is the Fisher matrix (Hessian of the mean KL), and
c = J_C(pi_k) - d is how far the current policy is above (> 0) or below (< 0) the budget.
The problem is solved in closed form through its two-variable dual (lambda for the trust
region, nu for the cost constraint), followed by a backtracking line search that checks the
KL, the reward surrogate and the cost surrogate on the batch. The value functions are fitted
separately with the usual minibatch SGD (see compute_cpo_value_loss).

Units. CRAX works per step (PPO-Lag compares the batch's mean per-step cost with d / T), and
the GAE cost advantage estimates the change of the *per-step* cost rate, so c is
c = mean per-step cost of the batch - d / T. This is the same rescaling as the reference
implementation in OpenAI's safety-starter-agents (c = (EpCost - d) / EpLen).

Everything here is pure JAX with fixed loop lengths, so it runs inside the jitted training
step. All functions are written for a single device and take an optional pmap axis name to
average across devices.
"""

from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from training import types
from training.agents.ppo import networks as ppo_networks
from training.agents.ppo.losses import PPONetworkParams, compute_gae, with_shared_latent

EPS = 1e-8

# Optimisation cases, as in Achiam et al. (2017) and safety-starter-agents.
CASE_INFEASIBLE_RECOVERY = 0   # x=0 infeasible and no feasible point in the trust region: only reduce cost
CASE_FEASIBLE_RECOVERY = 1     # x=0 infeasible, but part of the trust region is feasible
CASE_CONSTRAINED = 2           # x=0 feasible and the cost constraint cuts through the trust region
CASE_TR_FEASIBLE = 3           # the whole trust region is feasible: plain TRPO step
CASE_NO_COST_GRADIENT = 4      # cost gradient ~ 0 and x=0 feasible: plain TRPO step


# ---------------------------------------------------------------------------------------
# Small numerical building blocks
# ---------------------------------------------------------------------------------------

def diag_gaussian_kl(loc_p: jnp.ndarray, scale_p: jnp.ndarray,
                     loc_q: jnp.ndarray, scale_q: jnp.ndarray) -> jnp.ndarray:
    """KL(p || q) between diagonal Gaussians, summed over the last (action) axis.

    CRAX's policy is a Gaussian followed by tanh. tanh is a bijection, and KL divergence is
    unchanged by applying the same bijection to both distributions, so the KL of the
    pre-tanh Gaussians is exactly the KL of the action distributions.
    """
    return jnp.sum(
        jnp.log(scale_q / scale_p)
        + (jnp.square(scale_p) + jnp.square(loc_p - loc_q)) / (2.0 * jnp.square(scale_q))
        - 0.5,
        axis=-1,
    )


def conjugate_gradient(matvec: Callable[[jnp.ndarray], jnp.ndarray], b: jnp.ndarray,
                       iters: int, residual_tol: float = 1e-10) -> jnp.ndarray:
    """Approximately solve A x = b for symmetric positive-definite A, given only x -> A x.

    A fixed number of iterations keeps it jittable; once the squared residual is below
    residual_tol the iterate is frozen.
    """
    def body(carry, _):
        x, r, p, rr = carry
        ap = matvec(p)
        alpha = rr / (jnp.dot(p, ap) + EPS)
        x_new = x + alpha * p
        r_new = r - alpha * ap
        rr_new = jnp.dot(r_new, r_new)
        p_new = r_new + (rr_new / (rr + EPS)) * p
        done = rr < residual_tol
        keep = lambda new, old: jnp.where(done, old, new)
        return (keep(x_new, x), keep(r_new, r), keep(p_new, p), keep(rr_new, rr)), None

    x0 = jnp.zeros_like(b)
    (x, _, _, _), _ = jax.lax.scan(body, (x0, b, b, jnp.dot(b, b)), None, length=iters)
    return x


class CPOStep(NamedTuple):
    """Search direction of one CPO update and the quantities that produced it."""
    direction: jnp.ndarray   # x*: change of the flat policy parameters (before the line search)
    optim_case: jnp.ndarray  # one of the CASE_* constants
    lam: jnp.ndarray         # trust-region multiplier lambda*
    nu: jnp.ndarray          # cost-constraint multiplier nu*
    q: jnp.ndarray           # g^T H^-1 g
    r: jnp.ndarray           # g^T H^-1 b
    s: jnp.ndarray           # b^T H^-1 b
    A: jnp.ndarray           # q - r^2 / s
    B: jnp.ndarray           # 2 delta - c^2 / s  (> 0: the constraint plane meets the trust region)


def cpo_direction(v: jnp.ndarray, w: jnp.ndarray, q: jnp.ndarray, r: jnp.ndarray,
                  s: jnp.ndarray, c: jnp.ndarray, target_kl: float,
                  b_is_zero: jnp.ndarray) -> CPOStep:
    """Closed-form solution of the CPO subproblem through its dual.

    Args:
      v: H^-1 g (from conjugate gradient), g = gradient of the reward surrogate.
      w: H^-1 b, b = gradient of the cost surrogate.
      q, r, s: g^T H^-1 g, g^T H^-1 b, b^T H^-1 b.
      c: constraint value J_C(pi_k) - d (in the units of the cost surrogate).
      target_kl: trust-region size delta.
      b_is_zero: True if the cost gradient is (numerically) zero.

    The primal solution is x* = (1/lambda*) H^-1 (g - nu* b). For fixed lambda the best nu is
    nu(lambda) = max(0, lambda c + r) / s. Substituting it gives a one-dimensional problem in
    lambda with two pieces (nu > 0 on region A, nu = 0 on region B):
      f_A(lambda) = -0.5 (A / lambda + B lambda) + r c / s,   on {lambda : lambda c + r > 0}
      f_B(lambda) = -0.5 (q / lambda + 2 delta lambda),        on {lambda : lambda c + r <= 0}
    Each piece is maximised by projecting its unconstrained maximiser onto its region, and
    the better of the two is taken. (The sign of r follows from g being an ascent direction;
    tests/test_cpo.py checks the result against a generic constrained solver.)
    """
    s_safe = s + EPS
    A = q - jnp.square(r) / s_safe
    B = 2.0 * target_kl - jnp.square(c) / s_safe

    optim_case = jnp.where(
        b_is_zero & (c < 0), CASE_NO_COST_GRADIENT,
        jnp.where((c < 0) & (B < 0), CASE_TR_FEASIBLE,
                  jnp.where((c < 0) & (B >= 0), CASE_CONSTRAINED,
                            jnp.where((c >= 0) & (B >= 0), CASE_FEASIBLE_RECOVERY,
                                      CASE_INFEASIBLE_RECOVERY))))

    # Pure trust-region (TRPO) step size: maximiser of f_B without the region restriction.
    lam_trpo = jnp.sqrt(jnp.maximum(q, 0.0) / (2.0 * target_kl))

    # Boundary between the regions: lambda c + r = 0  <=>  lambda = -r / c.
    c_safe = jnp.where(c >= 0, jnp.maximum(c, EPS), jnp.minimum(c, -EPS))
    boundary = -r / c_safe
    big = jnp.inf
    # c < 0: region A = [0, boundary], region B = [boundary, inf)
    # c >= 0: region A = [boundary, inf), region B = [0, boundary]
    a_lo, a_hi = jnp.where(c < 0, 0.0, boundary), jnp.where(c < 0, boundary, big)
    b_lo, b_hi = jnp.where(c < 0, boundary, 0.0), jnp.where(c < 0, big, boundary)
    proj = lambda x, lo, hi: jnp.maximum(lo, jnp.minimum(hi, x))
    lam_a = proj(jnp.sqrt(jnp.maximum(A, 0.0) / jnp.maximum(B, EPS)), a_lo, a_hi)
    lam_b = proj(lam_trpo, b_lo, b_hi)
    lam_a, lam_b = jnp.maximum(lam_a, 0.0), jnp.maximum(lam_b, 0.0)
    f_a = -0.5 * (A / (lam_a + EPS) + B * lam_a) + r * c / s_safe
    f_b = -0.5 * (q / (lam_b + EPS) + 2.0 * target_kl * lam_b)
    lam_dual = jnp.where(f_a >= f_b, lam_a, lam_b)
    nu_dual = jnp.maximum(0.0, lam_dual * c + r) / s_safe

    trpo_case = optim_case >= CASE_TR_FEASIBLE
    lam = jnp.where(trpo_case, lam_trpo, jnp.where(optim_case == CASE_INFEASIBLE_RECOVERY, 0.0, lam_dual))
    nu = jnp.where(trpo_case, 0.0,
                   jnp.where(optim_case == CASE_INFEASIBLE_RECOVERY, jnp.sqrt(2.0 * target_kl / s_safe), nu_dual))

    normal_step = (v - nu * w) / (lam + EPS)
    recovery_step = -nu * w     # steepest decrease of the cost surrogate within the trust region
    direction = jnp.where(optim_case == CASE_INFEASIBLE_RECOVERY, recovery_step, normal_step)
    return CPOStep(direction=direction, optim_case=optim_case, lam=lam, nu=nu,
                   q=q, r=r, s=s, A=A, B=B)


def backtracking_line_search(
        flat_old: jnp.ndarray, direction: jnp.ndarray,
        evaluate: Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
        optim_case: jnp.ndarray, c: jnp.ndarray, target_kl: float,
        surr_r_old: jnp.ndarray, surr_c_old: jnp.ndarray,
        backtrack_coeff: float, backtrack_iters: int,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Shrink the step until it passes CPO's acceptance test (safety-starter-agents rules).

    Step sizes 1, coeff, coeff^2, ... are all evaluated (fixed loop, jittable) and the
    largest one that passes is used. A step passes if, on the batch,
      * KL(pi_old || pi_new) <= target_kl,
      * the reward surrogate did not get worse (only required when x=0 is feasible,
        cases 2-4; in recovery the reward is ignored),
      * the cost surrogate rose by at most max(-c, 0): it may use up the slack below the
        budget, and must not increase when the policy is already over budget.
    If no step passes, the policy is left unchanged.
    """
    fractions = backtrack_coeff ** jnp.arange(backtrack_iters, dtype=jnp.float32)
    kl, surr_r, surr_c = jax.lax.map(lambda f: evaluate(flat_old + f * direction), fractions)
    ok = (
        (kl <= target_kl)
        & jnp.where(optim_case >= CASE_CONSTRAINED, surr_r >= surr_r_old, True)
        & (surr_c - surr_c_old <= jnp.maximum(-c, 0.0))
        & jnp.isfinite(kl) & jnp.isfinite(surr_r) & jnp.isfinite(surr_c)
    )
    accepted = jnp.any(ok)
    idx = jnp.argmax(ok)   # first (largest) passing step
    frac = jnp.where(accepted, fractions[idx], 0.0)
    info = {
        'step_accepted': accepted.astype(jnp.float32),
        'step_fraction': frac,
        'backtrack_steps': jnp.where(accepted, idx, backtrack_iters).astype(jnp.float32),
        'kl': jnp.where(accepted, kl[idx], 0.0),
        'surr_reward_change': jnp.where(accepted, surr_r[idx] - surr_r_old, 0.0),
        'surr_cost_change': jnp.where(accepted, surr_c[idx] - surr_c_old, 0.0),
    }
    return flat_old + frac * direction, info


# ---------------------------------------------------------------------------------------
# The full CPO policy update on one batch
# ---------------------------------------------------------------------------------------

def _advantages(params: PPONetworkParams, normalizer_params: Any, data: types.Transition,
                ppo_network: ppo_networks.PPONetworks, discounting: float,
                reward_scaling: float, gae_lambda: float):
    """Reward and cost GAE (time-major), plus the policy inputs, using the current critics."""
    value_apply = ppo_network.value_network.apply
    cost_value_apply = ppo_network.cost_value_network.apply
    obs = with_shared_latent(ppo_network, params, normalizer_params, data.observation)
    terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    terminal_obs = with_shared_latent(ppo_network, params, normalizer_params, terminal_obs)

    rewards = data.reward * reward_scaling
    costs = data.extras['state_extras']['cost']
    truncation = data.extras['state_extras']['truncation']
    termination = (1 - data.discount) * (1 - truncation)

    _, adv = compute_gae(
        truncation=truncation, termination=termination, rewards=rewards,
        values=value_apply(normalizer_params, params.value, obs),
        bootstrap_value=value_apply(normalizer_params, params.value, terminal_obs),
        lambda_=gae_lambda, discount=discounting)
    _, cadv = compute_gae(
        truncation=truncation, termination=termination, rewards=costs,
        values=cost_value_apply(normalizer_params, params.cost_value, obs),
        bootstrap_value=cost_value_apply(normalizer_params, params.cost_value, terminal_obs),
        lambda_=gae_lambda, discount=discounting)
    return obs, adv, cadv, costs


def cpo_policy_update(
        params: PPONetworkParams,
        normalizer_params: Any,
        data: types.Transition,
        key: jnp.ndarray,
        *,
        ppo_network: ppo_networks.PPONetworks,
        pmap_axis_name: Optional[str],
        per_step_safety_bound: float,
        target_kl: float,
        cg_iters: int,
        cg_damping: float,
        backtrack_coeff: float,
        backtrack_iters: int,
        fvp_subsample: int,
        discounting: float,
        reward_scaling: float,
        gae_lambda: float,
) -> Tuple[Any, Dict[str, jnp.ndarray]]:
    """One CPO policy update on a batch of shape (batch, unroll_length).

    Returns the new policy parameters and scalar metrics (prefixed 'cpo/').
    """
    del key  # deterministic given the batch
    pmean = (lambda x: jax.lax.pmean(x, axis_name=pmap_axis_name)) if pmap_axis_name else (lambda x: x)
    dist = ppo_network.parametric_action_distribution
    policy_apply = ppo_network.policy_network.apply

    # Time-major, as compute_gae expects.
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    obs, adv, cadv, costs = _advantages(params, normalizer_params, data, ppo_network,
                                        discounting, reward_scaling, gae_lambda)
    raw_action = data.extras['policy_extras']['raw_action']

    # Reward advantages are standardised (only the direction matters; the trust region sets
    # the step length). Cost advantages are only centred: their scale is what links the
    # cost surrogate to the constraint value c, so it must be kept.
    adv = (adv - pmean(adv.mean())) / (jnp.sqrt(pmean(jnp.square(adv - pmean(adv.mean())).mean())) + EPS)
    cadv = cadv - pmean(cadv.mean())

    # Constraint value in per-step units: c = J_C(pi_k) - d, with J_C estimated by the
    # batch's mean per-step cost (every step of the batch was played by pi_k).
    mean_cost = pmean(jnp.mean(costs))
    c = mean_cost - per_step_safety_bound

    flat_old, unravel = ravel_pytree(params.policy)
    old_logits = jax.lax.stop_gradient(policy_apply(normalizer_params, params.policy, obs))
    old_dist = dist.create_dist(old_logits)
    old_logp = dist.log_prob(old_logits, raw_action)

    def evaluate(flat, sl=slice(None)):
        """(KL(old || new), reward surrogate, cost surrogate) of flat params on the batch."""
        logits = policy_apply(normalizer_params, unravel(flat),
                              jax.tree_util.tree_map(lambda o: o[:, sl], obs))
        new = dist.create_dist(logits)
        kl = diag_gaussian_kl(old_dist.loc[:, sl], old_dist.scale[:, sl], new.loc, new.scale).mean()
        ratio = jnp.exp(dist.log_prob(logits, raw_action[:, sl]) - old_logp[:, sl])
        return pmean(kl), pmean(jnp.mean(ratio * adv[:, sl])), pmean(jnp.mean(ratio * cadv[:, sl]))

    surr_r = lambda f: evaluate(f)[1]
    surr_c = lambda f: evaluate(f)[2]
    g = pmean(jax.grad(surr_r)(flat_old))
    b = pmean(jax.grad(surr_c)(flat_old))

    # Fisher-vector product: Hessian of the mean KL at the old parameters, times a vector
    # (forward-over-reverse differentiation), plus damping for numerical stability.
    # Optionally on every fvp_subsample-th unroll segment only (cheaper, standard practice).
    fvp_slice = slice(None, None, max(int(fvp_subsample), 1))
    kl_sub = lambda f: evaluate(f, fvp_slice)[0]
    grad_kl = jax.grad(kl_sub)

    def fvp(x):
        return pmean(jax.jvp(grad_kl, (flat_old,), (x,))[1]) + cg_damping * x

    v = conjugate_gradient(fvp, g, cg_iters)          # H^-1 g
    w = conjugate_gradient(fvp, b, cg_iters)          # H^-1 b
    hv, hw = fvp(v), fvp(w)
    # As in safety-starter-agents, q, r, s use H applied to the CG solutions, so they are
    # consistent with the (approximately) inverted H.
    q, r, s = jnp.dot(v, hv), jnp.dot(w, hv), jnp.dot(w, hw)
    b_is_zero = jnp.dot(b, b) <= 1e-8

    step = cpo_direction(v, w, q, r, s, c, target_kl, b_is_zero)
    _, surr_r_old, surr_c_old = evaluate(flat_old)
    flat_new, ls_info = backtracking_line_search(
        flat_old, step.direction, evaluate, step.optim_case, c, target_kl,
        surr_r_old, surr_c_old, backtrack_coeff, backtrack_iters)

    metrics = {
        'cpo/optim_case': step.optim_case.astype(jnp.float32),
        'cpo/c': c,
        'cpo/lambda': step.lam, 'cpo/nu': step.nu,
        'cpo/q': step.q, 'cpo/r': step.r, 'cpo/s': step.s, 'cpo/A': step.A, 'cpo/B': step.B,
        'cpo/grad_norm': jnp.linalg.norm(g), 'cpo/cost_grad_norm': jnp.linalg.norm(b),
        **{f'cpo/{k}': val for k, val in ls_info.items()},
        'mean_cost': mean_cost,
    }
    return unravel(flat_new), metrics


def compute_cpo_value_loss(
        params: PPONetworkParams,
        normalizer_params: Any,
        data: types.Transition,
        rng: jnp.ndarray,
        aux_state: Any = None,
        ppo_network: ppo_networks.PPONetworks = None,
        entropy_cost: float = 0.0,
        discounting: float = 0.99,
        reward_scaling: float = 1.0,
        gae_lambda: float = 0.95,
        clipping_epsilon: float = 0.3,
        normalize_advantage: bool = True,
) -> Tuple[jnp.ndarray, types.Metrics]:
    """Critic loss for CPO: reward-value and cost-value regression only.

    The policy is updated by cpo_policy_update, so it does not appear here: its gradient is
    exactly zero, and Adam leaves parameters with an all-zero gradient history unchanged.
    The loss mirrors the value terms of compute_ppo_lagrange_loss.
    """
    del rng, aux_state, entropy_cost, clipping_epsilon, normalize_advantage
    value_apply = ppo_network.value_network.apply
    cost_value_apply = ppo_network.cost_value_network.apply
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    obs = with_shared_latent(ppo_network, params, normalizer_params, data.observation)
    terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    terminal_obs = with_shared_latent(ppo_network, params, normalizer_params, terminal_obs)

    rewards = data.reward * reward_scaling
    costs = data.extras['state_extras']['cost']
    truncation = data.extras['state_extras']['truncation']
    termination = (1 - data.discount) * (1 - truncation)

    baseline = value_apply(normalizer_params, params.value, obs)
    cost_baseline = cost_value_apply(normalizer_params, params.cost_value, obs)
    vs, _ = compute_gae(
        truncation=truncation, termination=termination, rewards=rewards, values=baseline,
        bootstrap_value=value_apply(normalizer_params, params.value, terminal_obs),
        lambda_=gae_lambda, discount=discounting)
    cost_vs, _ = compute_gae(
        truncation=truncation, termination=termination, rewards=costs, values=cost_baseline,
        bootstrap_value=cost_value_apply(normalizer_params, params.cost_value, terminal_obs),
        lambda_=gae_lambda, discount=discounting)

    v_loss = jnp.mean(jnp.square(vs - baseline)) * 0.5 * 0.5
    cost_v_loss = jnp.mean(jnp.square(cost_vs - cost_baseline)) * 0.5 * 0.5
    total_loss = v_loss + cost_v_loss
    return total_loss, {
        'total_loss': total_loss,
        'v_loss': v_loss,
        'cost_v_loss': cost_v_loss,
        'mean_cost': jnp.mean(costs),
    }
