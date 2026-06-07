"""Fairness, recovery, and reach metrics.

* ``equity`` (EQ)            -- min/max per-capita coverage ratio across groups.
* ``fair_reach`` (FR)        -- share of reach carried by the fair channel.
* ``psbias_policy``          -- certified PS-Bias-IM of a seeding policy, scored
                                by the *frozen oracle* on a planted network.
* ``recovery_error``         -- gap between inferred and planted latent channels.
* ``audit_proxy``            -- the uncertified PS* used on real networks: the
                                counterfactual shift in reach when ``S`` is
                                flipped under a single shared IC model.
"""
from __future__ import annotations

import numpy as np

from ..diffusion.ic import LiveEdgeWorlds, activation_probability


def equity(reach_per_group: np.ndarray, group_sizes: np.ndarray) -> float:
    """EQ = min_i,j (xi_i / xi_j) with xi_i = I_G(A, V_i) / |V_i|."""
    per_capita = reach_per_group / np.maximum(group_sizes, 1)
    per_capita = per_capita[per_capita > 0]
    if per_capita.size == 0:
        return 0.0
    return float(per_capita.min() / per_capita.max())


def fair_reach(activation: np.ndarray, u_f: np.ndarray, u_b: np.ndarray,
               eps: float = 1e-9) -> float:
    """FR: fraction of total activated 'mass' attributable to the fair channel.

    Each activated node's reach is split between channels in proportion to the
    planted ``U_f`` vs ``U_b`` propensities.
    """
    share_f = u_f / (u_f + u_b + eps)
    total = float((activation).sum())
    if total <= 0:
        return 0.0
    return float((activation * share_f).sum() / total)


def psbias_policy(seeds, worlds: LiveEdgeWorlds, psbias_node: np.ndarray) -> float:
    """Certified PS-Bias-IM of a policy: activation-weighted oracle node bias.

    The policy reaches nodes with probabilities ``a_v``; its path-specific bias
    is ``sum_v a_v * psbias_node(v) / sum_v a_v`` -- the average unfair-channel
    sensitivity of the population the policy actually activates.
    """
    activation = activation_probability(worlds, seeds)
    denom = float(activation.sum())
    if denom <= 0:
        return 0.0
    return float((activation * psbias_node).sum() / denom)


def recovery_error(u_hat: np.ndarray, u_true: np.ndarray) -> float:
    """Mean absolute gap between an inferred channel and its planted counterpart.

    Both vectors are min-max normalised first so the error is scale-free and
    comparable across channel strengths.
    """
    def _norm(x: np.ndarray) -> np.ndarray:
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)

    return float(np.mean(np.abs(_norm(u_hat) - _norm(u_true))))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, robust to constant inputs."""
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def audit_proxy(reach_factual: np.ndarray, reach_counterfactual: np.ndarray,
                group_sizes: np.ndarray) -> float:
    """PS*: mean absolute per-capita reach shift when ``S`` is flipped.

    Computed under a single fixed, method-agnostic IC model applied to every
    method.  Conflates fair with unfair S-dependence, hence *suggestive*.
    """
    pc_f = reach_factual / np.maximum(group_sizes, 1)
    pc_cf = reach_counterfactual / np.maximum(group_sizes, 1)
    return float(np.mean(np.abs(pc_f - pc_cf)))
