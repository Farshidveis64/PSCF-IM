"""Shared evaluation harness used by every experiment script.

The frozen *oracle* defines everything external to a method's seed-selection
objective, so the comparison is free of metric circularity:

* the planted channels ``U_f, U_b`` (and the S-flip counterfactual ``U_b_cf``);
* a **biased** edge model ``p^bias_uv`` whose probability rises with the unfair
  channel ``U_b`` -- fairness-unaware / statistical baselines commit to this
  world (their reach is ``U_b``-skewed);
* a **fair** edge model ``p^fair_uv`` with the unfair channel removed -- the
  world PSCF-IM commits to after the minimal-change edit.

**PS-Bias-IM (counterfactual flip).**  A policy's path-specific bias is the mean
absolute change in per-node activation when ``S`` is flipped *only along the
unfair channel*, evaluated under the policy's own committed diffusion with
common random numbers (so only the unfair counterfactual differs)::

    PS-Bias-IM = mean_v | A_phi[seeds](v ; U_b) - A_phi[seeds](v ; U_b_cf) | .

Baselines commit to ``p^bias`` -> the flip moves edge probabilities -> non-zero.
PSCF-IM commits to ``p^fair`` built from its *inferred* ``U_f`` -> the flip moves
the policy only through the residual ``U_b`` leakage in ``U_f`` (zero under
perfect disentanglement).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data.planted import PlantedNetwork
from .diffusion.ic import (LiveEdgeWorlds, activation_probability,
                           reach_by_group, sample_live_edge_worlds, spread)
from .metrics.fairness import equity, fair_reach


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def oracle_edge_weights(net: PlantedNetwork, *, fair: bool,
                        u_b: np.ndarray | None = None) -> dict:
    """Oracle IC weights; ``fair=False`` lets the unfair channel raise p_uv.

    ``p_uv = sigmoid(-1 + g_f*(U_f^u+U_f^v-1) + [g_b*(U_b^u+U_b^v-1)])``.
    Pass ``u_b`` to score the counterfactual world (e.g. ``net.U_b_cf``).
    """
    cfg = net.config
    uf = net.U_f
    ub = net.U_b if u_b is None else u_b
    g_b = 0.0 if fair else cfg.g_b
    weights = {}
    for u, v in net.graph.edges():
        z = (-1.0 + cfg.g_f * (uf[u] + uf[v] - 1.0)
             + g_b * (ub[u] + ub[v] - 1.0))
        weights[(int(u), int(v))] = float(np.clip(_sigmoid(z), 1e-4, 1 - 1e-4))
    return weights


@dataclass
class ScoringContext:
    biased_worlds: LiveEdgeWorlds      # baselines select & are scored here
    groups: np.ndarray
    n_groups: int
    group_sizes: np.ndarray
    u_f: np.ndarray
    u_b: np.ndarray
    net: PlantedNetwork


def make_scoring_context(net: PlantedNetwork, n_worlds: int = 100,
                         seed: int = 0) -> ScoringContext:
    """Build the frozen oracle biased-world pool for a planted network."""
    biased = oracle_edge_weights(net, fair=False)
    worlds = sample_live_edge_worlds(net.graph, biased, n_worlds=n_worlds,
                                     seed=seed)
    groups = net.S.astype(int)
    n_groups = int(groups.max()) + 1
    sizes = np.array([(groups == g).sum() for g in range(n_groups)], float)
    return ScoringContext(worlds, groups, n_groups, sizes,
                          net.U_f, net.U_b, net)


def fair_worlds(net: PlantedNetwork, fair_weights: dict, n_worlds: int = 100,
                seed: int = 0) -> LiveEdgeWorlds:
    """Live-edge pool under a method's fair edge weights (for scoring PSCF-IM)."""
    return sample_live_edge_worlds(net.graph, fair_weights, n_worlds=n_worlds,
                                   seed=seed)


def psbias_counterfactual(net: PlantedNetwork, seeds, weights_factual: dict,
                          weights_cf: dict, n_worlds: int = 100,
                          seed: int = 0) -> float:
    """Counterfactual-flip PS-Bias-IM under a policy's committed diffusion.

    ``weights_factual`` / ``weights_cf`` are the policy's edge weights computed
    with the factual vs S-flipped unfair channel.  Common random numbers (same
    ``seed``) couple the two world pools so the activation difference isolates
    the unfair counterfactual.
    """
    w_f = sample_live_edge_worlds(net.graph, weights_factual, n_worlds=n_worlds,
                                  seed=seed)
    w_c = sample_live_edge_worlds(net.graph, weights_cf, n_worlds=n_worlds,
                                  seed=seed)
    a_f = activation_probability(w_f, seeds)
    a_c = activation_probability(w_c, seeds)
    return float(np.mean(np.abs(a_f - a_c)))


def evaluate(seeds, worlds: LiveEdgeWorlds, ctx: ScoringContext) -> dict:
    """Score a seed set under the supplied world pool: IS, EQ, FR."""
    act = activation_probability(worlds, seeds)
    grp_reach = reach_by_group(worlds, seeds, ctx.groups, ctx.n_groups)
    return {
        "IS": spread(worlds, seeds),
        "EQ": equity(grp_reach, ctx.group_sizes),
        "FR": fair_reach(act, ctx.u_f, ctx.u_b),
    }
