"""Baseline seed-selection policies, all run under one shared live-edge pool.

Families (mirroring the paper):

* Fairness-unaware : ``imm`` (RIS), ``greedy`` (plain spread greedy; an
  ``EPIC``-style unconstrained upper bound proxy).
* Statistical FIM  : ``fimm`` (group-balanced greedy), ``total_fairness``
  (suppress all S-dependence -> per-group round-robin greedy).

These are intentionally compact, single-file references -- not the original
authors' code -- sufficient to position PSCF-IM under an identical evaluation.
Heavy learned baselines (PIANO, FAIM-RL, UpLift) are represented by documented
stubs that fall back to a principled greedy variant.
"""
from __future__ import annotations

import numpy as np

from ..diffusion.ic import LiveEdgeWorlds, spread
from ..diffusion.seeding import greedy_celf, ris_select


def imm(worlds: LiveEdgeWorlds, k: int, seed: int = 0, **_) -> list[int]:
    """Reverse-influence-sampling selection (IMM family)."""
    return ris_select(worlds, k, seed=seed)


def greedy(worlds: LiveEdgeWorlds, k: int, **_) -> list[int]:
    """Plain CELF greedy maximising spread (unconstrained reach)."""
    return greedy_celf(worlds, k)


def group_balanced_greedy(worlds: LiveEdgeWorlds, k: int, groups: np.ndarray,
                          n_groups: int, balance: float = 1.0,
                          candidates=None, **_) -> list[int]:
    """Greedy with a per-capita group-coverage penalty (FIMM-style).

    At each step pick the node maximising ``Delta spread - balance * imbalance``
    where ``imbalance`` is the spread in per-capita reach across groups that the
    candidate would induce.  ``candidates`` optionally restricts the search to a
    node subset (e.g. top-degree) to bound cost on large graphs.
    """
    from ..diffusion.ic import _bfs_reach

    cand = list(range(worlds.n)) if candidates is None else list(candidates)
    sizes = np.array([(groups == g).sum() for g in range(n_groups)], float)
    seeds: list[int] = []
    cur = 0.0
    cur_reach = np.zeros(worlds.n, dtype=float)
    for _ in range(k):
        best_v, best_score = -1, -np.inf
        for v in cand:
            if v in seeds:
                continue
            # incremental reach estimate of adding v
            add = np.zeros(worlds.n)
            for adj in worlds.adj:
                add += _bfs_reach(adj, seeds + [v])
            add /= worlds.n_worlds
            gain = add.sum() - cur
            grp_reach = np.array([add[groups == g].sum() for g in range(n_groups)])
            pc = grp_reach / np.maximum(sizes, 1)
            imbalance = pc.max() - pc.min()
            score = gain - balance * imbalance
            if score > best_score:
                best_score, best_v, best_add = score, v, add
        if best_v < 0:
            break
        seeds.append(best_v)
        cur_reach = best_add
        cur = cur_reach.sum()
    return seeds


def total_fairness(worlds: LiveEdgeWorlds, k: int, groups: np.ndarray,
                   n_groups: int, **_) -> list[int]:
    """Over-fairness straw man: round-robin greedy that forces equal coverage."""
    from ..diffusion.ic import _bfs_reach

    seeds: list[int] = []
    per_group_quota = [k // n_groups + (1 if i < k % n_groups else 0)
                       for i in range(n_groups)]
    for gid in range(n_groups):
        members = np.nonzero(groups == gid)[0]
        for _ in range(per_group_quota[gid]):
            best_v, best_gain = -1, -np.inf
            cur = spread(worlds, seeds)
            for v in members:
                if v in seeds:
                    continue
                gain = spread(worlds, seeds + [int(v)]) - cur
                if gain > best_gain:
                    best_gain, best_v = gain, int(v)
            if best_v >= 0:
                seeds.append(best_v)
    return seeds


# Documented stubs for learned baselines -> principled greedy fallbacks.
def piano(worlds, k, **kw):
    """Deep-RL IM (PIANO) stub -> spread greedy fallback."""
    return greedy(worlds, k)


def epic(worlds, k, **kw):
    """Adaptive-greedy upper bound (EPIC) stub -> spread greedy fallback."""
    return greedy(worlds, k)


def faim_rl(worlds, k, groups, n_groups, **kw):
    """FAIM-RL stub -> group-balanced greedy fallback."""
    return group_balanced_greedy(worlds, k, groups, n_groups, balance=1.5)


def uplift(worlds, k, groups, n_groups, **kw):
    """Individually-fair UpLift stub -> strong group-balanced greedy fallback."""
    return group_balanced_greedy(worlds, k, groups, n_groups, balance=2.5)


REGISTRY = {
    "IMM": imm, "PIANO": piano, "EPIC": epic,
    "FIMM": group_balanced_greedy, "UpLift": uplift, "FAIM-RL": faim_rl,
    "Total-Fairness": total_fairness,
}
