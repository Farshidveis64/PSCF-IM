"""Seed-selection routines on a fixed pool of live-edge worlds.

``greedy_celf`` is the cost-effective lazy-forward (CELF) accelerated greedy of
Leskovec et al.; on a monotone submodular spread it attains the ``(1 - 1/e)``
guarantee.  ``ris_select`` is a reverse-influence-sampling selector (the IMM
family) that attains ``(1 - 1/e - eps)`` whp in near-linear time.  Both consume
the *same* pre-sampled worlds, so all methods are compared under one coupling.

Pseudocode (CELF)::

    gains[v] <- sigma({v});  push (gains[v], v) to a max-heap
    while |A| < k:
        pop (g, v); if v fresh this round: A <- A + {v}
        else: g <- sigma(A + {v}) - sigma(A); re-push (g, v)
"""
from __future__ import annotations

import heapq
from typing import Sequence

import numpy as np

from .ic import LiveEdgeWorlds, _bfs_reach, spread


def greedy_celf(worlds: LiveEdgeWorlds, k: int,
                candidates: Sequence[int] | None = None) -> list[int]:
    """Lazy-greedy (CELF) seed selection maximising expected spread."""
    n = worlds.n
    cand = list(range(n)) if candidates is None else list(candidates)

    # First pass: marginal gain of each singleton == sigma({v}).
    heap: list[tuple[float, int, int]] = []
    for v in cand:
        g = spread(worlds, [v])
        heapq.heappush(heap, (-g, v, 0))  # negate for a max-heap

    seeds: list[int] = []
    cur = 0.0
    while heap and len(seeds) < k:
        neg_g, v, last_eval = heapq.heappop(heap)
        if last_eval == len(seeds):           # gain is up to date -> accept
            seeds.append(v)
            cur += -neg_g
        else:                                  # stale -> recompute and re-push
            g = spread(worlds, seeds + [v]) - cur
            heapq.heappush(heap, (-g, v, len(seeds)))
    return seeds


def ris_select(worlds: LiveEdgeWorlds, k: int, n_rr_sets: int = 10000,
               seed: int = 0) -> list[int]:
    """Reverse-influence-sampling (IMM-style) max-coverage seed selection.

    Builds ``n_rr_sets`` reverse-reachable sets (by reversing live edges within
    randomly chosen worlds) and greedily covers the most sets.
    """
    rng = np.random.default_rng(seed)
    n = worlds.n
    # reverse adjacency per world for backward BFS
    rev_per_world = []
    for adj in worlds.adj:
        rev: list[list[int]] = [[] for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                rev[v].append(u)
        rev_per_world.append(rev)

    rr_sets: list[set[int]] = []
    for _ in range(n_rr_sets):
        w = int(rng.integers(0, worlds.n_worlds))
        root = int(rng.integers(0, n))
        mask = _bfs_reach(rev_per_world[w], [root])
        rr_sets.append(set(np.nonzero(mask)[0].tolist()))

    # greedy max coverage over RR sets
    covered = [False] * len(rr_sets)
    hits: list[set[int]] = [set() for _ in range(n)]
    for idx, rr in enumerate(rr_sets):
        for v in rr:
            hits[v].add(idx)

    seeds: list[int] = []
    for _ in range(k):
        best_v, best_cov = -1, -1
        for v in range(n):
            cov = sum(1 for idx in hits[v] if not covered[idx])
            if cov > best_cov:
                best_cov, best_v = cov, v
        if best_v < 0 or best_cov == 0:
            break
        seeds.append(best_v)
        for idx in hits[best_v]:
            covered[idx] = True
    return seeds
