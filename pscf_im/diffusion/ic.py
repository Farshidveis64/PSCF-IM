"""Independent Cascade (IC) diffusion under the live-edge / possible-world view.

The spread of a seed set ``A`` is ``sigma(A) = E_phi[|R_phi(A)|]`` where each
edge ``(u, v)`` is "live" independently with probability ``p_uv`` in a random
world ``phi`` and ``R_phi(A)`` is the set reachable from ``A``.  We precompute a
fixed pool of live-edge worlds so that spread, group reach, and greedy gains are
all evaluated against the *same* coupling -- this is what makes the empirical
spread function monotone and submodular in ``A`` (Prop. 1 in the paper).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import networkx as nx
import numpy as np

from ..utils.seeding import new_rng

EdgeProb = dict[tuple[int, int], float]


def uniform_ic_weights(graph: nx.DiGraph) -> EdgeProb:
    """Standard ``p_uv = 1 / in_degree(v)`` weights used by every baseline."""
    indeg = dict(graph.in_degree())
    return {
        (u, v): (1.0 / indeg[v] if indeg[v] > 0 else 0.0)
        for u, v in graph.edges()
    }


@dataclass
class LiveEdgeWorlds:
    """A frozen pool of ``R`` live-edge realisations of a weighted digraph."""
    adj: list[list[list[int]]]   # adj[r][u] -> live out-neighbours of u in world r
    n: int

    @property
    def n_worlds(self) -> int:
        return len(self.adj)


def sample_live_edge_worlds(graph: nx.DiGraph, weights: EdgeProb,
                            n_worlds: int = 100, seed: int = 0) -> LiveEdgeWorlds:
    """Pre-sample ``n_worlds`` live-edge realisations (shared across methods)."""
    rng = new_rng(seed)
    n = graph.number_of_nodes()
    edges = list(graph.edges())
    probs = np.array([weights.get(e, 0.0) for e in edges])
    worlds: list[list[list[int]]] = []
    for _ in range(n_worlds):
        live = rng.random(len(edges)) < probs
        adj: list[list[int]] = [[] for _ in range(n)]
        for keep, (u, v) in zip(live, edges):
            if keep:
                adj[u].append(v)
        worlds.append(adj)
    return LiveEdgeWorlds(adj=worlds, n=n)


def _bfs_reach(adj: list[list[int]], seeds: Iterable[int]) -> np.ndarray:
    """Boolean reachable mask from ``seeds`` in a single live-edge world."""
    n = len(adj)
    reached = np.zeros(n, dtype=bool)
    frontier = list(seeds)
    for s in frontier:
        reached[s] = True
    while frontier:
        nxt: list[int] = []
        for u in frontier:
            for w in adj[u]:
                if not reached[w]:
                    reached[w] = True
                    nxt.append(w)
        frontier = nxt
    return reached


def spread(worlds: LiveEdgeWorlds, seeds: Sequence[int]) -> float:
    """Expected number of activated nodes ``sigma(A)`` over the world pool."""
    if not seeds:
        return 0.0
    total = sum(int(_bfs_reach(adj, seeds).sum()) for adj in worlds.adj)
    return total / worlds.n_worlds


def reach_by_group(worlds: LiveEdgeWorlds, seeds: Sequence[int],
                   groups: np.ndarray, n_groups: int) -> np.ndarray:
    """Expected reach into each group, ``I_G(A, V_i)`` for ``i = 0..n_groups-1``."""
    acc = np.zeros(n_groups)
    for adj in worlds.adj:
        reached = _bfs_reach(adj, seeds)
        for gid in range(n_groups):
            acc[gid] += reached[groups == gid].sum()
    return acc / worlds.n_worlds


def activation_probability(worlds: LiveEdgeWorlds,
                           seeds: Sequence[int]) -> np.ndarray:
    """Per-node activation probability under the seed set (Monte-Carlo)."""
    n = worlds.n
    acc = np.zeros(n)
    for adj in worlds.adj:
        acc += _bfs_reach(adj, seeds)
    return acc / worlds.n_worlds
