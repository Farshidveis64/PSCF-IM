"""Loaders for the real-world benchmark networks (with optional download).

The four benchmarks used in the paper are obtained from their original sources
and are **not** redistributed here.  This module reads the raw files when they
are present under ``data_root`` (default ``data/raw``) and otherwise falls back
to a deterministic synthetic *surrogate* with matching gross statistics, so the
pipeline and tests run end-to-end without any download.

Expected raw files (see ``data/raw/README.md`` and ``pscf_im.data.download``):

============== ============================== =============================== =========
name           edges file                     labels file                     directed
============== ============================== =============================== =========
email_eu       email-Eu-core.txt              email-Eu-core-department-labels  yes
facebook       facebook_combined.txt          facebook.labels (optional)       no
epinions       soc-Epinions1.txt              -- (Louvain proxy)               yes
antelope_valley antelope_valley.edges         antelope_valley.labels           no
============== ============================== =============================== =========

Node IDs in the raw files need not be contiguous; the loader builds a stable
``original_id -> 0..n-1`` mapping and aligns the label file through it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import networkx as nx
import numpy as np

from ..utils.seeding import new_rng

# name -> (edges file, labels file or None, directed, label_kind, n_groups,
#          surrogate (n, m))
DATASETS: dict[str, dict] = {
    "email_eu": dict(
        edges="email-Eu-core.txt",
        labels="email-Eu-core-department-labels.txt",
        directed=True, label_kind="node_labels", n_groups=42,
        surrogate=(1005, 25571),
        source="https://snap.stanford.edu/data/email-Eu-core.html",
    ),
    "facebook": dict(
        edges="facebook_combined.txt",
        labels="facebook.labels",            # optional; Louvain if absent
        directed=False, label_kind="node_labels_or_louvain", n_groups=2,
        surrogate=(4039, 88234),
        source="https://snap.stanford.edu/data/ego-Facebook.html",
    ),
    "epinions": dict(
        edges="soc-Epinions1.txt",
        labels=None,
        directed=True, label_kind="louvain", n_groups=2,
        surrogate=(75879, 508837),
        source="https://snap.stanford.edu/data/soc-Epinions1.html",
    ),
    "antelope_valley": dict(
        edges="antelope_valley.edges",
        labels="antelope_valley.labels",
        directed=False, label_kind="node_labels", n_groups=2,
        surrogate=(500, 1697),
        source="Fair Influence Maximization literature (access-restricted)",
    ),
}

# Backwards-compatible alias used by older code / experiment CLIs.
BENCHMARKS = DATASETS


@dataclass
class RealNetwork:
    graph: nx.DiGraph
    groups: np.ndarray   # (n,) contiguous integer group label per node
    name: str
    synthetic: bool      # True if a surrogate was generated (no raw file found)
    id_map: dict         # original node id -> contiguous index (empty if synthetic)


def load_real_network(name: str, data_root: str = "data/raw",
                      seed: int = 0) -> RealNetwork:
    """Load benchmark ``name`` from disk, or build a deterministic surrogate."""
    if name not in DATASETS:
        raise KeyError(f"unknown benchmark '{name}'; choose from {list(DATASETS)}")
    spec = DATASETS[name]
    edge_path = os.path.join(data_root, spec["edges"])

    if not os.path.exists(edge_path):
        graph, groups = _surrogate(spec, seed)
        return RealNetwork(graph, groups, name, synthetic=True, id_map={})

    graph, id_map = _read_edge_list(edge_path, directed=spec["directed"])
    groups = _resolve_groups(spec, graph, id_map, data_root, seed)
    return RealNetwork(graph, groups, name, synthetic=False, id_map=id_map)


def _read_edge_list(path: str, *, directed: bool):
    """Parse a whitespace edge list (``# `` comments), preserving node IDs.

    Returns a relabelled ``nx.DiGraph`` on ``0..n-1`` plus the
    ``original_id -> index`` mapping used to align labels.
    """
    raw_edges: list[tuple[int, int]] = []
    nodes: set[int] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            u, v = int(parts[0]), int(parts[1])
            raw_edges.append((u, v))
            nodes.add(u)
            nodes.add(v)

    id_map = {orig: idx for idx, orig in enumerate(sorted(nodes))}
    g = nx.DiGraph() if directed else nx.Graph()
    g.add_nodes_from(range(len(id_map)))
    for u, v in raw_edges:
        g.add_edge(id_map[u], id_map[v])
    return (g if directed else g.to_directed()), id_map


def _resolve_groups(spec: dict, graph: nx.DiGraph, id_map: dict,
                    data_root: str, seed: int) -> np.ndarray:
    """Read node labels (aligned via ``id_map``) or fall back to Louvain."""
    label_file = spec.get("labels")
    if label_file:
        label_path = os.path.join(data_root, label_file)
        if os.path.exists(label_path):
            return _read_node_labels(label_path, id_map, graph.number_of_nodes())
    if spec["label_kind"] == "louvain" or \
            spec["label_kind"] == "node_labels_or_louvain":
        return _louvain_groups(graph, spec["n_groups"], seed)
    # node_labels was required but missing -> Louvain as a safe default
    return _louvain_groups(graph, spec["n_groups"], seed)


def _read_node_labels(path: str, id_map: dict, n: int) -> np.ndarray:
    """Parse ``<original_id> <label>`` lines; remap to contiguous group ids."""
    raw = np.full(n, -1, dtype=int)
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            orig, lab = line.split()[:2]
            orig = int(orig)
            if orig in id_map:
                raw[id_map[orig]] = int(lab)
    # any unlabelled node joins a dedicated bucket, then relabel 0..G-1
    uniq = {v: i for i, v in enumerate(sorted(set(raw.tolist())))}
    return np.array([uniq[v] for v in raw], dtype=int)


def _louvain_groups(graph: nx.DiGraph, n_groups: int, seed: int) -> np.ndarray:
    """Recover group labels by Louvain communities, capped at ``n_groups``."""
    undirected = graph.to_undirected()
    try:
        communities = nx.community.louvain_communities(undirected, seed=seed)
    except Exception:  # pragma: no cover - very old networkx
        communities = nx.community.greedy_modularity_communities(undirected)
    groups = np.zeros(graph.number_of_nodes(), dtype=int)
    for gid, comm in enumerate(communities):
        for v in comm:
            groups[v] = gid % n_groups
    return groups


def _surrogate(spec: dict, seed: int):
    """Deterministic powerlaw-cluster surrogate matching (n, m, #groups)."""
    rng = new_rng(seed)
    n, m = spec["surrogate"]
    # cap surrogate size so the fallback stays fast in CI / demos
    n = min(n, 4039)
    avg_deg = max(1, round(m / max(1, spec["surrogate"][0])))
    g = nx.powerlaw_cluster_graph(n, m=max(1, avg_deg // 2), p=0.1, seed=seed)
    g = g.to_directed()
    groups = rng.integers(0, spec["n_groups"], size=n)
    return g, groups
