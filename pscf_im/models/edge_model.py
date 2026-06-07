"""Edge-level fair diffusion model (Eq. 10 of the paper).

The minimal-change edit severs ``U_b -> Y`` by defining the fair-world edge
activation probability purely from the fair propensities::

    p^psf_uv = sigmoid( beta0 + <U_f^u, W U_f^v> )  in (0, 1).

Because ``sigmoid`` maps to the open unit interval, every ``p^psf_uv`` is a
valid Bernoulli parameter with *no* side condition, so the induced process is a
bona-fide IC instance and the spread stays monotone submodular (Prop. 1).  The
unfair channel ``U_b`` never enters this map, which is exactly how the edit
removes only unfair-pathway influence.

``W`` and ``beta0`` are fit by logistic regression to reproduce observed
activations along edges (a faithful, biased-cascade target), so the fair-world
model stays *close* to the data while being structurally fair.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression

from ..diffusion.ic import EdgeProb


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class FairEdgeModel:
    """Logistic edge model on fair propensities; yields valid IC weights."""

    def __init__(self, latent_dim: int = 1, l2: float = 1.0):
        self.latent_dim = latent_dim
        self.l2 = l2
        self.clf: LogisticRegression | None = None

    @staticmethod
    def _pair_features(u_f: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Bilinear-compatible features ``vec(U_f^u outer U_f^v)`` per edge."""
        uf_u = u_f[edges[:, 0]]
        uf_v = u_f[edges[:, 1]]
        if uf_u.ndim == 1:
            uf_u = uf_u[:, None]
            uf_v = uf_v[:, None]
        # outer products flattened -> linear weights == vec(W); bias == beta0.
        return np.einsum("ei,ej->eij", uf_u, uf_v).reshape(len(edges), -1)

    def fit(self, graph: nx.DiGraph, u_f: np.ndarray,
            edge_activation: dict[tuple[int, int], int]) -> "FairEdgeModel":
        """Fit ``(W, beta0)`` to observed per-edge activation labels in {0, 1}."""
        edges = np.array(list(graph.edges()))
        feats = self._pair_features(u_f, edges)
        labels = np.array([edge_activation.get((int(u), int(v)), 0)
                           for u, v in edges])
        if labels.sum() == 0 or labels.sum() == len(labels):
            # degenerate supervision -> fall back to an identity-like map.
            self.clf = None
            self._u_f = u_f
            return self
        self.clf = LogisticRegression(C=1.0 / self.l2, max_iter=500)
        self.clf.fit(feats, labels)
        self._u_f = u_f
        return self

    def edge_weights(self, graph: nx.DiGraph, u_f: np.ndarray) -> EdgeProb:
        """Return ``{(u, v): p^psf_uv}`` for every edge, all strictly in (0, 1)."""
        edges = np.array(list(graph.edges()))
        if self.clf is None:                      # graceful fallback
            uf_u, uf_v = u_f[edges[:, 0]], u_f[edges[:, 1]]
            probs = _sigmoid(4.0 * (uf_u * uf_v) - 1.0)
        else:
            feats = self._pair_features(u_f, edges)
            probs = self.clf.predict_proba(feats)[:, 1]
        probs = np.clip(probs, 1e-4, 1 - 1e-4)    # keep open-interval validity
        return {(int(u), int(v)): float(p) for (u, v), p in zip(edges, probs)}
