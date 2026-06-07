"""NumPy reference disentangler (the dependency-free estimator backend).

This is a faithful *simplification* of the PSCF-IM-VAE that captures its three
load-bearing mechanisms without a neural network, so it runs anywhere and is
exercised by the unit tests:

1. **Graph encoder**        -> neighbourhood smoothing of node signals.
2. **Weak supervision**     -> ``U_b`` is pinned by ridge-regressing the sparse
                               flags ``R_b`` onto the (smoothed) signals + ``S``.
3. **Adversarial disentangle** -> the fair channel is fit on the residual after
                               *orthogonally projecting out* the recovered
                               ``U_b`` direction (the linear analogue of the GRL
                               discriminator that enforces ``U_f ⟂ U_b | S``).

Two-phase schedule (mirrors Alg. 1): phase 1 fits and *freezes* ``U_b`` from the
flags; phase 2 fits ``U_f`` on the deflated residual.  Ablation switches expose
``use_adversary`` (deflation), ``use_weak_supervision`` (flags), and
``use_fair_exposure`` (neighbour smoothing of the fair channel).

The production backend with the real GNN-VAE lives in ``estimator_torch.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from sklearn.linear_model import Ridge

from ..data.planted import PlantedNetwork


@dataclass
class EstimatorConfig:
    use_weak_supervision: bool = True
    use_adversary: bool = True          # orthogonal deflation of U_b from U_f
    use_fair_exposure: bool = True      # neighbour-smooth the fair channel
    smoothing_hops: int = 2
    ridge_alpha: float = 1.0
    seed: int = 0


@dataclass
class InferredChannels:
    u_f: np.ndarray
    u_b: np.ndarray


def _row_normalised_adjacency(graph: nx.DiGraph) -> np.ndarray:
    """Row-stochastic adjacency (with self-loops) for neighbourhood smoothing."""
    n = graph.number_of_nodes()
    A = np.zeros((n, n))
    for u, v in graph.edges():
        A[v, u] = 1.0          # aggregate from in-neighbours
    A += np.eye(n)
    A /= A.sum(axis=1, keepdims=True)
    return A


def _smooth(A: np.ndarray, x: np.ndarray, hops: int) -> np.ndarray:
    out = x.copy()
    for _ in range(hops):
        out = A @ out
    return out


def _minmax(x: np.ndarray) -> np.ndarray:
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)


def _observed_signals(net: PlantedNetwork, rng: np.random.Generator) -> np.ndarray:
    """Build the per-node observed signal a GNN encoder would ingest.

    Cascades expose a *mixture* of both latent channels (a random mixing of
    ``U_f`` and ``U_b``) plus the covariate ``X`` -- the disentangler must undo
    this mixing using the flags and the orthogonality constraint.
    """
    n = net.n
    mix = rng.normal(0, 1, size=(2, 3))            # 3 observed signal channels
    base = np.stack([net.U_f, net.U_b])            # (2, n)
    signals = (mix.T @ base).T                     # (n, 3)
    signals += 0.05 * rng.normal(size=signals.shape)
    return np.column_stack([signals, net.X])       # (n, 4)


class ReferenceEstimator:
    """Weak-supervised linear disentangler with orthogonal deflation."""

    def __init__(self, cfg: EstimatorConfig | None = None):
        self.cfg = cfg or EstimatorConfig()

    def fit_infer(self, net: PlantedNetwork) -> InferredChannels:
        """Run the two-phase schedule and return inferred ``(U_f, U_b)``."""
        rng = np.random.default_rng(self.cfg.seed)
        A = _row_normalised_adjacency(net.graph)
        H = _observed_signals(net, rng)

        # encoder: optional neighbourhood smoothing.  We deliberately do *not*
        # feed the raw binary S as a feature -- the continuous signals already
        # encode its effect, and a raw binary input would make the recovered
        # channels step-like rather than continuous.
        hops = self.cfg.smoothing_hops if self.cfg.use_fair_exposure else 0
        H_enc = _smooth(A, H, hops)
        feat = H_enc

        # ---- Phase 1: pin U_b from sparse flags, then freeze ----
        if self.cfg.use_weak_supervision and net.R_b_mask.sum() >= 5:
            ridge = Ridge(alpha=self.cfg.ridge_alpha)
            ridge.fit(feat[net.R_b_mask], net.R_b[net.R_b_mask].astype(float))
            u_b_hat = ridge.predict(feat)
        else:
            # no supervision -> a generic component (cannot isolate the channel)
            u_b_hat = H_enc[:, 0]
        u_b_hat = _minmax(u_b_hat)

        # ---- Phase 2: fit U_f on the deflated residual ----
        resid = H_enc.copy()
        if self.cfg.use_adversary:
            # orthogonally project out the frozen U_b direction (disentangle)
            b = (u_b_hat - u_b_hat.mean())
            denom = float(b @ b) + 1e-9
            for j in range(resid.shape[1]):
                col = resid[:, j] - resid[:, j].mean()
                resid[:, j] = col - (col @ b) / denom * b

        # leading covariate-aligned component of the residual -> fair channel
        ridge_f = Ridge(alpha=self.cfg.ridge_alpha)
        ridge_f.fit(resid, net.X)             # X is a clean (non-S) handle on U_f
        u_f_hat = _minmax(ridge_f.predict(resid))

        return InferredChannels(u_f=u_f_hat, u_b=u_b_hat)
