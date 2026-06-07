"""End-to-end PSCF-IM pipeline: infer -> edit -> fair-world seed.

This ties the modules together (Fig. 2 of the paper):

1. **Factual inference**     -- recover ``(U_f, U_b)`` from the planted network
   using the selected backend (``torch`` if available, else the NumPy
   reference).
2. **Counterfactual edit**   -- fit the edge-level fair model on the fair channel
   only (``U_b`` is severed by construction) -> fair-world edge weights.
3. **Fair-world seeding**    -- sample live-edge worlds under those weights and
   run greedy/RIS, inheriting the ``(1 - 1/e - eps)`` guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from .data.planted import PlantedNetwork
from .diffusion.ic import sample_live_edge_worlds, uniform_ic_weights
from .diffusion.seeding import greedy_celf
from .models.edge_model import FairEdgeModel
from .models.estimator_numpy import EstimatorConfig, ReferenceEstimator


@dataclass
class PSCFIMResult:
    seeds: list[int]
    u_f_hat: np.ndarray
    u_b_hat: np.ndarray
    fair_weights: dict


def _edge_activation_labels(graph: nx.DiGraph, weights, seed: int = 0):
    """One Bernoulli draw per edge as the cascade-derived activation target."""
    rng = np.random.default_rng(seed)
    return {(int(u), int(v)): int(rng.random() < weights[(u, v)])
            for u, v in graph.edges()}


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def build_fair_weights(net: PlantedNetwork, u_f: np.ndarray, seed: int = 0) -> dict:
    """Fit the edge-level fair model on ``u_f`` and return fair-world weights.

    Shared by :func:`run_pscf_im` and the counterfactual PS-Bias-IM scoring so
    the factual and counterfactual fair worlds are built identically.
    """
    biased_weights = uniform_ic_weights(net.graph)
    edge_labels = _edge_activation_labels(net.graph, biased_weights, seed=seed)
    fair_model = FairEdgeModel().fit(net.graph, u_f, edge_labels)
    return fair_model.edge_weights(net.graph, u_f)


def infer_channels(net: PlantedNetwork, backend: str = "auto",
                   est_cfg: EstimatorConfig | None = None):
    """Infer ``(U_f, U_b)`` with the requested backend.

    ``backend='auto'`` uses torch when importable, else the NumPy reference.
    """
    if backend == "auto":
        backend = "torch" if torch_available() else "numpy"

    if backend == "numpy":
        est = ReferenceEstimator(est_cfg or EstimatorConfig(seed=net.config.seed))
        ch = est.fit_infer(net)
        return ch.u_f, ch.u_b, backend

    # torch backend: build features from the planted signals + S/X
    from .models.estimator_torch import build_torch_estimator
    rng = np.random.default_rng(net.config.seed)
    base = np.stack([net.U_f, net.U_b])
    mix = rng.normal(0, 1, size=(2, 3))
    features = np.column_stack([(mix.T @ base).T, net.X, net.S.astype(float)])
    y_target = (net.psbias_node > np.median(net.psbias_node)).astype(float)
    est = build_torch_estimator()
    est.fit(net.graph, features.astype("float32"), net.S.astype("float32"),
            y_target, net.R_b.astype("float32"), net.R_b_mask)
    u_f, u_b = est.infer()
    return u_f, u_b, backend


def run_pscf_im(net: PlantedNetwork, k: int, n_worlds: int = 100,
                backend: str = "auto", est_cfg: EstimatorConfig | None = None,
                seed: int = 0) -> PSCFIMResult:
    """Run the full PSCF-IM pipeline and return seeds + inferred channels."""
    u_f_hat, u_b_hat, _ = infer_channels(net, backend=backend, est_cfg=est_cfg)

    # counterfactual edit: fair-world edge weights from the fair channel only
    fair_weights = build_fair_weights(net, u_f_hat, seed=seed)

    # fair-world seeding
    worlds = sample_live_edge_worlds(net.graph, fair_weights,
                                     n_worlds=n_worlds, seed=seed)
    seeds = greedy_celf(worlds, k)
    return PSCFIMResult(seeds=seeds, u_f_hat=u_f_hat, u_b_hat=u_b_hat,
                        fair_weights=fair_weights)
