"""Planted-mechanism networks with a known structural causal model (SCM).

This generator is the *only* setting in which path-specific bias can be scored
against ground truth, because the fair and unfair channels are produced by an
SCM we control.  The SCM is:

    S  ~ Bernoulli(0.5)                      (sensitive attribute / group)
    X  ~ Normal(0, 1)                        (non-sensitive covariate)
    U_f = sigmoid( w_f * (a_f X + b_f S_centered) + eps_f )   (fair channel)
    U_b = sigmoid( w_b * S_centered          + eps_b )        (unfair channel)
    Y-propensity = sigmoid( g_f U_f + g_b U_b )               (biased outcome)

with ``S_centered = 2 S - 1``.  ``U_f`` legitimately depends on ``S`` only
through genuine homophily mixed with the covariate ``X``; ``U_b`` depends on
``S`` *directly* and is the channel a fair method must neutralise.

The closed-form oracle path-specific bias for a node is the change in its
outcome propensity when ``S`` is flipped *only along the unfair channel*::

    psbias(v) = | f_Y(U_f(s), U_b(s')) - f_Y(U_f(s), U_b(s)) | .

Weak supervision ``R_b`` flags a fraction ``c_r`` of nodes whose unfair channel
is high; flags may be corrupted (``flag_noise``) or group-skewed
(``flag_skew``) to study robustness.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from ..utils.seeding import new_rng


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class PlantedConfig:
    """Configuration for :func:`generate_planted_network`."""
    n: int = 1005
    n_communities: int = 5
    w_f: float = 0.8           # fair-channel strength
    w_b: float = 0.4           # unfair-channel strength
    c_r: float = 0.3           # flagged fraction (weak supervision coverage)
    p_in: float = 0.05         # intra-community edge prob (SBM)
    p_out: float = 0.005       # inter-community edge prob (SBM)
    g_f: float = 2.0           # outcome weight on the fair channel
    g_b: float = 2.0           # outcome weight on the unfair channel
    noise_f: float = 0.15
    noise_b: float = 0.15
    flag_noise: float = 0.0    # symmetric mislabelling rate of R_b
    flag_skew: float = 0.0     # extra under-flagging in group S=0 (0..1)
    multihop_rho: float = 0.0  # share of unfair effect routed multi-hop (Panel C)
    seed: int = 0


@dataclass
class PlantedNetwork:
    """A planted network plus its ground-truth latent channels and oracle bias."""
    graph: nx.DiGraph
    S: np.ndarray                # (n,) sensitive attribute in {0, 1}
    X: np.ndarray                # (n,) non-sensitive covariate
    U_f: np.ndarray              # (n,) planted fair propensity in (0, 1)
    U_b: np.ndarray              # (n,) planted unfair propensity in (0, 1)
    U_b_cf: np.ndarray           # (n,) unfair channel under the S-flip counterfactual
    R_b: np.ndarray              # (n,) observed unfair flags in {0, 1}
    R_b_mask: np.ndarray         # (n,) bool: which nodes carry a flag at all
    psbias_node: np.ndarray      # (n,) closed-form oracle path-specific bias
    config: PlantedConfig = field(repr=False)

    @property
    def n(self) -> int:
        return self.graph.number_of_nodes()


def _outcome_propensity(u_f: np.ndarray, u_b: np.ndarray,
                        cfg: PlantedConfig) -> np.ndarray:
    """Oracle outcome propensity ``f_Y(U_f, U_b)`` (the *biased* world)."""
    return _sigmoid(cfg.g_f * (u_f - 0.5) + cfg.g_b * (u_b - 0.5))


def generate_planted_network(cfg: PlantedConfig) -> PlantedNetwork:
    """Sample a planted SBM with a known fair/unfair SCM and oracle PS-bias."""
    rng = new_rng(cfg.seed)

    # --- community structure via a directed SBM ---
    sizes = [cfg.n // cfg.n_communities] * cfg.n_communities
    sizes[-1] += cfg.n - sum(sizes)
    probs = np.full((cfg.n_communities, cfg.n_communities), cfg.p_out)
    np.fill_diagonal(probs, cfg.p_in)
    g_undirected = nx.stochastic_block_model(
        sizes, probs.tolist(), seed=int(cfg.seed), directed=False,
    )
    graph = g_undirected.to_directed()
    n = graph.number_of_nodes()

    # --- sensitive attribute aligned (softly) with communities, plus covariate ---
    block_of = np.array([graph.nodes[v]["block"] for v in range(n)])
    base_group = (block_of % 2).astype(float)
    flip = rng.random(n) < 0.25                      # soft, not deterministic
    S = np.where(flip, 1.0 - base_group, base_group).astype(int)
    X = rng.normal(0.0, 1.0, size=n)
    s_centered = 2 * S - 1

    # --- planted latent channels ---
    # U_f is driven mostly by the non-sensitive covariate X with only *mild*
    # homophily on S (legitimate structure); U_b is driven by S *and* a
    # continuous unfair-susceptibility latent Z_b, so the unfair channel varies
    # continuously while still depending on the sensitive attribute.
    eps_f = rng.normal(0.0, cfg.noise_f, size=n)
    eps_b = rng.normal(0.0, cfg.noise_b, size=n)
    Z_b = rng.normal(0.0, 1.0, size=n)
    a_f, b_f = 1.4, 0.25
    s_coef_b, z_coef_b = 1.6, 1.3
    U_f = _sigmoid(cfg.w_f * (a_f * X + b_f * s_centered) + eps_f)
    U_b = _sigmoid(cfg.w_b * (s_coef_b * s_centered + z_coef_b * Z_b) + eps_b)

    # --- closed-form oracle path-specific bias (flip S only along U_b) ---
    s_prime = -s_centered
    U_b_cf = _sigmoid(cfg.w_b * (s_coef_b * s_prime + z_coef_b * Z_b) + eps_b)
    y_fact = _outcome_propensity(U_f, U_b, cfg)
    y_cf = _outcome_propensity(U_f, U_b_cf, cfg)
    psbias_node = np.abs(y_cf - y_fact)

    # --- weak supervision flags R_b (sparse, possibly noisy / group-skewed) ---
    R_b, R_b_mask = _make_flags(U_b, S, cfg, rng)

    return PlantedNetwork(
        graph=graph, S=S, X=X, U_f=U_f, U_b=U_b, U_b_cf=U_b_cf,
        R_b=R_b, R_b_mask=R_b_mask, psbias_node=psbias_node, config=cfg,
    )


def make_unfair_counterfactual(net: PlantedNetwork) -> PlantedNetwork:
    """Return a copy of ``net`` with the unfair channel set to its S-flip value.

    Everything else (graph, ``S``, ``X``, ``U_f``, flags) is held fixed, so an
    estimator re-run on this network differs only through the unfair channel --
    exactly the path-specific counterfactual used to score PS-Bias-IM.
    """
    import dataclasses

    return dataclasses.replace(net, U_b=net.U_b_cf, U_b_cf=net.U_b)


def _make_flags(U_b: np.ndarray, S: np.ndarray, cfg: PlantedConfig,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample sparse unfair-exposure flags with optional noise and group skew.

    A node is *eligible* to be flagged with base rate ``c_r``; group ``S == 0``
    is additionally under-monitored by a factor ``(1 - flag_skew)``.  A flag's
    value is ``1`` when the true ``U_b`` is in the top quantile, then corrupted
    with symmetric probability ``flag_noise``.
    """
    n = U_b.shape[0]
    base = np.full(n, cfg.c_r)
    base = np.where(S == 0, base * (1.0 - cfg.flag_skew), base)
    R_b_mask = rng.random(n) < base

    thresh = np.quantile(U_b, 0.65)
    flag_value = (U_b >= thresh).astype(int)
    if cfg.flag_noise > 0:
        corrupt = rng.random(n) < cfg.flag_noise
        flag_value = np.where(corrupt, 1 - flag_value, flag_value)

    R_b = np.where(R_b_mask, flag_value, 0).astype(int)
    return R_b, R_b_mask
