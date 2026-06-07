"""PSCF-IM: Path-Specific Counterfactual Fairness for Influence Maximization.

A reproducible reference implementation of the PSCF-IM framework: latent
fair/unfair mediators inferred from cascades, a path-specific bias (PS-Bias-IM)
defined via nested potential outcomes, a minimal-change edit to the diffusion
model, and an edge-level fair Independent-Cascade model whose submodularity
(and the ``(1 - 1/e - eps)`` greedy guarantee) holds by construction.

Two estimator backends are provided:

* ``numpy``  -- a lightweight, dependency-free reference disentangler that runs
  anywhere and is exercised by the unit tests.
* ``torch``  -- the full GNN-VAE with an adversarial (GRL) disentangler and
  weak supervision; this is the backend used for the paper's numbers and is
  selected automatically when PyTorch is importable.

See ``README.md`` for installation and reproduction instructions.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
