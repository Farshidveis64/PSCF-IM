"""Reproducibility helpers: a single entry point to fix every RNG."""
from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 0, *, deterministic_torch: bool = True) -> int:
    """Seed every source of randomness used in the project.

    Seeds the ``random`` module, NumPy, the ``PYTHONHASHSEED`` environment
    variable, and -- if PyTorch is importable -- the CPU/CUDA generators and
    cuDNN determinism flags.

    Parameters
    ----------
    seed:
        The integer seed to apply everywhere.
    deterministic_torch:
        If ``True`` and PyTorch is present, request deterministic cuDNN
        kernels (slower, but bit-reproducible on a fixed device).

    Returns
    -------
    int
        The seed that was applied (echoed for logging convenience).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is optional; only seed it when available.
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - torch may be absent
        pass

    return seed


def new_rng(seed: int) -> np.random.Generator:
    """Return an isolated NumPy ``Generator`` (preferred over global state)."""
    return np.random.default_rng(seed)
