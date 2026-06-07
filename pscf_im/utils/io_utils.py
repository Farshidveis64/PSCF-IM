"""Small IO helpers for publication-ready output (CSV + LaTeX + figures)."""
from __future__ import annotations

import json
import os
from typing import Mapping

import pandas as pd


def ensure_dir(path: str) -> str:
    """Create ``path`` (a directory) if needed and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def save_table(df: pd.DataFrame, stem: str, *, float_fmt: str = "%.4f",
               caption: str | None = None, label: str | None = None) -> None:
    """Persist a results table as ``stem.csv`` and ``stem.tex`` (booktabs).

    The LaTeX export uses ``booktabs`` rules so the file can be ``\\input{}``
    directly into a manuscript.
    """
    ensure_dir(os.path.dirname(stem) or ".")
    df.to_csv(f"{stem}.csv", index=False)
    try:
        latex = df.to_latex(
            index=False, float_format=lambda x: float_fmt % x,
            escape=False, caption=caption, label=label, position="t",
        )
    except TypeError:  # pandas<1.4 lacks caption/label kwargs
        latex = df.to_latex(index=False, float_format=lambda x: float_fmt % x,
                            escape=False)
    with open(f"{stem}.tex", "w", encoding="utf-8") as fh:
        fh.write(latex)


def save_metrics(metrics: Mapping[str, object], path: str) -> None:
    """Dump a flat metrics dict to JSON (for exact-value reproducibility)."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(metrics), fh, indent=2, sort_keys=True, default=float)
