#!/usr/bin/env python3
"""PSCF-IM unified command-line entry point.

A single dispatcher over the unit tests, every experiment script, and the
figure generator, so the whole project is driven from one command::

    python main.py test                       # run the 13 sanity checks
    python main.py certified --n 400 --k 10    # one table (args pass through)
    python main.py figures --n 800             # the disentanglement figure
    python main.py download --all              # fetch public SNAP datasets
    python main.py all                         # tests + all tables + figure
    python main.py all --profile paper         # paper-scale (needs torch)
    python main.py --list                      # show available commands

Each subcommand simply forwards its remaining arguments to the corresponding
``experiments/*.py`` script (or to ``tests/run_tests.py``), so every script's
own ``--help`` / flags remain authoritative and there is no duplicated config.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"

# subcommand -> script to execute
COMMANDS: dict[str, Path] = {
    "test": ROOT / "tests" / "run_tests.py",
    "certified": EXP / "run_certified.py",   # Table III
    "ablation": EXP / "run_ablation.py",      # Table IV
    "recovery": EXP / "run_recovery.py",      # Table V
    "robustness": EXP / "run_robustness.py",  # Table VI
    "realnet": EXP / "run_realnet.py",        # Table II
    "figures": EXP / "make_figures.py",       # Fig. 3
}

# profile -> (n, k, worlds, seeds, cand_cap, backend) for the `all` pipeline
PROFILES = {
    "demo": dict(n=400, k=10, worlds=60, seeds=3, cand_cap=120, backend="auto"),
    "paper": dict(n=1005, k=20, worlds=10000, seeds=5, cand_cap=400,
                  backend="torch"),
}


def _run(script: Path, extra: list[str]) -> int:
    """Execute a project script with the active interpreter; stream its output."""
    cmd = [sys.executable, str(script), *extra]
    print(f">> {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def _run_all(profile: str) -> int:
    """Reproduce tests + every table + the figure for the chosen profile."""
    p = PROFILES[profile]
    common = ["--n", str(p["n"]), "--k", str(p["k"]),
              "--worlds", str(p["worlds"]), "--seeds", str(p["seeds"])]
    steps: list[tuple[str, list[str]]] = [
        ("test", []),
        ("certified", common + ["--cand_cap", str(p["cand_cap"]),
                                "--backend", p["backend"]]),
        ("ablation", common),
        ("recovery", ["--n", str(p["n"]), "--seeds", str(p["seeds"])]),
        ("robustness", common),
        ("realnet", ["--networks", "antelope_valley", "email_eu",
                     "--k", str(p["k"]), "--worlds", str(p["worlds"])]),
        ("figures", ["--n", "800"]),
    ]
    for name, extra in steps:
        rc = _run(COMMANDS[name], extra)
        if rc != 0:
            print(f"!! step '{name}' failed (exit {rc}); aborting.", flush=True)
            return rc
    print(">> all steps complete. See results/", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true",
                        help="list available commands and exit")
    parser.add_argument("command", nargs="?",
                        choices=list(COMMANDS) + ["download", "all"],
                        help="which task to run")
    parser.add_argument("--profile", choices=list(PROFILES), default="demo",
                        help="size profile for 'all' (default: demo)")
    args, extra = parser.parse_known_args(argv)

    if args.list or args.command is None:
        print("Available commands:")
        for name in list(COMMANDS) + ["download", "all"]:
            print(f"  {name}")
        print("\nExample:  python main.py certified --n 400 --k 10 --seeds 3")
        return 0

    if args.command == "all":
        return _run_all(args.profile)
    if args.command == "download":
        cmd = [sys.executable, "-m", "pscf_im.data.download", *extra]
        print(f">> {' '.join(cmd)}", flush=True)
        return subprocess.call(cmd, cwd=str(ROOT))
    return _run(COMMANDS[args.command], extra)


if __name__ == "__main__":
    raise SystemExit(main())
