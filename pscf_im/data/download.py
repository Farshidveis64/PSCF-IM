#!/usr/bin/env python3
"""Download the *public* SNAP benchmark datasets into ``data/raw/``.

Fetches and decompresses the publicly available networks used in the paper.
Requires network access (it will not run in an offline sandbox).  The Antelope
Valley network is access-restricted and is **not** downloaded here -- obtain it
separately and place ``antelope_valley.edges`` / ``.labels`` under ``data/raw``.

Source pages (verified against SNAP):
    email_eu  -> https://snap.stanford.edu/data/email-Eu-core.html
    facebook  -> https://snap.stanford.edu/data/ego-Facebook.html
    epinions  -> https://snap.stanford.edu/data/soc-Epinions1.html

Usage::

    python -m pscf_im.data.download --all
    python -m pscf_im.data.download --datasets email_eu epinions
    python -m pscf_im.data.download --all --data_root data/raw
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import urllib.request

BASE = "https://snap.stanford.edu/data"

# dataset -> list of (remote .gz url, local decompressed filename)
DOWNLOADS: dict[str, list[tuple[str, str]]] = {
    "email_eu": [
        (f"{BASE}/email-Eu-core.txt.gz", "email-Eu-core.txt"),
        (f"{BASE}/email-Eu-core-department-labels.txt.gz",
         "email-Eu-core-department-labels.txt"),
    ],
    "facebook": [
        (f"{BASE}/facebook_combined.txt.gz", "facebook_combined.txt"),
    ],
    "epinions": [
        (f"{BASE}/soc-Epinions1.txt.gz", "soc-Epinions1.txt"),
    ],
}


def _fetch_and_gunzip(url: str, dest_txt: str) -> None:
    """Download a ``.gz`` and write the decompressed text to ``dest_txt``."""
    if os.path.exists(dest_txt):
        print(f"  [skip] {os.path.basename(dest_txt)} already present")
        return
    gz_path = dest_txt + ".gz"
    print(f"  [get ] {url}")
    urllib.request.urlretrieve(url, gz_path)  # noqa: S310 (trusted SNAP host)
    with gzip.open(gz_path, "rb") as fin, open(dest_txt, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    os.remove(gz_path)
    print(f"  [ok  ] -> {dest_txt}")


def download(datasets: list[str], data_root: str) -> None:
    os.makedirs(data_root, exist_ok=True)
    for name in datasets:
        if name not in DOWNLOADS:
            print(f"[warn] '{name}' is not publicly downloadable here; skipping.")
            continue
        print(f"[{name}]")
        for url, fname in DOWNLOADS[name]:
            _fetch_and_gunzip(url, os.path.join(data_root, fname))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=list(DOWNLOADS),
                    help="which public datasets to fetch")
    ap.add_argument("--all", action="store_true", help="fetch all public datasets")
    ap.add_argument("--data_root", default="data/raw")
    args = ap.parse_args()

    targets = list(DOWNLOADS) if args.all else (args.datasets or [])
    if not targets:
        ap.error("specify --all or --datasets ...")
    download(targets, args.data_root)
    print("\nDone. Antelope Valley is access-restricted and must be added "
          "manually (see data/raw/README.md).")


if __name__ == "__main__":
    main()
