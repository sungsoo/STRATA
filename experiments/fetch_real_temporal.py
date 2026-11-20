#!/usr/bin/env python3
"""Fetch the standard public temporal-graph suite and standardize it.

WHY THIS EXISTS
---------------
The build/CI sandbox in which this package was assembled blocks the public
dataset hosts: snap.stanford.edu, the TGB/TGB2 object store, and zenodo.org all
return HTTP 403 (``x-deny-reason: host_not_allowed``).  The two real-timestamp
graphs shipped under ``datasets_real_temporal/`` (CollegeMsg, Bitcoin-OTC)
were obtained from GitHub mirrors, which the sandbox does allow, and are enough
to demonstrate the temporal-vs-static gap and exact bit-for-bit correctness at
n up to ~5.9k.

This script lets you reproduce the *large-scale* real-temporal numbers on an
unrestricted machine (e.g. the target MacBook Pro, Apple M5 Max).  It downloads the
canonical SNAP temporal datasets, including one with ~63M edges
(``sx-stackoverflow``) that provides a genuine >=10^7-edge real stream, and
standardizes each to the ``src dst timestamp`` format the benchmarks consume.

USAGE (on a machine with internet access)
-----------------------------------------
    python3 experiments/fetch_real_temporal.py --which small     # <1M edges
    python3 experiments/fetch_real_temporal.py --which large     # incl. 10^7+
    python3 experiments/fetch_real_temporal.py --which all

Then run the same experiments used in the paper, e.g. the >=10^7 throughput
point on real data:

    python3 experiments/numba_kernel.py \
        --edges datasets_real_temporal/standard/sx-stackoverflow.txt \
        --repeats 3 --out results/real/numba_sx_stackoverflow.csv

or the full all-pairs comparison on the medium graphs:

    python3 experiments/real_temporal_benchmark.py

For the very large graphs, all-pairs is infeasible; use source blocking or the
sampled-source / throughput modes described in the paper's tiered execution
policy (Sec. Execution Tiers).
"""
from __future__ import annotations

import argparse
import gzip
import io
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from temporal_strata.io import convert_snap_to_standard

# name -> (url, edges_approx, note)
CATALOG = {
    "small": {
        "CollegeMsg": ("https://snap.stanford.edu/data/CollegeMsg.txt.gz", 59835, "already shipped"),
        "email-Eu-core-temporal": (
            "https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz",
            332334,
            "n=986, ideal for full all-pairs",
        ),
        "sx-mathoverflow": ("https://snap.stanford.edu/data/sx-mathoverflow.txt.gz", 506550, ""),
    },
    "large": {
        "sx-askubuntu": ("https://snap.stanford.edu/data/sx-askubuntu.txt.gz", 964437, ""),
        "sx-superuser": ("https://snap.stanford.edu/data/sx-superuser.txt.gz", 1443339, ""),
        "wiki-talk-temporal": (
            "https://snap.stanford.edu/data/wiki-talk-temporal.txt.gz",
            7833140,
            "~7.8M edges",
        ),
        "sx-stackoverflow": (
            "https://snap.stanford.edu/data/sx-stackoverflow.txt.gz",
            63497050,
            ">=10^7 real temporal stream (the target large-scale point)",
        ),
    },
}


def fetch_one(name, url, out_dir, log=None):
    raw_dir = out_dir / "raw"
    std_dir = out_dir / "standard"
    raw_dir.mkdir(parents=True, exist_ok=True)
    std_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / (name + ".txt")
    if not raw_path.exists():
        print(f"[fetch] {name} <- {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "strata-fetch/1.0"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        if url.endswith(".gz"):
            data = gzip.decompress(data)
        raw_path.write_bytes(data)
    else:
        print(f"[cached] {name}")
    meta = convert_snap_to_standard(raw_path, std_dir / (name + ".txt"), dataset_name=name)
    print(f"  standardized: n={meta['n']} m={meta['temporal_edges']} groups={meta['unique_timestamps']}")
    if log is not None:  # provenance of every fetched file
        import datetime
        import hashlib
        log.append({"dataset": name, "url": url,
                    "fetched": datetime.datetime.now().isoformat(timespec="seconds"),
                    "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                    "standard_sha256": hashlib.sha256((std_dir / (name + ".txt")).read_bytes()).hexdigest(),
                    "meta": meta})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["small", "large", "all"], default="small")
    ap.add_argument("--names", default=None,
                    help="comma list of catalog names (overrides --which)")
    ap.add_argument("--log", default=None, help="write a JSON provenance log (URL, date, sha256)")
    ap.add_argument("--out", default="datasets_real_temporal")
    args = ap.parse_args()
    out_dir = Path(args.out)
    groups = ["small", "large"] if args.which == "all" else [args.which]
    todo = [(nm, spec) for g in ("small", "large") for nm, spec in CATALOG[g].items()]
    if args.names:
        want = [x.strip() for x in args.names.split(",") if x.strip()]
        unknown = set(want) - {nm for nm, _ in todo}
        if unknown:
            raise SystemExit(f"unknown dataset(s): {sorted(unknown)}")
        todo = [(nm, spec) for nm, spec in todo if nm in want]
    else:
        todo = [(nm, spec) for g in groups for nm, spec in CATALOG[g].items()]
    log, failed = [], []
    for name, (url, approx, note) in todo:
        try:
            fetch_one(name, url, out_dir, log)
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            print(f"[error] {name}: {e}\n   (download failed here; the paper records this instead of substituting data)")
    if args.log:
        import json
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log).write_text(json.dumps({"fetched": log, "failed": failed}, indent=2))
        print(f"wrote {args.log}")
    if args.names and failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
