#!/usr/bin/env python3
"""Cross-backend benchmark: Numba core vs PyTorch tensor backend.

Same integrity contract as ``parallel_scaling.py`` (invariants I2/I3/I4):

1. Every backend x device x semantics configuration is verified **bit-identical**
   to the sequential core on every dataset *before any timing*; a single
   mismatch aborts the run.
2. Timing uses warm-up + best-of-3 and records the machine string; numbers are
   never compared across machines.
3. On hosts unfit for meaningful backend timing (this repository's 1-vCPU
   build container, or when the requested accelerator is unavailable), the
   script refuses to emit speedups and writes a PENDING table instead. The
   artifact table for the paper is produced only on the disclosed evaluation
   machine (Apple M5 Max: run with ``python3 experiments/backend_bench.py --device mps``).

Outputs: ``results/real/backend_bench.json`` and
``results/real/table_backend_bench.tex`` (artifact table; not a paper table).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from temporal_strata.packed_reachability import temporal_strata_packed  # noqa: E402

DATASETS = {
    "CollegeMsg": "datasets_real_temporal/standard/CollegeMsg.txt",
    "BitcoinOTC": "datasets_real_temporal/standard/BitcoinOTC.txt",
}
ROOT = os.path.join(os.path.dirname(__file__), "..")


def load(name):
    raw = np.loadtxt(os.path.join(ROOT, DATASETS[name]), dtype=np.int64)
    ids = np.unique(raw[:, :2])
    remap = {int(x): i for i, x in enumerate(ids)}
    edges = [(int(t), remap[int(u)], remap[int(v)]) for u, v, t in raw]
    return len(ids), edges


def best_of(fn, k=3):
    fn()  # warm-up (JIT / kernel launch / cache)
    ts = []
    for _ in range(k):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), ts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="torch device to bench (cpu|mps|cuda)")
    ap.add_argument("--min-cores", type=int, default=4,
                    help="refuse timing table below this core count")
    args = ap.parse_args()

    try:
        from backends.torch_backend import (
            packed_torch_to_numpy,
            resolve_device,
            temporal_strata_torch,
        )
        import backends.torch_backend as tb
        import torch
    except ImportError as e:
        print(f"[backend_bench] torch backend unavailable: {e}")
        print("[backend_bench] nothing to bench; install requirements-optional.txt")
        return 1

    dev = resolve_device(args.device)
    fallback = tb.last_fallback_reason
    cores = os.cpu_count() or 1
    machine = (f"{platform.platform()}; {cores} cores; "
               f"torch {torch.__version__}; device {dev}")
    os.makedirs(os.path.join(ROOT, "results", "real"), exist_ok=True)
    out_json = os.path.join(ROOT, "results", "real", "backend_bench.json")
    out_tex = os.path.join(ROOT, "results", "real", "table_backend_bench.tex")

    record = {"machine": machine, "requested_device": args.device,
              "fallback_reason": fallback, "rows": [], "verified": []}

    # ---- Phase 1 (mandatory): bit-identity of every configuration ----------
    for ds in DATASETS:
        n, edges = load(ds)
        for mode in ("strict", "closure"):
            ref = temporal_strata_packed(n, edges, same_time_mode=mode)
            got = packed_torch_to_numpy(
                temporal_strata_torch(n, edges, same_time_mode=mode,
                                      device=str(dev)))
            if not np.array_equal(ref, got):
                print(f"[backend_bench] BIT MISMATCH {ds}/{mode} on {dev}; aborting")
                return 2
            record["verified"].append(f"{ds}/{mode}/{dev}")
            print(f"[verified bit-identical] {ds:11s} {mode:7s} on {dev}")

    # ---- Phase 2 (conditional): timing ------------------------------------
    refuse = []
    if fallback:
        refuse.append(f"requested device fell back: {fallback}")
    if cores < args.min_cores and dev.type == "cpu":
        refuse.append(f"host has {cores} < {args.min_cores} cores")
    if refuse:
        record["status"] = "PENDING"
        record["refusal"] = refuse
        with open(out_json, "w") as f:
            json.dump(record, f, indent=2)
        with open(out_tex, "w") as f:
            f.write("% PENDING: backend timing refused on this host: "
                    + "; ".join(refuse) + "\n% verified bit-identical: "
                    + ", ".join(record["verified"]) + "\n")
        print("[backend_bench] timing REFUSED (" + "; ".join(refuse)
              + "); PENDING table written, verification recorded")
        return 0

    lines = ["% Emitted by backend_bench.py -- artifact table (machine: "
             + machine + ")",
             "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}llrr@{}}",
             "\\toprule",
             "Dataset & Mode & Numba core (s) & Torch " + dev.type
             + " (s) \\\\", "\\midrule"]
    for ds in DATASETS:
        n, edges = load(ds)
        for mode in ("strict", "closure"):
            t_core, _ = best_of(lambda: temporal_strata_packed(
                n, edges, same_time_mode=mode))
            t_torch, _ = best_of(lambda: temporal_strata_torch(
                n, edges, same_time_mode=mode, device=str(dev)))
            record["rows"].append({"dataset": ds, "mode": mode,
                                   "numba_s": t_core, "torch_s": t_torch})
            lines.append(f"{ds} & {mode} & {t_core:.3f} & {t_torch:.3f} \\\\")
            print(f"[timed] {ds}/{mode}: core {t_core:.3f}s torch {t_torch:.3f}s")
    lines += ["\\bottomrule", "\\end{tabular*}"]
    record["status"] = "MEASURED"
    with open(out_json, "w") as f:
        json.dump(record, f, indent=2)
    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[backend_bench] wrote {out_json} and {out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
