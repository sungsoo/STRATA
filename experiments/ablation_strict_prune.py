#!/usr/bin/env python3
"""Strict-mode pruning ablation: the frontier-delta test is pure overhead
in the interpreted NumPy core.

Background
----------
The strict merge R[v] |= R[u] is unconditional in the shipped core. A reviewer
noted (correctly) that the earlier claim "scanned word-by-word with a
short-circuit at the first nonzero word" was false for the NumPy path: gating the
merge with `np.any(R[u] & ~R[v])` allocates and scans a full W-word temporary
that does NOT short-circuit, so testing before merging is strictly more work than
merging. This script measures the effect on the real-timestamp streams by
toggling the internal `_STRICT_PRUNE` flag; both settings are bit-identical.

Usage:
    python3 experiments/ablation_strict_prune.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import temporal_strata.packed_reachability as pr
from temporal_strata.io import load_standard_edges
try:
    from thread_record import thread_record  # run as a script
except ImportError:  # imported as experiments.<module>
    from experiments.thread_record import thread_record

DATA = Path("datasets_real_temporal/standard")
CASES = [("CollegeMsg", "CollegeMsg.txt"), ("Bitcoin-OTC", "BitcoinOTC.txt"),
         ("Bitcoin-OTC(+)", "BitcoinOTCpos.txt")]


def best_of(fn, reps=3):
    fn()
    b = float("inf")
    r = None
    for _ in range(reps):
        t = time.perf_counter()
        r = fn()
        b = min(b, time.perf_counter() - t)
    return b, r


def main():
    rows = []
    for name, fname in CASES:
        p = DATA / fname
        if not p.exists():
            continue
        n, edges, _ = load_standard_edges(p)
        pr._STRICT_PRUNE = False
        t_np, R_np = best_of(lambda: pr.temporal_strata_packed(n, edges, same_time_mode="strict"))
        pr._STRICT_PRUNE = True
        t_pr, R_pr = best_of(lambda: pr.temporal_strata_packed(n, edges, same_time_mode="strict"))
        pr._STRICT_PRUNE = False
        ok = bool(np.array_equal(R_np, R_pr))
        assert ok, f"{name}: prune vs no-prune bit mismatch"
        rows.append({
            "dataset": name, "n": n, "m": len(edges),
            "no_prune_s": round(t_np, 4), "prune_s": round(t_pr, 4),
            "prune_over_noprune_x": round(t_pr / t_np, 3),
            "bit_identical": ok,
            "pairs": pr.reachable_pairs(R_np, n),
        })
        print(f"{name}: no-prune {t_np:.4f}s | prune {t_pr:.4f}s | "
              f"prune is {t_pr/t_np:.2f}x slower | bit-identical={ok}")
    out = Path("results/real/ablation_strict_prune.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": rows, "threads": thread_record()}, out.open("w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
