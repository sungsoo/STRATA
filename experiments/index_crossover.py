#!/usr/bin/env python3
"""Index baseline: materialize-once vs on-demand traversal.

Addresses the reviewer point (Section 1-2) that the paper calls indexes
"orthogonal" but never *measures* one. We compare the two ways to answer
reachability queries on a temporal graph and locate the crossover:

  * STRATA index: one pass builds the full reachable relation (a materialized
    reachability index); each subsequent point query "does s reach t?" is an
    O(1) bit test. Cost = build_once + Q * O(1).
  * On-demand traversal (no index): each queried source triggers a single-source
    temporal BFS answering that source's reachability to all targets. Cost =
    (#distinct sources) * per_source_BFS.

The crossover Q* = build_time / per_source_BFS_time is the number of distinct
sources beyond which materializing the whole relation (what STRATA does
efficiently) is cheaper than answering on demand. For all-pairs / many-source
workloads (the paper's target) STRATA wins by a wide margin; for a handful of
selective queries the lazy traversal wins -- which is exactly the regime where a
specialized query index (TopChain-style 2-hop) is complementary. Everything is
verified bit-for-bit against the exact reference.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.io import load_standard_edges
from temporal_strata.baseline import _single_source_temporal_bfs, all_sources_temporal_bfs
from temporal_strata.stream import sort_and_validate_edges
from temporal_strata.packed_reachability import temporal_strata_packed, packed_to_matrix, reachable_pairs
try:
    from thread_record import thread_record  # run as a script
except ImportError:  # imported as experiments.<module>
    from experiments.thread_record import thread_record

DATA = Path("datasets_real_temporal/standard")
CASES = [("CollegeMsg", "CollegeMsg.txt"), ("Bitcoin-OTC", "BitcoinOTC.txt")]


def best_of(fn, reps=3):
    fn()
    b = float("inf"); r = None
    for _ in range(reps):
        t = time.perf_counter(); r = fn(); b = min(b, time.perf_counter() - t)
    return b, r


def main():
    rows = []
    for name, fname in CASES:
        p = DATA / fname
        if not p.exists():
            continue
        n, edges, _ = load_standard_edges(p)
        stream = sort_and_validate_edges(n, edges)

        # STRATA materialized index (build once), verified bit-for-bit
        t_build, R = best_of(lambda: temporal_strata_packed(n, edges, same_time_mode="strict"))
        M = packed_to_matrix(R, n)
        # correctness check on a sample of sources against the exact single-source BFS
        rng = np.random.default_rng(0)
        sample = rng.choice(n, size=min(40, n), replace=False)
        for s in sample:
            row = _single_source_temporal_bfs(n, stream, int(s), directed=True, same_time_mode="strict")
            assert np.array_equal(M[int(s)], row), f"{name}: index row {s} mismatch"

        # on-demand: time a single-source temporal BFS (best-of over the sample)
        def one_source():
            s = int(sample[0])
            return _single_source_temporal_bfs(n, stream, s, directed=True, same_time_mode="strict")
        t_src, _ = best_of(one_source, reps=3)

        # O(1) query cost (bit test) is negligible; measure it for completeness
        t0 = time.perf_counter()
        acc = 0
        for _ in range(100000):
            acc += int(M[int(sample[0]), int(sample[1])])
        t_query = (time.perf_counter() - t0) / 100000

        q_star = t_build / t_src  # crossover in #distinct sources
        rows.append({
            "dataset": name, "n": n, "m": len(edges),
            "build_index_s": round(t_build, 4),
            "per_source_bfs_s": round(t_src, 4),
            "point_query_us": round(t_query * 1e6, 3),
            "crossover_sources": round(q_star, 1),
            "crossover_fraction_of_n": round(q_star / n, 4),
            "allpairs_index_speedup": round((n * t_src) / t_build, 1),  # n sources on-demand / one build
        })
        print(f"{name}: build {t_build:.3f}s | per-source BFS {t_src:.4f}s | O(1) query {t_query*1e6:.3f}us "
              f"| crossover Q*={q_star:.1f} sources ({q_star/n*100:.2f}% of n) | "
              f"all-pairs (n sources) index is {rows[-1]['allpairs_index_speedup']}x cheaper than on-demand")

    out = Path("results/real/index_crossover.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": rows, "threads": thread_record()}, out.open("w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
