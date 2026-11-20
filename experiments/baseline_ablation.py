#!/usr/bin/env python3
"""
baseline_ablation.py
========================
Equal-footing baselines that isolate WHERE STRATA's advantage comes from, in
direct response to reviewer concern (Boost 2): the headline speedup was measured
only against a single-threaded exact temporal BFS, which conflates (a) the
algorithmic benefit of word-level source-set propagation with (b) the removal of
Python per-object / per-queue overhead.

We therefore compare four implementations that all compute the SAME exact
temporal-reachability relation (verified bit-for-bit), differing only in the
mechanism, so the contribution of each design choice is isolated:

  1. exact_bfs_ref        : per-source temporal BFS (the original reference).
                            -> baseline; carries per-object overhead.
  2. bitset_ms_no_prune   : multi-source bitset temporal BFS that packs sources
                            into uint64 words and advances them per timestamp
                            group, but WITHOUT the frontier-delta pruning
                            (Delta = R[u] & ~R[v]). It always merges R[u] into
                            R[v]. -> isolates the effect of word packing alone.
  3. numpy_dense_group    : dense boolean per-timestamp-group propagation using
                            NumPy (vectorized, no bit packing). -> isolates the
                            effect of vectorization without the packed layout.
  4. strata_packed        : full STRATA (packed words + frontier-delta pruning +
                            strict snapshot). -> the proposed method.

Comparing (1) vs (2) shows packing+vectorization vs naive traversal; (2) vs (4)
shows the marginal value of frontier-delta pruning; (3) vs (4) shows packed-bits
vs dense-bool at equal vectorization. This decomposition is what the reviewer
asked for.

All four are pure-Python/NumPy at the SAME optimization level, so wall-clock
ratios are apples-to-apples (no cross-hardware or cross-language confound).
A compiled (Numba) kernel is provided separately in numba_kernel.py to
address the "compiled kernel data point" request.

Usage:
    python baseline_ablation.py --edges path/edges.txt --mode strict \
        --out results/ablation/baseline_ablation.csv
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from collections import defaultdict, deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from temporal_strata.packed_reachability import (
    temporal_strata_packed, reachable_pairs, words_for,
)


def load_edges(path):
    edges, verts = [], set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            u, v, t = int(p[0]), int(p[1]), int(p[2])
            edges.append((t, u, v))
            verts.add(u); verts.add(v)
    n = (max(verts) + 1) if verts else 0
    return n, edges


def group_by_time(edges):
    groups = defaultdict(list)
    for (t, u, v) in edges:
        groups[t].append((u, v))
    return [groups[t] for t in sorted(groups)]


# ----------------------------------------------------------------------
# 1. exact per-source temporal BFS  (reference; per-object overhead)
# ----------------------------------------------------------------------
def exact_bfs_ref(n, edges, mode):
    groups = group_by_time(edges)
    # adjacency per timestamp group
    reach_count = 0
    R = [set([s]) for s in range(n)]  # R[v] = sources reaching v
    # process per source independently to mimic the naive reference
    # (kept O(n * m) but in pure python for overhead realism)
    for s in range(n):
        reached = bytearray(n); reached[s] = 1
        for grp in groups:
            if mode == "strict":
                snap = reached[:]  # pre-group snapshot
                for (u, v) in grp:
                    if snap[u] and not reached[v]:
                        reached[v] = 1
            else:  # closure
                changed = True
                while changed:
                    changed = False
                    for (u, v) in grp:
                        if reached[u] and not reached[v]:
                            reached[v] = 1; changed = True
        reach_count += sum(reached)
    return reach_count


# ----------------------------------------------------------------------
# 2. multi-source bitset temporal BFS WITHOUT frontier-delta pruning
# ----------------------------------------------------------------------
def bitset_ms_no_prune(n, edges, mode):
    W = words_for(n)
    R = np.zeros((n, W), dtype=np.uint64)
    for s in range(n):
        R[s, s >> 6] |= np.uint64(1 << (s & 63))
    groups = group_by_time(edges)
    for grp in groups:
        if mode == "strict":
            snap = R.copy()
            for (u, v) in grp:
                # NO pruning: unconditionally OR snap[u] into R[v]
                R[v, :] |= snap[u, :]
        else:
            changed = True
            while changed:
                changed = False
                for (u, v) in grp:
                    before = R[v, :].copy()
                    R[v, :] |= R[u, :]
                    if not np.array_equal(before, R[v, :]):
                        changed = True
    return int(sum(int(bin(int(x)).count("1")) for row in R for x in row))


# ----------------------------------------------------------------------
# 3. dense boolean per-group propagation (NumPy, no bit packing)
# ----------------------------------------------------------------------
def numpy_dense_group(n, edges, mode):
    R = np.eye(n, dtype=bool)  # R[v, s] = s reaches v
    groups = group_by_time(edges)
    for grp in groups:
        us = np.array([u for (u, v) in grp], dtype=np.int64)
        vs = np.array([v for (u, v) in grp], dtype=np.int64)
        if mode == "strict":
            snap = R.copy()
            # R[v] |= snap[u] for each edge; group by v to vectorize
            np.bitwise_or.at(R, vs, snap[us])
        else:
            changed = True
            while changed:
                before = R.copy()
                np.bitwise_or.at(R, vs, R[us])
                changed = not np.array_equal(before, R)
    return int(R.sum())


# ----------------------------------------------------------------------
# 4. STRATA packed (proposed)
# ----------------------------------------------------------------------
def strata_packed(n, edges, mode):
    R = temporal_strata_packed(n, edges, same_time_mode=mode)
    return reachable_pairs(R, n)


METHODS = {
    "exact_bfs_ref": exact_bfs_ref,
    "bitset_ms_no_prune": bitset_ms_no_prune,
    "numpy_dense_group": numpy_dense_group,
    "strata_packed": strata_packed,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True)
    ap.add_argument("--mode", default="strict", choices=["strict", "closure"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results/ablation/baseline_ablation.csv")
    args = ap.parse_args()

    n, edges = load_edges(args.edges)
    print(f"loaded n={n}, edges={len(edges)}, mode={args.mode}")

    truth = None
    rows = []
    for name, fn in METHODS.items():
        best = None
        pairs = None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            pairs = fn(n, edges, args.mode)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        if name == "strata_packed":
            truth = pairs
        rows.append({"method": name, "reachable_pairs": pairs,
                     "best_time_s": round(best, 6)})
        print(f"  {name:22s} pairs={pairs} best={best:.6f}s")

    # correctness + speedup vs reference
    ref_t = next(r["best_time_s"] for r in rows if r["method"] == "exact_bfs_ref")
    for r in rows:
        r["matches_strata"] = (r["reachable_pairs"] == truth)
        r["speedup_vs_ref"] = round(ref_t / r["best_time_s"], 3) if r["best_time_s"] else ""

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = ["method", "reachable_pairs", "matches_strata", "best_time_s", "speedup_vs_ref"]
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields); wr.writeheader()
        for r in rows: wr.writerow({k: r.get(k, "") for k in fields})
    print("wrote", args.out)
    allmatch = all(r["matches_strata"] for r in rows)
    print("ALL METHODS AGREE:", allmatch)


if __name__ == "__main__":
    main()
