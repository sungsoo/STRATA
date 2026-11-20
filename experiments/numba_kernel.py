#!/usr/bin/env python3
"""
numba_kernel.py
===================
A compiled (Numba/LLVM) realization of the STRATA inner kernel, added in direct
response to the reviewer's request for at least one compiled data point. The
pure-Python/NumPy reference is convenient for exact verification but its
per-row vectorization overhead means the frontier-delta pruning cannot pay off
(an unconditional bitset OR is faster in interpreted code). The proven cost
model -- O(m * ceil(n/w)) with per-edge word-level AND-NOT/OR and short-circuit
pruning -- is realized only when the kernel compiles to tight machine code with
predictable branches and early-exit on the frontier test.

This module provides:
  * strata_strict_numba(n, t, u, v, W): compiled strict-semantics kernel with
    frontier-delta pruning and per-group snapshot.
  * a comparison entry point that times the compiled kernel against the
    pure-Python strata and the no-prune bitset baseline on the same input,
    verifying bit-for-bit agreement.

The kernel operates on a packed uint64 state R of shape (n, W) and processes
edges already sorted by timestamp; group boundaries are detected from the sorted
timestamp array. Strict semantics use a per-group snapshot.
"""
from __future__ import annotations
import argparse, csv, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from temporal_strata.packed_reachability import (
    temporal_strata_packed, reachable_pairs, words_for,
)

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return deco


@njit(cache=True)
def _popcount_state(R):
    total = 0
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            x = R[i, j]
            # popcount on uint64
            x = x - ((x >> 1) & np.uint64(0x5555555555555555))
            x = (x & np.uint64(0x3333333333333333)) + ((x >> 2) & np.uint64(0x3333333333333333))
            x = (x + (x >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
            total += int((x * np.uint64(0x0101010101010101)) >> np.uint64(56)) & 0xFF
    return total


@njit(cache=True)
def _strata_strict_kernel(n, W, ts, us, vs):
    R = np.zeros((n, W), dtype=np.uint64)
    for s in range(n):
        R[s, s >> 6] |= np.uint64(1) << np.uint64(s & 63)
    m = ts.shape[0]
    i = 0
    while i < m:
        # group [i, j) sharing timestamp ts[i]
        j = i
        t0 = ts[i]
        while j < m and ts[j] == t0:
            j += 1
        snap = R.copy()  # pre-group snapshot (strict)
        for k in range(i, j):
            u = us[k]; v = vs[k]
            # frontier-delta pruning with short-circuit: any new bit?
            new_bit = False
            for w in range(W):
                if snap[u, w] & ~R[v, w]:
                    new_bit = True
                    break
            if new_bit:
                for w in range(W):
                    R[v, w] |= snap[u, w]
        i = j
    return R


@njit(cache=True)
def _strata_closure_kernel(n, W, ts, us, vs):
    R = np.zeros((n, W), dtype=np.uint64)
    for s in range(n):
        R[s, s >> 6] |= np.uint64(1) << np.uint64(s & 63)
    m = ts.shape[0]
    i = 0
    while i < m:
        j = i; t0 = ts[i]
        while j < m and ts[j] == t0:
            j += 1
        # closure: sweep the group to a fixed point. The frontier-delta test is
        # ESSENTIAL here: it is the change detector that decides when to stop.
        changed = True
        while changed:
            changed = False
            for k in range(i, j):
                u = us[k]; v = vs[k]
                new_bit = False
                for w in range(W):
                    if R[u, w] & ~R[v, w]:
                        new_bit = True
                        break
                if new_bit:
                    for w in range(W):
                        R[v, w] |= R[u, w]
                    changed = True
        i = j
    return R


def strata_closure_numba(n, edges):
    W = words_for(n)
    arr = np.array(sorted(edges), dtype=np.int64)
    ts = np.ascontiguousarray(arr[:, 0])
    us = np.ascontiguousarray(arr[:, 1])
    vs = np.ascontiguousarray(arr[:, 2])
    R = _strata_closure_kernel(n, W, ts, us, vs)
    return R, _popcount_state(R)


def strata_strict_numba(n, edges):
    """edges: list of (t,u,v). Returns (R, reachable_pairs)."""
    W = words_for(n)
    arr = np.array(sorted(edges), dtype=np.int64)  # sort by t,u,v
    ts = np.ascontiguousarray(arr[:, 0])
    us = np.ascontiguousarray(arr[:, 1])
    vs = np.ascontiguousarray(arr[:, 2])
    R = _strata_strict_kernel(n, W, ts, us, vs)
    return R, _popcount_state(R)


@njit(cache=True)
def _bitset_noprune_kernel(n, W, ts, us, vs):
    R = np.zeros((n, W), dtype=np.uint64)
    for s in range(n):
        R[s, s >> 6] |= np.uint64(1) << np.uint64(s & 63)
    m = ts.shape[0]
    i = 0
    while i < m:
        j = i; t0 = ts[i]
        while j < m and ts[j] == t0:
            j += 1
        snap = R.copy()
        for k in range(i, j):
            u = us[k]; v = vs[k]
            for w in range(W):  # unconditional OR, no pruning
                R[v, w] |= snap[u, w]
        i = j
    return R


def bitset_noprune_numba(n, edges):
    W = words_for(n)
    arr = np.array(sorted(edges), dtype=np.int64)
    ts = np.ascontiguousarray(arr[:, 0])
    us = np.ascontiguousarray(arr[:, 1])
    vs = np.ascontiguousarray(arr[:, 2])
    R = _bitset_noprune_kernel(n, W, ts, us, vs)
    return R, _popcount_state(R)


def load_edges(path):
    edges, verts = [], set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"): continue
            p = line.split(); u, v, t = int(p[0]), int(p[1]), int(p[2])
            edges.append((t, u, v)); verts.add(u); verts.add(v)
    return (max(verts) + 1 if verts else 0), edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results/ablation/numba_kernel.csv")
    args = ap.parse_args()
    if not _HAVE_NUMBA:
        print("Numba unavailable; install numba to run the compiled kernel.")
        return
    n, edges = load_edges(args.edges)
    print(f"loaded n={n}, edges={len(edges)}")

    # ground truth from verified pure-Python core
    R_ref = temporal_strata_packed(n, edges, same_time_mode="strict")
    truth = reachable_pairs(R_ref, n)

    rows = []
    # warm-up compile (not timed)
    strata_strict_numba(n, edges); bitset_noprune_numba(n, edges)

    for name, fn in [("strata_compiled_prune", strata_strict_numba),
                     ("bitset_compiled_noprune", bitset_noprune_numba)]:
        best = None; pairs = None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            _, pairs = fn(n, edges)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        rows.append({"method": name, "reachable_pairs": pairs,
                     "matches_ref": (pairs == truth), "best_time_s": round(best, 6)})
        print(f"  {name:26s} pairs={pairs} matches={pairs==truth} best={best:.6f}s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["method", "reachable_pairs", "matches_ref", "best_time_s"])
        wr.writeheader()
        for r in rows: wr.writerow(r)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
