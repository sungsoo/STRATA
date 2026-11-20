"""ablation_touched_row.py -- strict-mode snapshot: touched-row vs full-copy.

Reviewer Q1: the strict-mode per-group snapshot is described as "touched-row"
copying, but the data structures and overhead were not specified or measured.

This script implements BOTH strict-mode snapshot strategies explicitly and
measures them on the same streams, verifying bit-identical output:

  (A) full-copy   : before = R.copy()             -- O(n W) words copied / group
  (B) touched-row : snapshot only rows that a group WRITES to, keyed by a
                    per-group dirty set (a Python set of target rows), restoring
                    the read-view lazily. Copied words / group = |dirty_t| W.

We report, across streams with a growing number of timestamp groups T (so the
per-group work shrinks and snapshot overhead becomes relatively larger):
  - words copied per group (mean) for each strategy,
  - total snapshot words copied,
  - wall time of the snapshot component vs the per-edge kernel,
  - correctness: pairs and full-bitset equality against the verified core.

The dirty set is a plain hash set of touched target-row indices; its maintenance
cost is one set-insert per written row per group. The measured trade is: full
copy touches n*W words regardless of activity, touched-row touches
|dirty_t|*W + O(|dirty_t|) set overhead, a net win exactly when groups touch far
fewer than n rows -- the common case for many short timestamp groups.
"""
import sys, os, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.io import load_standard_edges
from temporal_strata.packed_reachability import (
    temporal_strata_packed, reachable_pairs, words_for)
from itertools import groupby


def _identity(n, W):
    R = np.zeros((n, W), dtype=np.uint64)
    for s in range(n):
        R[s, s >> 6] |= np.uint64(1) << np.uint64(s & 63)
    return R


def strict_full_copy(n, stream):
    """Strict mode with a full O(nW) snapshot per group. Returns (R, stats)."""
    W = words_for(n)
    R = _identity(n, W)
    copied_words = 0
    n_groups = 0
    t_snap = 0.0
    t_kernel = 0.0
    for _, git in groupby(stream, key=lambda e: e.t):
        group = list(git)
        n_groups += 1
        s0 = time.perf_counter()
        before = R.copy()
        t_snap += time.perf_counter() - s0
        copied_words += R.shape[0] * R.shape[1]
        s1 = time.perf_counter()
        for e in group:
            cand = before[e.u]
            nz = cand & ~R[e.v]
            if nz.any():
                R[e.v] |= cand
        t_kernel += time.perf_counter() - s1
    return R, dict(strategy="full_copy", groups=n_groups,
                   copied_words=copied_words,
                   copied_per_group=copied_words / max(1, n_groups),
                   t_snapshot_s=t_snap, t_kernel_s=t_kernel)


def strict_touched_row(n, stream):
    """Strict mode snapshotting only rows a group writes, via a dirty set.

    Two-phase per group: (1) scan the group to collect the target rows it will
    write (the dirty set) and snapshot exactly those rows; (2) apply the kernel
    reading from the snapshot of the dirty rows and R elsewhere. Because a source
    row that is only READ (never written) in the group is identical in R and its
    snapshot, snapshotting only written rows is sufficient for strict semantics:
    a row's pre-group value is needed only where that row is subsequently
    overwritten within the same group.
    """
    W = words_for(n)
    R = _identity(n, W)
    copied_words = 0
    n_groups = 0
    t_snap = 0.0
    t_kernel = 0.0
    dirty_sizes = []
    for _, git in groupby(stream, key=lambda e: e.t):
        group = list(git)
        n_groups += 1
        # Phase 1: dirty set = rows this group writes.
        s0 = time.perf_counter()
        dirty = set()
        for e in group:
            dirty.add(e.v)
        snap = {v: R[v].copy() for v in dirty}
        t_snap += time.perf_counter() - s0
        copied_words += len(dirty) * W
        dirty_sizes.append(len(dirty))
        # Phase 2: read u from snapshot if u is dirty (its pre-group value),
        # else from R (unchanged this group); write v from snapshot-derived cand.
        s1 = time.perf_counter()
        for e in group:
            cand = snap[e.u] if e.u in snap else R[e.u]
            nz = cand & ~R[e.v]
            if nz.any():
                R[e.v] |= cand
        t_kernel += time.perf_counter() - s1
    return R, dict(strategy="touched_row", groups=n_groups,
                   copied_words=copied_words,
                   copied_per_group=copied_words / max(1, n_groups),
                   t_snapshot_s=t_snap, t_kernel_s=t_kernel,
                   mean_dirty_rows=float(np.mean(dirty_sizes)) if dirty_sizes else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", help="standard edge file (src dst t)")
    ap.add_argument("--out", default="results/ablation/touched_row.csv")
    args = ap.parse_args()

    if args.edges:
        n, edges, _ = load_standard_edges(args.edges)
        from temporal_strata.stream import sort_and_validate_edges
        streams = [("input", n, sort_and_validate_edges(n, edges))]
    else:
        # Synthetic: fixed n, vary number of timestamp groups T to expose the
        # snapshot/kernel balance. More groups => smaller per-group work =>
        # full-copy snapshot overhead dominates, touched-row stays proportional
        # to activity.
        from temporal_strata.stream import sort_and_validate_edges
        rng = np.random.default_rng(7)
        n = 1024
        streams = []
        for T in (16, 64, 256, 1024):
            per = 4096 // 1  # ~constant total edges
            edges = []
            for t in range(T):
                for _ in range(max(1, 4096 // T)):
                    u = int(rng.integers(0, n)); v = int(rng.integers(0, n))
                    if u != v:
                        edges.append((t, u, v))
            streams.append((f"T={T}", n, sort_and_validate_edges(n, edges)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    print(f"{'stream':>10} {'T':>6} {'strat':>12} {'copied/grp':>12} "
          f"{'snap_s':>9} {'kernel_s':>9} {'pairs':>10} {'agree':>6}")
    for name, n, stream in streams:
        Rf, sf = strict_full_copy(n, stream)
        Rt, st = strict_touched_row(n, stream)
        ref = temporal_strata_packed(n, [(e.t, e.u, e.v) for e in stream],
                                     same_time_mode="strict")
        pf, pt, pr = reachable_pairs(Rf, n), reachable_pairs(Rt, n), reachable_pairs(ref, n)
        agree = bool(np.array_equal(Rf, ref) and np.array_equal(Rt, ref) and pf == pt == pr)
        for stat in (sf, st):
            print(f"{name:>10} {stat['groups']:>6} {stat['strategy']:>12} "
                  f"{stat['copied_per_group']:>12.0f} {stat['t_snapshot_s']:>9.4f} "
                  f"{stat['t_kernel_s']:>9.4f} {pr:>10} {str(agree):>6}")
            stat.update(stream=name, n=n, pairs=pr, agree=agree)
            rows.append(stat)

    # write CSV
    import csv
    keys = ["stream", "n", "strategy", "groups", "copied_words",
            "copied_per_group", "mean_dirty_rows", "t_snapshot_s",
            "t_kernel_s", "pairs", "agree"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print("wrote", args.out)
    print("ALL AGREE:", all(r["agree"] for r in rows))


if __name__ == "__main__":
    main()
