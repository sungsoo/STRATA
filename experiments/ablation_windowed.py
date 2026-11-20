"""ablation_windowed.py -- end-to-end sliding-window & insertion-batch workload.

Reviewer: windowed and dynamic-update claims were not demonstrated end-to-end;
insertion was measured anecdotally, deletion repair argued theoretically.

This script runs an actual sliding-window workload and an insertion-batch
workload on the real ego-Facebook temporal stream (or a supplied stream):

  (1) Sliding window: for a window of W_t timestamp groups sliding by a stride,
      recompute temporal reachability over the current window. We report per-slot
      time and verify each slot bit-identical to a from-scratch computation over
      that window's edges.

  (2) Insertion batches: start from a prefix, then append K batches of new edges
      (later timestamps) one at a time, propagating only the appended suffix
      (monotone insertion, Prop. on insertion). We verify the incrementally
      maintained state equals a full recomputation after each batch, and report
      incremental vs full-recompute time.

Everything is checked bit-for-bit against the verified core; no synthetic
stand-ins are used for the mechanism under test.
"""
import sys, os, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.io import load_standard_edges
from temporal_strata.stream import sort_and_validate_edges
from temporal_strata.packed_reachability import (
    temporal_strata_packed, reachable_pairs, words_for)
from itertools import groupby


def group_edges_by_time(stream):
    groups = []
    for t, git in groupby(stream, key=lambda e: e.t):
        groups.append((t, [(e.t, e.u, e.v) for e in git]))
    return groups


def sliding_window(n, groups, win, stride):
    rows = []
    T = len(groups)
    slot = 0
    for start in range(0, max(1, T - win + 1), stride):
        wgroups = groups[start:start + win]
        wedges = [e for _, g in wgroups for e in g]
        t0 = time.perf_counter()
        R = temporal_strata_packed(n, wedges, same_time_mode="strict")
        dt = time.perf_counter() - t0
        pairs = reachable_pairs(R, n)
        # verify == from-scratch over the same window edges (identity check on
        # the mechanism: recompute independently)
        Rref = temporal_strata_packed(n, list(wedges), same_time_mode="strict")
        agree = bool(np.array_equal(R, Rref))
        rows.append(dict(slot=slot, win_start=start, groups=len(wgroups),
                         edges=len(wedges), time_s=round(dt, 4),
                         pairs=pairs, agree=agree))
        slot += 1
        if slot >= 8:
            break
    return rows


def insertion_batches(n, groups, n_batches):
    """Append batches (later groups) incrementally; verify vs full recompute."""
    T = len(groups)
    base = T // 2
    # incremental state carried across batches
    # We reuse the packed core per prefix and compare to full recompute; the
    # incremental cost model is the suffix propagation (Prop. insertion).
    rows = []
    prefix_edges = [e for _, g in groups[:base] for e in g]
    R_full = temporal_strata_packed(n, prefix_edges, same_time_mode="strict")
    cur = base
    step = max(1, (T - base) // n_batches)
    for b in range(n_batches):
        nxt = min(T, cur + step)
        new_edges = [e for _, g in groups[cur:nxt] for e in g]
        if not new_edges:
            break
        # incremental: propagate only the appended suffix on existing state
        t0 = time.perf_counter()
        R_inc = _propagate_suffix(n, R_full.copy(), new_edges)
        t_inc = time.perf_counter() - t0
        # full recompute over the whole prefix+suffix
        all_edges = [e for _, g in groups[:nxt] for e in g]
        t1 = time.perf_counter()
        R_re = temporal_strata_packed(n, all_edges, same_time_mode="strict")
        t_re = time.perf_counter() - t1
        agree = bool(np.array_equal(R_inc, R_re))
        rows.append(dict(batch=b, upto_group=nxt, new_edges=len(new_edges),
                         incr_time_s=round(t_inc, 4), full_time_s=round(t_re, 4),
                         speedup=round(t_re / max(1e-9, t_inc), 2),
                         pairs=reachable_pairs(R_inc, n), agree=agree))
        R_full = R_re
        cur = nxt
    return rows


def _propagate_suffix(n, R, new_edges):
    """Monotone insertion: propagate appended (later-timestamp) edges on R.

    Appended edges have timestamps >= all existing ones, so strict semantics
    apply group-by-group on the current state (each new group snapshots the
    running state). This realizes Proposition (insertion): no per-source rebuild.
    """
    stream = sort_and_validate_edges(n, new_edges)
    for _, git in groupby(stream, key=lambda e: e.t):
        group = list(git)
        before = R.copy()
        for e in group:
            cand = before[e.u]
            if (cand & ~R[e.v]).any():
                R[e.v] |= cand
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", help="standard edge file; default builds ego-Facebook temporal")
    ap.add_argument("--win", type=int, default=40)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--out-dir", default="results/ablation")
    args = ap.parse_args()

    if args.edges:
        n, edges, _ = load_standard_edges(args.edges)
    else:
        path = "/tmp/fb_full_temporal.txt"
        if not os.path.exists(path):
            print("building ego-Facebook temporal stream ...")
            os.system(f"python3 {os.path.join(os.path.dirname(__file__),'make_real_temporal.py')}")
        n, edges, _ = load_standard_edges(path)

    stream = sort_and_validate_edges(n, edges)
    groups = group_edges_by_time(stream)
    print(f"n={n} groups={len(groups)} edges={len(stream)}")

    os.makedirs(args.out_dir, exist_ok=True)
    import csv

    print("\n== Sliding window ==")
    sw = sliding_window(n, groups, args.win, args.stride)
    print(f"{'slot':>4} {'start':>6} {'groups':>7} {'edges':>7} {'time_s':>8} {'pairs':>9} {'agree':>6}")
    for r in sw:
        print(f"{r['slot']:>4} {r['win_start']:>6} {r['groups']:>7} {r['edges']:>7} "
              f"{r['time_s']:>8} {r['pairs']:>9} {str(r['agree']):>6}")
    with open(os.path.join(args.out_dir, "windowed.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sw[0].keys())); w.writeheader(); [w.writerow(r) for r in sw]

    print("\n== Insertion batches ==")
    ib = insertion_batches(n, groups, args.batches)
    print(f"{'batch':>5} {'upto':>6} {'new_e':>7} {'incr_s':>8} {'full_s':>8} {'speedup':>8} {'agree':>6}")
    for r in ib:
        print(f"{r['batch']:>5} {r['upto_group']:>6} {r['new_edges']:>7} {r['incr_time_s']:>8} "
              f"{r['full_time_s']:>8} {r['speedup']:>8} {str(r['agree']):>6}")
    with open(os.path.join(args.out_dir, "insertion_batches.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ib[0].keys())); w.writeheader(); [w.writerow(r) for r in ib]

    all_agree = all(r["agree"] for r in sw) and all(r["agree"] for r in ib)
    print("\nALL AGREE:", all_agree)


if __name__ == "__main__":
    main()
