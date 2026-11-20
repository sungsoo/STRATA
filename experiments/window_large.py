#!/usr/bin/env python3
"""Rolling windows and monotone insertion on a large real stream (v34, R109).

Why
---
The paper motivates STRATA with rolling-window provenance (an AML team asks, for
every account, who could have funded it inside a sliding window), but measures
windows and insertion only on CollegeMsg (1.9k vertices). This harness repeats the
two operations with the compiled parallel kernel on sx-askubuntu (159k vertices,
964k edges, a 3.2 GB packed state).

How
---
* Windows: the time-sorted stream is cut at group boundaries into ``--slots``
  windows of ``--window-frac`` of the edges, advancing by an equal step. Each
  window is preprocessed and computed from scratch (tile state, static schedule,
  p threads); 16 sampled sources per window are checked against a per-source
  temporal BFS on the same window. Times: preprocessing and kernel per slot.
* Insertion: the state after the first half of the stream is advanced by
  ``--batches`` later batches (Proposition 5: only the new edges propagate); after
  the last batch the state must equal a full recomputation bit for bit. Times: the
  kernel per batch against the full recomputation's kernel.
* v36: ``--insert-layouts tiled,rowmajor`` advances one copy per layout. The tiled
  layout gathers and scatters the whole state on every call; row-major updates
  only the rows a batch writes. ``--slots 0`` skips the windows.

What
---
    NUMBA_NUM_THREADS=18 python3 experiments/window_large.py --dataset sx-askubuntu \
        --threads 18 --slots 8 --window-frac 0.25 --batches 5 \
        --out results/real/window_large.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def group_cut(ts, i):
    """Smallest index >= i that starts a new timestamp group (or len(ts))."""
    i = int(min(max(i, 0), len(ts)))
    while 0 < i < len(ts) and ts[i] == ts[i - 1]:
        i += 1
    return i


def run_full(n, ts, us, vs, p):
    from temporal_strata.packed_reachability import words_for
    from temporal_strata.parallel import (_set_identity_vec, prepare_stream_arrays,
                                          resolve_plan, run_prepared)
    t0 = time.perf_counter()
    prep = prepare_stream_arrays(n, ts, us, vs, directed=True, same_time_mode="strict")
    prep_s = time.perf_counter() - t0
    W = words_for(n)
    R = np.zeros((n, W), dtype=np.uint64)
    _set_identity_vec(R, 0, n)
    t1 = time.perf_counter()
    run_prepared(R, prep, resolve_plan(W, p, layout="tiled", schedule="static", chunks=p))
    return R, prep, prep_s, time.perf_counter() - t1


def advance(R, n, ts, us, vs, p, layout="tiled"):
    """Monotone insertion: propagate only the batch on the resident state.

    v36: ``layout="rowmajor"`` updates the rows the batch writes in place; the tiled
    layout gathers the whole state into tiles and scatters it back on every call."""
    from temporal_strata.parallel import prepare_stream_arrays, resolve_plan, run_prepared
    prep = prepare_stream_arrays(n, ts, us, vs, directed=True, same_time_mode="strict")
    t0 = time.perf_counter()
    run_prepared(R, prep, resolve_plan(R.shape[1], p, layout=layout, schedule="static", chunks=p))
    return time.perf_counter() - t0


def windows(n, ts, us, vs, p, slots, frac, n_samples=16):
    from experiments.bench_common import _bfs_sources_strict
    from temporal_strata.parallel import prepare_stream_arrays, run_block_tiles
    m = len(ts)
    L = max(1, int(m * frac))
    step = max(1, (m - L) // max(1, slots - 1))
    rng = np.random.default_rng(5)
    out = []
    for k in range(slots):
        lo = group_cut(ts, k * step)
        hi = group_cut(ts, min(m, lo + L))
        t0 = time.perf_counter()
        prep = prepare_stream_arrays(n, ts[lo:hi], us[lo:hi], vs[lo:hi], directed=True,
                                     same_time_mode="strict")
        prep_s = time.perf_counter() - t0
        src = np.sort(rng.choice(n, size=min(n_samples, n), replace=False))
        t1 = time.perf_counter()
        pairs, cols, _ = run_block_tiles(prep, 0, n, p, sample_local=[int(s) for s in src])
        kern_s = time.perf_counter() - t1
        bfs = _bfs_sources_strict(n, prep.gptr, prep.us, prep.vs, src)
        for i, s in enumerate(src):
            if not np.array_equal(cols[i], bfs[i]):
                raise SystemExit(f"window {k}: source {s} differs from per-source BFS")
        out.append({"slot": k, "lo": lo, "hi": hi, "edges": hi - lo, "prep_s": prep_s,
                    "kernel_s": kern_s, "pairs": int(pairs), "verified_sources": len(src)})
        print(f"  window {k}: edges {hi - lo:,}, prep {prep_s:.2f} s, kernel {kern_s:.2f} s")
    return out


def insertion(n, ts, us, vs, p, batches, layouts=("tiled",)):
    """v36: each layout in ``layouts`` advances its own copy of the half-stream
    state; every copy must equal the full recomputation bit for bit. The first
    layout's batches stay under "batches" (v35 record shape)."""
    m = len(ts)
    cut = group_cut(ts, m // 2)
    R0, _, _, base_k = run_full(n, ts[:cut], us[:cut], vs[:cut], p)
    bounds = [cut] + [group_cut(ts, cut + (m - cut) * (b + 1) // batches) for b in range(batches)]
    bounds[-1] = m
    per = {}
    states = {}
    for lay in layouts:
        R = R0.copy()
        out = []
        for b in range(batches):
            lo, hi = bounds[b], bounds[b + 1]
            if hi <= lo:
                continue
            dt = advance(R, n, ts[lo:hi], us[lo:hi], vs[lo:hi], p, layout=lay)
            out.append({"batch": b, "edges": hi - lo, "kernel_s": dt})
            print(f"  {lay} batch {b}: edges {hi - lo:,}, kernel {dt:.3f} s")
        per[lay] = out
        states[lay] = R
    del R0
    Rf, _, _, full_k = run_full(n, ts, us, vs, p)
    for lay, R in states.items():
        if not np.array_equal(R, Rf):
            raise SystemExit(f"insertion ({lay}): state differs from full recomputation")
    return {"base_edges": cut, "base_kernel_s": base_k, "batches": per[layouts[0]],
            "layouts": per, "full_kernel_s": full_k, "bit_identical": True}


def main(argv=None):
    from experiments.bench_common import DATA
    from temporal_strata.io import load_standard_arrays
    from temporal_strata.parallel import machine_info, set_threads
    try:
        from thread_record import thread_record
    except ImportError:
        from experiments.thread_record import thread_record
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sx-askubuntu")
    ap.add_argument("--threads", type=int, default=18)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--window-frac", type=float, default=0.25)
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--insert-layouts", default="tiled",
                    help="comma list of layouts for the insertion batches (v36: tiled,rowmajor)")
    ap.add_argument("--out", default=str(HERE.parent / "results" / "real" / "window_large.json"))
    args = ap.parse_args(argv)
    path = DATA / f"{args.dataset}.txt"
    if not path.exists():
        raise SystemExit(f"missing {path}; run experiments/fetch_real_temporal.py --names {args.dataset}")
    n, ts, us, vs, meta = load_standard_arrays(path)
    order = np.argsort(ts, kind="stable")
    ts, us, vs = ts[order], us[order], vs[order]
    p = set_threads(args.threads)
    print(f"{args.dataset}: n={n:,} m={len(ts):,} p={p}")
    # warm-up (JIT compilation) outside every timed region
    k0 = group_cut(ts, min(len(ts), 1000))
    run_full(n, ts[:k0], us[:k0], vs[:k0], p)
    windows(n, ts[:k0], us[:k0], vs[:k0], p, 1, 1.0, n_samples=1)
    for lay in args.insert_layouts.split(","):
        if lay:
            Rw, _, _, _ = run_full(n, ts[:k0], us[:k0], vs[:k0], p)
            advance(Rw, n, ts[k0:2 * k0], us[k0:2 * k0], vs[k0:2 * k0], p, layout=lay)
            del Rw
    win = windows(n, ts, us, vs, p, args.slots, args.window_frac) if args.slots > 0 else []
    ins = insertion(n, ts, us, vs, p, args.batches,
                    tuple(x for x in args.insert_layouts.split(",") if x))
    payload = {"dataset": args.dataset, "n": int(n), "m": int(len(ts)), "threads": p,
               "machine": machine_info(), "threads_record": thread_record(),
               "windows": win, "insertion": ins}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
