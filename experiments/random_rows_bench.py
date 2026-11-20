#!/usr/bin/env python3
"""How fast this machine updates rows at random, by thread count (v25, R69).

Why: on sx-stackoverflow (v24 session, step 20) the block kernel slows down as
concurrent threads grow past six even with the work split into the same 18
chunks (17.5 s at p = 6, 35.4 s at p = 18), while a STREAM triad keeps rising.
A triad reads memory in order; the kernel reads and writes one random row per
edge. This benchmark measures only that access shape, without timestamps,
snapshots, or reachability, so a slowdown here is a property of the machine's
random-access memory path, not of STRATA's algorithm.

How: each column chunk owns a private n x cw tile of uint64 words (the tiled
layout of the kernel, cw = W / chunks words per row), starts from an identity
block, and applies ``tile[v] |= tile[u]`` for each (u, v) of an edge list, in
order. Two edge lists of the same length are used:
  * ``stream``: the first M edges of the prepared sx-stackoverflow stream (its
    real locality);
  * ``random``: M uniformly random pairs (fixed seed).
Static schedule, chunks = threads, median of ``--repeats`` runs after a warm-up.
The wrapping sum of all tile words must be the same at every thread count (the
updates of one column do not depend on the chunking), which is checked.
Reported: median seconds, row updates per second, touched bytes per second
(3 x 8 x cw per update per chunk), next to a STREAM triad at the same p.
This is a hardware measurement, not a correctness check of STRATA.

What:
    NUMBA_NUM_THREADS=18 python3 experiments/random_rows_bench.py \
        --dataset sx-stackoverflow --words 1024 --edges 16000000 \
        --threads 1,6,12,18 --repeats 3 --out results/real/random_rows.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False
    prange = range


def _rows_kernel(n, W, us, vs, nchunks):
    """Per-chunk tiles; returns the wrapping sum of every chunk's tile words (the
    total is the same for every chunking, since each column evolves alone)."""
    counts = np.zeros(nchunks, dtype=np.uint64)
    for c in prange(nchunks):
        lo = (c * W) // nchunks
        hi = ((c + 1) * W) // nchunks
        cw = hi - lo
        if cw <= 0:
            continue
        tile = np.zeros((n, cw), dtype=np.uint64)
        for w in range(cw):                      # identity block: source s in column s // 64
            for b in range(64):
                s = (lo + w) * 64 + b
                if s < n:
                    tile[s, w] = np.uint64(1) << np.uint64(b)
        for e in range(us.shape[0]):
            u = us[e]
            v = vs[e]
            for w in range(cw):
                tile[v, w] |= tile[u, w]
        total = np.uint64(0)
        for r in range(n):
            for w in range(cw):
                total += tile[r, w]
        counts[c] = total
    return counts


if HAVE_NUMBA:
    _rows_kernel = njit(parallel=True, cache=True)(_rows_kernel)


def random_pairs(n, m, seed=7):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, n, size=m, dtype=np.int64),
            rng.integers(0, n, size=m, dtype=np.int64))


def touched_bytes(m, W):
    """Load source segment, load and store target segment: 3 x 8 bytes per word."""
    return 3 * 8 * int(W) * int(m)


def summarize(rows):
    """Per edge order: median seconds by p and the slowdown of the largest p over
    the fastest p."""
    out = {}
    for order in sorted({r["order"] for r in rows}):
        sub = {r["threads"]: r["seconds_median"] for r in rows if r["order"] == order}
        best = min(sub, key=sub.get)
        out[order] = {"seconds": {str(p): t for p, t in sorted(sub.items())},
                      "best_threads": best,
                      "largest_over_best": sub[max(sub)] / sub[best]}
    return out


def bench(n, W, us, vs, threads, repeats, set_threads):
    rows = []
    for order, (a, b) in (("stream", (us, vs)), ("random", random_pairs(n, us.shape[0]))):
        ref = None
        for p in threads:
            eff = set_threads(p)
            if eff != p:
                print(f"  threads={p} unavailable; skipped")
                continue
            samples = []
            for rep in range(repeats + 1):                 # first run = warm-up + check
                t0 = time.perf_counter()
                counts = _rows_kernel(np.int64(n), np.int64(W), a, b, np.int64(eff))
                dt = time.perf_counter() - t0
                bits = int(counts.sum(dtype=np.uint64))
                ref = bits if ref is None else ref
                if bits != ref:
                    raise SystemExit(f"{order} p={eff}: word checksum {bits} != {ref}")
                if rep:
                    samples.append(dt)
            med = statistics.median(samples)
            rows.append({"order": order, "threads": eff, "chunks": eff, "edges": int(a.shape[0]),
                         "words": int(W), "seconds_samples": samples, "seconds_median": med,
                         "row_updates_per_s": a.shape[0] * eff / med,
                         "touched_GBps": touched_bytes(a.shape[0], W) / med / 1e9,
                         "checksum": ref})
            print(f"  {order} p={eff}: median {med:.3f} s, {rows[-1]['touched_GBps']:.1f} GB/s touched")
    return rows


def run(args):
    from experiments.bandwidth_roofline import stream_triad
    from experiments.bench_common import DATA
    from experiments.large_real_blocked import peak_rss_bytes
    from experiments.thread_record import thread_record
    from temporal_strata.io import load_standard_arrays
    from temporal_strata.parallel import machine_info, prepare_stream_arrays, set_threads

    threads = [int(x) for x in args.threads.split(",")]
    path = DATA / f"{args.dataset}.txt"
    if not path.exists():
        raise SystemExit(f"missing {path}; run experiments/fetch_real_temporal.py --names {args.dataset}")
    n, ts, us, vs, _ = load_standard_arrays(path)
    set_threads(max(threads))
    prep = prepare_stream_arrays(n, ts, us, vs, directed=True, same_time_mode="strict")
    del ts, us, vs
    m = min(args.edges, prep.num_edges) if args.edges else prep.num_edges
    su = np.ascontiguousarray(prep.us[:m]).astype(np.int64)
    sv = np.ascontiguousarray(prep.vs[:m]).astype(np.int64)
    del prep
    print(f"{args.dataset}: n={n:,}, {m:,} edges, W={args.words}")
    triad = stream_triad(threads, args.array_mib, args.repeats)
    triad_at = {r["threads"]: r["GBps"] for r in triad["rows"]}
    rows = bench(n, args.words, su, sv, threads, args.repeats, set_threads)
    for r in rows:
        r["triad_GBps"] = triad_at.get(r["threads"])
    out = {"dataset": args.dataset, "n": int(n), "edges": int(m), "words": args.words,
           "repeats": args.repeats, "schedule": "static, chunks = threads",
           "stream_triad": triad, "rows": rows, "summary": summarize(rows),
           "note": "hardware measurement of the kernel's access shape (tile[v] |= tile[u]); "
                   "no timestamps, snapshots, or reachability semantics",
           "peak_rss_bytes": peak_rss_bytes(), "machine": machine_info(),
           "threads_record": thread_record()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="sx-stackoverflow")
    ap.add_argument("--words", type=int, default=1024)
    ap.add_argument("--edges", type=int, default=16_000_000, help="0 = every edge")
    ap.add_argument("--threads", default="1,6,12,18")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--array-mib", type=int, default=1024)
    ap.add_argument("--out", default="results/real/random_rows.json")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
