#!/usr/bin/env python3
"""Word-packed temporal multi-source BFS baseline (v21, R54-2b).

Why: a reviewer asked for MS-BFS (Then et al., PVLDB 2014) adapted to time order
as a baseline. Scanning the whole stream with 64 sources per word is STRATA with
source blocking at beta = 64, so this baseline takes the MS-BFS idea further:
each batch of 64 sources keeps one uint64 per vertex and *skips* every edge whose
source word is still empty, so a batch pays only for the part of the stream its
sources have reached. Batches run in parallel over threads (one batch per task).

How: the shared preprocessing (sorted stream, timestamp groups) is the same as
STRATA's. Strict semantics read the pre-group words of a group before writing any
(two passes); closure repeats a group until no word changes. The result is packed
into the same n x W layout as STRATA and must match it bit for bit before any
time is recorded.

What: ``bench`` writes results/real/temporal_msbfs.json with, per dataset and
thread count, the MS-BFS time, the STRATA kernel time on the same prepared stream,
both verified identical.

    python3 experiments/temporal_msbfs.py bench --threads 1,18 --repeats 5 \
        --out results/real/temporal_msbfs.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.packed_reachability import words_for  # noqa: E402

try:
    import numba
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False


def _batch_strict(n, gptr, us, vs, first, count, te, tx):
    """One batch of ``count`` sources starting at ``first``: one word per vertex."""
    R = np.zeros(n, dtype=np.uint64)
    for j in range(count):
        R[first + j] |= np.uint64(1) << np.uint64(j)
    for g in range(gptr.shape[0] - 1):
        a, b = gptr[g], gptr[g + 1]
        c = 0
        for e in range(a, b):                 # read pre-group words
            x = R[us[e]]
            if x != 0:
                te[c] = vs[e]
                tx[c] = x
                c += 1
        for k in range(c):                    # then write
            R[te[k]] |= tx[k]
    return R


def _batch_closure(n, gptr, us, vs, first, count):
    R = np.zeros(n, dtype=np.uint64)
    for j in range(count):
        R[first + j] |= np.uint64(1) << np.uint64(j)
    for g in range(gptr.shape[0] - 1):
        a, b = gptr[g], gptr[g + 1]
        changed = True
        while changed:
            changed = False
            for e in range(a, b):
                x = R[us[e]]
                if x != 0 and (x & ~R[vs[e]]) != 0:
                    R[vs[e]] |= x
                    changed = True
    return R


def _msbfs_packed(n, gptr, us, vs, strict):
    W = (n + 63) // 64
    out = np.zeros((n, W), dtype=np.uint64)
    gmax = 1
    for g in range(gptr.shape[0] - 1):
        gmax = max(gmax, gptr[g + 1] - gptr[g])
    for w in prange(W):
        first = w * 64
        count = min(64, n - first)
        if strict:
            te = np.empty(gmax, dtype=np.int64)
            tx = np.empty(gmax, dtype=np.uint64)
            R = _batch_strict(n, gptr, us, vs, first, count, te, tx)
        else:
            R = _batch_closure(n, gptr, us, vs, first, count)
        for v in range(n):
            out[v, w] = R[v]
    return out


if HAVE_NUMBA:
    _batch_strict = njit(cache=True)(_batch_strict)
    _batch_closure = njit(cache=True)(_batch_closure)
    _msbfs_packed_jit = njit(parallel=True, cache=True)(_msbfs_packed)
else:  # pragma: no cover
    prange = range
    _msbfs_packed_jit = _msbfs_packed


def msbfs_relation(n, prep, mode="strict"):
    """Packed n x W relation from a PreparedStream (same layout as STRATA)."""
    return _msbfs_packed_jit(np.int64(n), prep.gptr, prep.us, prep.vs, mode == "strict")


def _time_best(fn, repeats):
    fn()                                          # warm-up (JIT, allocation)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return min(samples), samples


def bench(args):
    from experiments.bench_common import select_datasets_arrays
    from experiments.thread_record import thread_record
    from temporal_strata.parallel import (_empty_state, _set_identity_vec, machine_info,
                                          prepare_stream_arrays, resolve_plan, run_prepared,
                                          set_threads)

    threads = [int(x) for x in args.threads.split(",")]
    rows = []
    for name, n, arrays, _load in select_datasets_arrays(
            args.datasets.split(",") if args.datasets else [], args.large or [],
            include_default_synth=args.synthetic):
        for mode in args.modes.split(","):
            prep = prepare_stream_arrays(n, *arrays, directed=True, same_time_mode=mode,
                                         prep_mode="off")
            W = words_for(n)
            ref = None
            for p in threads:
                eff = set_threads(p)

                def strata():
                    R = _empty_state(n, n)
                    _set_identity_vec(R, 0, n)
                    run_prepared(R, prep, resolve_plan(W, eff, layout="tiled", schedule="static"))
                    return R
                Rs = strata()
                Rm = msbfs_relation(n, prep, mode)
                if not np.array_equal(Rs, Rm):
                    raise SystemExit(f"{name} {mode} p={p}: MS-BFS differs from STRATA; not timed")
                if ref is None:
                    ref = Rs
                elif not np.array_equal(ref, Rs):
                    raise SystemExit(f"{name} {mode}: STRATA differs across thread counts")
                ts_best, ts_s = _time_best(strata, args.repeats)
                tm_best, tm_s = _time_best(lambda: msbfs_relation(n, prep, mode), args.repeats)
                rows.append({"dataset": name, "mode": mode, "n": int(n), "m": prep.num_edges,
                             "W": int(W), "threads": int(eff), "verified_bit_identical": True,
                             "strata_kernel_s": ts_best, "strata_samples_s": ts_s,
                             "msbfs_s": tm_best, "msbfs_samples_s": tm_s,
                             "ratio_msbfs_over_strata": tm_best / ts_best})
                print(f"{name} {mode} p={eff}: STRATA {ts_best*1e3:.2f} ms, "
                      f"MS-BFS {tm_best*1e3:.2f} ms ({tm_best/ts_best:.2f}x)")
    out = {"machine": machine_info(), "threads": thread_record(), "repeats": args.repeats,
           "note": "MS-BFS = 64 sources per word, empty source words skipped, batches over "
                   "threads; STRATA = tiled/static kernel on the same prepared stream",
           "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--datasets", default="CollegeMsg,BitcoinOTC,BitcoinOTCpos")
    b.add_argument("--large", action="append", help="n:m:groups[:seed], repeatable")
    b.add_argument("--synthetic", action="store_true", help="add the n=8000, 1e6-edge stream")
    b.add_argument("--modes", default="strict")
    b.add_argument("--threads", default="1,18")
    b.add_argument("--repeats", type=int, default=5)
    b.add_argument("--out", default="results/real/temporal_msbfs.json")
    args = ap.parse_args(argv)
    if not HAVE_NUMBA:
        raise SystemExit("numba is required")
    if args.cmd == "bench":
        bench(args)


if __name__ == "__main__":
    main()
