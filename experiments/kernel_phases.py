#!/usr/bin/env python3
"""Which phase of the block kernel slows down past six threads (v27).

Why: on sx-stackoverflow (beta = 65,536) the strict tiled kernel takes 2.0x as
long at p = 18 as at p = 6 with the same 18 chunks (v24, step 20), while the
kernel's access shape alone (random row updates, v25, step 21) speeds up to
p = 18. The paper's Threats paragraph points to what that benchmark leaves out
-- the touched-row snapshot and the copies between the shared state and the
per-chunk tiles -- but only as an inference.

How: the same prepared stream and block, static schedule, chunks = p, and four
variants of the column kernel, each timed at p = 1 / 6 / 12 / 18:
  (a) ``strict``    the production strict tiled kernel (reference);
  (b) ``nosnap``    the same tiles and copies, every edge reads the live tile
                    (no snapshot; not strict semantics, timing comparison only);
  (c) ``copy``      allocate each tile, copy the state in and back out, no edges;
  (d) ``prealloc``  (a) with all tiles in one buffer allocated and touched
                    before the clock starts (no allocation / page faults inside).
Checks before any time is kept: (a) matches a per-source BFS on sampled sources
and keeps one pair count at every p; (d) gives the same state checksum as (a);
(b) gives one checksum at every p; (c) leaves the state unchanged.

v28 adds the follow-up the v27 run asked for. The copy phase alone grew from 3.3 s
at p = 1 to 32.6 s at p = 18, so four more variants separate its possible causes
and test the remedy:
  (e) ``copy_prefault``  copy after every page of the fresh state was touched
                         outside the clock (first-touch page faults removed);
  (f) ``copy_reuse``     one state allocated once and refilled outside the clock;
  (g) ``copy_aligned``   chunk borders on 128-byte cache-line multiples;
  (h) ``tiles``          the block run with the tiles as the state
                         (temporal_strata.parallel.run_block_tiles): no shared
                         state, no copies; checked against the per-source BFS
                         and against the strict pair count.

What:
    NUMBA_NUM_THREADS=18 python3 experiments/kernel_phases.py \
        --dataset sx-stackoverflow --block-bits 65536 --threads 1,6,12,18 \
        --repeats 3 --out results/real/kernel_phases.json
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

VARIANTS = ("strict", "nosnap", "copy", "prealloc")
# v28 (copy phase, tiles as state): run with --variants
COPY_VARIANTS = ("strict", "copy", "copy_prefault", "copy_reuse", "copy_aligned", "tiles")
ALL_VARIANTS = VARIANTS + ("copy_prefault", "copy_reuse", "copy_aligned", "tiles")
LINE_WORDS = 16


def _nosnap_kernel(R, gptr, us, vs, nchunks):
    n = R.shape[0]
    W = R.shape[1]
    m = us.shape[0]
    for c in prange(nchunks):
        lo = (c * W) // nchunks
        hi = ((c + 1) * W) // nchunks
        cw = hi - lo
        if cw <= 0:
            continue
        tile = np.empty((n, cw), dtype=np.uint64)
        for r in range(n):
            for w in range(cw):
                tile[r, w] = R[r, lo + w]
        for e in range(m):
            u = us[e]
            v = vs[e]
            for w in range(cw):
                tile[v, w] |= tile[u, w]
        for r in range(n):
            for w in range(cw):
                R[r, lo + w] = tile[r, w]


def _copy_kernel(R, nchunks):
    n = R.shape[0]
    W = R.shape[1]
    for c in prange(nchunks):
        lo = (c * W) // nchunks
        hi = ((c + 1) * W) // nchunks
        cw = hi - lo
        if cw <= 0:
            continue
        tile = np.empty((n, cw), dtype=np.uint64)
        for r in range(n):
            for w in range(cw):
                tile[r, w] = R[r, lo + w]
        for r in range(n):
            for w in range(cw):
                R[r, lo + w] = tile[r, w]


def _copy_bounds_kernel(R, bounds):
    n = R.shape[0]
    for c in prange(bounds.shape[0] - 1):
        lo = bounds[c]
        cw = bounds[c + 1] - lo
        if cw <= 0:
            continue
        tile = np.empty((n, cw), dtype=np.uint64)
        for r in range(n):
            for w in range(cw):
                tile[r, w] = R[r, lo + w]
        for r in range(n):
            for w in range(cw):
                R[r, lo + w] = tile[r, w]


def _touch2d(R, z):
    # v29: OR with a runtime zero so that the compiler cannot drop the store
    # (the v28 self-assignment R[i, j] = R[i, j] may be removed as a no-op, which
    # would leave the fresh pages untouched and make copy_prefault equal copy).
    for i in prange(R.shape[0]):
        for j in range(R.shape[1]):
            R[i, j] = R[i, j] | z


def aligned_bounds(W, nchunks, line=LINE_WORDS):
    """Chunk borders rounded to multiples of ``line`` words (first 0, last W)."""
    b = [0]
    for c in range(1, nchunks):
        x = int(round(c * W / nchunks / line)) * line
        b.append(min(max(x, b[-1]), W))
    b.append(W)
    return np.asarray(b, dtype=np.int64)


def _strict_prealloc_kernel(R, buf, gptr, us, vs, src_slot, dirty_idx, dptr, nchunks, max_dirty):
    """The production strict tiled kernel with tile c = buf[n*lo : n*hi] (n x cw)."""
    n = R.shape[0]
    W = R.shape[1]
    G = gptr.shape[0] - 1
    for c in prange(nchunks):
        lo = (c * W) // nchunks
        hi = ((c + 1) * W) // nchunks
        cw = hi - lo
        if cw <= 0:
            continue
        tile = buf[n * lo:n * hi].reshape((n, cw))
        for r in range(n):
            for w in range(cw):
                tile[r, w] = R[r, lo + w]
        snap = np.empty((max_dirty, cw), dtype=np.uint64)
        for g in range(G):
            d0 = dptr[g]
            d1 = dptr[g + 1]
            for k in range(d0, d1):
                row = dirty_idx[k]
                for w in range(cw):
                    snap[k - d0, w] = tile[row, w]
            for e in range(gptr[g], gptr[g + 1]):
                v = vs[e]
                sl = src_slot[e]
                if sl >= 0:
                    s = sl - d0
                    for w in range(cw):
                        tile[v, w] |= snap[s, w]
                else:
                    u = us[e]
                    for w in range(cw):
                        tile[v, w] |= tile[u, w]
        for r in range(n):
            for w in range(cw):
                R[r, lo + w] = tile[r, w]


def _checksum(R):
    """(wrapping sum, xor) of all words: equal states give equal pairs."""
    s = np.uint64(0)
    x = np.uint64(0)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            s += R[i, j]
            x ^= R[i, j]
    return s, x


def _touch(buf):
    for i in prange(buf.shape[0]):
        buf[i] = np.uint64(0)


if HAVE_NUMBA:
    _nosnap_kernel = njit(parallel=True, cache=True)(_nosnap_kernel)
    _copy_kernel = njit(parallel=True, cache=True)(_copy_kernel)
    _strict_prealloc_kernel = njit(parallel=True, cache=True)(_strict_prealloc_kernel)
    _checksum = njit(cache=True)(_checksum)
    _touch = njit(parallel=True, cache=True)(_touch)
    _copy_bounds_kernel = njit(parallel=True, cache=True)(_copy_bounds_kernel)
    _touch2d = njit(parallel=True, cache=True)(_touch2d)


def checksum(R):
    s, x = _checksum(R)
    return [int(s), int(x)]


def run_variant(variant, R, prep, nch, buf=None):
    """Advance R in place with one variant; returns nothing (timed by the caller)."""
    from temporal_strata.parallel import resolve_plan, run_prepared
    if variant == "strict":
        run_prepared(R, prep, resolve_plan(R.shape[1], int(nch), layout="tiled",
                                           schedule="static", chunks=int(nch)))
    elif variant == "nosnap":
        _nosnap_kernel(R, prep.gptr, prep.us, prep.vs, np.int64(nch))
    elif variant == "copy":
        _copy_kernel(R, np.int64(nch))
    elif variant in ("copy_prefault", "copy_reuse"):
        _copy_kernel(R, np.int64(nch))
    elif variant == "copy_aligned":
        _copy_bounds_kernel(R, aligned_bounds(R.shape[1], int(nch)))
    elif variant == "prealloc":
        _strict_prealloc_kernel(R, buf, prep.gptr, prep.us, prep.vs, prep.src_slot,
                                prep.dirty_idx, prep.dptr, np.int64(nch), np.int64(prep.max_dirty))
    else:
        raise ValueError(variant)


def summarize(rows):
    """Per variant: median seconds by p and the p_max / p=6 ratio (or vs the
    smallest p above one when 6 was not run)."""
    out = {}
    for v in ALL_VARIANTS:
        sub = {r["threads"]: r["seconds_median"] for r in rows if r["variant"] == v}
        if not sub:
            continue
        ps = sorted(sub)
        ref = 6 if 6 in sub else (ps[1] if len(ps) > 1 else ps[0])
        out[v] = {"seconds": {str(p): sub[p] for p in ps}, "ref_threads": ref,
                  "pmax_over_ref": sub[ps[-1]] / sub[ref]}
    return out


def measure(n, width, prep, threads, repeats, set_threads, src=None, bfs=None, variants=VARIANTS):
    from temporal_strata.packed_reachability import _empty_state
    from temporal_strata.parallel import _set_identity_vec
    from experiments.large_real_blocked import column_of

    def fresh():
        Rb = _empty_state(n, width)
        _set_identity_vec(Rb, 0, width)
        return Rb

    ident = checksum(fresh())
    rows, ref = [], {}
    buf = None
    shared = fresh() if "copy_reuse" in variants else None
    if "prealloc" in variants:
        buf = np.empty(n * fresh().shape[1], dtype=np.uint64)
        _touch(buf)
    for p in threads:
        eff = set_threads(p)
        if eff != p:
            print(f"  threads={p} unavailable; skipped")
            continue
        for v in variants:
            samples = []
            for rep in range(repeats + 1):                 # first run = warm-up + check
                if v == "tiles":
                    from temporal_strata.parallel import run_block_tiles
                    local = [] if src is None else [int(x) for x in src]
                    t0 = time.perf_counter()
                    pairs, cols, _ = run_block_tiles(prep, 0, width, eff, sample_local=local)
                    dt = time.perf_counter() - t0
                    if rep == 0:
                        for i, s in enumerate(local):
                            if not np.array_equal(cols[i], bfs[i]):
                                raise SystemExit(f"tiles p={eff}: source {s} differs from BFS")
                        ref.setdefault("tiles_pairs", pairs)
                        if pairs != ref["tiles_pairs"] or ("strict_pairs" in ref and pairs != ref["strict_pairs"]):
                            raise SystemExit(f"tiles p={eff}: pair count differs")
                    else:
                        samples.append(dt)
                    continue
                if v == "copy_reuse":
                    shared[:] = 0                          # refilled outside the clock
                    _set_identity_vec(shared, 0, width)
                    Rb = shared
                else:
                    Rb = fresh()
                if v == "prealloc":
                    _touch(buf)                            # touched again, outside the clock
                if v == "copy_prefault":
                    _touch2d(Rb, np.uint64(0))             # every page written, outside the clock
                t0 = time.perf_counter()
                run_variant(v, Rb, prep, eff, buf)
                dt = time.perf_counter() - t0
                if rep == 0:
                    cs = checksum(Rb)
                    if v == "strict":
                        ref.setdefault("strict_pairs", int(np.bitwise_count(Rb).sum()))
                    if v == "strict" and src is not None:
                        for i, s in enumerate(src):
                            if not np.array_equal(column_of(Rb, 0, int(s)), bfs[i]):
                                raise SystemExit(f"strict p={eff}: source {s} differs from BFS")
                    if v.startswith("copy") and cs != ident:
                        raise SystemExit(f"copy p={eff}: state changed")
                    if v == "prealloc" and cs != ref.get("strict"):
                        raise SystemExit(f"prealloc p={eff}: state differs from strict")
                    if v in ("strict", "nosnap"):
                        ref.setdefault(v, cs)
                        if cs != ref[v]:
                            raise SystemExit(f"{v} p={eff}: state differs between thread counts")
                else:
                    samples.append(dt)
                if v != "copy_reuse":
                    del Rb
            med = statistics.median(samples)
            rows.append({"variant": v, "threads": eff, "chunks": eff, "seconds_samples": samples,
                         "seconds_median": med})
            print(f"  {v:8s} p={eff}: median {med:.3f} s")
    return rows, {k: v for k, v in ref.items()}


def run(args):
    from experiments.bench_common import DATA, _bfs_sources_strict
    from experiments.large_real_blocked import peak_rss_bytes, pick_sources
    from experiments.thread_record import thread_record
    from temporal_strata.io import load_standard_arrays
    from temporal_strata.parallel import machine_info, prepare_stream_arrays, set_threads

    threads = [int(x) for x in args.threads.split(",")]
    path = DATA / f"{args.dataset}.txt"
    if not path.exists():
        raise SystemExit(f"missing {path}; run experiments/fetch_real_temporal.py --names {args.dataset}")
    n, ts, us, vs, meta = load_standard_arrays(path)
    set_threads(max(threads))
    prep = prepare_stream_arrays(n, ts, us, vs, directed=True, same_time_mode="strict")
    del ts, us, vs
    width = min(args.block_bits, n)
    src = pick_sources([(0, width)], args.verify_sources)
    bfs = _bfs_sources_strict(n, prep.gptr, prep.us, prep.vs, src)
    print(f"{args.dataset}: n={n:,} m={prep.num_edges:,} block {width}")
    variants = tuple(args.variants.split(",")) if args.variants else VARIANTS
    bad = [v for v in variants if v not in ALL_VARIANTS]
    if bad:
        raise SystemExit(f"unknown variant(s) {bad}")
    rows, ref = measure(n, width, prep, threads, args.repeats, set_threads, src, bfs, variants)
    out = {"dataset": args.dataset, "n": int(n), "m": prep.num_edges, "meta": meta,
           "block_bits": args.block_bits, "repeats": args.repeats,
           "schedule": "static, chunks = threads", "verified_sources": len(src),
           "variants": list(variants), "checksums": ref, "rows": rows, "summary": summarize(rows),
           "note": "nosnap is not strict semantics (timing comparison only); copy has no edges; "
                   "prealloc allocates and touches all tiles before the clock starts",
           "peak_rss_bytes": peak_rss_bytes(), "machine": machine_info(),
           "threads_record": thread_record()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="sx-stackoverflow")
    ap.add_argument("--block-bits", type=int, default=65536)
    ap.add_argument("--threads", default="1,6,12,18")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--verify-sources", type=int, default=16)
    ap.add_argument("--variants", default=None,
                    help="comma list (default: strict,nosnap,copy,prealloc); v28 adds "
                         "copy_prefault, copy_reuse, copy_aligned, tiles")
    ap.add_argument("--out", default="results/real/kernel_phases.json")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
