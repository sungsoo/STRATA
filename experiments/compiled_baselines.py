#!/usr/bin/env python3
"""Compiled-vs-compiled comparison and integrated main table.

Addresses two reviewer points:

  * (Section 2-2, compiled-vs-compiled) The headline speedup was measured against
    an interpreted per-source temporal BFS, which conflates the algorithmic gain
    with Python overhead. Here we compile BOTH the per-source temporal BFS and
    STRATA with Numba and compare them at C speed, isolating the pure algorithmic
    effect (word-packed many-source propagation vs one traversal per source).

  * (Section 4-4, integrated table) All comparison points are gathered into ONE
    table so a reader can separate the language effect from the algorithm effect
    at a glance: exact BFS (interpreted ref) | dense NumPy | STRATA-py |
    STRATA-numba | GraphBLAS (static reduction, compiled C).

v28 (backlog R79): the STRATA-numba column used the early experimental kernel of
experiments/numba_kernel.py, which copies the whole n x W state at every timestamp
group (``snap = R.copy()``) instead of the touched rows, and both compiled columns
timed a Python-level ``sorted(edges)`` of the edge list. Both compiled methods now
receive the same time-sorted int64 arrays prepared outside the clock; STRATA runs
the production core (``prepare_stream_arrays`` + ``run_prepared``, tiled/static,
one thread, touched-row snapshot). ``strata_numba_ms`` is that core end to end
from the sorted arrays (preprocessing + kernel), ``strata_numba_kernel_ms`` the
kernel alone; the compiled-vs-compiled ratio divides kernel by kernel (the
per-source BFS has no preprocessing). The legacy kernel's time stays in
``strata_numba_legacy_ms`` for comparison.

Everything is verified bit-for-bit (or by reachable-pair count against the exact
reference) before any timing is reported. Wall-clock is warm-up + best-of-3 for
the fast methods; the interpreted exact reference is a single warm measurement.
Writes results/perf/compiled_baselines.json and nothing else (``--no-tex`` is
accepted for compatibility and does nothing).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.stream import generate_temporal_stream, TemporalEdge
from temporal_strata.baseline import all_sources_temporal_bfs
from temporal_strata.packed_reachability import (
    temporal_strata_packed, packed_to_matrix, reachable_pairs, words_for,
)

# reuse the shipped compiled STRATA kernel
from importlib import import_module
try:
    from thread_record import thread_record  # run as a script
except ImportError:  # imported as experiments.<module>
    from experiments.thread_record import thread_record
_nk = import_module("experiments.numba_kernel") if os.path.exists("experiments/numba_kernel.py") else None
if _nk is None:
    sys.path.insert(0, "experiments")
    import numba_kernel as _nk  # type: ignore

from temporal_strata.datasets import FACEBOOK as FB  # noqa: E402


# ---------------------------------------------------------------------------
# Compiled per-source temporal BFS (the exact reference algorithm, compiled).
# Strict semantics via deferred within-group writes (O(m) per source, no O(n)
# per-group snapshot): edges read the pre-group `reached`, and newly reachable
# targets are committed only at the group boundary, so same-timestamp edges do
# not chain. Returns the packed relation so it can be checked bit-for-bit.
# ---------------------------------------------------------------------------
@njit(cache=True)
def _per_source_strict_bfs(n, W, ts, us, vs):
    R = np.zeros((n, W), dtype=np.uint64)
    m = ts.shape[0]
    reached = np.zeros(n, dtype=np.uint8)
    pend = np.empty(n, dtype=np.int64)  # deferred newly-reached buffer
    for s in range(n):
        # reset
        for x in range(n):
            reached[x] = 0
        reached[s] = 1
        i = 0
        while i < m:
            j = i
            t0 = ts[i]
            while j < m and ts[j] == t0:
                j += 1
            np_ = 0
            for k in range(i, j):
                u = us[k]; v = vs[k]
                if reached[u] == 1 and reached[v] == 0:
                    # defer: mark pending (avoid same-group chaining)
                    pend[np_] = v
                    np_ += 1
            for q in range(np_):
                reached[pend[q]] = 1
            i = j
        # pack row s's reach into R[:, .] as a source bit? We store by target:
        # bit s of R[v] set iff s reaches v -> set for every reached v.
        sw = s >> 6
        sb = np.uint64(1) << np.uint64(s & 63)
        for v in range(n):
            if reached[v] == 1:
                R[v, sw] |= sb
    return R


def sorted_arrays(edges):
    """(ts, us, vs) int64 arrays in (t, u, v) order, built once outside the clock."""
    arr = np.array(sorted(edges), dtype=np.int64)
    return (np.ascontiguousarray(arr[:, 0]), np.ascontiguousarray(arr[:, 1]),
            np.ascontiguousarray(arr[:, 2]))


def per_source_bfs_numba(n, edges=None, arrays=None):
    W = words_for(n)
    ts, us, vs = arrays if arrays is not None else sorted_arrays(edges)
    R = _per_source_strict_bfs(n, W, ts, us, vs)
    return R


def strata_core(n, arrays, kernel_only_prep=None):
    """Production compiled core on one thread: (R, pairs, prep_s, kernel_s)."""
    from temporal_strata.packed_reachability import _empty_state
    from temporal_strata.parallel import (_set_identity_vec, prepare_stream_arrays,
                                          resolve_plan, run_prepared)
    t0 = time.perf_counter()
    prep = kernel_only_prep or prepare_stream_arrays(n, *arrays, directed=True,
                                                    same_time_mode="strict", prep_mode="off")
    t1 = time.perf_counter()
    R = _empty_state(n, n)
    _set_identity_vec(R, 0, n)
    run_prepared(R, prep, resolve_plan(R.shape[1], 1, layout="tiled", schedule="static"))
    t2 = time.perf_counter()
    return R, int(_nk._popcount_state(R)), t1 - t0, t2 - t1, prep


def dense_numpy_group(n, stream):
    """Dense boolean per-timestamp-group strict propagation (no bit packing).
    Returns the relation in the reference's [source, target] orientation."""
    from itertools import groupby
    R = np.eye(n, dtype=bool)  # R[v] accumulates the sources that reach v
    s = sorted(stream, key=lambda e: (e.t, e.u, e.v))
    for _, git in groupby(s, key=lambda e: e.t):
        grp = list(git)
        before = R.copy()
        for e in grp:
            R[e.v] |= before[e.u]
    return R.T.copy()  # -> [source, target], matching all_sources_temporal_bfs


def graphblas_static(n, stream):
    """Static reachability via GraphBLAS (compiled C): transitive closure of the
    timestamp-collapsed adjacency, computed with the efficient masked-frontier
    idiom (the same one used in ablation_graphblas.py). This is the *static
    reduction* (an upper bound on temporal reachability), included as the
    compiled-C honesty anchor; STRATA in closure mode computes the same object."""
    import graphblas as gb
    from graphblas import Matrix, semiring
    # pin GraphBLAS to one OpenMP thread. NUMBA_NUM_THREADS=1 does not
    # limit it, so the v10 M5 run used its default thread count; those values are
    # not reported as single-thread numbers.
    gb.ss.config["nthreads"] = 1
    rows = np.fromiter((e.u for e in stream), dtype=np.int64)
    cols = np.fromiter((e.v for e in stream), dtype=np.int64)
    vals = np.ones(len(rows), dtype=bool)
    A = Matrix.from_coo(rows, cols, vals, nrows=n, ncols=n, dup_op=gb.binary.lor)
    R = Matrix.from_coo(np.arange(n), np.arange(n), np.ones(n, dtype=bool), nrows=n, ncols=n)
    prev = -1
    while R.nvals != prev:
        prev = R.nvals
        R(mask=~R.S) << semiring.lor_land(R @ A)  # masked frontier expansion
    return R.nvals


def best_of(fn, reps=3):
    fn()
    b = float("inf"); r = None
    for _ in range(reps):
        t = time.perf_counter(); r = fn(); b = min(b, time.perf_counter() - t)
    return b, r


def ego_fb_512():
    edges = []
    with open(FB) as f:
        for line in f:
            u, v = map(int, line.split())
            if u < 512 and v < 512:
                edges.append((u, v))
    return 512, [TemporalEdge(t=i, u=u, v=v) for i, (u, v) in enumerate(edges)]


SYNTH = [("synth-PA $n{=}128$", 128, 1500, 400),
         ("synth-PA $n{=}256$", 256, 4000, 800),
         ("synth-PA $n{=}512$", 512, 12000, 1500),
         ("synth-PA $n{=}1024$", 1024, 30000, 4000),
         ("synth-PA $n{=}2048$", 2048, 60000, 8000)]


def run_stream(label, n, stream):
    edges_tuv = [(e.t, e.u, e.v) for e in stream]
    W = words_for(n)
    # ground truth (interpreted exact BFS) + timing
    t0 = time.perf_counter(); ref = all_sources_temporal_bfs(n, stream, same_time_mode="strict"); t_ref = time.perf_counter() - t0
    truth = int(ref.sum())
    # STRATA-py
    t_spy, Rspy = best_of(lambda: temporal_strata_packed(n, stream, same_time_mode="strict"))
    assert np.array_equal(packed_to_matrix(Rspy, n), ref), f"{label}: STRATA-py mismatch"
    # dense NumPy
    t_dn, Rdn = best_of(lambda: dense_numpy_group(n, stream))
    assert np.array_equal(Rdn, ref), f"{label}: dense-numpy mismatch"
    # STRATA-numba: production core from the same sorted arrays (v28, R79)
    arrays = sorted_arrays(edges_tuv)
    t_snb, (Rsnb, p_snb, _, _, prep) = best_of(lambda: strata_core(n, arrays))
    assert p_snb == truth, f"{label}: STRATA-numba pair mismatch ({p_snb} vs {truth})"
    assert np.array_equal(packed_to_matrix(Rsnb, n), ref), f"{label}: STRATA-numba mismatch"
    t_snk = min(strata_core(n, arrays, kernel_only_prep=prep)[3] for _ in range(5))
    # the early experimental kernel (whole-state snapshot per group), for comparison
    t_leg, (_, p_leg) = best_of(lambda: _nk.strata_strict_numba(n, edges_tuv))
    assert p_leg == truth, f"{label}: legacy kernel pair mismatch"
    # compiled per-source BFS (compiled reference), same sorted arrays
    t_cbfs, Rcb = best_of(lambda: per_source_bfs_numba(n, arrays=arrays))
    assert np.array_equal(packed_to_matrix(Rcb, n), ref), f"{label}: compiled-BFS mismatch"
    # GraphBLAS static reduction (compiled C) -- static pairs (>= temporal)
    try:  # GraphBLAS is optional; a missing package is recorded, not fatal
        t_gb, gb_pairs = best_of(lambda: graphblas_static(n, stream))
    except ImportError as exc:
        print(f"  GraphBLAS unavailable ({exc}); column left empty")
        t_gb, gb_pairs = None, None
    row = {
        "label": label, "n": n, "m": len(stream), "pairs": truth,
        "exact_bfs_ms": round(t_ref * 1000, 1),
        "dense_numpy_ms": round(t_dn * 1000, 1),
        "strata_py_ms": round(t_spy * 1000, 1),
        "strata_numba_ms": round(t_snb * 1000, 2),
        "strata_numba_kernel_ms": round(t_snk * 1000, 3),
        "strata_numba_legacy_ms": round(t_leg * 1000, 2),
        "strata_numba_core": "production tiled/static, touched-row snapshot (v28)",
        "compiled_bfs_ms": round(t_cbfs * 1000, 2),
        "graphblas_static_ms": None if t_gb is None else round(t_gb * 1000, 2),
        # actual pool size read back from GraphBLAS, not the requested value
        "graphblas_threads": None if t_gb is None else int(sys.modules["graphblas"].ss.config["nthreads"]),
        "graphblas_static_pairs": gb_pairs,
        "speedup_py_vs_exact": round(t_ref / t_spy, 1),
        "compiled_vs_compiled": round(t_cbfs / t_snk, 1),  # per-source-BFS / STRATA kernel, both Numba
    }
    print(f"{label}: STRATA-numba {row['strata_numba_ms']}ms vs compiled-BFS {row['compiled_bfs_ms']}ms "
          f"=> compiled-vs-compiled {row['compiled_vs_compiled']}x | GraphBLAS(static) {row['graphblas_static_ms']}ms")
    return row


def main():
    rows = []
    for label, n, m, ts in SYNTH:
        rows.append(run_stream(label, n, generate_temporal_stream(n=n, m=m, timestamps=ts, seed=7)))
    n, fb = ego_fb_512()
    rows.append(run_stream("ego-Facebook", n, fb))

    cvc = np.array([r["compiled_vs_compiled"] for r in rows], float)
    geo_cvc = float(np.exp(np.mean(np.log(cvc))))
    out = Path("results/perf/compiled_baselines.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": rows, "geomean_compiled_vs_compiled": round(geo_cvc, 2), "threads": thread_record()}, out.open("w"), indent=2)
    print(f"\ngeomean compiled-vs-compiled (per-source-BFS / STRATA, both Numba) = {geo_cvc:.2f}x")

    print("wrote", out)


if __name__ == "__main__":
    main()
