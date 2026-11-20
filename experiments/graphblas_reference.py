#!/usr/bin/env python3
"""Multithreaded GraphBLAS as a reference point.

**Not the same problem.** SuiteSparse:GraphBLAS (via ``python-graphblas``)
computes the *static* transitive closure of the timestamp-collapsed graph, an
easier object than temporal reachability. The paper reports it only as a
reference point for a mature multithreaded sparse library on the same machine,
never as a same-problem competitor (the external-baseline rule of R13 applies:
only the same relation would be timed as a competitor).

Protocol per real dataset:
  1. collapse timestamps; compute the static closure with STRATA (closure
     semantics on one artificial timestamp) and with GraphBLAS (repeated
     ``R <- R or R*A`` over the boolean semiring until no new pair);
  2. require both closures to be equal entry for entry before any timing;
  3. time GraphBLAS at every thread count (``gb.ss.config["nthreads"]``) and the
     STRATA closure kernel at the same counts; keep all samples.

Usage (evaluation machine; the package is optional)::

  pip install python-graphblas
  NUMBA_NUM_THREADS=18 python3 experiments/graphblas_reference.py \\
      --threads 1,2,4,6,8,12,16,18 --repeats 5 --out results/real/graphblas_reference.json

If ``graphblas`` cannot be imported the script writes nothing and exits 3.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from experiments.bench_common import DATA, REAL_SETS, samples_of  # noqa: E402
from experiments.thread_record import thread_record  # noqa: E402
from temporal_strata.io import load_standard_arrays  # noqa: E402
from temporal_strata.packed_reachability import packed_to_matrix, words_for  # noqa: E402
from temporal_strata.parallel import (  # noqa: E402
    _set_identity_vec,
    machine_info,
    prepare_stream_arrays,
    resolve_plan,
    run_prepared,
    set_threads,
)

HERE = Path(__file__).resolve().parent


def strata_static_closure(n, us, vs, threads=1):
    """STRATA closure on one artificial timestamp = static closure (packed)."""
    ts0 = np.zeros(len(us), dtype=np.int64)
    prep = prepare_stream_arrays(n, ts0, us, vs, directed=True, same_time_mode="closure",
                                 prep_mode="off")
    W = words_for(n)
    plan = resolve_plan(W, threads, layout="tiled", schedule="static", chunk_policy="threads")

    def run():
        R = np.zeros((n, W), dtype=np.uint64)
        _set_identity_vec(R, 0, n)
        run_prepared(R, prep, plan)
        return R
    return run


def graphblas_closure_fn(gb, n, us, vs):
    """Closure matrix C with C[s, v] = 1 iff s reaches v (self included)."""
    # boolean scalar values: duplicates carry the same value, so no dup_op
    # (python-graphblas rejects dup_op with a scalar value; v9 M5 session)
    A = gb.Matrix.from_coo(np.asarray(us, dtype=np.int64), np.asarray(vs, dtype=np.int64), True,
                           nrows=n, ncols=n, dtype=bool)
    I = gb.Matrix.from_coo(np.arange(n), np.arange(n), True, nrows=n, ncols=n, dtype=bool)

    def run():
        R = I.dup()
        while True:
            nv = R.nvals
            R(accum=gb.binary.lor) << R.mxm(A, gb.semiring.lor_land)
            if R.nvals == nv:
                return R
    return run


def dense_closure_fn(n, us, vs):
    """NumPy fallback with the same contract (used by the tests)."""
    def run():
        C = np.eye(n, dtype=bool)
        A = np.zeros((n, n), dtype=bool)
        A[us, vs] = True
        while True:
            nxt = C | ((C.astype(np.uint8) @ A.astype(np.uint8)) > 0)
            if np.array_equal(nxt, C):
                return C
            C = nxt
    return run


def to_dense(C, n):
    if isinstance(C, np.ndarray):
        return C
    r, c, _v = C.to_coo()
    D = np.zeros((n, n), dtype=bool)
    D[r, c] = True
    return D


def verify(n, R_strata, C):
    """STRATA packed state (R[v] = sources reaching v) vs closure C[s, v]."""
    M = packed_to_matrix(R_strata, n).astype(bool)   # M[s, v]
    return bool(np.array_equal(M, to_dense(C, n)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(REAL_SETS))
    ap.add_argument("--threads", default="1,2,4,6,8,12,16,18")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=str(HERE.parent / "results" / "real" / "graphblas_reference.json"))
    args = ap.parse_args(argv)
    try:
        import graphblas as gb
    except Exception as exc:
        print(f"GRAPHBLAS UNAVAILABLE: {exc}")
        return 3
    threads = [int(x) for x in args.threads.split(",") if x]
    results = []
    for name in [x for x in args.datasets.split(",") if x]:
        n, _ts, us, vs, _meta = load_standard_arrays(DATA / f"{name}.txt")
        R = strata_static_closure(n, us, vs, 1)()
        gbf = graphblas_closure_fn(gb, n, us, vs)
        if not verify(n, R, gbf()):
            raise AssertionError(f"{name}: GraphBLAS closure differs from STRATA static closure")
        rows = []
        for T in threads:
            eff = set_threads(T)
            if eff != T:
                continue
            gb.ss.config["nthreads"] = eff
            g_s = samples_of(gbf, args.repeats)
            s_s = samples_of(strata_static_closure(n, us, vs, eff), args.repeats)
            rows.append({"threads": eff, "graphblas_threads_actual": int(gb.ss.config["nthreads"]),
                         "graphblas_s": min(g_s), "graphblas_samples_s": g_s,
                         "strata_static_s": min(s_s), "strata_static_samples_s": s_s})
            print(f"[{name}] T={eff:>2} graphblas={min(g_s):.4f}s strata-static={min(s_s):.4f}s")
        results.append({"dataset": name, "n": int(n), "m": int(len(us)),
                        "static_pairs": int(packed_to_matrix(R, n).sum()),
                        "verified_equal": True, "rows": rows})
    machine = machine_info()
    payload = {"machine": machine, "graphblas_version": getattr(gb, "__version__", "unknown"),
               "problem": "static transitive closure (NOT temporal reachability)",
               "repeats": args.repeats, "results": results, "threads": thread_record()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
