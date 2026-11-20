#!/usr/bin/env python3
"""Memory-bandwidth ceiling (roofline) for the strict column kernel.

Why
---
The bit-packed OR kernel moves far more bytes than it computes on, so its
strong scaling is expected to flatten at the memory-bandwidth ceiling. A speedup
curve alone cannot say where that ceiling is; this script measures it on the
same machine and reports how close the kernel gets.

How
---
1. STREAM-triad bandwidth: ``a[i] = b[i] + s * c[i]`` over float64 arrays
   (Numba ``prange``), best-of-R after warm-up, at every thread count of the
   grid. Bytes per triad are ``3 * 8 * N`` (``4 * 8 * N`` with write-allocate,
   also recorded). The vendor's nominal peak is never used as the ceiling.
2. Kernel bytes: the touched-byte model of temporal_strata/traffic.py, an exact
   function of the prepared stream (Corollary 1's work count times 8 bytes).
3. Achieved kernel bandwidth = model bytes / measured kernel time, at the same
   thread counts and for each requested layout; every timed configuration is
   first verified bit-identical to the sequential core.
4. Ratio = achieved / STREAM at the same thread count, and achieved / best
   STREAM over all thread counts.

Integrity rules
---------------
* The model counts *touched* bytes; cache-resident reuse can make the ratio
  exceed 1. The JSON says so; never present such a ratio as DRAM efficiency.
* On a single-hardware-thread host the script still verifies correctness but
  marks the run PENDING and emits no ratios.
* Results are artifact-only (JSON + optional supplementary figure via
  experiments/plot_parallel.py); the machine is recorded verbatim.

Usage
-----
  NUMBA_NUM_THREADS=<P> python3 experiments/bandwidth_roofline.py \
      [--threads 1,2,4,6,8,12,18] [--array-mib 1024] [--repeats 5] \
      [--datasets CollegeMsg,BitcoinOTC,BitcoinOTCpos] [--synthetic] \
      [--layouts rowmajor,tiled] [--chunk-policy threads] [--large n:m:G[:seed]] \
      [--out results/real/bandwidth_roofline.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.io import load_standard_edges
from temporal_strata.packed_reachability import temporal_strata_packed, words_for
from temporal_strata.stream import generate_temporal_stream
from temporal_strata.traffic import strict_kernel_bytes, triad_bytes

try:
    from numba import njit, prange

    from temporal_strata.parallel import (
        _set_identity_vec,
        machine_info,
        max_threads,
        prepare_stream_arrays,
        resolve_plan,
        run_prepared,
        set_threads,
    )
except Exception as exc:  # pragma: no cover
    print(f"numba unavailable ({exc}); bandwidth measurement cannot run", file=sys.stderr)
    sys.exit(2)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "datasets_real_temporal" / "standard"


@njit(parallel=True, cache=True)
def _triad(a, b, c, s):
    for i in prange(a.shape[0]):
        a[i] = b[i] + s * c[i]


SAMPLES = {}  # last samples per call key


def _best_of(fn, repeats, key=None):
    fn()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    if key is not None:
        SAMPLES[key] = out
    return min(out)


def stream_triad(threads, array_mib, repeats):
    n_elems = int(array_mib * 2**20 // 8)
    a = np.zeros(n_elems)
    b = np.full(n_elems, 1.0)
    c = np.full(n_elems, 2.0)
    rows = []
    for T in threads:
        eff = set_threads(T)
        if eff != T:
            print(f"  STREAM threads={T} unavailable (max={max_threads()}); skipping")
            continue
        t = _best_of(lambda: _triad(a, b, c, 3.0), repeats, key="triad")
        assert a[0] == 7.0 and a[-1] == 7.0
        rows.append({"threads": eff, "seconds": t, "samples_s": SAMPLES["triad"],
                     "GBps": triad_bytes(n_elems) / t / 1e9,
                     "GBps_write_allocate": triad_bytes(n_elems, True) / t / 1e9})
        print(f"  STREAM T={eff:>3} {rows[-1]['GBps']:.1f} GB/s")
    return {"array_mib": array_mib, "n_elems": n_elems, "rows": rows}


def kernel_bandwidth(name, n, edges, threads, layouts, repeats, chunk_policy="threads"):
    from experiments.bench_common import num_edges, reference_state

    print(f"[{name}] n={n} m={num_edges(edges)}")

    ref, ref_kind = reference_state(n, edges, "strict", "auto")
    prep = prepare_stream_arrays(n, *edges, directed=True, same_time_mode="strict")
    W = words_for(n)
    rows = []
    for T in threads:
        eff = set_threads(T)
        if eff != T:
            continue
        for layout in layouts:
            plan = resolve_plan(W, eff, layout=layout, chunk_policy=chunk_policy)
            R = np.zeros((n, W), dtype=np.uint64)
            _set_identity_vec(R, 0, n)
            run_prepared(R, prep, plan)
            assert np.array_equal(R, ref), f"diverged at T={eff} layout={layout}"

            def _k(plan=plan):
                Rk = np.zeros((n, W), dtype=np.uint64)
                _set_identity_vec(Rk, 0, n)
                run_prepared(Rk, prep, plan)

            t = _best_of(_k, repeats, key="kernel")
            model = strict_kernel_bytes(prep.num_edges, prep.dirty_total, n, W,
                                        layout=plan.resolved_layout)
            rows.append({"threads": eff, "layout": layout,
                         "resolved_layout": plan.resolved_layout,
                         "plan": plan.as_dict(), "kernel_s": t, "samples_s": SAMPLES["kernel"],
                         "model": model,
                         "achieved_GBps": model["total_bytes"] / t / 1e9})
            print(f"  T={eff:>3} {layout:>8} kernel={t:.4f}s "
                  f"touched={model['total_bytes'] / 1e9:.3f} GB "
                  f"achieved={rows[-1]['achieved_GBps']:.1f} GB/s")
    return {"dataset": name, "n": n, "m": num_edges(edges), "W": W, "reference": ref_kind,
            "state_bytes": int(n) * W * 8,
            "num_directed_edges": prep.num_edges, "dirty_total": prep.dirty_total,
            "verified_bit_identical": True, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=str, default=None)
    ap.add_argument("--array-mib", type=float, default=1024.0,
                    help="MiB per STREAM array (choose >= 4x the last-level cache)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--datasets", type=str, default="CollegeMsg,BitcoinOTC,BitcoinOTCpos")
    ap.add_argument("--synthetic", action="store_true",
                    help="also run the n=8000, 1e6-edge synthetic stream")
    ap.add_argument("--layouts", type=str, default="rowmajor,tiled")
    ap.add_argument("--chunk-policy", type=str, default="threads", choices=["threads", "auto"],
                    help="threads: min(p, W) chunks (harness protocol, R8); auto: library rule")
    ap.add_argument("--large", action="append", default=[],
                    help="extra synthetic stream n:m:groups[:seed] (repeatable; state "
                         "larger than the last-level cache, so DRAM bandwidth is exercised)")
    ap.add_argument("--out", type=str,
                    default=str(HERE.parent / "results" / "real" / "bandwidth_roofline.json"))
    args = ap.parse_args()

    hw = max_threads()
    if args.threads:
        grid = sorted({int(x) for x in args.threads.split(",")})
    else:
        grid, t = [], 1
        while t <= hw:
            grid.append(t)
            t *= 2
        if grid[-1] != hw:
            grid.append(hw)
    layouts = [x for x in args.layouts.split(",") if x]
    machine = machine_info()
    pending = int(machine["cpu_count"]) <= 1
    print("machine:", machine["summary"])

    stream = stream_triad(grid, args.array_mib, args.repeats)
    from experiments.bench_common import select_datasets_arrays

    # array-native datasets; "synthetic" is the same n=8000, 1e6-edge stream
    sets = [(nm, n, e) for nm, n, e, _load in select_datasets_arrays(
        [x for x in args.datasets.split(",") if x], args.large,
        include_default_synth=args.synthetic)]
    kernels = [kernel_bandwidth(nm, n, e, grid, layouts, args.repeats, args.chunk_policy)
               for nm, n, e in sets]

    best_stream = max((r["GBps"] for r in stream["rows"]), default=None)
    by_t = {r["threads"]: r["GBps"] for r in stream["rows"]}
    if not pending:
        for k in kernels:
            for r in k["rows"]:
                r["ratio_same_threads"] = r["achieved_GBps"] / by_t[r["threads"]]
                r["ratio_best_stream"] = r["achieved_GBps"] / best_stream
    payload = {
        "machine": machine, "thread_grid": grid, "pending": pending,
        "note": ("achieved_GBps uses touched bytes (traffic.py); cache reuse can make "
                 "ratios exceed 1, which indicates cache residency, not DRAM efficiency"),
        "stream_triad": stream, "best_stream_GBps": best_stream, "kernels": kernels,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}" + ("  [PENDING: single-core host, no ratios]" if pending else ""))


if __name__ == "__main__":
    main()
