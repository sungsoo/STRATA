#!/usr/bin/env python3
"""Closure semantics on real data with bucketed timestamps.

The observed-timestamp datasets have almost unique timestamps, so closure and
strict coincide and the closure path (fixed-point sweeps, Proposition 2,
Lemma 4) is barely exercised. This harness buckets the timestamps (hour / day /
week by default, ``t // bucket``), then for every bucketed stream

* reports the groups (count, mean / max size);
* counts the sweeps the sequential closure loop needs per group (the last sweep
  confirms the fixed point, so sweeps - 1 are productive) and checks Lemma 4:
  for column partitions into 2 / 6 / 18 chunks, every chunk needs at most the
  sequential number of sweeps in every group, and the chunks together give the
  sequential state bit for bit;
* times strict and closure kernels (tiled / static) at every thread count after
  verifying each configuration bit for bit against the sequential compiled
  reference of the same semantics; every repeat sample is stored.

No number is written into the paper by this script; experiments/m5_numbers.py
reads its JSON after the M5 run. Usage (evaluation machine)::

  NUMBA_NUM_THREADS=18 python3 experiments/closure_grouped.py \\
      --threads 1,2,4,6,8,12,16,18 --repeats 5 --out results/real/closure_grouped.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from experiments.bench_common import REAL_SETS, DATA, samples_of  # noqa: E402
from temporal_strata.io import load_standard_arrays  # noqa: E402
from temporal_strata.packed_reachability import words_for  # noqa: E402
from temporal_strata.parallel import (  # noqa: E402
    _HAVE_NUMBA,
    _set_identity_vec,
    machine_info,
    njit,
    prepare_stream_arrays,
    resolve_plan,
    run_prepared,
    set_threads,
)

HERE = Path(__file__).resolve().parent
BUCKETS = {"hour": 3600, "day": 86400, "week": 604800}


@njit(cache=True)
def closure_sweeps(R, gptr, us, vs, c0, c1, sweeps):
    """Sequential closure over columns [c0, c1); sweeps[g] = sweeps of group g."""
    for g in range(gptr.shape[0] - 1):
        k = 0
        while True:
            k += 1
            changed = False
            for e in range(gptr[g], gptr[g + 1]):
                u = us[e]
                v = vs[e]
                for j in range(c0, c1):
                    x = R[u, j] & ~R[v, j]
                    if x != 0:
                        R[v, j] |= x
                        changed = True
            if not changed:
                break
        sweeps[g] = k


def bucketed(ts, bucket):
    return (np.asarray(ts, dtype=np.int64) // np.int64(bucket)).astype(np.int64)


def sweep_study(n, prep, chunk_counts=(2, 6, 18)):
    W = words_for(n)
    G = prep.gptr.shape[0] - 1
    R = np.zeros((n, W), dtype=np.uint64)
    _set_identity_vec(R, 0, n)
    seq = np.zeros(G, dtype=np.int64)
    closure_sweeps(R, prep.gptr, prep.us, prep.vs, 0, W, seq)
    out = {"groups": int(G), "sweeps_mean": float(seq.mean()) if G else 0.0,
           "sweeps_max": int(seq.max()) if G else 0,
           "groups_with_chain": int((seq > 2).sum()), "lemma4": []}
    for c in chunk_counts:
        c = min(c, W)
        bounds = (np.arange(c + 1) * W) // c
        Rc = np.zeros((n, W), dtype=np.uint64)
        _set_identity_vec(Rc, 0, n)
        worst = 0
        ok = True
        for i in range(c):
            sw = np.zeros(G, dtype=np.int64)
            closure_sweeps(Rc, prep.gptr, prep.us, prep.vs, int(bounds[i]), int(bounds[i + 1]), sw)
            ok &= bool(np.all(sw <= seq))
            worst = max(worst, int((sw - seq).max()) if G else 0)
        same = bool(np.array_equal(Rc, R))
        out["lemma4"].append({"chunks": int(c), "chunk_sweeps_le_sequential": ok,
                              "max_excess": worst, "bit_identical": same})
        if not (ok and same):
            raise AssertionError(f"Lemma 4 / determinacy check failed for {c} chunks")
    return out, R


def time_kernels(n, edges_b, threads, repeats, R_closure):
    W = words_for(n)
    rows = []
    preps = {m: prepare_stream_arrays(n, *edges_b, directed=True, same_time_mode=m,
                                      prep_mode="off") for m in ("strict", "closure")}
    plan1 = resolve_plan(W, 1, layout="rowmajor", schedule="static", chunks=1)
    refs = {"closure": R_closure}
    Rs = np.zeros((n, W), dtype=np.uint64)
    _set_identity_vec(Rs, 0, n)
    run_prepared(Rs, preps["strict"], plan1)
    refs["strict"] = Rs
    for T in threads:
        eff = set_threads(T)
        if eff != T:
            continue
        for mode in ("strict", "closure"):
            plan = resolve_plan(W, eff, layout="tiled", schedule="static", chunk_policy="threads")
            R = np.zeros((n, W), dtype=np.uint64)
            _set_identity_vec(R, 0, n)
            run_prepared(R, preps[mode], plan)
            assert np.array_equal(R, refs[mode]), f"{mode} diverged at T={eff}"

            def _k(mode=mode, plan=plan):
                Rk = np.zeros((n, W), dtype=np.uint64)
                _set_identity_vec(Rk, 0, n)
                run_prepared(Rk, preps[mode], plan)

            ss = samples_of(_k, repeats)
            rows.append({"threads": eff, "mode": mode, "kernel_s": min(ss), "kernel_samples_s": ss})
    pairs = {m: int(sum(bin(int(x)).count("1") for x in refs[m].ravel())) for m in refs}
    return rows, pairs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(REAL_SETS))
    ap.add_argument("--buckets", default="hour,day,week")
    ap.add_argument("--threads", default="1,2,4,6,8,12,16,18")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=str(HERE.parent / "results" / "real" / "closure_grouped.json"))
    args = ap.parse_args(argv)
    if not _HAVE_NUMBA:
        raise SystemExit("numba is required")
    threads = [int(x) for x in args.threads.split(",") if x]
    results = []
    for name in [x for x in args.datasets.split(",") if x]:
        n, ts, us, vs, _meta = load_standard_arrays(DATA / f"{name}.txt")
        for bname in [x for x in args.buckets.split(",") if x]:
            tb = bucketed(ts, BUCKETS[bname])
            prep_c = prepare_stream_arrays(n, tb, us, vs, directed=True, same_time_mode="closure",
                                           prep_mode="off")
            sizes = np.diff(prep_c.gptr)
            study, Rc = sweep_study(n, prep_c)
            rows, pairs = time_kernels(n, (tb, us, vs), threads, args.repeats, Rc)
            rec = {"dataset": name, "bucket": bname, "bucket_s": BUCKETS[bname], "n": int(n),
                   "m": int(len(us)), "W": int(words_for(n)),
                   "group_size_mean": float(sizes.mean()), "group_size_max": int(sizes.max()),
                   "pairs_strict": pairs["strict"], "pairs_closure": pairs["closure"],
                   **study, "rows": rows}
            results.append(rec)
            print(f"[{name}/{bname}] groups={study['groups']} mean|G|={rec['group_size_mean']:.1f} "
                  f"sweeps mean={study['sweeps_mean']:.2f} max={study['sweeps_max']} "
                  f"pairs strict={pairs['strict']} closure={pairs['closure']}")
    machine = machine_info()
    payload = {"machine": machine, "repeats": args.repeats, "thread_grid": threads,
               "pending": int(machine["cpu_count"]) <= 1, "results": results,
               "note": "every configuration verified bit-identical before timing; all samples kept"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
