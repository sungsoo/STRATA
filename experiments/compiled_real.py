#!/usr/bin/env python3
"""Compiled-vs-compiled on the real-timestamp datasets.

Why
---
The paper's headline speedups on the real datasets (894x-2782x) are measured
against the *interpreted* per-source reference. The algorithmic effect alone is
the ratio with both sides compiled by the same tool, which the paper reports
only for the synthetic streams (4.4x geometric mean). This script measures it
on CollegeMsg, Bitcoin-OTC, and Bitcoin-OTC(+), on one machine.

How
---
* Both sides are Numba-compiled, single-threaded, and share the same sorted,
  grouped edge arrays:
    - STRATA: the column kernel of temporal_strata/parallel.py run with one
      thread and one chunk (identical to the sequential compiled core, with the
      touched-row snapshot);
    - per-source strict temporal BFS: one traversal per source with deferred
      within-group commits (same algorithm as experiments/compiled_baselines.py).
* Each result is compared bit-for-bit with the interpreted reference before
  timing; timing is warm-up + best-of-R.
* Kernel time and end-to-end time are reported separately: STRATA's end-to-end
  adds the full preprocessing (incl. its touched-row plan), the BFS's adds the
  shared sort / flatten / group phases only.

Integrity: one machine per run; the JSON records it; no ratio is ever formed
between machines.

Usage
-----
  python3 experiments/compiled_real.py [--repeats 5] \
      [--datasets CollegeMsg,BitcoinOTC,BitcoinOTCpos] [--large n:m:G[:seed]] \
      [--out results/real/compiled_real.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.packed_reachability import reachable_pairs, words_for  # noqa: E402
from experiments.bench_common import (  # noqa: E402
    best_of,
    samples_of,
    num_edges,
    reference_state,
    select_datasets_arrays,
)

try:
    from numba import njit

    from temporal_strata.parallel import (
        _set_identity_vec,
        machine_info,
        prepare_stream_arrays,
        resolve_plan,
        run_prepared,
        set_threads,
    )
except Exception as exc:  # pragma: no cover
    print(f"numba unavailable ({exc})", file=sys.stderr)
    sys.exit(2)

HERE = Path(__file__).resolve().parent


@njit(cache=True)
def per_source_strict_bfs_grouped(n, W, gptr, us, vs):
    """Sequential compiled per-source strict temporal BFS over grouped arrays."""
    R = np.zeros((n, W), dtype=np.uint64)
    G = gptr.shape[0] - 1
    reached = np.zeros(n, dtype=np.uint8)
    pend = np.empty(n, dtype=np.int64)
    for s in range(n):
        for x in range(n):
            reached[x] = 0
        reached[s] = 1
        for g in range(G):
            np_ = 0
            for k in range(gptr[g], gptr[g + 1]):
                u = us[k]
                v = vs[k]
                if reached[u] == 1 and reached[v] == 0:
                    pend[np_] = v
                    np_ += 1
            for q in range(np_):
                reached[pend[q]] = 1
        word = s >> 6
        bit = np.uint64(1) << np.uint64(s & 63)
        for v in range(n):
            if reached[v] == 1:
                R[v, word] |= bit
    return R


def strata_compiled_single(prep, n):
    W = words_for(n)
    plan = resolve_plan(W, 1, layout="rowmajor", schedule="static", chunks=1)
    R = np.zeros((n, W), dtype=np.uint64)
    _set_identity_vec(R, 0, n)
    run_prepared(R, prep, plan)
    return R


def measure(name, n, edges, repeats, load_s=None):
    """``edges``: (t, u, v) arrays (array-native path, R17)."""
    print(f"[{name}] n={n} m={num_edges(edges)}")
    ref, ref_kind = reference_state(n, edges, "strict", "auto")
    prep_cold = prepare_stream_arrays(n, *edges, directed=True, same_time_mode="strict")
    prep = min((prepare_stream_arrays(n, *edges, directed=True, same_time_mode="strict")
                for _ in range(max(1, repeats))), key=lambda p: p.timings["total_s"])
    W = words_for(n)
    set_threads(1)

    Rs = strata_compiled_single(prep, n)
    assert np.array_equal(Rs, ref), f"{name}: compiled STRATA diverged"
    Rb = per_source_strict_bfs_grouped(n, W, prep.gptr, prep.us, prep.vs)
    assert np.array_equal(Rb, ref), f"{name}: compiled per-source BFS diverged"

    s_strata = samples_of(lambda: strata_compiled_single(prep, n), repeats)
    s_bfs = samples_of(lambda: per_source_strict_bfs_grouped(n, W, prep.gptr, prep.us, prep.vs),
                       repeats)
    t_strata, t_bfs = min(s_strata), min(s_bfs)
    ph = prep.timings
    shared = ph["sort_validate_s"] + ph["flatten_s"] + ph["group_boundaries_s"]
    row = {
        "dataset": name, "n": n, "m": num_edges(edges), "W": W,
        "stream_path": "arrays", "load_s": load_s,
        "pairs": int(reachable_pairs(ref, n)), "reference": ref_kind,
        "verified_bit_identical": True,
        "strata_kernel_s": t_strata, "bfs_kernel_s": t_bfs,
        "strata_kernel_samples_s": s_strata, "bfs_kernel_samples_s": s_bfs,
        "strata_end_to_end_s": t_strata + ph["total_s"],
        "bfs_end_to_end_s": t_bfs + shared,
        "prep_phases_s": ph, "prep_cold_s": prep_cold.timings["total_s"],
        "prep_timing": "best-of-R after a cold call (cold call includes JIT compilation)",
    }
    row["kernel_ratio_bfs_over_strata"] = t_bfs / t_strata
    row["end_to_end_ratio_bfs_over_strata"] = row["bfs_end_to_end_s"] / row["strata_end_to_end_s"]
    print(f"  STRATA kernel={t_strata:.4f}s  BFS kernel={t_bfs:.4f}s  "
          f"kernel ratio={row['kernel_ratio_bfs_over_strata']:.2f}  "
          f"e2e ratio={row['end_to_end_ratio_bfs_over_strata']:.2f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--datasets", type=str, default="CollegeMsg,BitcoinOTC,BitcoinOTCpos")
    ap.add_argument("--large", action="append", default=[])
    ap.add_argument("--out", type=str,
                    default=str(HERE.parent / "results" / "real" / "compiled_real.json"))
    args = ap.parse_args()
    machine = machine_info()
    print("machine:", machine["summary"])
    sets = select_datasets_arrays([x for x in args.datasets.split(",") if x], args.large,
                                  include_default_synth=False)
    rows = [measure(nm, n, e, args.repeats, load_s) for nm, n, e, load_s in sets]
    k = [r["kernel_ratio_bfs_over_strata"] for r in rows]
    payload = {
        "machine": machine, "repeats": args.repeats, "rows": rows,
        "geomean_kernel_ratio": math.exp(sum(map(math.log, k)) / len(k)) if k else None,
        "note": "single machine; both sides Numba-compiled, single-threaded; "
                "bit-identity verified before timing",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
