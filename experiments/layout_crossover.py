#!/usr/bin/env python3
"""Row-major vs tiled crossover check for ``layout="auto"``.

Why
---
``layout="auto"`` picks tiled iff ``floor(W/L) < p`` (``L`` = cache-line words,
``p`` = threads): below that boundary the row-major state cannot give every
worker a whole cache line, so chunk borders share lines; above it row-major
already uses all cores and tiling only adds a gather/scatter. Whether the
measured crossover sits at this boundary is an empirical question.

How
---
For a fixed thread count ``p`` the script sweeps synthetic streams whose word
width ``W`` spans the boundary ``W = L * p`` (default: ``W`` from ``L*p/4`` to
``4*L*p``, edges per vertex held constant), times the strict kernel under the
row-major and tiled layouts with the same static chunking, verifies both
bit-identical to the sequential core before timing, and records for each ``W``
which layout was faster and what the rule would choose.

What it reports
---------------
* ``agreement``: fraction of sweep points where the rule picks the faster layout.
* ``measured_crossover_lines``: the smallest ``floor(W/L)`` from which row-major
  is faster at every larger sweep point (``None`` if never).
* ``suggested_rule``: if the measured crossover differs from ``p``, the rule
  ``floor(W/L) < c`` with the measured ``c``; the paper then states the
  corrected boundary rather than the analytical one.

Integrity: single-core hosts are marked PENDING (no crossover reported); the
machine and cache-line size are recorded verbatim; nothing here is a paper
number until it is run on the evaluation machine.

Usage
-----
  NUMBA_NUM_THREADS=<P> python3 experiments/layout_crossover.py \
      --threads 18 [--min-factor 0.25] [--max-factor 4] [--points 9] \
      [--edges-per-vertex 120] [--groups 400] [--repeats 5] \
      [--out results/real/layout_crossover.json]
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

from temporal_strata.packed_reachability import temporal_strata_packed, words_for
from temporal_strata.stream import arrays_to_edges, generate_temporal_arrays

try:
    from temporal_strata.parallel import (
        _set_identity_vec,
        cache_line_words,
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


def summarize(points, p):
    """Agreement and measured crossover from per-W timings (pure function)."""
    if not points:
        return {"agreement": None, "measured_crossover_lines": None, "suggested_rule": None}
    agree = sum(1 for x in points if x["rule_choice"] == x["faster"]) / len(points)
    pts = sorted(points, key=lambda x: x["W"])
    crossover = None
    for i, x in enumerate(pts):
        if all(y["faster"] == "rowmajor" for y in pts[i:]):
            crossover = x["lines"]
            break
    suggested = None
    if crossover is not None and crossover != p:
        suggested = f"tiled iff floor(W/L) < {crossover} (measured; analytical boundary p={p})"
    return {"agreement": agree, "measured_crossover_lines": crossover,
            "suggested_rule": suggested}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--min-factor", type=float, default=0.25)
    ap.add_argument("--max-factor", type=float, default=4.0)
    ap.add_argument("--points", type=int, default=9)
    ap.add_argument("--edges-per-vertex", type=int, default=120)
    ap.add_argument("--groups", type=int, default=400)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=str,
                    default=str(HERE.parent / "results" / "real" / "layout_crossover.json"))
    args = ap.parse_args()

    p = set_threads(args.threads)
    if p != args.threads:
        raise SystemExit(f"requested {args.threads} threads, pool allows {p}; "
                         f"set NUMBA_NUM_THREADS>={args.threads}")
    L = cache_line_words()
    machine = machine_info()
    pending = int(machine["cpu_count"]) <= 1
    boundary_W = L * p
    Ws = sorted({max(1, int(round(boundary_W * f)))
                 for f in np.geomspace(args.min_factor, args.max_factor, args.points)})
    print("machine:", machine["summary"])
    print(f"p={p} L={L} boundary W=L*p={boundary_W}; sweep W={Ws}")

    points = []
    for W in Ws:
        n = 64 * W
        m = args.edges_per_vertex * n
        arrays = generate_temporal_arrays(n, m, timestamps=args.groups, directed=True,
                                          seed=args.seed)
        ref = temporal_strata_packed(n, arrays_to_edges(*arrays), directed=True,
                                     same_time_mode="strict")
        prep = prepare_stream_arrays(n, *arrays, directed=True, same_time_mode="strict")
        times = {}
        for layout in ("rowmajor", "tiled"):
            # same chunk count for both layouts: the comparison isolates layout
            plan = resolve_plan(W, p, layout=layout, chunks=min(p, W))
            R = np.zeros((n, W), dtype=np.uint64)
            _set_identity_vec(R, 0, n)
            run_prepared(R, prep, plan)
            assert np.array_equal(R, ref), f"diverged W={W} layout={layout}"

            def _k(plan=plan):
                Rk = np.zeros((n, W), dtype=np.uint64)
                _set_identity_vec(Rk, 0, n)
                run_prepared(Rk, prep, plan)

            times[layout] = _best_of(_k, args.repeats, key=layout)
            times[layout + "_samples"] = SAMPLES[layout]
        rule = resolve_plan(W, p, layout="auto", line_words=L).resolved_layout
        faster = "rowmajor" if times["rowmajor"] <= times["tiled"] else "tiled"
        points.append({"W": W, "n": n, "m": m, "lines": W // L, "rule_choice": rule,
                       "faster": faster, "rowmajor_s": times["rowmajor"],
                       "tiled_s": times["tiled"],
                       "rowmajor_samples_s": times["rowmajor_samples"],
                       "tiled_samples_s": times["tiled_samples"]})
        print(f"  W={W:>5} floor(W/L)={W // L:>4} rule={rule:>8} faster={faster:>8} "
              f"rowmajor={times['rowmajor']:.4f}s tiled={times['tiled']:.4f}s")

    summary = None if pending else summarize(points, p)
    payload = {"machine": machine, "threads": p, "line_words": L,
               "boundary_W": boundary_W, "pending": pending,
               "points": points, "summary": summary}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}" + ("  [PENDING: single-core host]" if pending else ""))
    if summary:
        print("summary:", summary)


if __name__ == "__main__":
    main()
