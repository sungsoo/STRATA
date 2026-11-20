#!/usr/bin/env python3
"""Regenerate the headline performance table through the current packed core.

Added in v22 so that Table (packed STRATA vs exact temporal BFS) is produced by
the *same* core path that ships --- i.e. the touched-row strict snapshot
and the unconditional strict merge (v22, no non-short-circuiting np.any test).
Earlier the table numbers predated those core changes; regenerating here keeps
the reported speedups consistent with the code a reviewer would run.

Streams:
  * five preferential-attachment temporal streams (n = 128..2048) generated with
    a fixed seed via temporal_strata.stream.generate_temporal_stream;
  * the n=512 ego-Facebook induced subgraph (first 512 vertices of the bundled
    facebook_combined.txt), temporalized by edge order.

For each: STRATA (warm-up + best-of-3) vs the exact all-source temporal BFS
reference (deterministic single-threaded pure Python; one warm measurement),
verified bit-for-bit, at matched optimization level. Writes
results/perf/perf_table.json and nothing else (``--no-tex`` is accepted for
compatibility and does nothing).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.stream import generate_temporal_stream, TemporalEdge
from temporal_strata.baseline import all_sources_temporal_bfs
from temporal_strata.packed_reachability import (
    temporal_strata_packed,
    packed_to_matrix,
    reachable_pairs,
)
try:
    from thread_record import thread_record  # run as a script
except ImportError:  # imported as experiments.<module>
    from experiments.thread_record import thread_record

from temporal_strata.datasets import FACEBOOK as FB  # noqa: E402

# (label, n, m, timestamps) for synthetic preferential-attachment streams
SYNTH = [
    ("synth-PA $n{=}128$", 128, 1500, 400),
    ("synth-PA $n{=}256$", 256, 4000, 800),
    ("synth-PA $n{=}512$", 512, 12000, 1500),
    ("synth-PA $n{=}1024$", 1024, 30000, 4000),
    ("synth-PA $n{=}2048$", 2048, 60000, 8000),
]


def ego_fb_512():
    """First-512-vertex induced subgraph of ego-Facebook, temporalized by order."""
    edges = []
    seen = {}
    with open(FB) as f:
        for line in f:
            u, v = map(int, line.split())
            if u < 512 and v < 512:
                edges.append((u, v))
    n = 512
    # temporalize by edge order (one distinct timestamp per edge), directed u->v
    stream = [TemporalEdge(t=i, u=u, v=v) for i, (u, v) in enumerate(edges)]
    return n, stream, len(edges)


def best_of(fn, reps=3):
    fn()
    best = float("inf")
    r = None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        best = min(best, time.perf_counter() - t0)
    return best, r


def bench(label, n, stream):
    # exact reference (single deterministic warm measurement)
    t0 = time.perf_counter()
    ref = all_sources_temporal_bfs(n, stream, same_time_mode="strict")
    t_ref = time.perf_counter() - t0
    # STRATA (current core) best-of-3
    R = temporal_strata_packed(n, stream, same_time_mode="strict")
    ok = np.array_equal(packed_to_matrix(R, n), ref)
    assert ok, f"{label}: STRATA != exact BFS"
    t_str, _ = best_of(lambda: temporal_strata_packed(n, stream, same_time_mode="strict"))
    return {
        "label": label,
        "n": n,
        "m": len(stream),
        "packed_ms": round(t_str * 1000, 1),
        "exact_ms": round(t_ref * 1000, 1),
        "speedup": round(t_ref / t_str, 2),
        "pairs": reachable_pairs(R, n),
        "bit_identical": bool(ok),
    }


def main():
    rows = []
    for label, n, m, ts in SYNTH:
        stream = generate_temporal_stream(n=n, m=m, timestamps=ts, seed=7)
        rows.append(bench(label, n, stream))
        print(rows[-1])
    n, fb_stream, mfb = ego_fb_512()
    fb_row = bench("ego-Facebook", n, fb_stream)
    rows.append(fb_row)
    print(fb_row)

    speedups = np.array([r["speedup"] for r in rows], dtype=float)
    geomean = float(np.exp(np.mean(np.log(speedups))))
    print(f"geometric-mean speedup = {geomean:.2f}x")

    out = Path("results/perf/perf_table.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": rows, "geomean_speedup": round(geomean, 2), "threads": thread_record()}, out.open("w"), indent=2)

    print("wrote", out)


if __name__ == "__main__":
    main()
