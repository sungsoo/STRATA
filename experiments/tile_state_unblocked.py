#!/usr/bin/env python3
"""Unblocked runs with a fresh, a pre-written, and no shared state (v34, R102).

Why
---
On sx-stackoverflow the block kernel lost time past six threads because the tiles
were scattered back into a freshly allocated shared state, whose every page all
threads write for the first time inside the kernel (v31, R90). The unblocked tiled
kernel behind the paper's large synthetic streams (Table 8, Fig. 6) also scatters
its tiles into a freshly allocated state once per run, so the same cost may limit
its scaling. This harness measures that directly.

How
---
For each stream (``--large n:m:groups[:seed]``, the generator of
experiments/parallel_scaling.py) and each thread count p, three variants run the
same strict relation, tiled layout, static schedule, p chunks:
  * ``shared``     the production path: fresh zero state + identity, run_prepared;
  * ``prewritten`` the same, but every page of the state is written before the
                   clock (diagnostic: isolates first writes, not a remedy);
  * ``tiles``      the tiles are the state (parallel.run_block_tiles with one block
                   of width n): identity bits set inside the timed kernel, no
                   shared state, no scatter.
Checks before any time is kept: all variants give the same pair count at every p,
and sampled source columns of ``tiles`` equal those of ``shared``. Each variant is
warmed up once, then timed ``--repeats`` times (median kept, all samples stored).

What
---
    NUMBA_NUM_THREADS=18 python3 experiments/tile_state_unblocked.py \
        --large 50000:1000000:400 --large 100000:2000000:400 \
        --threads 1,6,12,18 --repeats 3 --out results/real/tile_state_unblocked.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

VARIANTS = ("shared", "prewritten", "tiles")


def _column(R, s):
    w, b = s >> 6, np.uint64(1) << np.uint64(s & 63)
    return (R[:, w] & b) != 0


def run_variant(variant, prep, n, p, sample):
    """Returns (seconds, pairs, sample columns k x n)."""
    from temporal_strata.packed_reachability import words_for
    from temporal_strata.parallel import (_set_identity_vec, resolve_plan, run_block_tiles,
                                          run_prepared)
    W = words_for(n)
    if variant == "tiles":
        t0 = time.perf_counter()
        pairs, cols, _ = run_block_tiles(prep, 0, n, p, sample_local=list(sample))
        return time.perf_counter() - t0, int(pairs), cols
    if variant == "shared":
        R = np.zeros((n, W), dtype=np.uint64)
    else:
        R = np.empty((n, W), dtype=np.uint64)
        R.fill(0)                                   # every page written, outside the clock
    _set_identity_vec(R, 0, n)
    plan = resolve_plan(W, p, layout="tiled", schedule="static", chunks=p)
    t0 = time.perf_counter()
    run_prepared(R, prep, plan)
    dt = time.perf_counter() - t0
    pairs = int(np.bitwise_count(R).sum())
    cols = np.stack([_column(R, int(s)) for s in sample]) if len(sample) else np.zeros((0, n), bool)
    return dt, pairs, cols


def measure(name, n, edges, threads, repeats, set_threads, n_samples=16, variants=VARIANTS):
    from temporal_strata.parallel import prepare_stream_arrays
    prep = prepare_stream_arrays(n, *edges, directed=True, same_time_mode="strict")
    rng = np.random.default_rng(11)
    sample = np.sort(rng.choice(n, size=min(n_samples, n), replace=False))
    rows, ref = [], {}
    for p in threads:
        eff = set_threads(p)
        if eff != p:
            print(f"  threads={p} unavailable; skipped")
            continue
        for v in variants:
            samples = []
            for rep in range(repeats + 1):
                dt, pairs, cols = run_variant(v, prep, n, eff, sample)
                if rep == 0:
                    ref.setdefault("pairs", pairs)
                    if pairs != ref["pairs"]:
                        raise SystemExit(f"{name} {v} p={eff}: pair count differs")
                    if v == "shared":
                        ref.setdefault("cols", cols)
                    if "cols" in ref and not np.array_equal(cols, ref["cols"]):
                        raise SystemExit(f"{name} {v} p={eff}: sampled columns differ")
                else:
                    samples.append(dt)
            med = statistics.median(samples)
            rows.append({"variant": v, "threads": eff, "seconds_samples": samples,
                         "seconds_median": med})
            print(f"  {name} {v:10s} p={eff}: median {med:.3f} s")
    return {"dataset": name, "n": int(n), "m": int(prep.num_edges), "pairs": ref["pairs"],
            "sampled_sources": [int(x) for x in sample], "rows": rows}


def main(argv=None):
    from bench_common import select_datasets_arrays
    from temporal_strata.parallel import machine_info, set_threads
    try:
        from thread_record import thread_record
    except ImportError:
        from experiments.thread_record import thread_record
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="append", default=[])
    ap.add_argument("--threads", default="1,6,12,18")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(HERE.parent / "results" / "real" / "tile_state_unblocked.json"))
    args = ap.parse_args(argv)
    threads = [int(x) for x in args.threads.split(",") if x]
    set_threads(max(threads))
    sets = select_datasets_arrays([], args.large, include_default_synth=False)
    results = [measure(nm, n, e, threads, args.repeats, set_threads) for nm, n, e, _ in sets]
    payload = {"machine": machine_info(), "threads_record": thread_record(), "repeats": args.repeats,
               "variants": list(VARIANTS), "results": results,
               "note": "prewritten is diagnostic (page writes outside the clock); tiles includes "
                       "identity setup inside the clock"}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
