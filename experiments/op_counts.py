#!/usr/bin/env python3
"""Elementary-step counts of compiled per-source BFS and compiled STRATA (v34).

Why
---
Packing 64 sources per word bounds the gain of STRATA's merges at about 64x, yet
the measured compiled-versus-compiled kernel ratios on the observed-timestamp data
are 165-238x. This script separates the two factors without any timing: how many
elementary steps each kernel performs on the same input (machine independent), and
therefore how much of the measured ratio is left for the cost of one step.

How
---
Both kernels are exactly those timed by experiments/compiled_real.py.
  * per-source strict BFS (``per_source_strict_bfs_grouped``), per source: n
    resets, one check of every edge (m), one commit per newly reached vertex, and
    a final pass over n vertices to write the row. Steps = n*m + 2*n*n + pairs,
    where ``pairs`` (reachable pairs, sources included) is the total commits plus
    the n sources.
  * STRATA (``run_prepared``, one thread, row-major, one chunk), word operations:
    one W-word merge per edge and one W-word copy per touched row of the strict
    snapshot. Word ops = W * (m + snapshot rows).
The counts are deterministic functions of the input; no clock is read. The ratio
of steps is compared with the measured kernel ratio of results/real/compiled_real.json
(M5 Max), and their quotient is the cost of one BFS step relative to one STRATA
word operation on that machine.

What
---
    python3 experiments/op_counts.py --out results/real/op_counts.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def counts(n, edges, pairs):
    from temporal_strata.packed_reachability import words_for
    from temporal_strata.parallel import prepare_stream_arrays
    prep = prepare_stream_arrays(n, *edges, directed=True, same_time_mode="strict")
    m = int(prep.num_edges)
    W = int(words_for(n))
    snap_rows = int(prep.dptr[-1])
    bfs_steps = n * m + 2 * n * n + int(pairs)
    strata_ops = W * (m + snap_rows)
    return {"n": int(n), "m": m, "W": W, "groups": int(prep.gptr.shape[0] - 1),
            "snapshot_rows": snap_rows, "pairs": int(pairs),
            "bfs_edge_checks": n * m, "bfs_vertex_steps": 2 * n * n,
            "bfs_steps": bfs_steps, "strata_merge_words": W * m,
            "strata_snapshot_words": W * snap_rows, "strata_word_ops": strata_ops,
            "step_ratio": bfs_steps / strata_ops}


def main(argv=None):
    from bench_common import select_datasets_arrays
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="CollegeMsg,BitcoinOTC,BitcoinOTCpos")
    ap.add_argument("--compiled", default=str(HERE.parent / "results" / "real" / "compiled_real.json"))
    ap.add_argument("--out", default=str(HERE.parent / "results" / "real" / "op_counts.json"))
    args = ap.parse_args(argv)
    cr = json.loads(Path(args.compiled).read_text())
    meas = {r["dataset"]: r for r in cr["rows"]}
    rows = []
    for name, n, edges, _ in select_datasets_arrays(
            [x for x in args.datasets.split(",") if x], [], include_default_synth=False):
        if name not in meas:
            raise SystemExit(f"{name}: no measured row in {args.compiled}")
        mr = meas[name]
        if mr["n"] != n:
            raise SystemExit(f"{name}: n differs from the measured run")
        r = counts(n, edges, mr["pairs"])
        if r["m"] != mr["m"]:
            raise SystemExit(f"{name}: m differs from the measured run")
        r["dataset"] = name
        r["measured_kernel_ratio"] = mr["kernel_ratio_bfs_over_strata"]
        r["cost_per_step_ratio"] = r["measured_kernel_ratio"] / r["step_ratio"]
        rows.append(r)
        print(f"{name}: steps BFS/STRATA {r['step_ratio']:.1f}x, measured {r['measured_kernel_ratio']:.0f}x, "
              f"per-step cost {r['cost_per_step_ratio']:.1f}x")
    Path(args.out).write_text(json.dumps({"rows": rows, "machine_independent": True,
                                          "measured_from": Path(args.compiled).name}, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
