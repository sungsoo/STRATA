#!/usr/bin/env python3
"""Multicore strong-scaling measurement for column-partitioned STRATA (TPDS).

What this measures
------------------
1. STRATA-parallel (temporal_strata.parallel.temporal_strata_parallel), the
   communication-free column-partitioned kernel, at thread counts
   T in {1, 2, 4, ..., max}, on:
     * the three real-timestamp datasets (CollegeMsg, Bitcoin-OTC, Bitcoin-OTC+),
     * one dense synthetic DAG stream (labelled synthetic) whose large timestamp
       groups stress the intra-group path (n=8000, 10^6 edges, 200 groups).
2. A *parallel* compiled per-source strict temporal BFS baseline (prange over
   sources) at the same thread grid -- the fair parallel-vs-parallel
   competitor: both sides are Numba-compiled and both are embarrassingly
   parallel along their natural axis (columns vs sources).

Every timed configuration is first verified BIT-IDENTICAL against the
sequential interpreted core; timing is warm-up + best-of-R (default R=3),
kernel-phase and end-to-end reported separately (the preprocessing phase is
sequential and identical across T; reporting both makes the Amdahl fraction
explicit instead of hiding it).

Integrity rules (non-negotiable)
--------------------------------
* If this host exposes a single hardware thread, the script records the
  machine info and STILL verifies correctness, but REFUSES to emit a speedup
  table: single-core "scaling" numbers would be noise presented as evidence.
  The emitted LaTeX table then contains PENDING dashes.
* The machine that produced any number is recorded verbatim in the JSON and in
  the LaTeX table footnote. No cross-machine speedups.

Realization sweep
--------------------------------------------------
Every thread count is timed for each requested memory layout
(``--layouts rowmajor,tiled,auto``) and chunk schedule
(``--schedules static,dynamic``); the resolved plan (layout, schedule, chunk
count, cache-line words, rule) is recorded in every JSON row next to its
timing, together with the full machine description (incl. Apple-silicon
performance levels). The LaTeX table is emitted for one configuration
(``--table-config``); the JSON keeps all of them.

STRATA-v3 additions
-------------------------------------
* ``--chunk-policy threads`` (default) restores the v1 measurement protocol:
  ``p`` chunks (static) or ``oversub * p`` (dynamic), capped only by ``W``.
  ``--chunk-policy auto`` measures the library's automatic rule instead, which
  caps row-major chunks at ``floor(W/L)`` (one cache line per chunk).
* ``--plan-only`` prints, without timing, the resolved plan (layout, chunks,
  idle workers) for every dataset x threads x layout x schedule; run it on the
  evaluation machine before the measurement session.
* ``--large n:m:groups[:seed]`` (repeatable) adds synthetic streams with a
  large word width; ``--block-bits B`` runs datasets with ``n > B`` source-
  blocked (each block is one kernel launch over ``W_beta`` columns);
  ``--reference`` chooses the bit-identity reference (interpreted NumPy core,
  or the independent compiled full-copy kernel for large streams);
  ``--baseline-max-work`` skips the per-source BFS baseline when ``n * m``
  exceeds it (the skip and its reason are recorded).
* Every dataset records the sequential preprocessing phases (sort, flatten,
  group boundaries, touched-row plan) next to the kernel time (Amdahl term).

Usage
-----
  NUMBA_NUM_THREADS=<P> python3 experiments/parallel_scaling.py \
      [--repeats 3] [--threads 1,2,4,8] [--skip-synthetic] [--skip-baseline] \
      [--layouts rowmajor,tiled,auto] [--schedules static,dynamic] [--oversub 4] \
      [--table-config tiled:static] [--chunk-policy threads] [--plan-only] \
      [--large 100000:2000000:400] [--block-bits 65536] [--reference auto] \
      [--baseline-max-work 5e10] [--skip-real] \
      [--out results/real/parallel_scaling.json] \
      [--tex-out results/real/table_parallel_scaling.tex]

The emitted ``table_parallel_scaling.tex`` is a LaTeX rendering of the JSON
results; the script writes only the paths given on the command line.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.io import load_standard_edges
from temporal_strata.packed_reachability import (
    _scatter_block,
    reachable_pairs,
    temporal_strata_packed,
    words_for,
)
from temporal_strata.stream import generate_temporal_stream
from experiments.bench_common import (
    REFERENCES,
    num_edges,
    reference_state,
    select_datasets_arrays,
)

try:
    import numba
    from numba import njit, prange

    from temporal_strata.parallel import (
        CHUNK_POLICIES,
        LAYOUTS,
        SCHEDULES,
        _set_identity_vec,
        machine_info,
        max_threads,
        prepare_stream_arrays,
        resolve_plan,
        run_prepared,
        set_threads,
    )
except Exception as exc:  # pragma: no cover
    print(f"numba unavailable ({exc}); parallel scaling cannot run", file=sys.stderr)
    sys.exit(2)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "datasets_real_temporal" / "standard"
REAL_SETS = ["CollegeMsg", "BitcoinOTC", "BitcoinOTCpos"]


# --------------------------------------------------------------------------- #
# Parallel compiled per-source strict temporal BFS (fair parallel baseline)
# --------------------------------------------------------------------------- #
@njit(parallel=True, cache=True)
def _per_source_strict_bfs_parallel(n, W, gptr, us, vs):
    """prange over sources; strict semantics by deferred within-group commits.

    Same algorithm as experiments/compiled_baselines.py::_per_source_strict_bfs
    but with the source loop parallelized (per-source scratch arrays are
    thread-private by allocation inside the loop body)."""
    R = np.zeros((n, W), dtype=np.uint64)
    G = gptr.shape[0] - 1
    for s in prange(n):
        reached = np.zeros(n, dtype=np.uint8)
        pend = np.empty(n, dtype=np.int64)
        reached[s] = 1
        for g in range(G):
            np_ = 0
            for k in range(gptr[g], gptr[g + 1]):
                u = us[k]
                v = vs[k]
                if reached[u] == 1 and reached[v] == 0:
                    ok = True
                    for q in range(np_):
                        if pend[q] == v:
                            ok = False
                            break
                    if ok:
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


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #
def _samples(fn, repeats: int):
    """Warm-up, then ``repeats`` timed calls; returns every sample."""
    fn()  # warm-up (also triggers/JIT-warms)
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def _best_of(fn, repeats: int) -> float:
    return min(_samples(fn, repeats))


def _stats(samples):
    """min / median / max of timing samples (one place, STRATA-v9 R26)."""
    xs = sorted(samples)
    k = len(xs)
    med = xs[k // 2] if k % 2 else 0.5 * (xs[k // 2 - 1] + xs[k // 2])
    return {"min_s": xs[0], "median_s": med, "max_s": xs[-1], "n": k}


def _load_real(name: str):
    n, edges, meta = load_standard_edges(DATA / f"{name}.txt")
    return n, edges, meta


def _make_synth():
    # Dense-group synthetic DAG stream (label: synthetic), matching the
    # compiled-throughput datapoint in the paper: n=8000, 10^6 edges, 200 groups.
    n, m, ngroups = 8000, 1_000_000, 200
    edges = generate_temporal_stream(n, m, timestamps=ngroups, directed=True, seed=7)
    return n, edges, {"dataset": "synthetic-PA(n=8000,m=1e6,G=200)", "synthetic": True}


def _run_plan(prep, n, W, plan_for, block_bits):
    """Run the kernel (full or source-blocked) and return the packed state."""
    if block_bits is None or block_bits >= n:
        R = np.zeros((n, W), dtype=np.uint64)
        _set_identity_vec(R, 0, n)
        run_prepared(R, prep, plan_for(W))
        return R
    full = np.zeros((n, W), dtype=np.uint64)
    for start in range(0, n, block_bits):
        nb = min(block_bits, n - start)
        Wb = words_for(nb)
        Rb = np.zeros((n, Wb), dtype=np.uint64)
        _set_identity_vec(Rb, start, nb)
        run_prepared(Rb, prep, plan_for(Wb))
        _scatter_block(full, Rb, start, nb)
    return full


def bench_dataset(name, n, edges, threads_grid, repeats, run_baseline,
                  configs=(("tiled", "static"),), oversub=4,
                  chunk_policy="threads", block_bits=None, reference="auto",
                  baseline_max_work=5e10, load_s=None, prep_parallel="off"):
    """``edges`` is a ``(t, u, v)`` array triple (array-native path, R17)."""
    m_edges = num_edges(edges)
    print(f"[{name}] n={n} m={m_edges}")
    t0 = time.perf_counter()
    R_ref, ref_used = reference_state(n, edges, "strict", reference)
    ref_s = time.perf_counter() - t0
    pairs = reachable_pairs(R_ref, n)
    W = words_for(n)
    blocked = block_bits is not None and block_bits < n

    # preprocessing: first (cold) call includes one-time JIT compilation of the
    # linear touched-row plan; then warm-up + best-of-R like the kernel
    # (sequential preprocessing, prep_mode="off": the classic Amdahl term)
    prep_cold = prepare_stream_arrays(n, *edges, directed=True, same_time_mode="strict",
                                      prep_mode="off")
    prep_cold_s = prep_cold.timings["total_s"]
    preps = [prepare_stream_arrays(n, *edges, directed=True, same_time_mode="strict",
                                   prep_mode="off") for _ in range(max(1, repeats))]
    prep = min(preps, key=lambda p: p.timings["total_s"])
    prep_s = prep.timings["total_s"]
    prep_samples = [p_.timings["total_s"] for p_ in preps]
    gptr, us, vs = prep.gptr, prep.us, prep.vs

    baseline_skip = None
    if run_baseline and n * m_edges > baseline_max_work:
        baseline_skip = (f"n*m={n * m_edges:.3g} exceeds --baseline-max-work "
                         f"{baseline_max_work:.3g}")
        print(f"  per-source BFS skipped: {baseline_skip}")
        run_baseline = False

    rows = []
    for T in threads_grid:
        eff = set_threads(T)
        if eff != T:
            print(f"  threads={T} unavailable on this host (max={max_threads()}); skipping")
            continue
        bfs_s = None
        if run_baseline:
            Rb = _per_source_strict_bfs_parallel(n, W, gptr, us, vs)
            assert np.array_equal(Rb, R_ref), f"baseline diverged at T={eff}"
            bfs_samples = _samples(lambda: _per_source_strict_bfs_parallel(n, W, gptr, us, vs),
                                   repeats)
            bfs_s = min(bfs_samples)
        prep_par = None
        if prep_parallel in ("on", "compare"):
            # parallel preprocessing at this thread count: verified array
            # for array against the sequential plan before it is timed
            pp = prepare_stream_arrays(n, *edges, directed=True, same_time_mode="strict",
                                       prep_mode="on", prep_threads=eff)
            for nm_ in ("gptr", "us", "vs", "dirty_idx", "dptr", "src_slot"):
                assert np.array_equal(getattr(pp, nm_), getattr(prep, nm_)), \
                    f"parallel preprocessing diverged ({nm_}) at T={eff}"
            pps = [prepare_stream_arrays(n, *edges, directed=True, same_time_mode="strict",
                                         prep_mode="on", prep_threads=eff)
                   for _ in range(max(1, repeats))]
            best_pp = min(pps, key=lambda p_: p_.timings["total_s"])
            prep_par = {"total_s": best_pp.timings["total_s"], "phases_s": best_pp.timings,
                        "samples_s": [p_.timings["total_s"] for p_ in pps],
                        "blocks": best_pp.prep_blocks, "sort_path": best_pp.prep_sort}
        for layout, schedule in configs:
            def plan_for(Wx, layout=layout, schedule=schedule, eff=eff):
                return resolve_plan(Wx, eff, layout=layout, schedule=schedule,
                                    oversub=oversub, chunk_policy=chunk_policy)
            plan = plan_for(words_for(min(block_bits, n)) if blocked else W)

            Rp = _run_plan(prep, n, W, plan_for, block_bits if blocked else None)
            assert np.array_equal(Rp, R_ref), \
                f"parallel diverged at T={eff} layout={layout} schedule={schedule}"
            del Rp

            def _kern(plan_for=plan_for):
                _run_plan(prep, n, W, plan_for, block_bits if blocked else None)

            k_samples = _samples(_kern, repeats)
            k_best = min(k_samples)
            row = {"threads": eff, "layout": layout,
                   "resolved_layout": plan.resolved_layout, "schedule": schedule,
                   "chunks": plan.nchunks, "chunk_policy": chunk_policy,
                   "plan": plan.as_dict(), "kernel_s": k_best,
                   "end_to_end_s": k_best + prep_s,
                   "kernel_samples_s": k_samples, "kernel_stats": _stats(k_samples)}
            if bfs_s is not None:
                row["persrc_bfs_s"] = bfs_s
                row["persrc_bfs_samples_s"] = bfs_samples
            if prep_par is not None:
                row["prep_par"] = prep_par
                row["end_to_end_par_s"] = k_best + prep_par["total_s"]
            rows.append(row)
            msg = (f"  T={eff:>3} {layout}/{schedule} ({plan.resolved_layout}, "
                   f"chunks={plan.nchunks}) kernel={k_best:.4f}s e2e={row['end_to_end_s']:.4f}s")
            if bfs_s is not None:
                msg += f" per-src-BFS={bfs_s:.4f}s"
            print(msg)

    return {
        "dataset": name, "n": n, "m": int(m_edges), "W": W, "pairs": int(pairs),
        "stream_path": "arrays", "load_s": load_s,
        "reference": ref_used, "reference_s": ref_s,
        "seq_interpreted_s": ref_s if ref_used == "interpreted" else None,
        "prep_s": prep_s, "prep_phases_s": prep.timings, "prep_cold_s": prep_cold_s,
        "prep_timing": "best-of-R after a cold call (cold call includes JIT compilation)",
        "prep_samples_s": prep_samples, "prep_parallel_mode": prep_parallel,
        "block_bits": block_bits if blocked else None,
        "baseline_skipped": baseline_skip,
        "num_directed_edges": prep.num_edges, "dirty_total": prep.dirty_total,
        "verified_bit_identical": True, "rows": rows,
    }


def print_plans(datasets, grid, configs, oversub, chunk_policy, block_bits, machine):
    """--plan-only: resolved plans without any timing."""
    from temporal_strata.parallel import cache_line_words

    print(f"PLAN ONLY (no timing) -- {machine['summary']}")
    print(f"cache-line words L={cache_line_words()}  chunk_policy={chunk_policy}  "
          f"oversub={oversub}  block_bits={block_bits}")
    out = []
    hdr = f"{'dataset':38s} {'W':>6s} {'p':>3s} {'layout/schedule':18s} {'resolved':9s} {'chunks':>6s} {'idle':>4s}"
    print(hdr)
    for name, n, _edges in datasets:
        W = words_for(min(block_bits, n)) if (block_bits and block_bits < n) else words_for(n)
        for T in grid:
            for layout, schedule in configs:
                pl = resolve_plan(W, T, layout=layout, schedule=schedule, oversub=oversub,
                                  chunk_policy=chunk_policy)
                idle = max(0, T - pl.nchunks)
                flag = "  <-- workers idle" if idle else ""
                print(f"{name[:38]:38s} {W:>6d} {T:>3d} {layout + '/' + schedule:18s} "
                      f"{pl.resolved_layout:9s} {pl.nchunks:>6d} {idle:>4d}{flag}")
                out.append({"dataset": name, "W": W, "threads": T, "idle_workers": idle,
                            "plan": pl.as_dict()})
    return out


def __sorted(n, edges):
    from temporal_strata.stream import sort_and_validate_edges
    return sort_and_validate_edges(n, edges)


def emit_table(results, machine, path: Path, pending: bool, table_config=None):
    """Emit tab:parscaling in the paper's fill-width layout.

    Layout: booktabs rules,
    ``tabular*{\\columnwidth}`` with ``\\extracolsep{\\fill}`` so the table fills
    exactly one column with uniform inter-column spacing, \\scriptsize,
    tabcolsep 2pt, and seven columns -- Dataset, p, kernel time, kernel
    speedup, end-to-end time (kernel + shared sequential preprocessing),
    end-to-end speedup, and the equal-thread parallel-BFS ratio.
    """
    lines = [
        "% Auto-generated by experiments/parallel_scaling.py -- do not edit by hand.",
        f"% Machine: {machine['summary']}",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Strong scaling of the compiled column-partitioned kernel"
        " (main paper, Sec.~4.4) versus a compiled \\emph{parallel} per-source"
        " temporal BFS at equal thread counts (best-of-%d after warm-up)."
        " Kernel phase and end-to-end (kernel + shared sequential preprocessing)"
        " reported separately; every configuration verified bit-identical to the"
        " sequential core before timing. Machine: %s.}"
        % (machine["repeats"], machine["tex_summary"]),
        "\\label{tab:parscaling}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lrrrrrr@{}}",
        "\\toprule",
        "\\textbf{Dataset} & $\\boldsymbol{p}$ & \\shortstack[r]{Kernel\\\\(ms)} &"
        " \\shortstack[r]{Kern.\\\\sp.} & \\shortstack[r]{E2E\\\\(ms)} &"
        " \\shortstack[r]{E2E\\\\sp.} & \\shortstack[r]{BFS/\\\\E2E} \\\\",
        "\\midrule",
    ]
    if pending:
        lines.append(
            "\\multicolumn{7}{c}{\\emph{PENDING -- populate by running"
            " \\texttt{experiments/parallel\\_scaling.py} on the evaluation"
            " machine (multi-core host required)}} \\\\")
    else:
        for r in results:
            sel = [x for x in r["rows"]
                   if table_config is None
                   or (x.get("layout"), x.get("schedule")) == tuple(table_config)]
            base = next((x for x in sel if x["threads"] == 1), None)
            for i, x in enumerate(sel):
                ksp = (base["kernel_s"] / x["kernel_s"]) if base else float("nan")
                esp = (base["end_to_end_s"] / x["end_to_end_s"]) if base else float("nan")
                bfs = x.get("persrc_bfs_s")
                ratio = (bfs / x["end_to_end_s"]) if bfs else None  # conservative E2E
                nm = r["dataset"] if i == 0 else ""
                lines.append(
                    f"{nm} & {x['threads']} & {x['kernel_s'] * 1e3:.3f} & "
                    f"{ksp:.2f}$\\times$ & {x['end_to_end_s'] * 1e3:.3f} & "
                    f"{esp:.2f}$\\times$ & "
                    + (f"{ratio:.1f}$\\times$" if ratio else "--") + " \\\\")
            if r is not results[-1]:
                lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def _parse_list(text, allowed, what):
    items = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in items if x not in allowed]
    if bad or not items:
        raise SystemExit(f"--{what}: expected comma list from {allowed}, got {text!r}")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=str, default=None,
                    help="comma list; default 1,2,4,... up to NUMBA max")
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--layouts", type=str, default="tiled",
                    help="comma list from rowmajor,tiled,auto")
    ap.add_argument("--schedules", type=str, default="static",
                    help="comma list from static,dynamic")
    ap.add_argument("--oversub", type=int, default=4,
                    help="chunks per thread under the dynamic schedule")
    ap.add_argument("--chunk-policy", type=str, default="threads",
                    choices=list(CHUNK_POLICIES),
                    help="threads: p (or oversub*p) chunks capped by W (v1 protocol, default); "
                         "auto: library rule with the row-major cache-line cap")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the resolved plans and exit without timing")
    ap.add_argument("--plan-out", type=str, default=None,
                    help="with --plan-only: also write the plans as JSON")
    ap.add_argument("--large", action="append", default=[],
                    help="extra synthetic stream n:m:groups[:seed] (repeatable)")
    ap.add_argument("--extra-real", action="append", default=[],
                    help="also run datasets_real_temporal/standard/NAME.txt")
    ap.add_argument("--skip-real", action="store_true",
                    help="omit the three real-timestamp datasets")
    ap.add_argument("--block-bits", type=int, default=None,
                    help="source-block width for datasets with n > B")
    ap.add_argument("--reference", type=str, default="auto", choices=list(REFERENCES))
    ap.add_argument("--baseline-max-work", type=float, default=5e10,
                    help="skip the per-source BFS baseline when n*m exceeds this")
    ap.add_argument("--table-config", type=str, default=None,
                    help="layout:schedule emitted to the LaTeX table "
                         "(default: first requested layout and schedule)")
    ap.add_argument("--prep-parallel", choices=["off", "on", "compare"], default="off",
                    help="also time the parallel preprocessing at every thread count "
                         "(verified identical to the sequential plan first); 'off' keeps "
                         "the v5-v8 protocol")
    ap.add_argument("--out", type=str, default=str(HERE.parent / "results" / "real" / "parallel_scaling.json"))
    ap.add_argument("--tex-out", type=str, default=str(HERE.parent / "results" / "real" / "table_parallel_scaling.tex"))
    args = ap.parse_args()

    layouts = _parse_list(args.layouts, LAYOUTS, "layouts")
    schedules = _parse_list(args.schedules, SCHEDULES, "schedules")
    configs = [(l, s) for l in layouts for s in schedules]
    table_config = (tuple(args.table_config.split(":")) if args.table_config
                    else configs[0])
    if table_config not in configs:
        raise SystemExit(f"--table-config {table_config} is not among the swept configs {configs}")

    hw_max = max_threads()
    if args.threads:
        grid = sorted({int(x) for x in args.threads.split(",")})
    else:
        grid, t = [], 1
        while t <= hw_max:
            grid.append(t)
            t *= 2
        if grid[-1] != hw_max:
            grid.append(hw_max)

    machine = machine_info()
    cpu_count = int(machine["cpu_count"])
    single_core = cpu_count <= 1
    machine["repeats"] = args.repeats
    machine["tex_summary"] = machine["summary"].replace("_", "\\_")
    print("machine:", machine["summary"])
    print("thread grid:", grid)
    print("configs (layout, schedule):", configs, "| table:", table_config)
    if single_core:
        print("WARNING: single hardware thread detected -- correctness will be "
              "verified but NO speedup table will be populated (integrity rule).")

    datasets = select_datasets_arrays([] if args.skip_real else REAL_SETS, args.large,
                                      include_default_synth=not args.skip_synthetic,
                                      materialize=not args.plan_only,
                                      extra=args.extra_real)
    if args.plan_only:
        plans = print_plans([d[:3] for d in datasets], grid, configs, args.oversub,
                            args.chunk_policy,
                            args.block_bits, machine)
        if args.plan_out:
            Path(args.plan_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.plan_out).write_text(json.dumps({"machine": machine, "plans": plans}, indent=2))
            print(f"wrote {args.plan_out}")
        return

    results = [bench_dataset(name, n, edges, grid, args.repeats,
                             run_baseline=not args.skip_baseline,
                             configs=configs, oversub=args.oversub,
                             chunk_policy=args.chunk_policy, block_bits=args.block_bits,
                             reference=args.reference,
                             baseline_max_work=args.baseline_max_work, load_s=load_s,
                             prep_parallel=args.prep_parallel)
               for (name, n, edges, load_s) in datasets]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"machine": machine, "thread_grid": grid, "configs": configs,
               "chunk_policy": args.chunk_policy, "block_bits": args.block_bits,
               "table_config": list(table_config), "oversub": args.oversub,
               "pending": single_core, "prep_parallel": args.prep_parallel,
               "samples": "every timed configuration stores all repeat samples",
               "results": results}
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    emit_table(results, machine, Path(args.tex_out), pending=single_core,
               table_config=table_config)
    if single_core:
        print("STATUS: PENDING -- rerun on a multi-core machine "
              "(the evaluation machine is an Apple M5 Max; NUMBA_NUM_THREADS=18) "
              "to populate the table.")


if __name__ == "__main__":
    main()
