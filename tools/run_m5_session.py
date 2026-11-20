#!/usr/bin/env python3
"""Run the Apple M5 Max measurement session in one go (backlog R15).

Why
---
The session (REPRODUCTION_GUIDE_EN.md / _KO.md, Section 10) has seven steps
and must succeed in one sitting. Typing the commands by hand invites option
typos, skipped steps, and lost logs; one wrong step can void the session.

How
---
* ``STEPS`` below is the single source of the session commands; the guide lists
  the same commands (tests/test_session_runner.py checks that every command
  here appears verbatim in both guides).
* Each command's console output and exit code go to
  ``results/real/logs/step<k>_<i>.<stamp>.log``;
  ``results/real/logs/session.<stamp>.json`` records start / end times, the machine, the git commit (if any), and every
  command's status.
* Stop conditions, checked before any timing:
    1. any step-1 test or ``tools/check_numbers.py`` fails;
    2. the ``--plan-only`` output shows idle workers for a real dataset at
       ``p <= W`` (backlog R8);
    3. step 6: timed external-baseline runs only if ``--check-only`` reports a
       matching semantics on every dataset; if ``raphtory`` cannot be
       installed or imported, only that part is skipped, with the reason.
* Existing result files are never overwritten: before a step runs, each of its
  output files that already exists is renamed with a time-stamp suffix.
* ``--dry-run`` prints the plan; ``--from-step N`` / ``--only-step N`` resume;
  ``--steps 1,2,3,6,7`` runs a subset (the STRATA-v5 re-measurement);
  STRATA-v9 adds steps 8-11 (single-thread suite on the M5, large public
  temporal graphs, bucketed closure, GraphBLAS reference point) and stores
  every repeat sample; the v9 re-measurement is
  ``--steps 1,2,3,4,5,6,8,9,10,11``;
  ``--only-external`` runs only the external-baseline part of step 6 (install,
  check, timed run), e.g. from a separate Python 3.13 environment where the
  external package has wheels.

Run inside the activated virtual environment from
``STRATA-code/05_Temporal_STRATA``::

    python3 tools/run_m5_session.py --dry-run
    python3 tools/run_m5_session.py
"""
from __future__ import annotations

import argparse
import re
import datetime as _dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "real"
LOGS = RES / "logs"
THREADS = "1,2,4,6,8,12,16,18"
REAL_W = {"CollegeMsg": 30, "BitcoinOTC": 92, "BitcoinOTCpos": 88}

# STRATA-v13 (R42): the last step rebuilds every generated paper file (macros,
# tables, data figures) from the results just measured and checks them, so a
# session can never leave the paper's copy stale. main.tex prose is not touched.
FIG = "../../STRATA-latex/STRATA-paper-EN/figures"
REGEN_COMMANDS = [
    "python3 experiments/m5_numbers.py",
    "python3 experiments/make_paper_figures.py",
    "python3 experiments/plot_parallel.py paper --json results/real/parallel_scaling.json "
    f"--large results/real/parallel_scaling_large.json --out {FIG}/fig_scaling.pdf",
    "python3 experiments/plot_parallel.py breakdown --json results/real/parallel_scaling.json "
    f"--large results/real/parallel_scaling_large.json --out {FIG}/fig_breakdown.pdf",
    "python3 experiments/plot_parallel.py roofline --json results/real/bandwidth_roofline.json "
    f"--out {FIG}/fig_roofline.pdf",
    "python3 tools/check_numbers.py",
]
REGEN_STEP = 13
MEASURE_STEPS = {2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}


def regenerated_files():
    """Paths and sha256 of the generated paper files after step 13 (R42)."""
    import hashlib
    paper = ROOT.parent.parent / "STRATA-latex" / "STRATA-paper-EN"
    names = ["m5_numbers.tex", "table_parallel_scaling.tex", "table_parallel_scaling_large.tex",
             "table_m5_summary.tex", "table_maincompare.tex", "table_real_temporal.tex",
             "table_closure_grouped.tex"]
    names += [f"figures/{f}.{ext}" for f in ("fig_scaling", "fig_breakdown", "fig_roofline",
                                             "fig_speedup_runtime") for ext in ("pdf", "layout.json")]
    out = []
    for name in names:
        p = paper / name
        if p.exists():
            out.append({"path": name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    return out


def with_regen(wanted, no_regen=False):
    """R42: add the regeneration step whenever a measurement step is selected."""
    if wanted is None or no_regen:
        return wanted
    return wanted | {REGEN_STEP} if wanted & MEASURE_STEPS else wanted


# Each step: (number, title, [command strings exactly as in the guide], outputs)
STEPS = [
    (1, "environment, correctness, and plan check", [
        "pip install -r requirements-mac.txt -r requirements-optional.txt",
        "python3 -m pytest tests/ -q",
        "NUMBA_NUM_THREADS=18 python3 -m pytest tests/test_parallel.py tests/test_parallel_layout.py -q",
        "python3 tools/check_numbers.py",
        "NUMBA_NUM_THREADS=18 python3 experiments/parallel_scaling.py --plan-only "
        f"--threads {THREADS} --layouts rowmajor,tiled,auto --schedules static,dynamic "
        "--chunk-policy threads --large 50000:1000000:400 --large 100000:2000000:400 "
        "--block-bits 65536 --plan-out results/real/plan_check.json",
    ], ["plan_check.json"]),
    (2, "strong scaling: layout x schedule", [
        "NUMBA_NUM_THREADS=18 python3 experiments/parallel_scaling.py "
        f"--threads {THREADS} --repeats 5 "
        "--layouts rowmajor,tiled,auto --schedules static,dynamic --oversub 4 "
        "--chunk-policy threads --table-config tiled:static --prep-parallel compare "
        "--out results/real/parallel_scaling.json "
        "--tex-out results/real/table_parallel_scaling.tex",
    ], ["parallel_scaling.json", "table_parallel_scaling.tex"]),
    (3, "large graphs and the sequential fraction", [
        "NUMBA_NUM_THREADS=18 python3 experiments/parallel_scaling.py "
        "--skip-real --skip-synthetic "
        "--large 50000:1000000:400 --large 100000:2000000:400 --block-bits 65536 "
        "--reference compiled --baseline-max-work 1e10 "
        f"--threads {THREADS} --repeats 3 "
        "--layouts rowmajor,tiled,auto --schedules static,dynamic --chunk-policy threads "
        "--prep-parallel compare "
        "--out results/real/parallel_scaling_large.json "
        "--tex-out results/real/table_parallel_scaling_large.tex",
    ], ["parallel_scaling_large.json", "table_parallel_scaling_large.tex"]),
    (4, "memory-bandwidth ceiling", [
        "NUMBA_NUM_THREADS=18 python3 experiments/bandwidth_roofline.py "
        f"--threads {THREADS} --array-mib 2048 --repeats 5 "
        "--layouts rowmajor,tiled --chunk-policy threads --synthetic "
        "--large 100000:2000000:400 "
        "--out results/real/bandwidth_roofline.json",
    ], ["bandwidth_roofline.json"]),
    (5, "row-major vs tiled crossover", [
        "NUMBA_NUM_THREADS=18 python3 experiments/layout_crossover.py "
        "--threads 18 --points 9 --repeats 5 --out results/real/layout_crossover_p18.json",
        "NUMBA_NUM_THREADS=18 python3 experiments/layout_crossover.py "
        "--threads 6 --points 9 --repeats 5 --out results/real/layout_crossover_p6.json",
    ], ["layout_crossover_p18.json", "layout_crossover_p6.json"]),
    (6, "compiled-vs-compiled on real data and an external baseline", [
        "python3 experiments/compiled_real.py --repeats 5 --out results/real/compiled_real.json",
        "pip install raphtory",
        "python3 experiments/external_baselines.py --adapter raphtory --check-only --diagnose "
        "--out results/real/external_check.json",
        "python3 experiments/external_baselines.py --adapter raphtory "
        "--datasets CollegeMsg,BitcoinOTC,BitcoinOTCpos --sources all --repeats 3 "
        "--out results/real/external_baselines.json",
    ], ["compiled_real.json", "external_check.json", "external_baselines.json"]),
    (7, "MPS probe and preview figures", [
        "python3 experiments/backend_bench.py --device mps",
        "python3 experiments/plot_parallel.py scaling "
        "--json results/real/parallel_scaling.json --out results/real/fig_parallel_scaling.pdf",
        "python3 experiments/plot_parallel.py scaling "
        "--json results/real/parallel_scaling_large.json "
        "--out results/real/fig_parallel_scaling_large.pdf",
        "python3 experiments/plot_parallel.py roofline "
        "--json results/real/bandwidth_roofline.json --out results/real/fig_roofline.pdf",
    ], ["backend_bench.json", "table_backend_bench.tex", "fig_parallel_scaling.pdf",
        "fig_parallel_scaling_large.pdf", "fig_roofline.pdf"]),
    # ---- STRATA-v9 (backlog R25 / R27 / R29 / R31): steps 8-11 ----
    (8, "single-thread experiments on the evaluation machine (R25)", [
        "NUMBA_NUM_THREADS=1 python3 experiments/perf_table.py --no-tex",
        "NUMBA_NUM_THREADS=1 python3 experiments/compiled_baselines.py --no-tex",
        "NUMBA_NUM_THREADS=1 python3 experiments/real_temporal_benchmark.py --reps 5",
        "NUMBA_NUM_THREADS=1 python3 experiments/ablation_snapshot_necessity.py",
        "NUMBA_NUM_THREADS=1 python3 experiments/ablation_strict_prune.py",
        "NUMBA_NUM_THREADS=1 python3 experiments/index_crossover.py",
        "NUMBA_NUM_THREADS=1 python3 experiments/real_dynamic_and_blocking.py",
        "python3 tools/machine_stamp.py --label single_thread "
        "--out results/real/single_thread_machine.json "
        "results/perf/perf_table.json results/perf/compiled_baselines.json "
        "results/real/real_temporal_results.json results/real/snapshot_necessity.json "
        "results/real/ablation_strict_prune.json results/real/index_crossover.json "
        "results/real/real_dynamic_blocking.json",
    ], ["perf/perf_table.json", "perf/compiled_baselines.json", "real_temporal_results.json",
        "real_temporal_results.csv", "snapshot_necessity.json", "ablation_strict_prune.json",
        "index_crossover.json", "real_dynamic_blocking.json", "single_thread_machine.json"]),
    (9, "large public temporal graphs (R27)", [
        "python3 experiments/fetch_real_temporal.py --names sx-askubuntu,sx-superuser "
        "--log results/real/fetch_large.json",
        "NUMBA_NUM_THREADS=18 python3 experiments/parallel_scaling.py "
        "--skip-real --skip-synthetic --extra-real sx-askubuntu --extra-real sx-superuser "
        "--reference sampled --baseline-max-work 1e10 "
        f"--threads {THREADS} --repeats 3 "
        "--layouts tiled --schedules static,dynamic --chunk-policy threads "
        "--prep-parallel compare "
        "--out results/real/parallel_scaling_realbig.json "
        "--tex-out results/real/table_parallel_scaling_realbig.tex",
    ], ["fetch_large.json", "parallel_scaling_realbig.json", "table_parallel_scaling_realbig.tex"]),
    (10, "closure semantics on bucketed real timestamps (R29)", [
        "NUMBA_NUM_THREADS=18 python3 experiments/closure_grouped.py "
        f"--threads {THREADS} --repeats 5 --out results/real/closure_grouped.json",
    ], ["closure_grouped.json"]),
    (11, "multithreaded GraphBLAS reference point, static closure (R31)", [
        "pip install python-graphblas",
        "NUMBA_NUM_THREADS=18 python3 experiments/graphblas_reference.py "
        f"--threads {THREADS} --repeats 5 --out results/real/graphblas_reference.json",
    ], ["graphblas_reference.json"]),
    # ---- STRATA-v13 (backlog R39 / R42) ----
    (12, "external baseline: Temporal Ligra earliest arrival (R39)", [
        "python3 experiments/temporal_ligra.py fetch --dest external/temporal_ligra",
        "python3 experiments/temporal_ligra.py build --src external/temporal_ligra",
        "python3 experiments/temporal_ligra.py check --src external/temporal_ligra "
        "--out results/real/temporal_ligra_check.json",
        "python3 experiments/temporal_ligra.py time --src external/temporal_ligra "
        "--threads 1,18 --out results/real/temporal_ligra_time.json",
        "python3 experiments/temporal_ligra.py time-batch --src external/temporal_ligra "
        "--threads 1,6,12,18 --out results/real/temporal_ligra_batch.json",
    ], ["temporal_ligra_check.json", "temporal_ligra_time.json", "temporal_ligra_batch.json"]),
    # ---- STRATA-v21 (backlog R54): listed before step 13 so regeneration runs last ----
    (14, "baseline: word-packed temporal MS-BFS (64 sources per word, empty words skipped)", [
        "NUMBA_NUM_THREADS=18 python3 experiments/temporal_msbfs.py bench "
        "--datasets CollegeMsg,BitcoinOTC,BitcoinOTCpos --synthetic --threads 1,18 --repeats 5 "
        "--out results/real/temporal_msbfs.json",
    ], ["temporal_msbfs.json"]),
    (15, "public temporal-index baselines: fetch and inspect (tgindex, TGLib)", [
        "python3 experiments/temporal_index_baseline.py probe --dest external "
        "--out results/real/index_baseline_probe.json",
    ], ["index_baseline_probe.json"]),
    (16, "real stream above 10^7 edges: sx-stackoverflow, source-blocked", [
        "python3 experiments/fetch_real_temporal.py --names sx-stackoverflow "
        "--log results/real/fetch_stackoverflow.json",
        "NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py "
        "--dataset sx-stackoverflow --block-bits 4096 --blocks 8 --threads 1,18 "
        "--schedule dynamic --state rows --verify-sources 64 --out results/real/large_real_blocked.json",
    ], ["fetch_stackoverflow.json", "large_real_blocked.json"]),
    # ---- STRATA-v22 (backlog R55 / R56 / R57) ----
    (17, "sx-stackoverflow: block width and thread sweep, then every block", [
        "NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py "
        "--dataset sx-stackoverflow --block-bits 4096 --blocks 2 --threads 1,6,12,18 "
        "--schedule static --state rows --verify-sources 64 --out results/real/large_real_b4096.json",
        "NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py "
        "--dataset sx-stackoverflow --block-bits 65536 --blocks 2 --threads 1,6,12,18 "
        "--schedule static --state rows --verify-sources 64 --out results/real/large_real_b65536.json",
        "NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py "
        "--dataset sx-stackoverflow --block-bits 65536 --blocks all "
        "--threads best:results/real/large_real_b65536.json "
        "--schedule static --state rows --verify-sources 64 --out results/real/large_real_full.json",
    ], ["large_real_b4096.json", "large_real_b65536.json", "large_real_full.json"]),
    (18, "external baseline: TGLib earliest arrival (MIT)", [
        "python3 experiments/tglib_baseline.py build --dest external "
        "--out results/real/tglib_build.json",
        "NUMBA_THREADING_LAYER=workqueue python3 experiments/tglib_baseline.py check "
        "--out results/real/tglib_check.json",
        "NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=1 python3 experiments/tglib_baseline.py time "
        "--check results/real/tglib_check.json --procs 1,18 --out results/real/tglib_time.json",
    ], ["tglib_build.json", "tglib_check.json", "tglib_time.json"]),
    # ---- STRATA-v23 (backlog R60) ----
    (19, "sx-stackoverflow: why the block kernel slows down beyond six threads", [
        "NUMBA_NUM_THREADS=18 python3 experiments/thread_diagnosis.py "
        "--dataset sx-stackoverflow --block-bits 4096,65536 --threads 1,6,12,18 "
        "--schedules static,dynamic --repeats 3 --out results/real/thread_diagnosis.json",
    ], ["thread_diagnosis.json"]),
    # ---- STRATA-v24 (backlog R64) ----
    (20, "sx-stackoverflow: column chunks versus concurrent threads", [
        "NUMBA_NUM_THREADS=18 python3 experiments/thread_diagnosis.py "
        "--dataset sx-stackoverflow --block-bits 65536 --pairs 6:6,6:12,6:18,12:18,18:18 "
        "--repeats 3 --out results/real/thread_chunks.json",
    ], ["thread_chunks.json"]),
    # ---- STRATA-v25 (backlog R69) ----
    (21, "random row updates by thread count (hardware access shape of the kernel)", [
        "NUMBA_NUM_THREADS=18 python3 experiments/random_rows_bench.py "
        "--dataset sx-stackoverflow --words 1024 --edges 16000000 --threads 1,6,12,18 "
        "--repeats 3 --out results/real/random_rows.json",
    ], ["random_rows.json"]),
    # ---- STRATA-v27 (backlog R73) ----
    (22, "sx-stackoverflow: which kernel phase slows down past six threads", [
        "NUMBA_NUM_THREADS=18 python3 experiments/kernel_phases.py "
        "--dataset sx-stackoverflow --block-bits 65536 --threads 1,6,12,18 "
        "--repeats 3 --out results/real/kernel_phases.json",
    ], ["kernel_phases.json"]),
    # ---- STRATA-v28 (backlog R75 / R76 / R79) ----
    (23, "sx-stackoverflow: the copy phase and the tiles-as-state run, by threads", [
        "NUMBA_NUM_THREADS=18 python3 experiments/kernel_phases.py "
        "--dataset sx-stackoverflow --block-bits 65536 --threads 1,6,12,18 "
        "--variants strict,copy,copy_prefault,copy_reuse,copy_aligned,tiles "
        "--repeats 3 --out results/real/kernel_copy.json",
    ], ["kernel_copy.json"]),
    (24, "sx-stackoverflow: every block with the tiles as the state", [
        "NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py "
        "--dataset sx-stackoverflow --block-bits 65536 --blocks 2 --threads 1,6,12,18 "
        "--schedule static --state tiles --verify-sources 64 "
        "--out results/real/large_real_tiles_b65536.json",
        "NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py "
        "--dataset sx-stackoverflow --block-bits 65536 --blocks all "
        "--threads best:results/real/large_real_tiles_b65536.json "
        "--schedule static --state tiles --verify-sources 64 "
        "--out results/real/large_real_tiles_full.json",
    ], ["large_real_tiles_b65536.json", "large_real_tiles_full.json"]),
    (25, "single-thread compiled baselines with the production core (R79)", [
        "NUMBA_NUM_THREADS=1 python3 experiments/compiled_baselines.py --no-tex",
    ], ["perf/compiled_baselines.json"]),
    # ---- STRATA-v30 (backlog R90 / R91) ----
    (26, "sx-stackoverflow: copies after every page of the fresh state was written", [
        "NUMBA_NUM_THREADS=18 python3 experiments/kernel_phases.py "
        "--dataset sx-stackoverflow --block-bits 65536 --threads 1,6,12,18 "
        "--variants copy,copy_prefault,copy_reuse "
        "--repeats 3 --out results/real/kernel_prefault.json",
    ], ["kernel_prefault.json"]),
    (27, "sx-stackoverflow: tile state on narrow blocks and with the dynamic schedule", [
        "NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py "
        "--dataset sx-stackoverflow --block-bits 4096 --blocks 2 --threads 1,6,12,18 "
        "--schedule static --state tiles --verify-sources 64 "
        "--out results/real/large_real_tiles_b4096.json",
        "NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py "
        "--dataset sx-stackoverflow --block-bits 4096 --blocks 2 --threads 1,6,12,18 "
        "--schedule dynamic --state tiles --verify-sources 64 "
        "--out results/real/large_real_tiles_dyn_b4096.json",
        "NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py "
        "--dataset sx-stackoverflow --block-bits 65536 --blocks 2 --threads 1,6,12,18 "
        "--schedule dynamic --state tiles --verify-sources 64 "
        "--out results/real/large_real_tiles_dyn_b65536.json",
    ], ["large_real_tiles_b4096.json", "large_real_tiles_dyn_b4096.json",
        "large_real_tiles_dyn_b65536.json"]),
    # ---- STRATA-v31 (backlog R94) ----
    (28, "sx-stackoverflow: tile state across block widths (memory versus threads)", [
        "NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py "
        "--dataset sx-stackoverflow --block-bits 16384 --blocks 2 --threads 1,6,12,18 "
        "--schedule static --state tiles --verify-sources 64 "
        "--out results/real/large_real_tiles_b16384.json",
        "NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py "
        "--dataset sx-stackoverflow --block-bits 32768 --blocks 2 --threads 1,6,12,18 "
        "--schedule static --state tiles --verify-sources 64 "
        "--out results/real/large_real_tiles_b32768.json",
    ], ["large_real_tiles_b16384.json", "large_real_tiles_b32768.json"]),
    # ---- STRATA-v32 (backlog R98) ----
    (29, "sx-stackoverflow: every block at half the block width (beta = 32,768)", [
        "NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py "
        "--dataset sx-stackoverflow --block-bits 32768 --blocks all --threads 18 "
        "--schedule static --state tiles --verify-sources 64 "
        "--out results/real/large_real_tiles_full_b32768.json",
    ], ["large_real_tiles_full_b32768.json"]),
    # ---- STRATA-v34 (backlog R102 / R109 / R105) ----
    (30, "large synthetic streams: fresh, pre-written, and no shared state (unblocked)", [
        "NUMBA_NUM_THREADS=18 python3 experiments/tile_state_unblocked.py "
        "--large 50000:1000000:400 --large 100000:2000000:400 "
        "--threads 1,6,12,18 --repeats 3 --out results/real/tile_state_unblocked.json",
    ], ["tile_state_unblocked.json"]),
    (31, "sx-askubuntu: rolling windows and monotone insertion, compiled parallel", [
        "python3 experiments/fetch_real_temporal.py --names sx-askubuntu "
        "--log results/real/fetch_askubuntu_window.json",
        "NUMBA_NUM_THREADS=18 python3 experiments/window_large.py --dataset sx-askubuntu "
        "--threads 18 --slots 8 --window-frac 0.25 --batches 5 "
        "--out results/real/window_large.json",
    ], ["fetch_askubuntu_window.json", "window_large.json"]),
    (32, "elementary-step counts of compiled BFS and STRATA (machine independent)", [
        "python3 experiments/op_counts.py --out results/real/op_counts.json",
    ], ["op_counts.json"]),
    # ---- STRATA-v36 (backlog R113) ----
    (33, "sx-askubuntu: insertion batches, tiled versus row-major in place", [
        "NUMBA_NUM_THREADS=18 python3 experiments/window_large.py --dataset sx-askubuntu "
        "--threads 18 --slots 0 --batches 5 --insert-layouts tiled,rowmajor "
        "--out results/real/window_insert.json",
    ], ["window_insert.json"]),
    (13, "regenerate the paper's generated files from the new results (R42)", REGEN_COMMANDS,
     []),
]

# Optional commands (STRATA-v9): a failure with one of the listed exit codes
# (None = any) is recorded and the rest of that step is skipped instead of
# stopping the session; any other failure still stops it.
# (step, index) -> (reason, exit codes)
OPTIONAL_COMMANDS = {
    (9, 0): ("dataset download failed; the paper records the failure instead of "
             "substituting data", None),
    (11, 0): ("python-graphblas could not be installed", None),
    (11, 1): ("python-graphblas not importable", {3}),
    (12, 0): ("Temporal Ligra could not be fetched (network); recorded, not substituted", None),
    (12, 1): ("Temporal Ligra did not compile on this machine; recorded", None),
    (12, 2): ("Temporal Ligra semantics check failed to run; recorded", None),
    (12, 3): ("Temporal Ligra does not compute the strict relation; not timed", {4}),
    (12, 4): ("Temporal Ligra does not compute the strict relation; not timed", {4}),
    # check_numbers fails when the abstract's typed numbers no longer match the
    # regenerated macros; the files are rebuilt anyway and the next batch fixes
    # the prose, so the session records the failure instead of stopping
    (13, 5): ("check_numbers reports prose to update in the next batch", None),
    (15, 0): ("temporal-index repositories could not be fetched; recorded", None),
    (16, 0): ("sx-stackoverflow download failed; recorded, not substituted", None),
    # v23 (R58): PyPI has no macOS arm64 wheel; the pinned source is built instead
    # v24 (R63): on macOS the build needs Homebrew's libomp (brew install libomp)
    (18, 0): ("TGLib could not be built from source on this machine; recorded", None),
    (18, 1): ("TGLib does not compute the strict relation (or is not importable); not timed",
              {2, 3}),
}

# step-6 commands that depend on the external check (index within step 6)
OPTIONAL_EXTERNAL = {1: "install", 2: "check", 3: "timed"}


class StopSession(RuntimeError):
    pass


def to_argv(command: str):
    """Split 'VAR=val python3 ...' into (env overrides, argv for this interpreter)."""
    toks = shlex.split(command)
    env = {}
    while toks and "=" in toks[0] and not toks[0].startswith("-"):
        k, v = toks.pop(0).split("=", 1)
        env[k] = v
    if toks[0] == "python3":
        toks[0] = sys.executable
    elif toks[0] == "pip":
        toks = [sys.executable, "-m", "pip"] + toks[1:]
    return env, toks


def idle_real_workers(plan_json: Path):
    """Real-dataset plans with idle workers although p <= W (R8 stop condition)."""
    plans = json.loads(plan_json.read_text())["plans"]
    return [p for p in plans
            if p["dataset"] in REAL_W and p["threads"] <= p["W"] and p["idle_workers"] > 0]


def external_check_matches(check_json: Path) -> bool:
    rows = json.loads(check_json.read_text()).get("rows", [])
    return bool(rows) and all(r.get("matched_semantics") for r in rows)


def preserve_outputs(outputs, stamp):
    moved = []
    for name in outputs:
        p = (ROOT / "results" / name) if "/" in name else RES / name
        if p.exists():
            q = p.with_name(f"{p.stem}.{stamp}{p.suffix}")
            p.rename(q)
            moved.append([str(p.name), str(q.name)])
    return moved


def git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                           text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


VERSION_RE = re.compile(r"^STRATA-v(\d+)$")


def version_folders(root=None):
    """(this STRATA-vN folder, newest sibling STRATA-vN folder), or (None, None)
    outside that layout (a public copy of the code, for example). v26, R71."""
    here = Path(root or ROOT).resolve()
    for d in [here, *here.parents]:
        m = VERSION_RE.match(d.name)
        if m:
            sib = [(int(VERSION_RE.match(x.name).group(1)), x) for x in d.parent.iterdir()
                   if x.is_dir() and VERSION_RE.match(x.name)]
            return d, max(sib)[1]
    return None, None


def check_version(root=None, allow_old=False):
    """Message if this is not the newest STRATA-vN folder (None if fine)."""
    cur, new = version_folders(root)
    if cur is None or cur == new or allow_old:
        return None
    return (f"this is {cur.name}, but {new.name} exists; run the session in "
            f"{new / 'STRATA-code' / '05_Temporal_STRATA'} (or pass --allow-old)")


def unknown_steps(wanted):
    """Requested step numbers that the runner does not have (v26, R71)."""
    have = {s[0] for s in STEPS}
    return sorted(set(wanted or ()) - have)


def _version_of(path):
    for part in Path(path).parts:
        m = VERSION_RE.match(part)
        if m:
            return part
    return None


def environment_record(cwd=None, prefix=None, executable=None, runner=None):
    """Where the interpreter, the working folder, and this runner live, with the
    home folder shown as '~', and a warning if they name different STRATA-vN
    folders (v26, R72; the v25 session ran the v24 environment unnoticed)."""
    home = os.path.expanduser("~")
    def tidy(p):
        p = str(p)
        return "~" + p[len(home):] if home and p.startswith(home) else p
    rec = {"cwd": tidy(cwd or os.getcwd()), "sys_prefix": tidy(prefix or sys.prefix),
           "sys_executable": tidy(executable or sys.executable),
           "runner": tidy(runner or Path(__file__).resolve())}
    names = {k: _version_of(v) for k, v in rec.items()}
    distinct = {v for v in names.values() if v}
    if len(distinct) > 1:
        rec["warning"] = "different STRATA-vN folders: " + ", ".join(
            f"{k}={v}" for k, v in names.items() if v)
    return rec


def run(args):
    wanted = ({int(x) for x in args.steps.split(",") if x.strip()} if args.steps else None)
    bad = unknown_steps(wanted)
    if bad:
        print(f"unknown step(s) {bad}; this runner has steps {sorted(s[0] for s in STEPS)}")
        return 4
    msg = check_version(allow_old=getattr(args, "allow_old", False))
    if msg:
        print("STOPPED before any measurement: " + msg)
        return 4
    wanted = with_regen(wanted, getattr(args, "no_regen", False))
    selected = [s for s in STEPS
                if (args.only_step is None or s[0] == args.only_step)
                and (wanted is None or s[0] in wanted)
                and s[0] >= args.from_step]
    if args.only_external:
        selected = [(num, title + " (external baseline only)",
                     [c for i, c in enumerate(cmds) if i in OPTIONAL_EXTERNAL], outs[1:])
                    for num, title, cmds, outs in STEPS if num == 6]
    if args.dry_run:
        for num, title, cmds, outs in selected:
            print(f"Step {num} -- {title}")
            for c in cmds:
                print(f"  $ {c}")
            print(f"  outputs: {', '.join(outs)}")
        return 0

    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    sys.path.insert(0, str(ROOT))
    from temporal_strata.parallel import machine_info

    cur, new = version_folders()
    envrec = environment_record()
    if envrec.get("warning"):
        print("WARNING: " + envrec["warning"])
    session = {"started": _dt.datetime.now().isoformat(timespec="seconds"),
               "machine": machine_info(), "git_commit": git_commit(),
               "python": sys.executable, "environment": envrec,
               "version_folder": cur.name if cur else None,
               "latest_version_folder": new.name if new else None,
               "steps_available": sorted(s[0] for s in STEPS),
               "steps_requested": sorted(wanted) if wanted else None, "steps": []}
    spath = LOGS / f"session.{stamp}.json"

    def save():
        spath.write_text(json.dumps(session, indent=2))

    try:
        for num, title, cmds, outs in selected:
            rec = {"step": num, "title": title,
                   "preserved": preserve_outputs(outs, stamp), "commands": []}
            session["steps"].append(rec)
            print(f"=== Step {num}: {title}")
            external_ok = None
            offset = 1 if args.only_external else 0  # keep step-6 command indices
            for i0, c in enumerate(cmds):
                i = i0 + offset
                if num == 6 and OPTIONAL_EXTERNAL.get(i) == "timed" and not external_ok:
                    rec["commands"].append({"command": c, "status": "skipped",
                                            "reason": "external check did not report a "
                                                      "matching semantics (or was skipped)"})
                    print("  skipped timed external run (no matching semantics)")
                    continue
                if rec.get("skip_rest"):
                    rec["commands"].append({"command": c, "status": "skipped",
                                            "reason": rec["skip_rest"]})
                    print(f"  skipped: {rec['skip_rest']}")
                    continue
                env_over, argv = to_argv(c)
                log = LOGS / f"step{num}_{i + 1}.{stamp}.log"
                print(f"  $ {c}\n    -> {log.name}")
                with open(log, "w") as fh:
                    r = subprocess.run(argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                       env={**os.environ, **env_over})
                entry = {"command": c, "exit_code": r.returncode, "log": log.name}
                rec["commands"].append(entry)
                save()
                if num == 6 and OPTIONAL_EXTERNAL.get(i) in ("install", "check"):
                    if r.returncode != 0:
                        entry["status"] = "skipped-external"
                        external_ok = False
                        print(f"    external baseline unavailable (exit {r.returncode}); "
                              "continuing without it")
                        continue
                    if OPTIONAL_EXTERNAL[i] == "check":
                        external_ok = external_check_matches(RES / "external_check.json")
                    continue
                opt = OPTIONAL_COMMANDS.get((num, i))
                if r.returncode != 0 and opt and (opt[1] is None or r.returncode in opt[1]):
                    entry["status"] = "skipped-optional"
                    rec["skip_rest"] = opt[0]
                    print(f"    optional command failed (exit {r.returncode}): "
                          f"{opt[0]}; rest of step {num} skipped")
                    save()
                    continue
                if r.returncode != 0:
                    raise StopSession(f"step {num} command failed (exit {r.returncode}): {c}")
                if num == 1 and "--plan-only" in c:
                    bad = idle_real_workers(RES / "plan_check.json")
                    if bad:
                        entry["idle_workers"] = bad
                        raise StopSession("plan check shows idle workers on real data at "
                                          "p <= W; send results/real/plan_check.json "
                                          "instead of timing (backlog R8)")
            if num == REGEN_STEP:
                rec["regenerated"] = regenerated_files()
                save()
        session["status"] = "completed"
        return_code = 0
    except StopSession as exc:
        session["status"] = "stopped"
        session["stop_reason"] = str(exc)
        print(f"STOPPED: {exc}")
        return_code = 1
    session["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    save()
    print(f"session record: {spath}")
    return return_code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-regen", action="store_true",
                    help="do not add step 13 (regenerate paper files) after measurement steps")
    ap.add_argument("--from-step", type=int, default=1)
    ap.add_argument("--only-step", type=int, default=None)
    ap.add_argument("--steps", type=str, default=None,
                    help="comma list of steps to run, e.g. 1,2,3,6,7")
    ap.add_argument("--allow-old", action="store_true",
                    help="run even if a newer STRATA-vN folder exists (v26)")
    ap.add_argument("--only-external", action="store_true",
                    help="run only the external-baseline commands of step 6")
    sys.exit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
