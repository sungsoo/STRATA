#!/usr/bin/env python3
"""Every M5 Max number the paper cites, derived from the measured JSON files.

STRATA-v8 (backlog B / R4-R6 / R9 / R12 / R13). The manuscript never types an
M5 performance number by hand: it uses the LaTeX macros this script writes to
``STRATA-latex/STRATA-paper-EN/m5_numbers.tex``, and the supplementary scaling
tables (milliseconds) it writes next to them. ``tools/check_numbers.py`` runs
this script in ``--check`` mode and fails if any emitted file differs from a
fresh derivation, so a hand edit or a stale file cannot pass the gate.

Inputs (all measured on the same Apple M5 Max by tools/run_m5_session.py):
  results/real/parallel_scaling.json        real data + synthetic n=8000
  results/real/parallel_scaling_large.json  synthetic n=50000 / n=100000
  results/real/compiled_real.json           compiled-vs-compiled, one thread
  results/real/bandwidth_roofline.json      STREAM triad + kernel bandwidth
  results/real/layout_crossover_p6.json / _p18.json
  results/real/backend_bench.json           PyTorch MPS vs compiled core
  results/real/external_check.json / external_witness.json
  results/real/temporal_msbfs.json          time-ordered MS-BFS baseline (v23, R59)
  results/real/large_real_blocked.json      sx-stackoverflow, beta=4096, lexsort prep (v21)
  results/real/large_real_b4096.json / _b65536.json / _full.json
                                            block-width x thread sweep, every block (v22)
  results/real/thread_diagnosis.json        static / dynamic x threads, bandwidth (v24, R65)
  results/real/thread_chunks.json           chunks versus concurrent threads (v25, R70)
  results/real/tglib_build / _check / _time.json   TGLib built from source (v26, R57 / R68)
  results/real/random_rows.json             random row updates by threads (v26, R69)
  results/real/kernel_phases.json           kernel phases by threads (v28, R77)
  results/real/kernel_copy.json             copy-phase causes and the tile state (v29, R75)
  results/real/large_real_tiles_b65536.json tile-state thread sweep, two blocks (v29, R76)
  results/real/large_real_tiles_full.json   tile-state run over every block (v29, R76)
  results/real/kernel_prefault.json         copies after every page was written (v31, R90)
  results/real/large_real_tiles_b4096.json, large_real_tiles_dyn_b4096.json,
  large_real_tiles_dyn_b65536.json          tile state: narrow blocks, dynamic (v31, R91)
  results/real/large_real_tiles_b16384.json, large_real_tiles_b32768.json
                                            tile state across block widths (v32, R94)
  results/real/large_real_tiles_full_b32768.json  every block at beta = 32,768 (v32, R98)
  results/real/op_counts.json               elementary-step counts, BFS vs STRATA (v34, R105)
  results/real/tile_state_unblocked.json    unblocked: fresh / pre-written / tile state (v35, R102 / R112)
  results/real/window_large.json            rolling windows and insertion, sx-askubuntu (v35, R109)
  results/real/window_insert.json           insertion, tiled vs row-major in place (v36, R113)

Definitions used in the text and tables
  * speedup at p threads = time of the fastest one-thread configuration
    (tiled/static on every dataset; asserted below) / time at p threads;
  * end-to-end = shared sequential preprocessing + kernel (file loading is
    recorded separately and excluded, as in the harness);
  * parallel-vs-parallel ratio = compiled parallel per-source BFS time /
    STRATA end-to-end time, both at the same p.
No number from another machine is read, and no two machines are compared.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REAL = ROOT / "results" / "real"
PAPER = ROOT.parent.parent / "STRATA-latex" / "STRATA-paper-EN"

SHORT = {"CollegeMsg": "College", "BitcoinOTC": "Otc", "BitcoinOTCpos": "OtcPos"}
LABEL = {"CollegeMsg": "CollegeMsg", "BitcoinOTC": "Bitcoin-OTC",
         "BitcoinOTCpos": "Bitcoin-OTC(+)"}


def load(name):
    return json.loads((REAL / name).read_text())


def synth_label(r):
    return f"Synthetic $n{{=}}{r['n']:,}$".replace(",", "{,}")


def short(r):
    if r["dataset"] in SHORT:
        return SHORT[r["dataset"]]
    return {8000: "SynA", 50000: "SynB", 100000: "SynC"}[r["n"]]


def label(r):
    return LABEL.get(r["dataset"]) or synth_label(r)


def rows_of(r, layout, schedule):
    return sorted((x for x in r["rows"] if x["layout"] == layout and x["schedule"] == schedule),
                  key=lambda x: x["threads"])


def base_one_thread(r):
    """Fastest p=1 configuration; asserts it is tiled/static (the default)."""
    ones = [x for x in r["rows"] if x["threads"] == 1]
    best = min(ones, key=lambda x: x["kernel_s"])
    if (best["layout"], best["schedule"]) != ("tiled", "static"):
        raise SystemExit(f"{r['dataset']}: fastest p=1 config is {best['layout']}/{best['schedule']}")
    return best


PLATEAU_TOL = 0.05     # R34: thread counts whose median is within 5% of the lowest median


def plateau(rows, tol=PLATEAU_TOL, sep=", "):
    """Thread counts of one configuration (same layout / schedule) whose median
    kernel time is within ``tol`` of the lowest median, written as runs over the
    measured thread grid ("12--18", "6, 12--18"). The best single p moves from
    run to run on flat curves (v9 / v10 re-measurements); this band does not."""
    grid = sorted(rows, key=lambda x: x["threads"])
    med = [median(x.get("kernel_samples_s") or [x["kernel_s"]]) for x in grid]
    best = min(med)
    sel = [i for i, v in enumerate(med) if v <= (1 + tol) * best]
    runs, start = [], sel[0]
    for a, b in zip(sel, sel[1:] + [None]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    out = []
    for a, b in runs:
        pa, pb = grid[a]["threads"], grid[b]["threads"]
        out.append(str(pa) if a == b else f"{pa}--{pb}")
    return sep.join(out)


def config_rows(r, x):
    """All rows of the configuration (layout, schedule) that row x belongs to."""
    return [y for y in r["rows"] if y["layout"] == x["layout"] and y["schedule"] == x["schedule"]]


def fmt(x, d=1):
    return f"{x:.{d}f}"


def rng(vals, d=1, sep="--"):
    lo, hi = min(vals), max(vals)
    a, b = fmt(lo, d), fmt(hi, d)
    return a if a == b else f"{a}{sep}{b}"


def derive():
    m = {}          # macro name -> text
    ps = load("parallel_scaling.json")
    pl = load("parallel_scaling_large.json")
    machine = ps["machine"]
    m["MMachine"] = machine["summary"].split(", Python")[0]
    m["MSuper"] = str(machine["perf_levels"][0]["physicalcpu"])
    m["MThreads"] = str(machine["cpu_count"])
    m["MRepeats"] = str(ps["machine"].get("repeats", ps.get("repeats", 5)))

    real_k, real_e, rt, bfs1, bfsP, prep_share = [], [], [], [], [], []
    for r in ps["results"]:
        s = short(r)
        b = base_one_thread(r)
        ts = rows_of(r, "tiled", "static")
        rs = rows_of(r, "rowmajor", "static")
        td = rows_of(r, "tiled", "dynamic")
        kbest = min(ts, key=lambda x: x["kernel_s"])
        ebest = min(ts, key=lambda x: x["end_to_end_s"])
        ksp = b["kernel_s"] / kbest["kernel_s"]
        esp = b["end_to_end_s"] / ebest["end_to_end_s"]
        m[f"M{s}W"] = str(r["W"])
        m[f"M{s}KOne"] = fmt(b["kernel_s"] * 1e3, 2)
        m[f"M{s}Ksp"] = fmt(ksp, 1)
        m[f"M{s}Kp"] = str(kbest["threads"])
        m[f"M{s}Plat"] = plateau(ts)
        m[f"M{s}Esp"] = fmt(esp, 1)
        m[f"M{s}Prep"] = fmt(r["prep_s"] * 1e3, 2)
        share = 100 * r["prep_s"] / ebest["end_to_end_s"]
        m[f"M{s}PrepShare"] = fmt(share, 0)
        rowtile = rs[0]["kernel_s"] / ts[0]["kernel_s"]
        m[f"M{s}RowTile"] = fmt(rowtile, 1)
        dyn_slow = td[-1]["kernel_s"] / ts[-1]["kernel_s"]
        m[f"M{s}DynSlow"] = fmt(dyn_slow, 1)
        rt.append(rowtile)
        if r["dataset"] in SHORT:
            real_k.append(ksp)
            real_e.append(esp)
            prep_share.append(share)
        if ts[0].get("persrc_bfs_s"):
            bfs1.append(ts[0]["persrc_bfs_s"] / ts[0]["end_to_end_s"])
            bfsP.append(ts[-1]["persrc_bfs_s"] / ts[-1]["end_to_end_s"])
    m["MRealKspRange"] = rng(real_k)
    m["MRealEspRange"] = rng(real_e)
    m["MRealPrepShareRange"] = rng(prep_share, 0)
    m["MBfsOneRange"] = rng(bfs1, 0)
    m["MBfsMaxRange"] = rng(bfsP, 0)
    m["MPmax"] = str(max(x["threads"] for x in ps["results"][0]["rows"]))

    for r in pl["results"]:
        s = short(r)
        b = base_one_thread(r)
        ts = rows_of(r, "tiled", "static")
        td = rows_of(r, "tiled", "dynamic")
        rs = rows_of(r, "rowmajor", "static")
        ks = min(ts, key=lambda x: x["kernel_s"])
        kd = min(td, key=lambda x: x["kernel_s"])
        ed = min(ts + td, key=lambda x: x["end_to_end_s"])
        m[f"M{s}W"] = str(r["W"])
        m[f"M{s}KOne"] = fmt(b["kernel_s"], 2)          # seconds
        m[f"M{s}KspStatic"] = fmt(b["kernel_s"] / ks["kernel_s"], 1)
        m[f"M{s}KpStatic"] = str(ks["threads"])
        m[f"M{s}PlatStatic"] = plateau(ts)
        m[f"M{s}KspDyn"] = fmt(b["kernel_s"] / kd["kernel_s"], 1)
        m[f"M{s}KpDyn"] = str(kd["threads"])
        m[f"M{s}PlatDyn"] = plateau(td)
        m[f"M{s}Esp"] = fmt(b["end_to_end_s"] / ed["end_to_end_s"], 1)
        m[f"M{s}Prep"] = fmt(r["prep_s"] * 1e3, 0)
        m[f"M{s}PrepShare"] = fmt(100 * r["prep_s"] / ed["end_to_end_s"], 0)
        rt.append(rs[0]["kernel_s"] / ts[0]["kernel_s"])
    m["MRowTileRange"] = rng(rt)
    m["MDynGain"] = fmt(max(float(m[f"M{x}KspDyn"]) / float(m[f"M{x}KspStatic"])
                            for x in ("SynB", "SynC")), 1)

    cr = load("compiled_real.json")
    m["MCompKernelRange"] = rng([x["kernel_ratio_bfs_over_strata"] for x in cr["rows"]], 0)
    m["MCompEtoERange"] = rng([x["end_to_end_ratio_bfs_over_strata"] for x in cr["rows"]], 0)
    m["MCompKernelGeo"] = fmt(math.exp(sum(math.log(x["kernel_ratio_bfs_over_strata"])
                                       for x in cr["rows"]) / len(cr["rows"])), 0)

    bw = load("bandwidth_roofline.json")
    peak = max(bw["stream_triad"]["rows"], key=lambda x: x["GBps"])
    m["MStreamPeak"] = fmt(peak["GBps"], 0)
    m["MStreamP"] = str(peak["threads"])
    big = max(bw["kernels"], key=lambda k: k["n"])
    bb = max(big["rows"], key=lambda x: x["achieved_GBps"])
    m["MBigBw"] = fmt(bb["achieved_GBps"], 0)
    m["MBigBwPct"] = fmt(100 * bb["achieved_GBps"] / peak["GBps"], 0)
    m["MBigBwN"] = f"{big['n']:,}".replace(",", "{,}")
    cache = [k for k in bw["kernels"] if max(x["ratio_best_stream"] for x in k["rows"]) > 1]
    m["MCacheBwRatio"] = rng([max(x["ratio_best_stream"] for x in k["rows"]) for k in cache], 2) \
        if cache else "--"

    xo = {}
    for p in (6, 18):
        d = load(f"layout_crossover_p{p}.json")
        ratios = [q["rowmajor_s"] / q["tiled_s"] for q in d["points"]]
        xo[p] = d
        m[f"MXover{'Six' if p == 6 else 'Max'}"] = rng(ratios, 1)
        if not all(q["faster"] == "tiled" for q in d["points"]):
            raise SystemExit(f"layout crossover p={p}: tiled not faster everywhere; revise text")
    m["MXoverPoints"] = str(len(xo[6]["points"]))
    m["MXoverAgree"] = fmt(100 * xo[6]["summary"]["agreement"], 0)

    be = load("backend_bench.json")
    for mode, key in (("strict", "MMpsStrict"), ("closure", "MMpsClosure")):
        m[key] = rng([x["torch_s"] / x["numba_s"] for x in be["rows"] if x["mode"] == mode], 0)

    ex = load("external_check.json")
    wi = load("external_witness.json")
    m["MExtVersion"] = ex["tool"]["version"]
    m["MExtToys"] = str(len(ex["toy_diagnosis"]))
    m["MExtToysClosure"] = str(sum(1 for t in ex["toy_diagnosis"] if t.get("matches_closure")))
    m["MExtTotal"] = str(wi["summary"]["sampled_sources"])
    m["MExtAgree"] = str(wi["summary"]["agreeing_sources"])
    m["MExtMissed"] = str(wi["summary"]["missed_vertices"])
    m["MExtWitness"] = str(wi["summary"]["missed_with_valid_witness"])
    m["MExtExtra"] = str(wi["summary"]["external_only_vertices"])
    hops = [x["hops"] for r in wi["rows"] for d in r["disagreements"] for x in d["missed_by_external"]]
    m["MExtHops"] = rng(hops, 0)
    m["MExtPerSet"] = str(wi["rows"][0]["sources"])
    return m, ps, pl


def median(xs):
    xs = sorted(xs)
    k = len(xs)
    return xs[k // 2] if k % 2 else 0.5 * (xs[k // 2 - 1] + xs[k // 2])


def has_samples(*result_sets):
    """True once the harness stores every repeat (STRATA-v9, R26)."""
    return all("kernel_samples_s" in x for rs in result_sets for r in rs["results"]
               for x in r["rows"])


def cell(x, key, digits, scale=1e3):
    """Best time; with stored samples also the median, as 'best (median)'."""
    v = f"{x[key] * scale:.{digits}f}"
    sk = {"kernel_s": "kernel_samples_s"}.get(key)
    if sk and sk in x:
        v += f" ({median(x[sk]) * scale:.{digits}f})"
    return v


def macros_tex(m):
    out = ["% Auto-generated by experiments/m5_numbers.py from the M5 Max result files.",
           "% Do not edit by hand: tools/check_numbers.py regenerates and compares.",
           f"% Machine: {m['MMachine']}"]
    for k in sorted(m):
        out.append(f"\\newcommand{{\\{k}}}{{{m[k]}}}")
    return "\n".join(out) + "\n"


def table_small(ps):
    lines = [
        "% Auto-generated by experiments/m5_numbers.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Strong scaling on %s (tiled layout, static schedule; best-of-%s after"
        " warm-up). Times in ms; speedups relative to the fastest one-thread"
        " configuration; E2E = shared sequential preprocessing + kernel; last column ="
        " compiled \\emph{parallel} per-source BFS time over \\strata\\ E2E at the same $p$."
        " Every configuration was verified bit-identical to the sequential core before"
        " timing.%s}" % (ps["machine"]["summary"].split(", Python")[0],
                         ps["machine"].get("repeats", 5),
                         " Kernel cells give best (median) over the repeats." if has_samples(ps) else ""),
        "\\label{tab:parscaling}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lrrrrrr@{}}",
        "\\toprule",
        "Dataset & $p$ & Kernel & Sp. & E2E & Sp. & BFS/E2E \\\\",
        "\\midrule",
    ]
    for r in ps["results"]:
        b = base_one_thread(r)
        for i, x in enumerate(rows_of(r, "tiled", "static")):
            bfs = x.get("persrc_bfs_s")
            lines.append(
                f"{label(r) if i == 0 else ''} & {x['threads']} & {cell(x, 'kernel_s', 3)} & "
                f"{b['kernel_s']/x['kernel_s']:.2f} & {x['end_to_end_s']*1e3:.3f} & "
                f"{b['end_to_end_s']/x['end_to_end_s']:.2f} & "
                + (f"{bfs/x['end_to_end_s']:.1f}" if bfs else "--") + " \\\\")
        if r is not ps["results"][-1]:
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def table_large(pl):
    lines = [
        "% Auto-generated by experiments/m5_numbers.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Strong scaling on the large synthetic streams (same machine; tiled"
        " layout). Kernel times in ms for the static and dynamic schedules; speedups"
        " relative to the fastest one-thread configuration (tiled/static); E2E uses the"
        " faster schedule at each $p$. Correctness reference: the independent compiled"
        " full-copy kernel (bit-identical before timing).%s}" % (
            " Kernel cells give best (median) over the repeats." if has_samples(pl) else ""),
        "\\label{tab:parscaling-large}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lrrrrrr@{}}",
        "\\toprule",
        "Dataset & $p$ & Static & Sp. & Dynamic & Sp. & E2E \\\\",
        "\\midrule",
    ]
    for r in pl["results"]:
        b = base_one_thread(r)
        st, dy = rows_of(r, "tiled", "static"), rows_of(r, "tiled", "dynamic")
        for i, (x, y) in enumerate(zip(st, dy)):
            e = min(x["end_to_end_s"], y["end_to_end_s"])
            lines.append(
                f"{label(r) if i == 0 else ''} & {x['threads']} & {cell(x, 'kernel_s', 0)} & "
                f"{b['kernel_s']/x['kernel_s']:.2f} & {cell(y, 'kernel_s', 0)} & "
                f"{b['kernel_s']/y['kernel_s']:.2f} & {e*1e3:.0f} \\\\")
        if r is not pl["results"][-1]:
            lines.append("\\addlinespace")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def table_summary(ps, pl, rb=None, so=None):
    """Main-paper summary table: one row per dataset (STRATA-v8)."""
    lines = [
        "% Auto-generated by experiments/m5_numbers.py -- do not edit by hand.",
        "\\setlength{\\tabcolsep}{1.2pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lrrrrrrr@{}}",
        "\\toprule",
        "Dataset & $W$ & \\shortstack[r]{Kernel\\\\$p{=}1$ (ms)} & \\shortstack[r]{Kernel\\\\sp.\\ ($p$)}"
        " & \\shortstack[r]{E2E\\\\sp.} & \\shortstack[r]{Prep.\\\\(\\%%)$^\\P$} & \\shortstack[r]{BFS/E2E\\\\$p{=}1$}"
        " & \\shortstack[r]{BFS/E2E\\\\$p{=}%d$} \\\\" % max(x["threads"] for x in ps["results"][0]["rows"]),
        "\\midrule",
    ]
    for r in ps["results"] + pl["results"]:
        b = base_one_thread(r)
        cand = rows_of(r, "tiled", "static") + (rows_of(r, "tiled", "dynamic") if r in pl["results"] else [])
        kb = min(cand, key=lambda x: x["kernel_s"])
        eb = min(cand, key=lambda x: x["end_to_end_s"])
        ts = rows_of(r, "tiled", "static")
        bfs = ts[0].get("persrc_bfs_s")
        tag = "$^\\ddagger$" if r in pl["results"] and kb["schedule"] == "dynamic" else ""
        lines.append(
            f"{label(r).replace('Synthetic', 'Synth.')} & {r['W']} & {b['kernel_s']*1e3:.2f} & "
            f"{b['kernel_s']/kb['kernel_s']:.1f} ({plateau(config_rows(r, kb), sep=',')}){tag} & "
            f"{b['end_to_end_s']/eb['end_to_end_s']:.1f} & "
            f"{100*r['prep_s']/b['end_to_end_s']:.0f}/{100*r['prep_s']/eb['end_to_end_s']:.0f} & "
            + (f"{bfs/ts[0]['end_to_end_s']:.0f} & {ts[-1]['persrc_bfs_s']/ts[-1]['end_to_end_s']:.0f}"
               if bfs else "-- & --") + " \\\\")
    for r in (rb or {}).get("results", []):
        one = min((x for x in r["rows"] if x["threads"] == 1), key=lambda x: x["kernel_s"])
        kb = min(r["rows"], key=lambda x: x["kernel_s"])
        eb = min(r["rows"], key=lambda x: x["end_to_end_s"])
        e1 = min((x for x in r["rows"] if x["threads"] == 1), key=lambda x: x["end_to_end_s"])
        tag = "$^\\ddagger$" if kb["schedule"] == "dynamic" else ""
        lines.append(f"{r['dataset']}$^\\S$ & {r['W']} & {one['kernel_s']*1e3:.2f} & "
                     f"{one['kernel_s']/kb['kernel_s']:.1f} ({plateau(config_rows(r, kb), sep=',')}){tag} & "
                     f"{e1['end_to_end_s']/eb['end_to_end_s']:.1f} & "
                     f"{100*r['prep_s']/e1['end_to_end_s']:.0f}/{100*r['prep_s']/eb['end_to_end_s']:.0f} & -- & -- \\\\")
    if so is not None:
        # v31 (R92): sx-stackoverflow, tile state, kernel per source block; the
        # E2E / BFS columns do not apply to a blocked run (no one-thread full run).
        tb, tf = so
        t = {r["threads"]: r["kernel_s_mean_block"] for r in tb["runs"]}
        best = min(t, key=t.get)
        full = tf["runs"][0]["all_blocks_s"]
        lines.append(f"sx-stackoverflow$^\\ast$ & {(tb['block_bits'] + 63) // 64} & "
                     f"{t[1]*1e3:.0f} & {t[1]/t[best]:.1f} ({best}) & -- & "
                     f"--/{100*tf['prep']['total_s']/(tf['prep']['total_s'] + full):.0f} & -- & -- \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# STRATA-v10 (backlog R20 / R25 / R26 / R27 / R29 / R31): single-thread suite,
# repeat spread, large public graphs, closure on bucketed timestamps, parallel
# preprocessing, GraphBLAS reference point. Same machine, same rules as above.
# ---------------------------------------------------------------------------
PERF = ROOT / "results" / "perf"
sys.path.insert(0, str(HERE))
from thread_record import check_one_thread  # noqa: E402  (R37)
SAMPLED_SOURCES = 64          # experiments/bench_common.sampled_reference default
# v9 M5 session result for the large public graphs, preserved by the v10 run
V9_REALBIG = "parallel_scaling_realbig.20261116-144107.json"


def load_perf(name):
    return json.loads((PERF / name).read_text())


def maybe(name):
    p = REAL / name
    return json.loads(p.read_text()) if p.exists() else None


def big(x):
    return f"{x:,}".replace(",", "{,}")


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def derive_v10(m, ps, pl):
    extra = {}
    # ---- R25: single-thread suite on the M5 (NUMBA_NUM_THREADS=1) ----------
    cb = load_perf("compiled_baselines.json")
    pt = load_perf("perf_table.json")
    # R37: every single-thread file must record one Numba thread (and one
    # GraphBLAS thread where GraphBLAS was timed) before a number is printed
    check_one_thread(cb.get("threads"), "compiled_baselines.json")
    check_one_thread(pt.get("threads"), "perf_table.json")
    for r in cb["rows"]:
        if r.get("graphblas_static_ms") is not None and r.get("graphblas_threads") != 1:
            raise SystemExit(f"compiled_baselines.json {r['label']}: GraphBLAS ran with "
                             f"{r.get('graphblas_threads')} threads, not 1")
    syn_cb = [r for r in cb["rows"] if r["label"].startswith("synth")]
    top = max(syn_cb, key=lambda r: r["n"])
    m["MStTopN"] = big(top["n"])
    m["MStPyMs"] = fmt(top["strata_py_ms"], 1)
    m["MStNumbaMs"] = fmt(top["strata_numba_ms"], 1)
    m["MStDenseMs"] = fmt(top["dense_numpy_ms"], 0)
    m["MStDenseOverPy"] = fmt(top["dense_numpy_ms"] / top["strata_py_ms"], 0)
    m["MStPyNumbaRatio"] = fmt(max(top["strata_py_ms"], top["strata_numba_ms"])
                               / min(top["strata_py_ms"], top["strata_numba_ms"]), 1)
    # v28 (R79): new records time the production core; the ratio divides kernel by
    # kernel. Older records (legacy whole-state-snapshot kernel) keep the old field.
    m["MStNumbaCore"] = "production" if "strata_numba_core" in top else "legacy"
    cc = [r["exact_bfs_ms"] and r["compiled_bfs_ms"] / r.get("strata_numba_kernel_ms", r["strata_numba_ms"])
          for r in syn_cb]
    m["MCcRange"] = rng(cc, 1)
    m["MCcGeo"] = fmt(geo(cc), 2)
    # R36: GraphBLAS pinned to one thread (checked above) -> interpreted STRATA
    # time over single-thread GraphBLAS time on the same static closure
    gbr = [r["strata_py_ms"] / r["graphblas_static_ms"] for r in cb["rows"]
           if r.get("graphblas_static_ms")]
    if len(gbr) != len(cb["rows"]):
        raise SystemExit("compiled_baselines.json: GraphBLAS column incomplete")
    m["MGbPyRange"] = rng(gbr, 1)
    syn_pt = [r for r in pt["rows"] if r["label"].startswith("synth")]
    ego = [r for r in pt["rows"] if not r["label"].startswith("synth")][0]
    m["MStGeoSpeed"] = fmt(geo([r["speedup"] for r in pt["rows"]]), 1)
    m["MStSpeedRange"] = rng([r["speedup"] for r in syn_pt], 0)
    m["MStEgoSpeed"] = fmt(ego["speedup"], 0)
    rt = load("real_temporal_results.json")
    check_one_thread(rt.get("threads"), "real_temporal_results.json")
    for r in rt["datasets"]:
        key = {"CollegeMsg": "College", "Bitcoin-OTC": "Otc", "Bitcoin-OTC(+)": "OtcPos"}[r["dataset"]]
        m[f"MRt{key}Speed"] = fmt(r["speedup_strict"], 0)
    # touched-row snapshot on the observed streams: rows written per group (a count)
    groups = {"CollegeMsg": "CollegeMsg", "BitcoinOTC": "Bitcoin-OTC", "BitcoinOTCpos": "Bitcoin-OTC(+)"}
    tg = {r["dataset"]: r["timestamp_groups"] for r in rt["datasets"]}
    dmean = []
    for r in ps["results"]:
        if r["dataset"] in groups:
            dmean.append(r["dirty_total"] / tg[groups[r["dataset"]]])
    m["MDirtyMeanRange"] = rng(dmean, 2)
    sp = load("ablation_strict_prune.json")
    check_one_thread(sp.get("threads"), "ablation_strict_prune.json")
    for r in sp["rows"]:
        key = {"CollegeMsg": "College", "Bitcoin-OTC": "Otc", "Bitcoin-OTC(+)": "OtcPos"}[r["dataset"]]
        m[f"MPrune{key}"] = fmt(r["prune_over_noprune_x"], 2)
    ix = load("index_crossover.json")
    check_one_thread(ix.get("threads"), "index_crossover.json")
    m["MIdxBuildRange"] = rng([r["build_index_s"] / r["per_source_bfs_s"] for r in ix["rows"]], 1)
    m["MIdxQueryUs"] = rng([r["point_query_us"] for r in ix["rows"]], 2)
    m["MIdxCross"] = fmt(max(r["crossover_sources"] for r in ix["rows"]), 0)
    m["MIdxAllCollege"] = fmt(ix["rows"][0]["allpairs_index_speedup"], 0)
    m["MIdxAllOtc"] = fmt(ix["rows"][1]["allpairs_index_speedup"], 0)
    dy = load("real_dynamic_blocking.json")
    check_one_thread(dy.get("threads"), "real_dynamic_blocking.json")
    m["MDynInsRange"] = rng([r["speedup"] for r in dy["insertion_CollegeMsg"]], 1)
    m["MDynWinRange"] = rng([r["seconds"] * 1e3 for r in dy["window_CollegeMsg"]], 1)
    m["MDynWinSlots"] = str(len(dy["window_CollegeMsg"]))
    dl = {r["cut"]: r for r in dy["deletion_CollegeMsg"]}
    m["MDynDel"] = "/".join(fmt(dl[c]["repair_s"] * 1e3, 1) for c in ("early", "middle", "late"))
    m["MDynPerK"] = rng([r["repair_s"] * 1e3 / (r["suffix_edges"] / 1e3) for r in dy["deletion_CollegeMsg"]], 2)
    bl = sorted(dy["blocking_BitcoinOTC"], key=lambda r: r["passes"])
    m["MBlkOneMs"] = fmt(bl[0]["seconds"] * 1e3, 0)
    m["MBlkMaxMs"] = fmt(bl[-1]["seconds"] * 1e3, 0)
    m["MBlkMaxPasses"] = str(bl[-1]["passes"])
    m["MBlkOneWords"] = f"{bl[0]['resident_words'] / 1e3:.0f}k"
    m["MBlkMaxWords"] = f"{bl[-1]['resident_words'] / 1e3:.0f}k"
    if not all(r["bit_identical"] for k in dy if k != "threads" for r in dy[k] if "bit_identical" in r):
        raise SystemExit("dynamic / blocking run not bit-identical")
    # per-edge throughput at p=1 versus word width (strict kernel, tiled/static)
    thr = []
    for r in ps["results"] + pl["results"]:
        if r["dataset"] in SHORT:
            continue
        b = base_one_thread(r)
        thr.append((r["W"], r["num_directed_edges"] / b["kernel_s"] / 1e6))
    thr.sort()
    m["MThruHi"] = fmt(thr[0][1], 1)
    m["MThruLo"] = fmt(thr[-1][1], 2)
    m["MThruWlo"] = str(thr[0][0])
    m["MThruWhi"] = big(thr[-1][0])
    wps = [w * e for w, e in thr]
    m["MThruWordRatio"] = fmt(max(wps) / min(wps), 1)

    # ---- R26: spread of the stored repeats ----------------------------------
    spread = []
    for rs in (ps, pl, maybe("parallel_scaling_realbig.json") or {"results": []}):
        for r in rs["results"]:
            for x in r["rows"]:
                s = x["kernel_samples_s"]
                spread.append(median(s) / min(s) - 1)
    spread.sort()
    m["MSpreadMed"] = fmt(100 * median(spread), 1)
    m["MSpreadNinety"] = fmt(100 * spread[int(0.9 * (len(spread) - 1))], 1)
    m["MSpreadMax"] = fmt(100 * spread[-1], 0)
    m["MSpreadN"] = str(len(spread))

    # ---- R20: parallel preprocessing (synthetic streams above the auto cut) --
    gains, e2e = [], []
    for r in ps["results"] + pl["results"]:
        if r["m"] < (1 << 17):
            continue
        par = [x for x in r["rows"] if x.get("prep_par")]
        pbest = min(x["prep_par"]["total_s"] for x in par)
        gains.append(r["prep_s"] / pbest)
        e_seq = min(x["end_to_end_s"] for x in r["rows"])
        e_par = min(x["end_to_end_par_s"] for x in par)
        e2e.append(e_seq / e_par)
    m["MPrepParRange"] = rng(gains, 1)
    m["MPrepEtoEMax"] = fmt(max(e2e), 1)

    # ---- R27: large public temporal graphs (sampled verification) -----------
    rb = maybe("parallel_scaling_realbig.json")
    if rb:
        for r in rb["results"]:
            key = {"sx-askubuntu": "Ask", "sx-superuser": "Sup"}[r["dataset"]]
            if not r["verified_bit_identical"]:
                raise SystemExit(f"{r['dataset']}: sampled verification failed")
            one = min((x for x in r["rows"] if x["threads"] == 1), key=lambda x: x["kernel_s"])
            kb = min(r["rows"], key=lambda x: x["kernel_s"])
            eb = min(r["rows"], key=lambda x: x["end_to_end_s"])
            par = [x for x in r["rows"] if x.get("prep_par")]
            pbest = min(x["prep_par"]["total_s"] for x in par) if par else r["prep_s"]
            m[f"M{key}N"] = big(r["n"])
            m[f"M{key}M"] = big(r["m"])
            m[f"M{key}W"] = big(r["W"])
            m[f"M{key}Ksp"] = fmt(one["kernel_s"] / kb["kernel_s"], 1)
            m[f"M{key}Kp"] = str(kb["threads"])
            m[f"M{key}Plat"] = plateau(config_rows(r, kb))
            m[f"M{key}PrepShare"] = fmt(100 * r["prep_s"] / eb["end_to_end_s"], 0)
            m[f"M{key}SortShare"] = fmt(100 * r["prep_phases_s"]["sort_validate_s"] / r["prep_s"], 0)
            m[f"M{key}PrepGain"] = fmt(r["prep_s"] / pbest, 1)
        m["MSampledSources"] = str(SAMPLED_SOURCES)
        # R35: best end-to-end with the parallel preprocessing path over the best
        # with the sequential path (same runs, same thread grid)
        pe = []
        for r in rb["results"]:
            par = [x for x in r["rows"] if x.get("prep_par")]
            if par:
                pe.append(min(x["end_to_end_par_s"] for x in par) / min(x["end_to_end_s"] for x in r["rows"]))
        if pe:
            m["MBigParEtoE"] = rng(pe, 2)
            paths = {x["prep_par"].get("sort_path") for r in rb["results"] for x in r["rows"] if x.get("prep_par")}
            m["MBigSortPath"] = "/".join(sorted(p_ for p_ in paths if p_))
        # R33: sequential preprocessing before (v9 session, raw-timestamp key
        # overflowed to a three-key lexsort) and after the dense-rank key
        old = maybe(V9_REALBIG)
        if old:
            for r_old, r_new in zip(old["results"], rb["results"]):
                key = {"sx-askubuntu": "Ask", "sx-superuser": "Sup"}[r_new["dataset"]]
                assert r_old["dataset"] == r_new["dataset"]
                m[f"M{key}PrepOld"] = fmt(r_old["prep_s"] * 1e3, 0)
                m[f"M{key}PrepNew"] = fmt(r_new["prep_s"] * 1e3, 0)
        extra["realbig"] = rb

    # ---- R29: closure semantics on bucketed real timestamps -----------------
    cg = maybe("closure_grouped.json")
    if cg:
        infl, cost1, csp, gap, sw_mean, ratio_g, lem = [], [], [], [], [], [], 0
        for r in cg["results"]:
            st = {x["threads"]: x["kernel_s"] for x in r["rows"] if x["mode"] == "strict"}
            cl = {x["threads"]: x["kernel_s"] for x in r["rows"] if x["mode"] == "closure"}
            infl.append(100 * (r["pairs_closure"] / r["pairs_strict"] - 1))
            cost1.append(cl[1] / st[1])
            csp.append(cl[1] / min(cl.values()))
            gap.append(min(cl.values()) / min(st.values()))
            sw_mean.append(r["sweeps_mean"])
            ratio_g.append(r["group_size_mean"] / r["sweeps_mean"])
            for x in r["lemma4"]:
                if not (x["chunk_sweeps_le_sequential"] and x["bit_identical"]):
                    raise SystemExit(f"{r['dataset']}/{r['bucket']}: Lemma 4 check failed")
                lem += 1
        m["MClStreams"] = str(len(cg["results"]))
        m["MClInflRange"] = rng(infl, 1)
        m["MClCostRange"] = rng(cost1, 1)
        m["MClSpRange"] = rng(csp, 1)
        m["MClGapRange"] = rng(gap, 1)
        m["MClSweepRange"] = rng(sw_mean, 1)
        m["MClSweepMax"] = str(max(r["sweeps_max"] for r in cg["results"]))
        m["MClDepthRange"] = rng([s - 1 for s in sw_mean], 1)
        m["MClDepthMax"] = str(max(r["sweeps_max"] for r in cg["results"]) - 1)
        m["MClSaveRange"] = rng(ratio_g, 0)
        m["MClSaveMax"] = fmt(max(ratio_g), 0)
        m["MClLemma"] = str(lem)
        extra["closure"] = cg

    # ---- R39: Temporal Ligra (same relation verified before timing) -----------
    tc, tt = maybe("temporal_ligra_check.json"), maybe("temporal_ligra_time.json")
    if tc and tt:
        if not (tc["same_relation_strict"] and tc["same_relation_closure"]):
            raise SystemExit("temporal_ligra_check.json: not the same relation; do not time")
        m["MTlCommit"] = tt["tool"]["commit"][:8]
        m["MTlSources"] = str(tc["datasets"][0]["sources"])
        one, best, slow = [], [], []
        for d in tt["datasets"]:
            r = next(x for x in ps["results"] if x["dataset"] == d["dataset"])
            lig = {}
            for x in d["rows"]:
                if x["workers_reported"] != [x["threads"]] or not x["verified_all_sources"]:
                    raise SystemExit(f"temporal_ligra_time.json {d['dataset']}: workers / verification (R37)")
                lig[x["threads"]] = x["sum_compute_s"]
            e1 = min(x["end_to_end_s"] for x in r["rows"] if x["threads"] == 1)
            eb = min(x["end_to_end_s"] for x in r["rows"])
            one.append(lig[1] / e1)
            best.append(min(lig.values()) / eb)
            slow.append(lig[max(lig)] / lig[1])
        m["MTlOneRange"] = rng(one, 0)
        m["MTlBestRange"] = rng(best, 0)
        m["MTlSlowRange"] = rng(slow, 1)
        m["MTlPmax"] = str(max(x["threads"] for d in tt["datasets"] for x in d["rows"]))
        # R43: the same tool run over p source batches in p one-thread processes
        tb = maybe("temporal_ligra_batch.json")
        if tb:
            crit, wall, sp = [], [], []
            for d in tb["datasets"]:
                r = next(x for x in ps["results"] if x["dataset"] == d["dataset"])
                eb = min(x["end_to_end_s"] for x in r["rows"])
                for x in d["rows"]:
                    if x["workers_reported"] != [1] or not x["verified_all_sources"] \
                            or x["sources"] != d["n"]:
                        raise SystemExit(f"temporal_ligra_batch.json {d['dataset']}: workers / coverage (R37)")
                c = {x["processes"]: x["critical_compute_s"] for x in d["rows"]}
                crit.append(min(c.values()) / eb)
                wall.append(min(x["wall_s"] for x in d["rows"]) / eb)
                sp.append(c[1] / min(c.values()))
            m["MTlBatchRange"] = rng(crit, 0)
            m["MTlBatchWallRange"] = rng(wall, 0)
            m["MTlBatchSp"] = rng(sp, 1)
            m["MTlBatchPmax"] = str(max(x["processes"] for d in tb["datasets"] for x in d["rows"]))

    # ---- R31: GraphBLAS static closure, a different problem ------------------
    gb = maybe("graphblas_reference.json")
    if gb:
        best_g, best_s, sp_g = [], [], []
        for r in gb["results"]:
            if not r["verified_equal"]:
                raise SystemExit("GraphBLAS closure not verified")
            for x in r["rows"]:
                if x.get("graphblas_threads_actual") != x["threads"]:
                    raise SystemExit(f"graphblas_reference.json {r['dataset']}: GraphBLAS pool "
                                     f"{x.get('graphblas_threads_actual')} != {x['threads']} (R37)")
            g = {x["threads"]: x["graphblas_s"] for x in r["rows"]}
            s = {x["threads"]: x["strata_static_s"] for x in r["rows"]}
            best_g.append(min(g.values()) / min(s.values()))
            sp_g.append(g[1] / min(g.values()))
        m["MGbVersion"] = gb["graphblas_version"]
        m["MGbOverStrata"] = rng(best_g, 0)
        m["MGbSpRange"] = rng(sp_g, 1)
        extra["graphblas"] = gb

    # ---- R59 (v23): time-ordered MS-BFS baseline (v21 session) ----------------
    mb = maybe("temporal_msbfs.json")
    if mb:
        r1, rp = [], []
        pmax = max(x["threads"] for x in mb["rows"])
        for x in mb["rows"]:
            if not x["verified_bit_identical"]:
                raise SystemExit(f"temporal_msbfs.json {x['dataset']} p={x['threads']}: not verified")
            (r1 if x["threads"] == 1 else rp if x["threads"] == pmax else []).append(
                x["ratio_msbfs_over_strata"])
        if mb.get("threads", {}).get("numba") != pmax:
            raise SystemExit("temporal_msbfs.json: Numba pool differs from the largest p (R37)")
        m["MMsOneRange"] = rng(r1, 1)
        m["MMsMaxRange"] = rng(rp, 2)
        m["MMsPmax"] = str(pmax)
        m["MMsSets"] = str(len(r1))

    # ---- R59 (v23): sx-stackoverflow, every source block (v22 session) --------
    full = maybe("large_real_full.json")
    sw = {b: maybe(f"large_real_b{b}.json") for b in (4096, 65536)}
    if full and all(sw.values()):
        recs = [full] + list(sw.values())
        for x in recs:
            for r in x["runs"]:
                if r["verified_sources"] != len(x["verified_sources"]) or r["schedule"] != "static":
                    raise SystemExit(f"sx-stackoverflow {x['block_bits']}: verification / schedule")
            if x["threads_record"].get("numba") != x["runs"][-1]["threads"]:
                raise SystemExit("large_real_*.json: Numba pool differs from the last run's p (R37)")
        fr = full["runs"][0]
        if "all_blocks_s" not in fr or fr["blocks_run"] != full["blocks_total"]:
            raise SystemExit("large_real_full.json: not every block ran (extrapolation)")
        ks = [b["kernel_s"] for b in fr["blocks"]]
        pairs = sum(b["pairs"] for b in fr["blocks"])
        m["MSoN"] = big(full["n"])
        m["MSoM"] = big(full["m"])
        m["MSoMm"] = fmt(full["m"] / 1e6, 1)
        m["MSoBeta"] = big(full["block_bits"])
        m["MSoBlocks"] = str(full["blocks_total"])
        m["MSoFullP"] = str(fr["threads"])
        m["MSoFullS"] = fmt(fr["all_blocks_s"], 0)
        m["MSoBlockRange"] = rng(ks, 1)
        m["MSoPairs"] = fmt(pairs / 1e12, 2)
        m["MSoRelTbit"] = fmt(full["relation_bits"] / 1e12, 1)
        m["MSoPairsPct"] = fmt(100 * pairs / full["relation_bits"], 1)
        m["MSoVerified"] = str(len(full["verified_sources"]))
        m["MSoRssGB"] = fmt(full["peak_rss_bytes"] / 1e9, 1)
        m["MSoStateGB"] = fmt(full["resident_block_bytes"] / 1e9, 1)
        m["MSoPrepNew"] = fmt(full["prep"]["total_s"], 1)
        sp, slow, mean = [], [], {}
        for b, x in sw.items():
            t = {r["threads"]: r["kernel_s_mean_block"] for r in x["runs"]}
            best = min(t, key=t.get)
            if best != fr["threads"]:
                raise SystemExit(f"beta={b}: best p={best} differs from the full run's p")
            sp.append(t[1] / t[best])
            slow.append(t[max(t)] / t[best])
            mean[b] = t
        m["MSoSweepSp"] = rng(sp, 1)
        m["MSoPmaxSlow"] = rng(slow, 1)
        m["MSoPmax"] = str(max(mean[65536]))
        m["MSoWideOne"] = fmt(mean[65536][1], 1)
        m["MSoWidePmax"] = fmt(mean[65536][max(mean[65536])], 1)
        m["MSoNarrowRss"] = fmt(sw[4096]["peak_rss_bytes"] / 1e9, 1)
        m["MSoWideRss"] = fmt(sw[65536]["peak_rss_bytes"] / 1e9, 1)
        # per-source cost of the wide block over the narrow one at the full run's p
        p = fr["threads"]
        m["MSoWidePerSrc"] = fmt((mean[4096][p] / 4096) / (mean[65536][p] / 65536), 2)
        old = maybe("large_real_blocked.json")
        if old and old["prep_sort"] == "lexsort" and full["prep_sort"] == "time-then-group":
            m["MSoSortOld"] = fmt(old["prep"]["sort_validate_s"], 1)
            m["MSoSortNew"] = fmt(full["prep"]["sort_validate_s"], 1)
            m["MSoSortGain"] = fmt(old["prep"]["sort_validate_s"] / full["prep"]["sort_validate_s"], 1)
        extra["stackoverflow"] = (full, sw)

    # ---- R65 (v24): why the block kernel slows down beyond six threads -------
    td = maybe("thread_diagnosis.json")
    if td:
        rows = td["rows"]
        if td["threads_record"].get("numba") != rows[-1]["threads"]:
            raise SystemExit("thread_diagnosis.json: Numba pool differs from the last run's p (R37)")
        if len({(r["block_bits"], r["pairs"]) for r in rows}) != len({r["block_bits"] for r in rows}):
            raise SystemExit("thread_diagnosis.json: pair counts differ inside a block width")
        get = {(r["block_bits"], r["threads"], r["schedule"]): r for r in rows}
        dyn = [get[(b, p, "dynamic")]["kernel_s_median"] / get[(b, p, "static")]["kernel_s_median"]
               for (b, p, sch) in get if sch == "static" and (b, p, "dynamic") in get]
        if min(dyn) < 1:
            raise SystemExit("thread_diagnosis.json: dynamic wins somewhere; revise the R65 text")
        m["MTdDynRange"] = rng(dyn, 2)
        wide = max(r["block_bits"] for r in rows)
        pth = sorted({r["threads"] for r in rows})
        best = min((r for r in rows if r["block_bits"] == wide and r["schedule"] == "static"),
                   key=lambda r: r["kernel_s_median"])
        top = get[(wide, pth[-1], "static")]
        m["MTdBestP"] = str(best["threads"])
        m["MTdBwBest"] = fmt(best["achieved_GBps"], 0)
        m["MTdBwBestPct"] = fmt(100 * best["ratio_to_triad"], 0)
        m["MTdBwMax"] = fmt(top["achieved_GBps"], 0)
        m["MTdTriadMax"] = fmt(top["triad_GBps"], 0)
        m["MTdPmax"] = str(pth[-1])
        m["MTdVerified"] = str(td["verified_sources_per_width"])
        one = [get[(b, 1, "dynamic")]["kernel_s_median"] / get[(b, 1, "static")]["kernel_s_median"]
               for b in sorted({r["block_bits"] for r in rows})]
        m["MTdChunkOne"] = rng(one, 2)
        m["MTdChunkOneN"] = str(get[(wide, 1, "dynamic")]["chunks"])
        extra["thread_diag"] = td

    # ---- R70 (v25): chunks versus concurrent threads --------------------------
    tc = maybe("thread_chunks.json")
    if tc:
        if tc["threads_record"].get("numba") != tc["rows"][-1]["threads"]:
            raise SystemExit("thread_chunks.json: Numba pool differs from the last run's p (R37)")
        if len({r["pairs"] for r in tc["rows"]}) != 1:
            raise SystemExit("thread_chunks.json: pair counts differ between configurations")
        sm = next(iter(tc["summary"].values()))
        fx = sm["chunks_at_fixed_threads"]
        ft = sm["threads_at_fixed_chunks"]
        cv = list(fx["series"].values())
        spread = 100 * (max(cv) - min(cv)) / min(cv)
        if spread > 10:
            raise SystemExit("thread_chunks.json: chunk count matters (> 10%); revise the R70 text")
        tv = {int(k): v for k, v in ft["series"].items()}
        get = {(r["threads"], r["chunks"]): r for r in tc["rows"]}
        m["MTcP"] = str(fx["threads"])
        m["MTcChunks"] = str(ft["chunks"])
        m["MTcChunkMax"] = str(max(int(k) for k in fx["series"]))
        m["MTcChunkVar"] = fmt(spread, 0)
        m["MTcThreadSlow"] = fmt(tv[max(tv)] / tv[min(tv)], 1)
        m["MTcPmax"] = str(max(tv))
        m["MTcBwLo"] = fmt(get[(max(tv), ft["chunks"])]["achieved_GBps"], 0)
        m["MTcBwHi"] = fmt(get[(min(tv), ft["chunks"])]["achieved_GBps"], 0)

    # ---- v26 (R57 / R68): TGLib, built from its pinned source -----------------
    gb_, gc_, gt_ = maybe("tglib_build.json"), maybe("tglib_check.json"), maybe("tglib_time.json")
    if gb_ and gc_ and gt_:
        if not (gb_.get("ok") and gc_.get("strict_matches")):
            raise SystemExit("TGLib: build not ok or strict relation not matched; do not time")
        if (gc_.get("package") or {}).get("commit") != gb_.get("commit"):
            raise SystemExit("TGLib: check ran a different build than the build record")
        m["MTgCommit"] = gb_["commit"][:7]
        m["MTgVersion"] = gc_["library_version"]
        data = [r for r in gc_["rows"] if r["sources"] == max(x["sources"] for x in gc_["rows"])]
        m["MTgSources"] = str(data[0]["sources"])
        cl = [r for r in data if r["mode"] == "closure"]
        m["MTgClosureMiss"] = str(sum(r["n_mismatched"] for r in cl))
        m["MTgClosureTotal"] = str(sum(r["sources"] for r in cl))
        one, best = [], []
        for d in sorted({r["dataset"] for r in gt_["rows"]}):
            rows = [r for r in gt_["rows"] if r["dataset"] == d]
            p1 = next(r for r in rows if r["procs"] == 1)
            one.append(p1["ratio_critical_over_strata"])
            sr = next(x for x in ps["results"] if x["dataset"] == d)
            eb = min(x["end_to_end_s"] for x in sr["rows"])
            best.append(min(r["critical_path_s"] for r in rows) / eb)
        m["MTgOneRange"] = rng(one, 0)
        m["MTgBestRange"] = rng(best, 1)
        m["MTgPmax"] = str(max(r["procs"] for r in gt_["rows"]))

    # ---- v26 (R69): random row updates by thread count ------------------------
    rr = maybe("random_rows.json")
    if rr:
        for order in ("random", "stream"):
            if len({r["checksum"] for r in rr["rows"] if r["order"] == order}) != 1:
                raise SystemExit(f"random_rows.json: {order} checksum differs between thread counts")
        sm = rr["summary"]
        if sm["random"]["largest_over_best"] > 1.05:
            raise SystemExit("random_rows.json: random updates slow down with threads; revise the text")
        sec = {o: {int(k): v for k, v in sm[o]["seconds"].items()} for o in ("random", "stream")}
        pmax = max(sec["random"])
        m["MRrPmax"] = str(pmax)
        m["MRrRandSp"] = fmt(sec["random"][1] / sec["random"][pmax], 1)
        m["MRrStreamSp"] = fmt(sec["stream"][1] / sec["stream"][pmax], 1)
        m["MRrEdgesM"] = fmt(rr["edges"] / 1e6, 0)

    # ---- v28 (R77): which kernel phase slows down past six threads -------------
    kp = maybe("kernel_phases.json")
    if kp:
        if kp["threads_record"].get("numba") != kp["rows"][-1]["threads"]:
            raise SystemExit("kernel_phases.json: Numba pool differs from the last run's p (R37)")
        sec = {}
        for r in kp["rows"]:
            sec.setdefault(r["variant"], {})[r["threads"]] = r["seconds_median"]
        st, cp = sec["strict"], sec["copy"]
        p1, pm = min(st), max(st)
        grow_copy, grow_strict = cp[pm] / cp[p1], st[pm] / st[min(st, key=st.get)]
        edge = {p: st[p] - cp[p] for p in st}
        if grow_copy <= grow_strict or edge[pm] >= edge[p1]:
            raise SystemExit("kernel_phases.json: the copy phase no longer explains the slowdown; revise R77 text")
        m["MKpPmax"] = str(pm)
        m["MKpCopyOne"] = fmt(cp[p1], 1)
        m["MKpCopyMax"] = fmt(cp[pm], 1)
        m["MKpCopyGrow"] = fmt(grow_copy, 0)
        m["MKpEdgeSp"] = fmt(edge[p1] / edge[pm], 0)
        m["MKpStrictMax"] = fmt(st[pm], 1)
        extra["kernel_phases"] = kp

    # ---- v29 (R75 / R76): copy-phase cause and the tile state ----------------
    kc = maybe("kernel_copy.json")
    if kc:
        if kc["threads_record"].get("numba") != kc["rows"][-1]["threads"]:
            raise SystemExit("kernel_copy.json: Numba pool differs from the last run's p (R37)")
        ck = kc["checksums"]
        if ck.get("tiles_pairs") != ck.get("strict_pairs"):
            raise SystemExit("kernel_copy.json: tile state and strict kernel disagree on pairs")
        sec = {}
        for r in kc["rows"]:
            sec.setdefault(r["variant"], {})[r["threads"]] = r["seconds_median"]
        cp, ru, ti, st = sec["copy"], sec["copy_reuse"], sec["tiles"], sec["strict"]
        ths = sorted(ti)
        p1, pm = ths[0], ths[-1]
        if not (cp[pm] > cp[p1] and ru[pm] < ru[p1]):
            raise SystemExit("kernel_copy.json: copies into a written state no longer scale "
                             "while fresh-state copies slow down; revise R75 text")
        if any(ti[a] <= ti[b] for a, b in zip(ths, ths[1:])):
            raise SystemExit("kernel_copy.json: tile-state kernel not faster at every added thread")
        m["MKcReuseOne"] = fmt(ru[p1], 1)
        m["MKcReuseMax"] = fmt(ru[pm], 1)
        m["MKcTilesOne"] = fmt(ti[p1], 1)
        m["MKcTilesMax"] = fmt(ti[pm], 1)
        m["MKcTilesOverStrict"] = fmt(min(st.values()) / ti[pm], 1)
        extra["kernel_copy"] = kc
    tb = maybe("large_real_tiles_b65536.json")
    tf = maybe("large_real_tiles_full.json")
    sofull = maybe("large_real_full.json")
    if tb and tf and sofull:
        for x in (tb, tf):
            for r in x["runs"]:
                if (r.get("state") != "tiles" or r["schedule"] != "static"
                        or r["verified_sources"] != len(x["verified_sources"])):
                    raise SystemExit("large_real_tiles_*.json: state / schedule / verification")
            if x["threads_record"].get("numba") != x["runs"][-1]["threads"]:
                raise SystemExit("large_real_tiles_*.json: Numba pool differs from the last run's p (R37)")
        t = {r["threads"]: r["kernel_s_mean_block"] for r in tb["runs"]}
        best = min(t, key=t.get)
        tr = tf["runs"][0]
        if "all_blocks_s" not in tr or tr["blocks_run"] != tf["blocks_total"]:
            raise SystemExit("large_real_tiles_full.json: not every block ran (extrapolation)")
        if best != tr["threads"]:
            raise SystemExit("large_real_tiles: full run's p is not the sweep's best p")
        fr = sofull["runs"][0]
        if sum(b["pairs"] for b in tr["blocks"]) != sum(b["pairs"] for b in fr["blocks"]):
            raise SystemExit("large_real_tiles_full.json: pair count differs from the row-state run")
        m["MStSweepSp"] = fmt(t[1] / t[best], 1)
        m["MStFullP"] = str(tr["threads"])
        m["MStFullS"] = fmt(tr["all_blocks_s"], 0)
        m["MStFullSp"] = fmt(fr["all_blocks_s"] / tr["all_blocks_s"], 1)
        m["MStRssGB"] = fmt(tf["peak_rss_bytes"] / 1e9, 1)
        m["MStBlockRange"] = rng([b["kernel_s"] for b in tr["blocks"]], 1)   # v34 (R106)
        extra["stackoverflow_tiles"] = (tb, tf)

    # ---- v31 (R90): copies after every page of the fresh state was written ----
    kf = maybe("kernel_prefault.json")
    if kf:
        if kf["threads_record"].get("numba") != kf["rows"][-1]["threads"]:
            raise SystemExit("kernel_prefault.json: Numba pool differs from the last run's p (R37)")
        sec = {}
        for r in kf["rows"]:
            sec.setdefault(r["variant"], {})[r["threads"]] = r["seconds_median"]
        cp, pf = sec["copy"], sec["copy_prefault"]
        p1, pm = min(pf), max(pf)
        if not (cp[pm] > cp[p1] and pf[pm] < pf[p1]):
            raise SystemExit("kernel_prefault.json: pre-written pages no longer make the copies "
                             "scale; revise the R90 text")
        m["MKfPreOne"] = fmt(pf[p1], 1)
        m["MKfPreMax"] = fmt(pf[pm], 1)
        m["MKfPreGain"] = fmt(cp[pm] / pf[pm], 0)
        extra["kernel_prefault"] = kf

    # ---- v31 (R91): tile state on narrow blocks and with dynamic chunks -------
    tn, tnd, twd = (maybe(f) for f in ("large_real_tiles_b4096.json",
                                       "large_real_tiles_dyn_b4096.json",
                                       "large_real_tiles_dyn_b65536.json"))
    rows_n = maybe("large_real_b4096.json")
    if tn and tnd and twd and rows_n and tb:
        def mean(x):
            for r in x["runs"]:
                if r["verified_sources"] != len(x["verified_sources"]):
                    raise SystemExit("tile-state sweeps: verification")
            return {r["threads"]: r["kernel_s_mean_block"] for r in x["runs"]}
        def pairs(x):
            return [b["pairs"] for b in x["runs"][0]["blocks"]]
        if not (pairs(tn) == pairs(tnd) == pairs(rows_n)[:len(pairs(tn))] and pairs(twd) == pairs(tb)):
            raise SystemExit("tile-state sweeps: pair counts differ between states / schedules")
        mn, mnd, mwd, mr, mw = mean(tn), mean(tnd), mean(twd), mean(rows_n), mean(tb)
        if any(r["schedule"] != "dynamic" for x in (tnd, twd) for r in x["runs"]):
            raise SystemExit("tile-state sweeps: dynamic files must hold dynamic runs")
        ratio = [mn[p] / mr[p] for p in mn]
        dyn = [min(mnd.values()) / min(mn.values()), min(mwd.values()) / min(mw.values())]
        if min(dyn) <= 1:
            raise SystemExit("tile-state sweeps: dynamic no longer slower; revise the R91 text")
        m["MNtRatioRange"] = rng(ratio, 2)
        m["MNtBest"] = fmt(min(mn.values()), 2)
        m["MNtRowsBest"] = fmt(min(mr.values()), 2)
        m["MDynTileRange"] = rng(dyn, 1)

    # ---- v32 (R94 / R98): block width versus threads and memory --------------
    w16, w32, f32 = (maybe(f) for f in ("large_real_tiles_b16384.json",
                                        "large_real_tiles_b32768.json",
                                        "large_real_tiles_full_b32768.json"))
    if w16 and w32 and f32 and tn and tb and tf and sofull:
        widths = {}
        for x in (tn, w16, w32, tb):
            for r in x["runs"]:
                if r.get("state") != "tiles" or r["schedule"] != "static" or \
                        r["verified_sources"] != len(x["verified_sources"]):
                    raise SystemExit("width sweep: state / schedule / verification")
            widths[x["block_bits"]] = {r["threads"]: r["kernel_s_mean_block"] for r in x["runs"]}
        r32 = f32["runs"][0]
        if r32["blocks_run"] != f32["blocks_total"] or "all_blocks_s" not in r32:
            raise SystemExit("large_real_tiles_full_b32768.json: not every block ran")
        if r32.get("state") != "tiles" or r32["verified_sources"] != len(f32["verified_sources"]):
            raise SystemExit("large_real_tiles_full_b32768.json: state / verification")
        if sum(b["pairs"] for b in r32["blocks"]) != sum(b["pairs"] for b in sofull["runs"][0]["blocks"]):
            raise SystemExit("large_real_tiles_full_b32768.json: pair count differs")
        per_src = {b: min(t.values()) / b for b, t in widths.items()}
        if per_src[max(per_src)] >= per_src[min(per_src)]:
            raise SystemExit("width sweep: wider blocks no longer cheaper per source")
        best = {b: min(t, key=t.get) for b, t in widths.items()}
        m["MSoHalfBeta"] = big(f32["block_bits"])
        m["MSoHalfS"] = fmt(r32["all_blocks_s"], 0)
        m["MSoHalfP"] = str(r32["threads"])
        m["MSoHalfGB"] = fmt(f32["resident_block_bytes"] / 1e9, 1)
        m["MSoHalfRss"] = fmt(f32["peak_rss_bytes"] / 1e9, 1)
        m["MSoWideGB"] = fmt(tf["resident_block_bytes"] / 1e9, 1)
        m["MSoPerSrcGain"] = fmt(per_src[min(per_src)] / per_src[max(per_src)], 1)
        # words per thread at the best p: narrow blocks peak below the wide ones'
        m["MSoNarrowBestP"] = str(best[min(best)])
        m["MSoNarrowWpt"] = fmt(((min(best) + 63) // 64) / best[min(best)], 1)
        mid = sorted(best)[1]
        m["MSoMidBeta"] = big(mid)
        m["MSoMidBestP"] = str(best[mid])
        m["MSoMidSlowWpt"] = fmt(((mid + 63) // 64) / max(widths[mid]), 1)
        extra["width_sweep"] = (widths, f32)

    # ---- v34 (R105): why compiled ratios exceed the 64-source packing --------
    oc = maybe("op_counts.json")
    if oc:
        cr_rows = {r["dataset"]: r for r in load("compiled_real.json")["rows"]}
        steps, cost = [], []
        for r in oc["rows"]:
            mr = cr_rows.get(r["dataset"])
            if mr is None or abs(mr["kernel_ratio_bfs_over_strata"] - r["measured_kernel_ratio"]) > 1e-9:
                raise SystemExit("op_counts.json: measured ratio differs from compiled_real.json; rerun op_counts")
            if not (1 < r["step_ratio"] <= 64 * 1.1):
                raise SystemExit("op_counts.json: step ratio outside (1, ~64]; revise the R105 text")
            steps.append(r["step_ratio"])
            cost.append(r["measured_kernel_ratio"] / r["step_ratio"])
        m["MOpStepRange"] = rng(steps, 0)
        m["MOpCostRange"] = rng(cost, 1)

    # ---- v35 (R102 / R112): unblocked state variants, rule fixed beforehand ----
    # R112 (fixed 2026-11-19 before the results were opened): if, on both large
    # synthetic streams and at every p, the tile state's median is at least 5%
    # below the shared state's (tiles <= 0.95 x shared), the production path
    # changes to the tile state and steps 2 / 3 are re-measured; otherwise the
    # abstract keeps the shared-state result and the paper states why.
    tu = maybe("tile_state_unblocked.json")
    if tu:
        if tu["threads_record"].get("numba") != max(r["threads"] for x in tu["results"] for r in x["rows"]):
            raise SystemExit("tile_state_unblocked.json: Numba pool differs from the largest p (R37)")
        win, ratios, pre = True, [], []
        for x in tu["results"]:
            sec = {}
            for r in x["rows"]:
                sec.setdefault(r["variant"], {})[r["threads"]] = r["seconds_median"]
            for p_ in sec["shared"]:
                q = sec["tiles"][p_] / sec["shared"][p_]
                ratios.append(q)
                pre.append(sec["prewritten"][p_] / sec["shared"][p_])
                if q > 0.95:
                    win = False
        m["MTuRule"] = "met" if win else "not met"
        m["MTuRatioRange"] = rng(ratios, 2)
        m["MTuPreRange"] = rng(pre, 2)
        extra["tile_unblocked"] = (tu, win)

    # ---- v35 (R109): rolling windows and insertion on sx-askubuntu -----------
    wl = maybe("window_large.json")
    if wl:
        if wl["threads_record"].get("numba") != wl["threads"]:
            raise SystemExit("window_large.json: Numba pool differs from the run's p (R37)")
        if not wl["insertion"]["bit_identical"] or any(w["verified_sources"] < 1 for w in wl["windows"]):
            raise SystemExit("window_large.json: verification missing")
        ins = wl["insertion"]
        m["MWlSlots"] = str(len(wl["windows"]))
        m["MWlEdgesK"] = fmt(sum(w["edges"] for w in wl["windows"]) / len(wl["windows"]) / 1e3, 0)
        m["MWlKernRange"] = rng([w["kernel_s"] * 1e3 for w in wl["windows"]], 0)
        m["MWlVerified"] = str(wl["windows"][0]["verified_sources"])
        m["MWlBatches"] = str(len(ins["batches"]))
        m["MWlBatchK"] = fmt(sum(b["edges"] for b in ins["batches"]) / len(ins["batches"]) / 1e3, 0)
        m["MWlBatchRange"] = rng([b["kernel_s"] * 1e3 for b in ins["batches"]], 0)
        m["MWlFull"] = fmt(ins["full_kernel_s"] * 1e3, 0)
        m["MWlP"] = str(wl["threads"])

    # ---- v36 (R113): insertion batches, tiled versus row-major in place -------
    wi = maybe("window_insert.json")
    if wi:
        ins = wi["insertion"]
        if not ins["bit_identical"] or set(ins.get("layouts", {})) != {"tiled", "rowmajor"}:
            raise SystemExit("window_insert.json: needs bit-identical tiled and rowmajor runs")
        if wi["threads_record"].get("numba") != wi["threads"]:
            raise SystemExit("window_insert.json: Numba pool differs from the run's p (R37)")
        t = [b["kernel_s"] for b in ins["layouts"]["tiled"]]
        r = [b["kernel_s"] for b in ins["layouts"]["rowmajor"]]
        m["MWiTiledRange"] = rng([x * 1e3 for x in t], 0)
        m["MWiRowRange"] = rng([x * 1e3 for x in r], 0)
        m["MWiFull"] = fmt(ins["full_kernel_s"] * 1e3, 0)
        m["MWiGain"] = fmt(sum(t) / sum(r), 1)
        m["MWiBatchK"] = fmt(sum(b["edges"] for b in ins["layouts"]["rowmajor"]) / len(r) / 1e3, 0)
        print(f"[R112] tile state on the unblocked streams: rule {'MET' if win else 'NOT met'} "
              f"(tiles/shared {min(ratios):.2f}-{max(ratios):.2f})")
    return extra


def table_maincompare():
    """Table 5 on the M5 (single thread), replacing the workstation values (R25)."""
    cb = load_perf("compiled_baselines.json")

    def ms(x, d):
        return "--" if x is None else f"{x:.{d}f}"
    lines = [
        "% Auto-generated by experiments/m5_numbers.py from results/perf/compiled_baselines.json -- do not edit.",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lrrrrrrr@{}}",
        "\\toprule",
        "Dataset & $n$ & $m$ & \\shortstack[r]{Exact\\\\BFS} & \\shortstack[r]{dense\\\\NumPy} & "
        "\\shortstack[r]{STRATA\\\\py} & \\shortstack[r]{STRATA\\\\numba} & \\shortstack[r]{Graph\\\\BLAS$^\\dagger$} \\\\",
        " & & & (ms) & (ms) & (ms) & (ms) & (ms) \\\\",
        "\\midrule",
    ]
    for r in cb["rows"]:
        lines.append(f"{r['label']} & {r['n']} & {r['m'] / 1e3:.1f}k & {r['exact_bfs_ms']:.0f} & "
                     f"{ms(r['dense_numpy_ms'], 1)} & {ms(r['strata_py_ms'], 1)} & "
                     f"{ms(r['strata_numba_ms'], 2)} & {ms(r.get('graphblas_static_ms'), 1)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}"]
    return "\n".join(lines) + "\n"


COMPILED_NAME = {"CollegeMsg": "CollegeMsg", "Bitcoin-OTC": "BitcoinOTC", "Bitcoin-OTC(+)": "BitcoinOTCpos"}


def compiled_speedup_cell(cr, dataset):
    """v21 (R53): compiled STRATA over compiled per-source BFS, kernel, one thread:
    best-of-R ratio with the ratio of medians in parentheses."""
    from statistics import median as _med
    (row,) = [r for r in cr["rows"] if r["dataset"] == COMPILED_NAME.get(dataset, dataset)]
    if not row.get("verified_bit_identical"):
        raise SystemExit(f"{dataset}: compiled run not verified")
    best = row["kernel_ratio_bfs_over_strata"]
    med = _med(row["bfs_kernel_samples_s"]) / _med(row["strata_kernel_samples_s"])
    return f"{best:.0f} ({med:.0f})"


def table_real_temporal():
    rt = load("real_temporal_results.json")
    cr = load("compiled_real.json")
    lines = [
        "% Auto-generated by experiments/m5_numbers.py from results/real/real_temporal_results.json"
        " and results/real/compiled_real.json -- do not edit.",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lrrrrrr@{}}",
        "\\toprule",
        "Dataset & $n$ & $m$ & \\shortstack[r]{Temporal\\\\pairs (strict)} & \\shortstack[r]{Static\\\\pairs} & "
        "\\shortstack[r]{Static\\\\over-count} & \\shortstack[r]{Comp.\\\\sp.\\ ($\\times$)} \\\\",
        "\\midrule",
    ]
    for r in rt["datasets"]:
        if not r["bit_identical_strict"]:
            raise SystemExit(f"{r['dataset']}: strict run not bit-identical")
        lines.append(f"{r['dataset']} & {big(r['n'])} & {big(r['m'])} & {big(r['temporal_pairs_strict'])} & "
                     f"{big(r['static_pairs'])} & $+{r['static_overcount_vs_strict_pct']:.1f}\\%$ & "
                     f"{compiled_speedup_cell(cr, r['dataset'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}"]
    return "\n".join(lines) + "\n"


def table_closure(cg):
    lines = [
        "% Auto-generated by experiments/m5_numbers.py from results/real/closure_grouped.json -- do not edit.",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}llrrrrrr@{}}",
        "\\toprule",
        "Dataset & Bucket & Groups & $\\overline{|\\grp_t|}$ & \\shortstack[r]{Sweeps\\\\mean/max} & "
        "\\shortstack[r]{Pairs\\\\cl./str.} & \\shortstack[r]{Cl./str.\\\\$p{=}1$} & \\shortstack[r]{Cl.\\\\sp.\\ ($p$)} \\\\",
        "\\midrule",
    ]
    for r in cg["results"]:
        st = {x["threads"]: x["kernel_s"] for x in r["rows"] if x["mode"] == "strict"}
        cl = {x["threads"]: x["kernel_s"] for x in r["rows"] if x["mode"] == "closure"}
        pb = min(cl, key=cl.get)
        lines.append(f"{LABEL.get(r['dataset'], r['dataset'])} & {r['bucket']} & {big(r['groups'])} & "
                     f"{r['group_size_mean']:.1f} & {r['sweeps_mean']:.2f}/{r['sweeps_max']} & "
                     f"{r['pairs_closure'] / r['pairs_strict']:.3f} & {cl[1] / st[1]:.1f} & "
                     f"{cl[1] / cl[pb]:.1f} ({pb}) \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}"]
    return "\n".join(lines) + "\n"


def table_large_real(full, sw, tiles=None, extra=None):
    """Supplementary Table S4: sx-stackoverflow, mean kernel seconds per source block by
    state / schedule / block width and threads, the time of every block where it was
    run, and the resident block state (v32, R97: one table for all block runs)."""
    extra = extra or {}
    ths = sorted({r["threads"] for x in sw.values() for r in x["runs"]})
    lines = [
        "% Auto-generated by experiments/m5_numbers.py from results/real/large_real_*.json -- do not edit.",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lr" + "r" * len(ths) + "rr@{}}",
        "\\toprule",
        "State & $\\beta$ & " + " & ".join(f"$p{{=}}{p}$" for p in ths)
        + " & All (s) & GB \\\\",
        "\\midrule",
    ]
    def row(label, x, all_run=None):
        if x is None or sorted(r["threads"] for r in x["runs"]) != ths:
            return
        t = {r["threads"]: r["kernel_s_mean_block"] for r in x["runs"]}
        best = min(t, key=t.get)
        cells = [(f"\\textbf{{{t[p]:.2f}}}" if p == best else f"{t[p]:.2f}") for p in ths]
        allc = "--"
        if all_run is not None:
            r0 = all_run["runs"][0]
            allc = f"{r0['all_blocks_s']:.0f} ({r0['threads']})"
        lines.append(f"{label} & {big(x['block_bits'])} & " + " & ".join(cells)
                     + f" & {allc} & {x['resident_block_bytes'] / 1e9:.1f} \\\\")
    row("shared", sw.get(4096))
    row("shared", sw.get(65536), full)
    lines.append("\\addlinespace")
    for f, full_f in (("large_real_tiles_b4096.json", None), ("large_real_tiles_b16384.json", None),
                      ("large_real_tiles_b32768.json", "large_real_tiles_full_b32768.json"),
                      ("large_real_tiles_b65536.json", "large_real_tiles_full.json")):
        row("tiles", maybe(f), maybe(full_f) if full_f else None)
    lines.append("\\addlinespace")
    for f in ("large_real_tiles_dyn_b4096.json", "large_real_tiles_dyn_b65536.json"):
        row("tiles$^d$", maybe(f))
    lines += ["\\bottomrule", "\\end{tabular*}"]
    return "\n".join(lines) + "\n"


def table_thread_diag(td):
    """Supplementary: static versus dynamic by threads, both block widths (R65)."""
    ths = sorted({r["threads"] for r in td["rows"]})
    bs = sorted({r["block_bits"] for r in td["rows"]})
    get = {(r["block_bits"], r["threads"], r["schedule"]): r for r in td["rows"]}
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{$\\beta{{=}}{big(b)}$}}" for b in bs)
    lines = [
        "% Auto-generated by experiments/m5_numbers.py from results/real/thread_diagnosis.json -- do not edit.",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}r" + "rr" * len(bs) + "rr@{}}",
        "\\toprule",
        f" & {head} & \\multicolumn{{2}}{{c}}{{GB/s}} \\\\",
        "$p$ & " + " & ".join(["Stat.", "Dyn."] * len(bs)) + f" & $\\beta{{=}}{big(bs[-1])}$ & Triad \\\\",
        "\\midrule",
    ]
    for p in ths:
        cells = []
        for b in bs:
            cells += [f"{get[(b, p, 'static')]['kernel_s_median']:.2f}",
                      f"{get[(b, p, 'dynamic')]['kernel_s_median']:.2f}"]
        st = get[(bs[-1], p, "static")]
        lines.append(f"{p} & " + " & ".join(cells) + f" & {st['achieved_GBps']:.0f} & {st['triad_GBps']:.0f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}"]
    return "\n".join(lines) + "\n"


KP_LABEL = {"strict": "strict kernel", "nosnap": "no snapshot", "copy": "copies only",
            "prealloc": "tiles pre-allocated", "copy_prefault": "copies, pages pre-written",
            "copy_reuse": "copies, written state",
            "tiles": "tiles as state"}


def table_kernel_phases(kp, kc=None, kf=None):
    """Supplementary: median seconds of each kernel variant by threads (R77);
    v29 (R75 / R76) appends two rows from kernel_copy.json when it covers the
    same thread counts."""
    ths = sorted({r["threads"] for r in kp["rows"]})
    sec = {}
    for r in kp["rows"]:
        sec.setdefault(r["variant"], {})[r["threads"]] = r["seconds_median"]
    if kc is not None and sorted({r["threads"] for r in kc["rows"]}) == ths:
        for r in kc["rows"]:
            if r["variant"] in ("copy_reuse", "tiles"):
                sec.setdefault(r["variant"], {})[r["threads"]] = r["seconds_median"]
    if kf is not None and sorted({r["threads"] for r in kf["rows"]}) == ths:
        for r in kf["rows"]:                       # v31 (R90): corrected prefault write
            if r["variant"] == "copy_prefault":
                sec.setdefault(r["variant"], {})[r["threads"]] = r["seconds_median"]
    lines = [
        "% Auto-generated by experiments/m5_numbers.py from results/real/kernel_phases.json"
        " and results/real/kernel_copy.json, kernel_prefault.json -- do not edit.",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}l" + "r" * len(ths) + "@{}}",
        "\\toprule",
        "Variant & " + " & ".join(f"$p{{=}}{p}$" for p in ths) + " \\\\",
        "\\midrule",
    ]
    for v in ("strict", "nosnap", "copy", "prealloc", "copy_prefault", "copy_reuse", "tiles"):
        if v in sec:
            lines.append(KP_LABEL[v] + " & " + " & ".join(f"{sec[v][p]:.1f}" for p in ths) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}"]
    return "\n".join(lines) + "\n"


def outputs():
    m, ps, pl = derive()
    extra = derive_v10(m, ps, pl)
    files = {
        PAPER / "m5_numbers.tex": macros_tex(m),
        PAPER / "table_parallel_scaling.tex": table_small(ps),
        PAPER / "table_parallel_scaling_large.tex": table_large(pl),
        PAPER / "table_m5_summary.tex": table_summary(ps, pl, extra.get("realbig"), extra.get("stackoverflow_tiles")),
        PAPER / "table_maincompare.tex": table_maincompare(),
        PAPER / "table_real_temporal.tex": table_real_temporal(),
    }
    if "closure" in extra:
        files[PAPER / "table_closure_grouped.tex"] = table_closure(extra["closure"])
    if "kernel_phases" in extra:
        files[PAPER / "table_kernel_phases.tex"] = table_kernel_phases(extra["kernel_phases"], extra.get("kernel_copy"),
                                                                 extra.get("kernel_prefault"))
    if "thread_diag" in extra:
        files[PAPER / "table_thread_diag.tex"] = table_thread_diag(extra["thread_diag"])
    if "stackoverflow" in extra:
        files[PAPER / "table_large_real.tex"] = table_large_real(*extra["stackoverflow"],
            tiles=extra["stackoverflow_tiles"][0] if "stackoverflow_tiles" in extra else None)
    return files, m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="do not write; fail if an emitted file differs from a fresh derivation")
    ap.add_argument("--print", action="store_true", help="print the macro values")
    args = ap.parse_args(argv)
    files, m = outputs()
    if args.print:
        for k in sorted(m):
            print(f"{k:24s} {m[k]}")
    if args.check:
        bad = [p.name for p, txt in files.items() if not p.exists() or p.read_text() != txt]
        for b in bad:
            print(f"[STALE] {b} differs from experiments/m5_numbers.py output")
        return 1 if bad else 0
    for p, txt in files.items():
        p.write_text(txt)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
