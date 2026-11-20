#!/usr/bin/env python3
"""Within-machine numbers for a second machine (v35).

Why
---
Project rule (2026-11-19): results from a second machine that the user measures
are reported only as comparisons inside that machine, never as run times against
the M5 Max. The paper generator (experiments/m5_numbers.py) reads results/real/
only; this generator reads results/server/<tag>/ only and can emit nothing but
dimensionless ratios, thread counts, and the machine description.

How
---
Inputs (each optional; a macro is written only when its file exists):
  parallel_scaling_large.json   tiled kernel speedup over the fastest one-thread
                                configuration, static and dynamic, per stream
  tile_state_unblocked.json     tile / shared and pre-written / shared medians
  kernel_prefault.json          fresh-state copies at p_max over p=1, and the same
                                for copies after every page was written
  large_real_tiles_b*.json      tile-state block speedup over one thread, best p
Every value passes ``_ratio`` or ``_count``; a key that does not end in one of the
allowed suffixes stops the run, so a time can never be written by mistake.

What
---
    python3 experiments/server_numbers.py --tag <name>
        -> STRATA-latex/STRATA-paper-EN/server_numbers_<name>.tex
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAPER = ROOT.parent.parent / "STRATA-latex" / "STRATA-paper-EN"
ALLOWED = ("Sp", "Ratio", "Range", "P", "Cores", "Machine", "Rule")


def _ratio(x):
    return f"{x:.2f}" if x < 10 else f"{x:.1f}"


def _rng(xs):
    lo, hi = min(xs), max(xs)
    return _ratio(lo) if f"{lo:.2f}" == f"{hi:.2f}" else f"{_ratio(lo)}--{_ratio(hi)}"


def _check(macros):
    for k in macros:
        if not re.fullmatch(r"MSv[A-Za-z]+", k) or not k.endswith(ALLOWED):
            raise SystemExit(f"server_numbers: macro {k} is not a within-machine quantity")


def derive(d: Path):
    m = {}

    def load(name):
        p = d / name
        return json.loads(p.read_text()) if p.exists() else None

    machine = None
    pl = load("parallel_scaling_large.json")
    if pl:
        machine = pl["machine"]
        for r, tag in zip(pl["results"], ("B", "C", "D", "E")):
            tiled = [x for x in r["rows"] if x["layout"] == "tiled"]
            one = min(x["kernel_s"] for x in tiled if x["threads"] == 1 and x["schedule"] == "static")
            for sched, s in (("static", "Static"), ("dynamic", "Dyn")):
                rows = [x for x in tiled if x["schedule"] == sched]
                if rows:
                    best = min(rows, key=lambda x: x["kernel_s"])
                    m[f"MSv{tag}{s}Sp"] = _ratio(one / best["kernel_s"])
                    m[f"MSv{tag}{s}P"] = str(best["threads"])
    tu = load("tile_state_unblocked.json")
    if tu:
        machine = machine or tu["machine"]
        q, pre, win = [], [], True
        for x in tu["results"]:
            sec = {}
            for r in x["rows"]:
                sec.setdefault(r["variant"], {})[r["threads"]] = r["seconds_median"]
            for p in sec["shared"]:
                q.append(sec["tiles"][p] / sec["shared"][p])
                pre.append(sec["prewritten"][p] / sec["shared"][p])
                win = win and q[-1] <= 0.95
        m["MSvTileRatioRange"] = _rng(q)
        m["MSvPreRatioRange"] = _rng(pre)
        m["MSvTileRule"] = "met" if win else "not met"
    kf = load("kernel_prefault.json")
    if kf:
        machine = machine or kf["machine"]
        sec = {}
        for r in kf["rows"]:
            sec.setdefault(r["variant"], {})[r["threads"]] = r["seconds_median"]
        p1, pm = min(sec["copy"]), max(sec["copy"])
        m["MSvCopyGrowRatio"] = _ratio(sec["copy"][pm] / sec["copy"][p1])
        m["MSvPrefaultGrowRatio"] = _ratio(sec["copy_prefault"][pm] / sec["copy_prefault"][p1])
        m["MSvCopyMaxP"] = str(pm)
    for b, word in ((16384, "Quarter"), (32768, "Half"), (65536, "Wide")):
        x = load(f"large_real_tiles_b{b}.json")
        if x:
            machine = machine or x["machine"]
            t = {r["threads"]: r["kernel_s_mean_block"] for r in x["runs"]}
            best = min(t, key=t.get)
            m[f"MSvTiles{word}Sp"] = _ratio(t[1] / t[best])
            m[f"MSvTiles{word}P"] = str(best)
    if machine:
        m["MSvMachine"] = machine.get("summary", "").split(", Python")[0].replace("_", r"\_")
        m["MSvCores"] = str(machine.get("cpu_count", ""))
    _check(m)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default=str(ROOT / "results" / "server"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    d = Path(args.root) / args.tag
    if not d.is_dir():
        raise SystemExit(f"no results in {d}; run tools/run_server_session.py --tag {args.tag}")
    m = derive(d)
    out = Path(args.out) if args.out else PAPER / f"server_numbers_{args.tag}.tex"
    lines = [f"% Auto-generated by experiments/server_numbers.py from results/server/{args.tag}/ "
             "-- within-machine ratios only; do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(m)} macros)")
    return m


if __name__ == "__main__":
    main()
