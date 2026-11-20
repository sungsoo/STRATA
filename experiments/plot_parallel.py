#!/usr/bin/env python3
"""Figures for the multicore measurements.

Subcommands
-----------
scaling   speedup vs threads from results/real/parallel_scaling.json, one line
          per (layout, schedule), one panel per dataset; vertical reference
          lines at the performance-level core boundaries recorded by the harness
          (Apple M5 Max: 6 super cores, 6 + 12 = 18 total). macOS does not
          guarantee which cores a thread occupies, so the lines are reference
          marks, not placement claims.
paper     main-paper figure: (a) real data + n=8000, (b) large
          streams with static vs dynamic schedules.
roofline  achieved kernel bandwidth vs threads against the measured STREAM-triad
          curve from results/real/bandwidth_roofline.json (supplementary).

Both refuse to plot a PENDING (single-core) JSON. Figures are generated at the
final printed width (column width 3.50 in) with 8 pt Palatino-family text
(TeX Gyre Pagella) and TrueType (Type 42) font embedding, so LaTeX includes
them at scale 1.

Usage
-----
  python3 experiments/plot_parallel.py scaling \
      --json results/real/parallel_scaling.json \
      --out results/real/fig_parallel_scaling.pdf
  python3 experiments/plot_parallel.py roofline \
      --json results/real/bandwidth_roofline.json \
      --out results/real/fig_roofline.pdf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from experiments.figstyle import COLUMN_WIDTH_IN, apply_style, save_with_layout  # noqa: E402


def core_boundaries(machine, override=None):
    if override:
        return [int(x) for x in override.split(",") if x]
    levels = machine.get("perf_levels") or []
    out, acc = [], 0
    for lvl in levels:
        try:
            acc += int(lvl["physicalcpu"])
        except (TypeError, ValueError):
            return []
        out.append(acc)
    return out


def plot_scaling(data, out, boundaries):
    plt = apply_style()
    results = data["results"]
    ncols = 2
    nrows = (len(results) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(COLUMN_WIDTH_IN, 1.35 * nrows),
                             sharex=True, squeeze=False)
    for ax, r in zip(axes.flat, results):
        configs = sorted({(x["layout"], x["schedule"]) for x in r["rows"]})
        tmax = max(x["threads"] for x in r["rows"])
        for layout, schedule in configs:
            rows = sorted((x for x in r["rows"]
                           if (x["layout"], x["schedule"]) == (layout, schedule)),
                          key=lambda x: x["threads"])
            base = next((x for x in rows if x["threads"] == 1), None)
            if base is None:
                continue
            ax.plot([x["threads"] for x in rows],
                    [base["kernel_s"] / x["kernel_s"] for x in rows],
                    marker="o", ms=2.5, lw=1.0, label=f"{layout}/{schedule}")
        ax.plot([1, tmax], [1, tmax], ls=":", lw=0.7, color="0.5")
        for b in boundaries:
            ax.axvline(b, ls="--", lw=0.6, color="0.35")
        ax.set_title(r["dataset"].split("(")[0], pad=2)
        ax.set_xlabel("threads $p$")
        ax.set_ylabel("kernel speedup")
    for ax in list(axes.flat)[len(results):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out)
    print(f"wrote {out}")


NAMES = {"CollegeMsg": "CollegeMsg", "BitcoinOTC": "Bitcoin-OTC", "BitcoinOTCpos": "Bitcoin-OTC(+)"}


def _label(r):
    return NAMES.get(r["dataset"]) or f"synthetic n={r['n']:,}"


def _series(r, layout, schedule):
    rows = sorted((x for x in r["rows"] if (x["layout"], x["schedule"]) == (layout, schedule)),
                  key=lambda x: x["threads"])
    base = min((x for x in r["rows"] if x["threads"] == 1), key=lambda x: x["kernel_s"])
    return [x["threads"] for x in rows], [base["kernel_s"] / x["kernel_s"] for x in rows]


def plot_paper(small, large, out, boundaries):
    """Main-paper figure: kernel speedup over the fastest one-thread
    configuration. (a) real datasets and synthetic n=8000, default tiled/static;
    (b) large synthetic streams, tiled static (solid) and dynamic (dashed)."""
    plt = apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN, 1.9), sharey=True)
    marks = ["o", "s", "^", "D"]
    for i, r in enumerate(small["results"]):
        x, y = _series(r, "tiled", "static")
        a.plot(x, y, marker=marks[i % 4], ms=2.6, lw=0.9, label=_label(r))
    for i, r in enumerate(large["results"]):
        x, y = _series(r, "tiled", "static")
        line, = b.plot(x, y, marker=marks[i % 4], ms=2.6, lw=0.9,
                       label=f"n={r['n']:,} static")
        x, y = _series(r, "tiled", "dynamic")
        b.plot(x, y, marker=marks[i % 4], ms=2.6, lw=0.9, ls="--", color=line.get_color(),
               mfc="white", label=f"n={r['n']:,} dynamic")
    tmax = max(x["threads"] for x in small["results"][0]["rows"])
    for ax, title in ((a, "(a) real data, n=8,000"), (b, "(b) n=50,000 / 100,000")):
        ax.plot([1, tmax], [1, tmax], ls=":", lw=0.7, color="0.55", gid="reference")
        for bd in boundaries[:-1]:
            ax.axvline(bd, ls="--", lw=0.6, color="0.35", gid="reference")
        ax.set_xlim(0.5, tmax + 0.5)
        ax.set_ylim(0, 10)
        ax.set_xticks([1, 6, 12, 18])
        ax.set_xlabel("threads $p$")
        ax.set_title(title, pad=2)
        ax.legend(frameon=False, fontsize=6.5, loc="upper right", handlelength=1.6,
                  borderaxespad=0.1, labelspacing=0.2)
    a.set_ylabel("kernel speedup")
    fig.tight_layout(pad=0.3, w_pad=0.6)
    save_with_layout(fig, out)
    print(f"wrote {out}")


def plot_breakdown(small, large, out):
    """Main-paper figure: share of sequential preprocessing and
    kernel in the end-to-end time, at p=1 and at the fastest end-to-end p."""
    plt = apply_style()
    results = small["results"] + large["results"]
    labels, pre1, ker1, preb, kerb, pbest = [], [], [], [], [], []
    for r in results:
        cand = [x for x in r["rows"] if x["layout"] == "tiled"]
        one = min((x for x in cand if x["threads"] == 1), key=lambda x: x["end_to_end_s"])
        best = min(cand, key=lambda x: x["end_to_end_s"])
        prep = r["prep_s"]
        for lst, x in ((pre1, one), (preb, best)):
            lst.append(100 * prep / x["end_to_end_s"])
        ker1.append(100 - pre1[-1])
        kerb.append(100 - preb[-1])
        pbest.append(best["threads"])
        labels.append(_label(r).replace("synthetic ", ""))
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 1.85))
    import numpy as np
    y = np.arange(len(results))
    h = 0.36
    ax.barh(y - h / 2, pre1, h, color="0.25", label="preprocessing, $p{=}1$")
    ax.barh(y - h / 2, ker1, h, left=pre1, color="0.75", label="kernel, $p{=}1$")
    ax.barh(y + h / 2, preb, h, color="#c0392b", label="preprocessing, best $p$")
    ax.barh(y + h / 2, kerb, h, left=preb, color="#f5b7b1", label="kernel, best $p$")
    for i, pb in enumerate(pbest):
        ax.text(101, y[i] + h / 2, f"p={pb}", va="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of end-to-end time (%)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.5, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(pad=0.3, rect=(0, 0.17, 1, 1))
    save_with_layout(fig, out)
    print(f"wrote {out}")


def plot_roofline(data, out):
    plt = apply_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.6))
    s = data["stream_triad"]["rows"]
    ax.plot([r["threads"] for r in s], [r["GBps"] for r in s], color="0.2", lw=1.2,
            marker="s", ms=2.5, label="STREAM triad (measured)")
    for k in data["kernels"]:
        for layout in ("tiled",):
            rows = sorted((r for r in k["rows"] if r["layout"] == layout),
                          key=lambda r: r["threads"])
            ax.plot([r["threads"] for r in rows], [r["achieved_GBps"] for r in rows],
                    marker="o", ms=2.5, lw=0.9,
                    label=_label(k) if "n" in k else k["dataset"])
    ax.set_xlabel("threads $p$")
    ax.set_ylabel("GB/s (touched bytes)")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.25),
              fontsize=7)
    fig.tight_layout()
    save_with_layout(fig, out)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("scaling")
    a.add_argument("--json", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--boundaries", default=None,
                   help="comma list overriding the recorded core boundaries, e.g. 6,18")
    c = sub.add_parser("paper")
    c.add_argument("--json", required=True, help="parallel_scaling.json")
    c.add_argument("--large", required=True, help="parallel_scaling_large.json")
    c.add_argument("--out", required=True)
    d = sub.add_parser("breakdown")
    d.add_argument("--json", required=True, help="parallel_scaling.json")
    d.add_argument("--large", required=True, help="parallel_scaling_large.json")
    d.add_argument("--out", required=True)
    b = sub.add_parser("roofline")
    b.add_argument("--json", required=True)
    b.add_argument("--out", required=True)
    args = ap.parse_args()
    from experiments import figstyle
    figstyle.FIGURE_SOURCES[:] = [args.json] + ([args.large] if getattr(args, "large", None) else [])
    data = json.loads(Path(args.json).read_text())
    if data.get("pending"):
        raise SystemExit("refusing to plot a PENDING (single-core) result file")
    if args.cmd == "breakdown":
        plot_breakdown(data, json.loads(Path(args.large).read_text()), args.out)
    elif args.cmd == "paper":
        large = json.loads(Path(args.large).read_text())
        plot_paper(data, large, args.out, core_boundaries(data["machine"]))
    elif args.cmd == "scaling":
        plot_scaling(data, args.out, core_boundaries(data["machine"], args.boundaries))
    else:
        plot_roofline(data, args.out)


if __name__ == "__main__":
    main()
