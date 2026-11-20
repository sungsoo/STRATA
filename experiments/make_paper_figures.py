#!/usr/bin/env python3
"""Regenerate the paper's measured-performance figure from its evidence file (R2).

Why
---
The v1 figure files fig_speedup_vs_n.pdf / fig_runtime_bar.pdf had no generator
in the artifact and their plotted values did not match the numbers the text
quotes for Fig. ``fig:speedup`` (24.5x -> 312x, ego-Facebook 205x, from
results/perf/perf_table.json). This script draws the figure directly from that
JSON, so figure and text share one source, and draws it at the printed column
width with 8 pt Palatino-family text and TrueType fonts (no Type 3).

What
----
fig_speedup_runtime.pdf: grouped bars of exact temporal BFS vs packed STRATA
wall-clock (log y) for the five preferential-attachment streams and
ego-Facebook, each pair labelled with its speedup. No number is computed here
beyond the ratio already stored in the JSON; the script asserts that the stored
speedup equals exact/packed to the stored precision.

Usage
-----
  python3 experiments/make_paper_figures.py \
      [--perf results/perf/perf_table.json] \
      [--outdir ../../STRATA-latex/STRATA-paper-EN/figures]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from experiments.figstyle import COLUMN_WIDTH_IN, apply_style, save_with_layout  # noqa: E402

HERE = Path(__file__).resolve().parent
EXACT = "#D1495B"
PACKED = "#1F6FEB"


def speedup_runtime(perf, out):
    plt = apply_style()
    import numpy as np

    rows = perf["rows"]
    for r in rows:
        ratio = r["exact_ms"] / r["packed_ms"]
        # the JSON rounds times to 0.1 ms; allow that rounding (v10: M5 runs reach 1.0 ms)
        tol = 0.01 + 0.05 / r["packed_ms"] + 0.05 / r["exact_ms"]
        assert abs(ratio - r["speedup"]) / r["speedup"] < tol, r["label"]
    labels = [(str(r["n"]) if r["label"].startswith("synth") else f"ego-FB\n{r['n']}")
              for r in rows]
    x = np.arange(len(rows))
    wbar = 0.38
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 1.95))
    ax.bar(x - wbar / 2, [r["exact_ms"] for r in rows], wbar, color=EXACT,
           label="exact temporal BFS")
    ax.bar(x + wbar / 2, [r["packed_ms"] for r in rows], wbar, color=PACKED,
           label="packed STRATA")
    ax.set_yscale("log")
    ax.set_ylim(1, 1.5e6)
    # plain tick labels (no 10^k superscripts, which would print below 8 pt)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1e3, 1e4, 1e5, 1e6]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: {1: "1", 10: "10", 100: "100", 1e3: "1k", 1e4: "10k",
                      1e5: "100k", 1e6: "1M"}.get(v, "")))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("vertices $n$ (synthetic PA streams)  |  real graph", labelpad=2)
    ax.set_ylabel("runtime (ms, log)")
    for xi, r in zip(x, rows):
        ax.text(xi, r["exact_ms"] * 1.6, f"{r['speedup']:.1f}$\\times$" if r["speedup"] < 100
                else f"{r['speedup']:.0f}$\\times$", ha="center", va="bottom")
    ax.axvline(len(rows) - 1.5, color="0.6", lw=0.6, ls=":", gid="reference")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=2,
              handlelength=1.2, columnspacing=1.5, borderaxespad=0.0)
    fig.tight_layout(pad=0.3)
    save_with_layout(fig, out)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", default=str(HERE.parent / "results" / "perf" / "perf_table.json"))
    ap.add_argument("--outdir", default=str(HERE.parent.parent.parent / "STRATA-latex"
                                            / "STRATA-paper-EN" / "figures"))
    args = ap.parse_args()
    from experiments import figstyle
    figstyle.FIGURE_SOURCES[:] = [args.perf]
    perf = json.loads(Path(args.perf).read_text())
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    speedup_runtime(perf, Path(args.outdir) / "fig_speedup_runtime.pdf")


if __name__ == "__main__":
    main()
