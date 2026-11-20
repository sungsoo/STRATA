"""Shared matplotlib style for paper figures.

* Figures are created at their final printed size: one IEEEtran compsoc
  column is 252.94 pt = 3.50 in, the full text width 517.94 pt = 7.17 in.
  LaTeX then includes them at scale 1, so 8 pt in the figure is 8 pt in print.
* Text is 8 pt (the paper's \\footnotesize is 8.03 pt), in the body's Palatino
  family: TeX Gyre Pagella if available (TeX Live ships it), else Palatino.
* Fonts are embedded as TrueType (pdf.fonttype 42): no Type 3 fonts.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess

COLUMN_WIDTH_IN = 252.94 / 72.27
TEXT_WIDTH_IN = 517.94 / 72.27
FONT_PT = 8.0

_FAMILY = ["TeX Gyre Pagella", "Palatino", "Palatino Linotype", "URW Palladio L",
           "P052", "serif"]


def _register_pagella(font_manager) -> None:
    candidates = []
    kpse = shutil.which("kpsewhich")
    if kpse:
        try:
            p = subprocess.run([kpse, "texgyrepagella-regular.otf"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
            if p:
                candidates += glob.glob(os.path.join(os.path.dirname(p), "texgyrepagella-*.otf"))
        except Exception:
            pass
    for root in ("/usr/share/texmf/fonts/opentype/public/tex-gyre",
                 "/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre",
                 "/Library/TeX/Root/texmf-dist/fonts/opentype/public/tex-gyre",
                 "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/tex-gyre"):
        candidates += glob.glob(os.path.join(root, "texgyrepagella-*.otf"))
    for c in sorted(set(candidates)):
        try:
            font_manager.fontManager.addfont(c)
        except Exception:
            pass


def apply_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    _register_pagella(font_manager)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": _FAMILY,
        "mathtext.fontset": "custom",
        "mathtext.rm": _FAMILY[0],
        "mathtext.it": f"{_FAMILY[0]}:italic",
        "mathtext.bf": f"{_FAMILY[0]}:bold",
        "mathtext.sf": _FAMILY[0],
        "mathtext.cal": _FAMILY[0],
        "mathtext.fallback": "cm",
        "font.size": FONT_PT,
        "axes.titlesize": FONT_PT,
        "axes.labelsize": FONT_PT,
        "xtick.labelsize": FONT_PT,
        "ytick.labelsize": FONT_PT,
        "legend.fontsize": FONT_PT,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "standard",
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


# --------------------------------------------------------------------------- #
# Layout sidecar for tools/check_figures.py
# --------------------------------------------------------------------------- #
def _seg_hits_box(p, q, box):
    """True if segment p-q intersects the axis-aligned box (x0, y0, x1, y1)."""
    (x0, y0, x1, y1) = box
    t0, t1 = 0.0, 1.0
    dx, dy = q[0] - p[0], q[1] - p[1]
    for pk, qk in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pk == 0:
            if qk < 0:
                return False
            continue
        t = qk / pk
        if pk < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def file_sha256(path):
    import hashlib
    from pathlib import Path
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# set by the plotting scripts before saving: the result files a
# figure is drawn from, recorded with their sha256 in the layout sidecar so
# that tools/check_numbers.py can refuse a figure drawn from stale results
FIGURE_SOURCES = []


def save_with_layout(fig, out):
    """Save ``fig`` and write ``<out>.layout.json`` with text overlaps and
    legend-over-data hits (display coordinates, after drawing). Lines with
    ``gid == "reference"`` (ideal scaling, core boundaries) may pass under a
    legend; data series may not."""
    import json
    from pathlib import Path

    fig.savefig(out)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    for t in fig.findobj(match=lambda a: a.__class__.__name__ == "Text"):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(renderer)
        if bb.width > 0 and bb.height > 0:
            texts.append((t.get_text(), bb))
    overlaps = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i][1], texts[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 1.0 and oy > 1.0:
                overlaps.append([texts[i][0], texts[j][0]])
    legend_hits = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(renderer)
        box = (lb.x0, lb.y0, lb.x1, lb.y1)
        for line in ax.get_lines():
            if line.get_gid() == "reference" or not line.get_visible():
                continue
            xy = ax.transData.transform(line.get_xydata())
            for k in range(len(xy) - 1):
                if _seg_hits_box(tuple(xy[k]), tuple(xy[k + 1]), box):
                    legend_hits.append(line.get_label())
                    break
        for patch in ax.patches:
            pb = patch.get_window_extent(renderer)
            if min(pb.x1, box[2]) - max(pb.x0, box[0]) > 1 and min(pb.y1, box[3]) - max(pb.y0, box[1]) > 1:
                legend_hits.append(patch.get_label() or "bar")
    side = Path(str(out)).with_suffix(".layout.json")
    root = Path(__file__).resolve().parents[1]
    sources = []
    for src in FIGURE_SOURCES:
        sp = Path(src).resolve()
        try:
            rel = str(sp.relative_to(root))
        except ValueError:
            rel = str(sp)
        sources.append({"path": rel, "sha256": file_sha256(sp)})
    side.write_text(json.dumps({"figure": Path(str(out)).name, "text_overlaps": overlaps,
                                "legend_over_data": sorted(set(legend_hits)),
                                "sources": sources}, indent=1))
    return side
