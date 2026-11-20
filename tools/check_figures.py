#!/usr/bin/env python3
"""Figure overlap check (STRATA-v9; backlog R22).

TikZ figures (``figures/fig_*.tex`` that ``\\input{geomhook}``) are compiled in
a temporary directory; ``geomhook.tex`` makes every node write its bounding box
and every ``geomarrow`` / ``geomcurve`` path write its tip, tip direction, and
the point a fixed distance back along the path. Matplotlib figures write a
``<figure>.layout.json`` sidecar at save time (``experiments/figstyle.py``).

Checks (thresholds in THRESH, one place):
  1. arrowhead overlap  -- the arrowhead segment enters a node other than its
                           target (containers excluded);
  2. branch length      -- a straight-ending arrow is not straight over
                           arrowhead + 2 mm before the tip;
  3. tip contact        -- the tip is farther than 0.3 mm from the nearest node
                           border, or sits deeper than 0.3 mm inside a node;
  4. text overlap       -- two non-container nodes (TikZ) or two text artists
                           (matplotlib) overlap, or a legend covers a data line;
  5. width              -- the figure is wider than its printed width.

Usage::

  python3 tools/check_figures.py                 # all paper figures
  python3 tools/check_figures.py --render        # also 400 dpi PNGs in figures/_check/
  python3 tools/check_figures.py --tex FILE.tex  # one TikZ file (tests)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent.parent.parent / "STRATA-latex" / "STRATA-paper-EN"
FIGS = PAPER / "figures"

MM = 72.27 / 25.4   # TeX points per millimetre
THRESH = {
    "tip_touch_mm": 0.3,      # check 3
    "tip_inside_mm": 0.3,     # check 3
    "straight_tol_pt": 0.5,   # check 2 (back point off the tip line)
    "overlap_pt": 2.5,        # check 4 (TikZ): overlap depth in both axes
    "border_pt": 2.0,         # standalone border added to every TikZ figure
    "width_tol_pt": 0.5,      # check 5
}
COLUMN_PT, TEXT_PT = 252.94, 517.94
WIDE = {"fig_architecture"}   # figure* (text width); everything else is one column


def _f(tok):
    return float(tok.replace("pt", ""))


def _dim_mm(tok):
    if tok.endswith("mm"):
        return float(tok[:-2])
    if tok.endswith("pt"):
        return float(tok[:-2]) / MM
    return float(tok)


shapes = {}


def parse_geom(text):
    shapes.clear()
    nodes, arrows = {}, {}
    for line in text.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "N" and len(p) >= 6:
            x1, y1, x2, y2 = map(_f, p[2:6])
            box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            if len(p) >= 10:
                wx, ex, ny, sy = map(_f, p[6:10])
                full = (min(wx, ex), min(ny, sy), max(wx, ex), max(ny, sy))
                # round shapes: the corner anchors lie inside the west / east extent
                if full[0] < box[0] - 0.1 and full[2] > box[2] + 0.1:
                    shapes[p[1]] = "circle"
                    box = full
            nodes[p[1]] = box
        elif p[0] == "L":
            arrows.setdefault(p[1], {})["len_mm"] = _dim_mm(p[2])
            arrows[p[1]]["need_mm"] = _dim_mm(p[3])
        elif p[0] == "B":
            arrows.setdefault(p[1], {})["back"] = (_f(p[2]), _f(p[3]))
        elif p[0] == "T":
            a = arrows.setdefault(p[1], {})
            a["kind"] = p[2]
            ex, ey, dx, dy = map(_f, p[3:7])
            n = math.hypot(dx - ex, dy - ey) or 1.0
            a["tip"] = (ex, ey)
            a["dir"] = ((dx - ex) / n, (dy - ey) / n)
    return nodes, arrows


def containers(nodes):
    out = set()
    for a, A in nodes.items():
        for b, B in nodes.items():
            if a != b and A[0] <= B[0] + 0.01 and A[1] <= B[1] + 0.01 \
                    and A[2] >= B[2] - 0.01 and A[3] >= B[3] - 0.01:
                out.add(a)
    return out


def _inside_depth(pt, box):
    x, y = pt
    return min(x - box[0], box[2] - x, y - box[1], box[3] - y)  # >0 inside


def _circle(box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return cx, cy, min(box[2] - box[0], box[3] - box[1]) / 2


def _depth(pt, name, box):
    """Inside depth (>0 inside) using the true outline for circle nodes."""
    if shapes.get(name) == "circle":
        cx, cy, r = _circle(box)
        return r - math.hypot(pt[0] - cx, pt[1] - cy)
    return _inside_depth(pt, box)


def _bdist(pt, name, box):
    if shapes.get(name) == "circle":
        return abs(_depth(pt, name, box))
    return _border_dist(pt, box)


def _border_dist(pt, box):
    d = _inside_depth(pt, box)
    if d >= 0:
        return d
    x, y = pt
    dx = max(box[0] - x, 0, x - box[2])
    dy = max(box[1] - y, 0, y - box[3])
    return math.hypot(dx, dy)


def check_geometry(nodes, arrows, name):
    issues = []
    cont = containers(nodes)
    plain = {k: v for k, v in nodes.items() if k not in cont}
    for aid, a in sorted(arrows.items(), key=lambda kv: int(kv[0])):
        if "tip" not in a:
            continue
        tip, d = a["tip"], a["dir"]
        L = a.get("len_mm", 2.0) * MM
        # target: node (container or not) whose border is nearest to the tip
        target, tdist = None, float("inf")
        for k, box in nodes.items():
            bd = _bdist(tip, k, box)
            if bd < tdist - 1e-6 or (abs(bd - tdist) <= 1e-6 and k in plain):
                target, tdist = k, bd
        if target is None:
            continue
        depth = max([_depth(tip, k, b) for k, b in plain.items() if k == target] or [-1.0])
        if tdist > THRESH["tip_touch_mm"] * MM and depth < 0:
            issues.append(f"{name}: arrow {aid} tip is {tdist / MM:.2f} mm from the nearest node ({target})")
        if depth > THRESH["tip_inside_mm"] * MM:
            issues.append(f"{name}: arrow {aid} tip sits {depth / MM:.2f} mm inside node {target}")
        # check 1: arrowhead samples must not enter another plain node
        for i in range(1, 11):
            s = (tip[0] - d[0] * L * i / 10, tip[1] - d[1] * L * i / 10)
            for k, box in plain.items():
                if k != target and _depth(s, k, box) > 0.2:
                    issues.append(f"{name}: arrow {aid} head overlaps node {k}")
                    break
            else:
                continue
            break
        # check 2: straight over the required length
        if a.get("kind") == "S" and a.get("need_mm", 0) > 0 and "back" in a:
            need = a["need_mm"] * MM
            exp = (tip[0] - d[0] * need, tip[1] - d[1] * need)
            off = math.hypot(a["back"][0] - exp[0], a["back"][1] - exp[1])
            if off > THRESH["straight_tol_pt"]:
                issues.append(f"{name}: arrow {aid} to {target} is straight for less than "
                              f"{a['need_mm']:.1f} mm before its tip (arrowhead {a.get('len_mm', 2.0):.1f} mm + 2 mm)")
    # check 4: plain node overlaps
    keys = sorted(plain)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            A, B = plain[a], plain[b]
            ox = min(A[2], B[2]) - max(A[0], B[0])
            oy = min(A[3], B[3]) - max(A[1], B[1])
            if ox > THRESH["overlap_pt"] and oy > THRESH["overlap_pt"]:
                issues.append(f"{name}: nodes {a} and {b} overlap ({ox:.1f} x {oy:.1f} pt)")
    return issues


def pdf_width(pdf):
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", out)
    return float(m.group(1)) if m else None


def width_issue(name, width, tikz):
    limit = TEXT_PT if name in WIDE else COLUMN_PT
    allow = limit + (2 * THRESH["border_pt"] if tikz else 0) + THRESH["width_tol_pt"]
    if width is not None and width > allow:
        return [f"{name}: {width:.1f} pt wide, printed width {limit:.1f} pt"]
    return []


def check_tex(tex: Path, render_dir: Path | None = None):
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        shutil.copy(tex, tdir / tex.name)
        hook = tex.parent / "geomhook.tex"
        if not hook.exists():
            hook = FIGS / "geomhook.tex"
        shutil.copy(hook, tdir / "geomhook.tex")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", tex.name], cwd=tdir,
                           capture_output=True, text=True)
        pdf = tdir / (tex.stem + ".pdf")
        geom = tdir / (tex.stem + ".geom")
        if r.returncode != 0 or not pdf.exists():
            return [f"{tex.stem}: pdflatex failed"]
        if not geom.exists():
            return [f"{tex.stem}: no geometry dump (does it \\input{{geomhook}}?)"]
        nodes, arrows = parse_geom(geom.read_text())
        issues = check_geometry(nodes, arrows, tex.stem)
        issues += width_issue(tex.stem, pdf_width(pdf), tikz=True)
        if render_dir is not None:
            render_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["pdftoppm", "-png", "-r", "400", "-singlefile", str(pdf),
                            str(render_dir / tex.stem)], capture_output=True)
        return issues


def check_layout_json(js: Path):
    data = json.loads(js.read_text())
    name = js.name.split(".")[0]
    issues = [f"{name}: text overlap {o}" for o in data.get("text_overlaps", [])]
    issues += [f"{name}: legend covers data line {o}" for o in data.get("legend_over_data", [])]
    return issues


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=None, help="check one TikZ file")
    ap.add_argument("--render", action="store_true", help="write 400 dpi PNGs to figures/_check/")
    args = ap.parse_args(argv)
    render = FIGS / "_check" if args.render else None
    issues, checked = [], []
    if args.tex:
        issues += check_tex(Path(args.tex), render)
        checked.append(Path(args.tex).stem)
    else:
        used = set(re.findall(r"figures/(fig_[A-Za-z0-9_]+)\.pdf",
                              (PAPER / "main.tex").read_text()))
        for name in sorted(used):
            tex, pdf = FIGS / f"{name}.tex", FIGS / f"{name}.pdf"
            js = FIGS / f"{name}.layout.json"
            if tex.exists():
                issues += check_tex(tex, render)
            elif js.exists():
                issues += check_layout_json(js)
                issues += width_issue(name, pdf_width(pdf), tikz=False)
            else:
                issues.append(f"{name}: no TikZ source and no layout sidecar (regenerate it)")
            checked.append(name)
    for i in issues:
        print(f"[FIG] {i}")
    print(f"[check_figures] {len(checked)} figures, {len(issues)} issues:",
          "PASS" if not issues else "FAIL")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
