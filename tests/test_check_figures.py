"""Figure overlap check (STRATA-v9, backlog R22)."""
from __future__ import annotations

import json
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import check_figures as cf  # noqa: E402

needs_tex = pytest.mark.skipif(shutil.which("pdflatex") is None or shutil.which("pdfinfo") is None,
                               reason="pdflatex / pdfinfo not installed")

TOY = r"""\documentclass[border=2pt,10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc}
\input{geomhook}
\begin{document}
\begin{tikzpicture}[>={Stealth[length=2.0mm]}]
\node[draw, anchor=north west] (rel) at (0,2) {relation};
\coordinate (trunk) at ($(rel.south west)+(0.1,0)$);
\node[draw, anchor=north west] (a) at ($(trunk)+(%(gap)s,-0.2)$) {box A};
\node[draw, anchor=north west] (b) at ($(trunk)+(%(gap)s,-1.0)$) {box B};
\foreach \t in {a,b} \draw[->, geomarrow] (trunk) |- (\t.west);
%(extra)s
\end{tikzpicture}
\end{document}
"""


def _toy(tmp_path, gap, extra=""):
    p = tmp_path / "fig_toy.tex"
    p.write_text(TOY % {"gap": gap, "extra": extra})
    return p


@needs_tex
def test_short_branch_fails_and_long_branch_passes(tmp_path):
    bad = cf.check_tex(_toy(tmp_path, "0.05"))
    assert any("straight for less than" in i for i in bad)
    good = cf.check_tex(_toy(tmp_path, "0.6"))
    assert good == []


@needs_tex
def test_arrowhead_into_other_node_and_text_overlap(tmp_path):
    extra = r"\node[draw] (c) at ($(a.west)+(-0.12,0)$) {X};"
    issues = cf.check_tex(_toy(tmp_path, "0.6", extra))
    assert any("head overlaps node c" in i or "overlap" in i for i in issues)


@needs_tex
def test_tip_must_touch_its_box(tmp_path):
    extra = r"\draw[->, geomarrow] (rel.east) -- ++(0.6,0);"   # ends in empty space
    issues = cf.check_tex(_toy(tmp_path, "0.6", extra))
    assert any("from the nearest node" in i for i in issues)


def test_geometry_parser_and_checks_without_tex():
    geom = "\n".join([
        "N a 0pt 0pt 20pt 10pt 0pt 20pt 10pt 0pt",
        "N b 30pt 0pt 50pt 10pt 30pt 50pt 10pt 0pt",
        "L 1 2.0mm 4.0mm", "B 1 18.6pt 5pt", "T 1 S 30pt 5pt 31pt 5pt",
    ])
    nodes, arrows = cf.parse_geom(geom)
    assert cf.check_geometry(nodes, arrows, "toy") == []          # straight 11.4 pt, touches b
    arrows["1"]["back"] = (25.0, 0.0)                            # bent before the tip
    assert any("straight" in i for i in cf.check_geometry(nodes, arrows, "toy"))


def test_layout_sidecar_flags_legend_over_data(tmp_path):
    pytest.importorskip("matplotlib")
    from experiments.figstyle import apply_style, save_with_layout

    plt = apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 2))
    ax.plot([0, 1], [0.95, 0.95], label="data")                  # runs under the legend
    ax.plot([0, 1], [0, 1], gid="reference", label="ideal")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    side = save_with_layout(fig, tmp_path / "fig_x.pdf")
    data = json.loads(side.read_text())
    assert "data" in data["legend_over_data"] and "ideal" not in data["legend_over_data"]
    assert cf.check_layout_json(side)
