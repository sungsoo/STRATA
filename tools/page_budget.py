#!/usr/bin/env python3
"""How many more lines fit before the page limit (STRATA-v26, R66).

Why: since v19 every batch learned only after compiling that the paper had grown
past 12 pages and then cut sentences; in v24 a manual experiment showed that two
more lines would make page 13. The next batch should know its budget up front.

How: copy the LaTeX folder to a temporary directory, insert k filler lines of
nine short words at a fixed place (main: end of Future Work, before the
acknowledgment block; supplementary: before \\end{document}), compile with the
existing .bbl, and count pages; the largest k that stays within the limit is
found by doubling and then bisection. The original files are never touched.
The figure is informational (``run_gates.py`` prints it); it is not a gate.

What:
    python3 tools/page_budget.py            # main <= 12, supplementary <= 4
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent.parent / "STRATA-latex" / "STRATA-paper-EN"
ANCHORS = {"main": "% AI-use disclosure", "supplementary": "\\end{document}"}
LIMITS = {"main": 12, "supplementary": 4}
FILLER = " ".join(["word"] * 9)


def with_filler(text, anchor, k):
    """The document text with k filler lines (one paragraph) before ``anchor``."""
    if k <= 0:
        return text
    i = text.index(anchor)
    return text[:i] + "\n" + " ".join([FILLER] * k) + "\n\n" + text[i:]


def largest_fitting(pages_for, limit, cap=200):
    """Largest k in [0, cap] with pages_for(k) <= limit (pages_for is monotone);
    -1 if even k = 0 does not fit."""
    if pages_for(0) > limit:
        return -1
    lo, hi = 0, 1
    while hi <= cap and pages_for(hi) <= limit:
        lo, hi = hi, hi * 2
    hi = min(hi, cap + 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pages_for(mid) <= limit:
            lo = mid
        else:
            hi = mid
    return lo


def _pdf_pages(pdf):
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    return 10 ** 6


def compiled_pages(doc, k, workdir):
    src = (PAPER / f"{doc}.tex").read_text()
    name = f"{doc}_budget"
    (workdir / f"{name}.tex").write_text(with_filler(src, ANCHORS[doc], k))
    bbl = PAPER / f"{doc}.bbl"
    if bbl.exists():
        shutil.copy(bbl, workdir / f"{name}.bbl")
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", name], cwd=workdir,
                       capture_output=True, text=True)
    return _pdf_pages(workdir / f"{name}.pdf")


def budget(doc):
    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "paper"
        shutil.copytree(PAPER, work)
        return largest_fitting(lambda k: compiled_pages(doc, k, work), LIMITS[doc])


def main():
    res = {doc: budget(doc) for doc in ("main", "supplementary")}
    print("[page budget] " + ", ".join(
        f"{doc}: about {k} more line(s) within {LIMITS[doc]} pages" if k >= 0
        else f"{doc}: over {LIMITS[doc]} pages" for doc, k in res.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
