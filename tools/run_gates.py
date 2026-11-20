#!/usr/bin/env python3
"""Run the release gates in order and stop at the first failure (STRATA-v12, R38).

  1. compile clean   main.tex + supplementary.tex: no LaTeX error, no Overfull box,
                     no undefined reference or citation
  2. page budget     main <= MAIN_PAGES, supplementary <= SUPP_PAGES
  3. pytest          the full suite
  4. check_numbers   every M5 number equals a fresh derivation from the result files
  5. check_figures   no overlapping arrows / boxes / text in any figure (user
                     decision 2026-11-16, R38)
  (info) page budget: lines that still fit before each page limit (R66, v26)
  (optional) privacy scan of the submission bundle: --scan (local paths only)
  (optional) reference check against Crossref / DBLP: --bib (R46; needs network
             once, then compares from reports/bib_cache.json; fails on a mismatch)
  The double-anonymous scan left the gates in v15 (TPDS is single-blind, R45).

Needs pdflatex and bibtex. Usage: python3 tools/run_gates.py [--skip-compile] [--scan] [--bib]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent.parent / "STRATA-latex" / "STRATA-paper-EN"
MAIN_PAGES = 12          # backlog R3 / R18: 12 pages including references
SUPP_PAGES = 4

GATES = ["compile", "pages", "pytest", "check_numbers", "check_figures"]


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def pages(pdf):
    """Page count via pypdf, else pdfinfo; -1 (a failed gate) if neither works."""
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf)).pages)
    except Exception:
        pass
    try:
        out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        return int(re.search(r"^Pages:\s+(\d+)", out, re.M).group(1))
    except Exception:
        return -1


def compile_doc(name):
    for cmd in (["pdflatex", "-interaction=nonstopmode", name],
                ["bibtex", name] if name == "main" else None,
                ["pdflatex", "-interaction=nonstopmode", name],
                ["pdflatex", "-interaction=nonstopmode", name]):
        if cmd:
            run(cmd, PAPER)
    log = (PAPER / f"{name}.log").read_text(errors="replace")
    bad = {"errors": len(re.findall(r"^! ", log, re.M)),
           "overfull": log.count("Overfull"),
           "undefined": len(re.findall(r"undefined", log, re.I))}
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--scan", action="store_true", help="also run the (optional) privacy scan")
    ap.add_argument("--bib", action="store_true",
                    help="also run the (optional) reference check, tools/check_bib.py --fetch")
    args = ap.parse_args(argv)
    results = []
    if not args.skip_compile:
        for doc in ("main", "supplementary"):
            bad = compile_doc(doc)
            ok = not any(bad.values())
            results.append(("compile", doc, ok, str(bad)))
            if not ok:
                return report(results)
    pm, ps_ = pages(PAPER / "main.pdf"), pages(PAPER / "supplementary.pdf")
    results.append(("pages", "main / supplementary", 0 < pm <= MAIN_PAGES and 0 < ps_ <= SUPP_PAGES,
                     f"{pm} / {ps_} (limit {MAIN_PAGES} / {SUPP_PAGES})"))
    if not args.skip_compile and results[-1][2]:
        # v26 (R66): informational, not a gate -- lines left before the page limits
        code, out = run([sys.executable, "tools/page_budget.py"], ROOT)
        print((out.strip().splitlines() or ["[page budget] unavailable"])[-1])
    for gate, cmd in (("pytest", [sys.executable, "-m", "pytest", "tests/", "-q"]),
                      ("check_numbers", [sys.executable, "tools/check_numbers.py"]),
                      ("check_figures", [sys.executable, "tools/check_figures.py"])) + (
                     (("privacy (optional)", [sys.executable, "tools/make_submission_bundle.py",
                                              "--scan-only"]),) if args.scan else ()) + (
                     (("bib (optional)", [sys.executable, "tools/check_bib.py", "--fetch"]),)
                     if args.bib else ()):
        if not results[-1][2]:
            break
        code, out = run(cmd, ROOT)
        tail = [line for line in out.strip().splitlines() if line.strip()][-1:] or [""]
        results.append((gate, "", code == 0, tail[0]))
    return report(results)


def report(results):
    for gate, what, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {gate} {what} {detail}".rstrip())
    passed = all(r[2] for r in results) and {r[0] for r in results} >= set(GATES)
    print(f"[gates] {'ALL PASS' if passed else 'STOPPED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
