#!/usr/bin/env python3
"""Gate C5: machine-independent number <-> evidence consistency check (I9).

Recomputes the paper's machine-independent invariants directly from the real
datasets (no cached JSON trusted), then verifies that the manuscript states
exactly these values. Scope is deliberately the *invariant* claims -- pair
counts, static over-count percentages, the worked-example pair counts, and the
required PENDING markers; wall-clock numbers are machine-dependent by design
(I4) and are covered instead by the harness emission discipline. Exits
non-zero on any mismatch.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from temporal_strata.packed_reachability import (  # noqa: E402
    packed_to_matrix,
    temporal_strata_packed,
)

PAPER = os.path.join(ROOT, "..", "..", "STRATA-latex", "STRATA-paper-EN",
                     "main.tex")
DATA = os.path.join(ROOT, "datasets_real_temporal", "standard")

REGISTERED_TODO = "TODO(M5)"
REGISTERED_TODO_COUNT = 0  # STRATA-v8: both M5 markers resolved from measured files

EXPECT = {  # dataset -> (strict pairs, static-closure pairs)
    "CollegeMsg.txt": (1_794_244, 2_464_598),
    "BitcoinOTC.txt": (15_294_857, 27_690_498),
    "BitcoinOTCpos.txt": (13_351_977, 25_288_198),
}


DATA_FIGURES = ("fig_scaling", "fig_breakdown", "fig_roofline", "fig_speedup_runtime")


def compute(name):
    raw = np.loadtxt(os.path.join(DATA, name), dtype=np.int64)
    ids = np.unique(raw[:, :2])
    remap = {int(x): i for i, x in enumerate(ids)}
    n = len(ids)
    edges = [(int(t), remap[int(u)], remap[int(v)]) for u, v, t in raw]
    strict = int(packed_to_matrix(
        temporal_strata_packed(n, edges, same_time_mode="strict"), n).sum())
    static = int(packed_to_matrix(
        temporal_strata_packed(n, [(0, u, v) for _, u, v in edges],
                               same_time_mode="closure"), n).sum())
    return strict, static


def main() -> int:
    latex_dir = os.path.dirname(PAPER)
    tex = ""
    for f in sorted(os.listdir(latex_dir)):
        if f.endswith(".tex"):
            tex += open(os.path.join(latex_dir, f)).read()
    ok = True

    def need(pattern, why):
        nonlocal ok
        if not re.search(pattern, tex):
            print(f"[MISSING] {why}: /{pattern}/")
            ok = False
        else:
            print(f"[ok] {why}")

    for name, (e_strict, e_static) in EXPECT.items():
        c_strict, c_static = compute(name)
        if (c_strict, c_static) != (e_strict, e_static):
            print(f"[MISMATCH] {name}: computed {(c_strict, c_static)} "
                  f"!= expected {(e_strict, e_static)}")
            ok = False
            continue
        over = 100.0 * (c_static - c_strict) / c_strict
        print(f"[recomputed] {name}: strict={c_strict:,} static={c_static:,} "
              f"over-count={over:.1f}%")
        for v in (c_strict, c_static):
            grp = f"{v:,}".replace(",", r"[,{}\\]*")
            need(grp, f"{name} count {v:,} appears in the LaTeX sources")
        need(re.escape(f"{over:.1f}") + r"\\?%", f"{name} over-count {over:.1f}% in the LaTeX sources")

    # Worked example (strict=7 / closure=10 pairs) as stated in the ablation.
    need(r"\(correct: \$7\$\)", "worked-example strict pair count (7)")
    need(r"yields \$10\$ or \$7\$ strict pairs", "order-dependence counterexample (10 vs 7)")

    # M5 numbers (STRATA-v8): every multicore number in the paper is a macro
    # emitted by experiments/m5_numbers.py from the M5 Max result files; the
    # emitted files must equal a fresh derivation and must be \input by main.tex.
    import subprocess
    rc = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "m5_numbers.py"),
                         "--check"]).returncode
    if rc:
        print("[FAIL] m5_numbers.tex / scaling tables are stale or hand-edited")
        ok = False
    else:
        print("[ok] M5 macros and scaling tables equal a fresh derivation from results/real")
    need(r"\\input\{m5_numbers(\.tex)?\}", "main.tex inputs m5_numbers.tex")
    scaling = open(os.path.join(ROOT, "..", "..", "STRATA-latex",
                                "STRATA-paper-EN", "table_parallel_scaling.tex")).read()
    if "PENDING" in scaling:
        print("[FAIL] scaling table still PENDING")
        ok = False
    else:
        print("[ok] scaling table filled from the evaluation machine")

    # Supplementary cross-references (v28): the supplement no longer repeats
    # Algorithm 1 and pins its line numbers with \deflabel; they must equal
    # the numbers LaTeX assigned in the main paper (main.aux).
    aux = os.path.join(os.path.dirname(PAPER), "main.aux")
    supp = os.path.join(os.path.dirname(PAPER), "supplementary.tex")
    if os.path.exists(aux) and os.path.exists(supp):
        auxs = open(aux).read()
        for key, val in re.findall(r"\\deflabel\{([^}]+)\}\{([^}]+)\}", open(supp).read()):
            mm = re.search(r"\\newlabel\{" + re.escape(key) + r"\}\{\{([^}]*)\}", auxs)
            if mm and mm.group(1) == val:
                print(f"[ok] supplementary {key} = {val} matches main.aux")
            else:
                print(f"[FAIL] supplementary {key} = {val}, main.aux {mm.group(1) if mm else None}")
                ok = False

    # Abstract (R7, STRATA-v8): 150-190 words, no math / macros / citations,
    # and every M5 number in it equals the macro derived from the result files.
    main_tex = open(PAPER).read()
    mabs = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_tex, re.S)
    if not mabs:
        print("[FAIL] abstract not found")
        ok = False
    else:
        abstract = mabs.group(1)
        words = len(abstract.split())
        clean = "$" not in abstract and "\\" not in abstract and "cite" not in abstract
        if 150 <= words <= 190 and clean:
            print(f"[ok] abstract: {words} words, no math / macros / citations")
        else:
            print(f"[FAIL] abstract: {words} words, clean={clean}")
            ok = False
        sys.path.insert(0, ROOT)
        from experiments import m5_numbers
        mac, ps_, pl_ = m5_numbers.derive()
        m5_numbers.derive_v10(mac, ps_, pl_)          # v23: sx-stackoverflow macros
        for key, fmt_ in (("MCompKernelRange", "{} times faster"), ("MSynBKspDyn", "{} times its"),
                          ("MSynBPlatDyn", "(at {} threads)"), ("MThreads", "single-socket {}-core system"),
                          ("MSoMm", "a {}-million-edge"), ("MStFullS", "in {} seconds")):
            phrase = fmt_.format(mac[key].replace("--", " to "))
            if phrase in abstract:
                print(f"[ok] abstract states {phrase!r} (= \\{key})")
            else:
                print(f"[FAIL] abstract lacks {phrase!r} derived from \\{key}")
                ok = False

    # Registered TODO markers only (I8). The registered marker names the
    # evaluation machine; STRATA-v2 renamed it from TODO(M3) to TODO(M5)
    # (Apple M5 Max) together with the two markers in main.tex.
    main_only = open(PAPER).read()
    todos = re.findall(r"%% " + re.escape(REGISTERED_TODO), main_only)
    other = re.findall(r"TODO(?!" + re.escape(REGISTERED_TODO[len("TODO"):]) + r")", main_only)
    if len(todos) == REGISTERED_TODO_COUNT and not other and "PENDING" not in main_only:
        print(f"[ok] TODO markers: exactly the {REGISTERED_TODO_COUNT} registered %% {REGISTERED_TODO}")
    else:
        print(f"[FAIL] TODO markers: {len(todos)} registered, {len(other)} unregistered")
        ok = False

    # Stated test count and code size (STRATA-v15, R44)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import doc_counts
    n_tests, sizes = doc_counts.collected_tests(), doc_counts.nonblank_lines()
    bad = doc_counts.check(n_tests, sizes)
    for path, what, stated, measured in bad:
        print(f"[FAIL] {os.path.relpath(path, ROOT)}: {what} stated {stated}, measured {measured}")
        ok = False
    if not bad:
        print(f"[ok] {n_tests} tests and code size "
              f"{'/'.join(doc_counts.kfmt(sizes[k]) for k in ('core', 'harness', 'tests'))}k "
              f"match every document")

    # Range macros ("16--18") must not sit in math mode (STRATA-v21, R53)
    import tex_lint
    import glob as _glob
    tex = [os.path.join(latex_dir, x) for x in ("main.tex", "supplementary.tex", "algorithm_and_proofs.tex")]
    tex += sorted(_glob.glob(os.path.join(latex_dir, "table_*.tex")))
    bad = tex_lint.range_macros_in_math(tex, os.path.join(latex_dir, "m5_numbers.tex"))
    for f, name, snippet in bad:
        print(f"[FAIL] {os.path.basename(f)}: range macro \\{name} in math mode: {snippet}")
        ok = False
    if not bad:
        print("[ok] no range macro in math mode")

    # Every refs.bib entry is cited (STRATA-v17, R47): uncited entries live in
    # refs_unused.bib, which the submission bundle leaves out.
    import check_bib
    unused = check_bib.unused_keys()
    if unused:
        print(f"[FAIL] refs.bib: {len(unused)} uncited entries ({', '.join(unused[:5])}"
              f"{', ...' if len(unused) > 5 else ''}); move them to refs_unused.bib")
        ok = False
    else:
        print("[ok] refs.bib: every entry is cited")

    # Data figures (STRATA-v12, R41): each figure drawn from M5 result files
    # records the sha256 of those files in its layout sidecar; a figure drawn
    # from older results than the current files fails here (v10 / v11 shipped
    # fig_scaling / fig_breakdown / fig_roofline drawn from v8-era files).
    import hashlib
    fig_dir = os.path.join(latex_dir, "figures")
    for fig in DATA_FIGURES:
        side = os.path.join(fig_dir, fig + ".layout.json")
        try:
            srcs = json.load(open(side)).get("sources") or []
        except OSError:
            srcs = []
        if not srcs:
            print(f"[FAIL] {fig}: no source record; redraw it with the plotting script")
            ok = False
            continue
        stale = [x["path"] for x in srcs
                 if hashlib.sha256(open(os.path.join(ROOT, x["path"]), "rb").read()).hexdigest()
                 != x["sha256"]]
        if stale:
            print(f"[FAIL] {fig}: drawn from older versions of {', '.join(stale)}")
            ok = False
        else:
            print(f"[ok] {fig} drawn from the current {', '.join(x['path'] for x in srcs)}")

    print("[check_numbers]", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
