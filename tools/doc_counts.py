"""Test count and code size as stated in the documents vs. measured (STRATA-v15, R44).

The manuscript, the supplementary material, both reproduction guides, and the
submission notes each state the number of tests, and the manuscript states the
code size. They were edited by hand and drifted once (v13 said 244 tests while
the suite had 245). ``check(...)`` returns every place whose stated value
differs from the measured one; ``tools/check_numbers.py`` fails on any.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
PAPER = CODE.parent.parent / "STRATA-latex" / "STRATA-paper-EN"

GROUPS = {"core": ("temporal_strata", "backends", "consumers"),
          "harness": ("experiments", "tools"),
          "tests": ("tests",)}

# (file, regex with one group per stated value, what each group means)
TEST_STATEMENTS = [
    (PAPER / "main.tex", r"lines in (\d+) tests", "tests"),
    (PAPER / "supplementary.tex", r"The suite has (\d+) tests", "tests"),
    (PAPER / "TPDS_SUBMISSION_NOTES.md", r"the (\d+)-test suite", "tests"),
    (CODE / "REPRODUCTION_GUIDE_EN.md", r"\((\d+) tests since STRATA-v\d+;", "tests"),
    (CODE / "REPRODUCTION_GUIDE_EN.md", r"\(expect (\d+) passed\)", "tests"),
    (CODE / "REPRODUCTION_GUIDE_KO.md", r"STRATA-v\d+부터 (\d+)개", "tests"),
    (CODE / "REPRODUCTION_GUIDE_KO.md", r"\(기대: (\d+) passed\)", "tests"),
]
SIZE_STATEMENT = (PAPER / "main.tex",
                  r"The artifact core is ([\d.]+)k non-blank lines of Python, the harnesses "
                  r"([\d.]+)k, and the test suite ([\d.]+)k lines")


def collected_tests(code=CODE) -> int:
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/"],
                       cwd=code, capture_output=True, text=True)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    if not m:
        raise RuntimeError("could not count tests:\n" + r.stdout[-500:] + r.stderr[-500:])
    return int(m.group(1))


def nonblank_lines(code=CODE) -> dict:
    out = {}
    for key, dirs in GROUPS.items():
        n = 0
        for d in dirs:
            for f in sorted((code / d).rglob("*.py")):
                if "__pycache__" in f.parts:
                    continue
                n += sum(1 for line in f.read_text(errors="replace").splitlines() if line.strip())
        out[key] = n
    return out


def kfmt(n: int) -> str:
    return f"{n / 1000:.1f}"


def check(n_tests: int, sizes: dict, statements=TEST_STATEMENTS, size_statement=SIZE_STATEMENT):
    """List of (file, what, stated, measured) mismatches; a missing statement counts."""
    bad = []
    for path, rx, what in statements:
        found = re.findall(rx, Path(path).read_text()) if Path(path).exists() else []
        if not found:
            bad.append((str(path), what, None, n_tests))
        for v in found:
            if int(v) != n_tests:
                bad.append((str(path), what, int(v), n_tests))
    path, rx = size_statement
    m = re.search(rx, Path(path).read_text()) if Path(path).exists() else None
    want = (kfmt(sizes["core"]), kfmt(sizes["harness"]), kfmt(sizes["tests"]))
    if not m:
        bad.append((str(path), "code size", None, want))
    elif m.groups() != want:
        bad.append((str(path), "code size", m.groups(), want))
    return bad
