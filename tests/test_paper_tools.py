"""Paper tooling: generated macros / tables, release gates, figure sources, stated counts.

Not part of the public reproduction repository (it needs the manuscript and the
measured result files)."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)


def test_m5_tables_switch_to_best_median_when_samples_exist():
    from experiments import m5_numbers

    x = {"kernel_s": 0.002, "kernel_samples_s": [0.002, 0.003, 0.004]}
    assert m5_numbers.cell(x, "kernel_s", 3) == "2.000 (3.000)"
    assert m5_numbers.cell({"kernel_s": 0.002}, "kernel_s", 3) == "2.000"


def test_m5_numbers_refuses_unrecorded_single_thread_files(tmp_path, monkeypatch):
    from experiments import m5_numbers

    real = m5_numbers.load_perf

    def fake(name):
        d = real(name)
        d.pop("threads", None)
        return d
    monkeypatch.setattr(m5_numbers, "load_perf", fake)
    with pytest.raises(SystemExit):
        m5_numbers.outputs()


def test_plateau_band_over_the_thread_grid():
    from experiments.m5_numbers import plateau

    def rows(meds):
        return [{"threads": p, "kernel_s": v, "kernel_samples_s": [v, v * 1.001, v]}
                for p, v in zip([1, 2, 4, 6, 8, 12, 16, 18], meds)]
    assert plateau(rows([8, 5, 3, 2.0, 2.2, 2.05, 2.08, 2.5])) == "6, 12--16"
    assert plateau(rows([8, 5, 3, 2.5, 2.2, 2.0, 2.01, 2.02])) == "12--18"
    assert plateau(rows([1, 2, 3, 4, 5, 6, 7, 8])) == "1"


def test_gate_runner_lists_the_gates():
    """R38 / R45: five gates; the anonymity scan is optional since TPDS is single-blind."""
    sys.path.insert(0, ROOT)
    from tools import run_gates
    assert run_gates.GATES == ["compile", "pages", "pytest", "check_numbers", "check_figures"]


def test_data_figures_record_their_result_files(tmp_path):
    """R41: the layout sidecar of a data figure names its source files with sha256."""
    import hashlib
    out = tmp_path / "fig_roofline.pdf"
    r = subprocess.run([sys.executable, "experiments/plot_parallel.py", "roofline",
                        "--json", "results/real/bandwidth_roofline.json", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    side = json.loads(out.with_suffix(".layout.json").read_text())
    (src,) = side["sources"]
    assert src["path"] == "results/real/bandwidth_roofline.json"
    data = open(os.path.join(ROOT, src["path"]), "rb").read()
    assert src["sha256"] == hashlib.sha256(data).hexdigest()


def test_doc_counts_flag_drift(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import doc_counts
    main = tmp_path / "main.tex"
    main.write_text("The artifact core is 2.8k non-blank lines of Python, the harnesses 8.1k, "
                    "and the test suite 1.8k lines in 247 tests.")
    supp = tmp_path / "supp.tex"
    supp.write_text("The suite has 245 tests: ...")
    stm = [(main, r"lines in (\d+) tests", "tests"), (supp, r"The suite has (\d+) tests", "tests")]
    size = (main, doc_counts.SIZE_STATEMENT[1])
    sizes = {"core": 2754, "harness": 8107, "tests": 1790}
    bad = doc_counts.check(247, sizes, stm, size)
    assert bad == [(str(supp), "tests", 245, 247)]
    bad = doc_counts.check(247, {"core": 2754, "harness": 8200, "tests": 1790}, stm[:1], size)
    assert bad and bad[0][1] == "code size"
    assert doc_counts.check(247, sizes, stm[:1], size) == []



def test_range_macro_in_math_is_flagged(tmp_path):
    """v21 (R53): "$p{=}\\MSupPlat$" printed "16 − −18" in v20."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import tex_lint
    mac = tmp_path / "m.tex"
    mac.write_text("\\newcommand{\\MSupPlat}{16--18}\n\\newcommand{\\MP}{18}\n")
    bad = tmp_path / "a.tex"
    bad.write_text("scales (p=\\MSupPlat) and $p{=}\\MSupPlat$ but $p{=}\\MP$; % $\\MSupPlat$\n")
    hits = tex_lint.range_macros_in_math([bad], mac)
    assert [h[1] for h in hits] == ["MSupPlat"]
    bad.write_text("$p{=}$\\MSupPlat and 50\\$ \\MSupPlat\n")
    assert tex_lint.range_macros_in_math([bad], mac) == []
