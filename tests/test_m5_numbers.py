"""M5 numbers are derived, not typed (STRATA-v8, backlog B)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments import m5_numbers  # noqa: E402


def test_emitted_files_equal_fresh_derivation():
    assert m5_numbers.main(["--check"]) == 0


def test_speedup_definition_uses_fastest_one_thread_config():
    r = {"dataset": "x", "rows": [
        {"threads": 1, "layout": "tiled", "schedule": "static", "kernel_s": 1.0, "end_to_end_s": 2.0},
        {"threads": 1, "layout": "rowmajor", "schedule": "static", "kernel_s": 1.5, "end_to_end_s": 2.5},
    ]}
    assert m5_numbers.base_one_thread(r)["layout"] == "tiled"
    r["rows"][1]["kernel_s"] = 0.5
    try:
        m5_numbers.base_one_thread(r)
    except SystemExit:
        pass
    else:
        raise AssertionError("a non-default fastest p=1 config must stop the derivation")


def test_ranges_and_macro_names():
    assert m5_numbers.rng([1.71, 2.26]) == "1.7--2.3"
    assert m5_numbers.rng([2.0, 2.04]) == "2.0"
    m, _, _ = m5_numbers.derive()
    assert all(k.isalpha() for k in m)          # valid LaTeX control words
    assert m["MExtAgree"] == "189" and m["MExtTotal"] == "192"
