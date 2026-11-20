"""Within-machine generator for a second machine (v35)."""
import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def test_server_numbers_emits_only_ratios(tmp_path):
    from experiments import server_numbers as sn
    real = os.path.join(HERE, "..", "results", "real")
    d = tmp_path / "srv"
    d.mkdir()
    for f in ("parallel_scaling_large.json", "tile_state_unblocked.json", "kernel_prefault.json",
              "large_real_tiles_b32768.json"):
        if os.path.exists(os.path.join(real, f)):
            shutil.copy(os.path.join(real, f), d / f)
    out = tmp_path / "s.tex"
    m = sn.main(["--tag", "srv", "--root", str(tmp_path), "--out", str(out)])
    assert m and all(k.startswith("MSv") for k in m)
    assert not any(k.endswith(("S", "Ms", "Sec", "GB")) for k in m)
    text = out.read_text()
    assert "\\newcommand{\\MSvMachine}" in text


def test_server_numbers_rejects_a_time_macro():
    from experiments import server_numbers as sn
    with pytest.raises(SystemExit):
        sn._check({"MSvFullS": "257"})
