"""Temporal Ligra adapter pieces that do not run the third-party binary."""
from __future__ import annotations

import os
import stat
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from experiments import temporal_ligra as tl  # noqa: E402


def test_graph_file_is_csr_with_durations(tmp_path):
    p = tmp_path / "g.adj"
    # (t, u, v): two edges from 0, one from 2; vertex 1 has none
    tl.write_graph(p, 3, [5, 3, 9], [0, 0, 2], [2, 1, 0], duration=1)
    w = p.read_text().split()
    assert w[:3] == ["WeightedTemporalAdjacencyGraph", "3", "3"]
    n, m = 3, 3
    offsets = w[3:3 + n]
    targets = w[3 + n:3 + n + m]
    weights = w[3 + n + m:3 + n + 2 * m]
    start = w[3 + n + 2 * m:3 + n + 3 * m]
    end = w[3 + n + 3 * m:3 + n + 4 * m]
    assert offsets == ["0", "2", "2"]
    assert targets == ["1", "2", "0"] and weights == ["1"] * 3
    assert start == ["3", "5", "9"] and end == ["4", "6", "10"]
    assert len(w) == 3 + n + 4 * m


def test_patch_inserts_print_once(tmp_path):
    app = tmp_path / "apps" / "temporal"
    app.mkdir(parents=True)
    (app / "EarliestArrival.C").write_text("int x;\n" + tl.PATCH_ANCHOR + "\n}\n")
    assert tl.patch(tmp_path) is True
    assert tl.patch(tmp_path) is False          # idempotent
    text = (app / "EarliestArrival.C").read_text()
    assert text.count(tl.PATCH_MARK) == 1 and text.index(tl.PATCH_MARK) < text.index("free(Visited)")


def test_run_source_parses_patched_output(tmp_path):
    fake = tmp_path / "fake_ea"
    fake.write_text("#!/usr/bin/env python3\nimport os\n"
                    "print('num_vertices=4')\nprint('runtime : 0.00125')\n"
                    "print('WORKERS', os.environ.get('OMP_NUM_THREADS', '1'))\n"
                    "print('REACH 0 2 3')\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    reach, workers, runtime = tl.run_source(str(fake), tmp_path / "g", 1, 0, threads=4)
    assert reach == {0, 1, 2, 3} and workers == 4 and abs(runtime - 0.00125) < 1e-12


def test_duration_maps_to_semantics_on_toys():
    """Duration 1 must give strict and duration 0 closure on the same-time chain
    (checked here with STRATA itself; the binary is checked on the M5)."""
    raw = [(1, 0, 1), (1, 1, 2), (1, 2, 3)]
    ts = np.array([t for t, _, _ in raw]); us = np.array([u for _, u, _ in raw])
    vs = np.array([v for _, _, v in raw])
    assert set(np.flatnonzero(tl.strata_reach(4, ts, us, vs, "strict")[0])) == {0, 1}
    assert set(np.flatnonzero(tl.strata_reach(4, ts, us, vs, "closure")[0])) == {0, 1, 2, 3}


def test_build_passes_absolute_paths_to_the_compiler(tmp_path, monkeypatch):
    """v13 M5 run: relative -I / -o paths broke the build inside apps/temporal."""
    src = tmp_path / "tl"
    app = src / "apps" / "temporal"
    app.mkdir(parents=True)
    (src / "ligra").mkdir()
    (src / "ligra" / "gettime.h").write_text("// header\n")
    (app / "EarliestArrival.C").write_text("int x;\n" + tl.PATCH_ANCHOR + "\n}\n")
    log = tmp_path / "argv.txt"
    cc = tmp_path / "fakecxx"
    cc.write_text("#!/usr/bin/env python3\nimport sys, os, pathlib\n"
                  f"open({str(log)!r}, 'a').write(os.getcwd() + ' ' + ' '.join(sys.argv[1:]) + chr(10))\n"
                  "if '--version' in sys.argv: print('fakecxx 1.0'); sys.exit(0)\n"
                  "inc = [a[2:] for a in sys.argv if a.startswith('-I')]\n"
                  "ok = all(os.path.isabs(i) and os.path.exists(os.path.join(i, 'gettime.h')) for i in inc)\n"
                  "out = sys.argv[sys.argv.index('-o') + 1]\n"
                  "if ok and os.path.isabs(out): pathlib.Path(out).write_text('bin')\n"
                  "sys.exit(0 if ok and os.path.isabs(out) else 1)\n")
    cc.chmod(cc.stat().st_mode | stat.S_IEXEC)
    monkeypatch.chdir(tmp_path)
    info = tl.build(os.path.relpath(src, tmp_path), compiler=str(cc))
    assert os.path.isabs(info["binary"]) and os.path.exists(info["binary"])
    assert info["mode"] == "openmp"


# ---- source batches -----------------------------------------

def test_batches_cover_every_source_once():
    for n, p in ((10, 3), (7, 7), (5, 8), (1899, 18)):
        parts = tl.batches(n, p)
        flat = sorted(s for b in parts for s in b)
        assert flat == list(range(n)) and len(parts) == p


def test_run_batches_runs_every_source_with_one_thread(tmp_path):
    fake = tmp_path / "fake_ea"
    fake.write_text("#!/usr/bin/env python3\nimport os, sys\n"
                    "s = int(sys.argv[2])\n"
                    "print('runtime : 0.001')\n"
                    "print('WORKERS', os.environ.get('OMP_NUM_THREADS', '?'))\n"
                    "print('REACH', s, (s + 1) % 9)\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    wall, results = tl.run_batches(str(fake), tmp_path / "g", 0, 9, 4)
    seen = {}
    for batch in results:
        for s_, out in batch:
            reach, workers, rt = tl.parse_output(out, s_)
            seen[s_] = (reach, workers, rt)
    assert sorted(seen) == list(range(9)) and wall > 0
    assert all(w == 1 and r == {s, (s + 1) % 9} for s, (r, w, _rt) in seen.items())
