"""Measurement harness: references, thread records, provenance stamps. No timing claims."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from temporal_strata.parallel import _HAVE_NUMBA  # noqa: E402
from temporal_strata.packed_reachability import temporal_strata_packed  # noqa: E402
from temporal_strata.stream import arrays_to_edges, generate_temporal_arrays  # noqa: E402

needs_numba = pytest.mark.skipif(not _HAVE_NUMBA, reason="numba not installed")


@needs_numba
def test_sampled_reference_matches_interpreted_core():
    from experiments.bench_common import reference_state, sampled_reference

    n = 260
    a = generate_temporal_arrays(n, 5000, 400, seed=11)
    R, info = sampled_reference(n, a, k=30)
    ref = temporal_strata_packed(n, arrays_to_edges(*a), directed=True, same_time_mode="strict")
    assert np.array_equal(R, ref) and info["sampled_sources"] == 30
    R2, used = reference_state(n, a, "strict", "sampled")
    assert used == "sampled" and np.array_equal(R2, ref)
    with pytest.raises(ValueError):
        reference_state(n, a, "closure", "sampled")


@needs_numba
def test_closure_sweeps_and_lemma4_on_bucketed_stream():
    from experiments.closure_grouped import bucketed, sweep_study
    from temporal_strata.parallel import prepare_stream_arrays

    n = 200
    ts, us, vs = generate_temporal_arrays(n, 6000, 3000, seed=4)
    tb = bucketed(ts, 300)                     # ~10 groups: long same-time chains
    prep = prepare_stream_arrays(n, tb, us, vs, same_time_mode="closure", prep_mode="off")
    study, R = sweep_study(n, prep, chunk_counts=(2, 3))
    ref = temporal_strata_packed(n, arrays_to_edges(tb, us, vs), directed=True,
                                 same_time_mode="closure")
    assert np.array_equal(R, ref)
    assert study["sweeps_max"] >= 2
    assert all(x["chunk_sweeps_le_sequential"] and x["bit_identical"] for x in study["lemma4"])


@needs_numba
def test_graphblas_reference_verification_with_dense_fallback():
    from experiments.graphblas_reference import dense_closure_fn, strata_static_closure, verify

    rng = np.random.default_rng(2)
    n = 70
    us, vs = rng.integers(0, n, 300), rng.integers(0, n, 300)
    R = strata_static_closure(n, us, vs, 1)()
    C = dense_closure_fn(n, us, vs)()
    assert verify(n, R, C)
    C2 = C.copy()
    C2[0, n - 1] = not C2[0, n - 1]
    assert not verify(n, R, C2)


@needs_numba
def test_graphblas_reference_real_library_path():
    """the python-graphblas call path itself (skipped where it is not installed).

    Runs in a fresh interpreter, as session step 11 does. In the pytest process
    torch is already loaded (backend tests), and on macOS its bundled OpenMP
    runtime and the one bundled with suitesparse-graphblas abort the process
    when GraphBLAS initializes (v10 M5 run, exit -6 in GxB_init)."""
    import importlib.util
    if importlib.util.find_spec("graphblas") is None:      # do not import it here
        pytest.skip("python-graphblas not installed")
    code = (
        "import sys, numpy as np\n"
        f"sys.path.insert(0, {ROOT!r})\n"
        "import graphblas as gb\n"
        "from experiments.graphblas_reference import graphblas_closure_fn, strata_static_closure, verify\n"
        "rng = np.random.default_rng(5)\n"
        "n = 60\n"
        "us, vs = rng.integers(0, n, 240), rng.integers(0, n, 240)\n"
        "us = np.concatenate([us, us[:20]]); vs = np.concatenate([vs, vs[:20]])  # duplicate edges\n"
        "R = strata_static_closure(n, us, vs, 1)()\n"
        "assert verify(n, R, graphblas_closure_fn(gb, n, us, vs)())\n"
        "print('GRAPHBLAS-OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "GRAPHBLAS-OK" in r.stdout, (
        f"exit {r.returncode}\nstdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-4000:]}")


def test_machine_stamp_records_hashes(tmp_path):
    f = tmp_path / "x.json"
    f.write_text('{"a": 1}')
    out = tmp_path / "stamp.json"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "machine_stamp.py"),
                        "--label", "t", "--out", str(out), str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rec = json.loads(out.read_text())
    assert rec["files"][0]["bytes"] == 8 and len(rec["files"][0]["sha256"]) == 64
    assert "summary" in rec["machine"]


def test_fetch_rejects_unknown_names():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "fetch_real_temporal.py"),
                        "--names", "no-such-graph"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "unknown dataset" in (r.stdout + r.stderr)


@needs_numba
def test_parallel_scaling_stores_samples_and_parallel_prep(tmp_path):
    out, tex = tmp_path / "ps.json", tmp_path / "ps.tex"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "parallel_scaling.py"),
                        "--skip-real", "--skip-synthetic", "--large", "600:8000:20",
                        "--threads", "1", "--repeats", "2", "--prep-parallel", "compare",
                        "--out", str(out), "--tex-out", str(tex)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    row = json.loads(out.read_text())["results"][0]["rows"][0]
    assert len(row["kernel_samples_s"]) == 2 and row["kernel_stats"]["n"] == 2
    assert "prep_par" in row and len(row["prep_par"]["samples_s"]) == 2


def test_thread_record_and_one_thread_check():
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    from thread_record import check_one_thread, thread_record

    rec = thread_record()
    assert set(rec) == {"numba", "graphblas", "threadpools"}
    check_one_thread({"numba": 1, "graphblas": None}, "x")
    check_one_thread({"numba": 1, "graphblas": 1}, "x", need_graphblas=True)
    for bad, gbneed in (({}, False), (None, False), ({"numba": 18}, False),
                        ({"numba": 1, "graphblas": 18}, True)):
        with pytest.raises(SystemExit):
            check_one_thread(bad, "x", need_graphblas=gbneed)

