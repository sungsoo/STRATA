"""Thread-drop diagnosis: bookkeeping and an end-to-end run on a small real graph."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def test_summary_picks_best_and_ratios():
    from experiments.thread_diagnosis import summarize
    rows = [{"block_bits": 64, "threads": p, "schedule": s, "kernel_s_median": t}
            for p, s, t in [(1, "static", 4.0), (1, "dynamic", 4.4), (6, "static", 1.0),
                            (6, "dynamic", 1.1), (18, "static", 2.0), (18, "dynamic", 0.9)]]
    s = summarize(rows)["64"]
    assert (s["best_threads"], s["best_schedule"]) == (18, "dynamic")
    assert s["dynamic_over_static"]["18"] == pytest.approx(0.45)
    assert s["dynamic_over_static"]["1"] == pytest.approx(1.1)


def test_stream_bytes_scale_with_chunks():
    from types import SimpleNamespace
    from experiments.thread_diagnosis import stream_read_bytes
    a = np.zeros(10, dtype=np.int64)
    prep = SimpleNamespace(us=a, vs=a, gptr=a, src_slot=a, dirty_idx=None, dptr=a)
    assert stream_read_bytes(prep, 1) == 5 * 80
    assert stream_read_bytes(prep, 3) == 3 * 5 * 80


def test_end_to_end_small_real_graph(tmp_path):
    pytest.importorskip("numba")
    from experiments.bench_common import DATA
    if not (DATA / "CollegeMsg.txt").exists():
        pytest.skip("CollegeMsg not present")
    from experiments import thread_diagnosis as td
    out = tmp_path / "td.json"
    td.main(["--dataset", "CollegeMsg", "--block-bits", "128", "--threads", "1,2",
             "--repeats", "1", "--array-mib", "8", "--verify-sources", "8", "--out", str(out)])
    rec = json.loads(out.read_text())
    assert len(rec["rows"]) == 4
    assert len({r["pairs"] for r in rec["rows"]}) == 1          # same block every time
    assert all(r["achieved_GBps"] > 0 for r in rec["rows"])
    assert set(rec["summary"]["128"]["dynamic_over_static"]) == {"1", "2"}


def test_pairs_parse_and_summary():
    from experiments.thread_diagnosis import parse_pairs, summarize_pairs
    assert parse_pairs("6:6,6:12,18:18") == [(6, 6), (6, 12), (18, 18)]
    rows = [{"block_bits": 64, "threads": p, "chunks": c, "kernel_s_median": t}
            for p, c, t in [(6, 6, 1.0), (6, 12, 1.5), (6, 18, 2.0), (12, 18, 2.5), (18, 18, 3.0)]]
    s = summarize_pairs(rows)["64"]
    assert s["chunks_at_fixed_threads"] == {"threads": 6, "series": {"6": 1.0, "12": 1.5, "18": 2.0}}
    assert s["threads_at_fixed_chunks"] == {"chunks": 18, "series": {"6": 2.0, "12": 2.5, "18": 3.0}}
    assert s["grid"]["6"]["12"] == 1.5


def test_pairs_end_to_end_uses_explicit_chunks(tmp_path):
    pytest.importorskip("numba")
    from experiments.bench_common import DATA
    if not (DATA / "CollegeMsg.txt").exists():
        pytest.skip("CollegeMsg not present")
    from experiments import thread_diagnosis as td
    out = tmp_path / "tc.json"
    td.main(["--dataset", "CollegeMsg", "--block-bits", "1024", "--pairs", "1:1,1:4,2:4",
             "--repeats", "1", "--array-mib", "8", "--verify-sources", "8", "--out", str(out)])
    rec = json.loads(out.read_text())
    assert [(r["threads"], r["chunks"], r["schedule"]) for r in rec["rows"]] == \
        [(1, 1, "static"), (1, 4, "static"), (2, 4, "static")]
    assert len({r["pairs"] for r in rec["rows"]}) == 1
    assert rec["summary"]["1024"]["chunks_at_fixed_threads"]["threads"] == 1
