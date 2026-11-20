"""Random row-update benchmark: bookkeeping and invariance of the word checksum."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def test_checksum_does_not_depend_on_chunking():
    pytest.importorskip("numba")
    from experiments.random_rows_bench import _rows_kernel, random_pairs
    n, W = 300, 5
    us, vs = random_pairs(n, 4000, seed=3)
    sums = [int(_rows_kernel(np.int64(n), np.int64(W), us, vs, np.int64(c)).sum(dtype=np.uint64))
            for c in (1, 2, 5)]
    assert len(set(sums)) == 1


def test_matches_packed_reference():
    pytest.importorskip("numba")
    from experiments.random_rows_bench import _rows_kernel
    n, W = 130, 3                                    # sources 0..129 in 3 words
    us = np.array([0, 1, 2, 5], dtype=np.int64)
    vs = np.array([1, 2, 3, 0], dtype=np.int64)
    R = np.zeros((n, W), dtype=np.uint64)
    for s in range(n):
        R[s, s // 64] = np.uint64(1) << np.uint64(s % 64)
    for u, v in zip(us, vs):                         # in-order packed propagation
        R[v] |= R[u]
    want = int(R.sum(dtype=np.uint64))
    assert int(_rows_kernel(np.int64(n), np.int64(W), us, vs, np.int64(2)).sum(dtype=np.uint64)) == want


def test_summary_and_bytes():
    from experiments.random_rows_bench import summarize, touched_bytes
    assert touched_bytes(10, 4) == 3 * 8 * 4 * 10
    rows = [{"order": "random", "threads": p, "seconds_median": t}
            for p, t in [(1, 4.0), (6, 1.0), (18, 2.0)]]
    s = summarize(rows)["random"]
    assert s["best_threads"] == 6 and s["largest_over_best"] == pytest.approx(2.0)


def test_end_to_end_small_graph(tmp_path):
    pytest.importorskip("numba")
    from experiments.bench_common import DATA
    if not (DATA / "CollegeMsg.txt").exists():
        pytest.skip("CollegeMsg not present")
    from experiments import random_rows_bench as rb
    out = tmp_path / "rr.json"
    rb.main(["--dataset", "CollegeMsg", "--words", "8", "--edges", "5000", "--threads", "1,2",
             "--repeats", "1", "--array-mib", "8", "--out", str(out)])
    rec = json.loads(out.read_text())
    assert {r["order"] for r in rec["rows"]} == {"stream", "random"}
    for order in ("stream", "random"):
        assert len({r["checksum"] for r in rec["rows"] if r["order"] == order}) == 1
