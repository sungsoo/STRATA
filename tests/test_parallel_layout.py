"""STRATA-v2 realization choices: layout / schedule / chunk plan.

Every test here is machine-independent. The determinacy tests check that the
tiled and auto layouts and the dynamic schedule return the sequential result
word-for-word (Theorem 3); the plan tests check the documented layout and
chunk rules with a forced cache-line size, so they do not depend on the host.

Run with an enlarged pool to exercise genuinely multi-threaded schedules::

    NUMBA_NUM_THREADS=8 python3 -m pytest tests/test_parallel_layout.py -q
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.packed_reachability import (  # noqa: E402
    reachable_pairs,
    temporal_strata_packed,
    words_for,
)
from temporal_strata.stream import generate_temporal_stream  # noqa: E402

numba = pytest.importorskip("numba")
from temporal_strata import parallel as par  # noqa: E402
from temporal_strata.parallel import (  # noqa: E402
    LAYOUTS,
    SCHEDULES,
    machine_info,
    max_threads,
    prepare_stream,
    resolve_plan,
    run_prepared,
    set_threads,
    temporal_strata_parallel,
)

REAL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "datasets_real_temporal", "standard", "CollegeMsg.txt",
)


# --------------------------------------------------------------------------- #
# Determinacy across layout x schedule x semantics x direction x chunks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("mode", ["strict", "closure"])
@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_layout_schedule_determinacy(layout, schedule, mode, directed):
    n, m = 300, 2200  # W = 5 words: chunk counts cover 1, non-divisors, W, > W
    edges = generate_temporal_stream(n, m, 50, directed=directed, seed=31)
    ref = temporal_strata_packed(n, edges, directed=directed, same_time_mode=mode)
    for chunks in (None, 1, 2, 3, 5, 9):
        got = temporal_strata_parallel(
            n, edges, directed=directed, same_time_mode=mode,
            layout=layout, schedule=schedule, chunks=chunks,
        )
        assert np.array_equal(ref, got), f"{layout}/{schedule}/chunks={chunks} diverged"


@pytest.mark.parametrize("layout", ["tiled", "auto"])
@pytest.mark.parametrize("block_bits", [64, 50, 200])
def test_blocked_layouts_match_sequential(layout, block_bits):
    n, m = 260, 1800
    edges = generate_temporal_stream(n, m, 40, directed=True, seed=13)
    ref = temporal_strata_packed(n, edges, directed=True, same_time_mode="strict")
    plans = []
    got = temporal_strata_parallel(
        n, edges, directed=True, same_time_mode="strict", block_bits=block_bits,
        layout=layout, schedule="dynamic", plans_out=plans,
    )
    assert np.array_equal(ref, got)
    assert len(plans) == -(-n // block_bits)  # one plan per source block
    assert all(p.words <= words_for(block_bits) for p in plans)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_thread_counts_dynamic(layout):
    n, m = 200, 1500
    edges = generate_temporal_stream(n, m, 60, directed=True, seed=29)
    ref = temporal_strata_packed(n, edges, directed=True, same_time_mode="strict")
    for k in sorted({1, 2, max_threads()}):
        eff = set_threads(k)
        got = temporal_strata_parallel(n, edges, num_threads=eff, layout=layout,
                                       schedule="dynamic")
        assert np.array_equal(ref, got), f"threads={eff} {layout} diverged"


def test_same_timestamp_chain_tiled():
    n = 5
    edges = [(1, 0, 1), (1, 1, 2), (1, 2, 3), (2, 3, 4)]
    for mode in ("strict", "closure"):
        ref = temporal_strata_packed(n, edges, same_time_mode=mode)
        got = temporal_strata_parallel(n, edges, same_time_mode=mode, layout="tiled")
        assert np.array_equal(ref, got)


def test_prepared_stream_reuse_across_plans():
    """One shared O(m) preprocessing serves every plan (harness usage)."""
    n, m = 190, 1300
    edges = generate_temporal_stream(n, m, 30, directed=True, seed=3)
    ref = temporal_strata_packed(n, edges, directed=True, same_time_mode="strict")
    prep = prepare_stream(n, edges, directed=True, same_time_mode="strict")
    W = words_for(n)
    assert prep.num_edges == len(edges)
    assert prep.dirty_total >= 1
    for layout in ("rowmajor", "tiled"):
        for schedule in SCHEDULES:
            plan = resolve_plan(W, 2, layout=layout, schedule=schedule)
            R = np.zeros((n, W), dtype=np.uint64)
            par._set_identity_vec(R, 0, n)
            run_prepared(R, prep, plan)
            assert np.array_equal(ref, R)


@pytest.mark.skipif(not os.path.exists(REAL), reason="real dataset not present")
def test_real_collegemsg_tiled_dynamic_pairs():
    from temporal_strata.io import load_standard_edges

    n, edges, _meta = load_standard_edges(REAL)
    R = temporal_strata_parallel(n, edges, layout="tiled", schedule="dynamic")
    assert reachable_pairs(R, n) == 1_794_244


# --------------------------------------------------------------------------- #
# Plan rules (forced cache-line size: host-independent)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "W,L,p,expected",
    [
        (30, 8, 4, "tiled"),       # floor(30/8)=3 < 4
        (30, 8, 3, "rowmajor"),    # 3 >= 3 (boundary)
        (30, 16, 2, "tiled"),      # floor(30/16)=1 < 2 (128-byte lines)
        (92, 16, 6, "tiled"),      # floor(92/16)=5 < 6
        (125, 16, 6, "rowmajor"),  # floor(125/16)=7 >= 6
        (1, 8, 1, "tiled"),        # floor(1/8)=0 < 1
    ],
)
def test_auto_layout_rule(W, L, p, expected):
    plan = resolve_plan(W, p, layout="auto", line_words=L)
    assert plan.resolved_layout == expected
    assert plan.line_words == L and plan.words == W and plan.threads == p
    assert "auto: floor(W/L)" in plan.rule


def test_chunk_rules():
    # row-major static: min(p, W, floor(W/L))
    assert resolve_plan(30, 8, layout="rowmajor", line_words=8).nchunks == 3
    assert resolve_plan(125, 6, layout="rowmajor", line_words=16).nchunks == 6
    # row-major dynamic is still capped at one cache line per chunk
    assert resolve_plan(125, 6, layout="rowmajor", schedule="dynamic",
                        line_words=16).nchunks == 7
    # tiled: no cache-line cap; dynamic = oversub * p, capped at W
    assert resolve_plan(30, 8, layout="tiled", line_words=8).nchunks == 8
    assert resolve_plan(125, 6, layout="tiled", schedule="dynamic",
                        line_words=16).nchunks == 24
    assert resolve_plan(30, 18, layout="tiled", schedule="dynamic",
                        line_words=16).nchunks == 30
    assert resolve_plan(125, 6, layout="tiled", schedule="dynamic", oversub=2,
                        line_words=16).nchunks == 12
    # explicit chunks override, clamped to [1, W]
    assert resolve_plan(30, 8, chunks=100, line_words=8).nchunks == 30
    assert resolve_plan(30, 8, chunks=0, line_words=8).nchunks == 1


def test_plan_validation_and_dict():
    with pytest.raises(ValueError):
        resolve_plan(10, 2, layout="columnmajor")
    with pytest.raises(ValueError):
        resolve_plan(10, 2, schedule="guided")
    with pytest.raises(ValueError):
        temporal_strata_parallel(4, [(0, 0, 1)], layout="bad")
    d = resolve_plan(10, 2, layout="auto", line_words=8).as_dict()
    assert set(d) >= {"layout", "resolved_layout", "schedule", "threads", "words",
                      "line_words", "nchunks", "oversub", "rule"}
    json.dumps(d)


def test_cache_line_detection(monkeypatch):
    par.cache_line_bytes.cache_clear()
    monkeypatch.setenv("STRATA_CACHELINE_BYTES", "128")
    assert par.cache_line_bytes() == 128 and par.cache_line_words() == 16
    par.cache_line_bytes.cache_clear()
    monkeypatch.setenv("STRATA_CACHELINE_BYTES", "garbage")
    assert par.cache_line_bytes() >= 8
    monkeypatch.delenv("STRATA_CACHELINE_BYTES")
    par.cache_line_bytes.cache_clear()
    assert par.cache_line_bytes() in (32, 64, 128, 256)


def test_machine_info_records_required_fields():
    info = machine_info()
    for key in ("platform", "cpu_count", "cache_line_bytes", "python", "numpy",
                "numba", "numba_max_threads", "summary"):
        assert key in info, key
    json.dumps(info)


# --------------------------------------------------------------------------- #
# R5 byte-traffic model (exact counts, no timing)
# --------------------------------------------------------------------------- #
def test_strict_traffic_model_counts():
    from temporal_strata.traffic import strict_kernel_bytes, triad_bytes

    b = strict_kernel_bytes(num_edges=10, dirty_total=4, n=7, words=3)
    assert b == {"edge_bytes": 720, "snapshot_bytes": 192, "tile_bytes": 0,
                 "total_bytes": 912}
    t = strict_kernel_bytes(num_edges=10, dirty_total=4, n=7, words=3, layout="tiled")
    assert t["tile_bytes"] == 4 * 8 * 7 * 3 and t["total_bytes"] == 912 + 672
    with pytest.raises(ValueError):
        strict_kernel_bytes(1, 1, 1, 1, layout="auto")
    assert triad_bytes(1000) == 24000 and triad_bytes(1000, write_allocate=True) == 32000


def test_traffic_model_on_prepared_stream():
    from temporal_strata.traffic import strict_kernel_bytes

    n, m = 150, 900
    edges = generate_temporal_stream(n, m, 25, directed=False, seed=8)
    prep = prepare_stream(n, edges, directed=False, same_time_mode="strict")
    assert prep.num_edges == 2 * m  # undirected expansion
    W = words_for(n)
    b = strict_kernel_bytes(prep.num_edges, prep.dirty_total, n, W)
    assert b["edge_bytes"] == 24 * W * 2 * m
    assert b["snapshot_bytes"] == 16 * W * prep.dirty_total
