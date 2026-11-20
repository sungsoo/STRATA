"""Kernel phase variants: outputs and bookkeeping on small inputs."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def _prep(n=200, m=3000, seed=4):
    from temporal_strata.parallel import prepare_stream_arrays
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(0, 300, size=m))
    us = rng.integers(0, n, size=m)
    vs = rng.integers(0, n, size=m)
    return prepare_stream_arrays(n, ts, us, vs, directed=True, same_time_mode="strict")


def _state(n, width):
    from temporal_strata.packed_reachability import _empty_state
    from temporal_strata.parallel import _set_identity_vec
    R = _empty_state(n, width)
    _set_identity_vec(R, 0, width)
    return R


def test_prealloc_equals_strict_and_copy_is_identity():
    pytest.importorskip("numba")
    from experiments import kernel_phases as kp
    n, width = 200, 130
    prep = _prep(n)
    ref = _state(n, width)
    kp.run_variant("strict", ref, prep, 1)
    W = ref.shape[1]
    for nch in (1, 2, 3):
        R = _state(n, width)
        buf = np.zeros(n * W, dtype=np.uint64)
        kp.run_variant("prealloc", R, prep, nch, buf)
        assert np.array_equal(R, ref)
        C = _state(n, width)
        kp.run_variant("copy", C, prep, nch)
        assert np.array_equal(C, _state(n, width))


def test_nosnap_is_partition_independent_and_matches_in_order_reference():
    pytest.importorskip("numba")
    from experiments import kernel_phases as kp
    n, width = 200, 130
    prep = _prep(n)
    R1 = _state(n, width)
    kp.run_variant("nosnap", R1, prep, 1)
    R3 = _state(n, width)
    kp.run_variant("nosnap", R3, prep, 3)
    assert np.array_equal(R1, R3)
    ref = _state(n, width)
    for u, v in zip(prep.us, prep.vs):          # every edge reads the live row
        ref[v] |= ref[u]
    assert np.array_equal(R1, ref)


def test_measure_end_to_end_and_summary():
    pytest.importorskip("numba")
    from experiments import kernel_phases as kp
    from temporal_strata.parallel import set_threads
    n, width = 200, 130
    prep = _prep(n)
    rows, ref = kp.measure(n, width, prep, [1, 2], 1, set_threads)
    assert {r["variant"] for r in rows} == set(kp.VARIANTS) and {"strict", "nosnap"} <= set(ref)
    s = kp.summarize(rows)
    assert s["strict"]["ref_threads"] == 2 and set(s["copy"]["seconds"]) == {"1", "2"}


def test_summary_uses_six_threads_when_present():
    from experiments.kernel_phases import summarize
    rows = [{"variant": "copy", "threads": p, "seconds_median": t}
            for p, t in [(1, 4.0), (6, 1.0), (18, 1.5)]]
    assert summarize(rows)["copy"]["pmax_over_ref"] == pytest.approx(1.5)


# ---- v28: copy-phase variants and tiles as the state -------------------------
def test_aligned_bounds():
    from experiments.kernel_phases import aligned_bounds
    b = aligned_bounds(1024, 6)
    assert b[0] == 0 and b[-1] == 1024 and all(x % 16 == 0 for x in b)
    assert list(aligned_bounds(10, 3)) == [0, 0, 0, 10]          # narrower than a line
    assert all(np.diff(aligned_bounds(1024, 18)) >= 0)


def test_copy_variants_keep_the_state_and_tiles_match_strict():
    pytest.importorskip("numba")
    from experiments import kernel_phases as kp
    from experiments.bench_common import _bfs_sources_strict
    from temporal_strata.parallel import set_threads
    n, width = 200, 130
    prep = _prep(n)
    src = np.array([0, 5, 129])
    bfs = _bfs_sources_strict(n, prep.gptr, prep.us, prep.vs, src)
    rows, ref = kp.measure(n, width, prep, [1, 2], 1, set_threads, src, bfs, kp.COPY_VARIANTS)
    assert {r["variant"] for r in rows} == set(kp.COPY_VARIANTS)
    assert ref["tiles_pairs"] == ref["strict_pairs"]


def test_prefault_subset_keeps_the_state():
    """v30: the copy / copy_prefault / copy_reuse subset (session step 26) runs
    without the strict variant, and the prefault write leaves the state unchanged."""
    pytest.importorskip("numba")
    from experiments import kernel_phases as kp
    from temporal_strata.parallel import set_threads
    n, width = 200, 130
    prep = _prep(n)
    R = _state(n, width)
    before = kp.checksum(R)
    kp._touch2d(R, np.uint64(0))
    assert kp.checksum(R) == before
    rows, _ = kp.measure(n, width, prep, [1, 2], 1, set_threads,
                         variants=("copy", "copy_prefault", "copy_reuse"))
    assert {r["variant"] for r in rows} == {"copy", "copy_prefault", "copy_reuse"}


def test_tile_state_unblocked_variants_agree():
    """v34: shared / pre-written / tile-state unblocked runs give one relation."""
    pytest.importorskip("numba")
    from experiments import tile_state_unblocked as tsu
    from temporal_strata.parallel import set_threads
    from temporal_strata.stream import generate_temporal_arrays
    n = 260
    t, u, v = generate_temporal_arrays(n, 3000, 60, seed=3)
    rec = tsu.measure("tiny", n, (t, u, v), [1, 2], 1, set_threads, n_samples=5)
    assert {r["variant"] for r in rec["rows"]} == set(tsu.VARIANTS)


def test_op_counts_formula():
    """v34: step counts follow the two kernels' loops."""
    from experiments.op_counts import counts
    from temporal_strata.stream import generate_temporal_arrays
    n = 200
    t, u, v = generate_temporal_arrays(n, 2000, 40, seed=2)
    r = counts(n, (t, u, v), pairs=1000)
    assert r["bfs_steps"] == n * r["m"] + 2 * n * n + 1000
    assert r["strata_word_ops"] == r["W"] * (r["m"] + r["snapshot_rows"])
