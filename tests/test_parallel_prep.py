"""Parallel preprocessing equals the sequential paths."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from temporal_strata.parallel import (  # noqa: E402
    _HAVE_NUMBA,
    _group_ptr,
    _strict_snapshot_plan,
    prepare_stream_arrays,
    set_threads,
    temporal_strata_parallel,
)
from temporal_strata.parallel_prep import (  # noqa: E402
    group_blocks,
    parallel_sort_keys,
    resolve_prep,
    sort_validate_arrays_parallel,
    strict_snapshot_plan_parallel,
)
from temporal_strata.stream import generate_temporal_arrays, sort_validate_arrays  # noqa: E402

pytestmark = pytest.mark.skipif(not _HAVE_NUMBA, reason="numba not installed")
BLOCKS = (1, 2, 6, 18)


def _streams():
    rng = np.random.default_rng(3)
    out = []
    for n, m, T in ((50, 600, 40), (200, 5000, 7), (300, 4000, 4000), (64, 3000, 1)):
        out.append((n, rng.integers(0, T, m), rng.integers(0, n, m), rng.integers(0, n, m)))
    out.append((10, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)))
    n = 120
    ts, us, vs = generate_temporal_arrays(n, 2500, 30, seed=5)
    out.append((n, ts, us, vs))
    return out


@pytest.mark.parametrize("blocks", BLOCKS)
def test_parallel_sort_matches_sequential(blocks):
    for n, t, u, v in _streams():
        a = sort_validate_arrays(n, t, u, v)
        b = sort_validate_arrays_parallel(n, t, u, v, blocks)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)


def test_parallel_sort_keys_and_validation():
    rng = np.random.default_rng(1)
    k = rng.integers(-10**12, 10**12, 10001)
    for c in (1, 2, 3, 7, 16):
        assert np.array_equal(parallel_sort_keys(k, c), np.sort(k))
    with pytest.raises(ValueError, match="out of range"):
        sort_validate_arrays_parallel(5, np.array([1, 2]), np.array([0, 5]), np.array([1, 1]), 4)
    # float timestamps fall back to the sequential path (same truncation)
    a = sort_validate_arrays(6, np.array([2.5, 1.0]), np.array([0, 1]), np.array([1, 2]))
    b = sort_validate_arrays_parallel(6, np.array([2.5, 1.0]), np.array([0, 1]), np.array([1, 2]), 4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("blocks", BLOCKS)
def test_parallel_plan_matches_sequential(blocks):
    for n, t, u, v in _streams():
        ts, us, vs = sort_validate_arrays(n, t, u, v)
        g = _group_ptr(ts)
        a = _strict_snapshot_plan(g, us, vs)
        b = strict_snapshot_plan_parallel(g, us, vs, blocks)
        for x, y in zip(a[:3], b[:3]):
            assert np.array_equal(x, y)
        assert a[3] == b[3]


def test_group_blocks_cover_all_groups():
    g = np.array([0, 5, 6, 30, 31, 32, 60])
    for k in (1, 2, 3, 6, 50):
        b = group_blocks(g, k)
        assert b[0] == 0 and b[-1] == 6 and np.all(np.diff(b) > 0)


@pytest.mark.parametrize("threads", (1, 2, 6, 18))
def test_prepare_modes_identical_and_results_bit_identical(threads):
    eff = set_threads(threads)
    n = 150
    ts, us, vs = generate_temporal_arrays(n, 4000, 25, seed=9)
    for directed in (True, False):
        base = prepare_stream_arrays(n, ts, us, vs, directed=directed, prep_mode="off")
        par = prepare_stream_arrays(n, ts, us, vs, directed=directed, prep_mode="on",
                                    prep_threads=threads)
        assert par.prep_parallel and not base.prep_parallel
        for name in ("gptr", "us", "vs", "dirty_idx", "dptr", "src_slot"):
            assert np.array_equal(getattr(base, name), getattr(par, name)), name
        assert base.max_dirty == par.max_dirty
    R1 = temporal_strata_parallel(n, (ts, us, vs), num_threads=eff)
    R2 = temporal_strata_parallel(n, (ts, us, vs), num_threads=1)
    assert np.array_equal(R1, R2)


def test_auto_mode_threshold():
    assert resolve_prep("auto", 10, 18) is False
    assert resolve_prep("auto", 1 << 20, 1) is False
    assert resolve_prep("auto", 1 << 20, 18) is True
    assert resolve_prep("off", 1 << 20, 18) is False
    with pytest.raises(ValueError):
        resolve_prep("fast", 1, 1)


# ---- wide timestamp ranges --------------------------------

def _wide_stream(seed, m, n, presorted):
    rng = np.random.default_rng(seed)
    t = rng.integers(1_200_000_000, 1_460_000_000, m)      # seconds over ~8 years
    if presorted:
        t = np.sort(t)
    t[3:40] = t[3]                                           # timestamp ties
    u = rng.integers(0, n, m)
    v = rng.integers(0, n, m)
    u[7:15], v[7:15] = u[7], v[7]                            # duplicate edges
    return t, u, v


@pytest.mark.parametrize("presorted", [True, False])
def test_wide_span_sort_matches_lexsort(presorted):
    from temporal_strata.parallel_prep import sort_validate_arrays_parallel
    from temporal_strata.stream import sort_validate_arrays

    n, m = 150_000, 50_000                    # span * n^2 > 2^62: the packed t key overflows
    t, u, v = _wide_stream(11, m, n, presorted)
    assert (int(t.max()) - int(t.min()) + 1) * n * n >= 2**62
    o = np.lexsort((v, u, t))
    ref = (t[o], u[o], v[o])
    for got in (sort_validate_arrays(n, t, u, v), sort_validate_arrays_parallel(n, t, u, v, 4)):
        for a, b in zip(ref, got):
            assert np.array_equal(a, b)


def test_dense_time_rank_keeps_order_and_ties():
    from temporal_strata.stream import dense_time_rank

    t = np.array([5, 9, 9, 2, 5, 100])
    r = dense_time_rank(t)
    assert r.tolist() == [1, 2, 2, 0, 1, 3]
    s = np.sort(t)
    assert dense_time_rank(s).tolist() == [0, 1, 1, 2, 2, 3]
    assert dense_time_rank(np.array([], dtype=np.int64)).size == 0


def test_wide_span_falls_back_when_rank_key_overflows():
    from temporal_strata.parallel_prep import sort_validate_arrays_parallel
    from temporal_strata.stream import sort_validate_arrays

    n = 3_000_000_000 // 1000                 # n^2 * m still >= 2^62 after rank compression
    m = 600_000
    rng = np.random.default_rng(3)
    t = np.arange(m, dtype=np.int64) * 10_000_000_000   # every timestamp distinct, huge span
    u = rng.integers(0, n, m)
    v = rng.integers(0, n, m)
    assert m * n * n >= 2**62
    o = np.lexsort((v, u, t))
    for got in (sort_validate_arrays(n, t, u, v), sort_validate_arrays_parallel(n, t, u, v, 4)):
        assert np.array_equal(got[0], t[o]) and np.array_equal(got[1], u[o]) and np.array_equal(got[2], v[o])


# ---- nearly sorted input takes the adaptive sort ----------

def _perturbed(seed, m, n, frac, wide):
    rng = np.random.default_rng(seed)
    base = 1_200_000_000 if wide else 0
    span = 260_000_000 if wide else 5_000
    t = np.sort(rng.integers(base, base + span, m))
    if frac >= 0.5:
        t = rng.permutation(t)                                 # not a time log
    else:
        k = int(frac * m)
        i = rng.integers(0, m - 1, k)
        t[i], t[i + 1] = t[i + 1].copy(), t[i].copy()        # local descents
    return t, rng.integers(0, n, m), rng.integers(0, n, m)


@pytest.mark.parametrize("wide", [True, False])
@pytest.mark.parametrize("frac,path", [(0.005, "adaptive"), (0.5, "parallel")])
def test_sort_path_by_presortedness_and_identical_arrays(wide, frac, path):
    from temporal_strata.parallel_prep import sort_validate_arrays_parallel
    from temporal_strata.stream import sort_validate_arrays

    n = 150_000 if wide else 300
    m = 40_000
    t, u, v = _perturbed(21, m, n, frac, wide)
    o = np.lexsort((v, u, t))
    info = {}
    got = sort_validate_arrays_parallel(n, t, u, v, 4, info=info)
    assert info["sort_path"] == path
    for a, b in zip((t[o], u[o], v[o]), got):
        assert np.array_equal(a, b)
    for a, b in zip((t[o], u[o], v[o]), sort_validate_arrays(n, t, u, v)):
        assert np.array_equal(a, b)


def test_nearly_sorted_threshold():
    from temporal_strata.parallel_prep import NEAR_SORTED_DIV, nearly_sorted

    a = np.arange(3200)
    assert nearly_sorted(a)
    b = a.copy()
    b[1:3200 // NEAR_SORTED_DIV * 2:2] = 0          # 100 descents in 3200 -> exactly m/32
    assert np.count_nonzero(b[1:] < b[:-1]) * NEAR_SORTED_DIV <= b.size
    assert nearly_sorted(b)
    assert not nearly_sorted(np.arange(3200)[::-1])


def test_prepared_stream_records_sort_path():
    t, u, v = _perturbed(4, 200_000, 2_000, 0.002, False)
    prep = prepare_stream_arrays(2_000, t, u, v, directed=True, same_time_mode="strict",
                                 prep_mode="on", prep_threads=4)
    assert prep.prep_parallel and prep.prep_sort == "adaptive"
    seq = prepare_stream_arrays(2_000, t, u, v, directed=True, same_time_mode="strict", prep_mode="off")
    assert seq.prep_sort == "sequential"
    assert np.array_equal(prep.us, seq.us) and np.array_equal(prep.gptr, seq.gptr)


# ---- v22: no packed key fits -> stable time sort + per-group sort ------------

def test_time_then_group_equals_lexsort():
    """sx-stackoverflow needs 70 key bits (2 x 22 for vertices, 26 for time ranks)."""
    from temporal_strata.parallel_prep import sort_validate_arrays_parallel
    from temporal_strata.stream import sort_validate_arrays, time_then_group_order

    rng = np.random.default_rng(3)
    n, m = 3_000_000, 700_000
    t = rng.integers(0, 2**40, m)
    t[100:400] = t[99]                      # one large group
    t[1000:1003] = t[999]                   # and a small one
    u = rng.integers(0, n, m)
    v = rng.integers(0, n, m)
    u[200:210] = u[199]                     # duplicate edges inside a group
    v[200:210] = v[199]
    ref = np.lexsort((v, u, t))
    assert np.array_equal(time_then_group_order(t, u, v), ref)
    for fn in (lambda info: sort_validate_arrays(n, t, u, v, info=info),
               lambda info: sort_validate_arrays_parallel(n, t, u, v, 4, info=info)):
        info = {}
        out = fn(info)
        assert info["sort_path"] == "time-then-group"
        assert all(np.array_equal(a, b) for a, b in zip(out, (t[ref], u[ref], v[ref])))
