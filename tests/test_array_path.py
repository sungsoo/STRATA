"""Array-native stream path equals the object path.

Loading, validation, sorting, flattening and the touched-row plan can now run
on three int64 arrays without TemporalEdge objects. Every output must equal
the object path array for array, and results must stay bit-identical.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.io import load_standard_arrays, load_standard_edges  # noqa: E402
from temporal_strata.packed_reachability import temporal_strata_packed  # noqa: E402
from temporal_strata.stream import (  # noqa: E402
    TemporalEdge,
    arrays_to_edges,
    edges_to_arrays,
    generate_temporal_arrays,
    generate_temporal_stream,
    sort_and_validate_edges,
    sort_validate_arrays,
)

numba = pytest.importorskip("numba")
from temporal_strata import parallel as par  # noqa: E402

REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "datasets_real_temporal", "standard")
FIELDS = ("gptr", "us", "vs", "dirty_idx", "dptr", "src_slot")


def _same_prep(a, b):
    for f in FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        if x is None or y is None:
            assert x is None and y is None
            continue
        assert x.dtype == y.dtype == np.int64, f
        assert np.array_equal(x, y), f
    assert a.max_dirty == b.max_dirty


@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("mode", ["strict", "closure"])
@pytest.mark.parametrize("n,m,ts,seed", [(2, 40, 3, 1), (130, 900, 40, 11),
                                         (300, 2200, 2, 31), (500, 3000, 3000, 7)])
def test_prepare_arrays_equals_object_path(n, m, ts, seed, mode, directed):
    edges = generate_temporal_stream(n, m, ts, directed=directed, seed=seed)
    obj = par.prepare_stream(n, edges, directed=directed, same_time_mode=mode)
    arr = par.prepare_stream_arrays(n, *edges_to_arrays(edges), directed=directed,
                                    same_time_mode=mode)
    _same_prep(obj, arr)
    assert set(arr.timings) == set(obj.timings)


def test_unsorted_and_duplicate_input_sorts_like_object_path():
    edges = [(5, 1, 2), (1, 3, 0), (5, 0, 2), (1, 3, 0), (2, 2, 2), (5, 1, 1)]
    ref = sort_and_validate_edges(4, edges)
    ts, us, vs = sort_validate_arrays(4, *edges_to_arrays(edges))
    assert [(e.t, e.u, e.v) for e in ref] == list(zip(ts.tolist(), us.tolist(), vs.tolist()))


def test_float_timestamps_match_object_path():
    edges = [TemporalEdge(1.7, 0, 1), TemporalEdge(1.2, 2, 0), TemporalEdge(0.5, 1, 2)]
    obj = par.prepare_stream(3, edges)
    arr = par.prepare_stream_arrays(3, *edges_to_arrays(edges))
    _same_prep(obj, arr)
    tuples = [(1.7, 0, 1), (1.2, 2, 0)]
    a = sort_and_validate_edges(3, tuples)
    b = sort_validate_arrays(3, *edges_to_arrays(tuples))
    assert [(e.t, e.u, e.v) for e in a] == list(zip(*(x.tolist() for x in b)))


def test_validation_errors_match():
    for bad in ([(0, 0, 5)], [(0, -1, 1)]):
        with pytest.raises(ValueError) as e1:
            sort_and_validate_edges(5, bad)
        with pytest.raises(ValueError) as e2:
            sort_validate_arrays(5, *edges_to_arrays(bad))
        assert str(e1.value) == str(e2.value)


def test_empty_stream():
    obj = par.prepare_stream(4, [])
    arr = par.prepare_stream_arrays(4, *edges_to_arrays([]))
    _same_prep(obj, arr)


@pytest.mark.parametrize("n,m,ts,directed,seed", [(130, 900, 40, True, 11),
                                                  (300, 2200, 50, False, 31),
                                                  (8000, 3000, 200, True, 7)])
def test_generator_arrays_identical(n, m, ts, directed, seed):
    edges = generate_temporal_stream(n, m, ts, directed=directed, seed=seed)
    a = generate_temporal_arrays(n, m, ts, directed=directed, seed=seed)
    assert [(e.t, e.u, e.v) for e in edges] == list(zip(*(x.tolist() for x in a)))
    assert arrays_to_edges(*a) == edges


@pytest.mark.skipif(not os.path.isdir(REAL), reason="real datasets not present")
@pytest.mark.parametrize("name", ["CollegeMsg", "BitcoinOTC", "BitcoinOTCpos"])
def test_loader_and_prep_on_real_data(name):
    path = os.path.join(REAL, f"{name}.txt")
    n0, edges, meta0 = load_standard_edges(path)
    n1, ts, us, vs, meta1 = load_standard_arrays(path)
    assert n0 == n1 and meta0 == meta1
    assert [(e.t, e.u, e.v) for e in edges] == list(zip(ts.tolist(), us.tolist(), vs.tolist()))
    _same_prep(par.prepare_stream(n0, edges), par.prepare_stream_arrays(n1, ts, us, vs))


@pytest.mark.parametrize("directed", [True, False])
def test_parallel_accepts_arrays_bit_identical(directed):
    n, m = 260, 2000
    a = generate_temporal_arrays(n, m, 30, directed=directed, seed=21)
    edges = arrays_to_edges(*a)
    for mode in ("strict", "closure"):
        ref = temporal_strata_packed(n, edges, directed=directed, same_time_mode=mode)
        for layout in ("rowmajor", "tiled"):
            got = par.temporal_strata_parallel(n, a, directed=directed, same_time_mode=mode,
                                               layout=layout)
            assert np.array_equal(ref, got)


def test_default_layout_is_tiled():
    plans = []
    par.temporal_strata_parallel(130, generate_temporal_arrays(130, 500, 20), plans_out=plans)
    assert plans[0].resolved_layout == "tiled"
    assert par.resolve_plan(30, 4).resolved_layout == "tiled"


def test_wide_timestamp_span_uses_fallback_sort_identically():
    big = 2**60
    edges = [(big, 1, 2), (0, 2, 1), (big, 0, 2), (5, 1, 0), (big // 3, 2, 2)]
    a = sort_and_validate_edges(3, edges)
    b = sort_validate_arrays(3, *edges_to_arrays(edges))
    assert [(e.t, e.u, e.v) for e in a] == list(zip(*(x.tolist() for x in b)))
