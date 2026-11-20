"""Vectorized sequential preprocessing equals the pre-v4 loops.

The flatten step and the strict touched-row plan were rewritten without Python
loops or dictionaries. Every output array must equal the reference
implementation (kept in temporal_strata/parallel.py as ``*_ref``) exactly, and
the full packed result must stay bit-identical to the sequential core.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.packed_reachability import temporal_strata_packed  # noqa: E402
from temporal_strata.stream import (  # noqa: E402
    TemporalEdge,
    generate_temporal_stream,
    sort_and_validate_edges,
)

numba = pytest.importorskip("numba")
from temporal_strata import parallel as par  # noqa: E402

REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "datasets_real_temporal", "standard")


def _check(n, edges, directed):
    stream = sort_and_validate_edges(n, edges)
    a = par._flatten_stream(n, stream, directed)
    b = par._flatten_stream_ref(n, stream, directed)
    for x, y in zip(a, b):
        assert x.dtype == y.dtype == np.int64
        assert np.array_equal(x, y)
    ts, us, vs = a
    gptr = par._group_ptr(ts)
    p_new = par._strict_snapshot_plan(gptr, us, vs)
    p_ref = par._strict_snapshot_plan_ref(gptr, us, vs)
    for x, y in zip(p_new[:3], p_ref[:3]):
        assert x.dtype == y.dtype == np.int64
        assert np.array_equal(x, y)
    assert p_new[3] == p_ref[3]


@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("n,m,ts,seed", [
    (2, 30, 3, 1), (130, 900, 40, 11), (300, 2200, 2, 31),   # few, very large groups
    (500, 3000, 3000, 7), (64, 5000, 10, 3),                 # near-unique groups / dense
])
def test_prep_arrays_identical(n, m, ts, seed, directed):
    _check(n, generate_temporal_stream(n, m, ts, directed=directed, seed=seed), directed)


def test_prep_edge_cases():
    _check(5, [], True)
    _check(5, [(1, 0, 1), (1, 1, 2), (1, 2, 3), (2, 3, 4)], True)   # same-time chain
    _check(5, [(0, 2, 2), (0, 2, 3), (0, 3, 2)], False)             # self-contact, mutual
    _check(4, [TemporalEdge(7, 0, 3)], False)
    plan = par._strict_snapshot_plan(np.zeros(1, dtype=np.int64),
                                     np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    assert plan[0].size == 0 and plan[1].tolist() == [0] and plan[3] == 1


@pytest.mark.skipif(not os.path.isdir(REAL), reason="real datasets not present")
@pytest.mark.parametrize("name", ["CollegeMsg", "BitcoinOTC", "BitcoinOTCpos"])
def test_prep_identical_on_real_data(name):
    from temporal_strata.io import load_standard_edges

    n, edges, _ = load_standard_edges(os.path.join(REAL, f"{name}.txt"))
    _check(n, edges, True)


@pytest.mark.parametrize("directed", [True, False])
def test_end_to_end_still_bit_identical(directed):
    n, m = 260, 2000
    edges = generate_temporal_stream(n, m, 30, directed=directed, seed=21)
    for mode in ("strict", "closure"):
        ref = temporal_strata_packed(n, edges, directed=directed, same_time_mode=mode)
        for layout in ("rowmajor", "tiled"):
            got = par.temporal_strata_parallel(n, edges, directed=directed,
                                               same_time_mode=mode, layout=layout)
            assert np.array_equal(ref, got)
