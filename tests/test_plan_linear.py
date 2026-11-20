"""Linear-time touched-row plan equals the vectorized and reference plans."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.packed_reachability import temporal_strata_packed  # noqa: E402
from temporal_strata.stream import generate_temporal_arrays, arrays_to_edges  # noqa: E402

numba = pytest.importorskip("numba")
from temporal_strata import parallel as par  # noqa: E402

REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "datasets_real_temporal", "standard")


def _plans(n, ts, us, vs, directed=True):
    prep = par.prepare_stream_arrays(n, ts, us, vs, directed=directed, same_time_mode="closure")
    g, u, v = prep.gptr, prep.us, prep.vs
    return (par._strict_snapshot_plan(g, u, v), par._strict_snapshot_plan_vec(g, u, v),
            par._strict_snapshot_plan_ref(g, u, v))


def _same(a, b):
    for x, y in zip(a[:3], b[:3]):
        assert x.dtype == y.dtype == np.int64
        assert np.array_equal(x, y)
    assert a[3] == b[3]


@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("n,m,ts,seed", [
    (2, 40, 3, 1), (130, 900, 40, 11), (300, 3000, 1, 31),   # one huge group (> 32 targets)
    (500, 3000, 3000, 7), (64, 6000, 10, 3), (1000, 4000, 80, 5),
])
def test_linear_equals_vectorized_and_reference(n, m, ts, seed, directed):
    a = generate_temporal_arrays(n, m, ts, directed=directed, seed=seed)
    lin, vec, ref = _plans(n, *a, directed=directed)
    _same(lin, vec)
    _same(lin, ref)


def test_edge_cases():
    empty = (np.zeros(0, np.int64),) * 3
    lin = par._strict_snapshot_plan(np.zeros(1, np.int64), empty[1], empty[2])
    assert lin[0].size == 0 and lin[1].tolist() == [0] and lin[3] == 1
    ts = np.array([1, 1, 1, 2]); us = np.array([0, 1, 2, 3]); vs = np.array([1, 2, 3, 4])
    lin, vec, ref = _plans(5, ts, us, vs)
    _same(lin, ref)
    ts = np.array([0, 0, 0]); us = np.array([2, 2, 3]); vs = np.array([2, 3, 2])  # self-contact
    lin, vec, ref = _plans(4, ts, us, vs, directed=False)
    _same(lin, ref)


@pytest.mark.skipif(not os.path.isdir(REAL), reason="real datasets not present")
@pytest.mark.parametrize("name", ["CollegeMsg", "BitcoinOTC", "BitcoinOTCpos"])
def test_linear_plan_on_real_data(name):
    from temporal_strata.io import load_standard_arrays

    n, t, u, v, _ = load_standard_arrays(os.path.join(REAL, f"{name}.txt"))
    lin, vec, ref = _plans(n, t, u, v)
    _same(lin, vec)
    _same(lin, ref)


def test_results_bit_identical_with_linear_plan():
    n, m = 300, 2500
    a = generate_temporal_arrays(n, m, 25, directed=True, seed=17)
    ref = temporal_strata_packed(n, arrays_to_edges(*a), directed=True, same_time_mode="strict")
    for layout in ("rowmajor", "tiled"):
        got = par.temporal_strata_parallel(n, a, layout=layout)
        assert np.array_equal(ref, got)
