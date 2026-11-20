"""Bit-identity of the column-partitioned parallel core vs the sequential core.

Run with an enlarged thread pool to genuinely exercise multi-threaded
schedules even on small CI machines, e.g.::

    NUMBA_NUM_THREADS=8 python3 -m pytest tests/test_parallel.py -q

The claims verified here are machine-independent: for every chunk count,
thread count, block width, semantics, and edge-direction mode, the packed
output must equal the sequential reference word-for-word.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.packed_reachability import (
    reachable_pairs,
    temporal_strata_packed,
)
from temporal_strata.stream import generate_temporal_stream

numba = pytest.importorskip("numba")
from temporal_strata.parallel import (  # noqa: E402
    max_threads,
    set_threads,
    temporal_strata_parallel,
)

REAL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "datasets_real_temporal",
    "standard",
    "CollegeMsg.txt",
)


def _rand_stream(n, m, timestamps, seed, directed):
    return generate_temporal_stream(n, m, timestamps, directed=directed, seed=seed)


@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("mode", ["strict", "closure"])
def test_partition_determinacy_matches_sequential(directed, mode):
    """Any chunk count yields the sequential result bit-for-bit."""
    n, m = 130, 900  # W = 3 words: chunk counts below cover 1..W and > W
    edges = _rand_stream(n, m, timestamps=40, seed=11, directed=directed)
    ref = temporal_strata_packed(n, edges, directed=directed, same_time_mode=mode)
    for chunks in (1, 2, 3, 7):
        got = temporal_strata_parallel(
            n, edges, directed=directed, same_time_mode=mode, chunks=chunks
        )
        assert np.array_equal(ref, got), f"chunks={chunks} diverged"


@pytest.mark.parametrize("mode", ["strict", "closure"])
def test_thread_count_determinacy(mode):
    """Every admissible thread count yields the identical packed state."""
    n, m = 200, 1400
    edges = _rand_stream(n, m, timestamps=60, seed=23, directed=True)
    ref = temporal_strata_packed(n, edges, directed=True, same_time_mode=mode)
    tried = []
    for k in (1, 2, 4, max_threads()):
        eff = set_threads(k)
        tried.append(eff)
        got = temporal_strata_parallel(
            n, edges, directed=True, same_time_mode=mode, num_threads=eff, chunks=max(eff, 3)
        )
        assert np.array_equal(ref, got), f"threads={eff} diverged"
    assert len(set(tried)) >= 1  # at least the single-thread schedule ran


@pytest.mark.parametrize("block_bits", [64, 50, 128])
def test_blocked_parallel_matches_sequential(block_bits):
    """Blocking (time-multiplexed columns) composes with thread parallelism
    (space-multiplexed columns); aligned and unaligned block widths."""
    n, m = 170, 1100
    edges = _rand_stream(n, m, timestamps=35, seed=5, directed=True)
    ref = temporal_strata_packed(n, edges, directed=True, same_time_mode="strict")
    got = temporal_strata_parallel(
        n, edges, directed=True, same_time_mode="strict", block_bits=block_bits, chunks=2
    )
    assert np.array_equal(ref, got)
    assert reachable_pairs(got, n) == reachable_pairs(ref, n)


def test_same_timestamp_no_chaining_strict():
    """Adversarial same-timestamp chain: strict must not chain, closure must."""
    n = 5
    edges = [(1, 0, 1), (1, 1, 2), (1, 2, 3), (2, 3, 4)]
    ref_s = temporal_strata_packed(n, edges, same_time_mode="strict")
    got_s = temporal_strata_parallel(n, edges, same_time_mode="strict", chunks=1)
    assert np.array_equal(ref_s, got_s)
    ref_c = temporal_strata_packed(n, edges, same_time_mode="closure")
    got_c = temporal_strata_parallel(n, edges, same_time_mode="closure", chunks=1)
    assert np.array_equal(ref_c, got_c)
    assert reachable_pairs(got_c, n) > reachable_pairs(got_s, n)


@pytest.mark.skipif(not os.path.exists(REAL), reason="real dataset not present")
def test_real_collegemsg_strict_pairs():
    """Real-timestamp regression anchor (machine-independent invariant)."""
    from temporal_strata.io import load_standard_edges

    n, edges, _meta = load_standard_edges(REAL)
    R = temporal_strata_parallel(n, edges, directed=True, same_time_mode="strict")
    assert reachable_pairs(R, n) == 1_794_244
