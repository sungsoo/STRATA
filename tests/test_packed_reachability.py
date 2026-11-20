"""Packed-core correctness tests.

Cross-checks the packed uint64 temporal STRATA against:
  1. the exact all-source temporal BFS baseline (ground truth), and
  2. the big-integer reference (if importable),
across strict/closure x directed/undirected x blocked/unblocked, over many
random streams. Also checks that source blocking is invariant to block width.

Run with:  python -m pytest tests/test_packed_reachability.py -q
or standalone:  python tests/test_packed_reachability.py
"""

from __future__ import annotations

import numpy as np

from temporal_strata.stream import generate_temporal_stream
from temporal_strata.baseline import all_sources_temporal_bfs
from temporal_strata.packed_reachability import (
    temporal_strata_packed,
    packed_to_matrix,
    reachable_pairs,
    words_for,
    packed_state_bytes,
)

try:
    from temporal_strata.bitset_reachability import (
        temporal_strata_reachability as _bigint_bits,
        bits_to_matrix as _bigint_to_matrix,
    )
    _HAVE_V9 = True
except Exception:  # pragma: no cover
    _HAVE_V9 = False


CONFIGS = [
    (40, 200, 30),
    (64, 400, 50),
    (65, 400, 50),    # forces W=2 words (boundary at 64)
    (130, 800, 80),   # forces W=3 words
    (200, 1500, 120),
]


def _matrix_from_packed(n, edges, **kw):
    R = temporal_strata_packed(n, edges, **kw)
    return packed_to_matrix(R, n)


def test_packed_matches_exact_baseline():
    for directed in (True, False):
        for mode in ("strict", "closure"):
            for (n, m, ts) in CONFIGS:
                edges = generate_temporal_stream(
                    n, m, ts, directed=directed, seed=11 + n
                )
                exact = all_sources_temporal_bfs(
                    n, edges, directed=directed, same_time_mode=mode
                )
                got = _matrix_from_packed(
                    n, edges, directed=directed, same_time_mode=mode
                )
                assert got.shape == exact.shape
                assert np.array_equal(got, exact), (
                    f"mismatch directed={directed} mode={mode} n={n}"
                )


def test_blocking_invariant():
    """All-pairs result must be identical regardless of source block width."""
    for (n, m, ts) in CONFIGS:
        edges = generate_temporal_stream(n, m, ts, directed=True, seed=99)
        full = _matrix_from_packed(n, edges, same_time_mode="strict")
        for block in (16, 32, 64, 96, n):
            blocked = _matrix_from_packed(
                n, edges, same_time_mode="strict", block_bits=block
            )
            assert np.array_equal(full, blocked), (
                f"block_bits={block} differs at n={n}"
            )


def test_reachable_pair_count_consistent():
    for (n, m, ts) in CONFIGS:
        edges = generate_temporal_stream(n, m, ts, directed=True, seed=7)
        R = temporal_strata_packed(n, edges, same_time_mode="strict")
        cnt = reachable_pairs(R, n)
        mat = all_sources_temporal_bfs(n, edges, same_time_mode="strict")
        assert cnt == int(mat.sum())


def test_matches_bigint_if_available():
    if not _HAVE_V9:
        return
    for (n, m, ts) in CONFIGS:
        edges = generate_temporal_stream(n, m, ts, directed=True, seed=3)
        # the big-integer variant takes (t,u,v) tuples too; reuse the same list.
        bigint_bits = _bigint_bits(n, edges, directed=True, same_time_mode="strict")
        bigint_mat = _bigint_to_matrix(bigint_bits, n)
        packed_mat = _matrix_from_packed(n, edges, same_time_mode="strict")
        assert np.array_equal(bigint_mat, packed_mat)


def test_touched_row_equals_full_copy():
    """The touched-row strict snapshot must be bit-identical to the full-copy
    snapshot across semantics, directedness, and stream shapes -- including
    near-unique-timestamp streams (many tiny groups) where touched-row is the
    performance-critical path (Sec. Strict Snapshot)."""
    import temporal_strata.packed_reachability as pr

    for seed in range(6):
        for many_groups in (True, False):
            n = 60 + seed * 7
            stream = generate_temporal_stream(
                n=n,
                m=8 * n,
                timestamps=(6 * n if many_groups else 5),
                seed=seed,
            )
            for mode in ("strict", "closure"):
                for directed in (True, False):
                    pr._USE_TOUCHED_ROW = True
                    A = temporal_strata_packed(
                        n, stream, directed=directed, same_time_mode=mode
                    )
                    pr._USE_TOUCHED_ROW = False
                    B = temporal_strata_packed(
                        n, stream, directed=directed, same_time_mode=mode
                    )
                    pr._USE_TOUCHED_ROW = True
                    assert np.array_equal(A, B), (
                        f"touched-row != full-copy (seed={seed}, mode={mode}, "
                        f"directed={directed}, many_groups={many_groups})"
                    )


def test_memory_formula():
    # Full all-pairs: n * ceil(n/64) * 8 bytes.
    assert packed_state_bytes(64) == 64 * 1 * 8
    assert packed_state_bytes(65) == 65 * 2 * 8
    assert words_for(64) == 1 and words_for(65) == 2
    # Blocking reduces per-pass footprint.
    assert packed_state_bytes(10000, block_bits=1024) < packed_state_bytes(10000)


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passed")
