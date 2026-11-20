"""Packed uint64 bit-parallel temporal STRATA core.

This module is the temporal STRATA core. It replaces Python
Python arbitrary-precision big-integer bitsets (``1 << i`` / ``int.bit_count``)
with a fixed-width packed representation::

    R : np.ndarray of shape (n, W), dtype=uint64,  W = ceil(n / 64)

Bit ``s`` of target ``v`` lives in word ``s // 64`` at bit ``s % 64`` of row
``R[v]``. Source ``s`` reaches target ``v`` iff that bit is set.

Why this matters for the paper's cost model
--------------------------------------------
Theorem (Termination and Complexity) in the manuscript claims a per-scan cost of
``O(m * ceil(n / w))`` for word size ``w`` and a packed reachability-state memory
of ``O(n * ceil(n / w))`` *words*. A big-integer implementation does **not**
realize this model: every ``&``, ``~`` and ``|`` allocated a fresh unbounded
Python integer object, there was no fixed word ``w``, and the companion static
``TC1`` path materialized a dense ``(n, n)`` uint8 footprint (O(n^2) bytes), which
contradicts the advertised ~8x packed-state saving and is the first thing to run
out of memory on large graphs.

This module fixes all three issues:

1. Fixed word size ``w = 64`` (uint64), so one OR over a row touches exactly
   ``W = ceil(n / 64)`` machine words. The cost model now matches Theorem 3
   literally with ``w = 64``.
2. No dense ``(n, n)`` materialization anywhere. The reachability *footprint* used
   for the AND-NOT pruning test is the packed state itself.
3. Optional **source blocking**: instead of carrying all ``n`` source bits at
   once (which is fundamentally ``O(n^2 / w)`` for all-pairs), sources are
   processed in blocks of ``block_bits`` columns, bounding peak state memory to
   ``O(n * block_bits / w)`` words per pass. This relieves the "n^2 pressure"
   limitation into a tunable knob.

Semantics
---------
* ``strict``  : an edge at timestamp ``t`` may only consume source bits that were
                present *before* the timestamp group ``t`` began. Two edges at the
                same ``t`` cannot chain. Implemented with a pre-group snapshot.
* ``closure`` : zero-wait chains inside a single timestamp group are allowed; a
                least fixed point is computed per timestamp group.

The public functions return the packed matrix; helper functions convert to a
dense boolean matrix or count reachable pairs for validation against the exact
baseline in :mod:`temporal_strata.baseline`.
"""

from __future__ import annotations

from itertools import groupby
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .stream import TemporalEdge, sort_and_validate_edges

WORD_BITS = 64
_DTYPE = np.uint64

# When True (default), strict semantics snapshot only the rows written within a
# timestamp group (touched-row materialization) whenever the group touches at
# most half the vertices; otherwise a full-state copy is used. Set False to
# force the legacy full-copy snapshot for A/B comparison (see
# experiments/ablation_touched_row_core.py). Bit-identical either way.
_USE_TOUCHED_ROW = True

# When False (default), the strict merge is unconditional (R[v] |= cand). When
# True, the frontier-delta test Delta = R[u] & ~R[v] gates the merge (legacy
# behavior). The test is neutral for strict-mode correctness and, in the NumPy
# path, is strictly extra work because `np.any(cand & ~foot)` materializes a
# full W-word temporary rather than short-circuiting. Kept only for A/B
# measurement; bit-identical either way.
_STRICT_PRUNE = False


def words_for(n: int) -> int:
    """Number of uint64 words needed to hold ``n`` source bits."""
    return (n + WORD_BITS - 1) // WORD_BITS


def packed_state_bytes(n: int, block_bits: int | None = None) -> int:
    """Exact packed-state footprint in bytes for the chosen block width.

    With ``block_bits`` source columns per pass and ``n`` target rows, the state
    is ``n * ceil(block_bits / 64)`` uint64 words. For full all-pairs,
    ``block_bits == n``.
    """
    cols = n if block_bits is None else min(block_bits, n)
    return n * words_for(cols) * 8


def _empty_state(n_rows: int, n_cols_bits: int) -> np.ndarray:
    return np.zeros((n_rows, words_for(n_cols_bits)), dtype=_DTYPE)


def _set_identity(R: np.ndarray, source_offset: int, n_block: int) -> None:
    """Set R[source_offset + j][bit j] = 1 for j in [0, n_block).

    The identity is placed so that within this source block, local source index
    ``j`` corresponds to global vertex ``source_offset + j``.
    """
    for j in range(n_block):
        word, bit = divmod(j, WORD_BITS)
        R[source_offset + j, word] |= _DTYPE(1) << _DTYPE(bit)


def _row_any_new(cand_row: np.ndarray, foot_row: np.ndarray) -> bool:
    """True iff cand_row has at least one bit not already in foot_row."""
    # cand & ~foot, word-parallel; .any() short-circuits at C level.
    return bool(np.any(cand_row & ~foot_row))


def _temporal_pass(
    n: int,
    stream: List[TemporalEdge],
    *,
    source_offset: int,
    n_block: int,
    directed: bool,
    same_time_mode: str,
) -> np.ndarray:
    """Run one packed propagation pass over a block of ``n_block`` sources.

    Returns R of shape (n, W_block) where W_block = words_for(n_block). Bit j of
    R[v] is set iff global source (source_offset + j) reaches v.
    """
    R = _empty_state(n, n_block)
    _set_identity(R, source_offset, n_block)

    for _, group_iter in groupby(stream, key=lambda e: e.t):
        group = list(group_iter)

        if same_time_mode == "strict":
            # Pre-group snapshot enforcing Eq. (strict): every read within the
            # group must see the pre-group state R^-, so same-timestamp edges
            # cannot chain. Two bit-identical realizations of R^-:
            #
            #   * touched-row: snapshot only the rows written in this group
            #     (the "dirty set" D_t = {targets, and sources if undirected}).
            #     Rows not written are unchanged during the group, so reading R
            #     directly returns their pre-group value. Snapshot word-traffic
            #     is |D_t| * W instead of n * W.
            #   * full-copy: snapshot the whole state (better when a group
            #     touches a constant fraction of n).
            #
            # This selection is the Theorem-complexity touched-row materialization
            # (Sec. Strict Snapshot). It is verified bit-identical to full-copy in
            # the test suite and is what makes near-unique-timestamp real streams
            # (many tiny groups) tractable: without it strict pays an O(T n W)
            # snapshot cost that dominates on CollegeMsg / Bitcoin-OTC.
            #
            # NOTE on pruning: the strict merge is UNCONDITIONAL (R[v] |= cand).
            # The frontier-delta test Delta = R[u] & ~R[v] is *not* applied here
            # because (i) an ablation shows it gives no measurable strict-mode
            # speedup once packing makes the merge cheap, and (ii) in the
            # interpreted NumPy path `np.any(cand & ~foot)` allocates and scans a
            # full W-word temporary (it does NOT short-circuit at the C level),
            # so testing before merging is strictly more work than merging. The
            # frontier-delta test is retained only in the closure branch, where
            # it is the fixed-point change detector, and in the compiled kernel,
            # where the word-loop can genuinely early-exit. (Set _STRICT_PRUNE
            # to re-enable the test for A/B measurement; bit-identical either way.)
            dirty = {e.v for e in group}
            if not directed:
                dirty.update(e.u for e in group)
            if _USE_TOUCHED_ROW and 2 * len(dirty) <= n:
                snap = {row: R[row].copy() for row in dirty}

                def _pre(x, snap=snap, R=R):
                    r = snap.get(x)
                    return R[x] if r is None else r

                for e in group:
                    cand = _pre(e.u)
                    if _STRICT_PRUNE:
                        if _row_any_new(cand, R[e.v]):
                            R[e.v] |= cand
                    else:
                        R[e.v] |= cand
                    if not directed:
                        cand2 = _pre(e.v)
                        if _STRICT_PRUNE:
                            if _row_any_new(cand2, R[e.u]):
                                R[e.u] |= cand2
                        else:
                            R[e.u] |= cand2
            else:
                before = R.copy()
                for e in group:
                    cand = before[e.u]
                    if _STRICT_PRUNE:
                        if _row_any_new(cand, R[e.v]):
                            R[e.v] |= cand
                    else:
                        R[e.v] |= cand
                    if not directed:
                        cand2 = before[e.v]
                        if _STRICT_PRUNE:
                            if _row_any_new(cand2, R[e.u]):
                                R[e.u] |= cand2
                        else:
                            R[e.u] |= cand2

        elif same_time_mode == "closure":
            changed = True
            while changed:
                changed = False
                for e in group:
                    cand = R[e.u]
                    if _row_any_new(cand, R[e.v]):
                        R[e.v] |= cand
                        changed = True
                    if not directed:
                        cand2 = R[e.v]
                        if _row_any_new(cand2, R[e.u]):
                            R[e.u] |= cand2
                            changed = True
        else:
            raise ValueError("same_time_mode must be 'strict' or 'closure'")

    return R


def temporal_strata_packed(
    n: int,
    edges: Iterable[TemporalEdge | Tuple[int, int, int]],
    *,
    directed: bool = True,
    same_time_mode: str = "strict",
    block_bits: int | None = None,
) -> np.ndarray:
    """All-source temporal reachability via packed uint64 causal frontier algebra.

    Parameters
    ----------
    n:
        Number of vertices.
    edges:
        Timestamped contacts; tuples are read as ``(t, u, v)``.
    directed:
        If False each contact is usable in both directions at the same timestamp.
    same_time_mode:
        ``strict`` or ``closure`` (see module docstring).
    block_bits:
        If None, all ``n`` sources are carried at once (full all-pairs). Otherwise
        sources are processed in blocks of ``block_bits`` columns and the results
        are stitched into the full ``(n, ceil(n/64))`` matrix. Blocking bounds
        peak state memory to ``O(n * block_bits / 64)`` words per pass at the cost
        of re-scanning the stream ``ceil(n / block_bits)`` times.

    Returns
    -------
    np.ndarray
        Packed matrix ``R`` of shape ``(n, ceil(n/64))``, dtype uint64. Bit ``s``
        of row ``v`` (word ``s//64``, bit ``s%64``) is set iff ``s`` reaches ``v``.
    """
    if same_time_mode not in {"strict", "closure"}:
        raise ValueError("same_time_mode must be 'strict' or 'closure'")

    stream = sort_and_validate_edges(n, edges)

    if block_bits is None or block_bits >= n:
        return _temporal_pass(
            n,
            stream,
            source_offset=0,
            n_block=n,
            directed=directed,
            same_time_mode=same_time_mode,
        )

    # Source-blocked execution. Each pass owns a contiguous range of source bits.
    full = _empty_state(n, n)
    for start in range(0, n, block_bits):
        n_block = min(block_bits, n - start)
        Rb = _temporal_pass(
            n,
            stream,
            source_offset=start,
            n_block=n_block,
            directed=directed,
            same_time_mode=same_time_mode,
        )
        # Scatter the local block (source bits 0..n_block-1) into global columns
        # starting at `start`.
        _scatter_block(full, Rb, start, n_block)
    return full


def _scatter_block(full: np.ndarray, block: np.ndarray, start: int, n_block: int) -> None:
    """Copy local source bits [0, n_block) of `block` into global columns
    [start, start+n_block) of `full`. Handles arbitrary (unaligned) start."""
    if start % WORD_BITS == 0:
        # Word-aligned fast path: copy whole words.
        w0 = start // WORD_BITS
        nw = block.shape[1]
        full[:, w0 : w0 + nw] |= block[:, :nw]
        # Mask any overshoot beyond start + n_block (last word of the block may
        # carry padding bits; they are zero by construction, so nothing to do).
        return
    # General path: bit-by-bit relocation (rare; only when block_bits % 64 != 0).
    for j in range(n_block):
        sw, sb = divmod(j, WORD_BITS)
        if np.any(block[:, sw] & (_DTYPE(1) << _DTYPE(sb))):
            gw, gb = divmod(start + j, WORD_BITS)
            colbit = (block[:, sw] >> _DTYPE(sb)) & _DTYPE(1)
            full[:, gw] |= colbit << _DTYPE(gb)


# --------------------------------------------------------------------------- #
# Conversion / validation helpers
# --------------------------------------------------------------------------- #
def packed_to_matrix(R: np.ndarray, n: int | None = None) -> np.ndarray:
    """Convert packed (n, W) uint64 state to a bool matrix M[source, target].

    M[s, v] is True iff source s reaches target v. Vectorized via np.unpackbits.
    """
    n_rows = R.shape[0]
    if n is None:
        n = n_rows
    # View uint64 words as little-endian bytes, unpack to bits (LSB-first).
    as_bytes = R.view(np.uint8).reshape(n_rows, -1)
    bits = np.unpackbits(as_bytes, axis=1, bitorder="little")  # (n_rows, W*64)
    target_by_source = bits[:, :n].astype(bool)  # rows=target, cols=source
    # M[source, target] = target_by_source[target, source].T
    return target_by_source[:, :n].T.copy()


def reachable_pairs(R: np.ndarray, n: int | None = None) -> int:
    """Number of reachable ordered (source, target) pairs, including self-pairs."""
    n_rows = R.shape[0]
    if n is None:
        n = n_rows
    as_bytes = R.view(np.uint8).reshape(n_rows, -1)
    bits = np.unpackbits(as_bytes, axis=1, bitorder="little")
    return int(bits[:, :n].sum())


def estimate_packed_memory_bytes(n: int, block_bits: int | None = None) -> int:
    """Public alias mirroring the ``estimate_bitset_memory_bytes`` API."""
    return packed_state_bytes(n, block_bits)
