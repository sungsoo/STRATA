from __future__ import annotations

from itertools import groupby
from typing import Iterable, List, Sequence, Tuple
import numpy as np

from .stream import TemporalEdge, sort_and_validate_edges


def _set_bit(i: int) -> int:
    return 1 << i


def estimate_bitset_memory_bytes(n: int) -> int:
    """Ideal packed memory for n target bitsets with n source bits each."""
    return n * ((n + 7) // 8)


def temporal_strata_reachability(
    n: int,
    edges: Iterable[TemporalEdge | Tuple[int, int, int]],
    *,
    directed: bool = True,
    same_time_mode: str = "strict",
) -> List[int]:
    """All-source temporal reachability by bit-parallel causal frontier algebra.

    The returned list ``bits`` has length n. Bit s in ``bits[v]`` is one iff
    source s can reach target v through a time-respecting path.

    Parameters
    ----------
    n:
        Number of vertices.
    edges:
        Timestamped contacts. Tuple inputs are interpreted as (t, u, v).
    directed:
        If False, each temporal contact is usable in both directions at the
        same timestamp.
    same_time_mode:
        ``strict`` means paths may not traverse two edges with the same
        timestamp; all updates at a timestamp use the pre-timestamp state.
        ``closure`` means zero-wait chains inside the same timestamp are allowed;
        a fixed point is computed for each timestamp group.

    Notes
    -----
    This is a CPU-first temporal extension of STRATA's reachability idea. It
    replaces source-by-source temporal BFS with a target-indexed vector of source
    bitsets. The update for a temporal edge (u,v,t) is simply R[v] |= R[u]
    under the chosen time semantics.
    """
    if same_time_mode not in {"strict", "closure"}:
        raise ValueError("same_time_mode must be 'strict' or 'closure'")

    stream = sort_and_validate_edges(n, edges)
    bits: List[int] = [_set_bit(i) for i in range(n)]

    for _, group_iter in groupby(stream, key=lambda e: e.t):
        group = list(group_iter)
        if same_time_mode == "strict":
            before = bits.copy()
            after = bits.copy()
            for e in group:
                new_v = before[e.u] & ~after[e.v]
                if new_v:
                    after[e.v] |= before[e.u]
                if not directed:
                    new_u = before[e.v] & ~after[e.u]
                    if new_u:
                        after[e.u] |= before[e.v]
            bits = after
        else:
            changed = True
            # Same-time closure is typically tiny because groups are timestamp buckets.
            while changed:
                changed = False
                for e in group:
                    merged = bits[e.v] | bits[e.u]
                    if merged != bits[e.v]:
                        bits[e.v] = merged
                        changed = True
                    if not directed:
                        merged = bits[e.u] | bits[e.v]
                        if merged != bits[e.u]:
                            bits[e.u] = merged
                            changed = True
    return bits


def bits_to_matrix(bits: Sequence[int], n: int | None = None) -> np.ndarray:
    """Convert target-indexed bitsets to a bool matrix M[source, target]."""
    if n is None:
        n = len(bits)
    M = np.zeros((n, n), dtype=bool)
    for v, mask in enumerate(bits):
        x = mask
        while x:
            lsb = x & -x
            s = lsb.bit_length() - 1
            if s < n:
                M[s, v] = True
            x ^= lsb
    return M


def reachability_count(bits: Sequence[int]) -> int:
    """Return the number of reachable ordered source-target pairs."""
    return sum(int(mask).bit_count() for mask in bits)
