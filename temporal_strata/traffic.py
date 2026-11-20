"""Byte-traffic model of the strict column kernel (backlog R5, roofline input).

Corollary 1 gives the work of the strict kernel as
``O((m + sum_t |D_t|) * W)`` word operations. This module turns that count into
the number of bytes the kernel *touches* (loads + stores of 64-bit words),
which experiments/bandwidth_roofline.py divides by a measured kernel time to
obtain an achieved bandwidth and compares with a measured STREAM-triad
bandwidth on the same machine.

Per directed edge (after undirected expansion) the kernel loads one source
segment (``R[u]`` or its snapshot), loads ``R[v]`` and stores ``R[v]``:
``3 * 8 * W`` bytes. Per dirty row and group the touched-row snapshot loads the
row and stores the snapshot copy: ``2 * 8 * W`` bytes. The tiled layout adds a
one-time gather and scatter of the whole state (load + store each way):
``4 * 8 * n * W`` bytes.

These are *touched* bytes, an upper bound on DRAM traffic: rows that stay
cache-resident are touched repeatedly without reaching memory, so an achieved
"bandwidth" above the STREAM figure means cache residency, not a measurement
error. The model is exact for the counts it states (no timing involved).
"""
from __future__ import annotations

from typing import Dict

WORD_BYTES = 8


def strict_kernel_bytes(num_edges: int, dirty_total: int, n: int, words: int,
                        layout: str = "rowmajor") -> Dict[str, int]:
    """Touched bytes of one strict kernel launch over ``words`` columns."""
    if layout not in ("rowmajor", "tiled"):
        raise ValueError("layout must be 'rowmajor' or 'tiled' (resolve 'auto' first)")
    W = int(words)
    edge_bytes = 3 * WORD_BYTES * W * int(num_edges)
    snap_bytes = 2 * WORD_BYTES * W * int(dirty_total)
    tile_bytes = 4 * WORD_BYTES * int(n) * W if layout == "tiled" else 0
    return {
        "edge_bytes": edge_bytes,
        "snapshot_bytes": snap_bytes,
        "tile_bytes": tile_bytes,
        "total_bytes": edge_bytes + snap_bytes + tile_bytes,
    }


def triad_bytes(n_elems: int, write_allocate: bool = False) -> int:
    """Bytes moved by one STREAM triad ``a = b + s*c`` over float64 arrays."""
    return (4 if write_allocate else 3) * WORD_BYTES * int(n_elems)


__all__ = ["strict_kernel_bytes", "triad_bytes", "WORD_BYTES"]
