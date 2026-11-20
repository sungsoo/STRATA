"""Parallel sequential-preprocessing paths.

On the Apple M5 Max the column-parallel kernel left the one-time preprocessing
as the end-to-end bottleneck (paper Fig. 7: up to 75% of the fastest run on the
n=8,000 stream), and inside it the touched-row plan dominates. Two parts are
parallelized here; both return arrays identical to the sequential paths for
every block count (tests/test_parallel_prep.py), so the output stays a pure
function of the input (paper Sec. 4.3, "Determinacy").

* Touched-row plan: timestamp groups are independent. Groups are split into
  ``nblocks`` contiguous blocks of about equal edge count; each block runs the
  sequential linear-time scan with its own marker arrays. A first pass counts
  the distinct targets per group, a prefix sum fixes every group's output
  offset, and a second pass fills the sorted lists and snapshot slots.
* Sort / validate: when the packed (t, u, v) key fits in int64 (the case the
  sequential path also uses), the keys are sorted by a parallel chunk sort plus
  pairwise parallel merges and decoded back; equal keys are identical edges, so
  the decoded arrays equal the sequential stable sort. When the timestamp range
  overflows the key (SNAP seconds over years), the dense time rank replaces t
 ; only if even that overflows does a stable time sort plus a per-group sort
  run (v22, R56; it replaced a three-key lexsort that took 88 s on 6.3e7 edges).

Group boundaries stay the vectorized NumPy scan (0.3-0.5 ms of 20-56 ms on the
M5 streams). Whether the parallel paths pay off is a measurement question
(``auto`` uses them only above PREP_PARALLEL_MIN_EDGES with more than one
thread); no speed claim is made before the M5 re-measurement.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .parallel import _HAVE_NUMBA, njit, prange

PREP_PARALLEL_MIN_EDGES = 1 << 17   # below this, "auto" keeps the sequential paths
PREP_MODES = ("auto", "on", "off")


# --------------------------------------------------------------------------- #
# Touched-row plan
# --------------------------------------------------------------------------- #
def group_blocks(gptr: np.ndarray, nblocks: int) -> np.ndarray:
    """Split groups into <= nblocks contiguous ranges of about equal edge count.

    Returns block boundaries over group indices (int64[k+1], 0 .. G)."""
    G = int(gptr.shape[0] - 1)
    nblocks = max(1, min(int(nblocks), max(G, 1)))
    if G <= 0:
        return np.zeros(2, dtype=np.int64)
    m = int(gptr[-1])
    targets = (np.arange(1, nblocks) * m) // nblocks
    cuts = np.searchsorted(gptr, targets, side="left")
    cuts = np.clip(cuts, 1, G - 1) if G > 1 else cuts[:0]
    return np.unique(np.concatenate([[0], cuts, [G]]).astype(np.int64))


@njit(parallel=True, cache=True)
def _plan_count_kernel(gptr, vs, nv, bptr):
    B = bptr.shape[0] - 1
    G = gptr.shape[0] - 1
    cnt = np.zeros(G, dtype=np.int64)
    for b in prange(B):
        last = np.full(nv, -1, dtype=np.int64)
        for g in range(bptr[b], bptr[b + 1]):
            c = 0
            for e in range(gptr[g], gptr[g + 1]):
                v = vs[e]
                if last[v] != g:
                    last[v] = g
                    c += 1
            cnt[g] = c
    return cnt


@njit(parallel=True, cache=True)
def _plan_fill_kernel(gptr, us, vs, nv, bptr, dptr, dirty, src_slot):
    B = bptr.shape[0] - 1
    for b in prange(B):
        last = np.full(nv, -1, dtype=np.int64)
        slot = np.empty(nv, dtype=np.int64)
        for g in range(bptr[b], bptr[b + 1]):
            lo = gptr[g]
            hi = gptr[g + 1]
            start = dptr[g]
            total = start
            for e in range(lo, hi):
                v = vs[e]
                if last[v] != g:
                    last[v] = g
                    dirty[total] = v
                    total += 1
            cnt = total - start
            if cnt <= 32:
                for a in range(start + 1, total):
                    x = dirty[a]
                    k = a - 1
                    while k >= start and dirty[k] > x:
                        dirty[k + 1] = dirty[k]
                        k -= 1
                    dirty[k + 1] = x
            else:
                dirty[start:total] = np.sort(dirty[start:total])
            for k in range(start, total):
                slot[dirty[k]] = k
            for e in range(lo, hi):
                u = us[e]
                if last[u] == g:
                    src_slot[e] = slot[u]


def strict_snapshot_plan_parallel(gptr, us, vs, nblocks: int):
    """Parallel touched-row plan; same outputs as parallel._strict_snapshot_plan."""
    G = int(gptr.shape[0] - 1)
    m = int(us.shape[0])
    if not _HAVE_NUMBA or G <= 0 or m == 0:
        from .parallel import _strict_snapshot_plan
        return _strict_snapshot_plan(gptr, us, vs)
    gptr = np.ascontiguousarray(gptr, dtype=np.int64)
    us = np.ascontiguousarray(us, dtype=np.int64)
    vs = np.ascontiguousarray(vs, dtype=np.int64)
    nv = np.int64(max(int(us.max()), int(vs.max())) + 1)
    bptr = group_blocks(gptr, nblocks)
    cnt = _plan_count_kernel(gptr, vs, nv, bptr)
    dptr = np.zeros(G + 1, dtype=np.int64)
    np.cumsum(cnt, out=dptr[1:])
    dirty = np.empty(int(dptr[-1]), dtype=np.int64)
    src_slot = np.full(m, -1, dtype=np.int64)
    _plan_fill_kernel(gptr, us, vs, nv, bptr, dptr, dirty, src_slot)
    max_dirty = int(max(1, cnt.max())) if G else 1
    return dirty, dptr, src_slot, max_dirty


# --------------------------------------------------------------------------- #
# Sort / validate
# --------------------------------------------------------------------------- #
@njit(parallel=True, cache=True)
def _chunk_sort(keys, bounds):
    for c in prange(bounds.shape[0] - 1):
        keys[bounds[c]:bounds[c + 1]] = np.sort(keys[bounds[c]:bounds[c + 1]])


@njit(cache=True)
def _merge_into(src, lo, mid, hi, dst):
    i, j, k = lo, mid, lo
    while i < mid and j < hi:
        if src[i] <= src[j]:
            dst[k] = src[i]
            i += 1
        else:
            dst[k] = src[j]
            j += 1
        k += 1
    while i < mid:
        dst[k] = src[i]
        i += 1
        k += 1
    while j < hi:
        dst[k] = src[j]
        j += 1
        k += 1


@njit(parallel=True, cache=True)
def _merge_round(src, dst, bounds, width):
    npairs = (bounds.shape[0] - 1 + 2 * width - 1) // (2 * width)
    nb = bounds.shape[0] - 1
    for q in prange(npairs):
        a = q * 2 * width
        lo = bounds[a]
        mid = bounds[min(a + width, nb)]
        hi = bounds[min(a + 2 * width, nb)]
        _merge_into(src, lo, mid, hi, dst)


def parallel_sort_keys(keys: np.ndarray, nchunks: int) -> np.ndarray:
    """Sort int64 keys: parallel chunk sort, then pairwise parallel merges."""
    keys = np.array(keys, dtype=np.int64, copy=True)
    m = keys.shape[0]
    nchunks = max(1, min(int(nchunks), max(m, 1)))
    if not _HAVE_NUMBA or nchunks == 1 or m < 2:
        keys.sort(kind="stable")
        return keys
    bounds = (np.arange(nchunks + 1, dtype=np.int64) * m) // nchunks
    _chunk_sort(keys, bounds)
    src, dst = keys, np.empty_like(keys)
    width = 1
    while width < nchunks:
        _merge_round(src, dst, bounds, width)
        src, dst = dst, src
        width *= 2
    return src


NEAR_SORTED_DIV = 32   # at most m/32 descents -> "nearly sorted"


def nearly_sorted(a) -> bool:
    """True when at most len(a)/NEAR_SORTED_DIV adjacent pairs descend. An
    adaptive stable sort (Timsort) is then near-linear and beats the parallel
    chunk sort + merges, which do not exploit existing order (v10 M5: SNAP logs
    have 0.4-1.2% descents; synthetic streams arrive sorted)."""
    a = np.asarray(a)
    if a.size < 2:
        return True
    return int(np.count_nonzero(a[1:] < a[:-1])) * NEAR_SORTED_DIV <= a.size


def _sort_keys(key, nchunks, info, primary):
    """Sort packed keys. The presortedness test looks at the primary key (time):
    unordered (u, v) inside a timestamp group are short runs that Timsort
    absorbs, while descents in time decide whether the input is a time log."""
    if nearly_sorted(primary):
        if info is not None:
            info["sort_path"] = "adaptive"
        return key[np.argsort(key, kind="stable")]
    if info is not None:
        info["sort_path"] = "parallel"
    return parallel_sort_keys(key, nchunks)


def sort_validate_arrays_parallel(n: int, t_raw, us, vs, nchunks: int, info=None
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same result as stream.sort_validate_arrays (validation and values).
    ``info`` (optional dict) receives ``sort_path``: adaptive / parallel /
    time-then-group (no packed key fits, v22) / lexsort (non-integer t)."""
    from .stream import TemporalEdge, sort_validate_arrays

    us = np.asarray(us, dtype=np.int64)
    vs = np.asarray(vs, dtype=np.int64)
    t_raw = np.asarray(t_raw)
    if not (t_raw.dtype.kind in "iu" and t_raw.size):
        if info is not None:
            info["sort_path"] = "lexsort"
        return sort_validate_arrays(n, t_raw, us, vs, info=info)
    bad = (us < 0) | (vs < 0) | (us >= n) | (vs >= n)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        te = TemporalEdge(t_raw[i].item(), int(us[i]), int(vs[i]))
        raise ValueError(f"edge endpoint out of range for n={n}: {te}")
    tmin = int(t_raw.min())
    span = int(t_raw.max()) - tmin + 1
    nn = np.int64(n) * np.int64(n)
    if span * n * n < 2**62:
        key = (t_raw.astype(np.int64) - tmin) * nn + us * np.int64(n) + vs
        key = _sort_keys(key, nchunks, info, t_raw)
        t_part, rem = np.divmod(key, nn)
        ts = t_part + tmin
        u_out, v_out = np.divmod(rem, np.int64(n))
        return ts.astype(np.int64), u_out.astype(np.int64), v_out.astype(np.int64)
    # wide timestamp range. Pack the dense time rank instead
    # of t (same order), sort in parallel, and map ranks back to timestamps.
    m = t_raw.size
    tt = t_raw.astype(np.int64)
    d = np.diff(tt)
    if (d >= 0).all():
        rank = np.empty(m, dtype=np.int64)
        rank[0] = 0
        np.cumsum(d > 0, out=rank[1:])
        t_of_rank = tt[np.concatenate(([True], d > 0))]
    elif nearly_sorted(tt):
        from .stream import dense_time_rank
        rank = dense_time_rank(tt)
        t_of_rank = np.empty(int(rank.max()) + 1, dtype=np.int64)
        t_of_rank[rank] = tt
    elif span * m < 2**62:
        k1 = parallel_sort_keys((tt - tmin) * np.int64(m) + np.arange(m, dtype=np.int64), nchunks)
        t_sorted, pos = np.divmod(k1, np.int64(m))
        step = np.concatenate(([True], np.diff(t_sorted) > 0))
        rank = np.empty(m, dtype=np.int64)
        rank[pos] = np.cumsum(step) - 1
        t_of_rank = t_sorted[step] + tmin
    else:   # no packed key fits: time-then-group sort (v22, R56)
        return sort_validate_arrays(n, t_raw, us, vs, info=info)
    if (int(rank.max()) + 1) * n * n >= 2**62:
        return sort_validate_arrays(n, t_raw, us, vs, info=info)
    key = _sort_keys(rank * nn + us * np.int64(n) + vs, nchunks, info, tt)
    r_part, rem = np.divmod(key, nn)
    u_out, v_out = np.divmod(rem, np.int64(n))
    return t_of_rank[r_part].astype(np.int64), u_out.astype(np.int64), v_out.astype(np.int64)


def resolve_prep(mode: str, m: int, threads: int) -> bool:
    """Whether the parallel preprocessing paths are used (``auto`` / ``on`` / ``off``)."""
    if mode not in PREP_MODES:
        raise ValueError(f"prep mode must be one of {PREP_MODES}")
    if mode == "off" or not _HAVE_NUMBA:
        return False
    if mode == "on":
        return True
    return threads > 1 and m >= PREP_PARALLEL_MIN_EDGES


__all__ = ["NEAR_SORTED_DIV", "PREP_PARALLEL_MIN_EDGES", "PREP_MODES", "group_blocks",
           "nearly_sorted", "parallel_sort_keys",
           "resolve_prep", "sort_validate_arrays_parallel", "strict_snapshot_plan_parallel"]
