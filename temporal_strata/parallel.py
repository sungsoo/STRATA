"""Thread-parallel packed temporal STRATA (column-partitioned, deterministic).

This module adds the shared-memory multicore realization of the packed
bit-parallel temporal STRATA core in :mod:`temporal_strata.packed_reachability`.
It exists because the sequential core already exposes -- but does not exploit --
a *communication-free* parallel decomposition:

    Every state update in both semantics is COLUMN-LOCAL.
    Strict:   R[v][j] |= R^-(u)[j]      (Eq. strict; snapshot read)
    Closure:  R[v][j] |= R[u][j]        (least fixed point per group)

    for every word column j independently. Therefore the evolution of any set
    of word columns depends only on those columns. Partition the W = ceil(n/64)
    columns into disjoint chunks, give each worker one chunk, and let every
    worker scan the SAME time-ordered edge stream over its own columns:

      * no worker ever reads or writes another worker's columns
        (zero inter-thread communication, no atomics, no locks);
      * the final state is BIT-IDENTICAL to the sequential result for every
        chunk count, every thread count, and every scheduling order
        (partition determinacy);
      * total work equals the sequential work O(m * ceil(n/w))
        (work efficiency -- nothing is recomputed).

    Source blocking (``block_bits``) is the *time-multiplexed* form of exactly
    the same column decomposition; this module is the *space-multiplexed*
    (thread-parallel) form, and the two compose.

Within a strict timestamp group the merges commute (all reads go to the
pre-group snapshot R^-, and OR is commutative/associative/idempotent), so the
per-group snapshot that enforces temporal exactness is the same invariant that
makes parallel execution race-free: temporal correctness and parallel safety
come from one mechanism. The closure branch is a monotone fixed point over a
finite lattice per column chunk, hence its per-chunk fixed point is the
restriction of the unique global fixed point to those columns.

The kernels are Numba-compiled (LLVM) and use the same touched-row snapshot
materialization as the sequential core: per group, only the rows written in
that group are snapshotted, restricted to the worker's own columns.

Determinism is enforced by construction and verified by the test suite
(tests/test_parallel.py): the packed output is compared word-for-word against
the sequential core across chunk counts, thread counts, block widths, both
semantics, and directed/undirected inputs.

Realization choices
----------------------------------------------------
Three choices change *how* the columns are executed and never *what* is
computed (Theorem 3 holds for every partition and every interleaving):

* ``layout="rowmajor"`` (the v1-v4 default): workers update their
  column slice in place inside the shared row-major ``n x W`` state.
  ``layout="tiled"``: each worker gathers its slice into a private contiguous
  ``n x cw`` tile, runs the scan there, and scatters the tile back once, so no
  two workers ever write the same cache line (cost: one gather + one scatter
  of the slice, and a transient second copy of the state). Default since
  STRATA-v5 (measured faster than row-major everywhere on the M5 Max).
  ``layout="auto"``: tiled iff ``floor(W / L) < p`` (``L`` = cache-line size in
  64-bit words, detected at run time; ``p`` = worker threads), i.e. iff the
  row-major state cannot give every worker at least one whole cache line.
  (Kept for reproducibility; the M5 Max crossover measurement did not support
  it -- tiled won at every point, so the advantage is locality, not false
  sharing.)
* ``schedule="static"``: ``p`` equal chunks, Numba's default static division.
  ``schedule="dynamic"``: ``oversub * p`` smaller chunks handed to idle workers
  one at a time (``numba.parallel_chunksize(1)``), intended for heterogeneous
  cores (e.g. super + performance cores) whose speeds differ.
* ``chunks``: explicit chunk count; overrides the automatic choice.

:func:`resolve_plan` makes the choice explicit and is recorded by the
measurement harnesses next to every timing.

Honesty note: this module makes *correctness and work-efficiency* claims that
are machine-independent and verified bit-for-bit. It makes no built-in speedup
claim; wall-clock scaling is measured only by experiments/parallel_scaling.py
on a disclosed machine, and that script refuses to report speedups on a
single-core host.
"""

from __future__ import annotations

import os
import platform
import subprocess
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .packed_reachability import (
    WORD_BITS,
    _DTYPE,
    _empty_state,
    _scatter_block,
    words_for,
)
from .stream import TemporalEdge, sort_and_validate_edges

try:  # pragma: no cover - exercised through public API
    import numba
    from numba import njit, prange

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*a, **k):  # type: ignore
        def deco(f):
            return f

        return deco

    prange = range  # type: ignore


# --------------------------------------------------------------------------- #
# Thread control helpers
# --------------------------------------------------------------------------- #
def max_threads() -> int:
    """Upper bound on usable worker threads in this process.

    Numba fixes the thread-pool size at first parallel launch from
    ``NUMBA_NUM_THREADS`` (default: CPU count). ``set_num_threads`` may only
    lower the active count below that bound.
    """
    if not _HAVE_NUMBA:
        return 1
    return int(numba.config.NUMBA_NUM_THREADS)


def set_threads(k: int) -> int:
    """Clamp-and-set the active Numba thread count; return the effective value."""
    if not _HAVE_NUMBA:
        return 1
    eff = max(1, min(int(k), max_threads()))
    numba.set_num_threads(eff)
    return eff


# --------------------------------------------------------------------------- #
# Cache-line detection and the realization plan (layout / schedule / chunks)
# --------------------------------------------------------------------------- #
LAYOUTS = ("rowmajor", "tiled", "auto")
SCHEDULES = ("static", "dynamic")
CHUNK_POLICIES = ("auto", "threads")
DEFAULT_OVERSUB = 4
_FALLBACK_LINE_BYTES = 64


def _sysctl(name: str) -> Optional[str]:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True,
                             text=True, timeout=2, check=False)
    except Exception:
        return None
    val = out.stdout.strip()
    return val if out.returncode == 0 and val else None


@lru_cache(maxsize=1)
def cache_line_bytes() -> int:
    """Cache-line size in bytes, detected at run time.

    Order: ``STRATA_CACHELINE_BYTES`` environment override; macOS
    ``sysctl hw.cachelinesize``; Linux sysfs ``coherency_line_size``; else 64.
    """
    env = os.environ.get("STRATA_CACHELINE_BYTES")
    if env:
        try:
            v = int(env)
            if v > 0:
                return v
        except ValueError:
            pass
    if platform.system() == "Darwin":
        v = _sysctl("hw.cachelinesize")
        if v and v.isdigit() and int(v) > 0:
            return int(v)
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    try:
        with open(path) as fh:
            v = int(fh.read().strip())
            if v > 0:
                return v
    except Exception:
        pass
    return _FALLBACK_LINE_BYTES


def cache_line_words() -> int:
    """Cache-line size in 64-bit state words (``L`` in the layout rule)."""
    return max(1, cache_line_bytes() // 8)


@dataclass(frozen=True)
class ParallelPlan:
    """The realization chosen for one kernel launch (recorded by harnesses)."""

    layout: str          # requested layout: rowmajor | tiled | auto
    resolved_layout: str # rowmajor | tiled
    schedule: str        # static | dynamic
    threads: int         # p
    words: int           # W (or W_beta under source blocking)
    line_words: int      # L
    nchunks: int
    oversub: int
    rule: str            # human-readable reason for the layout / chunk choice
    chunk_policy: str = "auto"  # auto (cache-line cap for row-major) | threads

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def resolve_plan(
    words: int,
    threads: int,
    *,
    layout: str = "tiled",
    schedule: str = "static",
    chunks: Optional[int] = None,
    oversub: int = DEFAULT_OVERSUB,
    line_words: Optional[int] = None,
    chunk_policy: str = "auto",
) -> ParallelPlan:
    """Choose layout and chunk count for ``words`` columns on ``threads`` workers.

    Layout rule (``layout="auto"``): tiled iff ``floor(W/L) < p``. Chunk rule
    (``chunks=None``): ``p`` chunks (static) or ``oversub * p`` chunks
    (dynamic), never more than ``W``; under the row-major layout additionally
    at most ``floor(W/L)`` so every chunk spans at least one whole cache line
    (no false sharing at chunk borders). Every choice is a performance choice
    only: Theorem 3 makes the output identical for all of them.

    ``chunk_policy="threads"`` (the measurement protocol of the v1 harness,
    restored in STRATA-v3 for backlog R8) drops the cache-line cap: ``p``
    (static) or ``oversub * p`` (dynamic) chunks, capped only by ``W``. Without
    it, a 128-byte-line machine (L = 16 words) would cap row-major runs on the
    real datasets (W = 30 / 92 / 88) at 1 / 5 / 5 chunks for every thread count.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    if chunk_policy not in CHUNK_POLICIES:
        raise ValueError(f"chunk_policy must be one of {CHUNK_POLICIES}")
    W = max(1, int(words))
    p = max(1, int(threads))
    L = max(1, int(line_words)) if line_words is not None else cache_line_words()
    k = max(1, int(oversub))
    lines = W // L
    if layout == "auto":
        resolved = "tiled" if lines < p else "rowmajor"
        rule = f"auto: floor(W/L)={lines} {'<' if lines < p else '>='} p={p} -> {resolved}"
    else:
        resolved = layout
        rule = f"explicit layout={layout}"
    if chunks is not None:
        nchunks = max(1, min(int(chunks), W))
        rule += f"; explicit chunks={nchunks}"
    else:
        base = p if schedule == "static" else k * p
        capped = resolved == "rowmajor" and chunk_policy == "auto"
        cap = max(1, lines) if capped else W
        nchunks = max(1, min(base, W, cap))
        rule += (f"; chunks=min({'p' if schedule == 'static' else 'oversub*p'}={base}, "
                 f"W={W}" + (f", floor(W/L)={max(1, lines)}" if capped else "")
                 + f")={nchunks}")
        if chunk_policy == "threads":
            rule += " [chunk_policy=threads]"
    return ParallelPlan(layout=layout, resolved_layout=resolved, schedule=schedule,
                        threads=p, words=W, line_words=L, nchunks=nchunks,
                        oversub=k, rule=rule, chunk_policy=chunk_policy)


def machine_info() -> Dict[str, object]:
    """Machine description recorded verbatim next to every timing.

    On Apple silicon this includes the per-performance-level core counts
    (``hw.perflevelN.physicalcpu`` / ``.name``), which the scaling plots use as
    reference lines; macOS does not guarantee which cores a thread runs on.
    """
    info: Dict[str, object] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count() or 1,
        "cache_line_bytes": cache_line_bytes(),
        "cache_line_env_override": bool(os.environ.get("STRATA_CACHELINE_BYTES")),
    }
    if _HAVE_NUMBA:
        info["numba"] = numba.__version__
        info["numba_max_threads"] = max_threads()
        try:
            info["numba_threading_layer_config"] = str(numba.config.THREADING_LAYER)
        except Exception:  # pragma: no cover
            pass
    if platform.system() == "Darwin":
        brand = _sysctl("machdep.cpu.brand_string")
        if brand:
            info["cpu_brand"] = brand
        mem = _sysctl("hw.memsize")
        if mem and mem.isdigit():
            info["memory_bytes"] = int(mem)
        levels = []
        for lvl in range(4):
            ncpu = _sysctl(f"hw.perflevel{lvl}.physicalcpu")
            if ncpu is None:
                break
            levels.append({"level": lvl,
                           "name": _sysctl(f"hw.perflevel{lvl}.name") or f"perflevel{lvl}",
                           "physicalcpu": int(ncpu) if ncpu.isdigit() else ncpu})
        if levels:
            info["perf_levels"] = levels
    else:
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        info["cpu_brand"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
        try:
            with open("/proc/meminfo") as fh:
                first = fh.readline().split()
                if first and first[0] == "MemTotal:":
                    info["memory_bytes"] = int(first[1]) * 1024
        except Exception:
            pass
    info["summary"] = (
        f"{info.get('cpu_brand') or info['processor'] or info['platform']}, "
        f"{info['cpu_count']} hardware threads"
        + (f" ({' + '.join(str(l['physicalcpu']) + ' ' + str(l['name']) for l in info['perf_levels'])})"
           if info.get("perf_levels") else "")
        + (f", {int(info['memory_bytes']) // 2**30} GiB" if info.get("memory_bytes") else "")
        + f", {info['cache_line_bytes']}-byte lines, Python {info['python']}, NumPy {info['numpy']}"
        + (f", Numba {info['numba']}" if info.get("numba") else "")
    )
    return info


# --------------------------------------------------------------------------- #
# Stream preprocessing (pure NumPy; done once per stream, reused per block)
# --------------------------------------------------------------------------- #
def _flatten_stream(
    n: int,
    stream: List[TemporalEdge],
    directed: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ts, us, vs) int64 arrays; undirected contacts are expanded into
    two directed edges inside the same timestamp group (semantics-equivalent:
    under strict, both directions read the same pre-group snapshot; under
    closure, the least fixed point is direction-symmetric by construction).

    Vectorized since STRATA-v3/one pass extracts the three
    fields, the undirected expansion is a strided interleave. The output is
    array-for-array identical to :func:`_flatten_stream_ref` (the pre-v4 loop,
    kept as the reference; tests/test_prep_vectorized.py).
    """
    m = len(stream)
    t_raw = np.array([e.t for e in stream]) if m else np.zeros(0, dtype=np.int64)
    ts1 = t_raw.astype(np.int64, copy=False) if t_raw.dtype != np.int64 else t_raw
    us1 = np.fromiter((e.u for e in stream), dtype=np.int64, count=m)
    vs1 = np.fromiter((e.v for e in stream), dtype=np.int64, count=m)
    if directed:
        return ts1, us1, vs1
    ts = np.empty(2 * m, dtype=np.int64)
    us = np.empty(2 * m, dtype=np.int64)
    vs = np.empty(2 * m, dtype=np.int64)
    ts[0::2] = ts1
    ts[1::2] = ts1
    us[0::2] = us1
    us[1::2] = vs1
    vs[0::2] = vs1
    vs[1::2] = us1
    return ts, us, vs


def _flatten_stream_ref(
    n: int,
    stream: List[TemporalEdge],
    directed: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ts, us, vs) int64 arrays; undirected contacts are expanded into
    two directed edges inside the same timestamp group (semantics-equivalent:
    under strict, both directions read the same pre-group snapshot; under
    closure, the least fixed point is direction-symmetric by construction)."""
    if directed:
        m = len(stream)
        ts = np.empty(m, dtype=np.int64)
        us = np.empty(m, dtype=np.int64)
        vs = np.empty(m, dtype=np.int64)
        for i, e in enumerate(stream):
            ts[i] = e.t
            us[i] = e.u
            vs[i] = e.v
        return ts, us, vs
    m = 2 * len(stream)
    ts = np.empty(m, dtype=np.int64)
    us = np.empty(m, dtype=np.int64)
    vs = np.empty(m, dtype=np.int64)
    for i, e in enumerate(stream):
        ts[2 * i] = e.t
        us[2 * i] = e.u
        vs[2 * i] = e.v
        ts[2 * i + 1] = e.t
        us[2 * i + 1] = e.v
        vs[2 * i + 1] = e.u
    return ts, us, vs


def _group_ptr(ts: np.ndarray) -> np.ndarray:
    """CSR-style pointer over maximal equal-timestamp groups of a sorted array."""
    m = ts.shape[0]
    if m == 0:
        return np.zeros(1, dtype=np.int64)
    cuts = np.flatnonzero(ts[1:] != ts[:-1]) + 1
    return np.concatenate(
        [np.zeros(1, dtype=np.int64), cuts.astype(np.int64), np.array([m], dtype=np.int64)]
    )


@njit(cache=True)
def _plan_linear_kernel(gptr, us, vs, nv):
    """One pass over the contiguous timestamp groups.

    ``last[v] == g`` marks v as a target already collected in group g, so each
    group's distinct targets are gathered once without a global sort; only the
    (usually short) per-group list is sorted ascending, which makes the output
    identical to the reference's per-group ``np.unique``.
    """
    G = gptr.shape[0] - 1
    m = us.shape[0]
    last = np.full(nv, -1, dtype=np.int64)
    slot = np.empty(nv, dtype=np.int64)
    dirty = np.empty(m, dtype=np.int64)
    dptr = np.zeros(G + 1, dtype=np.int64)
    src_slot = np.full(m, -1, dtype=np.int64)
    total = 0
    max_dirty = 1
    for g in range(G):
        lo = gptr[g]
        hi = gptr[g + 1]
        start = total
        for e in range(lo, hi):
            v = vs[e]
            if last[v] != g:
                last[v] = g
                dirty[total] = v
                total += 1
        cnt = total - start
        if cnt <= 32:  # insertion sort: no allocation for the common short group
            for a in range(start + 1, total):
                x = dirty[a]
                b = a - 1
                while b >= start and dirty[b] > x:
                    dirty[b + 1] = dirty[b]
                    b -= 1
                dirty[b + 1] = x
        else:
            dirty[start:total] = np.sort(dirty[start:total])
        for k in range(start, total):
            slot[dirty[k]] = k
        for e in range(lo, hi):
            u = us[e]
            if last[u] == g:
                src_slot[e] = slot[u]
        dptr[g + 1] = total
        if cnt > max_dirty:
            max_dirty = cnt
    return dirty[:total].copy(), dptr, src_slot, max_dirty


def _strict_snapshot_plan(
    gptr: np.ndarray, us: np.ndarray, vs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-group touched-row (dirty) lists and per-edge snapshot slots.

    Since STRATA-v6 a linear-time compiled pass
    (:func:`_plan_linear_kernel`, O(m + sum_t |D_t| log |D_t|) time, O(n)
    extra memory) replaces the global sort of :func:`_strict_snapshot_plan_vec`
    (O(m log m)); without Numba the vectorized path is used. All three
    implementations (linear, vectorized, the pre-v4 reference loop
    :func:`_strict_snapshot_plan_ref`) return identical arrays
    (tests/test_prep_vectorized.py, tests/test_plan_linear.py).

    Returns (dirty_idx int64[Dtot], dptr int64[G+1], src_slot int64[m], max_dirty).
    """
    G = gptr.shape[0] - 1
    m = us.shape[0]
    if not _HAVE_NUMBA or G <= 0 or m == 0:
        return _strict_snapshot_plan_vec(gptr, us, vs)
    nv = int(max(int(us.max()), int(vs.max())) + 1)
    dirty_idx, dptr, src_slot, max_dirty = _plan_linear_kernel(
        np.ascontiguousarray(gptr, dtype=np.int64), np.ascontiguousarray(us, dtype=np.int64),
        np.ascontiguousarray(vs, dtype=np.int64), np.int64(nv))
    return dirty_idx, dptr, src_slot, int(max_dirty)


def _strict_snapshot_plan_vec(
    gptr: np.ndarray, us: np.ndarray, vs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-group touched-row (dirty) lists and per-edge snapshot slots.

    dirty rows of a group = the rows *written* in that group (all targets vs;
    after undirected expansion every written row is a target). A source read
    inside the group must come from the pre-group snapshot iff that source row
    is dirty in the same group; otherwise the live row still holds its
    pre-group value (touched-row invariant, identical to the sequential core).

    Vectorized since no Python loop over groups and no
    dictionaries. Each edge gets its group id; the distinct (group, target)
    pairs in (group, target) order are exactly the concatenated per-group
    ``np.unique`` lists of the reference; each source slot is a binary search
    of the edge's (group, source) key among those pairs. Array-for-array
    identical to :func:`_strict_snapshot_plan_ref` (the pre-v4 loop, kept as
    the reference; tests/test_prep_vectorized.py).

    Returns
    -------
    dirty_idx : int64[Dtot]   concatenated per-group dirty row ids
    dptr      : int64[G+1]    group boundaries into dirty_idx
    src_slot  : int64[m]      global dirty_idx position of us[e] within e's
                              group, or -1 if us[e] is not dirty there
    max_dirty : int           max per-group dirty count (>=1 for allocation)
    """
    G = gptr.shape[0] - 1
    m = us.shape[0]
    if G <= 0 or m == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(max(G, 0) + 1, dtype=np.int64),
                np.full(m, -1, dtype=np.int64), 1)
    gid = np.repeat(np.arange(G, dtype=np.int64), np.diff(gptr).astype(np.int64))
    base = np.int64(max(int(us.max()), int(vs.max())) + 1)
    tkey = gid * base + vs
    dkey = np.unique(tkey)                      # sorted distinct (group, target)
    dirty_idx = (dkey % base).astype(np.int64)
    dgid = dkey // base
    counts = np.bincount(dgid, minlength=G).astype(np.int64)
    dptr = np.zeros(G + 1, dtype=np.int64)
    np.cumsum(counts, out=dptr[1:])
    qkey = gid * base + us
    pos = np.searchsorted(dkey, qkey)
    hit = pos < dkey.shape[0]
    hit[hit] = dkey[pos[hit]] == qkey[hit]
    src_slot = np.where(hit, pos, -1).astype(np.int64)
    max_dirty = max(1, int(counts.max()))
    return dirty_idx, dptr, src_slot, max_dirty


def _strict_snapshot_plan_ref(
    gptr: np.ndarray, us: np.ndarray, vs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-group touched-row (dirty) lists and per-edge snapshot slots.

    dirty rows of a group = the rows *written* in that group (all targets vs;
    after undirected expansion every written row is a target). A source read
    inside the group must come from the pre-group snapshot iff that source row
    is dirty in the same group; otherwise the live row still holds its
    pre-group value (touched-row invariant, identical to the sequential core).

    Returns
    -------
    dirty_idx : int64[Dtot]   concatenated per-group dirty row ids
    dptr      : int64[G+1]    group boundaries into dirty_idx
    src_slot  : int64[m]      global dirty_idx position of us[e] within e's
                              group, or -1 if us[e] is not dirty there
    max_dirty : int           max per-group dirty count (>=1 for allocation)
    """
    G = gptr.shape[0] - 1
    dirty_chunks: List[np.ndarray] = []
    dptr = np.zeros(G + 1, dtype=np.int64)
    src_slot = np.full(us.shape[0], -1, dtype=np.int64)
    total = 0
    max_dirty = 1
    for g in range(G):
        lo, hi = int(gptr[g]), int(gptr[g + 1])
        tgt = np.unique(vs[lo:hi])
        dirty_chunks.append(tgt)
        pos = {int(r): total + k for k, r in enumerate(tgt)}
        for e in range(lo, hi):
            sl = pos.get(int(us[e]))
            if sl is not None:
                src_slot[e] = sl
        total += tgt.shape[0]
        dptr[g + 1] = total
        if tgt.shape[0] > max_dirty:
            max_dirty = int(tgt.shape[0])
    dirty_idx = (
        np.concatenate(dirty_chunks) if dirty_chunks else np.zeros(0, dtype=np.int64)
    ).astype(np.int64)
    return dirty_idx, dptr, src_slot, max_dirty


# --------------------------------------------------------------------------- #
# Compiled column-partitioned kernels
# --------------------------------------------------------------------------- #
@njit(parallel=True, cache=True)
def _strict_cols_kernel(R, gptr, us, vs, src_slot, dirty_idx, dptr, nchunks, max_dirty):
    W = R.shape[1]
    G = gptr.shape[0] - 1
    for c in prange(nchunks):
        lo = (c * W) // nchunks
        hi = ((c + 1) * W) // nchunks
        cw = hi - lo
        if cw <= 0:
            continue
        snap = np.empty((max_dirty, cw), dtype=np.uint64)
        for g in range(G):
            d0 = dptr[g]
            d1 = dptr[g + 1]
            # pre-group snapshot of the rows this group will write,
            # restricted to this worker's columns (touched-row form of R^-)
            for k in range(d0, d1):
                row = dirty_idx[k]
                for w in range(cw):
                    snap[k - d0, w] = R[row, lo + w]
            for e in range(gptr[g], gptr[g + 1]):
                v = vs[e]
                sl = src_slot[e]
                if sl >= 0:
                    s = sl - d0
                    for w in range(cw):
                        R[v, lo + w] |= snap[s, w]
                else:
                    u = us[e]
                    for w in range(cw):
                        R[v, lo + w] |= R[u, lo + w]


@njit(parallel=True, cache=True)
def _closure_cols_kernel(R, gptr, us, vs, nchunks):
    W = R.shape[1]
    G = gptr.shape[0] - 1
    for c in prange(nchunks):
        lo = (c * W) // nchunks
        hi = ((c + 1) * W) // nchunks
        if hi <= lo:
            continue
        for g in range(G):
            changed = True
            while changed:
                changed = False
                for e in range(gptr[g], gptr[g + 1]):
                    u = us[e]
                    v = vs[e]
                    has_new = False
                    for w in range(lo, hi):
                        if R[u, w] & ~R[v, w]:
                            has_new = True
                            break
                    if has_new:
                        for w in range(lo, hi):
                            R[v, w] |= R[u, w]
                        changed = True


# Tiled layout: each worker gathers its column slice into a private contiguous
# n x cw tile, runs the identical scan on the tile, and scatters it back once.
# The tile is allocated inside the prange body, so it is thread-private; no two
# workers ever write the same cache line of shared memory during the scan.
@njit(parallel=True, cache=True)
def _strict_tiled_kernel(R, gptr, us, vs, src_slot, dirty_idx, dptr, nchunks, max_dirty):
    n = R.shape[0]
    W = R.shape[1]
    G = gptr.shape[0] - 1
    for c in prange(nchunks):
        lo = (c * W) // nchunks
        hi = ((c + 1) * W) // nchunks
        cw = hi - lo
        if cw <= 0:
            continue
        tile = np.empty((n, cw), dtype=np.uint64)
        for r in range(n):
            for w in range(cw):
                tile[r, w] = R[r, lo + w]
        snap = np.empty((max_dirty, cw), dtype=np.uint64)
        for g in range(G):
            d0 = dptr[g]
            d1 = dptr[g + 1]
            for k in range(d0, d1):
                row = dirty_idx[k]
                for w in range(cw):
                    snap[k - d0, w] = tile[row, w]
            for e in range(gptr[g], gptr[g + 1]):
                v = vs[e]
                sl = src_slot[e]
                if sl >= 0:
                    s = sl - d0
                    for w in range(cw):
                        tile[v, w] |= snap[s, w]
                else:
                    u = us[e]
                    for w in range(cw):
                        tile[v, w] |= tile[u, w]
        for r in range(n):
            for w in range(cw):
                R[r, lo + w] = tile[r, w]


@njit(parallel=True, cache=True)
def _closure_tiled_kernel(R, gptr, us, vs, nchunks):
    n = R.shape[0]
    W = R.shape[1]
    G = gptr.shape[0] - 1
    for c in prange(nchunks):
        lo = (c * W) // nchunks
        hi = ((c + 1) * W) // nchunks
        cw = hi - lo
        if cw <= 0:
            continue
        tile = np.empty((n, cw), dtype=np.uint64)
        for r in range(n):
            for w in range(cw):
                tile[r, w] = R[r, lo + w]
        for g in range(G):
            changed = True
            while changed:
                changed = False
                for e in range(gptr[g], gptr[g + 1]):
                    u = us[e]
                    v = vs[e]
                    has_new = False
                    for w in range(cw):
                        if tile[u, w] & ~tile[v, w]:
                            has_new = True
                            break
                    if has_new:
                        for w in range(cw):
                            tile[v, w] |= tile[u, w]
                        changed = True
        for r in range(n):
            for w in range(cw):
                R[r, lo + w] = tile[r, w]


@njit(cache=True)
def _popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


@njit(parallel=True, cache=True)
def _strict_block_tiles_kernel(n, start, width, gptr, us, vs, src_slot, dirty_idx, dptr,
                               nchunks, max_dirty, sample_local, out_cols, R_out, write_back):
    """Source-blocked strict run with the tiles as the state (v28, backlog R76).

    Block sources start .. start+width-1 occupy W = ceil(width/64) word columns,
    split into ``nchunks`` chunks; each chunk builds its tile in place (identity
    bits of its own sources), applies the same group / snapshot loop as
    ``_strict_tiled_kernel``, and counts its set bits. No shared n x W state is
    gathered or scattered. ``sample_local`` lists block-local source indices whose
    columns go to ``out_cols`` (k x n, for checking); with ``write_back`` the tiles
    are also stored into ``R_out`` (n x W, tests only). Returns per-chunk bit counts.
    """
    W = (width + 63) // 64
    G = gptr.shape[0] - 1
    counts = np.zeros(nchunks, dtype=np.int64)
    for c in prange(nchunks):
        lo = (c * W) // nchunks
        hi = ((c + 1) * W) // nchunks
        cw = hi - lo
        if cw <= 0:
            continue
        tile = np.zeros((n, cw), dtype=np.uint64)
        for w in range(cw):
            for b in range(64):
                j = (lo + w) * 64 + b
                if j < width:
                    tile[start + j, w] |= np.uint64(1) << np.uint64(b)
        snap = np.empty((max_dirty, cw), dtype=np.uint64)
        for g in range(G):
            d0 = dptr[g]
            d1 = dptr[g + 1]
            for k in range(d0, d1):
                row = dirty_idx[k]
                for w in range(cw):
                    snap[k - d0, w] = tile[row, w]
            for e in range(gptr[g], gptr[g + 1]):
                v = vs[e]
                sl = src_slot[e]
                if sl >= 0:
                    s = sl - d0
                    for w in range(cw):
                        tile[v, w] |= snap[s, w]
                else:
                    u = us[e]
                    for w in range(cw):
                        tile[v, w] |= tile[u, w]
        total = np.int64(0)
        for r in range(n):
            for w in range(cw):
                total += np.int64(_popcount64(tile[r, w]))
        counts[c] = total
        for i in range(sample_local.shape[0]):
            j = sample_local[i]
            wj = j // 64
            if lo <= wj < hi:
                bit = np.uint64(j % 64)
                for r in range(n):
                    out_cols[i, r] = (tile[r, wj - lo] >> bit) & np.uint64(1)
        if write_back:
            for r in range(n):
                for w in range(cw):
                    R_out[r, lo + w] = tile[r, w]
    return counts


def run_block_tiles(prep: "PreparedStream", start: int, width: int, nchunks: int,
                    sample_local=None, return_state: bool = False):
    """Strict source block with tiles as the state: (pairs, sample columns, state).

    ``sample_local`` are block-local source indices; their columns come back as a
    bool array (k x n). ``return_state`` assembles the n x W block state (tests)."""
    if prep.mode != "strict":
        raise ValueError("run_block_tiles supports strict semantics")
    n = prep.n
    W = (int(width) + 63) // 64
    samp = np.asarray(sample_local if sample_local is not None else [], dtype=np.int64)
    cols = np.zeros((samp.shape[0], n), dtype=np.uint64)
    R_out = np.zeros((n, W) if return_state else (1, 1), dtype=np.uint64)
    counts = _strict_block_tiles_kernel(np.int64(n), np.int64(start), np.int64(width), prep.gptr,
                                        prep.us, prep.vs, prep.src_slot, prep.dirty_idx, prep.dptr,
                                        np.int64(max(1, min(int(nchunks), W))),
                                        np.int64(prep.max_dirty), samp, cols, R_out, return_state)
    return int(counts.sum()), cols.astype(np.bool_), (R_out if return_state else None)


def _set_identity_vec(R: np.ndarray, source_offset: int, n_block: int) -> None:
    """Vectorized identity: local source j (global source_offset+j) is placed at
    row source_offset+j, column bit j."""
    j = np.arange(n_block, dtype=np.uint64)
    rows = (np.arange(n_block, dtype=np.int64) + source_offset,)
    R[rows[0], (j >> np.uint64(6)).astype(np.int64)] |= _DTYPE(1) << (j & np.uint64(63))


# --------------------------------------------------------------------------- #
# Prepared stream + kernel dispatch (shared by the public API and harnesses)
# --------------------------------------------------------------------------- #
@dataclass
class PreparedStream:
    """Sequential O(m) preprocessing, computed once and shared read-only."""

    n: int
    mode: str
    gptr: np.ndarray
    us: np.ndarray
    vs: np.ndarray
    src_slot: Optional[np.ndarray] = None
    dirty_idx: Optional[np.ndarray] = None
    dptr: Optional[np.ndarray] = None
    max_dirty: int = 1
    timings: Dict[str, float] = field(default_factory=dict)
    prep_parallel: bool = False   # parallel sort / touched-row plan used
    prep_blocks: int = 1          # group blocks of the parallel plan
    prep_sort: str = "sequential"  # adaptive / parallel / lexsort / sequential

    @property
    def num_edges(self) -> int:
        return int(self.us.shape[0])

    @property
    def dirty_total(self) -> int:
        return int(self.dirty_idx.shape[0]) if self.dirty_idx is not None else 0


def prepare_stream(
    n: int,
    edges: Iterable[TemporalEdge | Tuple[int, int, int]],
    *,
    directed: bool = True,
    same_time_mode: str = "strict",
) -> PreparedStream:
    """Sort, group, and (strict) build the touched-row snapshot plan."""
    if same_time_mode not in {"strict", "closure"}:
        raise ValueError("same_time_mode must be 'strict' or 'closure'")
    import time as _time

    t0 = _time.perf_counter()
    stream = sort_and_validate_edges(n, edges)
    t1 = _time.perf_counter()
    ts, us, vs = _flatten_stream(n, stream, directed)
    t2 = _time.perf_counter()
    gptr = _group_ptr(ts)
    t3 = _time.perf_counter()
    prep = PreparedStream(n=n, mode=same_time_mode, gptr=gptr, us=us, vs=vs)
    if same_time_mode == "strict":
        prep.dirty_idx, prep.dptr, prep.src_slot, prep.max_dirty = _strict_snapshot_plan(gptr, us, vs)
    t4 = _time.perf_counter()
    # wall-clock of each sequential preprocessing phase (the Amdahl term, R12);
    # recorded by the harnesses next to the kernel time, never compared across machines
    prep.timings = {"sort_validate_s": t1 - t0, "flatten_s": t2 - t1,
                    "group_boundaries_s": t3 - t2, "snapshot_plan_s": t4 - t3,
                    "total_s": t4 - t0}
    return prep


def _is_edge_arrays(edges) -> bool:
    return (isinstance(edges, tuple) and len(edges) == 3
            and all(isinstance(a, np.ndarray) for a in edges))


def prepare_stream_arrays(
    n: int,
    ts: np.ndarray,
    us: np.ndarray,
    vs: np.ndarray,
    *,
    directed: bool = True,
    same_time_mode: str = "strict",
    prep_mode: str = "auto",
    prep_threads: Optional[int] = None,
) -> PreparedStream:
    """Array-native preprocessing.

    ``prep_mode`` = ``auto`` / ``on`` / ``off`` selects
    the parallel sort and touched-row plan of :mod:`temporal_strata.parallel_prep`
    (``auto``: only above ``PREP_PARALLEL_MIN_EDGES`` edges with more than one
    thread). Outputs are identical in every mode; ``prep_threads`` (default: the
    active Numba thread count) sets the number of sort chunks and group blocks.

    Same output as :func:`prepare_stream` on the equivalent edge list (array
    for array; tests/test_array_path.py), without TemporalEdge objects:
    vectorized validation + lexicographic sort, strided undirected expansion,
    vectorized group boundaries and touched-row plan. Phase timings use the
    same keys as :func:`prepare_stream`.
    """
    import time as _time

    from .stream import sort_validate_arrays

    if same_time_mode not in {"strict", "closure"}:
        raise ValueError("same_time_mode must be 'strict' or 'closure'")
    from .parallel_prep import (resolve_prep, sort_validate_arrays_parallel,
                                strict_snapshot_plan_parallel)

    k = int(prep_threads) if prep_threads else (int(numba.get_num_threads()) if _HAVE_NUMBA else 1)
    use_par = resolve_prep(prep_mode, int(np.asarray(us).shape[0]), k)
    t0 = _time.perf_counter()
    sort_info = {"sort_path": "sequential"}
    if use_par:
        ts1, us1, vs1 = sort_validate_arrays_parallel(n, ts, us, vs, k, info=sort_info)
    else:
        ts1, us1, vs1 = sort_validate_arrays(n, ts, us, vs)
    t1 = _time.perf_counter()
    if directed:
        tsf, usf, vsf = ts1, us1, vs1
    else:
        m = ts1.shape[0]
        tsf = np.empty(2 * m, dtype=np.int64)
        usf = np.empty(2 * m, dtype=np.int64)
        vsf = np.empty(2 * m, dtype=np.int64)
        tsf[0::2] = ts1
        tsf[1::2] = ts1
        usf[0::2] = us1
        usf[1::2] = vs1
        vsf[0::2] = vs1
        vsf[1::2] = us1
    t2 = _time.perf_counter()
    gptr = _group_ptr(tsf)
    t3 = _time.perf_counter()
    prep = PreparedStream(n=n, mode=same_time_mode, gptr=gptr, us=usf, vs=vsf,
                          prep_parallel=use_par, prep_blocks=k if use_par else 1,
                          prep_sort=sort_info["sort_path"])
    if same_time_mode == "strict":
        if use_par:
            prep.dirty_idx, prep.dptr, prep.src_slot, prep.max_dirty = \
                strict_snapshot_plan_parallel(gptr, usf, vsf, k)
        else:
            prep.dirty_idx, prep.dptr, prep.src_slot, prep.max_dirty = _strict_snapshot_plan(gptr, usf, vsf)
    t4 = _time.perf_counter()
    prep.timings = {"sort_validate_s": t1 - t0, "flatten_s": t2 - t1,
                    "group_boundaries_s": t3 - t2, "snapshot_plan_s": t4 - t3,
                    "total_s": t4 - t0}
    return prep


def run_prepared(R: np.ndarray, prep: PreparedStream, plan: ParallelPlan) -> None:
    """Advance the packed state ``R`` (n x W, identity already set) in place."""
    if not _HAVE_NUMBA:
        raise RuntimeError("run_prepared requires numba")
    nch = np.int64(plan.nchunks)
    ctx = (numba.parallel_chunksize(1)
           if plan.schedule == "dynamic" and hasattr(numba, "parallel_chunksize")
           else nullcontext())
    with ctx:
        if prep.mode == "strict":
            kern = _strict_tiled_kernel if plan.resolved_layout == "tiled" else _strict_cols_kernel
            kern(R, prep.gptr, prep.us, prep.vs, prep.src_slot, prep.dirty_idx,
                 prep.dptr, nch, np.int64(prep.max_dirty))
        else:
            kern = _closure_tiled_kernel if plan.resolved_layout == "tiled" else _closure_cols_kernel
            kern(R, prep.gptr, prep.us, prep.vs, nch)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def temporal_strata_parallel(
    n: int,
    edges: Iterable[TemporalEdge | Tuple[int, int, int]],
    *,
    directed: bool = True,
    same_time_mode: str = "strict",
    block_bits: int | None = None,
    num_threads: int | None = None,
    chunks: int | None = None,
    layout: str = "tiled",
    schedule: str = "static",
    oversub: int = DEFAULT_OVERSUB,
    plans_out: Optional[List[ParallelPlan]] = None,
) -> np.ndarray:
    """Column-partitioned thread-parallel packed temporal reachability.

    Drop-in parallel counterpart of
    :func:`temporal_strata.packed_reachability.temporal_strata_packed`;
    returns the identical packed ``(n, ceil(n/64))`` uint64 matrix for every
    ``chunks``, ``num_threads``, ``block_bits``, ``layout`` and ``schedule``
    (partition determinacy; verified in tests/test_parallel.py and
    tests/test_parallel_layout.py).

    Parameters
    ----------
    n, edges, directed, same_time_mode, block_bits:
        As in the sequential core. Tuples are read as ``(t, u, v)``.
    num_threads:
        Active Numba worker threads; clamped to :func:`max_threads`.
        ``None`` uses the full pool.
    chunks:
        Number of column chunks the ``W`` (or per-block ``W_beta``) words are
        split into. ``None`` lets :func:`resolve_plan` choose. Any value >= 1 is
        valid and does not change the result.
    layout, schedule, oversub:
        Realization choices (see the module docstring and :func:`resolve_plan`).
        Default layout is ``"tiled"`` since on the Apple M5 Max the
        tiled layout was faster than row-major at every measured word width and
        thread count, including one thread (results/real/layout_crossover_p*.json),
        so the analytical ``layout="auto"`` rule was not supported; ``auto`` and
        ``rowmajor`` remain available for reproducing earlier runs.
    edges:
        A TemporalEdge / tuple list, or a ``(ts, us, vs)`` tuple of numpy arrays
        (array-native path, backlog R17; same result).
    plans_out:
        If a list is given, the :class:`ParallelPlan` of every kernel launch
        (one per source block) is appended to it.
    """
    if same_time_mode not in {"strict", "closure"}:
        raise ValueError("same_time_mode must be 'strict' or 'closure'")
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    if not _HAVE_NUMBA:
        raise RuntimeError(
            "temporal_strata_parallel requires numba; "
            "use temporal_strata_packed for the interpreted sequential core"
        )

    eff_threads = set_threads(num_threads) if num_threads is not None else set_threads(max_threads())
    if _is_edge_arrays(edges):
        prep = prepare_stream_arrays(n, *edges, directed=directed, same_time_mode=same_time_mode)
    else:
        prep = prepare_stream(n, edges, directed=directed, same_time_mode=same_time_mode)

    def _run(Rloc: np.ndarray) -> None:
        plan = resolve_plan(Rloc.shape[1], eff_threads, layout=layout,
                            schedule=schedule, chunks=chunks, oversub=oversub)
        if plans_out is not None:
            plans_out.append(plan)
        run_prepared(Rloc, prep, plan)

    if block_bits is None or block_bits >= n:
        R = _empty_state(n, n)
        _set_identity_vec(R, 0, n)
        _run(R)
        return R

    full = _empty_state(n, n)
    for start in range(0, n, block_bits):
        n_block = min(block_bits, n - start)
        Rb = _empty_state(n, n_block)
        _set_identity_vec(Rb, start, n_block)
        _run(Rb)
        _scatter_block(full, Rb, start, n_block)
    return full


__all__ = [
    "temporal_strata_parallel",
    "max_threads",
    "set_threads",
    "LAYOUTS",
    "SCHEDULES",
    "CHUNK_POLICIES",
    "ParallelPlan",
    "resolve_plan",
    "cache_line_bytes",
    "cache_line_words",
    "machine_info",
    "PreparedStream",
    "prepare_stream",
    "prepare_stream_arrays",
    "run_prepared",
]
