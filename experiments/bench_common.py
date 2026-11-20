"""Shared helpers for the measurement harnesses.

* Dataset selection: the three real-timestamp datasets plus synthetic streams
  given as ``--large n:m:groups[:seed]`` (graphs whose word width W is
  several times the core count, so column parallelism is not starved).
* Correctness references: every timed configuration is compared bit-for-bit
  against a reference state before any timing. ``interpreted`` is the NumPy
  packed core (the paper's reference); ``compiled`` is the independent Numba
  full-copy-snapshot kernel of experiments/numba_kernel.py, used when the
  interpreted core would take too long (large synthetic streams). ``auto``
  picks interpreted when ``m * W`` is at most ``INTERPRETED_LIMIT``.
* Timing: warm-up + best-of-R; nothing here compares machines.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.io import load_standard_edges  # noqa: E402
from temporal_strata.packed_reachability import (  # noqa: E402
    temporal_strata_packed,
    words_for,
)
from temporal_strata.io import load_standard_arrays  # noqa: E402
from temporal_strata.stream import (  # noqa: E402
    generate_temporal_arrays,
    generate_temporal_stream,
)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "datasets_real_temporal" / "standard"
REAL_SETS = ["CollegeMsg", "BitcoinOTC", "BitcoinOTCpos"]
INTERPRETED_LIMIT = 3e8  # m * W word-operations handled by the interpreted reference
REFERENCES = ("auto", "interpreted", "compiled", "sampled")


def load_real(name: str):
    n, edges, _meta = load_standard_edges(DATA / f"{name}.txt")
    return n, edges


def parse_large(spec: str, materialize: bool = True):
    """``n:m:groups[:seed]`` -> (label, n, edges) synthetic directed stream.

    With ``materialize=False`` the edges are not generated (``None``); used by
    ``--plan-only``, which needs only ``n``.
    """
    parts = [int(float(x)) for x in spec.split(":")]
    if len(parts) not in (3, 4) or min(parts[:3]) <= 0:
        raise ValueError(f"--large expects n:m:groups[:seed], got {spec!r}")
    n, m, groups = parts[:3]
    seed = parts[3] if len(parts) == 4 else 7
    edges = (generate_temporal_stream(n, m, timestamps=groups, directed=True, seed=seed)
             if materialize else None)
    return f"synthetic-PA(n={n},m={m},G={groups},seed={seed})", n, edges


def select_datasets(real: List[str], large: List[str], include_default_synth: bool,
                    materialize: bool = True):
    out: List[Tuple[str, int, list]] = []
    for name in real:
        p = DATA / f"{name}.txt"
        if p.exists():
            n, e = load_real(name)
            out.append((name, n, e))
        else:
            print(f"missing {p}; skipping")
    if include_default_synth:
        out.append(parse_large("8000:1000000:200:7", materialize))
    for spec in large:
        out.append(parse_large(spec, materialize))
    return out


# ---------------- array-native datasets ------------ #
def _is_arrays(x) -> bool:
    return isinstance(x, tuple) and len(x) == 3 and all(isinstance(a, np.ndarray) for a in x)


def num_edges(edges) -> int:
    return int(edges[0].shape[0]) if _is_arrays(edges) else len(edges)


def as_tuples(edges):
    """(t, u, v) tuples for the references, from arrays or edge objects."""
    if _is_arrays(edges):
        t, u, v = edges
        return list(zip(t.tolist(), u.tolist(), v.tolist()))
    return [(e.t, e.u, e.v) if hasattr(e, "t") else tuple(e) for e in edges]


def parse_large_arrays(spec: str, materialize: bool = True):
    """Like :func:`parse_large`, returning ``(t, u, v)`` arrays (same stream)."""
    parts = [int(float(x)) for x in spec.split(":")]
    if len(parts) not in (3, 4) or min(parts[:3]) <= 0:
        raise ValueError(f"--large expects n:m:groups[:seed], got {spec!r}")
    n, m, groups = parts[:3]
    seed = parts[3] if len(parts) == 4 else 7
    arrays = (generate_temporal_arrays(n, m, timestamps=groups, directed=True, seed=seed)
              if materialize else None)
    return f"synthetic-PA(n={n},m={m},G={groups},seed={seed})", n, arrays


def select_datasets_arrays(real: List[str], large: List[str], include_default_synth: bool,
                           materialize: bool = True, extra: Optional[List[str]] = None):
    """[(name, n, (t, u, v) arrays in file / generator order, load_seconds)]."""
    out = []
    for name in real:
        p = DATA / f"{name}.txt"
        if not p.exists():
            print(f"missing {p}; skipping")
            continue
        t0 = time.perf_counter()
        n, t, u, v, _meta = load_standard_arrays(p)
        out.append((name, n, (t, u, v), time.perf_counter() - t0))
    for name in (extra or []):   # fetched public temporal graphs
        p = DATA / f"{name}.txt"
        if not p.exists():
            raise SystemExit(f"missing {p}; run experiments/fetch_real_temporal.py --names {name}")
        t0 = time.perf_counter()
        n, t, u, v, _meta = load_standard_arrays(p)
        out.append((name, n, (t, u, v), time.perf_counter() - t0))
    specs = (["8000:1000000:200:7"] if include_default_synth else []) + list(large)
    for spec in specs:
        t0 = time.perf_counter()
        label, n, arrays = parse_large_arrays(spec, materialize)
        out.append((label, n, arrays, time.perf_counter() - t0))
    return out


def choose_reference(n: int, m: int, kind: str = "auto") -> str:
    if kind not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}")
    if kind != "auto":
        return kind
    return "interpreted" if m * words_for(n) <= INTERPRETED_LIMIT else "compiled"


def _bfs_sources_strict_py(n, gptr, us, vs, sources):
    """Independent per-source strict temporal BFS for a few sources (bool k x n)."""
    k = len(sources)
    reached = np.zeros((k, n), dtype=np.bool_)
    newv = np.empty(max(1, int(us.shape[0])), dtype=np.int64)
    for i in range(k):
        r = reached[i]
        r[sources[i]] = True
        for g in range(gptr.shape[0] - 1):
            c = 0
            for e in range(gptr[g], gptr[g + 1]):
                if r[us[e]] and not r[vs[e]]:
                    newv[c] = vs[e]
                    c += 1
            for j in range(c):        # strict: set only after the whole group read
                r[newv[j]] = True
    return reached


try:
    from numba import njit as _njit
    _bfs_sources_strict = _njit(cache=True)(_bfs_sources_strict_py)
except Exception:  # pragma: no cover
    _bfs_sources_strict = _bfs_sources_strict_py


def sampled_reference(n, edges, k=64, seed=5):
    """Reference for graphs too large for the other references.

    The full state comes from the sequential compiled core (row-major, one
    chunk, touched-row snapshot); before it is trusted, the columns of ``k``
    sampled sources are checked bit for bit against an independent per-source
    strict temporal BFS. Returns (state, info)."""
    from temporal_strata.parallel import (_set_identity_vec, prepare_stream_arrays,
                                          resolve_plan, run_prepared)

    prep = prepare_stream_arrays(n, *edges, directed=True, same_time_mode="strict",
                                 prep_mode="off")
    W = words_for(n)
    R = np.zeros((n, W), dtype=np.uint64)
    _set_identity_vec(R, 0, n)
    run_prepared(R, prep, resolve_plan(W, 1, layout="rowmajor", schedule="static", chunks=1))
    rng = np.random.default_rng(seed)
    src = np.sort(rng.choice(n, size=min(k, n), replace=False)).astype(np.int64)
    bfs = _bfs_sources_strict(n, prep.gptr, prep.us, prep.vs, src)
    for i, s_ in enumerate(src):
        col = ((R[:, s_ // 64] >> np.uint64(s_ % 64)) & np.uint64(1)).astype(np.bool_)
        if not np.array_equal(col, bfs[i]):
            raise AssertionError(f"sequential core disagrees with per-source BFS at source {s_}")
    return R, {"sampled_sources": int(len(src)), "seed": seed}


def reference_state(n: int, edges, mode: str = "strict", kind: str = "auto"):
    """(packed state, reference kind actually used)."""
    used = choose_reference(n, num_edges(edges), kind)
    if used == "sampled":
        if mode != "strict":
            raise ValueError("the sampled reference covers strict semantics")
        R, _info = sampled_reference(n, edges)
        return R, used
    if used == "interpreted":
        src = as_tuples(edges) if _is_arrays(edges) else edges
        return temporal_strata_packed(n, src, directed=True, same_time_mode=mode), used
    from experiments.numba_kernel import strata_closure_numba, strata_strict_numba

    tuples = as_tuples(edges)
    fn = strata_strict_numba if mode == "strict" else strata_closure_numba
    R, _pairs = fn(n, tuples)
    return R, used


def samples_of(fn: Callable[[], object], repeats: int) -> List[float]:
    """Warm-up, then ``repeats`` timed calls; every sample is kept."""
    fn()  # warm-up (JIT, allocation)
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def best_of(fn: Callable[[], object], repeats: int) -> float:
    return min(samples_of(fn, repeats))


__all__ = ["DATA", "REAL_SETS", "REFERENCES", "INTERPRETED_LIMIT", "load_real",
           "parse_large", "parse_large_arrays", "select_datasets", "select_datasets_arrays",
           "num_edges", "as_tuples", "choose_reference", "reference_state", "best_of",
           "samples_of", "sampled_reference"]
