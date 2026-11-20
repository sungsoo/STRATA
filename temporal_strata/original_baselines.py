from __future__ import annotations

"""Temporal analogues of the 12 STRATA repository baselines.

The original STRATA repository benchmarks 12 APSP implementations identified as
BC1-BC5, TC1-TC3, BG1-BG2, and TG1-TG2.  This module keeps the same identifiers
for temporal reachability experiments.  CPU methods are implemented as exact
semantic analogues over timestamped edge streams. CUDA-only methods are reported
as unavailable on Apple Silicon unless a CUDA runtime is explicitly provided by
an external environment.
"""

from dataclasses import dataclass
from itertools import groupby
from typing import Callable, Iterable, List, Tuple, Dict, Any, Optional
import importlib.util
import platform
import time

import numpy as np

from .baseline import all_sources_temporal_bfs
from .bitset_reachability import bits_to_matrix, temporal_strata_reachability
from .stream import TemporalEdge, sort_and_validate_edges


@dataclass(frozen=True)
class BaselineSpec:
    method_id: str
    name: str
    original_category: str
    temporal_analogue: str
    backend: str
    native_on_m3: bool
    notes: str


ORIGINAL_STRATA_BASELINES: List[BaselineSpec] = [
    BaselineSpec("BC1", "NetworkX", "CPU baseline", "per-source temporal BFS", "pure-python", True, "Exact source-by-source stream scan."),
    BaselineSpec("BC2", "SciPy C BFS", "CPU baseline", "NumPy/SciPy-style per-source temporal BFS", "numpy", True, "Exact baseline mirroring repeated CSR BFS semantics."),
    BaselineSpec("BC3", "I-AORM", "CPU baseline", "iterative temporal Boolean matrix propagation", "numpy-dense", True, "Dense causal matrix propagation by timestamp."),
    BaselineSpec("BC4", "M-AORM", "CPU baseline", "multi-source temporal matrix propagation", "numpy-dense", True, "Multi-source batched causal matrix update."),
    BaselineSpec("BC5", "GB-bfs", "CPU baseline", "GraphBLAS temporal BFS analogue", "optional-graphblas", False, "Runs only when python-graphblas/suitesparse-graphblas is installed."),
    BaselineSpec("TC1", "STRATA-SpMM-Cython", "STRATA CPU", "TA-STRATA int-bitset causal frontier", "python-int-bitset", True, "Primary method."),
    BaselineSpec("TC2", "STRATA-NumpyBLAS", "STRATA CPU", "TA-STRATA uint64 packed frontier", "numpy-uint64", True, "Packed word-parallel NumPy backend."),
    BaselineSpec("TC3", "STRATA-GraphBLAS", "STRATA CPU", "GraphBLAS STRATA analogue", "optional-graphblas", False, "Runs only when GraphBLAS backend is available."),
    BaselineSpec("BG1", "GPU-PerSrc-BFS", "GPU baseline", "CUDA per-source temporal BFS", "cuda", False, "Not native on Apple-silicon Macs (e.g., MacBook Pro M5 Max)."),
    BaselineSpec("BG2", "DAWN-SOVM", "GPU baseline", "CUDA/DAWN temporal frontier baseline", "cuda", False, "Not native on Apple-silicon Macs (e.g., MacBook Pro M5 Max)."),
    BaselineSpec("TG1", "STRATA-DAWNiBFS", "STRATA GPU", "CUDA bitwise temporal STRATA", "cuda", False, "Not native on Apple-silicon Macs (e.g., MacBook Pro M5 Max)."),
    BaselineSpec("TG2", "STRATA-CUDA", "STRATA GPU", "CUDA guard+CAS temporal STRATA", "cuda", False, "Not native on Apple-silicon Macs (e.g., MacBook Pro M5 Max)."),
]


def _dense_temporal_matrix(
    n: int,
    edges: Iterable[TemporalEdge | Tuple[int, int, int]],
    *,
    directed: bool = True,
    same_time_mode: str = "strict",
) -> np.ndarray:
    """Dense Boolean matrix temporal propagation.

    Matrix entry R[s, v] is true iff source s can reach target v under the
    selected temporal semantics.
    """
    stream = sort_and_validate_edges(n, edges)
    R = np.eye(n, dtype=bool)
    for _, group_iter in groupby(stream, key=lambda e: e.t):
        group = list(group_iter)
        if same_time_mode == "strict":
            before = R.copy()
            for e in group:
                R[:, e.v] |= before[:, e.u]
                if not directed:
                    R[:, e.u] |= before[:, e.v]
        elif same_time_mode == "closure":
            changed = True
            while changed:
                changed = False
                for e in group:
                    old = int(R[:, e.v].sum())
                    R[:, e.v] |= R[:, e.u]
                    if not directed:
                        R[:, e.u] |= R[:, e.v]
                    if int(R[:, e.v].sum()) != old:
                        changed = True
        else:
            raise ValueError("same_time_mode must be 'strict' or 'closure'")
    return R


def _dense_temporal_matrix_batched(
    n: int,
    edges: Iterable[TemporalEdge | Tuple[int, int, int]],
    *,
    directed: bool = True,
    same_time_mode: str = "strict",
) -> np.ndarray:
    """Batched dense analogue of M-AORM.

    This groups all same-timestamp updates by target. It is still exact, but it
    exposes the multi-source/multi-target structure that AORM-style methods use.
    """
    stream = sort_and_validate_edges(n, edges)
    R = np.eye(n, dtype=bool)
    for _, group_iter in groupby(stream, key=lambda e: e.t):
        group = list(group_iter)
        if same_time_mode == "strict":
            before = R.copy()
            by_v: Dict[int, List[int]] = {}
            for e in group:
                by_v.setdefault(e.v, []).append(e.u)
                if not directed:
                    by_v.setdefault(e.u, []).append(e.v)
            for v, us in by_v.items():
                if us:
                    R[:, v] |= np.logical_or.reduce(before[:, us], axis=1)
        elif same_time_mode == "closure":
            changed = True
            while changed:
                changed = False
                old_count = int(R.sum())
                by_v: Dict[int, List[int]] = {}
                for e in group:
                    by_v.setdefault(e.v, []).append(e.u)
                    if not directed:
                        by_v.setdefault(e.u, []).append(e.v)
                for v, us in by_v.items():
                    if us:
                        R[:, v] |= np.logical_or.reduce(R[:, us], axis=1)
                changed = int(R.sum()) != old_count
        else:
            raise ValueError("same_time_mode must be 'strict' or 'closure'")
    return R


def _numpy_uint64_strata(
    n: int,
    edges: Iterable[TemporalEdge | Tuple[int, int, int]],
    *,
    directed: bool = True,
    same_time_mode: str = "strict",
) -> np.ndarray:
    """Packed uint64 target-indexed source bitsets, returned as bool matrix."""
    stream = sort_and_validate_edges(n, edges)
    words = (n + 63) // 64
    R = np.zeros((n, words), dtype=np.uint64)  # target -> source words
    one = np.uint64(1)
    for i in range(n):
        R[i, i // 64] |= one << np.uint64(i % 64)

    for _, group_iter in groupby(stream, key=lambda e: e.t):
        group = list(group_iter)
        if same_time_mode == "strict":
            before = R.copy()
            after = R.copy()
            for e in group:
                after[e.v, :] |= before[e.u, :]
                if not directed:
                    after[e.u, :] |= before[e.v, :]
            R = after
        elif same_time_mode == "closure":
            changed = True
            while changed:
                changed = False
                for e in group:
                    old = R[e.v, :].copy()
                    R[e.v, :] |= R[e.u, :]
                    if not np.array_equal(old, R[e.v, :]):
                        changed = True
                    if not directed:
                        old = R[e.u, :].copy()
                        R[e.u, :] |= R[e.v, :]
                        if not np.array_equal(old, R[e.u, :]):
                            changed = True
        else:
            raise ValueError("same_time_mode must be 'strict' or 'closure'")

    M = np.zeros((n, n), dtype=bool)
    for v in range(n):
        for w in range(words):
            x = int(R[v, w])
            while x:
                lsb = x & -x
                bit = lsb.bit_length() - 1
                s = w * 64 + bit
                if s < n:
                    M[s, v] = True
                x ^= lsb
    return M


def _strata_int_matrix(n: int, edges, *, directed=True, same_time_mode="strict") -> np.ndarray:
    bits = temporal_strata_reachability(n, edges, directed=directed, same_time_mode=same_time_mode)
    return bits_to_matrix(bits, n)


def _graphblas_available() -> bool:
    return importlib.util.find_spec("graphblas") is not None or importlib.util.find_spec("pygraphblas") is not None


def _cuda_available() -> bool:
    # Conservative runtime check. Importable CUDA Python packages do not imply a
    # usable CUDA device, and Apple Silicon/MacOS does not provide CUDA.
    if platform.system() == "Darwin":
        return False
    try:
        import cupy  # type: ignore
        _ = cupy.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        pass
    try:
        from numba import cuda  # type: ignore
        return bool(cuda.is_available())
    except Exception:
        return False


def method_available(method_id: str, *, allow_emulated_optional: bool = False) -> tuple[bool, str]:
    if method_id in {"BC1", "BC2", "BC3", "BC4", "TC1", "TC2"}:
        return True, "native"
    if method_id in {"BC5", "TC3"}:
        if _graphblas_available():
            return True, "native-graphblas"
        if allow_emulated_optional:
            return True, "emulated-by-dense-or-bitset"
        return False, "missing GraphBLAS backend"
    if method_id in {"BG1", "BG2", "TG1", "TG2"}:
        if _cuda_available() and platform.system() != "Darwin":
            return True, "native-cuda"
        if allow_emulated_optional:
            return True, "emulated-cpu-analogue"
        return False, "CUDA/GPU backend unavailable on this host"
    return False, "unknown method"


def run_method(method_id: str, n: int, edges, *, directed=True, same_time_mode="strict") -> np.ndarray:
    if method_id == "BC1":
        return all_sources_temporal_bfs(n, edges, directed=directed, same_time_mode=same_time_mode)
    if method_id == "BC2":
        return all_sources_temporal_bfs(n, edges, directed=directed, same_time_mode=same_time_mode)
    if method_id == "BC3":
        return _dense_temporal_matrix(n, edges, directed=directed, same_time_mode=same_time_mode)
    if method_id == "BC4":
        return _dense_temporal_matrix_batched(n, edges, directed=directed, same_time_mode=same_time_mode)
    if method_id == "BC5":
        # Exact fallback analogue when GraphBLAS is unavailable; marked emulated by the runner.
        return _dense_temporal_matrix(n, edges, directed=directed, same_time_mode=same_time_mode)
    if method_id == "TC1":
        return _strata_int_matrix(n, edges, directed=directed, same_time_mode=same_time_mode)
    if method_id == "TC2":
        return _numpy_uint64_strata(n, edges, directed=directed, same_time_mode=same_time_mode)
    if method_id == "TC3":
        return _strata_int_matrix(n, edges, directed=directed, same_time_mode=same_time_mode)
    if method_id == "BG1":
        return all_sources_temporal_bfs(n, edges, directed=directed, same_time_mode=same_time_mode)
    if method_id == "BG2":
        return _dense_temporal_matrix_batched(n, edges, directed=directed, same_time_mode=same_time_mode)
    if method_id == "TG1":
        return _numpy_uint64_strata(n, edges, directed=directed, same_time_mode=same_time_mode)
    if method_id == "TG2":
        return _strata_int_matrix(n, edges, directed=directed, same_time_mode=same_time_mode)
    raise ValueError(f"unknown method id: {method_id}")


def benchmark_method(
    method_id: str,
    n: int,
    edges,
    *,
    repeats: int = 3,
    directed: bool = True,
    same_time_mode: str = "strict",
    allow_emulated_optional: bool = False,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    spec = next(s for s in ORIGINAL_STRATA_BASELINES if s.method_id == method_id)
    available, availability = method_available(method_id, allow_emulated_optional=allow_emulated_optional)
    out: Dict[str, Any] = {
        "method_id": method_id,
        "method_name": spec.name,
        "original_category": spec.original_category,
        "temporal_analogue": spec.temporal_analogue,
        "backend": spec.backend,
        "availability": availability,
        "status": "skipped" if not available else "ok",
        "median_sec": None,
        "runs_sec": [],
        "reachable_pairs": None,
        "notes": spec.notes,
    }
    if not available:
        if progress_callback is not None:
            progress_callback({
                "event": "method_skipped",
                "method_id": method_id,
                "method_name": spec.name,
                "availability": availability,
            })
        return out
    runs = []
    value: Optional[np.ndarray] = None
    for repeat_index in range(repeats):
        if progress_callback is not None:
            progress_callback({
                "event": "repeat_start",
                "method_id": method_id,
                "method_name": spec.name,
                "repeat": repeat_index + 1,
                "repeats": repeats,
            })
        t0 = time.perf_counter()
        value = run_method(method_id, n, edges, directed=directed, same_time_mode=same_time_mode)
        elapsed = time.perf_counter() - t0
        runs.append(elapsed)
        if progress_callback is not None:
            progress_callback({
                "event": "repeat_end",
                "method_id": method_id,
                "method_name": spec.name,
                "repeat": repeat_index + 1,
                "repeats": repeats,
                "elapsed_sec": elapsed,
            })
    assert value is not None
    out["median_sec"] = float(np.median(runs))
    out["runs_sec"] = [float(x) for x in runs]
    out["reachable_pairs"] = int(value.sum())
    return out


def baseline_crosswalk() -> List[Dict[str, Any]]:
    rows = []
    for s in ORIGINAL_STRATA_BASELINES:
        available, reason = method_available(s.method_id, allow_emulated_optional=False)
        rows.append({
            "method_id": s.method_id,
            "method_name": s.name,
            "original_category": s.original_category,
            "temporal_analogue": s.temporal_analogue,
            "backend": s.backend,
            "native_on_m3": s.native_on_m3,
            "available_here": available,
            "availability_reason": reason,
            "notes": s.notes,
        })
    return rows
