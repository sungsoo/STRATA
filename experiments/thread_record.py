"""Actual thread counts of every parallel runtime in this process.

Each timing harness stores ``thread_record()`` next to its results, so that
``m5_numbers.py`` can refuse to print a table labeled "one thread" unless every
runtime that took part really ran with one thread. Environment variables alone
are not enough: ``NUMBA_NUM_THREADS=1`` does not limit GraphBLAS's OpenMP pool
(found in v10, backlog R36).
"""
from __future__ import annotations

import sys


def thread_record() -> dict:
    rec = {"numba": None, "graphblas": None, "threadpools": []}
    try:
        import numba
        rec["numba"] = int(numba.get_num_threads())
    except Exception:
        pass
    gb = sys.modules.get("graphblas")
    if gb is not None:
        try:
            rec["graphblas"] = int(gb.ss.config["nthreads"])
        except Exception:
            pass
    try:
        from threadpoolctl import threadpool_info
        rec["threadpools"] = [{"user_api": p.get("user_api"), "internal_api": p.get("internal_api"),
                               "num_threads": p.get("num_threads")} for p in threadpool_info()]
    except Exception:
        pass
    return rec


def check_one_thread(rec, name, need_graphblas=False):
    """Raise SystemExit unless the record shows one Numba (and GraphBLAS) thread."""
    if not rec:
        raise SystemExit(f"{name}: no thread record; re-run the single-thread step")
    if rec.get("numba") != 1:
        raise SystemExit(f"{name}: Numba ran with {rec.get('numba')} threads, not 1")
    if need_graphblas and rec.get("graphblas") != 1:
        raise SystemExit(f"{name}: GraphBLAS ran with {rec.get('graphblas')} threads, not 1")
