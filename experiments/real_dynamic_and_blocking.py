#!/usr/bin/env python3
"""Real-timestamp dynamic-update and source-blocking experiments.

Replaces the ego-Facebook-with-assigned-timestamps dynamic study with genuinely
timestamped data (CollegeMsg).  Three things are measured, all bit-for-bit
against a from-scratch recomputation:

  1. Monotone insertion: append later-timestamp edge batches by suffix
     propagation on the resident state (Prop. insertion) and compare to full
     recomputation.
  2. Sliding window: recompute successive windows and confirm each is
     bit-identical to a from-scratch window computation.
  3. Deletion repair: replay the affected suffix from a pre-deletion snapshot
     (Prop. deletion) at early / middle / late cut points.

Also verifies source blocking is bit-identical across block widths on a real
graph (Bitcoin-OTC), reporting the resident-word footprint per width.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.io import load_standard_edges
from temporal_strata.packed_reachability import (
    temporal_strata_packed,
    reachable_pairs,
    words_for,
)
from temporal_strata.stream import TemporalEdge, sort_and_validate_edges
try:
    from thread_record import thread_record  # run as a script
except ImportError:  # imported as experiments.<module>
    from experiments.thread_record import thread_record


def _run_prefix(n, stream):
    return temporal_strata_packed(n, stream, same_time_mode="strict")


def insertion_experiment(n, stream, n_batches=5):
    """Append edge batches; suffix-propagate vs full recompute, bit-identical."""
    total = len(stream)
    cut = total // 2
    base_stream = stream[:cut]
    R = _run_prefix(n, base_stream)  # resident state after prefix
    batch_edges = stream[cut:]
    bsz = max(1, len(batch_edges) // n_batches)
    out = []
    cur = list(base_stream)
    Rres = R.copy()
    for b in range(n_batches):
        lo = b * bsz
        hi = total - cut if b == n_batches - 1 else (b + 1) * bsz
        batch = batch_edges[lo:hi]
        if not batch:
            break
        cur = cur + batch
        # full recompute
        t0 = time.perf_counter()
        R_full = _run_prefix(n, cur)
        t_full = time.perf_counter() - t0
        # incremental: propagate only the appended batch on resident state
        t0 = time.perf_counter()
        Rres = _propagate_suffix(n, Rres, batch)
        t_inc = time.perf_counter() - t0
        ok = bool(np.array_equal(Rres, R_full))
        out.append(
            {
                "batch": b + 1,
                "batch_edges": len(batch),
                "full_recompute_s": round(t_full, 4),
                "incremental_s": round(t_inc, 4),
                "speedup": round(t_full / max(t_inc, 1e-9), 1),
                "bit_identical": ok,
                "pairs": reachable_pairs(R_full, n),
            }
        )
        assert ok, f"insertion batch {b+1} bit mismatch"
    return out


def _propagate_suffix(n, R, batch):
    """Continue strict propagation of an already-sorted later-timestamp suffix
    on top of resident state R (Prop. insertion)."""
    R = R.copy()
    from itertools import groupby

    stream = sort_and_validate_edges(n, batch)
    for _, git in groupby(stream, key=lambda e: e.t):
        group = list(git)
        dirty = {e.v for e in group}
        snap = {row: R[row].copy() for row in dirty}
        for e in group:
            cand = snap.get(e.u, R[e.u])
            R[e.v] |= cand
    return R


def window_experiment(n, stream, n_slots=8):
    """Sliding window over timestamp range; each slot bit-identical to scratch."""
    ts = sorted({e.t for e in stream})
    T = len(ts)
    win = max(1, T // 4)
    stride = max(1, T // (2 * n_slots))
    out = []
    for s in range(n_slots):
        lo_i = s * stride
        hi_i = min(T - 1, lo_i + win)
        if lo_i >= T:
            break
        lo, hi = ts[lo_i], ts[hi_i]
        wedges = [e for e in stream if lo <= e.t <= hi]
        if not wedges:
            continue
        t0 = time.perf_counter()
        R = temporal_strata_packed(n, wedges, same_time_mode="strict")
        dt = time.perf_counter() - t0
        out.append(
            {
                "slot": s + 1,
                "window_edges": len(wedges),
                "seconds": round(dt, 4),
                "pairs": reachable_pairs(R, n),
            }
        )
    return out


def deletion_experiment(n, stream):
    """Deletion repair via affected-suffix replay from pre-deletion snapshot."""
    stream = sort_and_validate_edges(n, stream)
    ts = [e.t for e in stream]
    m = len(stream)
    out = []
    for frac, label in [(0.05, "early"), (0.5, "middle"), (0.95, "late")]:
        idx = int(frac * m)
        td = ts[idx]
        # split at first edge with timestamp >= td (the affected suffix)
        cut = next(i for i, t in enumerate(ts) if t >= td)
        R_pre = _run_prefix(n, stream[:cut])  # pre-deletion snapshot (excluded from repair cost)
        suffix = stream[cut:]
        # delete one edge inside the suffix, then replay the suffix
        suffix_del = suffix[1:] if len(suffix) > 1 else suffix
        t0 = time.perf_counter()
        R_rep = _propagate_suffix(n, R_pre, suffix_del)
        dt = time.perf_counter() - t0
        # ground truth: full recompute on stream minus that edge
        full = stream[:cut] + suffix_del
        R_full = _run_prefix(n, full)
        ok = bool(np.array_equal(R_rep, R_full))
        out.append(
            {
                "cut": label,
                "suffix_edges": len(suffix),
                "suffix_fraction_pct": round(100.0 * len(suffix) / m, 1),
                "repair_s": round(dt, 4),
                "per_1k_suffix_s": round(dt / max(len(suffix) / 1000.0, 1e-9), 4),
                "bit_identical": ok,
            }
        )
        assert ok, f"deletion {label} bit mismatch"
    return out


def blocking_experiment(n, stream, betas):
    ref = temporal_strata_packed(n, stream, same_time_mode="strict")
    ref_pairs = reachable_pairs(ref, n)
    out = []
    for beta in betas:
        t0 = time.perf_counter()
        R = temporal_strata_packed(n, stream, same_time_mode="strict", block_bits=beta)
        dt = time.perf_counter() - t0
        ok = bool(np.array_equal(R, ref))
        passes = (n + beta - 1) // beta
        resident_words = n * words_for(min(beta, n))
        out.append(
            {
                "beta": beta,
                "passes": passes,
                "resident_words": resident_words,
                "seconds": round(dt, 4),
                "pairs": reachable_pairs(R, n),
                "bit_identical": ok,
            }
        )
        assert ok, f"blocking beta={beta} bit mismatch (pairs {reachable_pairs(R,n)} vs {ref_pairs})"
    return out


def main():
    base = Path("datasets_real_temporal/standard")
    n_cm, cm, _ = load_standard_edges(base / "CollegeMsg.txt")
    n_otc, otc, _ = load_standard_edges(base / "BitcoinOTC.txt")

    print("=== Real dynamic on CollegeMsg ===")
    res = {
        "insertion_CollegeMsg": insertion_experiment(n_cm, cm),
        "window_CollegeMsg": window_experiment(n_cm, cm),
        "deletion_CollegeMsg": deletion_experiment(n_cm, cm),
        "blocking_BitcoinOTC": blocking_experiment(n_otc, otc, betas=[5881, 2048, 1024, 512]),
    }
    ins = res["insertion_CollegeMsg"]
    print(
        "  insertion speedup range:",
        f"{min(b['speedup'] for b in ins)}x-{max(b['speedup'] for b in ins)}x",
        "all bit-identical:",
        all(b["bit_identical"] for b in ins),
    )
    win = res["window_CollegeMsg"]
    print(
        "  window slots:",
        len(win),
        "per-slot seconds:",
        f"{min(w['seconds'] for w in win)}-{max(w['seconds'] for w in win)}",
    )
    dele = res["deletion_CollegeMsg"]
    for d in dele:
        print(
            f"  deletion {d['cut']}: suffix {d['suffix_fraction_pct']}% "
            f"repair {d['repair_s']}s ({d['per_1k_suffix_s']}s/1k) bit-id={d['bit_identical']}"
        )
    blk = res["blocking_BitcoinOTC"]
    for b in blk:
        print(
            f"  blocking beta={b['beta']}: {b['passes']} passes, "
            f"{b['resident_words']} words, {b['seconds']}s, bit-id={b['bit_identical']}"
        )

    out = Path("results/real/real_dynamic_blocking.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res["threads"] = thread_record()
    json.dump(res, out.open("w"), indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
