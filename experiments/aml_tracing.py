#!/usr/bin/env python3
"""AML / provenance tracing case study on REAL Bitcoin-OTC.

Replaces the earlier synthetic drug-discovery hit-prioritization case study with
a case study on genuinely timestamped data. The Bitcoin-OTC who-trusts-whom
graph (real UNIX timestamps) is a natural provenance / anti-money-laundering
(AML) setting: edges are trust/rating events between accounts, and an
investigator asks "which accounts could a flow originating at account a have
reached along a *time-respecting* path?" --- a static reachability query answers
the wrong question because it admits trails that travel backward in time.

We compute, on the real graph (all numbers bit-verified against exact temporal
BFS elsewhere):
  * temporal (strict) vs static reachable account pairs -> the "false-trail"
    fraction: statically-connected pairs with NO time-respecting path;
  * per-account downstream reach (out-reach) under temporal vs static semantics;
  * audit ranking: top-k accounts by out-reach, temporal vs static, and the
    rank disagreement (Spearman) an investigator would incur by ignoring time;
  * a recency window: reachability restricted to the last W days.
No synthetic labels; every quantity is a direct function of the real stream.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.io import load_standard_edges
from temporal_strata.packed_reachability import temporal_strata_packed, packed_to_matrix, reachable_pairs
from temporal_strata.stream import TemporalEdge


def out_reach_counts(M):
    """M[v,s] = 1 iff s reaches v. Out-reach of account s = #targets it reaches
    (column sum), excluding the trivial self-pair."""
    col = M.sum(axis=0).astype(np.int64) - 1  # minus self
    return col


def spearman(a, b):
    ra = np.argsort(np.argsort(-a))
    rb = np.argsort(np.argsort(-b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def main():
    p = Path("datasets_real_temporal/standard/BitcoinOTC.txt")
    n, edges, meta = load_standard_edges(p)
    tmin = meta.get("timestamp_min"); tmax = meta.get("timestamp_max")

    # temporal (strict) and static (timestamp-collapsed closure)
    Rt = temporal_strata_packed(n, edges, same_time_mode="strict")
    Mt = packed_to_matrix(Rt, n)
    flat = [TemporalEdge(t=0, u=e.u, v=e.v) for e in edges]
    Rs = temporal_strata_packed(n, flat, same_time_mode="closure")
    Ms = packed_to_matrix(Rs, n)

    temporal_pairs = int(Mt.sum()) - n  # minus diagonal
    static_pairs = int(Ms.sum()) - n
    false_trail_frac = (static_pairs - temporal_pairs) / static_pairs

    ot = out_reach_counts(Mt)
    os_ = out_reach_counts(Ms)

    k = 20
    top_t = set(np.argsort(-ot)[:k].tolist())
    top_s = set(np.argsort(-os_)[:k].tolist())
    overlap = len(top_t & top_s)
    rho = spearman(ot.astype(float), os_.astype(float))

    # per-top-static-account inflation: fraction of its static reach that is
    # NOT time-respecting
    top_s_list = np.argsort(-os_)[:k]
    infl = []
    for a in top_s_list:
        if os_[a] > 0:
            infl.append((os_[a] - ot[a]) / os_[a])
    infl = float(np.mean(infl)) if infl else 0.0

    # recency window: last 90 days
    win = None
    if isinstance(tmin, int) and isinstance(tmax, int):
        cutoff = tmax - 90 * 86400
        wedges = [e for e in edges if e.t >= cutoff]
        Rw = temporal_strata_packed(n, wedges, same_time_mode="strict")
        win_pairs = reachable_pairs(Rw, n) - n
        win = {"window_days": 90, "window_edges": len(wedges), "window_pairs": win_pairs}

    rec = {
        "dataset": "Bitcoin-OTC (real timestamps)",
        "n": n, "m": len(edges),
        "temporal_pairs": temporal_pairs,
        "static_pairs": static_pairs,
        "false_trail_fraction_pct": round(100 * false_trail_frac, 1),
        "static_overcount_pct": round(100 * (static_pairs - temporal_pairs) / temporal_pairs, 1),
        "topk": k,
        "topk_overlap_temporal_vs_static": overlap,
        "spearman_out_reach": round(rho, 3),
        "mean_static_reach_inflation_top_static_pct": round(100 * infl, 1),
        "recency_window": win,
    }
    print(json.dumps(rec, indent=2))
    out = Path("results/real/aml_tracing.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, out.open("w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
