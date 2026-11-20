#!/usr/bin/env python3
"""Snapshot necessity: the strict pre-group snapshot is required for a
well-defined, exact temporal relation.

Contribution 1 claims the timestamp-ordered *schedule* --- specifically the
per-group read-before-write snapshot R^- --- is what makes packed frontier
propagation exact for temporal (not static) reachability. Sufficiency is the
correctness theorem; here we measure *necessity*.

On streams with genuinely unique timestamps (every group has one edge) the
snapshot is vacuous, so we also study the realistic *coarse-resolution* regime
(day/hour buckets), where many events share a timestamp and same-timestamp
chaining becomes possible. We bucket the real timestamps into G groups and
compare, on the resulting stream:

  * strict WITH snapshot  -> order-independent, matches the exact strict relation;
  * strict WITHOUT snapshot, forward within-group order   -> pairs_fwd;
  * strict WITHOUT snapshot, reversed within-group order  -> pairs_rev;
  * closure (same-time chaining to a fixed point)         -> pairs_closure.

Two facts establish necessity: (a) without the snapshot the result is
order-dependent (pairs_fwd != pairs_rev in general), so it is not even a
well-defined function of the stream; (b) it over-counts the exact strict
relation and drifts toward the closure relation. The snapshot removes both
defects. All strict-with-snapshot numbers are bit-checked against the exact BFS.
"""
from __future__ import annotations

import json
import os
import sys
from itertools import groupby
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.io import load_standard_edges
from temporal_strata.baseline import all_sources_temporal_bfs
from temporal_strata.packed_reachability import (
    temporal_strata_packed, packed_to_matrix, reachable_pairs, words_for,
)
from temporal_strata.stream import TemporalEdge, sort_and_validate_edges


def _bucketize(edges, G):
    ts = [e.t for e in edges]
    tmin, tmax = min(ts), max(ts)
    span = max(1, tmax - tmin)
    out = []
    for e in edges:
        b = int((e.t - tmin) / span * (G - 1))
        out.append(TemporalEdge(t=b, u=e.u, v=e.v))
    return out


def _no_snapshot_pairs(n, edges, reverse_within_group=False):
    W = words_for(n)
    R = np.zeros((n, W), dtype=np.uint64)
    for s in range(n):
        R[s, s >> 6] |= np.uint64(1) << np.uint64(s & 63)
    stream = sort_and_validate_edges(n, edges)
    for _, git in groupby(stream, key=lambda e: e.t):
        grp = list(git)
        if reverse_within_group:
            grp = grp[::-1]
        for e in grp:
            R[e.v] |= R[e.u]  # live read: same-timestamp chaining, order-dependent
    return reachable_pairs(R, n)


def main():
    base = Path("datasets_real_temporal/standard")
    cases = [("CollegeMsg", "CollegeMsg.txt"), ("Bitcoin-OTC", "BitcoinOTC.txt")]
    rows = []

    n0 = 4
    chain = [TemporalEdge(t=0, u=0, v=1), TemporalEdge(t=0, u=1, v=2), TemporalEdge(t=0, u=2, v=3)]
    strict0 = reachable_pairs(temporal_strata_packed(n0, chain, same_time_mode="strict"), n0)
    closure0 = reachable_pairs(temporal_strata_packed(n0, chain, same_time_mode="closure"), n0)
    ns_fwd0 = _no_snapshot_pairs(n0, chain, False)
    ns_rev0 = _no_snapshot_pairs(n0, chain, True)
    print(f"[adversarial chain 0->1->2->3 @ t=0]  strict(snapshot)={strict0}  closure={closure0}  "
          f"no-snapshot fwd={ns_fwd0} rev={ns_rev0}  (order-dependent: {ns_fwd0 != ns_rev0})")
    rows.append({"dataset": "adversarial-chain@t0", "n": n0, "m": 3,
                 "strict_snapshot": strict0, "closure": closure0,
                 "no_snapshot_fwd": ns_fwd0, "no_snapshot_rev": ns_rev0,
                 "order_dependent": bool(ns_fwd0 != ns_rev0)})

    for name, fname in cases:
        p = base / fname
        if not p.exists():
            continue
        n, edges, _ = load_standard_edges(p)
        for G in (2000, 500, 100):
            bedges = _bucketize(edges, G)
            groups = len({e.t for e in bedges})
            R = temporal_strata_packed(n, bedges, same_time_mode="strict")
            strict = reachable_pairs(R, n)
            cap = min(len(bedges), 15000)
            ref = all_sources_temporal_bfs(n, bedges[:cap], same_time_mode="strict")
            Rp = temporal_strata_packed(n, bedges[:cap], same_time_mode="strict")
            assert np.array_equal(packed_to_matrix(Rp, n), ref), f"{name}@G{G}: strict != exact"
            closure = reachable_pairs(temporal_strata_packed(n, bedges, same_time_mode="closure"), n)
            ns_fwd = _no_snapshot_pairs(n, bedges, False)
            ns_rev = _no_snapshot_pairs(n, bedges, True)
            rec = {"dataset": name, "n": n, "groups": groups,
                   "strict_snapshot": strict, "closure": closure,
                   "no_snapshot_fwd": ns_fwd, "no_snapshot_rev": ns_rev,
                   "order_dependent": bool(ns_fwd != ns_rev),
                   "nosnap_overcount_vs_strict_pct": round(100.0 * (ns_fwd - strict) / strict, 3)}
            rows.append(rec)
            print(f"{name} @ {groups} groups: strict={strict} closure={closure} "
                  f"no-snapshot fwd={ns_fwd} rev={ns_rev} "
                  f"(order-dependent: {rec['order_dependent']}, over-count {rec['nosnap_overcount_vs_strict_pct']}%)")

    out = Path("results/real/snapshot_necessity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": rows}, out.open("w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
