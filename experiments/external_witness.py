#!/usr/bin/env python3
"""Witness paths for external-baseline disagreements.

``experiments/external_baselines.py --check-only --diagnose`` records, per
sampled source, the vertices that an external tool reports but STRATA does not
(``only_external_vs_<sem>``) and the vertices STRATA reports but the tool misses
(``only_strata_<sem>``). This script turns every such disagreement into
checkable evidence, offline and without the external tool:

* For each vertex STRATA reaches and the tool misses, it computes an explicit
  time-respecting path from the source with an earliest-arrival search that is
  independent of the packed STRATA core, and validates the path edge by edge
  against the input stream (every edge exists, consecutive timestamps are
  non-decreasing for ``closure`` / strictly increasing for ``strict``).
* For each vertex the tool reports and STRATA does not, it records whether the
  independent search finds any path (it should not; a path there would be a
  STRATA error).
* It also re-checks that the independent search reproduces STRATA's reachable
  set size for every disagreeing source.

No timing is involved, so the output carries no performance number; it is a
correctness record. Usage::

  python3 experiments/external_witness.py \
      --check results/real/external_check.json \
      --out results/real/external_witness.json
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

NEG_INF = float("-inf")


def build_adjacency(edges) -> Dict[int, List[Tuple[int, int]]]:
    """``u -> [(t, v), ...]`` from TemporalEdge-like objects or (t, u, v) tuples."""
    adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for e in edges:
        t, u, v = (e.t, e.u, e.v) if hasattr(e, "t") else e
        adj[int(u)].append((int(t), int(v)))
    return adj


def earliest_arrival(adj, source: int, semantics: str):
    """Independent earliest-arrival search (Dijkstra on arrival times).

    Returns ``(arrival, pred)``: ``arrival[v]`` is the earliest timestamp at
    which ``v`` is reached (``-inf`` for the source) and ``pred[v] = (u, t)``
    the last edge of one earliest path. ``closure`` allows equal consecutive
    timestamps, ``strict`` requires strictly increasing ones.
    """
    if semantics not in ("strict", "closure"):
        raise ValueError(f"unknown semantics {semantics!r}")
    strict = semantics == "strict"
    arrival = {source: NEG_INF}
    pred: Dict[int, Optional[Tuple[int, int]]] = {source: None}
    heap = [(NEG_INF, source)]
    while heap:
        a, u = heapq.heappop(heap)
        if a > arrival[u]:
            continue
        for t, v in adj.get(u, ()):
            usable = t > a if strict else t >= a
            if usable and t < arrival.get(v, float("inf")):
                arrival[v] = t
                pred[v] = (u, t)
                heapq.heappush(heap, (t, v))
    return arrival, pred


def path_to(pred, target: int) -> List[Tuple[int, int, int]]:
    """Edges ``(u, v, t)`` of the recorded path from the source to ``target``."""
    out = []
    v = target
    while pred.get(v) is not None:
        u, t = pred[v]
        out.append((u, v, t))
        v = u
    return out[::-1]


def validate_path(path: Sequence[Tuple[int, int, int]], source: int, target: int,
                  edge_set, semantics: str) -> bool:
    if not path:
        return source == target
    if path[0][0] != source or path[-1][1] != target:
        return False
    for i, (u, v, t) in enumerate(path):
        if (t, u, v) not in edge_set:
            return False
        if i:
            pu, pv, pt = path[i - 1]
            if pv != u:
                return False
            if (semantics == "strict" and not t > pt) or (semantics == "closure" and not t >= pt):
                return False
    return True


def witness_row(row, n, edges, semantics: str):
    adj = build_adjacency(edges)
    edge_set = {(int(e.t), int(e.u), int(e.v)) if hasattr(e, "t") else tuple(map(int, e))
                for e in edges}
    out = []
    for d in row.get("diagnosis", []):
        missed = d.get(f"only_strata_{semantics}", {"count": 0, "examples": []})
        extra = d.get(f"only_external_vs_{semantics}", {"count": 0, "examples": []})
        if not missed["count"] and not extra["count"]:
            continue
        s = int(d["source"])
        arrival, pred = earliest_arrival(adj, s, semantics)
        entry = {
            "source": s,
            "strata_size": d.get(f"{semantics}_size"),
            "independent_size": len(arrival),
            "independent_matches_strata": len(arrival) == d.get(f"{semantics}_size"),
            "external_size": d.get("external_size"),
            "missed_by_external": [],
            "reported_only_by_external": [],
            "examples_complete": (len(missed["examples"]) == missed["count"]
                                  and len(extra["examples"]) == extra["count"]),
        }
        for v in missed["examples"]:
            v = int(v)
            p = path_to(pred, v) if v in arrival else []
            entry["missed_by_external"].append({
                "vertex": v,
                "reachable_independent": v in arrival,
                "path": [list(x) for x in p],
                "hops": len(p),
                "path_valid": validate_path(p, s, v, edge_set, semantics) if v in arrival else False,
            })
        for v in extra["examples"]:
            v = int(v)
            entry["reported_only_by_external"].append({
                "vertex": v, "reachable_independent": v in arrival})
        out.append(entry)
    return out


def summarize(rows):
    total = sum(r["sources"] for r in rows)
    agree = sum(r["sources"] - len(r["disagreements"]) for r in rows)
    missed = [m for r in rows for d in r["disagreements"] for m in d["missed_by_external"]]
    extra = [x for r in rows for d in r["disagreements"] for x in d["reported_only_by_external"]]
    return {
        "sampled_sources": total,
        "agreeing_sources": agree,
        "disagreeing_sources": total - agree,
        "missed_vertices": len(missed),
        "missed_with_valid_witness": sum(1 for m in missed if m["path_valid"]),
        "external_only_vertices": len(extra),
        "external_only_with_path": sum(1 for x in extra if x["reachable_independent"]),
        "independent_matches_strata_all": all(
            d["independent_matches_strata"] for r in rows for d in r["disagreements"]),
    }


def main(argv=None):
    from experiments.bench_common import select_datasets

    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent.parent
    ap.add_argument("--check", default=str(here / "results" / "real" / "external_check.json"))
    ap.add_argument("--out", default=str(here / "results" / "real" / "external_witness.json"))
    ap.add_argument("--semantics", choices=["closure", "strict"], default=None,
                    help="default: the semantics the external tool matched on its toy streams")
    args = ap.parse_args(argv)

    check = json.loads(Path(args.check).read_text())
    sem = args.semantics
    if sem is None:
        toys = check.get("toy_diagnosis") or []
        if toys and all(t.get("matches_closure") for t in toys):
            sem = "closure"
        elif toys and all(t.get("matches_strict") for t in toys):
            sem = "strict"
        else:
            raise SystemExit("toy streams do not identify one semantics; pass --semantics")
    names = [r["dataset"] for r in check["rows"]]
    sets = {nm: (n, e) for nm, n, e in select_datasets(names, [], include_default_synth=False)}
    rows = []
    for r in check["rows"]:
        n, edges = sets[r["dataset"]]
        dis = witness_row(r, n, edges, sem)
        rows.append({"dataset": r["dataset"], "n": n, "m": len(edges),
                     "sources": r["sources"], "disagreements": dis})
        print(f"[{r['dataset']}] {r['sources'] - len(dis)}/{r['sources']} sources agree under {sem}")
        for d in dis:
            for m in d["missed_by_external"]:
                print(f"  source {d['source']}: external misses {m['vertex']} "
                      f"(witness {m['hops']} hops, valid={m['path_valid']})")
            for x in d["reported_only_by_external"]:
                print(f"  source {d['source']}: external-only {x['vertex']} "
                      f"(independent path: {x['reachable_independent']})")
    payload = {"source_check": os.path.basename(args.check), "tool": check.get("tool"),
               "semantics": sem, "rows": rows, "summary": summarize(rows),
               "note": "offline correctness record; no timing"}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print("summary:", payload["summary"])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
