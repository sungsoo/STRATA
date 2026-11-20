#!/usr/bin/env python3
"""External (third-party) temporal-reachability baselines.

Why
---
Every baseline in the paper so far is implemented in this artifact. Reviewers
expect at least one publicly available implementation, run on the same data
under the same semantics, with its version and the comparison conditions
recorded.

How
---
1. An *adapter* runs the external tool for a set of source vertices and returns,
   per source, the set of vertices it reports as temporally reachable.
2. Before any timing, the answers are compared with STRATA's exact relation
   under BOTH semantics (strict and same-time closure). The run is labelled
   with the semantics it matches exactly; if it matches neither, the script
   records the mismatch (first differing sources) and REFUSES to time it --
   a tool computing a different relation is not a same-problem competitor.
3. Matching runs are timed (warm-up + best-of-R) next to STRATA's all-pairs
   pass on the same machine. With ``--sources all`` both sides answer all
   ``n`` sources; with a sample, the external time covers only that sample and
   is reported as such (no extrapolation).

Adapters
--------
* ``command``: any executable, given as ``--command "TOOL {edges} {sources} {out}"``.
  Contract: ``{edges}`` is a text file of ``u v t`` lines (0-based ids, sorted
  by t), ``{sources}`` a file with one source id per line; the tool writes
  ``s v`` lines to ``{out}`` for every vertex ``v`` reachable from ``s``
  (``s`` itself optional). This lets any compiled index or traversal (C++,
  Rust, Java, ...) be plugged in without code changes; record its repository,
  commit, build flags, and license in ``--tool-note``.
* ``raphtory``: the Raphtory temporal-graph library (``pip install raphtory``),
  through ``raphtory.algorithms.temporally_reachable_nodes``. NOTE: this adapter
  was written against the library's documented API but could not be executed
  in the build environment (installing the package was not permitted there).
  Run ``--adapter raphtory --check-only`` first on the evaluation machine; any
  API or semantic difference stops the run before timing.

Diagnosis
-------------------------------------------
``--diagnose`` records, per sampled source, the sizes of the external / strict
/ closure sets, the size of each set difference with example vertices (only in
the external answer / only in STRATA's), and, for the first source, the raw
result shape the external tool returned. It also runs five toy streams whose
answers are known by hand (same-timestamp chain, time-reversed edge,
self-contact, waiting path, isolated source) and reports which semantics, if
any, the tool matches on each. This separates a definition difference from an
adapter interpretation error.

Usage
-----
  python3 experiments/external_baselines.py --adapter raphtory --check-only --diagnose
  python3 experiments/external_baselines.py --adapter raphtory \
      --datasets CollegeMsg,BitcoinOTC --sources all --repeats 3
  python3 experiments/external_baselines.py --adapter command \
      --command "./tool {edges} {sources} {out}" --tool-note "repo@commit, -O3, MIT"
      [--out results/real/external_baselines.json]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.packed_reachability import (  # noqa: E402
    packed_to_matrix,
    temporal_strata_packed,
    words_for,
)
from experiments.bench_common import best_of, select_datasets  # noqa: E402

HERE = Path(__file__).resolve().parent


class AdapterError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
class CommandAdapter:
    name = "command"

    def __init__(self, command: str, note: str = ""):
        if not command or "{edges}" not in command or "{out}" not in command:
            raise AdapterError("--command must contain {edges}, {sources} and {out}")
        self.command = command
        self.note = note

    def describe(self):
        return {"adapter": self.name, "command": self.command, "tool_note": self.note}

    def run(self, n, edges, sources) -> Dict[int, Set[int]]:
        with tempfile.TemporaryDirectory() as d:
            ep, sp, op = (os.path.join(d, x) for x in ("edges.txt", "sources.txt", "out.txt"))
            with open(ep, "w") as f:
                for e in sorted(edges, key=lambda e: (e.t, e.u, e.v)):
                    f.write(f"{e.u} {e.v} {e.t}\n")
            with open(sp, "w") as f:
                f.write("\n".join(str(s) for s in sources) + "\n")
            cmd = self.command.format(edges=ep, sources=sp, out=op)
            r = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
            if r.returncode != 0:
                raise AdapterError(f"command failed ({r.returncode}): {r.stderr[-500:]}")
            out: Dict[int, Set[int]] = {s: {s} for s in sources}
            with open(op) as f:
                for line in f:
                    a = line.split()
                    if len(a) >= 2:
                        out.setdefault(int(a[0]), set()).add(int(a[1]))
            return out


class RaphtoryAdapter:
    """Written against the documented API; verify with --check-only first."""

    name = "raphtory"

    def __init__(self):
        try:
            import raphtory  # noqa: F401
            from raphtory import algorithms  # noqa: F401
        except Exception as exc:
            raise AdapterError(f"raphtory not importable: {exc}")
        import raphtory

        self.raphtory = raphtory
        if not hasattr(raphtory.algorithms, "temporally_reachable_nodes"):
            raise AdapterError("raphtory.algorithms.temporally_reachable_nodes not found; "
                               "update this adapter for the installed version")

    def describe(self):
        return {"adapter": self.name,
                "version": getattr(self.raphtory, "__version__", "unknown"),
                "function": "raphtory.algorithms.temporally_reachable_nodes"}

    def run(self, n, edges, sources) -> Dict[int, Set[int]]:
        g = self.raphtory.Graph()
        for e in edges:
            g.add_edge(int(e.t), int(e.u), int(e.v))
        t0 = min(int(e.t) for e in edges)
        out: Dict[int, Set[int]] = {}
        for s in sources:
            res = self.raphtory.algorithms.temporally_reachable_nodes(
                g, max_hops=n, start_time=t0, seed_nodes=[s])
            out[s] = self._reached(res, s)
        return out

    def raw_sample(self, n, edges, s, limit=8):
        """Shape of the raw library result for one source (diagnosis only)."""
        g = self.raphtory.Graph()
        for e in edges:
            g.add_edge(int(e.t), int(e.u), int(e.v))
        t0 = min(int(e.t) for e in edges)
        res = self.raphtory.algorithms.temporally_reachable_nodes(
            g, max_hops=n, start_time=t0, seed_nodes=[s])
        out = {"type": type(res).__name__,
               "attrs": [a for a in dir(res) if not a.startswith("_")][:40]}
        try:
            items = None
            for attr in ("items", "get_all_with_names", "to_dict"):
                if hasattr(res, attr):
                    items = getattr(res, attr)()
                    out["accessor"] = attr
                    break
            if isinstance(items, dict):
                items = list(items.items())
            items = list(items)[:limit] if items is not None else []
            out["first_items"] = [[repr(k)[:80], type(k).__name__, repr(v)[:160],
                                   type(v).__name__] for k, v in items]
        except Exception as exc:  # pragma: no cover - diagnosis must not fail the run
            out["error"] = repr(exc)
        return out

    @staticmethod
    def _reached(res, s) -> Set[int]:
        """Vertices Raphtory reports as temporally reachable from the seed.

        Raphtory 0.17 returns an ``OutputNodeState`` with one entry per vertex
        of the graph, ``{'reachable_nodes': [ {'0': time, '1': predecessor}, ... ]}``;
        a vertex is reached iff that list is non-empty (the seed itself carries
        a ``'start'`` record). STRATA-v7 fix: the v3-v6 adapter
        tested the truthiness of the whole value, and a dict with an empty list
        is truthy, so every vertex was counted as reached (diagnosed on the M5,
        results/real/external_check.json of 2026-11-15).
        """
        items = None
        for attr in ("items", "get_all_with_names", "to_dict"):
            if hasattr(res, attr):
                items = getattr(res, attr)()
                break
        if items is None:
            raise AdapterError(f"unrecognized result type {type(res).__name__}; update adapter")
        if isinstance(items, dict):
            items = items.items()
        reached = {s}
        for node, val in items:
            name = getattr(node, "name", node)
            if isinstance(val, dict):
                if "reachable_nodes" not in val:
                    raise AdapterError(f"unexpected value keys {sorted(val)}; update adapter")
                records = val["reachable_nodes"]
                if not isinstance(records, (list, tuple)):
                    raise AdapterError(f"unexpected reachable_nodes type "
                                       f"{type(records).__name__}; update adapter")
            elif isinstance(val, (list, tuple)):
                records = val
            else:
                raise AdapterError(f"unexpected value type {type(val).__name__}; update adapter")
            if len(records) > 0:
                reached.add(int(name))
        return reached


def make_adapter(args):
    if args.adapter == "command":
        return CommandAdapter(args.command, args.tool_note)
    if args.adapter == "raphtory":
        return RaphtoryAdapter()
    raise AdapterError(f"unknown adapter {args.adapter}")


# --------------------------------------------------------------------------- #
# Semantics check and timing
# --------------------------------------------------------------------------- #
def reach_sets_from_packed(R, n, sources) -> Dict[int, Set[int]]:
    M = packed_to_matrix(R, n)  # M[s, v] == 1 iff s reaches v
    return {s: set(np.flatnonzero(M[s]).tolist()) for s in sources}


TOY_STREAMS = [
    # (name, n, edges (t, u, v), source, strict reach, closure reach)
    ("same_time_chain", 4, [(1, 0, 1), (1, 1, 2), (1, 2, 3)], 0, {0, 1}, {0, 1, 2, 3}),
    ("time_reversed", 3, [(5, 0, 1), (2, 1, 2)], 0, {0, 1}, {0, 1}),
    ("waiting_path", 3, [(1, 0, 1), (7, 1, 2)], 0, {0, 1, 2}, {0, 1, 2}),
    ("self_contact", 3, [(1, 0, 0), (2, 0, 1), (3, 1, 2)], 0, {0, 1, 2}, {0, 1, 2}),
    ("isolated_source", 4, [(1, 1, 2), (2, 2, 3)], 0, {0}, {0}),
]


def diagnose_toys(adapter):
    from temporal_strata.stream import TemporalEdge

    rows = []
    for name, n, raw, src, strict, closure in TOY_STREAMS:
        edges = [TemporalEdge(t, u, v) for t, u, v in raw]
        try:
            got = adapter.run(n, edges, [src]).get(src, {src}) | {src}
            rows.append({"toy": name, "source": src, "external": sorted(got),
                         "strict": sorted(strict), "closure": sorted(closure),
                         "matches_strict": got == strict, "matches_closure": got == closure})
        except Exception as exc:
            rows.append({"toy": name, "error": repr(exc)})
    return rows


def diagnose_sources(ext, exact, limit=5):
    out = []
    for s in sorted(exact["strict"]):
        e = ext.get(s, {s}) | {s}
        row = {"source": s, "external_size": len(e)}
        for sem in ("strict", "closure"):
            x = exact[sem][s] | {s}
            only_ext = sorted(e - x)
            only_strata = sorted(x - e)
            row[f"{sem}_size"] = len(x)
            row[f"only_external_vs_{sem}"] = {"count": len(only_ext), "examples": only_ext[:limit]}
            row[f"only_strata_{sem}"] = {"count": len(only_strata), "examples": only_strata[:limit]}
        out.append(row)
    return out


def suspicious_all_reached(ext, exact, n) -> bool:
    """True if every sampled source 'reaches' all n vertices externally while
    STRATA's exact relation (either semantics) does not: the signature of the
    v3-v6 Raphtory adapter error, flagged so it cannot pass silently again."""
    if not exact["strict"]:
        return False
    all_ext = all(len(ext.get(s, {s}) | {s}) == n for s in exact["strict"])
    all_exact = all(len(exact["closure"][s] | {s}) == n for s in exact["closure"])
    return all_ext and not all_exact


def compare(ext: Dict[int, Set[int]], exact: Dict[int, Set[int]], limit=5):
    bad = [s for s in exact if ext.get(s, {s}) | {s} != exact[s] | {s}]
    return {"match": not bad, "mismatched_sources": len(bad), "examples": bad[:limit]}


def run_dataset(adapter, name, n, edges, sources_spec, repeats, seed, check_only,
                diagnose=False):
    print(f"[{name}] n={n} m={len(edges)}")
    if sources_spec == "all":
        sources = list(range(n))
    else:
        rng = random.Random(seed)
        sources = sorted(rng.sample(range(n), min(int(sources_spec), n)))
    R_strict = temporal_strata_packed(n, edges, directed=True, same_time_mode="strict")
    R_closure = temporal_strata_packed(n, edges, directed=True, same_time_mode="closure")
    exact = {"strict": reach_sets_from_packed(R_strict, n, sources),
             "closure": reach_sets_from_packed(R_closure, n, sources)}
    ext = adapter.run(n, edges, sources)
    checks = {sem: compare(ext, exact[sem]) for sem in exact}
    suspicious = suspicious_all_reached(ext, exact, n)
    matched = [sem for sem in ("strict", "closure") if checks[sem]["match"]]
    row = {"dataset": name, "n": n, "m": len(edges), "sources": len(sources),
           "all_sources": sources_spec == "all", "semantics_check": checks,
           "matched_semantics": matched[0] if matched else None,
           "suspicious_all_reached": suspicious}
    if suspicious:
        print("  WARNING: the external tool reports every vertex as reachable from every "
              "sampled source although STRATA does not -- likely an adapter / API "
              "interpretation error, not a semantic difference")
    print(f"  semantics: strict={checks['strict']['match']} closure={checks['closure']['match']}")
    if diagnose:
        row["diagnosis"] = diagnose_sources(ext, exact)
        if hasattr(adapter, "raw_sample"):
            try:
                row["raw_sample"] = adapter.raw_sample(n, edges, sources[0])
            except Exception as exc:  # pragma: no cover
                row["raw_sample"] = {"error": repr(exc)}
        sizes = [(d["external_size"], d["strict_size"], d["closure_size"]) for d in row["diagnosis"][:5]]
        print(f"  diagnose: (external, strict, closure) sizes for first sources: {sizes}")
    if not matched:
        row["timed"] = False
        row["reason"] = "external relation matches neither semantics; not a same-problem run"
        return row
    if check_only:
        row["timed"] = False
        row["reason"] = "--check-only"
        return row
    sem = matched[0]
    row["external_s"] = best_of(lambda: adapter.run(n, edges, sources), repeats)
    row["strata_all_pairs_s"] = best_of(
        lambda: temporal_strata_packed(n, edges, directed=True, same_time_mode=sem), repeats)
    row["strata_backend"] = "interpreted NumPy packed core (all pairs)"
    row["timed"] = True
    print(f"  external({len(sources)} sources)={row['external_s']:.3f}s  "
          f"STRATA all-pairs={row['strata_all_pairs_s']:.3f}s")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", choices=["command", "raphtory"], required=True)
    ap.add_argument("--command", default="")
    ap.add_argument("--tool-note", default="")
    ap.add_argument("--datasets", default="CollegeMsg,BitcoinOTC,BitcoinOTCpos")
    ap.add_argument("--sources", default="64", help="'all' or a sample size")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--diagnose", action="store_true",
                    help="record set sizes / differences / raw result shape and run toy streams")
    ap.add_argument("--check-only", action="store_true",
                    help="semantics check on a small sample, no timing")
    ap.add_argument("--out", default=str(HERE.parent / "results" / "real" / "external_baselines.json"))
    args = ap.parse_args()
    try:
        adapter = make_adapter(args)
    except AdapterError as exc:
        print(f"ADAPTER UNAVAILABLE: {exc}")
        sys.exit(3)
    from temporal_strata.parallel import machine_info

    machine = machine_info()
    print("machine:", machine["summary"], "|", adapter.describe())
    sets = select_datasets([x for x in args.datasets.split(",") if x], [],
                           include_default_synth=False)
    if args.check_only:
        args.sources = "16" if args.sources == "all" else args.sources
    toys = diagnose_toys(adapter) if args.diagnose else None
    if toys:
        for t in toys:
            print(f"  toy {t.get('toy')}: external={t.get('external')} strict={t.get('strict')} "
                  f"closure={t.get('closure')} {t.get('error', '')}")
    rows = [run_dataset(adapter, nm, n, e, args.sources, args.repeats, args.seed,
                        args.check_only, args.diagnose) for nm, n, e in sets]
    payload = {"machine": machine, "tool": adapter.describe(), "rows": rows,
               "toy_diagnosis": toys,
               "note": "timed only when the external relation equals STRATA's exact "
                       "relation under the stated semantics; sampled-source times "
                       "cover the sample only"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
