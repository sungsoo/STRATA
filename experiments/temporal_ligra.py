#!/usr/bin/env python3
"""Temporal Ligra as an external baseline.

Temporal Ligra (MIT license; github.com/jshun/ligra, branch ``temporal``) is the
parallel temporal-graph system the Kairos paper compares against. Its
``apps/temporal/EarliestArrival.C`` computes earliest arrival times from ONE
source with a parallel label-correcting traversal; an edge (u, v, start, end)
can be taken when ``start >= arrival(u)`` and then gives ``arrival(v) <= end``.
Hence, on integer timestamps:

  * duration 1 (end = start + 1) is STRATA's *strict* semantics (each hop
    advances the clock: the next start >= t + 1 > t), and
  * duration 0 (end = start) is STRATA's *closure* semantics (same-time chains).

The adapter never vendors the source. It clones the pinned commit, applies one
small patch (prints the reachable vertices and the worker count after
``Compute``; the upstream print block is commented out), and compiles
``EarliestArrival`` with OpenMP when the compiler supports it.

Rule (same as R13): times are compared only after the reachable sets agree with
STRATA's relation. ``check`` runs the toy streams of
``external_baselines.py`` and sampled sources on the observed-timestamp data;
``time`` runs EVERY source (no extrapolation), verifies every set against
STRATA, and records the tool's own Compute time per source (graph loading is
excluded, as for STRATA's kernel) together with the worker count it reports.

Usage (evaluation machine)::

  python3 experiments/temporal_ligra.py fetch --dest external/temporal_ligra
  python3 experiments/temporal_ligra.py build --src external/temporal_ligra
  python3 experiments/temporal_ligra.py check --src external/temporal_ligra \\
      --out results/real/temporal_ligra_check.json
  python3 experiments/temporal_ligra.py time --src external/temporal_ligra --threads 1,18 \\
      --out results/real/temporal_ligra_time.json
  python3 experiments/temporal_ligra.py time-batch --src external/temporal_ligra \\
      --threads 1,6,12,18 --out results/real/temporal_ligra_batch.json   # v14, R43

``time`` exits 4 without timing when the check file does not report the same
relation for the strict semantics.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

REPO = "https://github.com/jshun/ligra.git"
BRANCH = "temporal"
PIN = "75eb1a29ce6fe39e77d08fce138e8aad486f5022"      # 2023-03-31, branch head when added
APP = "apps/temporal/EarliestArrival.C"
PATCH_MARK = "// STRATA-v13 adapter patch"
PATCH_ANCHOR = "  Frontier.del();\n  free(Visited);\n  free(EarliestArrivalTime);"
PATCH_CODE = (
    f"  {PATCH_MARK}: print the reachable set and the worker count\n"
    "  std::cout << \"WORKERS \" << getWorkers() << std::endl;\n"
    "  std::cout << \"REACH\";\n"
    "  for (long i = 0; i < n; i++)\n"
    "    if (EarliestArrivalTime[i] != INT_MAX) std::cout << ' ' << i;\n"
    "  std::cout << std::endl;\n")
INT_MAX32 = 2**31 - 1


# --------------------------------------------------------------------------- #
# fetch / patch / build
# --------------------------------------------------------------------------- #
def fetch(dest: Path):
    if not (dest / ".git").exists():
        subprocess.run(["git", "clone", "--branch", BRANCH, REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", PIN], check=True)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != PIN:
        raise SystemExit(f"temporal ligra at {head}, expected {PIN}")
    return head


def patch(src: Path) -> bool:
    p = src / APP
    text = p.read_text()
    if PATCH_MARK in text:
        return False
    if PATCH_ANCHOR not in text:
        raise SystemExit("patch anchor not found; the pinned source changed")
    p.write_text(text.replace(PATCH_ANCHOR, PATCH_CODE + PATCH_ANCHOR, 1))
    return True


def _omp_flag_sets():
    yield ["-fopenmp"]
    prefix = None
    try:
        prefix = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True,
                                text=True, timeout=20).stdout.strip() or None
    except Exception:
        pass
    if prefix:                       # Apple clang + Homebrew libomp
        yield ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include", f"-L{prefix}/lib", "-lomp"]


def build(src: Path, compiler: str = None) -> dict:
    # absolute paths: the compiler runs inside apps/temporal (v13 M5 run failed
    # with relative -I / -o paths: "gettime.h file not found")
    src = Path(src).resolve()
    patch(src)
    cxx = compiler or os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++") or "c++"
    app_dir = src / "apps" / "temporal"
    out = app_dir / "EarliestArrival"
    base = [cxx, "-std=c++14", "-O3", f"-I{src / 'ligra'}", "EarliestArrival.C", "-o", str(out)]
    tried = []
    for flags in _omp_flag_sets():
        cmd = [cxx, "-std=c++14", "-O3", "-DOPENMP", *flags, f"-I{src / 'ligra'}",
               "EarliestArrival.C", "-o", str(out)]
        r = subprocess.run(cmd, cwd=app_dir, capture_output=True, text=True)
        tried.append({"flags": flags, "ok": r.returncode == 0, "stderr": r.stderr[-400:]})
        if r.returncode == 0:
            mode = "openmp"
            break
    else:
        r = subprocess.run(base, cwd=app_dir, capture_output=True, text=True)
        tried.append({"flags": [], "ok": r.returncode == 0, "stderr": r.stderr[-400:]})
        if r.returncode != 0:
            raise SystemExit("could not compile EarliestArrival:\n" + r.stderr[-2000:])
        mode = "sequential"
    ver = subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
    return {"binary": str(out), "mode": mode, "compiler": cxx, "compiler_version": ver,
            "attempts": tried, "commit": PIN}


# --------------------------------------------------------------------------- #
# graph file and runs
# --------------------------------------------------------------------------- #
def write_graph(path: Path, n: int, ts, us, vs, duration: int):
    """WeightedTemporalAdjacencyGraph: n, m, offsets, targets, weights, start, end."""
    ts, us, vs = (np.asarray(a, dtype=np.int64) for a in (ts, us, vs))
    if ts.size and (ts.min() < -INT_MAX32 or ts.max() + duration >= INT_MAX32 - 1):
        raise SystemExit("timestamps do not fit Temporal Ligra's 32-bit times")
    order = np.lexsort((ts, vs, us))
    us, vs, ts = us[order], vs[order], ts[order]
    offsets = np.searchsorted(us, np.arange(n), side="left")
    m = us.size
    lines = ["WeightedTemporalAdjacencyGraph", str(n), str(m)]
    lines += [str(x) for x in offsets]
    lines += [str(x) for x in vs]
    lines += ["1"] * m
    lines += [str(x) for x in ts]
    lines += [str(x + duration) for x in ts]
    path.write_text("\n".join(lines) + "\n")


def run_raw(binary, graph, source, t0, threads=None):
    """One EarliestArrival process; returns its stdout (parsed by parse_output)."""
    env = dict(os.environ)
    if threads:
        env["OMP_NUM_THREADS"] = str(threads)
    r = subprocess.run([binary, str(graph), str(source), str(t0), str(INT_MAX32 - 1)],
                       capture_output=True, text=True, env=env, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"EarliestArrival failed ({r.returncode}): {r.stderr[-500:]}")
    return r.stdout


def run_source(binary, graph, source, t0, threads=None):
    return parse_output(run_raw(binary, graph, source, t0, threads), source)


def parse_output(stdout, source):
    reach = workers = runtime = None
    for line in stdout.splitlines():
        if line.startswith("REACH"):
            reach = {int(x) for x in line.split()[1:]}
        elif line.startswith("WORKERS"):
            workers = int(line.split()[1])
        else:
            mt = re.match(r"\s*runtime\s*:\s*([0-9.eE+-]+)", line)
            if mt:
                runtime = float(mt.group(1))
    if reach is None:
        raise RuntimeError("no REACH line; is the binary patched?")
    return reach | {source}, workers, runtime


def strata_reach(n, ts, us, vs, mode):
    from temporal_strata.packed_reachability import packed_to_matrix, temporal_strata_packed
    from temporal_strata.stream import arrays_to_edges

    R = temporal_strata_packed(n, arrays_to_edges(ts, us, vs), same_time_mode=mode)
    return packed_to_matrix(R, n).astype(bool)          # M[s, v]


def _real(names):
    from experiments.bench_common import DATA
    from temporal_strata.io import load_standard_arrays

    for name in names:
        n, ts, us, vs, _meta = load_standard_arrays(DATA / f"{name}.txt")
        yield name, n, ts, us, vs


# --------------------------------------------------------------------------- #
# check / time
# --------------------------------------------------------------------------- #
def check(binary, datasets, per_set, seed):
    from experiments.external_baselines import TOY_STREAMS

    tmp = Path(tempfile.mkdtemp(prefix="tligra_"))
    toys = []
    for name, n, raw, src, strict, closure in TOY_STREAMS:
        ts = [t for t, _u, _v in raw]
        us = [u for _t, u, _v in raw]
        vs = [v for _t, _u, v in raw]
        row = {"toy": name}
        for dur, sem, want in ((1, "strict", strict), (0, "closure", closure)):
            g = tmp / f"{name}_{dur}.adj"
            write_graph(g, n, ts, us, vs, dur)
            got, _w, _rt = run_source(binary, g, src, min(ts))
            row[f"duration{dur}"] = sorted(got)
            row[f"duration{dur}_matches_{sem}"] = got == want
        toys.append(row)
    sets = []
    rng = random.Random(seed)
    for name, n, ts, us, vs in _real(datasets):
        srcs = sorted(rng.sample(range(n), min(per_set, n)))
        row = {"dataset": name, "n": int(n), "m": int(len(us)), "sources": len(srcs)}
        for dur, sem in ((1, "strict"), (0, "closure")):
            M = strata_reach(n, ts, us, vs, sem)
            g = tmp / f"{name}_{dur}.adj"
            write_graph(g, n, ts, us, vs, dur)
            bad = []
            for s in srcs:
                got, _w, _rt = run_source(binary, g, s, int(np.min(ts)))
                want = set(np.flatnonzero(M[s]).tolist()) | {s}
                if got != want:
                    bad.append({"source": s, "only_ligra": len(got - want), "only_strata": len(want - got)})
            row[f"duration{dur}_vs_{sem}"] = {"agree": len(srcs) - len(bad), "disagree": bad[:10]}
        sets.append(row)
    shutil.rmtree(tmp, ignore_errors=True)
    same_strict = (all(t.get("duration1_matches_strict") for t in toys)
                   and all(r["duration1_vs_strict"]["agree"] == r["sources"] for r in sets))
    same_closure = (all(t.get("duration0_matches_closure") for t in toys)
                    and all(r["duration0_vs_closure"]["agree"] == r["sources"] for r in sets))
    return {"toys": toys, "datasets": sets, "same_relation_strict": same_strict,
            "same_relation_closure": same_closure}


def batches(n, p):
    """Round-robin split of sources 0..n-1 into p batches (every source exactly once)."""
    return [list(range(i, n, p)) for i in range(p)]


def run_batches(binary, graph, t0, n, p):
    """source-level parallelism the tool itself does not offer. p worker
    threads each run the one-thread binary over their batch, one process per
    source; outputs are kept raw and parsed only after the clock stops, so the
    wall time contains the tool's processes (start-up and graph reading
    included) and nothing of ours."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    parts = batches(n, p)

    def work(batch):
        out = []
        for s_ in batch:
            out.append((s_, run_raw(binary, graph, s_, t0, 1)))
        return out

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=p) as ex:
        results = list(ex.map(work, parts))
    wall = time.perf_counter() - start
    return wall, results


def time_batch(binary, datasets, threads):
    out = []
    tmp = Path(tempfile.mkdtemp(prefix="tligra_b_"))
    for name, n, ts, us, vs in _real(datasets):
        M = strata_reach(n, ts, us, vs, "strict")
        g = tmp / f"{name}.adj"
        write_graph(g, n, ts, us, vs, 1)
        t0 = int(np.min(ts))
        rows = []
        for p in threads:
            wall, results = run_batches(binary, g, t0, n, p)
            seen, worker_compute, workers = 0, [], set()
            for batch in results:
                tot = 0.0
                for s_, stdout in batch:
                    got, w, rt = parse_output(stdout, s_)
                    if got != set(np.flatnonzero(M[s_]).tolist()) | {s_}:
                        raise SystemExit(f"{name}: source {s_} disagrees with STRATA strict (p={p})")
                    tot += rt
                    workers.add(w)
                    seen += 1
                worker_compute.append(tot)
            if seen != n:
                raise SystemExit(f"{name}: {seen} of {n} sources run")
            rows.append({"processes": p, "workers_reported": sorted(workers), "wall_s": wall,
                         "critical_compute_s": max(worker_compute),
                         "sum_compute_s": float(sum(worker_compute)), "sources": n,
                         "verified_all_sources": True})
            print(f"[{name}] batch p={p} wall={wall:.3f}s critical-path compute="
                  f"{max(worker_compute):.3f}s workers={sorted(workers)}")
        out.append({"dataset": name, "n": int(n), "m": int(len(us)), "rows": rows})
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def time_all(binary, datasets, threads):
    out = []
    tmp = Path(tempfile.mkdtemp(prefix="tligra_t_"))
    for name, n, ts, us, vs in _real(datasets):
        M = strata_reach(n, ts, us, vs, "strict")
        g = tmp / f"{name}.adj"
        write_graph(g, n, ts, us, vs, 1)
        t0 = int(np.min(ts))
        rows = []
        for p in threads:
            per, workers, mism = [], set(), 0
            for s in range(n):
                got, w, rt = run_source(binary, g, s, t0, p)
                if got != set(np.flatnonzero(M[s]).tolist()) | {s}:
                    mism += 1
                per.append(rt)
                workers.add(w)
            if mism:
                raise SystemExit(f"{name}: {mism} sources disagree with STRATA strict at p={p}")
            rows.append({"threads": p, "workers_reported": sorted(workers),
                         "sum_compute_s": float(np.sum(per)), "median_source_s": float(np.median(per)),
                         "sources": n, "verified_all_sources": True})
            print(f"[{name}] p={p} workers={sorted(workers)} sum={np.sum(per):.3f}s "
                  f"median/source={np.median(per) * 1e3:.3f} ms")
        out.append({"dataset": name, "n": int(n), "m": int(len(us)), "rows": rows})
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--dest", default=str(ROOT / "external" / "temporal_ligra"))
    b = sub.add_parser("build")
    b.add_argument("--src", default=str(ROOT / "external" / "temporal_ligra"))
    b.add_argument("--cxx", default=None)
    for name in ("check", "time", "time-batch"):
        c = sub.add_parser(name)
        c.add_argument("--src", default=str(ROOT / "external" / "temporal_ligra"))
        c.add_argument("--datasets", default="CollegeMsg,BitcoinOTC,BitcoinOTCpos")
        c.add_argument("--out", required=True)
        if name == "check":
            c.add_argument("--per-set", type=int, default=64)
            c.add_argument("--seed", type=int, default=5)
        else:
            c.add_argument("--threads", default="1,18" if name == "time" else "1,6,12,18")
            c.add_argument("--check-file", default=str(ROOT / "results" / "real" / "temporal_ligra_check.json"))
    args = ap.parse_args(argv)

    if args.cmd == "fetch":
        print(f"temporal ligra at {fetch(Path(args.dest))}")
        return 0
    src = Path(args.src).resolve()
    if args.cmd == "build":
        info = build(src, args.cxx)
        (src / "strata_build.json").write_text(json.dumps(info, indent=2))
        print(f"built {info['binary']} ({info['mode']}, {info['compiler_version']})")
        return 0
    info = json.loads((src / "strata_build.json").read_text())
    from experiments.thread_record import thread_record
    from temporal_strata.parallel import machine_info

    datasets = [x for x in args.datasets.split(",") if x]
    if args.cmd == "check":
        res = check(info["binary"], datasets, args.per_set, args.seed)
    else:
        chk = json.loads(Path(args.check_file).read_text())
        if not chk.get("same_relation_strict"):
            print("NOT TIMED: Temporal Ligra (duration 1) does not compute STRATA's strict relation")
            return 4
        grid = [int(x) for x in args.threads.split(",") if x]
        if args.cmd == "time":
            res = {"datasets": time_all(info["binary"], datasets, grid)}
        else:
            res = {"mode": "source batches: p one-thread processes run concurrently",
                   "datasets": time_batch(info["binary"], datasets, grid)}
    payload = {"tool": {"name": "Temporal Ligra EarliestArrival", "repo": REPO, "branch": BRANCH,
                        "commit": PIN, "build": {k: info[k] for k in ("mode", "compiler_version")}},
               "machine": machine_info(), "threads": thread_record(), **res}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    if args.cmd == "check":
        print(f"same relation: strict={res['same_relation_strict']} closure={res['same_relation_closure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
