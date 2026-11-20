#!/usr/bin/env python3
"""TGLib earliest-arrival as an external baseline (v22, R57).

Why: the paper's temporal indexes (TopChain, TILL) have no public implementation,
and the public Substream index (tgindex) ships without a license, so it is not
used without the authors' permission. TGLib (Oettershagen and Mutzel, MIT
license, PyPI package ``temporalgraphlib``) computes earliest-arrival times from
one source by one pass over the time-ordered edge list; reachable means an arrival
time below infinity. It is a per-source tool, like Temporal Ligra.

How:
* Edges are written as ``u v t tt`` lines. TGLib relabels vertices in order of
  appearance; the node map is read back, so sets are compared on our vertex ids.
* Semantics: an edge (u, v, t, tt) is taken when arrival[u] <= t, and sets
  arrival[v] = t + tt. With integer timestamps, tt = 1 gives strict semantics
  (the next edge needs t' >= t + 1). tt = 0 is checked as well, but a single pass
  cannot follow every same-time chain, so closure is expected to differ.
* ``check`` compares TGLib's reachable sets with STRATA (bit for bit, all sources on
  toy streams, 64 sampled sources per dataset). ``time`` runs only if the strict
  check matched: every source, summed compute time of the library calls (graph
  loading excluded), one process, and the same work split over p processes
  (critical path = slowest process), next to STRATA's end-to-end time.

v23: PyPI publishes only Linux x86_64 wheels of temporalgraphlib (no source
distribution), so pip cannot install it on macOS arm64. ``build`` compiles the
pinned MIT source instead (commit below, recursive clone for the PyBind11
submodule): first ``pip install`` of the checkout (its pyproject), else the README
route (CMake Release build of tglib_cpp, then a .pth file that puts the built
module on this interpreter's path). The record keeps the commit, the route, the
tool versions, any local diff, and the log tail; nothing is substituted when the
build fails.

What:
    python3 experiments/tglib_baseline.py build --dest external --out results/real/tglib_build.json
    python3 experiments/tglib_baseline.py check --out results/real/tglib_check.json
    python3 experiments/tglib_baseline.py time --check results/real/tglib_check.json \
        --procs 1,18 --out results/real/tglib_time.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PACKAGE = "temporalgraphlib==1.7.1"
REPO_URL = "https://gitlab.com/tgpublic/tglib.git"
COMMIT = "e56c3ddb2b665455fb16aacca1044b99a62850f0"   # probed in session 20261117-142149
DATASETS = ["CollegeMsg", "BitcoinOTC", "BitcoinOTCpos"]
INF_BOUND = 2 ** 62          # TGLib's infinity is the largest Time value


def library_source(build_record="results/real/tglib_build.json"):
    """How pytglib got here: the source build record when one succeeded, else PyPI."""
    try:
        rec = json.loads(open(build_record).read())
        if rec.get("ok"):
            return {"source": "build", "commit": rec.get("commit"), "route": rec.get("route")}
    except (OSError, ValueError):
        pass
    return {"source": "pypi", "package": PACKAGE}


def write_tg(path, ts, us, vs, tt):
    with open(path, "w") as fh:
        for t, u, v in zip(ts.tolist(), us.tolist(), vs.tolist()):
            fh.write(f"{u} {v} {t} {tt}\n")


def reachable_from(tgl, tgs, to_internal, to_ours, s):
    """Our vertex ids reachable from our source s (s itself included)."""
    if s not in to_internal:          # isolated in the file: reaches only itself
        return {s}
    arr = np.asarray(list(tgl.earliest_arrival_times(tgs, to_internal[s], tgs.getTimeInterval())))
    return {to_ours[i] for i in np.flatnonzero(arr < INF_BOUND)}


def node_maps(tgs):
    """(our id -> TGLib id, TGLib id -> our id). TGLib numbers vertices in order of
    first appearance in the file; its node map gives the correspondence."""
    try:
        to_internal = {int(k): int(v) for k, v in dict(tgs.getNodeMap()).items()}
    except Exception:                               # binding without the map type
        rev = list(tgs.getReverseNodeMap())
        to_internal = {int(orig): i for i, orig in enumerate(rev)}
    to_ours = {v: k for k, v in to_internal.items()}
    return to_internal, to_ours


def strata_sets(n, ts, us, vs, mode, sources):
    from temporal_strata.packed_reachability import temporal_strata_packed
    from temporal_strata.stream import arrays_to_edges

    R = temporal_strata_packed(n, arrays_to_edges(ts, us, vs), directed=True, same_time_mode=mode)
    out = {}
    for s in sources:
        col = (R[:, s // 64] >> np.uint64(s % 64)) & np.uint64(1)
        out[s] = set(np.flatnonzero(col).tolist())
    return out


def toy_streams():
    """(name, n, ts, us, vs): same-time chains, ties, a time-ordered path."""
    a = np.array
    return [("chain-same-time", 4, a([1, 1, 1]), a([0, 1, 2]), a([1, 2, 3])),
            ("reverse-order", 4, a([1, 1, 2]), a([1, 0, 2]), a([2, 1, 3])),
            ("late-entry", 4, a([1, 1, 2, 3]), a([0, 1, 2, 0]), a([1, 2, 3, 2])),
            ("star-ties", 5, a([2, 2, 2, 3]), a([0, 0, 1, 4]), a([1, 2, 3, 0])),
            ("path", 5, a([1, 2, 3, 4]), a([0, 1, 2, 3]), a([1, 2, 3, 4]))]


def check_one(tgl, name, n, ts, us, vs, sources, workdir):
    rows = []
    for mode, tt in (("strict", 1), ("closure", 0)):
        path = os.path.join(workdir, f"{name}.{tt}.tg")
        write_tg(path, ts, us, vs, tt)
        tgs = tgl.load_ordered_edge_list(path)
        to_internal, to_ours = node_maps(tgs)
        ref = strata_sets(n, ts, us, vs, mode, sources)
        bad = [int(s) for s in sources if reachable_from(tgl, tgs, to_internal, to_ours, int(s)) != ref[s]]
        rows.append({"dataset": name, "mode": mode, "tt": tt, "sources": len(sources),
                     "mismatched_sources": bad[:10], "n_mismatched": len(bad),
                     "matched": not bad})
        print(f"[check] {name} {mode} (tt={tt}): {len(sources) - len(bad)}/{len(sources)} match")
    return rows


def check(args):
    import pytglib as tgl
    from experiments.bench_common import select_datasets_arrays
    from temporal_strata.stream import sort_validate_arrays

    rows = []
    with tempfile.TemporaryDirectory() as wd:
        for name, n, ts, us, vs in toy_streams():
            rows += check_one(tgl, name, n, ts, us, vs, list(range(n)), wd)
        rng = np.random.default_rng(5)
        for name, n, arrays, _ in select_datasets_arrays(args.datasets.split(","), [], False):
            ts, us, vs = sort_validate_arrays(n, *arrays)
            src = sorted(rng.choice(n, size=min(args.sample, n), replace=False).tolist())
            rows += check_one(tgl, name, n, ts, us, vs, src, wd)
    strict_ok = all(r["matched"] for r in rows if r["mode"] == "strict")
    out = {"package": library_source(), "library_version": getattr(tgl, "__version__", None),
           "strict_matches": strict_ok, "rows": rows}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out)
    return 0 if strict_ok else 3


def _time_slice(path, sources):
    """Worker: load once, then time only the library calls for these sources."""
    import pytglib as tgl
    tgs = tgl.load_ordered_edge_list(path)
    to_internal, _ = node_maps(tgs)
    ti = tgs.getTimeInterval()
    total, reach = 0.0, 0
    for s in sources:
        if s not in to_internal:
            reach += 1
            continue
        t0 = time.perf_counter()
        arr = tgl.earliest_arrival_times(tgs, to_internal[s], ti)
        total += time.perf_counter() - t0
        reach += int(np.count_nonzero(np.asarray(list(arr)) < INF_BOUND))
    return total, reach


def strata_end_to_end(n, arrays, repeats=5):
    from temporal_strata.parallel import (_empty_state, _set_identity_vec, prepare_stream_arrays,
                                          resolve_plan, run_prepared, set_threads)
    from temporal_strata.packed_reachability import words_for

    eff = set_threads(1)
    W = words_for(n)

    def once():
        prep = prepare_stream_arrays(n, *arrays, directed=True, same_time_mode="strict", prep_mode="off")
        R = _empty_state(n, n)
        _set_identity_vec(R, 0, n)
        run_prepared(R, prep, resolve_plan(W, eff, layout="tiled", schedule="static"))
        return R
    once()
    best = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        R = once()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best, int(np.bitwise_count(R).sum())


def time_all(args):
    from experiments.bench_common import select_datasets_arrays
    from temporal_strata.parallel import machine_info
    from temporal_strata.stream import sort_validate_arrays

    chk = json.loads(open(args.check).read())
    if not chk.get("strict_matches"):
        raise SystemExit("TGLib did not compute the strict relation; not timed")
    procs = [int(x) for x in args.procs.split(",")]
    rows = []
    with tempfile.TemporaryDirectory() as wd:
        for name, n, arrays, _ in select_datasets_arrays(args.datasets.split(","), [], False):
            ts, us, vs = sort_validate_arrays(n, *arrays)
            path = os.path.join(wd, f"{name}.tg")
            write_tg(path, ts, us, vs, 1)
            s_e2e, s_pairs = strata_end_to_end(n, (ts, us, vs))
            for p in procs:
                slices = [list(range(i, n, p)) for i in range(p)]
                t0 = time.perf_counter()
                with ProcessPoolExecutor(max_workers=p) as ex:
                    res = list(ex.map(_time_slice, [path] * p, slices))
                wall = time.perf_counter() - t0
                pairs = sum(r[1] for r in res)
                if pairs != s_pairs:
                    raise SystemExit(f"{name}: TGLib pairs {pairs} != STRATA {s_pairs}")
                crit = max(r[0] for r in res)
                rows.append({"dataset": name, "n": int(n), "m": int(ts.size), "procs": p,
                             "compute_s_sum": sum(r[0] for r in res), "critical_path_s": crit,
                             "wall_s": wall, "pairs": pairs, "strata_end_to_end_s_p1": s_e2e,
                             "ratio_critical_over_strata": crit / s_e2e})
                print(f"[time] {name} p={p}: critical {crit:.2f} s, wall {wall:.2f} s, "
                      f"STRATA {s_e2e*1e3:.1f} ms ({crit/s_e2e:.0f}x)")
    out = {"package": library_source(), "machine": machine_info(), "rows": rows,
           "note": "compute = library calls only (graph loading excluded); critical path = "
                   "slowest of p processes over interleaved source slices; STRATA = "
                   "end-to-end strict, one thread, best of 5"}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out)
    return 0


# --------------------------------------------------------------------------- #
# build from source (v23)
# --------------------------------------------------------------------------- #
def build_plan(repo, python, jobs=8, cmake="cmake", extra=(), build_dir=None):
    """Commands of the two build routes, in order (no side effects).

    Route ``pip``: ``python -m pip install --no-deps <repo>`` (the checkout's
    pyproject; ``extra`` CMake arguments reach it through ``CMAKE_ARGS``). Route
    ``cmake``: the README's Release build of ``tglib_cpp`` linked against this
    interpreter, with ``extra`` on the configure line."""
    repo = str(repo)
    bdir = build_dir or os.path.join(repo, "tglib_cpp", "build-release")
    return {
        "fetch": [["git", "clone", "--quiet", "--recurse-submodules", REPO_URL, repo]],
        "pin": [["git", "-C", repo, "fetch", "--quiet", "origin", COMMIT],
                ["git", "-C", repo, "checkout", "--quiet", COMMIT],
                ["git", "-C", repo, "submodule", "update", "--init", "--recursive", "--quiet"]],
        "pip": [[python, "-m", "pip", "install", "--no-deps", repo]],
        "cmake": [[cmake, "-S", os.path.join(repo, "tglib_cpp"), "-B", bdir,
                   "-DCMAKE_BUILD_TYPE=Release", f"-DPYTHON_EXECUTABLE={python}",
                   f"-DPython_EXECUTABLE={python}", POLICY_MIN, *extra],
                  [cmake, "--build", bdir, "--target", BINDING_TARGET, "-j", str(jobs)]],
        "build_dir": bdir,
    }


# v25: CMake 4 removed compatibility with cmake_minimum_required < 3.5, which the
# pinned pybind11 submodule still declares; TGLib's own pyproject passes this flag
# to scikit-build-core, so the CMake route passes it too. At this commit the
# binding target is ``_pytglib`` and ``pytglib/__init__.py`` imports it.
POLICY_MIN = "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
BINDING_TARGET = "_pytglib"


def openmp_args(prefix):
    """CMake arguments that let FindOpenMP use Homebrew's libomp with Apple clang
    (which ships no OpenMP runtime; v23 session: "Could NOT find OpenMP_C")."""
    flags = f"-Xpreprocessor -fopenmp -I{prefix}/include"
    lib = f"{prefix}/lib/libomp.dylib"
    # v25: TGLib adds the OpenMP flags to the compile line only, so every link
    # (test executables, the binding module) also needs the runtime library;
    # the v24 session stopped at "symbol(s) not found for architecture arm64".
    link = f"-L{prefix}/lib -lomp"
    return [f"-DOpenMP_ROOT={prefix}",
            f"-DOpenMP_C_FLAGS={flags}", f"-DOpenMP_CXX_FLAGS={flags}",
            "-DOpenMP_C_LIB_NAMES=omp", "-DOpenMP_CXX_LIB_NAMES=omp",
            f"-DOpenMP_omp_LIBRARY={lib}",
            f"-DCMAKE_EXE_LINKER_FLAGS={link}", f"-DCMAKE_SHARED_LINKER_FLAGS={link}",
            f"-DCMAKE_MODULE_LINKER_FLAGS={link}"]


NO_WERROR_CMAKE = """# Written by experiments/tglib_baseline.py: drop -Werror from TGLib's
# add_compile_options without editing its sources (a newer compiler may warn
# where the authors' did not). Warnings are still printed.
function(add_compile_options)
  set(_opts ${ARGV})
  list(REMOVE_ITEM _opts -Werror)
  _add_compile_options(${_opts})
endfunction()
"""


def no_werror_args(dest):
    """CMake argument that loads the -Werror filter right after project()."""
    path = os.path.abspath(os.path.join(dest, "tglib_no_werror.cmake"))
    with open(path, "w") as fh:
        fh.write(NO_WERROR_CMAKE)
    return [f"-DCMAKE_PROJECT_INCLUDE={path}"]


def find_module(build_dir):
    """Path of the compiled binding (``_pytglib*.so``) under the CMake build folder."""
    hits = []
    for root, _, files in os.walk(build_dir):
        for f in files:
            if f.startswith(BINDING_TARGET) and (f.endswith(".so") or f.endswith(".pyd")):
                hits.append(os.path.join(root, f))
    return sorted(hits)[0] if hits else None


def stage_package(repo, module, dest):
    """Put the checkout's ``pytglib/__init__.py`` and the built binding into one
    package folder ``<dest>/tglib_py/pytglib``; return the folder for the .pth."""
    import shutil
    root = os.path.join(dest, "tglib_py")
    pkg = os.path.join(root, "pytglib")
    os.makedirs(pkg, exist_ok=True)
    shutil.copy2(os.path.join(repo, "pytglib", "__init__.py"), os.path.join(pkg, "__init__.py"))
    shutil.copy2(module, os.path.join(pkg, os.path.basename(module)))
    return os.path.abspath(root)


def undefined_symbols(text, keep=12):
    """The lines after the linker's 'Undefined symbols' header (cause of a failed link)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "Undefined symbols" in line:
            return lines[i:i + keep]
    return []


def importable(python, extra_path=None):
    """(ok, version or error) of ``import pytglib`` in a fresh interpreter."""
    import subprocess
    code = ("import sys\n"
            + (f"sys.path.insert(0, {extra_path!r})\n" if extra_path else "")
            + "import pytglib as t; print(getattr(t, '__version__', 'unknown'))")
    r = subprocess.run([python, "-c", code], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout.strip() or r.stderr.strip()[-300:])


def _tool_version(cmd):
    import subprocess
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else None
    except (OSError, subprocess.SubprocessError):
        return None


def redact_home(text, home=None):
    """Replace the user's home folder with '~' so the record carries no local
    account path (the submission bundle's privacy scan rejects them)."""
    home = home or os.path.expanduser("~")
    return text.replace(home.rstrip("/") + "/", "~/") if home and home != "/" else text


def build(args, runner=None, python=None, platform=None):
    """Fetch, pin, build and register pytglib; write the record; 0 ok / 2 failed.

    On macOS the OpenMP runtime comes from Homebrew's libomp (``brew install
    libomp``); without it the record says so and nothing else is tried. The CMake
    route runs as written first and, only if that fails, once more with -Werror
    filtered out (recorded as ``werror_relaxed``). Without a cmake on PATH, it
    first installs PyPI's cmake wheel into this interpreter."""
    import shutil
    import subprocess
    import sysconfig

    python = python or sys.executable
    platform = platform or sys.platform

    def default_runner(c, env=None):
        return subprocess.run(c, capture_output=True, text=True,
                              env=None if env is None else {**os.environ, **env})
    runner = runner or default_runner
    repo = os.path.join(args.dest, "tglib")
    plan = build_plan(repo, python, args.jobs)
    rec = {"url": REPO_URL, "commit_pinned": COMMIT, "route": None, "ok": False,
           "steps": [], "python": python, "platform": platform, "werror_relaxed": False,
           "tools": {"cmake": _tool_version(["cmake", "--version"]),
                     "cxx": _tool_version(["c++", "--version"]),
                     "git": _tool_version(["git", "--version"])}}

    def do(stage, cmds, env=None):
        for c in cmds:
            r = runner(c, env=env) if env else runner(c)
            text = (r.stdout or "") + (r.stderr or "")
            tail = text[-1500:]
            rec["steps"].append({"stage": stage, "cmd": c, "rc": r.returncode, "tail": tail,
                                 **({"env": env} if env else {}),
                                 **({"undefined_symbols": undefined_symbols(text)}
                                    if r.returncode and "Undefined symbols" in text else {})})
            print(f"[build] {stage}: rc={r.returncode}  {' '.join(c[:4])} ...")
            if r.returncode != 0:
                return False
        return True

    def cmake_ready():
        rec["cmake_exe"] = shutil.which("cmake")
        if rec["cmake_exe"]:
            return True
        rec["cmake_exe"] = os.path.join(os.path.dirname(python), "cmake")
        return do("cmake-tool", [[python, "-m", "pip", "install", "cmake"]])

    def finish():
        ok, ver = importable(python) if rec["route"] else (False, "not built")
        rec["ok"], rec["import"] = ok, ver
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(redact_home(json.dumps(rec, indent=2)))
        print(f"[build] route={rec['route']} import={'ok' if ok else 'failed'} ({ver}); wrote {args.out}")
        return 0 if ok else 2

    os.makedirs(args.dest, exist_ok=True)
    extra = []
    if platform == "darwin":
        r = runner(["brew", "--prefix", "libomp"])
        prefix = (r.stdout or "").strip() if r.returncode == 0 else ""
        if not prefix or not os.path.isfile(os.path.join(prefix, "lib", "libomp.dylib")):
            rec["openmp"] = {"found": False, "hint": "brew install libomp"}
            print("[build] libomp not found; run: brew install libomp")
            return finish()
        rec["openmp"] = {"found": True, "prefix": prefix}
        extra = openmp_args(prefix)

    fetched = os.path.isdir(os.path.join(repo, ".git")) or do("fetch", plan["fetch"])
    if fetched and do("pin", plan["pin"]):
        rc_head = runner(["git", "-C", repo, "rev-parse", "HEAD"])
        rec["commit"] = (rc_head.stdout or "").strip() or None
        rec["local_diff"] = (runner(["git", "-C", repo, "diff", "--stat"]).stdout or "").strip()
        pip_env = {"CMAKE_ARGS": " ".join(f'"{a}"' if " " in a else a for a in extra)} if extra else None
        if do("pip", plan["pip"], env=pip_env) and importable(python)[0]:
            rec["route"] = "pip"
        elif cmake_ready():
            built = do("cmake", build_plan(repo, python, args.jobs, rec["cmake_exe"], extra)["cmake"])
            bdir = plan["build_dir"]
            if not built:
                bdir = os.path.join(repo, "tglib_cpp", "build-release-nowerror")
                rec["werror_relaxed"] = True
                built = do("cmake-nowerror", build_plan(
                    repo, python, args.jobs, rec["cmake_exe"],
                    extra + no_werror_args(args.dest), build_dir=bdir)["cmake"])
            mod = find_module(bdir) if built else None
            rec["module"] = mod
            if mod:
                pkg_root = stage_package(repo, mod, args.dest)
                rec["package_root"] = pkg_root
                site_dir = sysconfig.get_paths()["purelib"]
                pth = os.path.join(site_dir, "strata_tglib_build.pth")
                with open(pth, "w") as fh:
                    fh.write(pkg_root + "\n")
                rec["pth_file"] = pth
                rec["route"] = "cmake"
    return finish()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--datasets", default=",".join(DATASETS))
    c.add_argument("--sample", type=int, default=64)
    c.add_argument("--out", default="results/real/tglib_check.json")
    b = sub.add_parser("build")
    b.add_argument("--dest", default="external")
    b.add_argument("--jobs", type=int, default=8)
    b.add_argument("--out", default="results/real/tglib_build.json")
    t = sub.add_parser("time")
    t.add_argument("--datasets", default=",".join(DATASETS))
    t.add_argument("--check", default="results/real/tglib_check.json")
    t.add_argument("--procs", default="1,18")
    t.add_argument("--out", default="results/real/tglib_time.json")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        return build(args)
    try:
        import pytglib  # noqa: F401
    except ImportError:
        print("pytglib not importable; run: python3 experiments/tglib_baseline.py build")
        return 2
    return check(args) if args.cmd == "check" else time_all(args)


if __name__ == "__main__":
    sys.exit(main())
