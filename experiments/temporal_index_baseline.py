#!/usr/bin/env python3
"""Public temporal-index baselines: fetch and inspect (v21, R54-2a, phase 1).

Why: reviewers asked for a temporal reachability index as a baseline. The indexes
the paper cites (TopChain, TILL) have no public implementation; two public tools
from the same group answer single-source temporal queries:

* tgindex -- the Substream index for single-source-all-destination temporal
  distances (Oettershagen and Mutzel, arXiv 2111.10095), with earliest arrival;
* TGLib -- an MIT-licensed temporal graph library with earliest-arrival
  distances (arXiv 2209.12587).

An all-pairs relation from either tool is n single-source queries, with arrival
< infinity read as reachable. Their command-line interfaces and input formats are
not documented outside the repositories, which the cloud workspace cannot reach,
so this phase only fetches each repository at a recorded commit and records its
license, README, build files, and source tree. The adapter (build, semantics
check against STRATA, timing) is written from that record in the next batch, in
the same way as the Temporal Ligra adapter (experiments/temporal_ligra.py).

What:
    python3 experiments/temporal_index_baseline.py probe --dest external \
        --out results/real/index_baseline_probe.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

REPOS = {
    "tgindex": "https://gitlab.com/tgpublic/tgindex.git",
    "tglib": "https://gitlab.com/tgpublic/tglib.git",
}
BUILD_FILES = ("CMakeLists.txt", "Makefile", "setup.py", "pyproject.toml", "meson.build",
               "configure", "build.sh")
LICENSE_NAMES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.txt")
README_NAMES = ("README.md", "README", "README.txt", "readme.md")
SOURCE_SUFFIXES = {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".py"}


def describe(repo_dir):
    """What the adapter needs to know about a fetched repository (no network)."""
    d = Path(repo_dir)
    rec = {"license_file": None, "license_head": None, "readme_head": None,
           "build_files": [], "sources": 0, "top_level": sorted(p.name for p in d.iterdir()
                                                               if not p.name.startswith("."))}
    for name in LICENSE_NAMES:
        if (d / name).is_file():
            rec["license_file"] = name
            rec["license_head"] = (d / name).read_text(errors="replace")[:600]
            break
    for name in README_NAMES:
        if (d / name).is_file():
            rec["readme_head"] = (d / name).read_text(errors="replace")[:6000]
            break
    for p in sorted(d.rglob("*")):
        if ".git" in p.parts or not p.is_file():
            continue
        if p.name in BUILD_FILES:
            rec["build_files"].append(str(p.relative_to(d)))
        if p.suffix in SOURCE_SUFFIXES:
            rec["sources"] += 1
    return rec


def git(args, cwd=None):
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=600)
    return r.returncode, (r.stdout + r.stderr).strip()


def probe(args):
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = {"fetched": _dt.datetime.now().isoformat(timespec="seconds"), "repos": {}}
    ok_all = True
    for name in args.repos.split(","):
        url = REPOS[name]
        d = dest / name
        if not d.exists():
            code, msg = git(["clone", "--quiet", url, str(d)])
            if code != 0:
                out["repos"][name] = {"url": url, "error": msg[-500:]}
                ok_all = False
                print(f"[probe] {name}: clone failed")
                continue
        code, commit = git(["rev-parse", "HEAD"], cwd=d)
        rec = {"url": url, "commit": commit if code == 0 else None}
        rec.update(describe(d))
        out["repos"][name] = rec
        print(f"[probe] {name}: commit {rec['commit']}, license {rec['license_file']}, "
              f"build {rec['build_files'][:3]}, {rec['sources']} source files")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out)
    return 0 if ok_all else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--dest", default="external")
    p.add_argument("--repos", default="tgindex,tglib")
    p.add_argument("--out", default="results/real/index_baseline_probe.json")
    args = ap.parse_args(argv)
    return probe(args)


if __name__ == "__main__":
    sys.exit(main())
