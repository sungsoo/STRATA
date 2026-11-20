#!/usr/bin/env python3
"""Measurement session on a second machine (v34, R103).

Why
---
The project rule (2026-11-19, R101 decision B) allows results from a second
machine that the user measures, reported only as comparisons within that machine
(speedups, variant against variant), never as run times against the M5 Max. The
paper's scaling claims come from one 18-core laptop; reviewers ask how column
partitioning and the tile state behave on a server (more cores, other OS, NUMA).

How
---
The server session reuses the M5 session's commands (tools/run_m5_session.py) for
the steps that carry the scaling claims, rewritten for the machine at hand:
  * thread counts: 1, c/8, c/4, c/2, c for c = logical CPUs (``--cores`` overrides);
    "NUMBA_NUM_THREADS=18" becomes c;
  * outputs go to results/server/<tag>/ instead of results/real/, so M5 files are
    never touched and the paper generator (experiments/m5_numbers.py) cannot mix
    machines;
  * every run records the machine (platform, CPU, cores, sockets where visible).
Steps: 16 (fetch sx-stackoverflow), 3 (large synthetic scaling), 30 (fresh /
pre-written / no shared state, unblocked), 26 (copy causes), 24 (tile state over
every block), 28 (tile state across block widths).

What
---
    python3 tools/run_server_session.py --tag <name> --dry-run   # print the commands
    python3 tools/run_server_session.py --tag <name>             # run them
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

SERVER_STEPS = (16, 3, 30, 26, 24, 28)


def thread_grid(cores: int):
    return sorted({max(1, x) for x in (1, cores // 8, cores // 4, cores // 2, cores)})


def rewrite(command: str, cores: int, tag: str) -> str:
    grid = ",".join(str(x) for x in thread_grid(cores))
    out = command.replace("NUMBA_NUM_THREADS=18", f"NUMBA_NUM_THREADS={cores}")
    out = re.sub(r"--threads (?!best:)[0-9][0-9,]*", f"--threads {grid}", out)
    out = out.replace("results/real/", f"results/server/{tag}/")
    return out


def server_commands(cores: int, tag: str):
    from tools import run_m5_session as rs
    steps = {num: (desc, cmds) for num, desc, cmds, _ in rs.STEPS}
    plan = []
    for num in SERVER_STEPS:
        desc, cmds = steps[num]
        if num == 16:
            cmds = [c for c in cmds if "fetch_real_temporal" in c]
        plan.append((num, desc, [rewrite(c, cores, tag) for c in cmds]))
    return plan


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="short machine name, e.g. xeon64")
    ap.add_argument("--cores", type=int, default=os.cpu_count())
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
        raise SystemExit("--tag: letters, digits, '.', '_' or '-' only")
    plan = server_commands(args.cores, args.tag)
    if args.dry_run:
        for num, desc, cmds in plan:
            print(f"# step {num}: {desc}")
            for c in cmds:
                print(c)
        return 0
    out_dir = ROOT / "results" / "server" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    from temporal_strata.parallel import machine_info
    stamp = time.strftime("%Y%m%d-%H%M%S")
    record = {"tag": args.tag, "cores": args.cores, "started": stamp,
              "machine": machine_info(), "steps": []}
    for num, desc, cmds in plan:
        for i, c in enumerate(cmds, 1):
            env = dict(os.environ)
            parts = shlex.split(c)
            while parts and "=" in parts[0] and not parts[0].startswith("-"):
                k, v = parts.pop(0).split("=", 1)
                env[k] = v
            if parts and parts[0] == "python3":
                parts[0] = sys.executable
            t0 = time.time()
            log = out_dir / f"step{num}_{i}.{stamp}.log"
            with open(log, "w") as fh:
                rc = subprocess.call(parts, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
            record["steps"].append({"step": num, "index": i, "command": c, "rc": rc,
                                    "seconds": time.time() - t0, "log": log.name})
            (out_dir / f"session.{stamp}.json").write_text(json.dumps(record, indent=2))
            if rc != 0:
                print(f"step {num}.{i} failed (rc={rc}); see {log}")
                return rc
    record["status"] = "completed"
    (out_dir / f"session.{stamp}.json").write_text(json.dumps(record, indent=2))
    print("completed; results in", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
