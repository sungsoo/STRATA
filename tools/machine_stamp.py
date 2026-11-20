#!/usr/bin/env python3
"""Attach the machine description to result files that do not record it.

the single-thread scripts (perf_table,
compiled_baselines, real_temporal_benchmark, ablations) predate the harness
convention of storing the machine next to every timing. After they run in the
M5 session, this tool writes one JSON with the machine description, the thread
setting, and the SHA-256 and size of each output, so every single-thread number
can be traced to the machine that produced it.

Usage::

  python3 tools/machine_stamp.py --label single_thread \\
      --out results/real/single_thread_machine.json results/perf/perf_table.json ...
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def stamp(paths, label):
    from temporal_strata.parallel import machine_info

    files = []
    for p in paths:
        q = (ROOT / p) if not os.path.isabs(p) else Path(p)
        if not q.exists():
            raise SystemExit(f"missing output {p}")
        data = q.read_bytes()
        files.append({"path": str(p), "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest(),
                      "mtime": _dt.datetime.fromtimestamp(q.stat().st_mtime).isoformat(timespec="seconds")})
    return {"label": label, "stamped": _dt.datetime.now().isoformat(timespec="seconds"),
            "machine": machine_info(), "numba_num_threads_env": os.environ.get("NUMBA_NUM_THREADS"),
            "files": files}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)
    rec = stamp(args.paths, args.label)
    out = ROOT / args.out if not os.path.isabs(args.out) else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(f"wrote {out} ({len(rec['files'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
