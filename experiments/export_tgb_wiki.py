#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Export tgbl-wiki-v2 from py-tgb cache to TA-STRATA standard edge format.")
    p.add_argument("--root", default="data/raw/tgb")
    p.add_argument("--out", default="data/standard/tgbl-wiki-v2.edges")
    p.add_argument("--max-nodes", type=int, default=None)
    p.add_argument("--max-edges", type=int, default=None)
    args = p.parse_args()

    try:
        from tgb.linkproppred.dataset import LinkPropPredDataset  # type: ignore
    except Exception as e:
        raise SystemExit("py-tgb is required: pip install py-tgb") from e

    ds = LinkPropPredDataset(name="tgbl-wiki-v2", root=args.root)
    data = ds.full_data
    src = data["sources"]
    dst = data["destinations"]
    ts = data["timestamps"]

    selected: dict[int, int] = {}
    rows = []
    for u0, v0, t0 in zip(src, dst, ts):
        u0, v0, t0 = int(u0), int(v0), int(t0)
        if args.max_nodes is not None:
            for x in (u0, v0):
                if x not in selected and len(selected) < args.max_nodes:
                    selected[x] = len(selected)
            if u0 not in selected or v0 not in selected:
                continue
        else:
            for x in (u0, v0):
                if x not in selected:
                    selected[x] = len(selected)
        rows.append((selected[u0], selected[v0], t0))
        if args.max_edges is not None and len(rows) >= args.max_edges:
            break

    meta = {
        "dataset": "tgbl-wiki-v2",
        "n": len(selected),
        "temporal_edges": len(rows),
        "unique_timestamps": len(set(t for _, _, t in rows)),
        "timestamp_min": min((t for _, _, t in rows), default=None),
        "timestamp_max": max((t for _, _, t in rows), default=None),
        "source": "py-tgb LinkPropPredDataset.full_data",
        "max_nodes": args.max_nodes,
        "max_edges": args.max_edges,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wt", encoding="utf-8") as f:
        f.write("# TA-STRATA standardized temporal edge file\n")
        f.write("# columns: src dst timestamp\n")
        f.write("# meta: " + json.dumps(meta, sort_keys=True) + "\n")
        for u, v, t in rows:
            f.write(f"{u} {v} {t}\n")
    print(json.dumps({"ok": True, "standard_path": str(out), "metadata": meta}, indent=2))


if __name__ == "__main__":
    main()
