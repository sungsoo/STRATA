from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator, Tuple, Dict, Any, Optional

from .stream import TemporalEdge


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def parse_temporal_edge_lines(path: str | Path, *, has_header: bool = False, delimiter: Optional[str] = None) -> Iterator[tuple[int, int, int]]:
    """Parse raw temporal edges from files whose rows contain src dst timestamp.

    Output order is (t, u, v), matching TemporalEdge tuple interpretation in the
    TA-STRATA package.
    """
    p = Path(path)
    with _open_text(p) as f:
        if has_header:
            next(f, None)
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.split(delimiter) if delimiter else line.split()
            if len(parts) < 3:
                continue
            u, v, t = int(parts[0]), int(parts[1]), int(float(parts[2]))
            yield (t, u, v)


def load_standard_edges(path: str | Path) -> tuple[int, list[TemporalEdge], dict[str, Any]]:
    """Load a standardized TA-STRATA edge file with optional metadata header.

    Standard files are plain text or gzip files with lines:
      src dst timestamp
    If a line starts with '# meta:', it must contain a JSON metadata object.
    """
    p = Path(path)
    meta: dict[str, Any] = {}
    raw: list[tuple[int, int, int]] = []
    with _open_text(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# meta:"):
                try:
                    meta.update(json.loads(line[len("# meta:"):].strip()))
                except Exception:
                    pass
                continue
            if line.startswith("#"):
                continue
            a = line.split()
            if len(a) < 3:
                continue
            u, v, t = int(a[0]), int(a[1]), int(float(a[2]))
            raw.append((t, u, v))
    max_node = max((max(u, v) for _, u, v in raw), default=-1)
    n = int(meta.get("n", max_node + 1))
    edges = [TemporalEdge(t=t, u=u, v=v) for t, u, v in raw]
    return n, edges, meta


def convert_snap_to_standard(
    raw_path: str | Path,
    out_path: str | Path,
    *,
    dataset_name: str,
    max_nodes: int | None = None,
    max_edges: int | None = None,
    relabel_compact: bool = True,
) -> dict[str, Any]:
    """Convert SNAP src dst timestamp text/gz file to TA-STRATA standard format.

    If max_nodes is set, the converter keeps only temporal edges whose endpoints
    are among the first max_nodes distinct raw node IDs encountered in timestamp
    order. Compact relabeling maps selected raw IDs to 0..n-1.
    """
    selected: dict[int, int] = {}
    rows: list[tuple[int, int, int]] = []
    for t, u_raw, v_raw in parse_temporal_edge_lines(raw_path):
        if max_nodes is not None:
            for x in (u_raw, v_raw):
                if x not in selected and len(selected) < max_nodes:
                    selected[x] = len(selected)
            if u_raw not in selected or v_raw not in selected:
                continue
        elif relabel_compact:
            for x in (u_raw, v_raw):
                if x not in selected:
                    selected[x] = len(selected)
        if relabel_compact:
            u, v = selected[u_raw], selected[v_raw]
        else:
            u, v = u_raw, v_raw
        rows.append((u, v, t))
        if max_edges is not None and len(rows) >= max_edges:
            break
    if not relabel_compact:
        n = max((max(u, v) for u, v, _ in rows), default=-1) + 1
    else:
        n = len(selected)
    meta = {
        "dataset": dataset_name,
        "n": n,
        "temporal_edges": len(rows),
        "unique_timestamps": len(set(t for _, _, t in rows)),
        "timestamp_min": min((t for _, _, t in rows), default=None),
        "timestamp_max": max((t for _, _, t in rows), default=None),
        "relabel_compact": relabel_compact,
        "max_nodes": max_nodes,
        "max_edges": max_edges,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wt", encoding="utf-8") as f:
        f.write("# TA-STRATA standardized temporal edge file\n")
        f.write("# columns: src dst timestamp\n")
        f.write("# meta: " + json.dumps(meta, sort_keys=True) + "\n")
        for u, v, t in rows:
            f.write(f"{u} {v} {t}\n")
    return meta


def convert_static_to_standard(
    raw_path: str | Path,
    out_path: str | Path,
    *,
    dataset_name: str,
    max_nodes: int | None = None,
    max_edges: int | None = None,
    relabel_compact: bool = True,
    timestamp_policy: str = "edge_order",
    undirected: bool = False,
) -> dict[str, Any]:
    """Convert a static edge-list to the STRATA temporal-standard file.

    Static datasets are represented as temporal streams for unified runners.
    ``timestamp_policy=edge_order`` preserves a deterministic one-edge-per-step
    order; ``timestamp_policy=zero`` puts all edges at timestamp 0 and is useful
    for strict snapshot reachability. Comment lines starting with # or % are
    ignored.
    """
    p = Path(raw_path)
    selected: dict[int, int] = {}
    rows: list[tuple[int, int, int]] = []
    with _open_text(p) as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u_raw, v_raw = int(parts[0]), int(parts[1])
            if u_raw == v_raw:
                continue
            if max_nodes is not None:
                for x in (u_raw, v_raw):
                    if x not in selected and len(selected) < max_nodes:
                        selected[x] = len(selected)
                if u_raw not in selected or v_raw not in selected:
                    continue
            elif relabel_compact:
                for x in (u_raw, v_raw):
                    if x not in selected:
                        selected[x] = len(selected)
            u, v = (selected[u_raw], selected[v_raw]) if relabel_compact else (u_raw, v_raw)
            t = 0 if timestamp_policy == "zero" else idx
            rows.append((u, v, t))
            if undirected:
                rows.append((v, u, t))
            idx += 1
            if max_edges is not None and len(rows) >= max_edges:
                break
    n = len(selected) if relabel_compact else max((max(u, v) for u, v, _ in rows), default=-1) + 1
    meta = {
        "dataset": dataset_name,
        "source_mode": "static_as_temporal",
        "n": n,
        "temporal_edges": len(rows),
        "unique_timestamps": len(set(t for _, _, t in rows)),
        "timestamp_policy": timestamp_policy,
        "undirected_expanded": undirected,
        "relabel_compact": relabel_compact,
        "max_nodes": max_nodes,
        "max_edges": max_edges,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wt", encoding="utf-8") as f:
        f.write("# TA-STRATA standardized temporal edge file\n")
        f.write("# columns: src dst timestamp\n")
        f.write("# meta: " + json.dumps(meta, sort_keys=True) + "\n")
        for u, v, t in rows:
            f.write(f"{u} {v} {t}\n")
    return meta


def load_standard_arrays(path: str | Path):
    """Array-native loader.

    Returns ``(n, t_raw, us, vs, meta)`` with int64 arrays in file order, the
    same edges, ``n`` and metadata as :func:`load_standard_edges` (timestamps
    parsed as ``int(float(x))``), without building TemporalEdge objects.
    Standard files have three numeric columns; comment lines start with '#'.
    """
    import numpy as np

    p = Path(path)
    meta: dict[str, Any] = {}
    with _open_text(p) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if not s.startswith("#"):
                break
            if s.startswith("# meta:"):
                try:
                    meta.update(json.loads(s[len("# meta:"):].strip()))
                except Exception:
                    pass
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # empty file -> empty array, not a warning
        a = np.loadtxt(p, dtype=np.float64, comments="#", usecols=(0, 1, 2), ndmin=2)
    us = a[:, 0].astype(np.int64)
    vs = a[:, 1].astype(np.int64)
    ts = np.trunc(a[:, 2]).astype(np.int64)
    max_node = int(max(us.max(), vs.max())) if us.size else -1
    n = int(meta.get("n", max_node + 1))
    return n, ts, us, vs, meta
