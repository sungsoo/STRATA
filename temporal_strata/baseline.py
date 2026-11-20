from __future__ import annotations

from itertools import groupby
from typing import Iterable, List, Tuple
import numpy as np

from .stream import TemporalEdge, sort_and_validate_edges


def _single_source_temporal_bfs(
    n: int,
    stream: List[TemporalEdge],
    source: int,
    *,
    directed: bool,
    same_time_mode: str,
) -> np.ndarray:
    reach = np.zeros(n, dtype=bool)
    reach[source] = True

    for _, group_iter in groupby(stream, key=lambda e: e.t):
        group = list(group_iter)
        if same_time_mode == "strict":
            before = reach.copy()
            after = reach.copy()
            for e in group:
                if before[e.u]:
                    after[e.v] = True
                if not directed and before[e.v]:
                    after[e.u] = True
            reach = after
        elif same_time_mode == "closure":
            changed = True
            while changed:
                changed = False
                for e in group:
                    if reach[e.u] and not reach[e.v]:
                        reach[e.v] = True
                        changed = True
                    if not directed and reach[e.v] and not reach[e.u]:
                        reach[e.u] = True
                        changed = True
        else:
            raise ValueError("same_time_mode must be 'strict' or 'closure'")
    return reach


def all_sources_temporal_bfs(
    n: int,
    edges: Iterable[TemporalEdge | Tuple[int, int, int]],
    *,
    directed: bool = True,
    same_time_mode: str = "strict",
) -> np.ndarray:
    """Reference all-source temporal BFS.

    This baseline intentionally repeats the temporal scan once per source. It is
    simple and exact, but it performs redundant stream scans and is therefore a
    useful comparison target for Temporal STRATA's bit-parallel propagation.
    """
    stream = sort_and_validate_edges(n, edges)
    out = np.zeros((n, n), dtype=bool)
    for s in range(n):
        out[s, :] = _single_source_temporal_bfs(
            n, stream, s, directed=directed, same_time_mode=same_time_mode
        )
    return out
