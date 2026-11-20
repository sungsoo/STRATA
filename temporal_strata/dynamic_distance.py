from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path


def apsp_scipy(n: int, edges: Iterable[Tuple[int, int]], *, directed: bool = False) -> np.ndarray:
    """Compute dense unweighted APSP distances using SciPy as a robust baseline."""
    rows = []
    cols = []
    for u, v in edges:
        rows.append(int(u)); cols.append(int(v))
        if not directed:
            rows.append(int(v)); cols.append(int(u))
    data = np.ones(len(rows), dtype=np.int8)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.int8)
    D = shortest_path(A, directed=directed, unweighted=True, return_predecessors=False)
    D = np.where(np.isinf(D), np.iinfo(np.int32).max // 4, D).astype(np.int32)
    return D


def insert_edge_update_distance(
    D: np.ndarray,
    u: int,
    v: int,
    *,
    directed: bool = False,
) -> np.ndarray:
    """Closed-form STRATA distance update for an inserted unweighted edge.

    For directed graphs, only u->v is inserted. For undirected graphs, both
    u->v and v->u are considered. The input D is not modified.
    """
    INF = np.iinfo(np.int32).max // 4
    base = D.astype(np.int64, copy=False)
    cand = base[:, [u]] + 1 + base[[v], :]
    out = np.minimum(base, cand)
    if not directed:
        cand2 = base[:, [v]] + 1 + base[[u], :]
        out = np.minimum(out, cand2)
    out = np.minimum(out, INF).astype(np.int32)
    np.fill_diagonal(out, 0)
    return out
