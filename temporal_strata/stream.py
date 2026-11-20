from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import itertools
import random


@dataclass(frozen=True, order=True)
class TemporalEdge:
    """Timestamped directed or undirected contact.

    Parameters
    ----------
    t:
        Discrete or continuous timestamp. The implementation only requires
        sortable values.
    u, v:
        Zero-based endpoint identifiers.
    """

    t: int
    u: int
    v: int


def sort_and_validate_edges(n: int, edges: Iterable[TemporalEdge | Tuple[int, int, int]]) -> List[TemporalEdge]:
    """Return a timestamp-sorted edge stream after basic validation.

    Tuple inputs are interpreted as (t, u, v). The function rejects negative
    vertices and self-loops are kept because temporal self-contacts can be used
    by some generators to model waiting events.
    """
    out: List[TemporalEdge] = []
    for e in edges:
        if isinstance(e, TemporalEdge):
            te = e
        else:
            t, u, v = e
            te = TemporalEdge(int(t), int(u), int(v))
        if te.u < 0 or te.v < 0 or te.u >= n or te.v >= n:
            raise ValueError(f"edge endpoint out of range for n={n}: {te}")
        out.append(te)
    out.sort(key=lambda x: (x.t, x.u, x.v))
    return out


def generate_temporal_stream(
    n: int,
    m: int,
    timestamps: int,
    *,
    directed: bool = True,
    seed: int = 7,
    skew: float = 1.2,
) -> List[TemporalEdge]:
    """Generate a reproducible synthetic temporal edge stream.

    The generator uses a light preferential attachment bias so that the produced
    stream has skewed degrees, which is a more realistic stressor for temporal
    reachability than uniform Erdos-Renyi contacts. It deliberately has no GPU
    dependency and is suitable for a MacBook Pro (Apple M5 Max) workstation.
    """
    if n <= 1:
        raise ValueError("n must be larger than 1")
    if m < 0:
        raise ValueError("m must be nonnegative")
    if timestamps <= 0:
        raise ValueError("timestamps must be positive")

    rng = random.Random(seed)
    # Preferential weights: a small number of high-activity vertices.
    weights = [1.0 / ((i + 1) ** (1.0 / max(skew, 1e-9))) for i in range(n)]
    # Precomputed cumulative weights: random.choices
    # with weights= rebuilds this O(n) list on every call, making generation
    # O(m*n). Passing the identical list as cum_weights= performs exactly the
    # same random draws (same RNG calls, same bisect), so every stream is
    # bit-identical to the pre-v3 generator (tests/test_stream_generator.py).
    cum_weights = list(itertools.accumulate(weights))
    vertices = list(range(n))
    edges: List[TemporalEdge] = []
    for _ in range(m):
        u = rng.choices(vertices, cum_weights=cum_weights, k=1)[0]
        v = rng.choices(vertices, cum_weights=cum_weights, k=1)[0]
        while v == u:
            v = rng.choices(vertices, cum_weights=cum_weights, k=1)[0]
        if not directed and rng.random() < 0.5:
            u, v = v, u
        t = rng.randrange(timestamps)
        edges.append(TemporalEdge(t, u, v))
    edges.sort(key=lambda x: (x.t, x.u, x.v))
    return edges


# --------------------------------------------------------------------------- #
# Array-native stream path
# --------------------------------------------------------------------------- #
# The object path (TemporalEdge lists, sort with a key lambda, attribute reads)
# dominated end-to-end time in the M5 Max measurement. The functions below keep
# the stream as three int64 arrays from loading to the kernel. They are
# equivalent to the object path: same validation (same ValueError for an
# out-of-range endpoint, self-contacts allowed), same order as
# ``sort(key=(t, u, v))`` (stable lexicographic sort; fully equal edges are
# indistinguishable), and the same timestamp handling (TemporalEdge timestamps
# are sorted by their raw value and truncated to int64 afterwards, tuple
# timestamps are truncated with int() first, exactly as before).
# tests/test_array_path.py checks array-for-array equality with the object path.

def edges_to_arrays(edges) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """(t_raw, us, vs) arrays from TemporalEdge objects or (t, u, v) tuples.

    ``t_raw`` keeps TemporalEdge timestamps as given (so non-integer values
    still sort exactly as in the object path); tuple timestamps go through
    ``int()`` like :func:`sort_and_validate_edges`.
    """
    import numpy as np

    if isinstance(edges, tuple) and len(edges) == 3 and all(hasattr(a, "shape") for a in edges):
        t, u, v = edges
        return np.asarray(t), np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64)
    t_raw, us, vs = [], [], []
    for e in edges:
        if isinstance(e, TemporalEdge):
            t_raw.append(e.t)
            us.append(int(e.u))
            vs.append(int(e.v))
        else:
            t, u, v = e
            t_raw.append(int(t))
            us.append(int(u))
            vs.append(int(v))
    t_arr = np.array(t_raw) if t_raw else np.zeros(0, dtype=np.int64)
    return t_arr, np.array(us, dtype=np.int64), np.array(vs, dtype=np.int64)


def dense_time_rank(t_raw) -> "np.ndarray":
    """Dense rank of each timestamp (0 for the earliest); O(m) when t is
    already non-decreasing, else one stable (adaptive) argsort. Equal timestamps get equal ranks, so ordering by rank equals
    ordering by t."""
    import numpy as np

    t = np.asarray(t_raw)
    if t.size == 0:
        return np.zeros(0, dtype=np.int64)
    d = np.diff(t)
    if (d >= 0).all():
        r = np.empty(t.size, dtype=np.int64)
        r[0] = 0
        np.cumsum(d > 0, out=r[1:])
        return r
    # a stable argsort is Timsort for int64, near-linear on the
    # nearly time-ordered logs of real data (np.unique sorted from scratch)
    order = np.argsort(t, kind="stable")
    ts = t[order]
    r = np.empty(t.size, dtype=np.int64)
    r[order] = np.concatenate(([0], np.cumsum(ts[1:] != ts[:-1])))
    return r


def time_then_group_order(t, us, vs) -> "np.ndarray":
    """Permutation that sorts by (t, u, v), equal to ``np.lexsort((vs, us, t))``.

    Used when no packed int64 key fits (v22, R56: sx-stackoverflow needs 70 bits).
    One stable argsort on t alone (Timsort: near-linear on nearly time-ordered
    logs), then a lexicographic sort only inside timestamp groups with more than
    one edge, which are few on real logs. Ties keep input order in both steps,
    as lexsort does, so the permutation is identical."""
    import numpy as np

    t = np.asarray(t)
    order = np.argsort(t, kind="stable")
    if order.size < 2:
        return order
    ts = t[order]
    newg = np.empty(ts.size, dtype=np.bool_)
    newg[0] = True
    np.not_equal(ts[1:], ts[:-1], out=newg[1:])
    gid = np.cumsum(newg) - 1
    sizes = np.bincount(gid)
    sel = np.flatnonzero(sizes[gid] > 1)
    if sel.size:
        o = order[sel]
        order[sel] = o[np.lexsort((vs[o], us[o], gid[sel]))]
    return order


def sort_validate_arrays(n: int, t_raw, us, vs, info=None) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Validate endpoints and sort by (t, u, v); returns int64 (ts, us, vs).
    ``info`` (optional dict) receives ``sort_path`` when no packed key fits:
    time-then-group (integer timestamps) or lexsort."""
    import numpy as np

    us = np.asarray(us, dtype=np.int64)
    vs = np.asarray(vs, dtype=np.int64)
    t_raw = np.asarray(t_raw)
    if us.size:
        bad = (us < 0) | (vs < 0) | (us >= n) | (vs >= n)
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            te = TemporalEdge(t_raw[i].item(), int(us[i]), int(vs[i]))
            raise ValueError(f"edge endpoint out of range for n={n}: {te}")
    order = None
    if t_raw.dtype.kind in "iu" and t_raw.size:
        # one packed int64 key (t, u, v) when it fits: a single stable argsort
        # instead of a three-key lexsort; identical order
        tmin = int(t_raw.min())
        span = int(t_raw.max()) - tmin + 1
        if span * n * n < 2**62:
            key = (t_raw.astype(np.int64) - tmin) * (n * n) + us * n + vs
            order = np.argsort(key, kind="stable")
        else:
            # a wide timestamp range (e.g. SNAP seconds over
            # years) overflows the packed key. Replace t by its dense rank
            # (same order, at most m distinct values) and pack again.
            rank = dense_time_rank(t_raw)
            if (int(rank.max()) + 1) * n * n < 2**62:
                key = rank * np.int64(n * n) + us * np.int64(n) + vs
                order = np.argsort(key, kind="stable")
    if order is None:
        if t_raw.dtype.kind in "iu" and t_raw.size:
            order = time_then_group_order(t_raw, us, vs)
            if info is not None:
                info["sort_path"] = "time-then-group"
        else:
            order = np.lexsort((vs, us, t_raw))
            if info is not None:
                info["sort_path"] = "lexsort"
    ts = t_raw[order]
    ts = ts.astype(np.int64) if ts.dtype != np.int64 else ts
    return ts, us[order], vs[order]


def generate_temporal_arrays(
    n: int,
    m: int,
    timestamps: int,
    *,
    directed: bool = True,
    seed: int = 7,
    skew: float = 1.2,
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Same stream as :func:`generate_temporal_stream`, as sorted int64 arrays.

    The random draws are identical; only the TemporalEdge objects and the
    key-lambda sort are replaced by arrays and a lexicographic sort.
    """
    import numpy as np

    if n <= 1:
        raise ValueError("n must be larger than 1")
    if m < 0:
        raise ValueError("m must be nonnegative")
    if timestamps <= 0:
        raise ValueError("timestamps must be positive")
    rng = random.Random(seed)
    weights = [1.0 / ((i + 1) ** (1.0 / max(skew, 1e-9))) for i in range(n)]
    cum_weights = list(itertools.accumulate(weights))
    vertices = list(range(n))
    ts = np.empty(m, dtype=np.int64)
    us = np.empty(m, dtype=np.int64)
    vs = np.empty(m, dtype=np.int64)
    choices, rand, randrange = rng.choices, rng.random, rng.randrange
    for i in range(m):
        u = choices(vertices, cum_weights=cum_weights, k=1)[0]
        v = choices(vertices, cum_weights=cum_weights, k=1)[0]
        while v == u:
            v = choices(vertices, cum_weights=cum_weights, k=1)[0]
        if not directed and rand() < 0.5:
            u, v = v, u
        ts[i] = randrange(timestamps)
        us[i] = u
        vs[i] = v
    order = np.lexsort((vs, us, ts))
    return ts[order], us[order], vs[order]


def arrays_to_edges(ts, us, vs) -> List[TemporalEdge]:
    """TemporalEdge list from arrays (for references and object-only consumers)."""
    return [TemporalEdge(int(t), int(u), int(v)) for t, u, v in zip(ts.tolist(), us.tolist(), vs.tolist())]
