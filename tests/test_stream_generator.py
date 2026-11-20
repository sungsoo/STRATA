"""The synthetic generator is bit-identical to its pre-v3 form.

STRATA-v3 passes precomputed cumulative weights to ``random.choices`` instead
of recomputing them per call. This test re-implements the pre-v3 loop verbatim
and requires identical streams, so every synthetic dataset (and every number
derived from one) is unchanged.
"""
from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.stream import TemporalEdge, generate_temporal_stream  # noqa: E402


def _pre_v3(n, m, timestamps, directed=True, seed=7, skew=1.2):
    rng = random.Random(seed)
    weights = [1.0 / ((i + 1) ** (1.0 / max(skew, 1e-9))) for i in range(n)]
    vertices = list(range(n))
    edges = []
    for _ in range(m):
        u = rng.choices(vertices, weights=weights, k=1)[0]
        v = rng.choices(vertices, weights=weights, k=1)[0]
        while v == u:
            v = rng.choices(vertices, weights=weights, k=1)[0]
        if not directed and rng.random() < 0.5:
            u, v = v, u
        t = rng.randrange(timestamps)
        edges.append(TemporalEdge(t, u, v))
    edges.sort(key=lambda x: (x.t, x.u, x.v))
    return edges


@pytest.mark.parametrize("n,m,ts,directed,seed", [
    (2, 50, 3, True, 1), (130, 900, 40, True, 11), (300, 2200, 50, False, 31),
    (2048, 5000, 200, True, 7), (8000, 3000, 200, True, 7),
])
def test_generator_bit_identical_to_pre_v3(n, m, ts, directed, seed):
    new = generate_temporal_stream(n, m, ts, directed=directed, seed=seed)
    old = _pre_v3(n, m, ts, directed=directed, seed=seed)
    assert [(e.t, e.u, e.v) for e in new] == [(e.t, e.u, e.v) for e in old]
