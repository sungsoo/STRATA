"""Witness paths for external-baseline disagreements."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.external_witness import (  # noqa: E402
    build_adjacency,
    earliest_arrival,
    path_to,
    summarize,
    validate_path,
    witness_row,
)
from temporal_strata.packed_reachability import packed_to_matrix, temporal_strata_packed  # noqa: E402
from temporal_strata.stream import generate_temporal_stream  # noqa: E402

# s=0: 0->1 at 1, 1->2 at 1 (same-time chain), 2->3 at 5, 0->4 at 9, 4->3 at 2 (too early)
EDGES = [(1, 0, 1), (1, 1, 2), (5, 2, 3), (9, 0, 4), (2, 4, 3)]


def test_semantics_on_toy():
    adj = build_adjacency(EDGES)
    closure, _ = earliest_arrival(adj, 0, "closure")
    strict, _ = earliest_arrival(adj, 0, "strict")
    assert set(closure) == {0, 1, 2, 3, 4}
    assert set(strict) == {0, 1, 4}


def test_witness_path_is_valid_and_validator_rejects_bad_paths():
    adj = build_adjacency(EDGES)
    arr, pred = earliest_arrival(adj, 0, "closure")
    p = path_to(pred, 3)
    es = set(EDGES)
    assert p == [(0, 1, 1), (1, 2, 1), (2, 3, 5)]
    assert validate_path(p, 0, 3, es, "closure")
    assert not validate_path(p, 0, 3, es, "strict")          # equal timestamps
    assert not validate_path([(0, 4, 9), (4, 3, 2)], 0, 3, es, "closure")  # time goes back
    assert not validate_path([(0, 3, 5)], 0, 3, es, "closure")  # edge not in stream


def test_independent_search_matches_strata_on_random_streams():
    for seed in range(4):
        n = 40
        edges = generate_temporal_stream(n, 300, timestamps=25, directed=True, seed=seed)
        adj = build_adjacency(edges)
        for sem in ("strict", "closure"):
            M = packed_to_matrix(temporal_strata_packed(n, edges, same_time_mode=sem), n)
            for s in range(0, n, 7):
                arr, _ = earliest_arrival(adj, s, sem)
                assert set(arr) == set(int(v) for v in M[s].nonzero()[0]) | {s}


def test_witness_row_and_summary_from_a_diagnosis_record():
    row = {"dataset": "toy", "sources": 2, "diagnosis": [
        {"source": 0, "external_size": 4, "closure_size": 5,
         "only_strata_closure": {"count": 1, "examples": [3]},
         "only_external_vs_closure": {"count": 0, "examples": []}},
        {"source": 2, "external_size": 2, "closure_size": 2,
         "only_strata_closure": {"count": 0, "examples": []},
         "only_external_vs_closure": {"count": 0, "examples": []}},
    ]}
    dis = witness_row(row, 5, EDGES, "closure")
    assert len(dis) == 1 and dis[0]["source"] == 0
    m = dis[0]["missed_by_external"][0]
    assert m["vertex"] == 3 and m["path_valid"] and m["hops"] == 3
    assert dis[0]["independent_matches_strata"]
    s = summarize([{"sources": 2, "disagreements": dis}])
    assert s["agreeing_sources"] == 1 and s["missed_with_valid_witness"] == 1
    assert s["external_only_vertices"] == 0
