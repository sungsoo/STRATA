from temporal_strata import temporal_strata_reachability, bits_to_matrix, all_sources_temporal_bfs


def test_strict_temporal_reachability_matches_bfs():
    n = 5
    edges = [
        (0, 0, 1),
        (0, 1, 2),  # not chainable under strict same-time semantics
        (1, 2, 3),
        (2, 3, 4),
    ]
    bits = temporal_strata_reachability(n, edges, directed=True, same_time_mode="strict")
    M = bits_to_matrix(bits, n)
    B = all_sources_temporal_bfs(n, edges, directed=True, same_time_mode="strict")
    assert (M == B).all()
    assert M[0, 1]
    assert not M[0, 2]  # same timestamp chain is disallowed
    assert not M[0, 4]


def test_closure_same_timestamp_allows_chaining():
    n = 3
    edges = [(0, 0, 1), (0, 1, 2)]
    bits = temporal_strata_reachability(n, edges, directed=True, same_time_mode="closure")
    M = bits_to_matrix(bits, n)
    assert M[0, 2]
