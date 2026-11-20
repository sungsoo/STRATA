"""STRATA: benchmark-ready temporal reachability extensions for STRATA.

The package implements STRATA: Bit-Parallel Causal Frontier Algebra for
Temporal Reachability Graphs. It includes the TA-STRATA bit-parallel temporal
reachability engine, exact per-source temporal BFS baseline, Apple-silicon-safe public
benchmark download/standardization helpers, and reproducibility scripts.
"""

from .stream import TemporalEdge, sort_and_validate_edges, generate_temporal_stream
from .bitset_reachability import temporal_strata_reachability, bits_to_matrix, estimate_bitset_memory_bytes
from .baseline import all_sources_temporal_bfs
from .dynamic_distance import apsp_scipy, insert_edge_update_distance

__all__ = [
    "TemporalEdge",
    "sort_and_validate_edges",
    "generate_temporal_stream",
    "temporal_strata_reachability",
    "bits_to_matrix",
    "estimate_bitset_memory_bytes",
    "all_sources_temporal_bfs",
    "apsp_scipy",
    "insert_edge_update_distance",
]
