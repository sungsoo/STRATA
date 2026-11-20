# Temporal STRATA — Bit-Parallel Causal Frontier Algebra for Temporal Reachability

This directory is the **temporal reachability extension** of STRATA. While the
original STRATA (see the repository root: `02_Implementations`,
`02_Implementation_BRANDES`, `03_Experiments`, and https://dknife.github.io/STRATA/)
targets static all-pairs shortest paths, this extension computes exact
many-source / all-pairs **temporal reachability**: a journey is valid only if its
edge timestamps are non-decreasing.

Each vertex holds a packed bitset of the sources that causally reach it;
timestamped contacts propagate frontier deltas in time order, giving
`O(m⌈n/w⌉)` time and `O(n⌈n/w⌉)`-word memory (`w = 64`).

## Layout

```
05_Temporal_STRATA/
├── temporal_strata/          # core library
│   ├── packed_reachability.py  #   temporal_strata_packed(): the algorithm
│   ├── io.py                   #   standard edge-file loader
│   ├── baseline.py             #   exact per-source temporal BFS reference
│   ├── bitset_reachability.py  #   big-integer bitset variant
│   ├── original_baselines.py   #   baseline registry
│   ├── stream.py               #   temporal-edge utilities
│   └── dynamic_distance.py     #   dynamic-distance helpers
├── experiments/              # one script per experiment
│   ├── baseline_ablation.py        #   speedup decomposition
│   ├── ablation_pruning.py         #   pruning neutral for strict
│   ├── ablation_closure_detector.py#   closure termination signal
│   ├── ablation_naive_temporal.py  #   non-triviality of scheduling
│   ├── ablation_blocking.py        #   memory/pass trade-off
│   ├── ablation_deletion.py        #   deletion repair cost
│   ├── ablation_throughput_curve.py#   compiled throughput (needs numba)
│   ├── ablation_graphblas.py       #   fair GraphBLAS comparison
│   ├── ablation_real_temporal.py   #   real ego-Facebook (n=4039)
│   ├── make_real_temporal.py       #   build the real temporal stream
│   ├── hit_prioritization.py       #   prospective hit-prioritization case study
│   ├── numba_kernel.py             #   compiled (JIT) kernel
│   └── export_tgb_wiki.py          #   TGB2 → standard-format exporter
├── tests/                    # correctness self-checks (pytest)
├── REPRODUCTION_GUIDE_KO.md  # step-by-step reproduction (Korean)
├── REPRODUCTION_GUIDE_EN.md  # step-by-step reproduction (English)
├── requirements-mac.txt
└── requirements-ubuntu-cuda.txt
```

The bundled real dataset (SNAP ego-Facebook, n=4039) lives at
`datasets_real_temporal/static/facebook_combined.txt` (since STRATA-v19; the
scripts use `temporal_strata.datasets.FACEBOOK`). The legacy copy under the
repository root's `04_Datasets/real-world/` is unchanged.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt          # CPU / macOS
pip install numba                             # optional: compiled kernel / throughput
pip install python-graphblas                  # optional: GraphBLAS comparison

# correctness first
python3 -m pytest tests/ -q

# reproduce (writes results/ablation/*.csv)
python3 experiments/make_real_temporal.py
python3 experiments/ablation_real_temporal.py
python3 experiments/ablation_graphblas.py
```

See `REPRODUCTION_GUIDE_KO.md` / `REPRODUCTION_GUIDE_EN.md` for full step-by-step
instructions, including downloading and running large public graphs locally.

## Standard input format

Plain text (or `.gz`), one timestamped contact per line:

```
src dst timestamp
```

`src`/`dst` are 0-based integer vertex IDs; `timestamp` is an integer. A
`# meta: {"n": <count>}` header line is optional.

```python
from temporal_strata.io import load_standard_edges
n, edges, meta = load_standard_edges("graph.edges")
```

## Core API

```python
from temporal_strata.packed_reachability import (
    temporal_strata_packed, reachable_pairs, words_for, packed_state_bytes)

R = temporal_strata_packed(
        n, edges,                 # edges: iterable of (t, u, v)
        directed=True,
        same_time_mode="strict",  # "strict" or "closure"
        block_bits=None)          # set β to bound per-pass memory
pairs = reachable_pairs(R, n)
```

- `strict` blocks same-timestamp chaining via a per-group snapshot.
- `closure` iterates to the least fixed point within a timestamp group.
- `block_bits=β` processes β source-columns per pass; results are identical to the
  single-pass run (blocking is correctness-invariant).


## Thread-parallel core (TPDS v1)

- `temporal_strata/parallel.py` — deterministic column-partitioned parallel kernel (strict/closure, directed/undirected); bit-identical to the sequential core for any thread count / chunking (paper Lemma 3, Theorem 3, Corollary 1). No atomics, no locks, zero inter-worker communication.
- `tests/test_parallel.py` — 11 determinacy/anchoring tests (suite total 40).
- `experiments/parallel_scaling.py` — strong-scaling harness; verifies bit-identity before timing, compares against a compiled *parallel* per-source temporal BFS at equal thread counts, refuses to emit speedups on single-core hosts, and emits `table_parallel_scaling.tex` for the paper.
