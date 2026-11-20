# STRATA-final — Detailed Reproduction Guide

This document explains, step by step, how to reproduce the **temporal reachability**
experiments of the STRATA paper **from scratch** using the `STRATA-final.zip`
package, and how to **download large public graphs and run them locally**.

> Package overview
> - **Original STRATA (static APSP)** — `02_Implementations/`,
>   `02_Implementation_BRANDES/`, `03_Experiments/`, `docs/`: code from the initial
>   shared repository (https://dknife.github.io/STRATA/). Not the target of this
>   guide; see its own docs (`docs/howto.html`).
> - **Temporal reachability extension** — `05_Temporal_STRATA/`: the code this guide
>   reproduces (the paper's experiments).
> - **Data** — `05_Temporal_STRATA/datasets_real_temporal/static/facebook_combined.txt` (real SNAP ego-Facebook; since STRATA-v19, a copy of the legacy `04_Datasets/real-world/` file).

Shell commands start with `$`. Unless noted otherwise, run every command **inside the
`05_Temporal_STRATA/` directory**.

---

## 0. TL;DR

```bash
# (1) Unzip and enter the temporal-extension directory
unzip STRATA-final.zip
cd STRATA-final/05_Temporal_STRATA

# (2) Virtualenv + dependencies (pyenv + virtualenv — see 2.2 for full steps)
pyenv install 3.12.3 && pyenv local 3.12.3
pip install --upgrade pip virtualenv
virtualenv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt          # CPU / macOS
# compiled kernel too:  pip install -r requirements-ubuntu-cuda.txt
# GraphBLAS comparison: pip install python-graphblas

# (3) Correctness self-check FIRST
python3 -m pytest tests/ -q

# (4) Reproduce the key experiments (paper §6)
python3 experiments/make_real_temporal.py                                   # temporalize real graph
python3 experiments/ablation_real_temporal.py                               # real ego-Facebook (§6.5)
python3 experiments/baseline_ablation.py --edges /tmp/fb_full_temporal.txt --mode strict \
        --out results/ablation/baseline_ablation.csv                        # speedup decomposition (§6.5)
python3 experiments/ablation_graphblas.py                                   # fair GraphBLAS comparison (§6.5)
python3 experiments/ablation_deletion.py                                    # deletion repair (§6.4)
python3 experiments/ablation_throughput_curve.py                            # throughput 1/W (§6.6)
python3 experiments/hit_prioritization.py --synthetic --out-dir results/hitsel  # case study (§6.7)

# (5) NEW in v21 — real-timestamp evaluation (§"Real-Timestamp Datasets")
#     Two real SNAP temporal graphs ship under datasets_real_temporal/;
#     standardize them and run the exact-verified all-pairs benchmark:
python3 experiments/real_temporal_benchmark.py        # temporal-vs-static + speedup, 3 real datasets
python3 experiments/real_dynamic_and_blocking.py      # real insertion/window/deletion + blocking
python3 experiments/numba_kernel.py --edges datasets_real_temporal/standard/CollegeMsg.txt \
       --out results/real/numba_collegemsg.csv        # compiled kernel, bit-verified on real data
#     To reproduce the real >=10^7-edge point on an unrestricted machine:
python3 experiments/fetch_real_temporal.py --which large   # downloads sx-stackoverflow (~63M edges) etc.

# (6) v22 — regenerate the headline table through the current core + strict-prune ablation
python3 experiments/perf_table.py                     # rewrites table_honest_perf.tex (both LaTeX trees)
python3 experiments/ablation_strict_prune.py          # strict np.any test is pure overhead (2.3-2.5x)

# (7) v23 — compiled-vs-compiled, index baseline, integrated main table
python3 experiments/compiled_baselines.py             # STRATA-numba vs compiled per-source BFS (4.36x geomean); rewrites table_maincompare.tex
python3 experiments/index_crossover.py                # STRATA-index (build+O(1)) vs on-demand BFS; crossover ~2 sources

# (8) v24 — snapshot necessity (backs Contribution 1)
python3 experiments/ablation_snapshot_necessity.py    # no-snapshot is order-dependent + over-counts strict up to 17.7%

# (9) v25 — real-data AML/provenance case study (replaces synthetic drug study)
python3 experiments/aml_tracing.py                    # Bitcoin-OTC: 44.8% false trails, top-20 overlap 0
```

Numbers are written to `results/ablation/*.csv` for comparison against the paper.

---

## 1. Prerequisites

| Item | Recommended |
|------|-------------|
| OS | macOS 13+ (incl. Apple Silicon) or Ubuntu 22.04+ |
| Python | 3.10 – 3.12 |
| RAM | 16GB+ (the real n=4039 all-pairs run fits well) |
| Disk | 2GB (code + results); +several GB for large graphs |
| Network | (optional) only for downloading large public datasets |

Core deps: `numpy`, `pandas`, `scipy`, `matplotlib`, `pytest` (pure-CPU reproduction).
The GraphBLAS comparison needs `python-graphblas`; the compiled kernel / throughput
curve need `numba`.

---

## 2. Step 1 — Install

### 2.1 Unpack and layout

```bash
$ unzip STRATA-final.zip
$ cd STRATA-final/05_Temporal_STRATA
```

`05_Temporal_STRATA/` is the **working directory for the temporal experiments**.

```
STRATA-final/
├── 02_Implementations/         # (original) static APSP, 12 methods — not this guide
├── 02_Implementation_BRANDES/  # (original) Matrix Brandes centrality
├── 03_Experiments/             # (original) static experiments
├── docs/                       # (original) dknife.github.io site
├── 04_Datasets/
│   └── real-world/facebook_combined.txt   # real SNAP ego-Facebook (n=4039)
└── 05_Temporal_STRATA/         # ← the target of this guide
    ├── temporal_strata/          #   core library
    │   ├── packed_reachability.py#     temporal_strata_packed() core
    │   ├── io.py                 #     standard edge-file loader
    │   ├── baseline.py           #     exact temporal BFS reference
    │   └── ...
    ├── experiments/              #   per-experiment scripts
    ├── tests/                    #   correctness self-checks (pytest)
    ├── requirements-mac.txt
    └── requirements-ubuntu-cuda.txt
```

The data (`04_Datasets/`) sits **one level above** `05_Temporal_STRATA/`; scripts
reference it via a relative path (`../04_Datasets/...`).

### 2.2 Virtualenv and dependencies (pyenv + virtualenv)

To pin the reproduction Python version, install Python with `pyenv` and create an
isolated environment with `virtualenv`. (This leaves the system Python untouched and
prevents reproduction failures caused by version mismatches.)

#### (1) Install pyenv

```bash
# macOS (Homebrew)
$ brew install pyenv

# Ubuntu/Debian — install build dependencies first
$ sudo apt-get update && sudo apt-get install -y \
    make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev
$ curl -fsSL https://pyenv.run | bash
```

Then add pyenv to your shell config and restart the shell (or `source` it):

```bash
# append to ~/.bashrc (or ~/.zshrc)
$ echo 'export PYENV_ROOT="$HOME/.pyenv"'              >> ~/.bashrc
$ echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
$ echo 'eval "$(pyenv init -)"'                        >> ~/.bashrc
$ exec "$SHELL"                                        # restart the shell
$ pyenv --version                                      # verify install
```

#### (2) Install Python and pin the local version

```bash
$ pyenv install 3.12.3          # recommended (compatible with 3.10–3.12)
# pin this directory to always use 3.12.3 via .python-version
$ cd STRATA-final/05_Temporal_STRATA
$ pyenv local 3.12.3
$ python --version              # confirm Python 3.12.3
```

#### (3) Create an isolated environment with virtualenv

```bash
# install virtualenv (into the Python pyenv selected)
$ pip install --upgrade pip virtualenv

# create & activate a project-local virtual environment
$ virtualenv .venv
$ source .venv/bin/activate          # Windows: .venv\Scripts\activate
$ python --version                   # confirm the venv's Python 3.12.3
```

> Tip: with the `pyenv-virtualenv` plugin you can do this in one step:
> ```bash
> $ pyenv virtualenv 3.12.3 strata-final
> $ pyenv local strata-final          # auto-activates on entering the directory
> ```

#### (4) Install dependencies

```bash
$ pip install --upgrade pip
$ pip install -r requirements-mac.txt          # macOS / CPU
$ pip install -r requirements-ubuntu-cuda.txt  # (optional) numba — compiled kernel/throughput
$ pip install python-graphblas                 # (optional) fair GraphBLAS comparison
```

When finished, leave the environment with `deactivate`.

> `python-graphblas` installs the SuiteSparse:GraphBLAS binary. If it is hard to
> install, skip only the GraphBLAS comparison (Step 6); everything else reproduces.

---

## 3. Step 2 — Correctness Self-Check (run this first)

Every STRATA performance number rests on being **bit-identical to an exact temporal
BFS reference**.

```bash
$ python3 -m pytest tests/ -q
```

Checks:
- `test_packed_reachability.py` — STRATA's packed result matches the exact BFS
  reference in **reachable-pair count and full bitset** (strict and closure).
- `test_temporal_strata.py` — time-ordering / same-timestamp (strict vs closure) edge cases.
- `test_original_baselines.py` — baseline-implementation consistency.

> Quick library-only sanity check:
> ```bash
> $ python3 -c "
> from temporal_strata.packed_reachability import temporal_strata_packed, reachable_pairs
> edges=[(1,0,1),(2,1,2),(3,2,3)]   # (t,u,v)
> R=temporal_strata_packed(4, edges, same_time_mode='strict')
> print('reachable pairs =', reachable_pairs(R,4))
> "
> ```

---

## 4. Step 3 — The Standard Input Format

The loader (`temporal_strata.io.load_standard_edges`) and most scripts read a
**standard temporal edge format**: one line, three whitespace-separated integers:

```
src dst timestamp
```

- `src`, `dst`: 0-based integer vertex IDs
- `timestamp`: integer time label (smaller = earlier)
- `#` lines are comments; `# meta: {"n": 4039}` is parsed as JSON metadata
- gzip (`.gz`) loads directly

```python
from temporal_strata.io import load_standard_edges
n, edges, meta = load_standard_edges("graph.edges")   # edges: [(t,u,v), ...]
```

> **Note — two input conventions across scripts.**
> - Scripts using `load_standard_edges` (e.g. `ablation_real_temporal.py`) read the format above.
> - Some ablation scripts (`ablation_closure_detector.py`, `baseline_ablation.py`) take
>   path(s) as command-line arguments. Just follow each step's command.

---

## 5. Step 4 — Real Graph (ego-Facebook n=4039) (§6.5)

Reproduce the §6.5 "real topology" result first (representative and fast).

### 5.1 Build the temporal stream

```bash
$ python3 experiments/make_real_temporal.py
# → /tmp/fb_full_temporal.txt (n=4039, 88,234 edges, discovery order into 200 groups)
```

> **Honesty note:** the *topology* is real (SNAP ego-Facebook) but the *timestamps* are
> a **controlled** discovery order ("real topology + controlled timestamps"). A graph
> with genuine temporal dynamics (original timestamps) is downloaded in Appendix A.

### 5.2 Blocking scaling

```bash
$ python3 experiments/ablation_real_temporal.py
```
**Check (paper §6.5):**

| β | passes | words/pass | pairs |
|---|--------|-----------|-------|
| 4039 | 1 | 258,496 | 1,535,364 |
| 2019 | 3 | 129,248 | 1,535,364 |
| 1009 | 5 | 64,624 | 1,535,364 |
| 504 | 9 | 32,312 | 1,535,364 |

Pairs must be **1,535,364 at every β** (blocking is correctness-invariant).
Output: `results/ablation/real_temporal.csv`. (Absolute seconds vary by hardware.)

---

## 6. Step 5 — Speedup Decomposition (§6.5)

Compare four implementations on the same graph to see where the speedup comes from.

```bash
$ python3 experiments/baseline_ablation.py \
    --edges /tmp/fb_full_temporal.txt --mode strict \
    --out results/ablation/baseline_ablation.csv
```
**Check (real ego-Facebook example):**

```
exact_bfs_ref       pairs=1535364   (slowest, exact BFS reference)
numpy_dense_group   pairs=1535364
strata_packed       pairs=1535364   (word packing)
bitset_ms_no_prune  pairs=1535364   (bitset without pruning — fastest)
ALL METHODS AGREE: True
```
All four must report **1,535,364 pairs** (word packing dominates; pruning is neutral
for strict). Use `--mode closure` for the closure semantics.

---

## 7. Step 6 — Fair GraphBLAS Comparison (§6.5)

```bash
$ pip install python-graphblas        # if not yet installed
$ python3 experiments/ablation_graphblas.py
```

Applies warm-up + best-of-3 to STRATA and GraphBLAS **identically**, with a masked-mxm
closure idiom.

**Check (paper §6.5):**

| n | pairs | ratio (STRATA/GraphBLAS) |
|---|-------|--------------------------|
| 256 | 5,870 | 2.33× |
| 512 | 30,501 | 2.78× |
| 1024 | 93,715 | 2.02× |
| 2048 | 744,796 | 0.93× |
| 4039 | 2,512,141 | 0.66× |

Pairs must match **exactly** at every n; the ratio trends from 2–2.8× (slower) on
small graphs to parity at n=2048 to 0.66× (faster) at n=4039.
Output: `results/ablation/graphblas_static.csv`.

---

## 8. Step 7 — Remaining Experiments

Mind each script's input convention.

### 8.1 Deletion repair cost (§6.4)
```bash
$ python3 experiments/ablation_deletion.py        # uses /tmp/fb_full_temporal.txt by default
```
**Check:** ≈0.008s per 1k suffix edges (constant), scaling with suffix size.

### 8.2 Throughput 1/W curve (§6.6) — needs numba
```bash
$ python3 experiments/ablation_throughput_curve.py
```
**Check:** 27/5.3/1.5/0.43 M edges/s at n=2k/4k/8k/16k (~1/W).

### 8.3 Non-triviality vs naive baseline (§2.2/§6.5)
```bash
$ python3 experiments/ablation_naive_temporal.py
```
**Check:** 16–28× (bit-identical).

### 8.4 Blocking scaling (synthetic, §6.5)
```bash
$ python3 experiments/ablation_blocking.py
```
**Check:** smaller β ⇒ more passes / fewer words-per-pass; identical pair counts.

### 8.5 Closure detector (§6.5) — pass input graph(s) as arguments
```bash
# pass a stream with clear group structure (u v t format)
$ python3 experiments/ablation_closure_detector.py <graph1.txt> [graph2.txt ...]
```
**Check:** change-detected closure is 23–290× faster than unconditional sweeps
(bit-identical). With no argument the result is empty, so always pass graph path(s).

### 8.6 Case study — prospective hit prioritization (§6.7)
```bash
$ python3 experiments/hit_prioritization.py --synthetic --out-dir results/hitsel
```
**Check:** EF strict 6.1 > closure 4.8 ≫ popularity 0.9; 100% of static's top hits are leakage.

---

## 9. Reproduction Checklist

1. **Correctness:** `pytest tests/` all pass → STRATA is bit-identical to exact BFS.
2. **Pair-count match:** each experiment's "pairs" equals the tables above (hardware-independent).
3. **Trend match:** the *direction and magnitude band* of speedups/ratios agree.
4. **CSV written:** the corresponding `*.csv` appears under `results/ablation/`.

> The paper claims **same-machine relative comparison** and **bit-for-bit pair-count
> agreement**, so judge reproduction on those two (absolute wall-clock varies by hardware).

---

# Appendix A. Download & Run Large Public Graphs Locally

Download **large public graphs with genuine temporal dynamics (original timestamps)**
and run them through STRATA (unlike Step 5's controlled timestamps).

## A.1 Recommended datasets

| Dataset | Size (approx) | Source | Format |
|---------|---------------|--------|--------|
| **CollegeMsg** | 1,899 nodes / 59,835 edges | SNAP | `src dst unixts` |
| **email-Eu-core-temporal** | 986 / 332,334 | SNAP | `src dst time` |
| **sx-stackoverflow** (large) | 2.6M / 63M | SNAP | `src dst time` |
| **tgbl-wiki-v2** | ~10⁵ / ~10⁶ | TGB2 | py-tgb API |

## A.2 Path 1 — SNAP graphs (simplest, recommended)

SNAP temporal graphs use `src dst timestamp`, nearly identical to STRATA's standard format.

### (1) Download
```bash
$ cd STRATA-final/05_Temporal_STRATA
$ mkdir -p data/real

$ curl -L -o data/real/CollegeMsg.txt.gz \
    https://snap.stanford.edu/data/CollegeMsg.txt.gz
$ gunzip -f data/real/CollegeMsg.txt.gz
```
> If blocked, download from https://snap.stanford.edu/data/ in a browser, place under `data/real/`, and `gunzip`.

### (2) Normalize to the standard format
SNAP CollegeMsg lines are `SRC DST UNIXTS`. Densify vertex IDs from 0 and sort timestamps:

```bash
$ cat > to_standard.py << 'EOF'
import sys
src_path, out_path = sys.argv[1], sys.argv[2]
edges = []; nodes = {}
def nid(x):
    if x not in nodes: nodes[x] = len(nodes)
    return nodes[x]
for line in open(src_path):
    line = line.strip()
    if not line or line.startswith("#"): continue
    a = line.replace(",", " ").split()
    u, v, t = int(a[0]), int(a[1]), int(float(a[2]))
    edges.append((t, nid(u), nid(v)))
edges.sort()
times = sorted(set(t for t,_,_ in edges))
tmap = {t:i for i,t in enumerate(times)}
with open(out_path, "w") as f:
    f.write(f'# meta: {{"n": {len(nodes)}}}\n')
    for t,u,v in edges:
        f.write(f"{u} {v} {tmap[t]}\n")
print(f"n={len(nodes)} edges={len(edges)} groups={len(times)} -> {out_path}")
EOF

$ python3 to_standard.py data/real/CollegeMsg.txt data/real/collegemsg.edges
```

### (3) Run with STRATA
Either pass the path to the real-graph script, or call the library directly:

```bash
# blocking-scaling script (path as first argument)
$ python3 experiments/ablation_real_temporal.py data/real/collegemsg.edges

# or call the library directly
$ python3 << 'EOF'
import time, math
from temporal_strata.io import load_standard_edges
from temporal_strata.packed_reachability import (
    temporal_strata_packed, reachable_pairs, words_for)

n, edges, meta = load_standard_edges("data/real/collegemsg.edges")
print(f"loaded n={n} edges={len(edges)}")

t0 = time.perf_counter()
R = temporal_strata_packed(n, edges, same_time_mode="strict")
dt = time.perf_counter() - t0
print(f"strict: pairs={reachable_pairs(R,n)}  time={dt:.3f}s  words/row={words_for(n)}")

beta = max(64, n // 4)
t0 = time.perf_counter()
Rb = temporal_strata_packed(n, edges, same_time_mode="strict", block_bits=beta)
dt = time.perf_counter() - t0
assert reachable_pairs(Rb, n) == reachable_pairs(R, n)   # blocking is correctness-invariant
print(f"blocked(beta={beta}, passes={math.ceil(n/beta)}): time={dt:.3f}s  (pairs match: OK)")

Rc = temporal_strata_packed(n, edges, same_time_mode="closure")
print(f"closure: pairs={reachable_pairs(Rc,n)}")
EOF
```

This confirms, on a **real-timestamp public graph**, STRATA's correctness (blocking
invariance) and performance (single pass vs blocks).

## A.3 Path 2 — TGB2 (tgbl-wiki-v2)

### (1) Install py-tgb and cache the data
```bash
$ pip install py-tgb
$ python3 -c "
from tgb.linkproppred.dataset import LinkPropPredDataset
LinkPropPredDataset(name='tgbl-wiki-v2', root='data/raw/tgb')
"   # one-time download (cached)
```

### (2) Export to the standard format (bundled script)
```bash
$ python3 experiments/export_tgb_wiki.py \
    --root data/raw/tgb \
    --out  data/standard/tgbl-wiki.edges
# (optional) cap size: --max-nodes 20000 --max-edges 1000000
```

### (3) Run (use blocking when vertices are many)
```bash
$ python3 << 'EOF'
import time, math
from temporal_strata.io import load_standard_edges
from temporal_strata.packed_reachability import temporal_strata_packed, reachable_pairs, packed_state_bytes

n, edges, meta = load_standard_edges("data/standard/tgbl-wiki.edges")
print(f"n={n} edges={len(edges)}  full-state≈{packed_state_bytes(n)/1e6:.1f} MB")

beta = 2048
t0 = time.perf_counter()
R = temporal_strata_packed(n, edges, same_time_mode="strict", block_bits=beta)
dt = time.perf_counter() - t0
print(f"blocked(beta={beta}, passes={math.ceil(n/beta)}): pairs={reachable_pairs(R,n)} time={dt:.2f}s")
EOF
```

> Memory estimate: the full state is `n·⌈n/64⌉·8` bytes (`packed_state_bytes(n)`).
> E.g. n=100k ⇒ ~1.25GB. Use `block_bits=β` to bound per-pass footprint to `n·⌈β/64⌉·8`.

## A.4 Tips for large runs

- **Memory first:** check `packed_state_bytes(n)`; if it exceeds budget, set `block_bits=β`.
  Per-pass footprint ∝ β; passes = ⌈n/β⌉.
- **Correctness check:** always compare `reachable_pairs` for small vs large β.
- **Compiled speedup:** with `numba` installed, the kernel in `experiments/numba_kernel.py`
  accelerates big streams.
- **Fair timing:** report **best-of-three after warm-up** (the paper's global discipline).
- **Windowing:** for rolling windows, slice the edge list by time interval before feeding it in.

---

## Troubleshooting

- Correctness tests fail: pin Python/numpy to `requirements` and reinstall.
- GraphBLAS install fails: skip only Step 6; everything else reproduces.
- Out-of-memory on large graphs: lower `block_bits` (e.g. 1024 → 512).
- Empty result CSV: `ablation_closure_detector.py` takes input graph path(s) as **arguments**.


## 9. Multicore parallel verification and strong scaling (added in the pre-v1 history)

The column-partitioned thread-parallel core (`temporal_strata/parallel.py`) is proven bit-identical to the sequential core for every thread count, chunk partition, and interleaving (paper Theorem 3), and the tests verify it.

```bash
# Full suite including parallel tests (331 tests since STRATA-v36; optional deps skip with reason)
python3 -m pytest tests/ -q

# Strong-scaling harness: kernel vs end-to-end, vs parallel per-source BFS, emits the LaTeX table
NUMBA_NUM_THREADS=8 python3 experiments/parallel_scaling.py
```

Harness rules: (1) every parallel configuration is bit-checked against the sequential core before any timing, aborting on mismatch; (2) on a single-core host it refuses to emit speedups and produces the PENDING table; (3) results go to `results/real/parallel_scaling.json` and `table_parallel_scaling.tex` with the machine string in the caption. Copy the emitted table over `STRATA-latex/STRATA-paper-EN/table_parallel_scaling.tex`, resolve the two `%% TODO(M5)` markers in main.tex (since STRATA-v2 the table is Table S1 of the supplementary; see Section 10), rebuild, and confirm <= 14 pages.

## 10. Apple M5 Max measurement session (first run on STRATA-v4; re-measurements in 10.1 (v5), 10.2 (v6), 10.3 (v7); paper numbers in 10.4 (v8); v9 re-measurement in 10.5)

Machine: Apple M5 Max, 6 super + 12 performance cores, 128 GB. Close other
applications, keep the machine on AC power, and run all steps in one session
from `STRATA-v7/STRATA-code/05_Temporal_STRATA`. Every script verifies
bit-identity before timing and records the machine (incl. `hw.perflevel*` core
counts and the detected cache-line size) in its JSON. macOS does not guarantee
which cores a thread runs on; do not pin or interpret placement.


**One-command option (backlog R15).** `python3 tools/run_m5_session.py` (or
`tools/run_m5_session.sh`) runs exactly the commands below in this order, writes
each command's log to `results/real/logs/`, keeps older result files under a
time-stamped name instead of overwriting them, and stops before any timing if
a step-1 check fails or the plan check shows idle workers. Use `--dry-run` to
print the plan and `--from-step N` / `--only-step N` to resume. Create and
activate the virtual environment first (the first two lines of Step 1).

**Step 1 — environment, correctness, and plan check (expect 331 passed)**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt -r requirements-optional.txt
python3 -m pytest tests/ -q
NUMBA_NUM_THREADS=18 python3 -m pytest tests/test_parallel.py tests/test_parallel_layout.py -q
python3 tools/check_numbers.py
NUMBA_NUM_THREADS=18 python3 experiments/parallel_scaling.py --plan-only \
  --threads 1,2,4,6,8,12,16,18 --layouts rowmajor,tiled,auto --schedules static,dynamic \
  --chunk-policy threads --large 50000:1000000:400 --large 100000:2000000:400 \
  --block-bits 65536 --plan-out results/real/plan_check.json
```
Stop if the plan output shows `<-- workers idle` for any real dataset at
p <= W (R8); send the plan output instead of timing.

**Step 2 — strong scaling: layout × schedule (backlog B, R4, R6, R8)**
```bash
NUMBA_NUM_THREADS=18 python3 experiments/parallel_scaling.py \
  --threads 1,2,4,6,8,12,16,18 --repeats 5 \
  --layouts rowmajor,tiled,auto --schedules static,dynamic --oversub 4 \
  --chunk-policy threads --table-config tiled:static \
  --out results/real/parallel_scaling.json \
  --tex-out results/real/table_parallel_scaling.tex
```

**Step 3 — large graphs and the sequential fraction (R12)**
```bash
NUMBA_NUM_THREADS=18 python3 experiments/parallel_scaling.py \
  --skip-real --skip-synthetic \
  --large 50000:1000000:400 --large 100000:2000000:400 --block-bits 65536 \
  --reference compiled --baseline-max-work 1e10 \
  --threads 1,2,4,6,8,12,16,18 --repeats 3 \
  --layouts rowmajor,tiled,auto --schedules static,dynamic --chunk-policy threads \
  --out results/real/parallel_scaling_large.json \
  --tex-out results/real/table_parallel_scaling_large.tex
```

**Step 4 — memory-bandwidth ceiling (R5)**
```bash
NUMBA_NUM_THREADS=18 python3 experiments/bandwidth_roofline.py \
  --threads 1,2,4,6,8,12,16,18 --array-mib 2048 --repeats 5 \
  --layouts rowmajor,tiled --chunk-policy threads --synthetic \
  --large 100000:2000000:400 \
  --out results/real/bandwidth_roofline.json
```

**Step 5 — row-major vs tiled crossover for `layout="auto"` (R6)**
```bash
NUMBA_NUM_THREADS=18 python3 experiments/layout_crossover.py \
  --threads 18 --points 9 --repeats 5 --out results/real/layout_crossover_p18.json
NUMBA_NUM_THREADS=18 python3 experiments/layout_crossover.py \
  --threads 6 --points 9 --repeats 5 --out results/real/layout_crossover_p6.json
```

**Step 6 — compiled-vs-compiled on real data and an external baseline (R9, R13)**
```bash
python3 experiments/compiled_real.py --repeats 5 --out results/real/compiled_real.json
pip install raphtory
python3 experiments/external_baselines.py --adapter raphtory --check-only --diagnose \
  --out results/real/external_check.json
# only if the check reports a matching semantics:
python3 experiments/external_baselines.py --adapter raphtory \
  --datasets CollegeMsg,BitcoinOTC,BitcoinOTCpos --sources all --repeats 3 \
  --out results/real/external_baselines.json
```
If the Raphtory adapter reports `ADAPTER UNAVAILABLE` or a semantics mismatch,
keep the console output; do not edit the adapter during the session.

**Step 7 — MPS probe and preview figures**
```bash
python3 experiments/backend_bench.py --device mps
python3 experiments/plot_parallel.py scaling \
  --json results/real/parallel_scaling.json --out results/real/fig_parallel_scaling.pdf
python3 experiments/plot_parallel.py scaling \
  --json results/real/parallel_scaling_large.json --out results/real/fig_parallel_scaling_large.pdf
python3 experiments/plot_parallel.py roofline \
  --json results/real/bandwidth_roofline.json --out results/real/fig_roofline.pdf
```

Hand back the whole `results/real/` folder (plan_check, parallel_scaling,
parallel_scaling_large, bandwidth_roofline, layout_crossover_p18 / _p6,
compiled_real, external_check / external_baselines, backend_bench JSON files,
the emitted tables, and the preview PDFs) plus the console logs. These files
are the only source of performance numbers for the next manuscript version; nothing is copied into the
paper by hand.

### 10.1 STRATA-v5 re-measurement (backlog R17, R13; completed 2026-11-15)

The first session (STRATA-v4, 2026-11-15) is complete. STRATA-v5 adds the
array-native preprocessing path (R17) and makes the tiled layout the default,
so steps 1, 2, 3, 6, and 7 are re-run; steps 4 and 5 (kernel bandwidth and
layout crossover) do not depend on preprocessing and are not repeated. Earlier
result files are kept automatically under time-stamped names.

**A. Main environment (the same Python as the first session)**
```bash
cd ~/Research/STRATA/STRATA-v5/STRATA-code/05_Temporal_STRATA
python3 -m venv .venv && source .venv/bin/activate
python3 tools/run_m5_session.py --dry-run --steps 1,2,3,6,7
python3 tools/run_m5_session.py --steps 1,2,3,6,7
deactivate
```

**B. External baseline in a Python 3.13 environment (R13)**
Raphtory had no wheel for Python 3.15 in the first session. Run only the
external-baseline part of step 6 from a Python 3.13 virtual environment:
```bash
cd ~/Research/STRATA/STRATA-v5/STRATA-code/05_Temporal_STRATA
python3.13 -m venv .venv313 && source .venv313/bin/activate
pip install -r requirements-mac.txt
python3 tools/run_m5_session.py --only-external
deactivate
```
If `python3.13` is missing, install it first (e.g., `brew install python@3.13`
or the python.org installer). If the adapter still reports
`ADAPTER UNAVAILABLE` or a semantics mismatch, keep the log; do not edit the
adapter during the session.

### 10.2 STRATA-v6 re-measurement (backlog R19, R13 addition; completed 2026-11-15)

STRATA-v6 replaces the global sort of the touched-row plan with a linear pass
(R19; preprocessing is now timed as warm-up + best-of-R, and the first, cold
call that includes JIT compilation is recorded separately) and adds the
external-baseline diagnosis (`--diagnose`, R13 addition). Run the same two
parts as in 10.1, from the v6 folder:

**A. Main environment**
```bash
cd ~/Research/STRATA/STRATA-v6/STRATA-code/05_Temporal_STRATA
python3 -m venv .venv && source .venv/bin/activate
python3 tools/run_m5_session.py --steps 1,2,3,6,7
deactivate
```

**B. Python 3.13 environment (external baseline with diagnosis)**
```bash
cd ~/Research/STRATA/STRATA-v6/STRATA-code/05_Temporal_STRATA
python3.13 -m venv .venv313 && source .venv313/bin/activate
pip install -r requirements-mac.txt
python3 tools/run_m5_session.py --only-external
deactivate
```
The diagnosis lands in `results/real/external_check.json` (`toy_diagnosis`,
and per dataset `diagnosis` / `raw_sample`). The timed external run still
happens only if the semantics check matches.

### 10.3 STRATA-v7: external baseline only (backlog R13, adapter fix)

The v6 diagnosis showed that the Raphtory adapter counted every vertex as
reached (a dict with an empty `reachable_nodes` list is truthy). STRATA-v7
fixes the adapter and adds a guard that flags "every source reaches every
vertex" results. Nothing else changed, so only part B is re-run:

```bash
cd ~/Research/STRATA/STRATA-v7/STRATA-code/05_Temporal_STRATA
python3.13 -m venv .venv313 && source .venv313/bin/activate
pip install -r requirements-mac.txt
python3 tools/run_m5_session.py --only-external
deactivate
```
If the semantics check now matches strict or closure, the same command also
runs the timed comparison (`results/real/external_baselines.json`).

Result (2026-11-15): Raphtory 0.17.0 follows closure semantics on all five toy
streams and agrees with STRATA's closure relation on 189 of 192 sampled
sources; on the other 3 it misses one reachable vertex each. By decision (a)
of backlog R13 the tool is **not timed**.

### 10.4 STRATA-v8: paper numbers, tables, and figures from the M5 files

STRATA-v8 measures nothing new. Every M5 number in the paper is a LaTeX macro
derived from the result files; the supplementary tables (milliseconds) and the
three multicore figures are drawn from the same files. After any future
session run, regenerate them (no timing involved):

```bash
cd ~/Research/STRATA/STRATA-v8/STRATA-code/05_Temporal_STRATA
python3 experiments/external_witness.py          # witness paths for external disagreements
python3 experiments/m5_numbers.py                # m5_numbers.tex, Table 8, Tables S1-S2
P=../../STRATA-latex/STRATA-paper-EN/figures
python3 experiments/plot_parallel.py paper --json results/real/parallel_scaling.json \
    --large results/real/parallel_scaling_large.json --out $P/fig_scaling.pdf
python3 experiments/plot_parallel.py breakdown --json results/real/parallel_scaling.json \
    --large results/real/parallel_scaling_large.json --out $P/fig_breakdown.pdf
python3 experiments/plot_parallel.py roofline --json results/real/bandwidth_roofline.json \
    --out $P/fig_roofline.pdf
python3 tools/check_numbers.py                   # fails if the paper's copy is stale
python3 tools/make_submission_bundle.py --scan-only   # anonymity scan of the bundle
```
`check_numbers.py` now also checks the abstract (150-190 words, no math,
numbers equal to the macros). Because it compares the paper with the current
result files, run `m5_numbers.py` right after a new measurement, then rebuild
the paper and re-check the page count (12 pages).

### 10.5 STRATA-v9: re-measurement with samples, parallel preprocessing, and new steps

STRATA-v9 prepares (but does not run) the measurements requested by the v8
mock review (backlog R20, R25, R26, R27, R29, R31):

- every timed configuration now stores **all repeat samples** (R26); steps 2
  and 3 also time the **parallel preprocessing** at every thread count after
  checking it array for array against the sequential plan (R20,
  `--prep-parallel compare`);
- step 8 re-runs the **single-thread** experiments on the M5 Max (R25;
  `--no-tex` keeps the scripts from writing paper tables directly, and
  `tools/machine_stamp.py` records the machine and file hashes);
- step 9 downloads two **public temporal graphs with n >= 10^5** (SNAP
  sx-askubuntu, n = 159,316, 964,437 edges; sx-superuser, n = 194,085,
  1,443,339 edges) and runs the scaling harness with the **sampled reference**
  (sequential compiled core, its columns for 64 sampled sources checked against
  an independent per-source BFS); a failed download is recorded and the rest of
  the step is skipped (no substitute data);
- step 10 measures **closure** on hour / day / week-bucketed real timestamps
  (sweep counts, Lemma 4 check, strict vs closure timing) (R29);
- step 11 installs `python-graphblas` and times the **static** closure as a
  reference point only (R31); if the package is unavailable the step is skipped.

Run in the main environment (about 1-2 hours; steps 4-6 are repeated so that
every timing in the paper has samples):

```bash
cd ~/Research/STRATA/STRATA-v9/STRATA-code/05_Temporal_STRATA
source .venv/bin/activate
python3 tools/run_m5_session.py --dry-run --steps 1,2,3,4,5,6,8,9,10,11
python3 tools/run_m5_session.py --steps 1,2,3,4,5,6,8,9,10,11
```

The commands the runner executes for the changed and new steps:

```bash
# step 2: strong scaling: layout x schedule
NUMBA_NUM_THREADS=18 python3 experiments/parallel_scaling.py --threads 1,2,4,6,8,12,16,18 --repeats 5 --layouts rowmajor,tiled,auto --schedules static,dynamic --oversub 4 --chunk-policy threads --table-config tiled:static --prep-parallel compare --out results/real/parallel_scaling.json --tex-out results/real/table_parallel_scaling.tex
# step 3: large graphs and the sequential fraction
NUMBA_NUM_THREADS=18 python3 experiments/parallel_scaling.py --skip-real --skip-synthetic --large 50000:1000000:400 --large 100000:2000000:400 --block-bits 65536 --reference compiled --baseline-max-work 1e10 --threads 1,2,4,6,8,12,16,18 --repeats 3 --layouts rowmajor,tiled,auto --schedules static,dynamic --chunk-policy threads --prep-parallel compare --out results/real/parallel_scaling_large.json --tex-out results/real/table_parallel_scaling_large.tex
# step 8: single-thread experiments on the evaluation machine (R25)
NUMBA_NUM_THREADS=1 python3 experiments/perf_table.py --no-tex
NUMBA_NUM_THREADS=1 python3 experiments/compiled_baselines.py --no-tex
NUMBA_NUM_THREADS=1 python3 experiments/real_temporal_benchmark.py --reps 5
NUMBA_NUM_THREADS=1 python3 experiments/ablation_snapshot_necessity.py
NUMBA_NUM_THREADS=1 python3 experiments/ablation_strict_prune.py
NUMBA_NUM_THREADS=1 python3 experiments/index_crossover.py
NUMBA_NUM_THREADS=1 python3 experiments/real_dynamic_and_blocking.py
python3 tools/machine_stamp.py --label single_thread --out results/real/single_thread_machine.json results/perf/perf_table.json results/perf/compiled_baselines.json results/real/real_temporal_results.json results/real/snapshot_necessity.json results/real/ablation_strict_prune.json results/real/index_crossover.json results/real/real_dynamic_blocking.json
# step 9: large public temporal graphs (R27)
python3 experiments/fetch_real_temporal.py --names sx-askubuntu,sx-superuser --log results/real/fetch_large.json
NUMBA_NUM_THREADS=18 python3 experiments/parallel_scaling.py --skip-real --skip-synthetic --extra-real sx-askubuntu --extra-real sx-superuser --reference sampled --baseline-max-work 1e10 --threads 1,2,4,6,8,12,16,18 --repeats 3 --layouts tiled --schedules static,dynamic --chunk-policy threads --prep-parallel compare --out results/real/parallel_scaling_realbig.json --tex-out results/real/table_parallel_scaling_realbig.tex
# step 10: closure semantics on bucketed real timestamps (R29)
NUMBA_NUM_THREADS=18 python3 experiments/closure_grouped.py --threads 1,2,4,6,8,12,16,18 --repeats 5 --out results/real/closure_grouped.json
# step 11: multithreaded GraphBLAS reference point, static closure (R31)
pip install python-graphblas
NUMBA_NUM_THREADS=18 python3 experiments/graphblas_reference.py --threads 1,2,4,6,8,12,16,18 --repeats 5 --out results/real/graphblas_reference.json
```

After the session, send "재실측 끝". The paper numbers are then regenerated
with `experiments/m5_numbers.py` (Section 10.4) in the next batch; nothing in
the paper changes before that.

### 10.6 STRATA-v10: re-run of steps 1, 8, 9, and 11 (backlog R32 / R33)

The v9 session (2026-11-16) completed steps 1-6 and 8-10; step 11 stopped on a
harness defect in `experiments/graphblas_reference.py` (fixed in v10, R32), and
step 8 ran before python-graphblas was installed, so its GraphBLAS column is
empty. v10 also changes the edge sort for wide timestamp ranges (R33), which
affects step 9. Run, in the STRATA-v10 folder:

```bash
cd STRATA-v10/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 1,8,9,11
```

- The copied `.venv313/bin/activate` still points at the STRATA-v9 folder, so
  call the v10 interpreter by path as above; the runner uses it for every
  `python3` / `pip` command.
- Step 1 now installs python-graphblas from `requirements-optional.txt` and runs
  the 227-test suite, including the real python-graphblas path.
- Step 8 re-measures the single-thread suite with the GraphBLAS column.
- Step 9 re-measures sx-askubuntu / sx-superuser with the rank-compressed sort.
- Step 11 measures the GraphBLAS static-closure reference point.

About 15 minutes. Then send "재실측 끝"; the paper numbers are regenerated with
`experiments/m5_numbers.py` and the sentences that depend on steps 9 and 11 are
written from the new files.

### 10.7 STRATA-v11: thread records, single-thread GraphBLAS, adaptive sort (backlog R35 / R36 / R37)

v11 makes every single-thread harness store the actual thread count of each
runtime (Numba, GraphBLAS, OpenMP pools; R37), pins GraphBLAS to one thread in
`compiled_baselines.py` (R36), and sorts nearly time-ordered input with an
adaptive stable sort (R35). `experiments/m5_numbers.py` now refuses
single-thread result files without a thread record, so the test suite
(step 1) passes only after step 8 has written new files. Run, in this order:

```bash
cd STRATA-v11/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 8,11
.venv313/bin/python3 tools/run_m5_session.py --steps 1,9
```

- Step 8: single-thread suite with thread records and the one-thread GraphBLAS column.
- Step 11: GraphBLAS reference point with the pool size read back per row.
- Step 1: environment and the 235-test suite (now including the thread-record check).
- Step 9: sx-askubuntu / sx-superuser with the adaptive sort (the record shows `sort_path`).

About 20 minutes in total. Then send "재실측 끝".

Correction (2026-11-16, found in the v11 run): step 1 also checks that the
paper's generated files equal a fresh derivation, so after step 8 writes new
measurements it fails until those files are regenerated. Either run
`.venv313/bin/python3 experiments/m5_numbers.py` between the two commands, or,
as done in v11, run `--steps 1,9` again after the next batch has regenerated
them. The thread-record check itself passed on the first try.

### 10.8 STRATA-v12: synthetic streams with the v11 preprocessing path (backlog R40)

The synthetic preprocessing numbers in v11 were measured before the adaptive
sort (R35) existed. Re-run only the two synthetic scaling steps; do not run
step 1 in the same command, because the paper's generated files are rebuilt
from the new files afterwards (see the correction in 10.7):

```bash
cd STRATA-v12/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 2,3
```

About 12 minutes. Then send "재실측 끝". The kernel is re-measured too, so the
abstract numbers may move in the last digit; `tools/check_numbers.py` checks them.
All six release gates, including `tools/check_figures.py` (R38), run with
`python3 tools/run_gates.py`.

### 10.9 STRATA-v13: Temporal Ligra baseline and automatic regeneration (backlog R39 / R42)

Two new steps. Step 12 builds Temporal Ligra (MIT, github.com/jshun/ligra,
branch `temporal`, pinned commit `75eb1a29`) on this machine, checks that its
earliest-arrival traversal computes STRATA's relation (duration 1 = strict,
duration 0 = closure; toy streams + 64 sampled sources), and only then times
every source at 1 and 18 threads, verifying every reachable set:

```bash
python3 experiments/temporal_ligra.py fetch --dest external/temporal_ligra
python3 experiments/temporal_ligra.py build --src external/temporal_ligra
python3 experiments/temporal_ligra.py check --src external/temporal_ligra --out results/real/temporal_ligra_check.json
python3 experiments/temporal_ligra.py time --src external/temporal_ligra --threads 1,18 --out results/real/temporal_ligra_time.json
```

The clone stays in `external/` (not packaged). Apple clang needs Homebrew
`libomp` for OpenMP (`brew install libomp`); without it the adapter builds the
sequential variant and records that, so do install it before the run if possible.

Step 13 (R42) runs automatically after any measurement step (`--no-regen`
turns it off) and rebuilds every generated paper file from the new results:

```bash
python3 experiments/m5_numbers.py
python3 experiments/make_paper_figures.py
python3 experiments/plot_parallel.py paper --json results/real/parallel_scaling.json --large results/real/parallel_scaling_large.json --out ../../STRATA-latex/STRATA-paper-EN/figures/fig_scaling.pdf
python3 experiments/plot_parallel.py breakdown --json results/real/parallel_scaling.json --large results/real/parallel_scaling_large.json --out ../../STRATA-latex/STRATA-paper-EN/figures/fig_breakdown.pdf
python3 experiments/plot_parallel.py roofline --json results/real/bandwidth_roofline.json --out ../../STRATA-latex/STRATA-paper-EN/figures/fig_roofline.pdf
python3 tools/check_numbers.py
```

The session record lists every regenerated file with its sha256. Run for v13:

```bash
cd STRATA-v13/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 12
```

(step 13 is added automatically). About 15-30 minutes, mostly the per-source
Temporal Ligra runs. Then send "재실측 끝".

### 10.10 STRATA-v14: source-batch parallel baseline (backlog R43)

Temporal Ligra parallelizes inside one traversal, which slowed it down on the
observed-timestamp graphs (v13). Step 12 now also runs the one-thread binary
over p batches of sources in p concurrent processes, the most favorable
parallelization for a per-source tool, and records the wall time (process
start-up and graph reading included) and the critical-path compute time (the
largest per-batch sum of the tool's own compute times), verifying every source:

```bash
python3 experiments/temporal_ligra.py time-batch --src external/temporal_ligra --threads 1,6,12,18 --out results/real/temporal_ligra_batch.json
```

Run (step 13 is added automatically):

```bash
cd STRATA-v14/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 12
```

About 20-40 minutes. Then send "재실측 끝".

### 10.11 STRATA-v16: reference check before submission (backlog R46)

`tools/check_bib.py` compares every cited refs.bib entry with its Crossref
record (by DOI; entries without a DOI by title, then DBLP, as candidates):
author family names, given names, order and count, title, year, volume, and
pages. Neither the cloud workspace nor the Mac VM reaches Crossref / DBLP, so
run it once on macOS directly (needs network, no measurement, under a minute):

```bash
cd STRATA-v16/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/check_bib.py --fetch
```

It writes `reports/bib_check.md` (match / mismatch / unverified per entry) and
`reports/bib_cache.json` (the fetched records; later runs compare offline).
Exit 1 means at least one mismatch. Then send "bib 검사 끝".

### 10.12 STRATA-v17: reference check again (backlog R46 / R47)

v17 fixes how the tool reads a bad answer (DBLP's bot page no longer stops the
run), strips name particles, and picks title candidates of the same work and
year; refs.bib now has DOIs for 15 more entries. Run the same check again on
macOS (cached DOI answers are reused; title answers made by the v16 rule are
fetched again):

```bash
cd STRATA-v17/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/check_bib.py --fetch
```

If the report still lists entries as "fetch incomplete", run the same command
again a few minutes later. Then send "bib 검사 끝".

The same check also works from the v18 folder (`cd STRATA-v18/STRATA-code/05_Temporal_STRATA`); v18 changes nothing in refs.bib or the tool.

### 10.13 STRATA-v19: reference check after the corrections (backlog R50)

v19 corrects four references (tang2010temporalreachability, restless2020,
steer2024raphtory, and wen2020spanreach, formerly wen2020topkreach), adds three
DOIs, and changes the tool: Crossref subtitles are part of the title, OpenAlex is
searched before DBLP, a differing title-search record is reported as "check"
instead of "mismatch", and known record errors can be accepted with a reason
(`reports/bib_manual.json`). Every cached answer is fetched again once:

```bash
cd STRATA-v19/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/check_bib.py --fetch
```

Then send "bib 검사 끝".

### 10.14 STRATA-v21: stronger baselines and a real stream above 10^7 edges (backlog R54)

Three new session steps answer the v20 mock review (`reports/mock_review_v20.md`):
step 14 times a word-packed temporal MS-BFS (64 sources per word, empty source
words skipped, batches over threads) against STRATA on the same prepared stream,
after checking that both give the identical relation; step 15 fetches the two
public temporal-index tools (tgindex, TGLib) and records their license, README,
and build files, so that their adapter can be written in the next batch; step 16
downloads SNAP sx-stackoverflow (n = 2,601,977, m = 63,497,050) and runs the first
8 source blocks of width 4096 at 1 and 18 threads, verifying 64 sampled sources
against an independent per-source BFS. The relation has 636 blocks; the record
gives the measured blocks and, labeled as such, their mean times 636.

```bash
NUMBA_NUM_THREADS=18 python3 experiments/temporal_msbfs.py bench --datasets CollegeMsg,BitcoinOTC,BitcoinOTCpos --synthetic --threads 1,18 --repeats 5 --out results/real/temporal_msbfs.json
python3 experiments/temporal_index_baseline.py probe --dest external --out results/real/index_baseline_probe.json
python3 experiments/fetch_real_temporal.py --names sx-stackoverflow --log results/real/fetch_stackoverflow.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 4096 --blocks 8 --threads 1,18 --schedule dynamic --state rows --verify-sources 64 --out results/real/large_real_blocked.json
```

Run (step 13 is added automatically and runs last):

```bash
cd STRATA-v21/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 14,15,16
```

The download and conversion handle about 63 million lines (no time measured yet);
the harness prints the time of every block as it runs. Steps 15 and 16 need network
access (gitlab.com, snap.stanford.edu); a failed download is recorded and the rest
of that step is skipped. Then send "재실측 끝".

### 10.15 STRATA-v22: wider blocks on sx-stackoverflow and the TGLib baseline (backlog R55 / R56 / R57)

The v21 run verified sx-stackoverflow but, with blocks of 4096 sources (64 words),
18 threads were 1.84x slower than one. Step 17 sweeps the block width (4096 and
65536 sources, i.e. 64 and 1,024 words; about 21 GB resident at 65536) over 1, 6,
12, and 18 threads with the static schedule, then runs every block (40 at 65536)
at the thread count that was fastest in the sweep, so the total time is measured
rather than extrapolated. Preprocessing now uses a stable time sort plus a sort
inside multi-edge timestamp groups when no packed key fits (R56); the record keeps
`prep_sort` and the phase times. Step 18 installs TGLib (MIT, PyPI
`temporalgraphlib==1.7.1`), checks that its earliest-arrival sets equal STRATA's
(strict: unit edge durations), and only then times it from every source in one
and in 18 processes. tgindex is not used until its authors allow it (draft
request: `reports/tgindex_permission_request.md`).

```bash
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 4096 --blocks 2 --threads 1,6,12,18 --schedule static --state rows --verify-sources 64 --out results/real/large_real_b4096.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 65536 --blocks 2 --threads 1,6,12,18 --schedule static --state rows --verify-sources 64 --out results/real/large_real_b65536.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 65536 --blocks all --threads best:results/real/large_real_b65536.json --schedule static --state rows --verify-sources 64 --out results/real/large_real_full.json
pip install temporalgraphlib==1.7.1
python3 experiments/tglib_baseline.py check --out results/real/tglib_check.json
NUMBA_NUM_THREADS=1 python3 experiments/tglib_baseline.py time --check results/real/tglib_check.json --procs 1,18 --out results/real/tglib_time.json
```

Run (step 13 is added automatically and runs last):

```bash
cd STRATA-v22/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 17,18
```

Session `20261117-155538` ran both steps: the full step-17 run took 708.9 s of
kernel time over all 40 blocks at 6 threads; step 18 stopped at the install
because PyPI has no macOS arm64 wheel (see 10.16).

### 10.16 STRATA-v23: TGLib from source and the six-thread optimum (backlog R58 / R60)

PyPI publishes temporalgraphlib only as Linux x86_64 wheels, so step 18 now builds
the pinned MIT source (commit e56c3dd, recursive clone for PyBind11): first
`pip install` of the checkout, else the README's CMake Release build linked
against this interpreter, registered with a `.pth` file. The record
`results/real/tglib_build.json` keeps the route, the commit, the tool versions,
and the log tail; a failed build is recorded and the check / time commands are
skipped. The build needs `git`, `cmake`, and a C++ compiler (Xcode command-line
tools).

In the v22 sweep, 6 threads were fastest for both block widths, and the M5 Max
reports 6 "Super" and 12 "Performance" cores. Step 19 separates two explanations
on one block of each width: it times static and dynamic schedules at 1, 6, 12, and
18 threads (dynamic lets faster cores take more column chunks) and divides the
touched-byte model plus the stream bytes each chunk walks by the kernel time,
next to the STREAM-triad bandwidth at the same thread count. Every configuration
is checked against a per-source BFS on sampled sources.

```bash
python3 experiments/tglib_baseline.py build --dest external --out results/real/tglib_build.json
python3 experiments/tglib_baseline.py check --out results/real/tglib_check.json
NUMBA_NUM_THREADS=1 python3 experiments/tglib_baseline.py time --check results/real/tglib_check.json --procs 1,18 --out results/real/tglib_time.json
NUMBA_NUM_THREADS=18 python3 experiments/thread_diagnosis.py --dataset sx-stackoverflow --block-bits 4096,65536 --threads 1,6,12,18 --schedules static,dynamic --repeats 3 --out results/real/thread_diagnosis.json
```

Run (step 13 is added automatically and runs last):

```bash
cd STRATA-v23/STRATA-code/05_Temporal_STRATA
xcode-select -p && cmake --version     # build tools for step 18
.venv313/bin/python3 tools/run_m5_session.py --steps 18,19
```

Session `20261117-191557` ran both steps: step 18 stopped at CMake's OpenMP
check (Apple clang has no OpenMP runtime), and step 19 measured the thread sweep
(see 10.17).

### 10.17 STRATA-v24: TGLib with libomp, and chunks versus threads (backlog R63 / R64)

Apple clang ships no OpenMP runtime, so TGLib's `find_package(OpenMP REQUIRED)`
failed in the v23 session. Install Homebrew's libomp once; `build` then finds it
with `brew --prefix libomp` and passes it to CMake (and to the pip route through
`CMAKE_ARGS`). If the CMake build still fails, it retries once with TGLib's
`-Werror` filtered out by a CMake include (no source edit), recorded as
`werror_relaxed`. The check and time commands run with Numba's `workqueue`
threading layer so that only TGLib loads an OpenMP runtime.

The v23 thread diagnosis showed that, at a fixed thread count, more column chunks
alone cost time, while with the static schedule the chunk count equals the thread
count. Step 20 separates the two on one block of 65,536 sources: 6 threads with 6,
12, and 18 chunks (more passes over the stream), then 18 chunks with 6, 12, and 18
threads (more concurrent threads).

```bash
python3 experiments/tglib_baseline.py build --dest external --out results/real/tglib_build.json
NUMBA_THREADING_LAYER=workqueue python3 experiments/tglib_baseline.py check --out results/real/tglib_check.json
NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=1 python3 experiments/tglib_baseline.py time --check results/real/tglib_check.json --procs 1,18 --out results/real/tglib_time.json
NUMBA_NUM_THREADS=18 python3 experiments/thread_diagnosis.py --dataset sx-stackoverflow --block-bits 65536 --pairs 6:6,6:12,6:18,12:18,18:18 --repeats 3 --out results/real/thread_chunks.json
```

Run (step 13 is added automatically and runs last):

```bash
brew install libomp                    # once, for step 18
cd STRATA-v24/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 18,20
```

Session `20261117-220420` ran both steps: step 18 found libomp but stopped at the
link of TGLib's test executables (pip route) and at pybind11's CMake minimum
version (CMake route); step 20 measured the pairs (see 10.18).

### 10.18 STRATA-v25: TGLib link fix and random row updates (backlog R68 / R69)

TGLib adds the OpenMP flags to the compile line only, so the v24 build failed to
link its test executables. `build` now also passes `-L<libomp>/lib -lomp` to every
link, passes `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` on the CMake route (as TGLib's
own pyproject does for pip), builds the `_pytglib` binding, and installs it next
to the checkout's `pytglib/__init__.py` in `external/tglib_py/`. A failed link
records the linker's "Undefined symbols" lines.

The v24 pairs showed that the slowdown follows the number of concurrent threads,
not the number of chunks, while a STREAM triad keeps rising. Step 21 measures the
kernel's access shape alone (`tile[v] |= tile[u]` over private per-chunk tiles,
1,024 words per row, no timestamps or snapshots) with the first 16 million edges
of the sx-stackoverflow stream and with as many random pairs, at 1, 6, 12, and 18
threads. The word checksum must not change with the thread count.

```bash
python3 experiments/tglib_baseline.py build --dest external --out results/real/tglib_build.json
NUMBA_THREADING_LAYER=workqueue python3 experiments/tglib_baseline.py check --out results/real/tglib_check.json
NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=1 python3 experiments/tglib_baseline.py time --check results/real/tglib_check.json --procs 1,18 --out results/real/tglib_time.json
NUMBA_NUM_THREADS=18 python3 experiments/random_rows_bench.py --dataset sx-stackoverflow --words 1024 --edges 16000000 --threads 1,6,12,18 --repeats 3 --out results/real/random_rows.json
```

Run (step 13 is added automatically and runs last):

```bash
cd STRATA-v25/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 18,21
```

Sessions `20261118-064755` and `20261118-070247` ran both steps in STRATA-v25:
TGLib built through pip and matched the strict relation, and the random row
updates sped up with every thread count (see 10.19). Two earlier sessions that
day ran in STRATA-v24 before v25 existed, with v24 code.

### 10.19 STRATA-v26: session guards and the page budget (backlog R71 / R72 / R66)

`run_m5_session.py` now stops before any measurement (exit code 4) when it runs
in an older `STRATA-vN` folder than the newest one next to it (pass `--allow-old`
to override), or when `--steps` names a step it does not have. The session record
also keeps the version folder, the newest folder, the available and requested
steps, and where the interpreter, the working folder, and the runner live, with a
warning when they name different `STRATA-vN` folders. `run_gates.py` prints how
many lines still fit before the 12-page and 4-page limits (`tools/page_budget.py`;
information, not a gate). No new measurement step.

### 10.20 STRATA-v27: kernel phases past six threads (backlog R73)

The random row updates of step 21 speed up to 18 threads, but the strict kernel
on the same block does not. Step 22 times four variants of the column kernel on
one block of 65,536 sources (static, chunks = threads, 1, 6, 12, 18 threads):
the production strict kernel; the same tiles and copies without the snapshot
(not strict semantics, timing only); only the copies between the state and the
tiles; and the strict kernel with every tile allocated and touched before the
clock starts. The strict run is checked against a per-source BFS on sampled
sources, and every variant against its own reference state.

```bash
NUMBA_NUM_THREADS=18 python3 experiments/kernel_phases.py --dataset sx-stackoverflow --block-bits 65536 --threads 1,6,12,18 --repeats 3 --out results/real/kernel_phases.json
```

Run (step 13 is added automatically and runs last):

```bash
cd STRATA-v27/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 22
```

Sessions `20261118-081108` and `20261118-093215` ran step 22 twice with matching
results: the copy-only variant grew from about 3.3 s to 32.6 s between 1 and 18
threads, while the edge phase kept scaling (see 10.21).

### 10.21 STRATA-v28: the copy phase, tiles as the state, and the compiled baseline (backlog R75 / R76 / R79)

Step 23 separates the causes of the growing copy phase on the same block: copies
after every page of the fresh state was touched outside the clock, copies into a
state allocated once and refilled outside the clock, copies with chunk borders on
128-byte cache-line multiples, and the run with the tiles as the state (no shared
state, no copies), checked against the per-source BFS and the strict pair count.
Step 24 runs sx-stackoverflow with the tiles as the state: two blocks at 1, 6, 12,
and 18 threads, then every block at the fastest thread count. Step 25 reruns the
single-thread compiled baselines: the STRATA-numba column of Table 5 used an early
experimental kernel that copies the whole state at every timestamp group, and
both compiled columns timed a Python sort of the edge list; both now start from
the same sorted arrays, and STRATA runs the production core.

```bash
NUMBA_NUM_THREADS=18 python3 experiments/kernel_phases.py --dataset sx-stackoverflow --block-bits 65536 --threads 1,6,12,18 --variants strict,copy,copy_prefault,copy_reuse,copy_aligned,tiles --repeats 3 --out results/real/kernel_copy.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 65536 --blocks 2 --threads 1,6,12,18 --schedule static --state tiles --verify-sources 64 --out results/real/large_real_tiles_b65536.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 65536 --blocks all --threads best:results/real/large_real_tiles_b65536.json --schedule static --state tiles --verify-sources 64 --out results/real/large_real_tiles_full.json
NUMBA_NUM_THREADS=1 python3 experiments/compiled_baselines.py --no-tex
```

Run (step 13 is added automatically and runs last):

```bash
cd STRATA-v28/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 23,24,25
```

Steps 23-25 take about 33 minutes together (sessions 20261118-153811 / 172302 /
202159, all three consistent).

Results (used in the paper from STRATA-v29 on): copies into a freshly allocated block
state grow from 3.0 to 30.9 s between p = 1 and 18, while the same copies into a state
written before the clock fall from 2.2 to 0.6 s; the tile state falls from 31.3 to
6.5 s (2.7x below the strict kernel's best), and all 40 blocks take 257 s at p = 18
instead of 709 s at p = 6, with the same 2.68e12 pairs. The compiled baseline now
times the production core: the compiled STRATA kernel is 14.7-349.7x faster than the
compiled per-source BFS on the synthetic streams (v27: 2.5-5.3x with the legacy kernel).
In the v28 harness the copy_prefault variant gave the same times as copy, because its
touch loop (a self-assignment) may be removed by the compiler; STRATA-v29 writes a
runtime zero instead. The paper does not use copy_prefault.

### 10.22 STRATA-v30: the prefault check and the tile state on narrow blocks and with dynamic chunks (backlog R90 / R91)

Why: v29 places the shared-state loss in writing the tiles back into a freshly allocated
state, but the variant that writes every page before the clock (copy_prefault) did not
test this in v28: its touch loop could be removed by the compiler. v30 writes a runtime
zero instead. If copy_prefault now falls with the threads like copy_reuse, the cause is
the first write to fresh pages; if it still grows like copy, it is something else. The
tile state was measured only on wide blocks (beta = 65,536) with the static schedule;
step 27 adds narrow blocks (beta = 4,096, compare with the rows sweep of step 17) and the
dynamic schedule (4p chunks, at most W).

How: step 26 runs only the three copy variants on one block; step 27 runs two blocks per
configuration, verifies 64 sampled sources against a per-source BFS and keeps the pair
counts. ``large_real_blocked.py`` uses the tile state by default from v30; the earlier
session steps now say ``--state rows`` explicitly, so their commands still measure what
they measured.

What:

```bash
NUMBA_NUM_THREADS=18 python3 experiments/kernel_phases.py --dataset sx-stackoverflow --block-bits 65536 --threads 1,6,12,18 --variants copy,copy_prefault,copy_reuse --repeats 3 --out results/real/kernel_prefault.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 4096 --blocks 2 --threads 1,6,12,18 --schedule static --state tiles --verify-sources 64 --out results/real/large_real_tiles_b4096.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 4096 --blocks 2 --threads 1,6,12,18 --schedule dynamic --state tiles --verify-sources 64 --out results/real/large_real_tiles_dyn_b4096.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 65536 --blocks 2 --threads 1,6,12,18 --schedule dynamic --state tiles --verify-sources 64 --out results/real/large_real_tiles_dyn_b65536.json
```

Run (step 13 is added automatically and runs last):

```bash
cd STRATA-v30/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 26,27
```

Steps 26-27 took 11 minutes (session 20261119-041615). Results (in the paper from
STRATA-v31 on): with every page written before the clock the copies fall from 2.5 to
0.8 s between p = 1 and 18, like copies into a reused state, while fresh-state copies
grow from 3.3 to 32.0 s, so the loss is the first write to fresh pages inside the
kernel. On narrow blocks (beta = 4,096) the tile state is within 0.98-1.09x of the
shared state at every p and neither speeds up past six threads; the dynamic schedule
slows the tile state 2.1x (beta = 4,096) and 1.6x (beta = 65,536) at the best p.

### 10.23 STRATA-v31: tile state across block widths, and a second run of steps 26-27 (backlog R94 / R95)

Why: the tile state pays off at beta = 65,536 (6.6 s per block at p = 18, 4.7x over one
thread) but not at beta = 4,096 (64 words: about four words per thread at p = 18). Running
several narrow blocks at once is the same computation as one wider block whose column
chunks follow the block borders (Theorem 3), so the question is where between the two
widths the threads start to pay off, at what resident memory. Steps 26-27 ran once; a
second run tells whether the small narrow-block differences are real.

How: step 28 runs the tile state at beta = 16,384 and 32,768 (two blocks each, static,
64 sampled sources verified). R95 needs no new command: run steps 26 and 27 again in the
same session. ``large_real_blocked.py`` now defaults to the static schedule (the dynamic
schedule was slower with the tile state at both widths); step 16 names
``--schedule dynamic`` so that it still measures what it measured.

What:

```bash
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 16384 --blocks 2 --threads 1,6,12,18 --schedule static --state tiles --verify-sources 64 --out results/real/large_real_tiles_b16384.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 32768 --blocks 2 --threads 1,6,12,18 --schedule static --state tiles --verify-sources 64 --out results/real/large_real_tiles_b32768.json
```

Run (step 13 is added automatically and runs last):

```bash
cd STRATA-v31/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 26,27,28
```

Steps 26-28 took 15 minutes (session 20261119-050516). Steps 26-27 repeated the first
session within 7%. Step 28: the best mean block time is 4.96 s at p = 12 for beta =
16,384 (5.3 GB block state) and 3.82 s at p = 18 for beta = 32,768 (10.7 GB); with
beta = 4,096 and 65,536 the cost per source falls 3.5x from the narrowest to the widest
block. Words per thread alone do not predict the best p (beta = 4,096 is fastest at
10.7 words per thread, beta = 16,384 already slows at 14.2).

### 10.24 STRATA-v32: every block at beta = 32,768 (backlog R98)

Why: the half-width block halves the block state; its total time was only an
extrapolation from two blocks. How: the same full run as step 24 at beta = 32,768 and
18 threads (the best p of step 28). Result (run directly on 2026-11-19, now session
step 29): all 80 blocks in 280 s with the same 2.68e12 pairs, 19.3 GB peak RSS, against
257 s and 30.0 GB at beta = 65,536. ``large_real_blocked.py --memory-gb G`` picks the
widest power-of-two block whose state fits in G GB (11 -> 32,768; 25 -> 65,536 here).

```bash
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 32768 --blocks all --threads 18 --schedule static --state tiles --verify-sources 64 --out results/real/large_real_tiles_full_b32768.json
```

```bash
cd STRATA-v32/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 29
```

### 10.25 STRATA-v34: the unblocked kernel's state, large-graph windows, step counts (backlog R102 / R109 / R105)

Why: mock review 2 (reports/mock_review_v32.md) asks whether the first-write cost
found on sx-stackoverflow also limits the unblocked tiled kernel behind the large
synthetic streams (AE required item 2), asks for rolling windows on a large real
graph (R4-1), and asks why compiled ratios exceed the 64-source packing (R3-1).

How: step 30 runs the large synthetic streams with a fresh state (the production
path), a state whose pages were all written before the clock (diagnostic), and the
tiles as the state (one block of width n); all three must give the same pairs and
sampled columns. Step 31 recomputes 8 rolling windows (a quarter of the edges each)
of sx-askubuntu with the tile state at 18 threads, checking 16 sampled sources per
window against a per-source BFS, and advances the half-stream state by 5 later
batches, which must equal a full recomputation bit for bit. Step 32 counts the
elementary steps of the two compiled kernels of results/real/compiled_real.json
(no clock): BFS/STRATA steps are 34-45x on the observed data, so each BFS step costs
4.8-5.3x a STRATA word operation on the M5 Max.

What:

```bash
NUMBA_NUM_THREADS=18 python3 experiments/tile_state_unblocked.py --large 50000:1000000:400 --large 100000:2000000:400 --threads 1,6,12,18 --repeats 3 --out results/real/tile_state_unblocked.json
python3 experiments/fetch_real_temporal.py --names sx-askubuntu --log results/real/fetch_askubuntu_window.json
NUMBA_NUM_THREADS=18 python3 experiments/window_large.py --dataset sx-askubuntu --threads 18 --slots 8 --window-frac 0.25 --batches 5 --out results/real/window_large.json
python3 experiments/op_counts.py --out results/real/op_counts.json
```

```bash
cd STRATA-v34/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 30,31,32
```

Steps 30-32 took 1.5 minutes (session 20261119-180704). Results (in the paper from
STRATA-v35 on): on the large synthetic streams the tile state takes 0.89-1.03x and
pre-written pages 0.97-1.03x the shared state's time, so the rule fixed beforehand
(R112: tiles at least 5% faster at every p on both streams) is not met and the
paper keeps the shared-state results; the unblocked runs write their state once, so
the first-write cost of the block runs does not appear. On sx-askubuntu at 18
threads, 8 windows of 241k edges take 40-50 ms each; 96k-edge insertions take
102-110 ms against 320 ms for a full recomputation, because the tiled kernel
gathers and scatters the whole 3.2 GB state on every call.

### 10.26 STRATA-v34: a second machine (backlog R103)

Why: the project rule (R101 decision B) allows results that the user measures on a
second machine, reported only as comparisons within that machine.

How: ``tools/run_server_session.py --tag <name>`` reruns steps 16, 3, 30, 26, 24
and 28 with thread counts 1, c/8, c/4, c/2, c for c logical CPUs and writes every
file under results/server/<name>/; results/real/ is never touched. Check the plan
first with ``--dry-run``.

```bash
python3 tools/run_server_session.py --tag <name> --dry-run
python3 tools/run_server_session.py --tag <name>
```

Within-machine numbers: ``python3 experiments/server_numbers.py --tag <name>`` reads
results/server/<name>/ only and writes STRATA-latex/STRATA-paper-EN/server_numbers_<name>.tex
with ratios, thread counts, and the machine description; any other macro stops it.

### 10.27 STRATA-v36: insertion batches in place (backlog R113)

Why: step 31 showed that a 96k-edge insertion on sx-askubuntu costs 102-110 ms
against 320 ms for a full recomputation, because the tiled kernel gathers and
scatters the whole 3.2 GB state on every call. Row-major updates only the rows a
batch writes, so insertion cost should follow the batch, as Proposition 5 implies.

How: step 33 advances two copies of the half-stream state by the same 5 batches,
one with the tiled layout and one row-major in place (static, 18 chunks), and
requires both to equal a full recomputation bit for bit. The windows are skipped.

```bash
NUMBA_NUM_THREADS=18 python3 experiments/window_large.py --dataset sx-askubuntu --threads 18 --slots 0 --batches 5 --insert-layouts tiled,rowmajor --out results/real/window_insert.json
```

```bash
cd STRATA-v36/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 33
```

Step 33 took one minute (session 20261120-043121). Result (in the paper from STRATA-v37
on): row-major insertion takes 24-26 ms per 96k-edge batch, 4.1x less than tiled
(97-105 ms), against 191 ms for a full recomputation in the same run, so insertion cost
follows the batch. The full recomputation took 320 ms in step 31 (after the windows) and
191 ms here; the paper compares only numbers from one run.

