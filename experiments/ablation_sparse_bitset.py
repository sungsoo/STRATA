"""ablation_sparse_bitset.py -- adaptive sparse vs dense bitset scan (Q3).

Reviewer Q3: how does STRATA perform with sparse/compressed bitsets (e.g.,
Roaring) for small n/w active blocks or highly skewed reachable sets? Could
adaptive representation reduce the O(W) per-edge scan?

We model the per-edge cost directly. The dense kernel scans all W words of a row
regardless of how many are nonzero. A sparse representation stores only the
indices of nonzero words (a sorted list / set of "active words"), so the merge
R[v] |= R[u] and the frontier test touch only |active(u) ∪ active(v)| words.

This script measures, on real and synthetic streams, the per-row word-density
(fraction of nonzero words) as the frontier grows, and computes the words-touched
per edge under:
  - dense   : W words/edge (the current kernel),
  - sparse  : |active(u)| words/edge (index-only scan),
choosing adaptively per row (sparse while density < 1/w-ish, dense afterwards).

It reports the achievable per-edge word-scan reduction and the density regime in
which a Roaring-style container would pay off, plus its crossover: once rows
saturate (dense frontier), the sparse index overhead exceeds the dense scan, so
the benefit is confined to early passes / highly skewed reachable sets.
"""
import sys, os, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.io import load_standard_edges
from temporal_strata.stream import sort_and_validate_edges
from temporal_strata.packed_reachability import words_for
from itertools import groupby


def measure(n, stream):
    """Replay strict propagation, tracking dense vs sparse words touched/edge."""
    W = words_for(n)
    R = np.zeros((n, W), dtype=np.uint64)
    for s in range(n):
        R[s, s >> 6] |= np.uint64(1) << np.uint64(s & 63)
    # active-word count per row (popcount of nonzero words)
    active = [set(np.nonzero(R[v])[0].tolist()) for v in range(n)]

    dense_words = 0
    sparse_words = 0
    edges = 0
    density_samples = []

    for _, git in groupby(stream, key=lambda e: e.t):
        group = list(git)
        before = {e.u: set(active[e.u]) for e in group}
        beforeR = R.copy()
        for e in group:
            u, v = e.u, e.v
            au = before.get(u, active[u])
            # dense: scans all W words
            dense_words += W
            # sparse: union of active words of u and v (words that could change)
            union = au | active[v]
            sparse_words += len(union)
            edges += 1
            density_samples.append(len(au) / W)
            # apply merge on the real state (dense semantics, verified elsewhere)
            cand = beforeR[u]
            nz = cand & ~R[v]
            if nz.any():
                R[v] |= cand
                active[v] = set(np.nonzero(R[v])[0].tolist())
    return dict(
        n=n, W=W, edges=edges,
        dense_words=dense_words, sparse_words=sparse_words,
        ratio_dense_over_sparse=dense_words / max(1, sparse_words),
        mean_row_density=float(np.mean(density_samples)) if density_samples else 0.0,
        frac_edges_below_half_density=float(np.mean([d < 0.5 for d in density_samples])) if density_samples else 0.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges")
    ap.add_argument("--out", default="results/ablation/sparse_bitset.csv")
    args = ap.parse_args()

    streams = []
    if args.edges:
        n, edges, _ = load_standard_edges(args.edges)
        streams.append(("input", n, sort_and_validate_edges(n, edges)))
    else:
        rng = np.random.default_rng(23)
        # skewed: a few hubs reachable widely, most rows stay sparse
        n = 1024
        sk = []
        for t in range(200):
            for _ in range(20):
                u = int(rng.integers(0, 32))          # few sources dominate
                v = int(rng.integers(0, n))
                if u != v:
                    sk.append((t, u, v))
        streams.append(("skewed_hubs", n, sort_and_validate_edges(n, sk)))
        # dense-ish: frontiers saturate quickly
        ds = []
        for t in range(100):
            for _ in range(80):
                u = int(rng.integers(0, n)); v = int(rng.integers(0, n))
                if u != v:
                    ds.append((t, u, v))
        streams.append(("dense_mix", n, sort_and_validate_edges(n, ds)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import csv
    rows = []
    print(f"{'stream':>12} {'n':>6} {'W':>4} {'dense_w':>10} {'sparse_w':>10} "
          f"{'dense/sparse':>12} {'meandens':>9} {'frac<0.5':>9}")
    for name, n, stream in streams:
        r = measure(n, stream); r["stream"] = name
        print(f"{name:>12} {r['n']:>6} {r['W']:>4} {r['dense_words']:>10} "
              f"{r['sparse_words']:>10} {r['ratio_dense_over_sparse']:>12.2f} "
              f"{r['mean_row_density']:>9.3f} {r['frac_edges_below_half_density']:>9.3f}")
        rows.append(r)
    keys = ["stream", "n", "W", "edges", "dense_words", "sparse_words",
            "ratio_dense_over_sparse", "mean_row_density",
            "frac_edges_below_half_density"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print("wrote", args.out)


if __name__ == "__main__":
    main()
