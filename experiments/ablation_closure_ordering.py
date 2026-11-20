"""ablation_closure_ordering.py -- closure sweeps: naive vs ordered (Q2).

Reviewer Q2: for closure mode, do you implement per-group topological ordering
or SCC condensation to bound sweeps to d_t instead of naive repeated passes?

This script implements two intra-group closure strategies and verifies they are
bit-identical, while measuring the number of edge-sweeps each performs:

  (A) naive    : repeat "for e in group: propagate" until a full sweep makes no
                 change. Sweep count is d_t + 1 where d_t is the longest
                 same-timestamp dependency chain the fixed point must traverse.

  (B) ordered  : condense the intra-group edge set into SCCs (Tarjan), process
                 SCCs in topological order; within an SCC iterate to local fixed
                 point, then push once along the DAG order. This bounds the work
                 to one topological sweep plus intra-SCC iteration, replacing
                 d_t global sweeps by a single ordered pass over the condensation.

We report per dataset the empirical d_t distribution (mean/median/max group
diameter, i.e. naive sweeps - 1) and the sweep-count reduction from ordering,
giving the empirical characterization of d_t the reviewer asked to expand.
"""
import sys, os, time, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from temporal_strata.io import load_standard_edges
from temporal_strata.stream import sort_and_validate_edges
from temporal_strata.packed_reachability import words_for, reachable_pairs
from itertools import groupby


def _identity(n, W):
    R = np.zeros((n, W), dtype=np.uint64)
    for s in range(n):
        R[s, s >> 6] |= np.uint64(1) << np.uint64(s & 63)
    return R


def _tarjan_scc(nodes, adj):
    """Iterative Tarjan SCC. Returns (comp_id dict, list of components in
    reverse-topological order)."""
    index = {}; low = {}; onstack = {}; stack = []; comps = []
    counter = [0]
    sys.setrecursionlimit(10000)
    for start in nodes:
        if start in index:
            continue
        work = [(start, iter(adj.get(start, ())))]
        while work:
            v, it = work[-1]
            if v not in index:
                index[v] = low[v] = counter[0]; counter[0] += 1
                stack.append(v); onstack[v] = True
            advanced = False
            for w in it:
                if w not in index:
                    work.append((w, iter(adj.get(w, ()))))
                    advanced = True
                    break
                elif onstack.get(w):
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop(); onstack[w] = False; comp.append(w)
                    if w == v:
                        break
                comps.append(comp)
            work.pop()
            if work:
                pv = work[-1][0]
                low[pv] = min(low[pv], low[v])
    comp_id = {}
    for i, comp in enumerate(comps):
        for v in comp:
            comp_id[v] = i
    return comp_id, comps  # comps already in reverse-topological order


def closure_naive(n, stream):
    W = words_for(n); R = _identity(n, W)
    sweeps = []
    for _, git in groupby(stream, key=lambda e: e.t):
        group = list(git)
        s = 0; changed = True
        while changed:
            changed = False; s += 1
            for e in group:
                cand = R[e.u]; nz = cand & ~R[e.v]
                if nz.any():
                    R[e.v] |= cand; changed = True
        sweeps.append(s)
    return R, sweeps


def closure_ordered(n, stream):
    """Condense each group into SCCs, process in topological order."""
    W = words_for(n); R = _identity(n, W)
    ordered_sweeps = []
    for _, git in groupby(stream, key=lambda e: e.t):
        group = list(git)
        nodes = set()
        adj = {}
        for e in group:
            adj.setdefault(e.u, []).append(e.v)
            nodes.add(e.u); nodes.add(e.v)
        comp_id, comps = _tarjan_scc(nodes, adj)
        # comps in reverse-topological order -> process reversed for topo order
        local_sweeps = 0
        for comp in reversed(comps):
            # intra-SCC fixed point (only matters if |comp|>1 or self-loop)
            if len(comp) > 1:
                cs = set(comp)
                changed = True
                while changed:
                    changed = False; local_sweeps += 1
                    for e in group:
                        if e.u in cs and e.v in cs:
                            cand = R[e.u]; nz = cand & ~R[e.v]
                            if nz.any():
                                R[e.v] |= cand; changed = True
            else:
                local_sweeps += 1
            # push along edges leaving this comp (DAG order guarantees targets
            # are processed after their sources)
            for e in group:
                if comp_id[e.u] == comp_id.get(e.v, -1):
                    continue
                if e.u in comp_id and comp_id[e.u] == comp_id[e.u] and e.u in set(comp):
                    cand = R[e.u]; nz = cand & ~R[e.v]
                    if nz.any():
                        R[e.v] |= cand
        ordered_sweeps.append(local_sweeps)
    return R, ordered_sweeps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges")
    ap.add_argument("--out", default="results/ablation/closure_ordering.csv")
    args = ap.parse_args()

    streams = []
    if args.edges:
        n, edges, _ = load_standard_edges(args.edges)
        streams.append(("input", n, sort_and_validate_edges(n, edges)))
    else:
        rng = np.random.default_rng(11)
        # chain group (long dependency, large d_t) and random group (small d_t)
        # long chain within a single timestamp:
        n = 512
        chain = [(0, i, i + 1) for i in range(n - 1)]
        streams.append(("long_chain_t0", n, sort_and_validate_edges(n, chain)))
        # random dense-ish single group
        rnd = [(0, int(rng.integers(0, n)), int(rng.integers(0, n))) for _ in range(2000)]
        rnd = [(t, u, v) for (t, u, v) in rnd if u != v]
        streams.append(("random_t0", n, sort_and_validate_edges(n, rnd)))
        # multi-group mild
        mg = []
        for t in range(50):
            for _ in range(40):
                u = int(rng.integers(0, n)); v = int(rng.integers(0, n))
                if u != v:
                    mg.append((t, u, v))
        streams.append(("multi_group", n, sort_and_validate_edges(n, mg)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import csv
    rows = []
    print(f"{'stream':>14} {'naive_sweeps':>12} {'d_t(mean)':>10} {'d_t(max)':>9} "
          f"{'ordered_sweeps':>14} {'pairs':>8} {'agree':>6}")
    for name, n, stream in streams:
        Rn, sn = closure_naive(n, stream)
        Ro, so = closure_ordered(n, stream)
        agree = bool(np.array_equal(Rn, Ro))
        pairs = reachable_pairs(Rn, n)
        dt_mean = float(np.mean([s - 1 for s in sn]))  # d_t = sweeps-1
        dt_max = int(max(s - 1 for s in sn))
        tot_naive = sum(sn); tot_ord = sum(so)
        print(f"{name:>14} {tot_naive:>12} {dt_mean:>10.2f} {dt_max:>9} "
              f"{tot_ord:>14} {pairs:>8} {str(agree):>6}")
        rows.append(dict(stream=name, n=n, naive_sweeps=tot_naive,
                         dt_mean=round(dt_mean, 3), dt_max=dt_max,
                         ordered_sweeps=tot_ord, pairs=pairs, agree=agree))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); [w.writerow(r) for r in rows]
    print("wrote", args.out)
    print("ALL AGREE:", all(r["agree"] for r in rows))


if __name__ == "__main__":
    main()
