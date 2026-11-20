#!/usr/bin/env python3
"""ablation_pruning.py -- isolate the ACTUAL effect of frontier-delta
pruning, addressing the re-review finding that v14 claimed (without direct
measurement) that pruning helps under compilation on sparse graphs.

We measure compiled (Numba) strict kernels WITH and WITHOUT pruning across
graphs spanning a wide reachability-density range, and report the speed ratio.
The honest conclusion (see results): in this word-packed realization, pruning
does NOT yield a measurable speedup even when reachability stays sparse, because
the frontier-delta test costs about as much as the unconditional OR it would
skip. Pruning's real necessity is as the CHANGE DETECTOR for the closure
fixed point, not as a strict-mode speed optimization."""
import sys, os, csv, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for numba_kernel
import numpy as np
from numba_kernel import strata_strict_numba, bitset_noprune_numba, load_edges
from temporal_strata.packed_reachability import temporal_strata_packed, reachable_pairs

def density(path):
    n, edges = load_edges(path)
    R = temporal_strata_packed(n, edges, same_time_mode="strict")
    pairs = reachable_pairs(R, n)
    return n, len(edges), pairs, pairs/(n*n)

def main():
    graphs = [g for g in sys.argv[1:]] or []
    rows=[]
    for path in graphs:
        n, m, pairs, dens = density(path)
        # warm-up
        strata_strict_numba(n, load_edges(path)[1]); bitset_noprune_numba(n, load_edges(path)[1])
        def best(fn):
            b=None
            for _ in range(5):
                e=load_edges(path)[1]; t0=time.perf_counter(); fn(n,e); dt=time.perf_counter()-t0
                b=dt if b is None else min(b,dt)
            return b
        tp=best(strata_strict_numba); tn=best(bitset_noprune_numba)
        rows.append({"graph":os.path.basename(path),"n":n,"edges":m,
                     "reach_density":round(dens,4),
                     "prune_s":round(tp,5),"noprune_s":round(tn,5),
                     "prune/noprune":round(tp/tn,3)})
        print(f"{os.path.basename(path):24s} dens={dens:.3f} prune={tp:.4f}s noprune={tn:.4f}s ratio={tp/tn:.3f}")
    out="results/ablation/pruning_effect.csv"
    os.makedirs(os.path.dirname(out),exist_ok=True)
    with open(out,"w",newline="") as fh:
        wr=csv.DictWriter(fh,fieldnames=["graph","n","edges","reach_density","prune_s","noprune_s","prune/noprune"])
        wr.writeheader(); [wr.writerow(r) for r in rows]
    print("wrote",out)

if __name__=="__main__": main()
