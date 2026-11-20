#!/usr/bin/env python3
"""ablation_naive_temporal.py -- quantify the non-triviality of STRATA's
temporal scheduling (review Must-fix 3 / Fix 1). A static bit-parallel BFS
already achieves O(nm/w); the question is what a NAIVE attempt to respect
timestamp order costs, and what STRATA saves over it.

Naive time-respecting baseline: at each timestamp group boundary, recompute the
reachable-source state by replaying ALL edges up to that point with a packed
bitset pass (i.e., no incremental carry-over of state across groups -- each
group's prefix is recomputed from scratch). This is the straightforward way to
'respect time' if one only has a static packed BFS primitive and applies it per
prefix. Cost ~ O(T * m_prefix * W) summed over T groups, i.e. O(T m W) worst case.

STRATA instead carries the state forward across groups with a per-group snapshot,
paying O(m W) once. We measure both (identical relation, bit-for-bit)."""
import sys, os, csv, time
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from temporal_strata.packed_reachability import (temporal_strata_packed,
    reachable_pairs, words_for)

def load(path):
    g=defaultdict(list); verts=set(); edges=[]
    for line in open(path):
        line=line.strip()
        if not line or line.startswith("#"): continue
        p=line.split(); u,v,t=int(p[0]),int(p[1]),int(p[2])
        edges.append((t,u,v)); g[t].append((u,v)); verts.add(u); verts.add(v)
    n=(max(verts)+1) if verts else 0
    return n, edges, [(t,g[t]) for t in sorted(g)]

def init_state(n,W):
    R=np.zeros((n,W),dtype=np.uint64)
    for s in range(n): R[s,s>>6]|=np.uint64(1)<<np.uint64(s&63)
    return R

def naive_prefix_recompute(n, groups, W):
    """At each group boundary, recompute state from scratch over the prefix.
    Strict semantics: within a group, read pre-group snapshot."""
    # final state after processing all groups, recomputing prefix each time
    R_final=None
    for gi in range(len(groups)):
        # recompute prefix [0..gi] from scratch
        R=init_state(n,W)
        for (t,grp) in groups[:gi+1]:
            snap=R.copy()
            for (u,v) in grp:
                for w in range(W): R[v,w]|=snap[u,w]
        R_final=R
    return R_final

def strata_carry(n, edges):
    return temporal_strata_packed(n, edges, same_time_mode="strict")

def main():
    rows=[]
    for path in sys.argv[1:]:
        n, edges, groups = load(path); W=words_for(n)
        t0=time.perf_counter(); Rn=naive_prefix_recompute(n,groups,W); tn=time.perf_counter()-t0
        t0=time.perf_counter(); Rs=strata_carry(n,edges); ts=time.perf_counter()-t0
        pn,ps=reachable_pairs(Rn,n),reachable_pairs(Rs,n)
        rows.append({"graph":os.path.basename(path),"n":n,"groups":len(groups),
                     "naive_prefix_s":round(tn,4),"strata_carry_s":round(ts,4),
                     "speedup":round(tn/ts,2) if ts else "","match":pn==ps})
        print(f"{os.path.basename(path):22s} groups={len(groups)} "
              f"naive_prefix={tn:.3f}s strata_carry={ts:.4f}s speedup={tn/ts:.1f}x match={pn==ps}")
    out="results/ablation/naive_temporal.csv"
    os.makedirs(os.path.dirname(out),exist_ok=True)
    with open(out,"w",newline="") as fh:
        wr=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); wr.writeheader(); [wr.writerow(r) for r in rows]
    print("wrote",out)

if __name__=="__main__": main()
