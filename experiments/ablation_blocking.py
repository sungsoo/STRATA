#!/usr/bin/env python3
"""ablation_blocking.py -- empirical source-blocking scaling curve, the
supplementary experiment requested in the earlier review and not yet provided:
vary the source-block width beta and measure (passes, per-pass resident words,
total wall-clock), confirming the locality--rescan trade-off and that blocking
trades time for a bounded per-pass footprint while preserving the exact result."""
import sys, os, csv, time, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from temporal_strata.packed_reachability import temporal_strata_packed, reachable_pairs

def load(path):
    edges=[]; verts=set()
    for line in open(path):
        line=line.strip()
        if not line or line.startswith("#"): continue
        p=line.split(); u,v,t=int(p[0]),int(p[1]),int(p[2])
        edges.append((t,u,v)); verts.add(u); verts.add(v)
    return (max(verts)+1 if verts else 0), edges

def main():
    path=sys.argv[1]
    n, edges = load(path)
    truth=reachable_pairs(temporal_strata_packed(n,edges,same_time_mode="strict"),n)
    rows=[]
    W_full=math.ceil(n/64)
    for beta in [n, n//2, n//4, n//8, n//16]:
        if beta<1: continue
        passes=math.ceil(n/beta)
        per_pass_words=n*math.ceil(beta/64)
        best=None
        for _ in range(3):
            t0=time.perf_counter()
            R=temporal_strata_packed(n,edges,same_time_mode="strict",block_bits=beta)
            dt=time.perf_counter()-t0
            best=dt if best is None else min(best,dt)
        pairs=reachable_pairs(R,n)
        rows.append({"beta":beta,"passes":passes,"per_pass_words":per_pass_words,
                     "total_s":round(best,5),"pairs":pairs,"matches":pairs==truth})
        print(f"beta={beta:6d} passes={passes:3d} per_pass_words={per_pass_words:9d} "
              f"time={best:.4f}s matches={pairs==truth}")
    out="results/ablation/blocking_scaling.csv"
    os.makedirs(os.path.dirname(out),exist_ok=True)
    with open(out,"w",newline="") as fh:
        wr=csv.DictWriter(fh,fieldnames=["beta","passes","per_pass_words","total_s","pairs","matches"])
        wr.writeheader(); [wr.writerow(r) for r in rows]
    print("wrote",out)

if __name__=="__main__": main()
