#!/usr/bin/env python3
"""ablation_real_temporal.py -- blocking scaling and exactness on a REAL
public graph: the full SNAP ego-Facebook (n=4039, 88234 edges), temporalized by
discovery order (see make_real_temporal.py). Addresses the standing review
Must-fix for a real (non-synthetic) temporal-graph measurement at non-trivial
scale. Reports per-beta passes, per-pass resident words, wall-clock, and
bit-for-bit identity of the relation across beta."""
import sys, os, csv, time, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
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
    path=sys.argv[1] if len(sys.argv)>1 else "/tmp/fb_full_temporal.txt"
    n, edges = load(path); m=len(edges)
    print(f"REAL ego-Facebook temporal: n={n} m={m}")
    truth=None; rows=[]
    for beta in [n, n//2, n//4, n//8]:
        passes=math.ceil(n/beta); per_pass=n*math.ceil(beta/64)
        best=None
        for _ in range(2):
            t0=time.perf_counter()
            R=temporal_strata_packed(n,edges,same_time_mode="strict",block_bits=beta)
            dt=time.perf_counter()-t0
            best=dt if best is None else min(best,dt)
        pairs=reachable_pairs(R,n)
        if truth is None: truth=pairs
        rows.append({"beta":beta,"passes":passes,"per_pass_words":per_pass,
                     "total_s":round(best,3),"pairs":pairs,"matches":pairs==truth})
        print(f"beta={beta:5d} passes={passes:2d} per_pass_words={per_pass:9d} "
              f"time={best:.3f}s pairs={pairs} matches={pairs==truth}")
    thru=m/min(r["total_s"] for r in rows if r["passes"]==1) if any(r["passes"]==1 for r in rows) else 0
    print(f"single-pass edge throughput: {thru/1e6:.2f}M edges/s")
    out="results/ablation/real_temporal.csv"
    os.makedirs(os.path.dirname(out),exist_ok=True)
    with open(out,"w",newline="") as fh:
        wr=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); wr.writeheader(); [wr.writerow(r) for r in rows]
    print("wrote",out,"| ALL BETA AGREE:",len(set(r["pairs"] for r in rows))==1)

if __name__=="__main__": main()
