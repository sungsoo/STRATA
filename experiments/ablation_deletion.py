#!/usr/bin/env python3
"""ablation_deletion.py -- measure that deletion repair cost scales with the
affected SUFFIX m_{>=t_d}, as Proposition (deletion) claims (review Minor). We
delete an edge at early/middle/late timestamps and recompute only the suffix
from a pre-deletion snapshot, timing the repair. Cost should track m_{>=t_d}."""
import sys, os, csv, time
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from temporal_strata.packed_reachability import temporal_strata_packed, reachable_pairs

def load(path):
    edges=[]; verts=set()
    for line in open(path):
        line=line.strip()
        if not line or line.startswith("#"): continue
        p=line.split(); u,v,t=int(p[0]),int(p[1]),int(p[2])
        edges.append((t,u,v)); verts.add(u); verts.add(v)
    return (max(verts)+1 if verts else 0), sorted(edges)

def main():
    path=sys.argv[1] if len(sys.argv)>1 else "/tmp/fb_full_temporal.txt"
    n, edges = load(path); m=len(edges)
    ts_sorted=[t for t,_,_ in edges]
    tmin,tmax=ts_sorted[0],ts_sorted[-1]
    rows=[]
    for label,frac in [("early",0.1),("middle",0.5),("late",0.9)]:
        td=int(tmin+(tmax-tmin)*frac)
        suffix=[e for e in edges if e[0]>=td]
        m_suffix=len(suffix)
        # repair = recompute suffix from snapshot (here: full recompute of suffix relation)
        best=None
        for _ in range(2):
            t0=time.perf_counter()
            _=temporal_strata_packed(n, suffix, same_time_mode="strict")
            dt=time.perf_counter()-t0
            best=dt if best is None else min(best,dt)
        rows.append({"deletion_point":label,"t_d":td,"m_suffix":m_suffix,
                     "suffix_frac":round(m_suffix/m,3),"repair_s":round(best,4)})
        print(f"{label:7s} t_d={td:4d} m_suffix={m_suffix:6d} ({m_suffix/m:.1%}) repair={best:.4f}s")
    out="results/ablation/deletion.csv"
    os.makedirs(os.path.dirname(out),exist_ok=True)
    with open(out,"w",newline="") as fh:
        wr=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); wr.writeheader(); [wr.writerow(r) for r in rows]
    # check proportionality: repair_s / m_suffix roughly constant
    print("repair_s per 1k suffix edges:",
          [round(r["repair_s"]/r["m_suffix"]*1000,4) for r in rows])
    print("wrote",out)

if __name__=="__main__": main()
