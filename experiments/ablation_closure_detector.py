#!/usr/bin/env python3
"""ablation_closure_detector.py -- prove that frontier-delta pruning is
ESSENTIAL as the closure fixed-point change detector (review Must-fix 2). v15
showed pruning gives no strict-mode speedup; here we show its real role.

We compare two ways to run same-time closure on each timestamp group:
  (A) change-detected closure: sweep the group repeatedly, but stop as soon as a
      full sweep sets no new bit (the frontier-delta test IS the detector).
  (B) fixed-count closure: sweep the group a safe worst-case number of times
      (|G_t| sweeps, the bound that guarantees the fixed point without a
      detector), since without the delta test there is no signal to stop early.

Both compute the identical relation (verified). The ratio (B)/(A) quantifies how
much work the change detector saves, i.e. why pruning is necessary for closure
even though it is neutral for strict."""
import sys, os, csv, time
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from temporal_strata.packed_reachability import words_for, reachable_pairs

def load_groups(path):
    g=defaultdict(list); verts=set()
    for line in open(path):
        line=line.strip()
        if not line or line.startswith("#"): continue
        p=line.split(); u,v,t=int(p[0]),int(p[1]),int(p[2])
        g[t].append((u,v)); verts.add(u); verts.add(v)
    n=(max(verts)+1) if verts else 0
    return n, [g[t] for t in sorted(g)]

def init_state(n,W):
    R=np.zeros((n,W),dtype=np.uint64)
    for s in range(n): R[s,s>>6]|=np.uint64(1)<<np.uint64(s&63)
    return R

def closure_detected(n, groups, W):
    """(A) stop a group as soon as a sweep changes nothing (delta-detected)."""
    R=init_state(n,W); total_sweeps=0
    for grp in groups:
        changed=True
        while changed:
            changed=False; total_sweeps+=1
            for (u,v) in grp:
                nb=False
                for w in range(W):
                    if R[u,w] & ~R[v,w]: nb=True; break
                if nb:
                    for w in range(W): R[v,w]|=R[u,w]
                    changed=True
    return R, total_sweeps

def closure_fixedcount(n, groups, W):
    """(B) no detector: sweep each group |grp| times (safe worst-case bound)."""
    R=init_state(n,W); total_sweeps=0
    for grp in groups:
        g=len(grp)
        reps=max(1,g)  # worst-case sweeps to guarantee fixed point w/o detector
        for _ in range(reps):
            total_sweeps+=1
            for (u,v) in grp:
                for w in range(W): R[v,w]|=R[u,w]
    return R, total_sweeps

def main():
    rows=[]
    for path in sys.argv[1:]:
        n, groups = load_groups(path); W=words_for(n)
        t0=time.perf_counter(); Ra,sa=closure_detected(n,groups,W); ta=time.perf_counter()-t0
        t0=time.perf_counter(); Rb,sb=closure_fixedcount(n,groups,W); tb=time.perf_counter()-t0
        pa,pb=reachable_pairs(Ra,n),reachable_pairs(Rb,n)
        rows.append({"graph":os.path.basename(path),"n":n,
                     "detected_sweeps":sa,"fixedcount_sweeps":sb,
                     "detected_s":round(ta,4),"fixedcount_s":round(tb,4),
                     "speedup_from_detector":round(tb/ta,2) if ta else "",
                     "match":pa==pb})
        print(f"{os.path.basename(path):22s} detected:{sa}sw/{ta:.3f}s "
              f"fixedcount:{sb}sw/{tb:.3f}s detector_speedup={tb/ta:.2f}x match={pa==pb}")
    out="results/ablation/closure_detector.csv"
    os.makedirs(os.path.dirname(out),exist_ok=True)
    with open(out,"w",newline="") as fh:
        wr=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); wr.writeheader(); [wr.writerow(r) for r in rows]
    print("wrote",out)

if __name__=="__main__": main()
