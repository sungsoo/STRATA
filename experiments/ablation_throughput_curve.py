#!/usr/bin/env python3
"""ablation_throughput_curve.py -- compiled-kernel edge throughput as a
function of n (hence W), addressing the review Minor on single-point throughput.
Shows whether word-parallel throughput stays flat or degrades as W grows
(memory-bandwidth pressure)."""
import sys, os, csv, time, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for numba_kernel
from numba_kernel import _strata_strict_kernel, _popcount_state
from temporal_strata.packed_reachability import words_for

def gen(n, m, seed=7):
    random.seed(seed); E=[]
    for t in range(500):
        for _ in range(m//500):
            u=random.randint(0,n-2); v=u+random.randint(1,20)
            if v<n: E.append((t,u,v))
    return E

def main():
    # warm-up compile
    _strata_strict_kernel(64, words_for(64),
        np.array([0]),np.array([0]),np.array([1]))
    rows=[]
    for n in [2000,4000,8000,16000]:
        E=gen(n, 500000)
        arr=np.array(sorted(E),dtype=np.int64)
        ts=np.ascontiguousarray(arr[:,0]); us=np.ascontiguousarray(arr[:,1]); vs=np.ascontiguousarray(arr[:,2])
        W=words_for(n); m=len(E)
        best=None
        for _ in range(3):
            t0=time.perf_counter(); R=_strata_strict_kernel(n,W,ts,us,vs); dt=time.perf_counter()-t0
            best=dt if best is None else min(best,dt)
        thru=m/best
        rows.append({"n":n,"W":W,"edges":m,"time_s":round(best,3),"throughput_Meps":round(thru/1e6,2)})
        print(f"n={n:6d} W={W:4d} m={m} time={best:.3f}s throughput={thru/1e6:.2f}M edges/s")
    out="results/ablation/throughput_curve.csv"
    os.makedirs(os.path.dirname(out),exist_ok=True)
    with open(out,"w",newline="") as fh:
        wr=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); wr.writeheader(); [wr.writerow(r) for r in rows]
    print("wrote",out)

if __name__=="__main__": main()
