#!/usr/bin/env python3
"""ablation_graphblas.py -- FAIR same-machine comparison of STRATA's static
reduction against SuiteSparse:GraphBLAS (review v18 Critical fix). v18's
comparison measured GraphBLAS single-shot (no warm-up), inflating its time with
JIT/allocation overhead and producing a non-credible 10x reversal. Here we apply
IDENTICAL measurement discipline to both: warm-up then best-of-3, and use a
proper closure idiom. We report the honest ratio.

STRATA reduces (in closure mode) to static all-pairs reachability, the same
transitive-closure object GraphBLAS computes via masked mxm to a fixed point."""
import sys, os, csv, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from temporal_strata.packed_reachability import temporal_strata_packed, reachable_pairs
from temporal_strata.datasets import FACEBOOK
import graphblas as gb
from graphblas import Matrix, semiring

def load_static(path, cap=None):
    es=set(); verts=set()
    for line in open(path):
        line=line.strip()
        if not line or line.startswith("#"): continue
        p=line.split(); u,v=int(p[0]),int(p[1])
        if cap and (u>=cap or v>=cap): continue
        es.add((u,v)); verts.add(u); verts.add(v)
    n=(max(verts)+1) if verts else 0
    return n, list(es)

def gb_closure_pairs(n, es):
    """Reachability transitive closure via masked mxm to a fixed point
    (only newly-discovered entries are added each round; standard idiom)."""
    A=Matrix.from_coo([u for u,v in es],[v for u,v in es],[True]*len(es),
                      nrows=n, ncols=n, dtype=bool)
    R=A.dup()
    for i in range(n): R[i,i]=True
    while True:
        before=R.nvals
        # frontier expansion: R = R | (R @ A), masked accumulation
        R(mask=~R.S) << semiring.lor_land(R @ A)
        if R.nvals==before: break
    return R.nvals

def strata_static_pairs(n, es):
    return reachable_pairs(
        temporal_strata_packed(n, [(0,u,v) for u,v in es], same_time_mode="closure"), n)

def best_of(fn, n, es, k=3):
    b=None
    for _ in range(k):
        t0=time.perf_counter(); fn(n, es); dt=time.perf_counter()-t0
        b=dt if b is None else min(b, dt)
    return b

def main():
    path=sys.argv[1] if len(sys.argv)>1 else \
        str(FACEBOOK)
    rows=[]
    for cap in [256, 512, 1024, 2048, 4096]:
        n, es = load_static(path, cap=cap)
        sp=strata_static_pairs(n,es); gp=gb_closure_pairs(n,es)  # warm-up + correctness
        assert sp==gp, f"mismatch at n={n}: {sp} vs {gp}"
        ts=best_of(strata_static_pairs, n, es)
        tg=best_of(gb_closure_pairs, n, es)
        rows.append({"n":n,"edges":len(es),"pairs":sp,"agree":sp==gp,
                     "strata_s":round(ts,4),"graphblas_s":round(tg,4),
                     "ratio_strata_over_gb":round(ts/tg,2) if tg else ""})
        print(f"n={n} edges={len(es)} pairs={sp} agree={sp==gp} "
              f"STRATA={ts:.4f}s GraphBLAS={tg:.4f}s ratio={ts/tg:.2f}x")
    out="results/ablation/graphblas_static.csv"
    os.makedirs(os.path.dirname(out),exist_ok=True)
    with open(out,"w",newline="") as fh:
        wr=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); wr.writeheader(); [wr.writerow(r) for r in rows]
    print("wrote",out)

if __name__=="__main__": main()
