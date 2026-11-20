#!/usr/bin/env python3
"""make_real_temporal.py -- build a temporal stream from the FULL real SNAP
ego-Facebook graph (n=4039, 88234 undirected edges; Leskovec & Mcauley 2012,
bundled as datasets_real_temporal/static/facebook_combined.txt). We assign timestamps to the real edges
so the topology is genuinely real (community structure, degree distribution),
not synthetic. Timestamps are assigned by a breadth-first discovery order from a
seed, approximating a plausible activation order; this yields a real-topology
temporal reachability instance far larger (n=4039) than the n=512 ego-net used
earlier. Output: standardized 'src dst timestamp'."""
import sys, os, random
from collections import defaultdict, deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from temporal_strata.datasets import FACEBOOK as SRC  # noqa: E402

def main(n_groups=200, out="/tmp/fb_full_temporal.txt"):
    adj=defaultdict(list); edges=[]
    for line in open(SRC):
        u,v=map(int,line.split()); adj[u].append(v); adj[v].append(u)
        edges.append((u,v))
    n=max(max(u,v) for u,v in edges)+1
    # BFS discovery order from vertex 0 to assign a plausible activation time
    order={}; seen=set([0]); q=deque([0]); t=0
    while q:
        x=q.popleft(); order[x]=t; t+=1
        for y in sorted(adj[x]):
            if y not in seen: seen.add(y); q.append(y)
    for x in range(n):
        if x not in order: order[x]=t; t+=1
    # edge timestamp = bucket of the later endpoint's discovery order
    maxo=max(order.values())
    with open(out,"w") as f:
        for (u,v) in edges:
            ts=int(order[max(u,v,key=lambda z:order[z])]/(maxo+1)*n_groups)
            f.write(f"{u} {v} {ts}\n")  # directed u->v in discovery direction
    print(f"wrote {out}: n={n}, edges={len(edges)}, groups~{n_groups}")

if __name__=="__main__":
    main()
