#!/usr/bin/env python3
"""Tiny stand-in external tool for tests of experiments/external_baselines.py.

Usage: ext_reach_tool.py MODE EDGES SOURCES OUT, MODE in {strict, static}.
strict: per-source strict temporal BFS (deferred within-timestamp commits);
static: ordinary reachability ignoring timestamps (a different relation).
"""
import sys
from collections import defaultdict

mode, ep, sp, op = sys.argv[1:5]
edges = [tuple(map(int, l.split())) for l in open(ep) if l.strip()]
sources = [int(l) for l in open(sp) if l.strip()]
with open(op, "w") as out:
    for s in sources:
        reached = {s}
        if mode == "static":
            adj = defaultdict(list)
            for u, v, _t in edges:
                adj[u].append(v)
            stack = [s]
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y not in reached:
                        reached.add(y)
                        stack.append(y)
        else:
            i = 0
            while i < len(edges):
                j = i
                while j < len(edges) and edges[j][2] == edges[i][2]:
                    j += 1
                new = {v for u, v, _t in edges[i:j] if u in reached}
                reached |= new
                i = j
        for v in sorted(reached):
            out.write(f"{s} {v}\n")
