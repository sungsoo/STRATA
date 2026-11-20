#!/usr/bin/env python3
"""Why the block kernel slows down beyond six threads (v23).

Why: on sx-stackoverflow (v22 session, static schedule) the mean block time is
lowest at p = 6 for both block widths, and p = 12 / 18 are slower; at beta = 65536,
p = 18 is about as slow as p = 1. The M5 Max has 6 "Super" and 12 "Performance"
cores (machine_info perf_levels). Two explanations compete, and the paper may
state neither until one is measured:
  (a) heterogeneous cores: a static split gives every thread the same columns, so
      the slowest core sets the block time once threads land on the slower level;
  (b) memory traffic: every thread reads the whole edge stream and its share of
      the touched rows, so the achieved bandwidth saturates.

How (one source block, the same prepared stream for every configuration):
* kernel time at threads x {static, dynamic}; dynamic hands out column chunks on
  demand, so a faster core takes more of them. (a) predicts dynamic beating static
  at p > 6; (b) predicts both flat or worse.
* touched-byte model (temporal_strata/traffic.py) plus the stream bytes every
  column chunk reads (us, vs, group pointers), divided by the kernel time, next to
  the measured STREAM-triad bandwidth at the same thread count.
* every configuration's block is checked against an independent per-source BFS on
  sampled sources and must give the same reachable-pair count.
v24 (``--pairs``): the v23 run showed that, at a fixed thread count, more column
chunks alone cost time (each chunk walks the whole stream once). With the static
schedule, chunks = p, so passes and concurrent threads grew together.
``--pairs p:c,...`` runs explicit (threads, chunks) pairs, static schedule, so a
fixed p with c = 6 / 12 / 18 isolates the passes and a fixed c = 18 with
p = 6 / 12 / 18 isolates the concurrent threads.
The JSON holds numbers only; the session analysis states which explanation they
support, and the paper uses that only after it is measured.

What:
    NUMBA_NUM_THREADS=18 python3 experiments/thread_diagnosis.py \
        --dataset sx-stackoverflow --block-bits 4096,65536 --threads 1,6,12,18 \
        --schedules static,dynamic --repeats 3 --out results/real/thread_diagnosis.json
    NUMBA_NUM_THREADS=18 python3 experiments/thread_diagnosis.py \
        --dataset sx-stackoverflow --block-bits 65536 --pairs 6:6,6:12,6:18,12:18,18:18 \
        --repeats 3 --out results/real/thread_chunks.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def stream_read_bytes(prep, chunks):
    """Bytes of the prepared stream every column chunk walks once (u, v, group
    pointers, snapshot slots and dirty-row lists)."""
    arrays = (getattr(prep, a) for a in ("us", "vs", "gptr", "src_slot", "dirty_idx", "dptr"))
    per_pass = sum(int(x.nbytes) for x in arrays if x is not None)
    return per_pass * int(chunks)


def summarize(rows):
    """Comparisons the analysis needs, computed only from measured rows.

    For each block width: the best (threads, schedule); per thread count the
    dynamic / static median-time ratio (below 1 = dynamic faster)."""
    out = {}
    for bits in sorted({r["block_bits"] for r in rows}):
        sub = [r for r in rows if r["block_bits"] == bits]
        best = min(sub, key=lambda r: r["kernel_s_median"])
        ratio = {}
        for p in sorted({r["threads"] for r in sub}):
            st = [r for r in sub if r["threads"] == p and r["schedule"] == "static"]
            dy = [r for r in sub if r["threads"] == p and r["schedule"] == "dynamic"]
            if st and dy:
                ratio[str(p)] = dy[0]["kernel_s_median"] / st[0]["kernel_s_median"]
        out[str(bits)] = {"best_threads": best["threads"], "best_schedule": best["schedule"],
                          "best_kernel_s_median": best["kernel_s_median"],
                          "dynamic_over_static": ratio}
    return out


def parse_pairs(spec):
    """'6:6,6:12' -> [(6, 6), (6, 12)] (threads, chunks)."""
    out = []
    for tok in spec.split(","):
        p, c = tok.split(":")
        out.append((int(p), int(c)))
    return out


def summarize_pairs(rows):
    """Per block width: median seconds by threads, then chunks; and the two
    one-factor series the pair grid is built for."""
    out = {}
    for bits in sorted({r["block_bits"] for r in rows}):
        sub = [r for r in rows if r["block_bits"] == bits]
        grid = {}
        for r in sub:
            grid.setdefault(str(r["threads"]), {})[str(r["chunks"])] = r["kernel_s_median"]
        p0 = min(r["threads"] for r in sub)
        c0 = max(r["chunks"] for r in sub)
        out[str(bits)] = {
            "grid": grid,
            "chunks_at_fixed_threads": {"threads": p0, "series": {
                str(r["chunks"]): r["kernel_s_median"] for r in sub if r["threads"] == p0}},
            "threads_at_fixed_chunks": {"chunks": c0, "series": {
                str(r["threads"]): r["kernel_s_median"] for r in sub if r["chunks"] == c0}},
        }
    return out


def run(args):
    from experiments.bandwidth_roofline import stream_triad
    from experiments.bench_common import DATA, _bfs_sources_strict
    from experiments.large_real_blocked import column_of, peak_rss_bytes, pick_sources
    from experiments.thread_record import thread_record
    from temporal_strata.io import load_standard_arrays
    from temporal_strata.parallel import (_empty_state, _set_identity_vec, machine_info,
                                          prepare_stream_arrays, resolve_plan, run_prepared,
                                          set_threads)
    from temporal_strata.traffic import strict_kernel_bytes

    pairs = parse_pairs(args.pairs) if args.pairs else None
    threads = sorted({p for p, _ in pairs}) if pairs else [int(x) for x in args.threads.split(",")]
    schedules = ["static"] if pairs else args.schedules.split(",")
    configs = ([(p, "static", c) for p, c in pairs] if pairs else
               [(p, s_, None) for p in threads for s_ in schedules])
    widths = [int(x) for x in args.block_bits.split(",")]
    path = DATA / f"{args.dataset}.txt"
    if not path.exists():
        raise SystemExit(f"missing {path}; run experiments/fetch_real_temporal.py --names {args.dataset}")
    n, ts, us, vs, meta = load_standard_arrays(path)
    set_threads(max(threads))
    prep = prepare_stream_arrays(n, ts, us, vs, directed=True, same_time_mode="strict")
    del ts, us, vs
    print(f"{args.dataset}: n={n:,} m={prep.num_edges:,} prep {prep.timings['total_s']:.2f} s")

    triad = stream_triad(threads, args.array_mib, args.repeats)
    triad_at = {r["threads"]: r["GBps"] for r in triad["rows"]}
    rows = []
    for bits in widths:
        width = min(bits, n)
        src = pick_sources([(0, width)], args.verify_sources)
        bfs = _bfs_sources_strict(n, prep.gptr, prep.us, prep.vs, src)
        pairs_ref = None
        for p, sched, nch in configs:
            eff = set_threads(p)
            if eff != p:
                print(f"  threads={p} unavailable; skipped")
                continue
            samples, plan = [], None
            for rep in range(args.repeats + 1):            # first run = warm-up + check
                Rb = _empty_state(n, width)
                _set_identity_vec(Rb, 0, width)
                plan = resolve_plan(Rb.shape[1], eff, layout=args.layout, schedule=sched,
                                    chunks=nch)
                t0 = time.perf_counter()
                run_prepared(Rb, prep, plan)
                dt = time.perf_counter() - t0
                if rep == 0:
                    for i, s in enumerate(src):
                        if not np.array_equal(column_of(Rb, 0, int(s)), bfs[i]):
                            raise SystemExit(f"beta={bits} p={eff} {sched} c={nch}: source {s} differs")
                    npairs = int(np.bitwise_count(Rb).sum())
                    pairs_ref = npairs if pairs_ref is None else pairs_ref
                    if npairs != pairs_ref:
                        raise SystemExit(f"beta={bits} p={eff} {sched} c={nch}: pair count differs")
                else:
                    samples.append(dt)
                del Rb
            med = statistics.median(samples)
            pd = plan.as_dict()
            chunks = int(pd["nchunks"])
            model = strict_kernel_bytes(prep.num_edges, prep.dirty_total, n,
                                        -(-width // 64), layout=plan.resolved_layout)
            sbytes = stream_read_bytes(prep, chunks)
            total = model["total_bytes"] + sbytes
            row = {"block_bits": bits, "width": width, "threads": eff, "schedule": sched,
                   "plan": pd, "chunks": chunks, "kernel_s_samples": samples,
                   "kernel_s_median": med, "kernel_s_best": min(samples),
                   "model": model, "stream_read_bytes": sbytes,
                   "achieved_GBps": total / med / 1e9,
                   "triad_GBps": triad_at.get(eff),
                   "ratio_to_triad": (total / med / 1e9) / triad_at[eff] if eff in triad_at else None,
                   "pairs": pairs_ref}
            rows.append(row)
            print(f"  beta={bits} p={eff} {sched} chunks={chunks}: median {med:.3f} s, "
                  f"{row['achieved_GBps']:.1f} GB/s touched")
    out = {"dataset": args.dataset, "n": int(n), "m": prep.num_edges, "meta": meta,
           "layout": args.layout, "repeats": args.repeats,
           "verified_sources_per_width": args.verify_sources, "reference": "per-source strict BFS",
           "stream_triad": triad, "rows": rows,
           "summary": summarize_pairs(rows) if pairs else summarize(rows),
           "pairs": [list(x) for x in pairs] if pairs else None,
           "note": "touched bytes are an upper bound on DRAM traffic (cache reuse can push "
                   "the ratio to triad above 1); stream_read_bytes = edge stream walked once "
                   "per column chunk",
           "peak_rss_bytes": peak_rss_bytes(), "machine": machine_info(),
           "threads_record": thread_record()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="sx-stackoverflow")
    ap.add_argument("--block-bits", default="4096,65536")
    ap.add_argument("--threads", default="1,6,12,18")
    ap.add_argument("--schedules", default="static,dynamic")
    ap.add_argument("--pairs", default=None,
                    help="explicit threads:chunks pairs, static schedule (v24), e.g. 6:6,6:12,18:18")
    ap.add_argument("--layout", default="tiled")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--array-mib", type=int, default=1024)
    ap.add_argument("--verify-sources", type=int, default=16)
    ap.add_argument("--out", default="results/real/thread_diagnosis.json")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
