#!/usr/bin/env python3
"""Source-blocked STRATA on a real temporal graph with more than 10^7 edges (v21, R54-3).

v22: wider blocks (beta = 65536, 1,024 words, about 21 GB resident) give every
thread enough columns; running k narrow blocks side by side is the same column
partition (Theorem 3), so the width sweep covers source-block parallelism too.

Why: the largest observed-timestamp graph in the v20 paper has 1.4 * 10^6 edges;
reviewers asked for a real stream of 10^7 edges or more. SNAP sx-stackoverflow
(n = 2,601,977, m = 63,497,050) fits in memory one source block at a time: with
block width beta = 4096 a block holds n * 64 words (about 1.3 GB), and the full
relation needs ceil(n / beta) = 636 blocks. Its n^2 bits (about 0.85 TB) are never
stored; each block reports its reachable-pair count.

How: the stream is loaded and preprocessed once (strict, touched-row plan); each
block runs the tiled column-parallel kernel from an identity block; before any
time is kept, sampled sources of the processed blocks are compared bit for bit
with an independent per-source temporal BFS. ``--blocks`` runs the first k
blocks (or ``all``); the record states how many blocks ran and, separately, the
mean block time times the block count, labeled as an extrapolation when not all
blocks ran.

v28: ``--state tiles`` runs each block with the tiles as the state
(temporal_strata.parallel.run_block_tiles): no shared n x W block is gathered into
tiles and scattered back, which the v27 phase measurement found to grow from
3.3 s to 32.6 s between 1 and 18 threads. The block's identity bits are set inside
the timed kernel in this mode (they were set before the clock in ``rows`` mode),
sampled columns come straight from the tiles, and the pair count from the tiles.

v30: ``tiles`` is the default state (the M5 run of every sx-stackoverflow block took
257 s at 18 threads with tiles against 709 s at 6 with rows, same pairs);
``--state rows`` keeps the shared-state path. In ``tiles`` mode ``--schedule`` now
sets the chunk count as in rows mode (static: p chunks; dynamic: 4p chunks, at most
W words), and each run records its chunk count.

v31: ``static`` is the default schedule. On sx-stackoverflow the dynamic schedule
slowed the tile state 1.6x (beta = 65,536) and 2.1x (beta = 4,096) at the best p, so
the v30 defaults (tiles + dynamic) were the slowest combination measured.

v32: ``--memory-gb G`` picks the block width: the largest power of two (at least 64,
at most n rounded up to 64) whose n x ceil(beta/64) x 8-byte block state fits in G
gigabytes. On sx-stackoverflow the cost per source fell 3.5x from beta = 4,096 to
65,536 with the tile state, so the widest block that fits is the choice; the thread
count stays a user choice (the best p did not follow words per thread alone).

What:
    python3 experiments/fetch_real_temporal.py --names sx-stackoverflow
    NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py \
        --dataset sx-stackoverflow --block-bits 4096 --blocks 8 --threads 1,18 \
        --verify-sources 64 --out results/real/large_real_blocked.json
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def block_ranges(n, block_bits, blocks):
    """[(start, width)] for the first ``blocks`` source blocks ('all' = every block)."""
    total = -(-n // block_bits)
    k = total if blocks == "all" else min(int(blocks), total)
    return [(b * block_bits, min(block_bits, n - b * block_bits)) for b in range(k)], total


def pick_sources(ranges, k, seed=5):
    """``k`` sampled global source ids spread over the processed blocks."""
    rng = np.random.default_rng(seed)
    pool = np.concatenate([np.arange(s, s + w) for s, w in ranges])
    return np.sort(rng.choice(pool, size=min(k, pool.size), replace=False)).astype(np.int64)


def column_of(Rb, start, s):
    j = s - start
    return ((Rb[:, j // 64] >> np.uint64(j % 64)) & np.uint64(1)).astype(np.bool_)


def parse_threads(spec):
    """'1,6,18' or 'best:FILE' (the thread count with the lowest mean block time in an
    earlier record, v22 / R55: the full run uses the best count of the sweep)."""
    if spec.startswith("best:"):
        rec = json.loads(open(spec[5:]).read())
        return [min(rec["runs"], key=lambda r: r["kernel_s_mean_block"])["threads"]]
    return [int(x) for x in spec.split(",")]


def peak_rss_bytes():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)   # bytes on macOS, KiB on Linux



def tile_chunks(width_bits, threads, schedule):
    """Chunk count of the tile-state kernel: p (static) or 4p (dynamic), at most W."""
    from temporal_strata.parallel import DEFAULT_OVERSUB
    W = (int(width_bits) + 63) // 64
    base = int(threads) if schedule == "static" else DEFAULT_OVERSUB * int(threads)
    return max(1, min(base, W))

def block_bits_for_memory(n, memory_gb):
    """Largest power-of-two block width whose block state (n x ceil(beta/64) x 8
    bytes) fits in ``memory_gb`` gigabytes; at least 64, at most n rounded up to 64."""
    cap = max(64, -(-int(n) // 64) * 64)
    beta = 64
    while beta * 2 <= cap and int(n) * (beta * 2 // 64) * 8 <= float(memory_gb) * 1e9:
        beta *= 2
    if int(n) * (beta // 64) * 8 > float(memory_gb) * 1e9:
        raise SystemExit(f"--memory-gb {memory_gb}: even a 64-bit block needs "
                         f"{int(n) * 8 / 1e9:.2f} GB")
    return min(beta, cap)


def run(args):
    from experiments.bench_common import DATA, _bfs_sources_strict
    from experiments.thread_record import thread_record
    from temporal_strata.io import load_standard_arrays
    from temporal_strata.parallel import (_empty_state, _set_identity_vec, machine_info,
                                          prepare_stream_arrays, resolve_plan, run_prepared,
                                          set_threads)

    path = DATA / f"{args.dataset}.txt"
    if not path.exists():
        raise SystemExit(f"missing {path}; run experiments/fetch_real_temporal.py --names {args.dataset}")
    t0 = time.perf_counter()
    n, ts, us, vs, meta = load_standard_arrays(path)
    load_s = time.perf_counter() - t0
    threads = parse_threads(args.threads)
    set_threads(max(threads))
    prep = prepare_stream_arrays(n, ts, us, vs, directed=True, same_time_mode="strict")
    del ts, us, vs
    if args.memory_gb is not None:
        args.block_bits = block_bits_for_memory(n, args.memory_gb)
        print(f"--memory-gb {args.memory_gb}: block width {args.block_bits}")
    ranges, total = block_ranges(n, args.block_bits, args.blocks)
    src = pick_sources(ranges, args.verify_sources)
    print(f"{args.dataset}: n={n:,} m={prep.num_edges:,} groups={prep.gptr.shape[0]-1:,} "
          f"prep {prep.timings['total_s']:.2f} s ({prep.prep_sort}); blocks {len(ranges)}/{total}")

    bfs = _bfs_sources_strict(n, prep.gptr, prep.us, prep.vs, src)   # independent reference
    runs = []
    for p in threads:
        eff = set_threads(p)
        blocks, verified = [], 0
        for start, width in ranges:
            if args.state == "tiles":
                from temporal_strata.parallel import run_block_tiles
                inb = [i for i, s in enumerate(src) if start <= s < start + width]
                t1 = time.perf_counter()
                pairs, cols, _ = run_block_tiles(prep, start, width, tile_chunks(width, eff, args.schedule),
                                                 sample_local=[int(src[i]) - start for i in inb])
                dt = time.perf_counter() - t1
                for k, i in enumerate(inb):
                    if not np.array_equal(cols[k], bfs[i]):
                        raise SystemExit(f"block at {start}: source {src[i]} differs from per-source BFS")
                    verified += 1
                blocks.append({"start": start, "width": width, "kernel_s": dt, "pairs": pairs})
                print(f"  p={eff} block {start // args.block_bits}: {dt:.2f} s, {pairs:,} pairs (tiles)")
                continue
            Rb = _empty_state(n, width)
            _set_identity_vec(Rb, start, width)
            plan = resolve_plan(Rb.shape[1], eff, layout=args.layout, schedule=args.schedule)
            t1 = time.perf_counter()
            run_prepared(Rb, prep, plan)
            dt = time.perf_counter() - t1
            for i, s in enumerate(src):
                if start <= s < start + width:
                    if not np.array_equal(column_of(Rb, start, int(s)), bfs[i]):
                        raise SystemExit(f"block at {start}: source {s} differs from per-source BFS")
                    verified += 1
            pairs = int(np.bitwise_count(Rb).sum())
            blocks.append({"start": start, "width": width, "kernel_s": dt, "pairs": pairs})
            print(f"  p={eff} block {start // args.block_bits}: {dt:.2f} s, {pairs:,} pairs")
            del Rb
        mean = sum(b["kernel_s"] for b in blocks) / len(blocks)
        runs.append({"threads": int(eff), "layout": args.layout, "schedule": args.schedule,
                     "state": args.state,
                     "chunks": (tile_chunks(args.block_bits, eff, args.schedule)
                                if args.state == "tiles" else None),
                     "blocks_run": len(blocks), "verified_sources": verified,
                     "kernel_s_total": sum(b["kernel_s"] for b in blocks),
                     "kernel_s_mean_block": mean,
                     "all_blocks_s" if len(blocks) == total else "extrapolated_all_blocks_s":
                         mean * total,
                     "blocks": blocks})
    if any(r["verified_sources"] != len(src) for r in runs):
        raise SystemExit("not every sampled source was verified")
    out = {"dataset": args.dataset, "n": int(n), "m": prep.num_edges, "meta": meta,
           "groups": int(prep.gptr.shape[0] - 1), "block_bits": args.block_bits,
           "blocks_total": total, "load_s": load_s, "prep": prep.timings,
           "prep_sort": prep.prep_sort, "prep_parallel": prep.prep_parallel,
           "resident_block_bytes": int(n * -(-args.block_bits // 64) * 8),
           "relation_bits": int(n) * int(n),
           "verified_sources": [int(s) for s in src], "reference": "per-source strict BFS",
           "peak_rss_bytes": peak_rss_bytes(), "machine": machine_info(),
           "threads_record": thread_record(), "runs": runs}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="sx-stackoverflow")
    ap.add_argument("--block-bits", type=int, default=4096)
    ap.add_argument("--memory-gb", type=float, default=None,
                    help="pick the widest power-of-two block whose state fits (v32)")
    ap.add_argument("--blocks", default="8", help="number of leading source blocks, or 'all'")
    ap.add_argument("--threads", default="1,18", help="comma list, or best:FILE")
    ap.add_argument("--layout", default="tiled")
    ap.add_argument("--schedule", default="static", choices=("static", "dynamic"),
                    help="static (default since v31): p chunks; dynamic: 4p chunks, at most W")
    ap.add_argument("--verify-sources", type=int, default=64)
    ap.add_argument("--state", default="tiles", choices=("rows", "tiles"),
                    help="tiles: the tiles are the state, no copies (default since v30); "
                         "rows: shared n x W block copied into tiles")
    ap.add_argument("--out", default="results/real/large_real_blocked.json")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
