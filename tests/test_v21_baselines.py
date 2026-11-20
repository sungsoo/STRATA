"""Baselines and the large-graph harness added for the v20 mock review."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from temporal_strata.parallel import _HAVE_NUMBA  # noqa: E402

needs_numba = pytest.mark.skipif(not _HAVE_NUMBA, reason="numba not installed")


@needs_numba
@pytest.mark.parametrize("mode", ["strict", "closure"])
def test_msbfs_matches_strata(mode):
    from experiments.temporal_msbfs import msbfs_relation
    from temporal_strata.packed_reachability import temporal_strata_packed
    from temporal_strata.parallel import prepare_stream_arrays
    from temporal_strata.stream import arrays_to_edges, generate_temporal_arrays

    for n, m, g, seed in [(70, 600, 40, 1), (130, 3000, 20, 2), (65, 900, 900, 4)]:
        a = generate_temporal_arrays(n, m, g, seed=seed)
        prep = prepare_stream_arrays(n, *a, directed=True, same_time_mode=mode, prep_mode="off")
        ref = temporal_strata_packed(n, arrays_to_edges(*a), directed=True, same_time_mode=mode)
        assert np.array_equal(msbfs_relation(n, prep, mode), ref)


def test_block_ranges_and_sampled_sources():
    from experiments.large_real_blocked import block_ranges, pick_sources

    r, total = block_ranges(10_000, 4096, "2")
    assert total == 3 and r == [(0, 4096), (4096, 4096)]
    r, total = block_ranges(10_000, 4096, "all")
    assert r[-1] == (8192, 1808)
    src = pick_sources([(0, 10), (100, 10)], 5)
    assert len(src) == 5 and all(s < 10 or 100 <= s < 110 for s in src)


@needs_numba
def test_large_real_blocked_end_to_end(tmp_path, monkeypatch):
    """Tiny standard file: every block verified against per-source BFS, pairs equal STRATA."""
    from experiments import bench_common, large_real_blocked
    from temporal_strata.packed_reachability import reachable_pairs, temporal_strata_packed
    from temporal_strata.stream import arrays_to_edges, generate_temporal_arrays

    n = 300
    t, u, v = generate_temporal_arrays(n, 4000, 500, seed=3)
    f = tmp_path / "tiny.txt"
    f.write_text("# meta: {\"dataset\": \"tiny\"}\n" +
                 "".join(f"{a} {b} {c}\n" for a, b, c in zip(u, v, t)))
    monkeypatch.setattr(bench_common, "DATA", tmp_path)
    out = tmp_path / "o.json"
    large_real_blocked.main(["--dataset", "tiny", "--block-bits", "128", "--blocks", "all",
                             "--threads", "1,2", "--verify-sources", "20", "--out", str(out)])
    rec = json.loads(out.read_text())
    assert rec["blocks_total"] == 3 and all(r["verified_sources"] == 20 for r in rec["runs"])
    pairs = sum(b["pairs"] for b in rec["runs"][0]["blocks"])
    ref = temporal_strata_packed(n, arrays_to_edges(t, u, v), directed=True, same_time_mode="strict")
    assert pairs == reachable_pairs(ref, n) and "all_blocks_s" in rec["runs"][0]


def test_index_probe_describe(tmp_path):
    from experiments.temporal_index_baseline import describe

    (tmp_path / "LICENSE").write_text("MIT License\nCopyright ...")
    (tmp_path / "README.md").write_text("# tool\nusage: tool -i graph.txt")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.cpp").write_text("int main(){}")
    (tmp_path / "CMakeLists.txt").write_text("project(x)")
    rec = describe(tmp_path)
    assert rec["license_file"] == "LICENSE" and rec["license_head"].startswith("MIT")
    assert rec["build_files"] == ["CMakeLists.txt"] and rec["sources"] == 1
    assert "usage" in rec["readme_head"]


def test_threads_best_of_an_earlier_record(tmp_path):
    from experiments.large_real_blocked import parse_threads

    rec = tmp_path / "sweep.json"
    rec.write_text(json.dumps({"runs": [{"threads": 1, "kernel_s_mean_block": 9.0},
                                        {"threads": 12, "kernel_s_mean_block": 2.0},
                                        {"threads": 18, "kernel_s_mean_block": 2.5}]}))
    assert parse_threads(f"best:{rec}") == [12]
    assert parse_threads("1,6") == [1, 6]


def test_large_real_blocked_tiles_state_matches_rows(tmp_path, monkeypatch):
    """v28: --state tiles verifies the same sources and gives the same pair counts."""
    from experiments import bench_common, large_real_blocked
    from temporal_strata.stream import generate_temporal_arrays

    n = 300
    t, u, v = generate_temporal_arrays(n, 4000, 500, seed=4)
    f = tmp_path / "tiny.txt"
    f.write_text("# meta: {\"dataset\": \"tiny\"}\n" +
                 "".join(f"{a} {b} {c}\n" for a, b, c in zip(u, v, t)))
    monkeypatch.setattr(bench_common, "DATA", tmp_path)
    recs = {}
    for state in ("rows", "tiles"):
        out = tmp_path / f"{state}.json"
        large_real_blocked.main(["--dataset", "tiny", "--block-bits", "128", "--blocks", "all",
                                 "--threads", "1,2", "--verify-sources", "20", "--state", state,
                                 "--out", str(out)])
        recs[state] = json.loads(out.read_text())
    for a, b in zip(recs["rows"]["runs"], recs["tiles"]["runs"]):
        assert [x["pairs"] for x in a["blocks"]] == [x["pairs"] for x in b["blocks"]]
        assert b["state"] == "tiles" and b["verified_sources"] == 20


def test_large_real_blocked_tiles_default_and_dynamic_schedule(tmp_path, monkeypatch):
    """v30: tiles is the default state; the dynamic schedule changes only the chunk
    count (4p, at most W) and gives the same pair counts as static."""
    from experiments import bench_common, large_real_blocked
    from temporal_strata.stream import generate_temporal_arrays

    assert large_real_blocked.tile_chunks(4096, 6, "static") == 6
    assert large_real_blocked.tile_chunks(4096, 6, "dynamic") == 24
    assert large_real_blocked.tile_chunks(128, 6, "dynamic") == 2      # capped at W
    n = 300
    t, u, v = generate_temporal_arrays(n, 4000, 500, seed=5)
    f = tmp_path / "tiny.txt"
    f.write_text("# meta: {\"dataset\": \"tiny\"}\n" +
                 "".join(f"{a} {b} {c}\n" for a, b, c in zip(u, v, t)))
    monkeypatch.setattr(bench_common, "DATA", tmp_path)
    recs = {}
    for sched in ("static", "dynamic"):
        out = tmp_path / f"{sched}.json"
        large_real_blocked.main(["--dataset", "tiny", "--block-bits", "256", "--blocks", "all",
                                 "--threads", "1,2", "--verify-sources", "20",
                                 "--schedule", sched, "--out", str(out)])
        recs[sched] = json.loads(out.read_text())
    for a, b in zip(recs["static"]["runs"], recs["dynamic"]["runs"]):
        assert a["state"] == b["state"] == "tiles"
        assert [x["pairs"] for x in a["blocks"]] == [x["pairs"] for x in b["blocks"]]
    assert [r["chunks"] for r in recs["dynamic"]["runs"]] == [4, 4]   # W = 4 caps 4p


def test_large_real_blocked_default_schedule_is_static():
    """v31: the block runner defaults to the tile state with the static schedule."""
    from experiments import large_real_blocked
    seen = {}
    orig = large_real_blocked.run
    try:
        large_real_blocked.run = lambda a: seen.update(vars(a))
        large_real_blocked.main(["--dataset", "x"])
    finally:
        large_real_blocked.run = orig
    assert seen["schedule"] == "static" and seen["state"] == "tiles"


def test_block_bits_for_memory():
    """v32: --memory-gb picks the widest power-of-two block whose state fits."""
    from experiments.large_real_blocked import block_bits_for_memory
    n = 2_601_977                                   # sx-stackoverflow
    assert block_bits_for_memory(n, 11.0) == 32768  # 10.66 GB state
    assert block_bits_for_memory(n, 25.0) == 65536  # 21.32 GB
    assert block_bits_for_memory(n, 1.5) == 4096    # 1.33 GB
    assert block_bits_for_memory(300, 100.0) == 256  # power of two within n rounded up
    import pytest
    with pytest.raises(SystemExit):
        block_bits_for_memory(n, 0.001)


def test_window_large_on_a_small_stream(tmp_path, monkeypatch):
    """v34: windows verify sampled sources; insertion equals recomputation."""
    from experiments import bench_common, window_large
    from temporal_strata.stream import generate_temporal_arrays
    n = 300
    t, u, v = generate_temporal_arrays(n, 4000, 300, seed=6)
    f = tmp_path / "tiny.txt"
    f.write_text("# meta: {\"dataset\": \"tiny\"}\n" +
                 "".join(f"{a} {b} {c}\n" for a, b, c in zip(u, v, t)))
    monkeypatch.setattr(bench_common, "DATA", tmp_path)
    out = tmp_path / "w.json"
    window_large.main(["--dataset", "tiny", "--threads", "2", "--slots", "3", "--batches", "3",
                       "--out", str(out)])
    rec = json.loads(out.read_text())
    assert len(rec["windows"]) == 3 and rec["insertion"]["bit_identical"]


def test_window_insert_layouts_agree(tmp_path, monkeypatch):
    """v36: tiled and row-major insertion batches both equal the recomputation."""
    from experiments import bench_common, window_large
    from temporal_strata.stream import generate_temporal_arrays
    n = 300
    t, u, v = generate_temporal_arrays(n, 4000, 300, seed=8)
    f = tmp_path / "tiny.txt"
    f.write_text("# meta: {\"dataset\": \"tiny\"}\n" +
                 "".join(f"{a} {b} {c}\n" for a, b, c in zip(u, v, t)))
    monkeypatch.setattr(bench_common, "DATA", tmp_path)
    out = tmp_path / "wi.json"
    window_large.main(["--dataset", "tiny", "--threads", "2", "--slots", "0", "--batches", "3",
                       "--insert-layouts", "tiled,rowmajor", "--out", str(out)])
    rec = json.loads(out.read_text())
    assert rec["windows"] == [] and set(rec["insertion"]["layouts"]) == {"tiled", "rowmajor"}
    assert rec["insertion"]["bit_identical"]
