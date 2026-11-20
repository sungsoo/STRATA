"""STRATA-v3 measurement tooling: machine-independent checks."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from temporal_strata.packed_reachability import temporal_strata_packed, words_for  # noqa: E402
from temporal_strata.stream import generate_temporal_stream  # noqa: E402

numba = pytest.importorskip("numba")
from temporal_strata.parallel import prepare_stream, resolve_plan  # noqa: E402


# ---------------- chunk policy ----------------------------------------- #
@pytest.mark.parametrize("W,p,expect", [(30, 18, 18), (92, 18, 18), (125, 18, 18), (30, 6, 6), (4, 18, 4)])
def test_threads_policy_restores_v1_protocol(W, p, expect):
    """With 128-byte lines (L=16) the harness protocol still gives min(p, W) chunks."""
    plan = resolve_plan(W, p, layout="rowmajor", line_words=16, chunk_policy="threads")
    assert plan.nchunks == expect and plan.chunk_policy == "threads"


def test_auto_policy_keeps_library_cap_and_dynamic():
    assert resolve_plan(30, 18, layout="rowmajor", line_words=16).nchunks == 1
    assert resolve_plan(92, 18, layout="rowmajor", line_words=16).nchunks == 5
    assert resolve_plan(92, 18, layout="rowmajor", schedule="dynamic", line_words=16,
                        chunk_policy="threads").nchunks == 72
    with pytest.raises(ValueError):
        resolve_plan(30, 2, chunk_policy="bogus")


def test_plan_only_reports_no_idle_workers_under_threads_policy(capsys):
    sys.path.insert(0, os.path.join(HERE, "..", "experiments"))
    from experiments.parallel_scaling import print_plans

    ds = [("CollegeMsg", 1899, None), ("BitcoinOTC", 5881, None)]
    os.environ["STRATA_CACHELINE_BYTES"] = "128"
    from temporal_strata import parallel as par
    par.cache_line_bytes.cache_clear()
    try:
        plans = print_plans(ds, [1, 6, 18], [("rowmajor", "static"), ("tiled", "dynamic")],
                            4, "threads", None, {"summary": "test"})
        assert all(p["idle_workers"] == 0 for p in plans)
        bad = print_plans(ds, [18], [("rowmajor", "static")], 4, "auto", None, {"summary": "t"})
        assert [p["idle_workers"] for p in bad] == [17, 13]
    finally:
        del os.environ["STRATA_CACHELINE_BYTES"]
        par.cache_line_bytes.cache_clear()


# ---------------- datasets, references, prep timings ------------------ #
def test_parse_large_and_lazy_selection():
    from experiments.bench_common import parse_large, select_datasets

    label, n, edges = parse_large("500:2000:20:3")
    assert n == 500 and len(edges) == 2000 and "seed=3" in label
    label2, n2, e2 = parse_large("100000:2000000:400", materialize=False)
    assert n2 == 100000 and e2 is None
    with pytest.raises(ValueError):
        parse_large("10:20")
    ds = select_datasets([], ["300:900:10"], include_default_synth=False)
    assert len(ds) == 1 and ds[0][1] == 300


def test_compiled_reference_matches_interpreted():
    from experiments.bench_common import choose_reference, reference_state

    n, m = 260, 1800
    edges = generate_temporal_stream(n, m, 40, directed=True, seed=13)
    for mode in ("strict", "closure"):
        Ri, ki = reference_state(n, edges, mode, "interpreted")
        Rc, kc = reference_state(n, edges, mode, "compiled")
        assert (ki, kc) == ("interpreted", "compiled")
        assert np.array_equal(Ri, Rc)
    assert choose_reference(1000, 10, "auto") == "interpreted"
    assert choose_reference(10**6, 10**7, "auto") == "compiled"


def test_prep_records_phase_timings():
    prep = prepare_stream(200, generate_temporal_stream(200, 900, 30, seed=2))
    assert set(prep.timings) == {"sort_validate_s", "flatten_s", "group_boundaries_s",
                                 "snapshot_plan_s", "total_s"}
    assert prep.timings["total_s"] >= 0


# ---------------- compiled per-source BFS ------------------------------ #
def test_compiled_real_kernels_bit_identical():
    from experiments.compiled_real import per_source_strict_bfs_grouped, strata_compiled_single

    n, m = 300, 2500
    edges = generate_temporal_stream(n, m, 60, directed=True, seed=19)
    ref = temporal_strata_packed(n, edges, directed=True, same_time_mode="strict")
    prep = prepare_stream(n, edges)
    assert np.array_equal(strata_compiled_single(prep, n), ref)
    assert np.array_equal(per_source_strict_bfs_grouped(n, words_for(n), prep.gptr,
                                                        prep.us, prep.vs), ref)


# ---------------- external-baseline harness --------------------------- #
def _cmd(mode):
    tool = os.path.join(HERE, "tools", "ext_reach_tool.py")
    return f"{sys.executable} {tool} {mode} {{edges}} {{sources}} {{out}}"


def test_external_command_adapter_matching_tool_is_timed():
    from experiments.external_baselines import CommandAdapter, run_dataset

    n, m = 120, 700
    edges = generate_temporal_stream(n, m, 30, directed=True, seed=4)
    row = run_dataset(CommandAdapter(_cmd("strict"), "test tool"), "toy", n, edges,
                      "all", repeats=1, seed=1, check_only=False)
    assert row["matched_semantics"] == "strict" and row["timed"] is True


def test_external_command_adapter_wrong_relation_is_refused():
    from experiments.external_baselines import CommandAdapter, run_dataset

    n, m = 120, 700
    edges = generate_temporal_stream(n, m, 30, directed=True, seed=4)
    row = run_dataset(CommandAdapter(_cmd("static"), "static tool"), "toy", n, edges,
                      "20", repeats=1, seed=1, check_only=False)
    assert row["matched_semantics"] is None and row["timed"] is False
    assert row["semantics_check"]["strict"]["mismatched_sources"] > 0


def test_external_adapter_validation():
    from experiments.external_baselines import AdapterError, CommandAdapter, RaphtoryAdapter

    with pytest.raises(AdapterError):
        CommandAdapter("tool-without-placeholders")
    try:
        import raphtory  # noqa: F401
    except Exception:
        with pytest.raises(AdapterError):
            RaphtoryAdapter()


# ---------------- diagnosis mode ------------------------------ #
def test_external_diagnosis_reports_sizes_and_toys():
    from experiments.external_baselines import CommandAdapter, TOY_STREAMS, diagnose_toys, run_dataset

    n, m = 120, 700
    edges = generate_temporal_stream(n, m, 30, directed=True, seed=4)
    static = CommandAdapter(_cmd("static"), "static tool")
    row = run_dataset(static, "toy", n, edges, "10", repeats=1, seed=1, check_only=True,
                      diagnose=True)
    d = row["diagnosis"]
    assert len(d) == 10 and {"external_size", "strict_size", "closure_size"} <= set(d[0])
    assert any(x["only_external_vs_strict"]["count"] > 0 for x in d)  # static over-reaches
    toys = diagnose_toys(CommandAdapter(_cmd("strict"), "strict tool"))
    assert len(toys) == len(TOY_STREAMS) and all(t["matches_strict"] for t in toys)
    toys_static = diagnose_toys(static)
    assert not next(t for t in toys_static if t["toy"] == "time_reversed")["matches_strict"]


def test_toy_expectations_match_strata():
    from experiments.external_baselines import TOY_STREAMS
    from temporal_strata.packed_reachability import packed_to_matrix

    for name, n, raw, src, strict, closure in TOY_STREAMS:
        for mode, want in (("strict", strict), ("closure", closure)):
            M = packed_to_matrix(temporal_strata_packed(n, raw, same_time_mode=mode), n)
            assert set(np.flatnonzero(M[src]).tolist()) == want, (name, mode)


# ---------------- Raphtory result interpretation ------------------ #
class _FakeNode:
    def __init__(self, name):
        self.name = name


class _FakeOutputNodeState:
    """Mirrors the Raphtory 0.17 result shape recorded on the M5 (raw_sample)."""

    def __init__(self, reach):
        self._reach = reach

    def items(self):
        return [(_FakeNode(v), {"reachable_nodes": recs}) for v, recs in self._reach.items()]


def test_raphtory_reached_uses_reachable_nodes_list():
    from experiments.external_baselines import AdapterError, RaphtoryAdapter

    res = _FakeOutputNodeState({0: [], 1: [], 2: [{"0": 5, "1": "3"}],
                                3: [{"0": 1, "1": "start"}], 4: []})
    assert RaphtoryAdapter._reached(res, 3) == {2, 3}   # empty lists are NOT reached
    with pytest.raises(AdapterError):
        RaphtoryAdapter._reached(_FakeOutputNodeState({0: None}), 0)

    class Odd:
        def items(self):
            return [(_FakeNode(0), {"other": []})]
    with pytest.raises(AdapterError):
        RaphtoryAdapter._reached(Odd(), 0)


def test_suspicious_all_reached_guard():
    from experiments.external_baselines import suspicious_all_reached

    exact = {"strict": {0: {0}, 1: {0, 1}}, "closure": {0: {0}, 1: {0, 1}}}
    assert suspicious_all_reached({0: {0, 1, 2}, 1: {0, 1, 2}}, exact, 3) is True
    assert suspicious_all_reached({0: {0}, 1: {0, 1}}, exact, 3) is False
    full = {"strict": {0: {0, 1, 2}}, "closure": {0: {0, 1, 2}}}
    assert suspicious_all_reached({0: {0, 1, 2}}, full, 3) is False
