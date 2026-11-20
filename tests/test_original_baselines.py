from temporal_strata.original_baselines import ORIGINAL_STRATA_BASELINES, run_method, method_available
from temporal_strata.stream import TemporalEdge
import numpy as np


def test_original_baseline_registry_has_12_methods():
    ids = [s.method_id for s in ORIGINAL_STRATA_BASELINES]
    assert ids == ["BC1", "BC2", "BC3", "BC4", "BC5", "TC1", "TC2", "TC3", "BG1", "BG2", "TG1", "TG2"]


def test_native_temporal_baseline_analogues_match_tc1():
    edges = [
        TemporalEdge(1, 0, 1),
        TemporalEdge(2, 1, 2),
        TemporalEdge(2, 0, 3),
        TemporalEdge(3, 3, 2),
    ]
    ref = run_method("TC1", 4, edges)
    for method_id in ["BC1", "BC2", "BC3", "BC4", "TC2"]:
        got = run_method(method_id, 4, edges)
        assert np.array_equal(ref, got), method_id


def test_optional_methods_report_availability():
    for method_id in ["BC5", "TC3", "BG1", "BG2", "TG1", "TG2"]:
        available, reason = method_available(method_id)
        assert isinstance(available, bool)
        assert isinstance(reason, str) and reason
