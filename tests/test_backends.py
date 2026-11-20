"""Backend and consumer tests (TPDS v2 loop).

Bit-identity of the PyTorch tensor backend against the sequential core is a
correctness gate (invariants I3/I10): every semantics x directedness x shape
below must produce the *identical packed uint64 matrix*, including unaligned
n (tail-bit masking) and adversarial same-timestamp groups. PyG consumer
tests verify the adapter is exact and the TGN demo runs end-to-end on real
CollegeMsg events within budget. All tests skip-with-reason when the optional
dependency is absent (requirements-optional.txt).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from temporal_strata.packed_reachability import (  # noqa: E402
    packed_to_matrix,
    temporal_strata_packed,
)

torch = pytest.importorskip(
    "torch", reason="PyTorch not installed (optional backend; see requirements-optional.txt)"
)

from backends.torch_backend import (  # noqa: E402
    packed_torch_to_numpy,
    resolve_device,
    temporal_strata_torch,
)

DATA = os.path.join(os.path.dirname(__file__), "..",
                    "datasets_real_temporal", "standard", "CollegeMsg.txt")


def _rand_stream(n, m, tmax, seed):
    rng = np.random.default_rng(seed)
    return list(zip(rng.integers(0, tmax, m).tolist(),
                    rng.integers(0, n, m).tolist(),
                    rng.integers(0, n, m).tolist()))


@pytest.mark.parametrize("mode", ["strict", "closure"])
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("n,seed", [(70, 1), (128, 2), (257, 3)])
def test_torch_backend_bit_identity_random(mode, directed, n, seed):
    """Unaligned and aligned n; dense same-time collisions stress groups."""
    edges = _rand_stream(n, 6 * n, max(3, n // 8), seed)
    ref = temporal_strata_packed(n, edges, directed=directed, same_time_mode=mode)
    out = packed_torch_to_numpy(
        temporal_strata_torch(n, edges, directed=directed,
                              same_time_mode=mode, device="cpu"))
    assert out.dtype == np.uint64 and out.shape == ref.shape
    assert np.array_equal(out, ref)  # bit-for-bit, tail words included


def test_torch_backend_same_tick_chain_semantics():
    """0->1->2->3 all at t=0: strict forbids chaining, closure completes it."""
    n, edges = 4, [(0, 0, 1), (0, 1, 2), (0, 2, 3)]
    for mode in ("strict", "closure"):
        ref = temporal_strata_packed(n, edges, same_time_mode=mode)
        out = packed_torch_to_numpy(
            temporal_strata_torch(n, edges, same_time_mode=mode))
        assert np.array_equal(out, ref)
    strict_pairs = int(packed_to_matrix(
        temporal_strata_packed(n, edges, same_time_mode="strict"), n).sum())
    closure_pairs = int(packed_to_matrix(
        temporal_strata_packed(n, edges, same_time_mode="closure"), n).sum())
    assert (strict_pairs, closure_pairs) == (7, 10)  # matches the paper


def test_torch_backend_collegemsg_anchor():
    """Real timestamps: identical packed matrix and the published pair count."""
    raw = np.loadtxt(DATA, dtype=np.int64)
    ids = np.unique(raw[:, :2])
    remap = {int(x): i for i, x in enumerate(ids)}
    edges = [(int(t), remap[int(u)], remap[int(v)]) for u, v, t in raw]
    n = len(ids)
    ref = temporal_strata_packed(n, edges, same_time_mode="strict")
    out = packed_torch_to_numpy(temporal_strata_torch(n, edges,
                                                      same_time_mode="strict"))
    assert np.array_equal(out, ref)
    assert int(packed_to_matrix(ref, n).sum()) == 1_794_244


def test_resolve_device_falls_back_loudly():
    import backends.torch_backend as tb
    dev = resolve_device("cuda") if not torch.cuda.is_available() else resolve_device("cpu")
    assert dev.type == "cpu"
    if not torch.cuda.is_available():
        assert tb.last_fallback_reason and "cuda" in tb.last_fallback_reason


# ---------------------------- PyG consumer ---------------------------------

pyg = pytest.importorskip(
    "torch_geometric",
    reason="PyTorch Geometric not installed (optional consumer; see requirements-optional.txt)",
)

from consumers.pyg_reachability import (  # noqa: E402
    reachability_mask,
    reachable_negatives,
    run_tgn_demo,
)


def test_reachability_mask_exact():
    n, edges = 5, [(1, 0, 1), (2, 1, 2), (3, 2, 3)]
    R = temporal_strata_packed(n, edges, same_time_mode="strict")
    mask = reachability_mask(R, n)
    dense = packed_to_matrix(R, n)
    assert mask.shape == (n, n) and mask.numpy().tolist() == dense.tolist()
    assert bool(mask[0, 3]) and not bool(mask[3, 0])


def test_reachable_negatives_never_reachable():
    n, edges = 40, _rand_stream(40, 200, 20, 7)
    R = temporal_strata_packed(n, edges, same_time_mode="strict")
    mask = reachability_mask(R, n)
    src = torch.arange(n)
    neg = reachable_negatives(mask, src, 3,
                              generator=torch.Generator().manual_seed(0))
    picked = mask[src.repeat_interleave(3), neg.flatten()]
    full_rows = mask.all(dim=1)[src.repeat_interleave(3)]  # uniform fallback rows
    assert not bool((picked & ~full_rows).any())


def test_tgn_demo_end_to_end_budget():
    stats = run_tgn_demo(DATA, max_events=600, memory_dim=16, seed=0)
    assert stats["num_events"] == 600 and stats["memory_parameters"] > 0
    assert stats["masked_negatives_reachable"] == 0
    assert 0 <= stats["uniform_negatives_actually_reachable"] <= stats[
        "uniform_negatives_total"]
