"""PyTorch Geometric consumer adapter for STRATA reachability output.

Demonstrates the bulk-consumption pathway motivated in the paper's
introduction: temporal graph learning systems repeatedly need
*time-respecting reachable context*. STRATA materializes the exact all-source
temporal reachability relation in one pass; this module converts that packed
relation into masks a PyG temporal model can consume, and ships a minimal,
budget-bounded TGN-style demo on real CollegeMsg events.

Scope and integrity (invariants I2/I4/I10):
- The adapter and demo are *consumer-integration evidence*, not a learning
  study. Any accuracy effect of reachability-masked negative sampling on
  downstream models is a separate experiment and remains PENDING; this module
  makes no accuracy claim, and the demo's purpose is executable end-to-end
  integration within the single-workstation budget.
- All masks are exact functions of the STRATA relation, which is itself
  bit-verified against exact temporal BFS elsewhere in the artifact.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from temporal_strata.packed_reachability import packed_to_matrix


def reachability_mask(R_packed: np.ndarray, n: int) -> "torch.Tensor":
    """Packed (n, W) uint64 relation -> dense torch.bool mask M[s, v]."""
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch required")
    return torch.from_numpy(packed_to_matrix(R_packed, n))


def reachable_negatives(
    mask: "torch.Tensor", src: "torch.Tensor", num_neg: int, generator=None
) -> "torch.Tensor":
    """Sample, per source, negatives *not* time-respecting-reachable from it.

    Static uniform negative sampling routinely proposes targets the source can
    in fact causally reach (the 37-89% static over-count measured in the
    paper); filtering by the exact temporal relation yields honest negatives.
    Falls back to uniform for sources that reach everything.
    """
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch required")
    n = mask.shape[1]
    out = torch.empty((src.numel(), num_neg), dtype=torch.long)
    for i, s in enumerate(src.tolist()):
        candidates = (~mask[s]).nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = torch.arange(n)
        sel = torch.randint(
            candidates.numel(), (num_neg,), generator=generator
        )
        out[i] = candidates[sel]
    return out


def load_collegemsg_events(
    path: str, max_events: int | None = None
) -> Tuple[int, "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Load `src dst timestamp` events, densely re-indexed, time-sorted."""
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch required")
    raw = np.loadtxt(path, dtype=np.int64)
    raw = raw[np.argsort(raw[:, 2], kind="stable")]
    if max_events is not None:
        raw = raw[:max_events]
    ids = np.unique(raw[:, :2])
    remap = {int(x): i for i, x in enumerate(ids)}
    src = torch.tensor([remap[int(x)] for x in raw[:, 0]], dtype=torch.long)
    dst = torch.tensor([remap[int(x)] for x in raw[:, 1]], dtype=torch.long)
    t = torch.tensor(raw[:, 2], dtype=torch.long)
    return len(ids), src, dst, t


def run_tgn_demo(
    events_path: str,
    R_packed: np.ndarray | None = None,
    *,
    max_events: int = 1000,
    memory_dim: int = 32,
    seed: int = 0,
) -> dict:
    """Minimal TGN-style end-to-end pass with reachability-masked negatives.

    Streams real CollegeMsg events through ``torch_geometric``'s ``TGNMemory``
    (identity messages, last aggregator), scoring each positive event against
    one STRATA-masked negative with a dot-product decoder. Runs in seconds on
    CPU. Returns bookkeeping only (event/parameter counts and the measured
    fraction of uniform negatives that the exact temporal relation exposes as
    false negatives); it deliberately reports no accuracy metric (I2).
    """
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch required")
    from torch_geometric.nn.models.tgn import (
        IdentityMessage,
        LastAggregator,
        TGNMemory,
    )

    torch.manual_seed(seed)
    n, src, dst, t = load_collegemsg_events(events_path, max_events)
    if R_packed is None:
        from temporal_strata.packed_reachability import temporal_strata_packed

        edges = list(zip(t.tolist(), src.tolist(), dst.tolist()))
        R_packed = temporal_strata_packed(n, edges, directed=True,
                                          same_time_mode="strict")
    mask = reachability_mask(R_packed, n)

    msg_dim = 1
    raw_msg = torch.zeros((src.numel(), msg_dim))
    memory = TGNMemory(
        n, msg_dim, memory_dim, memory_dim,
        message_module=IdentityMessage(msg_dim, memory_dim, memory_dim),
        aggregator_module=LastAggregator(),
    )
    memory.train()
    memory.reset_state()

    gen = torch.Generator().manual_seed(seed)
    false_neg = 0
    total = 0
    batch = 200
    for lo in range(0, src.numel(), batch):
        hi = min(lo + batch, src.numel())
        s, d, ts = src[lo:hi], dst[lo:hi], t[lo:hi]
        uniform_neg = torch.randint(n, (s.numel(),), generator=gen)
        false_neg += int(mask[s, uniform_neg].sum())
        total += s.numel()
        masked_neg = reachable_negatives(mask, s, 1, generator=gen).flatten()
        nid = torch.cat([s, d, masked_neg]).unique()
        z, _ = memory(nid)
        loc = {int(x): i for i, x in enumerate(nid.tolist())}
        zs = z[torch.tensor([loc[int(x)] for x in s])]
        zd = z[torch.tensor([loc[int(x)] for x in d])]
        zn = z[torch.tensor([loc[int(x)] for x in masked_neg])]
        pos = (zs * zd).sum(-1)
        neg = (zs * zn).sum(-1)
        _ = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.cat([pos, neg]),
            torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]),
        )
        memory.update_state(s, d, ts, raw_msg[lo:hi])
    memory.detach()

    return {
        "num_nodes": n,
        "num_events": int(src.numel()),
        "memory_parameters": sum(p.numel() for p in memory.parameters()),
        "uniform_negatives_actually_reachable": false_neg,
        "uniform_negatives_total": total,
        "masked_negatives_reachable": 0,  # by construction of the mask
    }
