"""PyTorch tensor backend for STRATA packed temporal reachability.

Reimplements the column kernel of ``temporal_strata.packed_reachability`` with
``torch.int64`` tensor bitwise ops (AND-NOT / OR), so the same causal-frontier
algebra can run on CPU, Apple MPS, or CUDA devices **without any semantic
change**. The bit layout is identical to the core: bit ``s`` (word ``s//64``,
bit ``s%64``) of row ``v`` is set iff source ``s`` reaches target ``v``.

Integrity rules (project invariants I2/I3/I6):
- This backend is a *correctness-verified alternative realization*: it must be
  bit-identical to the sequential Numba/NumPy core for every semantics,
  directedness, and device before any timing is reported
  (see ``tests/test_backends.py`` and ``experiments/backend_bench.py``).
- Devices that do not support int64 bitwise ops fall back to CPU with an
  explicit recorded reason (``last_fallback_reason``); no silent degradation.
- CUDA paths are code-complete but unmeasured in this repository's build
  environment; all accelerator numbers are PENDING until produced by the
  shipped harness on a disclosed machine.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

try:  # torch is an optional dependency (requirements-optional.txt)
    import torch
except Exception as _e:  # pragma: no cover - exercised via skip-with-reason
    torch = None
    _TORCH_IMPORT_ERROR = _e
else:
    _TORCH_IMPORT_ERROR = None

__all__ = [
    "torch_available",
    "resolve_device",
    "last_fallback_reason",
    "temporal_strata_torch",
    "packed_torch_to_numpy",
]

# Populated by resolve_device() whenever a requested device is refused.
last_fallback_reason: str | None = None


def torch_available() -> bool:
    return torch is not None


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise ImportError(
            f"PyTorch is required for the tensor backend: {_TORCH_IMPORT_ERROR}"
        )


def resolve_device(device: str = "cpu") -> "torch.device":
    """Resolve a device string, probing int64 bitwise support.

    Falls back to CPU (recording ``last_fallback_reason``) when the requested
    device is unavailable or cannot execute the int64 bitwise ops the kernel
    needs. This makes the backend safe to call with ``device='mps'`` on any
    machine: on the Apple M5 Max it runs on MPS iff the ops are supported, elsewhere it
    degrades loudly, never silently.
    """
    global last_fallback_reason
    _require_torch()
    last_fallback_reason = None
    want = torch.device(device)
    if want.type == "cpu":
        return want
    try:
        if want.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available on this host")
        if want.type == "mps" and not (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS not available on this host")
        probe = torch.tensor([3, 5], dtype=torch.int64, device=want)
        one = torch.tensor(1, dtype=torch.int64, device=want)
        _ = torch.bitwise_or(
            torch.bitwise_and(probe, torch.bitwise_not(probe)),
            torch.bitwise_left_shift(one, torch.tensor(63, device=want)),
        )
        return want
    except Exception as e:  # capability probe failed -> loud CPU fallback
        last_fallback_reason = f"{want}: {type(e).__name__}: {e}"
        return torch.device("cpu")


def _sorted_groups(
    edges: Iterable[Tuple[int, int, int]]
) -> List[Tuple[int, List[Tuple[int, int]]]]:
    """Sort contacts by timestamp and bucket into maximal timestamp groups.

    Tuples are read as ``(t, u, v)`` exactly as in the core. Intra-group order
    is irrelevant by construction (strict reads only the pre-group snapshot;
    closure iterates to a fixed point), which is the commutativity the paper's
    Lemma 3 proves.
    """
    es = [(int(t), int(u), int(v)) for (t, u, v) in edges]
    es.sort(key=lambda e: e[0])
    groups: List[Tuple[int, List[Tuple[int, int]]]] = []
    for t, u, v in es:
        if not groups or groups[-1][0] != t:
            groups.append((t, []))
        groups[-1][1].append((u, v))
    return groups


def temporal_strata_torch(
    n: int,
    edges: Iterable[Tuple[int, int, int]],
    *,
    directed: bool = True,
    same_time_mode: str = "strict",
    device: str = "cpu",
) -> "torch.Tensor":
    """All-source temporal reachability on a torch device.

    Returns an ``(n, W)`` int64 tensor whose bit pattern is identical to the
    uint64 matrix produced by ``temporal_strata_packed`` (verified in
    ``tests/test_backends.py``). ``W = ceil(n/64)``; tail bits beyond ``n`` in
    the last word are provably zero: the identity initialization sets only
    valid bits and OR-composition of zero tails keeps them zero, so packed
    equality across backends is well defined.
    """
    _require_torch()
    if same_time_mode not in {"strict", "closure"}:
        raise ValueError("same_time_mode must be 'strict' or 'closure'")
    dev = resolve_device(device)
    W = (n + 63) // 64
    R = torch.zeros((n, W), dtype=torch.int64, device=dev)
    one = torch.tensor(1, dtype=torch.int64, device=dev)
    idx = torch.arange(n, device=dev)
    R[idx, idx // 64] = torch.bitwise_left_shift(one, (idx % 64).to(torch.int64))

    for _t, contacts in _sorted_groups(edges):
        if same_time_mode == "strict":
            # Rows written in this group; every read of a written row must see
            # its pre-group value (Eq. strict / Lemma 1).
            touched = set()
            for u, v in contacts:
                touched.add(v)
                if not directed:
                    touched.add(u)
            snap = {x: R[x].clone() for x in touched}

            def pre(x: int) -> "torch.Tensor":
                return snap[x] if x in snap else R[x]

            for u, v in contacts:
                R[v] = torch.bitwise_or(R[v], pre(u))
                if not directed:
                    R[u] = torch.bitwise_or(R[u], pre(v))
        else:  # closure: intra-group fixed point on live state (Eq. closure)
            changed = True
            while changed:
                changed = False
                for u, v in contacts:
                    new_v = torch.bitwise_or(R[v], R[u])
                    if not torch.equal(new_v, R[v]):
                        R[v] = new_v
                        changed = True
                    if not directed:
                        new_u = torch.bitwise_or(R[u], R[v])
                        if not torch.equal(new_u, R[u]):
                            R[u] = new_u
                            changed = True
    return R


def packed_torch_to_numpy(R: "torch.Tensor") -> np.ndarray:
    """Reinterpret the int64 device tensor as the core's uint64 packed matrix."""
    _require_torch()
    return np.ascontiguousarray(R.detach().cpu().numpy()).view(np.uint64)
