"""Helpers for numerically stable discounted-return computation.

The training experiment supports both PyTorch and MLX backends.  Episodes can
span tens of thousands of steps which makes the usual
``gamma**step``-based formulation vulnerable to severe underflow.  Once the
discount power reaches zero the old implementation divided by zero and the
learner observed ``inf``/``nan`` returns, quickly pushing the policy toward the
lowest achievable reward.  These helpers re-compute the discounted return via a
backwards accumulation which is stable for very long episodes and agnostic to
the underlying numeric backend.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence


def _ensure_scalar_bootstrap(value: object) -> float:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            return 0.0
        return float(value[0])
    return float(value)


def discounted_returns_list(
    rewards: Sequence[float],
    gamma: float,
    dones: Optional[Iterable[object]] = None,
    bootstrap: Optional[object] = None,
) -> List[float]:
    """Return discounted returns for a Python sequence of rewards."""

    length = len(rewards)
    if length == 0:
        return []

    gamma_f = float(gamma)
    returns: List[float] = [0.0] * length
    if dones is None:
        dones_iter = [False] * length
    else:
        dones_iter = [bool(flag) for flag in dones]
        if len(dones_iter) < length:
            raise ValueError("dones must match rewards length")
        dones_iter = dones_iter[:length]

    running = 0.0 if bootstrap is None else _ensure_scalar_bootstrap(bootstrap)

    for idx in range(length - 1, -1, -1):
        mask = 0.0 if dones_iter[idx] else 1.0
        running = float(rewards[idx]) + gamma_f * running * mask
        returns[idx] = running

    return returns


def discounted_returns_torch(rewards_tensor, gamma: float, dones=None, bootstrap=None):
    """Discounted returns for torch tensors with optional terminals and bootstrap."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch optional
        raise RuntimeError("torch is required for discounted_returns_torch") from exc

    if rewards_tensor.dim() not in (1, 2):
        raise ValueError("rewards_tensor must be 1-D or 2-D [T] or [T, B]")

    rt = rewards_tensor
    T = int(rt.shape[0])
    B = 1 if rt.dim() == 1 else int(rt.shape[1])
    rt = rt.reshape(T, B)

    device = rt.device
    dtype = rt.dtype

    if dones is None:
        dones_tensor = torch.zeros((T, B), dtype=torch.bool, device=device)
    else:
        dones_tensor = torch.as_tensor(dones, dtype=torch.bool, device=device)
        if dones_tensor.numel() == 1:
            dones_tensor = dones_tensor.reshape(1, 1).expand(T, B)
        elif dones_tensor.dim() == 1:
            if int(dones_tensor.shape[0]) != T:
                raise ValueError("dones must have length T")
            dones_tensor = dones_tensor.reshape(T, 1).expand(T, B)
        elif dones_tensor.dim() == 2:
            if int(dones_tensor.shape[0]) != T:
                raise ValueError("dones must have shape [T, B] or [T, 1]")
            if int(dones_tensor.shape[1]) not in (1, B):
                raise ValueError("dones second dimension must be 1 or match batch B")
            if int(dones_tensor.shape[1]) == 1 and B > 1:
                dones_tensor = dones_tensor.expand(T, B)
        else:
            raise ValueError("dones must be scalar, [T], or [T, B]")

    if bootstrap is None:
        running = torch.zeros((B,), dtype=dtype, device=device)
    else:
        bootstrap_tensor = torch.as_tensor(bootstrap, dtype=dtype, device=device)
        if bootstrap_tensor.numel() == 1:
            running = bootstrap_tensor.reshape(1).expand(B)
        elif bootstrap_tensor.dim() == 1 and int(bootstrap_tensor.shape[0]) == B:
            running = bootstrap_tensor.reshape(B)
        else:
            raise ValueError("bootstrap must be scalar or have shape [B]")

    returns = torch.empty_like(rt)
    gamma_tensor = rt.new_tensor(float(gamma))

    for idx in range(T - 1, -1, -1):
        mask = (~dones_tensor[idx]).to(dtype)
        running = rt[idx] + gamma_tensor * running * mask
        returns[idx] = running

    return returns.reshape(rewards_tensor.shape)


def discounted_returns_mlx(rewards_tensor, gamma: float, dones=None, bootstrap=None):
    """Discounted returns for MLX arrays with optional terminals and bootstrap."""

    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - MLX optional
        raise RuntimeError("mlx is required for discounted_returns_mlx") from exc

    rt = rewards_tensor
    if rt.ndim not in (1, 2):
        raise ValueError("rewards_tensor must be 1-D or 2-D [T] or [T, B]")

    T = int(rt.shape[0])
    B = 1 if rt.ndim == 1 else int(rt.shape[1])
    rt = mx.reshape(rt, (T, B))

    if dones is None:
        dones_tensor = mx.zeros((T, B), dtype=mx.bool_)
    else:
        dones_tensor = mx.asarray(dones, dtype=mx.bool_)
        if dones_tensor.size == 1:
            dones_tensor = mx.broadcast_to(mx.reshape(dones_tensor, (1, 1)), (T, B))
        elif dones_tensor.ndim == 1:
            if int(dones_tensor.shape[0]) != T:
                raise ValueError("dones must have length T")
            dones_tensor = mx.broadcast_to(mx.reshape(dones_tensor, (T, 1)), (T, B))
        elif dones_tensor.ndim == 2:
            if int(dones_tensor.shape[0]) != T:
                raise ValueError("dones must have shape [T, B] or [T, 1]")
            if int(dones_tensor.shape[1]) not in (1, B):
                raise ValueError("dones second dimension must be 1 or match batch B")
            if int(dones_tensor.shape[1]) == 1 and B > 1:
                dones_tensor = mx.broadcast_to(dones_tensor, (T, B))
        else:
            raise ValueError("dones must be scalar, [T], or [T, B]")

    if bootstrap is None:
        running = mx.zeros((B,), dtype=rt.dtype)
    else:
        bootstrap_tensor = mx.asarray(bootstrap, dtype=rt.dtype)
        if bootstrap_tensor.size == 1:
            running = mx.broadcast_to(mx.reshape(bootstrap_tensor, (1,)), (B,))
        elif bootstrap_tensor.ndim == 1 and int(bootstrap_tensor.shape[0]) == B:
            running = mx.reshape(bootstrap_tensor, (B,))
        else:
            raise ValueError("bootstrap must be scalar or have shape [B]")

    returns = mx.zeros_like(rt)
    gamma_scalar = mx.array(float(gamma), dtype=rt.dtype)
    one = mx.ones((B,), dtype=rt.dtype)
    zero = mx.zeros((B,), dtype=rt.dtype)

    for idx in range(T - 1, -1, -1):
        mask = mx.where(dones_tensor[idx], zero, one)
        running = rt[idx] + gamma_scalar * running * mask
        returns = _mlx_set_row(returns, idx, running)

    return mx.reshape(returns, rewards_tensor.shape)


def _mlx_set_row(tensor, index: int, value):
    """Return a tensor with the given row replaced by value (fallback for lack of in-place)."""

    try:
        import mlx.core as mx
    except ImportError:  # pragma: no cover - handled earlier
        raise

    if hasattr(mx, "index_update"):
        return mx.index_update(tensor, mx.index[index], value)
    # Fallback for older MLX versions without index_update
    before = tensor[: index]
    after = tensor[index + 1 :]
    value_expanded = mx.reshape(value, (1,) + value.shape)
    return mx.concatenate((before, value_expanded, after), axis=0)


__all__ = [
    "discounted_returns_list",
    "discounted_returns_torch",
    "discounted_returns_mlx",
]
