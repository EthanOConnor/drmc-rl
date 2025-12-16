"""Device resolution utilities for PyTorch and MLX.

Provides unified device selection across training backends.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

# Try importing PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

# Try importing MLX
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    mx = None  # type: ignore
    MLX_AVAILABLE = False


@dataclass
class DeviceInfo:
    """Description of a compute device."""
    
    name: str
    kind: str  # "cpu", "cuda", "mps", "metal"
    ordinal: int = 0
    memory_bytes: Optional[int] = None
    is_default: bool = False
    
    def summary(self) -> str:
        """Human-readable summary."""
        mem = f" ({_format_bytes(self.memory_bytes)})" if self.memory_bytes else ""
        default = " [default]" if self.is_default else ""
        return f"{self.kind}:{self.ordinal} - {self.name}{mem}{default}"


def _format_bytes(num_bytes: Optional[int]) -> str:
    """Format bytes as human-readable string."""
    if num_bytes is None:
        return "?"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes //= 1024
    return f"{num_bytes} PB"


def list_torch_devices() -> List[DeviceInfo]:
    """List available PyTorch devices."""
    if not TORCH_AVAILABLE:
        return []
    
    devices = [DeviceInfo(name="CPU", kind="cpu", ordinal=0)]
    
    # CUDA devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.append(DeviceInfo(
                name=props.name,
                kind="cuda",
                ordinal=i,
                memory_bytes=props.total_memory,
            ))
    
    # MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append(DeviceInfo(name="Apple MPS", kind="mps", ordinal=0))
    
    return devices


def list_mlx_devices() -> List[DeviceInfo]:
    """List available MLX devices."""
    if not MLX_AVAILABLE:
        return []
    
    devices = [DeviceInfo(name="CPU", kind="cpu", ordinal=0)]
    
    # MLX GPU (Metal on Apple Silicon)
    try:
        # MLX uses implicit GPU; check if available
        default = mx.default_device()
        if hasattr(default, "type") and "gpu" in str(default.type).lower():
            devices.append(DeviceInfo(
                name="Apple GPU (Metal)",
                kind="metal",
                ordinal=0,
                is_default=True,
            ))
    except Exception:
        pass
    
    return devices


def resolve_device(spec: str, backend: str = "auto") -> str:
    """Resolve device specification to concrete device string.
    
    Args:
        spec: Device specification like "auto", "cpu", "cuda", "cuda:0", "mps", "mlx"
        backend: "torch", "mlx", or "auto"
    
    Returns:
        Concrete device string (e.g., "cuda:0", "mps", "cpu")
    """
    spec = spec.lower().strip()
    
    # Determine backend
    if backend == "auto":
        if MLX_AVAILABLE and spec in ("mlx", "metal", "gpu"):
            backend = "mlx"
        elif TORCH_AVAILABLE:
            backend = "torch"
        else:
            return "cpu"
    
    if backend == "mlx":
        # MLX uses implicit device management
        if spec in ("auto", "gpu", "metal", "mlx"):
            return "gpu"  # MLX GPU
        return "cpu"
    
    # PyTorch device resolution
    if spec == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    if spec == "cuda":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    
    if spec == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    # Explicit device string
    return spec


def get_device_summary(backend: str = "auto") -> str:
    """Get summary of available devices."""
    lines = []
    
    if TORCH_AVAILABLE:
        lines.append("PyTorch devices:")
        for d in list_torch_devices():
            lines.append(f"  - {d.summary()}")
    
    if MLX_AVAILABLE:
        lines.append("MLX devices:")
        for d in list_mlx_devices():
            lines.append(f"  - {d.summary()}")
    
    if not lines:
        lines.append("No accelerator backends available (CPU only)")
    
    return "\n".join(lines)


def configure_mlx_device(spec: Optional[str] = None) -> None:
    """Configure MLX default device.
    
    Args:
        spec: Device spec like "gpu", "cpu", or None for default
    """
    if not MLX_AVAILABLE:
        return
    
    if spec is None or spec in ("auto", "gpu", "metal"):
        # Use default GPU
        return
    
    if spec == "cpu":
        mx.set_default_device(mx.cpu)
