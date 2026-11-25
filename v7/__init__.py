"""V7 Pure Neural Preimage Inversion Suite (`v7`).

Strict numerical and evaluation discipline:
1. Pure FP32 everywhere (`allow_tf32=False`, zero AMP/FP16/BF16/`complex64`).
2. Parameter count guardrail (`15M <= params <= 35M` for production configs).
3. Per-process CUDA VRAM cap (`<= 11.0 GB`) so all 3 routes can run safely on
   separate runtimes or concurrently as 3 processes on one A100-40GB.
4. Zero classical solver (`V6A` / GL-RRR / Viterbi / BP / SAT / peel-back)
   or ground-truth `W*` leakage during evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

V7_PROD_MIN_PARAMS = 15_000_000
V7_PROD_MAX_PARAMS = 35_000_000
V7_MAX_VRAM_GB = 11.0


def enforce_v7_fp32() -> None:
    """Disable TF32 and enforce pure IEEE-754 float32 default dtype."""
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def assert_strict_fp32(module: nn.Module) -> None:
    """Verify every floating parameter/buffer is strictly `torch.float32`."""
    for name, param in module.named_parameters():
        if param.dtype != torch.float32:
            raise AssertionError(
                f"V7 strict FP32 violation in parameter '{name}': got {param.dtype}"
            )
    for name, buf in module.named_buffers():
        if buf.is_floating_point() and buf.dtype != torch.float32:
            raise AssertionError(
                f"V7 strict FP32 violation in buffer '{name}': got {buf.dtype}"
            )
    if torch.cuda.is_available():
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise AssertionError("V7 strict FP32 violation: TF32 must be False")


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Return total (or trainable) parameter count of `module`."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def configure_v7_vram_cap(max_gb: float = V7_MAX_VRAM_GB, device: int = 0) -> float:
    """Apply hard per-process CUDA memory fraction cap (`<= max_gb` of GPU total)."""
    if not torch.cuda.is_available():
        return 0.0
    if max_gb <= 0.0:
        raise ValueError(f"max_gb must be positive, got {max_gb}")
    props = torch.cuda.get_device_properties(device)
    total_gb = float(props.total_memory) / (1024.0 ** 3)
    if total_gb <= 0.0:
        return 1.0
    frac = min(0.95, float(max_gb) / total_gb)
    torch.cuda.set_per_process_memory_fraction(frac, device=device)
    return frac


__all__ = [
    "V7_PROD_MIN_PARAMS",
    "V7_PROD_MAX_PARAMS",
    "V7_MAX_VRAM_GB",
    "enforce_v7_fp32",
    "assert_strict_fp32",
    "count_parameters",
    "configure_v7_vram_cap",
]
