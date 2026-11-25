"""Phase-SHA V6A: Ouroboros Full-Stack Inverse Scrambling Engine (Strict Pure FP32)."""

from __future__ import annotations

import torch
import torch.nn as nn


def enforce_v6a_fp32() -> None:
    """Enforce strict pure FP32 globally (TF32 disabled on matmul and cuDNN)."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def assert_strict_fp32(obj: nn.Module | torch.Tensor) -> None:
    """Verify no module parameter/tensor uses low-precision, TF32, or complex64."""
    enforce_v6a_fp32()
    if torch.cuda.is_available():
        assert torch.backends.cuda.matmul.allow_tf32 is False, "TF32 matmul must be False"
        assert torch.backends.cudnn.allow_tf32 is False, "TF32 cuDNN must be False"
    if isinstance(obj, torch.Tensor):
        assert obj.dtype in (
            torch.float32,
            torch.float64,
            torch.int64,
            torch.int32,
            torch.uint8,
            torch.bool,
        ), f"Forbidden dtype {obj.dtype}"
        assert obj.dtype != torch.complex64, "Forbidden complex64 tensor"
        return
    for name, p in obj.named_parameters():
        assert p.dtype == torch.float32, f"Parameter {name} has non-FP32 dtype {p.dtype}"
    for name, b in obj.named_buffers():
        if b.is_floating_point():
            if "dual_basis" in name and b.dtype == torch.float64:
                continue
            assert b.dtype in (torch.float32, torch.float64), f"Buffer {name} has dtype {b.dtype}"
