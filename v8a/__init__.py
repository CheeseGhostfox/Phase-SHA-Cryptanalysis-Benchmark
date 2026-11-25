"""Phase-SHA V8A: Volume-Preserving Torus Flow Unscrambler."""

from __future__ import annotations

import torch


def configure_strict_fp32() -> None:
    """Enforce IEEE-754 float32 precision with TF32 disabled."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(torch.float32)


def assert_strict_fp32(module: torch.nn.Module | None = None) -> None:
    """Assert TF32 is disabled and all floating tensors in module use float32."""
    if torch.backends.cuda.matmul.allow_tf32:
        raise AssertionError("TF32 must be disabled for cuda.matmul")
    if torch.backends.cudnn.allow_tf32:
        raise AssertionError("TF32 must be disabled for cudnn")
    if torch.get_default_dtype() != torch.float32:
        raise AssertionError(f"Default dtype must be float32, got {torch.get_default_dtype()}")
    if module is not None:
        for name, param in module.named_parameters():
            if param.is_floating_point() and param.dtype != torch.float32:
                raise AssertionError(f"Parameter {name} has dtype {param.dtype}")
        for name, buf in module.named_buffers():
            if buf.is_floating_point() and buf.dtype != torch.float32:
                raise AssertionError(f"Buffer {name} has dtype {buf.dtype}")


__all__ = ["configure_strict_fp32", "assert_strict_fp32"]
