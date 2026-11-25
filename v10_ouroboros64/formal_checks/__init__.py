"""
R4-3 形式化边界与符号/数值核验套件 (`v10_ouroboros64/formal_checks/`):
  - verify_symbolic_sympy.py: SymPy 精确符号推导与代数/概率/复杂度核验 (S1-S5)
  - verify_smt_z3.py: Z3 SMT 位向量与布尔等价/反例见证核验 (Z1-Z5)
  - verify_torch_bounds.py: PyTorch FP32 (`allow_tf32=False`) 实测边界与 C++ GF(2) 局部窗缩放核验
  - run_formal_checks.py: 总控调度与 21 个基线存档 SHA-256 只读完整性校验
"""

from __future__ import annotations

from typing import Tuple

ARCHIVE_DIRS: Tuple[str, ...] = (
    "v10_ouroboros64/runs/a100_m2",
    "v10_ouroboros64/runs/a100_m3",
    "v10_ouroboros64/runs/a100_m3v2",
)

DEFAULT_DIAG_SUMMARY_REL = "v10_ouroboros64/diagnostics/diagnostics_summary.json"
DEFAULT_M3V2_CKPT = "v10_ouroboros64/runs/a100_m3v2/checkpoints/best_m3v2.pt"
DEFAULT_M2_CKPT = "v10_ouroboros64/runs/a100_m2/checkpoints/ouroboros_a100_step4000.pt"
