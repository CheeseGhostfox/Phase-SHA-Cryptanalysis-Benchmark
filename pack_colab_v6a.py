"""Build standalone Colab A100/H100 Linux launcher bundle for Phase-SHA V6A (Ouroboros)."""

from __future__ import annotations

import json
import os
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = ROOT_DIR / "colab_bundle_v6a"
ZIP_PATH = ROOT_DIR / "colab_bundle_v6a.zip"

V6A_FILES = [
    "v6a/__init__.py",
    "v6a/v6a_homotopy.py",
    "v6a/v6a_gl_rrr.py",
    "v6a/v6a_bottleneck_mps.py",
    "v6a/v6a_bm_escape.py",
    "v6a/v6a_navigator.py",
    "v6a/v6a_engine.py",
]

SRC_FILES = [
    "src/__init__.py",
    "src/sha256_ref.py",
    "src/boundary_clamp.py",
    "src/dual_basis.py",
    "src/anneal_solver.py",
    "src/graph_backbone.py",
    "src/bit_grid_backbone.py",
    "src/flow_trainer.py",
    "src/v3_solver.py",
    "src/v3_trainer.py",
    "src/v4_manifolds.py",
    "src/v4_boundary.py",
    "src/v4_backbone.py",
    "src/v4_solver.py",
    "src/v5_interfaces.py",
    "src/v5d_rrr.py",
]

# Note: strictly excludes all Colab authentication / remote relay scripts!
OTHER_FILES = [
    "train_v6a.py",
    "pytest.ini",
    "tests/__init__.py",
    "tests/test_v6a.py",
    "configs/v6a_frontier_push_40g.yaml",
    "configs/v6a_r18_colab_40g.yaml",
    "configs/v6a_r18_colab_80g.yaml",
    "scripts/colab_monitor_dashboard.py",
    "scripts/plot_v6a_comparison.py",
    "docs/V6A_DESIGN_SPEC.md",
]

OPTIONAL_CSV_FILES = [
    "runs/r18_32b_v3/metrics.csv",
    "runs/v4c_r18_colab/metrics.csv",
    "runs/v5d_pure_r18_colab/metrics.csv",
    "runs/v5d_neural_r18_colab/metrics.csv",
    "runs/v5c_r18_colab/metrics.csv",
]


def _force_remove_readonly(func, path, exc_info) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _md_cell(text: str) -> Dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").splitlines()],
    }


def _code_cell(code: str) -> Dict[str, object]:
    lines = code.strip("\n").splitlines()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + ("\n" if i < len(lines) - 1 else "") for i, line in enumerate(lines)],
    }


def build_notebook() -> Dict[str, object]:
    cells: List[Dict[str, object]] = [
        _md_cell(
            """
# Phase-SHA V6A (`Ouroboros`) 全栈闭环逆置乱引擎 — Colab A100 发射台

严格执行纯 FP32（`allow_tf32 = False`、零混合精度、零单比特边际 BCE）：
1. **L1 (`v6a_homotopy`)**：兼容携带显式 31-bit 进位松弛的 $\\mathbb{F}_2$ 线性骨架，$\\alpha=0$ 全窗无冲突初始化，2D $(\\alpha, m)$ 自适应连续同伦；
2. **L2 (`v6a_gl_rrr`)**：全局 Damped Min-Sum BP + Top-K OSD-0 + Exact 4-word Viterbi RRR，遵守 $P_B^{(\\text{nl},\\alpha)}(x) = (1-\\alpha)x + \\alpha P_B^{(\\text{nl})}(x)$ 不动点保真；
3. **L3 (`v6a_bottleneck_mps`)**：64 瓶颈位点无奇点 MGS2 + $O(\\log L)$ 并行树收缩 + Rényi-2 纠缠熵下界 ($S_{\\text{ent}} \\ge 0.55$)；
4. **L4 (`v6a_bm_escape`)**：滞留触发 $S^{N-1}$ 历史基正交 Cayley 逃逸；
5. **L5 (`v6a_navigator`)**：零刹车同伦切向预测器（仅自监督 MSE）+ VSIDS 权重焦点。
"""
        ),
        _md_cell("## ① 环境检查与工作目录切换"),
        _code_cell(
            """
import os
from pathlib import Path
import torch

CANDIDATES = [Path.cwd(), Path('/content/colab_bundle_v6a'), Path('/content/drive/MyDrive/colab_bundle_v6a')]
WORK_DIR = next((p for p in CANDIDATES if (p / 'train_v6a.py').exists()), None)
if WORK_DIR is None:
    raise FileNotFoundError("请解压 colab_bundle_v6a.zip 至 /content/")
os.chdir(WORK_DIR)
SAVE_ROOT = '/content/runs'
Path(SAVE_ROOT).mkdir(parents=True, exist_ok=True)
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if torch.cuda.is_available() else 0.0
TIER = '80g' if vram_gb >= 60.0 else '40g'
print(f"DIR={WORK_DIR} | PyTorch={torch.__version__} | GPU={gpu_name} ({vram_gb:.1f} GB) -> tier={TIER}")
"""
        ),
        _md_cell("## ② 运行 `pytest -v tests/test_v6a.py` 红线与 R=4/5 无作弊真哈希验真"),
        _code_cell("!python3 -u -m pytest -v tests/test_v6a.py"),
        _md_cell("## ③ Battlefield A: Frontier Push (`R=6,7,8,9,10`, `B=1 x M=1024`, 零剥离作弊)"),
        _code_cell("!python3 -u train_v6a.py --mode frontier_push --config configs/v6a_frontier_push_40g.yaml --save_dir {SAVE_ROOT}/v6a_frontier_push"),
        _md_cell("## ④ Battlefield B: R=18 2000 步深挖 (`B=4 x M=256`, 2D 交错同伦 + `--resume` 续训)"),
        _code_cell("!python3 -u train_v6a.py --mode attack --rounds 18 --config configs/v6a_r18_colab_{TIER}.yaml --steps 2000 --save_dir {SAVE_ROOT}/v6a_v2_r18_colab"),
        _md_cell("## ⑤ 绘制 V6A-v2 vs V3/V4C/V5 横向科研对比总图"),
        _code_cell(
            """
!python3 -u scripts/plot_v6a_comparison.py --v6a-csv {SAVE_ROOT}/v6a_v2_r18_colab/metrics.csv --out {SAVE_ROOT}/v6a_comparison_r18_colab.png
from IPython.display import Image, display
display(Image(filename=f"{SAVE_ROOT}/v6a_comparison_r18_colab.png"))
"""
        ),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def pack_bundle() -> None:
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR, onerror=_force_remove_readonly)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    for rel in V6A_FILES + SRC_FILES + OTHER_FILES:
        src = ROOT_DIR / rel
        if not src.exists():
            raise FileNotFoundError(f"Required V6A bundle file missing: {src}")
        dst = BUNDLE_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for rel in OPTIONAL_CSV_FILES:
        src = ROOT_DIR / rel
        if src.exists():
            dst = BUNDLE_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # Ensure scripts package marker cleanly present
    (BUNDLE_DIR / "scripts" / "__init__.py").write_text("", encoding="utf-8")

    (BUNDLE_DIR / "requirements_colab.txt").write_text(
        "torch>=2.3.0\npyyaml>=6.0\nnumpy>=1.26\nmatplotlib>=3.8\npytest>=8.0\n",
        encoding="utf-8",
    )
    (BUNDLE_DIR / "README_COLAB_V6A.md").write_text(
        """# Phase-SHA V6A (`Ouroboros`) Colab A100 自包含发射包

## 快速启动命令（严格纯 FP32 + 无单比特边际 BCE）
```bash
unzip -q colab_bundle_v6a.zip -d /content
cd /content/colab_bundle_v6a
python3 -u -m pytest -v tests/test_v6a.py

# Battlefield A (B=1 x M=1024 突击 R=6..10):
python3 -u train_v6a.py --mode frontier_push --config configs/v6a_frontier_push_40g.yaml --save_dir /content/runs/v6a_frontier_push

# Battlefield B (B=4 x M=256 R=18 全栈 2D 同伦深挖 2000 steps):
python3 -u train_v6a.py --mode attack --rounds 18 --config configs/v6a_r18_colab_40g.yaml --steps 2000 --save_dir /content/runs/v6a_v2_r18_colab

# 绘图对比 V3 / V4C / V5:
python3 -u scripts/plot_v6a_comparison.py --v6a-csv /content/runs/v6a_v2_r18_colab/metrics.csv --out /content/runs/v6a_comparison_r18_colab.png
```
""",
        encoding="utf-8",
    )
    (BUNDLE_DIR / "colab_v6a_launcher.ipynb").write_text(
        json.dumps(build_notebook(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(BUNDLE_DIR.rglob("*")):
            if file_path.is_file() and "__pycache__" not in file_path.parts:
                arcname = Path("colab_bundle_v6a") / file_path.relative_to(BUNDLE_DIR)
                zf.write(file_path, arcname=str(arcname))

    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"Built Colab V6A bundle at {BUNDLE_DIR} and archive {ZIP_PATH} ({size_mb:.2f} MB)", flush=True)


if __name__ == "__main__":
    pack_bundle()
