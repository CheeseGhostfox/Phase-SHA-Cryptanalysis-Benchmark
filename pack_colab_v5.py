"""Build standalone Colab Ultra (A100/H100 Linux) launcher bundle for Phase-SHA V5."""

from __future__ import annotations

import json
import os
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = ROOT_DIR / "colab_bundle_v5"
ZIP_PATH = ROOT_DIR / "colab_bundle_v5.zip"

SRC_FILES = [
    "src/__init__.py",
    "src/sha256_ref.py",
    "src/boundary_clamp.py",
    "src/dual_basis.py",
    "src/anneal_solver.py",
    "src/graph_backbone.py",
    "src/bit_grid_backbone.py",
    "src/flow_trainer.py",
    "src/v2_solver.py",
    "src/v2_trainer.py",
    "src/v3_solver.py",
    "src/v3_trainer.py",
    "src/v4_manifolds.py",
    "src/v4_boundary.py",
    "src/v4_backbone.py",
    "src/v4_solver.py",
    "src/v4_trainer.py",
    "src/v5_interfaces.py",
    "src/v5d_rrr.py",
    "src/v5c_mps.py",
    "src/v5b_bm.py",
    "src/v5a_octonion.py",
    "src/v5_trainer.py",
]

OTHER_FILES = [
    "train_v5.py",
    "pytest.ini",
    "tests/__init__.py",
    "tests/test_v4.py",
    "tests/test_v5.py",
    "configs/v5d_pure_r18_colab.yaml",
    "configs/v5d_neural_r18_colab.yaml",
    "configs/v5c_r18_colab.yaml",
    "configs/v5b_r18_colab.yaml",
    "configs/v5a_r18_colab.yaml",
    "scripts/plot_v5_comparison.py",
    "scripts/plot_v2_results.py",
    "docs/V5_ROADMAP_SPEC.md",
    "checkpoints/stage3b_r18_calibrated_best.pt",
    "runs/r18_32b_v3/metrics.csv",
]

OPTIONAL_CSV_FILES = [
    "runs/v4a_r18_colab/metrics.csv",
    "runs/v4b_r18_colab/metrics.csv",
    "runs/v4c_r18_colab/metrics.csv",
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
# Phase-SHA V5 四路控变量单卡破壁体系 — Colab A100/H100 满血发射台

本发射包严格对齐 Lean 4 Module 6A–6D 与 `V5_ROADMAP_SPEC.md`：
- **死守纯血 FP32**：`allow_tf32 = False`、零混合精度、零 `torch.complex64`；
- **四路单卡正交消融**：`V5D (RRR)` → `V5C (MPS)` → `V5B (BM S^{N-1})` → `V5A (Octonion S^7)`；
- **双轴公平对比**：`metrics.csv` 与监控面板同步记录 `step` 与累计GPU算力时间 `wall_s`。
"""
        ),
        _md_cell("## ① 环境自检、Drive 挂载与工作目录切换"),
        _code_cell(
            """
import os, platform, sys
from pathlib import Path

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except ImportError:
    print("非 Colab 环境，跳过 Google Drive 挂载")

CANDIDATES = [Path.cwd(), Path('/content/colab_bundle_v5'), Path('/content/drive/MyDrive/colab_bundle_v5')]
WORK_DIR = next((p for p in CANDIDATES if (p / 'train_v5.py').exists()), None)
if WORK_DIR is None:
    raise FileNotFoundError("请解压 colab_bundle_v5.zip 至 /content/")
os.chdir(WORK_DIR)
SAVE_ROOT = '/content/drive/MyDrive/phase_sha_v5' if Path('/content/drive/MyDrive').exists() else str(WORK_DIR / 'runs_v5')
Path(SAVE_ROOT).mkdir(parents=True, exist_ok=True)
import torch
print(f"DIR={WORK_DIR} | PyTorch={torch.__version__} | CUDA={torch.cuda.is_available()} | SAVE={SAVE_ROOT}")
"""
        ),
        _md_cell("## ② 运行 `pytest -v tests/test_v5.py` 全套数学与纯 FP32 红线验真"),
        _code_cell("!python -u -m pytest -v tests/test_v5.py"),
        _md_cell("## ③-D1 启动 V5D-Pure（纯 GPU 批量高频 RRR 反射）"),
        _code_cell(
            """
!python -u train_v5.py --variant v5d_pure --config configs/v5d_pure_r18_colab.yaml --steps 2000 --save_dir {SAVE_ROOT}/v5d_pure_r18_colab
"""
        ),
        _md_cell("## ③-D2 启动 V5D-Neural（神经引导度量 + 动态阻尼 RRR）"),
        _code_cell(
            """
!python -u train_v5.py --variant v5d_neural --config configs/v5d_neural_r18_colab.yaml --steps 2000 --save_dir {SAVE_ROOT}/v5d_neural_r18_colab
"""
        ),
        _md_cell("## ③-C 启动 V5C（闭环感知张量网络 MPS，可自由调 `--bond_dim 4/8/16/32`）"),
        _code_cell(
            """
!python -u train_v5.py --variant v5c --config configs/v5c_r18_colab.yaml --bond_dim 8 --steps 2000 --init_from checkpoints/stage3b_r18_calibrated_best.pt --save_dir {SAVE_ROOT}/v5c_r18_colab
"""
        ),
        _md_cell("## ③-B 启动 V5B（Burer-Monteiro 超高维球面，可自由调 `--manifold_dim 16/32/64`）"),
        _code_cell(
            """
!python -u train_v5.py --variant v5b --config configs/v5b_r18_colab.yaml --manifold_dim 32 --steps 2000 --init_from checkpoints/stage3b_r18_calibrated_best.pt --save_dir {SAVE_ROOT}/v5b_r18_colab
"""
        ),
        _md_cell("## ③-A 启动 V5A（8D 八元数 S^7 + 结合子/实轴双路强退火）"),
        _code_cell(
            """
!python -u train_v5.py --variant v5a --config configs/v5a_r18_colab.yaml --steps 2000 --init_from checkpoints/stage3b_r18_calibrated_best.pt --save_dir {SAVE_ROOT}/v5a_r18_colab
"""
        ),
        _md_cell("## ④ 生成双轴（同训练步数 + 同算力时间）横向破壁对比总图"),
        _code_cell(
            """
!python -u scripts/plot_v5_comparison.py --out {SAVE_ROOT}/v5_comparison_r18_colab.png
from IPython.display import Image, display
display(Image(filename=f"{SAVE_ROOT}/v5_comparison_r18_colab.png"))
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

    for rel in SRC_FILES + OTHER_FILES:
        src = ROOT_DIR / rel
        if not src.exists():
            raise FileNotFoundError(f"Required bundle file missing: {src}")
        dst = BUNDLE_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if dst.name.endswith(".pt"):
            os.chmod(dst, stat.S_IWRITE | stat.S_IREAD)

    for rel in OPTIONAL_CSV_FILES:
        src = ROOT_DIR / rel
        if src.exists():
            dst = BUNDLE_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    (BUNDLE_DIR / "requirements_colab.txt").write_text(
        "torch>=2.3.0\npyyaml>=6.0\nnumpy>=1.26\nmatplotlib>=3.8\npytest>=8.0\n",
        encoding="utf-8",
    )
    (BUNDLE_DIR / "README_COLAB_V5.md").write_text(
        """# Phase-SHA V5 Colab A100/H100 自包含发射包

## 快速启动命令（纯血 FP32 + 控变量 `R=18` 2048 留出集）
```bash
unzip -q colab_bundle_v5.zip -d /content
cd /content/colab_bundle_v5
python3 -u -m pytest -v tests/test_v5.py

# 四路正交破壁基模启动指令：
python3 -u train_v5.py --variant v5d_pure   --config configs/v5d_pure_r18_colab.yaml   --steps 2000 --save_dir /content/runs/v5d_pure_r18_colab
python3 -u train_v5.py --variant v5d_neural --config configs/v5d_neural_r18_colab.yaml --steps 2000 --save_dir /content/runs/v5d_neural_r18_colab
python3 -u train_v5.py --variant v5c        --config configs/v5c_r18_colab.yaml --bond_dim 8 --steps 2000 --init_from checkpoints/stage3b_r18_calibrated_best.pt --save_dir /content/runs/v5c_r18_colab
python3 -u train_v5.py --variant v5b        --config configs/v5b_r18_colab.yaml --manifold_dim 32 --steps 2000 --init_from checkpoints/stage3b_r18_calibrated_best.pt --save_dir /content/runs/v5b_r18_colab
python3 -u train_v5.py --variant v5a        --config configs/v5a_r18_colab.yaml --steps 2000 --init_from checkpoints/stage3b_r18_calibrated_best.pt --save_dir /content/runs/v5a_r18_colab
```
""",
        encoding="utf-8",
    )
    (BUNDLE_DIR / "colab_v5_launcher.ipynb").write_text(
        json.dumps(build_notebook(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(BUNDLE_DIR.rglob("*")):
            if file_path.is_file() and "__pycache__" not in file_path.parts:
                arcname = Path("colab_bundle_v5") / file_path.relative_to(BUNDLE_DIR)
                zf.write(file_path, arcname=str(arcname))

    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"Built Colab V5 bundle at {BUNDLE_DIR} and archive {ZIP_PATH} ({size_mb:.2f} MB)", flush=True)


if __name__ == "__main__":
    pack_bundle()
