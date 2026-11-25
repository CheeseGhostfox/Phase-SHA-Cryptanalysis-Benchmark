"""Build standalone Colab Ultra (A100/H100 Linux) launcher bundle for Phase-SHA V4A / V4B / V4C."""

from __future__ import annotations

import json
import os
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = ROOT_DIR / "colab_bundle_v4"
ZIP_PATH = ROOT_DIR / "colab_bundle_v4.zip"

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
]

OTHER_FILES = [
    "train_v4.py",
    "pytest.ini",
    "tests/__init__.py",
    "tests/test_v4.py",
    "configs/v4a_r18.yaml",
    "configs/v4b_r18.yaml",
    "configs/v4c_r18.yaml",
    "configs/v4a_r18_colab.yaml",
    "configs/v4b_r18_colab.yaml",
    "configs/v4c_r18_colab.yaml",
    "scripts/plot_v4_comparison.py",
    "scripts/plot_v2_results.py",
    "scripts/profile_v4_a100.py",
    "docs/V4_ROADMAP_SPEC.md",
    "checkpoints/stage3b_r18_calibrated_best.pt",
    "runs/r18_32b_v3/metrics.csv",
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
# Phase-SHA V4 (V4A / V4B / V4C) — Colab Ultra 满血发射台

本 Notebook 严格遵循 Phase-SHA V4 工程与科学契约：
- **全流程严格 FP32 精度**（自动禁用 TF32，严禁 FP16/BF16/INT8）；
- **Linux + CUDA 自动启用 `torch.compile`** 释放 A100 / H100 显存带宽；
- **防断连 Google Drive 自动存档**：支持掉线重连后自动检测 `latest.pt` 执行 `--resume` 无损续训；
- **面向 V4E（PoW 部分零前缀）的通用 `static_mask` 基模接口**，训完后可直接微调切入 V4E 或 V4D。
"""
        ),
        _md_cell("## ① 环境自检、Google Drive 挂载与工作目录初始化"),
        _code_cell(
            """
import os, platform, sys
from pathlib import Path

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except ImportError:
    print("非 Colab 环境，跳过 Google Drive 挂载")

# 自动定位解压后的 colab_bundle_v4 工作目录
CANDIDATES = [
    Path.cwd(),
    Path('/content/colab_bundle_v4'),
    Path('/content/drive/MyDrive/colab_bundle_v4'),
]
WORK_DIR = next((p for p in CANDIDATES if (p / 'train_v4.py').exists()), None)
if WORK_DIR is None:
    raise FileNotFoundError("未找到 train_v4.py，请上传并解压 colab_bundle_v4.zip 至 /content/colab_bundle_v4")
os.chdir(WORK_DIR)
print(f"工作目录已切换至: {WORK_DIR}")

DRIVE_SAVE_ROOT = '/content/drive/MyDrive/phase_sha_v4' if Path('/content/drive/MyDrive').exists() else str(WORK_DIR / 'runs_v4')
Path(DRIVE_SAVE_ROOT).mkdir(parents=True, exist_ok=True)
print(f"Google Drive 自动存档根目录: {DRIVE_SAVE_ROOT}")

import torch
print(f"OS={platform.system()} | Python={sys.version.split()[0]} | PyTorch={torch.__version__}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU={props.name} | VRAM={props.total_memory / (1024**3):.2f} GB | BF16=禁止(V4强制FP32)")
else:
    print("WARNING: 未检测到 CUDA GPU，建议在 Colab 切换为 A100 / H100 运行时！")
"""
        ),
        _md_cell("## ② 一键运行 `tests/test_v4.py` 验真（离散极限零误差 + Lean 4 恒等式 + 纯 FP32 + V4E 接口）"),
        _code_cell(
            """
!python -u -m pytest -v tests/test_v4.py
"""
        ),
        _md_cell(
            """
## ③ V4A / V4B / V4C 三个独立检查点满血训练与评估单元

每个单元独立检查 Google Drive 对应目录下的 `latest.pt`：
- 若存在 `latest.pt` → 自动执行 `--resume` 无损断点续训；
- 若首次启动 → 从冻结的 V3 校准权重 `checkpoints/stage3b_r18_calibrated_best.pt` 无损热启动。
"""
        ),
        _md_cell("### ③-A 启动检查点 V4A（$\\mathbb{S}^1 \\subset \\mathbb{C}$ 复单位圆阿贝尔消融基线）"),
        _code_cell(
            """
import os, subprocess
from pathlib import Path

variant = 'v4a'
save_dir = Path(DRIVE_SAVE_ROOT) / f'{variant}_r18_colab'
latest_ckpt = save_dir / 'latest.pt'
init_ckpt = 'checkpoints/stage3b_r18_calibrated_best.pt'

cmd = [
    'python', '-u', 'train_v4.py',
    '--variant', variant,
    '--config', f'configs/{variant}_r18_colab.yaml',
    '--save_dir', str(save_dir),
]
if latest_ckpt.exists():
    print(f"[续训] 检测到断点: {latest_ckpt}")
    cmd.extend(['--resume', str(latest_ckpt)])
else:
    print(f"[热启动] 从 V3 校准基座启动: {init_ckpt}")
    cmd.extend(['--init_from', init_ckpt])

print("执行命令:", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
"""
        ),
        _md_cell("### ③-B 启动检查点 V4B（$\\mathbb{S}^2 \\subset \\mathfrak{su}(2)$ Bloch 厄米酉对合流形，Lean 4 `Mat2C`）"),
        _code_cell(
            """
import subprocess
from pathlib import Path

variant = 'v4b'
save_dir = Path(DRIVE_SAVE_ROOT) / f'{variant}_r18_colab'
latest_ckpt = save_dir / 'latest.pt'
init_ckpt = 'checkpoints/stage3b_r18_calibrated_best.pt'

cmd = [
    'python', '-u', 'train_v4.py',
    '--variant', variant,
    '--config', f'configs/{variant}_r18_colab.yaml',
    '--save_dir', str(save_dir),
]
if latest_ckpt.exists():
    print(f"[续训] 检测到断点: {latest_ckpt}")
    cmd.extend(['--resume', str(latest_ckpt)])
else:
    print(f"[热启动] 从 V3 校准基座启动: {init_ckpt}")
    cmd.extend(['--init_from', init_ckpt])

print("执行命令:", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
"""
        ),
        _md_cell("### ③-C 启动检查点 V4C（$\\mathbb{S}^3 \\subset \\mathbb{H}$ 单位四元数流形，Lean 4 `Quat`）"),
        _code_cell(
            """
import subprocess
from pathlib import Path

variant = 'v4c'
save_dir = Path(DRIVE_SAVE_ROOT) / f'{variant}_r18_colab'
latest_ckpt = save_dir / 'latest.pt'
init_ckpt = 'checkpoints/stage3b_r18_calibrated_best.pt'

cmd = [
    'python', '-u', 'train_v4.py',
    '--variant', variant,
    '--config', f'configs/{variant}_r18_colab.yaml',
    '--save_dir', str(save_dir),
]
if latest_ckpt.exists():
    print(f"[续训] 检测到断点: {latest_ckpt}")
    cmd.extend(['--resume', str(latest_ckpt)])
else:
    print(f"[热启动] 从 V3 校准基座启动: {init_ckpt}")
    cmd.extend(['--init_from', init_ckpt])

print("执行命令:", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
"""
        ),
        _md_cell("## ④ 自动汇总日志生成 V3 vs V4A / V4B / V4C 横向破壁对比总图"),
        _code_cell(
            """
import subprocess
from pathlib import Path
from IPython.display import Image, display

out_png = Path(DRIVE_SAVE_ROOT) / 'v4_comparison_r18.png'
cmd = [
    'python', '-u', 'scripts/plot_v4_comparison.py',
    '--v3-csv', 'runs/r18_32b_v3/metrics.csv',
    '--v4a-csv', f'{DRIVE_SAVE_ROOT}/v4a_r18_colab/metrics.csv',
    '--v4b-csv', f'{DRIVE_SAVE_ROOT}/v4b_r18_colab/metrics.csv',
    '--v4c-csv', f'{DRIVE_SAVE_ROOT}/v4c_r18_colab/metrics.csv',
    '--out', str(out_png),
]
subprocess.run(cmd, check=True)
display(Image(filename=str(out_png)))

for v in ('v4a', 'v4b', 'v4c'):
    rdir = Path(DRIVE_SAVE_ROOT) / f'{v}_r18_colab'
    if (rdir / 'metrics.csv').exists():
        subprocess.run(['python', '-u', 'scripts/plot_v2_results.py', '--run-dir', str(rdir)], check=True)
"""
        ),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def write_aux_docs(bundle_dir: Path) -> None:
    (bundle_dir / "requirements_colab.txt").write_text(
        "\n".join(
            [
                "torch>=2.3.0",
                "pyyaml>=6.0",
                "numpy>=1.26",
                "matplotlib>=3.8",
                "pytest>=8.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "README_COLAB.md").write_text(
        """# Phase-SHA V4 Colab Ultra 一键满血发射包

## 快速启动（3 步）
1. 将本目录（或 `colab_bundle_v4.zip`）上传至 Google Drive 根目录或 `/content/`（若传 zip 请先执行 `!unzip -q colab_bundle_v4.zip -d /content`）；
2. 在 Colab 选择 **A100 / H100 High-RAM** 运行时，打开 `colab_v4_launcher.ipynb`；
3. 按顺序运行 **① 环境挂载 → ② `test_v4.py` 验真 → ③-A/B/C 训练单元 → ④ V3 vs V4A~C 横向对比总图**。

## 核心说明
- 训练全程强制 **FP32 全精度**（禁用 TF32/FP16/BF16）；
- Linux + CUDA 下自动启用 `torch.compile` 加速，并自动剥离编译前缀存档，确保与本地 Windows 无损互通；
- 权重与 `metrics.csv` 每 250 步自动落盘 Google Drive（`/content/drive/MyDrive/phase_sha_v4/<variant>_r18_colab/`），断连重跑对应单元即可无缝 `--resume`；
- 训完后得到的 `best.pt` / `latest.pt` 可零结构修改直接用于 V4E 部分零前缀 PoW 微调或 V4D 概率引导 SAT 混合求解。
""",
        encoding="utf-8",
    )


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
        # Ensure copied read-only checkpoint inside bundle is normal writable file in bundle if needed
        if dst.name.endswith(".pt"):
            os.chmod(dst, stat.S_IWRITE | stat.S_IREAD)

    write_aux_docs(BUNDLE_DIR)
    nb_path = BUNDLE_DIR / "colab_v4_launcher.ipynb"
    nb_path.write_text(
        json.dumps(build_notebook(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(BUNDLE_DIR.rglob("*")):
            if file_path.is_file() and "__pycache__" not in file_path.parts:
                arcname = Path("colab_bundle_v4") / file_path.relative_to(BUNDLE_DIR)
                zf.write(file_path, arcname=str(arcname))

    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"Built Colab bundle at {BUNDLE_DIR} and archive {ZIP_PATH} ({size_mb:.2f} MB)", flush=True)


if __name__ == "__main__":
    pack_bundle()
