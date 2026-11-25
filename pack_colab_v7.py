"""Build self-contained V7 Colab upload bundle (`colab_bundle_v7/` & `.zip`).

Zero OAuth tokens, credentials, local runs, or `checkpoints/` are packed.
"""

from __future__ import annotations

import os
import shutil
import stat
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "colab_bundle_v7"
ZIP_OUT = ROOT / "colab_bundle_v7.zip"

FIXED_FILES = [
    "v7/__init__.py",
    "v7/v7_blocks.py",
    "v7/v7_arena.py",
    "v7/v7a_autoregressive.py",
    "v7/v7b_looped_rl.py",
    "v7/univ_arx_vm.py",
    "v7/v7c_levin_meta.py",
    "configs/v7a_colab.yaml",
    "configs/v7b_colab.yaml",
    "configs/v7c_colab.yaml",
    "train_v7.py",
    "scripts/export_v6a_corpus.py",
    "scripts/colab_monitor_dashboard.py",
    "tests/test_v7.py",
]


def _force_remove_readonly(func, path, exc_info) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def build_bundle(root: Path = ROOT, stage: Path = STAGE, zip_out: Path = ZIP_OUT) -> Path:
    if stage.exists():
        shutil.rmtree(stage, onerror=_force_remove_readonly)
    stage.mkdir(parents=True, exist_ok=True)

    rel_files = list(FIXED_FILES)
    # Include src/*.py and v6a/*.py cleanly (never checkpoints/runs/credentials)
    for pkg in ("src", "v6a"):
        pkg_dir = root / pkg
        for p in sorted(pkg_dir.glob("*.py")):
            rel_files.append(f"{pkg}/{p.name}")

    missing = [rel for rel in FIXED_FILES if not (root / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files for V7 Colab bundle: {missing}")

    for rel in rel_files:
        src = root / rel
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    if zip_out.exists():
        zip_out.unlink()
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(stage.rglob("*")):
            if file_path.is_file() and "__pycache__" not in file_path.parts:
                zf.write(file_path, arcname=file_path.relative_to(stage).as_posix())

    size_kb = zip_out.stat().st_size / 1024.0
    print(f"[pack_colab_v7] built {zip_out} ({size_kb:.1f} KB, {len(rel_files)} files)", flush=True)
    return zip_out


if __name__ == "__main__":
    build_bundle()
