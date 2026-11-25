"""
R4-3 总控核验运行器：
  1. 顺序执行 verify_symbolic_sympy、verify_smt_z3、verify_torch_bounds；
  2. 校验全部子脚本退出码为 0；
  3. 对 runs/a100_m2、a100_m3、a100_m3v2 全部 21 个基线存档与检查点文件执行前后 SHA-256 比对，
     并与 diagnostics_summary.json 中记录的基线哈希逐一核对，确保 100% 只读零变动。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v10_ouroboros64.formal_checks import ARCHIVE_DIRS, DEFAULT_DIAG_SUMMARY_REL
from v10_ouroboros64.ouroboros_core.engine import enforce_fp32_determinism


DEFAULT_OUT_JSON = "v10_ouroboros64/formal_checks/formal_checks_summary.json"

SUB_CHECKS: Tuple[Tuple[str, str, str], ...] = (
    (
        "sympy",
        "v10_ouroboros64.formal_checks.verify_symbolic_sympy",
        "v10_ouroboros64/formal_checks/sympy_report.json",
    ),
    (
        "z3",
        "v10_ouroboros64.formal_checks.verify_smt_z3",
        "v10_ouroboros64/formal_checks/z3_report.json",
    ),
    (
        "torch_bounds",
        "v10_ouroboros64.formal_checks.verify_torch_bounds",
        "v10_ouroboros64/formal_checks/torch_bounds_report.json",
    ),
)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _snapshot_archive_hashes(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    snap: Dict[str, Dict[str, Any]] = {}
    for rel_dir in ARCHIVE_DIRS:
        dpath = repo_root / rel_dir
        if not dpath.exists():
            continue
        for fpath in sorted(dpath.rglob("*")):
            if fpath.is_file():
                rel = fpath.relative_to(repo_root).as_posix()
                snap[rel] = {
                    "size": int(fpath.stat().st_size),
                    "sha256": _sha256_file(fpath),
                }
    return snap


def run_all(
    fast: bool = False,
    verify_archive_ro: bool = True,
    out_json: str = DEFAULT_OUT_JSON,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    enforce_fp32_determinism()
    cuda_avail = torch.cuda.is_available()
    allow_tf32_matmul = bool(torch.backends.cuda.matmul.allow_tf32) if cuda_avail else False
    allow_tf32_cudnn = bool(torch.backends.cudnn.allow_tf32) if cuda_avail else False
    assert (not allow_tf32_matmul) and (not allow_tf32_cudnn), "TF32 必须处于关闭状态"

    repo_root = Path(__file__).resolve().parents[2]
    out_p = Path(out_json)
    out_abs = out_p if out_p.is_absolute() else (repo_root / out_p)
    default_out_abs = (repo_root / DEFAULT_OUT_JSON).resolve()
    is_isolated_out = out_abs.resolve() != default_out_abs
    out_abs.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[run_formal_checks] 启动 | python={sys.executable} | cuda={cuda_avail} | "
        f"allow_tf32=({allow_tf32_matmul}, {allow_tf32_cudnn}) | fast={fast}",
        flush=True,
    )

    snap_before: Dict[str, Dict[str, Any]] = {}
    if verify_archive_ro:
        snap_before = _snapshot_archive_hashes(repo_root)
        assert len(snap_before) == 21, f"预期 21 个基线存档文件，实际发现 {len(snap_before)}"
        diag_path = repo_root / DEFAULT_DIAG_SUMMARY_REL
        if diag_path.exists():
            diag_data = json.loads(diag_path.read_text(encoding="utf-8"))
            diag_hashes = diag_data.get("meta", {}).get("archive_integrity", {}).get("file_hashes", {})
            if diag_hashes:
                assert snap_before == diag_hashes, "执行前基线存档哈希与 M1 diagnostics_summary.json 不一致!"
        print(f"[run_formal_checks] 执行前 21 个基线存档 SHA-256 核验通过。", flush=True)

    sub_runs: List[Dict[str, Any]] = []
    loaded_reports: Dict[str, Any] = {}

    for key, mod_name, sub_json_rel in SUB_CHECKS:
        if is_isolated_out:
            sub_json_abs = out_abs.parent / Path(sub_json_rel).name
            report_ref = str(sub_json_abs)
        else:
            sub_json_abs = repo_root / sub_json_rel
            report_ref = sub_json_rel

        cmd = [sys.executable, "-u", "-m", mod_name, "--out_json", str(sub_json_abs)]
        if fast:
            cmd.append("--fast")
        print(f"\n[run_formal_checks] >>> 执行 {key}: {' '.join(cmd)}", flush=True)
        ts = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
        dt = round(time.perf_counter() - ts, 3)
        sub_runs.append(
            {
                "key": key,
                "module": mod_name,
                "cmd": cmd,
                "returncode": int(proc.returncode),
                "elapsed_sec": dt,
                "report_json": report_ref,
            }
        )
        if proc.returncode != 0:
            raise RuntimeError(f"核验脚本 {mod_name} 退出码非零: {proc.returncode}")
        if not sub_json_abs.exists():
            raise FileNotFoundError(f"未找到输出文件: {sub_json_abs}")
        loaded_reports[key] = json.loads(sub_json_abs.read_text(encoding="utf-8"))
        assert loaded_reports[key].get("all_passed") is True, f"{mod_name} all_passed != True"

    archive_integrity: Dict[str, Any] = {"checked": bool(verify_archive_ro)}
    if verify_archive_ro:
        snap_after = _snapshot_archive_hashes(repo_root)
        changed = [
            {"file": k, "before": v, "after": snap_after.get(k)}
            for k, v in snap_before.items()
            if snap_after.get(k) != v
        ]
        extra = [k for k in snap_after if k not in snap_before]
        assert not changed and not extra, f"基线存档被修改! changed={changed}, extra={extra}"
        archive_integrity.update(
            {
                "files_checked": len(snap_before),
                "all_unchanged": True,
                "changed_files": [],
                "extra_files": [],
                "file_hashes": snap_before,
            }
        )
        print(f"[run_formal_checks] 执行后 21 个基线存档 SHA-256 100% 零变动。", flush=True)

    total_elapsed = round(time.perf_counter() - t0, 3)
    summary_payload: Dict[str, Any] = {
        "meta": {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fast_mode": bool(fast),
            "python_executable": sys.executable,
            "device": "cuda" if cuda_avail else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU",
            "allow_tf32_matmul": allow_tf32_matmul,
            "allow_tf32_cudnn": allow_tf32_cudnn,
            "fp32_locked": (not allow_tf32_matmul) and (not allow_tf32_cudnn),
            "total_elapsed_sec": total_elapsed,
            "all_exit_zero": all(r["returncode"] == 0 for r in sub_runs),
            "sub_runs": sub_runs,
            "archive_integrity": archive_integrity,
        },
        "sympy": loaded_reports.get("sympy"),
        "z3": loaded_reports.get("z3"),
        "torch_bounds": loaded_reports.get("torch_bounds"),
    }
    _atomic_write_json(out_abs, summary_payload)
    print(f"\n[run_formal_checks] 全套形式化数值/符号核验通过! 总耗时 {total_elapsed:.2f}s，输出: {out_abs}", flush=True)
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="R4-3 SymPy/Z3/PyTorch 形式化核验总控脚本")
    parser.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no_verify_archive_ro", action="store_true")
    args = parser.parse_args()
    run_all(
        fast=bool(args.fast),
        verify_archive_ro=(not args.no_verify_archive_ro),
        out_json=args.out_json,
    )


if __name__ == "__main__":
    main()
