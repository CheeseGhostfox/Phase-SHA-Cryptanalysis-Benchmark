"""
R2 只读诊断总控脚本：顺序调度 4 个诊断子脚本，校验存档 SHA-256 只读完整性，
检查 verify_preimage 100% 门控，并原子输出 diagnostics_summary.json。
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

from v10_ouroboros64.ouroboros_core.engine import enforce_fp32_determinism


ARCHIVE_DIRS: Tuple[str, ...] = (
    "v10_ouroboros64/runs/a100_m2",
    "v10_ouroboros64/runs/a100_m3",
    "v10_ouroboros64/runs/a100_m3v2",
)
DEFAULT_OUT_JSON = "v10_ouroboros64/diagnostics/diagnostics_summary.json"

SUB_DIAGNOSTICS: Tuple[Tuple[str, str, str], ...] = (
    (
        "domain_gap",
        "v10_ouroboros64.diagnostics.diag_domain_gap",
        "v10_ouroboros64/diagnostics/domain_gap_report.json",
    ),
    (
        "dyn_cdcl",
        "v10_ouroboros64.diagnostics.diag_dyn_cdcl_degradation",
        "v10_ouroboros64/diagnostics/dyn_cdcl_degradation_report.json",
    ),
    (
        "xor_cliff",
        "v10_ouroboros64.diagnostics.diag_xor_cliff",
        "v10_ouroboros64/diagnostics/diag_xor_cliff_report.json",
    ),
    (
        "r64_wormhole",
        "v10_ouroboros64.diagnostics.diag_r64_wormhole_stall",
        "v10_ouroboros64/diagnostics/diag_r64_wormhole_stall_report.json",
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


def _build_cross_summary(reports: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    dg = reports.get("domain_gap")
    if dg and "models" in dg and "best_m3v2" in dg["models"]:
        m3 = dg["models"]["best_m3v2"]
        summary["domain_gap"] = {
            "m3v2_pair_medium_acc": m3["pair_entanglement_vs_copy_dummy"]["domain_A_R64_medium"]["overall_model_pair_acc"],
            "m3v2_pair_medium_copy_dummy_acc": m3["pair_entanglement_vs_copy_dummy"]["domain_A_R64_medium"]["overall_copy_cd_dummy_acc"],
            "m3v2_pair_medium_flipped_acc": m3["pair_entanglement_vs_copy_dummy"]["domain_A_R64_medium"]["corrupted_flipped_pairs"]["model_acc"],
            "domain_b_r24_40b_phase_bce": m3["phase_and_collapse"]["domain_B_R24_40b"]["bce_loss_mean"],
            "domain_b_r24_40b_phase_acc": m3["phase_and_collapse"]["domain_B_R24_40b"]["phase_accuracy"],
            "domain_a_zero_defects_false_wolff_hit": m3.get("ablation_domain_a_zero_defects", {}).get("zero_defects_false_intact", {}).get("wolff_any_reducing_hit_rate"),
            "domain_a_zero_defects_true_wolff_hit": m3.get("ablation_domain_a_zero_defects", {}).get("zero_defects_true_ablated", {}).get("wolff_any_reducing_hit_rate"),
        }

    dc = reports.get("dyn_cdcl")
    if dc and "tiers" in dc:
        r26_probe: Optional[Dict[str, Any]] = None
        for t in dc["tiers"]:
            if t["rounds"] == 26 and t["free_bits"] == 16:
                full_recs = t["per_case_records"]["full"]
                r26_probe = {
                    "cases": t["cases"],
                    "cube_attempts_triggered_cases": sum(
                        1 for r in full_recs if r.get("cube_attempts", 0) > 0
                    ),
                    "total_cube_attempts": sum(int(r.get("cube_attempts", 0)) for r in full_recs),
                    "total_cube_unsolved_wasted": sum(
                        int(r.get("cube_unsolved_wasted", 0)) for r in full_recs
                    ),
                    "pure_mean_confl": t["summary"]["pure_cadical"]["mean_conflicts"],
                    "full_mean_confl": t["summary"]["full"]["mean_conflicts"],
                    "disable_cube_mean_confl": t["summary"]["disable_cube"]["mean_conflicts"],
                    "no_midsearch_set_phases_mean_confl": t["summary"]["no_midsearch_set_phases"]["mean_conflicts"],
                    "zero_heatmap_mean_confl": t["summary"]["zero_heatmap"]["mean_conflicts"],
                    "no_any_set_phases_mean_confl": t["summary"]["no_any_set_phases"]["mean_conflicts"],
                }
        summary["dyn_cdcl_degradation"] = {
            "verify_pass_rate": dc.get("verify_pass_rate"),
            "tiers": [
                {
                    "rounds": t["rounds"],
                    "free_bits": t["free_bits"],
                    "cases": t["cases"],
                    "pure_mean_confl": t["summary"]["pure_cadical"]["mean_conflicts"],
                    "full_mean_confl": t["summary"]["full"]["mean_conflicts"],
                    "disable_cube_mean_confl": t["summary"]["disable_cube"]["mean_conflicts"],
                    "no_midsearch_set_phases_mean_confl": t["summary"]["no_midsearch_set_phases"]["mean_conflicts"],
                    "zero_heatmap_mean_confl": t["summary"]["zero_heatmap"]["mean_conflicts"],
                    "no_any_set_phases_mean_confl": t["summary"]["no_any_set_phases"]["mean_conflicts"],
                }
                for t in dc["tiers"]
            ],
            "r26_16b_cube_probe": r26_probe,
        }

    xc = reports.get("xor_cliff")
    if xc and "tiers" in xc:
        summary["xor_cliff"] = {
            "verify_pass_rate": xc.get("verify_pass_rate"),
            "tiers": [
                {
                    "rounds": t["rounds"],
                    "free_bits": t["free_bits"],
                    "dual_unknown_after": t["dual_plane_after_prop"]["unknown_after_mean"],
                    "dual_active_carry_gates": t["dual_plane_after_prop"]["active_carry_gates_mean"],
                    "gf2_residual_dof": t["dual_plane_after_prop"]["gf2_residual_dof_mean"],
                    "cms_xor_sat": f"{t['solver_cliff_empirical']['cms_xor_sat']}/{t['cases']}",
                    "cadical_sat": f"{t['solver_cliff_empirical']['cadical_sat']}/{t['cases']}",
                }
                for t in xc["tiers"]
            ],
        }

    rw = reports.get("r64_wormhole")
    if rw:
        b0 = rw["baseline_best_m3v2_cases"][0] if rw.get("baseline_best_m3v2_cases") else {}
        summary["r64_wormhole_stall"] = {
            "baseline_case0_drop": {
                "unk_drop": b0.get("unk_drop"),
                "gate_drop": b0.get("gate_drop"),
                "splice_drop": b0.get("splice_drop"),
                "pruned_conflicts": b0.get("pruned_conflicts"),
            },
            "force_gf2_root_dt_sec": rw["counterexample_force_gf2"]["root_single_call"]["elapsed_sec"],
            "force_gf2_root_ded": rw["counterexample_force_gf2"]["root_single_call"]["gf2_deductions"],
            "force_gf2_3s_sims": rw["counterexample_force_gf2"]["mcts_3s_budget"]["simulations"],
            "no_macro_unk_drop": rw["counterexample_disable_macro_and_splice_block"]["disable_all_macro_actions"]["unk_drop"],
        }
    return summary


def run_all(
    fast: bool = False,
    only: Optional[str] = None,
    verify_archive_ro: bool = True,
    out_json: str = DEFAULT_OUT_JSON,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    enforce_fp32_determinism()
    cuda_avail = torch.cuda.is_available()
    allow_tf32_matmul = bool(torch.backends.cuda.matmul.allow_tf32) if cuda_avail else False
    allow_tf32_cudnn = bool(torch.backends.cudnn.allow_tf32) if cuda_avail else False
    assert (not allow_tf32_matmul) and (not allow_tf32_cudnn), "TF32 必须处于关闭状态！"

    repo_root = Path(__file__).resolve().parents[2]
    out_p = Path(out_json)
    out_abs = out_p if out_p.is_absolute() else (repo_root / out_p)
    default_out_abs = (repo_root / DEFAULT_OUT_JSON).resolve()
    is_isolated_out = out_abs.resolve() != default_out_abs
    out_abs.parent.mkdir(parents=True, exist_ok=True)

    selected_keys = (
        {x.strip() for x in only.split(",") if x.strip()}
        if only
        else {k for k, _, _ in SUB_DIAGNOSTICS}
    )

    print(
        f"[run_all_diagnostics] 启动 | python={sys.executable} | cuda={cuda_avail} | "
        f"allow_tf32={allow_tf32_matmul} | fast={fast} | isolated_out={is_isolated_out} | "
        f"selected={sorted(selected_keys)}",
        flush=True,
    )

    snap_before: Dict[str, Dict[str, Any]] = {}
    if verify_archive_ro:
        print("[run_all_diagnostics] 计算基线存档执行前 SHA-256 快照...", flush=True)
        snap_before = _snapshot_archive_hashes(repo_root)
        print(f"[run_all_diagnostics] 快照完成，共 {len(snap_before)} 个只读存档文件。", flush=True)

    sub_runs: List[Dict[str, Any]] = []
    loaded_reports: Dict[str, Optional[Dict[str, Any]]] = {}

    for key, mod_name, sub_json_rel in SUB_DIAGNOSTICS:
        if key not in selected_keys:
            loaded_reports[key] = None
            continue
        if is_isolated_out:
            sub_json_abs = out_abs.parent / Path(sub_json_rel).name
            report_json_ref = str(sub_json_abs)
        else:
            sub_json_abs = repo_root / sub_json_rel
            report_json_ref = sub_json_rel
        cmd = [sys.executable, "-u", "-m", mod_name, "--out_json", str(sub_json_abs)]
        if fast:
            cmd.append("--fast")
        print(f"\n[run_all_diagnostics] >>> 执行 {key}: {' '.join(cmd)}", flush=True)
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
                "report_json": report_json_ref,
            }
        )
        if proc.returncode != 0:
            raise RuntimeError(f"子诊断脚本 {mod_name} 失败，退出码={proc.returncode}")
        if not sub_json_abs.exists():
            raise FileNotFoundError(f"未找到子诊断输出文件: {sub_json_abs}")
        loaded_reports[key] = json.loads(sub_json_abs.read_text(encoding="utf-8"))

    archive_integrity: Dict[str, Any] = {"checked": bool(verify_archive_ro)}
    if verify_archive_ro:
        print("\n[run_all_diagnostics] 校验基线存档执行后 SHA-256 零变动...", flush=True)
        snap_after = _snapshot_archive_hashes(repo_root)
        changed = []
        for f_rel, meta_b in snap_before.items():
            meta_a = snap_after.get(f_rel)
            if meta_a != meta_b:
                changed.append({"file": f_rel, "before": meta_b, "after": meta_a})
        extra = [f for f in snap_after if f not in snap_before]
        assert not changed and not extra, f"基线只读存档发生变动! changed={changed}, extra={extra}"
        archive_integrity.update(
            {
                "files_checked": len(snap_before),
                "all_unchanged": True,
                "changed_files": [],
                "extra_files": [],
                "file_hashes": snap_before,
            }
        )
        print(f"[run_all_diagnostics] 存档完整性校验通过：{len(snap_before)} 个文件 SHA-256 100% 一致。", flush=True)

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
            "sub_runs": sub_runs,
            "archive_integrity": archive_integrity,
        },
        "cross_dimension_summary": _build_cross_summary(loaded_reports),
        "domain_gap": loaded_reports.get("domain_gap"),
        "dyn_cdcl_degradation": loaded_reports.get("dyn_cdcl"),
        "xor_cliff": loaded_reports.get("xor_cliff"),
        "r64_wormhole_stall": loaded_reports.get("r64_wormhole"),
    }
    _atomic_write_json(out_abs, summary_payload)
    print(f"\n[run_all_diagnostics] 全部完成！总耗时 {total_elapsed:.2f}s，汇总报告写入 {out_abs}", flush=True)
    return summary_payload


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="R2 只读诊断套件总调度与存档完整性校验")
    p.add_argument("--fast", action="store_true", help="快速验证模式")
    p.add_argument("--only", type=str, default=None, help="逗号分隔子脚本: domain_gap,dyn_cdcl,xor_cliff,r64_wormhole")
    p.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    p.add_argument("--no_verify_archive_ro", action="store_true", help="跳过基线存档 SHA-256 校验")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_all(
        fast=bool(args.fast),
        only=args.only,
        verify_archive_ro=(not args.no_verify_archive_ro),
        out_json=args.out_json,
    )
