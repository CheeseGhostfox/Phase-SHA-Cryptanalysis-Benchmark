"""Main runner for P7 Dual-Group Mixed-Basis Sparsification Telescope."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from cipher_ops import assert_fp32_locked, enforce_fp32_lock
from probes import P7Config, run_probe1, run_probe2, run_probe3


def _collect_conservation_errors(summary: Dict[str, Any]) -> Dict[str, float]:
    ortho_errs: List[float] = []
    det_errs: List[float] = []
    norm_errs: List[float] = []

    for r_data in summary.get("probe1_bit_slice", {}).values():
        for s_data in r_data.values():
            for c_data in s_data.values():
                lrn = c_data["learned"]
                ortho_errs.append(float(lrn["ortho_err"]))
                det_errs.append(float(lrn["det_err"]))
                norm_errs.append(float(lrn["norm_err"]))

    for r_data in summary.get("probe2_io_coupling", {}).values():
        for c_data in r_data.values():
            lrn = c_data["learned"]
            ortho_errs.extend([float(lrn["ortho_err_L"]), float(lrn["ortho_err_R"])])
            det_errs.extend([float(lrn["det_err_L"]), float(lrn["det_err_R"])])
            norm_errs.append(float(lrn["norm_err"]))

    for r_data in summary.get("probe3_operator_transfer", {}).values():
        for t_data in r_data.values():
            for c_data in t_data.values():
                lrn = c_data["learned"]
                ortho_errs.append(float(lrn["ortho_err"]))
                det_errs.append(float(lrn["det_err"]))
                norm_errs.append(float(lrn["norm_err"]))

    return {
        "max_ortho_err": max(ortho_errs) if ortho_errs else 0.0,
        "max_det_err": max(det_errs) if det_errs else 0.0,
        "max_norm_err": max(norm_errs) if norm_errs else 0.0,
    }


def main() -> None:
    enforce_fp32_lock()
    assert_fp32_locked()

    parser = argparse.ArgumentParser(description="Run P7 Dual-Group Spectral Telescope")
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else Path(__file__).resolve().parent / "results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    use_cuda = torch.cuda.is_available() and (cuda_env is None or cuda_env.strip() != "")
    device = torch.device("cuda" if use_cuda else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"

    if args.mode == "smoke":
        cfg = P7Config(
            rounds=(2,),
            probe1_k_train=4,
            probe1_k_holdout=4,
            probe2_k_train=2,
            probe2_n_train=2048,
            probe2_k_holdout=2,
            probe2_n_pools=4,
            probe2_n_pool_samples=1024,
            probe3_k_train=2,
            probe3_k_holdout=4,
            opt_steps=20,
            chunk_size=2048,
        )
    else:
        cfg = P7Config()

    print(
        f"[P7] Starting mode={args.mode} device={device} gpu={gpu_name} "
        f"fp32_default={torch.get_default_dtype()} allow_tf32={torch.backends.cuda.matmul.allow_tf32}",
        flush=True,
    )

    t0 = time.time()

    print("[P7] Running Probe I (bit_slice_ensemble)...", flush=True)
    t_p1_start = time.time()
    p1_res, p1_dist = run_probe1(cfg, device=device)
    t_p1 = time.time() - t_p1_start
    print(f"[P7] Probe I finished in {t_p1:.2f}s", flush=True)

    print("[P7] Running Probe II (io_dual_coupling)...", flush=True)
    t_p2_start = time.time()
    p2_res, p2_dist = run_probe2(cfg, device=device)
    t_p2 = time.time() - t_p2_start
    print(f"[P7] Probe II finished in {t_p2:.2f}s", flush=True)

    print("[P7] Running Probe III (operator_transfer)...", flush=True)
    t_p3_start = time.time()
    p3_res, p3_dist = run_probe3(cfg, device=device)
    t_p3 = time.time() - t_p3_start
    print(f"[P7] Probe III finished in {t_p3:.2f}s", flush=True)

    total_wall = time.time() - t0

    summary: Dict[str, Any] = {
        "meta": {
            "mode": args.mode,
            "gpu_name": gpu_name,
            "fp32_locked": True,
            "tf32_disabled": (
                not torch.backends.cuda.matmul.allow_tf32
                and not torch.backends.cudnn.allow_tf32
            ),
            "complex_disabled": True,
            "rounds": list(cfg.rounds),
            "total_wall_sec": total_wall,
        },
        "probe1_bit_slice": p1_res,
        "probe2_io_coupling": p2_res,
        "probe3_operator_transfer": p3_res,
    }
    errs = _collect_conservation_errors(summary)
    summary["meta"].update(errs)

    conservation_passed = (
        errs["max_ortho_err"] <= 1e-4
        and errs["max_det_err"] <= 1e-4
        and errs["max_norm_err"] <= 1e-4
    )
    summary["meta"]["conservation_passed"] = conservation_passed

    distillation = {
        "probe1_bit_slice": p1_dist,
        "probe2_io_coupling": p2_dist,
        "probe3_operator_transfer": p3_dist,
    }

    run_meta = {
        "mode": args.mode,
        "gpu_name": gpu_name,
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "fp32_locked": True,
        "tf32_disabled": summary["meta"]["tf32_disabled"],
        "complex_disabled": True,
        "timing_sec": {
            "probe1": t_p1,
            "probe2": t_p2,
            "probe3": t_p3,
            "total": total_wall,
        },
        "conservation": {**errs, "passed": conservation_passed},
        "config": {
            "rounds": list(cfg.rounds),
            "probe1_k_train": cfg.probe1_k_train,
            "probe1_k_holdout": cfg.probe1_k_holdout,
            "probe2_k_train": cfg.probe2_k_train,
            "probe2_n_train": cfg.probe2_n_train,
            "probe2_k_holdout": cfg.probe2_k_holdout,
            "probe2_n_pools": cfg.probe2_n_pools,
            "probe2_n_pool_samples": cfg.probe2_n_pool_samples,
            "probe3_k_train": cfg.probe3_k_train,
            "probe3_k_holdout": cfg.probe3_k_holdout,
            "opt_steps": cfg.opt_steps,
            "lr_max": cfg.lr_max,
            "lr_min": cfg.lr_min,
            "seed": cfg.seed,
        },
    }

    (out_dir / "p7_sweep_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / "p7_distillation.json").write_text(
        json.dumps(distillation, indent=2), encoding="utf-8"
    )
    (out_dir / "p7_run_meta.json").write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )

    print(
        f"[P7] Completed in {total_wall:.2f}s | conservation={conservation_passed} "
        f"(ortho={errs['max_ortho_err']:.2e}, det={errs['max_det_err']:.2e}, norm={errs['max_norm_err']:.2e})",
        flush=True,
    )
    if not conservation_passed:
        raise RuntimeError(f"Orthogonal conservation gate failed: {errs}")


if __name__ == "__main__":
    main()
