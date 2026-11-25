"""Stage 1 audit & Step-0 physics-guided annealing baseline script."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import yaml

from src.anneal_solver import AnnealSolver
from src.boundary_clamp import build_clamped_boundary
from src.flow_trainer import FlowTrainer
from src.sha256_ref import configure_cuda_memory


def main() -> None:
    cfg_path = ROOT_DIR / "configs" / "r12_sanity.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        configure_cuda_memory(float(cfg.get("cuda_memory_fraction", 0.80)))

    out_dir = ROOT_DIR / "runs" / "r12_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = FlowTrainer(config=cfg, output_dir=out_dir, device=device)

    # 1. Domain & Anti-Leakage Audit
    num_bits = int(cfg["num_bits"])
    total_space = 1 << num_bits
    holdout_n = int(trainer.holdout_w16.shape[0])
    train_pool_n = (
        int(trainer.train_w0_pool.shape[0])
        if trainer.train_w0_pool is not None
        else total_space - holdout_n
    )
    overlap_count = (
        int(torch.isin(trainer.train_w0_pool, trainer.holdout_w0_gpu).sum().item())
        if (trainer.train_w0_pool is not None and trainer.holdout_w0_gpu is not None)
        else -1
    )
    anneal_k = int(cfg.get("anneal_K", 256))
    random_guess_rate = 1.0 - (1.0 - 1.0 / float(total_space)) ** anneal_k

    solve_sig = str(inspect.signature(AnnealSolver.solve))
    clamp_sig = str(inspect.signature(build_clamped_boundary))

    audit_info = {
        "rounds": int(cfg["rounds"]),
        "domain_mode": str(cfg["domain_mode"]),
        "num_bits": num_bits,
        "total_domain_cardinality": total_space,
        "train_pool_size": train_pool_n,
        "holdout_size": holdout_n,
        "train_holdout_overlap_count": overlap_count,
        "anneal_K": anneal_k,
        "anneal_eval_samples": int(cfg.get("anneal_eval_samples", 64)),
        "theoretical_random_preimage_rate_at_K": random_guess_rate,
        "anneal_solver_solve_signature": solve_sig,
        "build_clamped_boundary_signature": clamp_sig,
    }

    print("=== Stage 1 (r12_sanity) Domain & Leakage Audit ===", flush=True)
    for k, v in audit_info.items():
        print(f"  {k}: {v}", flush=True)

    assert overlap_count == 0, f"Leakage detected: train/holdout overlap={overlap_count}"
    assert "x_star" not in solve_sig and "w16" not in solve_sig
    assert "x_star" not in clamp_sig and "w16" not in clamp_sig

    # 2. Step-0 Baselines: (A) Pure physics-only (neural v_theta=0) vs (B) Untrained neural+physics
    print("\n=== Running Step-0 Baseline A: Physics-Only (use_neural_velocity=False) ===", flush=True)
    physics_only = trainer.evaluate_holdout(run_anneal=True, use_neural_velocity=False)
    print(
        f"  [Physics-Only] preimage@256={physics_only['exact_preimage_rate@256']:.4f} "
        f"w0_free_acc={physics_only['w0_free_bit_acc']:.4f} "
        f"E_m1={physics_only['phys_energy_m1']:.2f} "
        f"E_m32={physics_only['phys_energy_m32']:.2f}",
        flush=True,
    )
    pos_a = [round(physics_only[f"bit_acc_pos_{k}"], 4) for k in range(32)]
    print(f"  [Physics-Only] bit_acc[0..31]: {pos_a}", flush=True)

    print("\n=== Running Step-0 Baseline B: Untrained Neural+Physics (use_neural_velocity=True) ===", flush=True)
    untrained_hybrid = trainer.evaluate_holdout(run_anneal=True, use_neural_velocity=True)
    print(
        f"  [Step-0 Hybrid] preimage@256={untrained_hybrid['exact_preimage_rate@256']:.4f} "
        f"eval_fm={untrained_hybrid['eval_fm_loss']:.4f} "
        f"blind1_w0={untrained_hybrid['blind1_w0_bit_acc']:.4f} "
        f"w0_free_acc={untrained_hybrid['w0_free_bit_acc']:.4f} "
        f"E_m1={untrained_hybrid['phys_energy_m1']:.2f} "
        f"E_m32={untrained_hybrid['phys_energy_m32']:.2f}",
        flush=True,
    )
    pos_b = [round(untrained_hybrid[f"bit_acc_pos_{k}"], 4) for k in range(32)]
    print(f"  [Step-0 Hybrid] bit_acc[0..31]: {pos_b}", flush=True)

    payload = {
        "audit": audit_info,
        "physics_only": physics_only,
        "untrained_hybrid": untrained_hybrid,
    }
    json_path = out_dir / "step0_baseline.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved Step-0 audit & baselines to {json_path}", flush=True)


if __name__ == "__main__":
    main()
