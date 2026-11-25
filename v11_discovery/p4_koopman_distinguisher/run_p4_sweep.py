"""Full P4 Sweep Runner: KoopmanCrystalNet3D across R in {4,6,8,10,12,16} & 5 Tuple Families."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .eval_distill import (
    distill_koopman_model,
    evaluate_logits_pooled,
    evaluate_zero_param_baselines,
    predict_model_logits,
)
from .models import KoopmanCrystalNet3D
from .sha256_tuples import (
    DEFAULT_CONTROL_CONFIG,
    FAMILIES,
    FAMILY_C_IN,
    FAMILY_K,
    enforce_fp32_determinism,
    encode_crystal_features,
    generate_tuple_batch,
)


def train_and_eval_one(
    rounds: int,
    family: str,
    control: bool,
    steps: int,
    extend_steps: int,
    eval_n: int,
    pool_k: int,
    seed: int,
    device: torch.device,
    sr_eval_shared: torch.Tensor | None = None,
    c_eval_shared: torch.Tensor | None = None,
    labels_eval_shared: torch.Tensor | None = None,
) -> Tuple[KoopmanCrystalNet3D, Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Train one KoopmanCrystalNet3D run and evaluate on independent holdout tuples."""
    t0 = time.time()
    c_in = FAMILY_C_IN[family]
    batch_size = 256

    torch.manual_seed(seed)
    gen_train = torch.Generator(device=device)
    gen_train.manual_seed(seed + 101)

    model = KoopmanCrystalNet3D(c_in=c_in, d_model=32, n_blocks=4, num_heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, steps), eta_min=1e-5
    )
    criterion = nn.BCEWithLogitsLoss()

    final_loss = 0.0
    model.train()
    for step in range(1, steps + 1):
        _, c_tuples, labels = generate_tuple_batch(
            family=family,
            batch_size=batch_size,
            rounds=rounds,
            device=device,
            generator=gen_train,
            control=control,
        )
        psi = encode_crystal_features(c_tuples, family)
        optimizer.zero_grad(set_to_none=True)
        logits = model(psi)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        final_loss = float(loss.item())

    # Generate independent holdout set (or reuse shared holdout set for same domain)
    if sr_eval_shared is None or c_eval_shared is None or labels_eval_shared is None:
        gen_eval = torch.Generator(device=device)
        gen_eval.manual_seed(seed + 500003)
        sr_eval, c_eval, labels_eval = generate_tuple_batch(
            family=family,
            batch_size=eval_n,
            rounds=rounds,
            device=device,
            generator=gen_eval,
            control=control,
        )
    else:
        sr_eval, c_eval, labels_eval = sr_eval_shared, c_eval_shared, labels_eval_shared

    labels_eval_np = labels_eval.detach().cpu().numpy().astype(np.float64)
    logits_eval = predict_model_logits(model, c_eval, family, batch_size=1024)
    metrics = evaluate_logits_pooled(logits_eval, labels_eval_np, pool_k=pool_k)

    extended = False
    if rounds >= 6 and extend_steps > 0 and metrics["z_null"] >= 4.0:
        extended = True
        opt_ext = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        sched_ext = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_ext, T_max=extend_steps, eta_min=1e-5
        )
        model.train()
        for _ in range(extend_steps):
            _, c_tuples, labels = generate_tuple_batch(
                family=family,
                batch_size=batch_size,
                rounds=rounds,
                device=device,
                generator=gen_train,
                control=control,
            )
            psi = encode_crystal_features(c_tuples, family)
            opt_ext.zero_grad(set_to_none=True)
            logits = model(psi)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_ext.step()
            sched_ext.step()
            final_loss = float(loss.item())

        logits_eval = predict_model_logits(model, c_eval, family, batch_size=1024)
        metrics = evaluate_logits_pooled(logits_eval, labels_eval_np, pool_k=pool_k)

    wall_sec = float(time.time() - t0)
    metrics["final_train_loss"] = final_loss
    metrics["extended"] = extended
    metrics["wall_sec"] = wall_sec
    return model, metrics, sr_eval, c_eval, labels_eval, logits_eval


def run_sweep(
    out_dir: Path,
    rounds_list: List[int],
    families_list: List[str],
    steps: int = 1000,
    r4_steps: int = 400,
    extend_steps: int = 2000,
    eval_n: int = 65536,
    pool_k: int = 4096,
    cal_n: int = 32768,
    base_seed: int = 20261125,
    force_cpu: bool = False,
) -> Dict[str, Any]:
    enforce_fp32_determinism()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if (not force_cpu and torch.cuda.is_available()) else "CPU"
    tf32_disabled = (
        (not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32)
        if device.type == "cuda"
        else True
    )

    print(
        f"[P4-Init] device={device} ({gpu_name}), fp32_locked=True, tf32_disabled={tf32_disabled}, "
        f"rounds={rounds_list}, families={families_list}, r4_steps={r4_steps}, steps={steps}, "
        f"extend_steps={extend_steps}, batch_size=256, d_model=32, eval_n={eval_n}",
        flush=True,
    )

    t_start = time.time()
    summary: Dict[str, Any] = {
        "meta": {
            "gpu_name": gpu_name,
            "fp32_locked": True,
            "tf32_disabled": bool(tf32_disabled),
            "rounds": rounds_list,
            "families": families_list,
            "r4_steps": int(r4_steps),
            "steps": int(steps),
            "extend_steps": int(extend_steps),
            "batch_size": 256,
            "d_model": 32,
            "n_blocks": 4,
            "eval_n": int(eval_n),
            "eval_pool_k": int(pool_k),
            "eval_num_pools": int(eval_n // pool_k),
            "cal_n": int(cal_n),
            "control_config": {
                "k_seed": DEFAULT_CONTROL_CONFIG["k_seed"],
                "rotr_offsets": DEFAULT_CONTROL_CONFIG["rotr_offsets"],
            },
            "total_wall_sec": 0.0,
        },
        "results": {},
    }
    summary_path = out_dir / "p4_sweep_summary.json"

    for r_idx, rounds in enumerate(rounds_list):
        r_key = str(rounds)
        summary["results"][r_key] = {}
        run_steps = r4_steps if rounds == 4 else steps
        for f_idx, family in enumerate(families_list):
            pair_seed = base_seed + r_idx * 100000 + f_idx * 10000

            # 1) Main SHA-256 run
            model_main, main_m, sr_eval_main, c_eval_main, lbl_eval_main, logits_main = (
                train_and_eval_one(
                    rounds=rounds,
                    family=family,
                    control=False,
                    steps=run_steps,
                    extend_steps=extend_steps,
                    eval_n=eval_n,
                    pool_k=pool_k,
                    seed=pair_seed,
                    device=device,
                )
            )

            # 2) Isomorphic Random ROTR/K Control run
            model_ctrl, ctrl_m, _, _, _, _ = train_and_eval_one(
                rounds=rounds,
                family=family,
                control=True,
                steps=run_steps,
                extend_steps=extend_steps,
                eval_n=eval_n,
                pool_k=pool_k,
                seed=pair_seed + 5000,
                device=device,
            )
            del model_ctrl

            # 3) Zero-param & DL baselines on independent calibration + same main holdout
            gen_cal = torch.Generator(device=device)
            gen_cal.manual_seed(pair_seed + 777777)
            sr_cal, c_cal, lbl_cal = generate_tuple_batch(
                family=family,
                batch_size=cal_n,
                rounds=rounds,
                device=device,
                generator=gen_cal,
                control=False,
            )
            zp_m = evaluate_zero_param_baselines(
                sr_cal=sr_cal,
                labels_cal=lbl_cal,
                sr_eval=sr_eval_main,
                labels_eval=lbl_eval_main,
                family=family,
            )

            # 4) Bidirectional comparison metrics
            delta_auc = float(main_m["auc"] - ctrl_m["auc"])
            se_comb = math.sqrt(
                main_m["auc_pool_se"] ** 2 + ctrl_m["auc_pool_se"] ** 2 + 1e-12
            )
            z_ctrl = float(delta_auc / se_comb)
            best_zp_auc = max(
                zp_m["zero_param_hw_auc"],
                zp_m["dl_single_bit_auc"],
                zp_m["dl_linear_auc"],
            )
            comp_m = {
                "delta_auc": delta_auc,
                "z_ctrl": z_ctrl,
                "delta_vs_zero_param": float(main_m["auc"] - best_zp_auc),
            }

            # 5) Spectral Distillation if triggered
            trigger_distill = (
                (rounds == 4 and family in ("F1_xor2", "F3_rx2"))
                or (main_m["z_null"] >= 4.0)
                or (abs(z_ctrl) >= 3.0)
            )
            distill_m: Dict[str, Any] | None = None
            if trigger_distill:
                lbl_eval_np = lbl_eval_main.detach().cpu().numpy().astype(np.float64)
                distill_m = distill_koopman_model(
                    model=model_main,
                    family=family,
                    rounds=rounds,
                    sr_fit=sr_cal,
                    c_fit=c_cal,
                    sr_eval=sr_eval_main,
                    c_eval=c_eval_main,
                    labels_eval=lbl_eval_np,
                    z_eval=logits_main,
                )
                ckpt_path = out_dir / f"ckpt_R{rounds}_{family}_main.pt"
                torch.save(model_main.state_dict(), ckpt_path)

            del model_main
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            summary["results"][r_key][family] = {
                "main": main_m,
                "control": {k: v for k, v in ctrl_m.items() if k != "z_acc_null"},
                "zero_param_baseline": zp_m,
                "comparison": comp_m,
                "distillation": distill_m,
            }
            summary["meta"]["total_wall_sec"] = float(time.time() - t_start)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            distill_tag = (
                f" | W_R2={distill_m['walsh_r2']:.3f} W_AUC={distill_m['walsh_surrogate_auc']:.4f} "
                f"2A_R2={distill_m['two_adic_r2']:.3f} 2A_AUC={distill_m['two_adic_surrogate_auc']:.4f}"
                if distill_m is not None
                else ""
            )
            print(
                f"[R={rounds:2d} {family:8s}] "
                f"main_AUC={main_m['auc']:.5f} (z_null={main_m['z_null']:+.2f}, ext={main_m['extended']}) | "
                f"ctrl_AUC={ctrl_m['auc']:.5f} (z={ctrl_m['z_null']:+.2f}) | "
                f"z_ctrl={z_ctrl:+.2f} | zp_lin={zp_m['dl_linear_auc']:.5f} | "
                f"wall={main_m['wall_sec']+ctrl_m['wall_sec']:.1f}s{distill_tag}",
                flush=True,
            )

    summary["meta"]["total_wall_sec"] = float(time.time() - t_start)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[P4-Done] Total wall_sec={summary['meta']['total_wall_sec']:.2f}s -> {summary_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 3D Crystal Koopman Distinguisher Sweep")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--rounds", type=str, default="4,6,8,10,12,16")
    parser.add_argument("--families", type=str, default=",".join(FAMILIES))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--r4_steps", type=int, default=400)
    parser.add_argument("--extend_steps", type=int, default=2000)
    parser.add_argument("--eval_n", type=int, default=65536)
    parser.add_argument("--pool_k", type=int, default=4096)
    parser.add_argument("--cal_n", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()

    rounds_list = [int(x.strip()) for x in args.rounds.split(",") if x.strip()]
    families_list = [x.strip() for x in args.families.split(",") if x.strip()]

    run_sweep(
        out_dir=Path(args.out_dir),
        rounds_list=rounds_list,
        families_list=families_list,
        steps=args.steps,
        r4_steps=args.r4_steps,
        extend_steps=args.extend_steps,
        eval_n=args.eval_n,
        pool_k=args.pool_k,
        cal_n=args.cal_n,
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()
