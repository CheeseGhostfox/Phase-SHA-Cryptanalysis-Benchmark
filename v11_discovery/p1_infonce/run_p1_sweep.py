"""Main P1 Cross-Round InfoNCE Variational Mutual Information Sweep Runner."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .eval_distill import (
    distill_embeddings,
    evaluate_embeddings_pooled,
    evaluate_ridge_cca,
)
from .models import DualTowerInfoNCE, LinearInfoNCE, symmetric_infonce_loss
from .sha256_telescope import (
    ControlConfig,
    encode_multiscale_features,
    enforce_fp32_determinism,
    make_control_config,
    sample_uniform_w16,
    sha256_sr_control,
    sha256_sr_main,
)

DEFAULT_ROUNDS = (4, 6, 8, 10, 12, 16, 20)


def _train_infonce_model(
    model: nn.Module,
    rounds: int,
    trace_fn: Callable[[torch.Tensor, int], torch.Tensor],
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    log_prefix: str,
) -> Tuple[float, float, float]:
    """Train DualTowerInfoNCE or LinearInfoNCE on-the-fly in pure FP32."""
    model = model.to(device=device, dtype=torch.float32)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, steps), eta_min=lr * 0.05
    )

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    t0 = time.time()
    last_loss = 0.0
    log_interval = max(1, steps // 4)

    for step in range(1, steps + 1):
        with torch.no_grad():
            w16 = sample_uniform_w16(batch_size, device=device, generator=gen)
            sr = trace_fn(w16, rounds)
            phi_in = encode_multiscale_features(w16)
            phi_out = encode_multiscale_features(sr)

        optimizer.zero_grad(set_to_none=True)
        u, v, scale = model(phi_in, phi_out)
        loss = symmetric_infonce_loss(u, v, scale)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        last_loss = float(loss.item())
        if step % log_interval == 0 or step == steps:
            elapsed = time.time() - t0
            temp = float(1.0 / scale.item())
            print(
                f"  [{log_prefix}] step {step}/{steps} | loss={last_loss:.4f} | "
                f"tau={temp:.4f} | {elapsed:.1f}s",
                flush=True,
            )

    wall_sec = time.time() - t0
    learned_temp = float(1.0 / model.clamped_scale().item())
    return last_loss, learned_temp, wall_sec


@torch.no_grad()
def _encode_holdout(
    model: nn.Module,
    phi_in: torch.Tensor,
    phi_out: torch.Tensor,
    chunk_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Encode holdout features in chunks to bound peak VRAM."""
    model.eval()
    u_chunks: List[torch.Tensor] = []
    v_chunks: List[torch.Tensor] = []
    n = phi_in.shape[0]
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        u_c, v_c, _ = model(phi_in[start:end], phi_out[start:end])
        u_chunks.append(u_c)
        v_chunks.append(v_c)
    scale = float(model.clamped_scale().item())
    return torch.cat(u_chunks, dim=0), torch.cat(v_chunks, dim=0), scale


@torch.no_grad()
def _build_dataset_features(
    n_samples: int,
    rounds: int,
    trace_fn: Callable[[torch.Tensor, int], torch.Tensor],
    seed: int,
    device: torch.device,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic isolated (phi_in, phi_out) dataset on device."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    in_chunks: List[torch.Tensor] = []
    out_chunks: List[torch.Tensor] = []
    remaining = n_samples
    while remaining > 0:
        cur = min(chunk_size, remaining)
        w16 = sample_uniform_w16(cur, device=device, generator=gen)
        sr = trace_fn(w16, rounds)
        in_chunks.append(encode_multiscale_features(w16))
        out_chunks.append(encode_multiscale_features(sr))
        remaining -= cur
    return torch.cat(in_chunks, dim=0), torch.cat(out_chunks, dim=0)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def run_sweep(
    out_dir: Path,
    rounds_list: List[int],
    steps: int = 4000,
    batch_size: int = 1024,
    lr: float = 3e-4,
    eval_n: int = 16384,
    eval_pool_k: int = 1024,
    fit_n: int = 32768,
    perm_n: int = 64,
    hidden_dim: int = 512,
    emb_dim: int = 128,
    num_blocks: int = 4,
    control_seed: int = 20261125,
    save_weights: bool = True,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute full P1 sweep across rounds and write SPEC-compliant p1_sweep_summary.json."""
    enforce_fp32_determinism()
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        assert not torch.backends.cuda.matmul.allow_tf32, "TF32 matmul must be False"
        assert not torch.backends.cudnn.allow_tf32, "TF32 cudnn must be False"
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = "CPU"

    out_dir.mkdir(parents=True, exist_ok=True)
    control_cfg: ControlConfig = make_control_config(seed=control_seed)

    print(
        f"=== P1 InfoNCE Telescope Sweep ===\n"
        f"device={device} ({gpu_name}) | fp32_locked=True | rounds={rounds_list}\n"
        f"steps={steps} | batch_size={batch_size} | eval_n={eval_n} (K={eval_pool_k}) | fit_n={fit_n} | perm_n={perm_n}",
        flush=True,
    )

    t_global = time.time()
    summary: Dict[str, Any] = {
        "meta": {
            "gpu_name": gpu_name,
            "fp32_locked": True,
            "tf32_disabled": True,
            "batch_size": batch_size,
            "train_steps": steps,
            "eval_n": eval_n,
            "eval_pool_k": eval_pool_k,
            "eval_num_pools": eval_n // eval_pool_k,
            "fit_n": fit_n,
            "perm_n": perm_n,
            "feature_dims": {"in": 896, "out": 448},
            "model_config": {"hidden_dim": hidden_dim, "emb_dim": emb_dim, "num_blocks": num_blocks, "lr": lr},
            "control_config": control_cfg.to_meta_dict(),
            "total_wall_sec": 0.0,
        },
        "rounds": {},
    }

    summary_path = out_dir / "p1_sweep_summary.json"

    for r in rounds_list:
        t_r = time.time()
        print(f"\n--- Round R={r} ---", flush=True)

        # Build isolated holdout sets (main & control) and Ridge-CCA fit set (main)
        eval_seed = 900100 + r
        fit_seed = 800100 + r

        phi_in_eval, phi_out_eval_main = _build_dataset_features(
            eval_n, r, sha256_sr_main, seed=eval_seed, device=device
        )
        _, phi_out_eval_ctrl = _build_dataset_features(
            eval_n, r, lambda w, rnd: sha256_sr_control(w, rnd, control_cfg),
            seed=eval_seed, device=device
        )
        phi_in_fit, phi_out_fit_main = _build_dataset_features(
            fit_n, r, sha256_sr_main, seed=fit_seed, device=device
        )

        # 1. Main DualTowerInfoNCE on real SHA-256
        torch.manual_seed(100100 + r)
        model_main = DualTowerInfoNCE(896, 448, hidden_dim, emb_dim, num_blocks)
        main_loss, main_temp, main_wall = _train_infonce_model(
            model_main, r, sha256_sr_main, steps, batch_size, lr,
            seed=100100 + r, device=device, log_prefix=f"R={r} MAIN",
        )
        u_main, v_main, scale_main = _encode_holdout(model_main, phi_in_eval, phi_out_eval_main)
        eval_main = evaluate_embeddings_pooled(
            u_main, v_main, scale_main, pool_k=eval_pool_k, num_perm=perm_n, perm_seed=777000 + r
        )
        eval_main.update({
            "final_train_loss": main_loss,
            "learned_temp": main_temp,
            "wall_sec": main_wall,
        })

        # 2. Control DualTowerInfoNCE on random ROTR/K clone
        torch.manual_seed(200100 + r)
        model_ctrl = DualTowerInfoNCE(896, 448, hidden_dim, emb_dim, num_blocks)
        ctrl_fn = lambda w, rnd: sha256_sr_control(w, rnd, control_cfg)
        ctrl_loss, ctrl_temp, ctrl_wall = _train_infonce_model(
            model_ctrl, r, ctrl_fn, steps, batch_size, lr,
            seed=200100 + r, device=device, log_prefix=f"R={r} CTRL",
        )
        u_ctrl, v_ctrl, scale_ctrl = _encode_holdout(model_ctrl, phi_in_eval, phi_out_eval_ctrl)
        eval_ctrl = evaluate_embeddings_pooled(
            u_ctrl, v_ctrl, scale_ctrl, pool_k=eval_pool_k, num_perm=perm_n, perm_seed=787000 + r
        )
        eval_ctrl.update({
            "final_train_loss": ctrl_loss,
            "learned_temp": ctrl_temp,
            "wall_sec": ctrl_wall,
        })

        # 3. Linear InfoNCE baseline on real SHA-256
        torch.manual_seed(300100 + r)
        model_lin = LinearInfoNCE(896, 448, emb_dim)
        lin_loss, lin_temp, lin_wall = _train_infonce_model(
            model_lin, r, sha256_sr_main, steps, batch_size, lr,
            seed=300100 + r, device=device, log_prefix=f"R={r} LIN",
        )
        u_lin, v_lin, scale_lin = _encode_holdout(model_lin, phi_in_eval, phi_out_eval_main)
        eval_lin = evaluate_embeddings_pooled(
            u_lin, v_lin, scale_lin, pool_k=eval_pool_k, num_perm=0
        )

        # 4. Analytic Ridge-CCA baseline on real SHA-256
        ridge_res = evaluate_ridge_cca(
            phi_in_fit, phi_out_fit_main, phi_in_eval, phi_out_eval_main,
            reg=1e-3, emb_dim=emb_dim, pool_k=eval_pool_k,
        )

        linear_baseline = {
            "linear_infonce_mi_bits_mean": eval_lin["mi_bits_mean"],
            "linear_infonce_mi_bits_se": eval_lin["mi_bits_se"],
            "linear_infonce_top1_mean": eval_lin["top1_mean"],
            "linear_infonce_top5_mean": eval_lin["top5_mean"],
            "linear_infonce_train_loss": lin_loss,
            "linear_infonce_learned_temp": lin_temp,
            "linear_infonce_wall_sec": lin_wall,
            **ridge_res,
        }

        # 5. Comparisons
        delta_mi = eval_main["mi_bits_mean"] - eval_ctrl["mi_bits_mean"]
        z_ctrl = delta_mi / math.sqrt(
            eval_main["mi_bits_se"] ** 2 + eval_ctrl["mi_bits_se"] ** 2 + 1e-12
        )
        delta_vs_linear = eval_main["mi_bits_mean"] - eval_lin["mi_bits_mean"]
        comparison = {
            "delta_mi_bits": delta_mi,
            "z_ctrl": z_ctrl,
            "delta_vs_linear_bits": delta_vs_linear,
        }

        # 6. Conditional CCA / Sparse Lasso / 2nd-order Walsh Distillation
        trigger_distill = (
            (r == 4)
            or (eval_main["z_null"] >= 6.0)
            or (abs(z_ctrl) >= 3.0)
        )
        distillation: Optional[Dict[str, Any]] = None
        if trigger_distill:
            print(f"  [R={r} DISTILL] Triggered (z_null={eval_main['z_null']:.2f}, z_ctrl={z_ctrl:.2f})", flush=True)
            distillation = distill_embeddings(phi_in_eval, phi_out_eval_main, u_main, v_main)

        # Save weights for R=4 calibration or significant R>=8 hits
        if save_weights and ((r == 4) or (eval_main["z_null"] >= 6.0 and z_ctrl >= 3.0)):
            ckpt_path = out_dir / f"ckpt_R{r}_main.pt"
            torch.save(
                {"round": r, "state_dict": model_main.state_dict(), "eval_main": eval_main},
                ckpt_path,
            )

        summary["rounds"][str(r)] = {
            "main": eval_main,
            "control": eval_ctrl,
            "linear_baseline": linear_baseline,
            "comparison": comparison,
            "distillation": distillation,
        }
        summary["meta"]["total_wall_sec"] = round(time.time() - t_global, 2)
        _atomic_write_json(summary_path, summary)

        print(
            f"  [R={r} SUMMARY] main_MI={eval_main['mi_bits_mean']:.4f}b (Top1={eval_main['top1_mean']:.5f}, z_null={eval_main['z_null']:.2f}) | "
            f"ctrl_MI={eval_ctrl['mi_bits_mean']:.4f}b (Top1={eval_ctrl['top1_mean']:.5f}) | "
            f"lin_MI={eval_lin['mi_bits_mean']:.4f}b | cca_MI={ridge_res['ridge_cca_mi_bits_eval']:.4f}b | "
            f"z_ctrl={z_ctrl:.2f} | round_wall={time.time() - t_r:.1f}s",
            flush=True,
        )

        # Free GPU tensors before next round
        del phi_in_eval, phi_out_eval_main, phi_out_eval_ctrl, phi_in_fit, phi_out_fit_main
        del model_main, model_ctrl, model_lin, u_main, v_main, u_ctrl, v_ctrl, u_lin, v_lin
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n=== P1 Sweep Finished in {summary['meta']['total_wall_sec']:.1f}s ===", flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="P1 Cross-Round InfoNCE Telescope Sweep")
    p.add_argument("--out_dir", type=str, default="v11_discovery/p1_infonce/results")
    p.add_argument("--rounds", type=int, nargs="+", default=list(DEFAULT_ROUNDS))
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_n", type=int, default=16384)
    p.add_argument("--eval_pool_k", type=int, default=1024)
    p.add_argument("--fit_n", type=int, default=32768)
    p.add_argument("--perm_n", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--num_blocks", type=int, default=4)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--no_save_weights", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Fast CPU/tiny smoke config")
    args = p.parse_args()

    if args.smoke:
        run_sweep(
            out_dir=Path(args.out_dir),
            rounds_list=[4, 8],
            steps=6,
            batch_size=32,
            lr=args.lr,
            eval_n=128,
            eval_pool_k=32,
            fit_n=128,
            perm_n=4,
            hidden_dim=64,
            emb_dim=16,
            num_blocks=2,
            save_weights=False,
            device_str="cpu",
        )
    else:
        run_sweep(
            out_dir=Path(args.out_dir),
            rounds_list=list(args.rounds),
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_n=args.eval_n,
            eval_pool_k=args.eval_pool_k,
            fit_n=args.fit_n,
            perm_n=args.perm_n,
            hidden_dim=args.hidden_dim,
            emb_dim=args.emb_dim,
            num_blocks=args.num_blocks,
            save_weights=not args.no_save_weights,
            device_str=args.device or None,
        )


if __name__ == "__main__":
    main()
