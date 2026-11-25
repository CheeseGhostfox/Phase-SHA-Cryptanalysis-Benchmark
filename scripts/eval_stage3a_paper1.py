"""Paper I 2,048-hash full 32-bit R=16 ablation evaluation on checkpoints/stage3a_r16_perfect.pt."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn.functional as F
import yaml

from src.flow_trainer import prepare_ground_truth_nodes
from src.sha256_ref import configure_cuda_memory, sha256_digest
from src.v3_trainer import UnrolledTrainerV3, _compute_per_round_acc


def run_ablation_setting(
    base_cfg: Dict[str, Any],
    ckpt_path: Path,
    name: str,
    k_infer: int,
    infer_damp: float,
    use_dynamic_latch: bool,
    device: torch.device,
) -> Dict[str, Any]:
    cfg = dict(
        base_cfg,
        K_infer=k_infer,
        infer_damp=infer_damp,
        use_dynamic_latch=use_dynamic_latch,
        holdout_size=2048,
        eval_batch_size=256,
        anneal_eval_samples=64,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tr = UnrolledTrainerV3(cfg, output_dir=Path(tmpdir), device=device)
        tr.load_checkpoint(ckpt_path)
        tr.ema_model.eval()

        total = tr.holdout_w16.shape[0]
        eval_bs = 256
        rounds = tr.rounds

        bce_sum = 0.0
        m1_sum = 0.0
        m32_sum = 0.0
        latch_sum = 0.0
        n_batches = 0
        exact_1shot_hits = 0

        round_correct = torch.zeros(rounds, dtype=torch.float64, device=device)
        round_count = torch.zeros(rounds, dtype=torch.float64, device=device)
        w0_pos_correct = torch.zeros(32, dtype=torch.float64, device=device)
        w0_pos_count = torch.zeros(32, dtype=torch.float64, device=device)

        t0 = time.perf_counter()
        with torch.no_grad():
            for start in range(0, total, eval_bs):
                w16_b = tr.holdout_w16[start : start + eval_bs].to(device)
                target_dig = sha256_digest(w16_b, rounds=rounds)
                x_star, boundary = prepare_ground_truth_nodes(
                    w16_b,
                    rounds=rounds,
                    domain_mode=tr.domain_mode,
                    num_bits=tr.num_bits,
                    msg_bytes_len=tr.msg_bytes_len,
                )
                free_mask = 1.0 - boundary.clamp_mask
                free_count = free_mask.sum().clamp_min(1.0)

                logits_z, x_disc_z, m1_z, m32_z, latch_z = (
                    tr.solver.infer_deterministic_zero(boundary, use_neural_velocity=True)
                )
                y_target = 0.5 * (1.0 + x_star)
                bce_map = F.binary_cross_entropy_with_logits(
                    2.0 * logits_z, y_target, reduction="none"
                )
                bce_sum += float(((bce_map * free_mask).sum() / free_count).item())
                m1_sum += float(m1_z.mean().item())
                m32_sum += float(m32_z.mean().item())
                latch_sum += float(latch_z.mean().item())

                cand_w16 = boundary.extract_candidate_w16(x_disc_z)
                cand_dig = sha256_digest(cand_w16, rounds=rounds)
                hits_b = (cand_dig == target_dig).all(dim=-1)
                exact_1shot_hits += int(hits_b.sum().item())

                match_z = (
                    ((x_disc_z < 0) == (x_star < 0)).to(torch.float64)
                    * free_mask.to(torch.float64)
                )
                round_correct += match_z.sum(dim=(0, 2, 3))
                round_count += free_mask.to(torch.float64).sum(dim=(0, 2, 3))
                w0_free = free_mask[:, 0, 2, :].to(torch.float64)
                w0_pos_correct += match_z[:, 0, 2, :].sum(dim=0)
                w0_pos_count += w0_free.sum(dim=0)
                n_batches += 1

        zero_time_s = time.perf_counter() - t0
        w0_pos_acc = (w0_pos_correct / w0_pos_count.clamp_min(1.0)).cpu().tolist()
        zero_round_acc = [
            1.0 if float(round_count[t].item()) <= 0.0 else float(round_correct[t].item()) / float(round_count[t].item())
            for t in range(rounds)
        ]
        zero_w0_bit_acc = float(w0_pos_correct.sum().item() / max(1.0, float(w0_pos_count.sum().item())))

        # 64-digest K=256 SMC inference
        t1 = time.perf_counter()
        with torch.no_grad():
            w16_sub = tr.holdout_w16[:64].to(device)
            x_star_sub, boundary_sub = prepare_ground_truth_nodes(
                w16_sub,
                rounds=rounds,
                domain_mode=tr.domain_mode,
                num_bits=tr.num_bits,
                msg_bytes_len=tr.msg_bytes_len,
            )
            digests_sub = sha256_digest(w16_sub, rounds=rounds)
            rng = torch.Generator(device=device).manual_seed(77777)
            solve_res = tr.solver.solve(
                digests_sub,
                domain_mode=tr.domain_mode,
                num_bits=tr.num_bits,
                msg_bytes_len=tr.msg_bytes_len,
                generator=rng,
            )
            free_sub = 1.0 - boundary_sub.clamp_mask
            match_sub = (
                ((solve_res.best_x_disc < 0) == (x_star_sub < 0)).to(torch.float64)
                * free_sub.to(torch.float64)
            )
            w0_acc_k = float(
                match_sub[:, 0, 2, :].sum().item()
                / max(1.0, float(free_sub[:, 0, 2, :].to(torch.float64).sum().item()))
            )
            k256_round_acc = _compute_per_round_acc(solve_res.best_x_disc, x_star_sub, free_sub)
        k256_time_s = time.perf_counter() - t1

    res = {
        "name": name,
        "K_infer": k_infer,
        "infer_damp": infer_damp,
        "use_dynamic_latch": use_dynamic_latch,
        "holdout_size": total,
        "eval_bce_loss": bce_sum / max(1, n_batches),
        "zero_w0_bit_acc": zero_w0_bit_acc,
        "exact_1shot_hits": exact_1shot_hits,
        "exact_1shot_rate": exact_1shot_hits / float(total),
        "zero_w0_lsb03_mean": sum(w0_pos_acc[:4]) / 4.0,
        "zero_w0_msb431_mean": sum(w0_pos_acc[4:]) / 28.0,
        "zero_w0_bit_acc_pos": w0_pos_acc,
        "zero_round_acc": zero_round_acc,
        "zero_latch_ratio": latch_sum / max(1, n_batches),
        "zero_phys_energy_m1": m1_sum / max(1, n_batches),
        "zero_phys_energy_m32": m32_sum / max(1, n_batches),
        "zero_eval_time_s": round(zero_time_s, 2),
        "k256_digests": 64,
        "exact_preimage_hits@256": int(solve_res.solved.sum().item()),
        "exact_preimage_rate@256": solve_res.solve_rate,
        "w0_free_bit_acc": w0_acc_k,
        "anneal_latch_ratio": solve_res.mean_latch_ratio,
        "unique_ratio@256": solve_res.mean_unique_ratio,
        "phys_energy_m1": float(solve_res.best_m1_energy.mean().item()),
        "phys_energy_m32": float(solve_res.best_m32_energy.mean().item()),
        "min_phys_energy_m32": float(solve_res.best_m32_energy.min().item()),
        "k256_round_acc": k256_round_acc,
        "k256_eval_time_s": round(k256_time_s, 2),
    }
    print(
        f"[{name}] 1-shot(2048): hits={exact_1shot_hits}/2048 ({res['exact_1shot_rate']*100:.2f}%) "
        f"zero_w0={zero_w0_bit_acc:.6f} lsb0..3={res['zero_w0_lsb03_mean']:.5f} msb4..31={res['zero_w0_msb431_mean']:.5f} "
        f"bce={res['eval_bce_loss']:.4f} latch_z={res['zero_latch_ratio']:.3f} | "
        f"K=256(64): hits={res['exact_preimage_hits@256']}/64 ({res['exact_preimage_rate@256']*100:.2f}%) "
        f"latch_k={res['anneal_latch_ratio']:.3f} uniq={res['unique_ratio@256']:.3f}",
        flush=True,
    )
    return res


def main() -> None:
    configure_cuda_memory(0.80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = ROOT_DIR / "configs" / "r16_32b_v3.yaml"
    ckpt_path = ROOT_DIR / "checkpoints" / "stage3a_r16_perfect.pt"
    out_path = ROOT_DIR / "runs" / "r16_32b_v3" / "paper1_ablation.json"

    with cfg_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    settings = [
        ("A1_ode14_nolatch", 14, 0.30, False),
        ("A2_ode32_nolatch", 32, 0.20, False),
        ("A3_ode24_latch", 24, 0.25, True),
        ("A4_ode32_latch", 32, 0.20, True),
    ]
    results: List[Dict[str, Any]] = []
    for name, k_inf, damp, dl in settings:
        results.append(
            run_ablation_setting(
                base_cfg=base_cfg,
                ckpt_path=ckpt_path,
                name=name,
                k_infer=k_inf,
                infer_damp=damp,
                use_dynamic_latch=dl,
                device=device,
            )
        )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved Paper I Stage 3A ablation report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
