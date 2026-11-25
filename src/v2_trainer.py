"""Phase-SHA V2 unrolled trainer with BCE logit loss, mixed corruption, and per-round telemetry."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bit_grid_backbone import BitGridBackbone
from src.boundary_clamp import ClampedBoundary
from src.dual_basis import DualBasisEngine
from src.flow_trainer import prepare_ground_truth_nodes
from src.sha256_ref import (
    pack_int_messages_to_w16,
    sample_random_w16,
    sha256_digest,
)
from src.v2_solver import UnrolledSolverV2


def build_corrupted_init(
    x_star: torch.Tensor,
    boundary: ClampedBoundary,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Construct mixed initial states matching zero-init & diverse noisy inference:
    50% all-zero unknown state, 25% continuous noise, 25% masked/partially corrupted trace.
    """
    bsz = x_star.shape[0]
    device = x_star.device
    u = torch.rand((bsz,), dtype=torch.float32, device=device, generator=generator)

    x_init = torch.zeros_like(x_star)

    # 25% random continuous noise tanh(sigma * eps)
    noise_mask = (u >= 0.50) & (u < 0.75)
    if noise_mask.any():
        n_n = int(noise_mask.sum().item())
        sigmas = 0.05 + 1.15 * torch.rand(
            (n_n, 1, 1, 1), dtype=torch.float32, device=device, generator=generator
        )
        eps = torch.randn(
            (n_n, *x_star.shape[1:]),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        x_init[noise_mask] = torch.tanh(sigmas * eps)

    # 25% partially masked + lightly bit-corrupted x_star
    part_mask = u >= 0.75
    if part_mask.any():
        n_p = int(part_mask.sum().item())
        keep_prob = 0.10 + 0.75 * torch.rand(
            (n_p, 1, 1, 1), dtype=torch.float32, device=device, generator=generator
        )
        keep = (
            torch.rand(
                (n_p, *x_star.shape[1:]),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            < keep_prob
        )
        flip = torch.where(
            torch.rand(
                (n_p, *x_star.shape[1:]),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            < 0.05,
            -1.0,
            1.0,
        )
        x_init[part_mask] = torch.where(keep, x_star[part_mask] * flip, 0.0)

    return boundary.apply_clamp(x_init)


def compute_v2_step_loss(
    logits: torch.Tensor,
    x_star: torch.Tensor,
    boundary: ClampedBoundary,
    dual_basis: DualBasisEngine,
    alpha_phys: float = 0.25,
) -> tuple[torch.Tensor, float, float, float, float]:
    """Compute free-bit BCEWithLogitsLoss (blind ln2=0.693) + full-32 physics loss."""
    clamp_mask = boundary.clamp_mask
    free_mask = 1.0 - clamp_mask
    free_count = free_mask.sum().clamp_min(1.0)

    # +1 spin <-> bit 0 <-> target_prob = 1.0; -1 spin <-> bit 1 <-> target_prob = 0.0
    # sigma(2 * logits) = (1 + tanh(logits)) / 2
    y_target = 0.5 * (1.0 + x_star)
    bce_map = F.binary_cross_entropy_with_logits(
        2.0 * logits, y_target, reduction="none"
    )
    loss_bce = (bce_map * free_mask).sum() / free_count

    x_hat = clamp_mask * boundary.x_clamped + free_mask * torch.tanh(logits)
    a_full, e_full, w_full = boundary.assemble_full(x_hat)
    full_lam = torch.ones(32, dtype=torch.float32, device=logits.device)
    phys_energy, _, stats = dual_basis.compute_energy_and_features(
        a_full, e_full, w_full, lam=full_lam
    )
    lam_norm = 32.0 * float(dual_basis.num_eq_rounds)
    loss_phys = (phys_energy / lam_norm).mean()
    total = loss_bce + alpha_phys * loss_phys

    return (
        total,
        float(loss_bce.detach().item()),
        float(loss_phys.detach().item()),
        float(stats["m1_energy"].detach().mean().item()),
        float(stats["m32_energy"].detach().mean().item()),
    )


def run_unrolled_training_step(
    model: BitGridBackbone,
    dual_basis: DualBasisEngine,
    x_star: torch.Tensor,
    boundary: ClampedBoundary,
    x_init: torch.Tensor,
    k_unroll: int = 6,
    alpha_phys: float = 0.25,
    grad_scale: float = 2.0,
    damp: float = 0.5,
    eta_max: float = 0.03,
    beta_max: float = 0.08,
    do_backward: bool = True,
) -> Dict[str, float]:
    """Run K_unroll truncated-BPTT refinement steps with per-step deep supervision."""
    bsz = x_star.shape[0]
    device = x_star.device
    clamp_mask = boundary.clamp_mask
    free_mask = 1.0 - clamp_mask
    full_lam = torch.ones(32, dtype=torch.float32, device=device)

    raw_weights = [float(s + 1) for s in range(k_unroll)]
    w_sum = sum(raw_weights)
    step_weights = [w / w_sum for w in raw_weights]
    denom = float(max(1, k_unroll - 1))

    x_cur = boundary.apply_clamp(x_init.detach())
    h_prev: Optional[torch.Tensor] = None

    total_loss_val = 0.0
    total_bce_val = 0.0
    total_phys_val = 0.0
    final_bce_val = 0.0
    final_m1_val = 0.0
    final_m32_val = 0.0

    for s in range(k_unroll):
        frac = float(s) / denom
        step_tensor = torch.full((bsz,), frac, dtype=torch.float32, device=device)

        x_cur = x_cur.detach()
        if h_prev is not None:
            h_prev = h_prev.detach()

        grad_radar, res_feat, _, _, raw_grad = dual_basis.compute_radar_and_features(
            x_cur,
            boundary,
            lam=full_lam,
            grad_scale=grad_scale,
            return_raw_grad=True,
        )

        logits_s, h_next = model(
            x_nodes=x_cur,
            grad_radar=grad_radar,
            clamp_mask=clamp_mask,
            res_features=res_feat,
            k_spins=dual_basis.K_spins,
            step_frac=step_tensor,
            lam=full_lam,
            h_prev=h_prev,
        )

        step_loss, bce_val, phys_val, m1_val, m32_val = compute_v2_step_loss(
            logits=logits_s,
            x_star=x_star,
            boundary=boundary,
            dual_basis=dual_basis,
            alpha_phys=alpha_phys,
        )

        w_s = step_weights[s]
        if do_backward:
            (w_s * step_loss).backward()

        total_loss_val += w_s * float(step_loss.detach().item())
        total_bce_val += w_s * bce_val
        total_phys_val += w_s * phys_val
        final_bce_val = bce_val
        final_m1_val = m1_val
        final_m32_val = m32_val

        if s + 1 < k_unroll:
            with torch.no_grad():
                x_neural = torch.tanh(logits_s.detach())
                x_safe = torch.clamp(x_cur, -2.0, 2.0)
                grad_bin = -4.0 * x_safe * (1.0 - x_safe * x_safe)
                eta_s = eta_max * (0.2 + 0.8 * frac)
                beta_s = beta_max * (frac * frac)
                guide = torch.clamp(raw_grad + beta_s * grad_bin, -10.0, 10.0)
                x_next = torch.clamp(
                    (1.0 - damp) * x_cur + damp * x_neural - eta_s * guide * free_mask,
                    -1.0,
                    1.0,
                )
                x_cur = boundary.apply_clamp(x_next)
                h_prev = h_next.detach()

    return {
        "loss": total_loss_val,
        "bce_loss": total_bce_val,
        "bce_final": final_bce_val,
        "phys_loss": total_phys_val,
        "m1_energy": final_m1_val,
        "m32_energy": final_m32_val,
    }


def _compute_per_round_acc(
    x_pred_disc: torch.Tensor,
    x_star: torch.Tensor,
    free_mask: torch.Tensor,
) -> List[float]:
    """Compute accuracy per round t in [0..R-1]; fully clamped rounds report 1.0."""
    correct = (
        ((x_pred_disc < 0) == (x_star < 0)).to(torch.float64)
        * free_mask.to(torch.float64)
    ).sum(dim=(0, 2, 3))
    counts = free_mask.to(torch.float64).sum(dim=(0, 2, 3))
    out: List[float] = []
    for t in range(x_star.shape[1]):
        c = float(counts[t].item())
        out.append(1.0 if c <= 0.0 else float(correct[t].item()) / c)
    return out


class UnrolledTrainerV2:
    """Phase-SHA V2 unrolled trainer with strict disjoint splits and wavefront telemetry."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str | Path,
        device: torch.device | str = "cuda",
    ) -> None:
        self.config = dict(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

        self.rounds = int(config["rounds"])
        self.domain_mode = str(config.get("domain_mode", "int_range"))
        self.num_bits = int(config.get("num_bits", 32))
        self.msg_bytes_len = int(config.get("msg_bytes_len", 4))

        self.dual_basis = DualBasisEngine(rounds=self.rounds).to(self.device)
        self.model = BitGridBackbone(
            rounds=self.rounds,
            d_bit=int(config.get("d_bit", 64)),
            n_heads=int(config.get("n_heads", 4)),
            n_blocks=int(config.get("n_blocks", 4)),
            d_ff_mult=int(config.get("d_ff_mult", 3)),
            share_blocks=bool(config.get("share_blocks", False)),
            grad_checkpointing=bool(config.get("grad_checkpointing", False)),
        ).to(self.device)

        self.total_steps = int(config.get("steps", 5000))
        self.lr_total_steps = int(config.get("lr_total_steps", 15000))
        self.warmup_steps = int(config.get("warmup_steps", 500))
        self.lr = float(config.get("lr", 5.0e-4))
        self.min_lr_ratio = float(config.get("min_lr_ratio", 0.1))
        self.weight_decay = float(config.get("weight_decay", 0.05))
        self.grad_clip = float(config.get("grad_clip", 1.0))
        self.alpha_phys = float(config.get("alpha_phys", 0.25))
        self.grad_scale = float(config.get("grad_scale", 2.0))
        self.batch_size = int(config.get("batch_size", 256))
        self.k_unroll = int(config.get("K_unroll", 6))
        self.damp = float(config.get("damp", 0.45))
        self.infer_damp = float(config.get("infer_damp", 0.10))
        self.sde_noise_min = float(config.get("sde_noise_min", 0.10))
        self.sde_noise_max = float(config.get("sde_noise_max", 0.45))
        self.eta_max = float(config.get("anneal_eta_max", 0.02))
        self.beta_max = float(config.get("anneal_beta_max", 0.25))

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )

        self.train_rng = torch.Generator(device=self.device)
        self.train_rng.manual_seed(int(config.get("seed", 42)))

        holdout_size = int(config.get("holdout_size", 2048))
        holdout_rng = torch.Generator(device="cpu")
        holdout_rng.manual_seed(int(config.get("holdout_seed", 20261122)))

        self.train_w0_pool: Optional[torch.Tensor] = None
        self.holdout_w0_gpu: Optional[torch.Tensor] = None

        if self.domain_mode == "int_range" and self.num_bits <= 20:
            total_space = 1 << self.num_bits
            if holdout_size >= total_space:
                raise ValueError(
                    f"holdout_size ({holdout_size}) must be < 2^num_bits ({total_space})"
                )
            perm = torch.randperm(total_space, generator=holdout_rng, dtype=torch.int64)
            holdout_w0 = perm[:holdout_size]
            train_w0 = perm[holdout_size:]
            self.holdout_w16 = pack_int_messages_to_w16(holdout_w0)
            self.train_w0_pool = train_w0.to(self.device)
            self.holdout_w0_gpu = holdout_w0.to(self.device)
        elif self.domain_mode == "int_range":
            high = 1 << self.num_bits
            unique_vals: List[int] = []
            seen = set()
            while len(unique_vals) < holdout_size:
                cand = torch.randint(
                    0, high, (holdout_size * 2,), dtype=torch.int64, generator=holdout_rng
                ).tolist()
                for v in cand:
                    if v not in seen:
                        seen.add(v)
                        unique_vals.append(v)
                        if len(unique_vals) == holdout_size:
                            break
            holdout_w0 = torch.tensor(unique_vals, dtype=torch.int64)
            self.holdout_w16 = pack_int_messages_to_w16(holdout_w0)
            self.holdout_w0_gpu = holdout_w0.to(self.device)
        else:
            self.holdout_w16 = sample_random_w16(
                batch_size=holdout_size,
                mode=self.domain_mode,
                num_bits=self.num_bits,
                msg_bytes_len=self.msg_bytes_len,
                device="cpu",
                generator=holdout_rng,
            )

        self.solver = UnrolledSolverV2(
            model=self.model,
            dual_basis=self.dual_basis,
            num_candidates=int(config.get("anneal_K", 256)),
            num_steps=int(config.get("K_infer", 32)),
            damp=self.infer_damp,
            eta_max=self.eta_max,
            beta_max=self.beta_max,
            sde_noise_min=self.sde_noise_min,
            sde_noise_max=self.sde_noise_max,
            grad_scale=self.grad_scale,
            chunk_size=int(config.get("anneal_chunk_size", 1024)),
        )

        self.step = 0
        self.best_eval_bce = float("inf")
        self.csv_path = self.output_dir / "metrics.csv"
        self.csv_fields = self._build_csv_fields(self.rounds)

    def _sample_train_w16(self, batch_size: int) -> torch.Tensor:
        if self.train_w0_pool is not None:
            idx = torch.randint(
                0,
                self.train_w0_pool.shape[0],
                (batch_size,),
                dtype=torch.int64,
                device=self.device,
                generator=self.train_rng,
            )
            return pack_int_messages_to_w16(self.train_w0_pool[idx])

        w16 = sample_random_w16(
            batch_size=batch_size,
            mode=self.domain_mode,
            num_bits=self.num_bits,
            msg_bytes_len=self.msg_bytes_len,
            device=self.device,
            generator=self.train_rng,
        )
        if self.domain_mode == "int_range" and self.holdout_w0_gpu is not None:
            high = 1 << self.num_bits
            collision = torch.isin(w16[:, 0], self.holdout_w0_gpu)
            while bool(collision.any().item()):
                n_col = int(collision.sum().item())
                w16[collision, 0] = torch.randint(
                    0,
                    high,
                    (n_col,),
                    dtype=torch.int64,
                    device=self.device,
                    generator=self.train_rng,
                )
                collision = torch.isin(w16[:, 0], self.holdout_w0_gpu)
        return w16

    def _lr_lambda(self, current_step: int) -> float:
        if self.warmup_steps > 0 and current_step < self.warmup_steps:
            return float(current_step + 1) / float(self.warmup_steps)
        denom = max(1, self.lr_total_steps - self.warmup_steps)
        progress = min(1.0, float(current_step - self.warmup_steps) / float(denom))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    @staticmethod
    def _build_csv_fields(rounds: int) -> List[str]:
        fields = [
            "step",
            "lr",
            "train_loss",
            "train_bce_loss",
            "train_bce_final",
            "train_phys_loss",
            "eval_bce_loss",
            "zero_w0_bit_acc",
            "zero_phys_energy_m1",
            "zero_phys_energy_m32",
            "w0_free_bit_acc",
            "phys_energy_m1",
            "phys_energy_m32",
            "exact_preimage_rate@256",
            "unique_ratio@256",
            "peak_vram_gb",
            "step_time_ms",
        ]
        fields.extend([f"per_round_acc_{t}" for t in range(rounds)])
        fields.extend([f"zero_round_acc_{t}" for t in range(rounds)])
        fields.extend([f"bit_acc_pos_{k}" for k in range(32)])
        fields.extend([f"zero_w0_bit_acc_pos_{k}" for k in range(32)])
        fields.extend([f"zero_bit_acc_pos_{k}" for k in range(32)])
        return fields

    def _append_csv_row(self, row: Dict[str, Any]) -> None:
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fields)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self.csv_fields})

    def save_checkpoint(self, filename: str = "latest.pt") -> Path:
        ckpt_path = self.output_dir / filename
        payload = {
            "step": self.step,
            "best_eval_bce": self.best_eval_bce,
            "config": self.config,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        torch.save(payload, ckpt_path)
        return ckpt_path

    def load_checkpoint(self, ckpt_path: str | Path) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.step = int(ckpt.get("step", 0))
        self.best_eval_bce = float(ckpt.get("best_eval_bce", float("inf")))

    def load_weights_only(self, ckpt_path: str | Path) -> None:
        """Warm-start model weights across round/domain curricula with fresh optimizer/step."""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        src_cfg = ckpt.get("config", {})
        src_rounds = src_cfg.get("rounds", "?")
        src_step = ckpt.get("step", "?")
        print(
            f"[Phase-SHA V2] Warm-started weights from {ckpt_path} "
            f"(source R={src_rounds}, step={src_step}) -> target R={self.rounds}",
            flush=True,
        )

    def train_one_step(self) -> Dict[str, float]:
        """Execute one truncated-BPTT unrolled training step."""
        self.model.train()
        w16 = self._sample_train_w16(self.batch_size)
        x_star, boundary = prepare_ground_truth_nodes(
            w16,
            rounds=self.rounds,
            domain_mode=self.domain_mode,
            num_bits=self.num_bits,
            msg_bytes_len=self.msg_bytes_len,
        )
        x_init = build_corrupted_init(x_star, boundary, generator=self.train_rng)

        self.optimizer.zero_grad(set_to_none=True)
        metrics = run_unrolled_training_step(
            model=self.model,
            dual_basis=self.dual_basis,
            x_star=x_star,
            boundary=boundary,
            x_init=x_init,
            k_unroll=self.k_unroll,
            alpha_phys=self.alpha_phys,
            grad_scale=self.grad_scale,
            damp=self.damp,
            eta_max=self.eta_max,
            beta_max=self.beta_max,
            do_backward=True,
        )
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return metrics

    @torch.no_grad()
    def evaluate_holdout(
        self, run_anneal: bool = True, use_neural_velocity: bool = True
    ) -> Dict[str, float]:
        """Evaluate on 2048 zero-init holdout hashes and K=256 diverse inference."""
        self.model.eval()
        eval_bs = int(self.config.get("eval_batch_size", 256))
        total = self.holdout_w16.shape[0]

        bce_sum = 0.0
        zero_m1_sum = 0.0
        zero_m32_sum = 0.0
        zero_w0_correct = 0.0
        zero_w0_count = 0.0
        zero_round_correct = torch.zeros(
            self.rounds, dtype=torch.float64, device=self.device
        )
        zero_round_count = torch.zeros(
            self.rounds, dtype=torch.float64, device=self.device
        )
        zero_pos_correct = torch.zeros(32, dtype=torch.float64, device=self.device)
        zero_pos_count = torch.zeros(32, dtype=torch.float64, device=self.device)
        zero_w0_pos_correct = torch.zeros(32, dtype=torch.float64, device=self.device)
        zero_w0_pos_count = torch.zeros(32, dtype=torch.float64, device=self.device)
        n_batches = 0

        for start in range(0, total, eval_bs):
            w16_b = self.holdout_w16[start : start + eval_bs].to(self.device)
            x_star, boundary = prepare_ground_truth_nodes(
                w16_b,
                rounds=self.rounds,
                domain_mode=self.domain_mode,
                num_bits=self.num_bits,
                msg_bytes_len=self.msg_bytes_len,
            )
            free_mask = 1.0 - boundary.clamp_mask
            free_count = free_mask.sum().clamp_min(1.0)

            logits_z, x_disc_z, m1_z, m32_z = self.solver.infer_deterministic_zero(
                boundary, use_neural_velocity=use_neural_velocity
            )
            y_target = 0.5 * (1.0 + x_star)
            bce_map = F.binary_cross_entropy_with_logits(
                2.0 * logits_z, y_target, reduction="none"
            )
            eval_bce = (bce_map * free_mask).sum() / free_count
            bce_sum += float(eval_bce.item())
            zero_m1_sum += float(m1_z.mean().item())
            zero_m32_sum += float(m32_z.mean().item())

            match_z = (
                ((x_disc_z < 0) == (x_star < 0)).to(torch.float64)
                * free_mask.to(torch.float64)
            )
            zero_round_correct += match_z.sum(dim=(0, 2, 3))
            zero_round_count += free_mask.to(torch.float64).sum(dim=(0, 2, 3))
            zero_pos_correct += match_z.sum(dim=(0, 1, 2))
            zero_pos_count += free_mask.to(torch.float64).sum(dim=(0, 1, 2))

            w0_free_mask = free_mask[:, 0, 2, :].to(torch.float64)
            zero_w0_pos_correct += match_z[:, 0, 2, :].sum(dim=0)
            zero_w0_pos_count += w0_free_mask.sum(dim=0)

            zero_w0_correct += float(match_z[:, 0, 2, :].sum().item())
            zero_w0_count += float(w0_free_mask.sum().item())
            n_batches += 1

        zero_pos_acc = (zero_pos_correct / zero_pos_count.clamp_min(1.0)).cpu().tolist()
        zero_w0_pos_acc = (
            (zero_w0_pos_correct / zero_w0_pos_count.clamp_min(1.0)).cpu().tolist()
        )
        out: Dict[str, float] = {
            "eval_bce_loss": bce_sum / max(1, n_batches),
            "zero_w0_bit_acc": zero_w0_correct / max(1.0, zero_w0_count),
            "zero_phys_energy_m1": zero_m1_sum / max(1, n_batches),
            "zero_phys_energy_m32": zero_m32_sum / max(1, n_batches),
        }
        for t in range(self.rounds):
            c = float(zero_round_count[t].item())
            out[f"zero_round_acc_{t}"] = (
                1.0 if c <= 0.0 else float(zero_round_correct[t].item()) / c
            )
        for k in range(32):
            out[f"zero_bit_acc_pos_{k}"] = float(zero_pos_acc[k])
            out[f"zero_w0_bit_acc_pos_{k}"] = float(zero_w0_pos_acc[k])

        if run_anneal:
            n_anneal = min(
                int(self.config.get("anneal_eval_samples", 64)),
                self.holdout_w16.shape[0],
            )
            w16_sub = self.holdout_w16[:n_anneal].to(self.device)
            x_star_sub, boundary_sub = prepare_ground_truth_nodes(
                w16_sub,
                rounds=self.rounds,
                domain_mode=self.domain_mode,
                num_bits=self.num_bits,
                msg_bytes_len=self.msg_bytes_len,
            )
            digests = sha256_digest(w16_sub, rounds=self.rounds)
            infer_rng = torch.Generator(device=self.device)
            infer_rng.manual_seed(77777)

            solve_res = self.solver.solve(
                digests,
                domain_mode=self.domain_mode,
                num_bits=self.num_bits,
                msg_bytes_len=self.msg_bytes_len,
                generator=infer_rng,
                use_neural_velocity=use_neural_velocity,
            )

            free_sub = 1.0 - boundary_sub.clamp_mask
            match_sub = (
                ((solve_res.best_x_disc < 0) == (x_star_sub < 0)).to(torch.float64)
                * free_sub.to(torch.float64)
            )
            pos_correct = match_sub.sum(dim=(0, 1, 2))
            pos_counts = free_sub.to(torch.float64).sum(dim=(0, 1, 2)).clamp_min(1.0)
            pos_acc = (pos_correct / pos_counts).cpu().tolist()

            round_acc = _compute_per_round_acc(
                solve_res.best_x_disc, x_star_sub, free_sub
            )
            w0_free = free_sub[:, 0, 2, :].to(torch.float64)
            w0_acc = float(
                match_sub[:, 0, 2, :].sum().item()
                / max(1.0, float(w0_free.sum().item()))
            )

            out["exact_preimage_rate@256"] = solve_res.solve_rate
            out["unique_ratio@256"] = solve_res.mean_unique_ratio
            out["w0_free_bit_acc"] = w0_acc
            out["phys_energy_m1"] = float(solve_res.best_m1_energy.mean().item())
            out["phys_energy_m32"] = float(solve_res.best_m32_energy.mean().item())
            for t in range(self.rounds):
                out[f"per_round_acc_{t}"] = float(round_acc[t])
            for k in range(32):
                out[f"bit_acc_pos_{k}"] = float(pos_acc[k])

        return out

    def train(self, max_steps: Optional[int] = None) -> None:
        """Run V2 training loop from current step up to target steps."""
        target_steps = self.total_steps if max_steps is None else max_steps
        log_every = int(self.config.get("log_every", 100))
        eval_every = int(self.config.get("eval_every", 500))
        ckpt_every = int(self.config.get("checkpoint_every", 500))

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        print(
            f"[Phase-SHA V2] Starting training: R={self.rounds}, B={self.batch_size}, "
            f"K_unroll={self.k_unroll}, K_infer={self.solver.num_steps}, "
            f"device={self.device}, step={self.step}/{target_steps}",
            flush=True,
        )

        if self.step == 0:
            eval_0 = self.evaluate_holdout(run_anneal=True)
            round_str = " ".join(
                f"r{t}:{eval_0[f'per_round_acc_{t}']:.2f}"
                for t in range(self.rounds)
            )
            print(
                f"  [eval step=0] bce={eval_0['eval_bce_loss']:.4f} "
                f"zero_w0={eval_0['zero_w0_bit_acc']:.4f} "
                f"w0_acc={eval_0['w0_free_bit_acc']:.4f} "
                f"E_m1={eval_0['phys_energy_m1']:.2f} "
                f"E_m32={eval_0['phys_energy_m32']:.2f} "
                f"pre@256={eval_0['exact_preimage_rate@256']:.4f} "
                f"uniq@256={eval_0['unique_ratio@256']:.3f}",
                flush=True,
            )
            print(f"  [rounds step=0] {round_str}", flush=True)
            row0: Dict[str, Any] = {
                "step": 0,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
            row0.update(eval_0)
            self._append_csv_row(row0)

        while self.step < target_steps:
            t0 = time.perf_counter()
            train_metrics = self.train_one_step()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                peak_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            else:
                peak_gb = 0.0
            dt_ms = (time.perf_counter() - t0) * 1000.0
            cur_lr = float(self.optimizer.param_groups[0]["lr"])

            do_eval = (self.step % eval_every == 0) or (self.step == target_steps)
            do_log = (self.step % log_every == 0) or do_eval or (self.step == 1)

            eval_metrics: Dict[str, float] = {}
            if do_eval:
                eval_metrics = self.evaluate_holdout(run_anneal=True)
                if eval_metrics["eval_bce_loss"] < self.best_eval_bce:
                    self.best_eval_bce = eval_metrics["eval_bce_loss"]
                    self.save_checkpoint("best.pt")

            if do_log:
                print(
                    f"step={self.step:6d}/{target_steps} lr={cur_lr:.2e} "
                    f"loss={train_metrics['loss']:.4f} bce={train_metrics['bce_loss']:.4f} "
                    f"bce_end={train_metrics['bce_final']:.4f} "
                    f"phys={train_metrics['phys_loss']:.4f} vram={peak_gb:.2f}GB "
                    f"dt={dt_ms:.1f}ms",
                    flush=True,
                )
                if eval_metrics:
                    round_str = " ".join(
                        f"r{t}:{eval_metrics[f'per_round_acc_{t}']:.2f}"
                        for t in range(self.rounds)
                    )
                    print(
                        f"  [eval] bce={eval_metrics['eval_bce_loss']:.4f} "
                        f"zero_w0={eval_metrics['zero_w0_bit_acc']:.4f} "
                        f"w0_acc={eval_metrics['w0_free_bit_acc']:.4f} "
                        f"E_m1={eval_metrics['phys_energy_m1']:.2f} "
                        f"E_m32={eval_metrics['phys_energy_m32']:.2f} "
                        f"pre@256={eval_metrics['exact_preimage_rate@256']:.4f} "
                        f"uniq@256={eval_metrics['unique_ratio@256']:.3f}",
                        flush=True,
                    )
                    print(f"  [rounds] {round_str}", flush=True)
                row = {
                    "step": self.step,
                    "lr": cur_lr,
                    "train_loss": train_metrics["loss"],
                    "train_bce_loss": train_metrics["bce_loss"],
                    "train_bce_final": train_metrics["bce_final"],
                    "train_phys_loss": train_metrics["phys_loss"],
                    "peak_vram_gb": round(peak_gb, 4),
                    "step_time_ms": round(dt_ms, 2),
                }
                row.update(eval_metrics)
                self._append_csv_row(row)

            if (self.step % ckpt_every == 0) or (self.step == target_steps):
                self.save_checkpoint("latest.pt")
