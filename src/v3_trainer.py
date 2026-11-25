"""Phase-SHA V3 trainer: zero-leak init, chunked BPTT, wavefront weighting, logit reg, and EMA."""

from __future__ import annotations

import copy
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
from src.v3_solver import (
    UnrolledSolverV3,
    apply_closed_loop_schedule,
    spin_expand_schedule,
)


def build_v3_init(
    boundary: ClampedBoundary,
    zero_prob: float = 0.85,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Zero-leakage training initialization: 85% pure zero state + 15% high-entropy random noise."""
    x_clamped = boundary.x_clamped
    bsz = x_clamped.shape[0]
    device = x_clamped.device

    u = torch.rand((bsz,), dtype=torch.float32, device=device, generator=generator)
    is_zero = u < zero_prob

    x_init = torch.zeros_like(x_clamped)
    rand_idx = ~is_zero
    if bool(rand_idx.any().item()):
        n_rand = int(rand_idx.sum().item())
        sigmas = 0.3 + 1.2 * torch.rand(
            (n_rand, 1, 1, 1), dtype=torch.float32, device=device, generator=generator
        )
        eps = torch.randn(
            (n_rand,) + x_clamped.shape[1:],
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        x_init[rand_idx] = torch.tanh(sigmas * eps)

    return boundary.apply_clamp(x_init)


def compute_v3_step_loss(
    logits: torch.Tensor,
    x_star: torch.Tensor,
    boundary: ClampedBoundary,
    dual_basis: DualBasisEngine,
    alpha_phys: float = 0.25,
    logit_reg_weight: float = 0.0015,
    wavefront_weight: float = 3.0,
    closed_loop_schedule: bool = False,
) -> tuple[torch.Tensor, float, float, float, int, float, float]:
    """Compute wavefront-focused BCE + full-32 physics loss + logit L2 regularization."""
    clamp_mask = boundary.clamp_mask
    free_mask = 1.0 - clamp_mask
    free_count = free_mask.sum().clamp_min(1.0)
    rounds = x_star.shape[1]

    x_hat = clamp_mask * boundary.x_clamped + free_mask * torch.tanh(logits)
    if closed_loop_schedule and rounds > 16:
        w_exp = spin_expand_schedule(x_hat[:, :16, 2, :], rounds)
        x_hat_phys = x_hat.clone()
        x_hat_phys[:, 16:rounds, 2, :] = w_exp[:, 16:rounds, :]
        logits_eff = logits.clone()
        logits_eff[:, 16:rounds, 2, :] = torch.atanh(
            (0.995 * w_exp[:, 16:rounds, :]).clamp(-0.995, 0.995)
        )
    else:
        x_hat_phys = x_hat
        logits_eff = logits

    y_target = 0.5 * (1.0 + x_star)
    bce_map = F.binary_cross_entropy_with_logits(
        2.0 * logits_eff, y_target, reduction="none"
    )
    bce_unweighted = (bce_map * free_mask).sum() / free_count

    with torch.no_grad():
        r_cnt = free_mask.sum(dim=(0, 2, 3))  # (R,)
        r_cor = (
            ((logits_eff.detach() < 0) == (x_star < 0)).to(torch.float32) * free_mask
        ).sum(dim=(0, 2, 3))
        r_acc = r_cor / r_cnt.clamp_min(1.0)
        is_frontier = (r_cnt > 0.0) & (r_acc >= 0.55) & (r_acc <= 0.99)
        round_w = torch.where(
            is_frontier,
            torch.full_like(r_acc, float(wavefront_weight)),
            torch.ones_like(r_acc),
        )
        frontier_cnt = int(is_frontier.sum().item())

    weight_4d = free_mask * round_w.view(1, rounds, 1, 1)
    loss_bce_weighted = (bce_map * weight_4d).sum() / weight_4d.sum().clamp_min(1.0)

    logit_reg = (logits_eff.square() * free_mask).sum() / free_count

    a_full, e_full, w_full = boundary.assemble_full(x_hat_phys)
    full_lam = torch.ones(32, dtype=torch.float32, device=logits.device)
    phys_energy, _, stats = dual_basis.compute_energy_and_features(
        a_full, e_full, w_full, lam=full_lam
    )
    lam_norm = 32.0 * float(dual_basis.num_eq_rounds)
    loss_phys = (phys_energy / lam_norm).mean()

    total = (
        loss_bce_weighted
        + float(alpha_phys) * loss_phys
        + float(logit_reg_weight) * logit_reg
    )
    return (
        total,
        float(bce_unweighted.detach().item()),
        float(loss_phys.detach().item()),
        float(logit_reg.detach().item()),
        frontier_cnt,
        float(stats["m1_energy"].detach().mean().item()),
        float(stats["m32_energy"].detach().mean().item()),
    )


def run_chunked_bptt_step(
    model: BitGridBackbone,
    dual_basis: DualBasisEngine,
    x_star: torch.Tensor,
    boundary: ClampedBoundary,
    x_init: torch.Tensor,
    k_unroll: int = 14,
    bptt_chunk_len: int = 4,
    alpha_phys: float = 0.25,
    logit_reg_weight: float = 0.0015,
    wavefront_weight: float = 3.0,
    grad_scale: float = 2.0,
    damp: float = 0.30,
    eta_max: float = 0.02,
    beta_max: float = 0.25,
    sync_h_damp: bool = False,
    closed_loop_schedule: bool = False,
    do_backward: bool = True,
) -> Dict[str, float]:
    """Run K_unroll training steps with differentiable Chunked BPTT (truncated every bptt_chunk_len)."""
    bsz, rounds, _, _ = x_star.shape
    device = x_star.device
    clamp_mask = boundary.clamp_mask
    free_mask = 1.0 - clamp_mask
    full_lam = torch.ones(32, dtype=torch.float32, device=device)

    chunk_len = max(1, int(bptt_chunk_len))
    raw_weights = [float(s + 1) for s in range(k_unroll)]
    w_sum = sum(raw_weights)
    step_weights = [w / w_sum for w in raw_weights]
    denom = float(max(1, k_unroll - 1))

    x_cur = boundary.apply_clamp(x_init.detach())
    if closed_loop_schedule and rounds > 16:
        x_cur = boundary.apply_clamp(apply_closed_loop_schedule(x_cur, rounds))

    h_prev: Optional[torch.Tensor] = None
    chunk_loss: Optional[torch.Tensor] = None

    total_loss_val = 0.0
    total_bce_val = 0.0
    total_phys_val = 0.0
    total_lreg_val = 0.0
    final_bce_val = 0.0
    final_m1_val = 0.0
    final_m32_val = 0.0
    final_front_cnt = 0

    for s in range(k_unroll):
        frac = float(s) / denom
        step_tensor = torch.full((bsz,), frac, dtype=torch.float32, device=device)

        grad_radar, res_feat, _, _, raw_grad = dual_basis.compute_radar_and_features(
            x_cur,
            boundary,
            lam=full_lam,
            grad_scale=grad_scale,
            return_raw_grad=True,
        )

        logits_s, h_tilde = model(
            x_nodes=x_cur,
            grad_radar=grad_radar,
            clamp_mask=clamp_mask,
            res_features=res_feat,
            k_spins=dual_basis.K_spins,
            step_frac=step_tensor,
            lam=full_lam,
            h_prev=h_prev,
        )
        if sync_h_damp and h_prev is not None:
            h_next = (1.0 - damp) * h_prev + damp * h_tilde
        else:
            h_next = h_tilde

        (
            step_loss,
            bce_val,
            phys_val,
            lreg_val,
            front_cnt,
            m1_val,
            m32_val,
        ) = compute_v3_step_loss(
            logits=logits_s,
            x_star=x_star,
            boundary=boundary,
            dual_basis=dual_basis,
            alpha_phys=alpha_phys,
            logit_reg_weight=logit_reg_weight,
            wavefront_weight=wavefront_weight,
            closed_loop_schedule=closed_loop_schedule,
        )

        w_s = step_weights[s]
        weighted_step = w_s * step_loss
        chunk_loss = weighted_step if chunk_loss is None else (chunk_loss + weighted_step)

        total_loss_val += w_s * float(step_loss.detach().item())
        total_bce_val += w_s * bce_val
        total_phys_val += w_s * phys_val
        total_lreg_val += w_s * lreg_val
        final_bce_val = bce_val
        final_m1_val = m1_val
        final_m32_val = m32_val
        final_front_cnt = front_cnt

        is_chunk_end = ((s + 1) % chunk_len == 0) or (s + 1 == k_unroll)
        if is_chunk_end:
            if do_backward and chunk_loss is not None:
                chunk_loss.backward()
            chunk_loss = None
            if s + 1 < k_unroll:
                with torch.no_grad():
                    x_neural = torch.tanh(logits_s.detach())
                    x_safe = torch.clamp(x_cur.detach(), -2.0, 2.0)
                    grad_bin = -4.0 * x_safe * (1.0 - x_safe * x_safe)
                    eta_s = eta_max * (0.2 + 0.8 * frac)
                    beta_s = beta_max * (frac * frac)
                    guide = torch.clamp(raw_grad + beta_s * grad_bin, -10.0, 10.0)
                    x_next = torch.clamp(
                        (1.0 - damp) * x_cur.detach()
                        + damp * x_neural
                        - eta_s * guide * free_mask,
                        -1.0,
                        1.0,
                    )
                    if closed_loop_schedule and rounds > 16:
                        x_next = apply_closed_loop_schedule(x_next, rounds)
                    x_cur = boundary.apply_clamp(x_next).detach()
                    h_prev = h_next.detach()
        else:
            x_neural = torch.tanh(logits_s)
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
            if closed_loop_schedule and rounds > 16:
                x_next = apply_closed_loop_schedule(x_next, rounds)
            x_cur = boundary.apply_clamp(x_next)
            h_prev = h_next

    return {
        "loss": total_loss_val,
        "bce_loss": total_bce_val,
        "bce_final": final_bce_val,
        "phys_loss": total_phys_val,
        "logit_reg": total_lreg_val,
        "frontier_rounds": float(final_front_cnt),
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


class UnrolledTrainerV3:
    """Phase-SHA V3 trainer with carry BiGRU, zero-leak init, chunked BPTT, EMA, and SMC latch."""

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
            d_bit=int(config.get("d_bit", 48)),
            n_heads=int(config.get("n_heads", 3)),
            n_blocks=int(config.get("n_blocks", 2)),
            d_ff_mult=int(config.get("d_ff_mult", 2)),
            share_blocks=bool(config.get("share_blocks", False)),
            grad_checkpointing=bool(config.get("grad_checkpointing", False)),
            use_carry_gru=bool(config.get("use_carry_gru", True)),
        ).to(self.device)

        self.ema_decay = float(config.get("ema_decay", 0.995))
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        self.total_steps = int(config.get("steps", 5000))
        self.lr_total_steps = int(config.get("lr_total_steps", 15000))
        self.warmup_steps = int(config.get("warmup_steps", 300))
        self.lr = float(config.get("lr", 6.0e-4))
        self.min_lr_ratio = float(config.get("min_lr_ratio", 0.1))
        self.weight_decay = float(config.get("weight_decay", 0.02))
        self.grad_clip = float(config.get("grad_clip", 1.0))
        self.alpha_phys = float(config.get("alpha_phys", 0.25))
        self.logit_reg_weight = float(config.get("logit_reg_weight", 0.0015))
        self.wavefront_weight = float(config.get("wavefront_weight", 3.0))
        self.zero_init_prob = float(config.get("zero_init_prob", 0.85))
        self.grad_scale = float(config.get("grad_scale", 2.0))
        self.batch_size = int(config.get("batch_size", 128))
        self.k_unroll = int(config.get("K_unroll", 14))
        self.bptt_chunk_len = int(config.get("bptt_chunk_len", 4))
        self.damp = float(config.get("damp", 0.30))
        self.damp_jitter_min = config.get("damp_jitter_min", None)
        self.damp_jitter_max = config.get("damp_jitter_max", None)
        self.sync_h_damp = bool(config.get("sync_h_damp", False))
        self.closed_loop_schedule = bool(config.get("closed_loop_schedule", False))
        self.best_ckpt_name = str(config.get("best_ckpt_name", "best.pt"))
        self.latest_ckpt_name = str(config.get("latest_ckpt_name", "latest.pt"))
        self.infer_damp = float(config.get("infer_damp", 0.20))
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

        self.solver = UnrolledSolverV3(
            model=self.ema_model,
            dual_basis=self.dual_basis,
            num_candidates=int(config.get("anneal_K", 256)),
            num_steps=int(config.get("K_infer", 64)),
            damp=self.infer_damp,
            eta_max=self.eta_max,
            beta_max=self.beta_max,
            sde_noise_min=self.sde_noise_min,
            sde_noise_max=self.sde_noise_max,
            grad_scale=self.grad_scale,
            chunk_size=int(config.get("anneal_chunk_size", 1024)),
            use_dynamic_latch=bool(config.get("use_dynamic_latch", True)),
            latch_every=int(config.get("latch_every", 2)),
            latch_conf_thresh=float(config.get("latch_conf_thresh", 0.985)),
            latch_res_thresh=float(config.get("latch_res_thresh", 8.0)),
            latch_exact_tol=float(config.get("latch_exact_tol", 1e-6)),
            use_smc_resampling=bool(config.get("use_smc_resampling", True)),
            smc_resample_every=int(config.get("smc_resample_every", 8)),
            smc_elite_k=int(config.get("smc_elite_k", 64)),
            sync_h_damp=self.sync_h_damp,
            closed_loop_schedule=self.closed_loop_schedule,
        )

        self.step = 0
        self.best_eval_bce = float("inf")
        self.best_preimage_rate = 0.0
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

    @torch.no_grad()
    def _update_ema(self) -> None:
        decay = min(self.ema_decay, float(1 + self.step) / float(10 + self.step))
        for p_ema, p_src in zip(self.ema_model.parameters(), self.model.parameters()):
            p_ema.data.mul_(decay).add_(p_src.data, alpha=1.0 - decay)
        for b_ema, b_src in zip(self.ema_model.buffers(), self.model.buffers()):
            b_ema.copy_(b_src)

    @staticmethod
    def _build_csv_fields(rounds: int) -> List[str]:
        fields = [
            "step",
            "lr",
            "train_loss",
            "train_bce_loss",
            "train_bce_final",
            "train_phys_loss",
            "train_logit_reg",
            "frontier_rounds",
            "eval_bce_loss",
            "zero_w0_bit_acc",
            "zero_w0_lsb03_acc",
            "zero_w0_lsb03_zscore",
            "zero_latch_ratio",
            "zero_phys_energy_m1",
            "zero_phys_energy_m32",
            "w0_free_bit_acc",
            "anneal_latch_ratio",
            "phys_energy_m1",
            "phys_energy_m32",
            "min_phys_energy_m32",
            "best_global_m32_energy",
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
            "best_preimage_rate": self.best_preimage_rate,
            "config": self.config,
            "model_state": self.model.state_dict(),
            "ema_model_state": self.ema_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        torch.save(payload, ckpt_path)
        return ckpt_path

    def load_checkpoint(self, ckpt_path: str | Path) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        if "ema_model_state" in ckpt:
            self.ema_model.load_state_dict(ckpt["ema_model_state"])
        else:
            self.ema_model.load_state_dict(self.model.state_dict())
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.step = int(ckpt.get("step", 0))
        self.best_eval_bce = float(ckpt.get("best_eval_bce", float("inf")))
        self.best_preimage_rate = float(ckpt.get("best_preimage_rate", 0.0))

    def load_weights_only(self, ckpt_path: str | Path) -> None:
        """Warm-start weights from V2 or V3 checkpoints with fresh optimizer/step."""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        src_state = ckpt.get("ema_model_state", ckpt["model_state"])
        src_keys = set(src_state.keys())
        tgt_keys = set(self.model.state_dict().keys())

        if src_keys == tgt_keys:
            self.model.load_state_dict(src_state, strict=True)
            mode_desc = "strict=True"
        else:
            incompat = self.model.load_state_dict(src_state, strict=False)
            if incompat.unexpected_keys:
                raise RuntimeError(
                    f"Unexpected keys during V3 warm-start: {incompat.unexpected_keys}"
                )
            bad_missing = [k for k in incompat.missing_keys if "carry_gru" not in k]
            if bad_missing:
                raise RuntimeError(
                    f"Missing non-carry_gru keys during V3 warm-start: {bad_missing}"
                )
            mode_desc = (
                f"V2->V3 partial (loaded={len(src_keys)}, "
                f"fresh_carry_gru={len(incompat.missing_keys)})"
            )

        self.ema_model.load_state_dict(self.model.state_dict(), strict=True)
        src_cfg = ckpt.get("config", {})
        src_rounds = src_cfg.get("rounds", "?")
        src_step = ckpt.get("step", "?")
        print(
            f"[Phase-SHA V3] Warm-started weights from {ckpt_path} "
            f"[{mode_desc}] (source R={src_rounds}, step={src_step}) -> target R={self.rounds}",
            flush=True,
        )

    def train_one_step(self) -> Dict[str, float]:
        """Execute one Chunked-BPTT unrolled training step and update EMA."""
        self.model.train()
        w16 = self._sample_train_w16(self.batch_size)
        x_star, boundary = prepare_ground_truth_nodes(
            w16,
            rounds=self.rounds,
            domain_mode=self.domain_mode,
            num_bits=self.num_bits,
            msg_bytes_len=self.msg_bytes_len,
        )
        x_init = build_v3_init(
            boundary, zero_prob=self.zero_init_prob, generator=self.train_rng
        )

        step_damp = self.damp
        if self.damp_jitter_min is not None and self.damp_jitter_max is not None:
            lo = float(self.damp_jitter_min)
            hi = float(self.damp_jitter_max)
            u_d = float(torch.rand((), device=self.device, generator=self.train_rng).item())
            step_damp = lo + (hi - lo) * u_d

        self.optimizer.zero_grad(set_to_none=True)
        metrics = run_chunked_bptt_step(
            model=self.model,
            dual_basis=self.dual_basis,
            x_star=x_star,
            boundary=boundary,
            x_init=x_init,
            k_unroll=self.k_unroll,
            bptt_chunk_len=self.bptt_chunk_len,
            alpha_phys=self.alpha_phys,
            logit_reg_weight=self.logit_reg_weight,
            wavefront_weight=self.wavefront_weight,
            grad_scale=self.grad_scale,
            damp=step_damp,
            eta_max=self.eta_max,
            beta_max=self.beta_max,
            sync_h_damp=self.sync_h_damp,
            closed_loop_schedule=self.closed_loop_schedule,
            do_backward=True,
        )
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        self._update_ema()
        return metrics

    @torch.no_grad()
    def evaluate_holdout(
        self, run_anneal: bool = True, use_neural_velocity: bool = True
    ) -> Dict[str, float]:
        """Evaluate EMA model on 2048 deterministic zero-init hashes + K=256 SMC inference."""
        self.ema_model.eval()
        eval_bs = int(self.config.get("eval_batch_size", 256))
        total = self.holdout_w16.shape[0]

        bce_sum = 0.0
        zero_m1_sum = 0.0
        zero_m32_sum = 0.0
        zero_latch_sum = 0.0
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

            logits_z, x_disc_z, m1_z, m32_z, latch_z = (
                self.solver.infer_deterministic_zero(
                    boundary, use_neural_velocity=use_neural_velocity
                )
            )
            y_target = 0.5 * (1.0 + x_star)
            bce_map = F.binary_cross_entropy_with_logits(
                2.0 * logits_z, y_target, reduction="none"
            )
            eval_bce = (bce_map * free_mask).sum() / free_count
            bce_sum += float(eval_bce.item())
            zero_m1_sum += float(m1_z.mean().item())
            zero_m32_sum += float(m32_z.mean().item())
            zero_latch_sum += float(latch_z.mean().item())

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
        zero_w0_lsb03_acc = sum(zero_w0_pos_acc[:4]) / 4.0
        lsb_std = math.sqrt(0.25 / max(1.0, float(total) * 4.0))
        out: Dict[str, float] = {
            "eval_bce_loss": bce_sum / max(1, n_batches),
            "zero_w0_bit_acc": zero_w0_correct / max(1.0, zero_w0_count),
            "zero_w0_lsb03_acc": zero_w0_lsb03_acc,
            "zero_w0_lsb03_zscore": (zero_w0_lsb03_acc - 0.5) / lsb_std,
            "zero_latch_ratio": zero_latch_sum / max(1, n_batches),
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
            out["anneal_latch_ratio"] = solve_res.mean_latch_ratio
            out["w0_free_bit_acc"] = w0_acc
            out["phys_energy_m1"] = float(solve_res.best_m1_energy.mean().item())
            out["phys_energy_m32"] = float(solve_res.best_m32_energy.mean().item())
            out["min_phys_energy_m32"] = float(solve_res.min_m32_energy.mean().item())
            out["best_global_m32_energy"] = float(solve_res.min_m32_energy.min().item())
            for t in range(self.rounds):
                out[f"per_round_acc_{t}"] = float(round_acc[t])
            for k in range(32):
                out[f"bit_acc_pos_{k}"] = float(pos_acc[k])

        return out

    def _print_r18_probes(self, eval_metrics: Dict[str, float]) -> None:
        if self.rounds < 18:
            return
        w0_std = math.sqrt(0.25 / max(1.0, float(self.holdout_w16.shape[0]) * 32.0))
        w0_z = (eval_metrics["zero_w0_bit_acc"] - 0.5) / w0_std
        print(
            f"  [R=18 probes] "
            f"r17={eval_metrics['zero_round_acc_17']:.3f} "
            f"r16={eval_metrics['zero_round_acc_16']:.3f} "
            f"r14={eval_metrics['zero_round_acc_14']:.3f} "
            f"r13={eval_metrics['zero_round_acc_13']:.3f} "
            f"r12={eval_metrics['zero_round_acc_12']:.3f} "
            f"r10={eval_metrics['zero_round_acc_10']:.3f} "
            f"r0={eval_metrics['zero_round_acc_0']:.3f} | "
            f"W0_z={w0_z:+.2f} "
            f"lsb03={eval_metrics['zero_w0_lsb03_acc']:.4f}"
            f"(z={eval_metrics['zero_w0_lsb03_zscore']:+.2f}) "
            f"min_E32={eval_metrics['min_phys_energy_m32']:.2f} "
            f"global_min_E32={eval_metrics['best_global_m32_energy']:.2f}",
            flush=True,
        )

    def train(self, max_steps: Optional[int] = None) -> None:
        """Run V3 training loop from current step up to target steps."""
        target_steps = self.total_steps if max_steps is None else max_steps
        log_every = int(self.config.get("log_every", 100))
        eval_every = int(self.config.get("eval_every", 500))
        ckpt_every = int(self.config.get("checkpoint_every", 500))

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        print(
            f"[Phase-SHA V3] Starting training: R={self.rounds}, B={self.batch_size}, "
            f"K_unroll={self.k_unroll} (chunk={self.bptt_chunk_len}), K_infer={self.solver.num_steps}, "
            f"carry_gru={self.model.use_carry_gru}, ema={self.ema_decay}, "
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
                f"latch_z={eval_0['zero_latch_ratio']:.3f} "
                f"latch_k={eval_0['anneal_latch_ratio']:.3f} "
                f"E_m1={eval_0['phys_energy_m1']:.2f} "
                f"E_m32={eval_0['phys_energy_m32']:.2f} "
                f"pre@256={eval_0['exact_preimage_rate@256']:.4f} "
                f"uniq@256={eval_0['unique_ratio@256']:.3f}",
                flush=True,
            )
            print(f"  [rounds step=0] {round_str}", flush=True)
            self._print_r18_probes(eval_0)
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
                    self.save_checkpoint(self.best_ckpt_name)
                if eval_metrics["exact_preimage_rate@256"] > self.best_preimage_rate:
                    self.best_preimage_rate = eval_metrics["exact_preimage_rate@256"]
                    self.save_checkpoint("best_preimage.pt")

            if do_log:
                print(
                    f"step={self.step:6d}/{target_steps} lr={cur_lr:.2e} "
                    f"loss={train_metrics['loss']:.4f} bce={train_metrics['bce_loss']:.4f} "
                    f"bce_end={train_metrics['bce_final']:.4f} "
                    f"phys={train_metrics['phys_loss']:.4f} "
                    f"lreg={train_metrics['logit_reg']:.3f} "
                    f"fr={int(train_metrics['frontier_rounds'])} "
                    f"vram={peak_gb:.2f}GB dt={dt_ms:.1f}ms",
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
                        f"latch_z={eval_metrics['zero_latch_ratio']:.3f} "
                        f"latch_k={eval_metrics['anneal_latch_ratio']:.3f} "
                        f"E_m1={eval_metrics['phys_energy_m1']:.2f} "
                        f"E_m32={eval_metrics['phys_energy_m32']:.2f} "
                        f"pre@256={eval_metrics['exact_preimage_rate@256']:.4f} "
                        f"uniq@256={eval_metrics['unique_ratio@256']:.3f}",
                        flush=True,
                    )
                    print(f"  [rounds] {round_str}", flush=True)
                    self._print_r18_probes(eval_metrics)
                row = {
                    "step": self.step,
                    "lr": cur_lr,
                    "train_loss": train_metrics["loss"],
                    "train_bce_loss": train_metrics["bce_loss"],
                    "train_bce_final": train_metrics["bce_final"],
                    "train_phys_loss": train_metrics["phys_loss"],
                    "train_logit_reg": train_metrics["logit_reg"],
                    "frontier_rounds": int(train_metrics["frontier_rounds"]),
                    "peak_vram_gb": round(peak_gb, 4),
                    "step_time_ms": round(dt_ms, 2),
                }
                row.update(eval_metrics)
                self._append_csv_row(row)

            if (self.step % ckpt_every == 0) or (self.step == target_steps):
                self.save_checkpoint(self.latest_ckpt_name)
