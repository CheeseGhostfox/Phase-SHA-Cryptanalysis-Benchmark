"""Self-supervised Rectified Flow training loop, online GPU data generator, and telemetry."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.anneal_solver import AnnealSolver, homotopy_lambda
from src.boundary_clamp import ClampedBoundary, build_clamped_boundary
from src.dual_basis import DualBasisEngine
from src.graph_backbone import PhaseSHABackbone
from src.sha256_ref import (
    pack_int_messages_to_w16,
    sample_random_w16,
    sha256_digest,
    sha256_trace,
    words_to_spins,
)


def prepare_ground_truth_nodes(
    w16: torch.Tensor,
    rounds: int,
    domain_mode: str = "int_range",
    num_bits: int = 32,
    msg_bytes_len: int = 4,
) -> tuple[torch.Tensor, ClampedBoundary]:
    """Run vectorized SHA-256 trace and return (x_star, boundary)."""
    trace = sha256_trace(w16, rounds=rounds)
    boundary = build_clamped_boundary(
        digest=trace.digest,
        rounds=rounds,
        domain_mode=domain_mode,
        num_bits=num_bits,
        msg_bytes_len=msg_bytes_len,
    )
    a_mid = words_to_spins(trace.A_full[:, 3 : 3 + rounds], dtype=torch.float32)
    e_mid = words_to_spins(trace.E_full[:, 3 : 3 + rounds], dtype=torch.float32)
    w_all = words_to_spins(trace.W_full[:, :rounds], dtype=torch.float32)
    x_star = torch.stack([a_mid, e_mid, w_all], dim=2)
    return x_star, boundary


def sample_frequency_masks(
    batch_size: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample curriculum frequency masks lambda (B, 32).

    Mix:
      50% uniform full-spectrum lambda = 1
      25% random hard cutoff m_max ~ Uniform{1..32}
      25% random smooth homotopy wavefront
    """
    lam = torch.ones((batch_size, 32), dtype=torch.float32, device=device)
    u = torch.rand((batch_size,), dtype=torch.float32, device=device, generator=generator)
    m_idx = torch.arange(1, 33, dtype=torch.float32, device=device).unsqueeze(0)

    hard_mask = (u >= 0.50) & (u < 0.75)
    if hard_mask.any():
        n_hard = int(hard_mask.sum().item())
        cutoffs = torch.randint(
            1, 33, (n_hard, 1), dtype=torch.float32, device=device, generator=generator
        )
        lam[hard_mask] = (m_idx <= cutoffs).to(torch.float32)

    smooth_mask = u >= 0.75
    if smooth_mask.any():
        n_smooth = int(smooth_mask.sum().item())
        taus = torch.rand(
            (n_smooth,), dtype=torch.float32, device=device, generator=generator
        )
        lam[smooth_mask] = homotopy_lambda(taus, gamma=2.0, device=device)

    return lam


def compute_flow_loss(
    model: PhaseSHABackbone,
    dual_basis: DualBasisEngine,
    x_star: torch.Tensor,
    boundary: ClampedBoundary,
    tau: torch.Tensor,
    lam: torch.Tensor,
    alpha_phys: float = 0.5,
    grad_scale: float = 2.0,
    noise: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Compute Rectified Flow + 2-adic physics residual loss for a batch."""
    clamp_mask = boundary.clamp_mask
    free_mask = 1.0 - clamp_mask
    if noise is None:
        noise = torch.randn_like(x_star)

    tau_view = tau.view(-1, 1, 1, 1)
    x_interp = (1.0 - tau_view) * noise + tau_view * x_star
    x_tau = clamp_mask * boundary.x_clamped + free_mask * x_interp
    u_target = (x_star - noise) * free_mask

    grad_radar, res_feat, _, _ = dual_basis.compute_radar_and_features(
        x_tau, boundary, lam=lam, grad_scale=grad_scale
    )

    v_pred = model(
        x_nodes=x_tau,
        grad_radar=grad_radar,
        clamp_mask=clamp_mask,
        res_features=res_feat,
        k_spins=dual_basis.K_spins,
        tau=tau,
        lam=lam,
    )

    free_count = free_mask.sum().clamp_min(1.0)
    loss_fm = (((v_pred - u_target) * free_mask) ** 2).sum() / free_count

    # One-step projected clean state, strictly preserving clamped boundaries
    x1_free = torch.tanh(x_tau + (1.0 - tau_view) * v_pred)
    x1_hat = clamp_mask * boundary.x_clamped + free_mask * x1_free
    a_full, e_full, w_full = boundary.assemble_full(x1_hat)
    phys_energy, _, stats = dual_basis.compute_energy_and_features(
        a_full, e_full, w_full, lam=lam
    )

    lam_norm = lam.sum(dim=-1).clamp_min(1.0) * float(dual_basis.num_eq_rounds)
    loss_phys = (phys_energy / lam_norm).mean()
    total_loss = loss_fm + alpha_phys * loss_phys

    metrics = {
        "loss": float(total_loss.detach().item()),
        "fm_loss": float(loss_fm.detach().item()),
        "phys_loss": float(loss_phys.detach().item()),
        "m1_energy": float(stats["m1_energy"].detach().mean().item()),
        "m32_energy": float(stats["m32_energy"].detach().mean().item()),
    }
    return total_loss, metrics


class FlowTrainer:
    """Self-supervised trainer with disjoint train/holdout splits and CSV telemetry."""

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
        self.model = PhaseSHABackbone(
            rounds=self.rounds,
            d_model=int(config.get("d_model", 384)),
            n_heads=int(config.get("n_heads", 8)),
            n_layers=int(config.get("n_layers", 8)),
            d_ff=int(config.get("d_ff", 1024)),
            share_layers=bool(config.get("share_layers", False)),
            num_registers=int(config.get("num_registers", 2)),
            grad_checkpointing=bool(config.get("grad_checkpointing", False)),
        ).to(self.device)

        self.total_steps = int(config.get("steps", 20000))
        self.warmup_steps = int(config.get("warmup_steps", 500))
        self.lr = float(config.get("lr", 3.0e-4))
        self.weight_decay = float(config.get("weight_decay", 0.1))
        self.grad_clip = float(config.get("grad_clip", 1.0))
        self.alpha_phys = float(config.get("alpha_phys", 0.5))
        self.grad_scale = float(config.get("grad_scale", 2.0))
        self.batch_size = int(config.get("batch_size", 256))

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

        self.solver = AnnealSolver(
            model=self.model,
            dual_basis=self.dual_basis,
            num_candidates=int(config.get("anneal_K", 256)),
            num_steps=int(config.get("anneal_steps", 100)),
            gamma=float(config.get("anneal_gamma", 2.0)),
            eta_max=float(config.get("anneal_eta_max", 0.05)),
            beta_max=float(config.get("anneal_beta_max", 0.10)),
            grad_scale=self.grad_scale,
            chunk_size=int(config.get("anneal_chunk_size", 256)),
        )

        self.step = 0
        self.best_eval_loss = float("inf")
        self.csv_path = self.output_dir / "metrics.csv"
        self.csv_fields = self._build_csv_fields()

    def _sample_train_w16(self, batch_size: int) -> torch.Tensor:
        """Sample online training message blocks strictly disjoint from holdout_w16."""
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
        denom = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, float(current_step - self.warmup_steps) / float(denom))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    @staticmethod
    def _build_csv_fields() -> List[str]:
        fields = [
            "step",
            "lr",
            "train_loss",
            "train_fm_loss",
            "train_phys_loss",
            "eval_fm_loss",
            "blind1_w0_bit_acc",
            "blind1_phys_energy_m1",
            "blind1_phys_energy_m32",
            "w0_free_bit_acc",
            "phys_energy_m1",
            "phys_energy_m32",
            "exact_preimage_rate@256",
            "peak_vram_gb",
            "step_time_ms",
        ]
        fields.extend([f"bit_acc_pos_{k}" for k in range(32)])
        fields.extend([f"blind1_bit_acc_pos_{k}" for k in range(32)])
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
            "best_eval_loss": self.best_eval_loss,
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
        self.best_eval_loss = float(ckpt.get("best_eval_loss", float("inf")))

    def train_one_step(self) -> Dict[str, float]:
        """Execute one online GPU training step."""
        self.model.train()
        w16 = self._sample_train_w16(self.batch_size)
        x_star, boundary = prepare_ground_truth_nodes(
            w16,
            rounds=self.rounds,
            domain_mode=self.domain_mode,
            num_bits=self.num_bits,
            msg_bytes_len=self.msg_bytes_len,
        )
        tau = torch.rand(
            (self.batch_size,),
            dtype=torch.float32,
            device=self.device,
            generator=self.train_rng,
        )
        lam = sample_frequency_masks(
            self.batch_size, device=self.device, generator=self.train_rng
        )
        noise = torch.randn(
            x_star.shape,
            dtype=torch.float32,
            device=self.device,
            generator=self.train_rng,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss, metrics = compute_flow_loss(
            model=self.model,
            dual_basis=self.dual_basis,
            x_star=x_star,
            boundary=boundary,
            tau=tau,
            lam=lam,
            alpha_phys=self.alpha_phys,
            grad_scale=self.grad_scale,
            noise=noise,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return metrics

    @torch.no_grad()
    def evaluate_holdout(
        self, run_anneal: bool = True, use_neural_velocity: bool = True
    ) -> Dict[str, float]:
        """Evaluate on unseen holdout hashes (tau~U(0,1) FM, blind tau=0, and K=256 anneal)."""
        self.model.eval()
        eval_bs = int(self.config.get("eval_batch_size", 256))
        total = self.holdout_w16.shape[0]
        eval_rng = torch.Generator(device=self.device)
        eval_rng.manual_seed(99991)

        full_lam = torch.ones(32, dtype=torch.float32, device=self.device)

        fm_loss_sum = 0.0
        blind_m1_sum = 0.0
        blind_m32_sum = 0.0
        blind_correct_by_pos = torch.zeros(32, dtype=torch.float64, device=self.device)
        blind_count_by_pos = torch.zeros(32, dtype=torch.float64, device=self.device)
        blind_w0_correct = 0.0
        blind_w0_count = 0.0
        n_batches = 0

        for start in range(0, total, eval_bs):
            w16_b = self.holdout_w16[start : start + eval_bs].to(self.device)
            bs = w16_b.shape[0]
            x_star, boundary = prepare_ground_truth_nodes(
                w16_b,
                rounds=self.rounds,
                domain_mode=self.domain_mode,
                num_bits=self.num_bits,
                msg_bytes_len=self.msg_bytes_len,
            )
            clamp_mask = boundary.clamp_mask
            free_mask = 1.0 - clamp_mask
            free_count = free_mask.sum().clamp_min(1.0)

            # 1. Unbiased FM evaluation loss at tau ~ U(0, 1)
            tau_rand = torch.rand(
                (bs,), dtype=torch.float32, device=self.device, generator=eval_rng
            )
            noise_fm = torch.randn(
                x_star.shape,
                dtype=torch.float32,
                device=self.device,
                generator=eval_rng,
            )
            tau_view = tau_rand.view(-1, 1, 1, 1)
            x_tau = clamp_mask * boundary.x_clamped + free_mask * (
                (1.0 - tau_view) * noise_fm + tau_view * x_star
            )
            u_target = (x_star - noise_fm) * free_mask
            grad_radar_fm, res_feat_fm, _, _ = self.dual_basis.compute_radar_and_features(
                x_tau, boundary, lam=full_lam, grad_scale=self.grad_scale
            )
            v_pred_fm = self.model(
                x_nodes=x_tau,
                grad_radar=grad_radar_fm,
                clamp_mask=clamp_mask,
                res_features=res_feat_fm,
                k_spins=self.dual_basis.K_spins,
                tau=tau_rand,
                lam=full_lam,
            )
            fm_loss = (((v_pred_fm - u_target) * free_mask) ** 2).sum() / free_count
            fm_loss_sum += float(fm_loss.item())

            # 2. Pure blind 1-step reconstruction at tau = 0.0 (zero X* leaked)
            tau_zero = torch.zeros((bs,), dtype=torch.float32, device=self.device)
            noise_blind = torch.randn(
                x_star.shape,
                dtype=torch.float32,
                device=self.device,
                generator=eval_rng,
            )
            x_zero = boundary.apply_clamp(noise_blind)
            grad_radar_0, res_feat_0, _, _ = self.dual_basis.compute_radar_and_features(
                x_zero, boundary, lam=full_lam, grad_scale=self.grad_scale
            )
            v_pred_0 = self.model(
                x_nodes=x_zero,
                grad_radar=grad_radar_0,
                clamp_mask=clamp_mask,
                res_features=res_feat_0,
                k_spins=self.dual_basis.K_spins,
                tau=tau_zero,
                lam=full_lam,
            )
            x1_blind = clamp_mask * boundary.x_clamped + free_mask * torch.tanh(
                x_zero + v_pred_0
            )
            a_b, e_b, w_b = boundary.assemble_full(x1_blind)
            _, _, stats_b = self.dual_basis.compute_energy_and_features(
                a_b, e_b, w_b, lam=full_lam
            )
            blind_m1_sum += float(stats_b["m1_energy"].mean().item())
            blind_m32_sum += float(stats_b["m32_energy"].mean().item())

            match_blind = (
                (x1_blind < 0) == (x_star < 0)
            ).to(torch.float64) * free_mask.to(torch.float64)
            blind_correct_by_pos += match_blind.sum(dim=(0, 1, 2))
            blind_count_by_pos += free_mask.to(torch.float64).sum(dim=(0, 1, 2))

            w0_free = free_mask[:, 0, 2, :].to(torch.float64)
            blind_w0_correct += float(match_blind[:, 0, 2, :].sum().item())
            blind_w0_count += float(w0_free.sum().item())
            n_batches += 1

        blind_acc_pos = (
            (blind_correct_by_pos / blind_count_by_pos.clamp_min(1.0)).cpu().tolist()
        )
        out: Dict[str, float] = {
            "eval_fm_loss": fm_loss_sum / max(1, n_batches),
            "blind1_w0_bit_acc": blind_w0_correct / max(1.0, blind_w0_count),
            "blind1_phys_energy_m1": blind_m1_sum / max(1, n_batches),
            "blind1_phys_energy_m32": blind_m32_sum / max(1, n_batches),
        }
        for k in range(32):
            out[f"blind1_bit_acc_pos_{k}"] = float(blind_acc_pos[k])

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
            anneal_rng = torch.Generator(device=self.device)
            anneal_rng.manual_seed(77777)

            # Blind annealing: ONLY digests and static domain parameters are passed
            solve_res = self.solver.solve(
                digests,
                domain_mode=self.domain_mode,
                num_bits=self.num_bits,
                msg_bytes_len=self.msg_bytes_len,
                generator=anneal_rng,
                use_neural_velocity=use_neural_velocity,
            )
            free_sub = 1.0 - boundary_sub.clamp_mask
            match_anneal = (
                (solve_res.best_x_disc < 0) == (x_star_sub < 0)
            ).to(torch.float64) * free_sub.to(torch.float64)
            anneal_correct_pos = match_anneal.sum(dim=(0, 1, 2))
            anneal_count_pos = free_sub.to(torch.float64).sum(dim=(0, 1, 2))
            anneal_acc_pos = (
                (anneal_correct_pos / anneal_count_pos.clamp_min(1.0)).cpu().tolist()
            )

            w0_free_sub = free_sub[:, 0, 2, :].to(torch.float64)
            w0_acc = float(
                match_anneal[:, 0, 2, :].sum().item()
                / max(1.0, float(w0_free_sub.sum().item()))
            )

            out["exact_preimage_rate@256"] = solve_res.solve_rate
            out["phys_energy_m1"] = float(solve_res.best_m1_energy.mean().item())
            out["phys_energy_m32"] = float(solve_res.best_m32_energy.mean().item())
            out["w0_free_bit_acc"] = w0_acc
            for k in range(32):
                out[f"bit_acc_pos_{k}"] = float(anneal_acc_pos[k])
        else:
            out["exact_preimage_rate@256"] = 0.0
            out["phys_energy_m1"] = out["blind1_phys_energy_m1"]
            out["phys_energy_m32"] = out["blind1_phys_energy_m32"]
            out["w0_free_bit_acc"] = out["blind1_w0_bit_acc"]
            for k in range(32):
                out[f"bit_acc_pos_{k}"] = out[f"blind1_bit_acc_pos_{k}"]

        return out

    def train(self, max_steps: Optional[int] = None) -> None:
        """Run full training loop from current step up to target steps."""
        target_steps = self.total_steps if max_steps is None else max_steps
        log_every = int(self.config.get("log_every", 50))
        eval_every = int(self.config.get("eval_every", 500))
        ckpt_every = int(self.config.get("checkpoint_every", 500))

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        print(
            f"[Phase-SHA] Starting training: R={self.rounds}, B={self.batch_size}, "
            f"device={self.device}, step={self.step}/{target_steps}",
            flush=True,
        )

        # Record step=0 baseline before first gradient update
        if self.step == 0:
            eval_0 = self.evaluate_holdout(run_anneal=True)
            print(
                f"  [eval step=0] fm={eval_0['eval_fm_loss']:.4f} "
                f"blind1_w0={eval_0['blind1_w0_bit_acc']:.4f} "
                f"w0_acc={eval_0['w0_free_bit_acc']:.4f} "
                f"E_m1={eval_0['phys_energy_m1']:.2f} "
                f"E_m32={eval_0['phys_energy_m32']:.2f} "
                f"preimage@256={eval_0['exact_preimage_rate@256']:.4f}",
                flush=True,
            )
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
                if eval_metrics["eval_fm_loss"] < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics["eval_fm_loss"]
                    self.save_checkpoint("best.pt")

            if do_log:
                print(
                    f"step={self.step:6d}/{target_steps} lr={cur_lr:.2e} "
                    f"loss={train_metrics['loss']:.4f} fm={train_metrics['fm_loss']:.4f} "
                    f"phys={train_metrics['phys_loss']:.4f} vram={peak_gb:.2f}GB "
                    f"dt={dt_ms:.1f}ms",
                    flush=True,
                )
                if eval_metrics:
                    print(
                        f"  [eval] fm={eval_metrics['eval_fm_loss']:.4f} "
                        f"blind1_w0={eval_metrics['blind1_w0_bit_acc']:.4f} "
                        f"w0_acc={eval_metrics['w0_free_bit_acc']:.4f} "
                        f"E_m1={eval_metrics['phys_energy_m1']:.2f} "
                        f"E_m32={eval_metrics['phys_energy_m32']:.2f} "
                        f"preimage@256={eval_metrics['exact_preimage_rate@256']:.4f}",
                        flush=True,
                    )
                row = {
                    "step": self.step,
                    "lr": cur_lr,
                    "train_loss": train_metrics["loss"],
                    "train_fm_loss": train_metrics["fm_loss"],
                    "train_phys_loss": train_metrics["phys_loss"],
                    "peak_vram_gb": round(peak_gb, 4),
                    "step_time_ms": round(dt_ms, 2),
                }
                row.update(eval_metrics)
                self._append_csv_row(row)

            if (self.step % ckpt_every == 0) or (self.step == target_steps):
                self.save_checkpoint("latest.pt")
