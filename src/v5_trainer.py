"""Unified pure-FP32 Trainer for Phase-SHA V5 (v5d_pure, v5d_neural, v5c, v5b, v5a).

Inherits all verified A100 80GB lossless optimizations:
  - Strict pure FP32 (allow_tf32=False, zero TF32 / AMP / complex64).
  - AdamW(fused=True) on CUDA, zero_grad(set_to_none=True), foreach grad clip & EMA.
  - Background pinned-memory AsyncV4BatchPrefetcher.
  - Zero mid-step CPU-GPU syncs; torch.compile(mode="default") on Linux+CUDA modular kernels only.
  - Cumulative compute `wall_s` tracking + dashboard-compatible telemetry & lossless `--resume`.
"""

from __future__ import annotations

import copy
import math
import platform
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.flow_trainer import prepare_ground_truth_nodes
from src.sha256_ref import (
    bits_to_words,
    pack_int_messages_to_w16,
    sample_random_w16,
    sha256_digest,
    words_to_bits,
)
from src.v3_trainer import _compute_per_round_acc
from src.v4_boundary import (
    ManifoldBoundary,
    apply_clamp_prepared,
    assemble_full_prepared,
)
from src.v4_manifolds import (
    _spin_add32_flat,
    apply_manifold_closed_loop_schedule,
    normalize_manifold,
)
from src.v4_trainer import AsyncV4BatchPrefetcher, _bce_wta_loss_head, build_v4_init
from src.v5_interfaces import (
    GlobalProgressiveCollapseSchedule,
    V5MetricsLogger,
    build_v5_holdout_w16,
    enforce_v5_fp32,
)
from src.v5a_octonion import UnrolledSolverV5A, V5ABackbone, V5APhysicsEngine
from src.v5b_bm import (
    UnrolledSolverV5B,
    V5BBackbone,
    V5BPhysicsEngine,
    compute_bm_rank1_and_eff_rank,
)
from src.v5c_mps import V5CMPSSolver
from src.v5d_rrr import (
    DriftMonitor,
    RRRDynamicsOperator,
    SHA256ReplicaGraph,
    V5DNeuralController,
    V5DRRRSolver,
)


def _strip_orig_mod(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        (k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state.items()
    }


def _evaluate_common_holdout(
    rounds: int,
    holdout_w16: torch.Tensor,
    domain_mode: str,
    num_bits: int,
    msg_bytes_len: int,
    device: torch.device,
    eval_bs: int,
    anneal_samples: int,
    zero_infer_fn: Callable[[ManifoldBoundary], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]],
    solve_fn: Callable[[torch.Tensor, ManifoldBoundary, torch.Generator], Any],
) -> Dict[str, float]:
    """Shared strict-FP32 holdout evaluation matching V3/V4 exact telemetry columns + V5 probes."""
    total = holdout_w16.shape[0]
    bce_sum = 0.0
    zero_m1_sum = 0.0
    zero_m32_sum = 0.0
    zero_latch_sum = 0.0
    zero_align_sum = 0.0
    zero_comm_sum = 0.0
    zero_w0_correct = 0.0
    zero_w0_count = 0.0
    zero_round_correct = torch.zeros(rounds, dtype=torch.float64, device=device)
    zero_round_count = torch.zeros(rounds, dtype=torch.float64, device=device)
    zero_pos_correct = torch.zeros(32, dtype=torch.float64, device=device)
    zero_pos_count = torch.zeros(32, dtype=torch.float64, device=device)
    zero_w0_pos_correct = torch.zeros(32, dtype=torch.float64, device=device)
    zero_w0_pos_count = torch.zeros(32, dtype=torch.float64, device=device)
    extra_accum: Dict[str, List[float]] = {}
    n_batches = 0

    for start in range(0, total, eval_bs):
        w16_b = holdout_w16[start : start + eval_bs].to(device)
        x_star, cb = prepare_ground_truth_nodes(
            w16_b,
            rounds=rounds,
            domain_mode=domain_mode,
            num_bits=num_bits,
            msg_bytes_len=msg_bytes_len,
        )
        mb = ManifoldBoundary.from_clamped_boundary(cb)
        free_mask = 1.0 - mb.clamp_mask
        free_count = free_mask.sum().clamp_min(1.0)

        logits_z, x_disc_z, m1_z, m32_z, latch_z, align_z, comm_z, extra_z = zero_infer_fn(mb)
        y_target = 0.5 * (1.0 + x_star)
        bce_map = F.binary_cross_entropy_with_logits(
            2.0 * logits_z.to(torch.float32), y_target, reduction="none"
        )
        eval_bce = (bce_map * free_mask).sum() / free_count
        bce_sum += float(eval_bce.item())
        zero_m1_sum += float(m1_z.mean().item())
        zero_m32_sum += float(m32_z.mean().item())
        zero_latch_sum += float(latch_z.mean().item())
        zero_align_sum += float(align_z.mean().item())
        zero_comm_sum += float(comm_z.mean().item())

        for ek, ev in extra_z.items():
            extra_accum.setdefault(ek, []).append(float(ev))

        match_z = (((x_disc_z < 0) == (x_star < 0)).to(torch.float64) * free_mask.to(torch.float64))
        zero_round_correct += match_z.sum(dim=(0, 2, 3))
        zero_round_count += free_mask.to(torch.float64).sum(dim=(0, 2, 3))
        zero_pos_correct += match_z.sum(dim=(0, 1, 2))
        zero_pos_count += free_mask.to(torch.float64).sum(dim=(0, 1, 2))

        w0_free = free_mask[:, 0, 2, :].to(torch.float64)
        zero_w0_pos_correct += match_z[:, 0, 2, :].sum(dim=0)
        zero_w0_pos_count += w0_free.sum(dim=0)
        zero_w0_correct += float(match_z[:, 0, 2, :].sum().item())
        zero_w0_count += float(w0_free.sum().item())
        n_batches += 1

    zero_pos_acc = (zero_pos_correct / zero_pos_count.clamp_min(1.0)).cpu().tolist()
    zero_w0_pos_acc = (zero_w0_pos_correct / zero_w0_pos_count.clamp_min(1.0)).cpu().tolist()
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
        "zero_manifold_align": zero_align_sum / max(1, n_batches),
        "zero_manifold_comm": zero_comm_sum / max(1, n_batches),
    }
    for ek, vals in extra_accum.items():
        out[ek] = sum(vals) / max(1, len(vals))
    for t in range(rounds):
        c = float(zero_round_count[t].item())
        out[f"zero_round_acc_{t}"] = 1.0 if c <= 0.0 else float(zero_round_correct[t].item()) / c
    for k in range(32):
        out[f"zero_bit_acc_pos_{k}"] = float(zero_pos_acc[k])
        out[f"zero_w0_bit_acc_pos_{k}"] = float(zero_w0_pos_acc[k])

    n_anneal = min(int(anneal_samples), total)
    w16_sub = holdout_w16[:n_anneal].to(device)
    x_star_sub, cb_sub = prepare_ground_truth_nodes(
        w16_sub, rounds=rounds, domain_mode=domain_mode, num_bits=num_bits, msg_bytes_len=msg_bytes_len
    )
    mb_sub = ManifoldBoundary.from_clamped_boundary(cb_sub)
    digests = sha256_digest(w16_sub, rounds=rounds)
    infer_rng = torch.Generator(device=device)
    infer_rng.manual_seed(77777)

    solve_res = solve_fn(digests, mb_sub, infer_rng)
    free_sub = 1.0 - mb_sub.clamp_mask
    match_sub = (((solve_res.best_x_disc < 0) == (x_star_sub < 0)).to(torch.float64) * free_sub.to(torch.float64))
    pos_counts = free_sub.to(torch.float64).sum(dim=(0, 1, 2)).clamp_min(1.0)
    pos_acc = (match_sub.sum(dim=(0, 1, 2)) / pos_counts).cpu().tolist()
    round_acc = _compute_per_round_acc(solve_res.best_x_disc, x_star_sub, free_sub)
    w0_free_s = free_sub[:, 0, 2, :].to(torch.float64)
    w0_acc = float(match_sub[:, 0, 2, :].sum().item() / max(1.0, float(w0_free_s.sum().item())))

    out["exact_preimage_rate@256"] = solve_res.solve_rate
    out["unique_ratio@256"] = solve_res.mean_unique_ratio
    out["anneal_latch_ratio"] = solve_res.mean_latch_ratio
    out["w0_free_bit_acc"] = w0_acc
    out["phys_energy_m1"] = float(solve_res.best_m1_energy.mean().item())
    out["phys_energy_m32"] = float(solve_res.best_m32_energy.mean().item())
    out["min_phys_energy_m32"] = float(solve_res.min_m32_energy.mean().item())
    out["best_global_m32_energy"] = float(solve_res.min_m32_energy.min().item())
    for ek, ev in solve_res.extra_metrics.items():
        if ek == "escape_count":
            out[ek] = float(ev)
        else:
            out.setdefault(ek, float(ev))
            if ek in ("rank1_ratio", "eff_rank", "rrr_gap", "rrr_beta", "drift_flag", "drift_slope", "ent_entropy", "ent_energy", "zero_assoc"):
                out[ek] = float(ev)
    for t in range(rounds):
        out[f"per_round_acc_{t}"] = float(round_acc[t])
    for k in range(32):
        out[f"bit_acc_pos_{k}"] = float(pos_acc[k])
    return out


class UnifiedTrainerV5:
    """Unified high-throughput pure-FP32 trainer for V5D / V5C / V5B / V5A."""

    SUPPORTED_VARIANTS = ("v5d_pure", "v5d_neural", "v5c", "v5b", "v5a")

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path | str,
        device: torch.device | str = "cuda",
    ) -> None:
        enforce_v5_fp32()
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

        self.variant = str(config.get("variant", "v5d_pure")).lower()
        if self.variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported V5 variant '{self.variant}', expected {self.SUPPORTED_VARIANTS}")

        self.rounds = int(config.get("rounds", 18))
        self.domain_mode = str(config.get("domain_mode", "int_range"))
        self.num_bits = int(config.get("num_bits", 32))
        self.msg_bytes_len = int(config.get("msg_bytes_len", 4))

        self.total_steps = int(config.get("steps", 2000))
        self.lr_total_steps = int(config.get("lr_total_steps", max(self.total_steps, 4000)))
        self.warmup_steps = int(config.get("warmup_steps", 200))
        self.lr = float(config.get("lr", 4.0e-4))
        self.min_lr_ratio = float(config.get("min_lr_ratio", 0.1))
        self.weight_decay = float(config.get("weight_decay", 0.02))
        self.grad_clip = float(config.get("grad_clip", 1.0))

        self.batch_size = int(config.get("batch_size", 16))
        self.num_particles = int(config.get("num_particles", 4))
        self.k_unroll = int(config.get("K_unroll", 14))
        self.bptt_chunk_len = int(config.get("bptt_chunk_len", 4))
        self.backward_group_chunks = max(1, int(config.get("backward_group_chunks", 4)))
        self.damp = float(config.get("damp", 0.28))
        self.damp_jitter_min = config.get("damp_jitter_min", 0.20)
        self.damp_jitter_max = config.get("damp_jitter_max", 0.35)
        self.sync_h_damp = bool(config.get("sync_h_damp", True))
        self.closed_loop_schedule = bool(config.get("closed_loop_schedule", True))

        self.alpha_phys = float(config.get("alpha_phys", 1.0))
        self.wta_winner_weight = float(config.get("wta_winner_weight", 0.85))
        self.bce_weight = float(config.get("bce_weight", 0.25))
        self.closed_loop_wall_rounds = tuple(int(r) for r in config.get("closed_loop_wall_rounds", [0, 13, 16]))
        self.closed_loop_bce_scale = float(config.get("closed_loop_bce_scale", 0.0))
        self.anti_collapse_weight = float(config.get("anti_collapse_weight", 0.05))
        self.logit_reg_weight = float(config.get("logit_reg_weight", 0.001))
        self.wavefront_weight = float(config.get("wavefront_weight", 3.0))
        self.zero_init_prob = float(config.get("zero_init_prob", 0.85))
        self.grad_scale = float(config.get("grad_scale", 2.0))
        self.w_xor = float(config.get("w_xor", 1.0))
        self.eta_max = float(config.get("anneal_eta_max", 0.025))
        self.beta_max = float(config.get("anneal_beta_max", 0.25))

        # Global Progressive Collapse Schedules (prevents V4 continuous plateau)
        self.sched_align = GlobalProgressiveCollapseSchedule(
            start_val=float(config.get("lam_align_start", 0.02)),
            mid_val=float(config.get("lam_align_mid", 1.50)),
            end_val=float(config.get("lam_align_end", 6.00)),
        )
        self.sched_comm = GlobalProgressiveCollapseSchedule(
            start_val=float(config.get("lam_comm_start", 0.02)),
            mid_val=float(config.get("lam_comm_mid", 0.30)),
            end_val=float(config.get("lam_comm_end", 0.60)),
        )
        self.sched_assoc = GlobalProgressiveCollapseSchedule(
            start_val=float(config.get("lam_assoc_start", 0.02)),
            mid_val=float(config.get("lam_assoc_mid", 0.60)),
            end_val=float(config.get("lam_assoc_end", 2.50)),
        )

        # Build variant-specific architecture & physics
        self.physics: Any = None
        self.model: Optional[nn.Module] = None
        self.ema_model: Optional[nn.Module] = None
        self.train_model: Optional[nn.Module] = None
        self.solver: Any = None
        self._compiled_energy_core: Optional[Any] = None
        self._compiled_spin_add32: Optional[Any] = None
        self._compiled_retract_clamp: Optional[Any] = None
        self._compiled_loss_head: Optional[Any] = None
        self.compiled_active: bool = False
        self._use_fused_adamw: bool = False
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self.ema_decay = float(config.get("ema_decay", 0.995))
        self._ema_params: List[torch.Tensor] = []
        self._src_params: List[torch.Tensor] = []
        self._ema_buffers: List[torch.Tensor] = []
        self._src_buffers: List[torch.Tensor] = []

        self._build_variant_components()

        seed_val = int(config.get("seed", 5050))
        self.train_rng = torch.Generator(device=self.device)
        self.train_rng.manual_seed(seed_val)
        self.damp_rng = torch.Generator(device="cpu").manual_seed(seed_val + 1337)

        holdout_size = int(config.get("holdout_size", 2048))
        holdout_seed = int(config.get("holdout_seed", 20261122))
        holdout_w16_cpu, train_pool = build_v5_holdout_w16(
            holdout_size=holdout_size,
            holdout_seed=holdout_seed,
            domain_mode=self.domain_mode,
            num_bits=self.num_bits,
            msg_bytes_len=self.msg_bytes_len,
            device="cpu",
        )
        self.holdout_w16 = holdout_w16_cpu
        if train_pool is not None:
            self.train_w0_pool_cpu: Optional[torch.Tensor] = train_pool.contiguous()
            self.train_w0_pool: Optional[torch.Tensor] = train_pool.to(self.device)
            holdout_w0 = holdout_w16_cpu[:, 0]
            self.holdout_w0_gpu: Optional[torch.Tensor] = holdout_w0.to(self.device)
            self.holdout_w0_set: Optional[set[int]] = set(holdout_w0.tolist())
        else:
            self.train_w0_pool_cpu = None
            self.train_w0_pool = None
            self.holdout_w0_gpu = holdout_w16_cpu[:, 0].to(self.device) if self.domain_mode == "int_range" else None
            self.holdout_w0_set = set(holdout_w16_cpu[:, 0].tolist()) if self.domain_mode == "int_range" else None

        self._prefetcher: Optional[AsyncV4BatchPrefetcher] = None
        # Persistent GPU reflection state for v5d_pure progressive search
        self._pure_state: Optional[torch.Tensor] = None
        self._pure_mask: Optional[torch.Tensor] = None
        self._pure_vals: Optional[torch.Tensor] = None
        self._pure_beta: Optional[torch.Tensor] = None
        self._pure_drift: Optional[DriftMonitor] = None
        self._pure_b_sub: Optional[ManifoldBoundary] = None
        self._pure_best_m32: Optional[torch.Tensor] = None
        self._pure_best_m1: Optional[torch.Tensor] = None
        self._pure_best_x_disc: Optional[torch.Tensor] = None
        self._pure_zero_cache: Optional[Dict[str, float]] = None
        self._pure_last_gap_t: Optional[torch.Tensor] = None

        self.step = 0
        self.wall_s = 0.0
        self.best_eval_bce = float("inf")
        self.best_min_m32 = float("inf")
        self.best_preimage_rate = 0.0
        self.best_ckpt_name = str(config.get("best_ckpt_name", "best.pt"))
        self.latest_ckpt_name = str(config.get("latest_ckpt_name", "latest.pt"))
        self.logger = V5MetricsLogger(self.output_dir, rounds=self.rounds)

    def _build_variant_components(self) -> None:
        cfg = self.config
        if self.variant == "v5d_pure":
            self.solver = V5DRRRSolver(
                rounds=self.rounds,
                mode="pure",
                num_candidates=int(cfg.get("anneal_K", 256)),
                num_steps=int(cfg.get("K_infer", 48)),
                chunk_size=int(cfg.get("anneal_chunk_size", 1024)),
                beta_min=float(cfg.get("rrr_beta_min", 0.38)),
                beta_max=float(cfg.get("rrr_beta_max", 1.05)),
                init_scale=float(cfg.get("rrr_init_scale", 0.85)),
            ).to(self.device)
            return

        if self.variant == "v5d_neural":
            self.model = V5DNeuralController(
                rounds=self.rounds, hidden_dim=int(cfg.get("hidden_dim", 64))
            ).to(self.device)
            self.physics = self.model.graph
            self.ema_model = copy.deepcopy(self.model).to(self.device)
            self.solver = V5DRRRSolver(
                rounds=self.rounds,
                mode="neural",
                controller=self.ema_model,
                num_candidates=int(cfg.get("anneal_K", 128)),
                num_steps=int(cfg.get("K_infer", 40)),
                chunk_size=int(cfg.get("anneal_chunk_size", 512)),
                beta_min=float(cfg.get("rrr_beta_min", 0.38)),
                beta_max=float(cfg.get("rrr_beta_max", 1.05)),
                init_scale=float(cfg.get("rrr_init_scale", 0.85)),
            ).to(self.device)
        elif self.variant == "v5c":
            self.model = V5CMPSSolver(
                rounds=self.rounds,
                bond_dim=int(cfg.get("bond_dim", 8)),
                d_bit=int(cfg.get("d_bit", 48)),
                n_heads=int(cfg.get("n_heads", 3)),
                n_blocks=int(cfg.get("n_blocks", 2)),
                d_ff_mult=int(cfg.get("d_ff_mult", 2)),
                share_blocks=bool(cfg.get("share_blocks", False)),
                grad_checkpointing=bool(cfg.get("grad_checkpointing", False)),
                use_carry_gru=bool(cfg.get("use_carry_gru", True)),
                grad_scale=self.grad_scale,
                branch_mix=float(cfg.get("branch_mix", 0.25)),
                num_candidates=int(cfg.get("anneal_K", 256)),
                num_steps=int(cfg.get("K_infer", 16)),
                chunk_size=int(cfg.get("anneal_chunk_size", 16)),
            ).to(self.device).float()
            self.ema_model = copy.deepcopy(self.model).to(self.device).float()
            self.solver = self.ema_model
            self.physics = self.model.dual_basis
        elif self.variant == "v5b":
            ndim = int(cfg.get("manifold_dim", 32))
            self.physics = V5BPhysicsEngine(manifold_dim=ndim, rounds=self.rounds).to(self.device)
            self.model = V5BBackbone(
                manifold_dim=ndim,
                rounds=self.rounds,
                d_bit=int(cfg.get("d_bit", 48)),
                n_heads=int(cfg.get("n_heads", 3)),
                n_blocks=int(cfg.get("n_blocks", 2)),
                d_ff_mult=int(cfg.get("d_ff_mult", 2)),
                share_blocks=bool(cfg.get("share_blocks", False)),
                grad_checkpointing=bool(cfg.get("grad_checkpointing", False)),
                use_carry_gru=bool(cfg.get("use_carry_gru", True)),
            ).to(self.device).float()
            self.ema_model = copy.deepcopy(self.model).to(self.device).float()
            self.solver = UnrolledSolverV5B(
                model=self.ema_model,
                physics=self.physics,
                num_candidates=int(cfg.get("anneal_K", 256)),
                num_steps=int(cfg.get("K_infer", 48)),
                damp=float(cfg.get("infer_damp", 0.25)),
                eta_max=self.eta_max,
                beta_max=self.beta_max,
                grad_scale=self.grad_scale,
                chunk_size=int(cfg.get("anneal_chunk_size", 2048)),
                w_xor=self.w_xor,
                lam_comm_end=float(cfg.get("lam_comm_end", 0.60)),
                lam_rank1_end=float(cfg.get("lam_align_end", 6.00)),
                escape_patience=int(cfg.get("escape_patience", 6)),
                escape_step_t=float(cfg.get("escape_step_t", 0.35)),
            )
        elif self.variant == "v5a":
            self.physics = V5APhysicsEngine(rounds=self.rounds).to(self.device)
            self.model = V5ABackbone(
                rounds=self.rounds,
                d_bit=int(cfg.get("d_bit", 48)),
                n_heads=int(cfg.get("n_heads", 3)),
                n_blocks=int(cfg.get("n_blocks", 2)),
                d_ff_mult=int(cfg.get("d_ff_mult", 2)),
                share_blocks=bool(cfg.get("share_blocks", False)),
                grad_checkpointing=bool(cfg.get("grad_checkpointing", False)),
                use_carry_gru=bool(cfg.get("use_carry_gru", True)),
            ).to(self.device).float()
            self.ema_model = copy.deepcopy(self.model).to(self.device).float()
            self.solver = UnrolledSolverV5A(
                model=self.ema_model,
                physics=self.physics,
                num_candidates=int(cfg.get("anneal_K", 256)),
                num_steps=int(cfg.get("K_infer", 48)),
                damp=float(cfg.get("infer_damp", 0.25)),
                eta_max=self.eta_max,
                beta_max=self.beta_max,
                grad_scale=self.grad_scale,
                chunk_size=int(cfg.get("anneal_chunk_size", 2048)),
                w_xor=self.w_xor,
                lam_comm_end=float(cfg.get("lam_comm_end", 0.60)),
                lam_align_end=float(cfg.get("lam_align_end", 6.00)),
                lam_assoc_end=float(cfg.get("lam_assoc_end", 2.50)),
            )

        assert self.model is not None and self.ema_model is not None
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self._src_params = list(self.model.parameters())
        self._ema_params = list(self.ema_model.parameters())
        self._src_buffers = [b for b in self.model.buffers() if b.dtype == torch.float32]
        self._ema_buffers = [b for b in self.ema_model.buffers() if b.dtype == torch.float32]

        opt_kwargs: Dict[str, Any] = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "betas": (0.9, 0.999),
        }
        if self.device.type == "cuda":
            try:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), fused=True, **opt_kwargs)
                self._use_fused_adamw = True
            except (TypeError, RuntimeError):
                self.optimizer = torch.optim.AdamW(self.model.parameters(), **opt_kwargs)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), **opt_kwargs)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )
        self._configure_compile()

    def _configure_compile(self) -> None:
        if self.model is None:
            return
        pref = str(self.config.get("use_torch_compile", "auto")).lower()
        want = (
            (platform.system() == "Linux" and self.device.type == "cuda")
            if pref == "auto"
            else pref in ("true", "1", "yes", "on")
        )
        self.train_model = self.model
        if want and hasattr(torch, "compile"):
            try:
                if bool(self.config.get("allow_rnn", False)) and hasattr(torch, "_dynamo"):
                    torch._dynamo.config.allow_rnn = True
                # V5 Spec: Linux+CUDA ONLY mode="default" on training hot kernels; eval stays eager
                if self.variant in ("v5a", "v5b"):
                    self.train_model = torch.compile(self.model, mode="default")
                    if bool(self.config.get("compile_physics", True)):
                        self._compiled_energy_core = torch.compile(
                            self.physics.compute_energy_and_features_core, mode="default"
                        )
                        self._compiled_spin_add32 = torch.compile(
                            _spin_add32_flat, mode="default", dynamic=True
                        )
                        self._compiled_retract_clamp = torch.compile(
                            self.physics.retract_and_clamp, mode="default"
                        )
                        self._compiled_loss_head = torch.compile(_bce_wta_loss_head, mode="default")
                elif self.variant == "v5d_neural":
                    self.model.var_net = torch.compile(self.model.var_net, mode="default")
                    self.train_model = self.model
                self.compiled_active = True
            except Exception as exc:
                print(f"[Phase-SHA {self.variant.upper()}] compile fallback eager ({exc})", flush=True)
                self.train_model = self.model
                self.compiled_active = False

    def _lr_lambda(self, current_step: int) -> float:
        if self.warmup_steps > 0 and current_step < self.warmup_steps:
            return float(current_step + 1) / float(self.warmup_steps)
        denom = max(1, self.lr_total_steps - self.warmup_steps)
        progress = min(1.0, float(current_step - self.warmup_steps) / float(denom))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    @torch.no_grad()
    def _update_ema(self) -> None:
        if not self._ema_params:
            return
        decay = min(self.ema_decay, float(1 + self.step) / float(10 + self.step))
        torch._foreach_mul_(self._ema_params, decay)
        torch._foreach_add_(self._ema_params, self._src_params, alpha=1.0 - decay)
        for b_ema, b_src in zip(self._ema_buffers, self._src_buffers):
            b_ema.copy_(b_src)

    def start_prefetch(self) -> None:
        if self.variant == "v5d_pure":
            return
        if self._prefetcher is not None and self._prefetcher.is_running():
            return
        self._prefetcher = AsyncV4BatchPrefetcher(
            batch_size=self.batch_size,
            rounds=self.rounds,
            domain_mode=self.domain_mode,
            num_bits=self.num_bits,
            msg_bytes_len=self.msg_bytes_len,
            device=self.device,
            seed=int(self.config.get("seed", 5050)),
            train_w0_pool_cpu=self.train_w0_pool_cpu,
            holdout_w0_set=self.holdout_w0_set,
            queue_size=int(self.config.get("prefetch_queue_size", 3)),
        )
        self._prefetcher.start()

    def stop_prefetch(self) -> None:
        if self._prefetcher is not None:
            self._prefetcher.stop()
            self._prefetcher = None

    def _sample_train_batch(self) -> Tuple[torch.Tensor, Any]:
        if self._prefetcher is not None and self._prefetcher.is_running():
            return self._prefetcher.next_on_device()
        if self.train_w0_pool is not None:
            idx = torch.randint(
                0, self.train_w0_pool.shape[0], (self.batch_size,), device=self.device, generator=self.train_rng
            )
            w16 = pack_int_messages_to_w16(self.train_w0_pool[idx])
        else:
            w16 = sample_random_w16(
                batch_size=self.batch_size,
                mode=self.domain_mode,
                num_bits=self.num_bits,
                msg_bytes_len=self.msg_bytes_len,
                device=self.device,
                generator=self.train_rng,
            )
            if self.domain_mode == "int_range" and self.holdout_w0_gpu is not None:
                high = 1 << self.num_bits
                coll = torch.isin(w16[:, 0], self.holdout_w0_gpu)
                while bool(coll.any().item()):
                    n_c = int(coll.sum().item())
                    w16[coll, 0] = torch.randint(
                        0, high, (n_c,), dtype=torch.int64, device=self.device, generator=self.train_rng
                    )
                    coll = torch.isin(w16[:, 0], self.holdout_w0_gpu)
        return prepare_ground_truth_nodes(
            w16, rounds=self.rounds, domain_mode=self.domain_mode, num_bits=self.num_bits, msg_bytes_len=self.msg_bytes_len
        )

    def save_checkpoint(self, filename: str = "latest.pt") -> Path:
        ckpt_path = self.output_dir / filename
        payload: Dict[str, Any] = {
            "variant": self.variant,
            "step": self.step,
            "wall_s": self.wall_s,
            "best_eval_bce": self.best_eval_bce,
            "best_min_m32": self.best_min_m32,
            "best_preimage_rate": self.best_preimage_rate,
            "config": self.config,
        }
        if self.model is not None and self.ema_model is not None:
            payload["model_state"] = _strip_orig_mod(self.model.state_dict())
            payload["ema_model_state"] = _strip_orig_mod(self.ema_model.state_dict())
        if self.optimizer is not None:
            payload["optimizer_state"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            payload["scheduler_state"] = self.scheduler.state_dict()
        if self.variant == "v5d_pure" and self._pure_state is not None:
            payload["pure_state"] = self._pure_state.detach().cpu()
            payload["pure_best_m32"] = self._pure_best_m32.detach().cpu() if self._pure_best_m32 is not None else None
            payload["pure_best_m1"] = self._pure_best_m1.detach().cpu() if self._pure_best_m1 is not None else None
            payload["pure_best_x_disc"] = (
                self._pure_best_x_disc.detach().cpu() if self._pure_best_x_disc is not None else None
            )
            payload["pure_zero_cache"] = self._pure_zero_cache
        torch.save(payload, ckpt_path)
        return ckpt_path

    def load_checkpoint(self, ckpt_path: str | Path) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if self.model is not None and "model_state" in ckpt:
            self.model.load_state_dict(_strip_orig_mod(ckpt["model_state"]))
            self.model.float()
        if self.ema_model is not None:
            src = ckpt.get("ema_model_state", ckpt.get("model_state"))
            if src is not None:
                self.ema_model.load_state_dict(_strip_orig_mod(src))
                self.ema_model.float()
        if self.optimizer is not None and "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            if self._use_fused_adamw:
                for pg in self.optimizer.param_groups:
                    pg["fused"] = True
        if self.scheduler is not None and "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if self.variant == "v5d_pure" and ckpt.get("pure_state") is not None:
            self._init_pure_persistent_state()
            self._pure_state = ckpt["pure_state"].to(self.device)
            if ckpt.get("pure_best_m32") is not None:
                self._pure_best_m32 = ckpt["pure_best_m32"].to(self.device)
            if ckpt.get("pure_best_m1") is not None:
                self._pure_best_m1 = ckpt["pure_best_m1"].to(self.device)
            if ckpt.get("pure_best_x_disc") is not None:
                self._pure_best_x_disc = ckpt["pure_best_x_disc"].to(self.device)
            if ckpt.get("pure_zero_cache") is not None:
                self._pure_zero_cache = dict(ckpt["pure_zero_cache"])
        self.step = int(ckpt.get("step", 0))
        self.wall_s = float(ckpt.get("wall_s", 0.0))
        self.best_eval_bce = float(ckpt.get("best_eval_bce", float("inf")))
        self.best_min_m32 = float(ckpt.get("best_min_m32", float("inf")))
        self.best_preimage_rate = float(ckpt.get("best_preimage_rate", 0.0))

    def load_weights_only(self, ckpt_path: str | Path) -> None:
        if self.model is None or self.ema_model is None:
            return
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        src_state = _strip_orig_mod(ckpt.get("ema_model_state", ckpt.get("model_state")))
        if set(src_state.keys()) == set(self.model.state_dict().keys()):
            self.model.load_state_dict(src_state, strict=True)
            self.model.float()
            info_str = f"V5->V5 strict (step={ckpt.get('step', '?')})"
        elif hasattr(self.model, "load_v3_warmstart"):
            info = self.model.load_v3_warmstart(ckpt_path, map_location=self.device)
            info_str = f"V3->{self.variant.upper()} lossless (loaded={info['loaded_keys']}, fresh={len(info['missing_adapter_keys'])})"
        else:
            raise RuntimeError(f"Incompatible warm-start checkpoint for {self.variant}")
        self.ema_model.load_state_dict(self.model.state_dict(), strict=True)
        self.ema_model.float()
        print(f"[Phase-SHA {self.variant.upper()}] Warm-started from {ckpt_path} [{info_str}]", flush=True)

    # --------------------------------------------------------------------------
    # Variant Training Step Implementations
    # --------------------------------------------------------------------------

    def _init_pure_persistent_state(self) -> None:
        n_dig = min(int(self.config.get("pure_num_digests", 16)), self.holdout_w16.shape[0])
        k_traj = int(self.config.get("pure_trajs_per_digest", 128))
        w16_sub = self.holdout_w16[:n_dig].to(self.device)
        _, cb_sub = prepare_ground_truth_nodes(
            w16_sub, rounds=self.rounds, domain_mode=self.domain_mode, num_bits=self.num_bits, msg_bytes_len=self.msg_bytes_len
        )
        mb_rep = ManifoldBoundary.from_clamped_boundary(cb_sub).repeat_for_candidates(k_traj)
        var_m, var_v = self.solver.graph.build_boundary_clamps(mb_rep)
        x0 = self.solver._init_replicas(var_m, var_v, generator=self.train_rng)
        m_tot = x0.shape[0]
        beta_ladder = torch.linspace(self.solver.beta_min, self.solver.beta_max, k_traj, dtype=torch.float32, device=self.device)
        self._pure_beta = beta_ladder.repeat(n_dig)
        self._pure_state = x0
        self._pure_mask = var_m
        self._pure_vals = var_v
        self._pure_b_sub = mb_rep
        self._pure_drift = DriftMonitor()
        self._pure_best_m32 = torch.full((m_tot,), 1e9, dtype=torch.float32, device=self.device)
        self._pure_best_m1 = torch.full((m_tot,), 1e9, dtype=torch.float32, device=self.device)
        self._pure_best_x_disc = torch.zeros((m_tot, self.rounds, 3, 32), dtype=torch.float32, device=self.device)
        self._pure_last_gap_t = None

    @torch.no_grad()
    def _update_pure_persistent_snapshot(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._pure_state is not None and self._pure_mask is not None and self._pure_vals is not None
        assert self._pure_b_sub is not None and self._pure_best_m32 is not None and self._pure_best_m1 is not None
        assert self._pure_best_x_disc is not None
        m_tot = self._pure_state.shape[0]
        chunk_sz = max(64, int(self.config.get("anneal_chunk_size", 1024)))
        lam_full = torch.ones(32, dtype=torch.float32, device=self.device)

        cur_m1_parts: List[torch.Tensor] = []
        cur_m32_parts: List[torch.Tensor] = []
        cur_w0_parts: List[torch.Tensor] = []
        for c0 in range(0, m_tot, chunk_sz):
            c1 = min(c0 + chunk_sz, m_tot)
            _, _, _, x_nodes_c = self.solver.rrr.readout_shadow(
                self._pure_state[c0:c1], self._pure_mask[c0:c1], self._pure_vals[c0:c1]
            )
            x_disc_c = torch.where(x_nodes_c < 0.0, -torch.ones_like(x_nodes_c), torch.ones_like(x_nodes_c))
            mb_c = ManifoldBoundary(
                rounds=self.rounds,
                A_full_clamped=self._pure_b_sub.A_full_clamped[c0:c1],
                A_full_mask=self._pure_b_sub.A_full_mask[c0:c1],
                E_full_clamped=self._pure_b_sub.E_full_clamped[c0:c1],
                E_full_mask=self._pure_b_sub.E_full_mask[c0:c1],
                W_full_clamped=self._pure_b_sub.W_full_clamped[c0:c1],
                W_full_mask=self._pure_b_sub.W_full_mask[c0:c1],
                A_words_clamped=self._pure_b_sub.A_words_clamped[c0:c1],
                E_words_clamped=self._pure_b_sub.E_words_clamped[c0:c1],
                W16_clamped_words=self._pure_b_sub.W16_clamped_words[c0:c1],
                W16_mask_bits=self._pure_b_sub.W16_mask_bits[c0:c1],
                digest_target_bits=(
                    None if self._pure_b_sub.digest_target_bits is None else self._pure_b_sub.digest_target_bits[c0:c1]
                ),
                digest_target_mask=(
                    None if self._pure_b_sub.digest_target_mask is None else self._pure_b_sub.digest_target_mask[c0:c1]
                ),
                iv_words=self._pure_b_sub.iv_words,
                is_full_digest_clamped=self._pure_b_sub.is_full_digest_clamped,
            )
            cb_c = mb_c.to_clamped_boundary()
            x_disc_c = cb_c.apply_clamp(x_disc_c)
            a_d, e_d, w_d = cb_c.assemble_full(x_disc_c)
            _, _, stats = self.solver.dual_basis.compute_energy_and_features(a_d, e_d, w_d, lam=lam_full)
            m1_c = stats["per_scale"][:, 0]
            m32_c = stats["per_scale"][:, 31]

            improved = m32_c < self._pure_best_m32[c0:c1]
            if improved.any():
                self._pure_best_m32[c0:c1] = torch.where(improved, m32_c, self._pure_best_m32[c0:c1])
                self._pure_best_m1[c0:c1] = torch.where(improved, m1_c, self._pure_best_m1[c0:c1])
                self._pure_best_x_disc[c0:c1] = torch.where(
                    improved.view(-1, 1, 1, 1), x_disc_c, self._pure_best_x_disc[c0:c1]
                )
            cur_m1_parts.append(m1_c)
            cur_m32_parts.append(m32_c)
            cur_w0_parts.append(bits_to_words((x_disc_c[:, 0, 2, :] < 0.0).to(torch.int64)))

        return (
            torch.cat(cur_m1_parts, dim=0),
            torch.cat(cur_m32_parts, dim=0),
            torch.cat(cur_w0_parts, dim=0),
        )

    def _train_step_v5d_pure(self, sync_metrics: bool) -> Dict[str, float]:
        if self._pure_state is None:
            self._init_pure_persistent_state()
        assert self._pure_state is not None and self._pure_mask is not None and self._pure_vals is not None and self._pure_beta is not None
        assert self._pure_drift is not None and self._pure_b_sub is not None and self._pure_best_m32 is not None and self._pure_best_m1 is not None

        n_inner = max(1, int(self.config.get("pure_steps_per_tick", self.k_unroll)))
        gap_rms = torch.zeros(self._pure_state.shape[0], dtype=torch.float32, device=self.device)
        for _ in range(n_inner):
            self._pure_state, diag = self.solver.rrr.step(
                self._pure_state, self._pure_mask, self._pure_vals, beta=self._pure_beta
            )
            gap_rms = diag["gap_rms"]
        self._pure_drift.record(diag["x_rms_norm"])
        self._pure_last_gap_t = gap_rms.detach()
        self.step += 1
        if not sync_metrics:
            return {}

        n_dig = min(int(self.config.get("pure_num_digests", 16)), self.holdout_w16.shape[0])
        k_traj = int(self.config.get("pure_trajs_per_digest", 128))
        self._update_pure_persistent_snapshot()
        slope, flag = self._pure_drift.summarize()
        min_e32_mean = float(self._pure_best_m32.view(n_dig, k_traj).min(dim=1).values.mean().item())
        global_min_e32 = float(self._pure_best_m32.min().item())
        return {
            "loss": float(gap_rms.mean().item()),
            "phys_loss": min_e32_mean,
            "bce_loss": 0.6931,
            "bce_final": 0.6931,
            "xor_loss": float(gap_rms.min().item()),
            "comm_loss": global_min_e32,
            "align_loss": float(self._pure_beta.mean().item()),
            "logit_reg": 0.0,
            "frontier_rounds": 0.0,
            "rrr_gap": float(gap_rms.mean().item()),
            "rrr_beta": float(self._pure_beta.mean().item()),
            "drift_slope": float(slope.mean().item()),
            "drift_flag": float(flag.max().item()),
            "aux1_name": "beta",
            "aux1_val": float(self._pure_beta.mean().item()),
            "aux2_name": "minE",
            "aux2_val": min_e32_mean,
        }

    def _train_step_v5d_neural(self, sync_metrics: bool) -> Dict[str, float]:
        x_star_base, cb_base = self._sample_train_batch()
        m = max(1, self.num_particles)
        x_star = x_star_base.repeat_interleave(m, dim=0) if m > 1 else x_star_base
        mb = ManifoldBoundary.from_clamped_boundary(cb_base)
        if m > 1:
            mb = mb.repeat_for_candidates(m)
        bm = mb.batch_size
        b = bm // m

        var_mask, var_vals = self.model.graph.build_boundary_clamps(mb)
        x_cur = self.solver._init_replicas(var_mask, var_vals, generator=self.train_rng)
        rrr = RRRDynamicsOperator(self.model.graph, ste_temp=4.0)
        cb = mb.to_clamped_boundary()
        free_mask = 1.0 - cb.clamp_mask
        free_count = free_mask.sum().clamp_min(1.0)
        lam_full = torch.ones(32, dtype=torch.float32, device=self.device)
        b_offsets = torch.arange(b, device=self.device) * m

        self.optimizer.zero_grad(set_to_none=True)
        n_steps = max(2, self.k_unroll)
        chunk = max(1, self.bptt_chunk_len)
        acc_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        acc_phys = torch.zeros((), dtype=torch.float32, device=self.device)
        acc_bce = torch.zeros((), dtype=torch.float32, device=self.device)
        acc_gap = torch.zeros((), dtype=torch.float32, device=self.device)
        acc_beta = torch.zeros((), dtype=torch.float32, device=self.device)
        last_bce = acc_loss
        drift = DriftMonitor()
        grouped_loss: Optional[torch.Tensor] = None
        denom = float(max(1, n_steps - 1))
        w_sum = sum(float(s + 1) for s in range(n_steps))

        for s in range(n_steps):
            sf = torch.full((bm, 1), float(s) / denom, dtype=torch.float32, device=self.device)
            rep_w, beta_t, logits_s, _ = self.train_model(x_cur, var_mask, var_vals, sf)
            x_next, diag = rrr.step(x_cur, var_mask, var_vals, beta=beta_t, rep_weights=rep_w, ste_temp=4.0)
            drift.record(diag["x_rms_norm"])
            # Shadow P_A readout loss
            a_f, e_f, w_f, x_nodes = rrr.readout_shadow(x_next, var_mask, var_vals, rep_weights=rep_w)
            phys_e, _, _ = self.model.dual_basis.compute_energy_and_features(a_f, e_f, w_f, lam=lam_full)
            gap_sq = diag["gap_rms"].square()
            e_bm = (phys_e / (32.0 * float(self.model.dual_basis.num_eq_rounds)) + gap_sq).view(b, m)
            wta_e = self.wta_winner_weight * e_bm.min(dim=1).values + (1.0 - self.wta_winner_weight) * e_bm.mean(dim=1)
            loss_phys = wta_e.mean()

            y_tgt = 0.5 * (1.0 + x_star)
            bce_val = (F.binary_cross_entropy_with_logits(2.0 * logits_s, y_tgt, reduction="none") * free_mask).sum() / free_count
            drift_pen = 0.01 * F.relu(diag["x_rms_norm"] - 3.0).square().mean()
            step_l = self.alpha_phys * loss_phys + self.bce_weight * bce_val + drift_pen
            w_s = float(s + 1) / w_sum
            grouped_loss = (w_s * step_l) if grouped_loss is None else (grouped_loss + w_s * step_l)

            if sync_metrics:
                acc_loss = acc_loss + step_l.detach() * w_s
                acc_phys = acc_phys + loss_phys.detach() * w_s
                acc_bce = acc_bce + bce_val.detach() * w_s
                acc_gap = acc_gap + diag["gap_rms"].mean().detach() * w_s
                acc_beta = acc_beta + beta_t.mean().detach() * w_s
                last_bce = bce_val.detach()

            is_chunk_end = ((s + 1) % chunk == 0) or (s == n_steps - 1)
            if is_chunk_end and grouped_loss is not None:
                grouped_loss.backward()
                grouped_loss = None
                x_cur = x_next.detach()
            else:
                x_cur = x_next

        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip, foreach=True)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        self._update_ema()
        if not sync_metrics:
            return {}
        slope, flag = drift.summarize()
        return {
            "loss": float(acc_loss.item()),
            "phys_loss": float(acc_phys.item()),
            "bce_loss": float(acc_bce.item()),
            "bce_final": float(last_bce.item()),
            "xor_loss": float(acc_gap.item()),
            "comm_loss": float(slope.mean().item()),
            "align_loss": float(acc_beta.item()),
            "logit_reg": 0.0,
            "frontier_rounds": 0.0,
            "rrr_gap": float(acc_gap.item()),
            "rrr_beta": float(acc_beta.item()),
            "drift_slope": float(slope.mean().item()),
            "drift_flag": float(flag.max().item()),
            "aux1_name": "beta",
            "aux1_val": float(acc_beta.item()),
            "aux2_name": "gap",
            "aux2_val": float(acc_gap.item()),
        }

    def _train_step_v5c_mps(self, sync_metrics: bool) -> Dict[str, float]:
        x_star, cb = self._sample_train_batch()
        mb = ManifoldBoundary.from_clamped_boundary(cb)
        bsz = mb.batch_size
        free_mask = 1.0 - cb.clamp_mask
        free_count = free_mask.sum().clamp_min(1.0)
        target_chain = self.model.nodes_to_chain(x_star)

        cores = self.model.init_cores(mb)
        h_prev: Optional[torch.Tensor] = None
        self.optimizer.zero_grad(set_to_none=True)

        n_steps = max(2, self.k_unroll)
        chunk = max(1, self.bptt_chunk_len)
        denom = float(max(1, n_steps - 1))
        w_sum = sum(float(s + 1) for s in range(n_steps))
        lam_full = torch.ones(32, dtype=torch.float32, device=self.device)
        lam_norm = 32.0 * float(self.model.dual_basis.num_eq_rounds)

        acc_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        acc_phys = torch.zeros((), dtype=torch.float32, device=self.device)
        acc_bce = torch.zeros((), dtype=torch.float32, device=self.device)
        acc_ent = torch.zeros((), dtype=torch.float32, device=self.device)
        acc_lreg = torch.zeros((), dtype=torch.float32, device=self.device)
        last_bce = acc_loss
        grouped_loss: Optional[torch.Tensor] = None

        for s in range(n_steps):
            sf = torch.full((bsz,), float(s) / denom, dtype=torch.float32, device=self.device)
            cores_next, logits_s, _, h_next, _ = self.model.forward_step(cores, mb, sf, h_prev=h_prev)

            # Exact joint MPS NLL + loop energy + marginal BCE
            nll_joint = (-self.model.mps.joint_log_prob(cores_next, target_chain) / float(3 * self.rounds)).mean()
            ent_e = self.model.mps.compute_entangled_energy(cores_next, self.model.loop_pairs).mean()
            x_soft = torch.tanh(logits_s)
            a_f, e_f, w_f = cb.assemble_full(cb.apply_clamp(x_soft))
            phys_e, _, _ = self.model.dual_basis.compute_energy_and_features(a_f, e_f, w_f, lam=lam_full)
            loss_phys = (phys_e / lam_norm).mean()

            y_tgt = 0.5 * (1.0 + x_star)
            bce_val = (F.binary_cross_entropy_with_logits(2.0 * logits_s, y_tgt, reduction="none") * free_mask).sum() / free_count
            lreg = (logits_s.square() * free_mask).sum() / free_count
            step_l = (
                self.alpha_phys * loss_phys
                + 0.5 * nll_joint
                + 0.25 * ent_e
                + self.bce_weight * bce_val
                + self.logit_reg_weight * lreg
            )
            w_s = float(s + 1) / w_sum
            grouped_loss = (w_s * step_l) if grouped_loss is None else (grouped_loss + w_s * step_l)

            if sync_metrics:
                acc_loss = acc_loss + step_l.detach() * w_s
                acc_phys = acc_phys + loss_phys.detach() * w_s
                acc_bce = acc_bce + bce_val.detach() * w_s
                acc_ent = acc_ent + ent_e.detach() * w_s
                acc_lreg = acc_lreg + lreg.detach() * w_s
                last_bce = bce_val.detach()

            is_chunk_end = ((s + 1) % chunk == 0) or (s == n_steps - 1)
            if is_chunk_end and grouped_loss is not None:
                grouped_loss.backward()
                grouped_loss = None
                cores = cores_next.detach()
                h_prev = h_next.detach()
            else:
                cores = cores_next
                h_prev = h_next

        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip, foreach=True)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        self._update_ema()
        if not sync_metrics:
            return {}
        ent_s = float(self.model.mps.compute_entanglement_entropy(cores, cut_idx=1).mean().item())
        return {
            "loss": float(acc_loss.item()),
            "phys_loss": float(acc_phys.item()),
            "bce_loss": float(acc_bce.item()),
            "bce_final": float(last_bce.item()),
            "xor_loss": float(acc_ent.item()),
            "comm_loss": ent_s,
            "align_loss": float(acc_ent.item()),
            "logit_reg": float(acc_lreg.item()),
            "frontier_rounds": 0.0,
            "ent_entropy": ent_s,
            "ent_energy": float(acc_ent.item()),
            "aux1_name": "S_ent",
            "aux1_val": ent_s,
            "aux2_name": "E_ent",
            "aux2_val": float(acc_ent.item()),
        }

    def _train_step_v5_manifold_bptt(self, sync_metrics: bool) -> Dict[str, float]:
        """Shared optimized zero-sync Chunked-BPTT trainer for V5A (Octonion) & V5B (Burer-Monteiro)."""
        x_star_base, cb_base = self._sample_train_batch()
        m = max(1, self.num_particles)
        x_star = x_star_base.repeat_interleave(m, dim=0) if m > 1 else x_star_base
        mb = ManifoldBoundary.from_clamped_boundary(cb_base)
        if m > 1:
            mb = mb.repeat_for_candidates(m)

        manifold = self.physics.manifold
        u_init = build_v4_init(
            boundary=mb, manifold=manifold, num_particles=m, zero_prob=self.zero_init_prob, generator=self.train_rng
        )
        step_damp = self.damp
        if self.damp_jitter_min is not None and self.damp_jitter_max is not None:
            u_d = float(torch.rand((), generator=self.damp_rng).item())
            step_damp = float(self.damp_jitter_min) + (float(self.damp_jitter_max) - float(self.damp_jitter_min)) * u_d

        self.optimizer.zero_grad(set_to_none=True)
        bm = u_init.shape[0]
        b = bm // m
        prep = mb.prepare_for_manifold(manifold)
        free_count = prep.free_mask.sum().clamp_min(1.0)
        full_lam = torch.ones(32, dtype=torch.float32, device=self.device)
        wall_mask = torch.zeros((1, self.rounds, 1, 1), dtype=torch.float32, device=self.device)
        wall_bce_mul = torch.ones((self.rounds,), dtype=torch.float32, device=self.device)
        for wr in self.closed_loop_wall_rounds:
            if 0 <= int(wr) < self.rounds:
                wall_mask[:, int(wr)] = 1.0
                wall_bce_mul[int(wr)] = self.closed_loop_bce_scale
        wall_free = prep.free_mask * wall_mask
        wall_den = wall_free.sum().clamp_min(1.0)
        b_offsets = torch.arange(b, device=self.device) * m

        denom = float(max(1, self.k_unroll - 1))
        frac_vals = [float(s) / denom for s in range(self.k_unroll)]
        frac_vec = torch.tensor(frac_vals, dtype=torch.float32, device=self.device)
        step_frac_mat = frac_vec.unsqueeze(1).expand(self.k_unroll, bm).contiguous()
        lam_comm_vec = torch.tensor(
            [self.sched_comm.value(self.step, self.total_steps, f) for f in frac_vals],
            dtype=torch.float32,
            device=self.device,
        )
        lam_align_vec = torch.tensor(
            [self.sched_align.value(self.step, self.total_steps, f) for f in frac_vals],
            dtype=torch.float32,
            device=self.device,
        )
        lam_assoc_vec = torch.tensor(
            [self.sched_assoc.value(self.step, self.total_steps, f) for f in frac_vals],
            dtype=torch.float32,
            device=self.device,
        )
        eta_vec = torch.tensor([self.eta_max * (0.2 + 0.8 * f) for f in frac_vals], dtype=torch.float32, device=self.device)
        beta_vec = torch.tensor([self.beta_max * (f * f) for f in frac_vals], dtype=torch.float32, device=self.device)
        damp_t = torch.tensor(float(step_damp), dtype=torch.float32, device=self.device)
        w_xor_t = torch.tensor(float(self.w_xor), dtype=torch.float32, device=self.device)

        rc_fn = self.physics.retract_and_clamp if self._compiled_retract_clamp is None else self._compiled_retract_clamp
        energy_fn = self.physics.compute_energy_and_features_core if self._compiled_energy_core is None else self._compiled_energy_core
        head_fn = _bce_wta_loss_head if self._compiled_loss_head is None else self._compiled_loss_head
        add32_fn = self._compiled_spin_add32

        u_cur = apply_clamp_prepared(u_init.detach(), prep)
        if self.closed_loop_schedule and self.rounds > 16:
            u_cur = apply_clamp_prepared(
                apply_manifold_closed_loop_schedule(u_cur, manifold, self.rounds, add32_fn=add32_fn), prep
            )

        h_prev: Optional[torch.Tensor] = None
        grouped_loss: Optional[torch.Tensor] = None
        chunk_idx = 0
        chunk_len = max(1, self.bptt_chunk_len)
        group_chunks = self.backward_group_chunks
        raw_weights = [float(s + 1) for s in range(self.k_unroll)]
        w_sum = sum(raw_weights)
        step_weights = [w / w_sum for w in raw_weights]
        lam_norm = 32.0 * float(self.physics.dual_basis.num_eq_rounds)

        acc_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        acc_bce = torch.zeros((), dtype=torch.float32, device=self.device)
        acc_phys = torch.zeros((), dtype=torch.float32, device=self.device)
        acc_xor = torch.zeros((), dtype=torch.float32, device=self.device)
        acc_comm = torch.zeros((), dtype=torch.float32, device=self.device)
        acc_align = torch.zeros((), dtype=torch.float32, device=self.device)
        acc_assoc = torch.zeros((), dtype=torch.float32, device=self.device)
        acc_lreg = torch.zeros((), dtype=torch.float32, device=self.device)
        last_bce_t = acc_loss
        last_front_t = acc_loss
        lam_b = self.physics.dual_basis._normalize_lambda(full_lam, bm, self.device)

        for s in range(self.k_unroll):
            is_v5a = self.variant == "v5a"
            extra_kw = {"lam_assoc": lam_assoc_vec[s]} if is_v5a else {}
            grad_radar, res_feat, _, grad_riem = self.physics.compute_radar_prepared(
                u_nodes=u_cur,
                prep=prep,
                lam=full_lam,
                grad_scale=self.grad_scale,
                w_xor=w_xor_t,
                lam_comm=lam_comm_vec[s],
                lam_align=lam_align_vec[s],
                energy_fn=energy_fn,
                return_raw_grad=True,
                **extra_kw,
            )
            logits_s, delta_u, h_tilde = self.train_model(
                u_nodes=u_cur,
                grad_radar=grad_radar,
                clamp_mask=prep.clamp_mask,
                res_features=res_feat,
                k_spins=self.physics.K_spins,
                step_frac=step_frac_mat[s],
                lam=full_lam,
                h_prev=h_prev,
            )
            h_next = (1.0 - damp_t) * h_prev + damp_t * h_tilde if (self.sync_h_damp and h_prev is not None) else h_tilde
            u_next = rc_fn(
                u_cur, logits_s, delta_u, grad_riem, prep.u_clamped_masked, prep.free_mask_exp, damp_t, eta_vec[s], beta_vec[s]
            )
            if self.closed_loop_schedule and self.rounds > 16:
                u_sched = apply_manifold_closed_loop_schedule(u_next, manifold, self.rounds, add32_fn=add32_fn)
                u_next = normalize_manifold(prep.u_clamped_masked + prep.free_mask_exp * u_sched.to(torch.float32))

            u_a, u_e, u_w = assemble_full_prepared(u_next, prep, manifold)
            if is_v5a:
                core_out = energy_fn(
                    u_a, u_e, u_w, prep.free_mask, lam_b, w_xor_t, lam_comm_vec[s], lam_align_vec[s], lam_assoc_vec[s],
                    prep.digest_target_bits, prep.digest_target_mask, prep.iv_words, 4.0, prep.has_free_right, prep.has_digest_prefix
                )
                total_e, m1_e, m32_e, e_xor, e_comm, e_align, e_assoc = (
                    core_out[0], core_out[6], core_out[7], core_out[9], core_out[10], core_out[11], core_out[12]
                )
            else:
                core_out = energy_fn(
                    u_a, u_e, u_w, prep.free_mask, lam_b, w_xor_t, lam_comm_vec[s], lam_align_vec[s],
                    prep.digest_target_bits, prep.digest_target_mask, prep.iv_words, 4.0, prep.has_free_right, prep.has_digest_prefix
                )
                total_e, m1_e, m32_e, e_xor, e_comm, e_align = (
                    core_out[0], core_out[6], core_out[7], core_out[9], core_out[10], core_out[11]
                )
                e_assoc = torch.zeros_like(e_align)

            step_loss, bce_t, phys_t, _, lreg_t, front_t, _, _ = head_fn(
                logits_s, x_star, prep.free_mask, free_count, wall_free, wall_den, wall_bce_mul,
                b_offsets, total_e, m1_e, m32_e, frac_vec[s], m, self.wta_winner_weight,
                self.alpha_phys, self.bce_weight, self.anti_collapse_weight, self.logit_reg_weight,
                self.wavefront_weight, lam_norm,
            )
            w_s = step_weights[s]
            grouped_loss = (w_s * step_loss) if grouped_loss is None else (grouped_loss + w_s * step_loss)

            if sync_metrics:
                acc_loss = acc_loss + step_loss.detach() * w_s
                acc_bce = acc_bce + bce_t * w_s
                acc_phys = acc_phys + phys_t * w_s
                acc_xor = acc_xor + e_xor.mean().detach() * w_s
                acc_comm = acc_comm + e_comm.mean().detach() * w_s
                acc_align = acc_align + e_align.mean().detach() * w_s
                acc_assoc = acc_assoc + e_assoc.mean().detach() * w_s
                acc_lreg = acc_lreg + lreg_t * w_s
                last_bce_t = bce_t
                last_front_t = front_t

            is_chunk_end = ((s + 1) % chunk_len == 0) or (s == self.k_unroll - 1)
            is_group_end = (s == self.k_unroll - 1) or (is_chunk_end and ((chunk_idx + 1) % group_chunks == 0))
            if is_group_end and grouped_loss is not None:
                grouped_loss.backward()
                grouped_loss = None
            if is_chunk_end:
                chunk_idx += 1
                u_cur = u_next.detach()
                h_prev = h_next.detach()
            else:
                u_cur = u_next
                h_prev = h_next

        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip, foreach=True)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        self._update_ema()
        if not sync_metrics:
            return {}

        if self.variant == "v5b":
            r1, er = compute_bm_rank1_and_eff_rank(u_cur, prep.free_mask)
            return {
                "loss": float(acc_loss.item()),
                "phys_loss": float(acc_phys.item()),
                "bce_loss": float(acc_bce.item()),
                "bce_final": float(last_bce_t.item()),
                "xor_loss": float(acc_xor.item()),
                "comm_loss": float(acc_comm.item()),
                "align_loss": float(acc_align.item()),
                "logit_reg": float(acc_lreg.item()),
                "frontier_rounds": float(last_front_t.item()),
                "rank1_ratio": r1,
                "eff_rank": er,
                "aux1_name": "rank1",
                "aux1_val": r1,
                "aux2_name": "effR",
                "aux2_val": er,
            }
        return {
            "loss": float(acc_loss.item()),
            "phys_loss": float(acc_phys.item()),
            "bce_loss": float(acc_bce.item()),
            "bce_final": float(last_bce_t.item()),
            "xor_loss": float(acc_xor.item()),
            "comm_loss": float(acc_comm.item()),
            "align_loss": float(acc_align.item()),
            "logit_reg": float(acc_lreg.item()),
            "frontier_rounds": float(last_front_t.item()),
            "train_assoc_loss": float(acc_assoc.item()),
            "aux1_name": "assoc",
            "aux1_val": float(acc_assoc.item()),
            "aux2_name": "align",
            "aux2_val": float(acc_align.item()),
        }

    def train_one_step(self, sync_metrics: bool = True) -> Dict[str, float]:
        if self.variant == "v5d_pure":
            return self._train_step_v5d_pure(sync_metrics=sync_metrics)
        if self.variant == "v5d_neural":
            self.model.train()
            self.train_model.train()
            return self._train_step_v5d_neural(sync_metrics=sync_metrics)
        if self.variant == "v5c":
            self.model.train()
            return self._train_step_v5c_mps(sync_metrics=sync_metrics)
        self.model.train()
        self.train_model.train()
        return self._train_step_v5_manifold_bptt(sync_metrics=sync_metrics)

    @torch.no_grad()
    def _evaluate_v5d_pure_holdout(self, eval_bs: int, anneal_samples: int) -> Dict[str, float]:
        def zero_fn(mb: ManifoldBoundary):
            return self.solver.infer_deterministic_zero(mb)

        def solve_fn(digests: torch.Tensor, mb: ManifoldBoundary, rng: torch.Generator):
            return self.solver.solve(target_digests=digests, boundary=mb, generator=rng)

        if self.step == 0 or self._pure_zero_cache is None:
            base_res = _evaluate_common_holdout(
                rounds=self.rounds,
                holdout_w16=self.holdout_w16,
                domain_mode=self.domain_mode,
                num_bits=self.num_bits,
                msg_bytes_len=self.msg_bytes_len,
                device=self.device,
                eval_bs=eval_bs,
                anneal_samples=anneal_samples,
                zero_infer_fn=zero_fn,
                solve_fn=solve_fn,
            )
            self._pure_zero_cache = {
                k: float(v)
                for k, v in base_res.items()
                if k == "eval_bce_loss" or k.startswith("zero_")
            }
            if self.step == 0:
                return base_res

        if self._pure_state is None:
            self._init_pure_persistent_state()

        n_dig = min(int(self.config.get("pure_num_digests", 16)), self.holdout_w16.shape[0])
        k_traj = int(self.config.get("pure_trajs_per_digest", 128))
        _, _, cur_w0 = self._update_pure_persistent_snapshot()

        assert self._pure_best_m32 is not None and self._pure_best_m1 is not None and self._pure_best_x_disc is not None
        assert self._pure_mask is not None and self._pure_vals is not None and self._pure_beta is not None
        assert self._pure_drift is not None

        w16_sub = self.holdout_w16[:n_dig].to(self.device)
        x_star_sub, cb_sub = prepare_ground_truth_nodes(
            w16_sub, rounds=self.rounds, domain_mode=self.domain_mode, num_bits=self.num_bits, msg_bytes_len=self.msg_bytes_len
        )
        mb_sub = ManifoldBoundary.from_clamped_boundary(cb_sub)
        digests = sha256_digest(w16_sub, rounds=self.rounds)

        x_best_all = self._pure_best_x_disc.view(n_dig, k_traj, self.rounds, 3, 32)
        m32_mat = self._pure_best_m32.view(n_dig, k_traj)
        m1_mat = self._pure_best_m1.view(n_dig, k_traj)

        k_w = min(16, self.rounds)
        w_pred_bits = (x_best_all[:, :, :k_w, 2, :] < 0.0).to(torch.int64)
        if k_w < 16:
            pad_b = torch.zeros((n_dig, k_traj, 16 - k_w, 32), dtype=torch.int64, device=self.device)
            w_pred_bits = torch.cat([w_pred_bits, pad_b], dim=2)
        w16_m_int = (mb_sub.W16_mask_bits > 0.5).to(torch.int64).unsqueeze(1)
        bits_clamp = words_to_bits(mb_sub.W16_clamped_words).unsqueeze(1)
        w16_cand = bits_to_words(torch.where(w16_m_int > 0, bits_clamp, w_pred_bits))

        chunk_d = max(16, int(self.config.get("anneal_chunk_size", 1024)))
        flat_w16 = w16_cand.reshape(-1, 16)
        dig_parts: List[torch.Tensor] = []
        for c0 in range(0, flat_w16.shape[0], chunk_d):
            dig_parts.append(sha256_digest(flat_w16[c0 : c0 + chunk_d], rounds=self.rounds))
        dig_cand = torch.cat(dig_parts, dim=0).view(n_dig, k_traj, 8)
        if mb_sub.is_full_digest_clamped:
            hits_mat = (dig_cand == digests.unsqueeze(1)).all(dim=-1)
        elif mb_sub.digest_target_mask is not None and mb_sub.digest_target_bits is not None:
            dig_bits = words_to_bits(dig_cand).to(torch.float32)
            tgt_b = mb_sub.digest_target_bits.unsqueeze(1).to(torch.float32)
            tgt_m = mb_sub.digest_target_mask.unsqueeze(1).to(torch.float32)
            hits_mat = ((dig_bits - tgt_b).abs() * tgt_m).sum(dim=(-2, -1)) < 0.5
        else:
            hits_mat = torch.zeros((n_dig, k_traj), dtype=torch.bool, device=self.device)

        solved = hits_mat.sum(dim=1) > 0
        best_idx = m32_mat.argmin(dim=1)
        if solved.any():
            best_idx = torch.where(hits_mat, m32_mat - 1e6, m32_mat).argmin(dim=1)
        b_idx = torch.arange(n_dig, device=self.device)

        best_x_disc = x_best_all[b_idx, best_idx]
        best_m1 = m1_mat[b_idx, best_idx]
        best_m32 = m32_mat[b_idx, best_idx]

        cur_w0_list = cur_w0.view(n_dig, k_traj).detach().cpu().tolist()
        mean_uniq = sum(float(len(set(row))) / float(max(1, k_traj)) for row in cur_w0_list) / max(1, n_dig)

        free_sub = 1.0 - mb_sub.clamp_mask
        match_sub = (((best_x_disc < 0) == (x_star_sub < 0)).to(torch.float64) * free_sub.to(torch.float64))
        pos_counts = free_sub.to(torch.float64).sum(dim=(0, 1, 2)).clamp_min(1.0)
        pos_acc = (match_sub.sum(dim=(0, 1, 2)) / pos_counts).cpu().tolist()
        round_acc = _compute_per_round_acc(best_x_disc, x_star_sub, free_sub)
        w0_free_s = free_sub[:, 0, 2, :].to(torch.float64)
        w0_acc = float(match_sub[:, 0, 2, :].sum().item() / max(1.0, float(w0_free_s.sum().item())))

        n_gap = min(256, self._pure_state.shape[0])
        gap_val = (
            float(self._pure_last_gap_t.mean().item())
            if self._pure_last_gap_t is not None
            else float(self.solver.rrr.gap_norm(self._pure_state[:n_gap], self._pure_mask[:n_gap], self._pure_vals[:n_gap]).mean().item())
        )
        slope, flag = self._pure_drift.summarize()

        out: Dict[str, float] = dict(self._pure_zero_cache)
        out["exact_preimage_rate@256"] = float(solved.float().mean().item())
        out["unique_ratio@256"] = float(mean_uniq)
        out["anneal_latch_ratio"] = 0.0
        out["w0_free_bit_acc"] = w0_acc
        out["phys_energy_m1"] = float(best_m1.mean().item())
        out["phys_energy_m32"] = float(best_m32.mean().item())
        out["min_phys_energy_m32"] = float(m32_mat.min(dim=1).values.mean().item())
        out["best_global_m32_energy"] = float(m32_mat.min().item())
        out["rrr_gap"] = gap_val
        out["rrr_beta"] = float(self._pure_beta.mean().item())
        out["drift_slope"] = float(slope.mean().item())
        out["drift_flag"] = float(flag.max().item())
        for t in range(self.rounds):
            out[f"per_round_acc_{t}"] = float(round_acc[t])
        for k in range(32):
            out[f"bit_acc_pos_{k}"] = float(pos_acc[k])
        return out

    @torch.no_grad()
    def evaluate_holdout(self, holdout_w16_override: Optional[torch.Tensor] = None) -> Dict[str, float]:
        if self.ema_model is not None:
            self.ema_model.eval()
        eval_bs = int(self.config.get("eval_batch_size", 256))
        anneal_samples = int(self.config.get("anneal_eval_samples", 64))
        if self.variant == "v5d_pure" and holdout_w16_override is None:
            return self._evaluate_v5d_pure_holdout(eval_bs=eval_bs, anneal_samples=anneal_samples)
        target_w16 = self.holdout_w16 if holdout_w16_override is None else holdout_w16_override

        def zero_fn(mb: ManifoldBoundary):
            return self.solver.infer_deterministic_zero(mb)

        def solve_fn(digests: torch.Tensor, mb: ManifoldBoundary, rng: torch.Generator):
            return self.solver.solve(target_digests=digests, boundary=mb, generator=rng)

        return _evaluate_common_holdout(
            rounds=self.rounds,
            holdout_w16=target_w16,
            domain_mode=self.domain_mode,
            num_bits=self.num_bits,
            msg_bytes_len=self.msg_bytes_len,
            device=self.device,
            eval_bs=eval_bs,
            anneal_samples=anneal_samples,
            zero_infer_fn=zero_fn,
            solve_fn=solve_fn,
        )

    def _format_v5_probe_str(self, m: Dict[str, float]) -> str:
        if self.variant.startswith("v5d"):
            return (
                f"rrr_gap={m.get('rrr_gap', 0.0):.4f} "
                f"beta={m.get('rrr_beta', 0.0):.3f} "
                f"drift={m.get('drift_slope', 0.0):+.4f}(flag={int(m.get('drift_flag', 0.0))})"
            )
        if self.variant == "v5c":
            return f"ent_S={m.get('ent_entropy', 0.0):.4f} ent_E={m.get('ent_energy', 0.0):.4f}"
        if self.variant == "v5b":
            return (
                f"rank1={m.get('rank1_ratio', 0.0):.4f} "
                f"eff_rank={m.get('eff_rank', 0.0):.2f} "
                f"esc={int(m.get('escape_count', 0.0))}"
            )
        return f"assoc={m.get('zero_assoc', 0.0):.4f}"

    def train(self, max_steps: Optional[int] = None) -> None:
        target_steps = self.total_steps if max_steps is None else int(max_steps)
        log_every = int(self.config.get("log_every", 50))
        eval_every = int(self.config.get("eval_every", 250))
        ckpt_every = int(self.config.get("checkpoint_every", 250))

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        extra_desc = ""
        if self.variant == "v5c":
            extra_desc = f"chi={self.config.get('bond_dim', 8)}, "
        elif self.variant == "v5b":
            extra_desc = f"N={self.config.get('manifold_dim', 32)}, "
        elif self.variant == "v5a":
            extra_desc = "D=8(Oct), "

        print(
            f"[Phase-SHA {self.variant.upper()}] Starting FP32 run: {extra_desc}R={self.rounds}, "
            f"B={self.batch_size}xM={self.num_particles}, K_unroll={self.k_unroll}, "
            f"compiled={self.compiled_active}, device={self.device}, step={self.step}/{target_steps}",
            flush=True,
        )

        if self.step == 0:
            t_eval0 = time.perf_counter()
            eval_0 = self.evaluate_holdout()
            self.wall_s += time.perf_counter() - t_eval0
            V5MetricsLogger.print_eval_block(
                self.rounds, eval_0, self.holdout_w16.shape[0], self._format_v5_probe_str(eval_0), step=0
            )
            row0: Dict[str, Any] = {
                "step": 0,
                "wall_s": round(self.wall_s, 3),
                "lr": float(self.optimizer.param_groups[0]["lr"]) if self.optimizer else 0.0,
            }
            row0.update(eval_0)
            self.logger.append_row(row0)
            if eval_0["min_phys_energy_m32"] < self.best_min_m32:
                self.best_min_m32 = eval_0["min_phys_energy_m32"]
                self.best_eval_bce = eval_0["eval_bce_loss"]
                self.save_checkpoint(self.best_ckpt_name)

        if bool(self.config.get("use_async_prefetch", True)) and self.device.type == "cuda":
            self.start_prefetch()

        try:
            interval_t0 = time.perf_counter()
            interval_steps = 0
            while self.step < target_steps:
                next_step = self.step + 1
                do_eval = (next_step % eval_every == 0) or (next_step == target_steps)
                do_log = (next_step % log_every == 0) or do_eval or (self.step == 0)

                step_t0 = time.perf_counter() if do_log else 0.0
                train_metrics = self.train_one_step(sync_metrics=do_log)
                interval_steps += 1

                if do_log:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                        peak_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                    else:
                        peak_gb = 0.0
                    now = time.perf_counter()
                    elapsed_int = now - interval_t0
                    self.wall_s += elapsed_int
                    dt_ms = (elapsed_int * 1000.0) / float(max(1, interval_steps)) if interval_steps > 1 else (now - step_t0) * 1000.0
                    cur_lr = float(self.optimizer.param_groups[0]["lr"]) if self.optimizer else 0.0

                    V5MetricsLogger.print_train_line(
                        step=self.step,
                        target_steps=target_steps,
                        lr=cur_lr,
                        loss=train_metrics["loss"],
                        phys=train_metrics["phys_loss"],
                        bce=train_metrics["bce_loss"],
                        xor_v=train_metrics["xor_loss"],
                        aux1_name=train_metrics["aux1_name"],
                        aux1_val=train_metrics["aux1_val"],
                        aux2_name=train_metrics["aux2_name"],
                        aux2_val=train_metrics["aux2_val"],
                        vram_gb=peak_gb,
                        dt_ms=dt_ms,
                        wall_s=self.wall_s,
                    )

                    eval_metrics: Dict[str, float] = {}
                    if do_eval:
                        t_ev = time.perf_counter()
                        eval_metrics = self.evaluate_holdout()
                        self.wall_s += time.perf_counter() - t_ev
                        V5MetricsLogger.print_eval_block(
                            self.rounds, eval_metrics, self.holdout_w16.shape[0], self._format_v5_probe_str(eval_metrics)
                        )
                        if eval_metrics["eval_bce_loss"] < self.best_eval_bce:
                            self.best_eval_bce = eval_metrics["eval_bce_loss"]
                        if eval_metrics["min_phys_energy_m32"] < self.best_min_m32:
                            self.best_min_m32 = eval_metrics["min_phys_energy_m32"]
                            self.save_checkpoint(self.best_ckpt_name)
                        elif not (self.output_dir / self.best_ckpt_name).exists():
                            self.save_checkpoint(self.best_ckpt_name)
                        if eval_metrics["exact_preimage_rate@256"] > self.best_preimage_rate:
                            self.best_preimage_rate = eval_metrics["exact_preimage_rate@256"]
                            self.save_checkpoint("best_preimage.pt")

                    row: Dict[str, Any] = {
                        "step": self.step,
                        "wall_s": round(self.wall_s, 3),
                        "lr": cur_lr,
                        "train_loss": train_metrics["loss"],
                        "train_bce_loss": train_metrics["bce_loss"],
                        "train_bce_final": train_metrics["bce_final"],
                        "train_phys_loss": train_metrics["phys_loss"],
                        "train_logit_reg": train_metrics["logit_reg"],
                        "frontier_rounds": int(train_metrics["frontier_rounds"]),
                        "train_xor_loss": train_metrics["xor_loss"],
                        "train_comm_loss": train_metrics["comm_loss"],
                        "train_align_loss": train_metrics["align_loss"],
                        "train_assoc_loss": train_metrics.get("train_assoc_loss", ""),
                        "peak_vram_gb": round(peak_gb, 4),
                        "step_time_ms": round(dt_ms, 2),
                    }
                    for ek in ("rrr_gap", "rrr_beta", "drift_flag", "drift_slope", "ent_entropy", "ent_energy", "rank1_ratio", "eff_rank"):
                        if ek in train_metrics:
                            row[ek] = train_metrics[ek]
                    row.update(eval_metrics)
                    self.logger.append_row(row)
                    interval_t0 = time.perf_counter()
                    interval_steps = 0

                if (self.step % ckpt_every == 0) or (self.step == target_steps):
                    self.save_checkpoint(self.latest_ckpt_name)
                    if not do_log:
                        now = time.perf_counter()
                        self.wall_s += now - interval_t0
                        interval_t0 = now
                        interval_steps = 0
        finally:
            self.stop_prefetch()


__all__ = ["UnifiedTrainerV5"]
