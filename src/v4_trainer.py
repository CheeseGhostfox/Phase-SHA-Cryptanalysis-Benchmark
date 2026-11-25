"""Phase-SHA V4 strict-FP32 Best-of-M manifold trainer, Linux torch.compile & telemetry."""

from __future__ import annotations

import copy
import csv
import math
import platform
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.boundary_clamp import ClampedBoundary
from src.flow_trainer import prepare_ground_truth_nodes
from src.sha256_ref import pack_int_messages_to_w16, sample_random_w16, sha256_digest
from src.v3_trainer import _compute_per_round_acc
from src.v4_backbone import ManifoldBitGridBackbone
from src.v4_boundary import (
    ManifoldBoundary,
    PreparedManifoldBoundary,
    apply_clamp_prepared,
    assemble_full_prepared,
    build_manifold_boundary,
)
from src.v4_manifolds import (
    BaseManifold,
    ManifoldPhysicsEngine,
    _spin_add32_flat,
    apply_manifold_closed_loop_schedule,
    enforce_fp32_precision,
    normalize_manifold,
)
from src.v4_solver import UnrolledSolverV4, anneal_schedule


def build_v4_init(
    boundary: ManifoldBoundary,
    manifold: BaseManifold,
    num_particles: int = 1,
    zero_prob: float = 0.85,
    spin_std: float = 0.35,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Construct zero-leakage manifold initial states for B digests x M exploration particles.

    When num_particles >= 2, particle 0 starts at neutral equator e_eq and particles 1..M-1
    sample diverse points on S^{D-1} to drive parallel Best-of-M exploration.
    """
    bm = boundary.batch_size
    r = boundary.rounds
    device = boundary.device
    m = max(1, int(num_particles))
    if bm % m != 0:
        raise ValueError(f"Expected boundary batch size ({bm}) divisible by num_particles ({m})")
    b = bm // m

    u_eq = manifold.make_neutral_equator((b, m, r, 3, 32), device=device)
    u_rand = manifold.sample_random_manifold(
        (b, m, r, 3, 32), device=device, generator=generator, spin_std=spin_std
    )
    if m >= 2:
        p_idx = torch.arange(m, device=device).view(1, m, 1, 1, 1, 1)
        u_init = torch.where(p_idx == 0, u_eq, u_rand)
    else:
        use_eq = (
            torch.rand((b, 1, 1, 1, 1, 1), device=device, generator=generator) < float(zero_prob)
        )
        u_init = torch.where(use_eq, u_eq, u_rand)

    return boundary.apply_clamp_manifold(u_init.reshape(bm, r, 3, 32, manifold.dim), manifold)


@torch.amp.autocast("cuda", enabled=False)
def _bce_wta_loss_head(
    logits_spin: torch.Tensor,
    x_star: torch.Tensor,
    free_mask: torch.Tensor,
    free_count: torch.Tensor,
    wall_free: torch.Tensor,
    wall_den: torch.Tensor,
    wall_bce_mul: torch.Tensor,
    b_offsets: torch.Tensor,
    total_e: torch.Tensor,
    m1_energy: torch.Tensor,
    m32_energy: torch.Tensor,
    frac: torch.Tensor,
    num_particles: int,
    wta_winner_weight: float,
    alpha_phys: float,
    bce_weight: float,
    anti_collapse_weight: float,
    logit_reg_weight: float,
    wavefront_weight: float,
    lam_norm: float,
) -> Tuple[torch.Tensor, ...]:
    """Compilable pure-tensor BCE + wavefront + anti-collapse + WTA aggregation head."""
    bm, rounds, _, _ = x_star.shape
    m = max(1, int(num_particles))
    b = bm // m

    y_target = 0.5 * (1.0 + x_star.to(torch.float32))
    bce_map = F.binary_cross_entropy_with_logits(
        2.0 * logits_spin.to(torch.float32), y_target, reduction="none"
    )
    bce_unweighted = (bce_map * free_mask).sum() / free_count

    with torch.no_grad():
        pred_pos = logits_spin >= 0
        gt_pos = x_star > 0
        correct = (pred_pos == gt_pos).to(torch.float32) * free_mask
        round_cnt = free_mask.sum(dim=(0, 2, 3))
        round_acc = correct.sum(dim=(0, 2, 3)) / round_cnt.clamp_min(1.0)
        is_frontier = (round_cnt > 0.0) & (round_acc > 0.55) & (round_acc < 0.99)
        frontier_cnt_t = is_frontier.sum().to(torch.float32)
        w_round = torch.where(
            is_frontier,
            torch.full_like(round_acc, float(wavefront_weight)),
            torch.ones_like(round_acc),
        )
        w_round = w_round * wall_bce_mul

    weight_map = free_mask * w_round.view(1, rounds, 1, 1)
    loss_bce_weighted = (bce_map * weight_map).sum() / free_count

    anti_collapse = ((1.0 - torch.tanh(logits_spin).square()) * wall_free).sum() / wall_den
    logit_reg = (logits_spin.square() * free_mask).sum() / free_count

    if m > 1:
        e_bm = total_e.view(b, m)
        win_idx = torch.argmin(e_bm.detach(), dim=1, keepdim=True)
        e_min = torch.gather(e_bm, dim=1, index=win_idx).squeeze(1)
        e_mean = e_bm.mean(dim=1)
        wta_e = float(wta_winner_weight) * e_min + (1.0 - float(wta_winner_weight)) * e_mean
        loss_phys = (wta_e / float(lam_norm)).mean()
        win_flat = b_offsets + win_idx.squeeze(1)
        best_m1 = m1_energy[win_flat].mean()
        best_m32 = m32_energy[win_flat].mean()
    else:
        loss_phys = (total_e / float(lam_norm)).mean()
        best_m1 = m1_energy.mean()
        best_m32 = m32_energy.mean()

    total = (
        float(alpha_phys) * loss_phys
        + float(bce_weight) * loss_bce_weighted
        + float(anti_collapse_weight) * (0.2 + 0.8 * frac) * anti_collapse
        + float(logit_reg_weight) * logit_reg
    )
    return (
        total,
        bce_unweighted.detach(),
        loss_phys.detach(),
        anti_collapse.detach(),
        logit_reg.detach(),
        frontier_cnt_t.detach(),
        best_m1.detach(),
        best_m32.detach(),
    )


def _step_loss_from_parts(
    logits_spin: torch.Tensor,
    u_step_end: torch.Tensor,
    x_star: torch.Tensor,
    clamp_mask: torch.Tensor,
    free_mask: torch.Tensor,
    free_count: torch.Tensor,
    wall_free: torch.Tensor,
    wall_den: torch.Tensor,
    wall_bce_mul: torch.Tensor,
    b_offsets: torch.Tensor,
    full_lam: torch.Tensor,
    u_a_left: torch.Tensor,
    u_e_left: torch.Tensor,
    u_a_right: Optional[torch.Tensor],
    u_e_right: Optional[torch.Tensor],
    a_right_mask: torch.Tensor,
    e_right_mask: torch.Tensor,
    a_right_clamped: torch.Tensor,
    e_right_clamped: torch.Tensor,
    k_last_spins: Optional[torch.Tensor],
    digest_target_bits: Optional[torch.Tensor],
    digest_target_mask: Optional[torch.Tensor],
    iv_words: Optional[torch.Tensor],
    physics: ManifoldPhysicsEngine,
    num_particles: int,
    wta_winner_weight: float,
    alpha_phys: float,
    bce_weight: float,
    anti_collapse_weight: float,
    logit_reg_weight: float,
    wavefront_weight: float,
    w_xor: float | torch.Tensor,
    lam_comm: float | torch.Tensor,
    lam_align: float | torch.Tensor,
    frac: float | torch.Tensor,
    has_free_right: bool,
    has_digest_prefix: bool,
    energy_fn: Optional[Callable[..., Tuple[torch.Tensor, ...]]] = None,
    loss_head_fn: Optional[Callable[..., Tuple[torch.Tensor, ...]]] = None,
    add32_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    clone_cudagraph_outputs: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Pure-tensor zero-sync V4 step loss composing modular energy core and BCE/WTA head."""
    rounds = x_star.shape[1]
    manifold = physics.manifold
    u_f = u_step_end.to(torch.float32)
    u_a_pre = torch.cat([u_a_left, u_f[:, :, 0, :, :]], dim=1)
    u_e_pre = torch.cat([u_e_left, u_f[:, :, 1, :, :]], dim=1)
    u_w = u_f[:, :, 2, :, :]
    if not has_free_right and u_a_right is not None and u_e_right is not None:
        u_a = torch.cat([u_a_pre, u_a_right], dim=1)
        u_e = torch.cat([u_e_pre, u_e_right], dim=1)
    else:
        from src.dual_basis import spin_Ch, spin_Maj, spin_Sigma0, spin_Sigma1
        from src.v3_solver import spin_add32

        adder = spin_add32 if add32_fn is None else add32_fn
        a_pre_s = manifold.project_spin(u_a_pre)
        e_pre_s = manifold.project_spin(u_e_pre)
        w_last_s = manifold.project_spin(u_w[:, rounds - 1])
        s1_e = spin_Sigma1(e_pre_s[:, rounds + 2])
        ch_e = spin_Ch(e_pre_s[:, rounds + 2], e_pre_s[:, rounds + 1], e_pre_s[:, rounds])
        t1_s = adder(
            adder(adder(e_pre_s[:, rounds - 1], s1_e), adder(ch_e, k_last_spins)),
            w_last_s,
        )
        s0_a = spin_Sigma0(a_pre_s[:, rounds + 2])
        maj_a = spin_Maj(a_pre_s[:, rounds + 2], a_pre_s[:, rounds + 1], a_pre_s[:, rounds])
        t2_s = adder(s0_a, maj_a)
        e_r_dyn = adder(a_pre_s[:, rounds - 1], t1_s)
        a_r_dyn = adder(t1_s, t2_s)
        a_r_s = a_right_mask * a_right_clamped + (1.0 - a_right_mask) * a_r_dyn
        e_r_s = e_right_mask * e_right_clamped + (1.0 - e_right_mask) * e_r_dyn
        u_a = torch.cat([u_a_pre, manifold.embed_spins_with_equator(a_r_s).unsqueeze(1)], dim=1)
        u_e = torch.cat([u_e_pre, manifold.embed_spins_with_equator(e_r_s).unsqueeze(1)], dim=1)

    core_fn = energy_fn if energy_fn is not None else physics.compute_energy_and_features_core
    lam_b = physics.dual_basis._normalize_lambda(full_lam, u_a.shape[0], u_a.device)
    if clone_cudagraph_outputs and hasattr(torch, "compiler") and hasattr(
        torch.compiler, "cudagraph_mark_step_begin"
    ):
        torch.compiler.cudagraph_mark_step_begin()
    (
        total_e,
        _,
        _,
        _,
        _,
        _,
        m1_energy,
        m32_energy,
        _,
        energy_xor,
        energy_comm,
        energy_align,
        _,
    ) = core_fn(
        u_a,
        u_e,
        u_w,
        free_mask,
        lam_b,
        w_xor,
        lam_comm,
        lam_align,
        digest_target_bits,
        digest_target_mask,
        iv_words,
        4.0,
        has_free_right,
        has_digest_prefix,
    )
    if clone_cudagraph_outputs:
        total_e = total_e.clone()
        m1_energy = m1_energy.clone()
        m32_energy = m32_energy.clone()
        energy_xor = energy_xor.clone()
        energy_comm = energy_comm.clone()
        energy_align = energy_align.clone()

    frac_t = (
        frac
        if isinstance(frac, torch.Tensor)
        else torch.tensor(float(frac), dtype=torch.float32, device=logits_spin.device)
    )
    lam_norm = 32.0 * float(physics.dual_basis.num_eq_rounds)
    head_fn = loss_head_fn if loss_head_fn is not None else _bce_wta_loss_head
    if clone_cudagraph_outputs and hasattr(torch, "compiler") and hasattr(
        torch.compiler, "cudagraph_mark_step_begin"
    ):
        torch.compiler.cudagraph_mark_step_begin()
    (
        total,
        bce_t,
        phys_t,
        anti_t,
        lreg_t,
        front_t,
        best_m1,
        best_m32,
    ) = head_fn(
        logits_spin,
        x_star,
        free_mask,
        free_count,
        wall_free,
        wall_den,
        wall_bce_mul,
        b_offsets,
        total_e,
        m1_energy,
        m32_energy,
        frac_t,
        num_particles,
        wta_winner_weight,
        alpha_phys,
        bce_weight,
        anti_collapse_weight,
        logit_reg_weight,
        wavefront_weight,
        lam_norm,
    )
    if clone_cudagraph_outputs:
        total = total.clone()
    return (
        total,
        bce_t,
        phys_t,
        energy_xor.detach().mean(),
        energy_comm.detach().mean(),
        energy_align.detach().mean(),
        anti_t,
        lreg_t,
        front_t,
        best_m1,
        best_m32,
    )


def compute_v4_step_loss(
    logits_spin: torch.Tensor,
    u_step_end: torch.Tensor,
    x_star: torch.Tensor,
    boundary: ManifoldBoundary,
    physics: ManifoldPhysicsEngine,
    num_particles: int,
    wta_winner_weight: float = 0.85,
    alpha_phys: float = 1.0,
    bce_weight: float = 0.25,
    closed_loop_wall_rounds: Tuple[int, ...] = (0, 13, 16),
    closed_loop_bce_scale: float = 0.0,
    anti_collapse_weight: float = 0.05,
    logit_reg_weight: float = 0.001,
    wavefront_weight: float = 3.0,
    w_xor: float | torch.Tensor = 1.0,
    lam_comm: float | torch.Tensor = 0.0,
    lam_align: float | torch.Tensor = 0.0,
    frac: float | torch.Tensor = 1.0,
    prep: Optional[PreparedManifoldBoundary] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute Best-of-M exploration + manifold physics + calibrated auxiliary loss in FP32."""
    bm, rounds, _, _ = x_star.shape
    m = max(1, int(num_particles))
    b = bm // m
    device = x_star.device
    if prep is None:
        prep = boundary.prepare_for_manifold(physics.manifold)
    free_count = prep.free_mask.sum().clamp_min(1.0)
    full_lam = torch.ones(32, dtype=torch.float32, device=device)

    wall_mask = torch.zeros((1, rounds, 1, 1), dtype=torch.float32, device=device)
    wall_bce_mul = torch.ones((rounds,), dtype=torch.float32, device=device)
    for wr in closed_loop_wall_rounds:
        if 0 <= int(wr) < rounds:
            wall_mask[:, int(wr)] = 1.0
            wall_bce_mul[int(wr)] = float(closed_loop_bce_scale)
    wall_free = prep.free_mask * wall_mask
    wall_den = wall_free.sum().clamp_min(1.0)
    b_offsets = torch.arange(b, device=device) * m

    (
        total,
        bce_t,
        phys_t,
        xor_t,
        comm_t,
        align_t,
        anti_t,
        lreg_t,
        front_t,
        bm1_t,
        bm32_t,
    ) = _step_loss_from_parts(
        logits_spin=logits_spin,
        u_step_end=u_step_end,
        x_star=x_star,
        clamp_mask=prep.clamp_mask,
        free_mask=prep.free_mask,
        free_count=free_count,
        wall_free=wall_free,
        wall_den=wall_den,
        wall_bce_mul=wall_bce_mul,
        b_offsets=b_offsets,
        full_lam=full_lam,
        u_a_left=prep.u_a_left,
        u_e_left=prep.u_e_left,
        u_a_right=prep.u_a_right,
        u_e_right=prep.u_e_right,
        a_right_mask=prep.a_right_mask,
        e_right_mask=prep.e_right_mask,
        a_right_clamped=prep.a_right_clamped,
        e_right_clamped=prep.e_right_clamped,
        k_last_spins=prep.k_last_spins,
        digest_target_bits=prep.digest_target_bits,
        digest_target_mask=prep.digest_target_mask,
        iv_words=prep.iv_words,
        physics=physics,
        num_particles=m,
        wta_winner_weight=wta_winner_weight,
        alpha_phys=alpha_phys,
        bce_weight=bce_weight,
        anti_collapse_weight=anti_collapse_weight,
        logit_reg_weight=logit_reg_weight,
        wavefront_weight=wavefront_weight,
        w_xor=w_xor,
        lam_comm=lam_comm,
        lam_align=lam_align,
        frac=frac,
        has_free_right=prep.has_free_right,
        has_digest_prefix=prep.has_digest_prefix,
    )
    vals = torch.stack(
        [bce_t, phys_t, xor_t, comm_t, align_t, anti_t, lreg_t, front_t, bm1_t, bm32_t]
    ).cpu().tolist()
    diag = {
        "bce_loss": vals[0],
        "phys_loss": vals[1],
        "xor_loss": vals[2],
        "comm_loss": vals[3],
        "align_loss": vals[4],
        "anti_collapse": vals[5],
        "logit_reg": vals[6],
        "frontier_rounds": vals[7],
        "m1_energy": vals[8],
        "m32_energy": vals[9],
    }
    return total, diag


def _advance_and_loss_step(
    u_cur: torch.Tensor,
    logits_s: torch.Tensor,
    delta_u_ambient: torch.Tensor,
    grad_riem: torch.Tensor,
    x_star: torch.Tensor,
    u_clamped_masked: torch.Tensor,
    free_mask_exp: torch.Tensor,
    clamp_mask: torch.Tensor,
    free_mask: torch.Tensor,
    free_count: torch.Tensor,
    wall_free: torch.Tensor,
    wall_den: torch.Tensor,
    wall_bce_mul: torch.Tensor,
    b_offsets: torch.Tensor,
    full_lam: torch.Tensor,
    u_a_left: torch.Tensor,
    u_e_left: torch.Tensor,
    u_a_right: Optional[torch.Tensor],
    u_e_right: Optional[torch.Tensor],
    a_right_mask: torch.Tensor,
    e_right_mask: torch.Tensor,
    a_right_clamped: torch.Tensor,
    e_right_clamped: torch.Tensor,
    k_last_spins: Optional[torch.Tensor],
    digest_target_bits: Optional[torch.Tensor],
    digest_target_mask: Optional[torch.Tensor],
    iv_words: Optional[torch.Tensor],
    damp_t: torch.Tensor,
    eta_s: torch.Tensor,
    beta_s: torch.Tensor,
    w_xor: float | torch.Tensor,
    lam_comm_s: torch.Tensor,
    lam_align_s: torch.Tensor,
    frac_s: torch.Tensor,
    physics: ManifoldPhysicsEngine,
    num_particles: int,
    wta_winner_weight: float,
    alpha_phys: float,
    bce_weight: float,
    anti_collapse_weight: float,
    logit_reg_weight: float,
    wavefront_weight: float,
    closed_loop_schedule: bool,
    has_free_right: bool,
    has_digest_prefix: bool,
    retract_clamp_fn: Optional[Callable[..., torch.Tensor]] = None,
    add32_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    energy_fn: Optional[Callable[..., Tuple[torch.Tensor, ...]]] = None,
    loss_head_fn: Optional[Callable[..., Tuple[torch.Tensor, ...]]] = None,
    clone_cudagraph_outputs: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Compatibility helper delegating to modular manifold retraction + schedule + loss head."""
    rounds = u_cur.shape[1]
    rc_fn = physics.retract_and_clamp if retract_clamp_fn is None else retract_clamp_fn
    u_next = rc_fn(
        u_cur,
        logits_s,
        delta_u_ambient,
        grad_riem,
        u_clamped_masked,
        free_mask_exp,
        damp_t,
        eta_s,
        beta_s,
    )
    if clone_cudagraph_outputs:
        u_next = u_next.clone()
    if closed_loop_schedule and rounds > 16:
        u_sched = apply_manifold_closed_loop_schedule(
            u_next, physics.manifold, rounds, add32_fn=add32_fn
        )
        u_next = normalize_manifold(u_clamped_masked + free_mask_exp * u_sched.to(torch.float32))

    step_parts = _step_loss_from_parts(
        logits_spin=logits_s,
        u_step_end=u_next,
        x_star=x_star,
        clamp_mask=clamp_mask,
        free_mask=free_mask,
        free_count=free_count,
        wall_free=wall_free,
        wall_den=wall_den,
        wall_bce_mul=wall_bce_mul,
        b_offsets=b_offsets,
        full_lam=full_lam,
        u_a_left=u_a_left,
        u_e_left=u_e_left,
        u_a_right=u_a_right,
        u_e_right=u_e_right,
        a_right_mask=a_right_mask,
        e_right_mask=e_right_mask,
        a_right_clamped=a_right_clamped,
        e_right_clamped=e_right_clamped,
        k_last_spins=k_last_spins,
        digest_target_bits=digest_target_bits,
        digest_target_mask=digest_target_mask,
        iv_words=iv_words,
        physics=physics,
        num_particles=num_particles,
        wta_winner_weight=wta_winner_weight,
        alpha_phys=alpha_phys,
        bce_weight=bce_weight,
        anti_collapse_weight=anti_collapse_weight,
        logit_reg_weight=logit_reg_weight,
        wavefront_weight=wavefront_weight,
        w_xor=w_xor,
        lam_comm=lam_comm_s,
        lam_align=lam_align_s,
        frac=frac_s,
        has_free_right=has_free_right,
        has_digest_prefix=has_digest_prefix,
        energy_fn=energy_fn,
        loss_head_fn=loss_head_fn,
        add32_fn=add32_fn,
        clone_cudagraph_outputs=clone_cudagraph_outputs,
    )
    return (u_next, *step_parts)


@torch.amp.autocast("cuda", enabled=False)
def run_v4_chunked_bptt_step(
    train_model: nn.Module,
    raw_model: ManifoldBitGridBackbone,
    physics: ManifoldPhysicsEngine,
    x_star: torch.Tensor,
    boundary: ManifoldBoundary,
    u_init: torch.Tensor,
    num_particles: int = 2,
    k_unroll: int = 8,
    bptt_chunk_len: int = 4,
    alpha_phys: float = 1.0,
    wta_winner_weight: float = 0.85,
    bce_weight: float = 0.25,
    closed_loop_wall_rounds: Tuple[int, ...] = (0, 13, 16),
    closed_loop_bce_scale: float = 0.0,
    anti_collapse_weight: float = 0.05,
    logit_reg_weight: float = 0.001,
    wavefront_weight: float = 3.0,
    grad_scale: float = 2.0,
    damp: float = 0.28,
    eta_max: float = 0.025,
    beta_max: float = 0.25,
    sync_h_damp: bool = True,
    closed_loop_schedule: bool = True,
    w_xor: float = 1.0,
    lam_comm_start: float = 0.02,
    lam_comm_end: float = 0.60,
    lam_align_start: float = 0.02,
    lam_align_end: float = 0.60,
    anneal_mode: str = "cosine",
    do_backward: bool = True,
    backward_group_chunks: int = 1,
    sync_metrics: bool = True,
    radar_energy_fn: Optional[Callable[..., Tuple[torch.Tensor, ...]]] = None,
    retract_clamp_fn: Optional[Callable[..., torch.Tensor]] = None,
    add32_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    loss_head_fn: Optional[Callable[..., Tuple[torch.Tensor, ...]]] = None,
    clone_cudagraph_outputs: bool = False,
) -> Dict[str, float]:
    """Run K_unroll norm-preserving Chunked-BPTT step in strict FP32 with modular compiled kernels."""
    bm, rounds, _, _, _ = u_init.shape
    device = u_init.device
    manifold = physics.manifold
    prep = boundary.prepare_for_manifold(manifold)
    m = max(1, int(num_particles))
    b = bm // m

    free_count = prep.free_mask.sum().clamp_min(1.0)
    full_lam = torch.ones(32, dtype=torch.float32, device=device)
    wall_mask = torch.zeros((1, rounds, 1, 1), dtype=torch.float32, device=device)
    wall_bce_mul = torch.ones((rounds,), dtype=torch.float32, device=device)
    for wr in closed_loop_wall_rounds:
        if 0 <= int(wr) < rounds:
            wall_mask[:, int(wr)] = 1.0
            wall_bce_mul[int(wr)] = float(closed_loop_bce_scale)
    wall_free = prep.free_mask * wall_mask
    wall_den = wall_free.sum().clamp_min(1.0)
    b_offsets = torch.arange(b, device=device) * m

    chunk_len = max(1, int(bptt_chunk_len))
    group_chunks = max(1, int(backward_group_chunks))
    raw_weights = [float(s + 1) for s in range(k_unroll)]
    w_sum = sum(raw_weights)
    step_weights = [w / w_sum for w in raw_weights]
    denom = float(max(1, k_unroll - 1))

    frac_vals = [float(s) / denom for s in range(k_unroll)]
    frac_vec = torch.tensor(frac_vals, dtype=torch.float32, device=device)
    step_frac_mat = frac_vec.unsqueeze(1).expand(k_unroll, bm).contiguous()
    lam_comm_vec = torch.tensor(
        [anneal_schedule(f, lam_comm_start, lam_comm_end, anneal_mode) for f in frac_vals],
        dtype=torch.float32,
        device=device,
    )
    lam_align_vec = torch.tensor(
        [anneal_schedule(f, lam_align_start, lam_align_end, anneal_mode) for f in frac_vals],
        dtype=torch.float32,
        device=device,
    )
    eta_vec = torch.tensor(
        [float(eta_max) * (0.2 + 0.8 * f) for f in frac_vals],
        dtype=torch.float32,
        device=device,
    )
    beta_vec = torch.tensor(
        [float(beta_max) * (f * f) for f in frac_vals],
        dtype=torch.float32,
        device=device,
    )
    damp_t = torch.tensor(float(damp), dtype=torch.float32, device=device)
    w_xor_t = torch.tensor(float(w_xor), dtype=torch.float32, device=device)

    u_cur = apply_clamp_prepared(u_init.detach(), prep)
    if closed_loop_schedule and rounds > 16:
        u_cur = apply_clamp_prepared(
            apply_manifold_closed_loop_schedule(
                u_cur, manifold, rounds, add32_fn=add32_fn
            ),
            prep,
        )

    h_prev: Optional[torch.Tensor] = None
    grouped_loss: Optional[torch.Tensor] = None
    chunk_idx = 0

    acc_loss = torch.zeros((), dtype=torch.float32, device=device)
    acc_bce = torch.zeros((), dtype=torch.float32, device=device)
    acc_phys = torch.zeros((), dtype=torch.float32, device=device)
    acc_xor = torch.zeros((), dtype=torch.float32, device=device)
    acc_comm = torch.zeros((), dtype=torch.float32, device=device)
    acc_align = torch.zeros((), dtype=torch.float32, device=device)
    acc_logit_reg = torch.zeros((), dtype=torch.float32, device=device)
    last_bce_t = acc_loss
    last_front_t = acc_loss
    last_m1_t = acc_loss
    last_m32_t = acc_loss

    rc_fn = physics.retract_and_clamp if retract_clamp_fn is None else retract_clamp_fn

    for s in range(k_unroll):
        step_tensor = step_frac_mat[s]
        lam_comm_s = lam_comm_vec[s]
        lam_align_s = lam_align_vec[s]
        eta_s = eta_vec[s]
        beta_s = beta_vec[s]
        frac_s = frac_vec[s]

        grad_radar, res_feat, _, grad_riem = physics.compute_radar_prepared(
            u_nodes=u_cur,
            prep=prep,
            lam=full_lam,
            grad_scale=grad_scale,
            w_xor=w_xor_t,
            lam_comm=lam_comm_s,
            lam_align=lam_align_s,
            energy_fn=radar_energy_fn,
            return_raw_grad=True,
            clone_cudagraph_outputs=clone_cudagraph_outputs,
        )

        if clone_cudagraph_outputs and hasattr(torch, "compiler") and hasattr(
            torch.compiler, "cudagraph_mark_step_begin"
        ):
            torch.compiler.cudagraph_mark_step_begin()
        logits_s, delta_u_ambient, h_tilde = train_model(
            u_nodes=u_cur,
            grad_radar=grad_radar,
            clamp_mask=prep.clamp_mask,
            res_features=res_feat,
            k_spins=physics.K_spins,
            step_frac=step_tensor,
            lam=full_lam,
            h_prev=h_prev,
        )
        if clone_cudagraph_outputs:
            logits_s = logits_s.clone()
            delta_u_ambient = delta_u_ambient.clone()
            h_tilde = h_tilde.clone()

        if sync_h_damp and h_prev is not None:
            h_next = (1.0 - damp_t) * h_prev + damp_t * h_tilde
        else:
            h_next = h_tilde

        if clone_cudagraph_outputs and hasattr(torch, "compiler") and hasattr(
            torch.compiler, "cudagraph_mark_step_begin"
        ):
            torch.compiler.cudagraph_mark_step_begin()
        u_next = rc_fn(
            u_cur,
            logits_s,
            delta_u_ambient,
            grad_riem,
            prep.u_clamped_masked,
            prep.free_mask_exp,
            damp_t,
            eta_s,
            beta_s,
        )
        if clone_cudagraph_outputs:
            u_next = u_next.clone()

        if closed_loop_schedule and rounds > 16:
            u_sched = apply_manifold_closed_loop_schedule(
                u_next, manifold, rounds, add32_fn=add32_fn
            )
            u_next = normalize_manifold(
                prep.u_clamped_masked + prep.free_mask_exp * u_sched.to(torch.float32)
            )

        (
            step_loss,
            bce_t,
            phys_t,
            xor_t,
            comm_t,
            align_t,
            _,
            lreg_t,
            front_t,
            bm1_t,
            bm32_t,
        ) = _step_loss_from_parts(
            logits_spin=logits_s,
            u_step_end=u_next,
            x_star=x_star,
            clamp_mask=prep.clamp_mask,
            free_mask=prep.free_mask,
            free_count=free_count,
            wall_free=wall_free,
            wall_den=wall_den,
            wall_bce_mul=wall_bce_mul,
            b_offsets=b_offsets,
            full_lam=full_lam,
            u_a_left=prep.u_a_left,
            u_e_left=prep.u_e_left,
            u_a_right=prep.u_a_right,
            u_e_right=prep.u_e_right,
            a_right_mask=prep.a_right_mask,
            e_right_mask=prep.e_right_mask,
            a_right_clamped=prep.a_right_clamped,
            e_right_clamped=prep.e_right_clamped,
            k_last_spins=prep.k_last_spins,
            digest_target_bits=prep.digest_target_bits,
            digest_target_mask=prep.digest_target_mask,
            iv_words=prep.iv_words,
            physics=physics,
            num_particles=num_particles,
            wta_winner_weight=wta_winner_weight,
            alpha_phys=alpha_phys,
            bce_weight=bce_weight,
            anti_collapse_weight=anti_collapse_weight,
            logit_reg_weight=logit_reg_weight,
            wavefront_weight=wavefront_weight,
            w_xor=w_xor_t,
            lam_comm=lam_comm_s,
            lam_align=lam_align_s,
            frac=frac_s,
            has_free_right=prep.has_free_right,
            has_digest_prefix=prep.has_digest_prefix,
            energy_fn=radar_energy_fn,
            loss_head_fn=loss_head_fn,
            add32_fn=add32_fn,
            clone_cudagraph_outputs=clone_cudagraph_outputs,
        )

        w_s = step_weights[s]
        weighted_term = w_s * step_loss
        grouped_loss = weighted_term if grouped_loss is None else (grouped_loss + weighted_term)

        if sync_metrics:
            acc_loss = acc_loss + step_loss.detach() * w_s
            acc_bce = acc_bce + bce_t * w_s
            acc_phys = acc_phys + phys_t * w_s
            acc_xor = acc_xor + xor_t * w_s
            acc_comm = acc_comm + comm_t * w_s
            acc_align = acc_align + align_t * w_s
            acc_logit_reg = acc_logit_reg + lreg_t * w_s
            last_bce_t = bce_t
            last_front_t = front_t
            last_m1_t = bm1_t
            last_m32_t = bm32_t

        is_chunk_end = ((s + 1) % chunk_len == 0) or (s == k_unroll - 1)
        is_group_end = (s == k_unroll - 1) or (
            is_chunk_end and ((chunk_idx + 1) % group_chunks == 0)
        )
        if do_backward and is_group_end and grouped_loss is not None:
            grouped_loss.backward()
            grouped_loss = None

        if is_chunk_end:
            chunk_idx += 1
            u_cur = u_next.detach()
            h_prev = h_next.detach()
        else:
            u_cur = u_next
            h_prev = h_next

    if not sync_metrics:
        return {}

    vals = torch.stack(
        [
            acc_loss,
            acc_bce,
            last_bce_t,
            acc_phys,
            acc_xor,
            acc_comm,
            acc_align,
            acc_logit_reg,
            last_front_t,
            last_m1_t,
            last_m32_t,
        ]
    ).cpu().tolist()
    return {
        "loss": vals[0],
        "bce_loss": vals[1],
        "bce_final": vals[2],
        "phys_loss": vals[3],
        "xor_loss": vals[4],
        "comm_loss": vals[5],
        "align_loss": vals[6],
        "logit_reg": vals[7],
        "frontier_rounds": vals[8],
        "m1_energy": vals[9],
        "m32_energy": vals[10],
    }


class AsyncV4BatchPrefetcher:
    """Background CPU producer overlapping SHA-256 ground-truth trace generation & pinning with GPU training."""

    def __init__(
        self,
        batch_size: int,
        rounds: int,
        domain_mode: str,
        num_bits: int,
        msg_bytes_len: int,
        device: torch.device,
        seed: int,
        train_w0_pool_cpu: Optional[torch.Tensor] = None,
        holdout_w0_set: Optional[set[int]] = None,
        queue_size: int = 3,
    ) -> None:
        self.batch_size = int(batch_size)
        self.rounds = int(rounds)
        self.domain_mode = str(domain_mode)
        self.num_bits = int(num_bits)
        self.msg_bytes_len = int(msg_bytes_len)
        self.device = device
        self.train_w0_pool_cpu = train_w0_pool_cpu
        self.holdout_w0_set = holdout_w0_set
        self.pin = device.type == "cuda"
        self.rng = torch.Generator(device="cpu").manual_seed(int(seed) + 7919)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    def start(self) -> None:
        if self.is_running():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _sample_cpu_w16(self) -> torch.Tensor:
        if self.train_w0_pool_cpu is not None:
            idx = torch.randint(
                0,
                self.train_w0_pool_cpu.shape[0],
                (self.batch_size,),
                generator=self.rng,
            )
            return pack_int_messages_to_w16(self.train_w0_pool_cpu[idx])

        w16 = sample_random_w16(
            batch_size=self.batch_size,
            mode=self.domain_mode,
            num_bits=self.num_bits,
            msg_bytes_len=self.msg_bytes_len,
            device="cpu",
            generator=self.rng,
        )
        if self.domain_mode == "int_range" and self.holdout_w0_set:
            high = 1 << self.num_bits
            col = self.holdout_w0_set
            w0_list = w16[:, 0].tolist()
            bad = [i for i, val in enumerate(w0_list) if val in col]
            while bad:
                repl = torch.randint(0, high, (len(bad),), dtype=torch.int64, generator=self.rng).tolist()
                next_bad = []
                for pos, cand in zip(bad, repl):
                    if cand in col:
                        next_bad.append(pos)
                    else:
                        w16[pos, 0] = cand
                bad = next_bad
        return w16

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                w16 = self._sample_cpu_w16()
                x_star_base, cb_base = prepare_ground_truth_nodes(
                    w16,
                    rounds=self.rounds,
                    domain_mode=self.domain_mode,
                    num_bits=self.num_bits,
                    msg_bytes_len=self.msg_bytes_len,
                )
                if self.pin:
                    item = (
                        x_star_base.pin_memory(),
                        cb_base.A_full_clamped.pin_memory(),
                        cb_base.A_full_mask.pin_memory(),
                        cb_base.E_full_clamped.pin_memory(),
                        cb_base.E_full_mask.pin_memory(),
                        cb_base.W_full_clamped.pin_memory(),
                        cb_base.W_full_mask.pin_memory(),
                        cb_base.A_words_clamped.pin_memory(),
                        cb_base.E_words_clamped.pin_memory(),
                        cb_base.W16_clamped_words.pin_memory(),
                        cb_base.W16_mask_bits.pin_memory(),
                    )
                else:
                    item = (
                        x_star_base,
                        cb_base.A_full_clamped,
                        cb_base.A_full_mask,
                        cb_base.E_full_clamped,
                        cb_base.E_full_mask,
                        cb_base.W_full_clamped,
                        cb_base.W_full_mask,
                        cb_base.A_words_clamped,
                        cb_base.E_words_clamped,
                        cb_base.W16_clamped_words,
                        cb_base.W16_mask_bits,
                    )
                while not self._stop.is_set():
                    try:
                        self._queue.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue
            except Exception as exc:
                print(f"[AsyncV4BatchPrefetcher] error: {exc}", flush=True)
                self._stop.wait(0.2)

    def next_on_device(self) -> Tuple[torch.Tensor, ClampedBoundary]:
        (
            xs,
            afc,
            afm,
            efc,
            efm,
            wfc,
            wfm,
            awc,
            ewc,
            w16c,
            w16m,
        ) = self._queue.get()
        nb = self.pin
        dev = self.device
        cb = ClampedBoundary(
            rounds=self.rounds,
            A_full_clamped=afc.to(dev, non_blocking=nb),
            A_full_mask=afm.to(dev, non_blocking=nb),
            E_full_clamped=efc.to(dev, non_blocking=nb),
            E_full_mask=efm.to(dev, non_blocking=nb),
            W_full_clamped=wfc.to(dev, non_blocking=nb),
            W_full_mask=wfm.to(dev, non_blocking=nb),
            A_words_clamped=awc.to(dev, non_blocking=nb),
            E_words_clamped=ewc.to(dev, non_blocking=nb),
            W16_clamped_words=w16c.to(dev, non_blocking=nb),
            W16_mask_bits=w16m.to(dev, non_blocking=nb),
        )
        return xs.to(dev, non_blocking=nb), cb


class UnrolledTrainerV4:
    """V4 trainer with Linux torch.compile, Best-of-M exploration, and V3-identical telemetry."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path | str,
        device: torch.device | str = "cuda",
    ) -> None:
        enforce_fp32_precision()
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

        self.variant = str(config.get("variant", "v4b")).lower()
        self.rounds = int(config.get("rounds", 18))
        self.domain_mode = str(config.get("domain_mode", "int_range"))
        self.num_bits = int(config.get("num_bits", 32))
        self.msg_bytes_len = int(config.get("msg_bytes_len", 4))

        self.physics = ManifoldPhysicsEngine(
            variant=self.variant, rounds=self.rounds
        ).to(self.device)
        self.manifold: BaseManifold = self.physics.manifold
        self.model = ManifoldBitGridBackbone(
            variant=self.variant,
            rounds=self.rounds,
            d_bit=int(config.get("d_bit", 48)),
            n_heads=int(config.get("n_heads", 3)),
            n_blocks=int(config.get("n_blocks", 2)),
            d_ff_mult=int(config.get("d_ff_mult", 2)),
            share_blocks=bool(config.get("share_blocks", False)),
            grad_checkpointing=bool(config.get("grad_checkpointing", False)),
            use_carry_gru=bool(config.get("use_carry_gru", True)),
        ).to(self.device).float()

        self.ema_decay = float(config.get("ema_decay", 0.995))
        self.ema_model = copy.deepcopy(self.model).to(self.device).float()
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self._ema_params = list(self.ema_model.parameters())
        self._src_params = list(self.model.parameters())
        self._ema_buffers = list(self.ema_model.buffers())
        self._src_buffers = list(self.model.buffers())

        self.train_model: nn.Module = self.model
        self.compiled_active: bool = False
        self.compile_mode: str = str(config.get("torch_compile_mode", "default"))
        self.compile_physics: bool = bool(config.get("compile_physics", True))
        self.allow_rnn_compile: bool = bool(config.get("allow_rnn", False))
        self.eval_compile: bool = bool(config.get("eval_compile", True))
        self._compiled_energy_core: Optional[Any] = None
        self._compiled_spin_add32: Optional[Any] = None
        self._compiled_retract_clamp: Optional[Any] = None
        self._compiled_loss_head: Optional[Any] = None
        self._compiled_eval_model: Optional[nn.Module] = None
        self._clone_cudagraph_outputs: bool = False
        self._configure_compile()

        self.total_steps = int(config.get("steps", 10000))
        self.lr_total_steps = int(config.get("lr_total_steps", 20000))
        self.warmup_steps = int(config.get("warmup_steps", 300))
        self.lr = float(config.get("lr", 4.0e-4))
        self.min_lr_ratio = float(config.get("min_lr_ratio", 0.1))
        self.weight_decay = float(config.get("weight_decay", 0.02))
        self.grad_clip = float(config.get("grad_clip", 1.0))

        self.batch_size = int(config.get("batch_size", 16))
        self.num_particles = int(config.get("num_particles", 2))
        self.k_unroll = int(config.get("K_unroll", 8))
        self.bptt_chunk_len = int(config.get("bptt_chunk_len", 4))
        self.backward_group_chunks = max(1, int(config.get("backward_group_chunks", 1)))
        self.damp = float(config.get("damp", 0.28))
        self.damp_jitter_min = config.get("damp_jitter_min", 0.20)
        self.damp_jitter_max = config.get("damp_jitter_max", 0.35)
        self.sync_h_damp = bool(config.get("sync_h_damp", True))
        self.closed_loop_schedule = bool(config.get("closed_loop_schedule", True))

        # Exploration loss & annealing hyperparameters
        self.alpha_phys = float(config.get("alpha_phys", 1.0))
        self.wta_winner_weight = float(config.get("wta_winner_weight", 0.85))
        self.bce_weight = float(config.get("bce_weight", 0.25))
        self.closed_loop_wall_rounds = tuple(
            int(r) for r in config.get("closed_loop_wall_rounds", [0, 13, 16])
        )
        self.closed_loop_bce_scale = float(config.get("closed_loop_bce_scale", 0.0))
        self.anti_collapse_weight = float(config.get("anti_collapse_weight", 0.05))
        self.logit_reg_weight = float(config.get("logit_reg_weight", 0.001))
        self.wavefront_weight = float(config.get("wavefront_weight", 3.0))
        self.zero_init_prob = float(config.get("zero_init_prob", 0.85))
        self.grad_scale = float(config.get("grad_scale", 2.0))

        self.w_xor = float(config.get("w_xor", 1.0))
        self.lam_comm_start = float(config.get("lam_comm_start", 0.02))
        self.lam_comm_end = float(config.get("lam_comm_end", 0.60))
        self.lam_align_start = float(config.get("lam_align_start", 0.02))
        self.lam_align_end = float(config.get("lam_align_end", 0.60))
        self.anneal_mode = str(config.get("anneal_mode", "cosine"))

        self.best_ckpt_name = str(config.get("best_ckpt_name", "best.pt"))
        self.latest_ckpt_name = str(config.get("latest_ckpt_name", "latest.pt"))
        self.infer_damp = float(config.get("infer_damp", 0.25))
        self.sde_noise_min = float(config.get("sde_noise_min", 0.10))
        self.sde_noise_max = float(config.get("sde_noise_max", 0.45))
        self.eta_max = float(config.get("anneal_eta_max", 0.025))
        self.beta_max = float(config.get("anneal_beta_max", 0.25))

        opt_kwargs: Dict[str, Any] = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "betas": (0.9, 0.999),
        }
        self._use_fused_adamw: bool = False
        if self.device.type == "cuda":
            try:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), fused=True, **opt_kwargs
                )
                self._use_fused_adamw = True
            except (TypeError, RuntimeError):
                self.optimizer = torch.optim.AdamW(self.model.parameters(), **opt_kwargs)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), **opt_kwargs)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )

        seed_val = int(config.get("seed", 404))
        self.train_rng = torch.Generator(device=self.device)
        self.train_rng.manual_seed(seed_val)
        self.damp_rng = torch.Generator(device="cpu").manual_seed(seed_val + 1337)

        holdout_size = int(config.get("holdout_size", 2048))
        holdout_rng = torch.Generator(device="cpu")
        holdout_rng.manual_seed(int(config.get("holdout_seed", 20261122)))
        self.train_w0_pool: Optional[torch.Tensor] = None
        self.train_w0_pool_cpu: Optional[torch.Tensor] = None
        self.holdout_w0_gpu: Optional[torch.Tensor] = None
        self.holdout_w0_set: Optional[set[int]] = None

        if self.domain_mode == "int_range" and self.num_bits <= 20:
            total_space = 1 << self.num_bits
            perm = torch.randperm(total_space, generator=holdout_rng, dtype=torch.int64)
            holdout_w0 = perm[:holdout_size]
            train_w0 = perm[holdout_size:]
            self.holdout_w16 = pack_int_messages_to_w16(holdout_w0)
            self.train_w0_pool_cpu = train_w0.contiguous()
            self.train_w0_pool = train_w0.to(self.device)
            self.holdout_w0_gpu = holdout_w0.to(self.device)
            self.holdout_w0_set = set(holdout_w0.tolist())
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
            self.holdout_w0_set = set(unique_vals)
        else:
            self.holdout_w16 = sample_random_w16(
                batch_size=holdout_size,
                mode=self.domain_mode,
                num_bits=self.num_bits,
                msg_bytes_len=self.msg_bytes_len,
                device="cpu",
                generator=holdout_rng,
            )
        self._prefetcher: Optional[AsyncV4BatchPrefetcher] = None

        self.solver = UnrolledSolverV4(
            model=self.ema_model,
            physics=self.physics,
            num_candidates=int(config.get("anneal_K", 256)),
            num_steps=int(config.get("K_infer", 32)),
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
            smc_backtrack_ratio=float(config.get("smc_backtrack_ratio", 0.25)),
            sync_h_damp=self.sync_h_damp,
            closed_loop_schedule=self.closed_loop_schedule,
            w_xor=self.w_xor,
            lam_comm_start=self.lam_comm_start,
            lam_comm_end=self.lam_comm_end,
            lam_align_start=self.lam_align_start,
            lam_align_end=self.lam_align_end,
            anneal_mode=self.anneal_mode,
            eval_model=self._compiled_eval_model,
            compiled_energy_fn=self._compiled_energy_core,
            compiled_add32_fn=self._compiled_spin_add32,
        )

        self.step = 0
        self.best_eval_bce = float("inf")
        self.best_min_m32 = float("inf")
        self.best_preimage_rate = 0.0
        self.csv_path = self.output_dir / "metrics.csv"
        self.csv_fields = self._build_csv_fields(self.rounds)

    def _configure_compile(self) -> None:
        pref = str(self.config.get("use_torch_compile", "auto")).lower()
        want = (
            (platform.system() == "Linux" and self.device.type == "cuda")
            if pref == "auto"
            else pref in ("true", "1", "yes", "on")
        )
        self._compiled_energy_core = None
        self._compiled_spin_add32 = None
        self._compiled_retract_clamp = None
        self._compiled_loss_head = None
        self._compiled_eval_model = None
        self._clone_cudagraph_outputs = False

        if want and hasattr(torch, "compile"):
            try:
                if self.allow_rnn_compile and hasattr(torch, "_dynamo"):
                    torch._dynamo.config.allow_rnn = True
                mode = str(self.config.get("torch_compile_mode", "default"))
                self.compile_mode = mode
                self.train_model = torch.compile(self.model, mode=mode)
                self._clone_cudagraph_outputs = mode == "reduce-overhead"
                if self.compile_physics:
                    self._compiled_energy_core = torch.compile(
                        self.physics.compute_energy_and_features_core, mode=mode
                    )
                    self._compiled_spin_add32 = torch.compile(
                        _spin_add32_flat, mode="default", dynamic=True
                    )
                    self._compiled_retract_clamp = torch.compile(
                        self.physics.retract_and_clamp, mode=mode
                    )
                    self._compiled_loss_head = torch.compile(_bce_wta_loss_head, mode=mode)
                if self.eval_compile:
                    eval_mode = str(self.config.get("eval_compile_mode", "default"))
                    self._compiled_eval_model = torch.compile(self.ema_model, mode=eval_mode)
                self.compiled_active = True
            except Exception as exc:
                print(
                    f"[Phase-SHA V4] torch.compile fallback to eager ({exc})",
                    flush=True,
                )
                self.train_model = self.model
                self._compiled_energy_core = None
                self._compiled_spin_add32 = None
                self._compiled_retract_clamp = None
                self._compiled_loss_head = None
                self._compiled_eval_model = None
                self._clone_cudagraph_outputs = False
                self.compiled_active = False
        else:
            self.train_model = self.model
            self.compiled_active = False

    def start_prefetch(self) -> None:
        if self._prefetcher is not None and self._prefetcher.is_running():
            return
        self._prefetcher = AsyncV4BatchPrefetcher(
            batch_size=self.batch_size,
            rounds=self.rounds,
            domain_mode=self.domain_mode,
            num_bits=self.num_bits,
            msg_bytes_len=self.msg_bytes_len,
            device=self.device,
            seed=int(self.config.get("seed", 404)),
            train_w0_pool_cpu=self.train_w0_pool_cpu,
            holdout_w0_set=self.holdout_w0_set,
            queue_size=int(self.config.get("prefetch_queue_size", 3)),
        )
        self._prefetcher.start()

    def stop_prefetch(self) -> None:
        if self._prefetcher is not None:
            self._prefetcher.stop()
            self._prefetcher = None

    def _sample_train_w16(self, batch_size: int) -> torch.Tensor:
        if self.train_w0_pool is not None:
            idx = torch.randint(
                0,
                self.train_w0_pool.shape[0],
                (batch_size,),
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
        if self._ema_params:
            torch._foreach_mul_(self._ema_params, decay)
            torch._foreach_add_(self._ema_params, self._src_params, alpha=1.0 - decay)
        for b_ema, b_src in zip(self._ema_buffers, self._src_buffers):
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
        fields.extend(
            [
                "train_xor_loss",
                "train_comm_loss",
                "train_align_loss",
                "zero_manifold_align",
                "zero_manifold_comm",
            ]
        )
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
            "variant": self.variant,
            "step": self.step,
            "best_eval_bce": self.best_eval_bce,
            "best_min_m32": self.best_min_m32,
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
        self.model.float()
        if "ema_model_state" in ckpt:
            self.ema_model.load_state_dict(ckpt["ema_model_state"])
        else:
            self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_model.float()
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            if self._use_fused_adamw:
                for pg in self.optimizer.param_groups:
                    pg["fused"] = True
        if "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.step = int(ckpt.get("step", 0))
        self.best_eval_bce = float(ckpt.get("best_eval_bce", float("inf")))
        self.best_min_m32 = float(ckpt.get("best_min_m32", float("inf")))
        self.best_preimage_rate = float(ckpt.get("best_preimage_rate", 0.0))

    def load_weights_only(self, ckpt_path: str | Path) -> None:
        """Warm-start either from a V3 checkpoint or another V4 checkpoint with same variant."""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        src_state = ckpt.get("ema_model_state", ckpt.get("model_state"))
        if set(src_state.keys()) == set(self.model.state_dict().keys()):
            self.model.load_state_dict(src_state, strict=True)
            self.model.float()
            info_str = f"V4->V4 strict (R={ckpt.get('config', {}).get('rounds', '?')}, step={ckpt.get('step', '?')})"
        else:
            info = self.model.load_v3_warmstart(ckpt_path, map_location=self.device)
            info_str = (
                f"V3->V4 lossless (loaded={info['loaded_keys']}, "
                f"fresh_adapters={len(info['missing_adapter_keys'])}, "
                f"src_step={info['source_step']})"
            )
        self.ema_model.load_state_dict(self.model.state_dict(), strict=True)
        self.ema_model.float()
        print(
            f"[Phase-SHA {self.variant.upper()}] Warm-started from {ckpt_path} [{info_str}]",
            flush=True,
        )

    def train_one_step(self, sync_metrics: bool = True) -> Dict[str, float]:
        """Execute one FP32 Best-of-M manifold training step and update EMA."""
        self.model.train()
        self.train_model.train()

        if self._prefetcher is not None and self._prefetcher.is_running():
            x_star_base, cb_base = self._prefetcher.next_on_device()
        else:
            w16 = self._sample_train_w16(self.batch_size)
            x_star_base, cb_base = prepare_ground_truth_nodes(
                w16,
                rounds=self.rounds,
                domain_mode=self.domain_mode,
                num_bits=self.num_bits,
                msg_bytes_len=self.msg_bytes_len,
            )
        m = max(1, self.num_particles)
        if m > 1:
            x_star = x_star_base.repeat_interleave(m, dim=0)
            mb_base = ManifoldBoundary.from_clamped_boundary(cb_base)
            boundary = mb_base.repeat_for_candidates(m)
        else:
            x_star = x_star_base
            boundary = ManifoldBoundary.from_clamped_boundary(cb_base)

        u_init = build_v4_init(
            boundary=boundary,
            manifold=self.manifold,
            num_particles=m,
            zero_prob=self.zero_init_prob,
            generator=self.train_rng,
        )

        step_damp = self.damp
        if self.damp_jitter_min is not None and self.damp_jitter_max is not None:
            lo = float(self.damp_jitter_min)
            hi = float(self.damp_jitter_max)
            u_d = float(torch.rand((), generator=self.damp_rng).item())
            step_damp = lo + (hi - lo) * u_d

        self.optimizer.zero_grad(set_to_none=True)
        metrics = run_v4_chunked_bptt_step(
            train_model=self.train_model,
            raw_model=self.model,
            physics=self.physics,
            x_star=x_star,
            boundary=boundary,
            u_init=u_init,
            num_particles=m,
            k_unroll=self.k_unroll,
            bptt_chunk_len=self.bptt_chunk_len,
            alpha_phys=self.alpha_phys,
            wta_winner_weight=self.wta_winner_weight,
            bce_weight=self.bce_weight,
            closed_loop_wall_rounds=self.closed_loop_wall_rounds,
            closed_loop_bce_scale=self.closed_loop_bce_scale,
            anti_collapse_weight=self.anti_collapse_weight,
            logit_reg_weight=self.logit_reg_weight,
            wavefront_weight=self.wavefront_weight,
            grad_scale=self.grad_scale,
            damp=step_damp,
            eta_max=self.eta_max,
            beta_max=self.beta_max,
            sync_h_damp=self.sync_h_damp,
            closed_loop_schedule=self.closed_loop_schedule,
            w_xor=self.w_xor,
            lam_comm_start=self.lam_comm_start,
            lam_comm_end=self.lam_comm_end,
            lam_align_start=self.lam_align_start,
            lam_align_end=self.lam_align_end,
            anneal_mode=self.anneal_mode,
            do_backward=True,
            backward_group_chunks=self.backward_group_chunks,
            sync_metrics=sync_metrics,
            radar_energy_fn=self._compiled_energy_core,
            retract_clamp_fn=self._compiled_retract_clamp,
            add32_fn=self._compiled_spin_add32,
            loss_head_fn=self._compiled_loss_head,
            clone_cudagraph_outputs=self._clone_cudagraph_outputs,
        )
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip, foreach=True)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        self._update_ema()
        return metrics

    @torch.no_grad()
    def evaluate_holdout(
        self, run_anneal: bool = True, use_neural_velocity: bool = True
    ) -> Dict[str, float]:
        """Evaluate EMA model with V3-identical Stage 3B holdout metrics + manifold diagnostics."""
        self.ema_model.eval()
        eval_bs = int(self.config.get("eval_batch_size", 256))
        total = self.holdout_w16.shape[0]

        bce_sum = 0.0
        zero_m1_sum = 0.0
        zero_m32_sum = 0.0
        zero_latch_sum = 0.0
        zero_align_sum = 0.0
        zero_comm_sum = 0.0
        zero_w0_correct = 0.0
        zero_w0_count = 0.0
        zero_round_correct = torch.zeros(self.rounds, dtype=torch.float64, device=self.device)
        zero_round_count = torch.zeros(self.rounds, dtype=torch.float64, device=self.device)
        zero_pos_correct = torch.zeros(32, dtype=torch.float64, device=self.device)
        zero_pos_count = torch.zeros(32, dtype=torch.float64, device=self.device)
        zero_w0_pos_correct = torch.zeros(32, dtype=torch.float64, device=self.device)
        zero_w0_pos_count = torch.zeros(32, dtype=torch.float64, device=self.device)
        n_batches = 0

        for start in range(0, total, eval_bs):
            w16_b = self.holdout_w16[start : start + eval_bs].to(self.device)
            x_star, cb = prepare_ground_truth_nodes(
                w16_b,
                rounds=self.rounds,
                domain_mode=self.domain_mode,
                num_bits=self.num_bits,
                msg_bytes_len=self.msg_bytes_len,
            )
            mb = ManifoldBoundary.from_clamped_boundary(cb)
            free_mask = 1.0 - mb.clamp_mask
            free_count = free_mask.sum().clamp_min(1.0)

            logits_z, x_disc_z, m1_z, m32_z, latch_z, align_z, comm_z = (
                self.solver.infer_deterministic_zero(
                    mb, use_neural_velocity=use_neural_velocity
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
            zero_align_sum += float(align_z.mean().item())
            zero_comm_sum += float(comm_z.mean().item())

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
            x_star_sub, cb_sub = prepare_ground_truth_nodes(
                w16_sub,
                rounds=self.rounds,
                domain_mode=self.domain_mode,
                num_bits=self.num_bits,
                msg_bytes_len=self.msg_bytes_len,
            )
            mb_sub = ManifoldBoundary.from_clamped_boundary(cb_sub)
            digests = sha256_digest(w16_sub, rounds=self.rounds)
            infer_rng = torch.Generator(device=self.device)
            infer_rng.manual_seed(77777)

            solve_res = self.solver.solve(
                target_digests=digests,
                boundary=mb_sub,
                generator=infer_rng,
                use_neural_velocity=use_neural_velocity,
            )

            free_sub = 1.0 - mb_sub.clamp_mask
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
            f"global_min_E32={eval_metrics['best_global_m32_energy']:.2f} "
            f"align={eval_metrics['zero_manifold_align']:.4f} "
            f"comm={eval_metrics['zero_manifold_comm']:.4f}",
            flush=True,
        )

    def train(self, max_steps: Optional[int] = None) -> None:
        """Run strict-FP32 V4 training loop from current step up to target steps."""
        target_steps = self.total_steps if max_steps is None else max_steps
        log_every = int(self.config.get("log_every", 100))
        eval_every = int(self.config.get("eval_every", 250))
        ckpt_every = int(self.config.get("checkpoint_every", 250))

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        print(
            f"[Phase-SHA {self.variant.upper()}] Starting FP32 training: D={self.manifold.dim}, "
            f"R={self.rounds}, B={self.batch_size}xM={self.num_particles}, "
            f"K_unroll={self.k_unroll} (chunk={self.bptt_chunk_len}), K_infer={self.solver.num_steps}, "
            f"anneal_K={self.solver.num_candidates}, compiled={self.compiled_active}, "
            f"device={self.device}, step={self.step}/{target_steps}",
            flush=True,
        )

        if self.step == 0:
            eval_0 = self.evaluate_holdout(run_anneal=True)
            round_str = " ".join(
                f"r{t}:{eval_0[f'per_round_acc_{t}']:.2f}" for t in range(self.rounds)
            )
            print(
                f"  [eval step=0] bce={eval_0['eval_bce_loss']:.4f} "
                f"zero_w0={eval_0['zero_w0_bit_acc']:.4f} "
                f"w0_acc={eval_0['w0_free_bit_acc']:.4f} "
                f"latch_z={eval_0['zero_latch_ratio']:.3f} "
                f"E_m1={eval_0['phys_energy_m1']:.2f} "
                f"E_m32={eval_0['phys_energy_m32']:.2f} "
                f"pre@K={eval_0['exact_preimage_rate@256']:.4f} "
                f"uniq={eval_0['unique_ratio@256']:.3f}",
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
            # Initialize best.pt at baseline if no checkpoint yet
            if eval_0["min_phys_energy_m32"] < self.best_min_m32:
                self.best_min_m32 = eval_0["min_phys_energy_m32"]
                self.best_eval_bce = eval_0["eval_bce_loss"]
                self.save_checkpoint(self.best_ckpt_name)

        use_prefetch = (
            bool(self.config.get("use_async_prefetch", True))
            and self.device.type == "cuda"
        )
        if use_prefetch:
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
                    dt_ms = (
                        ((now - interval_t0) * 1000.0) / float(max(1, interval_steps))
                        if interval_steps > 1
                        else (now - step_t0) * 1000.0
                    )
                    cur_lr = float(self.optimizer.param_groups[0]["lr"])

                    eval_metrics: Dict[str, float] = {}
                    if do_eval:
                        eval_metrics = self.evaluate_holdout(run_anneal=True)
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

                    print(
                        f"step={self.step:6d}/{target_steps} lr={cur_lr:.2e} "
                        f"loss={train_metrics['loss']:.4f} phys={train_metrics['phys_loss']:.4f} "
                        f"bce={train_metrics['bce_loss']:.4f} xor={train_metrics['xor_loss']:.3f} "
                        f"comm={train_metrics['comm_loss']:.3f} align={train_metrics['align_loss']:.3f} "
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
                            f"E_m1={eval_metrics['phys_energy_m1']:.2f} "
                            f"E_m32={eval_metrics['phys_energy_m32']:.2f} "
                            f"pre@K={eval_metrics['exact_preimage_rate@256']:.4f} "
                            f"uniq={eval_metrics['unique_ratio@256']:.3f}",
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
                        "train_xor_loss": train_metrics["xor_loss"],
                        "train_comm_loss": train_metrics["comm_loss"],
                        "train_align_loss": train_metrics["align_loss"],
                        "frontier_rounds": int(train_metrics["frontier_rounds"]),
                        "peak_vram_gb": round(peak_gb, 4),
                        "step_time_ms": round(dt_ms, 2),
                    }
                    row.update(eval_metrics)
                    self._append_csv_row(row)
                    interval_t0 = time.perf_counter()
                    interval_steps = 0

                if (self.step % ckpt_every == 0) or (self.step == target_steps):
                    self.save_checkpoint(self.latest_ckpt_name)
                    interval_t0 = time.perf_counter()
                    interval_steps = 0
        finally:
            self.stop_prefetch()
