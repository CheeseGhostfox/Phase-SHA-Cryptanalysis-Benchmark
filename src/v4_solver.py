"""Phase-SHA V4 inference engine: norm-preserving manifold ODE/SDE, dynamic latch, and slow-thinking SMC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.dual_basis import DualBasisEngine
from src.sha256_ref import sha256_digest, words_to_bits
from src.v3_solver import (
    _build_flip_table_16,
    _eval_eq_e_candidates,
    _eval_eq_s_candidates,
    _eval_eq_w_for_e_candidates,
    _eval_eq_w_for_w_candidates,
)
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
    apply_manifold_closed_loop_schedule,
    normalize_manifold,
)


@dataclass
class SolveResultV4:
    """Results from Phase-SHA V4 unrolled manifold inference."""

    solved: torch.Tensor  # (B,) bool
    num_hits: torch.Tensor  # (B,) int64
    unique_ratio: torch.Tensor  # (B,) float32
    latch_ratio: torch.Tensor  # (B,) float32
    best_w16: torch.Tensor  # (B, 16) int64
    best_energy: torch.Tensor  # (B,) float32
    best_m1_energy: torch.Tensor  # (B,) float32
    best_m32_energy: torch.Tensor  # (B,) float32
    min_m32_energy: torch.Tensor  # (B,) float32
    best_x_disc: torch.Tensor  # (B, R, 3, 32) float32 spins
    mean_align: torch.Tensor  # (B,) float32
    mean_comm: torch.Tensor  # (B,) float32
    final_u_sample: Optional[torch.Tensor] = None

    @property
    def solve_rate(self) -> float:
        return float(self.solved.float().mean().item())

    @property
    def mean_unique_ratio(self) -> float:
        return float(self.unique_ratio.float().mean().item())

    @property
    def mean_latch_ratio(self) -> float:
        return float(self.latch_ratio.float().mean().item())


def anneal_schedule(
    frac: float,
    start: float,
    end: float,
    mode: str = "cosine",
) -> float:
    """Smoothly ramp manifold alignment / commutator weights along unrolled depth frac in [0, 1]."""
    t = max(0.0, min(1.0, float(frac)))
    if mode == "linear":
        w = t
    else:
        import math

        w = 0.5 * (1.0 - math.cos(math.pi * t))
    return float(start) + (float(end) - float(start)) * w


class UnrolledSolverV4:
    """V4 test-time compute manifold solver supporting full-digest and V4E arbitrary masks."""

    def __init__(
        self,
        model: ManifoldBitGridBackbone,
        physics: ManifoldPhysicsEngine,
        num_candidates: int = 256,
        num_steps: int = 32,
        damp: float = 0.25,
        eta_max: float = 0.025,
        beta_max: float = 0.25,
        sde_noise_min: float = 0.10,
        sde_noise_max: float = 0.45,
        grad_scale: float = 2.0,
        chunk_size: int = 1024,
        use_neural_velocity: bool = True,
        use_dynamic_latch: bool = True,
        latch_every: int = 2,
        latch_conf_thresh: float = 0.985,
        latch_res_thresh: float = 8.0,
        latch_exact_tol: float = 1e-6,
        use_smc_resampling: bool = True,
        smc_resample_every: int = 8,
        smc_elite_k: int = 64,
        smc_backtrack_ratio: float = 0.25,
        sync_h_damp: bool = True,
        closed_loop_schedule: bool = True,
        w_xor: float = 1.0,
        lam_comm_start: float = 0.02,
        lam_comm_end: float = 0.60,
        lam_align_start: float = 0.02,
        lam_align_end: float = 0.60,
        anneal_mode: str = "cosine",
        eval_model: Optional[nn.Module] = None,
        compiled_energy_fn: Optional[Any] = None,
        compiled_add32_fn: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.eval_model: nn.Module = eval_model if eval_model is not None else model
        self.compiled_energy_fn: Optional[Any] = compiled_energy_fn
        self.compiled_add32_fn: Optional[Any] = compiled_add32_fn
        self.physics = physics
        self.manifold: BaseManifold = physics.manifold
        self.dual_basis: DualBasisEngine = physics.dual_basis
        self.num_candidates = int(num_candidates)
        self.num_steps = int(num_steps)
        self.damp = float(damp)
        self.eta_max = float(eta_max)
        self.beta_max = float(beta_max)
        self.sde_noise_min = float(sde_noise_min)
        self.sde_noise_max = float(sde_noise_max)
        self.grad_scale = float(grad_scale)
        self.chunk_size = int(chunk_size)
        self.use_neural_velocity = bool(use_neural_velocity)

        self.use_dynamic_latch = bool(use_dynamic_latch)
        self.latch_every = max(1, int(latch_every))
        self.latch_conf_thresh = float(latch_conf_thresh)
        self.latch_res_thresh = float(latch_res_thresh)
        self.latch_exact_tol = float(latch_exact_tol)

        self.use_smc_resampling = bool(use_smc_resampling)
        self.smc_resample_every = max(1, int(smc_resample_every))
        self.smc_elite_k = max(1, int(smc_elite_k))
        self.smc_backtrack_ratio = float(smc_backtrack_ratio)

        self.sync_h_damp = bool(sync_h_damp)
        self.closed_loop_schedule = bool(closed_loop_schedule)

        self.w_xor = float(w_xor)
        self.lam_comm_start = float(lam_comm_start)
        self.lam_comm_end = float(lam_comm_end)
        self.lam_align_start = float(lam_align_start)
        self.lam_align_end = float(lam_align_end)
        self.anneal_mode = str(anneal_mode)

        self._flip_table_cache: dict[torch.device, torch.Tensor] = {}

    def _get_flip_table(self, device: torch.device) -> torch.Tensor:
        if device not in self._flip_table_cache:
            self._flip_table_cache[device] = _build_flip_table_16(device)
        return self._flip_table_cache[device]

    def _build_sigma_and_sde(
        self, num_digests: int, k_total: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sigmas = torch.linspace(0.25, 1.40, k_total, dtype=torch.float32, device=device)
        sigmas[0] = 0.0
        sde = torch.linspace(
            self.sde_noise_min,
            self.sde_noise_max,
            k_total,
            dtype=torch.float32,
            device=device,
        )
        sde[0] = 0.0
        return (
            sigmas.unsqueeze(0).expand(num_digests, k_total),
            sde.unsqueeze(0).expand(num_digests, k_total),
        )

    def _init_diverse_manifold_states(
        self,
        b_chunk: ManifoldBoundary,
        sigmas_flat: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Traj 0 starts at neutral equator e_eq; trajs 1..K-1 sample diverse directions on S^{D-1}."""
        device = b_chunk.device
        rounds = b_chunk.rounds
        n_traj = b_chunk.batch_size
        raw = self.manifold.sample_random_manifold(
            (n_traj, rounds, 3, 32), device=device, generator=generator, spin_std=0.45
        )
        # Blend spin_std with sigma scale and assign neutral equator to sigma == 0 trajectories
        eq = self.manifold.make_neutral_equator((n_traj, rounds, 3, 32), device=device)
        is_zero = (sigmas_flat <= 1e-6).view(n_traj, 1, 1, 1, 1)
        u0 = torch.where(is_zero, eq, raw)
        return b_chunk.apply_clamp_manifold(u0, self.manifold)

    def _build_candidates_16(
        self, logits_sub: torch.Tensor, flip_table: torch.Tensor
    ) -> torch.Tensor:
        cand0 = torch.where(
            logits_sub < 0,
            torch.full_like(logits_sub, -1.0),
            torch.full_like(logits_sub, 1.0),
        )
        cands = cand0.unsqueeze(1).repeat(1, 16, 1)
        _, low_idx = torch.topk(logits_sub.abs(), k=5, dim=-1, largest=False)
        sign_mult = torch.where(flip_table, -1.0, 1.0).unsqueeze(0)
        idx_exp = low_idx.unsqueeze(1).expand(-1, 16, -1)
        gathered = torch.gather(cands, dim=-1, index=idx_exp)
        cands.scatter_(dim=-1, index=idx_exp, src=gathered * sign_mult)
        return cands

    def _attempt_slot_latch(
        self,
        eligible_mask: torch.Tensor,
        logits: torch.Tensor,
        t: int,
        ch: int,
        eval_fn,
        a_full: torch.Tensor,
        e_full: torch.Tensor,
        w_full: torch.Tensor,
        dyn_clamped: torch.Tensor,
        dyn_mask: torch.Tensor,
        u_cur: torch.Tensor,
        x_cur: torch.Tensor,
        latched_logits: torch.Tensor,
        a_locked: torch.Tensor,
        e_locked: torch.Tensor,
        w_locked: torch.Tensor,
        flip_table: torch.Tensor,
    ) -> bool:
        if not bool(eligible_mask.any().item()):
            return False
        idx = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)
        logits_sub = logits[idx, t, ch, :]
        cands = self._build_candidates_16(logits_sub, flip_table)
        cand_res = eval_fn(idx, cands)

        cand0_exact = cand_res[:, 0] < self.latch_exact_tol
        best_res, best_c = cand_res.min(dim=-1)
        exact_mask = cand0_exact | (best_res < self.latch_exact_tol)
        if not bool(exact_mask.any().item()):
            return False

        chosen_c = torch.where(cand0_exact, torch.zeros_like(best_c), best_c)
        hit_pos = torch.nonzero(exact_mask, as_tuple=False).squeeze(-1)
        hit_batch = idx[hit_pos]
        exact_words = cands[hit_pos, chosen_c[hit_pos], :]

        dyn_clamped[hit_batch, t, ch, :] = exact_words
        dyn_mask[hit_batch, t, ch, :] = 1.0
        x_cur[hit_batch, t, ch, :] = exact_words
        u_cur[hit_batch, t, ch, :, :] = self.manifold.embed_spins(exact_words)
        latched_logits[hit_batch, t, ch, :] = 4.0 * exact_words

        if ch == 0:
            a_full[hit_batch, t + 3, :] = exact_words
            a_locked[hit_batch, t + 3] = True
        elif ch == 1:
            e_full[hit_batch, t + 3, :] = exact_words
            e_locked[hit_batch, t + 3] = True
        else:
            w_full[hit_batch, t, :] = exact_words
            w_locked[hit_batch, t] = True
        return True

    def _try_dynamic_latch(
        self,
        u: torch.Tensor,
        logits: torch.Tensor,
        b_chunk: ManifoldBoundary,
        dyn_clamped: torch.Tensor,
        dyn_mask: torch.Tensor,
        latched_logits: torch.Tensor,
    ) -> torch.Tensor:
        u_out, _ = self._try_dynamic_latch_with_flag(
            u=u,
            logits=logits,
            b_chunk=b_chunk,
            dyn_clamped=dyn_clamped,
            dyn_mask=dyn_mask,
            latched_logits=latched_logits,
        )
        return u_out

    def _try_dynamic_latch_with_flag(
        self,
        u: torch.Tensor,
        logits: torch.Tensor,
        b_chunk: ManifoldBoundary,
        dyn_clamped: torch.Tensor,
        dyn_mask: torch.Tensor,
        latched_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        """Sweep R-1..0 verifying single-unknown equations and latching discrete poles."""
        rounds = b_chunk.rounds
        device = u.device
        flip_table = self._get_flip_table(device)

        u_cur = normalize_manifold(
            dyn_mask.unsqueeze(-1) * self.manifold.embed_spins(dyn_clamped)
            + (1.0 - dyn_mask.unsqueeze(-1)) * u
        )
        conf_word = torch.sigmoid(2.0 * logits.abs()).mean(dim=-1)
        if not bool((conf_word > self.latch_conf_thresh).any().item()):
            return u_cur, False

        u_a_full, u_e_full, u_w_full = b_chunk.assemble_full_manifold(u_cur, self.manifold)
        a_full = self.manifold.project_spin(u_a_full)
        e_full = self.manifold.project_spin(u_e_full)
        w_full = self.manifold.project_spin(u_w_full)
        x_cur = dyn_mask * dyn_clamped + (1.0 - dyn_mask) * self.manifold.project_spin(u_cur)

        delta_e, delta_w, delta_s = self.dual_basis.compute_wrapped_deltas(
            a_full, e_full, w_full
        )
        res_e = (2.0 * torch.sin(0.5 * delta_e).square()).sum(dim=-1)
        res_w = (2.0 * torch.sin(0.5 * delta_w).square()).sum(dim=-1)
        res_s = (
            (2.0 * torch.sin(0.5 * delta_s).square()).sum(dim=-1)
            if delta_s is not None
            else None
        )

        a_word_mask = (dyn_mask[:, :, 0, :] > 0.5).all(dim=-1)
        e_word_mask = (dyn_mask[:, :, 1, :] > 0.5).all(dim=-1)
        w_locked = (dyn_mask[:, :, 2, :] > 0.5).all(dim=-1).clone()

        left_a_ok = (b_chunk.A_full_mask[:, 0:3] > 0.5).all(dim=-1)
        left_e_ok = (b_chunk.E_full_mask[:, 0:3] > 0.5).all(dim=-1)
        right_a_ok = (b_chunk.A_full_mask[:, rounds + 3 : rounds + 4] > 0.5).all(dim=-1)
        right_e_ok = (b_chunk.E_full_mask[:, rounds + 3 : rounds + 4] > 0.5).all(dim=-1)
        a_locked = torch.cat([left_a_ok, a_word_mask, right_a_ok], dim=1)
        e_locked = torch.cat([left_e_ok, e_word_mask, right_e_ok], dim=1)
        did_latch = False

        for t in range(rounds - 1, -1, -1):
            if t >= 16 and res_s is not None:
                prereq_s = (
                    w_locked[:, t - 2]
                    & w_locked[:, t - 7]
                    & w_locked[:, t - 15]
                    & w_locked[:, t - 16]
                )
                elig_s = (
                    (~w_locked[:, t])
                    & prereq_s
                    & (conf_word[:, t, 2] > self.latch_conf_thresh)
                    & (res_s[:, t - 16] < self.latch_res_thresh)
                )
                if self._attempt_slot_latch(
                    elig_s,
                    logits,
                    t,
                    2,
                    lambda idx, cands, tt=t: _eval_eq_s_candidates(
                        self.dual_basis, w_full[idx], tt, cands
                    ),
                    a_full,
                    e_full,
                    w_full,
                    dyn_clamped,
                    dyn_mask,
                    u_cur,
                    x_cur,
                    latched_logits,
                    a_locked,
                    e_locked,
                    w_locked,
                    flip_table,
                ):
                    did_latch = True

            if 1 <= t <= rounds - 8:
                prereq_a = (
                    e_locked[:, t + 7]
                    & a_locked[:, t + 7]
                    & a_locked[:, t + 6]
                    & a_locked[:, t + 5]
                    & a_locked[:, t + 4]
                )
                elig_a = (
                    (~a_locked[:, t + 3])
                    & prereq_a
                    & (conf_word[:, t, 0] > self.latch_conf_thresh)
                    & (res_e[:, t + 3] < self.latch_res_thresh)
                )
                if self._attempt_slot_latch(
                    elig_a,
                    logits,
                    t,
                    0,
                    lambda idx, cands, tt=t: _eval_eq_e_candidates(
                        self.dual_basis, a_full[idx], e_full[idx], tt, cands
                    ),
                    a_full,
                    e_full,
                    w_full,
                    dyn_clamped,
                    dyn_mask,
                    u_cur,
                    x_cur,
                    latched_logits,
                    a_locked,
                    e_locked,
                    w_locked,
                    flip_table,
                ):
                    did_latch = True

            if 1 <= t <= rounds - 4:
                prereq_e = (
                    w_locked[:, t + 3]
                    & e_locked[:, t + 7]
                    & e_locked[:, t + 6]
                    & e_locked[:, t + 5]
                    & e_locked[:, t + 4]
                    & a_locked[:, t + 3]
                )
                elig_e = (
                    (~e_locked[:, t + 3])
                    & prereq_e
                    & (conf_word[:, t, 1] > self.latch_conf_thresh)
                    & (res_w[:, t + 3] < self.latch_res_thresh)
                )
                if self._attempt_slot_latch(
                    elig_e,
                    logits,
                    t,
                    1,
                    lambda idx, cands, tt=t: _eval_eq_w_for_e_candidates(
                        self.dual_basis, a_full[idx], e_full[idx], w_full[idx], tt, cands
                    ),
                    a_full,
                    e_full,
                    w_full,
                    dyn_clamped,
                    dyn_mask,
                    u_cur,
                    x_cur,
                    latched_logits,
                    a_locked,
                    e_locked,
                    w_locked,
                    flip_table,
                ):
                    did_latch = True

            prereq_w = (
                e_locked[:, t + 4]
                & e_locked[:, t + 3]
                & e_locked[:, t + 2]
                & e_locked[:, t + 1]
                & e_locked[:, t]
                & a_locked[:, t]
            )
            elig_w = (
                (~w_locked[:, t])
                & prereq_w
                & (conf_word[:, t, 2] > self.latch_conf_thresh)
                & (res_w[:, t] < self.latch_res_thresh)
            )
            if self._attempt_slot_latch(
                elig_w,
                logits,
                t,
                2,
                lambda idx, cands, tt=t: _eval_eq_w_for_w_candidates(
                    self.dual_basis, a_full[idx], e_full[idx], tt, cands
                ),
                a_full,
                e_full,
                w_full,
                dyn_clamped,
                dyn_mask,
                u_cur,
                x_cur,
                latched_logits,
                a_locked,
                e_locked,
                w_locked,
                flip_table,
            ):
                did_latch = True

        u_out = normalize_manifold(
            dyn_mask.unsqueeze(-1) * self.manifold.embed_spins(dyn_clamped)
            + (1.0 - dyn_mask.unsqueeze(-1)) * u_cur
        )
        return u_out, did_latch

    def _compute_discrete_energies(
        self,
        u: torch.Tensor,
        dyn_clamped: torch.Tensor,
        dyn_mask: torch.Tensor,
        b_chunk: ManifoldBoundary,
    ) -> torch.Tensor:
        s_proj = self.manifold.project_spin(u)
        s_bin = torch.where(s_proj < 0, torch.full_like(s_proj, -1.0), torch.full_like(s_proj, 1.0))
        s_disc = dyn_mask * dyn_clamped + (1.0 - dyn_mask) * s_bin
        u_disc = self.manifold.embed_spins(s_disc)
        u_a, u_e, u_w = b_chunk.assemble_full_manifold(u_disc, self.manifold)
        full_lam = torch.ones(32, dtype=torch.float32, device=u.device)
        energies, _, _ = self.physics.compute_energy_and_features(
            u_a,
            u_e,
            u_w,
            free_mask=1.0 - b_chunk.clamp_mask,
            lam=full_lam,
            digest_target_bits=b_chunk.digest_target_bits,
            digest_target_mask=b_chunk.digest_target_mask,
            iv_words=b_chunk.iv_words,
            w_xor=self.w_xor,
            lam_comm=0.0,
            lam_align=0.0,
            include_tail_e=b_chunk.has_free_right_boundary,
        )
        return energies

    def _smc_resample_v4(
        self,
        u: torch.Tensor,
        h_prev: Optional[torch.Tensor],
        dyn_clamped: torch.Tensor,
        dyn_mask: torch.Tensor,
        latched_logits: torch.Tensor,
        archive_state: Optional[Tuple[torch.Tensor, ...]],
        b_chunk: ManifoldBoundary,
        num_digests: int,
        k_per_digest: int,
        temp: float,
        generator: Optional[torch.Generator],
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, ...],
    ]:
        """Slow-thinking SMC resampling with historical lowest-energy backtracking archive."""
        device = u.device
        rounds = b_chunk.rounds
        m = num_digests
        k = k_per_digest
        n_elite = min(self.smc_elite_k, max(1, k // 4))
        n_off = k - n_elite
        n_back = max(1, min(n_elite, int(round(self.smc_backtrack_ratio * n_elite))))

        cur_energies = self._compute_discrete_energies(u, dyn_clamped, dyn_mask, b_chunk)
        cur_e_mk = cur_energies.view(m, k)
        d = self.manifold.dim

        # Merge historical backtracking elites (if present) for sorting & archive update
        if archive_state is not None:
            arc_u, arc_h, arc_dc, arc_dm, arc_ll, arc_e = archive_state
            cand_u = torch.cat([u.view(m, k, rounds, 3, 32, d), arc_u], dim=1)
            cand_dc = torch.cat([dyn_clamped.view(m, k, rounds, 3, 32), arc_dc], dim=1)
            cand_dm = torch.cat([dyn_mask.view(m, k, rounds, 3, 32), arc_dm], dim=1)
            cand_ll = torch.cat([latched_logits.view(m, k, rounds, 3, 32), arc_ll], dim=1)
            cand_e = torch.cat([cur_e_mk, arc_e], dim=1)
            cand_h = (
                torch.cat([h_prev.view(m, k, rounds, 32, h_prev.shape[-1]), arc_h], dim=1)
                if (h_prev is not None and arc_h is not None)
                else None
            )
        else:
            cand_u = u.view(m, k, rounds, 3, 32, d)
            cand_dc = dyn_clamped.view(m, k, rounds, 3, 32)
            cand_dm = dyn_mask.view(m, k, rounds, 3, 32)
            cand_ll = latched_logits.view(m, k, rounds, 3, 32)
            cand_e = cur_e_mk
            cand_h = (
                h_prev.view(m, k, rounds, 32, h_prev.shape[-1])
                if h_prev is not None
                else None
            )

        order = torch.argsort(cand_e, dim=1, descending=False)
        elite_local = order[:, :n_elite]
        new_arc_idx = order[:, :n_back]

        total_pool = cand_u.shape[1]
        base_offsets = torch.arange(m, device=device).unsqueeze(1) * total_pool
        flat_arc = (base_offsets + new_arc_idx).reshape(-1)
        new_archive = (
            cand_u.reshape(-1, rounds, 3, 32, d)[flat_arc].view(m, n_back, rounds, 3, 32, d).clone(),
            (
                cand_h.reshape(-1, rounds, 32, cand_h.shape[-1])[flat_arc]
                .view(m, n_back, rounds, 32, cand_h.shape[-1])
                .clone()
                if cand_h is not None
                else None
            ),
            cand_dc.reshape(-1, rounds, 3, 32)[flat_arc].view(m, n_back, rounds, 3, 32).clone(),
            cand_dm.reshape(-1, rounds, 3, 32)[flat_arc].view(m, n_back, rounds, 3, 32).clone(),
            cand_ll.reshape(-1, rounds, 3, 32)[flat_arc].view(m, n_back, rounds, 3, 32).clone(),
            torch.gather(cand_e, dim=1, index=new_arc_idx).clone(),
        )

        reps = (n_off + n_elite - 1) // n_elite
        off_local = elite_local.repeat(1, reps)[:, :n_off]
        gather_local = torch.cat([elite_local, off_local], dim=1)
        flat_idx = (base_offsets + gather_local).reshape(-1)

        u_mk = cand_u.reshape(-1, rounds, 3, 32, d)[flat_idx].view(m, k, rounds, 3, 32, d).clone()
        dc_new = cand_dc.reshape(-1, rounds, 3, 32)[flat_idx].clone()
        dm_new = cand_dm.reshape(-1, rounds, 3, 32)[flat_idx].clone()
        ll_new = cand_ll.reshape(-1, rounds, 3, 32)[flat_idx].clone()

        n1 = n_off // 3
        n2 = (2 * n_off) // 3
        sigma_off = torch.empty(n_off, dtype=torch.float32, device=device)
        flip_off = torch.empty(n_off, dtype=torch.float32, device=device)
        h_decay_off = torch.empty(n_off, dtype=torch.float32, device=device)

        sigma_off[:n1] = 0.25 * temp
        flip_off[:n1] = 0.0
        h_decay_off[:n1] = 1.0
        sigma_off[n1:n2] = 0.55 * temp
        flip_off[n1:n2] = 0.08
        h_decay_off[n1:n2] = 0.7
        sigma_off[n2:] = 0.90 * temp
        flip_off[n2:] = 0.18
        h_decay_off[n2:] = 0.4

        eps = torch.randn(
            (m, n_off, rounds, 3, 32, d),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        rand_u = torch.rand(
            (m, n_off, rounds, 3, 32),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        # Optional spin reflection on unlatched bits followed by tangent geodesic retraction
        flip_mult = torch.where(
            rand_u < flip_off.view(1, n_off, 1, 1, 1), -1.0, 1.0
        )
        u_off = u_mk[:, n_elite:].clone()
        u_off[..., self.manifold.spin_axis] = u_off[..., self.manifold.spin_axis] * flip_mult
        u_off = self.manifold.retract(
            u_off, eps, step_size=sigma_off.view(1, n_off, 1, 1, 1, 1)
        )

        dm_free = (1.0 - dm_new).view(m, k, rounds, 3, 32, 1) > 0.5
        dc_poles = self.manifold.embed_spins(dc_new).view(m, k, rounds, 3, 32, d)
        u_mk[:, n_elite:] = torch.where(dm_free[:, n_elite:], u_off, dc_poles[:, n_elite:])
        u_out = normalize_manifold(
            dm_new.unsqueeze(-1) * self.manifold.embed_spins(dc_new)
            + (1.0 - dm_new.unsqueeze(-1)) * u_mk.reshape(-1, rounds, 3, 32, d)
        )

        h_out: Optional[torch.Tensor] = None
        if cand_h is not None:
            d_bit = cand_h.shape[-1]
            h_mk = (
                cand_h.reshape(-1, rounds, 32, d_bit)[flat_idx]
                .view(m, k, rounds, 32, d_bit)
                .clone()
            )
            h_mk[:, n_elite:] = h_mk[:, n_elite:] * h_decay_off.view(1, n_off, 1, 1, 1)
            h_out = h_mk.reshape(-1, rounds, 32, d_bit)

        return u_out, h_out, dc_new, dm_new, ll_new, new_archive

    @torch.amp.autocast("cuda", enabled=False)
    def infer_v4(
        self,
        b_chunk: ManifoldBoundary,
        u_init: torch.Tensor,
        use_neural: bool = True,
        sde_scales: Optional[torch.Tensor] = None,
        num_digests: int = 1,
        k_per_digest: int = 1,
        generator: Optional[torch.Generator] = None,
        return_dyn_mask: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Run K_infer norm-preserving manifold steps with dual-mask isolation & slow-thinking SMC."""
        device = b_chunk.device
        rounds = b_chunk.rounds
        n_traj = b_chunk.batch_size
        k_spins = self.dual_basis.K_spins
        full_lam = torch.ones(32, dtype=torch.float32, device=device)

        prep_static = b_chunk.prepare_for_manifold(self.manifold)
        static_mask = prep_static.clamp_mask
        dyn_mask = static_mask.clone()
        dyn_clamped = b_chunk.x_clamped.clone()
        latched_logits = torch.zeros_like(dyn_clamped)

        u = apply_clamp_prepared(u_init.to(torch.float32), prep_static)
        if self.closed_loop_schedule and rounds > 16:
            u = apply_clamp_prepared(
                apply_manifold_closed_loop_schedule(
                    u, self.manifold, rounds, add32_fn=self.compiled_add32_fn
                ),
                prep_static,
            )

        h_prev: Optional[torch.Tensor] = None
        last_logits = torch.zeros((n_traj, rounds, 3, 32), dtype=torch.float32, device=device)
        archive_state: Optional[Tuple[torch.Tensor, ...]] = None
        denom = float(max(1, self.num_steps - 1))
        any_latched = False
        prep_active = prep_static

        frac_vals = [float(step_idx) / denom for step_idx in range(self.num_steps)]
        frac_vec = torch.tensor(frac_vals, dtype=torch.float32, device=device)
        step_frac_mat = frac_vec.unsqueeze(1).expand(self.num_steps, n_traj).contiguous()
        lam_comm_vec = torch.tensor(
            [
                anneal_schedule(f, self.lam_comm_start, self.lam_comm_end, self.anneal_mode)
                for f in frac_vals
            ],
            dtype=torch.float32,
            device=device,
        )
        lam_align_vec = torch.tensor(
            [
                anneal_schedule(f, self.lam_align_start, self.lam_align_end, self.anneal_mode)
                for f in frac_vals
            ],
            dtype=torch.float32,
            device=device,
        )
        w_xor_t = torch.tensor(float(self.w_xor), dtype=torch.float32, device=device)

        def _build_active_prep() -> PreparedManifoldBoundary:
            b_active = ManifoldBoundary(
                rounds=b_chunk.rounds,
                A_full_clamped=torch.cat(
                    [
                        b_chunk.A_full_clamped[:, 0:3],
                        dyn_clamped[:, :, 0, :],
                        b_chunk.A_full_clamped[:, rounds + 3 : rounds + 4],
                    ],
                    dim=1,
                ),
                A_full_mask=torch.cat(
                    [
                        b_chunk.A_full_mask[:, 0:3],
                        dyn_mask[:, :, 0, :],
                        b_chunk.A_full_mask[:, rounds + 3 : rounds + 4],
                    ],
                    dim=1,
                ),
                E_full_clamped=torch.cat(
                    [
                        b_chunk.E_full_clamped[:, 0:3],
                        dyn_clamped[:, :, 1, :],
                        b_chunk.E_full_clamped[:, rounds + 3 : rounds + 4],
                    ],
                    dim=1,
                ),
                E_full_mask=torch.cat(
                    [
                        b_chunk.E_full_mask[:, 0:3],
                        dyn_mask[:, :, 1, :],
                        b_chunk.E_full_mask[:, rounds + 3 : rounds + 4],
                    ],
                    dim=1,
                ),
                W_full_clamped=dyn_clamped[:, :, 2, :],
                W_full_mask=dyn_mask[:, :, 2, :],
                A_words_clamped=b_chunk.A_words_clamped,
                E_words_clamped=b_chunk.E_words_clamped,
                W16_clamped_words=b_chunk.W16_clamped_words,
                W16_mask_bits=b_chunk.W16_mask_bits,
                digest_target_bits=b_chunk.digest_target_bits,
                digest_target_mask=b_chunk.digest_target_mask,
                iv_words=b_chunk.iv_words,
                is_full_digest_clamped=b_chunk.is_full_digest_clamped,
            )
            return b_active.prepare_for_manifold(self.manifold)

        for step_idx in range(self.num_steps):
            frac = frac_vals[step_idx]
            step_tensor = step_frac_mat[step_idx]
            free_mask = (1.0 - dyn_mask) if any_latched else prep_static.free_mask
            free_mask_exp = free_mask.unsqueeze(-1) if any_latched else prep_static.free_mask_exp

            grad_radar, res_feat, _, grad_riem = self.physics.compute_radar_prepared(
                u_nodes=u,
                prep=prep_active,
                lam=full_lam,
                grad_scale=self.grad_scale,
                w_xor=w_xor_t,
                lam_comm=lam_comm_vec[step_idx],
                lam_align=lam_align_vec[step_idx],
                energy_fn=self.compiled_energy_fn,
                return_raw_grad=True,
            )

            if use_neural:
                # Dual-mask isolation: neural backbone always observes static_mask
                logits, delta_u_ambient, h_tilde = self.eval_model(
                    u_nodes=u,
                    grad_radar=grad_radar,
                    clamp_mask=static_mask,
                    res_features=res_feat,
                    k_spins=k_spins,
                    step_frac=step_tensor,
                    lam=full_lam,
                    h_prev=h_prev,
                )
                if self.sync_h_damp:
                    h_prev = (
                        h_tilde
                        if h_prev is None
                        else (1.0 - self.damp) * h_prev + self.damp * h_tilde
                    )
                else:
                    h_prev = h_tilde
                last_logits = logits
                x_spin_target = torch.tanh(logits)
                spin_pull = self.manifold.embed_spins(x_spin_target - self.manifold.project_spin(u))
                neural_drive = spin_pull + delta_u_ambient
            else:
                logits = last_logits
                neural_drive = torch.zeros_like(u)

            x_safe = self.manifold.project_spin(u).clamp(-2.0, 2.0)
            grad_bin = self.manifold.embed_spins(-4.0 * x_safe * (1.0 - x_safe * x_safe))
            eta_s = self.eta_max * (0.2 + 0.8 * frac)
            beta_s = self.beta_max * (frac * frac)
            phys_drive = torch.clamp(grad_riem + beta_s * grad_bin, -10.0, 10.0)

            temp = max(0.0, 1.0 - frac / 0.85)
            if sde_scales is not None and temp > 0.0:
                eps = torch.randn(
                    u.shape, dtype=torch.float32, device=device, generator=generator
                )
                sde_term = sde_scales.view(-1, 1, 1, 1, 1) * temp * eps * free_mask_exp
            else:
                sde_term = 0.0

            total_delta = neural_drive - eta_s * phys_drive + sde_term
            u_next = self.manifold.retract(u, total_delta, step_size=self.damp)
            u = apply_clamp_prepared(u_next, prep_active)

            if self.closed_loop_schedule and rounds > 16:
                u = apply_clamp_prepared(
                    apply_manifold_closed_loop_schedule(
                        u, self.manifold, rounds, add32_fn=self.compiled_add32_fn
                    ),
                    prep_active,
                )
                w_spins_exp = self.manifold.project_spin(u[:, 16:rounds, 2, :, :])
                logits_for_latch = logits.clone()
                logits_for_latch[:, 16:rounds, 2, :] = torch.atanh(
                    (0.995 * w_spins_exp).clamp(-0.995, 0.995)
                )
                last_logits = logits_for_latch
            else:
                logits_for_latch = logits

            if (
                self.use_dynamic_latch
                and use_neural
                and step_idx >= 1
                and (step_idx + 1) % self.latch_every == 0
            ):
                u, did_latch = self._try_dynamic_latch_with_flag(
                    u=u,
                    logits=logits_for_latch,
                    b_chunk=b_chunk,
                    dyn_clamped=dyn_clamped,
                    dyn_mask=dyn_mask,
                    latched_logits=latched_logits,
                )
                if did_latch:
                    any_latched = True
                    prep_active = _build_active_prep()

            if (
                self.use_smc_resampling
                and k_per_digest >= 16
                and (step_idx + 1) % self.smc_resample_every == 0
                and step_idx < int(0.85 * self.num_steps)
            ):
                smc_temp = max(0.15, 1.0 - frac)
                u, h_prev, dyn_clamped, dyn_mask, latched_logits, archive_state = (
                    self._smc_resample_v4(
                        u=u,
                        h_prev=h_prev,
                        dyn_clamped=dyn_clamped,
                        dyn_mask=dyn_mask,
                        latched_logits=latched_logits,
                        archive_state=archive_state,
                        b_chunk=b_chunk,
                        num_digests=num_digests,
                        k_per_digest=k_per_digest,
                        temp=smc_temp,
                        generator=generator,
                    )
                )
                if any_latched:
                    prep_active = _build_active_prep()
                if self.closed_loop_schedule and rounds > 16:
                    u = apply_clamp_prepared(
                        apply_manifold_closed_loop_schedule(
                            u, self.manifold, rounds, add32_fn=self.compiled_add32_fn
                        ),
                        prep_active,
                    )

        dyn_only_bits = torch.clamp(dyn_mask - static_mask, min=0.0)
        orig_free = (1.0 - static_mask).sum(dim=(1, 2, 3)).clamp_min(1.0)
        latch_ratio = dyn_only_bits.sum(dim=(1, 2, 3)) / orig_free
        final_logits = torch.where(dyn_only_bits > 0.5, latched_logits, last_logits)
        if return_dyn_mask:
            return u, final_logits, latch_ratio, dyn_mask
        return u, final_logits, latch_ratio

    @torch.no_grad()
    def infer_deterministic_zero(
        self,
        boundary: ManifoldBoundary,
        use_neural_velocity: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run deterministic zero-spin equator init V4 inference over a batch of boundaries."""
        use_neural = (
            self.use_neural_velocity
            if use_neural_velocity is None
            else bool(use_neural_velocity)
        )
        was_training = self.model.training
        self.model.eval()
        try:
            u_eq = self.manifold.make_neutral_equator(
                (boundary.batch_size, boundary.rounds, 3, 32), boundary.device
            )
            u_eq = boundary.apply_clamp_manifold(u_eq, self.manifold)
            u_final, final_logits, latch_ratio = self.infer_v4(
                boundary,
                u_eq,
                use_neural=use_neural,
                sde_scales=None,
                num_digests=boundary.batch_size,
                k_per_digest=1,
            )
            # Continuous manifold alignment / commutator diagnostics at end of ODE
            ua_c, ue_c, uw_c = boundary.assemble_full_manifold(u_final, self.manifold)
            full_lam = torch.ones(32, dtype=torch.float32, device=boundary.device)
            _, _, c_stats = self.physics.compute_energy_and_features(
                ua_c,
                ue_c,
                uw_c,
                free_mask=1.0 - boundary.clamp_mask,
                lam=full_lam,
                digest_target_bits=boundary.digest_target_bits,
                digest_target_mask=boundary.digest_target_mask,
                iv_words=boundary.iv_words,
                w_xor=self.w_xor,
                lam_comm=1.0,
                lam_align=1.0,
                include_tail_e=boundary.has_free_right_boundary,
            )

            s_proj = self.manifold.project_spin(u_final)
            x_disc = torch.where(
                s_proj < 0,
                torch.full_like(s_proj, -1.0),
                torch.full_like(s_proj, 1.0),
            )
            x_disc = boundary.clamp_mask * boundary.x_clamped + (1.0 - boundary.clamp_mask) * x_disc
            u_disc = self.manifold.embed_spins(x_disc)
            u_a, u_e, u_w = boundary.assemble_full_manifold(u_disc, self.manifold)
            _, _, stats = self.physics.compute_energy_and_features(
                u_a,
                u_e,
                u_w,
                free_mask=1.0 - boundary.clamp_mask,
                lam=full_lam,
                digest_target_bits=boundary.digest_target_bits,
                digest_target_mask=boundary.digest_target_mask,
                iv_words=boundary.iv_words,
                w_xor=self.w_xor,
                lam_comm=0.0,
                lam_align=0.0,
                include_tail_e=boundary.has_free_right_boundary,
            )
        finally:
            if was_training:
                self.model.train()

        return (
            final_logits,
            x_disc,
            stats["m1_energy"],
            stats["m32_energy"],
            latch_ratio,
            c_stats["energy_align"],
            c_stats["energy_comm"],
        )

    @torch.no_grad()
    def predict_calibrated_bit_probs(
        self,
        target_digests: Optional[torch.Tensor] = None,
        boundary: Optional[ManifoldBoundary] = None,
        domain_mode: str = "int_range",
        num_bits: int = 32,
        msg_bytes_len: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """Export calibrated P(bit=1) probabilities & verified latch masks (for V4D / V4E)."""
        if boundary is None:
            if target_digests is None:
                raise ValueError("Either target_digests or boundary must be provided.")
            if target_digests.ndim == 1:
                target_digests = target_digests.unsqueeze(0)
            boundary = build_manifold_boundary(
                digest=target_digests,
                rounds=self.physics.rounds,
                domain_mode=domain_mode,
                num_bits=num_bits,
                msg_bytes_len=msg_bytes_len,
            )
        was_training = self.model.training
        self.model.eval()
        try:
            u_eq = self.manifold.make_neutral_equator(
                (boundary.batch_size, boundary.rounds, 3, 32), boundary.device
            )
            u_eq = boundary.apply_clamp_manifold(u_eq, self.manifold)
            _, final_logits, latch_ratio, dyn_mask = self.infer_v4(
                boundary,
                u_eq,
                use_neural=self.use_neural_velocity,
                sde_scales=None,
                num_digests=boundary.batch_size,
                k_per_digest=1,
                return_dyn_mask=True,
            )
        finally:
            if was_training:
                self.model.train()

        cal_logits = torch.where(
            boundary.clamp_mask > 0.5, 4.0 * boundary.x_clamped, final_logits
        )
        node_bit1_probs = torch.sigmoid(-2.0 * cal_logits)
        return {
            "w0_bit1_probs": node_bit1_probs[:, 0, 2, :],
            "node_bit1_probs": node_bit1_probs,
            "node_logits": cal_logits,
            "latched_mask": dyn_mask > 0.5,
            "latch_ratio": latch_ratio,
        }

    @torch.inference_mode()
    def _evaluate_chunk(
        self,
        b_chunk: ManifoldBoundary,
        u_final: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        rounds = b_chunk.rounds
        full_lam = torch.ones(32, dtype=torch.float32, device=u_final.device)
        cand_w16 = b_chunk.extract_candidate_w16(u_final, self.manifold)
        cand_digest = sha256_digest(cand_w16, rounds=rounds)

        s_proj = self.manifold.project_spin(u_final)
        x_disc = torch.where(
            s_proj < 0,
            torch.full_like(s_proj, -1.0),
            torch.full_like(s_proj, 1.0),
        )
        x_disc = b_chunk.clamp_mask * b_chunk.x_clamped + (1.0 - b_chunk.clamp_mask) * x_disc
        u_disc = self.manifold.embed_spins(x_disc)
        u_a, u_e, u_w = b_chunk.assemble_full_manifold(u_disc, self.manifold)
        energies, _, stats = self.physics.compute_energy_and_features(
            u_a,
            u_e,
            u_w,
            free_mask=1.0 - b_chunk.clamp_mask,
            lam=full_lam,
            digest_target_bits=b_chunk.digest_target_bits,
            digest_target_mask=b_chunk.digest_target_mask,
            iv_words=b_chunk.iv_words,
            w_xor=self.w_xor,
            lam_comm=0.0,
            lam_align=0.0,
            include_tail_e=b_chunk.has_free_right_boundary,
        )

        # Evaluate continuous manifold alignment/comm diagnostics cleanly before hard threshold
        ua_c, ue_c, uw_c = b_chunk.assemble_full_manifold(u_final, self.manifold)
        _, _, c_stats = self.physics.compute_energy_and_features(
            ua_c,
            ue_c,
            uw_c,
            free_mask=1.0 - b_chunk.clamp_mask,
            lam=full_lam,
            w_xor=self.w_xor,
            lam_comm=1.0,
            lam_align=1.0,
            include_tail_e=b_chunk.has_free_right_boundary,
        )
        return (
            cand_w16,
            cand_digest,
            x_disc,
            energies,
            stats["m1_energy"],
            stats["m32_energy"],
            c_stats["energy_align"],
            c_stats["energy_comm"],
        )

    @staticmethod
    def _check_matches(
        cand_digest: torch.Tensor,
        target_digests: Optional[torch.Tensor],
        b_chunk: ManifoldBoundary,
    ) -> torch.Tensor:
        if target_digests is not None:
            return (cand_digest == target_digests).all(dim=-1)
        if b_chunk.digest_target_mask is not None and b_chunk.digest_target_bits is not None:
            cand_bits = words_to_bits(cand_digest).to(torch.float32)
            t_mask = b_chunk.digest_target_mask > 0.5
            bit_ok = (cand_bits == b_chunk.digest_target_bits) | (~t_mask)
            return bit_ok.all(dim=(1, 2))
        raise ValueError("Must supply target_digests or digest_target_mask.")

    @torch.no_grad()
    def solve(
        self,
        target_digests: Optional[torch.Tensor] = None,
        boundary: Optional[ManifoldBoundary] = None,
        domain_mode: str = "int_range",
        num_bits: int = 32,
        msg_bytes_len: int = 4,
        generator: Optional[torch.Generator] = None,
        use_neural_velocity: Optional[bool] = None,
    ) -> SolveResultV4:
        """Slow-thinking parallel SMC solve over full digests or arbitrary V4E partial boundaries."""
        if boundary is None:
            if target_digests is None:
                raise ValueError("Either target_digests or boundary must be provided.")
            if target_digests.ndim == 1:
                target_digests = target_digests.unsqueeze(0)
            boundary = build_manifold_boundary(
                digest=target_digests,
                rounds=self.physics.rounds,
                domain_mode=domain_mode,
                num_bits=num_bits,
                msg_bytes_len=msg_bytes_len,
            )
        elif target_digests is not None and target_digests.ndim == 1:
            target_digests = target_digests.unsqueeze(0)

        m = boundary.batch_size
        k = self.num_candidates
        rounds = boundary.rounds
        device = boundary.device
        use_neural = (
            self.use_neural_velocity
            if use_neural_velocity is None
            else bool(use_neural_velocity)
        )

        total_traj = m * k
        if total_traj <= self.chunk_size or (m == 1 and k > self.chunk_size):
            # Evaluate m=1 in chunks of size <= self.chunk_size (no infinite recursion)
            was_training = self.model.training
            self.model.eval()
            sigmas_mk, sde_mk = self._build_sigma_and_sde(m, k, device)
            if total_traj <= self.chunk_size:
                try:
                    b_exp = boundary.repeat_for_candidates(k)
                    u0 = self._init_diverse_manifold_states(
                        b_exp, sigmas_flat=sigmas_mk.reshape(-1), generator=generator
                    )
                    u_final, _, latch_ratios = self.infer_v4(
                        b_exp,
                        u0,
                        use_neural=use_neural,
                        sde_scales=sde_mk.reshape(-1),
                        num_digests=m,
                        k_per_digest=k,
                        generator=generator,
                    )
                    (
                        cand_w16,
                        cand_digest,
                        x_disc,
                        energies,
                        m1_e,
                        m32_e,
                        align_e,
                        comm_e,
                    ) = self._evaluate_chunk(b_exp, u_final)
                finally:
                    if was_training:
                        self.model.train()
                u_sample_ret = (
                    u_final.view(m, k, rounds, 3, 32, self.manifold.dim)[0]
                    if m == 1
                    else None
                )
            else:
                # m == 1, k > chunk_size
                w16_parts, dig_parts, xd_parts = [], [], []
                e_parts, m1_parts, m32_parts, lr_parts, al_parts, cm_parts = [], [], [], [], [], []
                u_last = None
                try:
                    for start in range(0, k, self.chunk_size):
                        cur_k = min(self.chunk_size, k - start)
                        b_sub = boundary.repeat_for_candidates(cur_k)
                        u0 = self._init_diverse_manifold_states(
                            b_sub,
                            sigmas_flat=sigmas_mk[0, start : start + cur_k],
                            generator=generator,
                        )
                        u_fin_sub, _, lr_sub = self.infer_v4(
                            b_sub,
                            u0,
                            use_neural=use_neural,
                            sde_scales=sde_mk[0, start : start + cur_k],
                            num_digests=1,
                            k_per_digest=cur_k,
                            generator=generator,
                        )
                        cw, cd, xd, en, m1, m32, al, cm = self._evaluate_chunk(b_sub, u_fin_sub)
                        w16_parts.append(cw)
                        dig_parts.append(cd)
                        xd_parts.append(xd)
                        e_parts.append(en)
                        m1_parts.append(m1)
                        m32_parts.append(m32)
                        lr_parts.append(lr_sub)
                        al_parts.append(al)
                        cm_parts.append(cm)
                        u_last = u_fin_sub
                finally:
                    if was_training:
                        self.model.train()
                cand_w16 = torch.cat(w16_parts, dim=0)
                cand_digest = torch.cat(dig_parts, dim=0)
                x_disc = torch.cat(xd_parts, dim=0)
                energies = torch.cat(e_parts, dim=0)
                m1_e = torch.cat(m1_parts, dim=0)
                m32_e = torch.cat(m32_parts, dim=0)
                latch_ratios = torch.cat(lr_parts, dim=0)
                align_e = torch.cat(al_parts, dim=0)
                comm_e = torch.cat(cm_parts, dim=0)
                b_exp = boundary.repeat_for_candidates(k)
                u_sample_ret = u_last

            t_dig_exp = (
                target_digests.repeat_interleave(k, dim=0)
                if target_digests is not None
                else None
            )
            matches = self._check_matches(cand_digest, t_dig_exp, b_exp).view(m, k)
            hits = matches.sum(dim=-1)
            solved = hits > 0

            cand_w16_mk = cand_w16.view(m, k, 16)
            x_disc_mk = x_disc.view(m, k, rounds, 3, 32)
            energies_mk = energies.view(m, k)
            m1_mk = m1_e.view(m, k)
            m32_mk = m32_e.view(m, k)
            latch_mk = latch_ratios.view(m, k)
            align_mk = align_e.view(m, k)
            comm_mk = comm_e.view(m, k)

            min_indices = torch.argmin(energies_mk, dim=-1)
            hit_indices = torch.argmax(matches.to(torch.int64), dim=-1)
            chosen_idx = torch.where(solved, hit_indices, min_indices)
            b_idx = torch.arange(m, device=device)

            uniq_ratios = torch.tensor(
                [
                    float(torch.unique(cand_w16_mk[i, :, 0]).numel()) / float(k)
                    for i in range(m)
                ],
                dtype=torch.float32,
                device=device,
            )
            return SolveResultV4(
                solved=solved,
                num_hits=hits,
                unique_ratio=uniq_ratios,
                latch_ratio=latch_mk[b_idx, chosen_idx],
                best_w16=cand_w16_mk[b_idx, chosen_idx],
                best_energy=energies_mk[b_idx, chosen_idx],
                best_m1_energy=m1_mk[b_idx, chosen_idx],
                best_m32_energy=m32_mk[b_idx, chosen_idx],
                min_m32_energy=m32_mk.min(dim=-1).values,
                best_x_disc=x_disc_mk[b_idx, chosen_idx],
                mean_align=align_mk.mean(dim=-1),
                mean_comm=comm_mk.mean(dim=-1),
                final_u_sample=u_sample_ret,
            )

        # Multi-digest batched chunking when M * K > chunk_size
        best_w16_l = []
        best_e_l = []
        best_m1_l = []
        best_m32_l = []
        min_m32_l = []
        best_x_l = []
        solved_l = []
        hits_l = []
        uniq_l = []
        latch_l = []
        align_l = []
        comm_l = []
        m_step = max(1, self.chunk_size // max(1, k))
        for start_i in range(0, m, m_step):
            end_i = min(m, start_i + m_step)
            b_sub = ManifoldBoundary(
                rounds=boundary.rounds,
                A_full_clamped=boundary.A_full_clamped[start_i:end_i],
                A_full_mask=boundary.A_full_mask[start_i:end_i],
                E_full_clamped=boundary.E_full_clamped[start_i:end_i],
                E_full_mask=boundary.E_full_mask[start_i:end_i],
                W_full_clamped=boundary.W_full_clamped[start_i:end_i],
                W_full_mask=boundary.W_full_mask[start_i:end_i],
                A_words_clamped=boundary.A_words_clamped[start_i:end_i],
                E_words_clamped=boundary.E_words_clamped[start_i:end_i],
                W16_clamped_words=boundary.W16_clamped_words[start_i:end_i],
                W16_mask_bits=boundary.W16_mask_bits[start_i:end_i],
                digest_target_bits=(
                    None
                    if boundary.digest_target_bits is None
                    else boundary.digest_target_bits[start_i:end_i]
                ),
                digest_target_mask=(
                    None
                    if boundary.digest_target_mask is None
                    else boundary.digest_target_mask[start_i:end_i]
                ),
                iv_words=boundary.iv_words,
                is_full_digest_clamped=boundary.is_full_digest_clamped,
            )
            td_sub = None if target_digests is None else target_digests[start_i:end_i]
            r_sub = self.solve(
                target_digests=td_sub,
                boundary=b_sub,
                generator=generator,
                use_neural_velocity=use_neural,
            )
            solved_l.append(r_sub.solved)
            hits_l.append(r_sub.num_hits)
            uniq_l.append(r_sub.unique_ratio)
            latch_l.append(r_sub.latch_ratio)
            best_w16_l.append(r_sub.best_w16)
            best_e_l.append(r_sub.best_energy)
            best_m1_l.append(r_sub.best_m1_energy)
            best_m32_l.append(r_sub.best_m32_energy)
            min_m32_l.append(r_sub.min_m32_energy)
            best_x_l.append(r_sub.best_x_disc)
            align_l.append(r_sub.mean_align)
            comm_l.append(r_sub.mean_comm)

        return SolveResultV4(
            solved=torch.cat(solved_l, dim=0),
            num_hits=torch.cat(hits_l, dim=0),
            unique_ratio=torch.cat(uniq_l, dim=0),
            latch_ratio=torch.cat(latch_l, dim=0),
            best_w16=torch.cat(best_w16_l, dim=0),
            best_energy=torch.cat(best_e_l, dim=0),
            best_m1_energy=torch.cat(best_m1_l, dim=0),
            best_m32_energy=torch.cat(best_m32_l, dim=0),
            min_m32_energy=torch.cat(min_m32_l, dim=0),
            best_x_disc=torch.cat(best_x_l, dim=0),
            mean_align=torch.cat(align_l, dim=0),
            mean_comm=torch.cat(comm_l, dim=0),
        )
