"""Phase-SHA V3 inference engine: dynamic digital latch, local bit-flip check, and SMC resampling."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from src.bit_grid_backbone import BitGridBackbone
from src.boundary_clamp import ClampedBoundary, build_clamped_boundary
from src.dual_basis import (
    DualBasisEngine,
    spin_Ch,
    spin_Maj,
    spin_Sigma0,
    spin_Sigma1,
    spin_sigma0,
    spin_sigma1,
)
from src.sha256_ref import sha256_digest


def spin_add32(x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
    """Exact differentiable multilinear 32-bit modular ripple adder on bipolar spins (..., 32)."""
    c = torch.ones_like(x_a[..., 0])
    out_bits = []
    for k in range(32):
        ak = x_a[..., k]
        bk = x_b[..., k]
        prod = ak * bk * c
        out_bits.append(prod)
        if k < 31:
            c = 0.5 * (ak + bk + c - prod)
    return torch.stack(out_bits, dim=-1)


def spin_expand_schedule(w_base: torch.Tensor, rounds: int) -> torch.Tensor:
    """Synthesize continuous expanded schedule W_{0..R-1} from W_{0..15} spins (..., 16, 32)."""
    if rounds <= 16:
        return w_base[:, :rounds, :]
    w_cols = [w_base[:, t, :] for t in range(16)]
    for t in range(16, rounds):
        s1 = spin_sigma1(w_cols[t - 2])
        s0 = spin_sigma0(w_cols[t - 15])
        wt = spin_add32(
            spin_add32(s1, w_cols[t - 7]),
            spin_add32(s0, w_cols[t - 16]),
        )
        w_cols.append(wt)
    return torch.stack(w_cols, dim=1)


def apply_closed_loop_schedule(x_nodes: torch.Tensor, rounds: int) -> torch.Tensor:
    """Replace W_{16..R-1} channel spins with differentiable schedule expansion of W_{0..15}."""
    if rounds <= 16:
        return x_nodes
    w_exp = spin_expand_schedule(x_nodes[:, :16, 2, :], rounds)
    out = x_nodes.clone()
    out[:, 16:rounds, 2, :] = w_exp[:, 16:rounds, :]
    return out


@dataclass
class SolveResultV3:
    """Results from Phase-SHA V3 unrolled neural-physics inference."""

    solved: torch.Tensor  # (B,) bool
    num_hits: torch.Tensor  # (B,) int64
    unique_ratio: torch.Tensor  # (B,) float32
    latch_ratio: torch.Tensor  # (B,) float32
    best_w16: torch.Tensor  # (B, 16) int64
    best_energy: torch.Tensor  # (B,) float32
    best_m1_energy: torch.Tensor  # (B,) float32
    best_m32_energy: torch.Tensor  # (B,) float32
    min_m32_energy: torch.Tensor  # (B,) float32
    best_x_disc: torch.Tensor  # (B, R, 3, 32) float32
    final_x_sample: Optional[torch.Tensor] = None  # (K, R, 3, 32)

    @property
    def solve_rate(self) -> float:
        return float(self.solved.float().mean().item())

    @property
    def mean_unique_ratio(self) -> float:
        return float(self.unique_ratio.float().mean().item())

    @property
    def mean_latch_ratio(self) -> float:
        return float(self.latch_ratio.float().mean().item())


def _build_flip_table_16(device: torch.device) -> torch.Tensor:
    """Build (16, 5) bool mask: 0 flips (1) + 1-bit flips (5) + 2-bit flips (10)."""
    rows = [[False] * 5]
    for i in range(5):
        r = [False] * 5
        r[i] = True
        rows.append(r)
    for i, j in itertools.combinations(range(5), 2):
        r = [False] * 5
        r[i] = True
        r[j] = True
        rows.append(r)
    return torch.tensor(rows, dtype=torch.bool, device=device)


@torch.amp.autocast("cuda", enabled=False)
def _eval_eq_s_candidates(
    dual_basis: DualBasisEngine,
    w_full: torch.Tensor,
    t: int,
    cand_w: torch.Tensor,
) -> torch.Tensor:
    """Evaluate single-equation Eq S(t) 32-scale residual on (M, C, 32) W_t candidates."""
    sig1 = spin_sigma1(w_full[:, t - 2 : t - 1].to(torch.float32)).squeeze(1)
    sig0 = spin_sigma0(w_full[:, t - 15 : t - 14].to(torch.float32)).squeeze(1)
    rhs = (
        dual_basis.spins_to_phase_fp64(sig1)
        + dual_basis.spins_to_phase_fp64(w_full[:, t - 7])
        + dual_basis.spins_to_phase_fp64(sig0)
        + dual_basis.spins_to_phase_fp64(w_full[:, t - 16])
    )
    delta = dual_basis._wrap_to_fp32(
        dual_basis.spins_to_phase_fp64(cand_w) - rhs.unsqueeze(1)
    )
    return (2.0 * torch.sin(0.5 * delta).square()).sum(dim=-1)


@torch.amp.autocast("cuda", enabled=False)
def _eval_eq_e_candidates(
    dual_basis: DualBasisEngine,
    a_full: torch.Tensor,
    e_full: torch.Tensor,
    t: int,
    cand_a: torch.Tensor,
) -> torch.Tensor:
    """Evaluate Eq E(s=t+3) 32-scale residual on (M, C, 32) A_t candidates."""
    a6 = a_full[:, t + 6].to(torch.float32)
    a5 = a_full[:, t + 5].to(torch.float32)
    a4 = a_full[:, t + 4].to(torch.float32)
    fixed = (
        dual_basis.spins_to_phase_fp64(e_full[:, t + 7])
        + dual_basis.spins_to_phase_fp64(spin_Sigma0(a6))
        + dual_basis.spins_to_phase_fp64(spin_Maj(a6, a5, a4))
        - dual_basis.spins_to_phase_fp64(a_full[:, t + 7])
    )
    delta = dual_basis._wrap_to_fp32(
        fixed.unsqueeze(1) - dual_basis.spins_to_phase_fp64(cand_a)
    )
    return (2.0 * torch.sin(0.5 * delta).square()).sum(dim=-1)


@torch.amp.autocast("cuda", enabled=False)
def _eval_eq_w_for_e_candidates(
    dual_basis: DualBasisEngine,
    a_full: torch.Tensor,
    e_full: torch.Tensor,
    w_full: torch.Tensor,
    t: int,
    cand_e: torch.Tensor,
) -> torch.Tensor:
    """Evaluate Eq W(s=t+3) 32-scale residual on (M, C, 32) E_t candidates."""
    s = t + 3
    e3 = e_full[:, s + 3].to(torch.float32)
    e2 = e_full[:, s + 2].to(torch.float32)
    e1 = e_full[:, s + 1].to(torch.float32)
    fixed = (
        dual_basis.spins_to_phase_fp64(w_full[:, s])
        + dual_basis.theta_K[s].unsqueeze(0)
        + dual_basis.spins_to_phase_fp64(spin_Sigma1(e3))
        + dual_basis.spins_to_phase_fp64(spin_Ch(e3, e2, e1))
        + dual_basis.spins_to_phase_fp64(a_full[:, s])
        - dual_basis.spins_to_phase_fp64(e_full[:, s + 4])
    )
    delta = dual_basis._wrap_to_fp32(
        fixed.unsqueeze(1) + dual_basis.spins_to_phase_fp64(cand_e)
    )
    return (2.0 * torch.sin(0.5 * delta).square()).sum(dim=-1)


@torch.amp.autocast("cuda", enabled=False)
def _eval_eq_w_for_w_candidates(
    dual_basis: DualBasisEngine,
    a_full: torch.Tensor,
    e_full: torch.Tensor,
    t: int,
    cand_w: torch.Tensor,
) -> torch.Tensor:
    """Evaluate Eq W(t) 32-scale residual on (M, C, 32) W_t candidates."""
    e3 = e_full[:, t + 3].to(torch.float32)
    e2 = e_full[:, t + 2].to(torch.float32)
    e1 = e_full[:, t + 1].to(torch.float32)
    fixed = (
        dual_basis.theta_K[t].unsqueeze(0)
        + dual_basis.spins_to_phase_fp64(spin_Sigma1(e3))
        + dual_basis.spins_to_phase_fp64(spin_Ch(e3, e2, e1))
        + dual_basis.spins_to_phase_fp64(e_full[:, t])
        + dual_basis.spins_to_phase_fp64(a_full[:, t])
        - dual_basis.spins_to_phase_fp64(e_full[:, t + 4])
    )
    delta = dual_basis._wrap_to_fp32(
        fixed.unsqueeze(1) + dual_basis.spins_to_phase_fp64(cand_w)
    )
    return (2.0 * torch.sin(0.5 * delta).square()).sum(dim=-1)


class UnrolledSolverV3:
    """V3 iterative solver with Dynamic Digital Latch, local bit-flip search, and SMC resampling."""

    def __init__(
        self,
        model: BitGridBackbone,
        dual_basis: DualBasisEngine,
        num_candidates: int = 256,
        num_steps: int = 64,
        damp: float = 0.08,
        eta_max: float = 0.02,
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
        sync_h_damp: bool = False,
        closed_loop_schedule: bool = False,
    ) -> None:
        self.model = model
        self.dual_basis = dual_basis
        self.num_candidates = num_candidates
        self.num_steps = num_steps
        self.damp = damp
        self.eta_max = eta_max
        self.beta_max = beta_max
        self.sde_noise_min = sde_noise_min
        self.sde_noise_max = sde_noise_max
        self.grad_scale = grad_scale
        self.chunk_size = chunk_size
        self.use_neural_velocity = use_neural_velocity

        self.use_dynamic_latch = use_dynamic_latch
        self.latch_every = max(1, int(latch_every))
        self.latch_conf_thresh = float(latch_conf_thresh)
        self.latch_res_thresh = float(latch_res_thresh)
        self.latch_exact_tol = float(latch_exact_tol)

        self.use_smc_resampling = use_smc_resampling
        self.smc_resample_every = max(1, int(smc_resample_every))
        self.smc_elite_k = max(1, int(smc_elite_k))

        self.sync_h_damp = bool(sync_h_damp)
        self.closed_loop_schedule = bool(closed_loop_schedule)

        self._flip_table_cache: dict[torch.device, torch.Tensor] = {}

    def _get_flip_table(self, device: torch.device) -> torch.Tensor:
        if device not in self._flip_table_cache:
            self._flip_table_cache[device] = _build_flip_table_16(device)
        return self._flip_table_cache[device]

    def _build_sigma_and_sde(
        self, num_digests: int, k_total: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigmas = torch.linspace(0.25, 1.80, k_total, device=device)
        sigmas[0] = 0.0
        sde = torch.linspace(
            self.sde_noise_min, self.sde_noise_max, k_total, device=device
        )
        sde[0] = 0.0
        return (
            sigmas.unsqueeze(0).expand(num_digests, k_total),
            sde.unsqueeze(0).expand(num_digests, k_total),
        )

    def _init_diverse_states(
        self,
        b_chunk: ClampedBoundary,
        sigmas_flat: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Initialize index-0 trajectory at 0 and 1..K-1 with diverse continuous noise."""
        device = b_chunk.device
        rounds = b_chunk.rounds
        n_traj = b_chunk.batch_size
        eps = torch.randn(
            (n_traj, rounds, 3, 32),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        x0 = torch.tanh(sigmas_flat.view(n_traj, 1, 1, 1) * eps)
        return b_chunk.apply_clamp(x0)

    def _build_candidates_16(
        self, logits_sub: torch.Tensor, flip_table: torch.Tensor
    ) -> torch.Tensor:
        """Build 16 candidate discrete words per item: exact sign + 1..2 bit flips on top-5 uncertain."""
        cand0 = torch.where(
            logits_sub < 0,
            torch.full_like(logits_sub, -1.0),
            torch.full_like(logits_sub, 1.0),
        )  # (M, 32)
        cands = cand0.unsqueeze(1).repeat(1, 16, 1)  # (M, 16, 32)
        _, low_idx = torch.topk(logits_sub.abs(), k=5, dim=-1, largest=False)  # (M, 5)
        sign_mult = torch.where(flip_table, -1.0, 1.0).unsqueeze(0)  # (1, 16, 5)
        idx_exp = low_idx.unsqueeze(1).expand(-1, 16, -1)  # (M, 16, 5)
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
        x: torch.Tensor,
        latched_logits: torch.Tensor,
        a_locked: torch.Tensor,
        e_locked: torch.Tensor,
        w_locked: torch.Tensor,
        flip_table: torch.Tensor,
    ) -> None:
        """Check 16 discrete candidates for eligible trajectories and latch verified exact words."""
        if not bool(eligible_mask.any().item()):
            return
        idx = torch.nonzero(eligible_mask, as_tuple=False).squeeze(-1)
        logits_sub = logits[idx, t, ch, :]  # (M, 32)
        cands = self._build_candidates_16(logits_sub, flip_table)  # (M, 16, 32)
        cand_res = eval_fn(idx, cands)  # (M, 16)

        # Prefer candidate 0 (sign(logits)) when exact; otherwise pick lowest-residual flip
        cand0_exact = cand_res[:, 0] < self.latch_exact_tol
        best_res, best_c = cand_res.min(dim=-1)
        exact_mask = cand0_exact | (best_res < self.latch_exact_tol)
        if not bool(exact_mask.any().item()):
            return

        chosen_c = torch.where(cand0_exact, torch.zeros_like(best_c), best_c)
        hit_pos = torch.nonzero(exact_mask, as_tuple=False).squeeze(-1)
        hit_batch = idx[hit_pos]
        exact_words = cands[hit_pos, chosen_c[hit_pos], :]  # (H, 32)

        dyn_clamped[hit_batch, t, ch, :] = exact_words
        dyn_mask[hit_batch, t, ch, :] = 1.0
        x[hit_batch, t, ch, :] = exact_words
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

    def _try_dynamic_latch(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        b_chunk: ClampedBoundary,
        dyn_clamped: torch.Tensor,
        dyn_mask: torch.Tensor,
        latched_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Sweep R-1..0 checking single-unknown discrete equations and locking verified words."""
        rounds = b_chunk.rounds
        device = x.device
        flip_table = self._get_flip_table(device)

        x_cur = dyn_mask * dyn_clamped + (1.0 - dyn_mask) * x
        a_full, e_full, w_full = b_chunk.assemble_full(x_cur)

        delta_e, delta_w, delta_s = self.dual_basis.compute_wrapped_deltas(
            a_full, e_full, w_full
        )
        res_e = (2.0 * torch.sin(0.5 * delta_e).square()).sum(dim=-1)  # (B, R-4)
        res_w = (2.0 * torch.sin(0.5 * delta_w).square()).sum(dim=-1)  # (B, R)
        res_s = (
            (2.0 * torch.sin(0.5 * delta_s).square()).sum(dim=-1)
            if delta_s is not None
            else None
        )

        # Calibrated word confidence matching sigma(2*logits) prob parameterization
        conf_word = torch.sigmoid(2.0 * logits.abs()).mean(dim=-1)  # (B, R, 3)

        # Word-level lock status across (A_full, E_full, W_full)
        a_word_mask = (dyn_mask[:, :, 0, :] > 0.5).all(dim=-1)  # (B, R)
        e_word_mask = (dyn_mask[:, :, 1, :] > 0.5).all(dim=-1)  # (B, R)
        w_locked = (dyn_mask[:, :, 2, :] > 0.5).all(dim=-1).clone()  # (B, R)

        bsz = x.shape[0]
        left_ok = torch.ones((bsz, 3), dtype=torch.bool, device=device)
        right_ok = torch.ones((bsz, 1), dtype=torch.bool, device=device)
        a_locked = torch.cat([left_ok, a_word_mask, right_ok], dim=1)  # (B, R+4)
        e_locked = torch.cat([left_ok, e_word_mask, right_ok], dim=1)  # (B, R+4)

        for t in range(rounds - 1, -1, -1):
            # 1. W_t via Eq S(t) (for t >= 16)
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
                self._attempt_slot_latch(
                    eligible_mask=elig_s,
                    logits=logits,
                    t=t,
                    ch=2,
                    eval_fn=lambda idx, cands, tt=t: _eval_eq_s_candidates(
                        self.dual_basis, w_full[idx], tt, cands
                    ),
                    a_full=a_full,
                    e_full=e_full,
                    w_full=w_full,
                    dyn_clamped=dyn_clamped,
                    dyn_mask=dyn_mask,
                    x=x_cur,
                    latched_logits=latched_logits,
                    a_locked=a_locked,
                    e_locked=e_locked,
                    w_locked=w_locked,
                    flip_table=flip_table,
                )

            # 2. A_t via Eq E(t+3) (for 1 <= t <= R - 8)
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
                self._attempt_slot_latch(
                    eligible_mask=elig_a,
                    logits=logits,
                    t=t,
                    ch=0,
                    eval_fn=lambda idx, cands, tt=t: _eval_eq_e_candidates(
                        self.dual_basis, a_full[idx], e_full[idx], tt, cands
                    ),
                    a_full=a_full,
                    e_full=e_full,
                    w_full=w_full,
                    dyn_clamped=dyn_clamped,
                    dyn_mask=dyn_mask,
                    x=x_cur,
                    latched_logits=latched_logits,
                    a_locked=a_locked,
                    e_locked=e_locked,
                    w_locked=w_locked,
                    flip_table=flip_table,
                )

            # 3. E_t via Eq W(t+3) (for 1 <= t <= R - 4)
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
                self._attempt_slot_latch(
                    eligible_mask=elig_e,
                    logits=logits,
                    t=t,
                    ch=1,
                    eval_fn=lambda idx, cands, tt=t: _eval_eq_w_for_e_candidates(
                        self.dual_basis, a_full[idx], e_full[idx], w_full[idx], tt, cands
                    ),
                    a_full=a_full,
                    e_full=e_full,
                    w_full=w_full,
                    dyn_clamped=dyn_clamped,
                    dyn_mask=dyn_mask,
                    x=x_cur,
                    latched_logits=latched_logits,
                    a_locked=a_locked,
                    e_locked=e_locked,
                    w_locked=w_locked,
                    flip_table=flip_table,
                )

            # 4. W_t via Eq W(t) (for 0 <= t < R)
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
            self._attempt_slot_latch(
                eligible_mask=elig_w,
                logits=logits,
                t=t,
                ch=2,
                eval_fn=lambda idx, cands, tt=t: _eval_eq_w_for_w_candidates(
                    self.dual_basis, a_full[idx], e_full[idx], tt, cands
                ),
                a_full=a_full,
                e_full=e_full,
                w_full=w_full,
                dyn_clamped=dyn_clamped,
                dyn_mask=dyn_mask,
                x=x_cur,
                latched_logits=latched_logits,
                a_locked=a_locked,
                e_locked=e_locked,
                w_locked=w_locked,
                flip_table=flip_table,
            )

        return dyn_mask * dyn_clamped + (1.0 - dyn_mask) * x_cur

    def _smc_resample(
        self,
        x: torch.Tensor,
        h_prev: Optional[torch.Tensor],
        dyn_clamped: torch.Tensor,
        dyn_mask: torch.Tensor,
        latched_logits: torch.Tensor,
        b_chunk: ClampedBoundary,
        num_digests: int,
        k_per_digest: int,
        temp: float,
        generator: Optional[torch.Generator],
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Parallel SMC resampling: preserve Top-K elites untouched and mutate regenerated offspring."""
        device = x.device
        rounds = b_chunk.rounds
        m = num_digests
        k = k_per_digest
        n_elite = min(self.smc_elite_k, max(1, k // 4))
        n_off = k - n_elite

        x_bin = torch.where(
            x < 0,
            torch.full_like(x, -1.0),
            torch.full_like(x, 1.0),
        )
        x_disc = dyn_mask * dyn_clamped + (1.0 - dyn_mask) * x_bin
        a_full, e_full, w_full = b_chunk.assemble_full(x_disc)
        full_lam = torch.ones(32, dtype=torch.float32, device=device)
        energies, _, _ = self.dual_basis.compute_energy_and_features(
            a_full, e_full, w_full, lam=full_lam
        )
        energies_mk = energies.view(m, k)
        order = torch.argsort(energies_mk, dim=1, descending=False)
        elite_local = order[:, :n_elite]  # (m, n_elite)

        reps = (n_off + n_elite - 1) // n_elite
        off_local = elite_local.repeat(1, reps)[:, :n_off]
        gather_local = torch.cat([elite_local, off_local], dim=1)  # (m, k)
        base_offsets = torch.arange(m, device=device).unsqueeze(1) * k
        flat_idx = (base_offsets + gather_local).reshape(-1)

        x_mk = x[flat_idx].view(m, k, rounds, 3, 32)
        dyn_clamped_new = dyn_clamped[flat_idx]
        dyn_mask_new = dyn_mask[flat_idx]
        latched_logits_new = latched_logits[flat_idx]

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
            (m, n_off, rounds, 3, 32),
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
        flip_sign = torch.where(
            rand_u < flip_off.view(1, n_off, 1, 1, 1), -1.0, 1.0
        )
        x_off = torch.clamp(
            x_mk[:, n_elite:] * flip_sign + sigma_off.view(1, n_off, 1, 1, 1) * eps,
            -1.0,
            1.0,
        )
        dyn_free_mk = (1.0 - dyn_mask_new).view(m, k, rounds, 3, 32)
        dyn_clamped_mk = dyn_clamped_new.view(m, k, rounds, 3, 32)
        x_mk[:, n_elite:] = torch.where(
            dyn_free_mk[:, n_elite:] > 0.5, x_off, dyn_clamped_mk[:, n_elite:]
        )

        x_out = (
            dyn_mask_new * dyn_clamped_new
            + (1.0 - dyn_mask_new) * x_mk.reshape(-1, rounds, 3, 32)
        )

        h_out: Optional[torch.Tensor] = None
        if h_prev is not None:
            d_bit = h_prev.shape[-1]
            h_mk = h_prev[flat_idx].view(m, k, rounds, 32, d_bit)
            h_mk[:, n_elite:] = h_mk[:, n_elite:] * h_decay_off.view(1, n_off, 1, 1, 1)
            h_out = h_mk.reshape(-1, rounds, 32, d_bit)

        return x_out, h_out, dyn_clamped_new, dyn_mask_new, latched_logits_new

    def _compute_dynamic_boundary(
        self, b_chunk: ClampedBoundary, dyn_clamped: torch.Tensor, dyn_mask: torch.Tensor
    ) -> ClampedBoundary:
        """Build lightweight boundary view with updated dynamic clamp masks for dual_basis radar."""
        rounds = b_chunk.rounds
        a_full_clamped = b_chunk.A_full_clamped.clone()
        a_full_mask = b_chunk.A_full_mask.clone()
        e_full_clamped = b_chunk.E_full_clamped.clone()
        e_full_mask = b_chunk.E_full_mask.clone()
        w_full_clamped = b_chunk.W_full_clamped.clone()
        w_full_mask = b_chunk.W_full_mask.clone()

        a_full_clamped[:, 3 : 3 + rounds] = dyn_clamped[:, :, 0, :]
        a_full_mask[:, 3 : 3 + rounds] = dyn_mask[:, :, 0, :]
        e_full_clamped[:, 3 : 3 + rounds] = dyn_clamped[:, :, 1, :]
        e_full_mask[:, 3 : 3 + rounds] = dyn_mask[:, :, 1, :]
        w_full_clamped[:, :rounds] = dyn_clamped[:, :, 2, :]
        w_full_mask[:, :rounds] = dyn_mask[:, :, 2, :]

        return ClampedBoundary(
            rounds=rounds,
            A_full_clamped=a_full_clamped,
            A_full_mask=a_full_mask,
            E_full_clamped=e_full_clamped,
            E_full_mask=e_full_mask,
            W_full_clamped=w_full_clamped,
            W_full_mask=w_full_mask,
            A_words_clamped=b_chunk.A_words_clamped,
            E_words_clamped=b_chunk.E_words_clamped,
            W16_clamped_words=b_chunk.W16_clamped_words,
            W16_mask_bits=b_chunk.W16_mask_bits,
        )

    def infer_v3(
        self,
        b_chunk: ClampedBoundary,
        x_init: torch.Tensor,
        use_neural: bool = True,
        sde_scales: Optional[torch.Tensor] = None,
        num_digests: int = 1,
        k_per_digest: int = 1,
        generator: Optional[torch.Generator] = None,
        return_dyn_mask: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Run K_infer steps with Dynamic Digital Latch, optional closed-loop schedule, and SMC."""
        device = b_chunk.device
        rounds = b_chunk.rounds
        n_traj = b_chunk.batch_size
        k_spins = self.dual_basis.K_spins
        full_lam = torch.ones(32, dtype=torch.float32, device=device)

        static_mask = b_chunk.clamp_mask
        dyn_mask = static_mask.clone()
        dyn_clamped = b_chunk.x_clamped.clone()
        latched_logits = torch.zeros_like(dyn_clamped)

        x = dyn_mask * dyn_clamped + (1.0 - dyn_mask) * x_init
        if self.closed_loop_schedule and rounds > 16:
            x = apply_closed_loop_schedule(x, rounds)
            x = dyn_mask * dyn_clamped + (1.0 - dyn_mask) * x

        h_prev: Optional[torch.Tensor] = None
        last_logits = torch.zeros_like(x)
        denom = float(max(1, self.num_steps - 1))

        for step_idx in range(self.num_steps):
            frac = float(step_idx) / denom
            step_tensor = torch.full((n_traj,), frac, dtype=torch.float32, device=device)
            free_mask = 1.0 - dyn_mask

            if self.use_dynamic_latch and not torch.equal(dyn_mask, static_mask):
                b_active = self._compute_dynamic_boundary(
                    b_chunk, dyn_clamped, dyn_mask
                )
            else:
                b_active = b_chunk

            grad_radar, res_feat, _, _, raw_grad = (
                self.dual_basis.compute_radar_and_features(
                    x,
                    b_active,
                    lam=full_lam,
                    grad_scale=self.grad_scale,
                    return_raw_grad=True,
                )
            )

            if use_neural:
                logits, h_tilde = self.model(
                    x_nodes=x,
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
                x_neural = torch.tanh(logits)
                x_base = (1.0 - self.damp) * x + self.damp * x_neural
            else:
                logits = last_logits
                x_base = x

            x_safe = torch.clamp(x, -2.0, 2.0)
            grad_bin = -4.0 * x_safe * (1.0 - x_safe * x_safe)
            eta_s = self.eta_max * (0.2 + 0.8 * frac)
            beta_s = self.beta_max * (frac * frac)
            guide = torch.clamp(raw_grad + beta_s * grad_bin, -10.0, 10.0)

            temp = max(0.0, 1.0 - frac / 0.80)
            if sde_scales is not None and temp > 0.0:
                eps = torch.randn(
                    x.shape, dtype=torch.float32, device=device, generator=generator
                )
                sde_term = sde_scales.view(-1, 1, 1, 1) * temp * eps * free_mask
            else:
                sde_term = 0.0

            x_next = torch.clamp(
                x_base - eta_s * guide * free_mask + sde_term, -1.0, 1.0
            )
            x = dyn_mask * dyn_clamped + free_mask * x_next

            if self.closed_loop_schedule and rounds > 16:
                x_w_exp = spin_expand_schedule(x[:, :16, 2, :], rounds)[:, 16:rounds, :]
                w_dyn_m = dyn_mask[:, 16:rounds, 2, :]
                w_dyn_c = dyn_clamped[:, 16:rounds, 2, :]
                x[:, 16:rounds, 2, :] = w_dyn_m * w_dyn_c + (1.0 - w_dyn_m) * x_w_exp
                logits_for_latch = logits.clone()
                logits_for_latch[:, 16:rounds, 2, :] = torch.atanh(
                    (0.995 * x_w_exp).clamp(-0.995, 0.995)
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
                x = self._try_dynamic_latch(
                    x=x,
                    logits=logits_for_latch,
                    b_chunk=b_chunk,
                    dyn_clamped=dyn_clamped,
                    dyn_mask=dyn_mask,
                    latched_logits=latched_logits,
                )

            if (
                self.use_smc_resampling
                and k_per_digest >= 16
                and (step_idx + 1) % self.smc_resample_every == 0
                and step_idx < int(0.85 * self.num_steps)
            ):
                smc_temp = max(0.15, 1.0 - frac)
                x, h_prev, dyn_clamped, dyn_mask, latched_logits = self._smc_resample(
                    x=x,
                    h_prev=h_prev,
                    dyn_clamped=dyn_clamped,
                    dyn_mask=dyn_mask,
                    latched_logits=latched_logits,
                    b_chunk=b_chunk,
                    num_digests=num_digests,
                    k_per_digest=k_per_digest,
                    temp=smc_temp,
                    generator=generator,
                )
                if self.closed_loop_schedule and rounds > 16:
                    x_w_exp = spin_expand_schedule(x[:, :16, 2, :], rounds)[:, 16:rounds, :]
                    w_dyn_m = dyn_mask[:, 16:rounds, 2, :]
                    w_dyn_c = dyn_clamped[:, 16:rounds, 2, :]
                    x[:, 16:rounds, 2, :] = w_dyn_m * w_dyn_c + (1.0 - w_dyn_m) * x_w_exp

        dyn_only_bits = torch.clamp(dyn_mask - static_mask, min=0.0)
        orig_free = (1.0 - static_mask).sum(dim=(1, 2, 3)).clamp_min(1.0)
        latch_ratio = dyn_only_bits.sum(dim=(1, 2, 3)) / orig_free
        final_logits = torch.where(
            dyn_only_bits > 0.5, latched_logits, last_logits
        )
        if return_dyn_mask:
            return x, final_logits, latch_ratio, dyn_mask
        return x, final_logits, latch_ratio

    @torch.no_grad()
    def infer_deterministic_zero(
        self,
        boundary: ClampedBoundary,
        use_neural_velocity: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run deterministic zero-init V3 inference over a batch of boundaries."""
        use_neural = (
            self.use_neural_velocity
            if use_neural_velocity is None
            else bool(use_neural_velocity)
        )
        was_training = self.model.training
        self.model.eval()
        try:
            x_zero = boundary.apply_clamp(torch.zeros_like(boundary.x_clamped))
            x_final, final_logits, latch_ratio = self.infer_v3(
                boundary,
                x_zero,
                use_neural=use_neural,
                sde_scales=None,
                num_digests=boundary.batch_size,
                k_per_digest=1,
            )
            x_disc = torch.where(
                x_final < 0,
                torch.full_like(x_final, -1.0),
                torch.full_like(x_final, 1.0),
            )
            x_disc = boundary.apply_clamp(x_disc)
            a_full, e_full, w_full = boundary.assemble_full(x_disc)
            full_lam = torch.ones(32, dtype=torch.float32, device=boundary.device)
            _, _, stats = self.dual_basis.compute_energy_and_features(
                a_full, e_full, w_full, lam=full_lam
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
        )

    @torch.no_grad()
    def predict_calibrated_bit_probs(
        self,
        target_digests: torch.Tensor,
        domain_mode: str = "int_range",
        num_bits: int = 32,
        msg_bytes_len: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """Export calibrated P(bit=1) probabilities and verified latch masks for V4D SAT hybrids."""
        if target_digests.ndim == 1:
            target_digests = target_digests.unsqueeze(0)
        boundary = build_clamped_boundary(
            digest=target_digests,
            rounds=self.dual_basis.rounds,
            domain_mode=domain_mode,
            num_bits=num_bits,
            msg_bytes_len=msg_bytes_len,
        )
        was_training = self.model.training
        self.model.eval()
        try:
            x_zero = boundary.apply_clamp(torch.zeros_like(boundary.x_clamped))
            _, final_logits, latch_ratio, dyn_mask = self.infer_v3(
                boundary,
                x_zero,
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

    def _evaluate_chunk(
        self,
        b_chunk: ClampedBoundary,
        x_final: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        rounds = b_chunk.rounds
        full_lam = torch.ones(32, dtype=torch.float32, device=b_chunk.device)
        cand_w16 = b_chunk.extract_candidate_w16(x_final)
        cand_digest = sha256_digest(cand_w16, rounds=rounds)

        x_disc = torch.where(
            x_final < 0,
            torch.full_like(x_final, -1.0),
            torch.full_like(x_final, 1.0),
        )
        x_disc = b_chunk.apply_clamp(x_disc)
        a_full, e_full, w_full = b_chunk.assemble_full(x_disc)
        energies, _, stats = self.dual_basis.compute_energy_and_features(
            a_full, e_full, w_full, lam=full_lam
        )
        return (
            cand_w16,
            cand_digest,
            x_disc,
            energies,
            stats["m1_energy"],
            stats["m32_energy"],
        )

    @torch.no_grad()
    def solve_batch_boundaries(
        self,
        boundaries_batch: ClampedBoundary,
        target_digests_batch: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        use_neural_velocity: Optional[bool] = None,
    ) -> tuple[torch.Tensor, ...]:
        """Solve M digests in parallel using M*K trajectories with joint SMC resampling."""
        use_neural = (
            self.use_neural_velocity
            if use_neural_velocity is None
            else bool(use_neural_velocity)
        )
        m = boundaries_batch.batch_size
        k = self.num_candidates
        rounds = boundaries_batch.rounds
        device = target_digests_batch.device

        was_training = self.model.training
        self.model.eval()
        try:
            b_exp = boundaries_batch.repeat_for_candidates(k)
            sigmas_mk, sde_mk = self._build_sigma_and_sde(m, k, device)
            x0 = self._init_diverse_states(
                b_exp, sigmas_flat=sigmas_mk.reshape(-1), generator=generator
            )
            x_final, _, latch_ratios = self.infer_v3(
                b_exp,
                x0,
                use_neural=use_neural,
                sde_scales=sde_mk.reshape(-1),
                num_digests=m,
                k_per_digest=k,
                generator=generator,
            )
            cand_w16, cand_digest, x_disc, energies, m1_e, m32_e = (
                self._evaluate_chunk(b_exp, x_final)
            )
        finally:
            if was_training:
                self.model.train()

        cand_w16_mk = cand_w16.view(m, k, 16)
        cand_digest_mk = cand_digest.view(m, k, 8)
        x_disc_mk = x_disc.view(m, k, rounds, 3, 32)
        energies_mk = energies.view(m, k)
        m1_mk = m1_e.view(m, k)
        m32_mk = m32_e.view(m, k)
        latch_mk = latch_ratios.view(m, k)

        matches = (cand_digest_mk == target_digests_batch.unsqueeze(1)).all(dim=-1)
        hits = matches.sum(dim=-1)
        solved = hits > 0

        min_indices = torch.argmin(energies_mk, dim=-1)
        hit_indices = torch.argmax(matches.to(torch.int64), dim=-1)
        chosen_idx = torch.where(solved, hit_indices, min_indices)

        batch_idx = torch.arange(m, device=device)
        best_w16 = cand_w16_mk[batch_idx, chosen_idx]
        best_energy = energies_mk[batch_idx, chosen_idx]
        best_m1 = m1_mk[batch_idx, chosen_idx]
        best_m32 = m32_mk[batch_idx, chosen_idx]
        min_m32 = m32_mk.min(dim=-1).values
        best_x_disc = x_disc_mk[batch_idx, chosen_idx]
        best_latch = latch_mk[batch_idx, chosen_idx]

        unique_ratios = torch.tensor(
            [
                float(torch.unique(cand_w16_mk[i, :, 0]).numel()) / float(k)
                for i in range(m)
            ],
            dtype=torch.float32,
            device=device,
        )

        return (
            solved,
            hits,
            unique_ratios,
            best_latch,
            best_w16,
            best_energy,
            best_m1,
            best_m32,
            min_m32,
            best_x_disc,
        )

    @torch.no_grad()
    def solve_single_boundary(
        self,
        boundary_one: ClampedBoundary,
        target_digest_one: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_trajectories: bool = False,
        use_neural_velocity: Optional[bool] = None,
    ) -> tuple[
        bool,
        int,
        float,
        float,
        torch.Tensor,
        float,
        float,
        float,
        float,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Solve a single digest (chunking K if chunk_size < K)."""
        use_neural = (
            self.use_neural_velocity
            if use_neural_velocity is None
            else bool(use_neural_velocity)
        )
        k_total = self.num_candidates
        chunk = max(1, min(self.chunk_size, k_total))
        device = boundary_one.device
        sigmas_all, sde_all = self._build_sigma_and_sde(1, k_total, device)
        sigmas_1d = sigmas_all[0]
        sde_1d = sde_all[0]

        was_training = self.model.training
        self.model.eval()

        best_energy_val = float("inf")
        best_m1_val = float("inf")
        best_m32_val = float("inf")
        min_m32_val = float("inf")
        best_latch_val = 0.0
        best_w16 = boundary_one.W16_clamped_words[0].clone()
        best_x_disc = boundary_one.x_clamped[0].clone()
        total_hits = 0
        found_hit = False
        all_w0 = []
        last_x_chunk: Optional[torch.Tensor] = None

        try:
            start = 0
            while start < k_total:
                cur_k = min(chunk, k_total - start)
                sig_sub = sigmas_1d[start : start + cur_k]
                sde_sub = sde_1d[start : start + cur_k]
                start += cur_k
                b_chunk = boundary_one.repeat_for_candidates(cur_k)
                x0 = self._init_diverse_states(
                    b_chunk, sigmas_flat=sig_sub, generator=generator
                )
                x_final, _, latch_sub = self.infer_v3(
                    b_chunk,
                    x0,
                    use_neural=use_neural,
                    sde_scales=sde_sub,
                    num_digests=1,
                    k_per_digest=cur_k,
                    generator=generator,
                )
                cand_w16, cand_digest, x_disc, energies, m1_e, m32_e = (
                    self._evaluate_chunk(b_chunk, x_final)
                )
                all_w0.append(cand_w16[:, 0])
                chunk_min_m32 = float(m32_e.min().item())
                if chunk_min_m32 < min_m32_val:
                    min_m32_val = chunk_min_m32

                matches = (cand_digest == target_digest_one.unsqueeze(0)).all(dim=-1)
                hits = int(matches.sum().item())
                total_hits += hits

                if hits > 0 and not found_hit:
                    first_hit = int(torch.nonzero(matches, as_tuple=False)[0].item())
                    best_w16 = cand_w16[first_hit].clone()
                    best_x_disc = x_disc[first_hit].clone()
                    best_energy_val = float(energies[first_hit].item())
                    best_m1_val = float(m1_e[first_hit].item())
                    best_m32_val = float(m32_e[first_hit].item())
                    best_latch_val = float(latch_sub[first_hit].item())
                    found_hit = True
                elif not found_hit:
                    min_idx = int(torch.argmin(energies).item())
                    min_e = float(energies[min_idx].item())
                    if min_e < best_energy_val:
                        best_energy_val = min_e
                        best_m1_val = float(m1_e[min_idx].item())
                        best_m32_val = float(m32_e[min_idx].item())
                        best_latch_val = float(latch_sub[min_idx].item())
                        best_w16 = cand_w16[min_idx].clone()
                        best_x_disc = x_disc[min_idx].clone()

                if return_trajectories:
                    last_x_chunk = x_final
        finally:
            if was_training:
                self.model.train()

        unique_ratio = float(torch.unique(torch.cat(all_w0, dim=0)).numel()) / float(
            k_total
        )
        return (
            total_hits > 0,
            total_hits,
            unique_ratio,
            best_latch_val,
            best_w16,
            best_energy_val,
            best_m1_val,
            best_m32_val,
            min_m32_val,
            best_x_disc,
            last_x_chunk,
        )

    def solve(
        self,
        target_digests: torch.Tensor,
        domain_mode: str = "int_range",
        num_bits: int = 32,
        msg_bytes_len: int = 4,
        generator: Optional[torch.Generator] = None,
        return_last_trajectories: bool = False,
        use_neural_velocity: Optional[bool] = None,
    ) -> SolveResultV3:
        """Solve preimage candidates for a batch of digests (B, 8) or (8,)."""
        if target_digests.ndim == 1:
            target_digests = target_digests.unsqueeze(0)
        bsz = target_digests.shape[0]
        device = target_digests.device
        rounds = self.dual_basis.rounds
        m_per_batch = max(1, self.chunk_size // self.num_candidates)

        if m_per_batch >= 2 and not return_last_trajectories and bsz > 1:
            solved_l, hits_l, uniq_l, latch_l, w16_l, e_l, m1_l, m32_l, min_m32_l, x_l = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for start in range(0, bsz, m_per_batch):
                dig_sub = target_digests[start : start + m_per_batch]
                b_sub = build_clamped_boundary(
                    digest=dig_sub,
                    rounds=rounds,
                    domain_mode=domain_mode,
                    num_bits=num_bits,
                    msg_bytes_len=msg_bytes_len,
                )
                s_i, h_i, u_i, l_i, w_i, e_i, m1_i, m32_i, mm32_i, x_i = (
                    self.solve_batch_boundaries(
                        b_sub,
                        dig_sub,
                        generator=generator,
                        use_neural_velocity=use_neural_velocity,
                    )
                )
                solved_l.append(s_i)
                hits_l.append(h_i)
                uniq_l.append(u_i)
                latch_l.append(l_i)
                w16_l.append(w_i)
                e_l.append(e_i)
                m1_l.append(m1_i)
                m32_l.append(m32_i)
                min_m32_l.append(mm32_i)
                x_l.append(x_i)

            return SolveResultV3(
                solved=torch.cat(solved_l, dim=0),
                num_hits=torch.cat(hits_l, dim=0),
                unique_ratio=torch.cat(uniq_l, dim=0),
                latch_ratio=torch.cat(latch_l, dim=0),
                best_w16=torch.cat(w16_l, dim=0),
                best_energy=torch.cat(e_l, dim=0),
                best_m1_energy=torch.cat(m1_l, dim=0),
                best_m32_energy=torch.cat(m32_l, dim=0),
                min_m32_energy=torch.cat(min_m32_l, dim=0),
                best_x_disc=torch.cat(x_l, dim=0),
                final_x_sample=None,
            )

        solved_l, hits_l, uniq_l, latch_l, w16_l, e_l, m1_l, m32_l, min_m32_l, x_l = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        last_traj: Optional[torch.Tensor] = None
        for i in range(bsz):
            dig_i = target_digests[i]
            b_i = build_clamped_boundary(
                digest=dig_i,
                rounds=rounds,
                domain_mode=domain_mode,
                num_bits=num_bits,
                msg_bytes_len=msg_bytes_len,
            )
            s_i, h_i, u_i, l_i, w_i, e_i, m1_i, m32_i, mm32_i, x_i, traj_i = (
                self.solve_single_boundary(
                    b_i,
                    dig_i,
                    generator=generator,
                    return_trajectories=(return_last_trajectories and i == bsz - 1),
                    use_neural_velocity=use_neural_velocity,
                )
            )
            solved_l.append(s_i)
            hits_l.append(h_i)
            uniq_l.append(u_i)
            latch_l.append(l_i)
            w16_l.append(w_i)
            e_l.append(e_i)
            m1_l.append(m1_i)
            m32_l.append(m32_i)
            min_m32_l.append(mm32_i)
            x_l.append(x_i)
            if traj_i is not None:
                last_traj = traj_i

        return SolveResultV3(
            solved=torch.tensor(solved_l, dtype=torch.bool, device=device),
            num_hits=torch.tensor(hits_l, dtype=torch.int64, device=device),
            unique_ratio=torch.tensor(uniq_l, dtype=torch.float32, device=device),
            latch_ratio=torch.tensor(latch_l, dtype=torch.float32, device=device),
            best_w16=torch.stack(w16_l, dim=0),
            best_energy=torch.tensor(e_l, dtype=torch.float32, device=device),
            best_m1_energy=torch.tensor(m1_l, dtype=torch.float32, device=device),
            best_m32_energy=torch.tensor(m32_l, dtype=torch.float32, device=device),
            min_m32_energy=torch.tensor(min_m32_l, dtype=torch.float32, device=device),
            best_x_disc=torch.stack(x_l, dim=0),
            final_x_sample=last_traj,
        )
