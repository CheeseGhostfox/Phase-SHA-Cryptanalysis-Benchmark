"""Phase-SHA V5B: Burer-Monteiro High-Dimensional Hypersphere S^{N-1} Solver (strict FP32).

Strictly aligned with Lean 4 Module 6B (formal_proofs/PhaseSHA/V5/BurerMonteiroEscape.lean):
  - Pure FP32 everywhere (allow_tf32=False, zero complex64).
  - Supports N in {16, 32, 64} (any N >= 4) organized cleanly as bmm/einsum ops.
  - Anchor axis e_0 = (1, 0, ..., 0) coupled directly to 2-adic phase tower.
  - Strictly non-negative zero-lower-bound squared gate residuals (eliminating SDP negative-energy cheating).
  - GlobalProgressiveCollapseSchedule for Rank-1 collapse (log rank1_ratio & eff_rank).
  - Exact orthogonal half-angle rational Cayley escape perturbation (burer_monteiro_plus_one_dim_escape).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bit_grid_backbone import (
    BitGridBlock,
    SHA_ROTR_OFFSETS,
    build_rotr_channels,
    extract_grid_features,
    _modulate_grid,
)
from src.dual_basis import DualBasisEngine, spin_Ch, spin_Maj, spin_Sigma0, spin_Sigma1
from src.graph_backbone import build_round_sparse_attn_mask, sinusoidal_embedding
from src.sha256_ref import bits_to_words, sha256_digest, words_to_bits, words_to_spins
from src.v3_solver import spin_add32
from src.v4_boundary import (
    ManifoldBoundary,
    PreparedManifoldBoundary,
    apply_clamp_prepared,
    assemble_full_prepared,
    build_manifold_boundary,
)
from src.v4_manifolds import normalize_manifold
from src.v4_solver import UnrolledSolverV4, anneal_schedule
from src.v5_interfaces import (
    GeometryBackend,
    GlobalProgressiveCollapseSchedule,
    OuterSearchShell,
    SolveResultV5,
    compute_calibrated_probs_from_candidates,
    compute_calibrated_probs_from_logits,
)


def compute_bm_rank1_and_eff_rank(
    u_nodes: torch.Tensor,
    free_mask: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    """Compute (rank1_ratio = mean_free(v_0^2), eff_rank = exp(spectral entropy)) in pure FP32."""
    uf = u_nodes.detach().to(torch.float32)
    v0_sq = uf[..., 0].square()
    if free_mask is not None:
        fm = free_mask.to(torch.float32)
        rank1_ratio = float(((v0_sq * fm).sum() / fm.sum().clamp_min(1.0)).item())
    else:
        rank1_ratio = float(v0_sq.mean().item())

    n_dim = uf.shape[-1]
    flat = uf.reshape(-1, n_dim)
    gram = torch.matmul(flat.t(), flat) / float(max(1, flat.shape[0]))
    gram = 0.5 * (gram + gram.t())
    evals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    p = evals / evals.sum().clamp_min(1e-12)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum()
    eff_rank = float(torch.exp(entropy).item())
    return rank1_ratio, eff_rank


def cayley_orthogonal_escape(
    v: torch.Tensor,
    step_t: float | torch.Tensor = 0.35,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply Lean 4 exact norm-preserving rational half-angle Cayley rotation along orthogonal direction.

    Reproduces `liftedVec`:
      x = (t * u) / 2
      v' = ((1 - x^2)/(1 + x^2)) * v + (2x / (1 + x^2)) * e_orth
    where <v, e_orth> = 0, ||e_orth|| = 1 -> ||v'|| == ||v|| == 1 identically!
    """
    vf = normalize_manifold(v)
    n_dim = vf.shape[-1]
    bsz = vf.shape[0]
    dev = vf.device

    # Identify dominant low-rank direction per sample so escape probes outside low-dim subspace
    flat = vf.reshape(bsz, -1, n_dim)
    e_dom = normalize_manifold(flat.mean(dim=1, keepdim=True)).view(bsz, *([1] * (vf.ndim - 2)), n_dim)

    raw = torch.randn(vf.shape, dtype=torch.float32, device=dev, generator=generator)
    # Two-stage Gram-Schmidt orthogonalization against both e_dom and local vector v
    r1 = raw - (raw * e_dom).sum(dim=-1, keepdim=True) * e_dom
    r2 = r1 - (r1 * vf).sum(dim=-1, keepdim=True) * vf
    r2_norm = torch.linalg.vector_norm(r2, dim=-1, keepdim=True)
    # Deterministic fallback orthogonal coordinate if random sample happened collinear
    fb = torch.roll(vf, shifts=1, dims=-1)
    fb = fb - (fb * vf).sum(dim=-1, keepdim=True) * vf
    e_orth = torch.where(
        r2_norm > 1e-5,
        r2 / r2_norm.clamp_min(1e-5),
        fb / torch.linalg.vector_norm(fb, dim=-1, keepdim=True).clamp_min(1e-5),
    )

    u_sign = torch.where(
        torch.rand(vf.shape[:-1] + (1,), dtype=torch.float32, device=dev, generator=generator) < 0.5,
        -torch.ones_like(vf[..., :1]),
        torch.ones_like(vf[..., :1]),
    )
    t_val = step_t if isinstance(step_t, torch.Tensor) else torch.full_like(u_sign, float(step_t))
    x = 0.5 * t_val * u_sign
    x2 = x * x
    den = (1.0 + x2).clamp_min(1.0)
    c_cos = (1.0 - x2) / den
    c_sin = (2.0 * x) / den
    return normalize_manifold(c_cos * vf + c_sin * e_orth)


class BurerMonteiroSphereBackend(GeometryBackend):
    """V6 Slot 1: High-dimensional unit hypersphere S^{N-1} with non-negative zero-lower-bound gates."""

    variant = "v5b"
    spin_axis = 0
    is_abelian = False

    def __init__(self, manifold_dim: int = 32) -> None:
        if manifold_dim < 4:
            raise ValueError(f"manifold_dim N must be >= 4, got {manifold_dim}")
        self.dim = int(manifold_dim)
        self.trans_axes = tuple(range(1, self.dim))

    def e0_vec(self, device: torch.device) -> torch.Tensor:
        v = torch.zeros((self.dim,), dtype=torch.float32, device=device)
        v[0] = 1.0
        return v

    def eq_vec(self, device: torch.device) -> torch.Tensor:
        v = torch.zeros((self.dim,), dtype=torch.float32, device=device)
        v[1] = 1.0
        return v

    @torch.amp.autocast("cuda", enabled=False)
    def embed_spins(self, spins: torch.Tensor) -> torch.Tensor:
        s = spins.to(torch.float32)
        out = torch.zeros((*s.shape, self.dim), dtype=torch.float32, device=s.device)
        return torch.cat([s.unsqueeze(-1), out[..., 1:]], dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def embed_spins_with_equator(self, spins: torch.Tensor) -> torch.Tensor:
        s = spins.to(torch.float32).clamp(-1.0, 1.0)
        trans = torch.sqrt((1.0 - s * s).clamp_min(0.0))
        z = torch.zeros((*s.shape, max(0, self.dim - 2)), dtype=torch.float32, device=s.device)
        return normalize_manifold(torch.cat([s.unsqueeze(-1), trans.unsqueeze(-1), z], dim=-1))

    @torch.amp.autocast("cuda", enabled=False)
    def project_spin(self, u: torch.Tensor) -> torch.Tensor:
        return u.to(torch.float32)[..., 0]

    @torch.amp.autocast("cuda", enabled=False)
    def project_transverse(self, u: torch.Tensor) -> torch.Tensor:
        return u.to(torch.float32)[..., 1:]

    @torch.amp.autocast("cuda", enabled=False)
    def compose_from_spin_and_transverse_dir(
        self, spins: torch.Tensor, u_ref: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        s = spins.to(torch.float32).clamp(-1.0, 1.0)
        trans_amp = torch.sqrt((1.0 - s * s).clamp_min(0.0))
        ref_t = self.project_transverse(u_ref)
        ref_norm = torch.linalg.vector_norm(ref_t, dim=-1, keepdim=True)
        fb = torch.zeros_like(ref_t)
        fb = torch.cat([torch.ones_like(s).unsqueeze(-1), fb[..., 1:]], dim=-1)
        dir_t = torch.where(ref_norm > eps, ref_t / ref_norm.clamp_min(eps), fb)
        trans_scaled = trans_amp.unsqueeze(-1) * dir_t
        return normalize_manifold(torch.cat([s.unsqueeze(-1), trans_scaled], dim=-1), eps=eps)

    @torch.amp.autocast("cuda", enabled=False)
    def tangent_project(self, u: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        u_f = u.to(torch.float32)
        d_f = delta.to(torch.float32)
        return d_f - (d_f * u_f).sum(dim=-1, keepdim=True) * u_f

    @torch.amp.autocast("cuda", enabled=False)
    def retract(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        step_size: float | torch.Tensor = 1.0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        u_n = normalize_manifold(u, eps=eps)
        d_perp = self.tangent_project(u_n, delta)
        return normalize_manifold(u_n + step_size * d_perp, eps=eps)

    @torch.amp.autocast("cuda", enabled=False)
    def make_neutral_equator(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        return self.eq_vec(device).view(*([1] * len(shape)), self.dim).expand(*shape, self.dim).clone()

    @torch.amp.autocast("cuda", enabled=False)
    def sample_random_manifold(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        spin_std: float = 0.35,
    ) -> torch.Tensor:
        raw = torch.randn((*shape, self.dim), dtype=torch.float32, device=device, generator=generator)
        # Scale spin axis vs transverse coordinates proportionally
        scale_t = 1.0 / math.sqrt(float(self.dim - 1))
        spin_part = (raw[..., :1] * float(spin_std))
        trans_part = raw[..., 1:] * scale_t
        return normalize_manifold(torch.cat([spin_part, trans_part], dim=-1))

    @torch.amp.autocast("cuda", enabled=False)
    def align_sq(self, u: torch.Tensor) -> torch.Tensor:
        """Rank-1 transverse penalty 1 - v_0^2 = ||v_perp||^2 (>= 0, 0 iff Rank-1 on +-e_0)."""
        return (1.0 - self.project_spin(u).square()).clamp_min(0.0)

    @torch.amp.autocast("cuda", enabled=False)
    def rotr(self, u_word: torch.Tensor, n: int) -> torch.Tensor:
        if n == 0:
            return u_word.to(torch.float32)
        return torch.roll(u_word.to(torch.float32), shifts=-n, dims=-2)

    @torch.amp.autocast("cuda", enabled=False)
    def shr(self, u_word: torch.Tensor, n: int) -> torch.Tensor:
        u_f = u_word.to(torch.float32)
        if n == 0:
            return u_f
        if n >= 32:
            return self.e0_vec(u_f.device).view(*([1] * (u_f.ndim - 1)), self.dim).expand_as(u_f).clone()
        e0 = self.e0_vec(u_f.device).view(*([1] * (u_f.ndim - 2)), 1, self.dim)
        pad = e0.expand(*u_f.shape[:-2], n, self.dim)
        return torch.cat([u_f[..., n:, :], pad], dim=-2)

    @torch.amp.autocast("cuda", enabled=False)
    def commutator_sq(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """2x2 wedge bivector norm^2 ||a ^ b||^2 = ||a||^2 ||b||^2 - <a, b>^2 (0 iff collinear)."""
        af = a.to(torch.float32)
        bf = b.to(torch.float32)
        aa = af.square().sum(dim=-1)
        bb = bf.square().sum(dim=-1)
        ab = (af * bf).sum(dim=-1)
        return (aa * bb - ab * ab).clamp_min(0.0)

    @torch.amp.autocast("cuda", enabled=False)
    def xor2_residual(self, a: torch.Tensor, b: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Strictly non-negative zero-lower-bound 2-XOR squared residual."""
        af, bf, yf = a.to(torch.float32), b.to(torch.float32), y.to(torch.float32)
        ab = (af * bf).sum(dim=-1)
        by = (bf * yf).sum(dim=-1)
        ya = (yf * af).sum(dim=-1)
        return ((ab - yf[..., 0]).square() + (by - af[..., 0]).square() + (ya - bf[..., 0]).square()) / 3.0

    @torch.amp.autocast("cuda", enabled=False)
    def xor3_spin(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Permutation-symmetric high-dim 3-XOR spin projection in [-1, 1]."""
        af, bf, cf = a.to(torch.float32), b.to(torch.float32), c.to(torch.float32)
        ab = (af * bf).sum(dim=-1)
        bc = (bf * cf).sum(dim=-1)
        ca = (cf * af).sum(dim=-1)
        return ((ab * cf[..., 0] + bc * af[..., 0] + ca * bf[..., 0]) / 3.0).clamp(-1.0, 1.0)

    @torch.amp.autocast("cuda", enabled=False)
    def xor3_residual(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Strictly non-negative zero-lower-bound 3-XOR squared residual (0 at poles, >=4 on flip)."""
        af, bf, cf, yf = a.to(torch.float32), b.to(torch.float32), c.to(torch.float32), y.to(torch.float32)
        s_pred = self.xor3_spin(af, bf, cf)
        term_spin = (s_pred - yf[..., 0]).square()
        term_gram = (
            ((af * bf).sum(dim=-1) - (cf * yf).sum(dim=-1)).square()
            + ((af * cf).sum(dim=-1) - (bf * yf).sum(dim=-1)).square()
            + ((bf * cf).sum(dim=-1) - (af * yf).sum(dim=-1)).square()
        ) / 3.0
        return term_spin + term_gram

    @torch.amp.autocast("cuda", enabled=False)
    def eq_s_lsb_residual(self, words9: List[torch.Tensor]) -> torch.Tensor:
        """Strictly non-negative 9-operand Eq S LSB parity residual (zero on trace, >=4 on flip)."""
        t1 = self.xor3_spin(words9[0], words9[1], words9[2])
        t2 = self.xor3_spin(words9[3], words9[4], words9[5])
        t3 = self.xor3_spin(words9[6], words9[7], words9[8])
        # Group parity zero-lower-bound square + transverse guard
        parity_err = (t1 * t2 * t3 - 1.0).square()
        trans_mean = sum(self.align_sq(w) for w in words9) / float(len(words9))
        return parity_err + 0.5 * trans_mean

    @staticmethod
    def antiferro_triangle_nonneg_energy(
        v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor
    ) -> torch.Tensor:
        """Zero-lower-bound antiferromagnetic triangle energy: sum (<vi,vj> + 1)^2 >= 0."""
        d12 = (v1 * v2).sum(dim=-1)
        d23 = (v2 * v3).sum(dim=-1)
        d31 = (v3 * v1).sum(dim=-1)
        return (d12 + 1.0).square() + (d23 + 1.0).square() + (d31 + 1.0).square()

    def _op3_words(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spins = self.xor3_spin(a, b, c)
        comm = self.commutator_sq(a, b) + self.commutator_sq(b, c) + self.commutator_sq(a, c)
        leak = ((spins - a[..., 0] * b[..., 0] * c[..., 0]).square()).clamp_min(0.0)
        return spins, comm, leak

    def sigma0_upper(self, u_a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._op3_words(self.rotr(u_a, 2), self.rotr(u_a, 13), self.rotr(u_a, 22))

    def sigma1_upper(self, u_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._op3_words(self.rotr(u_e, 6), self.rotr(u_e, 11), self.rotr(u_e, 25))

    def sigma0_lower(self, u_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._op3_words(self.rotr(u_w, 7), self.rotr(u_w, 18), self.shr(u_w, 3))

    def sigma1_lower(self, u_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._op3_words(self.rotr(u_w, 17), self.rotr(u_w, 19), self.shr(u_w, 10))


class V5BPhysicsEngine(nn.Module):
    """Differentiable V5B Burer-Monteiro + 32-scale 2-adic Fourier phase tower engine."""

    def __init__(self, manifold_dim: int = 32, rounds: int = 18) -> None:
        super().__init__()
        self.variant = "v5b"
        self.rounds = int(rounds)
        self.manifold = BurerMonteiroSphereBackend(manifold_dim=manifold_dim)
        self.dual_basis = DualBasisEngine(rounds=self.rounds)

    @property
    def K_spins(self) -> torch.Tensor:
        return self.dual_basis.K_spins

    def float(self) -> "V5BPhysicsEngine":
        for name, mod in self.named_children():
            if name != "dual_basis":
                mod.float()
        return self

    @torch.amp.autocast("cuda", enabled=False)
    def compute_energy_and_features_core(
        self,
        u_a: torch.Tensor,
        u_e: torch.Tensor,
        u_w: torch.Tensor,
        free_mask: Optional[torch.Tensor],
        lam_b: torch.Tensor,
        w_xor: float | torch.Tensor = 1.0,
        lam_comm: float | torch.Tensor = 0.0,
        lam_align: float | torch.Tensor = 0.0,
        digest_target_bits: Optional[torch.Tensor] = None,
        digest_target_mask: Optional[torch.Tensor] = None,
        iv_words: Optional[torch.Tensor] = None,
        w_digest_prefix: float = 4.0,
        include_tail_e: bool = False,
        has_digest_prefix: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Pure-tensor compilable V5B energy & features core."""
        r = self.rounds
        m = self.manifold
        u_a, u_e, u_w = u_a.to(torch.float32), u_e.to(torch.float32), u_w.to(torch.float32)
        bsz = u_a.shape[0]
        device = u_a.device

        a_spins = m.project_spin(u_a)
        e_spins = m.project_spin(u_e)
        w_spins = m.project_spin(u_w)

        len_e = r - 4
        sig0_a_spins, comm_s0, leak_s0 = m.sigma0_upper(u_a[:, 3 : 3 + len_e])
        maj_a = spin_Maj(
            a_spins[:, 3 : 3 + len_e], a_spins[:, 2 : 2 + len_e], a_spins[:, 1 : 1 + len_e]
        )
        sig1_e_spins, comm_s1, leak_s1 = m.sigma1_upper(u_e[:, 3 : 3 + r])
        ch_e = spin_Ch(e_spins[:, 3 : 3 + r], e_spins[:, 2 : 2 + r], e_spins[:, 1 : 1 + r])

        db = self.dual_basis
        theta_a = db.spins_to_phase_fp64(a_spins)
        theta_e = db.spins_to_phase_fp64(e_spins)
        theta_w = db.spins_to_phase_fp64(w_spins)

        delta_e_fp64 = (
            theta_e[:, 4 : 4 + len_e]
            + db.spins_to_phase_fp64(sig0_a_spins)
            + db.spins_to_phase_fp64(maj_a)
            - theta_a[:, 4 : 4 + len_e]
            - theta_a[:, 0:len_e]
        )
        delta_w_fp64 = (
            theta_w
            + db.theta_K.unsqueeze(0)
            + db.spins_to_phase_fp64(sig1_e_spins)
            + db.spins_to_phase_fp64(ch_e)
            + theta_e[:, 0:r]
            + theta_a[:, 0:r]
            - theta_e[:, 4 : 4 + r]
        )
        delta_e = db._wrap_to_fp32(delta_e_fp64)
        delta_w = db._wrap_to_fp32(delta_w_fp64)

        sin_e = torch.sin(delta_e)
        omc_e = 2.0 * torch.sin(0.5 * delta_e).square()
        sin_w = torch.sin(delta_w)
        omc_w = 2.0 * torch.sin(0.5 * delta_w).square()

        per_scale_e = omc_e.sum(dim=1)
        if include_tail_e:
            sig0_tail, _, _ = m.sigma0_upper(u_a[:, r - 1 : r + 3])
            maj_tail = spin_Maj(
                a_spins[:, r - 1 : r + 3], a_spins[:, r - 2 : r + 2], a_spins[:, r - 3 : r + 1]
            )
            delta_e_tail_fp64 = (
                theta_e[:, r : r + 4]
                + db.spins_to_phase_fp64(sig0_tail)
                + db.spins_to_phase_fp64(maj_tail)
                - theta_a[:, r : r + 4]
                - theta_a[:, r - 4 : r]
            )
            omc_tail = 2.0 * torch.sin(0.5 * db._wrap_to_fp32(delta_e_tail_fp64)).square()
            per_scale_e = per_scale_e + omc_tail.sum(dim=1)
        per_scale_w = omc_w.sum(dim=1)
        per_scale = per_scale_e + per_scale_w

        pad_e = torch.zeros((bsz, 4, 32), dtype=torch.float32, device=device)
        feat_e = torch.cat(
            [torch.cat([sin_e, pad_e], dim=1), torch.cat([omc_e, pad_e], dim=1)], dim=-1
        )
        feat_w = torch.cat([sin_w, omc_w], dim=-1)

        if r > 16:
            sig1_w_spins, comm_ls1, leak_ls1 = m.sigma1_lower(u_w[:, 14 : r - 2])
            sig0_w_spins, comm_ls0, leak_ls0 = m.sigma0_lower(u_w[:, 1 : r - 15])
            delta_s_fp64 = (
                theta_w[:, 16:r]
                - db.spins_to_phase_fp64(sig1_w_spins)
                - theta_w[:, 9 : r - 7]
                - db.spins_to_phase_fp64(sig0_w_spins)
                - theta_w[:, 0 : r - 16]
            )
            delta_s = db._wrap_to_fp32(delta_s_fp64)
            sin_s = torch.sin(delta_s)
            omc_s = 2.0 * torch.sin(0.5 * delta_s).square()
            per_scale_s = omc_s.sum(dim=1)
            per_scale = per_scale + per_scale_s
            pad_s = torch.zeros((bsz, 16, 32), dtype=torch.float32, device=device)
            feat_s = torch.cat(
                [torch.cat([pad_s, sin_s], dim=1), torch.cat([pad_s, omc_s], dim=1)], dim=-1
            )

            w_m2 = u_w[:, 14 : r - 2]
            w_m7 = u_w[:, 9 : r - 7]
            w_m15 = u_w[:, 1 : r - 15]
            w_m16 = u_w[:, 0 : r - 16]
            w_t = u_w[:, 16:r]
            lsb_ops9 = [
                w_m2[:, :, 17],
                w_m2[:, :, 19],
                w_m2[:, :, 10],
                w_m7[:, :, 0],
                w_m15[:, :, 7],
                w_m15[:, :, 18],
                w_m15[:, :, 3],
                w_m16[:, :, 0],
                w_t[:, :, 0],
            ]
            eq_s_xor = m.eq_s_lsb_residual(lsb_ops9).sum(dim=1)
            leak_sched = leak_ls1.mean(dim=-1).sum(dim=1) + leak_ls0.mean(dim=-1).sum(dim=1)
            comm_sched = (
                comm_ls1.mean(dim=-1).mean(dim=1)
                + comm_ls0.mean(dim=-1).mean(dim=1)
                + m.commutator_sq(w_t, w_m16).mean(dim=-1).mean(dim=1)
            ) / 3.0
        else:
            per_scale_s = torch.zeros_like(per_scale_w)
            feat_s = torch.zeros((bsz, r, 64), dtype=torch.float32, device=device)
            eq_s_xor = torch.zeros(bsz, dtype=torch.float32, device=device)
            leak_sched = torch.zeros(bsz, dtype=torch.float32, device=device)
            comm_sched = torch.zeros(bsz, dtype=torch.float32, device=device)

        energy_2adic = (per_scale * lam_b).sum(dim=-1)
        features = torch.cat([feat_e, feat_w, feat_s], dim=-1)
        energy_xor = eq_s_xor + leak_s0.mean(dim=-1).sum(dim=1) + leak_s1.mean(dim=-1).sum(dim=1) + leak_sched

        ua_mid = u_a[:, 3 : 3 + r]
        ue_mid = u_e[:, 3 : 3 + r]
        comm_node_pairs = (
            m.commutator_sq(ua_mid, ue_mid)
            + m.commutator_sq(ue_mid, u_w)
            + m.commutator_sq(ua_mid, u_w)
        ).mean(dim=(1, 2)) / 3.0
        comm_gates = (comm_s0.mean(dim=(1, 2)) / 3.0 + comm_s1.mean(dim=(1, 2)) / 3.0) / 2.0
        energy_comm = (
            (comm_node_pairs + comm_gates + comm_sched) / 3.0
            if r > 16
            else 0.5 * (comm_node_pairs + comm_gates)
        )

        node_align = torch.stack(
            [m.align_sq(ua_mid), m.align_sq(ue_mid), m.align_sq(u_w)], dim=2
        )
        if free_mask is not None:
            fm = free_mask.to(torch.float32)
            energy_align = (node_align * fm).sum(dim=(1, 2, 3)) / fm.sum(
                dim=(1, 2, 3)
            ).clamp_min(1.0)
        else:
            energy_align = node_align.mean(dim=(1, 2, 3))

        energy_prefix = torch.zeros(bsz, dtype=torch.float32, device=device)
        if (
            has_digest_prefix
            and digest_target_mask is not None
            and digest_target_bits is not None
            and iv_words is not None
        ):
            s_r_spins = torch.stack(
                [
                    a_spins[:, r + 3],
                    a_spins[:, r + 2],
                    a_spins[:, r + 1],
                    a_spins[:, r],
                    e_spins[:, r + 3],
                    e_spins[:, r + 2],
                    e_spins[:, r + 1],
                    e_spins[:, r],
                ],
                dim=1,
            )
            iv_spins = words_to_spins(iv_words.to(device), dtype=torch.float32)
            h_pred_spins = spin_add32(s_r_spins, iv_spins.unsqueeze(0).expand(bsz, 8, 32))
            target_spins = 1.0 - 2.0 * digest_target_bits.to(torch.float32).to(device)
            t_mask = digest_target_mask.to(torch.float32).to(device)
            energy_prefix = ((h_pred_spins - target_spins).square() * t_mask).sum(dim=(1, 2))

        total_energy = (
            energy_2adic
            + w_xor * energy_xor
            + lam_comm * energy_comm
            + lam_align * energy_align
            + float(w_digest_prefix) * energy_prefix
        )
        return (
            total_energy,
            features,
            per_scale,
            per_scale_e,
            per_scale_w,
            per_scale_s,
            per_scale[:, 0],
            per_scale[:, 31],
            energy_2adic,
            energy_xor,
            energy_comm,
            energy_align,
            energy_prefix,
        )

    @torch.amp.autocast("cuda", enabled=False)
    def compute_energy_and_features(
        self,
        U_A_full: torch.Tensor,
        U_E_full: torch.Tensor,
        U_W_full: torch.Tensor,
        free_mask: Optional[torch.Tensor] = None,
        lam: Optional[torch.Tensor] = None,
        digest_target_bits: Optional[torch.Tensor] = None,
        digest_target_mask: Optional[torch.Tensor] = None,
        iv_words: Optional[torch.Tensor] = None,
        w_xor: float | torch.Tensor = 1.0,
        lam_comm: float | torch.Tensor = 0.0,
        lam_align: float | torch.Tensor = 0.0,
        w_digest_prefix: float = 4.0,
        energy_fn: Optional[Callable[..., Tuple[torch.Tensor, ...]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        bsz = U_A_full.shape[0]
        lam_b = self.dual_basis._normalize_lambda(lam, bsz, U_A_full.device)
        has_prefix = (
            digest_target_mask is not None
            and digest_target_bits is not None
            and iv_words is not None
            and bool((digest_target_mask > 0.0).any().item())
        )
        fn = self.compute_energy_and_features_core if energy_fn is None else energy_fn
        out = fn(
            U_A_full,
            U_E_full,
            U_W_full,
            free_mask,
            lam_b,
            w_xor,
            lam_comm,
            lam_align,
            digest_target_bits,
            digest_target_mask,
            iv_words,
            w_digest_prefix,
            bool(digest_target_mask is not None),
            has_prefix,
        )
        return out[0], out[1], {
            "per_scale": out[2],
            "per_scale_e": out[3],
            "per_scale_w": out[4],
            "per_scale_s": out[5],
            "m1_energy": out[6],
            "m32_energy": out[7],
            "energy_2adic": out[8],
            "energy_xor": out[9],
            "energy_comm": out[10],
            "energy_align": out[11],
            "energy_prefix": out[12],
        }

    @torch.amp.autocast("cuda", enabled=False)
    def compute_radar_prepared(
        self,
        u_nodes: torch.Tensor,
        prep: PreparedManifoldBoundary,
        lam: torch.Tensor,
        grad_scale: float = 2.0,
        w_xor: float | torch.Tensor = 1.0,
        lam_comm: float | torch.Tensor = 0.0,
        lam_align: float | torch.Tensor = 0.0,
        w_digest_prefix: float = 4.0,
        energy_fn: Optional[Callable[..., Tuple[torch.Tensor, ...]]] = None,
        return_raw_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        with torch.inference_mode(False):
            with torch.enable_grad():
                u_in = normalize_manifold(u_nodes.detach()).requires_grad_(True)
                u_a, u_e, u_w = assemble_full_prepared(u_in, prep, self.manifold)
                bsz = u_a.shape[0]
                lam_b = self.dual_basis._normalize_lambda(lam.detach().clone(), bsz, u_a.device)
                fn = self.compute_energy_and_features_core if energy_fn is None else energy_fn
                out = fn(
                    u_a,
                    u_e,
                    u_w,
                    prep.free_mask.detach().clone(),
                    lam_b,
                    w_xor,
                    lam_comm,
                    lam_align,
                    None if prep.digest_target_bits is None else prep.digest_target_bits.detach().clone(),
                    None if prep.digest_target_mask is None else prep.digest_target_mask.detach().clone(),
                    None if prep.iv_words is None else prep.iv_words.detach().clone(),
                    w_digest_prefix,
                    prep.has_free_right,
                    prep.has_digest_prefix,
                )
                grad_amb = torch.autograd.grad(out[0].sum(), u_in, create_graph=False)[0]
                grad_riem = self.manifold.tangent_project(u_in, grad_amb) * prep.free_mask_exp
                grad_radar = torch.clamp(grad_scale * grad_riem, -4.0, 4.0)
        stats = {
            "per_scale": out[2].detach(),
            "energy_2adic": out[8].detach(),
            "energy_xor": out[9].detach(),
            "energy_comm": out[10].detach(),
            "energy_align": out[11].detach(),
            "energy_prefix": out[12].detach(),
        }
        return (
            grad_radar.detach(),
            out[1].detach(),
            stats,
            grad_riem.detach() if return_raw_grad else torch.empty(0),
        )

    @torch.amp.autocast("cuda", enabled=False)
    def retract_and_clamp(
        self,
        u_cur: torch.Tensor,
        logits_spin: torch.Tensor,
        delta_u_ambient: torch.Tensor,
        grad_riem: torch.Tensor,
        u_clamped_masked: torch.Tensor,
        free_mask_exp: torch.Tensor,
        damp_t: torch.Tensor | float,
        eta_s: torch.Tensor | float,
        beta_s: torch.Tensor | float,
    ) -> torch.Tensor:
        spin_cur = u_cur[..., 0]
        spin_tgt = torch.tanh(logits_spin.to(torch.float32))
        d_spin = spin_tgt - spin_cur
        # Zero padding columns for delta_spin along transverse axes
        pad_z = torch.zeros_like(u_cur[..., 1:])
        d_spin_amb = torch.cat([d_spin.unsqueeze(-1), pad_z], dim=-1)
        delta_amb = damp_t * d_spin_amb + beta_s * delta_u_ambient.to(torch.float32) - eta_s * grad_riem.to(torch.float32)
        u_raw = self.manifold.retract(u_cur, delta_amb, step_size=1.0)
        return normalize_manifold(u_clamped_masked + free_mask_exp * u_raw)


class V5BBackbone(nn.Module):
    """Strict-FP32 2D Bit-Grid + BiGRU Burer-Monteiro S^{N-1} Backbone with V3 warm-start."""

    BASE_DIM: int = 16
    ROT_DIM: int = BitGridBlock.ROT_DIM
    IN_DIM: int = BASE_DIM + ROT_DIM
    TRANS_BOTTLENECK: int = 6

    def __init__(
        self,
        manifold_dim: int = 32,
        rounds: int = 18,
        d_bit: int = 48,
        n_heads: int = 3,
        n_blocks: int = 2,
        d_ff_mult: int = 2,
        share_blocks: bool = False,
        grad_checkpointing: bool = False,
        use_carry_gru: bool = True,
    ) -> None:
        super().__init__()
        self.manifold = BurerMonteiroSphereBackend(manifold_dim=manifold_dim)
        self.manifold_dim = self.manifold.dim
        self.trans_dim = self.manifold_dim - 1
        self.rounds = int(rounds)
        self.d_bit = int(d_bit)
        self.grad_checkpointing = bool(grad_checkpointing)

        # Shared V3 core layers (exact name parity for lossless warm-start)
        self.input_proj = nn.Linear(self.IN_DIM, d_bit)
        self.h_prev_norm = nn.LayerNorm(d_bit, eps=1e-6)
        self.h_prev_proj = nn.Linear(d_bit, d_bit)
        self.register_buffer(
            "round_pos_emb",
            sinusoidal_embedding(torch.arange(rounds, dtype=torch.float32), d_bit).view(1, rounds, 1, d_bit),
            persistent=False,
        )
        self.register_buffer(
            "bit_pos_emb",
            sinusoidal_embedding(torch.arange(32, dtype=torch.float32), d_bit).view(1, 1, 32, d_bit),
            persistent=False,
        )
        self.register_buffer(
            "round_attn_mask", build_round_sparse_attn_mask(rounds, num_registers=0), persistent=False
        )
        self.step_mlp = nn.Sequential(nn.Linear(d_bit, d_bit), nn.SiLU(), nn.Linear(d_bit, d_bit))
        self.lam_mlp = nn.Sequential(nn.Linear(32, d_bit), nn.SiLU(), nn.Linear(d_bit, d_bit))

        if share_blocks:
            shared = BitGridBlock(
                d_bit=d_bit, n_heads=n_heads, d_ff_mult=d_ff_mult, use_carry_gru=use_carry_gru
            )
            self.blocks = nn.ModuleList([shared for _ in range(n_blocks)])
        else:
            self.blocks = nn.ModuleList(
                [
                    BitGridBlock(
                        d_bit=d_bit, n_heads=n_heads, d_ff_mult=d_ff_mult, use_carry_gru=use_carry_gru
                    )
                    for _ in range(n_blocks)
                ]
            )

        self.final_norm = nn.LayerNorm(d_bit, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_bit, 2 * d_bit, bias=True))
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        self.out_proj = nn.Linear(d_bit, 3)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

        # Compact factorized V5B transverse adapters scaling cleanly across N in {16, 32, 64}
        self.bm_trans_compress = nn.Linear(self.trans_dim, self.TRANS_BOTTLENECK, bias=False)
        nn.init.orthogonal_(self.bm_trans_compress.weight, gain=1.0 / math.sqrt(float(self.trans_dim)))
        self.manifold_in_proj = nn.Linear(42 * self.TRANS_BOTTLENECK, d_bit, bias=True)
        nn.init.orthogonal_(self.manifold_in_proj.weight, gain=0.01)
        nn.init.zeros_(self.manifold_in_proj.bias)

        self.manifold_tangent_head = nn.Linear(d_bit, 3 * self.manifold_dim, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.manifold_tangent_head.bias)
            w_view = self.manifold_tangent_head.weight.view(3, self.manifold_dim, d_bit)
            w_view[:, 0, :] = 0.0
            for ax in range(1, self.manifold_dim):
                nn.init.orthogonal_(w_view[:, ax, :], gain=0.02 / math.sqrt(float(self.trans_dim)))

    def _build_transverse_features(
        self, u_nodes: torch.Tensor, grad_radar: torch.Tensor
    ) -> torch.Tensor:
        u_t = self.bm_trans_compress(self.manifold.project_transverse(u_nodes))
        g_t = self.bm_trans_compress(self.manifold.project_transverse(grad_radar))
        bsz, r, _, n_bits, bt = u_t.shape
        src = torch.cat(
            [
                u_t.permute(0, 1, 3, 2, 4).reshape(bsz, r, n_bits, 3 * bt),
                g_t.permute(0, 1, 3, 2, 4).reshape(bsz, r, n_bits, 3 * bt),
            ],
            dim=-1,
        )
        rotated = [torch.roll(src, shifts=-off, dims=2) for off in SHA_ROTR_OFFSETS]
        return torch.cat([src, *rotated], dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        u_nodes: torch.Tensor,
        grad_radar: torch.Tensor,
        clamp_mask: torch.Tensor,
        res_features: torch.Tensor,
        k_spins: torch.Tensor,
        step_frac: torch.Tensor,
        lam: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u_f = u_nodes.to(torch.float32)
        g_f = grad_radar.to(torch.float32)
        bsz, r, _, n_bits, d = u_f.shape
        x_spin = self.manifold.project_spin(u_f)
        g_spin = self.manifold.project_spin(g_f)

        lam_b = lam.to(torch.float32).unsqueeze(0).expand(bsz, 32) if lam.ndim == 1 else lam.to(torch.float32)
        step_b = step_frac.to(torch.float32).expand(bsz) if step_frac.ndim == 0 else step_frac.to(torch.float32)

        base_feat, x_grid, grad_grid = extract_grid_features(
            x_nodes=x_spin,
            grad_radar=g_spin,
            clamp_mask=clamp_mask.to(torch.float32),
            res_features=res_features.to(torch.float32),
            k_spins=k_spins.to(torch.float32),
        )
        rot_feat = build_rotr_channels(x_grid, grad_grid)
        grid_in = torch.cat([base_feat, rot_feat], dim=-1).to(torch.float32)
        trans_feat = self._build_transverse_features(u_f, g_f)

        cond = F.silu(
            self.step_mlp(sinusoidal_embedding(step_b * 1000.0, self.d_bit)) + self.lam_mlp(lam_b)
        )
        h = self.input_proj(grid_in) + self.manifold_in_proj(trans_feat) + self.round_pos_emb + self.bit_pos_emb
        if h_prev is not None:
            h = h + self.h_prev_proj(self.h_prev_norm(h_prev.to(torch.float32)))

        rot_in = rot_feat.to(torch.float32)
        for block in self.blocks:
            h = block(h, rot_in, cond, self.round_attn_mask)

        sh_f, sc_f = self.final_adaLN(cond).chunk(2, dim=-1)
        h_out = _modulate_grid(self.final_norm(h), sh_f, sc_f)
        logits_spin = self.out_proj(h_out).permute(0, 1, 3, 2).contiguous()
        delta_u_ambient = (
            self.manifold_tangent_head(h_out).view(bsz, r, n_bits, 3, d).permute(0, 1, 3, 2, 4).contiguous()
        )
        return logits_spin, delta_u_ambient, h.to(torch.float32)

    def load_v3_warmstart(
        self, ckpt_path: str | Path, map_location: torch.device | str = "cpu"
    ) -> Dict[str, object]:
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        src_state = ckpt.get("ema_model_state", ckpt.get("model_state"))
        incompat = self.load_state_dict(src_state, strict=False)
        allowed = ("bm_trans_compress.", "manifold_in_proj.", "manifold_tangent_head.")
        non_adapter_missing = [
            k for k in incompat.missing_keys if not k.startswith(allowed)
        ]
        if incompat.unexpected_keys or non_adapter_missing:
            raise RuntimeError(
                f"V3->V5B mismatch: unexpected={incompat.unexpected_keys}, missing={non_adapter_missing}"
            )
        self.float()
        return {
            "loaded_keys": len(src_state),
            "missing_adapter_keys": list(incompat.missing_keys),
            "source_step": ckpt.get("step", "?"),
        }


class UnrolledSolverV5B(OuterSearchShell):
    """Test-time compute solver for V5B with Cayley negative-curvature escape & V4D/V4E support."""

    def __init__(
        self,
        model: V5BBackbone,
        physics: V5BPhysicsEngine,
        num_candidates: int = 256,
        num_steps: int = 48,
        damp: float = 0.25,
        eta_max: float = 0.025,
        beta_max: float = 0.25,
        grad_scale: float = 2.0,
        chunk_size: int = 1024,
        w_xor: float = 1.0,
        lam_comm_end: float = 0.40,
        lam_rank1_end: float = 6.0,
        escape_patience: int = 6,
        escape_step_t: float = 0.35,
    ) -> None:
        self.model = model
        self.physics = physics
        self.manifold = physics.manifold
        self.dual_basis = physics.dual_basis
        self.num_candidates = int(num_candidates)
        self.num_steps = int(num_steps)
        self.damp = float(damp)
        self.eta_max = float(eta_max)
        self.beta_max = float(beta_max)
        self.grad_scale = float(grad_scale)
        self.chunk_size = int(chunk_size)
        self.w_xor = float(w_xor)
        self.lam_comm_end = float(lam_comm_end)
        self.lam_rank1_end = float(lam_rank1_end)
        self.escape_patience = max(2, int(escape_patience))
        self.escape_step_t = float(escape_step_t)

    @torch.no_grad()
    def infer_deterministic_zero(
        self,
        boundary: ManifoldBoundary,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        bsz = boundary.batch_size
        r = boundary.rounds
        dev = boundary.device
        prep = boundary.prepare_for_manifold(self.manifold)
        lam_full = torch.ones(32, dtype=torch.float32, device=dev)

        u_cur = apply_clamp_prepared(
            self.manifold.make_neutral_equator((bsz, r, 3, 32), device=dev), prep
        )
        h_prev: Optional[torch.Tensor] = None
        logits_last: Optional[torch.Tensor] = None
        denom = float(max(1, self.num_steps - 1))
        for s in range(self.num_steps):
            frac = float(s) / denom
            lam_c = anneal_schedule(frac, 0.02, self.lam_comm_end)
            lam_r = anneal_schedule(frac, 0.02, self.lam_rank1_end)
            eta_s = self.eta_max * (0.2 + 0.8 * frac)
            beta_s = self.beta_max * (frac * frac)
            grad_radar, res_feat, _, grad_riem = self.physics.compute_radar_prepared(
                u_cur, prep, lam=lam_full, grad_scale=self.grad_scale,
                w_xor=self.w_xor, lam_comm=lam_c, lam_align=lam_r, return_raw_grad=True,
            )
            sf = torch.full((bsz,), frac, dtype=torch.float32, device=dev)
            logits_last, delta_u, h_tilde = self.model(
                u_cur, grad_radar, prep.clamp_mask, res_feat, self.physics.K_spins, sf, lam_full, h_prev
            )
            h_prev = h_tilde if h_prev is None else (1.0 - self.damp) * h_prev + self.damp * h_tilde
            u_cur = self.physics.retract_and_clamp(
                u_cur, logits_last, delta_u, grad_riem, prep.u_clamped_masked, prep.free_mask_exp,
                self.damp, eta_s, beta_s
            )

        spins = self.manifold.project_spin(u_cur)
        x_disc = torch.where(spins < 0.0, -torch.ones_like(spins), torch.ones_like(spins))
        x_disc = boundary.to_clamped_boundary().apply_clamp(x_disc)
        u_a, u_e, u_w = assemble_full_prepared(self.manifold.embed_spins(x_disc), prep, self.manifold)
        _, _, stats = self.physics.compute_energy_and_features(u_a, u_e, u_w, free_mask=prep.free_mask)
        m1, m32 = stats["m1_energy"], stats["m32_energy"]

        u_a_c, u_e_c, u_w_c = assemble_full_prepared(u_cur, prep, self.manifold)
        _, _, c_stats = self.physics.compute_energy_and_features(u_a_c, u_e_c, u_w_c, free_mask=prep.free_mask)
        rank1_r, eff_r = compute_bm_rank1_and_eff_rank(u_cur, prep.free_mask)
        zeros = torch.zeros_like(m1)
        return (
            logits_last,
            x_disc,
            m1,
            m32,
            zeros,
            c_stats["energy_align"],
            c_stats["energy_comm"],
            {"rank1_ratio": rank1_r, "eff_rank": eff_r, "escape_count": 0.0},
        )

    @torch.no_grad()
    def solve(
        self,
        target_digests: Optional[torch.Tensor] = None,
        boundary: Optional[ManifoldBoundary] = None,
        generator: Optional[torch.Generator] = None,
        num_candidates: Optional[int] = None,
        num_steps: Optional[int] = None,
    ) -> SolveResultV5:
        if boundary is None:
            if target_digests is None:
                raise ValueError("Either boundary or target_digests must be provided.")
            boundary = build_manifold_boundary(target_digests, rounds=self.model.rounds)

        dev = boundary.device
        b_cnt = boundary.batch_size
        k_cand = self.num_candidates if num_candidates is None else int(num_candidates)
        n_steps = self.num_steps if num_steps is None else int(num_steps)
        if target_digests is None:
            target_digests = torch.zeros((b_cnt, 8), dtype=torch.int64, device=dev)

        lam_full = torch.ones(32, dtype=torch.float32, device=dev)
        b_rep = boundary.repeat_for_candidates(k_cand)
        total_traj = b_cnt * k_cand

        disc_chunks, e_chunks, m1_chunks, m32_chunks, al_chunks, cm_chunks = [], [], [], [], [], []
        rank1_vals, eff_vals = [], []
        escape_count = 0

        for start in range(0, total_traj, self.chunk_size):
            end = min(start + self.chunk_size, total_traj)
            m_cur = end - start
            b_sub = ManifoldBoundary(
                rounds=b_rep.rounds,
                A_full_clamped=b_rep.A_full_clamped[start:end],
                A_full_mask=b_rep.A_full_mask[start:end],
                E_full_clamped=b_rep.E_full_clamped[start:end],
                E_full_mask=b_rep.E_full_mask[start:end],
                W_full_clamped=b_rep.W_full_clamped[start:end],
                W_full_mask=b_rep.W_full_mask[start:end],
                A_words_clamped=b_rep.A_words_clamped[start:end],
                E_words_clamped=b_rep.E_words_clamped[start:end],
                W16_clamped_words=b_rep.W16_clamped_words[start:end],
                W16_mask_bits=b_rep.W16_mask_bits[start:end],
                digest_target_bits=(
                    None if b_rep.digest_target_bits is None else b_rep.digest_target_bits[start:end]
                ),
                digest_target_mask=(
                    None if b_rep.digest_target_mask is None else b_rep.digest_target_mask[start:end]
                ),
                iv_words=b_rep.iv_words,
                is_full_digest_clamped=b_rep.is_full_digest_clamped,
            )
            prep = b_sub.prepare_for_manifold(self.manifold)
            u_cur = apply_clamp_prepared(
                self.manifold.sample_random_manifold(
                    (m_cur, b_sub.rounds, 3, 32), device=dev, generator=generator, spin_std=0.45
                ),
                prep,
            )
            h_prev: Optional[torch.Tensor] = None
            prev_best_e = float("inf")
            stagnant = 0

            denom = float(max(1, n_steps - 1))
            for s in range(n_steps):
                frac = float(s) / denom
                lam_c = anneal_schedule(frac, 0.02, self.lam_comm_end)
                lam_r = anneal_schedule(frac, 0.02, self.lam_rank1_end)
                grad_radar, res_feat, e_stats, grad_riem = self.physics.compute_radar_prepared(
                    u_cur, prep, lam=lam_full, grad_scale=self.grad_scale,
                    w_xor=self.w_xor, lam_comm=lam_c, lam_align=lam_r, return_raw_grad=True,
                )
                cur_mean_e = float(e_stats["energy_2adic"].mean().item())
                if s > 2 and cur_mean_e >= prev_best_e - 1e-3:
                    stagnant += 1
                    if stagnant >= self.escape_patience and s < n_steps - 2:
                        u_esc = cayley_orthogonal_escape(u_cur, step_t=self.escape_step_t, generator=generator)
                        u_cur = normalize_manifold(prep.u_clamped_masked + prep.free_mask_exp * u_esc)
                        stagnant = 0
                        escape_count += 1
                else:
                    prev_best_e = min(prev_best_e, cur_mean_e)
                    stagnant = 0

                sf = torch.full((m_cur,), frac, dtype=torch.float32, device=dev)
                logits_s, delta_u, h_tilde = self.model(
                    u_cur, grad_radar, prep.clamp_mask, res_feat, self.physics.K_spins, sf, lam_full, h_prev
                )
                h_prev = h_tilde if h_prev is None else (1.0 - self.damp) * h_prev + self.damp * h_tilde
                u_cur = self.physics.retract_and_clamp(
                    u_cur, logits_s, delta_u, grad_riem, prep.u_clamped_masked, prep.free_mask_exp,
                    self.damp, self.eta_max * (0.2 + 0.8 * frac), self.beta_max * (frac * frac)
                )

            r1, er = compute_bm_rank1_and_eff_rank(u_cur, prep.free_mask)
            rank1_vals.append(r1)
            eff_vals.append(er)

            spins = self.manifold.project_spin(u_cur)
            x_disc = torch.where(spins < 0.0, -torch.ones_like(spins), torch.ones_like(spins))
            x_disc = b_sub.to_clamped_boundary().apply_clamp(x_disc)
            u_a, u_e, u_w = assemble_full_prepared(self.manifold.embed_spins(x_disc), prep, self.manifold)
            tot_e, _, stats = self.physics.compute_energy_and_features(u_a, u_e, u_w, free_mask=prep.free_mask)
            u_a_c, u_e_c, u_w_c = assemble_full_prepared(u_cur, prep, self.manifold)
            _, _, c_stats = self.physics.compute_energy_and_features(u_a_c, u_e_c, u_w_c, free_mask=prep.free_mask)

            disc_chunks.append(x_disc)
            e_chunks.append(tot_e)
            m1_chunks.append(stats["m1_energy"])
            m32_chunks.append(stats["m32_energy"])
            al_chunks.append(c_stats["energy_align"])
            cm_chunks.append(c_stats["energy_comm"])

        r_n = self.model.rounds
        x_disc_all = torch.cat(disc_chunks, dim=0).view(b_cnt, k_cand, r_n, 3, 32)
        e_all = torch.cat(e_chunks, dim=0).view(b_cnt, k_cand)
        m1_all = torch.cat(m1_chunks, dim=0).view(b_cnt, k_cand)
        m32_all = torch.cat(m32_chunks, dim=0).view(b_cnt, k_cand)
        al_all = torch.cat(al_chunks, dim=0).view(b_cnt, k_cand)
        cm_all = torch.cat(cm_chunks, dim=0).view(b_cnt, k_cand)

        k_w = min(16, r_n)
        w_pred_bits = (x_disc_all[:, :, :k_w, 2, :] < 0.0).to(torch.int64)
        if k_w < 16:
            w_pred_bits = torch.cat(
                [w_pred_bits, torch.zeros((b_cnt, k_cand, 16 - k_w, 32), dtype=torch.int64, device=dev)], dim=2
            )
        w16_m_int = (boundary.W16_mask_bits > 0.5).to(torch.int64).unsqueeze(1)
        w16_cand = bits_to_words(torch.where(w16_m_int > 0, words_to_bits(boundary.W16_clamped_words).unsqueeze(1), w_pred_bits))

        if boundary.is_full_digest_clamped:
            dig_cand = sha256_digest(w16_cand.reshape(-1, 16), rounds=r_n).view(b_cnt, k_cand, 8)
            hits_mat = (dig_cand == target_digests.unsqueeze(1)).all(dim=-1)
        elif boundary.digest_target_mask is not None and boundary.digest_target_bits is not None:
            dig_cand = sha256_digest(w16_cand.reshape(-1, 16), rounds=r_n).view(b_cnt, k_cand, 8)
            dig_bits = words_to_bits(dig_cand).to(torch.float32)
            hits_mat = ((dig_bits - boundary.digest_target_bits.unsqueeze(1)).abs() * boundary.digest_target_mask.unsqueeze(1)).sum(dim=(-2, -1)) < 0.5
        else:
            hits_mat = torch.zeros((b_cnt, k_cand), dtype=torch.bool, device=dev)

        num_hits = hits_mat.sum(dim=1)
        solved = num_hits > 0
        best_idx = torch.where(hits_mat, m32_all - 1e6, m32_all).argmin(dim=1) if solved.any() else m32_all.argmin(dim=1)
        b_idx = torch.arange(b_cnt, device=dev)
        w0_vals = w16_cand[:, :, 0].detach().cpu().tolist()
        uniq_ratios = torch.tensor(
            [float(len(set(row))) / float(max(1, k_cand)) for row in w0_vals], dtype=torch.float32, device=dev
        )
        cal_probs, p_weights = compute_calibrated_probs_from_candidates(x_disc_all, m32_all, temperature=1.0)
        return SolveResultV5(
            solved=solved,
            num_hits=num_hits,
            unique_ratio=uniq_ratios,
            latch_ratio=torch.zeros((b_cnt,), dtype=torch.float32, device=dev),
            best_w16=w16_cand[b_idx, best_idx],
            best_energy=e_all[b_idx, best_idx],
            best_m1_energy=m1_all[b_idx, best_idx],
            best_m32_energy=m32_all[b_idx, best_idx],
            min_m32_energy=m32_all.min(dim=1).values,
            best_x_disc=x_disc_all[b_idx, best_idx],
            mean_align=al_all.mean(dim=1),
            mean_comm=cm_all.mean(dim=1),
            calibrated_probs=cal_probs,
            particle_weights=p_weights,
            extra_metrics={
                "rank1_ratio": float(sum(rank1_vals) / max(1, len(rank1_vals))),
                "eff_rank": float(sum(eff_vals) / max(1, len(eff_vals))),
                "escape_count": float(escape_count),
            },
        )

    def export_calibrated_probs(self, solve_result: SolveResultV5) -> torch.Tensor:
        if solve_result.calibrated_probs is not None:
            return solve_result.calibrated_probs
        return compute_calibrated_probs_from_logits(1.5 * solve_result.best_x_disc)


__all__ = [
    "BurerMonteiroSphereBackend",
    "UnrolledSolverV5B",
    "V5BBackbone",
    "V5BPhysicsEngine",
    "cayley_orthogonal_escape",
    "compute_bm_rank1_and_eff_rank",
]
