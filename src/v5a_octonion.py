"""Phase-SHA V5A: 8D Non-Associative Octonion Manifold Lift (S^7 subset O, strict pure FP32).

Strictly aligned with Lean 4 Module 6A (formal_proofs/PhaseSHA/V5/OctonionLift.lean):
  - Pure-real FP32 8D Cayley-Dickson multiplication: (a, b)*(c, d) = (a*c - conj(d)*b, d*a + b*conj(c)).
  - Zero torch.complex64, zero graph breaks, Degen 8-square norm conservation.
  - Explicit 3-input XOR associator residual ||(xy)z - x(yz)||_2^2 + dual progressive collapse schedule.
  - Identical shared 2D Bit-Grid + BiGRU backbone & V3 warm-start to V4C for pure 4D -> 8D ablation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.bit_grid_backbone import (
    BitGridBlock,
    SHA_ROTR_OFFSETS,
    build_rotr_channels,
    extract_grid_features,
    _modulate_grid,
)
from src.dual_basis import DualBasisEngine, spin_Ch, spin_Maj
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
from src.v4_solver import anneal_schedule
from src.v5_interfaces import (
    GeometryBackend,
    OuterSearchShell,
    SolveResultV5,
    compute_calibrated_probs_from_candidates,
    compute_calibrated_probs_from_logits,
)


@torch.amp.autocast("cuda", enabled=False)
def _quat_mul_fp32(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ],
        dim=-1,
    )


@torch.amp.autocast("cuda", enabled=False)
def _quat_conj_fp32(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def _oct_mul_cayley_dickson_ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reference Cayley-Dickson octonion product matching Lean 4 `Oct.mul`.

    x = (a, b), y = (c, d) with a, b, c, d in H:
      (a, b) * (c, d) = (a*c - conj(d)*b, d*a + b*conj(c))
    """
    xf, yf = x.to(torch.float32), y.to(torch.float32)
    a, b = xf[..., :4], xf[..., 4:8]
    c, d = yf[..., :4], yf[..., 4:8]
    lo = _quat_mul_fp32(a, c) - _quat_mul_fp32(_quat_conj_fp32(d), b)
    hi = _quat_mul_fp32(d, a) + _quat_mul_fp32(b, _quat_conj_fp32(c))
    return torch.cat([lo, hi], dim=-1)


def _build_oct_structure_matrices() -> Tuple[torch.Tensor, torch.Tensor]:
    """Build exact FP32 (8, 64) bilinear left-mul and commutator structure matrices from Cayley-Dickson."""
    t_3d = torch.zeros((8, 8, 8), dtype=torch.float32)
    eye = torch.eye(8, dtype=torch.float32)
    for i in range(8):
        for j in range(8):
            t_3d[i, j, :] = _oct_mul_cayley_dickson_ref(eye[i], eye[j])
    t_mat = t_3d.reshape(8, 64).contiguous()
    c_mat = (t_3d - t_3d.transpose(0, 1)).reshape(8, 64).contiguous()
    return t_mat, c_mat


_OCT_T_MAT_CPU, _OCT_C_MAT_CPU = _build_oct_structure_matrices()
_OCT_T_CACHE: Dict[torch.device, torch.Tensor] = {}
_OCT_C_CACHE: Dict[torch.device, torch.Tensor] = {}


def _get_oct_t_mat(device: torch.device) -> torch.Tensor:
    t = _OCT_T_CACHE.get(device)
    if t is None:
        t = _OCT_T_MAT_CPU.to(device=device, dtype=torch.float32)
        if not torch.compiler.is_compiling():
            _OCT_T_CACHE[device] = t
    return t


def _get_oct_c_mat(device: torch.device) -> torch.Tensor:
    c = _OCT_C_CACHE.get(device)
    if c is None:
        c = _OCT_C_MAT_CPU.to(device=device, dtype=torch.float32)
        if not torch.compiler.is_compiling():
            _OCT_C_CACHE[device] = c
    return c


@torch.amp.autocast("cuda", enabled=False)
def oct_mul_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Exact pure-FP32 Cayley-Dickson octonion product matching Lean 4 `Oct.mul` (2 GEMMs)."""
    xf, yf = x.to(torch.float32), y.to(torch.float32)
    t_mat = _get_oct_t_mat(xf.device)
    l_x = torch.matmul(xf, t_mat).view(*xf.shape[:-1], 8, 8)
    return torch.matmul(yf.unsqueeze(-2), l_x).squeeze(-2)


@torch.amp.autocast("cuda", enabled=False)
def oct_comm_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Exact pure-FP32 octonion commutator [x, y] = x*y - y*x."""
    xf, yf = x.to(torch.float32), y.to(torch.float32)
    c_mat = _get_oct_c_mat(xf.device)
    c_x = torch.matmul(xf, c_mat).view(*xf.shape[:-1], 8, 8)
    return torch.matmul(yf.unsqueeze(-2), c_x).squeeze(-2)


@torch.amp.autocast("cuda", enabled=False)
def oct_associator_fp32(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Exact 8D associator (x * y) * z - x * (y * z) matching Lean 4 `assoc`."""
    return oct_mul_fp32(oct_mul_fp32(x, y), z) - oct_mul_fp32(x, oct_mul_fp32(y, z))


class OctonionS7Backend(GeometryBackend):
    """V6 Slot 1: 8D Non-Associative Division Algebra Unit Hypersphere S^7 subset O."""

    variant = "v5a"
    dim = 8
    spin_axis = 0
    trans_axes = (1, 2, 3, 4, 5, 6, 7)
    is_abelian = False

    def e0_vec(self, device: torch.device) -> torch.Tensor:
        v = torch.zeros((8,), dtype=torch.float32, device=device)
        v[0] = 1.0
        return v

    def eq_vec(self, device: torch.device) -> torch.Tensor:
        v = torch.zeros((8,), dtype=torch.float32, device=device)
        v[1] = 1.0
        return v

    @staticmethod
    def imag_basis(k: int, device: torch.device) -> torch.Tensor:
        """Exact e_{k+1} for k in {0..6} matching Lean 4 `imagBasis`."""
        if not (0 <= k < 7):
            raise ValueError(f"k must be in [0, 6], got {k}")
        v = torch.zeros((8,), dtype=torch.float32, device=device)
        v[k + 1] = 1.0
        return v

    @staticmethod
    def imag_path(k: int, t: torch.Tensor) -> torch.Tensor:
        """Singularity-free 8-Lipschitz unit curve connecting +1 (t=0) and -1 (t=1) along e_{k+1}."""
        tf = t.to(torch.float32)
        den = 1.0 - 2.0 * tf + 2.0 * tf * tf
        c_t = (1.0 - 2.0 * tf) / den
        s_t = (2.0 * tf * (1.0 - tf)) / den
        out = torch.zeros((*tf.shape, 8), dtype=torch.float32, device=tf.device)
        cols = [
            c_t if ax == 0 else (s_t if ax == (k + 1) else torch.zeros_like(tf))
            for ax in range(8)
        ]
        return torch.stack(cols, dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def embed_spins(self, spins: torch.Tensor) -> torch.Tensor:
        s = spins.to(torch.float32)
        z = torch.zeros_like(s)
        return torch.stack([s, z, z, z, z, z, z, z], dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def embed_spins_with_equator(self, spins: torch.Tensor) -> torch.Tensor:
        s = spins.to(torch.float32).clamp(-1.0, 1.0)
        trans = torch.sqrt((1.0 - s * s).clamp_min(0.0))
        z = torch.zeros_like(s)
        return normalize_manifold(torch.stack([s, trans, z, z, z, z, z, z], dim=-1))

    @torch.amp.autocast("cuda", enabled=False)
    def project_spin(self, u: torch.Tensor) -> torch.Tensor:
        return u.to(torch.float32)[..., 0]

    @torch.amp.autocast("cuda", enabled=False)
    def project_transverse(self, u: torch.Tensor) -> torch.Tensor:
        return u.to(torch.float32)[..., 1:8]

    @torch.amp.autocast("cuda", enabled=False)
    def compose_from_spin_and_transverse_dir(
        self, spins: torch.Tensor, u_ref: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        s = spins.to(torch.float32).clamp(-1.0, 1.0)
        trans_amp = torch.sqrt((1.0 - s * s).clamp_min(0.0))
        ref_t = self.project_transverse(u_ref)
        ref_norm = torch.linalg.vector_norm(ref_t, dim=-1, keepdim=True)
        z = torch.zeros_like(s)
        fb = torch.stack([torch.ones_like(s), z, z, z, z, z, z], dim=-1)
        dir_t = torch.where(ref_norm > eps, ref_t / ref_norm.clamp_min(eps), fb)
        return normalize_manifold(torch.cat([s.unsqueeze(-1), trans_amp.unsqueeze(-1) * dir_t], dim=-1), eps=eps)

    @torch.amp.autocast("cuda", enabled=False)
    def tangent_project(self, u: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        u_f, d_f = u.to(torch.float32), delta.to(torch.float32)
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
        return self.eq_vec(device).view(*([1] * len(shape)), 8).expand(*shape, 8).clone()

    @torch.amp.autocast("cuda", enabled=False)
    def sample_random_manifold(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        spin_std: float = 0.35,
    ) -> torch.Tensor:
        raw = torch.randn((*shape, 8), dtype=torch.float32, device=device, generator=generator)
        spin_p = raw[..., :1] * float(spin_std)
        return normalize_manifold(torch.cat([spin_p, raw[..., 1:8]], dim=-1))

    @torch.amp.autocast("cuda", enabled=False)
    def align_sq(self, u: torch.Tensor) -> torch.Tensor:
        return self.project_transverse(u).square().sum(dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def rotr(self, u_word: torch.Tensor, n: int) -> torch.Tensor:
        return u_word.to(torch.float32) if n == 0 else torch.roll(u_word.to(torch.float32), shifts=-n, dims=-2)

    @torch.amp.autocast("cuda", enabled=False)
    def shr(self, u_word: torch.Tensor, n: int) -> torch.Tensor:
        u_f = u_word.to(torch.float32)
        if n == 0:
            return u_f
        if n >= 32:
            return self.e0_vec(u_f.device).view(*([1] * (u_f.ndim - 1)), 8).expand_as(u_f).clone()
        e0 = self.e0_vec(u_f.device).view(*([1] * (u_f.ndim - 2)), 1, 8)
        pad = e0.expand(*u_f.shape[:-2], n, 8)
        return torch.cat([u_f[..., n:, :], pad], dim=-2)

    @torch.amp.autocast("cuda", enabled=False)
    def commutator_sq(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Exact ||[a, b]||_2^2 = ||a*b - b*a||_2^2."""
        return oct_comm_fp32(a, b).square().sum(dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def associator_sq(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Explicit 3-input non-associativity penalty ||(a*b)*c - a*(b*c)||_2^2."""
        return oct_associator_fp32(a, b, c).square().sum(dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def xor2_residual(self, a: torch.Tensor, b: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (oct_mul_fp32(a, b) - y.to(torch.float32)).square().sum(dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def xor3_spin(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        left = oct_mul_fp32(oct_mul_fp32(a, b), c)
        right = oct_mul_fp32(a, oct_mul_fp32(b, c))
        return 0.5 * (left[..., 0] + right[..., 0])

    @torch.amp.autocast("cuda", enabled=False)
    def xor3_residual(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        yf = y.to(torch.float32)
        left = oct_mul_fp32(oct_mul_fp32(a, b), c)
        right = oct_mul_fp32(a, oct_mul_fp32(b, c))
        return 0.5 * ((left - yf).square().sum(dim=-1) + (right - yf).square().sum(dim=-1))

    @torch.amp.autocast("cuda", enabled=False)
    def eq_s_lsb_residual(self, words9: List[torch.Tensor]) -> torch.Tensor:
        p0 = oct_mul_fp32(oct_mul_fp32(words9[0], words9[1]), words9[2])
        p1 = oct_mul_fp32(oct_mul_fp32(words9[3], words9[4]), words9[5])
        p2 = oct_mul_fp32(oct_mul_fp32(words9[6], words9[7]), words9[8])
        prod = oct_mul_fp32(oct_mul_fp32(p0, p1), p2)
        e0 = self.e0_vec(prod.device).view(*([1] * (prod.ndim - 1)), 8)
        return (prod - e0).square().sum(dim=-1)

    def _op3_words(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        left = oct_mul_fp32(oct_mul_fp32(a, b), c)
        right = oct_mul_fp32(a, oct_mul_fp32(b, c))
        spins = 0.5 * (left[..., 0] + right[..., 0])
        assoc = (left - right).square().sum(dim=-1)
        comm = self.commutator_sq(a, b) + self.commutator_sq(b, c) + self.commutator_sq(a, c)
        leak = 0.5 * (left[..., 1:8].square().sum(dim=-1) + right[..., 1:8].square().sum(dim=-1))
        return spins, comm, leak, assoc

    def sigma0_upper(self, u_a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._op3_words(self.rotr(u_a, 2), self.rotr(u_a, 13), self.rotr(u_a, 22))

    def sigma1_upper(self, u_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._op3_words(self.rotr(u_e, 6), self.rotr(u_e, 11), self.rotr(u_e, 25))

    def sigma0_lower(self, u_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._op3_words(self.rotr(u_w, 7), self.rotr(u_w, 18), self.shr(u_w, 3))

    def sigma1_lower(self, u_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._op3_words(self.rotr(u_w, 17), self.rotr(u_w, 19), self.shr(u_w, 10))


class V5APhysicsEngine(nn.Module):
    """Differentiable V5A S^7 Octonion + 32-scale 2-adic engine with explicit associator term."""

    def __init__(self, rounds: int = 18) -> None:
        super().__init__()
        self.variant = "v5a"
        self.rounds = int(rounds)
        self.manifold = OctonionS7Backend()
        self.dual_basis = DualBasisEngine(rounds=self.rounds)

    @property
    def K_spins(self) -> torch.Tensor:
        return self.dual_basis.K_spins

    def float(self) -> "V5APhysicsEngine":
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
        lam_assoc: float | torch.Tensor = 0.0,
        digest_target_bits: Optional[torch.Tensor] = None,
        digest_target_mask: Optional[torch.Tensor] = None,
        iv_words: Optional[torch.Tensor] = None,
        w_digest_prefix: float = 4.0,
        include_tail_e: bool = False,
        has_digest_prefix: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        r = self.rounds
        m = self.manifold
        u_a, u_e, u_w = u_a.to(torch.float32), u_e.to(torch.float32), u_w.to(torch.float32)
        bsz = u_a.shape[0]
        device = u_a.device

        a_spins = m.project_spin(u_a)
        e_spins = m.project_spin(u_e)
        w_spins = m.project_spin(u_w)

        len_e = r - 4
        sig0_a_spins, comm_s0, leak_s0, assoc_s0 = m.sigma0_upper(u_a[:, 3 : 3 + len_e])
        maj_a = spin_Maj(
            a_spins[:, 3 : 3 + len_e], a_spins[:, 2 : 2 + len_e], a_spins[:, 1 : 1 + len_e]
        )
        sig1_e_spins, comm_s1, leak_s1, assoc_s1 = m.sigma1_upper(u_e[:, 3 : 3 + r])
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
            sig0_tail, _, _, _ = m.sigma0_upper(u_a[:, r - 1 : r + 3])
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
            per_scale_e = per_scale_e + (
                2.0 * torch.sin(0.5 * db._wrap_to_fp32(delta_e_tail_fp64)).square()
            ).sum(dim=1)
        per_scale_w = omc_w.sum(dim=1)
        per_scale = per_scale_e + per_scale_w

        pad_e = torch.zeros((bsz, 4, 32), dtype=torch.float32, device=device)
        feat_e = torch.cat(
            [torch.cat([sin_e, pad_e], dim=1), torch.cat([omc_e, pad_e], dim=1)], dim=-1
        )
        feat_w = torch.cat([sin_w, omc_w], dim=-1)

        if r > 16:
            sig1_w_spins, comm_ls1, leak_ls1, assoc_ls1 = m.sigma1_lower(u_w[:, 14 : r - 2])
            sig0_w_spins, comm_ls0, leak_ls0, assoc_ls0 = m.sigma0_lower(u_w[:, 1 : r - 15])
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
            assoc_sched = (assoc_ls1.mean(dim=-1).mean(dim=1) + assoc_ls0.mean(dim=-1).mean(dim=1)) / 2.0
        else:
            per_scale_s = torch.zeros_like(per_scale_w)
            feat_s = torch.zeros((bsz, r, 64), dtype=torch.float32, device=device)
            eq_s_xor = torch.zeros(bsz, dtype=torch.float32, device=device)
            leak_sched = torch.zeros(bsz, dtype=torch.float32, device=device)
            comm_sched = torch.zeros(bsz, dtype=torch.float32, device=device)
            assoc_sched = torch.zeros(bsz, dtype=torch.float32, device=device)

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

        # Associator energy across Sigma0/1, sigma0/1, and cross-stream (A, E, W) triples
        assoc_nodes = m.associator_sq(ua_mid, ue_mid, u_w).mean(dim=(1, 2))
        assoc_gates = 0.5 * (assoc_s0.mean(dim=(1, 2)) + assoc_s1.mean(dim=(1, 2)))
        energy_assoc = (
            (assoc_nodes + assoc_gates + assoc_sched) / 3.0
            if r > 16
            else 0.5 * (assoc_nodes + assoc_gates)
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
            + lam_assoc * energy_assoc
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
            energy_assoc,
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
        lam_assoc: float | torch.Tensor = 0.0,
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
            lam_assoc,
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
            "energy_assoc": out[12],
            "energy_prefix": out[13],
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
        lam_assoc: float | torch.Tensor = 0.0,
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
                    lam_assoc,
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
            "energy_assoc": out[12].detach(),
            "energy_prefix": out[13].detach(),
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
        pad_z = torch.zeros_like(u_cur[..., 1:])
        d_spin_amb = torch.cat([d_spin.unsqueeze(-1), pad_z], dim=-1)
        delta_amb = damp_t * d_spin_amb + beta_s * delta_u_ambient.to(torch.float32) - eta_s * grad_riem.to(torch.float32)
        u_raw = self.manifold.retract(u_cur, delta_amb, step_size=1.0)
        return normalize_manifold(u_clamped_masked + free_mask_exp * u_raw)


class V5ABackbone(nn.Module):
    """Exact S^7 Octonion 2D Bit-Grid + BiGRU Backbone (100% shared-layer parity with V4C)."""

    BASE_DIM: int = 16
    ROT_DIM: int = BitGridBlock.ROT_DIM
    IN_DIM: int = BASE_DIM + ROT_DIM

    def __init__(
        self,
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
        self.manifold = OctonionS7Backend()
        self.manifold_dim = 8
        self.trans_dim = 7
        self.rounds = int(rounds)
        self.d_bit = int(d_bit)
        self.grad_checkpointing = bool(grad_checkpointing)

        # Shared V3/V4C core layers (exact name parity)
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

        # Pure direct 8D octonion transverse adapters (42 * 7 = 294 -> d_bit, tangent head d_bit -> 24)
        trans_feat_dim = 42 * self.trans_dim
        self.manifold_in_proj = nn.Linear(trans_feat_dim, d_bit, bias=True)
        nn.init.orthogonal_(self.manifold_in_proj.weight, gain=0.01)
        nn.init.zeros_(self.manifold_in_proj.bias)

        self.manifold_tangent_head = nn.Linear(d_bit, 3 * self.manifold_dim, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.manifold_tangent_head.bias)
            w_view = self.manifold_tangent_head.weight.view(3, self.manifold_dim, d_bit)
            w_view[:, 0, :] = 0.0
            for ax in self.manifold.trans_axes:
                nn.init.orthogonal_(w_view[:, ax, :], gain=0.02)

    def _build_transverse_features(
        self, u_nodes: torch.Tensor, grad_radar: torch.Tensor
    ) -> torch.Tensor:
        u_t = self.manifold.project_transverse(u_nodes)
        g_t = self.manifold.project_transverse(grad_radar)
        bsz, r, _, n_bits, td = u_t.shape
        src = torch.cat(
            [
                u_t.permute(0, 1, 3, 2, 4).reshape(bsz, r, n_bits, 3 * td),
                g_t.permute(0, 1, 3, 2, 4).reshape(bsz, r, n_bits, 3 * td),
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
        u_f, g_f = u_nodes.to(torch.float32), grad_radar.to(torch.float32)
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
            if self.grad_checkpointing and self.training:
                h = checkpoint(block, h, rot_in, cond, self.round_attn_mask, use_reentrant=False)
            else:
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
        non_adapter_missing = [
            k
            for k in incompat.missing_keys
            if not (k.startswith("manifold_in_proj.") or k.startswith("manifold_tangent_head."))
        ]
        if incompat.unexpected_keys or non_adapter_missing:
            raise RuntimeError(
                f"V3->V5A mismatch: unexpected={incompat.unexpected_keys}, missing={non_adapter_missing}"
            )
        self.float()
        return {
            "loaded_keys": len(src_state),
            "missing_adapter_keys": list(incompat.missing_keys),
            "source_step": ckpt.get("step", "?"),
        }


class UnrolledSolverV5A(OuterSearchShell):
    """Test-time solver for V5A S^7 Octonion with associator + progressive real-axis annealing."""

    def __init__(
        self,
        model: V5ABackbone,
        physics: V5APhysicsEngine,
        num_candidates: int = 256,
        num_steps: int = 48,
        damp: float = 0.25,
        eta_max: float = 0.025,
        beta_max: float = 0.25,
        grad_scale: float = 2.0,
        chunk_size: int = 1024,
        w_xor: float = 1.0,
        lam_comm_end: float = 0.60,
        lam_align_end: float = 6.00,
        lam_assoc_end: float = 2.50,
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
        self.lam_align_end = float(lam_align_end)
        self.lam_assoc_end = float(lam_assoc_end)

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
            grad_radar, res_feat, _, grad_riem = self.physics.compute_radar_prepared(
                u_cur,
                prep,
                lam=lam_full,
                grad_scale=self.grad_scale,
                w_xor=self.w_xor,
                lam_comm=anneal_schedule(frac, 0.02, self.lam_comm_end),
                lam_align=anneal_schedule(frac, 0.02, self.lam_align_end),
                lam_assoc=anneal_schedule(frac, 0.02, self.lam_assoc_end),
                return_raw_grad=True,
            )
            sf = torch.full((bsz,), frac, dtype=torch.float32, device=dev)
            logits_last, delta_u, h_tilde = self.model(
                u_cur, grad_radar, prep.clamp_mask, res_feat, self.physics.K_spins, sf, lam_full, h_prev
            )
            h_prev = h_tilde if h_prev is None else (1.0 - self.damp) * h_prev + self.damp * h_tilde
            u_cur = self.physics.retract_and_clamp(
                u_cur,
                logits_last,
                delta_u,
                grad_riem,
                prep.u_clamped_masked,
                prep.free_mask_exp,
                self.damp,
                self.eta_max * (0.2 + 0.8 * frac),
                self.beta_max * (frac * frac),
            )

        spins = self.manifold.project_spin(u_cur)
        x_disc = torch.where(spins < 0.0, -torch.ones_like(spins), torch.ones_like(spins))
        x_disc = boundary.to_clamped_boundary().apply_clamp(x_disc)
        u_a, u_e, u_w = assemble_full_prepared(self.manifold.embed_spins(x_disc), prep, self.manifold)
        _, _, stats = self.physics.compute_energy_and_features(u_a, u_e, u_w, free_mask=prep.free_mask)
        u_a_c, u_e_c, u_w_c = assemble_full_prepared(u_cur, prep, self.manifold)
        _, _, c_stats = self.physics.compute_energy_and_features(u_a_c, u_e_c, u_w_c, free_mask=prep.free_mask)

        return (
            logits_last,
            x_disc,
            stats["m1_energy"],
            stats["m32_energy"],
            torch.zeros_like(stats["m1_energy"]),
            c_stats["energy_align"],
            c_stats["energy_comm"],
            {"zero_assoc": float(c_stats["energy_assoc"].mean().item())},
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
        disc_chunks, e_chunks, m1_chunks, m32_chunks, al_chunks, cm_chunks, as_chunks = [], [], [], [], [], [], []

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
            denom = float(max(1, n_steps - 1))
            for s in range(n_steps):
                frac = float(s) / denom
                grad_radar, res_feat, _, grad_riem = self.physics.compute_radar_prepared(
                    u_cur,
                    prep,
                    lam=lam_full,
                    grad_scale=self.grad_scale,
                    w_xor=self.w_xor,
                    lam_comm=anneal_schedule(frac, 0.02, self.lam_comm_end),
                    lam_align=anneal_schedule(frac, 0.02, self.lam_align_end),
                    lam_assoc=anneal_schedule(frac, 0.02, self.lam_assoc_end),
                    return_raw_grad=True,
                )
                sf = torch.full((m_cur,), frac, dtype=torch.float32, device=dev)
                logits_s, delta_u, h_tilde = self.model(
                    u_cur, grad_radar, prep.clamp_mask, res_feat, self.physics.K_spins, sf, lam_full, h_prev
                )
                h_prev = h_tilde if h_prev is None else (1.0 - self.damp) * h_prev + self.damp * h_tilde
                u_cur = self.physics.retract_and_clamp(
                    u_cur,
                    logits_s,
                    delta_u,
                    grad_riem,
                    prep.u_clamped_masked,
                    prep.free_mask_exp,
                    self.damp,
                    self.eta_max * (0.2 + 0.8 * frac),
                    self.beta_max * (frac * frac),
                )

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
            as_chunks.append(c_stats["energy_assoc"])

        r_n = self.model.rounds
        x_disc_all = torch.cat(disc_chunks, dim=0).view(b_cnt, k_cand, r_n, 3, 32)
        e_all = torch.cat(e_chunks, dim=0).view(b_cnt, k_cand)
        m1_all = torch.cat(m1_chunks, dim=0).view(b_cnt, k_cand)
        m32_all = torch.cat(m32_chunks, dim=0).view(b_cnt, k_cand)
        al_all = torch.cat(al_chunks, dim=0).view(b_cnt, k_cand)
        cm_all = torch.cat(cm_chunks, dim=0).view(b_cnt, k_cand)
        as_all = torch.cat(as_chunks, dim=0)

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
            extra_metrics={"zero_assoc": float(as_all.mean().item())},
        )

    def export_calibrated_probs(self, solve_result: SolveResultV5) -> torch.Tensor:
        if solve_result.calibrated_probs is not None:
            return solve_result.calibrated_probs
        return compute_calibrated_probs_from_logits(1.5 * solve_result.best_x_disc)


__all__ = [
    "OctonionS7Backend",
    "UnrolledSolverV5A",
    "V5ABackbone",
    "V5APhysicsEngine",
    "oct_associator_fp32",
    "oct_mul_fp32",
]
