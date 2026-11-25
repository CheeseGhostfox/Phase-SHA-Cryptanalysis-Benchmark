"""Phase-SHA V4 high-dimensional manifold geometry and physics residual engine (strict FP32).

Implements the progressive Abelian / Non-Abelian manifold ladder aligned with
Lean 4 Modules 2 & 5 (formal_proofs/PhaseSHA/Algebraic/):
  - v4a: S^1 in C (unit complex numbers, 2D)
  - v4b: S^2 in su(2) (Hermitian unitary involutive Bloch matrices Mat2C, 3D)
  - v4c: S^3 in H (unit quaternions Quat, 4D)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.boundary_clamp import ClampedBoundary
from src.dual_basis import DualBasisEngine, spin_Ch, spin_Maj, spin_sigma0, spin_sigma1
from src.sha256_ref import words_to_spins
from src.v3_solver import spin_add32

if TYPE_CHECKING:
    from src.v4_boundary import PreparedManifoldBoundary


def enforce_fp32_precision() -> None:
    """Disable TF32 / reduced precision matmul globally as required by V4 contracts."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def normalize_manifold(u: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project vectors onto unit sphere S^{D-1} in strict FP32 without singularities."""
    u_f = u.to(torch.float32)
    norm = torch.linalg.vector_norm(u_f, dim=-1, keepdim=True).clamp_min(eps)
    return u_f / norm


class BaseManifold:
    """Abstract base interface for S^1 (V4A), S^2 Bloch (V4B), and S^3 Quat (V4C)."""

    variant: str
    dim: int
    spin_axis: int
    trans_axes: Tuple[int, ...]
    is_abelian: bool

    def e0_vec(self, device: torch.device) -> torch.Tensor:
        """Unit vector representing discrete spin +1 (bit 0) via pure functional stack."""
        one = torch.ones((), dtype=torch.float32, device=device)
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return torch.stack(
            [one if ax == self.spin_axis else zero for ax in range(self.dim)], dim=0
        )

    def eq_vec(self, device: torch.device) -> torch.Tensor:
        """Neutral equator unit vector with exactly zero spin and unit norm."""
        one = torch.ones((), dtype=torch.float32, device=device)
        zero = torch.zeros((), dtype=torch.float32, device=device)
        primary_t = self.trans_axes[0]
        return torch.stack(
            [one if ax == primary_t else zero for ax in range(self.dim)], dim=0
        )

    def embed_spins(self, spins: torch.Tensor) -> torch.Tensor:
        """Embed real spins s in [-1, 1] onto S^{D-1} poles s * e0 without slice-copy autograd overhead."""
        s = spins.to(torch.float32)
        z = torch.zeros_like(s)
        cols = [s if ax == self.spin_axis else z for ax in range(self.dim)]
        return torch.stack(cols, dim=-1)

    def embed_spins_with_equator(self, spins: torch.Tensor) -> torch.Tensor:
        """Lift bounded spins in [-1, 1] onto S^{D-1} along primary transverse axis."""
        s = spins.to(torch.float32).clamp(-1.0, 1.0)
        trans = torch.sqrt((1.0 - s * s).clamp_min(0.0))
        z = torch.zeros_like(s)
        primary_t = self.trans_axes[0]
        cols = [
            s if ax == self.spin_axis else (trans if ax == primary_t else z)
            for ax in range(self.dim)
        ]
        return normalize_manifold(torch.stack(cols, dim=-1))

    def project_spin(self, u: torch.Tensor) -> torch.Tensor:
        """Extract bipolar spin coordinate s in [-1, 1] along discrete axis."""
        return u.to(torch.float32)[..., self.spin_axis]

    def project_transverse(self, u: torch.Tensor) -> torch.Tensor:
        """Extract transverse coordinates u_perp of shape (..., D - 1)."""
        u_f = u.to(torch.float32)
        return torch.stack([u_f[..., idx] for idx in self.trans_axes], dim=-1)

    def compose_from_spin_and_transverse_dir(
        self, spins: torch.Tensor, u_ref: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Combine target spin s with transverse unit direction extracted from u_ref."""
        s = spins.to(torch.float32).clamp(-1.0, 1.0)
        trans_amp = torch.sqrt((1.0 - s * s).clamp_min(0.0))

        ref_t = self.project_transverse(u_ref)
        ref_norm = torch.linalg.vector_norm(ref_t, dim=-1, keepdim=True)
        z_s = torch.zeros_like(s)
        ones_s = torch.ones_like(s)
        fb_cols = [ones_s if i == 0 else z_s for i in range(len(self.trans_axes))]
        fallback = torch.stack(fb_cols, dim=-1)
        dir_t = torch.where(ref_norm > eps, ref_t / ref_norm.clamp_min(eps), fallback)

        trans_scaled = trans_amp.unsqueeze(-1) * dir_t
        ax_to_t = {ax: i for i, ax in enumerate(self.trans_axes)}
        cols = [
            s if ax == self.spin_axis else trans_scaled[..., ax_to_t[ax]]
            for ax in range(self.dim)
        ]
        return normalize_manifold(torch.stack(cols, dim=-1), eps=eps)

    def tangent_project(self, u: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Orthogonally project ambient delta onto T_u S^{D-1}: delta - <delta, u> u."""
        u_f = u.to(torch.float32)
        d_f = delta.to(torch.float32)
        dot = (d_f * u_f).sum(dim=-1, keepdim=True)
        return d_f - dot * u_f

    def retract(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        step_size: float | torch.Tensor = 1.0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Norm-preserving tangent-space retraction onto S^{D-1}."""
        u_n = normalize_manifold(u, eps=eps)
        d_perp = self.tangent_project(u_n, delta)
        cand = u_n + step_size * d_perp
        return normalize_manifold(cand, eps=eps)

    @torch.amp.autocast("cuda", enabled=False)
    def make_neutral_equator(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        """Construct unit manifold state with zero spin at every bit."""
        eq = self.eq_vec(device).view(*([1] * len(shape)), self.dim)
        return eq.expand(*shape, self.dim).clone()

    @torch.amp.autocast("cuda", enabled=False)
    def sample_random_manifold(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        spin_std: float = 0.35,
    ) -> torch.Tensor:
        """Sample diverse random points on S^{D-1} with non-degenerate transverse exploration."""
        raw = torch.randn(
            (*shape, self.dim), dtype=torch.float32, device=device, generator=generator
        )
        raw[..., self.spin_axis] = raw[..., self.spin_axis] * float(spin_std)
        return normalize_manifold(raw)

    @torch.amp.autocast("cuda", enabled=False)
    def align_sq(self, u: torch.Tensor) -> torch.Tensor:
        """Squared transverse magnitude ||u_perp||_2^2 (0 iff on discrete spin axis)."""
        return self.project_transverse(u).square().sum(dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def rotr(self, u_word: torch.Tensor, n: int) -> torch.Tensor:
        """Circular right rotation by n bits on bit axis dim=-2 for (..., 32, D)."""
        if n == 0:
            return u_word.to(torch.float32)
        return torch.roll(u_word.to(torch.float32), shifts=-n, dims=-2)

    @torch.amp.autocast("cuda", enabled=False)
    def shr(self, u_word: torch.Tensor, n: int) -> torch.Tensor:
        """Logical right shift toward bit 0, padding upper n bits with discrete bit-0 e_0."""
        u_f = u_word.to(torch.float32)
        if n == 0:
            return u_f
        if n >= 32:
            e0 = self.e0_vec(u_f.device).view(*([1] * (u_f.ndim - 1)), self.dim)
            return e0.expand_as(u_f).clone()
        e0 = self.e0_vec(u_f.device).view(*([1] * (u_f.ndim - 2)), 1, self.dim)
        pad_shape = (*u_f.shape[:-2], n, self.dim)
        pad = e0.expand(*pad_shape)
        return torch.cat([u_f[..., n:, :], pad], dim=-2)

    def commutator_sq(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def xor2_residual(
        self, a: torch.Tensor, b: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def xor3_spin(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def xor3_residual(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def xor3_leakage(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(a.shape[:-1], dtype=torch.float32, device=a.device)

    def eq_s_lsb_residual(self, words9: List[torch.Tensor]) -> torch.Tensor:
        """9-operand odd-parity Eq S LSB XOR loop residual against discrete +1 identity."""
        raise NotImplementedError

    def _op3_words(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spins = self.xor3_spin(a, b, c)
        comm = (
            self.commutator_sq(a, b)
            + self.commutator_sq(b, c)
            + self.commutator_sq(a, c)
        )
        leak = self.xor3_leakage(a, b, c)
        return spins, comm, leak

    def sigma0_upper(
        self, u_a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._op3_words(self.rotr(u_a, 2), self.rotr(u_a, 13), self.rotr(u_a, 22))

    def sigma1_upper(
        self, u_e: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._op3_words(self.rotr(u_e, 6), self.rotr(u_e, 11), self.rotr(u_e, 25))

    def sigma0_lower(
        self, u_w: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._op3_words(self.rotr(u_w, 7), self.rotr(u_w, 18), self.shr(u_w, 3))

    def sigma1_lower(
        self, u_w: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._op3_words(self.rotr(u_w, 17), self.rotr(u_w, 19), self.shr(u_w, 10))


class S1ComplexManifold(BaseManifold):
    """V4A: Abelian unit circle S^1 in C, z = (x, y), discrete poles (+-1, 0)."""

    variant = "v4a"
    dim = 2
    spin_axis = 0
    trans_axes = (1,)
    is_abelian = True

    @staticmethod
    @torch.amp.autocast("cuda", enabled=False)
    def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        af = a.to(torch.float32)
        bf = b.to(torch.float32)
        ax, ay = af[..., 0], af[..., 1]
        bx, by = bf[..., 0], bf[..., 1]
        return torch.stack([ax * bx - ay * by, ax * by + ay * bx], dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def commutator_sq(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.zeros(a.shape[:-1], dtype=torch.float32, device=a.device)

    @torch.amp.autocast("cuda", enabled=False)
    def xor2_residual(
        self, a: torch.Tensor, b: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        prod = self.mul(a, b)
        return (prod - y.to(torch.float32)).square().sum(dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def xor3_prod(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        return self.mul(self.mul(a, b), c)

    @torch.amp.autocast("cuda", enabled=False)
    def xor3_spin(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        return self.xor3_prod(a, b, c)[..., 0]

    @torch.amp.autocast("cuda", enabled=False)
    def xor3_residual(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        prod = self.xor3_prod(a, b, c)
        return (prod - y.to(torch.float32)).square().sum(dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def eq_s_lsb_residual(self, words9: List[torch.Tensor]) -> torch.Tensor:
        acc = words9[0].to(torch.float32)
        for op in words9[1:]:
            acc = self.mul(acc, op)
        e0 = self.e0_vec(acc.device).view(*([1] * (acc.ndim - 1)), 2)
        return (acc - e0).square().sum(dim=-1)


class S2BlochManifold(BaseManifold):
    """V4B: Non-Abelian S^2 Bloch Hermitian unitary matrices Mat2C, X = n . sigma."""

    variant = "v4b"
    dim = 3
    spin_axis = 2
    trans_axes = (0, 1)
    is_abelian = False

    @staticmethod
    @torch.amp.autocast("cuda", enabled=False)
    def pauli_triple(
        a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact real-FP32 representation of X_a X_b X_c = (i * tau) I_2 + p . sigma."""
        af = a.to(torch.float32)
        bf = b.to(torch.float32)
        cf = c.to(torch.float32)
        a_cross_b = torch.linalg.cross(af, bf, dim=-1)
        tau = (a_cross_b * cf).sum(dim=-1)  # (...,)
        ab = (af * bf).sum(dim=-1, keepdim=True)
        ac = (af * cf).sum(dim=-1, keepdim=True)
        bc = (bf * cf).sum(dim=-1, keepdim=True)
        p = ab * cf - ac * bf + bc * af  # (..., 3)
        return tau, p

    @torch.amp.autocast("cuda", enabled=False)
    def commutator_sq(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Exact ||[X_a, X_b]||_F^2 = 8 * ||a x b||_2^2."""
        cross = torch.linalg.cross(a.to(torch.float32), b.to(torch.float32), dim=-1)
        return 8.0 * cross.square().sum(dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def xor3_spin(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        """Exact (1/2) Tr(X_a X_b X_c sigma_z) = p_z."""
        _, p = self.pauli_triple(a, b, c)
        return p[..., 2]

    @torch.amp.autocast("cuda", enabled=False)
    def xor3_leakage(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        """Scalar imaginary part tau^2 = ((a x b) . c)^2 outside traceless space."""
        tau, _ = self.pauli_triple(a, b, c)
        return tau.square()

    @torch.amp.autocast("cuda", enabled=False)
    def xor3_residual(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Exact (1/2) ||X_a X_b X_c - X_y||_F^2 = tau^2 + ||p - y||_2^2."""
        tau, p = self.pauli_triple(a, b, c)
        return tau.square() + (p - y.to(torch.float32)).square().sum(dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def xor2_residual(
        self, a: torch.Tensor, b: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Exact (1/2) ||X_a X_b X_y - sigma_z||_F^2 (even-parity check with sigma_z)."""
        tau, p = self.pauli_triple(a, b, y)
        e_z = self.e0_vec(p.device).view(*([1] * (p.ndim - 1)), 3)
        return tau.square() + (p - e_z).square().sum(dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def eq_s_lsb_residual(self, words9: List[torch.Tensor]) -> torch.Tensor:
        """Exact (1/2) ||X_1 ... X_9 - sigma_z||_F^2 in pure real FP32 without complex dtypes.

        Represents M = (ar + i*ai)*I_2 + (vr + i*vi).sigma; right multiplication by X_c gives:
          ar' = vr . c,             ai' = vi . c
          vr' = ar*c - (vi x c),    vi' = ai*c + (vr x c).
        Note: X_1...X_9 is odd-length (9 factors), so discrete poles yield +-sigma_z.
        """
        c0 = words9[0].to(torch.float32)
        ar = torch.zeros(c0.shape[:-1], dtype=torch.float32, device=c0.device)
        ai = torch.zeros_like(ar)
        vr = c0
        vi = torch.zeros_like(c0)
        for op in words9[1:]:
            c = op.to(torch.float32)
            ar_next = (vr * c).sum(dim=-1)
            ai_next = (vi * c).sum(dim=-1)
            vr_next = ar.unsqueeze(-1) * c - torch.linalg.cross(vi, c, dim=-1)
            vi_next = ai.unsqueeze(-1) * c + torch.linalg.cross(vr, c, dim=-1)
            ar, ai, vr, vi = ar_next, ai_next, vr_next, vi_next
        e_z = self.e0_vec(vr.device).view(*([1] * (vr.ndim - 1)), 3)
        return ar.square() + ai.square() + (vr - e_z).square().sum(dim=-1) + vi.square().sum(dim=-1)


class S3QuatManifold(BaseManifold):
    """V4C: Non-Abelian unit quaternions S^3 in H (Lean 4 Quat), q = (w, x, y, z)."""

    variant = "v4c"
    dim = 4
    spin_axis = 0
    trans_axes = (1, 2, 3)
    is_abelian = False

    @staticmethod
    @torch.amp.autocast("cuda", enabled=False)
    def mul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Exact Hamilton quaternion product matching Quat.mul in NonAbelianLiftAndOGP.lean."""
        pf = p.to(torch.float32)
        qf = q.to(torch.float32)
        pw, px, py, pz = pf[..., 0], pf[..., 1], pf[..., 2], pf[..., 3]
        qw, qx, qy, qz = qf[..., 0], qf[..., 1], qf[..., 2], qf[..., 3]
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
    def commutator_sq(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Exact ||[q_a, q_b]||_2^2 = 4 * ||v_a x v_b||_2^2."""
        va = a.to(torch.float32)[..., 1:4]
        vb = b.to(torch.float32)[..., 1:4]
        cross = torch.linalg.cross(va, vb, dim=-1)
        return 4.0 * cross.square().sum(dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def xor2_residual(
        self, a: torch.Tensor, b: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return (self.mul(a, b) - y.to(torch.float32)).square().sum(dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def xor3_prod(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        return self.mul(self.mul(a, b), c)

    @torch.amp.autocast("cuda", enabled=False)
    def xor3_spin(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        return self.xor3_prod(a, b, c)[..., 0]

    @torch.amp.autocast("cuda", enabled=False)
    def xor3_residual(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return (self.xor3_prod(a, b, c) - y.to(torch.float32)).square().sum(dim=-1)

    @torch.amp.autocast("cuda", enabled=False)
    def eq_s_lsb_residual(self, words9: List[torch.Tensor]) -> torch.Tensor:
        acc = words9[0].to(torch.float32)
        for op in words9[1:]:
            acc = self.mul(acc, op)
        e0 = self.e0_vec(acc.device).view(*([1] * (acc.ndim - 1)), 4)
        return (acc - e0).square().sum(dim=-1)


def get_manifold(variant: str) -> BaseManifold:
    """Return manifold geometry object for v4a, v4b, or v4c."""
    key = str(variant).lower()
    if key == "v4a":
        return S1ComplexManifold()
    if key == "v4b":
        return S2BlochManifold()
    if key == "v4c":
        return S3QuatManifold()
    raise ValueError(f"Unsupported V4 variant '{variant}', expected v4a/v4b/v4c.")


@torch.amp.autocast("cuda", enabled=False)
def _spin_add32_flat(x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
    """Single 2D (N, 32) modular ripple-adder entrypoint for standalone torch.compile."""
    return spin_add32(x_a, x_b)


@torch.amp.autocast("cuda", enabled=False)
def apply_manifold_closed_loop_schedule(
    u_nodes: torch.Tensor,
    manifold: BaseManifold,
    rounds: int,
    add32_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """2-wave vectorized closed-loop W_{16..R-1} projection preserving transverse directions."""
    if rounds <= 16:
        return u_nodes.to(torch.float32)

    adder = spin_add32 if add32_fn is None else add32_fn
    u_f = normalize_manifold(u_nodes)
    bsz = u_f.shape[0]
    spins_16 = manifold.project_spin(u_f[:, :16, 2, :, :])  # (B, 16, 32)
    w_cols = [spins_16[:, t, :] for t in range(16)]
    for t_cur in range(16, rounds, 2):
        t_end = min(t_cur + 2, rounds)
        k_wave = t_end - t_cur
        w_m2 = torch.stack([w_cols[t - 2] for t in range(t_cur, t_end)], dim=1)
        w_m7 = torch.stack([w_cols[t - 7] for t in range(t_cur, t_end)], dim=1)
        w_m15 = torch.stack([w_cols[t - 15] for t in range(t_cur, t_end)], dim=1)
        w_m16 = torch.stack([w_cols[t - 16] for t in range(t_cur, t_end)], dim=1)

        s1 = spin_sigma1(w_m2)
        s0 = spin_sigma0(w_m15)
        left_a = torch.cat([s1, s0], dim=1).reshape(2 * bsz * k_wave, 32)
        right_b = torch.cat([w_m7, w_m16], dim=1).reshape(2 * bsz * k_wave, 32)
        inner = adder(left_a, right_b).view(bsz, 2 * k_wave, 32)

        sum_left = inner[:, :k_wave, :].reshape(bsz * k_wave, 32)
        sum_right = inner[:, k_wave:, :].reshape(bsz * k_wave, 32)
        wt_wave = adder(sum_left, sum_right).view(bsz, k_wave, 32)
        for i in range(k_wave):
            w_cols.append(wt_wave[:, i, :])

    w_exp_spins = torch.stack(w_cols[16:rounds], dim=1)  # (B, R-16, 32)
    u_ref = u_f[:, 16:rounds, 2, :, :]
    u_exp = manifold.compose_from_spin_and_transverse_dir(w_exp_spins, u_ref)

    u_w_full = torch.cat([u_f[:, :16, 2, :, :], u_exp], dim=1)
    return torch.stack(
        [u_f[:, :, 0, :, :], u_f[:, :, 1, :, :], u_w_full],
        dim=2,
    )


class ManifoldPhysicsEngine(nn.Module):
    """Differentiable V4 manifold + 32-scale 2-adic Fourier phase tower physics engine."""

    def __init__(self, variant: str = "v4b", rounds: int = 18) -> None:
        super().__init__()
        self.variant = str(variant).lower()
        self.rounds = int(rounds)
        self.manifold = get_manifold(self.variant)
        self.dual_basis = DualBasisEngine(rounds=self.rounds)

    @property
    def K_spins(self) -> torch.Tensor:
        return self.dual_basis.K_spins

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
        """Pure-tensor compilable core returning flat tuple (total_energy, features, ...)."""
        r = self.rounds
        m = self.manifold
        u_a = u_a.to(torch.float32)
        u_e = u_e.to(torch.float32)
        u_w = u_w.to(torch.float32)
        bsz = u_a.shape[0]
        device = u_a.device

        a_spins = m.project_spin(u_a)
        e_spins = m.project_spin(u_e)
        w_spins = m.project_spin(u_w)

        len_e = r - 4
        sig0_a_spins, comm_s0, leak_s0 = m.sigma0_upper(u_a[:, 3 : 3 + len_e])
        maj_a = spin_Maj(
            a_spins[:, 3 : 3 + len_e],
            a_spins[:, 2 : 2 + len_e],
            a_spins[:, 1 : 1 + len_e],
        )

        sig1_e_spins, comm_s1, leak_s1 = m.sigma1_upper(u_e[:, 3 : 3 + r])
        ch_e = spin_Ch(
            e_spins[:, 3 : 3 + r],
            e_spins[:, 2 : 2 + r],
            e_spins[:, 1 : 1 + r],
        )

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
                a_spins[:, r - 1 : r + 3],
                a_spins[:, r - 2 : r + 2],
                a_spins[:, r - 3 : r + 1],
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
            [torch.cat([sin_e, pad_e], dim=1), torch.cat([omc_e, pad_e], dim=1)],
            dim=-1,
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
                [torch.cat([pad_s, sin_s], dim=1), torch.cat([pad_s, omc_s], dim=1)],
                dim=-1,
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

        leak_comp = leak_s0.mean(dim=-1).sum(dim=1) + leak_s1.mean(dim=-1).sum(dim=1)
        energy_xor = eq_s_xor + leak_comp + leak_sched

        if m.is_abelian:
            energy_comm = torch.zeros(bsz, dtype=torch.float32, device=device)
        else:
            ua_mid = u_a[:, 3 : 3 + r]
            ue_mid = u_e[:, 3 : 3 + r]
            comm_node_pairs = (
                m.commutator_sq(ua_mid, ue_mid)
                + m.commutator_sq(ue_mid, u_w)
                + m.commutator_sq(ua_mid, u_w)
            ).mean(dim=(1, 2)) / 3.0
            comm_gates = (
                comm_s0.mean(dim=(1, 2)) / 3.0 + comm_s1.mean(dim=(1, 2)) / 3.0
            ) / 2.0
            if r > 16:
                energy_comm = (comm_node_pairs + comm_gates + comm_sched) / 3.0
            else:
                energy_comm = 0.5 * (comm_node_pairs + comm_gates)

        node_align = torch.stack(
            [
                m.align_sq(u_a[:, 3 : 3 + r]),
                m.align_sq(u_e[:, 3 : 3 + r]),
                m.align_sq(u_w),
            ],
            dim=2,
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
            energy_prefix = ((h_pred_spins - target_spins).square() * t_mask).sum(
                dim=(1, 2)
            )

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
        include_tail_e: bool = False,
        has_digest_prefix: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Evaluate total manifold physics energy (B,), 2-adic residual features (B, R, 192), and stats."""
        lam_b = self.dual_basis._normalize_lambda(lam, U_A_full.shape[0], U_A_full.device)
        if has_digest_prefix is None:
            has_digest_prefix = bool(
                digest_target_mask is not None
                and digest_target_bits is not None
                and iv_words is not None
                and float(digest_target_mask.max().item()) > 0.0
            )
        (
            total_energy,
            features,
            per_scale,
            per_scale_e,
            per_scale_w,
            per_scale_s,
            m1_e,
            m32_e,
            energy_2adic,
            energy_xor,
            energy_comm,
            energy_align,
            energy_prefix,
        ) = self.compute_energy_and_features_core(
            u_a=U_A_full,
            u_e=U_E_full,
            u_w=U_W_full,
            free_mask=free_mask,
            lam_b=lam_b,
            w_xor=w_xor,
            lam_comm=lam_comm,
            lam_align=lam_align,
            digest_target_bits=digest_target_bits,
            digest_target_mask=digest_target_mask,
            iv_words=iv_words,
            w_digest_prefix=w_digest_prefix,
            include_tail_e=include_tail_e,
            has_digest_prefix=has_digest_prefix,
        )
        stats = {
            "per_scale": per_scale,
            "per_scale_e": per_scale_e,
            "per_scale_w": per_scale_w,
            "per_scale_s": per_scale_s,
            "m1_energy": m1_e,
            "m32_energy": m32_e,
            "energy_2adic": energy_2adic,
            "energy_xor": energy_xor,
            "energy_comm": energy_comm,
            "energy_align": energy_align,
            "energy_prefix": energy_prefix,
        }
        return total_energy, features, stats

    @torch.amp.autocast("cuda", enabled=False)
    def retract_and_clamp(
        self,
        u_cur: torch.Tensor,
        logits_s: torch.Tensor,
        delta_u_ambient: torch.Tensor,
        grad_riem: torch.Tensor,
        u_clamped_masked: torch.Tensor,
        free_mask_exp: torch.Tensor,
        damp_t: torch.Tensor,
        eta_s: torch.Tensor,
        beta_s: torch.Tensor,
    ) -> torch.Tensor:
        """Compilable pure-tensor manifold retraction + boundary clamping."""
        m = self.manifold
        x_spin_target = torch.tanh(logits_s)
        spin_pull = m.embed_spins(x_spin_target - m.project_spin(u_cur))
        neural_drive = spin_pull + delta_u_ambient

        x_safe = m.project_spin(u_cur).detach().clamp(-2.0, 2.0)
        grad_bin = m.embed_spins(-4.0 * x_safe * (1.0 - x_safe * x_safe))
        phys_drive = torch.clamp(grad_riem + beta_s * grad_bin, -10.0, 10.0)

        u_next = m.retract(u_cur, neural_drive - eta_s * phys_drive, step_size=damp_t)
        return normalize_manifold(u_clamped_masked + free_mask_exp * u_next.to(torch.float32))

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
        energy_fn: Optional[Any] = None,
        return_raw_grad: bool = True,
        clone_cudagraph_outputs: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Fast prepared-boundary radar path compatible with compiled physics core."""
        from src.v4_boundary import assemble_full_prepared

        core_fn = energy_fn if energy_fn is not None else self.compute_energy_and_features_core
        lam_b = self.dual_basis._normalize_lambda(lam, u_nodes.shape[0], u_nodes.device)
        with torch.enable_grad():
            u_req = u_nodes.detach().to(torch.float32).requires_grad_(True)
            u_a, u_e, u_w = assemble_full_prepared(u_req, prep, self.manifold)
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            core_out = core_fn(
                u_a,
                u_e,
                u_w,
                prep.free_mask,
                lam_b,
                w_xor,
                lam_comm,
                lam_align,
                prep.digest_target_bits,
                prep.digest_target_mask,
                prep.iv_words,
                4.0,
                prep.has_free_right,
                prep.has_digest_prefix,
            )
            total_e, features = core_out[0], core_out[1]
            grad_euclid = torch.autograd.grad(
                total_e.sum(), u_req, create_graph=False, retain_graph=False
            )[0]
            grad_riem = self.manifold.tangent_project(u_req.detach(), grad_euclid.detach())

        grad_radar = torch.tanh(grad_riem / float(grad_scale))
        feat_out = features.detach().clone() if clone_cudagraph_outputs else features.detach()
        e_out = total_e.detach().clone() if clone_cudagraph_outputs else total_e.detach()
        if return_raw_grad:
            return grad_radar, feat_out, e_out, grad_riem.detach()
        return grad_radar, feat_out, e_out

    @torch.amp.autocast("cuda", enabled=False)
    def compute_manifold_radar_and_features(
        self,
        u_nodes: torch.Tensor,
        boundary: "Any",
        lam: Optional[torch.Tensor] = None,
        grad_scale: float = 2.0,
        w_xor: float = 1.0,
        lam_comm: float = 0.0,
        lam_align: float = 0.0,
        return_raw_grad: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Compute Riemannian gradient radar (B, R, 3, 32, D), residual features, and energies."""
        with torch.enable_grad():
            u_req = u_nodes.detach().to(torch.float32).requires_grad_(True)
            u_a, u_e, u_w = boundary.assemble_full_manifold(u_req, self.manifold)
            free_mask = 1.0 - boundary.clamp_mask
            total_e, features, stats = self.compute_energy_and_features(
                u_a,
                u_e,
                u_w,
                free_mask=free_mask,
                lam=lam,
                digest_target_bits=boundary.digest_target_bits,
                digest_target_mask=boundary.digest_target_mask,
                iv_words=boundary.iv_words,
                w_xor=w_xor,
                lam_comm=lam_comm,
                lam_align=lam_align,
                include_tail_e=boundary.has_free_right_boundary,
            )
            grad_euclid = torch.autograd.grad(
                total_e.sum(), u_req, create_graph=False, retain_graph=False
            )[0]
            grad_riem = self.manifold.tangent_project(u_req.detach(), grad_euclid.detach())

        grad_radar = torch.tanh(grad_riem / float(grad_scale))
        features_det = features.detach()
        energy_det = total_e.detach()
        stats_det = {k: v.detach() for k, v in stats.items()}
        if return_raw_grad:
            return grad_radar, features_det, energy_det, stats_det, grad_riem.detach()
        return grad_radar, features_det, energy_det, stats_det
