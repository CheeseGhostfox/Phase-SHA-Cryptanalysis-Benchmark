"""Bipolar spin operators and hybrid float64-folded 2-adic Fourier phase tower."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.boundary_clamp import ClampedBoundary
from src.sha256_ref import get_k_tensor, words_to_bits, words_to_spins


def spin_xor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Continuous bipolar XOR: x * y."""
    return x * y


def spin_rotr(x: torch.Tensor, n: int) -> torch.Tensor:
    """Circular right rotation by n bits along LSB-first trailing dim (..., 32)."""
    if n == 0:
        return x
    return torch.roll(x, shifts=-n, dims=-1)


def spin_shr(x: torch.Tensor, n: int) -> torch.Tensor:
    """Logical right shift toward index 0, padding upper n bits with +1.0 (logical 0)."""
    if n == 0:
        return x
    if n >= 32:
        return torch.ones_like(x)
    pad = torch.ones(*x.shape[:-1], n, dtype=x.dtype, device=x.device)
    return torch.cat([x[..., n:], pad], dim=-1)


def spin_Sigma0(x: torch.Tensor) -> torch.Tensor:
    """Continuous spin Sigma0(x) = ROTR(2) ^ ROTR(13) ^ ROTR(22)."""
    return spin_rotr(x, 2) * spin_rotr(x, 13) * spin_rotr(x, 22)


def spin_Sigma1(x: torch.Tensor) -> torch.Tensor:
    """Continuous spin Sigma1(x) = ROTR(6) ^ ROTR(11) ^ ROTR(25)."""
    return spin_rotr(x, 6) * spin_rotr(x, 11) * spin_rotr(x, 25)


def spin_sigma0(x: torch.Tensor) -> torch.Tensor:
    """Continuous spin sigma0(x) = ROTR(7) ^ ROTR(18) ^ SHR(3)."""
    return spin_rotr(x, 7) * spin_rotr(x, 18) * spin_shr(x, 3)


def spin_sigma1(x: torch.Tensor) -> torch.Tensor:
    """Continuous spin sigma1(x) = ROTR(17) ^ ROTR(19) ^ SHR(10)."""
    return spin_rotr(x, 17) * spin_rotr(x, 19) * spin_shr(x, 10)


def spin_Ch(e: torch.Tensor, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Continuous spin Ch(e, f, g) = 0.5 * (f + g + e * (g - f))."""
    return 0.5 * (f + g + e * (g - f))


def spin_Maj(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Continuous spin Maj(a, b, c) = 0.5 * (a + b + c - a * b * c)."""
    return 0.5 * (a + b + c - a * b * c)


def build_phase_projection_matrix() -> torch.Tensor:
    """Build P_T of shape (32, 32) float64 where P[m-1, k] = pi / 2^(m-1-k) for k < m."""
    p = torch.zeros((32, 32), dtype=torch.float64)
    for m in range(1, 33):
        for k in range(m):
            p[m - 1, k] = math.pi / float(1 << (m - 1 - k))
    return p.t().contiguous()


class DualBasisEngine(nn.Module):
    """Exact hybrid float64-folded 2-adic Fourier phase residual and gradient engine."""

    def __init__(self, rounds: int = 64) -> None:
        super().__init__()
        if not (8 <= rounds <= 64):
            raise ValueError(f"rounds must be in [8, 64], got {rounds}")
        self.rounds = rounds
        self.num_eq_rounds = (rounds - 4) + rounds + max(0, rounds - 16)

        p_t = build_phase_projection_matrix()
        self.register_buffer("P_T", p_t, persistent=False)

        k_words = get_k_tensor(rounds, device=None)
        k_bits = words_to_bits(k_words).to(torch.float64)  # (R, 32)
        theta_k = torch.matmul(k_bits, p_t)  # (R, 32) float64
        self.register_buffer("theta_K", theta_k, persistent=False)

        k_spins = words_to_spins(k_words, dtype=torch.float32)  # (R, 32)
        self.register_buffer("K_spins", k_spins, persistent=False)

    @torch.amp.autocast("cuda", enabled=False)
    def spins_to_phase_fp64(self, spins: torch.Tensor) -> torch.Tensor:
        """Project bipolar spins (..., 32) to 2-adic phase tower (..., 32) in float64."""
        bits_fp64 = 0.5 * (1.0 - spins.to(torch.float64))
        return torch.matmul(bits_fp64, self.P_T)

    @torch.amp.autocast("cuda", enabled=False)
    def _wrap_to_fp32(self, delta_fp64: torch.Tensor) -> torch.Tensor:
        """Wrap float64 phase deltas onto [-pi, pi) and immediately cast to float32."""
        two_pi = 2.0 * math.pi
        wrapped = torch.remainder(delta_fp64 + math.pi, two_pi) - math.pi
        return wrapped.to(torch.float32)

    @torch.amp.autocast("cuda", enabled=False)
    def compute_wrapped_deltas(
        self,
        A_full: torch.Tensor,
        E_full: torch.Tensor,
        W_full: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute wrapped float32 phase deltas for Equations E, W, and (if R>16) S.

        Inputs:
            A_full: (B, R + 4, 32) spins
            E_full: (B, R + 4, 32) spins
            W_full: (B, R, 32) spins
        """
        r = self.rounds
        a_f32 = A_full.to(torch.float32)
        e_f32 = E_full.to(torch.float32)
        w_f32 = W_full.to(torch.float32)

        # Continuous spin nonlinear terms in float32
        len_e = r - 4
        sig0_a = spin_Sigma0(a_f32[:, 3 : 3 + len_e])
        maj_a = spin_Maj(
            a_f32[:, 3 : 3 + len_e],
            a_f32[:, 2 : 2 + len_e],
            a_f32[:, 1 : 1 + len_e],
        )

        sig1_e = spin_Sigma1(e_f32[:, 3 : 3 + r])
        ch_e = spin_Ch(
            e_f32[:, 3 : 3 + r],
            e_f32[:, 2 : 2 + r],
            e_f32[:, 1 : 1 + r],
        )

        # Linear phase projections and equation sums/diffs in float64
        theta_a = self.spins_to_phase_fp64(a_f32)
        theta_e = self.spins_to_phase_fp64(e_f32)
        theta_w = self.spins_to_phase_fp64(w_f32)

        # Eq 1: Elimination / E-coupling (t = 0 .. R-5)
        delta_e_fp64 = (
            theta_e[:, 4 : 4 + len_e]
            + self.spins_to_phase_fp64(sig0_a)
            + self.spins_to_phase_fp64(maj_a)
            - theta_a[:, 4 : 4 + len_e]
            - theta_a[:, 0:len_e]
        )

        # Eq 2: Round / W-coupling (t = 0 .. R-1)
        delta_w_fp64 = (
            theta_w
            + self.theta_K.unsqueeze(0)
            + self.spins_to_phase_fp64(sig1_e)
            + self.spins_to_phase_fp64(ch_e)
            + theta_e[:, 0:r]
            + theta_a[:, 0:r]
            - theta_e[:, 4 : 4 + r]
        )

        delta_e = self._wrap_to_fp32(delta_e_fp64)
        delta_w = self._wrap_to_fp32(delta_w_fp64)

        delta_s: Optional[torch.Tensor] = None
        if r > 16:
            sig1_w = spin_sigma1(w_f32[:, 14 : r - 2])
            sig0_w = spin_sigma0(w_f32[:, 1 : r - 15])
            delta_s_fp64 = (
                theta_w[:, 16:r]
                - self.spins_to_phase_fp64(sig1_w)
                - theta_w[:, 9 : r - 7]
                - self.spins_to_phase_fp64(sig0_w)
                - theta_w[:, 0 : r - 16]
            )
            delta_s = self._wrap_to_fp32(delta_s_fp64)

        return delta_e, delta_w, delta_s

    def _normalize_lambda(
        self,
        lam: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if lam is None:
            return torch.ones((batch_size, 32), dtype=torch.float32, device=device)
        lam_f = lam.to(device=device, dtype=torch.float32)
        if lam_f.ndim == 1:
            if lam_f.shape != (32,):
                raise ValueError(f"Expected lam shape (32,), got {tuple(lam_f.shape)}")
            return lam_f.unsqueeze(0).expand(batch_size, 32)
        if lam_f.ndim == 2 and lam_f.shape == (batch_size, 32):
            return lam_f
        raise ValueError(
            f"Expected lam shape (32,) or ({batch_size}, 32), got {tuple(lam_f.shape)}"
        )

    def compute_energy_and_features(
        self,
        A_full: torch.Tensor,
        E_full: torch.Tensor,
        W_full: torch.Tensor,
        lam: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute physics energy (B,), round residual features (B, R, 192), and stats."""
        batch_size = A_full.shape[0]
        r = self.rounds
        device = A_full.device
        lam_b = self._normalize_lambda(lam, batch_size, device)

        delta_e, delta_w, delta_s = self.compute_wrapped_deltas(A_full, E_full, W_full)

        # Compute sin and 1 - cos = 2 * sin(delta/2)^2 in float32 (strictly >= 0)
        sin_e = torch.sin(delta_e)
        omc_e = 2.0 * torch.sin(0.5 * delta_e).square()

        sin_w = torch.sin(delta_w)
        omc_w = 2.0 * torch.sin(0.5 * delta_w).square()

        per_scale_e = omc_e.sum(dim=1)  # (B, 32)
        per_scale_w = omc_w.sum(dim=1)  # (B, 32)
        per_scale = per_scale_e + per_scale_w

        # Pad Eq E along rounds from R-4 to R
        pad_e = torch.zeros((batch_size, 4, 32), dtype=torch.float32, device=device)
        feat_e = torch.cat(
            [torch.cat([sin_e, pad_e], dim=1), torch.cat([omc_e, pad_e], dim=1)],
            dim=-1,
        )  # (B, R, 64)
        feat_w = torch.cat([sin_w, omc_w], dim=-1)  # (B, R, 64)

        if delta_s is not None:
            sin_s = torch.sin(delta_s)
            omc_s = 2.0 * torch.sin(0.5 * delta_s).square()
            per_scale_s = omc_s.sum(dim=1)
            per_scale = per_scale + per_scale_s
            pad_s = torch.zeros(
                (batch_size, 16, 32), dtype=torch.float32, device=device
            )
            feat_s = torch.cat(
                [torch.cat([pad_s, sin_s], dim=1), torch.cat([pad_s, omc_s], dim=1)],
                dim=-1,
            )  # (B, R, 64)
        else:
            per_scale_s = torch.zeros_like(per_scale_w)
            feat_s = torch.zeros(
                (batch_size, r, 64), dtype=torch.float32, device=device
            )

        energy = (per_scale * lam_b).sum(dim=-1)  # (B,)
        features = torch.cat([feat_e, feat_w, feat_s], dim=-1)  # (B, R, 192)

        stats = {
            "per_scale": per_scale,
            "per_scale_e": per_scale_e,
            "per_scale_w": per_scale_w,
            "per_scale_s": per_scale_s,
            "m1_energy": per_scale[:, 0],
            "m32_energy": per_scale[:, 31],
        }
        return energy, features, stats

    def compute_radar_and_features(
        self,
        x_nodes: torch.Tensor,
        boundary: ClampedBoundary,
        lam: Optional[torch.Tensor] = None,
        grad_scale: float = 2.0,
        return_raw_grad: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Compute normalized analytical gradient radar, residual features, and energy."""
        with torch.enable_grad():
            x_req = x_nodes.detach().to(torch.float32).requires_grad_(True)
            a_full, e_full, w_full = boundary.assemble_full(x_req)
            energy, features, stats = self.compute_energy_and_features(
                a_full, e_full, w_full, lam=lam
            )
            grad_x = torch.autograd.grad(
                energy.sum(), x_req, create_graph=False, retain_graph=False
            )[0]

        grad_radar = torch.tanh(grad_x.detach() / grad_scale)
        features_det = features.detach()
        energy_det = energy.detach()
        stats_det = {k: v.detach() for k, v in stats.items()}
        if return_raw_grad:
            return (
                grad_radar,
                features_det,
                energy_det,
                stats_det,
                grad_x.detach(),
            )
        return grad_radar, features_det, energy_det, stats_det
