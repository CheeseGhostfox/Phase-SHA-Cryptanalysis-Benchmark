"""Pure-FP32 Dual-Tower InfoNCE & Linear Baseline Models."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_SCALE_MIN: float = math.log(1.0)
LOG_SCALE_MAX: float = math.log(50.0)
LOG_SCALE_INIT: float = math.log(1.0 / 0.07)


class PreLNResBlock(nn.Module):
    """Pre-LayerNorm residual MLP block: x + Linear(SiLU(Linear(LN(x))))."""

    def __init__(self, hidden_dim: int = 512) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.act(self.fc1(self.ln(x))))
        return x + h


class TowerEncoder(nn.Module):
    """4-block Pre-LN residual MLP encoder projecting onto unit hypersphere S^{D-1}."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 128,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [PreLNResBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        z = self.out_proj(self.out_ln(h))
        return F.normalize(z, p=2, dim=-1, eps=1e-8)


class DualTowerInfoNCE(nn.Module):
    """Dual-tower nonlinear InfoNCE model for (S_0, S_R)."""

    def __init__(
        self,
        in_dim: int = 896,
        out_dim: int = 448,
        hidden_dim: int = 512,
        emb_dim: int = 128,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.f_theta = TowerEncoder(in_dim, hidden_dim, emb_dim, num_blocks)
        self.g_phi = TowerEncoder(out_dim, hidden_dim, emb_dim, num_blocks)
        self.logit_scale = nn.Parameter(
            torch.tensor(LOG_SCALE_INIT, dtype=torch.float32)
        )

    def clamped_scale(self) -> torch.Tensor:
        return torch.exp(self.logit_scale.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX))

    def forward(
        self, phi_in: torch.Tensor, phi_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.f_theta(phi_in)
        v = self.g_phi(phi_out)
        return u, v, self.clamped_scale()


class LinearInfoNCE(nn.Module):
    """Linear dual-tower InfoNCE baseline: u = Norm(W_f x), v = Norm(W_g y)."""

    def __init__(
        self,
        in_dim: int = 896,
        out_dim: int = 448,
        emb_dim: int = 128,
    ) -> None:
        super().__init__()
        self.f_lin = nn.Linear(in_dim, emb_dim)
        self.g_lin = nn.Linear(out_dim, emb_dim)
        self.logit_scale = nn.Parameter(
            torch.tensor(LOG_SCALE_INIT, dtype=torch.float32)
        )

    def clamped_scale(self) -> torch.Tensor:
        return torch.exp(self.logit_scale.clamp(LOG_SCALE_MIN, LOG_SCALE_MAX))

    def forward(
        self, phi_in: torch.Tensor, phi_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = F.normalize(self.f_lin(phi_in), p=2, dim=-1, eps=1e-8)
        v = F.normalize(self.g_lin(phi_out), p=2, dim=-1, eps=1e-8)
        return u, v, self.clamped_scale()


def symmetric_infonce_loss(
    u: torch.Tensor, v: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Symmetric InfoNCE loss in nats (float32)."""
    logits = (u @ v.T) * scale
    labels = torch.arange(u.shape[0], device=u.device, dtype=torch.long)
    loss_uv = F.cross_entropy(logits, labels)
    loss_vu = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_uv + loss_vu)
