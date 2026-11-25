"""256-D Volume-Preserving Flat-Torus Coupling Flow (T^256 Flow)."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v8a import configure_strict_fp32

TWO_PI = 2.0 * math.pi


def wrap_2pi(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angles cleanly into [0, 2*pi)."""
    return torch.remainder(theta, TWO_PI)


def circular_abs_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact shortest geodesic distance on S^1 in [0, pi]."""
    return torch.abs(torch.remainder(a - b + math.pi, TWO_PI) - math.pi)


def dequantize_bits(
    bits: torch.Tensor,
    deterministic: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Map binary bits {0, 1} to T^256 with uniform cell noise u ~ U(-pi/2, +pi/2)."""
    if deterministic:
        return torch.remainder(math.pi * bits.to(torch.float32), TWO_PI)
    u = (
        torch.rand(bits.shape, device=bits.device, dtype=torch.float32, generator=generator) - 0.5
    ) * math.pi
    return torch.remainder(math.pi * bits.to(torch.float32) + u, TWO_PI)


def build_coupling_masks(num_layers: int, seed: int = 42) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Build balanced 128/128 bipartition masks cycling across word, bit, and random splits."""
    all_idx = torch.arange(256, dtype=torch.long)
    word_id = all_idx // 32
    bit_in_word = all_idx % 32

    def _pair(mask_a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        idx_a = all_idx[mask_a]
        idx_b = all_idx[~mask_a]
        if idx_a.numel() != 128 or idx_b.numel() != 128:
            raise AssertionError("Mask must produce an exact 128/128 bipartition")
        return idx_a, idx_b

    base_patterns = [
        _pair((word_id % 2) == 0),          # word checkerboard: {0,2,4,6} vs {1,3,5,7}
        _pair(word_id < 4),                 # Feistel half-state: {0..3} vs {4..7}
        _pair((word_id // 2) % 2 == 0),     # word-pair split: {0,1,4,5} vs {2,3,6,7}
        _pair((bit_in_word % 2) == 0),      # intra-word bit checkerboard
        _pair(bit_in_word < 16),            # intra-word high-16 vs low-16
    ]

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    masks: List[Tuple[torch.Tensor, torch.Tensor]] = []
    pair_idx = 0
    while len(masks) < num_layers:
        pat_slot = pair_idx % 6
        if pat_slot < 5:
            ia, ib = base_patterns[pat_slot]
        else:
            perm = torch.randperm(256, generator=gen)
            ia, ib = perm[:128].sort().values, perm[128:].sort().values
        masks.append((ia, ib))
        if len(masks) < num_layers:
            masks.append((ib, ia))
        pair_idx += 1
    return masks


class SwiGLUResBlock(nn.Module):
    """Pre-LN multiplicative SwiGLU residual block matching torus trigonometric products."""

    def __init__(self, hidden_dim: int, ff_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.w_gate = nn.Linear(hidden_dim, ff_dim)
        self.w_up = nn.Linear(hidden_dim, ff_dim)
        self.w_down = nn.Linear(ff_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))


class CrossGroupAttention(nn.Module):
    """Lightweight 8-group self-attention enabling global cross-register coupling."""

    def __init__(self, hidden_dim: int, num_groups: int = 8, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_dim % num_groups != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_groups={num_groups}")
        group_dim = hidden_dim // num_groups
        if group_dim % num_heads != 0:
            raise ValueError(f"group_dim={group_dim} must be divisible by num_heads={num_heads}")
        self.num_groups = num_groups
        self.group_dim = group_dim
        self.num_heads = num_heads
        self.head_dim = group_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.group_pos = nn.Parameter(torch.randn(1, num_groups, group_dim) * 0.02)
        self.qkv = nn.Linear(group_dim, 3 * group_dim)
        self.proj = nn.Linear(group_dim, group_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        h = self.norm(x).view(bsz, self.num_groups, self.group_dim) + self.group_pos
        qkv = self.qkv(h).view(bsz, self.num_groups, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B, H, G, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(bsz, self.num_groups, self.group_dim)
        out = self.proj(out).view(bsz, self.num_groups * self.group_dim)
        return x + out


class TorusConditioner(nn.Module):
    """Conditioner network f_phi(cos theta_A, sin theta_A) -> bounded phase shift delta_B."""

    def __init__(
        self,
        hidden_dim: int = 384,
        ff_dim: int = 512,
        num_blocks: int = 2,
        use_attn: bool = True,
        max_shift: float = 2.0 * math.pi,
        beta_hi: float = 6.0,
    ) -> None:
        super().__init__()
        self.max_shift = float(max_shift)
        self.beta_hi = float(beta_hi)
        self.in_proj = nn.Linear(512, hidden_dim)
        self.attn = CrossGroupAttention(hidden_dim) if use_attn else None
        self.blocks = nn.ModuleList(
            [SwiGLUResBlock(hidden_dim, ff_dim) for _ in range(num_blocks)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 128)
        self.skip_linear = nn.Linear(128, 128)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.skip_linear.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.skip_linear.bias)

    def forward(self, theta_a: torch.Tensor) -> torch.Tensor:
        c = torch.cos(theta_a)
        s = torch.sin(theta_a)
        c3 = torch.tanh(3.0 * c)
        c_hi = torch.tanh(self.beta_hi * c)
        feat = torch.cat([c, s, c3, c_hi], dim=-1)
        h = self.in_proj(feat)
        if self.attn is not None:
            h = self.attn(h)
        for blk in self.blocks:
            h = blk(h)
        raw = self.out_proj(self.out_norm(h)) + self.skip_linear(c_hi)
        return self.max_shift * torch.tanh(raw / self.max_shift)


class TorusCouplingLayer(nn.Module):
    """Pure phase-translation affine-free coupling layer on T^256 (det J === +1)."""

    def __init__(
        self,
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
        hidden_dim: int = 384,
        ff_dim: int = 512,
        num_blocks: int = 2,
        use_attn: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("idx_a", idx_a.clone())
        self.register_buffer("idx_b", idx_b.clone())
        self.conditioner = TorusConditioner(
            hidden_dim=hidden_dim,
            ff_dim=ff_dim,
            num_blocks=num_blocks,
            use_attn=use_attn,
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        theta_a = theta.index_select(1, self.idx_a)
        theta_b = theta.index_select(1, self.idx_b)
        delta_b = self.conditioner(theta_a)
        out = theta.clone()
        out[:, self.idx_b] = wrap_2pi(theta_b + delta_b)
        return out

    def inverse(self, theta: torch.Tensor) -> torch.Tensor:
        theta_a = theta.index_select(1, self.idx_a)
        theta_b = theta.index_select(1, self.idx_b)
        delta_b = self.conditioner(theta_a)
        out = theta.clone()
        out[:, self.idx_b] = wrap_2pi(theta_b - delta_b)
        return out


class TorusFlow(nn.Module):
    """Exact volume-preserving bijective flow on the 256-dimensional flat torus T^256."""

    def __init__(
        self,
        num_layers: int = 24,
        hidden_dim: int = 384,
        ff_dim: int = 512,
        num_blocks: int = 2,
        use_attn: bool = True,
        mask_seed: int = 42,
    ) -> None:
        configure_strict_fp32()
        super().__init__()
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.ff_dim = int(ff_dim)
        self.num_blocks = int(num_blocks)
        self.use_attn = bool(use_attn)

        masks = build_coupling_masks(self.num_layers, seed=mask_seed)
        self.layers = nn.ModuleList(
            [
                TorusCouplingLayer(
                    idx_a=ia,
                    idx_b=ib,
                    hidden_dim=self.hidden_dim,
                    ff_dim=self.ff_dim,
                    num_blocks=self.num_blocks,
                    use_attn=self.use_attn,
                )
                for ia, ib in masks
            ]
        )

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        theta: torch.Tensor,
        stage_endpoints: Sequence[int] | None = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, list[torch.Tensor]]:
        x = wrap_2pi(theta)
        if stage_endpoints is None:
            for layer in self.layers:
                x = layer(x)
            return x

        endpoint_map: dict[int, list[int]] = {}
        for pos, ep in enumerate(stage_endpoints):
            endpoint_map.setdefault(int(ep), []).append(pos)
        collected: list[torch.Tensor | None] = [None] * len(stage_endpoints)

        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x)
            if idx in endpoint_map:
                for pos in endpoint_map[idx]:
                    collected[pos] = x
        return x, [t for t in collected if t is not None]

    def forward_slice(self, theta: torch.Tensor, start_layer: int, end_layer: int) -> torch.Tensor:
        """Apply contiguous coupling layers [start_layer, end_layer) for local transport."""
        x = wrap_2pi(theta)
        for idx in range(start_layer, end_layer):
            x = self.layers[idx](x)
        return x

    def inverse(self, theta: torch.Tensor) -> torch.Tensor:
        x = wrap_2pi(theta)
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def compute_tangent_jacobian(self, theta_single: torch.Tensor) -> torch.Tensor:
        """Compute exact 256x256 Riemannian tangent Jacobian at a single point on T^256."""
        if theta_single.shape != (256,):
            raise ValueError(f"Expected single point shape (256,), got {tuple(theta_single.shape)}")
        x0 = theta_single.detach().clone().requires_grad_(True)
        y = self.forward(x0.unsqueeze(0)).squeeze(0)
        cos_y = torch.cos(y)
        sin_y = torch.sin(y)

        rows: list[torch.Tensor] = []
        for i in range(256):
            g_sin = torch.autograd.grad(sin_y[i], x0, retain_graph=True)[0]
            g_cos = torch.autograd.grad(cos_y[i], x0, retain_graph=(i < 255))[0]
            # d(theta'_i)/d(theta) = cos(theta'_i) * d(sin)/d(theta) - sin(theta'_i) * d(cos)/d(theta)
            rows.append(cos_y[i].detach() * g_sin - sin_y[i].detach() * g_cos)
        return torch.stack(rows, dim=0)
