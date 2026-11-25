"""V8B Native Domino-ARX Volume-Preserving Torus Coupling Flow on T^256."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from v8a import configure_strict_fp32
from v8a.torus_flow import TWO_PI, SwiGLUResBlock

TWO_PI_F64 = 2.0 * math.pi
MAX_SHIFT = 1.5 * math.pi


def add_phase_exact(theta_b: torch.Tensor, delta_b: torch.Tensor) -> torch.Tensor:
    """Lossless float64 modular phase addition preserving original tensor dtype."""
    out = torch.remainder(theta_b.to(torch.float64) + delta_b.to(torch.float64), TWO_PI_F64)
    return out.to(theta_b.dtype)


def sub_phase_exact(theta_b: torch.Tensor, delta_b: torch.Tensor) -> torch.Tensor:
    """Lossless float64 modular phase subtraction preserving original tensor dtype."""
    out = torch.remainder(theta_b.to(torch.float64) - delta_b.to(torch.float64), TWO_PI_F64)
    return out.to(theta_b.dtype)


class DominoARXConditioner(nn.Module):
    """Explicit ARX bilinear basis + zero-init Kogge-Stone carry tree + global SwiGLU conditioner."""

    def __init__(
        self,
        mask_tgt: torch.Tensor,
        c_bit: int = 128,
        c_mid: int = 96,
        d_glob: int = 384,
        ff_glob: int = 512,
        num_arx_blocks: int = 2,
    ) -> None:
        super().__init__()
        mask_tgt_b = mask_tgt.to(torch.bool).view(8, 32)
        mask_ctx_b = ~mask_tgt_b
        self.register_buffer("mask_tgt_bool", mask_tgt_b)
        self.register_buffer("mask_ctx_bool", mask_ctx_b)
        self.register_buffer("mask_tgt", mask_tgt_b.to(torch.float32).unsqueeze(0))
        self.register_buffer("mask_ctx", mask_ctx_b.to(torch.float32).unsqueeze(0))

        pair_idx = [(i, j) for i in range(8) for j in range(i + 1, 8)]
        self.register_buffer("pi_i", torch.tensor([p[0] for p in pair_idx], dtype=torch.long))
        self.register_buffer("pi_j", torch.tensor([p[1] for p in pair_idx], dtype=torch.long))

        # 8*(cos, sin, soft, mask)=32 + 28 cross-word pairs + 8*31=248 circular shift pairs = 308
        in_ch = 32 + len(pair_idx) + 8 * 31
        self.in_c = nn.Conv1d(in_ch, c_bit, kernel_size=1)
        self.bit_pos = nn.Parameter(torch.randn(1, c_bit, 32) * 0.02)

        self.mul_norms = nn.ModuleList([nn.GroupNorm(8, c_bit) for _ in range(num_arx_blocks)])
        self.mul_in = nn.ModuleList(
            [nn.Conv1d(c_bit, 2 * c_mid, kernel_size=1) for _ in range(num_arx_blocks)]
        )
        self.mul_out = nn.ModuleList(
            [nn.Conv1d(c_mid, c_bit, kernel_size=1) for _ in range(num_arx_blocks)]
        )

        # Zero-init 3-stage Kogge-Stone causal LSB->MSB carry residual blocks (dilations 1, 2, 4)
        self.carry_dils = (1, 2, 4)
        self.carry_norms = nn.ModuleList([nn.GroupNorm(8, c_bit) for _ in self.carry_dils])
        self.carry_convs = nn.ModuleList(
            [nn.Conv1d(c_bit, c_bit, kernel_size=3, padding=0, groups=c_bit) for _ in self.carry_dils]
        )
        self.carry_in = nn.ModuleList(
            [nn.Conv1d(c_bit, 2 * c_mid, kernel_size=1) for _ in self.carry_dils]
        )
        self.carry_out = nn.ModuleList(
            [nn.Conv1d(c_mid, c_bit, kernel_size=1) for _ in self.carry_dils]
        )
        for co in self.carry_out:
            nn.init.zeros_(co.weight)
            nn.init.zeros_(co.bias)

        self.arx_out_norm = nn.GroupNorm(8, c_bit)
        self.arx_out = nn.Conv1d(c_bit, 8, kernel_size=1)
        nn.init.normal_(self.arx_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.arx_out.bias)

        # Register-global SwiGLU branch: 8*32*3 = 768
        self.glob_in = nn.Linear(768, d_glob)
        self.glob_block = SwiGLUResBlock(d_glob, ff_glob)
        self.glob_norm = nn.LayerNorm(d_glob)
        self.glob_out = nn.Linear(d_glob, 256)
        nn.init.normal_(self.glob_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.glob_out.bias)

    def forward(self, theta_w: torch.Tensor) -> torch.Tensor:
        bsz = theta_w.shape[0]
        theta_f32 = theta_w.to(torch.float32) if theta_w.dtype != torch.float32 else theta_w
        theta_ctx = torch.where(
            self.mask_ctx_bool.unsqueeze(0), theta_f32, torch.zeros_like(theta_f32)
        )
        c = torch.cos(theta_ctx) * self.mask_ctx
        s = torch.sin(theta_ctx) * self.mask_ctx
        soft = torch.tanh(2.0 * c) * self.mask_ctx
        mask_exp = self.mask_ctx.expand(bsz, -1, -1)

        cross = c[:, self.pi_i, :] * c[:, self.pi_j, :]
        shifts = torch.cat([c * torch.roll(c, shifts=sh, dims=-1) for sh in range(1, 32)], dim=1)
        feat = torch.cat([c, s, soft, mask_exp, cross, shifts], dim=1)

        h = self.in_c(feat) + self.bit_pos
        for gn, m_in, m_out in zip(self.mul_norms, self.mul_in, self.mul_out):
            u, v = m_in(gn(h)).chunk(2, dim=1)
            us = 3.0 * torch.tanh(u / 3.0)
            vs = 3.0 * torch.tanh(v / 3.0)
            h = h + m_out(us * vs)

        for dil, gn, c_conv, c_in, c_out in zip(
            self.carry_dils, self.carry_norms, self.carry_convs, self.carry_in, self.carry_out
        ):
            z = gn(h)
            z_pad = F.pad(z, (0, 2 * dil))
            zc = F.conv1d(
                z_pad, c_conv.weight, c_conv.bias, stride=1, padding=0, dilation=dil, groups=z.shape[1]
            )
            u, v = c_in(z + zc).chunk(2, dim=1)
            us = 3.0 * torch.tanh(u / 3.0)
            vs = 3.0 * torch.tanh(v / 3.0)
            h = h + c_out(us * vs + F.silu(u))

        arx_delta = self.arx_out(self.arx_out_norm(h))

        glob_in = torch.cat([c, s, soft], dim=1).reshape(bsz, 768)
        g = self.glob_block(self.glob_in(glob_in))
        glob_delta = self.glob_out(self.glob_norm(g)).view(bsz, 8, 32)

        raw = arx_delta + glob_delta
        return MAX_SHIFT * torch.tanh(raw / MAX_SHIFT) * self.mask_tgt


class V8BCouplingLayer(nn.Module):
    """Exact volume-preserving V8B ARX phase-shift layer on (B, 8, 32)."""

    def __init__(
        self,
        mask_tgt: torch.Tensor,
        c_bit: int = 128,
        c_mid: int = 96,
        d_glob: int = 384,
        ff_glob: int = 512,
        num_arx_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.conditioner = DominoARXConditioner(
            mask_tgt=mask_tgt,
            c_bit=c_bit,
            c_mid=c_mid,
            d_glob=d_glob,
            ff_glob=ff_glob,
            num_arx_blocks=num_arx_blocks,
        )
        self.tgt_count = float(mask_tgt.sum().item())

    def forward(self, theta_w: torch.Tensor) -> torch.Tensor:
        delta = self.conditioner(theta_w)
        return add_phase_exact(theta_w, delta)

    def inverse(self, theta_w: torch.Tensor) -> torch.Tensor:
        delta = self.conditioner(theta_w)
        return sub_phase_exact(theta_w, delta)

    def compute_local_lipschitz(
        self, theta_w_sub: torch.Tensor, sigma: float = 0.15
    ) -> torch.Tensor:
        x32 = theta_w_sub.to(torch.float32) if theta_w_sub.dtype != torch.float32 else theta_w_sub
        noise = torch.randn_like(x32) * sigma
        d_clean = self.conditioner(x32)
        d_noisy = self.conditioner(x32 + noise)
        diff = (d_noisy - d_clean) * self.conditioner.mask_tgt
        return diff.pow(2).sum(dim=(1, 2)).mean() / (self.tgt_count * (sigma * sigma))


class RegisterRollLayer(nn.Module):
    """Parameter-free exact volume-preserving 32-bit word permutation (det J = +1)."""

    def __init__(self, shift: int = -1) -> None:
        super().__init__()
        self.shift = int(shift)

    def forward(self, theta_w: torch.Tensor) -> torch.Tensor:
        return torch.roll(theta_w, shifts=self.shift, dims=1)

    def inverse(self, theta_w: torch.Tensor) -> torch.Tensor:
        return torch.roll(theta_w, shifts=-self.shift, dims=1)


def _mask_words(word_ids: Sequence[int]) -> torch.Tensor:
    m = torch.zeros((8, 32), dtype=torch.bool)
    for w in word_ids:
        m[int(w), :] = True
    return m


def _mask_words_bit_checkerboard(word_ids: Sequence[int], parity: int) -> torch.Tensor:
    m = torch.zeros((8, 32), dtype=torch.bool)
    b = (torch.arange(32) % 2) == (int(parity) % 2)
    for w in word_ids:
        m[int(w), b] = True
    return m


def _mask_bit_checkerboard(parity: int) -> torch.Tensor:
    w = torch.arange(8).unsqueeze(1).expand(8, 32)
    b = torch.arange(32).unsqueeze(0).expand(8, 32)
    return (((w // 4) + b) % 2) == (int(parity) % 2)


def _mask_bit_half(hi_or_lo: int) -> torch.Tensor:
    b = torch.arange(32).unsqueeze(0).expand(8, 32)
    return (b < 16) if (int(hi_or_lo) == 0) else (b >= 16)


class V8BTorusFlow(nn.Module):
    """V8B Native Domino-ARX Bijective Flow supporting Track-1 and Track-2 half-round transport."""

    def __init__(
        self,
        flow_layout: str = "v8b_t1",
        rounds: int = 2,
        num_layers: int = 24,
        c_bit: int = 128,
        c_mid: int = 96,
        d_glob: int = 384,
        ff_glob: int = 512,
        num_arx_blocks: int = 2,
        use_checkpoint: bool = False,
    ) -> None:
        configure_strict_fp32()
        super().__init__()
        if flow_layout not in ("v8b_t1", "v8b_t2"):
            raise ValueError(f"Unsupported flow_layout={flow_layout}")
        self.flow_layout = flow_layout
        self.rounds = int(rounds)
        self.use_checkpoint = bool(use_checkpoint)
        self.transport_stage_slices: List[Tuple[int, int]] = []
        self.transport_target_rounds: List[int] = []
        self.transport_stage_kinds: List[str] = []

        layers: List[nn.Module] = []

        def _add_coupling(mask: torch.Tensor) -> None:
            layers.append(
                V8BCouplingLayer(
                    mask_tgt=mask,
                    c_bit=c_bit,
                    c_mid=c_mid,
                    d_glob=d_glob,
                    ff_glob=ff_glob,
                    num_arx_blocks=num_arx_blocks,
                )
            )

        if flow_layout == "v8b_t1":
            cycle_masks = [
                _mask_words((4, 5, 6, 7)),
                _mask_words((0, 4)),
                _mask_words((1, 5)),
                _mask_words((2, 6)),
                _mask_words((3, 7)),
                _mask_bit_checkerboard(0),
                _mask_words((0, 1, 2, 3)),
                _mask_words((0, 4)),
                _mask_words((1, 5)),
                _mask_words((2, 6)),
                _mask_words((3, 7)),
                _mask_bit_checkerboard(1),
            ]
            for i in range(int(num_layers)):
                _add_coupling(cycle_masks[i % len(cycle_masks)])
        else:
            # Stage 0 (g_dm): H_R -> S_R (+IV removal via bit-split layers)
            s0_start = len(layers)
            for m in (
                _mask_bit_checkerboard(0),
                _mask_bit_checkerboard(1),
                _mask_bit_half(0),
                _mask_bit_half(1),
            ):
                _add_coupling(m)
            self.transport_stage_slices.append((s0_start, len(layers)))
            self.transport_target_rounds.append(self.rounds)
            self.transport_stage_kinds.append("full")

            # Stages 1..R-1: Half-round S_t -> S_t^{half} (W0=T1) -> S_{t-1} (6 copy words untouched!)
            for target_r in range(self.rounds - 1, 0, -1):
                src_round = target_r + 1
                half_start = len(layers)
                _add_coupling(_mask_words((0,)))
                if self.rounds <= 8:
                    _add_coupling(_mask_words_bit_checkerboard((0,), 0))
                half_end = len(layers)
                self.transport_stage_slices.append((half_start, half_end))
                self.transport_target_rounds.append(src_round)
                self.transport_stage_kinds.append("half_t1")

                full_start = len(layers)
                _add_coupling(_mask_words((4,)))
                _add_coupling(_mask_words((0,)))
                _add_coupling(_mask_words_bit_checkerboard((0, 4), 1))
                layers.append(RegisterRollLayer(shift=-1))
                full_end = len(layers)
                self.transport_stage_slices.append((full_start, full_end))
                self.transport_target_rounds.append(target_r)
                self.transport_stage_kinds.append("full")

            # Tail unsupervised disentanglement block g_tail on S_1 -> W0 condensed state
            tail_n = 4 if self.rounds >= 8 else 6
            tail_masks = [
                _mask_words((4, 5, 6, 7)),
                _mask_words((0, 4)),
                _mask_bit_checkerboard(0),
                _mask_bit_checkerboard(1),
                _mask_words((0, 1, 2, 3)),
                _mask_words((0, 4)),
            ]
            for i in range(tail_n):
                _add_coupling(tail_masks[i % len(tail_masks)])

        self.layers = nn.ModuleList(layers)
        self.num_coupling_layers = sum(
            1 for m in self.layers if isinstance(m, V8BCouplingLayer)
        )

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _run_layer(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and isinstance(layer, V8BCouplingLayer):
            return ckpt.checkpoint(layer, x, use_reentrant=False)
        return layer(x)

    def forward(
        self,
        theta: torch.Tensor,
        stage_endpoints: Sequence[int] | None = None,
        return_lip: bool = False,
        lip_sigma: float = 0.15,
        lip_samples: int = 64,
    ):
        bsz = theta.shape[0]
        x = torch.remainder(theta.to(torch.float64), TWO_PI_F64).view(bsz, 8, 32)
        lip_terms: List[torch.Tensor] = []
        n_lip = min(lip_samples, bsz)

        endpoint_map: dict[int, list[int]] = {}
        collected: list[torch.Tensor | None] = []
        if stage_endpoints is not None:
            for pos, ep in enumerate(stage_endpoints):
                endpoint_map.setdefault(int(ep), []).append(pos)
            collected = [None] * len(stage_endpoints)

        for idx, layer in enumerate(self.layers, start=1):
            if return_lip and self.training and isinstance(layer, V8BCouplingLayer):
                lip_terms.append(
                    layer.compute_local_lipschitz(x[:n_lip].to(torch.float32).detach(), sigma=lip_sigma)
                )
            x = self._run_layer(layer, x)
            if stage_endpoints is not None and idx in endpoint_map:
                flat_x = x.to(torch.float32).reshape(bsz, 256)
                for pos in endpoint_map[idx]:
                    collected[pos] = flat_x

        out_flat = x.to(torch.float32).reshape(bsz, 256)
        if stage_endpoints is not None and return_lip:
            lip_val = torch.stack(lip_terms).mean() if lip_terms else torch.zeros((), device=theta.device)
            return out_flat, [t for t in collected if t is not None], lip_val
        if stage_endpoints is not None:
            return out_flat, [t for t in collected if t is not None]
        if return_lip:
            lip_val = torch.stack(lip_terms).mean() if lip_terms else torch.zeros((), device=theta.device)
            return out_flat, lip_val
        return out_flat

    def forward_slice(self, theta: torch.Tensor, start_op: int, end_op: int) -> torch.Tensor:
        """Apply contiguous op slice [start_op, end_op) using lossless float64 phase wire."""
        bsz = theta.shape[0]
        x = torch.remainder(theta.to(torch.float64), TWO_PI_F64).view(bsz, 8, 32)
        for idx in range(start_op, end_op):
            x = self._run_layer(self.layers[idx], x)
        return x.to(torch.float32).reshape(bsz, 256)

    def inverse(self, theta: torch.Tensor) -> torch.Tensor:
        bsz = theta.shape[0]
        x = torch.remainder(theta.to(torch.float64), TWO_PI_F64).view(bsz, 8, 32)
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x.to(torch.float32).reshape(bsz, 256)

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
            rows.append(cos_y[i].detach() * g_sin - sin_y[i].detach() * g_cos)
        return torch.stack(rows, dim=0)
