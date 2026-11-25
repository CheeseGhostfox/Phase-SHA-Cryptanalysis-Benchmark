"""2D bit-grid native recursive backbone for Phase-SHA V2."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.graph_backbone import (
    SwiGLUFFN,
    build_round_sparse_attn_mask,
    sinusoidal_embedding,
)

SHA_ROTR_OFFSETS: Tuple[int, ...] = (2, 6, 11, 13, 22, 25)


def extract_grid_features(
    x_nodes: torch.Tensor,
    grad_radar: torch.Tensor,
    clamp_mask: torch.Tensor,
    res_features: torch.Tensor,
    k_spins: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert node tensors into (base_feat, x_grid, grad_grid) on shape (B, R, 32, C).

    In `res_features` (B, R, 192), the 6 blocks of 32 scales are:
      [sin_E, 1-cos_E, sin_W, 1-cos_W, sin_S, 1-cos_S]
    Reshaping via (B, R, 6, 32).permute(0, 1, 3, 2) places all 6 residual channels
    for scale m = k + 1 directly at grid cell (t, k).
    """
    bsz, r, _, _ = x_nodes.shape
    x_grid = x_nodes.permute(0, 1, 3, 2).contiguous()  # (B, R, 32, 3)
    grad_grid = grad_radar.permute(0, 1, 3, 2).contiguous()  # (B, R, 32, 3)
    mask_grid = clamp_mask.permute(0, 1, 3, 2).contiguous()  # (B, R, 32, 3)
    res_grid = res_features.view(bsz, r, 6, 32).permute(0, 1, 3, 2).contiguous()

    if k_spins.ndim == 2:
        k_grid = k_spins.unsqueeze(0).expand(bsz, r, 32).unsqueeze(-1)
    else:
        k_grid = k_spins.unsqueeze(-1)

    base_feat = torch.cat([x_grid, grad_grid, mask_grid, res_grid, k_grid], dim=-1)
    return base_feat, x_grid, grad_grid


def build_rotr_channels(
    x_grid: torch.Tensor, grad_grid: torch.Tensor
) -> torch.Tensor:
    """Concatenate explicit SHA-256 bit-rotated (ROTR 2,6,11,13,22,25) state+grad channels."""
    src = torch.cat([x_grid, grad_grid], dim=-1)  # (B, R, 32, 6)
    rotated = [
        torch.roll(src, shifts=-off, dims=2) for off in SHA_ROTR_OFFSETS
    ]
    return torch.cat(rotated, dim=-1)  # (B, R, 32, 36)


def _modulate_grid(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1.0 + scale.view(-1, 1, 1, x.shape[-1])) + shift.view(
        -1, 1, 1, x.shape[-1]
    )


def _conv1d_shift_gemm_seq(x_seq: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
    """Exact pure-FP32 1D convolution via shifted slices + F.linear on (N, L, C_in).

    Preserves conv.weight/conv.bias state_dict parity with nn.Conv1d while avoiding
    cuDNN FFT2D heuristic fallbacks and fusing cleanly under torch.compile.
    """
    stride = conv.stride[0]
    groups = conv.groups
    dilation = conv.dilation[0]
    pad = conv.padding[0]
    k_size = conv.kernel_size[0]
    l_seq = x_seq.shape[1]
    if stride != 1 or groups != 1 or conv.padding_mode != "zeros":
        return conv(x_seq.transpose(1, 2)).transpose(1, 2)
    x_pad = F.pad(x_seq, (0, 0, pad, pad)) if pad > 0 else x_seq
    expected_span = (l_seq - 1) + (k_size - 1) * dilation + 1
    if x_pad.shape[1] != expected_span:
        return conv(x_seq.transpose(1, 2)).transpose(1, 2)
    slices = [x_pad[:, k * dilation : k * dilation + l_seq, :] for k in range(k_size)]
    x_cat = torch.cat(slices, dim=-1)  # (N, L, K * C_in)
    w_flat = conv.weight.permute(0, 2, 1).reshape(conv.out_channels, k_size * conv.in_channels)
    return F.linear(x_cat, w_flat, conv.bias)


class BitGridBlock(nn.Module):
    """2D Bit-Axis (Carry Conv1d + Bit Attn) + Round-Axis Sparse Attn + SwiGLU block."""

    ROT_DIM: int = len(SHA_ROTR_OFFSETS) * 6  # 36

    def __init__(
        self,
        d_bit: int = 64,
        n_heads: int = 4,
        d_ff_mult: int = 3,
        use_carry_gru: bool = False,
    ) -> None:
        super().__init__()
        if d_bit % n_heads != 0:
            raise ValueError(f"d_bit ({d_bit}) must be divisible by n_heads ({n_heads})")
        if use_carry_gru and d_bit % 2 != 0:
            raise ValueError(f"d_bit ({d_bit}) must be even when use_carry_gru=True")
        self.d_bit = d_bit
        self.n_heads = n_heads
        self.head_dim = d_bit // n_heads
        self.use_carry_gru = use_carry_gru

        # Explicit rotation channel injection at block entry
        self.rot_norm = nn.LayerNorm(d_bit, eps=1e-6)
        self.rot_inject = nn.Linear(d_bit + self.ROT_DIM, d_bit, bias=True)

        # (a1) Bidirectional carry-propagation 1D conv along bit axis (non-circular zero pad)
        self.norm_conv = nn.LayerNorm(d_bit, elementwise_affine=False, eps=1e-6)
        self.conv_in = nn.Conv1d(d_bit, 2 * d_bit, kernel_size=5, padding=2)
        self.conv_dil = nn.Conv1d(d_bit, d_bit, kernel_size=3, padding=2, dilation=2)
        if use_carry_gru:
            self.carry_gru_norm = nn.LayerNorm(d_bit, eps=1e-6)
            self.carry_gru = nn.GRU(
                input_size=d_bit,
                hidden_size=d_bit // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.carry_gru_proj = nn.Linear(d_bit, d_bit, bias=True)
            nn.init.normal_(self.carry_gru_proj.weight, std=0.02)
            nn.init.zeros_(self.carry_gru_proj.bias)

        # (a2) Bit-axis multi-head attention
        self.norm_bit = nn.LayerNorm(d_bit, elementwise_affine=False, eps=1e-6)
        self.bit_qkv = nn.Linear(d_bit, 3 * d_bit, bias=True)
        self.bit_proj = nn.Linear(d_bit, d_bit, bias=True)

        # (b) Round-axis sparse graph attention
        self.norm_round = nn.LayerNorm(d_bit, elementwise_affine=False, eps=1e-6)
        self.round_qkv = nn.Linear(d_bit, 3 * d_bit, bias=True)
        self.round_proj = nn.Linear(d_bit, d_bit, bias=True)

        # (c) Pointwise SwiGLU FFN
        self.norm_ffn = nn.LayerNorm(d_bit, elementwise_affine=False, eps=1e-6)
        self.ffn = SwiGLUFFN(d_bit, d_bit * d_ff_mult)

        # AdaLN-Zero modulation for conv, bit_attn, round_attn, and ffn (12 chunks)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_bit, 12 * d_bit, bias=True),
        )
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def _attention_1d(
        self,
        x_seq: torch.Tensor,
        qkv_proj: nn.Linear,
        out_proj: nn.Linear,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_batch, seq_len, dim = x_seq.shape
        qkv = qkv_proj(x_seq).view(n_batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(n_batch, seq_len, dim)
        return out_proj(out)

    def forward(
        self,
        h: torch.Tensor,
        rot_feat: torch.Tensor,
        cond: torch.Tensor,
        round_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, rounds, n_bits, dim = h.shape

        # Inject explicit SHA-256 rotated channels
        h = h + self.rot_inject(torch.cat([self.rot_norm(h), rot_feat], dim=-1))

        (
            sh_c,
            sc_c,
            gt_c,
            sh_b,
            sc_b,
            gt_b,
            sh_r,
            sc_r,
            gt_r,
            sh_f,
            sc_f,
            gt_f,
        ) = self.adaLN(cond).chunk(12, dim=-1)

        # (a1) Bidirectional carry Conv1d (+ optional Bit-Axis BiGRU) along bit axis
        hc = _modulate_grid(self.norm_conv(h), sh_c, sc_c)
        hc_in = hc.reshape(bsz * rounds, n_bits, dim)  # (B*R, 32, C)
        g_c, u_c = _conv1d_shift_gemm_seq(hc_in, self.conv_in).chunk(2, dim=-1)
        hc_seq = _conv1d_shift_gemm_seq(F.silu(g_c) * u_c, self.conv_dil)  # (B*R, 32, C)
        if self.use_carry_gru:
            with torch.amp.autocast("cuda", enabled=False):
                gru_out, _ = self.carry_gru(self.carry_gru_norm(hc_seq.float()))
                gru_delta = self.carry_gru_proj(gru_out)
            hc_seq = hc_seq + gru_delta.to(hc_seq.dtype)
        hc = hc_seq.reshape(bsz, rounds, n_bits, dim)
        h = h + gt_c.view(bsz, 1, 1, dim) * hc

        # (a2) Bit-axis attention
        hb = _modulate_grid(self.norm_bit(h), sh_b, sc_b).reshape(
            bsz * rounds, n_bits, dim
        )
        hb = self._attention_1d(hb, self.bit_qkv, self.bit_proj, attn_mask=None)
        h = h + gt_b.view(bsz, 1, 1, dim) * hb.view(bsz, rounds, n_bits, dim)

        # (b) Round-axis sparse graph attention
        hr = (
            _modulate_grid(self.norm_round(h), sh_r, sc_r)
            .permute(0, 2, 1, 3)
            .reshape(bsz * n_bits, rounds, dim)
        )
        hr = self._attention_1d(
            hr, self.round_qkv, self.round_proj, attn_mask=round_attn_mask
        )
        hr = hr.view(bsz, n_bits, rounds, dim).permute(0, 2, 1, 3)
        h = h + gt_r.view(bsz, 1, 1, dim) * hr

        # (c) SwiGLU FFN
        hf = self.ffn(_modulate_grid(self.norm_ffn(h), sh_f, sc_f))
        h = h + gt_f.view(bsz, 1, 1, dim) * hf
        return h


class BitGridBackbone(nn.Module):
    """2D (Round x 32-Bit) recursive solver predicting clean free-bit logits."""

    BASE_DIM: int = 16
    ROT_DIM: int = BitGridBlock.ROT_DIM
    IN_DIM: int = BASE_DIM + ROT_DIM  # 52

    def __init__(
        self,
        rounds: int = 12,
        d_bit: int = 64,
        n_heads: int = 4,
        n_blocks: int = 4,
        d_ff_mult: int = 3,
        share_blocks: bool = False,
        grad_checkpointing: bool = False,
        use_carry_gru: bool = False,
    ) -> None:
        super().__init__()
        self.rounds = rounds
        self.d_bit = d_bit
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.share_blocks = share_blocks
        self.grad_checkpointing = grad_checkpointing
        self.use_carry_gru = use_carry_gru

        self.input_proj = nn.Linear(self.IN_DIM, d_bit)
        self.h_prev_norm = nn.LayerNorm(d_bit, eps=1e-6)
        self.h_prev_proj = nn.Linear(d_bit, d_bit)

        self.register_buffer(
            "round_pos_emb",
            sinusoidal_embedding(torch.arange(rounds, dtype=torch.float32), d_bit)
            .view(1, rounds, 1, d_bit),
            persistent=False,
        )
        self.register_buffer(
            "bit_pos_emb",
            sinusoidal_embedding(torch.arange(32, dtype=torch.float32), d_bit)
            .view(1, 1, 32, d_bit),
            persistent=False,
        )
        self.register_buffer(
            "round_attn_mask",
            build_round_sparse_attn_mask(rounds, num_registers=0),
            persistent=False,
        )

        self.step_mlp = nn.Sequential(
            nn.Linear(d_bit, d_bit),
            nn.SiLU(),
            nn.Linear(d_bit, d_bit),
        )
        self.lam_mlp = nn.Sequential(
            nn.Linear(32, d_bit),
            nn.SiLU(),
            nn.Linear(d_bit, d_bit),
        )

        if share_blocks:
            shared = BitGridBlock(
                d_bit=d_bit,
                n_heads=n_heads,
                d_ff_mult=d_ff_mult,
                use_carry_gru=use_carry_gru,
            )
            self.blocks = nn.ModuleList([shared for _ in range(n_blocks)])
        else:
            self.blocks = nn.ModuleList(
                [
                    BitGridBlock(
                        d_bit=d_bit,
                        n_heads=n_heads,
                        d_ff_mult=d_ff_mult,
                        use_carry_gru=use_carry_gru,
                    )
                    for _ in range(n_blocks)
                ]
            )

        self.final_norm = nn.LayerNorm(d_bit, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_bit, 2 * d_bit, bias=True),
        )
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)

        self.out_proj = nn.Linear(d_bit, 3)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x_nodes: torch.Tensor,
        grad_radar: torch.Tensor,
        clamp_mask: torch.Tensor,
        res_features: torch.Tensor,
        k_spins: torch.Tensor,
        step_frac: torch.Tensor,
        lam: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict clean free-bit logits (B, R, 3, 32) and updated grid state (B, R, 32, d_bit)."""
        bsz, r, _, _ = x_nodes.shape
        if r != self.rounds:
            raise ValueError(f"Expected rounds={self.rounds}, got {r}")

        if lam.ndim == 1:
            lam_b = lam.unsqueeze(0).expand(bsz, 32)
        else:
            lam_b = lam

        if step_frac.ndim == 0:
            step_b = step_frac.expand(bsz)
        else:
            step_b = step_frac

        base_feat, x_grid, grad_grid = extract_grid_features(
            x_nodes=x_nodes,
            grad_radar=grad_radar,
            clamp_mask=clamp_mask,
            res_features=res_features,
            k_spins=k_spins,
        )
        rot_feat = build_rotr_channels(x_grid, grad_grid)
        grid_in = torch.cat([base_feat, rot_feat], dim=-1)

        use_bf16 = x_nodes.is_cuda and torch.cuda.is_bf16_supported()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            step_emb = self.step_mlp(sinusoidal_embedding(step_b * 1000.0, self.d_bit))
            lam_emb = self.lam_mlp(lam_b.to(torch.float32))
            cond = F.silu(step_emb + lam_emb)

            h = self.input_proj(grid_in) + self.round_pos_emb + self.bit_pos_emb
            if h_prev is not None:
                h = h + self.h_prev_proj(self.h_prev_norm(h_prev))

            rot_in = rot_feat.to(h.dtype)
            for block in self.blocks:
                if self.grad_checkpointing and self.training:
                    h = checkpoint(
                        block,
                        h,
                        rot_in,
                        cond,
                        self.round_attn_mask,
                        use_reentrant=False,
                    )
                else:
                    h = block(h, rot_in, cond, self.round_attn_mask)

            sh_f, sc_f = self.final_adaLN(cond).chunk(2, dim=-1)
            h_out = _modulate_grid(self.final_norm(h), sh_f, sc_f)
            logits_grid = self.out_proj(h_out)  # (B, R, 32, 3)

        logits = logits_grid.permute(0, 1, 3, 2).contiguous().to(torch.float32)
        return logits, h.to(torch.float32)
