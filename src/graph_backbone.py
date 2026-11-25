"""Round-aligned sparse Graph Transformer backbone for Phase-SHA."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROUND_ATTN_OFFSETS: Tuple[int, ...] = (0, 1, 2, 3, 4, 7, 15, 16)


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal embeddings for 1D tensor values of shape (N,)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=values.device)
        / max(half, 1)
    )
    args = values.to(torch.float32).unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def build_round_sparse_attn_mask(
    rounds: int,
    num_registers: int = 2,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build (1, 1, S, S) boolean attention mask (True = allowed).

    Round nodes i, j in [0, R-1] attend iff |i - j| in {0, 1, 2, 3, 4, 7, 15, 16}.
    Global register tokens attend to and are attended by all tokens.
    """
    total_len = rounds + num_registers
    mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)
    idx = torch.arange(rounds, device=device)
    diff = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    for off in ROUND_ATTN_OFFSETS:
        mask[:rounds, :rounds] |= diff == off
    if num_registers > 0:
        mask[rounds:, :] = True
        mask[:, rounds:] = True
    return mask.unsqueeze(0).unsqueeze(0)


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network with multiplicative gating."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, 2 * d_ff, bias=True)
        self.w_down = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


class SparseAttention(nn.Module):
    """Multi-head self-attention enforcing round-aligned sparse graph topology."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, S, D_h)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.proj(out)


class SparseDiTBlock(nn.Module):
    """Pre-LN Sparse Attention + SwiGLU FFN block with AdaLN-Zero conditioning."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = SparseAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ffn = SwiGLUFFN(d_model, d_ff)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )
        h_attn = self.attn(
            _modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask
        )
        x = x + gate_msa.unsqueeze(1) * h_attn
        h_ffn = self.ffn(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * h_ffn
        return x


class PhaseSHABackbone(nn.Module):
    """Round-aligned sparse Graph Transformer predicting clamped velocity field."""

    NODE_FEAT_DIM: int = 96 + 96 + 96 + 192 + 32  # 512

    def __init__(
        self,
        rounds: int = 64,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 1024,
        share_layers: bool = False,
        num_registers: int = 2,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.rounds = rounds
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.share_layers = share_layers
        self.num_registers = num_registers
        self.grad_checkpointing = grad_checkpointing

        self.input_proj = nn.Linear(self.NODE_FEAT_DIM, d_model)

        # Fixed sinusoidal round positional encoding
        round_idx = torch.arange(rounds, dtype=torch.float32)
        self.register_buffer(
            "round_pos_emb",
            sinusoidal_embedding(round_idx, d_model).unsqueeze(0),
            persistent=False,
        )

        # Conditioning network for flow time tau and frequency mask lambda
        self.tau_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.lam_mlp = nn.Sequential(
            nn.Linear(32, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        if num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, num_registers, d_model) * 0.02
            )
        else:
            self.register_tokens = None

        self.register_buffer(
            "attn_mask",
            build_round_sparse_attn_mask(rounds, num_registers=num_registers),
            persistent=False,
        )

        if share_layers:
            self.shared_block = SparseDiTBlock(d_model, n_heads, d_ff)
            self.blocks = nn.ModuleList([self.shared_block for _ in range(n_layers)])
        else:
            self.blocks = nn.ModuleList(
                [SparseDiTBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
            )

        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)

        self.out_proj = nn.Linear(d_model, 96)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x_nodes: torch.Tensor,
        grad_radar: torch.Tensor,
        clamp_mask: torch.Tensor,
        res_features: torch.Tensor,
        k_spins: torch.Tensor,
        tau: torch.Tensor,
        lam: torch.Tensor,
    ) -> torch.Tensor:
        """Predict masked velocity field v of shape (B, R, 3, 32) in float32.

        Args:
            x_nodes: (B, R, 3, 32) current continuous spins
            grad_radar: (B, R, 3, 32) normalized analytical gradient radar
            clamp_mask: (B, R, 3, 32) binary mask (1=locked, 0=free)
            res_features: (B, R, 192) phase residual features
            k_spins: (R, 32) or (B, R, 32) round constant spins
            tau: (B,) flow time in [0, 1]
            lam: (B, 32) or (32,) frequency mask
        """
        bsz, r, _, _ = x_nodes.shape
        if r != self.rounds:
            raise ValueError(f"Expected rounds={self.rounds}, got {r}")

        if k_spins.ndim == 2:
            k_spins_b = k_spins.unsqueeze(0).expand(bsz, r, 32)
        else:
            k_spins_b = k_spins

        if lam.ndim == 1:
            lam_b = lam.unsqueeze(0).expand(bsz, 32)
        else:
            lam_b = lam

        if tau.ndim == 0:
            tau_b = tau.expand(bsz)
        else:
            tau_b = tau

        node_in = torch.cat(
            [
                x_nodes.reshape(bsz, r, 96),
                grad_radar.reshape(bsz, r, 96),
                clamp_mask.reshape(bsz, r, 96),
                res_features,
                k_spins_b,
            ],
            dim=-1,
        )

        use_bf16 = x_nodes.is_cuda and torch.cuda.is_bf16_supported()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            tau_emb = self.tau_mlp(sinusoidal_embedding(tau_b * 1000.0, self.d_model))
            lam_emb = self.lam_mlp(lam_b.to(torch.float32))
            cond = F.silu(tau_emb + lam_emb)

            h = self.input_proj(node_in) + self.round_pos_emb + tau_emb.unsqueeze(1)
            if self.register_tokens is not None:
                regs = self.register_tokens.expand(bsz, -1, -1)
                h = torch.cat([h, regs], dim=1)

            for block in self.blocks:
                if self.grad_checkpointing and self.training:
                    h = checkpoint(
                        block, h, cond, self.attn_mask, use_reentrant=False
                    )
                else:
                    h = block(h, cond, self.attn_mask)

            h_rounds = h[:, :r, :]
            shift_f, scale_f = self.final_adaLN(cond).chunk(2, dim=-1)
            h_rounds = _modulate(self.final_norm(h_rounds), shift_f, scale_f)
            v_flat = self.out_proj(h_rounds)

        v_nodes = v_flat.to(torch.float32).view(bsz, r, 3, 32)
        # Zero out velocity on all clamped/known bits
        return v_nodes * (1.0 - clamp_mask)
