"""Shared pure-FP32 neural building blocks for V7 (RoPE, RMSNorm, SwiGLU, KV-cache).

Zero `complex64`, zero TF32, zero AMP: all rotary math uses real `float32` cos/sin.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNormFP32(nn.Module):
    """Root-mean-square layer normalization strictly computed in float32."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * inv_rms * self.weight


class RotaryEmbeddingFP32(nn.Module):
    """Real-valued float32 Rotary Position Embedding (RoPE) without complex64."""

    def __init__(self, head_dim: int, max_seq_len: int = 1024, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / float(head_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len, device=inv_freq.device)

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, seq_len: int, offset: int = 0, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        need = offset + seq_len
        dev = device if device is not None else self.inv_freq.device
        if need > self.cos_cached.shape[0] or self.cos_cached.device != dev:
            self._build_cache(max(need, self.cos_cached.shape[0]), device=dev)
        cos = self.cos_cached[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_fp32(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply real-valued FP32 RoPE to `q`, `k` of shape `(B, H, T, D)`."""
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class RoPEAttentionFP32(nn.Module):
    """Multi-head self-attention with real FP32 RoPE and optional KV cache."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        is_causal: bool = True,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.is_causal = is_causal
        self.dropout = dropout

        self.wq = nn.Linear(d_model, d_model, bias=False, dtype=torch.float32)
        self.wk = nn.Linear(d_model, d_model, bias=False, dtype=torch.float32)
        self.wv = nn.Linear(d_model, d_model, bias=False, dtype=torch.float32)
        self.wo = nn.Linear(d_model, d_model, bias=False, dtype=torch.float32)
        self.rope = RotaryEmbeddingFP32(self.head_dim, max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_offset: int = 0,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(t, offset=rope_offset, device=x.device)
        q, k = apply_rope_fp32(q, k, cos, sin)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
        new_cache = (k, v)

        use_causal = self.is_causal and kv_cache is None
        attn_mask: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            # key_padding_mask: True where token is valid, False where PAD
            valid = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            if use_causal:
                s = k.shape[2]
                causal_mask = torch.tril(
                    torch.ones((t, s), dtype=torch.bool, device=x.device),
                    diagonal=s - t,
                )
                attn_mask = valid & causal_mask.unsqueeze(0).unsqueeze(0)
                use_causal = False
            else:
                attn_mask = valid

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_causal,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.wo(out), new_cache


class SwiGLUFFNFP32(nn.Module):
    """SwiGLU feed-forward block strictly in float32."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False, dtype=torch.float32)
        self.w_up = nn.Linear(d_model, d_ff, bias=False, dtype=torch.float32)
        self.w_down = nn.Linear(d_ff, d_model, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlockFP32(nn.Module):
    """Pre-norm RMSNorm + RoPE Attention + SwiGLU Transformer layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        is_causal: bool = True,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNormFP32(d_model)
        self.attn = RoPEAttentionFP32(
            d_model=d_model,
            n_heads=n_heads,
            is_causal=is_causal,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.norm2 = RMSNormFP32(d_model)
        self.ffn = SwiGLUFFNFP32(d_model=d_model, d_ff=d_ff)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_offset: int = 0,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, new_cache = self.attn(
            self.norm1(x),
            kv_cache=kv_cache,
            rope_offset=rope_offset,
            key_padding_mask=key_padding_mask,
        )
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class NibbleBitwiseEmbeddingFP32(nn.Module):
    """Embeds 4-bit nibble & special tokens with exact 4-bit bitwise inductive bias."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, dtype=torch.float32)
        self.bit_proj = nn.Linear(4, d_model, bias=False, dtype=torch.float32)

        # Precompute lookup table of signed bits in [-1, +1] for tokens 0..15, 0 for >=16
        lut = torch.zeros((vocab_size, 4), dtype=torch.float32)
        for v in range(16):
            for b in range(4):
                lut[v, b] = 1.0 if ((v >> b) & 1) else -1.0
        self.register_buffer("bit_lut", lut, persistent=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        tok = self.tok_emb(token_ids)
        bits = self.bit_lut[token_ids]
        return tok + 0.25 * self.bit_proj(bits)


__all__ = [
    "RMSNormFP32",
    "RotaryEmbeddingFP32",
    "apply_rope_fp32",
    "RoPEAttentionFP32",
    "SwiGLUFFNFP32",
    "TransformerBlockFP32",
    "NibbleBitwiseEmbeddingFP32",
]
