"""Pure FP32 3D Spacetime Crystal Koopman Neural Distinguisher (KoopmanCrystalNet3D)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sha256_tuples import enforce_fp32_determinism


class CrystalKoopmanBlock(nn.Module):
    """
    3D Spacetime Crystal Residual Block:
      1) Local Carry-Spacetime Branch: circular bit padding (k_B=3), zero time padding (k_T=3),
         via fast equivalent Conv2d over merged Track=2 batch followed by asymmetric (A, E) track mix.
      2) Global Bit-Rotation Coupling Branch: 4-head SDPA along Bit=32 axis + FFN.
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 1) Carry-Spacetime local branch
        self.norm_conv = nn.GroupNorm(8, d_model)
        self.conv2d = nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=0)
        self.track_mix = nn.Linear(2 * d_model, 2 * d_model)

        # 2) Global bit-rotation self-attention branch (along Bit=32 axis)
        self.norm_attn = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, 4, 2, 32)
        bsz, d, t_dim, r_dim, b_dim = x.shape

        # Local carry-spacetime branch via Conv2d on (B*2, D, 4, 32)
        h = self.norm_conv(x)
        h = h.permute(0, 3, 1, 2, 4).reshape(bsz * r_dim, d, t_dim, b_dim)
        h = torch.cat([h[..., -1:], h, h[..., :1]], dim=-1)  # circular pad along Bit
        h = F.pad(h, (0, 0, 1, 1), mode="constant", value=0.0)  # zero pad along Time
        h = F.silu(self.conv2d(h))
        h = h.view(bsz, r_dim, d, t_dim, b_dim).permute(0, 3, 4, 1, 2).reshape(
            bsz, t_dim, b_dim, r_dim * d
        )
        h = self.track_mix(h).view(bsz, t_dim, b_dim, r_dim, d).permute(0, 4, 1, 3, 2)
        x = x + h

        # Global bit-rotation attention branch: (B*8, 32, D)
        n_seq = bsz * t_dim * r_dim
        seq = x.permute(0, 2, 3, 4, 1).reshape(n_seq, b_dim, d)
        seq_n = self.norm_attn(seq)
        qkv = (
            self.qkv(seq_n)
            .view(n_seq, b_dim, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        attn_out = (
            F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
            .transpose(1, 2)
            .reshape(n_seq, b_dim, d)
        )
        seq = seq + self.proj(attn_out)
        seq = seq + self.ffn(self.norm_ffn(seq))

        return seq.view(bsz, t_dim, r_dim, b_dim, d).permute(0, 4, 1, 2, 3)


class KoopmanCrystalNet3D(nn.Module):
    """
    Symmetry-unbiased 3D Spacetime Crystal Neural Distinguisher in strict FP32.
    Input Psi_3D: (B, C_in, 4, 2, 32) -> Output scalar logit: (B,)
    """

    def __init__(
        self,
        c_in: int,
        d_model: int = 32,
        n_blocks: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        enforce_fp32_determinism()
        self.c_in = int(c_in)
        self.d_model = int(d_model)

        pos = self._build_pos_encoding()
        self.register_buffer("pos_enc", pos, persistent=False)  # (1, 4, 4, 2, 32)

        self.stem = nn.Sequential(
            nn.Linear(self.c_in + 4, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [CrystalKoopmanBlock(d_model=d_model, num_heads=num_heads) for _ in range(n_blocks)]
        )

        # Dual pooling + bit-level linear projection -> 96-dim representation -> 64 -> 1
        self.bit_proj = nn.Linear(d_model * 4 * 2 * 32, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _build_pos_encoding() -> torch.Tensor:
        t = torch.arange(4, dtype=torch.float32)
        t_norm = ((t - 1.5) / 1.5).view(1, 1, 4, 1, 1).expand(1, 1, 4, 2, 32)

        r_sign = torch.tensor([1.0, -1.0], dtype=torch.float32).view(1, 1, 1, 2, 1).expand(1, 1, 4, 2, 32)

        b = torch.arange(32, dtype=torch.float32)
        ang = 2.0 * math.pi * (b / 32.0)
        b_sin = torch.sin(ang).view(1, 1, 1, 1, 32).expand(1, 1, 4, 2, 32)
        b_cos = torch.cos(ang).view(1, 1, 1, 1, 32).expand(1, 1, 4, 2, 32)
        return torch.cat([t_norm, r_sign, b_sin, b_cos], dim=1)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        bsz = psi.shape[0]
        pos = self.pos_enc.expand(bsz, -1, -1, -1, -1)
        x = torch.cat([psi, pos], dim=1)
        x = self.stem(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        for blk in self.blocks:
            x = blk(x)

        mean_p = x.mean(dim=(2, 3, 4))
        max_p = x.amax(dim=(2, 3, 4))
        bit_p = self.bit_proj(x.flatten(1))
        feat = torch.cat([mean_p, max_p, bit_p], dim=1)
        return self.head(feat).squeeze(-1)
