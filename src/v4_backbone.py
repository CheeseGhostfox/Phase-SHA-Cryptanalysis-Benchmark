"""Strict FP32 2D Bit-Grid + BiGRU Manifold Backbone with lossless V3 warm-start."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

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
from src.graph_backbone import build_round_sparse_attn_mask, sinusoidal_embedding
from src.v4_manifolds import BaseManifold, get_manifold


class ManifoldBitGridBackbone(nn.Module):
    """Strict FP32 2D (Round x 32-Bit) recursive solver on S^{D-1} manifolds."""

    BASE_DIM: int = 16
    ROT_DIM: int = BitGridBlock.ROT_DIM  # 36
    IN_DIM: int = BASE_DIM + ROT_DIM  # 52 (identical to V3 for zero-loss warm-start)

    def __init__(
        self,
        variant: str = "v4b",
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
        self.variant = str(variant).lower()
        self.manifold: BaseManifold = get_manifold(self.variant)
        self.manifold_dim = self.manifold.dim
        self.trans_dim = self.manifold_dim - 1
        self.rounds = int(rounds)
        self.d_bit = int(d_bit)
        self.n_heads = int(n_heads)
        self.n_blocks = int(n_blocks)
        self.share_blocks = bool(share_blocks)
        self.grad_checkpointing = bool(grad_checkpointing)
        self.use_carry_gru = bool(use_carry_gru)

        # Shared V3 core layers (exact name parity with V3 BitGridBackbone)
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

        # V4 transverse manifold adapters: (u_trans, g_trans + 6x ROTR) -> d_bit, and tangent output
        trans_feat_dim = 42 * self.trans_dim  # (3 + 3)*(1 + 6) * trans_dim
        self.manifold_in_proj = nn.Linear(trans_feat_dim, d_bit, bias=True)
        nn.init.orthogonal_(self.manifold_in_proj.weight, gain=0.01)
        nn.init.zeros_(self.manifold_in_proj.bias)

        self.manifold_tangent_head = nn.Linear(d_bit, 3 * self.manifold_dim, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.manifold_tangent_head.bias)
            w_view = self.manifold_tangent_head.weight.view(3, self.manifold_dim, d_bit)
            w_view[:, self.manifold.spin_axis, :] = 0.0
            for ax in self.manifold.trans_axes:
                nn.init.orthogonal_(w_view[:, ax, :], gain=0.02)

    def _build_transverse_features(
        self, u_nodes: torch.Tensor, grad_radar: torch.Tensor
    ) -> torch.Tensor:
        """Build 42 * (D-1) transverse position + Riemannian gradient + ROTR features in FP32."""
        u_trans = self.manifold.project_transverse(u_nodes)  # (B, R, 3, 32, D-1)
        g_trans = self.manifold.project_transverse(grad_radar)  # (B, R, 3, 32, D-1)

        # Flatten (3, D-1) onto bit-cell feature dim -> (B, R, 32, 6*(D-1))
        bsz, r, _, n_bits, td = u_trans.shape
        u_grid = u_trans.permute(0, 1, 3, 2, 4).reshape(bsz, r, n_bits, 3 * td)
        g_grid = g_trans.permute(0, 1, 3, 2, 4).reshape(bsz, r, n_bits, 3 * td)
        src = torch.cat([u_grid, g_grid], dim=-1)  # (B, R, 32, 6*(D-1))
        rotated = [torch.roll(src, shifts=-off, dims=2) for off in SHA_ROTR_OFFSETS]
        return torch.cat([src, *rotated], dim=-1)  # (B, R, 32, 42*(D-1))

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
        """Strict FP32 forward returning (logits_spin (B,R,3,32), delta_u_ambient (B,R,3,32,D), h_out)."""
        u_f = u_nodes.to(torch.float32)
        g_f = grad_radar.to(torch.float32)
        mask_f = clamp_mask.to(torch.float32)
        res_f = res_features.to(torch.float32)
        k_f = k_spins.to(torch.float32)

        bsz, r, _, n_bits, d = u_f.shape
        if r != self.rounds or d != self.manifold_dim:
            raise ValueError(
                f"Expected shape (B, {self.rounds}, 3, 32, {self.manifold_dim}), got {tuple(u_f.shape)}"
            )

        x_spin = self.manifold.project_spin(u_f)
        g_spin = self.manifold.project_spin(g_f)

        lam_b = lam.to(torch.float32).unsqueeze(0).expand(bsz, 32) if lam.ndim == 1 else lam.to(torch.float32)
        step_b = step_frac.to(torch.float32).expand(bsz) if step_frac.ndim == 0 else step_frac.to(torch.float32)

        base_feat, x_grid, grad_grid = extract_grid_features(
            x_nodes=x_spin,
            grad_radar=g_spin,
            clamp_mask=mask_f,
            res_features=res_f,
            k_spins=k_f,
        )
        rot_feat = build_rotr_channels(x_grid, grad_grid)
        grid_in = torch.cat([base_feat, rot_feat], dim=-1).to(torch.float32)
        trans_feat = self._build_transverse_features(u_f, g_f)

        step_emb = self.step_mlp(sinusoidal_embedding(step_b * 1000.0, self.d_bit))
        lam_emb = self.lam_mlp(lam_b)
        cond = F.silu(step_emb + lam_emb)

        h = (
            self.input_proj(grid_in)
            + self.manifold_in_proj(trans_feat)
            + self.round_pos_emb
            + self.bit_pos_emb
        )
        if h_prev is not None:
            h = h + self.h_prev_proj(self.h_prev_norm(h_prev.to(torch.float32)))

        rot_in = rot_feat.to(torch.float32)
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
        logits_spin = logits_grid.permute(0, 1, 3, 2).contiguous()

        delta_grid = self.manifold_tangent_head(h_out).view(
            bsz, r, n_bits, 3, self.manifold_dim
        )
        delta_u_ambient = delta_grid.permute(0, 1, 3, 2, 4).contiguous()
        return logits_spin, delta_u_ambient, h.to(torch.float32)

    def load_v3_warmstart(
        self, ckpt_path: str | Path, map_location: torch.device | str = "cpu"
    ) -> Dict[str, object]:
        """Load shared V3 backbone weights losslessly, leaving manifold adapters freshly initialized."""
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        src_state = ckpt.get("ema_model_state", ckpt.get("model_state"))
        if src_state is None:
            raise KeyError(f"No ema_model_state or model_state in {ckpt_path}")

        incompat = self.load_state_dict(src_state, strict=False)
        if incompat.unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys during V3->V4 warm-start: {incompat.unexpected_keys}"
            )
        non_adapter_missing = [
            k
            for k in incompat.missing_keys
            if not (
                k.startswith("manifold_in_proj.")
                or k.startswith("manifold_tangent_head.")
            )
        ]
        if non_adapter_missing:
            raise RuntimeError(
                f"Missing shared V3 core keys during V3->V4 warm-start: {non_adapter_missing}"
            )
        # Ensure pure FP32 weights after load
        self.float()
        src_cfg = ckpt.get("config", {})
        info = {
            "loaded_keys": len(src_state),
            "missing_adapter_keys": list(incompat.missing_keys),
            "source_rounds": src_cfg.get("rounds", "?"),
            "source_step": ckpt.get("step", "?"),
        }
        return info
