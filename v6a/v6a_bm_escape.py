"""Phase-SHA V6A Layer 4: Burer-Monteiro S^{N-1} History-Basin Orthogonal Escape Layer.

Replaces unguided random restarts with geometric basin-orthogonal Cayley kicks:
  - Tracks a rolling W-step shadow consensus history window per trajectory.
  - When a trajectory's instantaneous gl_gap stagnates for T_stag consecutive ticks,
    builds the local low-dimensional basin subspace via singularity-free FP32 MGS2,
    extracts a VSIDS-conflict-focused perturbation in its orthogonal complement,
    lifts to unit hypersphere S^{N-1} (default N=16), and applies the exact Lean 4
    rational half-angle Cayley rotation (norm-preserving to FP32 machine precision)
    while strictly leaving all `var_clamp_mask == 1` coordinates invariant.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from v6a import enforce_v6a_fp32
from v6a.v6a_bottleneck_mps import batched_orthonormalize_mgs2


class BurerMonteiroEscapeLayer(nn.Module):
    """Stagnation-triggered S^{N-1} Cayley orthogonal escape for trapped GL-RRR particles."""

    def __init__(
        self,
        n_var_bits: int,
        manifold_dim: int = 16,
        window_size: int = 8,
        stag_patience: int = 12,
        stag_rel_tol: float = 0.01,
        cayley_t: float = 0.42,
    ) -> None:
        super().__init__()
        enforce_v6a_fp32()
        self.n_var_bits = int(n_var_bits)
        self.manifold_dim = max(4, int(manifold_dim))
        self.window_size = max(3, int(window_size))
        self.stag_patience = max(2, int(stag_patience))
        self.stag_rel_tol = float(stag_rel_tol)
        self.cayley_t = max(0.05, min(0.95, float(cayley_t)))

        self._hist_buf: Optional[torch.Tensor] = None  # (B, W, n_var_bits)
        self._hist_len: int = 0
        self._hist_ptr: int = 0
        self._prev_gap: Optional[torch.Tensor] = None
        self._stag_cnt: Optional[torch.Tensor] = None
        self.total_escapes: int = 0

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        self._hist_buf = torch.zeros(
            (batch_size, self.window_size, self.n_var_bits), dtype=torch.float32, device=device
        )
        self._hist_len = 0
        self._hist_ptr = 0
        self._prev_gap = None
        self._stag_cnt = torch.zeros((batch_size,), dtype=torch.int64, device=device)

    @torch.no_grad()
    @torch.amp.autocast("cuda", enabled=False)
    def cayley_rotate_s_n(
        self,
        v_sph: torch.Tensor,
        tangent_dir: torch.Tensor,
        t_param: float,
    ) -> torch.Tensor:
        """Exact norm-preserving S^{N-1} rational half-angle Cayley rotation (Lean 4 `liftedVec`).

        v_sph: (..., N) unit vectors, tangent_dir: (..., N) orthogonal unit tangent vectors.
        """
        vf = v_sph.to(torch.float32)
        qf = tangent_dir.to(torch.float32)
        # Ensure exact orthogonality & unit norm in pure FP32
        vn = vf / torch.linalg.vector_norm(vf, dim=-1, keepdim=True).clamp_min(1e-8)
        qf = qf - (qf * vn).sum(dim=-1, keepdim=True) * vn
        qn = qf / torch.linalg.vector_norm(qf, dim=-1, keepdim=True).clamp_min(1e-8)

        t2 = float(t_param) * float(t_param)
        denom = 1.0 + t2
        c_rot = (1.0 - t2) / denom
        s_rot = (2.0 * float(t_param)) / denom
        out = c_rot * vn + s_rot * qn
        return out / torch.linalg.vector_norm(out, dim=-1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def record_and_maybe_escape(
        self,
        v_a_vars: torch.Tensor,
        gl_gap: torch.Tensor,
        var_clamp_mask: torch.Tensor,
        var_clamp_vals: torch.Tensor,
        vsids_var_weights: Optional[torch.Tensor] = None,
        force_escape_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Dict[str, float]]:
        """Record shadow trajectory states; emit orthogonal Cayley-kicked variable vectors for stagnated indices.

        Returns (kicked_vars_subset (K, n_var_bits) or None, kicked_indices (K,), stats).
        Strictly preserves `var_clamp_mask == 1` values!
        """
        bsz, n_v = v_a_vars.shape
        dev = v_a_vars.device
        if self._hist_buf is None or self._hist_buf.shape[0] != bsz or self._hist_buf.device != dev:
            self.reset_state(bsz, dev)

        assert self._hist_buf is not None and self._stag_cnt is not None
        self._hist_buf[:, self._hist_ptr, :] = v_a_vars.detach().to(torch.float32)
        self._hist_ptr = (self._hist_ptr + 1) % self.window_size
        self._hist_len = min(self.window_size, self._hist_len + 1)

        if self._prev_gap is None:
            self._prev_gap = gl_gap.detach().to(torch.float32)
        else:
            rel_impr = (self._prev_gap - gl_gap) / self._prev_gap.clamp_min(1e-6)
            stalled = (rel_impr < self.stag_rel_tol) & (gl_gap > 1e-4)
            self._stag_cnt = torch.where(stalled, self._stag_cnt + 1, torch.zeros_like(self._stag_cnt))
            self._prev_gap = gl_gap.detach().to(torch.float32)

        escape_mask = self._stag_cnt >= self.stag_patience
        if force_escape_mask is not None:
            escape_mask = escape_mask | force_escape_mask.to(torch.bool)

        kicked_indices = torch.nonzero(escape_mask, as_tuple=False).reshape(-1)
        if kicked_indices.numel() == 0:
            return None, kicked_indices, {"esc_count": 0.0, "esc_orth_norm": 0.0}

        # Cap batch size per step so escape MGS stays ultra fast (< 10 ms)
        if kicked_indices.numel() > 64:
            kicked_indices = kicked_indices[:64]

        k_cnt = int(kicked_indices.numel())
        free_mask = (1.0 - var_clamp_mask[kicked_indices].to(torch.float32)).clamp(0.0, 1.0)
        cur_v = v_a_vars[kicked_indices].to(torch.float32)  # (K, n_v)

        # Build recent history basis (K, n_v, W_active) -> orthonormal Q_basin via FP32 MGS2
        w_act = max(1, self._hist_len)
        hist_sub = self._hist_buf[kicked_indices, :w_act, :].permute(0, 2, 1) * free_mask.unsqueeze(-1)
        q_basin = batched_orthonormalize_mgs2(hist_sub)  # (K, n_v, w_act)

        # Random conflict-weighted perturbation projected onto orthogonal complement of Q_basin
        z = torch.randn((k_cnt, n_v), dtype=torch.float32, device=dev, generator=generator)
        if vsids_var_weights is not None:
            z = z * vsids_var_weights[kicked_indices].to(torch.float32).sqrt()
        z = z * free_mask
        proj_c = torch.bmm(q_basin.transpose(-2, -1), z.unsqueeze(-1))  # (K, w_act, 1)
        z_perp = (z - torch.bmm(q_basin, proj_c).squeeze(-1)) * free_mask
        z_perp = z_perp / torch.linalg.vector_norm(z_perp, dim=-1, keepdim=True).clamp_min(1e-8)

        # Lift bit scalar coordinates onto S^{N-1} (e_0 = spin axis, e_1 = basin-orthogonal axis, rest = small gauge)
        # Vector shape (K, n_v, N)
        v_sph = torch.zeros((k_cnt, n_v, self.manifold_dim), dtype=torch.float32, device=dev)
        q_sph = torch.zeros((k_cnt, n_v, self.manifold_dim), dtype=torch.float32, device=dev)
        spin_clamped = cur_v.clamp(-1.0, 1.0)
        v_sph[..., 0] = spin_clamped
        v_sph[..., 1] = torch.sqrt((1.0 - spin_clamped * spin_clamped).clamp_min(0.0))
        # Orthogonal transverse rotation kicks along e_1 / e_2 and projects onto spin axis via Cayley angle
        # Normalize each bit vector on S^{N-1} then apply exact Cayley rotation toward z_perp in (e_0, e_2)
        q_sph[..., 0] = z_perp * math.sqrt(float(n_v))
        q_sph[..., 2] = 0.5

        v_rot = self.cayley_rotate_s_n(v_sph, q_sph, t_param=self.cayley_t)
        # Extract kicked real-axis coordinates + direct orthogonal basin displacement, clamp free coords to [-1.25, 1.25]
        v_new = torch.where(
            free_mask > 0.5,
            (v_rot[..., 0] + 0.65 * z_perp * math.sqrt(float(n_v))).clamp(-1.35, 1.35),
            var_clamp_vals[kicked_indices].to(torch.float32),
        )

        self._stag_cnt[kicked_indices] = 0
        self.total_escapes += k_cnt
        orth_norm = float(torch.linalg.vector_norm(z_perp, dim=-1).mean().item())
        return v_new, kicked_indices, {"esc_count": float(k_cnt), "esc_orth_norm": orth_norm}
