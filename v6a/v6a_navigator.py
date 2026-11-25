"""Phase-SHA V6A Layer 5: Zero-Brake Neural Navigator (Homotopy Predictor + VSIDS Conflict Focus).

Strict design contracts:
  - ZERO single-bit marginal BCE anywhere!
  - HomotopyPredictorNet: compact word-structured residual neural predictor forecasting tangent jump
    delta_x(alpha -> alpha + d_alpha), trained strictly online via self-supervised MSE
    ||x_corr - x_pred||_2^2 ONLY on converged/successful homotopy advances (never poisoned by dead pits).
  - VSIDSFocusHead: maps GL-RRR gap-oscillation EMA onto normalized positive replica weights
    w_i in [0.5, 4.0] (mean 1.0 per trajectory), physically forbidden from altering step size beta.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v6a import enforce_v6a_fp32


class HomotopyPredictorNet(nn.Module):
    """Word-structured tangent-step homotopy predictor (pure FP32, O(n_var_words) fast)."""

    def __init__(self, n_var_words: int, hidden: int = 64) -> None:
        super().__init__()
        enforce_v6a_fp32()
        self.n_var_words = int(n_var_words)
        # Input per word: [x_word(32), gap_word(32), clamp_word(32), scalars(3 broadcast->32)] -> 128 channels
        self.word_in = nn.Linear(128, hidden)
        self.word_conv = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, groups=1)
        self.word_out = nn.Linear(hidden, 32)
        nn.init.zeros_(self.word_out.weight)
        nn.init.zeros_(self.word_out.bias)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        v_a_vars: torch.Tensor,
        var_gap_ema: torch.Tensor,
        var_clamp_mask: torch.Tensor,
        alpha: float,
        d_alpha: float,
        mask_frac: float,
    ) -> torch.Tensor:
        """Predict variable-space tangent correction delta_x (B, n_var_bits) in [-0.75, 0.75]."""
        bsz, _ = v_a_vars.shape
        x_w = v_a_vars.to(torch.float32).view(bsz, self.n_var_words, 32)
        g_w = var_gap_ema.to(torch.float32).view(bsz, self.n_var_words, 32).clamp(0.0, 2.0)
        m_w = var_clamp_mask.to(torch.float32).view(bsz, self.n_var_words, 32)

        cond = torch.zeros((bsz, self.n_var_words, 32), dtype=torch.float32, device=v_a_vars.device)
        cond[:, :, 0] = float(alpha)
        cond[:, :, 1] = float(d_alpha) * 10.0
        cond[:, :, 2] = float(mask_frac)

        feat = torch.cat([x_w, g_w, m_w, cond], dim=-1)  # (B, W, 128)
        h = F.silu(self.word_in(feat))  # (B, W, H)
        h = h + F.silu(self.word_conv(h.transpose(1, 2)).transpose(1, 2))
        delta = 0.75 * torch.tanh(self.word_out(h)) * (1.0 - m_w)
        return delta.reshape(bsz, self.n_var_bits if hasattr(self, "n_var_bits") else self.n_var_words * 32)


class VSIDSFocusHead(nn.Module):
    """Conflict-activity EMA & positive consensus weight focus w_i in [0.5, 4.0] (mean 1.0)."""

    def __init__(self, n_var_bits: int, ema_decay: float = 0.85, w_min: float = 0.5, w_max: float = 4.0) -> None:
        super().__init__()
        enforce_v6a_fp32()
        self.n_var_bits = int(n_var_bits)
        self.ema_decay = float(ema_decay)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.gain_raw = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.bias_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self._gap_ema: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update_ema(self, var_gap_profile: torch.Tensor) -> torch.Tensor:
        gp = var_gap_profile.detach().to(torch.float32)
        if self._gap_ema is None or self._gap_ema.shape != gp.shape or self._gap_ema.device != gp.device:
            self._gap_ema = gp.clone()
        else:
            self._gap_ema.mul_(self.ema_decay).add_(gp * (1.0 - self.ema_decay))
        return self._gap_ema

    @torch.amp.autocast("cuda", enabled=False)
    def compute_weights(self, var_clamp_mask: torch.Tensor) -> torch.Tensor:
        """Compute per-variable positive consensus focus weights normalized per trajectory cleanly."""
        bsz, n_v = var_clamp_mask.shape
        if self._gap_ema is None or self._gap_ema.shape[0] != bsz:
            return torch.ones((bsz, n_v), dtype=torch.float32, device=var_clamp_mask.device)

        free_m = (1.0 - var_clamp_mask.to(torch.float32)).clamp(0.0, 1.0)
        gain = F.softplus(self.gain_raw).clamp(0.1, 6.0)
        raw_w = torch.sigmoid(gain * self._gap_ema + self.bias_raw) * (self.w_max - self.w_min) + self.w_min
        # Preserve unit weight 1.0 on clamped constants; normalize free mean smoothly inside [w_min, w_max]
        free_mean = (raw_w * free_m).sum(dim=-1, keepdim=True) / free_m.sum(dim=-1, keepdim=True).clamp_min(1.0)
        norm_w = (raw_w / free_mean.clamp_min(1e-4)).clamp(self.w_min, self.w_max)
        return torch.where(free_m > 0.5, norm_w, torch.ones_like(norm_w))


class ZeroBrakeNavigator(nn.Module):
    """V6A Layer 5 controller: predicts homotopy tangent jumps & VSIDS consensus weights with ZERO braking."""

    def __init__(
        self,
        n_var_words: int,
        hidden: int = 64,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        enforce_v6a_fp32()
        self.n_var_words = int(n_var_words)
        self.n_var_bits = self.n_var_words * 32
        self.predictor = HomotopyPredictorNet(n_var_words=self.n_var_words, hidden=hidden)
        self.vsids = VSIDSFocusHead(n_var_bits=self.n_var_bits)
        self.opt = torch.optim.Adam(self.parameters(), lr=float(lr))

        # Stashed prediction waiting for verified post-jump correction before self-supervised MSE update
        self._pending_pred_in: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]] = None
        self._pending_x_before: Optional[torch.Tensor] = None
        self.last_pred_mse: float = 0.0
        self.predictor_updates: int = 0

    @torch.no_grad()
    def record_gap(self, var_gap_profile: torch.Tensor) -> torch.Tensor:
        return self.vsids.update_ema(var_gap_profile)

    @torch.no_grad()
    def get_vsids_weights(self, var_clamp_mask: torch.Tensor) -> torch.Tensor:
        return self.vsids.compute_weights(var_clamp_mask)

    @torch.no_grad()
    def predict_tangent_jump(
        self,
        v_a_vars: torch.Tensor,
        var_clamp_mask: torch.Tensor,
        alpha: float,
        d_alpha: float,
        mask_frac: float,
    ) -> torch.Tensor:
        """Predict jump increment delta_x and cache inputs for safe post-convergence training."""
        gap_ema = (
            self.vsids._gap_ema
            if self.vsids._gap_ema is not None and self.vsids._gap_ema.shape == v_a_vars.shape
            else torch.zeros_like(v_a_vars)
        )
        delta_x = self.predictor(v_a_vars, gap_ema, var_clamp_mask, alpha, d_alpha, mask_frac)
        self._pending_pred_in = (
            v_a_vars.detach().clone(),
            gap_ema.detach().clone(),
            var_clamp_mask.detach().clone(),
            float(alpha),
            float(d_alpha),
            float(mask_frac),
        )
        self._pending_x_before = v_a_vars.detach().clone()
        return delta_x

    @torch.amp.autocast("cuda", enabled=False)
    def train_on_successful_jump(
        self,
        v_a_after_converged: torch.Tensor,
        best_mask: Optional[torch.Tensor] = None,
        max_train_samples: int = 128,
    ) -> float:
        """Update predictor strictly via self-supervised MSE ||delta_actual - delta_pred||^2 (ZERO BCE)."""
        if self._pending_pred_in is None or self._pending_x_before is None:
            return self.last_pred_mse
        x_in, gap_in, m_in, al, da, mf = self._pending_pred_in
        if x_in.shape != v_a_after_converged.shape:
            self._pending_pred_in = None
            return self.last_pred_mse

        free_m = (1.0 - m_in.to(torch.float32)).clamp(0.0, 1.0)
        target_delta = ((v_a_after_converged.detach().to(torch.float32) - self._pending_x_before) * free_m).clamp(-0.8, 0.8)
        if best_mask is not None and bool(best_mask.any().item()):
            x_in = x_in[best_mask]
            gap_in = gap_in[best_mask]
            m_in = m_in[best_mask]
            free_m = free_m[best_mask]
            target_delta = target_delta[best_mask]

        cap = max(8, int(max_train_samples))
        if x_in.shape[0] > cap:
            sel = torch.linspace(0, x_in.shape[0] - 1, steps=cap, device=x_in.device).round().to(torch.int64)
            x_in = x_in[sel]
            gap_in = gap_in[sel]
            m_in = m_in[sel]
            free_m = free_m[sel]
            target_delta = target_delta[sel]

        self.opt.zero_grad(set_to_none=True)
        pred_delta = self.predictor(x_in, gap_in, m_in, al, da, mf)
        mse = ((pred_delta - target_delta).square() * free_m).sum() / free_m.sum().clamp_min(1.0)
        mse.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.opt.step()

        self.last_pred_mse = float(mse.item())
        self.predictor_updates += 1
        self._pending_pred_in = None
        self._pending_x_before = None
        return self.last_pred_mse

    def discard_pending_jump(self) -> None:
        """Discard pending jump when bifurcation / spike occurred so dead pits do not poison predictor."""
        self._pending_pred_in = None
        self._pending_x_before = None
