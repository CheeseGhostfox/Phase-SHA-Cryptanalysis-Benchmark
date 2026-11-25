"""Unsupervised marginal entropy reduction & round-transport objectives on T^256."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v8a import configure_strict_fp32
from v8a.sha256_stream import compute_round_t1_bits
from v8a.torus_flow import TWO_PI, TorusFlow, dequantize_bits


class PeriodicMarginalDensity(nn.Module):
    """Exact normalized 1D mixture of von Mises densities per torus channel."""

    def __init__(
        self,
        num_channels: int = 256,
        num_components: int = 12,
        max_log_kappa: float = 5.0,
        seed: int = 77,
    ) -> None:
        configure_strict_fp32()
        super().__init__()
        self.num_channels = int(num_channels)
        self.num_components = int(num_components)
        self.max_log_kappa = float(max_log_kappa)

        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        logits_init = torch.randn((self.num_channels, self.num_components), generator=gen) * 0.05
        base_mu = torch.linspace(0.0, TWO_PI * (1.0 - 1.0 / self.num_components), self.num_components)
        mu_init = base_mu.unsqueeze(0).expand(self.num_channels, -1).clone()
        mu_init = mu_init + torch.randn((self.num_channels, self.num_components), generator=gen) * 0.08
        log_kappa_init = torch.ones((self.num_channels, self.num_components), dtype=torch.float32)

        self.logits = nn.Parameter(logits_init)
        self.mu = nn.Parameter(mu_init)
        self.log_kappa = nn.Parameter(log_kappa_init)

    def _log_components(self, theta: torch.Tensor) -> torch.Tensor:
        kappa = torch.exp(torch.clamp(self.log_kappa, min=-6.0, max=self.max_log_kappa))
        log_w = F.log_softmax(self.logits, dim=-1)
        log_norm = -math.log(TWO_PI) - torch.log(torch.special.i0e(kappa))
        diff = theta.unsqueeze(-1) - self.mu.unsqueeze(0)
        return log_w.unsqueeze(0) + kappa.unsqueeze(0) * (torch.cos(diff) - 1.0) + log_norm.unsqueeze(0)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute log q_{psi_i}(theta_{n,i}) of shape [B, C]."""
        return torch.logsumexp(self._log_components(theta), dim=-1)

    @torch.no_grad()
    def verify_normalization(self, grid_size: int = 2048) -> float:
        """Verify integral_0^{2*pi} q_i(theta) d_theta == 1 across all channels."""
        device = self.logits.device
        grid = torch.linspace(0.0, TWO_PI, grid_size + 1, device=device, dtype=torch.float32)[:-1]
        theta_grid = grid.unsqueeze(1).expand(grid_size, self.num_channels)
        log_q = self.log_prob(theta_grid)
        integrals = torch.exp(log_q).mean(dim=0) * TWO_PI
        return float(torch.max(torch.abs(integrals - 1.0)).item())

    def forward(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        log_q = self.log_prob(theta)
        per_channel_nll = -log_q.mean(dim=0)
        loss_marg = per_channel_nll.sum()
        per_channel_bits = (math.log(TWO_PI) - per_channel_nll) / math.log(2.0)
        extracted_bits = per_channel_bits.sum()
        return {
            "loss_marg": loss_marg,
            "per_channel_nll": per_channel_nll,
            "per_channel_bits": per_channel_bits,
            "extracted_bits": extracted_bits,
        }


@torch.no_grad()
def compute_circular_metrics(theta: torch.Tensor) -> Dict[str, object]:
    """Compute first and second harmonic circular resultant lengths R1, R2."""
    c1 = torch.cos(theta).mean(dim=0)
    s1 = torch.sin(theta).mean(dim=0)
    r1 = torch.sqrt(c1 * c1 + s1 * s1)

    c2 = torch.cos(2.0 * theta).mean(dim=0)
    s2 = torch.sin(2.0 * theta).mean(dim=0)
    r2 = torch.sqrt(c2 * c2 + s2 * s2)

    r_max = torch.maximum(r1, r2)
    return {
        "R1": r1,
        "R2": r2,
        "R_max": r_max,
        "collapsed_ch_35": int((r1 > 0.35).sum().item()),
        "collapsed_ch_55": int((r1 > 0.55).sum().item()),
        "collapsed_ch_max_35": int((r_max > 0.35).sum().item()),
        "collapsed_ch_max_55": int((r_max > 0.55).sum().item()),
        "max_R1": float(r1.max().item()),
        "max_R2": float(r2.max().item()),
        "mean_R1": float(r1.mean().item()),
    }


@torch.no_grad()
def compute_discrete_metrics(
    flow: TorusFlow,
    h_bits: torch.Tensor,
    batch_size: int = 2048,
) -> Dict[str, object]:
    """Evaluate deterministic (u=0) discrete collapse resultant R_disc through flow."""
    was_training = flow.training
    flow.eval()
    chunks: List[torch.Tensor] = []
    for start in range(0, h_bits.shape[0], batch_size):
        theta_0 = dequantize_bits(h_bits[start : start + batch_size], deterministic=True)
        chunks.append(flow(theta_0))
    theta_out = torch.cat(chunks, dim=0)
    if was_training:
        flow.train()

    c1 = torch.cos(theta_out).mean(dim=0)
    s1 = torch.sin(theta_out).mean(dim=0)
    r_disc = torch.sqrt(c1 * c1 + s1 * s1)
    return {
        "R_disc": r_disc,
        "collapsed_ch_disc_90": int((r_disc > 0.90).sum().item()),
        "max_R_disc": float(r_disc.max().item()),
    }


def estimate_raw_input_baseline(
    eval_h_bits: torch.Tensor,
    train_h_bits: torch.Tensor | None = None,
    num_components: int = 12,
    steps: int = 200,
    batch_size: int = 1024,
    lr: float = 0.03,
    seed: int = 999,
) -> Dict[str, object]:
    """Measure marginal redundancy already present in raw SHA-256 outputs (identity flow)."""
    device = eval_h_bits.device
    if train_h_bits is None:
        mid = eval_h_bits.shape[0] // 2
        fit_bits = eval_h_bits[:mid]
        val_bits = eval_h_bits[mid:]
    else:
        fit_bits = train_h_bits
        val_bits = eval_h_bits

    density = PeriodicMarginalDensity(
        num_channels=256, num_components=num_components, seed=seed
    ).to(device)
    opt = torch.optim.Adam(density.parameters(), lr=lr)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    n_fit = fit_bits.shape[0]

    density.train()
    for _ in range(steps):
        idx = torch.randint(0, n_fit, (min(batch_size, n_fit),), device=device, generator=gen)
        theta = dequantize_bits(fit_bits[idx], deterministic=False, generator=gen)
        out = density(theta)
        opt.zero_grad()
        out["loss_marg"].backward()
        opt.step()

    density.eval()
    with torch.no_grad():
        total_bits = 0.0
        r1_acc = torch.zeros(256, device=device)
        draws = 4
        for _ in range(draws):
            theta_val = dequantize_bits(val_bits, deterministic=False, generator=gen)
            out_val = density(theta_val)
            total_bits += float(out_val["extracted_bits"].item())
            cm = compute_circular_metrics(theta_val)
            r1_acc += cm["R1"]
        r1_avg = r1_acc / draws

    return {
        "baseline_input_bits": total_bits / draws,
        "baseline_collapsed_ch_35": int((r1_avg > 0.35).sum().item()),
        "baseline_collapsed_ch_55": int((r1_avg > 0.55).sum().item()),
        "baseline_max_R1": float(r1_avg.max().item()),
    }


class RoundTransportObjective:
    """256-bit reversible round-by-round transport warmup for Track-2."""

    def __init__(
        self,
        rounds: int,
        k_bits: int,
        total_layers: int,
        custom_stage_slices: List[Tuple[int, int]] | None = None,
        custom_target_rounds: List[int] | None = None,
        custom_stage_kinds: List[str] | None = None,
    ) -> None:
        self.rounds = int(rounds)
        self.k_bits = int(k_bits)
        self.total_layers = int(total_layers)

        if custom_stage_slices is not None and custom_target_rounds is not None:
            if len(custom_stage_slices) != len(custom_target_rounds):
                raise ValueError("custom_stage_slices and custom_target_rounds length mismatch")
            self.stage_slices = [(int(a), int(b)) for a, b in custom_stage_slices]
            self.target_rounds = [int(r) for r in custom_target_rounds]
            self.stage_kinds = (
                [str(k) for k in custom_stage_kinds]
                if custom_stage_kinds is not None
                else ["full"] * len(self.target_rounds)
            )
            self.num_stages = len(self.target_rounds)
            self.layers_per_stage = max((b - a) for a, b in self.stage_slices)
            self.stage_endpoints = [b for _, b in self.stage_slices]
            return

        r0 = min(self.rounds, 1 if self.k_bits == 32 else 2)
        if self.rounds > r0:
            self.target_rounds = list(range(self.rounds - 1, r0 - 1, -1))
        else:
            self.target_rounds = [r0]
        self.stage_kinds = ["full"] * len(self.target_rounds)
        self.num_stages = len(self.target_rounds)

        if self.num_stages == 1:
            self.layers_per_stage = self.total_layers
            self.stage_slices = [(0, self.total_layers)]
        else:
            if self.num_stages * 4 + 2 <= self.total_layers:
                lps = 4
            else:
                lps = max(1, (self.total_layers - 2) // self.num_stages)
                if lps * self.num_stages > self.total_layers:
                    lps = max(1, self.total_layers // self.num_stages)
            self.layers_per_stage = lps
            self.stage_slices = [
                (i * lps, (i + 1) * lps) for i in range(self.num_stages)
            ]
        self.stage_endpoints = [end for _, end in self.stage_slices]

    def _build_stage_target_bits(
        self,
        stage_idx: int,
        states_bits: torch.Tensor,
        t1_cache: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        tr = self.target_rounds[stage_idx]
        s_bits = states_bits[tr - 1]
        if self.stage_kinds[stage_idx] == "half_t1":
            if tr not in t1_cache:
                t1_cache[tr] = compute_round_t1_bits(s_bits)
            out = s_bits.clone()
            out[:, :32] = t1_cache[tr]
            return out
        return s_bits

    def _stage_loss_and_acc(
        self,
        pred_stage: torch.Tensor,
        target_bits: torch.Tensor,
        stage_idx: int,
        eps: float,
        w_bit: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kind = self.stage_kinds[stage_idx]
        bsz = pred_stage.shape[0]
        pred_w = pred_stage.view(bsz, 8, 32)
        tgt_w = target_bits.view(bsz, 8, 32)
        pred_b = (torch.cos(pred_w) < 0.0).to(torch.float32)

        if kind == "half_t1":
            diff = pred_w[:, 0, :] - math.pi * tgt_w[:, 0, :] - eps
            loss = ((1.0 - torch.cos(diff)) * w_bit).sum(dim=-1).mean()
            dh_acc = (pred_b[:, 0, :] == tgt_w[:, 0, :]).float().mean()
            full_acc = (pred_b == tgt_w).float().mean()
            return loss, full_acc, dh_acc

        if stage_idx > 0 and self.stage_kinds[stage_idx - 1] == "half_t1":
            # Focus gradient cleanly on the two active computed words (d_{t-1}=word 3, h_{t-1}=word 7)
            diff_dh = pred_w[:, (3, 7), :] - math.pi * tgt_w[:, (3, 7), :] - eps
            loss = ((1.0 - torch.cos(diff_dh)) * w_bit.unsqueeze(0)).sum(dim=(1, 2)).mean()
            dh_acc = (pred_b[:, (3, 7), :] == tgt_w[:, (3, 7), :]).float().mean()
            full_acc = (pred_b == tgt_w).float().mean()
            return loss, full_acc, dh_acc

        diff_all = pred_w - math.pi * tgt_w - eps
        loss = ((1.0 - torch.cos(diff_all)) * w_bit.unsqueeze(0)).sum(dim=(1, 2)).mean()
        full_acc = (pred_b == tgt_w).float().mean()
        return loss, full_acc, full_acc

    def compute_loss(
        self,
        flow: TorusFlow,
        density: PeriodicMarginalDensity,
        h_bits: torch.Tensor,
        states_bits: torch.Tensor,
        alpha: float,
        generator: torch.Generator | None = None,
        lip_sigma: float = 0.0,
        progress: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute blended teacher-forced + rollout round transport + weak marginal loss."""
        theta_in = dequantize_bits(h_bits, deterministic=False, generator=generator)
        prog = max(0.0, min(1.0, float(progress)))
        eps = 0.8 * max(0.0, 1.0 - 2.0 * prog)
        temp = 2.0 * max(0.0, 1.0 - 2.0 * prog)
        bit_pos = torch.arange(32, device=h_bits.device, dtype=torch.float32)
        w_bit = torch.exp(temp * (bit_pos - 31.0))
        w_bit = w_bit / w_bit.mean()

        t1_cache: Dict[int, torch.Tensor] = {}
        stage_targets = [
            self._build_stage_target_bits(i, states_bits, t1_cache)
            for i in range(self.num_stages)
        ]

        # 1. Parallel teacher-forced local stage losses
        tf_losses: List[torch.Tensor] = []
        tf_accs: List[torch.Tensor] = []
        tf_dh_accs: List[torch.Tensor] = []
        for stage_idx, (s_start, s_end) in enumerate(self.stage_slices):
            if stage_idx == 0:
                stage_in = theta_in
            else:
                stage_in = dequantize_bits(
                    stage_targets[stage_idx - 1], deterministic=False, generator=generator
                )
            pred_stage = flow.forward_slice(stage_in, s_start, s_end)
            s_loss, f_acc, d_acc = self._stage_loss_and_acc(
                pred_stage, stage_targets[stage_idx], stage_idx, eps=eps, w_bit=w_bit
            )
            tf_losses.append(s_loss)
            with torch.no_grad():
                tf_accs.append(f_acc)
                tf_dh_accs.append(d_acc)
        loss_tf = torch.stack(tf_losses).mean()
        tf_bit_acc = torch.stack(tf_accs).mean()
        tf_dh_acc = torch.stack(tf_dh_accs).mean()

        # 2. Chained rollout consistency + tail marginal entropy
        if lip_sigma > 0.0 and hasattr(flow, "flow_layout"):
            theta_final, stage_outputs, loss_lip = flow(
                theta_in,
                stage_endpoints=self.stage_endpoints,
                return_lip=True,
                lip_sigma=lip_sigma,
            )
        else:
            theta_final, stage_outputs = flow(theta_in, stage_endpoints=self.stage_endpoints)
            loss_lip = torch.zeros((), device=h_bits.device, dtype=torch.float32)

        rollout_losses: List[torch.Tensor] = []
        rollout_accs: List[torch.Tensor] = []
        rollout_dh_accs: List[torch.Tensor] = []
        for stage_idx, pred_out in enumerate(stage_outputs):
            r_loss, f_acc, d_acc = self._stage_loss_and_acc(
                pred_out, stage_targets[stage_idx], stage_idx, eps=eps, w_bit=w_bit
            )
            rollout_losses.append(r_loss)
            with torch.no_grad():
                rollout_accs.append(f_acc)
                rollout_dh_accs.append(d_acc)
        loss_rollout = torch.stack(rollout_losses).mean()
        rollout_bit_acc = torch.stack(rollout_accs).mean()
        rollout_dh_acc = torch.stack(rollout_dh_accs).mean()

        marg_out = density(theta_final)
        loss_marg = marg_out["loss_marg"]

        alpha_clamped = max(0.0, min(1.0, float(alpha)))
        total_loss = (1.0 - alpha_clamped) * loss_tf + alpha_clamped * loss_rollout + 0.10 * loss_marg

        return {
            "loss": total_loss,
            "loss_tf": loss_tf.detach(),
            "loss_rollout": loss_rollout.detach(),
            "loss_marg": loss_marg.detach(),
            "loss_lip": loss_lip,
            "tf_bit_acc": tf_bit_acc.detach(),
            "rollout_bit_acc": rollout_bit_acc.detach(),
            "tf_dh_acc": tf_dh_acc.detach(),
            "rollout_dh_acc": rollout_dh_acc.detach(),
            "extracted_bits": marg_out["extracted_bits"].detach(),
            "theta_final": theta_final.detach(),
        }
