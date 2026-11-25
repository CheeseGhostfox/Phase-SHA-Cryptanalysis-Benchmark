"""GrokkingResMLP architecture and Walsh/Fourier + SAE causal circuit dissection."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets_and_baselines import (
    TRACK_A_TARGET_NAMES,
    TrackAData,
    TrackBData,
    compute_auc_batched,
)


class ResidualMLPBlock(nn.Module):
    """Pre-LN residual MLP block with Dropout=0.0."""

    def __init__(self, d_model: int = 192, d_hidden: int = 384) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_hidden)
        self.w2 = nn.Linear(d_hidden, d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.w2(F.gelu(self.w1(self.ln(h))))


class GrokkingResMLP(nn.Module):
    """Unbiased Pre-LN residual MLP backbone (D=192, 4 blocks, Dropout=0)."""

    def __init__(self, d_in: int, d_out: int, d_model: int = 192, n_blocks: int = 4) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.in_ln = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(d_model=d_model, d_hidden=2 * d_model) for _ in range(n_blocks)]
        )
        self.head_ln = nn.LayerNorm(d_model)
        self.head_out = nn.Linear(d_model, d_out)

    def forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.in_ln(self.in_proj(x)))
        for blk in self.blocks:
            h = blk(h)
        return h

    def forward_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        return self.head_out(self.head_ln(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_from_hidden(self.forward_hidden(x))


def fwht_1d_numpy(vec: np.ndarray) -> np.ndarray:
    """Normalized Fast Walsh-Hadamard Transform over length 2^n vector.

    Bit b of index i corresponds to spin (1 - 2 * ((i >> b) & 1)), matching
    LSB-first ordering in words_to_spins.
    """
    n = vec.shape[0]
    assert (n & (n - 1)) == 0, "Length must be a power of 2"
    a = vec.astype(np.float64, copy=True)
    h = 1
    while h < n:
        a = a.reshape(-1, 2, h)
        x0 = a[:, 0, :].copy()
        x1 = a[:, 1, :].copy()
        a[:, 0, :] = x0 + x1
        a[:, 1, :] = x0 - x1
        a = a.reshape(n)
        h <<= 1
    return a / float(n)


def _popcount_array(n: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.int32)
    deg = np.zeros(n, dtype=np.int32)
    tmp = idx.copy()
    while np.any(tmp):
        deg += tmp & 1
        tmp >>= 1
    return deg


@torch.no_grad()
def dissect_track_a_walsh_fourier(
    model: GrokkingResMLP,
    data: TrackAData,
    target_idx: int,
    batch_size: int = 4096,
) -> Dict[str, object]:
    """Exact 2^14 WHT and 1D FFT on model logits vs true boolean target."""
    model.eval()
    logits_chunks = []
    for s in range(0, data.x_all.shape[0], batch_size):
        logits_chunks.append(model(data.x_all[s : s + batch_size])[:, target_idx])
    f_raw = torch.cat(logits_chunks, dim=0).detach().cpu().numpy().astype(np.float64)

    y_bits = data.y_bin_all[:, target_idx].detach().cpu().numpy().astype(np.float64)
    y_spin = 1.0 - 2.0 * y_bits

    f_centered = f_raw - f_raw.mean()
    y_centered = y_spin - y_spin.mean()

    f_hat = fwht_1d_numpy(f_centered)
    y_hat = fwht_1d_numpy(y_centered)

    power = f_hat * f_hat
    total_power = float(power.sum()) + 1e-18
    probs = power / total_power
    spectral_entropy = float(-np.sum(probs * np.log2(probs + 1e-18)))

    sorted_power = np.sort(power)[::-1]
    top_k_energy = {
        str(k): float(sorted_power[:k].sum() / total_power) for k in (8, 32, 128)
    }

    degs = _popcount_array(f_hat.shape[0])
    degree_energy = {
        str(d): float(power[degs == d].sum() / total_power) for d in range(15)
    }

    denom = (np.linalg.norm(f_hat) * np.linalg.norm(y_hat)) + 1e-18
    cosine_to_target = float(np.dot(f_hat, y_hat) / denom)

    # Also compute sign-aligned cosine (since logit = log(p/(1-p)) flips sign relative to 1-2y)
    top32_idx = np.argsort(power)[::-1][:32]
    top32_terms = []
    for m in top32_idx:
        mask_int = int(m)
        bits = [b for b in range(14) if (mask_int >> b) & 1]
        top32_terms.append(
            {
                "mask": mask_int,
                "degree": int(degs[mask_int]),
                "bits": bits,
                "f_hat": float(f_hat[mask_int]),
                "y_hat": float(y_hat[mask_int]),
                "energy_frac": float(power[mask_int] / total_power),
            }
        )

    # 1D real FFT along integer order x = 0..16383
    fft_vals = np.fft.rfft(f_centered)
    fft_power = np.abs(fft_vals) ** 2
    fft_power[0] = 0.0  # exclude DC
    fft_total = float(fft_power.sum()) + 1e-18
    fft_top_idx = np.argsort(fft_power)[::-1][:16]
    fft_top16_ratio = float(fft_power[fft_top_idx].sum() / fft_total)
    fft_top8 = [
        {"freq": int(k), "power_frac": float(fft_power[k] / fft_total)}
        for k in fft_top_idx[:8]
    ]

    return {
        "target_name": TRACK_A_TARGET_NAMES[target_idx],
        "target_idx": target_idx,
        "top_k_energy": top_k_energy,
        "spectral_entropy": spectral_entropy,
        "degree_energy": degree_energy,
        "cosine_to_target": cosine_to_target,
        "abs_cosine_to_target": abs(cosine_to_target),
        "top32_walsh_terms": top32_terms,
        "fft_top16_ratio": fft_top16_ratio,
        "fft_top8": fft_top8,
    }


def _track_b_bit_label(idx: int) -> str:
    family = "XOR" if idx < 128 else "MOD"
    local = idx % 128
    word_names = ("HR0", "HR1", "HR4", "HR5")
    w = local // 32
    b = local % 32
    return f"{family}_{word_names[w]}_b{b}"


@torch.no_grad()
def dissect_track_b_walsh(
    model: GrokkingResMLP,
    data: TrackBData,
    batch_size: int = 4096,
) -> Dict[str, object]:
    """Differential crystal Walsh interaction spectrum + Top-16 out-of-sample surrogate."""
    model.eval()

    def _eval_logits(x: torch.Tensor) -> torch.Tensor:
        out = []
        for s in range(0, x.shape[0], batch_size):
            out.append(model(x[s : s + batch_size]).reshape(-1))
        return torch.cat(out, dim=0)

    z_tr = _eval_logits(data.x_train)
    z_ho = _eval_logits(data.x_holdout)

    f_tr = z_tr - z_tr.mean()
    f_ho = z_ho - z_ho.mean()

    delta_tr = data.x_train[:, 256:512]  # (N_tr, 256) in {-1, +1}
    delta_ho = data.x_holdout[:, 256:512]  # (N_ho, 256)

    # 1st-order correlations on train (for feature selection) and holdout (for reporting)
    f_tr_norm = f_tr / (f_tr.std() + 1e-12)
    f_ho_norm = f_ho / (f_ho.std() + 1e-12)
    w1_tr = (delta_tr.T @ f_tr_norm) / float(delta_tr.shape[0])  # (256,)
    w1_ho = (delta_ho.T @ f_ho_norm) / float(delta_ho.shape[0])  # (256,)

    top24_idx = torch.topk(w1_tr.abs(), k=24).indices.tolist()

    # Build 276 pairwise interaction terms among top24
    pair_indices: List[Tuple[int, int]] = []
    for i in range(len(top24_idx)):
        for j in range(i + 1, len(top24_idx)):
            pair_indices.append((top24_idx[i], top24_idx[j]))

    idx_a = torch.tensor([p[0] for p in pair_indices], dtype=torch.long, device=delta_tr.device)
    idx_b = torch.tensor([p[1] for p in pair_indices], dtype=torch.long, device=delta_tr.device)

    phi2_tr = delta_tr[:, idx_a] * delta_tr[:, idx_b]  # (N_tr, 276)
    phi2_ho = delta_ho[:, idx_a] * delta_ho[:, idx_b]  # (N_ho, 276)

    w2_tr = (phi2_tr.T @ f_tr_norm) / float(delta_tr.shape[0])  # (276,)
    w2_ho = (phi2_ho.T @ f_ho_norm) / float(delta_ho.shape[0])  # (276,)

    cand_tr = torch.cat([delta_tr, phi2_tr], dim=1)  # (N_tr, 532)
    cand_ho = torch.cat([delta_ho, phi2_ho], dim=1)  # (N_ho, 532)
    w_all_tr = torch.cat([w1_tr, w2_tr], dim=0)
    w_all_ho = torch.cat([w1_ho, w2_ho], dim=0)

    top16_cand = torch.topk(w_all_tr.abs(), k=16).indices.tolist()
    top16_names = []
    for c_idx in top16_cand:
        if c_idx < 256:
            top16_names.append(
                {
                    "term": _track_b_bit_label(c_idx),
                    "order": 1,
                    "corr_holdout": float(w_all_ho[c_idx].item()),
                }
            )
        else:
            a, b = pair_indices[c_idx - 256]
            top16_names.append(
                {
                    "term": f"{_track_b_bit_label(a)}*{_track_b_bit_label(b)}",
                    "order": 2,
                    "corr_holdout": float(w_all_ho[c_idx].item()),
                }
            )

    # Fit linear surrogate on train, evaluate R^2 and AUC on holdout
    ones_tr = torch.ones((cand_tr.shape[0], 1), dtype=torch.float32, device=cand_tr.device)
    ones_ho = torch.ones((cand_ho.shape[0], 1), dtype=torch.float32, device=cand_ho.device)
    feat_tr = torch.cat([cand_tr[:, top16_cand], ones_tr], dim=1)
    feat_ho = torch.cat([cand_ho[:, top16_cand], ones_ho], dim=1)

    coef = torch.linalg.lstsq(feat_tr, f_tr.unsqueeze(1)).solution.squeeze(1)
    f_pred_ho = feat_ho @ coef
    ss_res = float((f_ho - f_pred_ho).pow(2).sum().item())
    ss_tot = float(f_ho.pow(2).sum().item()) + 1e-12
    r2_walsh = float(1.0 - ss_res / ss_tot)
    walsh_auc = float(compute_auc_batched(f_pred_ho, data.y_holdout).item())

    # Spectral concentration summary across the 532 terms on holdout
    p_ho = (w_all_ho ** 2).detach().cpu().numpy().astype(np.float64)
    p_tot = float(p_ho.sum()) + 1e-18
    p_norm = p_ho / p_tot
    spec_ent = float(-np.sum(p_norm * np.log2(p_norm + 1e-18)))

    return {
        "r2_walsh": r2_walsh,
        "walsh_surrogate_auc": walsh_auc,
        "spectral_entropy_532": spec_ent,
        "top_k_energy_532": {
            "8": float(np.sort(p_ho)[::-1][:8].sum() / p_tot),
            "16": float(np.sort(p_ho)[::-1][:16].sum() / p_tot),
            "32": float(np.sort(p_ho)[::-1][:32].sum() / p_tot),
        },
        "top16_walsh_terms": top16_names,
    }


class SparseAutoencoder(nn.Module):
    """4x overcomplete ReLU SAE (192 -> 768 -> 192) with unit-norm decoder columns."""

    def __init__(self, d_in: int = 192, d_dict: int = 768) -> None:
        super().__init__()
        self.b_pre = nn.Parameter(torch.zeros(d_in, dtype=torch.float32))
        self.w_enc = nn.Parameter(torch.empty(d_dict, d_in, dtype=torch.float32))
        self.b_enc = nn.Parameter(torch.zeros(d_dict, dtype=torch.float32))
        self.w_dec = nn.Parameter(torch.empty(d_in, d_dict, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.w_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.w_dec.copy_(self.w_enc.T)
            self.normalize_decoder_()

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        norms = self.w_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.w_dec.div_(norms)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return F.relu(F.linear(h - self.b_pre, self.w_enc, self.b_enc))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.w_dec, self.b_pre)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(h)
        h_hat = self.decode(z)
        return h_hat, z


def dissect_sae_and_ablate(
    model: GrokkingResMLP,
    x_train: torch.Tensor,
    x_holdout: torch.Tensor,
    y_holdout: torch.Tensor,
    target_idx: int = 0,
    sae_steps: int = 2000,
    lambda_l1: float = 5e-3,
    seed: int = 20260606,
) -> Dict[str, object]:
    """Train 4x overcomplete SAE on last residual activations and perform Top-8 causal ablation."""
    model.eval()
    device = x_train.device

    with torch.no_grad():
        h_tr_chunks = [model.forward_hidden(x_train[s : s + 4096]) for s in range(0, x_train.shape[0], 4096)]
        h_tr = torch.cat(h_tr_chunks, dim=0)
        h_ho_chunks = [model.forward_hidden(x_holdout[s : s + 4096]) for s in range(0, x_holdout.shape[0], 4096)]
        h_ho = torch.cat(h_ho_chunks, dim=0)

    torch.manual_seed(seed)
    sae = SparseAutoencoder(d_in=h_tr.shape[1], d_dict=4 * h_tr.shape[1]).to(device)
    with torch.no_grad():
        sae.b_pre.copy_(h_tr.mean(dim=0))

    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    n_tr = h_tr.shape[0]
    bs = min(2048, n_tr)
    gen = torch.Generator(device=device).manual_seed(seed)

    sae.train()
    for _ in range(sae_steps):
        if bs == n_tr:
            hb = h_tr
        else:
            idx = torch.randint(0, n_tr, (bs,), generator=gen, device=device)
            hb = h_tr[idx]
        h_hat, z = sae(hb)
        recon_loss = (hb - h_hat).pow(2).sum(dim=-1).mean()
        l1_loss = z.abs().sum(dim=-1).mean()
        loss = recon_loss + lambda_l1 * l1_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.normalize_decoder_()

    sae.eval()
    with torch.no_grad():
        h_hat_ho, z_ho = sae(h_ho)
        ss_res = float((h_ho - h_hat_ho).pow(2).sum().item())
        h_centered = h_ho - h_ho.mean(dim=0, keepdim=True)
        ss_tot = float(h_centered.pow(2).sum().item()) + 1e-12
        r2_sae = float(1.0 - ss_res / ss_tot)

        active_mask = z_ho > 0.0
        l0_mean = float(active_mask.to(torch.float32).sum(dim=-1).mean().item())
        act_freq = active_mask.to(torch.float32).mean(dim=0)  # (768,)
        dead_frac = float((act_freq < 1e-4).to(torch.float32).mean().item())

        # Correlate each SAE feature with target y_holdout
        y_c = y_holdout - y_holdout.mean()
        y_std = y_c.std() + 1e-12
        z_c = z_ho - z_ho.mean(dim=0, keepdim=True)
        z_std = z_ho.std(dim=0) + 1e-12
        corr = (z_c.T @ y_c) / (float(h_ho.shape[0]) * z_std * y_std)
        corr = torch.where(act_freq >= 1e-4, corr, torch.zeros_like(corr))

        top8_idx = torch.topk(corr.abs(), k=8).indices.tolist()
        top8_features = [
            {
                "feature_idx": int(fid),
                "corr_to_target": float(corr[fid].item()),
                "act_freq": float(act_freq[fid].item()),
                "mean_act": float(z_ho[:, fid].mean().item()),
            }
            for fid in top8_idx
        ]

        # Causal ablation: zero Top-8 features in z_ho and pass reconstructed h_abl to head
        z_abl = z_ho.clone()
        z_abl[:, top8_idx] = 0.0
        h_abl_ho = sae.decode(z_abl)

        logits_orig = model.forward_from_hidden(h_ho)[:, target_idx]
        logits_sae = model.forward_from_hidden(h_hat_ho)[:, target_idx]
        logits_abl = model.forward_from_hidden(h_abl_ho)[:, target_idx]

        auc_orig = float(compute_auc_batched(logits_orig, y_holdout).item())
        auc_sae = float(compute_auc_batched(logits_sae, y_holdout).item())
        auc_abl = float(compute_auc_batched(logits_abl, y_holdout).item())

        acc_orig = float(((logits_orig > 0.0).to(torch.float32) == y_holdout).to(torch.float32).mean().item())
        acc_sae = float(((logits_sae > 0.0).to(torch.float32) == y_holdout).to(torch.float32).mean().item())
        acc_abl = float(((logits_abl > 0.0).to(torch.float32) == y_holdout).to(torch.float32).mean().item())

    return {
        "r2_sae": r2_sae,
        "l0_mean": l0_mean,
        "dead_frac": dead_frac,
        "top8_features": top8_features,
        "auc_orig": auc_orig,
        "auc_sae_recon": auc_sae,
        "auc_ablated": auc_abl,
        "causal_delta_auc_ablate": float(auc_sae - auc_abl),
        "causal_delta_auc_vs_orig": float(auc_orig - auc_abl),
        "acc_orig": acc_orig,
        "acc_sae_recon": acc_sae,
        "acc_ablated": acc_abl,
        "causal_delta_acc_ablate": float(acc_sae - acc_abl),
    }
