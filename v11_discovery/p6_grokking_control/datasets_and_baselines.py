"""Datasets, vectorized tie-aware metrics, pooled SE, z-scores, and 0-param baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from cipher_variants import (
    MASK32,
    CipherVariant,
    variant_digest_and_state,
)

try:
    from src.sha256_ref import pack_int_messages_to_w16, words_to_bits, words_to_spins
except ImportError:
    from sha256_ref import pack_int_messages_to_w16, words_to_bits, words_to_spins  # type: ignore


SEED_SPLIT_A: int = 20261106
SEED_B_TRAIN: int = 20261116
SEED_B_HOLD: int = 998244353
SEED_B_RETEST: int = 777666555

TRACK_A_TARGET_NAMES: Tuple[str, ...] = (
    "HR0_b0",
    "HR0_b1",
    "HR0_b7",
    "HR0_b15",
    "HR0_b23",
    "HR0_b31",
    "HR4_b0",
    "HR4_b1",
    "HR4_b15",
    "HR4_b31",
    "HR0_parity",
    "HR4_parity",
    "Em1_b0",
    "Em1_b7",
    "Em1_b15",
    "Em1_b31",
)

N_POOLS: int = 16


def popcount32(x: torch.Tensor) -> torch.Tensor:
    """Exact bit count of uint32-in-int64 tensor."""
    v = x & MASK32
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    v = (v * 0x01010101) & MASK32
    return (v >> 24) & 0x3F


@dataclass
class TrackAData:
    """All 16,384 domain samples + split indices and target tensors."""

    x_all: torch.Tensor  # (16384, 14) float32 spins
    y_bin_all: torch.Tensor  # (16384, 16) float32 in {0, 1}
    y_cont_all: torch.Tensor  # (16384, 2) float32
    idx_train: torch.Tensor  # (6144,) int64
    idx_holdout: torch.Tensor  # (6144,) int64
    idx_retest: torch.Tensor  # (4096,) int64


@dataclass
class TrackBData:
    """Mutually exclusive train/holdout/retest sets for structural discrimination."""

    x_train: torch.Tensor  # (8192, 512) float32
    y_train: torch.Tensor  # (8192,) float32
    x_holdout: torch.Tensor  # (16384, 512) float32
    y_holdout: torch.Tensor  # (16384,) float32
    x_retest: torch.Tensor  # (16384, 512) float32
    y_retest: torch.Tensor  # (16384,) float32
    xor_popcnt_holdout: torch.Tensor  # (16384,) float32


def build_track_a_data(
    rounds: int,
    variant: CipherVariant,
    device: torch.device | str,
) -> TrackAData:
    """Build 14-bit fixed-padding slice dataset on target device."""
    n_total = 1 << 14
    x_int = torch.arange(n_total, dtype=torch.int64, device=device)
    w16 = pack_int_messages_to_w16(x_int)

    # Chunked forward just in case of tight memory, though 16384 is small
    s_r, h_r = variant_digest_and_state(w16, rounds=rounds, variant=variant)

    x_bits = words_to_bits(x_int)[:, :14]
    x_spins = (1.0 - 2.0 * x_bits.to(torch.float32)).contiguous()

    hr0_bits = words_to_bits(h_r[:, 0])
    hr4_bits = words_to_bits(h_r[:, 4])
    em1_bits = words_to_bits(s_r[:, 5])  # S_R[:, 5] == E_{R-1}

    hr0_pop = popcount32(h_r[:, 0])
    hr4_pop = popcount32(h_r[:, 4])

    y_bin = torch.stack(
        [
            hr0_bits[:, 0],
            hr0_bits[:, 1],
            hr0_bits[:, 7],
            hr0_bits[:, 15],
            hr0_bits[:, 23],
            hr0_bits[:, 31],
            hr4_bits[:, 0],
            hr4_bits[:, 1],
            hr4_bits[:, 15],
            hr4_bits[:, 31],
            hr0_pop & 1,
            hr4_pop & 1,
            em1_bits[:, 0],
            em1_bits[:, 7],
            em1_bits[:, 15],
            em1_bits[:, 31],
        ],
        dim=1,
    ).to(torch.float32)

    y_hw = ((hr0_pop + hr4_pop).to(torch.float32) - 32.0) / 4.0
    y_val = (
        (h_r[:, 0].to(torch.float64) / 4294967296.0 - 0.5) * math.sqrt(12.0)
    ).to(torch.float32)
    y_cont = torch.stack([y_hw, y_val], dim=1)

    gen = torch.Generator(device="cpu").manual_seed(SEED_SPLIT_A)
    perm = torch.randperm(n_total, generator=gen).to(device)
    idx_train = perm[:6144]
    idx_holdout = perm[6144 : 6144 + 6144]
    idx_retest = perm[6144 + 6144 :]

    return TrackAData(
        x_all=x_spins,
        y_bin_all=y_bin,
        y_cont_all=y_cont,
        idx_train=idx_train,
        idx_holdout=idx_holdout,
        idx_retest=idx_retest,
    )


def _sample_track_b_pool(
    n_total: int,
    seed: int,
    rounds: int,
    variant: CipherVariant,
    device: torch.device | str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate stratified (pos first half, neg second half, then shuffled) pool."""
    assert n_total % 2 == 0
    n_half = n_total // 2
    gen = torch.Generator(device="cpu").manual_seed(seed)

    # Positive pairs: (x, x ^ (1 << 31))
    x_pos = torch.randint(0, 1 << 32, (n_half,), dtype=torch.int64, generator=gen)
    x_pos_p = x_pos ^ (1 << 31)

    # Negative pairs: independent (x1, x2)
    x_neg0 = torch.randint(0, 1 << 32, (n_half,), dtype=torch.int64, generator=gen)
    x_neg1 = torch.randint(0, 1 << 32, (n_half,), dtype=torch.int64, generator=gen)

    x0_all = torch.cat([x_pos, x_neg0], dim=0).to(device)
    x1_all = torch.cat([x_pos_p, x_neg1], dim=0).to(device)
    y_all = torch.cat(
        [
            torch.ones(n_half, dtype=torch.float32),
            torch.zeros(n_half, dtype=torch.float32),
        ],
        dim=0,
    ).to(device)

    # Interleave deterministically so every contiguous pool of M=16 has exact 50/50 balance
    perm_pos = torch.randperm(n_half, generator=gen)
    perm_neg = torch.randperm(n_half, generator=gen) + n_half
    interleaved = torch.stack([perm_pos, perm_neg], dim=1).reshape(n_total).to(device)

    x0_all = x0_all[interleaved]
    x1_all = x1_all[interleaved]
    y_all = y_all[interleaved]

    # Run cipher on x0 and x1 in chunks of 8192 to keep peak VRAM small
    def _extract_u(x_words: torch.Tensor) -> torch.Tensor:
        out_chunks = []
        for s in range(0, x_words.shape[0], 8192):
            w16 = pack_int_messages_to_w16(x_words[s : s + 8192])
            _, h_r = variant_digest_and_state(w16, rounds=rounds, variant=variant)
            u = h_r[:, [0, 1, 4, 5]]  # H_R[0, 1, 4, 5] = (A_R, A_{R-1}, E_R, E_{R-1}) + IV
            out_chunks.append(u)
        return torch.cat(out_chunks, dim=0)

    u0 = _extract_u(x0_all)
    u1 = _extract_u(x1_all)
    u_xor = (u0 ^ u1) & MASK32
    u_mod = (u1 - u0) & MASK32

    xor_popcnt = popcount32(u_xor).sum(dim=1).to(torch.float32)

    psi = torch.cat(
        [
            words_to_spins(u0, torch.float32).reshape(n_total, 128),
            words_to_spins(u1, torch.float32).reshape(n_total, 128),
            words_to_spins(u_xor, torch.float32).reshape(n_total, 128),
            words_to_spins(u_mod, torch.float32).reshape(n_total, 128),
        ],
        dim=1,
    ).contiguous()
    return psi, y_all, xor_popcnt


def build_track_b_data(
    rounds: int,
    variant: CipherVariant,
    device: torch.device | str,
) -> TrackBData:
    """Build Track B train/holdout/retest datasets on device."""
    x_tr, y_tr, _ = _sample_track_b_pool(8192, SEED_B_TRAIN, rounds, variant, device)
    x_ho, y_ho, pop_ho = _sample_track_b_pool(16384, SEED_B_HOLD, rounds, variant, device)
    x_re, y_re, _ = _sample_track_b_pool(16384, SEED_B_RETEST, rounds, variant, device)
    return TrackBData(
        x_train=x_tr,
        y_train=y_tr,
        x_holdout=x_ho,
        y_holdout=y_ho,
        x_retest=x_re,
        y_retest=y_re,
        xor_popcnt_holdout=pop_ho,
    )


def compute_auc_batched(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Exact tie-aware Mann-Whitney U ROC-AUC for shapes (..., N).

    Computes average of pessimistic (positives sorted before negatives on ties)
    and optimistic (negatives sorted before positives on ties) rank sums.
    """
    n_pos = labels.sum(dim=-1)
    n_neg = labels.shape[-1] - n_pos
    valid = (n_pos > 0) & (n_neg > 0)

    # Order 1: sort by (1 - labels) ascending -> 1s before 0s -> stable sort by scores
    idx_1_first = torch.argsort(1.0 - labels, dim=-1, stable=True)
    scores_1 = torch.gather(scores, -1, idx_1_first)
    labels_1 = torch.gather(labels, -1, idx_1_first)
    order_pess = torch.argsort(scores_1, dim=-1, stable=True)
    y_pess = torch.gather(labels_1, -1, order_pess)

    # Order 2: sort by labels ascending -> 0s before 1s -> stable sort by scores
    idx_0_first = torch.argsort(labels, dim=-1, stable=True)
    scores_0 = torch.gather(scores, -1, idx_0_first)
    labels_0 = torch.gather(labels, -1, idx_0_first)
    order_opt = torch.argsort(scores_0, dim=-1, stable=True)
    y_opt = torch.gather(labels_0, -1, order_opt)

    ranks = torch.arange(1, scores.shape[-1] + 1, dtype=torch.float64, device=scores.device)
    rank_sum_pess = (y_pess.to(torch.float64) * ranks).sum(dim=-1)
    rank_sum_opt = (y_opt.to(torch.float64) * ranks).sum(dim=-1)
    rank_sum_avg = 0.5 * (rank_sum_pess + rank_sum_opt)

    n_pos_f = n_pos.to(torch.float64).clamp(min=1.0)
    n_neg_f = n_neg.to(torch.float64).clamp(min=1.0)
    u_stat = rank_sum_avg - 0.5 * n_pos_f * (n_pos_f + 1.0)
    auc = (u_stat / (n_pos_f * n_neg_f)).to(torch.float32)
    return torch.where(valid, auc, torch.full_like(auc, 0.5))


def z_null_auc(auc: float, n_pos: int, n_neg: int) -> float:
    """Theoretical Mann-Whitney U z-score against H0: AUC = 0.5."""
    if n_pos <= 0 or n_neg <= 0:
        return 0.0
    var = (n_pos + n_neg + 1.0) / (12.0 * float(n_pos) * float(n_neg))
    return float((auc - 0.5) / math.sqrt(var))


def z_null_acc(acc: float, n_eval: int) -> float:
    """Theoretical binomial z-score against H0: ACC = 0.5."""
    if n_eval <= 0:
        return 0.0
    se = math.sqrt(0.25 / float(n_eval))
    return float((acc - 0.5) / se)


def evaluate_track_a(
    logits: torch.Tensor,
    y_bin: torch.Tensor,
    y_cont: torch.Tensor,
) -> Dict[str, object]:
    """Evaluate Track A predictions on (N, 18) logits with M=16 pools."""
    n = logits.shape[0]
    assert n % N_POOLS == 0
    pool_sz = n // N_POOLS

    bin_logits = logits[:, :16]
    cont_preds = logits[:, 16:18]

    preds = (bin_logits > 0.0).to(torch.float32)
    correct = (preds == y_bin).to(torch.float32)  # (N, 16)
    bit_acc = correct.mean(dim=0)  # (16,)

    # Full-set per-bit AUC: transpose to (16, N)
    bit_auc = compute_auc_batched(bin_logits.T, y_bin.T)  # (16,)

    # Pool-wise metrics: (16_targets, M=16_pools, pool_sz)
    logits_p = bin_logits.T.reshape(16, N_POOLS, pool_sz)
    y_p = y_bin.T.reshape(16, N_POOLS, pool_sz)
    pool_bit_aucs = compute_auc_batched(logits_p, y_p)  # (16, 16)
    pool_bit_accs = correct.T.reshape(16, N_POOLS, pool_sz).mean(dim=-1)  # (16, 16)

    # Aggregate pool metrics across digest12 and all16
    pool_d12_auc = pool_bit_aucs[:12].mean(dim=0)  # (16,)
    pool_d12_acc = pool_bit_accs[:12].mean(dim=0)
    pool_all_auc = pool_bit_aucs.mean(dim=0)
    pool_all_acc = pool_bit_accs.mean(dim=0)

    # MSE per pool
    sq_err = (cont_preds - y_cont).pow(2).mean(dim=-1)  # (N,)
    mse = float(sq_err.mean().item())
    pool_mses = sq_err.reshape(N_POOLS, pool_sz).mean(dim=-1)
    mse_pool_se = float(pool_mses.std(unbiased=True).item() / math.sqrt(N_POOLS))
    z_mse = float((1.0 - mse) / max(mse_pool_se, 1e-12))

    # Per-bit z_null_auc
    n_pos_bits = y_bin.sum(dim=0).tolist()
    bit_z_list = [
        z_null_auc(float(bit_auc[j].item()), int(n_pos_bits[j]), n - int(n_pos_bits[j]))
        for j in range(16)
    ]

    # Best digest12 bit (for strict full-R evaluation) and best overall bit
    d12_aucs = bit_auc[:12]
    best_d12_idx = int(torch.argmax(d12_aucs).item())
    best_all_idx = int(torch.argmax(bit_auc).item())

    digest12_auc = float(d12_aucs.mean().item())
    digest12_acc = float(bit_acc[:12].mean().item())
    digest12_pool_se = float(pool_d12_auc.std(unbiased=True).item() / math.sqrt(N_POOLS))
    # Pooled z_null for 12-target mean (uses pool SE when >0, and also reports single-best z)
    digest12_z_null_pool = float((digest12_auc - 0.5) / max(digest12_pool_se, 1e-12))

    all_auc = float(bit_auc.mean().item())
    all_acc = float(bit_acc.mean().item())
    all_auc_pool_mean = float(pool_all_auc.mean().item())
    all_auc_pool_se = float(pool_all_auc.std(unbiased=True).item() / math.sqrt(N_POOLS))
    all_acc_pool_se = float(pool_all_acc.std(unbiased=True).item() / math.sqrt(N_POOLS))

    # Best digest12 bit pool SE
    best_d12_pool_se = float(
        pool_bit_aucs[best_d12_idx].std(unbiased=True).item() / math.sqrt(N_POOLS)
    )

    return {
        "acc": all_acc,
        "auc": all_auc,
        "auc_pool_mean": all_auc_pool_mean,
        "auc_pool_se": all_auc_pool_se,
        "acc_pool_se": all_acc_pool_se,
        "z_null_auc": float((all_auc - 0.5) / max(all_auc_pool_se, 1e-12)),
        "z_null_acc": z_null_acc(all_acc, n * 16),
        "digest12_acc": digest12_acc,
        "digest12_auc": digest12_auc,
        "digest12_auc_pool_mean": float(pool_d12_auc.mean().item()),
        "digest12_auc_pool_se": digest12_pool_se,
        "digest12_acc_pool_se": float(pool_d12_acc.std(unbiased=True).item() / math.sqrt(N_POOLS)),
        "digest12_z_null_auc": digest12_z_null_pool,
        "best_digest12_bit_name": TRACK_A_TARGET_NAMES[best_d12_idx],
        "best_digest12_bit_idx": best_d12_idx,
        "best_digest12_bit_auc": float(bit_auc[best_d12_idx].item()),
        "best_digest12_bit_pool_se": best_d12_pool_se,
        "best_digest12_bit_z_null": bit_z_list[best_d12_idx],
        "best_bit_name": TRACK_A_TARGET_NAMES[best_all_idx],
        "best_bit_idx": best_all_idx,
        "best_bit_auc": float(bit_auc[best_all_idx].item()),
        "best_bit_z_null": bit_z_list[best_all_idx],
        "per_bit_auc": {TRACK_A_TARGET_NAMES[j]: float(bit_auc[j].item()) for j in range(16)},
        "per_bit_z_null": {TRACK_A_TARGET_NAMES[j]: bit_z_list[j] for j in range(16)},
        "mse": mse,
        "mse_pool_se": mse_pool_se,
        "z_null_mse": z_mse,
    }


def evaluate_track_b(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, object]:
    """Evaluate Track B predictions on (N,) logits with M=16 pools."""
    s = logits.reshape(-1)
    n = s.shape[0]
    assert n % N_POOLS == 0
    pool_sz = n // N_POOLS

    preds = (s > 0.0).to(torch.float32)
    correct = (preds == y).to(torch.float32)
    acc = float(correct.mean().item())
    auc = float(compute_auc_batched(s, y).item())

    pool_aucs = compute_auc_batched(s.reshape(N_POOLS, pool_sz), y.reshape(N_POOLS, pool_sz))
    pool_accs = correct.reshape(N_POOLS, pool_sz).mean(dim=-1)

    auc_pool_mean = float(pool_aucs.mean().item())
    auc_pool_se = float(pool_aucs.std(unbiased=True).item() / math.sqrt(N_POOLS))
    acc_pool_se = float(pool_accs.std(unbiased=True).item() / math.sqrt(N_POOLS))

    n_pos = int(y.sum().item())
    n_neg = n - n_pos
    return {
        "acc": acc,
        "auc": auc,
        "auc_pool_mean": auc_pool_mean,
        "auc_pool_se": auc_pool_se,
        "acc_pool_se": acc_pool_se,
        "z_null_auc": z_null_auc(auc, n_pos, n_neg),
        "z_null_acc": z_null_acc(acc, n),
    }


def compute_zero_param_baselines_a(data: TrackAData) -> Dict[str, object]:
    """0-param null + closed-form 1st-order empirical correlation / OLS baseline."""
    x_tr = data.x_all[data.idx_train]  # (6144, 14)
    yb_tr = data.y_bin_all[data.idx_train]  # (6144, 16)
    yc_tr = data.y_cont_all[data.idx_train]  # (6144, 2)

    x_ho = data.x_all[data.idx_holdout]
    yb_ho = data.y_bin_all[data.idx_holdout]
    yc_ho = data.y_cont_all[data.idx_holdout]

    # Centered target correlation weights w_j = E_train[u_j * (2y - 1)]
    yb_signed = 2.0 * yb_tr - 1.0
    w_corr = (x_tr.T @ yb_signed) / float(x_tr.shape[0])  # (14, 16)
    logits_ho_bin = x_ho @ w_corr  # (6144, 16)

    # Closed-form OLS with bias for continuous targets
    ones_tr = torch.ones((x_tr.shape[0], 1), dtype=torch.float32, device=x_tr.device)
    ones_ho = torch.ones((x_ho.shape[0], 1), dtype=torch.float32, device=x_ho.device)
    x_tr_b = torch.cat([x_tr, ones_tr], dim=1)
    x_ho_b = torch.cat([x_ho, ones_ho], dim=1)
    w_ols = torch.linalg.lstsq(x_tr_b, yc_tr).solution  # (15, 2)
    preds_ho_cont = x_ho_b @ w_ols

    logits_ho = torch.cat([logits_ho_bin, preds_ho_cont], dim=1)
    res_lin = evaluate_track_a(logits_ho, yb_ho, yc_ho)
    return {
        "null_constant": {"acc": 0.5, "auc": 0.5, "mse": 1.0},
        "closed_form_linear": {
            "acc": res_lin["acc"],
            "auc": res_lin["auc"],
            "digest12_acc": res_lin["digest12_acc"],
            "digest12_auc": res_lin["digest12_auc"],
            "best_bit_name": res_lin["best_bit_name"],
            "best_bit_auc": res_lin["best_bit_auc"],
            "mse": res_lin["mse"],
        },
    }


def compute_zero_param_baselines_b(data: TrackBData) -> Dict[str, object]:
    """0-param negative Hamming weight + closed-form diff-linear matched filter."""
    # 1. Strictly 0-param negative Hamming weight (-popcnt + 64 threshold for ACC)
    s0 = 64.0 - data.xor_popcnt_holdout
    res_ham = evaluate_track_b(s0, data.y_holdout)

    # 2. Closed-form differential-linear matched filter (train-only mean diff)
    pos_mask = data.y_train > 0.5
    mu_pos = data.x_train[pos_mask].mean(dim=0)
    mu_neg = data.x_train[~pos_mask].mean(dim=0)
    mu_hat = mu_pos - mu_neg
    midpoint = 0.5 * (mu_pos + mu_neg)
    s_mf = (data.x_holdout - midpoint.unsqueeze(0)) @ mu_hat
    res_mf = evaluate_track_b(s_mf, data.y_holdout)

    return {
        "null_constant": {"acc": 0.5, "auc": 0.5},
        "zero_param_neg_hamming": {"acc": res_ham["acc"], "auc": res_ham["auc"]},
        "diff_linear_matched_filter": {
            "acc": res_mf["acc"],
            "auc": res_mf["auc"],
            "z_null_auc": res_mf["z_null_auc"],
        },
    }
