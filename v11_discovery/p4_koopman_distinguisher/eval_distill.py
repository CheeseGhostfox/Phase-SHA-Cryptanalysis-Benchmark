"""Holdout Pool Evaluation, Zero-Param/DL Baselines, and Walsh/2-adic Spectral Distillation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from .sha256_tuples import (
    CHANNEL_NAMES,
    FAMILY_C_IN,
    WORD_LABELS,
    encode_crystal_features,
    highest_order_diff_sr,
    structured_mod_residual_sr,
    words_to_bits_lsb,
    words_to_spins_fp32,
)


def bit_index_to_name(bit_idx: int) -> str:
    w_idx = bit_idx // 32
    b_idx = bit_idx % 32
    return f"{WORD_LABELS[w_idx]}_b{b_idx}"


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact tie-aware Mann-Whitney ROC-AUC in O(N log N) pure NumPy."""
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y > 0.5).sum())
    n_neg = int(y.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    starts = cum - counts
    avg_ranks = starts.astype(np.float64) + 0.5 * (counts.astype(np.float64) + 1.0)
    ranks = avg_ranks[inv]
    pos_rank_sum = float(ranks[y > 0.5].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def evaluate_logits_pooled(
    logits: np.ndarray,
    labels: np.ndarray,
    pool_k: int = 4096,
) -> Dict[str, float]:
    """Compute full-set ROC-AUC, Acc, bidirectional null z-scores, and pooled AUC mean/SE."""
    n_total = int(labels.shape[0])
    assert n_total % pool_k == 0, f"N={n_total} not divisible by pool_k={pool_k}"
    num_pools = n_total // pool_k

    auc_full = compute_roc_auc(labels, logits)
    preds = (logits > 0.0).astype(np.float64)
    acc_full = float((preds == labels).mean())

    n_pos = int((labels > 0.5).sum())
    n_neg = n_total - n_pos
    sigma_auc_0 = math.sqrt((n_pos + n_neg + 1.0) / (12.0 * n_pos * n_neg))
    z_null = (auc_full - 0.5) / sigma_auc_0
    z_acc_null = (acc_full - 0.5) / math.sqrt(0.25 / float(n_total))

    pool_aucs: List[float] = []
    # Pair equal pos/neg slices per pool if labels are [pos..., neg...], else slice directly
    pos_idx = np.where(labels > 0.5)[0]
    neg_idx = np.where(labels <= 0.5)[0]
    half_k = pool_k // 2
    if len(pos_idx) == n_total // 2 and len(neg_idx) == n_total // 2:
        for m in range(num_pools):
            idx_m = np.concatenate(
                [pos_idx[m * half_k : (m + 1) * half_k], neg_idx[m * half_k : (m + 1) * half_k]]
            )
            pool_aucs.append(compute_roc_auc(labels[idx_m], logits[idx_m]))
    else:
        for m in range(num_pools):
            s = slice(m * pool_k, (m + 1) * pool_k)
            pool_aucs.append(compute_roc_auc(labels[s], logits[s]))

    arr = np.asarray(pool_aucs, dtype=np.float64)
    auc_pool_mean = float(arr.mean())
    auc_pool_se = float(arr.std(ddof=1) / math.sqrt(num_pools)) if num_pools > 1 else 0.0

    return {
        "auc": float(auc_full),
        "auc_pool_mean": auc_pool_mean,
        "auc_pool_se": auc_pool_se,
        "acc": float(acc_full),
        "z_null": float(z_null),
        "z_acc_null": float(z_acc_null),
    }


@torch.no_grad()
def predict_model_logits(
    model: torch.nn.Module,
    c_tuples: torch.Tensor,
    family: str,
    batch_size: int = 1024,
) -> np.ndarray:
    """Run pure FP32 inference in chunks and return numpy float64 logits."""
    model.eval()
    n_total = c_tuples.shape[0]
    outs: List[np.ndarray] = []
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        psi = encode_crystal_features(c_tuples[start:end], family)
        logits = model(psi)
        outs.append(logits.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def evaluate_zero_param_baselines(
    sr_cal: torch.Tensor,
    labels_cal: torch.Tensor,
    sr_eval: torch.Tensor,
    labels_eval: torch.Tensor,
    family: str,
    chunk_size: int = 8192,
) -> Dict[str, Any]:
    """Evaluate 0-parameter Hamming weight bias and single-bit / linear DL probe."""
    y_eval = labels_eval.detach().cpu().numpy().astype(np.float64)
    n_total = int(sr_eval.shape[0])

    # 1) Calibration statistics for DL bit probe
    sum_pos = torch.zeros(256, dtype=torch.float64, device=sr_cal.device)
    sum_neg = torch.zeros(256, dtype=torch.float64, device=sr_cal.device)
    cnt_pos = 0
    cnt_neg = 0
    for start in range(0, sr_cal.shape[0], chunk_size):
        end = min(start + chunk_size, sr_cal.shape[0])
        diff_c = highest_order_diff_sr(sr_cal[start:end], family)
        b_c = words_to_spins_fp32(diff_c).reshape(-1, 256).to(torch.float64)
        lbl_c = labels_cal[start:end] > 0.5
        if lbl_c.any():
            sum_pos += b_c[lbl_c].sum(dim=0)
            cnt_pos += int(lbl_c.sum().item())
        if (~lbl_c).any():
            sum_neg += b_c[~lbl_c].sum(dim=0)
            cnt_neg += int((~lbl_c).sum().item())

    mu_hat = (sum_pos / max(1, cnt_pos)) - (sum_neg / max(1, cnt_neg))
    best_bit_idx = int(torch.argmax(mu_hat.abs()).item())
    best_sign = 1.0 if float(mu_hat[best_bit_idx].item()) >= 0.0 else -1.0

    # 2) Eval chunked pass for HW, single-bit, and linear matched filter
    hw_list: List[np.ndarray] = []
    single_list: List[np.ndarray] = []
    linear_list: List[np.ndarray] = []

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        diff_e = highest_order_diff_sr(sr_eval[start:end], family)
        bits_e = words_to_bits_lsb(diff_e)
        hw = bits_e.sum(dim=(1, 2)).cpu().numpy().astype(np.int64)
        hw_list.append(hw)

        b_e = (1.0 - 2.0 * bits_e.to(torch.float64)).reshape(-1, 256)
        single_s = (best_sign * b_e[:, best_bit_idx]).cpu().numpy()
        linear_s = (b_e @ mu_hat).cpu().numpy()
        single_list.append(single_s)
        linear_list.append(linear_s)

    hw_arr = np.concatenate(hw_list, axis=0)
    single_arr = np.concatenate(single_list, axis=0)
    linear_arr = np.concatenate(linear_list, axis=0)

    # Optimal threshold accuracy over integer HW in [0..256]
    pos_hist = np.bincount(hw_arr[y_eval > 0.5], minlength=257)
    neg_hist = np.bincount(hw_arr[y_eval <= 0.5], minlength=257)
    # Classify hw <= thr as positive (since s0 = -hw)
    tp_le = np.cumsum(pos_hist)
    fp_le = np.cumsum(neg_hist)
    tn_gt = int((y_eval <= 0.5).sum()) - fp_le
    acc_le = (tp_le + tn_gt) / float(n_total)
    # Also check opposite orientation just in case
    best_hw_acc = float(max(acc_le.max(), (1.0 - acc_le).max()))

    return {
        "zero_param_hw_auc": compute_roc_auc(y_eval, -hw_arr.astype(np.float64)),
        "zero_param_hw_acc": best_hw_acc,
        "dl_single_bit_auc": compute_roc_auc(y_eval, single_arr),
        "dl_linear_auc": compute_roc_auc(y_eval, linear_arr),
        "best_bit_idx": best_bit_idx,
        "best_bit_name": bit_index_to_name(best_bit_idx),
    }


def _solve_ridge_np(x_fit: np.ndarray, y_fit: np.ndarray, alpha: float = 1e-3) -> Tuple[np.ndarray, float]:
    """Closed-form float64 Ridge regression with intercept."""
    x_mean = x_fit.mean(axis=0, keepdims=True)
    y_mean = float(y_fit.mean())
    xc = x_fit - x_mean
    yc = y_fit - y_mean
    d = xc.shape[1]
    gram = xc.T @ xc + alpha * np.eye(d, dtype=np.float64)
    w = np.linalg.solve(gram, xc.T @ yc)
    b0 = y_mean - float((x_mean @ w).item())
    return w, b0


def _build_two_adic_features(
    sr_tuples: torch.Tensor, family: str, chunk_size: int = 8192
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Build 192-dim 2-adic modular phase features (8 words x 4 scales x 3 harmonics x 2)."""
    scales = (4, 8, 16, 32)
    kappas = (1, 2, 4)
    meta: List[Dict[str, Any]] = []
    for w_idx in range(8):
        for p in scales:
            for kap in kappas:
                meta.append({"word": WORD_LABELS[w_idx], "w_idx": w_idx, "p": p, "kappa": kap, "func": "cos"})
                meta.append({"word": WORD_LABELS[w_idx], "w_idx": w_idx, "p": p, "kappa": kap, "func": "sin"})

    chunks: List[np.ndarray] = []
    for start in range(0, sr_tuples.shape[0], chunk_size):
        end = min(start + chunk_size, sr_tuples.shape[0])
        dw = structured_mod_residual_sr(sr_tuples[start:end], family).cpu().numpy().astype(np.uint64)
        cols: List[np.ndarray] = []
        for p in scales:
            mask = (1 << p) - 1
            denom = float(1 << p)
            frac = ((dw & np.uint64(mask)).astype(np.float64)) / denom  # (B, 8)
            for kap in kappas:
                ang = (2.0 * math.pi * float(kap)) * frac
                cols.append(np.cos(ang))
                cols.append(np.sin(ang))
        # Reorder to match meta: outer w_idx, inner (p, kap, cos/sin)
        # Currently cols has 4*3*2 = 24 matrices of shape (B, 8). Stack -> (B, 24, 8) -> transpose -> (B, 8, 24)
        stacked = np.stack(cols, axis=1).transpose(0, 2, 1).reshape(end - start, 192)
        chunks.append(stacked)

    return np.concatenate(chunks, axis=0), meta


def distill_koopman_model(
    model: torch.nn.Module,
    family: str,
    rounds: int,
    sr_fit: torch.Tensor,
    c_fit: torch.Tensor,
    sr_eval: torch.Tensor,
    c_eval: torch.Tensor,
    labels_eval: np.ndarray,
    z_eval: np.ndarray,
    batch_size: int = 1024,
) -> Dict[str, Any]:
    """
    Perform Walsh-Hadamard (order 1 & 2), 2-adic phase harmonic, and 3D crystal gradient distillation.
    Selection and fitting happen strictly on (sr_fit, c_fit); R^2 and surrogate AUC are evaluated on holdout.
    """
    z_fit = predict_model_logits(model, c_fit, family, batch_size=batch_size)
    f_fit = z_fit - float(z_fit.mean())
    f_eval = z_eval - float(z_eval.mean())
    f_eval_ss = float(np.sum(f_eval * f_eval)) + 1e-12

    # --- 1) Walsh-Hadamard Boolean Spectrum ---
    with torch.no_grad():
        b_fit = (
            words_to_spins_fp32(highest_order_diff_sr(sr_fit, family))
            .reshape(-1, 256)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        b_eval = (
            words_to_spins_fp32(highest_order_diff_sr(sr_eval, family))
            .reshape(-1, 256)
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    f_fit_std = float(np.std(f_fit)) + 1e-12
    f_fit_norm = f_fit / f_fit_std
    # Order-1 correlations
    b1_c = b_fit - b_fit.mean(axis=0, keepdims=True)
    b1_std = b1_c.std(axis=0) + 1e-12
    corr1 = (f_fit_norm @ b1_c) / (float(b_fit.shape[0]) * b1_std)

    top24_idx = np.argsort(np.abs(corr1))[::-1][:24].tolist()
    pair_indices: List[Tuple[int, int]] = []
    pair_fit_cols: List[np.ndarray] = []
    for a_i in range(len(top24_idx)):
        for b_i in range(a_i + 1, len(top24_idx)):
            i, j = top24_idx[a_i], top24_idx[b_i]
            pair_indices.append((i, j))
            pair_fit_cols.append(b_fit[:, i] * b_fit[:, j])

    b2_fit = np.stack(pair_fit_cols, axis=1)  # (N_fit, 276)
    b2_c = b2_fit - b2_fit.mean(axis=0, keepdims=True)
    b2_std = b2_c.std(axis=0) + 1e-12
    corr2 = (f_fit_norm @ b2_c) / (float(b_fit.shape[0]) * b2_std)

    candidates: List[Tuple[float, int, Tuple[int, ...], float]] = []
    for j in range(256):
        candidates.append((abs(float(corr1[j])), 1, (j,), float(corr1[j])))
    for p_idx, (i, j) in enumerate(pair_indices):
        candidates.append((abs(float(corr2[p_idx])), 2, (i, j), float(corr2[p_idx])))

    candidates.sort(key=lambda t: t[0], reverse=True)
    top16 = candidates[:16]

    x_w_fit_cols = []
    x_w_eval_cols = []
    for _, order, idxs, _ in top16:
        if order == 1:
            x_w_fit_cols.append(b_fit[:, idxs[0]])
            x_w_eval_cols.append(b_eval[:, idxs[0]])
        else:
            x_w_fit_cols.append(b_fit[:, idxs[0]] * b_fit[:, idxs[1]])
            x_w_eval_cols.append(b_eval[:, idxs[0]] * b_eval[:, idxs[1]])
    x_w_fit = np.stack(x_w_fit_cols, axis=1)
    x_w_eval = np.stack(x_w_eval_cols, axis=1)

    w_walsh, b0_walsh = _solve_ridge_np(x_w_fit, f_fit, alpha=1e-3)
    pred_w_eval = x_w_eval @ w_walsh + b0_walsh
    walsh_r2 = float(1.0 - np.sum((f_eval - pred_w_eval) ** 2) / f_eval_ss)
    walsh_auc = compute_roc_auc(labels_eval, pred_w_eval)

    walsh_top16_json: List[Dict[str, Any]] = []
    for k_i, (_, order, idxs, c_val) in enumerate(top16):
        term_name = (
            bit_index_to_name(idxs[0])
            if order == 1
            else f"{bit_index_to_name(idxs[0])}*{bit_index_to_name(idxs[1])}"
        )
        walsh_top16_json.append(
            {
                "term": term_name,
                "order": order,
                "corr": float(c_val),
                "coef": float(w_walsh[k_i]),
            }
        )

    # --- 2) 2-adic Carry Phase Spectrum ---
    phi_fit, phi_meta = _build_two_adic_features(sr_fit, family)
    phi_eval, _ = _build_two_adic_features(sr_eval, family)

    phi_std = phi_fit.std(axis=0) + 1e-8
    phi_mean = phi_fit.mean(axis=0, keepdims=True)
    phi_fit_norm = (phi_fit - phi_mean) / phi_std
    w_all_2adic, _ = _solve_ridge_np(phi_fit_norm, f_fit, alpha=10.0)
    top12_idx = np.argsort(np.abs(w_all_2adic))[::-1][:12].tolist()

    w_2adic, b0_2adic = _solve_ridge_np(phi_fit[:, top12_idx], f_fit, alpha=1e-2)
    pred_2a_eval = phi_eval[:, top12_idx] @ w_2adic + b0_2adic
    two_adic_r2 = float(1.0 - np.sum((f_eval - pred_2a_eval) ** 2) / f_eval_ss)
    two_adic_auc = compute_roc_auc(labels_eval, pred_2a_eval)

    two_adic_top12_json: List[Dict[str, Any]] = []
    for rank_i, col_i in enumerate(top12_idx):
        m = phi_meta[col_i]
        two_adic_top12_json.append(
            {
                "word": m["word"],
                "p": m["p"],
                "kappa": m["kappa"],
                "func": m["func"],
                "coef": float(w_2adic[rank_i]),
            }
        )

    # --- 3) 3D Spacetime Crystal Gradient Attribution ---
    pos_mask = labels_eval > 0.5
    c_pos = c_eval[torch.from_numpy(pos_mask).to(c_eval.device)]
    n_grad = min(2048, int(c_pos.shape[0]))
    c_pos_sub = c_pos[:n_grad]
    grad_accum = torch.zeros(
        (FAMILY_C_IN[family], 4, 2, 32), dtype=torch.float32, device=c_eval.device
    )
    model.eval()
    for start in range(0, n_grad, 512):
        end = min(start + 512, n_grad)
        psi = encode_crystal_features(c_pos_sub[start:end], family).detach().requires_grad_(True)
        out = model(psi).sum()
        g = torch.autograd.grad(out, psi)[0].detach().abs().sum(dim=0)
        grad_accum += g
    grad_mean = (grad_accum / float(max(1, n_grad))).cpu().numpy()  # (C_in, 4, 2, 32)

    round_names = [f"R-{3 - t}" if t < 3 else "R" for t in range(4)]
    by_round_vals = grad_mean.mean(axis=(0, 2, 3))
    by_track_vals = grad_mean.mean(axis=(0, 1, 3))
    saliency_by_round = {round_names[t]: float(by_round_vals[t]) for t in range(4)}
    saliency_by_round["track_A"] = float(by_track_vals[0])
    saliency_by_round["track_E"] = float(by_track_vals[1])

    ch_names = CHANNEL_NAMES[FAMILY_C_IN[family]]
    by_ch_vals = grad_mean.mean(axis=(1, 2, 3))
    saliency_by_channel = {ch_names[i]: float(by_ch_vals[i]) for i in range(len(ch_names))}

    return {
        "rounds": int(rounds),
        "family": family,
        "walsh_top16": walsh_top16_json,
        "walsh_r2": walsh_r2,
        "walsh_surrogate_auc": walsh_auc,
        "two_adic_top12": two_adic_top12_json,
        "two_adic_r2": two_adic_r2,
        "two_adic_surrogate_auc": two_adic_auc,
        "crystal_saliency_by_round": saliency_by_round,
        "crystal_saliency_by_channel": saliency_by_channel,
    }
