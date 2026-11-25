"""Exact statistical utilities for P9 held-out evaluation, paired McNemar tests, and Bonferroni correction."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch

from .circuit_slp import count_packed_matches

# Normal quantile for two-sided 99.9% CI (alpha = 0.001, alpha/2 = 0.0005)
Z_999 = 3.2905267314919255


def norm_two_sided_p(z: float) -> float:
    """Two-sided normal p-value 2 * (1 - Phi(|z|)) with underflow safety."""
    az = abs(float(z))
    if az > 37.0:
        return 0.0
    return float(math.erfc(az / math.sqrt(2.0)))


def wilson_ci_999(matches: int, n_eval: int) -> Tuple[float, float]:
    """Exact 99.9% Wilson score confidence interval for binomial proportion."""
    if n_eval <= 0:
        return (0.0, 1.0)
    p = matches / float(n_eval)
    z2 = Z_999 * Z_999
    denom = 1.0 + z2 / n_eval
    center = (p + z2 / (2.0 * n_eval)) / denom
    half = (Z_999 * math.sqrt((p * (1.0 - p) / n_eval) + (z2 / (4.0 * n_eval * n_eval)))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def compute_eval_metrics(
    matches: int,
    n_eval: int,
    n_trials: int,
) -> Dict[str, Any]:
    """Compute p, bias, correlation r, z-score, 99.9% Wilson CI, and Bonferroni p_adj."""
    p = matches / float(n_eval)
    bias = abs(p - 0.5)
    r = 2.0 * p - 1.0
    sigma0 = 1.0 / (2.0 * math.sqrt(n_eval))
    z = bias / sigma0
    ci_lo, ci_hi = wilson_ci_999(matches, n_eval)
    p_raw = norm_two_sided_p(z)
    n_tr = max(1, int(n_trials))
    p_adj = min(1.0, n_tr * p_raw)
    return {
        "n_eval": int(n_eval),
        "matches": int(matches),
        "eval_p": float(p),
        "eval_bias": float(bias),
        "corr_r": float(r),
        "sigma0": float(sigma0),
        "z_score": float(z),
        "wilson_ci999": [float(ci_lo), float(ci_hi)],
        "p_raw": float(p_raw),
        "p_adj_bonferroni": float(p_adj),
        "n_trials": n_tr,
    }


def compute_paired_incremental(
    proxy_pred_packed: torch.Tensor,
    base_pred_packed: torch.Tensor,
    target_packed: torch.Tensor,
    n_eval: int,
) -> Dict[str, Any]:
    """Compute exact paired McNemar incremental bias delta_bias and z_paired on identical samples."""
    ones_word = torch.full_like(target_packed, -1)

    m_c = count_packed_matches(proxy_pred_packed, target_packed)
    p_c = m_c / float(n_eval)
    c_oriented = proxy_pred_packed if p_c >= 0.5 else (proxy_pred_packed ^ ones_word)

    m_b = count_packed_matches(base_pred_packed, target_packed)
    p_b = m_b / float(n_eval)
    b_oriented = base_pred_packed if p_b >= 0.5 else (base_pred_packed ^ ones_word)

    # Bit i is 1 iff oriented predictor agrees with target
    c_right = ~(c_oriented ^ target_packed)
    b_right = ~(b_oriented ^ target_packed)

    # n10: proxy right & base wrong -> match of (c_right & ~b_right) against ones_word
    only_c = c_right & (~b_right)
    only_b = (~c_right) & b_right

    n10 = count_packed_matches(only_c, ones_word)
    n01 = count_packed_matches(only_b, ones_word)

    bias_c = abs(p_c - 0.5)
    bias_b = abs(p_b - 0.5)
    delta_bias = bias_c - bias_b

    # Paired variance of (I_c - I_b) / n_eval
    diff_sq_sum = n10 + n01
    if diff_sq_sum == 0:
        sigma_paired = 1.0 / float(n_eval)
        z_paired = 0.0
    else:
        mean_diff = (n10 - n01) / float(n_eval)
        var_term = max(1e-18, (diff_sq_sum / float(n_eval)) - mean_diff * mean_diff)
        sigma_paired = math.sqrt(var_term / float(n_eval))
        z_paired = delta_bias / sigma_paired

    passed = bool((delta_bias > 0.005) and (z_paired >= 6.0))
    return {
        "proxy_bias": float(bias_c),
        "baseline_bias": float(bias_b),
        "delta_bias": float(delta_bias),
        "n10": int(n10),
        "n01": int(n01),
        "sigma_paired": float(sigma_paired),
        "z_paired": float(z_paired),
        "passed_known_check": passed,
    }
