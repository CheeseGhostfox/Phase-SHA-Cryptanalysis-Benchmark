"""Holdout Pool Evaluation, Ridge-CCA Baseline, and CCA/Lasso/Walsh Distillation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .sha256_telescope import IN_FEATURE_NAMES, OUT_FEATURE_NAMES


@torch.no_grad()
def evaluate_embeddings_pooled(
    u_all: torch.Tensor,
    v_all: torch.Tensor,
    scale: float,
    pool_k: int = 1024,
    num_perm: int = 64,
    perm_seed: int = 777777,
) -> Dict[str, float]:
    """Evaluate pooled bidirectional Top-1/Top-5, InfoNCE MI (bits), and permutation z_null."""
    n_total = u_all.shape[0]
    assert n_total % pool_k == 0, f"N={n_total} not divisible by pool_k={pool_k}"
    num_pools = n_total // pool_k
    device = u_all.device

    mi_list: List[float] = []
    top1_list: List[float] = []
    top5_list: List[float] = []

    log2_k = math.log2(pool_k)
    ln2 = math.log(2.0)

    u_pools = u_all.view(num_pools, pool_k, -1)
    v_pools = v_all.view(num_pools, pool_k, -1)

    for m in range(num_pools):
        um = u_pools[m]
        vm = v_pools[m]
        sim = um @ vm.T  # (K, K)
        logits = sim * scale
        labels = torch.arange(pool_k, device=device, dtype=torch.long)
        loss_uv = F.cross_entropy(logits, labels)
        loss_vu = F.cross_entropy(logits.T, labels)
        loss_sym = 0.5 * (loss_uv + loss_vu).item()
        mi_bits = log2_k - (loss_sym / ln2)
        mi_list.append(mi_bits)

        # Bidirectional Top-1 and Top-5 retrieval rates
        top5_uv = torch.topk(sim, k=min(5, pool_k), dim=1).indices
        top5_vu = torch.topk(sim.T, k=min(5, pool_k), dim=1).indices
        t1_uv = (top5_uv[:, 0] == labels).to(torch.float32).mean().item()
        t1_vu = (top5_vu[:, 0] == labels).to(torch.float32).mean().item()
        t5_uv = (top5_uv == labels.unsqueeze(1)).any(dim=1).to(torch.float32).mean().item()
        t5_vu = (top5_vu == labels.unsqueeze(1)).any(dim=1).to(torch.float32).mean().item()
        top1_list.append(0.5 * (t1_uv + t1_vu))
        top5_list.append(0.5 * (t5_uv + t5_vu))

    def _mean_se(vals: Sequence[float]) -> Tuple[float, float]:
        t = torch.tensor(vals, dtype=torch.float64)
        mean = float(t.mean().item())
        se = float(t.std(unbiased=True).item() / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        return mean, se

    mi_mean, mi_se = _mean_se(mi_list)
    top1_mean, top1_se = _mean_se(top1_list)
    top5_mean, top5_se = _mean_se(top5_list)

    # Null permutation test (P independent row permutations per pool)
    perm_mi_mean = 0.0
    perm_mi_std = 0.0
    z_null = 0.0
    if num_perm > 0:
        gen = torch.Generator(device=device)
        gen.manual_seed(perm_seed)
        perm_pool_means: List[float] = []
        labels = torch.arange(pool_k, device=device, dtype=torch.long)
        for _ in range(num_perm):
            acc = 0.0
            for m in range(num_pools):
                idx = torch.randperm(pool_k, device=device, generator=gen)
                logits_p = (u_pools[m] @ v_pools[m][idx].T) * scale
                l_uv = F.cross_entropy(logits_p, labels)
                l_vu = F.cross_entropy(logits_p.T, labels)
                acc += log2_k - (0.5 * (l_uv + l_vu).item() / ln2)
            perm_pool_means.append(acc / num_pools)
        pt = torch.tensor(perm_pool_means, dtype=torch.float64)
        perm_mi_mean = float(pt.mean().item())
        perm_mi_std = float(pt.std(unbiased=True).item()) if num_perm > 1 else 0.0
        se_perm = perm_mi_std / math.sqrt(num_perm)
        z_null = (mi_mean - perm_mi_mean) / math.sqrt(mi_se ** 2 + se_perm ** 2 + 1e-12)

    # Top-1 binomial z-score (2 * N_total directional queries, p0 = 1 / pool_k)
    p0 = 1.0 / float(pool_k)
    n_queries = 2.0 * float(n_total)
    z_top1_null = (top1_mean - p0) / math.sqrt(p0 * (1.0 - p0) / n_queries + 1e-18)

    return {
        "mi_bits_mean": mi_mean,
        "mi_bits_se": mi_se,
        "top1_mean": top1_mean,
        "top1_se": top1_se,
        "top5_mean": top5_mean,
        "top5_se": top5_se,
        "perm_mi_mean": perm_mi_mean,
        "perm_mi_std": perm_mi_std,
        "z_null": z_null,
        "z_top1_null": z_top1_null,
    }


def _solve_cca_matrices(
    x: torch.Tensor,
    y: torch.Tensor,
    reg: float,
    num_dims: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit regularized CCA in float64 for numerical stability; return float32 params."""
    x64 = x.to(torch.float64)
    y64 = y.to(torch.float64)
    n = float(x64.shape[0])
    mu_x = x64.mean(dim=0, keepdim=True)
    mu_y = y64.mean(dim=0, keepdim=True)
    xc = x64 - mu_x
    yc = y64 - mu_y

    dx = x64.shape[1]
    dy = y64.shape[1]
    cxx = (xc.T @ xc) / n + reg * torch.eye(dx, dtype=torch.float64, device=x.device)
    cyy = (yc.T @ yc) / n + reg * torch.eye(dy, dtype=torch.float64, device=x.device)
    cxy = (xc.T @ yc) / n

    evals_x, evecs_x = torch.linalg.eigh(cxx)
    evals_y, evecs_y = torch.linalg.eigh(cyy)
    inv_sqrt_x = evecs_x @ torch.diag(evals_x.clamp(min=1e-12).rsqrt()) @ evecs_x.T
    inv_sqrt_y = evecs_y @ torch.diag(evals_y.clamp(min=1e-12).rsqrt()) @ evecs_y.T

    t_mat = inv_sqrt_x @ cxy @ inv_sqrt_y
    u_svd, s_svd, vh_svd = torch.linalg.svd(t_mat, full_matrices=False)
    k = min(num_dims, s_svd.shape[0])

    wx = inv_sqrt_x @ u_svd[:, :k]
    wy = inv_sqrt_y @ vh_svd[:k, :].T
    return (
        mu_x.to(torch.float32),
        mu_y.to(torch.float32),
        wx.to(torch.float32),
        wy.to(torch.float32),
        s_svd[:k].to(torch.float32),
    )


@torch.no_grad()
def evaluate_ridge_cca(
    x_fit: torch.Tensor,
    y_fit: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    reg: float = 1e-3,
    emb_dim: int = 128,
    pool_k: int = 1024,
) -> Dict[str, Any]:
    """Fit analytic Ridge-CCA on fit set and evaluate Gaussian MI + cosine retrieval on holdout."""
    mu_x, mu_y, wx, wy, _ = _solve_cca_matrices(x_fit, y_fit, reg=reg, num_dims=emb_dim)

    zx = (x_eval - mu_x) @ wx
    zy = (y_eval - mu_y) @ wy

    # Holdout canonical correlations
    zx_c = zx - zx.mean(dim=0, keepdim=True)
    zy_c = zy - zy.mean(dim=0, keepdim=True)
    num = (zx_c * zy_c).sum(dim=0)
    den = torch.sqrt((zx_c ** 2).sum(dim=0) * (zy_c ** 2).sum(dim=0) + 1e-18)
    corrs = (num / den).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    r2 = (corrs ** 2).clamp(0.0, 1.0 - 1e-7)
    cca_mi_bits = float((-0.5 * torch.log2(1.0 - r2).sum()).item())

    # Cosine retrieval on L2-normalized CCA projections
    zx_norm = F.normalize(zx, p=2, dim=-1, eps=1e-8)
    zy_norm = F.normalize(zy, p=2, dim=-1, eps=1e-8)
    ret = evaluate_embeddings_pooled(
        zx_norm, zy_norm, scale=1.0 / 0.07, pool_k=pool_k, num_perm=0
    )

    return {
        "ridge_cca_mi_bits_eval": cca_mi_bits,
        "ridge_cca_top1_mean": ret["top1_mean"],
        "ridge_cca_top5_mean": ret["top5_mean"],
        "ridge_cca_top10_corrs": [float(v) for v in corrs[:10].tolist()],
    }


def _fista_lasso(
    x: torch.Tensor,
    y: torch.Tensor,
    l1_alpha: float = 0.02,
    iters: int = 250,
) -> Tuple[torch.Tensor, float]:
    """Pure-PyTorch FP32 FISTA Lasso for standardized features x (N, D) -> standardized target y (N,).

    Returns (weights (D,), explained_variance R^2).
    """
    n, d = x.shape
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-6)
    xs = (x - x_mean) / x_std

    y_mean = y.mean()
    y_std = y.std(unbiased=False).clamp(min=1e-6)
    ys = (y - y_mean) / y_std

    # Precompute Gram matrix X^T X / N and X^T y / N (since D <= 896, very fast on GPU/CPU)
    gram = (xs.T @ xs) / float(n)
    xty = (xs.T @ ys) / float(n)

    # Power iteration for Lipschitz constant L = lambda_max(Gram)
    v = torch.ones(d, dtype=torch.float32, device=x.device) / math.sqrt(d)
    for _ in range(15):
        v = gram @ v
        v_norm = v.norm()
        if v_norm > 0:
            v = v / v_norm
    lip = float((v @ (gram @ v)).item()) + 1e-4
    step = 1.0 / lip
    thresh = l1_alpha * step

    w = torch.zeros(d, dtype=torch.float32, device=x.device)
    z = w.clone()
    t_k = 1.0
    for _ in range(iters):
        grad = (gram @ z) - xty
        w_next_raw = z - step * grad
        w_next = torch.sign(w_next_raw) * torch.clamp(w_next_raw.abs() - thresh, min=0.0)
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_k * t_k))
        z = w_next + ((t_k - 1.0) / t_next) * (w_next - w)
        w = w_next
        t_k = t_next

    with torch.no_grad():
        pred = xs @ w
        ss_res = float(((ys - pred) ** 2).sum().item())
        ss_tot = float((ys ** 2).sum().item()) + 1e-12
        r2 = max(0.0, 1.0 - ss_res / ss_tot)
    return w.detach(), r2


@torch.no_grad()
def distill_embeddings(
    phi_in: torch.Tensor,
    phi_out: torch.Tensor,
    u_emb: torch.Tensor,
    v_emb: torch.Tensor,
    top_modes: int = 3,
    top_features: int = 16,
) -> Dict[str, Any]:
    """Perform embedding CCA spectrum, sparse Lasso/Pearson attribution, and 2nd-order Walsh probe."""
    mu_u, mu_v, wu, wv, s_cca = _solve_cca_matrices(
        u_emb, v_emb, reg=1e-5, num_dims=u_emb.shape[1]
    )
    zu = (u_emb - mu_u) @ wu
    zv = (v_emb - mu_v) @ wv

    # Compute exact canonical correlations on the embeddings
    zu_c = zu - zu.mean(dim=0, keepdim=True)
    zv_c = zv - zv.mean(dim=0, keepdim=True)
    corrs = ((zu_c * zv_c).sum(dim=0)) / torch.sqrt(
        (zu_c ** 2).sum(dim=0) * (zv_c ** 2).sum(dim=0) + 1e-18
    )
    corrs = corrs.clamp(0.0, 1.0)
    # Sort descending just in case
    order = torch.argsort(corrs, descending=True)
    corrs = corrs[order]
    zu = zu[:, order]
    zv = zv[:, order]

    r2_spec = corrs ** 2
    r2_sum = float(r2_spec.sum().item())
    if r2_sum > 1e-12:
        probs = (r2_spec / r2_sum).clamp(min=1e-18)
        eff_rank = float(torch.exp(-(probs * torch.log(probs)).sum()).item())
    else:
        eff_rank = 0.0
    num_sig = int((corrs > 0.10).sum().item())

    # Pre-standardize phi_in & phi_out for fast Pearson r computation
    in_c = phi_in - phi_in.mean(dim=0, keepdim=True)
    in_norm = torch.sqrt((in_c ** 2).sum(dim=0) + 1e-12)
    out_c = phi_out - phi_out.mean(dim=0, keepdim=True)
    out_norm = torch.sqrt((out_c ** 2).sum(dim=0) + 1e-12)

    modes_out: List[Dict[str, Any]] = []
    n_modes = min(top_modes, corrs.shape[0])

    # Split half for unbiased R^2 evaluation vs Lasso fit
    half = phi_in.shape[0] // 2

    for k in range(n_modes):
        zk_x = zu[:, k]
        zk_s = zv[:, k]

        # Pearson correlations on full eval
        zx_c = zk_x - zk_x.mean()
        zs_c = zk_s - zk_s.mean()
        r_in = ((in_c.T @ zx_c) / (in_norm * torch.sqrt((zx_c ** 2).sum() + 1e-12))).detach()
        r_out = ((out_c.T @ zs_c) / (out_norm * torch.sqrt((zs_c ** 2).sum() + 1e-12))).detach()

        # FISTA Lasso on first half, evaluate R^2 on second half
        with torch.enable_grad():
            w_in, r2_in_fit = _fista_lasso(phi_in[:half], zk_x[:half], l1_alpha=0.02, iters=250)
            w_out, r2_out_fit = _fista_lasso(phi_out[:half], zk_s[:half], l1_alpha=0.02, iters=250)

        def _eval_r2(x_tr: torch.Tensor, y_tr: torch.Tensor, x_te: torch.Tensor, y_te: torch.Tensor, w: torch.Tensor) -> float:
            xm = x_tr.mean(dim=0, keepdim=True)
            xs = x_tr.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-6)
            ym = y_tr.mean()
            ys = y_tr.std(unbiased=False).clamp(min=1e-6)
            pred = ((x_te - xm) / xs) @ w
            yt = (y_te - ym) / ys
            return max(0.0, 1.0 - float(((yt - pred) ** 2).sum().item()) / (float((yt ** 2).sum().item()) + 1e-12))

        r2_in_val = _eval_r2(phi_in[:half], zk_x[:half], phi_in[half:], zk_x[half:], w_in)
        r2_out_val = _eval_r2(phi_out[:half], zk_s[:half], phi_out[half:], zk_s[half:], w_out)

        # Rank top features combining |lasso_w| and |pearson_r|
        score_in = w_in.abs() + 0.25 * r_in.abs()
        top_in_idx = torch.topk(score_in, k=min(top_features, phi_in.shape[1])).indices.tolist()
        top_in_feats = [
            {
                "name": IN_FEATURE_NAMES[idx],
                "lasso_w": round(float(w_in[idx].item()), 5),
                "pearson_r": round(float(r_in[idx].item()), 5),
            }
            for idx in top_in_idx
        ]

        score_out = w_out.abs() + 0.25 * r_out.abs()
        top_out_idx = torch.topk(score_out, k=min(top_features, phi_out.shape[1])).indices.tolist()
        top_out_feats = [
            {
                "name": OUT_FEATURE_NAMES[idx],
                "lasso_w": round(float(w_out[idx].item()), 5),
                "pearson_r": round(float(r_out[idx].item()), 5),
            }
            for idx in top_out_idx
        ]

        # 2nd-order Walsh-Hadamard probe on top-16 bit spins (indices < 512 for in, < 256 for out)
        def _walsh_probe(phi_bits: torch.Tensor, bit_names: Sequence[str], r_bits: torch.Tensor, target: torch.Tensor) -> List[Dict[str, Any]]:
            top16 = torch.topk(r_bits.abs(), k=min(16, r_bits.shape[0])).indices.tolist()
            tc = target - target.mean()
            t_norm = torch.sqrt((tc ** 2).sum() + 1e-12)
            pairs: List[Tuple[float, str, str, float]] = []
            for ii in range(len(top16)):
                for jj in range(ii + 1, len(top16)):
                    a, b = top16[ii], top16[jj]
                    prod = phi_bits[:, a] * phi_bits[:, b]
                    pc = prod - prod.mean()
                    corr = float(((pc * tc).sum() / (torch.sqrt((pc ** 2).sum() + 1e-12) * t_norm)).item())
                    pairs.append((abs(corr), bit_names[a], bit_names[b], corr))
            pairs.sort(key=lambda x: x[0], reverse=True)
            return [
                {"bit_a": ba, "bit_b": bb, "walsh2_r": round(c, 5)}
                for _, ba, bb, c in pairs[:10]
            ]

        walsh_in = _walsh_probe(phi_in[:, :512], IN_FEATURE_NAMES[:512], r_in[:512], zk_x)
        walsh_out = _walsh_probe(phi_out[:, :256], OUT_FEATURE_NAMES[:256], r_out[:256], zk_s)

        modes_out.append(
            {
                "mode_idx": k,
                "canonical_corr": round(float(corrs[k].item()), 6),
                "r2_in": round(r2_in_val, 5),
                "r2_out": round(r2_out_val, 5),
                "r2_in_fit": round(r2_in_fit, 5),
                "r2_out_fit": round(r2_out_fit, 5),
                "top_in_features": top_in_feats,
                "top_out_features": top_out_feats,
                "top_walsh_pairs": {"in": walsh_in, "out": walsh_out},
            }
        )

    return {
        "cca_top10_corrs": [round(float(v), 6) for v in corrs[:10].tolist()],
        "effective_rank": round(eff_rank, 4),
        "num_sig_modes": num_sig,
        "modes": modes_out,
    }
