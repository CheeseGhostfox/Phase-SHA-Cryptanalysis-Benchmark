"""Probe I, II, III execution, statistical comparison, and U_theta distillation for P7."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from cipher_ops import (
    CipherSpec,
    feistel32_strong_bijection,
    make_ctrl_rotr_k_spec,
    make_main_spec,
    run_rounds_ae,
    sample_keys,
    sample_perms_256,
)
from ortho_basis import (
    DualGroup32Basis,
    build_256_init_bank,
    build_haar_ortho,
    build_hybrid_wht_dft_256,
    build_probe2_init_bank,
    build_real_dft,
    build_sylvester_walsh,
    check_orthogonal_conservation,
    sparsity_metrics_tensor,
    train_cayley_biortho_bank,
    train_cayley_ortho_bank,
)
from src.sha256_ref import MASK32, words_to_spins


@dataclass
class P7Config:
    rounds: Tuple[int, ...] = (2, 4, 6, 8, 10, 12)
    probe1_k_train: int = 16
    probe1_k_holdout: int = 32
    probe2_k_train: int = 4
    probe2_n_train: int = 65536
    probe2_k_holdout: int = 4
    probe2_n_pools: int = 16
    probe2_n_pool_samples: int = 8192
    probe3_k_train: int = 8
    probe3_k_holdout: int = 16
    opt_steps: int = 400
    lr_max: float = 3e-2
    lr_min: float = 1e-3
    chunk_size: int = 65536
    seed: int = 20261125


def _mean_se(vals: Sequence[float]) -> Tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    t = torch.tensor(vals, dtype=torch.float64)
    m = float(t.mean().item())
    if n == 1:
        return m, 0.0
    se = float(t.std(unbiased=True).item() / math.sqrt(float(n)))
    return m, se


def _summarize_pool_metrics(
    h_vals: List[float], r_vals: List[float], k_vals: List[float], top_key: str
) -> Dict[str, float]:
    hm, hse = _mean_se(h_vals)
    rm, rse = _mean_se(r_vals)
    km, kse = _mean_se(k_vals)
    return {
        "entropy_mean": hm,
        "entropy_se": hse,
        "l1_l2_mean": rm,
        "l1_l2_se": rse,
        f"{top_key}_mean": km,
        f"{top_key}_se": kse,
    }


def _compute_vs_classical_deltas(
    learned_pools: Tuple[List[float], List[float], List[float]],
    baseline_pools: Dict[str, Tuple[List[float], List[float], List[float]]],
    cl_names: Sequence[str],
) -> Dict[str, Any]:
    lh, lr, lk = learned_pools
    n = len(lh)
    dh, dr, dk = [], [], []
    for i in range(n):
        best_h = min(baseline_pools[b][0][i] for b in cl_names)
        best_r = min(baseline_pools[b][1][i] for b in cl_names)
        best_k = max(baseline_pools[b][2][i] for b in cl_names)
        dh.append(best_h - lh[i])
        dr.append(best_r - lr[i])
        dk.append(lk[i] - best_k)

    dhm, dhse = _mean_se(dh)
    drm, drse = _mean_se(dr)
    dkm, dkse = _mean_se(dk)
    return {
        "delta_entropy_mean": dhm,
        "delta_entropy_se": dhse,
        "z_entropy_vs_cl": dhm / (dhse + 1e-12),
        "delta_l1_l2_mean": drm,
        "delta_l1_l2_se": drse,
        "z_l1_l2_vs_cl": drm / (drse + 1e-12),
        "delta_topK_mean": dkm,
        "delta_topK_se": dkse,
        "z_topK_vs_cl": dkm / (dkse + 1e-12),
    }


def _z_diff(m1: float, se1: float, m2: float, se2: float) -> float:
    return (m1 - m2) / math.sqrt(se1 * se1 + se2 * se2 + 1e-12)


def _inject_slice_w0(
    w0_mask: torch.Tensor, x_vals: torch.Tensor, slice_name: str
) -> torch.Tensor:
    """Broadcast w0_mask (K,) and x_vals (K, 256) into (K, 256) W_0 words."""
    base = w0_mask.unsqueeze(1).expand_as(x_vals)
    if slice_name == "slice_lsb":
        return ((base & (~0xFF & MASK32)) | (x_vals & 0xFF)) & MASK32
    if slice_name == "slice_msb":
        return ((base & 0x00FFFFFF) | ((x_vals & 0xFF) << 24)) & MASK32
    if slice_name == "slice_strided":
        positions = (0, 4, 8, 12, 16, 20, 24, 28)
        clear_mask = MASK32
        for pos in positions:
            clear_mask &= ~(1 << pos)
        w0 = base & clear_mask
        for bit_i, pos in enumerate(positions):
            w0 = w0 | (((x_vals >> bit_i) & 1) << pos)
        return w0 & MASK32
    raise ValueError(f"Unknown slice_name: {slice_name}")


def _generate_probe1_columns(
    rounds: int,
    slice_name: str,
    cipher_name: str,
    spec: CipherSpec,
    w0_mask: torch.Tensor,
    w_rest: torch.Tensor,
    perms: Optional[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Generate per-key normalized (256, M_valid) matrices and word-group indices."""
    k_num = w0_mask.shape[0]
    device = w0_mask.device
    x_grid = torch.arange(256, dtype=torch.int64, device=device).unsqueeze(0).expand(k_num, 256)
    if cipher_name == "ctrl_rand_perm":
        assert perms is not None
        x_in = torch.gather(perms, 1, x_grid)
    else:
        x_in = x_grid

    w0 = _inject_slice_w0(w0_mask, x_in, slice_name)  # (K, 256)
    w16 = torch.empty((k_num, 256, 16), dtype=torch.int64, device=device)
    w16[:, :, 0] = w0
    w16[:, :, 1:] = w_rest.unsqueeze(1).expand(k_num, 256, 15)

    w16_flat = w16.reshape(k_num * 256, 16)
    a_r, e_r = run_rounds_ae(w16_flat, rounds, spec)
    out_words = torch.stack(
        [a_r, e_r, (a_r ^ e_r) & MASK32, (a_r - e_r) & MASK32], dim=1
    )  # (K*256, 4)
    spins = words_to_spins(out_words, dtype=torch.float32).reshape(k_num, 256, 128)

    # Center along x (dim=1) and normalize valid non-constant columns
    centered = spins - spins.mean(dim=1, keepdim=True)
    norms = torch.sqrt(centered.square().sum(dim=1))  # (K, 128)

    per_key_mats: List[torch.Tensor] = []
    per_key_group_ids: List[torch.Tensor] = []
    group_ids_full = torch.arange(128, device=device) // 32  # 0:A, 1:E, 2:A^E, 3:A-E

    for k_idx in range(k_num):
        valid = norms[k_idx] > 1e-6
        mat_k = centered[k_idx, :, valid] / norms[k_idx, valid].unsqueeze(0)
        per_key_mats.append(mat_k)
        per_key_group_ids.append(group_ids_full[valid])

    return per_key_mats, per_key_group_ids


def distill_256_matrix(
    u_star: torch.Tensor,
    q_wht: torch.Tensor,
    q_dft: torch.Tensor,
    q_hyb: torch.Tensor,
) -> Dict[str, Any]:
    """Distill structural overlap of U_* in O(256) against WHT, DFT, and Hybrid bases."""
    out: Dict[str, Any] = {}
    for bname, q_base in (("walsh", q_wht), ("dft", q_dft), ("hybrid", q_hyb)):
        m = q_base.transpose(-2, -1) @ u_star  # (256, 256), unit columns
        m_sq = m.square()
        diag_frac = float(torch.trace(m_sq).item() / 256.0)
        # Column-wise effective modes (inverse participation ratio)
        ipr = 1.0 / (m_sq.square().sum(dim=0) + 1e-12)
        avg_eff_modes = float(ipr.mean().item())
        # Matrix-level normalized spectral entropy
        p_mat = m_sq / (m_sq.sum() + 1e-12)
        mat_ent = float(-(p_mat * torch.log(p_mat + 1e-12)).sum().item() / math.log(65536.0))
        out[bname] = {
            "matrix_spec_entropy": mat_ent,
            "diag_energy_frac": diag_frac,
            "avg_eff_modes": avg_eff_modes,
        }
    return out


def run_probe1(
    cfg: P7Config,
    device: torch.device,
    rounds_list: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run Probe I: multi-key 8-bit bit-slice spectral telescope."""
    rounds_seq = list(rounds_list) if rounds_list is not None else list(cfg.rounds)
    spec_main = make_main_spec()
    spec_rotr_k = make_ctrl_rotr_k_spec(cfg.seed + 101)

    init_bank = build_256_init_bank(device=device)
    q_std = init_bank["init_std"]
    q_wht = init_bank["init_walsh"]
    q_dft = init_bank["init_dft"]
    q_hyb = init_bank["init_hybrid"]
    q_haar = build_haar_ortho(256, seed=cfg.seed + 303, device=device)

    basis_eval = {
        "std": q_std,
        "walsh": q_wht,
        "dft": q_dft,
        "haar": q_haar,
    }
    cl_names = ("std", "walsh", "dft", "haar")

    # Fixed strictly disjoint train & holdout key pools
    w0_tr, wrest_tr = sample_keys(cfg.probe1_k_train, cfg.seed + 1000, device)
    w0_ho, wrest_ho = sample_keys(cfg.probe1_k_holdout, cfg.seed + 2000, device)
    perms_tr = sample_perms_256(cfg.probe1_k_train, cfg.seed + 3000, device)
    perms_ho = sample_perms_256(cfg.probe1_k_holdout, cfg.seed + 4000, device)

    results: Dict[str, Any] = {}
    distill: Dict[str, Any] = {}

    slices = ("slice_lsb", "slice_msb", "slice_strided")
    ciphers = ("main", "ctrl_rotr_k", "ctrl_rand_perm")
    word_names = ("A_R", "E_R", "A_xor_E", "A_sub_E")

    for r in rounds_seq:
        r_key = str(r)
        results[r_key] = {}
        for s_name in slices:
            results[r_key][s_name] = {}
            pool_deltas_by_cipher: Dict[str, Dict[str, Any]] = {}

            for c_idx, c_name in enumerate(ciphers):
                spec = spec_rotr_k if c_name == "ctrl_rotr_k" else spec_main
                p_tr = perms_tr if c_name == "ctrl_rand_perm" else None
                p_ho = perms_ho if c_name == "ctrl_rand_perm" else None

                tr_mats, _ = _generate_probe1_columns(
                    r, s_name, c_name, spec, w0_tr, wrest_tr, p_tr
                )
                ho_mats, ho_gids = _generate_probe1_columns(
                    r, s_name, c_name, spec, w0_ho, wrest_ho, p_ho
                )

                g_train = torch.cat(tr_mats, dim=1)
                g_ho_all = torch.cat(ho_mats, dim=1)

                best_init, u_star, train_losses = train_cayley_ortho_bank(
                    q_init_dict=init_bank,
                    train_mat=g_train,
                    mode="columns",
                    top_k=16,
                    steps=cfg.opt_steps,
                    lr_max=cfg.lr_max,
                    lr_min=cfg.lr_min,
                    seed=cfg.seed + r * 100 + c_idx,
                )

                ortho_err, det_err, norm_err_tr = check_orthogonal_conservation(u_star, g_train)
                _, _, norm_err_ho = check_orthogonal_conservation(u_star, g_ho_all)
                norm_err = max(norm_err_tr, norm_err_ho)

                # Evaluate baselines & learned on each holdout key pool
                b_pools: Dict[str, Tuple[List[float], List[float], List[float]]] = {
                    bn: ([], [], []) for bn in cl_names
                }
                l_pools: Tuple[List[float], List[float], List[float]] = ([], [], [])
                cw_accum_learned: Dict[int, List[torch.Tensor]] = {0: [], 1: [], 2: [], 3: []}
                cw_accum_haar: Dict[int, List[torch.Tensor]] = {0: [], 1: [], 2: [], 3: []}

                with torch.no_grad():
                    for k_i, mat_k in enumerate(ho_mats):
                        for bn, q_b in basis_eval.items():
                            zb = q_b.transpose(-2, -1) @ mat_k
                            hb, rb, kb = sparsity_metrics_tensor(zb, top_k=16)
                            b_pools[bn][0].append(float(hb.mean().item()))
                            b_pools[bn][1].append(float(rb.mean().item()))
                            b_pools[bn][2].append(float(kb.mean().item()))
                            if bn == "haar":
                                gids = ho_gids[k_i]
                                for gid in range(4):
                                    mask = gids == gid
                                    if mask.any():
                                        cw_accum_haar[gid].append(
                                            torch.stack([hb[mask], rb[mask], kb[mask]], dim=0)
                                        )

                        zl = u_star.transpose(-2, -1) @ mat_k
                        hl, rl, kl = sparsity_metrics_tensor(zl, top_k=16)
                        l_pools[0].append(float(hl.mean().item()))
                        l_pools[1].append(float(rl.mean().item()))
                        l_pools[2].append(float(kl.mean().item()))
                        gids = ho_gids[k_i]
                        for gid in range(4):
                            mask = gids == gid
                            if mask.any():
                                cw_accum_learned[gid].append(
                                    torch.stack([hl[mask], rl[mask], kl[mask]], dim=0)
                                )

                baselines_json = {
                    bn: _summarize_pool_metrics(*b_pools[bn], top_key="top16")
                    for bn in cl_names
                }
                learned_json = _summarize_pool_metrics(*l_pools, top_key="top16")
                learned_json.update(
                    {
                        "best_init": best_init,
                        "ortho_err": ortho_err,
                        "det_err": det_err,
                        "norm_err": norm_err,
                        "train_loss_per_init": train_losses,
                        "valid_cols_mean": float(sum(m.shape[1] for m in ho_mats) / len(ho_mats)),
                    }
                )

                cw_json: Dict[str, Any] = {}
                for gid, wname in enumerate(word_names):
                    cat_l = torch.cat(cw_accum_learned[gid], dim=1)
                    cat_h = torch.cat(cw_accum_haar[gid], dim=1)
                    cw_json[wname] = {
                        "entropy_mean": float(cat_l[0].mean().item()),
                        "l1_l2_mean": float(cat_l[1].mean().item()),
                        "top16_mean": float(cat_l[2].mean().item()),
                        "haar_entropy_mean": float(cat_h[0].mean().item()),
                        "haar_l1_l2_mean": float(cat_h[1].mean().item()),
                    }

                delta_stats = _compute_vs_classical_deltas(l_pools, b_pools, cl_names)
                pool_deltas_by_cipher[c_name] = delta_stats

                entry: Dict[str, Any] = {
                    "baselines": baselines_json,
                    "learned": learned_json,
                    "cross_word_breakdown": cw_json,
                }
                results[r_key][s_name][c_name] = entry

                if c_name == "main":
                    main_u_star = u_star
                    main_best_init = best_init
                    main_delta = delta_stats

            # Fill comparison under main
            d_main = pool_deltas_by_cipher["main"]
            d_rotr = pool_deltas_by_cipher["ctrl_rotr_k"]
            d_perm = pool_deltas_by_cipher["ctrl_rand_perm"]
            comp = {
                "delta_entropy_vs_best_cl": d_main["delta_entropy_mean"],
                "z_entropy_vs_cl": d_main["z_entropy_vs_cl"],
                "delta_l1_l2_vs_best_cl": d_main["delta_l1_l2_mean"],
                "z_l1_l2_vs_cl": d_main["z_l1_l2_vs_cl"],
                "delta_top16_vs_best_cl": d_main["delta_topK_mean"],
                "z_top16_vs_cl": d_main["z_topK_vs_cl"],
                "z_entropy_vs_rotr_k": _z_diff(
                    d_main["delta_entropy_mean"],
                    d_main["delta_entropy_se"],
                    d_rotr["delta_entropy_mean"],
                    d_rotr["delta_entropy_se"],
                ),
                "z_entropy_vs_rand_perm": _z_diff(
                    d_main["delta_entropy_mean"],
                    d_main["delta_entropy_se"],
                    d_perm["delta_entropy_mean"],
                    d_perm["delta_entropy_se"],
                ),
                "z_l1_l2_vs_rotr_k": _z_diff(
                    d_main["delta_l1_l2_mean"],
                    d_main["delta_l1_l2_se"],
                    d_rotr["delta_l1_l2_mean"],
                    d_rotr["delta_l1_l2_se"],
                ),
                "z_l1_l2_vs_rand_perm": _z_diff(
                    d_main["delta_l1_l2_mean"],
                    d_main["delta_l1_l2_se"],
                    d_perm["delta_l1_l2_mean"],
                    d_perm["delta_l1_l2_se"],
                ),
            }
            results[r_key][s_name]["main"]["comparison"] = comp

            if r == 2 or comp["z_entropy_vs_cl"] >= 4.0 or comp["z_l1_l2_vs_cl"] >= 4.0:
                distill.setdefault(r_key, {})[s_name] = {
                    "best_init": main_best_init,
                    "z_entropy_vs_cl": comp["z_entropy_vs_cl"],
                    "z_l1_l2_vs_cl": comp["z_l1_l2_vs_cl"],
                    "basis_projection": distill_256_matrix(main_u_star, q_wht, q_dft, q_hyb),
                }
        print(f"  [Probe I] R={r} completed", flush=True)

    return results, distill


def _compute_coupling_matrix(
    basis: DualGroup32Basis,
    rounds: int,
    spec: CipherSpec,
    apply_feistel: bool,
    w0_samples: torch.Tensor,
    w_rest_row: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Compute centered cross-covariance matrix C_R of shape (93, 186) over samples."""
    n_total = w0_samples.shape[0]
    device = w0_samples.device

    sum_x = torch.zeros((93,), dtype=torch.float64, device=device)
    sum_y = torch.zeros((186,), dtype=torch.float64, device=device)
    sum_xy = torch.zeros((93, 186), dtype=torch.float64, device=device)

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        bs = end - start
        w0 = w0_samples[start:end]
        w0_circuit = feistel32_strong_bijection(w0) if apply_feistel else w0

        w16 = torch.empty((bs, 16), dtype=torch.int64, device=device)
        w16[:, 0] = w0_circuit
        w16[:, 1:] = w_rest_row.unsqueeze(0).expand(bs, 15)

        a_r, e_r = run_rounds_ae(w16, rounds, spec)

        phi_x = basis(w0)  # (bs, 93) float32
        phi_y = torch.cat([basis(a_r), basis(e_r)], dim=1)  # (bs, 186) float32

        px64 = phi_x.to(torch.float64)
        py64 = phi_y.to(torch.float64)
        sum_x += px64.sum(dim=0)
        sum_y += py64.sum(dim=0)
        sum_xy += px64.t() @ py64

    inv_n = 1.0 / float(n_total)
    mean_x = sum_x * inv_n
    mean_y = sum_y * inv_n
    cov = sum_xy * inv_n - torch.outer(mean_x, mean_y)
    return cov.to(torch.float32)


def _quadrant_and_sv_profile(c_mat: torch.Tensor) -> Tuple[Dict[str, float], float]:
    """Compute F2 / 2-adic quadrant energy fractions and top-5 SV energy ratio on (93, 186)."""
    in_f2 = slice(0, 32)
    in_2a = slice(32, 93)
    out_f2_idx = torch.cat([torch.arange(0, 32), torch.arange(93, 125)]).to(c_mat.device)
    out_2a_idx = torch.cat([torch.arange(32, 93), torch.arange(125, 186)]).to(c_mat.device)

    sq = c_mat.square()
    total = float(sq.sum().item()) + 1e-12
    e_f2_f2 = float(sq[in_f2, out_f2_idx].sum().item()) / total
    e_f2_2a = float(sq[in_f2, out_2a_idx].sum().item()) / total
    e_2a_f2 = float(sq[in_2a, out_f2_idx].sum().item()) / total
    e_2a_2a = float(sq[in_2a, out_2a_idx].sum().item()) / total

    sv = torch.linalg.svdvals(c_mat)
    sv_sq = sv.square()
    top5 = float(sv_sq[:5].sum().item() / (sv_sq.sum().item() + 1e-12))
    return (
        {
            "f2_f2": e_f2_f2,
            "f2_2adic": e_f2_2a,
            "2adic_f2": e_2a_f2,
            "2adic_2adic": e_2a_2a,
        },
        top5,
    )


def _eval_coupling_matrix(c_mat: torch.Tensor, top_k: int = 32) -> Tuple[float, float, float]:
    z = c_mat.reshape(-1, 1)
    h, r, k = sparsity_metrics_tensor(z, top_k=top_k)
    return float(h[0].item()), float(r[0].item()), float(k[0].item())


def run_probe2(
    cfg: P7Config,
    device: torch.device,
    rounds_list: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run Probe II: full 32-bit dual-group input-output coupling telescope."""
    rounds_seq = list(rounds_list) if rounds_list is not None else list(cfg.rounds)
    basis = DualGroup32Basis().to(device)
    spec_main = make_main_spec()
    spec_rotr_k = make_ctrl_rotr_k_spec(cfg.seed + 101)

    ql_dict, qr_dict = build_probe2_init_bank(device=device)
    ql_haar = build_haar_ortho(93, seed=cfg.seed + 501, device=device)
    qr_haar = build_haar_ortho(186, seed=cfg.seed + 502, device=device)

    _, wrest_tr = sample_keys(cfg.probe2_k_train, cfg.seed + 5000, device)
    _, wrest_ho = sample_keys(cfg.probe2_k_holdout, cfg.seed + 6000, device)

    # Disjoint input domains: Train in [0, 2^30), Holdout in [2^30, 2^32)
    gen_tr = torch.Generator(device="cpu")
    gen_tr.manual_seed(cfg.seed + 7000)
    w0_train_pools = [
        torch.randint(0, 1 << 30, (cfg.probe2_n_train,), dtype=torch.int64, generator=gen_tr).to(device)
        for _ in range(cfg.probe2_k_train)
    ]

    gen_ho = torch.Generator(device="cpu")
    gen_ho.manual_seed(cfg.seed + 8000)
    w0_holdout_pools = [
        [
            torch.randint(
                1 << 30, 1 << 32, (cfg.probe2_n_pool_samples,), dtype=torch.int64, generator=gen_ho
            ).to(device)
            for _ in range(cfg.probe2_k_holdout)
        ]
        for _ in range(cfg.probe2_n_pools)
    ]

    out_f2_idx = torch.cat([torch.arange(0, 32), torch.arange(93, 125)]).to(device)
    out_2a_idx = torch.cat([torch.arange(32, 93), torch.arange(125, 186)]).to(device)

    results: Dict[str, Any] = {}
    distill: Dict[str, Any] = {}
    ciphers = ("main", "ctrl_rotr_k", "ctrl_rand_perm")
    cl_names = ("dual_raw", "wht_block", "haar")

    for r in rounds_seq:
        r_key = str(r)
        results[r_key] = {}
        pool_deltas_by_cipher: Dict[str, Dict[str, Any]] = {}

        for c_idx, c_name in enumerate(ciphers):
            spec = spec_rotr_k if c_name == "ctrl_rotr_k" else spec_main
            apply_feistel = c_name == "ctrl_rand_perm"

            train_mats = torch.stack(
                [
                    _compute_coupling_matrix(
                        basis, r, spec, apply_feistel, w0_train_pools[k], wrest_tr[k], cfg.chunk_size
                    )
                    for k in range(cfg.probe2_k_train)
                ],
                dim=0,
            )  # (K_tr, 93, 186)

            best_init, ul_star, ur_star, train_losses = train_cayley_biortho_bank(
                ql_dict=ql_dict,
                qr_dict=qr_dict,
                train_mats=train_mats,
                top_k=32,
                steps=cfg.opt_steps,
                lr_max=cfg.lr_max,
                lr_min=cfg.lr_min,
                seed=cfg.seed + 500 + r * 20 + c_idx,
            )

            ortho_l, det_l, norm_l = check_orthogonal_conservation(ul_star, train_mats, ur_star)
            ortho_r, det_r, _ = check_orthogonal_conservation(ur_star)

            b_pools: Dict[str, Tuple[List[float], List[float], List[float]]] = {
                bn: ([], [], []) for bn in ("dual_raw", "wht_block", "haar", "std_bit", "2adic_raw")
            }
            l_pools: Tuple[List[float], List[float], List[float]] = ([], [], [])
            frob_list: List[float] = []
            top5_list: List[float] = []
            quad_accum = {"f2_f2": 0.0, "f2_2adic": 0.0, "2adic_f2": 0.0, "2adic_2adic": 0.0}

            ql_wht = ql_dict["init_wht_block"]
            qr_wht = qr_dict["init_wht_block"]

            with torch.no_grad():
                for p_idx in range(cfg.probe2_n_pools):
                    pool_metrics: Dict[str, List[Tuple[float, float, float]]] = {
                        bn: [] for bn in b_pools
                    }
                    pool_l: List[Tuple[float, float, float]] = []

                    for k_idx in range(cfg.probe2_k_holdout):
                        c_ho = _compute_coupling_matrix(
                            basis,
                            r,
                            spec,
                            apply_feistel,
                            w0_holdout_pools[p_idx][k_idx],
                            wrest_ho[k_idx],
                            cfg.chunk_size,
                        )
                        frob_list.append(float(c_ho.square().sum().item()))
                        q_prof, top5_sv = _quadrant_and_sv_profile(c_ho)
                        top5_list.append(top5_sv)
                        for qk, qv in q_prof.items():
                            quad_accum[qk] += qv

                        pool_metrics["dual_raw"].append(_eval_coupling_matrix(c_ho, 32))
                        pool_metrics["wht_block"].append(
                            _eval_coupling_matrix(ql_wht.t() @ c_ho @ qr_wht, 32)
                        )
                        pool_metrics["haar"].append(
                            _eval_coupling_matrix(ql_haar.t() @ c_ho @ qr_haar, 32)
                        )
                        pool_metrics["std_bit"].append(
                            _eval_coupling_matrix(c_ho[:32, out_f2_idx], 32)
                        )
                        pool_metrics["2adic_raw"].append(
                            _eval_coupling_matrix(c_ho[32:, out_2a_idx], 32)
                        )
                        pool_l.append(
                            _eval_coupling_matrix(ul_star.t() @ c_ho @ ur_star, 32)
                        )

                    for bn in b_pools:
                        vals = pool_metrics[bn]
                        b_pools[bn][0].append(sum(v[0] for v in vals) / len(vals))
                        b_pools[bn][1].append(sum(v[1] for v in vals) / len(vals))
                        b_pools[bn][2].append(sum(v[2] for v in vals) / len(vals))

                    l_pools[0].append(sum(v[0] for v in pool_l) / len(pool_l))
                    l_pools[1].append(sum(v[1] for v in pool_l) / len(pool_l))
                    l_pools[2].append(sum(v[2] for v in pool_l) / len(pool_l))

            n_tot_mats = float(cfg.probe2_n_pools * cfg.probe2_k_holdout)
            quad_mean = {qk: qv / n_tot_mats for qk, qv in quad_accum.items()}

            baselines_json = {
                bn: _summarize_pool_metrics(*b_pools[bn], top_key="top32")
                for bn in ("std_bit", "dual_raw", "2adic_raw", "haar", "wht_block")
            }
            learned_json = _summarize_pool_metrics(*l_pools, top_key="top32")
            learned_json.update(
                {
                    "best_init": best_init,
                    "ortho_err_L": ortho_l,
                    "det_err_L": det_l,
                    "ortho_err_R": ortho_r,
                    "det_err_R": det_r,
                    "norm_err": norm_l,
                    "train_loss_per_init": train_losses,
                }
            )

            delta_stats = _compute_vs_classical_deltas(l_pools, b_pools, cl_names)
            pool_deltas_by_cipher[c_name] = delta_stats

            results[r_key][c_name] = {
                "frobenius_norm_sq": float(sum(frob_list) / len(frob_list)),
                "mc_noise_floor": float((93 * 186) / float(cfg.probe2_n_pool_samples)),
                "quadrant_energy_frac": quad_mean,
                "top5_sv_ratio": float(sum(top5_list) / len(top5_list)),
                "baselines": baselines_json,
                "learned": learned_json,
            }

            if c_name == "main":
                main_ul = ul_star
                main_ur = ur_star
                main_quad = quad_mean
                main_top5 = float(sum(top5_list) / len(top5_list))
                main_train0 = train_mats[0]

        d_main = pool_deltas_by_cipher["main"]
        d_rotr = pool_deltas_by_cipher["ctrl_rotr_k"]
        d_perm = pool_deltas_by_cipher["ctrl_rand_perm"]
        comp = {
            "delta_entropy_vs_best_cl": d_main["delta_entropy_mean"],
            "z_entropy_vs_cl": d_main["z_entropy_vs_cl"],
            "delta_l1_l2_vs_best_cl": d_main["delta_l1_l2_mean"],
            "z_l1_l2_vs_cl": d_main["z_l1_l2_vs_cl"],
            "delta_top32_vs_best_cl": d_main["delta_topK_mean"],
            "z_top32_vs_cl": d_main["z_topK_vs_cl"],
            "z_vs_rotr_k": _z_diff(
                d_main["delta_entropy_mean"],
                d_main["delta_entropy_se"],
                d_rotr["delta_entropy_mean"],
                d_rotr["delta_entropy_se"],
            ),
            "z_vs_rand_perm": _z_diff(
                d_main["delta_entropy_mean"],
                d_main["delta_entropy_se"],
                d_perm["delta_entropy_mean"],
                d_perm["delta_entropy_se"],
            ),
            "z_l1_l2_vs_rotr_k": _z_diff(
                d_main["delta_l1_l2_mean"],
                d_main["delta_l1_l2_se"],
                d_rotr["delta_l1_l2_mean"],
                d_rotr["delta_l1_l2_se"],
            ),
            "z_l1_l2_vs_rand_perm": _z_diff(
                d_main["delta_l1_l2_mean"],
                d_main["delta_l1_l2_se"],
                d_perm["delta_l1_l2_mean"],
                d_perm["delta_l1_l2_se"],
            ),
        }
        results[r_key]["main"]["comparison"] = comp

        if r == 2 or comp["z_entropy_vs_cl"] >= 4.0 or comp["z_l1_l2_vs_cl"] >= 4.0:
            ul_f2_stay = float(main_ul[:32, :32].square().sum().item() / 32.0)
            ur_f2_stay = float(
                main_ur[out_f2_idx][:, out_f2_idx].square().sum().item() / 64.0
            )
            rot_quad, rot_top5 = _quadrant_and_sv_profile(main_ul.t() @ main_train0 @ main_ur)
            distill[r_key] = {
                "raw_quadrant_energy_frac": main_quad,
                "raw_top5_sv_ratio": main_top5,
                "learned_train0_quadrant_energy_frac": rot_quad,
                "learned_train0_top5_sv_ratio": rot_top5,
                "uL_f2_subspace_retention": ul_f2_stay,
                "uR_f2_subspace_retention": ur_f2_stay,
            }
        print(f"  [Probe II] R={r} completed", flush=True)

    return results, distill


def _build_transfer_matrices(
    rounds: int,
    target: str,
    cipher_name: str,
    spec: CipherSpec,
    w0_mask: torch.Tensor,
    w_rest: torch.Tensor,
    perms: Optional[torch.Tensor],
) -> torch.Tensor:
    """Build centered 256x256 transition matrices T_tilde of shape (K, 256, 256)."""
    k_num = w0_mask.shape[0]
    device = w0_mask.device

    x_grid = torch.arange(256, dtype=torch.int64, device=device)  # (256,)
    a_grid = torch.arange(64, dtype=torch.int64, device=device)  # (64,)
    xa = x_grid.unsqueeze(1) * 1 + (a_grid.unsqueeze(0) << 8)  # (256, 64)

    mats = []
    for k_idx in range(k_num):
        if cipher_name == "ctrl_rand_perm":
            assert perms is not None
            x_mapped = perms[k_idx][x_grid].unsqueeze(1).expand(256, 64)
        else:
            x_mapped = x_grid.unsqueeze(1).expand(256, 64)

        w0_vals = (w0_mask[k_idx] & (~0x3FFF & MASK32)) | x_mapped | (a_grid.unsqueeze(0) << 8)
        w0_flat = w0_vals.reshape(-1)  # 16384
        w16 = torch.empty((16384, 16), dtype=torch.int64, device=device)
        w16[:, 0] = w0_flat
        w16[:, 1:] = w_rest[k_idx].unsqueeze(0).expand(16384, 15)

        a_r, e_r = run_rounds_ae(w16, rounds, spec)
        if target == "A_sub_E_low8":
            y = ((a_r - e_r) & 0xFF).reshape(256, 64)
        elif target == "E_low8":
            y = (e_r & 0xFF).reshape(256, 64)
        else:
            raise ValueError(f"Unknown target: {target}")

        x_idx = x_grid.unsqueeze(1).expand(256, 64).reshape(-1)
        y_idx = y.reshape(-1)
        counts = torch.bincount(y_idx * 256 + x_idx, minlength=65536).reshape(256, 256)
        t_mat = counts.to(torch.float32) / 64.0
        t_tilde = t_mat - (1.0 / 256.0)
        mats.append(t_tilde)

    return torch.stack(mats, dim=0)


def run_probe3(
    cfg: P7Config,
    device: torch.device,
    rounds_list: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run Probe III: centered 8-bit transition operator conjugation sparsification."""
    rounds_seq = list(rounds_list) if rounds_list is not None else list(cfg.rounds)
    spec_main = make_main_spec()
    spec_rotr_k = make_ctrl_rotr_k_spec(cfg.seed + 101)

    init_bank = build_256_init_bank(device=device)
    q_std = init_bank["init_std"]
    q_wht = init_bank["init_walsh"]
    q_dft = init_bank["init_dft"]
    q_hyb = init_bank["init_hybrid"]
    q_haar = build_haar_ortho(256, seed=cfg.seed + 606, device=device)

    basis_eval = {"std": q_std, "walsh": q_wht, "dft": q_dft, "haar": q_haar}
    cl_names = ("std", "walsh", "dft", "haar")

    w0_tr, wrest_tr = sample_keys(cfg.probe3_k_train, cfg.seed + 9000, device)
    w0_ho, wrest_ho = sample_keys(cfg.probe3_k_holdout, cfg.seed + 10000, device)
    perms_tr = sample_perms_256(cfg.probe3_k_train, cfg.seed + 11000, device)
    perms_ho = sample_perms_256(cfg.probe3_k_holdout, cfg.seed + 12000, device)

    results: Dict[str, Any] = {}
    distill: Dict[str, Any] = {}

    targets = ("A_sub_E_low8", "E_low8")
    ciphers = ("main", "ctrl_rotr_k", "ctrl_rand_perm")

    for r in rounds_seq:
        r_key = str(r)
        results[r_key] = {}
        for tgt in targets:
            results[r_key][tgt] = {}
            pool_deltas_by_cipher: Dict[str, Dict[str, Any]] = {}

            for c_idx, c_name in enumerate(ciphers):
                spec = spec_rotr_k if c_name == "ctrl_rotr_k" else spec_main
                p_tr = perms_tr if c_name == "ctrl_rand_perm" else None
                p_ho = perms_ho if c_name == "ctrl_rand_perm" else None

                t_train = _build_transfer_matrices(
                    r, tgt, c_name, spec, w0_tr, wrest_tr, p_tr
                )
                t_ho = _build_transfer_matrices(
                    r, tgt, c_name, spec, w0_ho, wrest_ho, p_ho
                )

                best_init, u_star, train_losses = train_cayley_ortho_bank(
                    q_init_dict=init_bank,
                    train_mat=t_train,
                    mode="conjugate",
                    top_k=16,
                    steps=cfg.opt_steps,
                    lr_max=cfg.lr_max,
                    lr_min=cfg.lr_min,
                    seed=cfg.seed + 800 + r * 50 + c_idx,
                )

                ortho_err, det_err, norm_tr = check_orthogonal_conservation(u_star, t_train, u_star)
                _, _, norm_ho = check_orthogonal_conservation(u_star, t_ho, u_star)
                norm_err = max(norm_tr, norm_ho)

                b_pools: Dict[str, Tuple[List[float], List[float], List[float]]] = {
                    bn: ([], [], []) for bn in cl_names
                }
                l_pools: Tuple[List[float], List[float], List[float]] = ([], [], [])

                with torch.no_grad():
                    for bn, q_b in basis_eval.items():
                        zb = torch.einsum("ij,bjl,lm->bim", q_b.t(), t_ho, q_b)
                        zb_flat = zb.reshape(t_ho.shape[0], 65536).t()
                        hb, rb, kb = sparsity_metrics_tensor(zb_flat, top_k=16)
                        b_pools[bn][0].extend(hb.tolist())
                        b_pools[bn][1].extend(rb.tolist())
                        b_pools[bn][2].extend(kb.tolist())

                    zl = torch.einsum("ij,bjl,lm->bim", u_star.t(), t_ho, u_star)
                    zl_flat = zl.reshape(t_ho.shape[0], 65536).t()
                    hl, rl, kl = sparsity_metrics_tensor(zl_flat, top_k=16)
                    l_pools[0].extend(hl.tolist())
                    l_pools[1].extend(rl.tolist())
                    l_pools[2].extend(kl.tolist())

                baselines_json = {
                    bn: _summarize_pool_metrics(*b_pools[bn], top_key="top16")
                    for bn in cl_names
                }
                learned_json = _summarize_pool_metrics(*l_pools, top_key="top16")
                learned_json.update(
                    {
                        "best_init": best_init,
                        "ortho_err": ortho_err,
                        "det_err": det_err,
                        "norm_err": norm_err,
                        "train_loss_per_init": train_losses,
                    }
                )

                delta_stats = _compute_vs_classical_deltas(l_pools, b_pools, cl_names)
                pool_deltas_by_cipher[c_name] = delta_stats

                results[r_key][tgt][c_name] = {
                    "baselines": baselines_json,
                    "learned": learned_json,
                }

                if c_name == "main":
                    main_u_star = u_star
                    main_best_init = best_init

            d_main = pool_deltas_by_cipher["main"]
            d_rotr = pool_deltas_by_cipher["ctrl_rotr_k"]
            d_perm = pool_deltas_by_cipher["ctrl_rand_perm"]
            comp = {
                "delta_entropy_vs_best_cl": d_main["delta_entropy_mean"],
                "z_entropy_vs_cl": d_main["z_entropy_vs_cl"],
                "delta_l1_l2_vs_best_cl": d_main["delta_l1_l2_mean"],
                "z_l1_l2_vs_cl": d_main["z_l1_l2_vs_cl"],
                "delta_top16_vs_best_cl": d_main["delta_topK_mean"],
                "z_top16_vs_cl": d_main["z_topK_vs_cl"],
                "z_entropy_vs_rotr_k": _z_diff(
                    d_main["delta_entropy_mean"],
                    d_main["delta_entropy_se"],
                    d_rotr["delta_entropy_mean"],
                    d_rotr["delta_entropy_se"],
                ),
                "z_entropy_vs_rand_perm": _z_diff(
                    d_main["delta_entropy_mean"],
                    d_main["delta_entropy_se"],
                    d_perm["delta_entropy_mean"],
                    d_perm["delta_entropy_se"],
                ),
                "z_l1_l2_vs_rotr_k": _z_diff(
                    d_main["delta_l1_l2_mean"],
                    d_main["delta_l1_l2_se"],
                    d_rotr["delta_l1_l2_mean"],
                    d_rotr["delta_l1_l2_se"],
                ),
                "z_l1_l2_vs_rand_perm": _z_diff(
                    d_main["delta_l1_l2_mean"],
                    d_main["delta_l1_l2_se"],
                    d_perm["delta_l1_l2_mean"],
                    d_perm["delta_l1_l2_se"],
                ),
            }
            results[r_key][tgt]["main"]["comparison"] = comp

            if r == 2 or comp["z_entropy_vs_cl"] >= 4.0 or comp["z_l1_l2_vs_cl"] >= 4.0:
                distill.setdefault(r_key, {})[tgt] = {
                    "best_init": main_best_init,
                    "z_entropy_vs_cl": comp["z_entropy_vs_cl"],
                    "z_l1_l2_vs_cl": comp["z_l1_l2_vs_cl"],
                    "basis_projection": distill_256_matrix(main_u_star, q_wht, q_dft, q_hyb),
                }
        print(f"  [Probe III] R={r} completed", flush=True)

    return results, distill
