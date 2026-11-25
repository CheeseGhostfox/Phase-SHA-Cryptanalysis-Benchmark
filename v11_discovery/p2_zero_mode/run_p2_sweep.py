"""P2「时空晶体等零面零模波包」主训练、阶梯扫频、真值核验与 3D 波包结构提取脚本。

支持：
  - --mode smoke : 秒级本地 CPU 冒烟（CUDA_VISIBLE_DEVICES=""）
  - --mode full  : Colab L4/T4 完整 7x12=84 格点扫频（STD + BTC320 + CTRL_RAND_SHA + RAND_WALK）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from v10_ouroboros64.ouroboros_core.engine import (
    MASK32,
    enforce_fp32_determinism,
    popcnt32_torch,
    words_to_bits_lsb,
)

from .p2_core import (
    K_GRID,
    ROUNDS_GRID,
    CipherConfig,
    ParametricCrystalEngine,
    binomial_z_score,
    collect_base_crystals_for_grid,
    compute_wavepacket_kl_and_bias,
    exact_binomial_pvalue_greater,
    free_word_count,
    leading_zeros_u32_torch,
    make_random_control_cipher_config,
    make_standard_cipher_config,
    patch_model_for_cipher,
    two_proportion_z,
    verify_sha256_rounds_np,
    wilson_ci95,
)
from .p2_proposer import (
    ZeroModeCrystalProposer,
    deduplicate_candidates_per_base,
    generate_rand_walk_candidates,
)


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _popcnt_u32_np(arr: np.ndarray) -> np.ndarray:
    """对 uint32 数组逐元素计算 popcount。"""
    v = np.asarray(arr, dtype=np.uint32)
    v = v - ((v >> np.uint32(1)) & np.uint32(0x55555555))
    v = (v & np.uint32(0x33333333)) + ((v >> np.uint32(2)) & np.uint32(0x33333333))
    v = (v + (v >> np.uint32(4))) & np.uint32(0x0F0F0F0F)
    return ((v * np.uint32(0x01010101)) & np.uint32(0xFFFFFFFF)) >> np.uint32(24)


def _verify_hits_on_cpu_np(
    w0_all: np.ndarray,
    dw0_all: np.ndarray,
    h_gpu_all: np.ndarray,
    hit_mask: np.ndarray,
    rounds: int,
    k: int,
    cipher_cfg: CipherConfig,
) -> int:
    """对全部 GPU 判定命中的样本执行独立纯 NumPy 前向核验，确保 0 误差。"""
    idx = np.flatnonzero(hit_mask)
    if idx.size == 0:
        return 0
    w_pert = (w0_all[idx] ^ dw0_all[idx]).astype(np.uint32)
    h_np = verify_sha256_rounds_np(w_pert, rounds=rounds, cipher_cfg=cipher_cfg)
    if not np.array_equal(h_np, h_gpu_all[idx].astype(np.uint32)):
        raise AssertionError(
            f"GPU 与纯 NumPy SHA-256 真值不一致！R={rounds}, k={k}, cfg={cipher_cfg.name}"
        )
    shift = 31 - k
    verified = (h_np[:, 0] >> np.uint32(shift)) == np.uint32(0)
    if not np.all(verified):
        raise AssertionError(f"命中校验失败: R={rounds}, k={k}, {np.sum(~verified)}/{idx.size} 未过前缀零")
    return int(idx.size)


def _extract_dw_coords(dw_row: np.ndarray, w_free: int) -> List[Tuple[int, int]]:
    """从 (16,) uint32 微扰字提取排序后的 (t, bit) 坐标列表。"""
    coords: List[Tuple[int, int]] = []
    for t in range(w_free):
        word = int(dw_row[t])
        while word:
            lsb = word & -word
            b = lsb.bit_length() - 1
            coords.append((t, b))
            word ^= lsb
    return coords


def _compute_3d_wavepacket_record(
    eng: ParametricCrystalEngine,
    rounds: int,
    k: int,
    carrier: str,
    method: str,
    w_free: int,
    w0_flat: np.ndarray,
    dw0_flat: np.ndarray,
    h0_flat: np.ndarray,
    h1_flat: np.ndarray,
    hit_mask: np.ndarray,
    max_heatmap_samples: int = 512,
    max_verify_samples: int = 8,
) -> Dict[str, Any]:
    """按 SPEC 第 5 节提取 3D 差分波包热力图指标、Hamming 重分布与 Top-20 稀疏团簇模式。"""
    hit_idx = np.flatnonzero(hit_mask)
    miss_idx = np.flatnonzero(~hit_mask)

    hamming_all = _popcnt_u32_np(dw0_flat[:, :w_free]).sum(axis=1)
    max_bin = int(min(32 * w_free, max(12, int(hamming_all.max()) if hamming_all.size > 0 else 8)))

    def _hist_dict(vals: np.ndarray) -> Dict[str, float]:
        if vals.size == 0:
            return {}
        cnt = Counter(int(v) for v in vals)
        tot = float(vals.size)
        return {str(m): float(c) / tot for m, c in sorted(cnt.items()) if m <= max_bin}

    p_m_all = _hist_dict(hamming_all)
    p_m_hit = _hist_dict(hamming_all[hit_idx]) if hit_idx.size > 0 else {}

    def _accumulate_heatmap(indices: np.ndarray) -> Optional[np.ndarray]:
        if indices.size == 0:
            return None
        take = indices[:max_heatmap_samples]
        w0_t = torch.from_numpy(w0_flat[take].astype(np.int64))
        w1_t = torch.from_numpy((w0_flat[take] ^ dw0_flat[take]).astype(np.int64))
        c0 = eng.compress_exact(w0_t, rounds=rounds, hash_only=False)
        c1 = eng.compress_exact(w1_t, rounds=rounds, hash_only=False)

        dw_diff = c1["W"] ^ c0["W"]  # (N, R)
        ds_diff = c1["S"][:, 1 : rounds + 1, :] ^ c0["S"][:, 1 : rounds + 1, :]  # (N, R, 8)
        dc_diff = c1["C"] ^ c0["C"]  # (N, R, 10)
        planes = torch.cat([dw_diff.unsqueeze(2), ds_diff, dc_diff], dim=2)  # (N, R, 19)
        bits = words_to_bits_lsb(planes)  # (N, R, 19, 32) float32
        return bits.mean(dim=0).cpu().numpy()

    hm_hit = _accumulate_heatmap(hit_idx)
    hm_miss = _accumulate_heatmap(miss_idx)

    hit_stats = compute_wavepacket_kl_and_bias(hm_hit) if hm_hit is not None else None
    miss_stats = compute_wavepacket_kl_and_bias(hm_miss) if hm_miss is not None else None

    # 提取 Top-20 高频稀疏输入团簇模式
    cluster_counter: Counter[Tuple[Tuple[int, int], ...]] = Counter()
    cluster_example: Dict[Tuple[Tuple[int, int], ...], int] = {}
    for idx_val in hit_idx[:4096]:
        pat = tuple(_extract_dw_coords(dw0_flat[idx_val], w_free=w_free))
        cluster_counter[pat] += 1
        if pat not in cluster_example:
            cluster_example[pat] = int(idx_val)

    top_clusters = []
    for pat, cnt in cluster_counter.most_common(20):
        ex_i = cluster_example[pat]
        # 计算字内/跨字偏移特征
        offsets = []
        if len(pat) >= 2:
            t0, b0 = pat[0]
            for tj, bj in pat[1:]:
                offsets.append({"dt": int(tj - t0), "db": int(bj - b0)})
        top_clusters.append(
            {
                "coords_tb": [[int(t), int(b)] for t, b in pat],
                "hamming_w": len(pat),
                "hit_count": int(cnt),
                "offsets_from_anchor": offsets,
                "W0_hex": [f"0x{int(x):08x}" for x in w0_flat[ex_i].tolist()],
                "dW_hex": [f"0x{int(x):08x}" for x in dw0_flat[ex_i].tolist()],
                "H0_hex": [f"0x{int(x):08x}" for x in h0_flat[ex_i].tolist()],
                "H1_hex": [f"0x{int(x):08x}" for x in h1_flat[ex_i].tolist()],
            }
        )

    # 导出可直接重算验真的样本列表（含 uint32 与 hex）
    verifiable_samples = []
    for idx_val in hit_idx[:max_verify_samples]:
        verifiable_samples.append(
            {
                "R": int(rounds),
                "k": int(k),
                "carrier": carrier,
                "method": method,
                "W0_u32": [int(x) for x in w0_flat[idx_val].tolist()],
                "dW0_u32": [int(x) for x in dw0_flat[idx_val].tolist()],
                "H0_u32": [int(x) for x in h0_flat[idx_val].tolist()],
                "H1_u32": [int(x) for x in h1_flat[idx_val].tolist()],
                "W0_hex": [f"0x{int(x):08x}" for x in w0_flat[idx_val].tolist()],
                "dW_hex": [f"0x{int(x):08x}" for x in dw0_flat[idx_val].tolist()],
                "H0_hex": [f"0x{int(x):08x}" for x in h0_flat[idx_val].tolist()],
                "H1_hex": [f"0x{int(x):08x}" for x in h1_flat[idx_val].tolist()],
            }
        )

    return {
        "R": int(rounds),
        "k": int(k),
        "carrier": carrier,
        "method": method,
        "n_hits_total": int(hit_idx.size),
        "n_hits_heatmap": int(min(hit_idx.size, max_heatmap_samples)),
        "n_misses_heatmap": int(min(miss_idx.size, max_heatmap_samples)),
        "hamming_dist_all": p_m_all,
        "hamming_dist_hit": p_m_hit,
        "hit_heatmap_stats": hit_stats,
        "miss_heatmap_stats": miss_stats,
        "top20_clusters": top_clusters,
        "verifiable_samples": verifiable_samples,
    }


def train_proposer_for_round(
    eng: ParametricCrystalEngine,
    cipher_cfg: CipherConfig,
    rounds: int,
    carrier: str,
    train_bank: Dict[int, Dict[str, torch.Tensor]],
    k_grid: Tuple[int, ...],
    device: torch.device,
    d_node: int,
    n_layers: int,
    train_steps: int,
    batch_train: int,
    k_train: int,
    seed: int,
) -> Tuple[ZeroModeCrystalProposer, float]:
    """在独立训练基底晶体流形上用 REINFORCE + 前缀零保留软奖励训练 3D 零模提议器。"""
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    proposer = ZeroModeCrystalProposer(
        d_node=d_node,
        n_layers=n_layers,
        bond_dim=4,
        d_key=24,
        max_bits=8,
    ).to(device=device, dtype=torch.float32)

    if cipher_cfg.is_control:
        patch_model_for_cipher(proposer.net, cipher_cfg)

    # 校验全部参数均为纯 FP32
    for p_name, param in proposer.named_parameters():
        if param.dtype != torch.float32:
            raise AssertionError(f"参数 {p_name} 非 FP32: {param.dtype}")

    w_free = free_word_count(rounds, carrier)
    optimizer = torch.optim.AdamW(proposer.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, train_steps), eta_min=2e-4
    )

    rng = np.random.default_rng(seed)
    avail_ks = sorted(train_bank.keys())
    # 70% 倾向低-中阶 k<=6 保证充足正命中梯度，30% 覆盖高阶 k>6
    weights = np.array([1.8 if k <= 6 else 0.7 for k in avail_ks], dtype=np.float64)
    weights /= weights.sum()

    proposer.train()
    total_hits = 0
    total_cands = 0

    for step in range(train_steps):
        chosen_ks = rng.choice(avail_ks, size=batch_train, p=weights)
        w0_list = []
        for kv in chosen_ks:
            bank_w = train_bank[int(kv)]["W0"]
            idx = int(rng.integers(0, bank_w.shape[0]))
            w0_list.append(bank_w[idx])
        w0_t = torch.stack(w0_list, dim=0).to(device=device)
        k_t = torch.tensor(chosen_ks, device=device, dtype=torch.int64)

        with torch.no_grad():
            cr0 = eng.compress_exact(w0_t, rounds=rounds, hash_only=False)
            h0 = cr0["H"]
            h_target_star = h0.clone()
            h_target_star[:, 0] = h0[:, 0] ^ (torch.ones_like(k_t) << (31 - k_t))

        policy = proposer.encode_crystal_policy(
            H_target_star=h_target_star,
            W=cr0["W"],
            S=cr0["S"],
            C=cr0["C"],
            rounds=rounds,
            w_free=w_free,
            crystal_engine=eng,
        )
        samp = proposer.sample_from_policy(
            policy,
            num_samples=k_train,
            temperature=1.0,
            include_greedy=False,
            return_log_prob=True,
        )
        dw0 = samp["dW0"]  # (B, K, 16)
        log_prob = samp["log_prob"]  # (B, K)
        entropy = samp["entropy"]  # (B,)

        with torch.no_grad():
            w_pert = (w0_t.unsqueeze(1) ^ dw0).reshape(batch_train * k_train, 16)
            h_prime = eng.compress_exact(w_pert, rounds=rounds, hash_only=True)["H"]
            h_prime_0 = h_prime[:, 0].reshape(batch_train, k_train)

            lcz_p = leading_zeros_u32_torch(h_prime_0)
            k_exp = k_t.unsqueeze(1)
            is_hit = lcz_p >= (k_exp + 1)
            prefix_surv = torch.minimum(lcz_p, k_exp).to(torch.float32) / k_exp.to(torch.float32)

            shift = (31 - k_exp).clamp(min=0)
            xor_top = ((h_prime_0 ^ h_target_star[:, 0].unsqueeze(1)) & MASK32) >> shift
            mismatches = popcnt32_torch(xor_top).to(torch.float32)
            top_match = ((k_exp + 1).to(torch.float32) - mismatches) / (k_exp + 1).to(torch.float32)

            reward = 4.0 * is_hit.to(torch.float32) + 0.8 * prefix_surv + 0.4 * top_match
            adv = reward - reward.mean(dim=1, keepdim=True)
            std = reward.std(dim=1, keepdim=True) + 1e-5
            adv_norm = (adv / std).clamp(-4.0, 4.0)

            total_hits += int(is_hit.sum().item())
            total_cands += batch_train * k_train

        ent_coef = 0.015 * max(0.2, 1.0 - step / float(max(1, train_steps)))
        loss = -(adv_norm * log_prob).mean() - ent_coef * entropy.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proposer.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    proposer.eval()
    elapsed = time.perf_counter() - t0
    print(
        f"  [Train] R={rounds:2d} {carrier:6s} {'CTRL' if cipher_cfg.is_control else 'SHA ':4s} "
        f"steps={train_steps} train_hit_rate={total_hits}/{total_cands} "
        f"({100.0 * total_hits / max(1, total_cands):.3f}%) time={elapsed:.1f}s",
        flush=True,
    )
    return proposer, elapsed


@torch.no_grad()
def evaluate_cell(
    proposer: Optional[ZeroModeCrystalProposer],
    eng: ParametricCrystalEngine,
    cipher_cfg: CipherConfig,
    rounds: int,
    k: int,
    carrier: str,
    method: str,
    eval_w0: torch.Tensor,
    eval_h0: torch.Tensor,
    k_cand_per_base: int,
    rng: np.random.Generator,
    extract_wavepacket: bool,
    device: torch.device,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """对单个 (R, k, carrier, method) 格点执行严格冻结推理、基底内去重与精确前向真值核验。"""
    n_base = int(eval_w0.shape[0])
    w_free = free_word_count(rounds, carrier)

    w0_np = eval_w0.cpu().numpy().astype(np.uint32)
    h0_np = eval_h0.cpu().numpy().astype(np.uint32)

    if carrier == "BTC320":
        assert np.all(w0_np[:, 4] == np.uint32(0x80000000))
        assert np.all(w0_np[:, 5:15] == 0)
        assert np.all(w0_np[:, 15] == np.uint32(640))

    if method == "RAND_WALK":
        dw0_np = generate_rand_walk_candidates(
            rng=rng,
            bsz=n_base,
            k_target=k_cand_per_base,
            w_free=w_free,
            max_bits=6,
        )
        neural_unique_frac = 0.0
    else:
        assert proposer is not None
        dw_chunks: List[np.ndarray] = []
        fracs: List[float] = []
        base_chunk = 32
        k_pool = min(1024, max(k_cand_per_base * 2, k_cand_per_base + 128))
        for start in range(0, n_base, base_chunk):
            end = min(n_base, start + base_chunk)
            wb = eval_w0[start:end].to(device=device)
            cr_b = eng.compress_exact(wb, rounds=rounds, hash_only=False)
            h_star = cr_b["H"].clone()
            h_star[:, 0] = h_star[:, 0] ^ (1 << (31 - k))

            policy = proposer.encode_crystal_policy(
                H_target_star=h_star,
                W=cr_b["W"],
                S=cr_b["S"],
                C=cr_b["C"],
                rounds=rounds,
                w_free=w_free,
                crystal_engine=eng,
            )
            # 先采第一批温度 1.0 候选，若去重不足则补一批高温候选
            s1 = proposer.sample_from_policy(
                policy,
                num_samples=k_pool // 2,
                temperature=1.0,
                include_greedy=True,
                return_log_prob=False,
            )
            s2 = proposer.sample_from_policy(
                policy,
                num_samples=k_pool - (k_pool // 2),
                temperature=1.45,
                include_greedy=False,
                return_log_prob=False,
            )
            cand_free = torch.cat(
                [s1["dW0"][:, :, :w_free], s2["dW0"][:, :, :w_free]], dim=1
            ).cpu().numpy().astype(np.uint32)

            dedup_b, frac_b = deduplicate_candidates_per_base(
                cand_free, k_target=k_cand_per_base, w_free=w_free, rng=rng
            )
            dw_chunks.append(dedup_b)
            fracs.append(frac_b)

        dw0_np = np.concatenate(dw_chunks, axis=0)
        neural_unique_frac = float(np.mean(fracs))

    # 硬校验：非零微扰且严格位于活跃自由域内
    assert np.all(np.any(dw0_np != 0, axis=2)), "存在非法零微扰 dW0 == 0"
    if w_free < 16:
        assert np.all(dw0_np[:, :, w_free:] == 0), "存在越界微扰 (w_free 锁定区被修改)"

    w0_rep = np.repeat(w0_np[:, None, :], k_cand_per_base, axis=1).reshape(-1, 16)
    h0_rep = np.repeat(h0_np[:, None, :], k_cand_per_base, axis=1).reshape(-1, 8)
    dw0_flat = dw0_np.reshape(-1, 16)
    n_eval = int(dw0_flat.shape[0])

    # 在 GPU/引擎上批量精确计算前向 R 轮输出
    h1_chunks: List[np.ndarray] = []
    hash_chunk = 65536
    w_pert_all = w0_rep ^ dw0_flat
    for st in range(0, n_eval, hash_chunk):
        ed = min(n_eval, st + hash_chunk)
        w_t = torch.from_numpy(w_pert_all[st:ed].astype(np.int64))
        h_out = eng.compress_exact(w_t, rounds=rounds, hash_only=True)["H"]
        h1_chunks.append(h_out.cpu().numpy().astype(np.uint32))
    h1_flat = np.concatenate(h1_chunks, axis=0)

    shift = 31 - k
    hit_mask = (h1_flat[:, 0] >> np.uint32(shift)) == np.uint32(0)

    # 对全部命中样本执行独立纯 NumPy SHA-256 真值复核
    hits = _verify_hits_on_cpu_np(
        w0_all=w0_rep,
        dw0_all=dw0_flat,
        h_gpu_all=h1_flat,
        hit_mask=hit_mask,
        rounds=rounds,
        k=k,
        cipher_cfg=cipher_cfg,
    )

    p0 = 2.0 ** (-(k + 1))
    p_hit = float(hits) / float(n_eval)
    gain_g = p_hit / p0
    z_score = binomial_z_score(hits, n_eval, p0)
    p_binom = exact_binomial_pvalue_greater(hits, n_eval, p0)
    ci_lo, ci_hi = wilson_ci95(hits, n_eval)

    hamming_all = _popcnt_u32_np(dw0_flat[:, :w_free]).sum(axis=1)
    mean_hw = float(np.mean(hamming_all))
    mean_hw_hit = float(np.mean(hamming_all[hit_mask])) if hits > 0 else None

    cell_rec = {
        "R": int(rounds),
        "k": int(k),
        "carrier": carrier,
        "method": method,
        "n_base": int(n_base),
        "k_cand_per_base": int(k_cand_per_base),
        "N_eval": int(n_eval),
        "hits": int(hits),
        "p_hit": p_hit,
        "p_rand": p0,
        "gain_G": gain_g,
        "z_score": z_score,
        "p_binom": p_binom,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "mean_hamming_w": mean_hw,
        "mean_hamming_w_hit": mean_hw_hit,
        "neural_unique_frac": neural_unique_frac,
    }

    wp_rec = None
    if extract_wavepacket or z_score >= 3.5:
        wp_rec = _compute_3d_wavepacket_record(
            eng=eng,
            rounds=rounds,
            k=k,
            carrier=carrier,
            method=method,
            w_free=w_free,
            w0_flat=w0_rep,
            dw0_flat=dw0_flat,
            h0_flat=h0_rep,
            h1_flat=h1_flat,
            hit_mask=hit_mask,
        )

    return cell_rec, wp_rec


def run_sweep(args: argparse.Namespace) -> None:
    t_global = time.perf_counter()
    enforce_fp32_determinism()

    if args.mode == "smoke":
        device = torch.device("cpu")
        rounds_list = (4, 8)
        k_list = (1, 2)
        n_base_by_k = {1: 4, 2: 4}
        k_cand_per_base = 16
        train_counts_by_k = {1: 16, 2: 16}
        carriers_configs = [
            ("STD", False, ["NEURAL", "RAND_WALK"]),
            ("BTC320", False, ["NEURAL"]),
            ("STD", True, ["CTRL_RAND_SHA"]),
        ]
        train_steps = args.train_steps if args.train_steps > 0 else 3
        batch_train = 4
        k_train = 4
        d_node = 32
        n_layers = 2
        max_bank_draws = 5000
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rounds_list = ROUNDS_GRID
        k_list = K_GRID
        n_base_by_k = {k: (64 if k <= 10 else 256) for k in k_list}
        k_cand_per_base = 512
        train_counts_by_k = {k: (512 if k <= 8 else (256 if k <= 10 else 64)) for k in k_list}
        carriers_configs = [
            ("STD", False, ["NEURAL", "RAND_WALK"]),
            ("BTC320", False, ["NEURAL", "RAND_WALK"]),
            ("STD", True, ["CTRL_RAND_SHA"]),
        ]
        if args.include_btc_ctrl:
            carriers_configs.append(("BTC320", True, ["CTRL_RAND_SHA"]))
        train_steps = args.train_steps if args.train_steps > 0 else 220
        batch_train = args.batch_train
        k_train = args.k_train
        d_node = args.d_node
        n_layers = args.n_layers
        max_bank_draws = 12_000_000

    tf32_matmul = bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else False
    tf32_cudnn = bool(torch.backends.cudnn.allow_tf32) if torch.cuda.is_available() else False
    if tf32_matmul or tf32_cudnn:
        raise AssertionError(f"TF32 未禁用: matmul={tf32_matmul}, cudnn={tf32_cudnn}")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(
        f"[P2 Init] mode={args.mode} device={device} gpu={gpu_name} "
        f"tf32_matmul={tf32_matmul} tf32_cudnn={tf32_cudnn} seed={args.seed}",
        flush=True,
    )

    std_cfg = make_standard_cipher_config()
    ctrl_cfg = make_random_control_cipher_config(seed=args.seed + 777)

    eng_std = ParametricCrystalEngine(std_cfg, device=device)
    eng_ctrl = ParametricCrystalEngine(ctrl_cfg, device=device)

    all_cells: List[Dict[str, Any]] = []
    wavepackets: List[Dict[str, Any]] = []
    train_timings: List[Dict[str, Any]] = []
    eval_start_total = time.perf_counter()

    # 代表性波包提取档位（其余 z >= 3.5 的显著格点也会自动触发提取）
    rep_ks = {1, 4, 8, 12} if args.mode == "full" else {1, 2}

    for rounds in rounds_list:
        for carrier, is_ctrl, methods in carriers_configs:
            cfg = ctrl_cfg if is_ctrl else std_cfg
            eng = eng_ctrl if is_ctrl else eng_std
            c_id = 1 if carrier == "STD" else 2
            ctrl_off = 50000 if is_ctrl else 0

            # 严格零泄漏划分：训练流形与留出评估流形采用完全独立的种子
            train_seed = args.seed + 1000 * rounds + 17 * c_id + ctrl_off
            eval_seed = args.seed + 900000 + 1000 * rounds + 17 * c_id + ctrl_off

            t_bank0 = time.perf_counter()
            need_train = any(m in ("NEURAL", "CTRL_RAND_SHA") for m in methods)
            if need_train:
                train_bank = collect_base_crystals_for_grid(
                    eng=eng,
                    rounds=rounds,
                    carrier=carrier,
                    counts_by_k=train_counts_by_k,
                    seed=train_seed,
                    max_draws=max_bank_draws,
                )
            else:
                train_bank = {}

            eval_bank = collect_base_crystals_for_grid(
                eng=eng,
                rounds=rounds,
                carrier=carrier,
                counts_by_k=n_base_by_k,
                seed=eval_seed,
                max_draws=max_bank_draws,
            )
            t_bank = time.perf_counter() - t_bank0
            print(
                f"[Bank] R={rounds:2d} {carrier:6s} {'CTRL' if is_ctrl else 'SHA ':4s} "
                f"collected in {t_bank:.2f}s",
                flush=True,
            )

            trained_proposer: Optional[ZeroModeCrystalProposer] = None
            if need_train:
                trained_proposer, tr_sec = train_proposer_for_round(
                    eng=eng,
                    cipher_cfg=cfg,
                    rounds=rounds,
                    carrier=carrier,
                    train_bank=train_bank,
                    k_grid=k_list,
                    device=device,
                    d_node=d_node,
                    n_layers=n_layers,
                    train_steps=train_steps,
                    batch_train=batch_train,
                    k_train=k_train,
                    seed=train_seed + 31,
                )
                train_timings.append(
                    {
                        "R": int(rounds),
                        "carrier": carrier,
                        "is_control": is_ctrl,
                        "train_sec": tr_sec,
                    }
                )

            for method in methods:
                eval_rng = np.random.default_rng(
                    eval_seed + (333 if method == "RAND_WALK" else 111)
                )
                t_ev0 = time.perf_counter()
                for k in k_list:
                    do_wp = k in rep_ks
                    cell_rec, wp_rec = evaluate_cell(
                        proposer=trained_proposer if method != "RAND_WALK" else None,
                        eng=eng,
                        cipher_cfg=cfg,
                        rounds=rounds,
                        k=k,
                        carrier=carrier,
                        method=method,
                        eval_w0=eval_bank[k]["W0"],
                        eval_h0=eval_bank[k]["H0"],
                        k_cand_per_base=k_cand_per_base,
                        rng=eval_rng,
                        extract_wavepacket=do_wp,
                        device=device,
                    )
                    all_cells.append(cell_rec)
                    if wp_rec is not None:
                        wavepackets.append(wp_rec)

                t_ev = time.perf_counter() - t_ev0
                sub = [c for c in all_cells if c["R"] == rounds and c["carrier"] == carrier and c["method"] == method]
                max_z_cell = max(sub, key=lambda x: x["z_score"])
                print(
                    f"  [Eval] R={rounds:2d} {carrier:6s} {method:13s} time={t_ev:.1f}s | "
                    f"peak_z={max_z_cell['z_score']:+.2f} at k={max_z_cell['k']} "
                    f"(hits={max_z_cell['hits']}/{max_z_cell['N_eval']}, G={max_z_cell['gain_G']:.3f})",
                    flush=True,
                )

        # 每完成一轮 R 立即原子落盘增量进度，防止长时任务意外中断丢失数据
        _atomic_write_json(
            Path(args.out_dir) / "sweep_matrix_partial.json",
            {"completed_round": int(rounds), "cells": all_cells},
        )

    # 补充对齐统计：将 NEURAL 与同载体 CTRL_RAND_SHA 及 RAND_WALK 配对计算 G_net 与 z_vs_ctrl
    lookup: Dict[Tuple[int, int, str, str], Dict[str, Any]] = {
        (c["R"], c["k"], c["carrier"], c["method"]): c for c in all_cells
    }
    for cell in all_cells:
        r_v, k_v, car_v, m_v = cell["R"], cell["k"], cell["carrier"], cell["method"]
        ctrl_cell = lookup.get((r_v, k_v, car_v, "CTRL_RAND_SHA")) or lookup.get(
            (r_v, k_v, "STD", "CTRL_RAND_SHA")
        )
        rw_cell = lookup.get((r_v, k_v, car_v, "RAND_WALK"))

        if m_v == "NEURAL" and ctrl_cell is not None:
            p_ctrl_eff = max(ctrl_cell["p_hit"], 1.0 / float(cell["N_eval"]))
            cell["ctrl_hits"] = ctrl_cell["hits"]
            cell["ctrl_N_eval"] = ctrl_cell["N_eval"]
            cell["ctrl_p_hit"] = ctrl_cell["p_hit"]
            cell["gain_G_net"] = cell["p_hit"] / p_ctrl_eff
            cell["z_vs_ctrl"] = two_proportion_z(
                cell["hits"], cell["N_eval"], ctrl_cell["hits"], ctrl_cell["N_eval"]
            )
            cell["hit_light"] = bool(
                cell["z_score"] >= 6.0 and cell["gain_G"] > 1.0 and cell["z_vs_ctrl"] >= 4.0
            )
        else:
            cell["hit_light"] = False

        if m_v == "NEURAL" and rw_cell is not None:
            cell["rw_hits"] = rw_cell["hits"]
            cell["rw_p_hit"] = rw_cell["p_hit"]
            cell["z_vs_rand_walk"] = two_proportion_z(
                cell["hits"], cell["N_eval"], rw_cell["hits"], rw_cell["N_eval"]
            )

    total_sec = time.perf_counter() - t_global
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_payload = {
        "metadata": {
            "experiment": "P2_crystal_zero_mode_wavepacket",
            "mode": args.mode,
            "device": str(device),
            "gpu_name": gpu_name,
            "torch_version": torch.__version__,
            "fp32_locked": True,
            "tf32_matmul": tf32_matmul,
            "tf32_cudnn": tf32_cudnn,
            "seed": args.seed,
            "rounds_grid": list(rounds_list),
            "k_grid": list(k_list),
            "total_elapsed_sec": total_sec,
            "train_timings": train_timings,
            "control_cipher": {
                "name": ctrl_cfg.name,
                "sigma0_rots": list(ctrl_cfg.sigma0_rots),
                "sigma1_rots": list(ctrl_cfg.sigma1_rots),
                "lower_sigma0_rots": list(ctrl_cfg.lower_sigma0_rots),
                "lower_sigma0_shr": ctrl_cfg.lower_sigma0_shr,
                "lower_sigma1_rots": list(ctrl_cfg.lower_sigma1_rots),
                "lower_sigma1_shr": ctrl_cfg.lower_sigma1_shr,
                "k_words_hex": [f"0x{x:08x}" for x in ctrl_cfg.k_words],
            },
        },
        "cells": all_cells,
    }
    wavepacket_payload = {
        "metadata": {
            "experiment": "P2_wavepacket_structures",
            "mode": args.mode,
            "total_records": len(wavepackets),
        },
        "structures": wavepackets,
    }

    _atomic_write_json(out_dir / "sweep_matrix.json", sweep_payload)
    _atomic_write_json(out_dir / "wavepacket_structures.json", wavepacket_payload)
    print(
        f"[P2 Done] cells={len(all_cells)} wavepackets={len(wavepackets)} "
        f"total_elapsed={total_sec:.1f}s -> {out_dir}",
        flush=True,
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P2 Crystal Zero-Mode Wavepacket Sweep")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
    )
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--train_steps", type=int, default=0)
    p.add_argument("--batch_train", type=int, default=24)
    p.add_argument("--k_train", type=int, default=24)
    p.add_argument("--d_node", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--include_btc_ctrl", action="store_true", default=True)
    p.add_argument("--no_btc_ctrl", dest="include_btc_ctrl", action="store_false")
    return p


if __name__ == "__main__":
    run_sweep(build_argparser().parse_args())
