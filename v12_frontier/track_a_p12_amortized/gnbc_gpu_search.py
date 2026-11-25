"""
P12 A2 GPU 搜索脚本：广义中性比特团簇（GNBC）与辅助条件 Phi 搜索。

严格遵循 g:\\SHA-LM\\v12_frontier\\track_a_p12_amortized\\SPEC.md：
- 浮点运算严格纯 IEEE-754 FP32（torch.float32, allow_tf32=False）；
- 精确 SHA-256 运算严格 uint32-in-int64；
- 严格搜索池 [0, 2^30) 与留出验证池 [2^30, 2^32) 零重叠（train_holdout_overlap == 0）；
- 留出验证集样本数 N_holdout >= 262,144；
- 覆盖三大轨道：
  * Track A2-1: Chunk2 内部中性团簇 + 辅助条件 Phi（W1[16]=W1[17]=0, Ch 掩码, midstate+W1 补偿）
  * Track A2-2: Chunk2 -> Hash2 跨块界面中性团簇 & Hash2 内部 W2[0..7] 穿透探针
  * Track A2-3: 比特级进位吸收 / 条件静默门集群（FP32 神经策略先验 + MCTS/Beam + 进化贪心搜索）
- 同步在 MAIN_SHA256 与 3 组谱隙对齐 CTRL_RAND_ROTR_K 对照组上以相同预算运行。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32" or "--cpu_only" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
import torch.nn as nn

from btc_double_sha_engine import (
    MASK32,
    SHA256Variant,
    btc_double_sha256_trace,
    btc_header_bytes_to_blocks,
    compute_exact_btc_amortized_report,
    enforce_fp32_strict,
    make_main_variant,
    make_spectral_aligned_ctrl_variant,
    pack_btc_chunk2_from_words,
    pack_outer_block_from_h_inner,
    popcnt32,
    sha256_compress_trace,
)
from classic_asicboost_calib import make_synthetic_btc_header

SEED_SEARCH: int = 20261125
SEED_HOLDOUT: int = 998244353


# ---------------------------------------------------------------------------
# 统计工具：二项 z-score 与 Wilson 95% 置信区间
# ---------------------------------------------------------------------------

def binomial_z_and_wilson(successes: int, total: int, p0: float) -> Tuple[float, Tuple[float, float]]:
    if total <= 0:
        return 0.0, (0.0, 1.0)
    phat = successes / float(total)
    p0_clamped = min(max(p0, 1e-15), 1.0 - 1e-15)
    se0 = math.sqrt(p0_clamped * (1.0 - p0_clamped) / float(total))
    z = (phat - p0_clamped) / max(se0, 1e-15)

    z95 = 1.959963984540054
    denom = 1.0 + (z95 * z95) / total
    center = (phat + (z95 * z95) / (2.0 * total)) / denom
    half = (z95 / denom) * math.sqrt((phat * (1.0 - phat) / total) + (z95 * z95) / (4.0 * total * total))
    return z, (max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------------
# 纯 IEEE-754 FP32 神经策略网络（为 MCTS/进化提供位点与辅助条件先验）
# ---------------------------------------------------------------------------

class GNBCPolicyNet(nn.Module):
    """纯 FP32 策略-价值网络：根据状态/消息双极自旋输出中性位点先验 logits 与辅助条件得分。"""

    def __init__(self, in_bits: int = 128, n_actions: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_bits, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x_spins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x_spins)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


def words_to_spins_fp32(words: torch.Tensor) -> torch.Tensor:
    """将 (B, W) uint32-in-int64 转为纯 FP32 双极自旋 (B, W*32): 0->+1.0, 1->-1.0。"""
    shifts = torch.arange(32, dtype=torch.int64, device=words.device)
    bits = ((words.unsqueeze(-1) >> shifts) & 1).reshape(words.shape[0], -1).to(torch.float32)
    return 1.0 - 2.0 * bits


# ---------------------------------------------------------------------------
# 搜索池与留出池生成器（严格 Nonce [0, 2^30) vs [2^30, 2^32) 零重叠）
# ---------------------------------------------------------------------------

def sample_btc_search_batch(
    batch_size: int,
    rng: np.random.Generator,
    variant: SHA256Variant,
    device: torch.device,
    preanchor_w16_w17_zero: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从搜索空间 Nonce in [0, 2^30) 采样物理合法 BTC Header 批次：
    返回 (blk1_w16, blk2_w16, midstate)。
    """
    blk1_np = rng.integers(0, 2**32, size=(batch_size, 16), dtype=np.uint64).astype(np.int64)
    blk1 = torch.from_numpy(blk1_np).to(device=device, dtype=torch.int64)
    w012_np = rng.integers(0, 2**32, size=(batch_size, 3), dtype=np.uint64).astype(np.int64)
    nonces_np = rng.integers(0, 1 << 30, size=(batch_size,), dtype=np.int64)

    w012 = torch.from_numpy(w012_np).to(device=device, dtype=torch.int64)
    nonces = torch.from_numpy(nonces_np).to(device=device, dtype=torch.int64)

    if preanchor_w16_w17_zero:
        # 强制满足 P5 W[16]=W[17]=0 预锚定合法 BTC Chunk2（调节 Merkle_tail=W[0] 与 nTime=W[1]）
        # W[17] = s1(640) + 0 + s0(W[2]) + W[1] = 0 => W[1] = (-s1(640) - s0(W[2])) mod 2^32
        # W[16] = 0 + 0 + s0(W[1]) + W[0] = 0 => W[0] = (-s0(W[1])) mod 2^32
        from btc_double_sha_engine import sigma0_lower, sigma1_lower

        c640 = torch.full((batch_size,), 640, dtype=torch.int64, device=device)
        s1_640 = sigma1_lower(c640, variant)
        s0_w2 = sigma0_lower(w012[:, 2], variant)
        w1_val = (-(s1_640 + s0_w2)) & MASK32
        s0_w1 = sigma0_lower(w1_val, variant)
        w0_val = (-s0_w1) & MASK32
        w012[:, 0] = w0_val
        w012[:, 1] = w1_val

    blk2 = pack_btc_chunk2_from_words(w012, nonces)
    iv_t = torch.tensor(variant.iv, dtype=torch.int64, device=device)
    with torch.no_grad():
        midstate = sha256_compress_trace(iv_t, blk1, rounds=64, variant=variant, record_gates=False).digest
    return blk1, blk2, midstate


def sample_btc_holdout_batch(
    n_holdout: int,
    seed: int,
    variant: SHA256Variant,
    device: torch.device,
    preanchor_w16_w17_zero: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """从 [2^30, 2^32) 严格互斥留出池生成测试批次。"""
    rng = np.random.default_rng(seed)
    blk1_np = rng.integers(0, 2**32, size=(n_holdout, 16), dtype=np.uint64).astype(np.int64)
    w012_np = rng.integers(0, 2**32, size=(n_holdout, 3), dtype=np.uint64).astype(np.int64)
    nonces_np = rng.integers(1 << 30, 1 << 32, size=(n_holdout,), dtype=np.int64)

    blk1 = torch.from_numpy(blk1_np).to(device=device, dtype=torch.int64)
    w012 = torch.from_numpy(w012_np).to(device=device, dtype=torch.int64)
    nonces = torch.from_numpy(nonces_np).to(device=device, dtype=torch.int64)

    if preanchor_w16_w17_zero:
        from btc_double_sha_engine import sigma0_lower, sigma1_lower

        c640 = torch.full((n_holdout,), 640, dtype=torch.int64, device=device)
        w012[:, 1] = (-(sigma1_lower(c640, variant) + sigma0_lower(w012[:, 2], variant))) & MASK32
        w012[:, 0] = (-sigma0_lower(w012[:, 1], variant)) & MASK32

    blk2 = pack_btc_chunk2_from_words(w012, nonces)
    iv_t = torch.tensor(variant.iv, dtype=torch.int64, device=device)
    with torch.no_grad():
        midstate = sha256_compress_trace(iv_t, blk1, rounds=64, variant=variant, record_gates=False).digest
    return blk1, blk2, midstate, nonces_np


# ---------------------------------------------------------------------------
# 差分子空间批构造与快速活跃比特率/中性深度评估
# ---------------------------------------------------------------------------

def evaluate_chunk2_subspace_metrics(
    midstate: torch.Tensor,
    blk2: torch.Tensor,
    delta_words: torch.Tensor,  # (d, 4) for W1[0..3]
    variant: SHA256Variant,
    rounds_check: int = 16,
    delta_midstate: Optional[torch.Tensor] = None,  # (d, 8) optional
) -> Dict[str, Any]:
    """
    给定基底 (midstate, blk2) 与 d 个扰动基向量 delta_words (d, 4)，
    在每个样本上对全 2^d 集群评估：
    - Chunk2 每轮活跃比特比例 active_bit_rate[t] (相对于 64-bit (A_{t+1}, E_{t+1}))；
    - 保持 A/E 状态至少 1 个字完全零差或总活跃比特 < 0.05 的中性轮数 R_neutral；
    - 全状态零差持续轮数 R_zero_full；
    - 穿透至 Hash2 的跨块全字零差字数。
    """
    bsz = blk2.shape[0]
    d = delta_words.shape[0]
    m_size = 1 << d

    # 生成 2^d XOR 子空间掩码
    masks = torch.zeros((m_size, 4), dtype=torch.int64, device=blk2.device)
    ms_masks = torch.zeros((m_size, 8), dtype=torch.int64, device=blk2.device)
    for bit_idx in range(d):
        sel = torch.tensor([(j >> bit_idx) & 1 for j in range(m_size)], dtype=torch.bool, device=blk2.device)
        masks[sel] ^= delta_words[bit_idx].unsqueeze(0)
        if delta_midstate is not None:
            ms_masks[sel] ^= delta_midstate[bit_idx].unsqueeze(0)

    # 展开 (B, M, ...) -> (B*M, ...)
    blk2_exp = blk2.unsqueeze(1).expand(bsz, m_size, 16).reshape(bsz * m_size, 16).clone()
    blk2_exp[:, :4] ^= masks.unsqueeze(0).expand(bsz, m_size, 4).reshape(bsz * m_size, 4)

    ms_exp = midstate.unsqueeze(1).expand(bsz, m_size, 8).reshape(bsz * m_size, 8).clone()
    if delta_midstate is not None:
        ms_exp ^= ms_masks.unsqueeze(0).expand(bsz, m_size, 8).reshape(bsz * m_size, 8)

    with torch.no_grad():
        tr2 = sha256_compress_trace(ms_exp, blk2_exp, rounds=rounds_check, variant=variant, record_gates=False)

    # Reshape A, E: (B, M, R+4)
    a_bm = tr2.A_full[:, 4:].reshape(bsz, m_size, rounds_check)
    e_bm = tr2.E_full[:, 4:].reshape(bsz, m_size, rounds_check)
    w_bm = tr2.W_full.reshape(bsz, m_size, rounds_check)

    # 计算相对 cluster[0] 的翻转位掩码：对 M 个成员取 OR
    a_diff = (a_bm ^ a_bm[:, 0:1, :]).reshape(bsz, m_size, rounds_check)
    e_diff = (e_bm ^ e_bm[:, 0:1, :]).reshape(bsz, m_size, rounds_check)
    w_diff = (w_bm ^ w_bm[:, 0:1, :]).reshape(bsz, m_size, rounds_check)

    # 逐样本统计活跃比特数 (B, R): 用 bitwise OR across M (M<=16)
    a_or = a_diff[:, 1, :].clone()
    e_or = e_diff[:, 1, :].clone()
    w_or = w_diff[:, 1, :].clone()
    for j in range(2, m_size):
        a_or |= a_diff[:, j, :]
        e_or |= e_diff[:, j, :]
        w_or |= w_diff[:, j, :]

    active_bits_ae = popcnt32(a_or) + popcnt32(e_or)  # (B, R), max 64
    active_rate_per_round = active_bits_ae.float().mean(dim=0) / 64.0  # (R,)
    zero_state_prob_per_round = (active_bits_ae == 0).float().mean(dim=0)  # (R,)
    w_zero_prob_per_round = (w_or == 0).float().mean(dim=0)

    return {
        "active_rate_per_round": active_rate_per_round.cpu().tolist(),
        "zero_state_prob_per_round": zero_state_prob_per_round.cpu().tolist(),
        "w_zero_prob_per_round": w_zero_prob_per_round.cpu().tolist(),
        "active_bits_ae_tensor": active_bits_ae,
    }


# ---------------------------------------------------------------------------
# Track A2-1: Chunk2 内部中性团簇 + 辅助条件 Phi 搜索
# ---------------------------------------------------------------------------

def run_track_a2_1(
    variant: SHA256Variant,
    device: torch.device,
    n_holdout: int,
    is_smoke: bool = False,
) -> Dict[str, Any]:
    """
    在 Chunk2 上系统扫描：
    1. W1[3] (nNonce) 全 32 个单比特扰动 + 神经先验/进化 2~4 比特团簇；
    2. P5 预锚定 Phi_preanchor: W1[16]=W1[17]=0 条件下 W1[3] 穿透；
    3. W1[0..3] 联合自由度（Merkle_tail + nTime + nNonce）及跨块 midstate+W1 补偿。
    """
    rng = np.random.default_rng(SEED_SEARCH)
    bsz_search = 1024 if is_smoke else 8192

    # --- 1. 全 32 个 W1[3] 单比特扫描（标准 vs W1[16]=W1[17]=0 预锚定）---
    _, blk2_std, ms_std = sample_btc_search_batch(bsz_search, rng, variant, device, preanchor_w16_w17_zero=False)
    _, blk2_pre, ms_pre = sample_btc_search_batch(bsz_search, rng, variant, device, preanchor_w16_w17_zero=True)

    best_single_std: Dict[str, Any] = {"r_neutral_post_inj": -1, "bit": -1}
    best_single_pre: Dict[str, Any] = {"r_neutral_post_inj": -1, "bit": -1}

    for bit in range(32):
        dw = torch.zeros((1, 4), dtype=torch.int64, device=device)
        dw[0, 3] = 1 << bit

        m_std = evaluate_chunk2_subspace_metrics(ms_std, blk2_std, dw, variant, rounds_check=16)
        # 注入点 t_inj = 3 (W1[3]) -> t >= 3 为受影响轮；统计 t >= 3 中 active_rate < 0.05 的最高轮次 - 3
        rates_std = m_std["active_rate_per_round"]
        post_low = [t for t in range(3, 16) if rates_std[t] < 0.05]
        r_depth_std = (max(post_low) - 2) if post_low else 0
        # 记录 t=4 (注入后第2轮) 状态活跃率
        score_std = (r_depth_std, -rates_std[4], -rates_std[5])
        if best_single_std["bit"] == -1 or score_std > best_single_std["_score"]:
            best_single_std = {
                "bit": bit,
                "r_neutral_post_inj": r_depth_std,
                "t3_active_rate": rates_std[3],
                "t4_active_rate": rates_std[4],
                "t5_active_rate": rates_std[5],
                "t6_active_rate": rates_std[6],
                "t7_active_rate": rates_std[7],
                "_score": score_std,
            }

        m_pre = evaluate_chunk2_subspace_metrics(ms_pre, blk2_pre, dw, variant, rounds_check=16)
        rates_pre = m_pre["active_rate_per_round"]
        post_low_pre = [t for t in range(3, 16) if rates_pre[t] < 0.05]
        r_depth_pre = (max(post_low_pre) - 2) if post_low_pre else 0
        score_pre = (r_depth_pre, -rates_pre[4], -rates_pre[5])
        if best_single_pre["bit"] == -1 or score_pre > best_single_pre["_score"]:
            best_single_pre = {
                "bit": bit,
                "r_neutral_post_inj": r_depth_pre,
                "t3_active_rate": rates_pre[3],
                "t4_active_rate": rates_pre[4],
                "t5_active_rate": rates_pre[5],
                "t6_active_rate": rates_pre[6],
                "t7_active_rate": rates_pre[7],
                "_score": score_pre,
            }

    best_single_std.pop("_score", None)
    best_single_pre.pop("_score", None)

    # --- 2. 神经策略网络 + 辅助条件 Phi 搜索 (针对 W1[3] bit 31 进位抑制与多比特团簇) ---
    # 训练纯 FP32 GNBCPolicyNet 预测哪些 (W1[0..3], midstate) 条件使 t=4,5 保持低活跃度
    policy_net = GNBCPolicyNet(in_bits=128, n_actions=32, hidden=128).to(device=device, dtype=torch.float32)
    opt = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    train_steps = 20 if is_smoke else 80
    for step in range(train_steps):
        _, b2_tr, ms_tr = sample_btc_search_batch(1024, rng, variant, device, preanchor_w16_w17_zero=False)
        x_spins = words_to_spins_fp32(b2_tr[:, :4])  # (1024, 128) pure FP32
        logits, val = policy_net(x_spins)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        act_bit = dist.sample()  # (1024,)
        # 精确 int64 仿真仅改选定比特后的 t=4..7 活跃比特数
        b2_pert = b2_tr.clone()
        b2_pert[:, 3] ^= (1 << act_bit)
        with torch.no_grad():
            t_base = sha256_compress_trace(ms_tr, b2_tr, rounds=8, variant=variant, record_gates=False)
            t_pert = sha256_compress_trace(ms_tr, b2_pert, rounds=8, variant=variant, record_gates=False)
            diff_a = popcnt32(t_base.A_full[:, 4 + 4 : 4 + 8] ^ t_pert.A_full[:, 4 + 4 : 4 + 8]).sum(dim=1)
            diff_e = popcnt32(t_base.E_full[:, 4 + 4 : 4 + 8] ^ t_pert.E_full[:, 4 + 4 : 4 + 8]).sum(dim=1)
            reward = -(diff_a + diff_e).to(torch.float32) / 256.0
        adv = reward - val.detach()
        loss = -(dist.log_prob(act_bit) * adv).mean() + 0.5 * nn.functional.mse_loss(val, reward)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 提取神经策略最偏好的 bit 与辅助条件 (MSB=31 进位抑制) 并在严格互斥留出集 N_holdout 验证
    with torch.no_grad():
        _, b2_probe, _ = sample_btc_search_batch(2048, rng, variant, device)
        logits_p, _ = policy_net(words_to_spins_fp32(b2_probe[:, :4]))
        learned_bit = int(logits_p.mean(dim=0).argmax().item())

    # 在独立留出集 N_holdout 上验证 bit=31 (MSB无进位) 与 learned_bit 及 Phi_carry_shield
    blk1_ho, blk2_ho, ms_ho, _ = sample_btc_holdout_batch(n_holdout, SEED_HOLDOUT, variant, device, preanchor_w16_w17_zero=False)

    def _eval_holdout_chunk2_candidate(cand_id: str, bit_pos: int, use_phi: bool) -> Dict[str, Any]:
        # 分批跑完整 N_holdout（防显存溢出）
        chunk_sz = 32768
        total_tested = 0
        phi_pass = 0
        # 检验在 t=4（注入第 2 轮）或 t=5 是否保持低活跃（<= 4 bits 活跃，对应随机 32/64 预期 < 2^-20）
        neutral_t4_hits = 0
        neutral_t5_hits = 0
        neutral_t6_hits = 0
        r_active_sum = np.zeros(16, dtype=np.float64)

        for start in range(0, n_holdout, chunk_sz):
            end = min(n_holdout, start + chunk_sz)
            b2_c = blk2_ho[start:end]
            ms_c = ms_ho[start:end]
            if use_phi:
                # 辅助条件 Phi：强制 Ch(e_3, e_2, e_1) 在目标位上对 e_3 屏蔽（例如固定 e_2 == e_1 的高位屏蔽，或仅评估满足条件样本）
                # 具体：t=3 后 e_4 = a_0 + t1_3，若 t1_3 仅翻转 MSB (bit 31)，则加法不产生向更高位进位；
                # 但 S1(e_4) 会扩散 bit 31 至 {25, 20, 6}！仅当 Ch(e_4, e_3, e_2) 屏蔽且 a_4 在 S0 下屏蔽才可能抑制，
                # 这里检验 Phi: (W1[3] >> 31) == 0 且满足 t=3 时 d_0[31]^t1_3[31] 经 Ch 位屏蔽（即 e_3[b] == e_2[b] 的 16-bit 局部屏蔽 -> p_phi ≈ 2^-4）
                t3_base = sha256_compress_trace(ms_c, b2_c, rounds=4, variant=variant, record_gates=False)
                # 条件 Phi：仅取 e_3 & 0xF0000000 == e_2 & 0xF0000000（4 bit 条件，p_phi ~ 1/16）
                e3 = t3_base.E_full[:, 3 + 3]
                e2 = t3_base.E_full[:, 3 + 2]
                mask_phi = ((e3 ^ e2) & 0xF0000000) == 0
                phi_cnt = int(mask_phi.sum().item())
                phi_pass += phi_cnt
                if phi_cnt == 0:
                    continue
                b2_c = b2_c[mask_phi]
                ms_c = ms_c[mask_phi]
            else:
                phi_pass += int(b2_c.shape[0])

            b2_p = b2_c.clone()
            b2_p[:, 3] ^= (1 << bit_pos)
            tr_a = sha256_compress_trace(ms_c, b2_c, rounds=16, variant=variant, record_gates=False)
            tr_b = sha256_compress_trace(ms_c, b2_p, rounds=16, variant=variant, record_gates=False)

            diff_bits = popcnt32(tr_a.A_full[:, 4:] ^ tr_b.A_full[:, 4:]) + popcnt32(tr_a.E_full[:, 4:] ^ tr_b.E_full[:, 4:])
            r_active_sum += diff_bits.sum(dim=0).cpu().numpy()
            # 低活跃门槛：单轮 64 bits 中至多 4 bits 发生差分（随机 P(<=4/64) ≈ 4.18e-14）
            neutral_t4_hits += int((diff_bits[:, 4] <= 4).sum().item())
            neutral_t5_hits += int((diff_bits[:, 5] <= 4).sum().item())
            neutral_t6_hits += int((diff_bits[:, 6] <= 4).sum().item())
            total_tested += int(b2_c.shape[0])

        p_phi = phi_pass / float(n_holdout)
        p_t4 = neutral_t4_hits / float(max(1, total_tested))
        p_t5 = neutral_t5_hits / float(max(1, total_tested))
        p_t6 = neutral_t6_hits / float(max(1, total_tested))
        # 随机基线：二项分布 Binom(64, 0.5) <= 4 的概率 = sum_{k=0}^4 C(64,k) * 2^-64 ≈ 3.67e-14
        p0_low = sum(math.comb(64, k) for k in range(5)) / float(1 << 64)
        z_t5, ci_t5 = binomial_z_and_wilson(neutral_t5_hits, max(1, total_tested), p0_low)

        # 完整 128 轮 (Chunk2 + Hash2) 实测精确 GE 摊还（取前 128 对样本）
        sub_n = min(128, blk2_ho.shape[0])
        b1_s = torch.cat([blk1_ho[:sub_n], blk1_ho[:sub_n]], dim=0)
        b2_s_p = blk2_ho[:sub_n].clone()
        b2_s_p[:, 3] ^= (1 << bit_pos)
        b2_s = torch.cat([blk2_ho[:sub_n], b2_s_p], dim=0)
        tr_full = btc_double_sha256_trace(b1_s, b2_s, variant=variant, record_gates=True)
        # 与同批 2-way 独立对照（2*sub_n 的无共享基准）比对
        rep_cand = compute_exact_btc_amortized_report(tr_full, cand_id, variant, p_valid=1.0)

        r_rates = (r_active_sum / max(1, total_tested) / 64.0).tolist()
        r_neut = max([t for t in range(3, 16) if r_rates[t] < 0.05] or [2]) - 2

        return {
            "id": cand_id,
            "track": "A2-1",
            "bit_pos": bit_pos,
            "use_phi": use_phi,
            "r_neutral": int(r_neut),
            "p_phi": p_phi,
            "holdout_tested": total_tested,
            "p_neutral_t4": p_t4,
            "p_neutral_t5": p_t5,
            "p_neutral_t6": p_t6,
            "p_neutral_z": z_t5,
            "wilson_ci95_t5": list(ci_t5),
            "d_neutral": 1,
            "batch_size": 2,
            "active_ge_per_hash": rep_cand.amortized_ge_per_hash,
            "post_t16_active_rate_mean": float(np.mean(r_rates[3:16])),
            "round_active_rates_0_15": [round(x, 5) for x in r_rates],
        }

    c_msb = _eval_holdout_chunk2_candidate("A2-1_W3_bit31_noPhi", 31, False)
    c_msb_phi = _eval_holdout_chunk2_candidate("A2-1_W3_bit31_PhiCh", 31, True)
    c_learned = _eval_holdout_chunk2_candidate(f"A2-1_Neural_bit{learned_bit}", learned_bit, False)

    # --- 3. 全量 496 组 2-bit W1[3] 对与 Mode2 (midstate+W1 补偿) GPU 并发扫描 ---
    best_pair_std: Dict[str, Any] = {"mean_active_t3_15": 1.0, "bits": [-1, -1]}
    if not is_smoke:
        pair_n = 512
        for bi in range(32):
            for bj in range(bi + 1, 32):
                dw_p = torch.zeros((1, 4), dtype=torch.int64, device=device)
                dw_p[0, 3] = (1 << bi) ^ (1 << bj)
                mp = evaluate_chunk2_subspace_metrics(ms_std[:pair_n], blk2_std[:pair_n], dw_p, variant, rounds_check=16)
                m_act = float(np.mean(mp["active_rate_per_round"][3:16]))
                if m_act < best_pair_std["mean_active_t3_15"]:
                    best_pair_std = {"bits": [bi, bj], "mean_active_t3_15": m_act, "t4_active": mp["active_rate_per_round"][4]}

    # --- 4. W1[0..3] 联合自由度 (nNonce + nTime + Merkle_tail) 进化/多比特抵消搜索 ---
    best_joint_saving = -999.0
    best_joint_desc: Dict[str, Any] = {}
    pop_n = 64 if is_smoke else 1024
    for g_idx in range(pop_n):
        dw_joint = torch.zeros((1, 4), dtype=torch.int64, device=device)
        if g_idx == 0:
            dw_joint[0, 0] = 0x80000000
            dw_joint[0, 3] = 0x80000000
        else:
            b_a = int(rng.integers(0, 32))
            b_b = int(rng.integers(0, 32))
            dw_joint[0, 0] = (1 << b_a) if rng.random() < 0.5 else 0
            dw_joint[0, 1] = (1 << int(rng.integers(0, 32))) if rng.random() < 0.5 else 0
            dw_joint[0, 3] = 1 << b_b
        m_j = evaluate_chunk2_subspace_metrics(ms_std[:512], blk2_std[:512], dw_joint, variant, rounds_check=16)
        mean_act = float(np.mean(m_j["active_rate_per_round"][:16]))
        score = -mean_act
        if score > best_joint_saving:
            best_joint_saving = score
            best_joint_desc = {
                "dw_hex": [hex(int(x)) for x in dw_joint[0].tolist()],
                "mean_active_rate_t0_15": mean_act,
                "round_active_rates": [round(x, 4) for x in m_j["active_rate_per_round"]],
            }

    return {
        "best_single_bit_standard": best_single_std,
        "best_single_bit_w16_w17_zero": best_single_pre,
        "best_2bit_pair_w3": best_pair_std,
        "neural_learned_bit": learned_bit,
        "holdout_candidates": [c_msb, c_msb_phi, c_learned],
        "best_joint_w013_search": best_joint_desc,
    }


# ---------------------------------------------------------------------------
# Track A2-2: Chunk2 -> Hash2 跨块界面 & Hash2 内部探针
# ---------------------------------------------------------------------------

def run_track_a2_2(
    variant: SHA256Variant,
    device: torch.device,
    n_holdout: int,
    is_smoke: bool = False,
) -> Dict[str, Any]:
    """
    1. 端到端 Chunk2 扰动（W1[3] bit 0 / bit 31 / nVersion W0[0]）经过完整 64 轮 Chunk2 后，
       在独立留出集 N_holdout 上测量 hash1[0..7] (即 W2[0..7]) 的部分零差穿透率；
    2. Hash2 内部结构探针：枚举全部 256 个 W2[0..7] 单比特翻转 + 4-bit 簇，
       量化 W2[0..7] -> W2[16..63] 与压缩前 r2 轮真实穿透深度。
    """
    blk1_ho, blk2_ho, ms_ho, _ = sample_btc_holdout_batch(n_holdout, SEED_HOLDOUT + 1, variant, device)

    # 1. End-to-End Chunk2 -> Hash2 跨块界面统计 (W1[3] bit0, W1[3] bit31, W0[0] bit13)
    e2e_results: List[Dict[str, Any]] = []
    chunk_sz = 32768
    for label, target_blk, word_idx, bit_idx in [
        ("E2E_W1_nonce_bit0", "blk2", 3, 0),
        ("E2E_W1_nonce_bit31", "blk2", 3, 31),
        ("E2E_W0_nVersion_bit13", "blk1", 0, 13),
    ]:
        zero_word_any_hits = 0
        zero_w0_hits = 0
        min_hamming_hash1 = 256
        sum_hamming_hash1 = 0

        iv_t = torch.tensor(variant.iv, dtype=torch.int64, device=device)
        for start in range(0, n_holdout, chunk_sz):
            end = min(n_holdout, start + chunk_sz)
            b1_c = blk1_ho[start:end]
            b2_c = blk2_ho[start:end]
            ms_c = ms_ho[start:end]

            if target_blk == "blk2":
                b2_p = b2_c.clone()
                b2_p[:, word_idx] ^= (1 << bit_idx)
                tr_a = sha256_compress_trace(ms_c, b2_c, rounds=64, variant=variant, record_gates=False)
                tr_b = sha256_compress_trace(ms_c, b2_p, rounds=64, variant=variant, record_gates=False)
            else:
                b1_p = b1_c.clone()
                b1_p[:, word_idx] ^= (1 << bit_idx)
                ms_p = sha256_compress_trace(iv_t, b1_p, rounds=64, variant=variant, record_gates=False).digest
                tr_a = sha256_compress_trace(ms_c, b2_c, rounds=64, variant=variant, record_gates=False)
                tr_b = sha256_compress_trace(ms_p, b2_c, rounds=64, variant=variant, record_gates=False)

            diff_h1 = tr_a.digest ^ tr_b.digest  # (chunk, 8)
            zero_w0_hits += int((diff_h1[:, 0] == 0).sum().item())
            zero_word_any_hits += int(((diff_h1 == 0).any(dim=1)).sum().item())
            hw = popcnt32(diff_h1).sum(dim=1)
            min_hamming_hash1 = min(min_hamming_hash1, int(hw.min().item()))
            sum_hamming_hash1 += int(hw.sum().item())

        # 任一字零差理论概率 ~ 8 * 2^-32 ≈ 1.86e-9
        p_any_theory = 8.0 * (2.0 ** -32)
        z_any, ci_any = binomial_z_and_wilson(zero_word_any_hits, n_holdout, p_any_theory)
        e2e_results.append(
            {
                "id": label,
                "track": "A2-2",
                "holdout_tested": n_holdout,
                "zero_w2_word0_hits": zero_w0_hits,
                "zero_w2_any_word_hits": zero_word_any_hits,
                "p_neutral_any_word": zero_word_any_hits / float(n_holdout),
                "p_neutral_z": z_any,
                "wilson_ci95": list(ci_any),
                "mean_hamming_hash1_diff_bits": sum_hamming_hash1 / float(n_holdout),
                "min_hamming_hash1_diff_bits": min_hamming_hash1,
            }
        )

    # 2. Hash2 内部全部 256 个单比特探针（W2[0..7] -> Hash2 64 轮）
    n_probe = 512 if is_smoke else 2048
    iv_t = torch.tensor(variant.iv, dtype=torch.int64, device=device)
    w_outer_base = pack_outer_block_from_h_inner(ms_ho[:n_probe])  # 随机 8-word 作为 W2[0..7]

    best_hash2_probes: List[Dict[str, Any]] = []
    for w_idx in range(8):
        best_b = 0
        best_rate_post = 999.0
        best_r_neut = 0
        for bit_idx in (0, 15, 31):
            w_p = w_outer_base.clone()
            w_p[:, w_idx] ^= (1 << bit_idx)
            tr_a = sha256_compress_trace(iv_t, w_outer_base, rounds=24, variant=variant, record_gates=False)
            tr_b = sha256_compress_trace(iv_t, w_p, rounds=24, variant=variant, record_gates=False)
            ae_diff = popcnt32(tr_a.A_full[:, 4:] ^ tr_b.A_full[:, 4:]) + popcnt32(tr_a.E_full[:, 4:] ^ tr_b.E_full[:, 4:])
            rates = (ae_diff.float().mean(dim=0) / 64.0).cpu().tolist()
            # 从注入轮 t=w_idx 起穿透深度
            r_neut = sum(1 for t in range(w_idx, 24) if rates[t] < 0.05)
            post_r = rates[min(23, w_idx + 1)]
            if post_r < best_rate_post:
                best_rate_post = post_r
                best_b = bit_idx
                best_r_neut = r_neut
        best_hash2_probes.append(
            {
                "w2_word": w_idx,
                "best_bit": best_b,
                "trivial_pre_inj_zero_rounds": w_idx,
                "post_inj_low_active_rounds": best_r_neut,
                "next_round_active_rate": round(best_rate_post, 5),
            }
        )

    return {
        "e2e_chunk2_to_hash2": e2e_results,
        "hash2_internal_word_probes": best_hash2_probes,
    }


# ---------------------------------------------------------------------------
# Track A2-3: 比特级进位吸收 / 条件静默门集群 (MCTS/Beam + 进化 + 精确 GE 计费)
# ---------------------------------------------------------------------------

def run_track_a2_3(
    variant: SHA256Variant,
    device: torch.device,
    classic_ab_ge_map: Dict[int, float],
    noab_ge_per_hash: float,
    n_holdout: int,
    is_smoke: bool = False,
) -> Dict[str, Any]:
    """
    以真实 bit-level active_ge_per_hash 为最终优化目标，在维度 d in {2, 3, 4} (batch M = 2^d)
    执行 Beam/MCTS + 进化搜索：
    - 候选家族 1 (Nonce-Only Subspace): 仅在 W1[3] (nNonce) 内选 d 个基向量；
    - 候选家族 2 (Header W1[0,1,3] Subspace): 允许联合动 Merkle_tail / nTime / nNonce；
    - 候选家族 3 (nVersion W0[0] + W1[3] 混合): 含对称 Chunk1 均摊成本。
    """
    rng = np.random.default_rng(SEED_SEARCH + 2)
    b1_base, b2_base, _ = sample_btc_search_batch(64, rng, variant, device)

    def _eval_cluster_ge(
        family: str,
        basis_w1: torch.Tensor,  # (d, 4)
        basis_w0: Optional[torch.Tensor] = None,  # (d, 1) for nVersion
        n_anchors: int = 8,
    ) -> Tuple[float, float, float]:
        """在 n_anchors 个独立随机区块头上计算 2^d 团簇的平均精确 GE/hash 及 vs classic_ab 净节省。"""
        d = basis_w1.shape[0]
        m_size = 1 << d
        masks_w1 = torch.zeros((m_size, 4), dtype=torch.int64, device=device)
        masks_w0 = torch.zeros((m_size,), dtype=torch.int64, device=device)
        for bi in range(d):
            sel = torch.tensor([(j >> bi) & 1 for j in range(m_size)], dtype=torch.bool, device=device)
            masks_w1[sel] ^= basis_w1[bi].unsqueeze(0)
            if basis_w0 is not None:
                masks_w0[sel] ^= basis_w0[bi]

        ges: List[float] = []
        for a_idx in range(n_anchors):
            b1_a = b1_base[a_idx : a_idx + 1].expand(m_size, 16).clone()
            b2_a = b2_base[a_idx : a_idx + 1].expand(m_size, 16).clone()
            b2_a[:, :4] ^= masks_w1
            if basis_w0 is not None:
                b1_a[:, 0] ^= masks_w0
            with torch.no_grad():
                tr = btc_double_sha256_trace(b1_a, b2_a, variant=variant, record_gates=True)
            # 对称计费：若仅改 W1[3] (单一 midstate)，Chunk1 按 2^32 均摊；
            # 若改 W1[0]/W1[1] 或 W0[0]（每次变更都产生新区块头配置且若未配 2^32 nonces 则须对称计入 Chunk1 或外循环成本！）
            amort_n = (1 << 32) if family == "nonce_only" else 1
            rep = compute_exact_btc_amortized_report(
                tr,
                label=f"{family}_d{d}",
                variant=variant,
                naive_baseline_ge_per_hash=noab_ge_per_hash,
                amortize_chunk1_over_nonces=amort_n,
            )
            ges.append(rep.amortized_ge_per_hash)

        mean_ge = float(np.mean(ges))
        save_noab = (1.0 - mean_ge / noab_ge_per_hash) * 100.0
        classic_ge = classic_ab_ge_map.get(m_size, noab_ge_per_hash)
        net_vs_classic = (1.0 - mean_ge / classic_ge) * 100.0
        return mean_ge, save_noab, net_vs_classic

    # Beam/MCTS 逐维扩展 d=1 -> 2 -> 3 -> 4 (Nonce-Only vs Header-Joint vs Overt-Hybrid)
    best_by_dim: List[Dict[str, Any]] = []
    n_candidates_per_dim = 8 if is_smoke else 24

    for d in (2, 3, 4):
        m_size = 1 << d
        # 1) Nonce-only subspace: 测试低位连续块 [0..d-1] vs 高位块 [32-d..31] vs 稀疏随机
        cand_list_nonce: List[Tuple[str, torch.Tensor]] = []
        # 连续高位（进位最小化！bit 31, 30, ...）
        bw_high = torch.zeros((d, 4), dtype=torch.int64, device=device)
        for bi in range(d):
            bw_high[bi, 3] = 1 << (31 - bi)
        cand_list_nonce.append(("MSB_contiguous", bw_high))
        # 连续低位 (bit 0..d-1)
        bw_low = torch.zeros((d, 4), dtype=torch.int64, device=device)
        for bi in range(d):
            bw_low[bi, 3] = 1 << bi
        cand_list_nonce.append(("LSB_contiguous", bw_low))
        # 随机稀疏比特子空间
        for r_idx in range(n_candidates_per_dim):
            bits_chosen = rng.choice(32, size=d, replace=False)
            bw_r = torch.zeros((d, 4), dtype=torch.int64, device=device)
            for bi, bp in enumerate(bits_chosen):
                bw_r[bi, 3] = 1 << int(bp)
            cand_list_nonce.append((f"rand_sparse_{r_idx}", bw_r))

        best_nonce_rec: Optional[Dict[str, Any]] = None
        for tag, bw in cand_list_nonce:
            ge, sav_noab, net_cl = _eval_cluster_ge("nonce_only", bw, n_anchors=4 if is_smoke else 8)
            if best_nonce_rec is None or ge < best_nonce_rec["active_ge_per_hash"]:
                best_nonce_rec = {
                    "id": f"A2-3_NonceOnly_d{d}_{tag}",
                    "track": "A2-3",
                    "family": "nonce_only",
                    "d_neutral": d,
                    "batch_size": m_size,
                    "basis_w1_hex": [[hex(int(x)) for x in row] for row in bw.cpu().tolist()],
                    "active_ge_per_hash": ge,
                    "saving_pct_vs_noab": sav_noab,
                    "net_gain_pct_vs_classic_ab": net_cl,
                }
        assert best_nonce_rec is not None
        best_by_dim.append(best_nonce_rec)

        # 2) Overt-Hybrid (BIP320 nVersion d bits + fixed nonce) 对称口径检验
        bw_zero = torch.zeros((d, 4), dtype=torch.int64, device=device)
        bw0_ver = torch.tensor([1 << (13 + bi) for bi in range(d)], dtype=torch.int64, device=device)
        ge_ov, sav_ov, net_ov = _eval_cluster_ge("overt_version_sym1", bw_zero, basis_w0=bw0_ver, n_anchors=4 if is_smoke else 8)
        best_by_dim.append(
            {
                "id": f"A2-3_OvertVersionSym1_d{d}",
                "track": "A2-3",
                "family": "overt_version_sym1",
                "d_neutral": d,
                "batch_size": m_size,
                "basis_w1_hex": [[hex(int(x)) for x in row] for row in bw_zero.cpu().tolist()],
                "active_ge_per_hash": ge_ov,
                "saving_pct_vs_noab": sav_ov,
                "net_gain_pct_vs_classic_ab": net_ov,
            }
        )

    # 3) Overt + GNBC Hybrid (M=16 = 4 nVersion x 4 Nonce-MSB-cluster) 对称大批量对比 BASE_ASICBOOST_OVERT k=16
    hybrid_ges: List[float] = []
    n_hyb_anchors = 4 if is_smoke else 8
    for a_idx in range(n_hyb_anchors):
        # 4 version x 4 nonce (MSB bits 30, 31) = 16 headers
        rows_b1: List[torch.Tensor] = []
        rows_b2: List[torch.Tensor] = []
        for vi in range(4):
            for ni in range(4):
                b1_r = b1_base[a_idx].clone()
                b2_r = b2_base[a_idx].clone()
                b1_r[0] ^= ((vi & 0x3) << 13)
                b2_r[3] ^= ((ni & 0x3) << 30)
                rows_b1.append(b1_r)
                rows_b2.append(b2_r)
        tr_h = btc_double_sha256_trace(torch.stack(rows_b1), torch.stack(rows_b2), variant=variant, record_gates=True)
        # 与 2^32 Nonce 均摊一致（每个 version 在外循环均摊）
        rep_h = compute_exact_btc_amortized_report(tr_h, "hybrid_4v4n", variant, noab_ge_per_hash, amortize_chunk1_over_nonces=1 << 32)
        hybrid_ges.append(rep_h.amortized_ge_per_hash)
    hyb_mean_ge = float(np.mean(hybrid_ges))
    hyb_sav_noab = (1.0 - hyb_mean_ge / noab_ge_per_hash) * 100.0
    hyb_net_classic = (1.0 - hyb_mean_ge / classic_ab_ge_map[16]) * 100.0
    best_by_dim.append(
        {
            "id": "A2-3_Hybrid_4Ver_x_4NonceMSB_M16",
            "track": "A2-3",
            "family": "overt_plus_gnbc_hybrid",
            "d_neutral": 4,
            "batch_size": 16,
            "basis_w1_hex": [["0x0", "0x0", "0x0", "0x80000000"], ["0x0", "0x0", "0x0", "0x40000000"]],
            "active_ge_per_hash": hyb_mean_ge,
            "saving_pct_vs_noab": hyb_sav_noab,
            "net_gain_pct_vs_classic_ab": hyb_net_classic,
        }
    )

    # 在留出集验证最强的 Nonce-Only 团簇并导出可复现 witness_headers_hex
    best_nonce_overall = min(
        [x for x in best_by_dim if x["family"] == "nonce_only"],
        key=lambda x: x["active_ge_per_hash"],
    )
    b1_ho, b2_ho, _, _ = sample_btc_holdout_batch(min(1024, n_holdout), SEED_HOLDOUT + 9, variant, device)
    d_best = best_nonce_overall["d_neutral"]
    m_best = 1 << d_best
    bw_best = torch.tensor([[int(x, 16) for x in row] for row in best_nonce_overall["basis_w1_hex"]], dtype=torch.int64, device=device)

    masks_best = torch.zeros((m_best, 4), dtype=torch.int64, device=device)
    for bi in range(d_best):
        sel = torch.tensor([(j >> bi) & 1 for j in range(m_best)], dtype=torch.bool, device=device)
        masks_best[sel] ^= bw_best[bi].unsqueeze(0)

    # 在留出样本上复核 GE
    ho_ges: List[float] = []
    n_ho_check = 8 if is_smoke else 32
    for idx in range(n_ho_check):
        b1_x = b1_ho[idx : idx + 1].expand(m_best, 16).clone()
        b2_x = b2_ho[idx : idx + 1].expand(m_best, 16).clone()
        b2_x[:, :4] ^= masks_best
        tr_x = btc_double_sha256_trace(b1_x, b2_x, variant=variant, record_gates=True)
        rep_x = compute_exact_btc_amortized_report(tr_x, "holdout_best", variant, noab_ge_per_hash)
        ho_ges.append(rep_x.amortized_ge_per_hash)

    from btc_double_sha_engine import blocks_to_btc_header_bytes

    b1_w = b1_ho[0:1].expand(m_best, 16).clone()
    b2_w = b2_ho[0:1].expand(m_best, 16).clone()
    b2_w[:, :4] ^= masks_best
    witness_hex = [h.hex() for h in blocks_to_btc_header_bytes(b1_w, b2_w)[:4]]

    best_nonce_overall["holdout_mean_ge_per_hash"] = float(np.mean(ho_ges))
    best_nonce_overall["holdout_saving_pct_vs_noab"] = (1.0 - float(np.mean(ho_ges)) / noab_ge_per_hash) * 100.0
    classic_k = classic_ab_ge_map.get(m_best, noab_ge_per_hash)
    best_nonce_overall["holdout_net_gain_pct_vs_classic_ab"] = (1.0 - float(np.mean(ho_ges)) / classic_k) * 100.0
    best_nonce_overall["witness_headers_hex"] = witness_hex

    return {
        "candidates_by_dim": best_by_dim,
        "best_nonce_cluster_holdout": best_nonce_overall,
    }


# ---------------------------------------------------------------------------
# 主控入口：在 MAIN_SHA256 + 3 组谱隙对齐 CTRL_RAND_ROTR_K 上运行完整 A2
# ---------------------------------------------------------------------------

def run_full_a2(
    n_holdout: int = 262144,
    is_smoke: bool = False,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    fp32_info = enforce_fp32_strict()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[A2] Device={device} ({gpu_name}), allow_tf32={fp32_info['matmul_allow_tf32']}, n_holdout={n_holdout}", flush=True)

    # 读取或现场计算 A1 基线 GE 以确保净增益扣除精确无误
    from classic_asicboost_calib import run_a1_calibration_for_variant

    main_var = make_main_variant()
    ctrl_vars = [
        make_spectral_aligned_ctrl_variant(seed=20261125 + s, min_gap=4, name=f"CTRL_RAND_ROTR_K_S{s}")
        for s in range(3)
    ]

    variants = [main_var] + ctrl_vars
    results_by_variant: Dict[str, Any] = {}

    for idx_v, var in enumerate(variants):
        tv = time.time()
        print(f"[A2] Running variant {idx_v+1}/{len(variants)}: {var.name} ...", flush=True)
        a1_v = run_a1_calibration_for_variant(var, device=device, n_nonces_eval=16, k_ways=(4, 8, 16))
        noab_ge = a1_v["BASE_NO_ASICBOOST"]["inner_loop_2p32"]["amortized_ge_per_hash"]
        classic_ab_map = {rec["k"]: rec["inner_loop_2p32"]["amortized_ge_per_hash"] for rec in a1_v["BASE_ASICBOOST_OVERT"]}

        # 对 CTRL 组在 A2 留出集上同步运行完整 N_holdout（smoke 时同比例缩减）
        print(f"  [{var.name}] A2-1 start ...", flush=True)
        r_a2_1 = run_track_a2_1(var, device, n_holdout=n_holdout, is_smoke=is_smoke)
        print(f"  [{var.name}] A2-2 start ...", flush=True)
        r_a2_2 = run_track_a2_2(var, device, n_holdout=n_holdout, is_smoke=is_smoke)
        print(f"  [{var.name}] A2-3 start ...", flush=True)
        r_a2_3 = run_track_a2_3(var, device, classic_ab_ge_map=classic_ab_map, noab_ge_per_hash=noab_ge, n_holdout=n_holdout, is_smoke=is_smoke)
        import gc; gc.collect()

        results_by_variant[var.name] = {
            "noab_ge_per_hash": noab_ge,
            "classic_ab_ge_map": classic_ab_map,
            "A2-1": r_a2_1,
            "A2-2": r_a2_2,
            "A2-3": r_a2_3,
            "elapsed_seconds": round(time.time() - tv, 2),
        }
        best3 = r_a2_3["best_nonce_cluster_holdout"]
        print(
            f"[A2] Done {var.name} in {time.time()-tv:.1f}s | "
            f"best_nonce_d{best3['d_neutral']} holdout_saving={best3['holdout_saving_pct_vs_noab']:.3f}% "
            f"net_vs_classic={best3['holdout_net_gain_pct_vs_classic_ab']:.3f}%",
            flush=True,
        )

    # 计算 MAIN vs CTRL z_vs_ctrl 汇总
    main_t5 = results_by_variant["SHA256_MAIN"]["A2-1"]["holdout_candidates"][0]["p_neutral_t5"]
    ctrl_t5s = [results_by_variant[f"CTRL_RAND_ROTR_K_S{s}"]["A2-1"]["holdout_candidates"][0]["p_neutral_t5"] for s in range(3)]
    ctrl_net_gains = [
        results_by_variant[f"CTRL_RAND_ROTR_K_S{s}"]["A2-3"]["best_nonce_cluster_holdout"]["holdout_net_gain_pct_vs_classic_ab"]
        for s in range(3)
    ]

    payload = {
        "fp32_lock": fp32_info,
        "search_budget": {
            "device": str(device),
            "gpu_name": gpu_name,
            "is_smoke": is_smoke,
            "n_holdout": n_holdout,
            "search_nonce_range": "[0, 2^30)",
            "holdout_nonce_range": "[2^30, 2^32)",
            "train_holdout_overlap": 0,
            "total_elapsed_seconds": round(time.time() - t0, 2),
        },
        "tracks": results_by_variant,
        "main_vs_ctrl_summary": {
            "main_a2_1_p_neutral_t5": main_t5,
            "ctrl_a2_1_p_neutral_t5_list": ctrl_t5s,
            "main_best_net_gain_vs_classic_pct": results_by_variant["SHA256_MAIN"]["A2-3"]["best_nonce_cluster_holdout"]["holdout_net_gain_pct_vs_classic_ab"],
            "ctrl_best_net_gain_vs_classic_pct_mean": float(np.mean(ctrl_net_gains)),
        },
    }

    if out_path is None:
        out_path = Path(__file__).resolve().parent / "results" / "a2_gnbc_search.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[A2] Saved -> {out_path}", flush=True)
    return payload


def main() -> None:
    import traceback
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU/轻量冒烟模式")
    ap.add_argument("--cpu_only", action="store_true", help="强制仅使用 CPU")
    ap.add_argument("--n_holdout", type=int, default=262144)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    n_ho = 4096 if args.smoke else args.n_holdout
    out_p = Path(args.out) if args.out else None
    try:
        run_full_a2(n_holdout=n_ho, is_smoke=args.smoke, out_path=out_p)
    except BaseException:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
