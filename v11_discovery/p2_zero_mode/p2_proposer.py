"""P2「时空晶体等零面零模波包」3D 晶体零模提议器与去重波包采样器。

严格复用 v10_ouroboros64.ouroboros_net.model.Ouroboros64Net（纯 FP32），
并将扰动输出严格限制在活跃自由输入域 Omega_free(R, carrier) 内，保证 dW != 0 且无自抵消。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v10_ouroboros64.ouroboros_core.engine import enforce_fp32_determinism
from v10_ouroboros64.ouroboros_net.heads import PLANE_COUNT
from v10_ouroboros64.ouroboros_net.model import Ouroboros64Net

from .p2_core import ParametricCrystalEngine


class ZeroModeCrystalProposer(nn.Module):
    """
    基于 Ouroboros64Net 3D 时空晶体主干的等零面多比特波包提议器。

    工作流：
      1) 将基底晶体 (W, S, C) 与虚拟靶向边界 H_target^* 输入 Ouroboros64Net（纯 FP32）；
      2) 提取 plane=0 (W) 且 t < w_free 的隐状态、flip_logits 与 wolff_head 位点键向量；
      3) 先预测波包 Hamming 重 m ∈ {1..max_bits}，再经条件自回归选取 m 个互异自由比特组成 Delta_W。
    """

    def __init__(
        self,
        d_node: int = 64,
        n_layers: int = 4,
        bond_dim: int = 4,
        d_key: int = 24,
        max_bits: int = 8,
    ) -> None:
        super().__init__()
        enforce_fp32_determinism()
        self.d_node = int(d_node)
        self.d_key = int(d_key)
        self.max_bits = int(max_bits)

        self.net = Ouroboros64Net(
            d_node=d_node,
            n_layers=n_layers,
            bond_dim=bond_dim,
            d_key=d_key,
            max_cluster_size=max(2, min(6, max_bits)),
            dropout=0.0,
            use_checkpoint=False,
        )

        # 全局汇总隐状态预测微扰规模 m ∈ {1..max_bits}
        self.size_head = nn.Sequential(
            nn.LayerNorm(d_node * 3),
            nn.Linear(d_node * 3, d_node),
            nn.SiLU(),
            nn.Linear(d_node, self.max_bits),
        )
        # 针对输入自由比特位的轻量细化偏置（叠加到 wolff_head.anchor_logits 上）
        self.w_site_refine = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, 1),
        )

    def encode_crystal_policy(
        self,
        H_target_star: torch.Tensor,
        W: torch.Tensor,
        S: torch.Tensor,
        C: torch.Tensor,
        rounds: int,
        w_free: int,
        crystal_engine: ParametricCrystalEngine,
    ) -> Dict[str, torch.Tensor]:
        """对批量基底晶体执行一次 Ouroboros64Net 3D 前向并导出自由输入域策略分布特征。"""
        out = self.net(
            H_target=H_target_star,
            W=W,
            S=S,
            C=C,
            rounds=rounds,
            propose_clusters=False,
            return_entropy=False,
            crystal_engine=crystal_engine,
        )
        h_final = out["H"]  # (B, R, 32, D)
        flip_logits = out["flip_logits"]  # (B, R, 19, 32)
        site_def = out["site_defect_signal"]  # (B, R, 19, 32)

        cache = self.net.wolff_head.encode_sites(h_final, site_def, flip_logits=flip_logits)
        bsz = W.shape[0]
        device = W.device

        # 仅切片 plane=0 (W) 且 t ∈ [0, w_free) 的合法自由位点索引
        t_idx = torch.arange(w_free, device=device, dtype=torch.long).unsqueeze(1)
        b_idx = torch.arange(32, device=device, dtype=torch.long).unsqueeze(0)
        w_flat_idx = ((t_idx * PLANE_COUNT) * 32 + b_idx).reshape(-1)  # (N_w,)

        w_keys = cache["flat_keys"][:, w_flat_idx, :]  # (B, N_w, d_key)
        base_anchor = cache["anchor_logits"][:, w_flat_idx]  # (B, N_w)
        refine_w = self.w_site_refine(h_final[:, :w_free, :, :]).reshape(bsz, w_free * 32)
        anchor_logits = base_anchor + refine_w
        partner_prior = flip_logits[:, :w_free, 0, :].reshape(bsz, w_free * 32).clamp(-8.0, 8.0)

        pooled = torch.cat(
            [
                h_final[:, 0, :, :].mean(dim=1),
                h_final[:, -1, :, :].mean(dim=1),
                h_final[:, :w_free, :, :].mean(dim=(1, 2)),
            ],
            dim=-1,
        )
        size_logits = self.size_head(pooled)  # (B, max_bits)

        return {
            "w_keys": w_keys,
            "anchor_logits": anchor_logits,
            "partner_prior": partner_prior,
            "size_logits": size_logits,
            "w_free": w_free,
        }

    def sample_from_policy(
        self,
        policy: Dict[str, torch.Tensor],
        num_samples: int,
        temperature: float = 1.0,
        include_greedy: bool = False,
        return_log_prob: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """从策略分布批量采样 (B, K, 16) int64 非零微扰 dW0 与联合对数概率。"""
        w_keys = policy["w_keys"]  # (B, N_w, d_key)
        anchor_logits = policy["anchor_logits"]  # (B, N_w)
        partner_prior = policy["partner_prior"]  # (B, N_w)
        size_logits = policy["size_logits"]  # (B, max_bits)
        w_free = int(policy["w_free"])

        bsz, n_w, d_key = w_keys.shape
        device = w_keys.device
        temp = max(0.1, float(temperature))
        scale = 1.0 / math.sqrt(float(d_key))
        max_m = min(self.max_bits, n_w)

        # 1) 采样每个候选的微扰位数 m ∈ {1..max_m}
        scaled_size = size_logits[:, :max_m] / temp
        logp_size = F.log_softmax(scaled_size, dim=-1)
        probs_size = torch.exp(logp_size)
        ent_size = -(probs_size * logp_size).sum(dim=-1)  # (B,)

        m_idx = torch.multinomial(probs_size, num_samples=num_samples, replacement=True)  # (B, K)
        if include_greedy and num_samples >= 1:
            m_idx[:, 0] = torch.argmax(scaled_size, dim=-1)
        target_m = m_idx + 1  # (B, K) in 1..max_m

        total_logp = torch.gather(logp_size, 1, m_idx) if return_log_prob else None

        # 2) 采样锚点比特 idx_0 ∈ [0, N_w)
        scaled_anchor = anchor_logits / temp
        logp_0 = F.log_softmax(scaled_anchor, dim=-1)
        probs_0 = torch.exp(logp_0)
        ent_0 = -(probs_0 * logp_0).sum(dim=-1)

        idx_0 = torch.multinomial(probs_0, num_samples=num_samples, replacement=True)  # (B, K)
        if include_greedy and num_samples >= 1:
            idx_0[:, 0] = torch.argmax(scaled_anchor, dim=-1)

        if total_logp is not None:
            total_logp = total_logp + torch.gather(logp_0, 1, idx_0)

        # 追踪已选位点防止自抵消，并记录 bit 位向量 (B, K, N_w)
        chosen_mask = torch.zeros((bsz, num_samples, n_w), dtype=torch.bool, device=device)
        chosen_mask = chosen_mask.scatter(2, idx_0.unsqueeze(-1), True)
        selected_bits = chosen_mask.to(torch.int64)

        idx_0_exp = idx_0.unsqueeze(-1).expand(bsz, num_samples, d_key)
        h_gru = torch.gather(w_keys, 1, idx_0_exp).reshape(bsz * num_samples, d_key)

        wolff = self.net.wolff_head
        prior_bias = 0.5 * wolff.flip_logit_weight * partner_prior.unsqueeze(1)

        # 3) 条件自回归补充剩余 (m - 1) 个互异位点
        max_step = int(target_m.max().item()) if num_samples > 0 else 1
        for step in range(1, max_step):
            active = target_m > step  # (B, K)
            if not torch.any(active):
                break
            q = wolff.query_proj(wolff.query_norm(h_gru)).reshape(bsz, num_samples, d_key)
            p_logits = (torch.einsum("bkd,bnd->bkn", q, w_keys) * scale + prior_bias) / temp
            p_logits = p_logits.masked_fill(chosen_mask, -1e9)
            logp_s = F.log_softmax(p_logits, dim=-1)
            probs_s = torch.exp(logp_s).reshape(bsz * num_samples, n_w)

            idx_s = torch.multinomial(probs_s, num_samples=1).reshape(bsz, num_samples)
            if include_greedy and num_samples >= 1:
                idx_s[:, 0] = torch.argmax(p_logits[:, 0, :], dim=-1)

            if total_logp is not None:
                picked = torch.gather(logp_s, 2, idx_s.unsqueeze(-1)).squeeze(-1)
                total_logp = total_logp + torch.where(active, picked, torch.zeros_like(picked))

            add_bit = torch.zeros_like(chosen_mask).scatter(2, idx_s.unsqueeze(-1), True)
            add_active = add_bit & active.unsqueeze(-1)
            chosen_mask = chosen_mask | add_active
            selected_bits = selected_bits | add_active.to(torch.int64)

            if step + 1 < max_step:
                idx_s_exp = idx_s.unsqueeze(-1).expand(bsz, num_samples, d_key)
                step_key = torch.gather(w_keys, 1, idx_s_exp).reshape(bsz * num_samples, d_key)
                h_next = wolff.step_gru(step_key, h_gru)
                act_flat = active.reshape(bsz * num_samples, 1)
                h_gru = torch.where(act_flat, h_next, h_gru)

        # 将 (B, K, w_free*32) 比特位向量打包为 (B, K, 16) uint32-in-int64
        bit_grid = selected_bits.reshape(bsz, num_samples, w_free, 32)
        shifts = torch.arange(32, device=device, dtype=torch.int64).view(1, 1, 1, 32)
        dw_free = (bit_grid << shifts).sum(dim=-1)  # (B, K, w_free)
        dw_full = torch.zeros((bsz, num_samples, 16), dtype=torch.int64, device=device)
        dw_full[:, :, :w_free] = dw_free

        actual_hamming = selected_bits.sum(dim=-1)  # (B, K)
        return {
            "dW0": dw_full,
            "log_prob": total_logp,
            "entropy": ent_size + ent_0,
            "hamming": actual_hamming,
        }


def deduplicate_candidates_per_base(
    cand_free_np: np.ndarray,
    k_target: int,
    w_free: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """
    对单个批次 (B, K_pool, w_free) uint32 候选微扰执行严格基底内去重、过滤零微扰，
    保留首次出现顺序；若不足 k_target 条，则补充互异随机多比特微扰至精确 k_target 条。
    返回 (dW0_full (B, k_target, 16) uint32, neural_unique_frac)。
    """
    bsz = cand_free_np.shape[0]
    n_w = w_free * 32
    out = np.zeros((bsz, k_target, 16), dtype=np.uint32)
    neural_counts = 0

    for i in range(bsz):
        rows = cand_free_np[i]  # (K_pool, w_free)
        nz_mask = np.any(rows != 0, axis=1)
        rows_nz = rows[nz_mask]
        if rows_nz.shape[0] > 0:
            _, first_idx = np.unique(rows_nz, axis=0, return_index=True)
            first_idx.sort()
            uniq_rows = rows_nz[first_idx][:k_target]
        else:
            uniq_rows = np.zeros((0, w_free), dtype=np.uint32)

        n_neural = int(uniq_rows.shape[0])
        neural_counts += n_neural
        if n_neural > 0:
            out[i, :n_neural, :w_free] = uniq_rows

        if n_neural < k_target:
            seen = {uniq_rows[r].tobytes() for r in range(n_neural)}
            pos = n_neural
            while pos < k_target:
                need = k_target - pos
                # 批量生成随机 m ∈ {1..min(6, n_w)} 比特微扰填充剩余槽位
                batch_gen = max(need * 2, 64)
                extra = sample_random_perturbations_np(rng, batch_gen, w_free=w_free, max_bits=min(6, n_w))
                for r in range(batch_gen):
                    key = extra[r].tobytes()
                    if key not in seen:
                        seen.add(key)
                        out[i, pos, :w_free] = extra[r]
                        pos += 1
                        if pos >= k_target:
                            break

    neural_frac = float(neural_counts) / float(max(1, bsz * k_target))
    return out, neural_frac


def sample_random_perturbations_np(
    rng: np.random.Generator,
    count: int,
    w_free: int,
    max_bits: int = 6,
) -> np.ndarray:
    """在 Omega_free(w_free) 内向量化采样非零随机多比特微扰 m ∈ [1, max_bits]，返回 (count, w_free) uint32。"""
    n_w = w_free * 32
    m_hi = min(max_bits, n_w)
    ms = rng.integers(1, m_hi + 1, size=count)
    out = np.zeros((count, w_free), dtype=np.uint32)
    for m_val in range(1, m_hi + 1):
        idx = np.flatnonzero(ms == m_val)
        if idx.size == 0:
            continue
        if m_val == 1:
            bits = rng.integers(0, n_w, size=idx.size)
            words = bits >> 5
            bpos = (bits & 31).astype(np.uint32)
            out[idx, words] ^= np.uint32(1) << bpos
        else:
            # 向量化抽取无重复多比特索引（按随机浮点键取前 m_val 个）
            scores = rng.random((idx.size, n_w), dtype=np.float32)
            chosen = np.argpartition(scores, kth=m_val - 1, axis=1)[:, :m_val]  # (K_m, m_val)
            words = chosen >> 5
            bpos = (chosen & 31).astype(np.uint32)
            masks = np.uint32(1) << bpos
            for col in range(m_val):
                np.bitwise_xor.at(out, (idx, words[:, col]), masks[:, col])
    return out


def generate_rand_walk_candidates(
    rng: np.random.Generator,
    bsz: int,
    k_target: int,
    w_free: int,
    max_bits: int = 6,
) -> np.ndarray:
    """为 RAND_WALK 基线生成每基底晶体严格去重的 (B, k_target, 16) uint32 微扰。"""
    # 过采样 1.25x 后按基底晶体去重
    pool_k = int(math.ceil(k_target * 1.25)) + 16
    raw = sample_random_perturbations_np(rng, bsz * pool_k, w_free=w_free, max_bits=max_bits)
    raw = raw.reshape(bsz, pool_k, w_free)
    out, _ = deduplicate_candidates_per_base(raw, k_target=k_target, w_free=w_free, rng=rng)
    return out
