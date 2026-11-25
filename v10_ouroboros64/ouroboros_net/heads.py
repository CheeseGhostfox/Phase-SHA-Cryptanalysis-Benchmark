"""
Phase-SHA V10「衔尾蛇-64」三大专用决策头与缺陷愈合头：
  1. DefectHealingHead：19 平面逐比特缺陷修复 logits + 相邻进位对 4 态联合分布 logits
  2. WolffClusterProposer：两阶段（缺陷锚点 + 条件自回归协同位）神经 Wolff 团簇提议器，
     严格屏蔽焊死边界 S[0]/S[R] 与常量进位位，导出精确可微联合 log_prob（供 Stage A 蒸馏与 Stage C GRPO）
  3. CDCLBranchPhaseScorer：CDCL 瓶颈变量坍缩得分（替代 VSIDS）与条件相位极性概率
  4. CrystalEnergyCritic：全局晶体残余势能价值评估头
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ouroboros_core.engine import C_PLANES


PLANE_COUNT = 1 + 8 + C_PLANES  # 0:W, 1..8:S[t,0..7], 9..18:C[t,0..9]


def build_valid_site_mask(rounds: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """
    构建可变晶体位点的合法布尔掩码，形状 (R, 19, 32)。
    严格禁止修改：
      - S[0]（t=0，planes 1..8）：起点焊死于 IV（注：S[R] 不在 0..R-1 索引内，天然不可触及）
      - C[:, :, b=0]（planes 9..18, b=0）：模 2^32 加法最低位进位恒为 0
      - C[0..15, 0..2, :]（planes 9..11, t<16）：W[0..15] 无需消息扩展进位链
    """
    mask = torch.ones((rounds, PLANE_COUNT, 32), dtype=torch.bool, device=device)
    # 屏蔽 S[0] 焊死起点边界
    mask[0, 1:9, :] = False
    # 屏蔽全部进位平面的 b=0（常量 0）
    mask[:, 9:19, 0] = False
    # 屏蔽前 16 轮不参与计算的消息扩展进位平面 C[:, 0..2]
    t_msg_unused = min(16, rounds)
    mask[:t_msg_unused, 9:12, :] = False
    return mask


def flat_site_to_coord(flat_idx: torch.Tensor, rounds: int) -> torch.Tensor:
    """
    将扁平位点索引 flat_idx ∈ [0, R*19*32 - 1]（或 -1 填充）转换为 C++ 坐标
    (kind, t, word, bit) int32。
      - plane 0       -> kind=0 (W), word=0
      - plane 1..8    -> kind=1 (S), word=plane-1
      - plane 9..18   -> kind=2 (C), word=plane-9
      - flat_idx < 0  -> (-1, 0, 0, 0)
    """
    out = torch.zeros((*flat_idx.shape, 4), dtype=torch.int32, device=flat_idx.device)
    out[..., 0] = -1
    valid = flat_idx >= 0
    if not torch.any(valid):
        return out

    idx = flat_idx[valid].to(torch.int64)
    b = idx % 32
    rem = idx // 32
    p = rem % PLANE_COUNT
    t = rem // PLANE_COUNT

    kind = torch.where(p == 0, 0, torch.where(p < 9, 1, 2))
    word = torch.where(p == 0, 0, torch.where(p < 9, p - 1, p - 9))

    coords_v = torch.stack([kind, t, word, b], dim=-1).to(torch.int32)
    out[valid] = coords_v
    return out


def coord_to_flat_site(coords: torch.Tensor) -> torch.Tensor:
    """
    flat_site_to_coord 的精确逆变换：(..., 4) -> flat_idx int64，kind < 0 映射为 -1。
    """
    kind = coords[..., 0].to(torch.int64)
    t = coords[..., 1].to(torch.int64)
    word = coords[..., 2].to(torch.int64)
    bit = coords[..., 3].to(torch.int64)

    plane = torch.where(kind == 0, 0, torch.where(kind == 1, 1 + word, 9 + word))
    flat = ((t * PLANE_COUNT) + plane) * 32 + bit
    return torch.where(kind >= 0, flat, torch.full_like(flat, -1))


class DefectHealingHead(nn.Module):
    """
    缺陷定位与进位纠缠联合分布头：
      - flip_logits: (B, R, 19, 32) 预测每个位点被注入噪声/需翻转概率
      - pair_logits: (B, R, 31, 10, 4) 预测相邻进位比特对 (c_{k,b}, c_{k,b+1}) 的 4 态联合分布，
        显式条件于 J_carry 纠缠键张量，检验是否突破秩-1 边缘分解。
    """

    def __init__(self, d_node: int, bond_dim: int) -> None:
        super().__init__()
        self.bond_dim = bond_dim
        self.flip_head = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, d_node),
            nn.SiLU(),
            nn.Linear(d_node, PLANE_COUNT),
        )
        # 进位对联合分布条件输入：[H_b, H_{b+1}, flatten(J_carry)]
        pair_in_dim = d_node * 2 + bond_dim * bond_dim
        pair_hidden = max(64, d_node)
        self.pair_head = nn.Sequential(
            nn.LayerNorm(pair_in_dim),
            nn.Linear(pair_in_dim, pair_hidden),
            nn.SiLU(),
            nn.Linear(pair_hidden, C_PLANES * 4),
        )

    def forward(
        self,
        h: torch.Tensor,
        j_carry: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n_rounds, n_bits, _ = h.shape
        flip_raw = self.flip_head(h)  # (B, R, 32, 19)
        flip_logits = flip_raw.permute(0, 1, 3, 2).contiguous()  # (B, R, 19, 32)

        j_flat = j_carry.reshape(bsz, n_rounds, n_bits - 1, self.bond_dim * self.bond_dim)
        pair_in = torch.cat([h[:, :, :-1, :], h[:, :, 1:, :], j_flat], dim=-1)
        pair_logits = self.pair_head(pair_in).reshape(bsz, n_rounds, n_bits - 1, C_PLANES, 4)
        return flip_logits, pair_logits


class WolffClusterProposer(nn.Module):
    """
    Head 1：神经 Wolff 多比特协同团簇提议器。
    架构：两阶段「缺陷锚点选择 + 条件自回归协同位生成」。
    特性：
      - 输出大小 m ∈ [2, M_max] 的协同翻转团簇 (B, K, M_max, 4)，末尾 -1 填充；
      - 与 C++ eval_cluster_flips_batch 坐标完全对齐；
      - 同时支持采样提议 (`propose_clusters`) 与给定正团簇似然蒸馏 (`score_given_clusters`)。
    """

    def __init__(
        self,
        d_node: int,
        d_key: int = 32,
        max_cluster_size: int = 6,
    ) -> None:
        super().__init__()
        self.d_node = d_node
        self.d_key = d_key
        self.max_cluster_size = max(2, max_cluster_size)

        # 从节点隐状态映射到 19 个平面的位点 key 与 anchor score
        self.node_norm = nn.LayerNorm(d_node)
        self.site_key_proj = nn.Linear(d_node, PLANE_COUNT * d_key)
        self.anchor_proj = nn.Linear(d_node, PLANE_COUNT)
        self.plane_emb = nn.Embedding(PLANE_COUNT, d_key)

        # 融合局部缺陷残差信号与比特周期位置编码 (t/R, sin/cos bit)
        self.coord_proj = nn.Linear(4, d_key)
        self.key_fuse = nn.Sequential(
            nn.LayerNorm(d_key + 1),
            nn.Linear(d_key + 1, d_key),
            nn.Tanh(),
        )
        self.anchor_defect_weight = nn.Parameter(torch.tensor(6.0, dtype=torch.float32))
        self.partner_defect_weight = nn.Parameter(torch.tensor(4.5, dtype=torch.float32))
        self.partner_local_round_bias = nn.Parameter(torch.tensor(3.5, dtype=torch.float32))
        self.partner_lag_bias = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.flip_logit_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # 条件自回归协同位生成器
        self.step_gru = nn.GRUCell(d_key, d_key)
        self.query_norm = nn.LayerNorm(d_key)
        self.query_proj = nn.Linear(d_key, d_key)
        self.stop_proj = nn.Linear(d_key, 1)

    @staticmethod
    def _coord_features(rounds: int, device: torch.device) -> torch.Tensor:
        """构建每个 (t, b) 的归一化时空坐标特征 (1, R, 1, 32, 4)。"""
        t_norm = torch.linspace(0.0, 1.0, rounds, device=device, dtype=torch.float32)
        b_idx = torch.arange(32, device=device, dtype=torch.float32)
        angle = 2.0 * math.pi * (b_idx / 32.0)
        t_grid = t_norm.view(1, rounds, 1, 1, 1).expand(1, rounds, 1, 32, 1)
        t_rev = (1.0 - t_norm).view(1, rounds, 1, 1, 1).expand(1, rounds, 1, 32, 1)
        sin_b = torch.sin(angle).view(1, 1, 1, 32, 1).expand(1, rounds, 1, 32, 1)
        cos_b = torch.cos(angle).view(1, 1, 1, 32, 1).expand(1, rounds, 1, 32, 1)
        return torch.cat([t_grid, t_rev, sin_b, cos_b], dim=-1)

    def encode_sites(
        self,
        h: torch.Tensor,
        site_defect_signal: torch.Tensor,
        flip_logits: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        将骨干张量 H (B, R, 32, D) 与各平面缺陷信号 (B, R, 19, 32) 编码为
        扁平候选位点键向量与锚点分布 logits。
        """
        bsz, n_rounds, n_bits, _ = h.shape
        device = h.device
        hn = self.node_norm(h)

        # (B, R, 32, 19*d_key) -> (B, R, 19, 32, d_key)
        keys = (
            self.site_key_proj(hn)
            .reshape(bsz, n_rounds, n_bits, PLANE_COUNT, self.d_key)
            .permute(0, 1, 3, 2, 4)
        )
        p_ids = torch.arange(PLANE_COUNT, device=device)
        p_emb = self.plane_emb(p_ids).view(1, 1, PLANE_COUNT, 1, self.d_key)
        c_feat = self.coord_proj(self._coord_features(n_rounds, device))  # (1, R, 1, 32, d_key)
        keys = keys + p_emb + c_feat

        def_sig = site_defect_signal.unsqueeze(-1).to(torch.float32)  # (B, R, 19, 32, 1)
        keys_fused = self.key_fuse(torch.cat([keys, def_sig], dim=-1))
        n_sites = n_rounds * PLANE_COUNT * n_bits
        flat_keys = keys_fused.reshape(bsz, n_sites, self.d_key)
        flat_defect = site_defect_signal.reshape(bsz, n_sites).to(torch.float32)
        if flip_logits is not None:
            flat_flip_score = flip_logits.detach().reshape(bsz, n_sites).clamp(-8.0, 8.0).to(torch.float32)
        else:
            flat_flip_score = torch.zeros_like(flat_defect)
        site_round = (
            torch.arange(n_rounds, device=device, dtype=torch.long)
            .view(n_rounds, 1, 1)
            .expand(n_rounds, PLANE_COUNT, n_bits)
            .reshape(n_sites)
        )

        anchor_raw = self.anchor_proj(hn).permute(0, 1, 3, 2).reshape(bsz, n_sites)
        anchor_logits = (
            anchor_raw
            + self.anchor_defect_weight * flat_defect
            + self.flip_logit_weight * flat_flip_score
        )
        valid_mask = build_valid_site_mask(n_rounds, device=device).reshape(1, n_sites)
        anchor_logits = anchor_logits.masked_fill(~valid_mask, -1e9)

        return {
            "rounds": n_rounds,
            "n_sites": n_sites,
            "flat_keys": flat_keys,
            "flat_defect": flat_defect,
            "flat_flip_score": flat_flip_score,
            "site_round": site_round,
            "anchor_logits": anchor_logits,
            "valid_mask": valid_mask,
        }

    def propose_clusters(
        self,
        h: torch.Tensor,
        site_defect_signal: torch.Tensor,
        num_clusters: int = 16,
        temperature: float = 1.0,
        deterministic: bool = False,
        flip_logits: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        对批量内每个晶体提议 K 个非退化多比特协同团簇 (m ∈ [2, M_max])。
        Returns:
            clusters_coord: (B, K, M_max, 4) int32
            clusters_flat:  (B, K, M_max) int64（-1 填充）
            log_prob:       (B, K) float32 精确联合对数概率
            cluster_sizes:  (B, K) int64
        """
        cache = self.encode_sites(h, site_defect_signal, flip_logits=flip_logits)
        return self._decode_clusters(
            cache,
            num_clusters=num_clusters,
            temperature=temperature,
            deterministic=deterministic,
            target_flat=None,
        )

    def score_given_clusters(
        self,
        h: torch.Tensor,
        site_defect_signal: torch.Tensor,
        clusters_coord: torch.Tensor,
        flip_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        给定外部已知有效团簇坐标 clusters_coord (B, K, M_max, 4)，计算精确条件对数似然 (B, K)，
        用于 Stage A C++ 筛选正团簇（ΔE < 0）的监督蒸馏。
        """
        cache = self.encode_sites(h, site_defect_signal, flip_logits=flip_logits)
        target_flat = coord_to_flat_site(clusters_coord)
        out = self._decode_clusters(
            cache,
            num_clusters=target_flat.shape[1],
            temperature=1.0,
            deterministic=True,
            target_flat=target_flat,
        )
        return out["log_prob"]

    def _decode_clusters(
        self,
        cache: Dict[str, torch.Tensor],
        num_clusters: int,
        temperature: float,
        deterministic: bool,
        target_flat: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor]:
        flat_keys = cache["flat_keys"]  # (B, N, d_key)
        flat_defect = cache["flat_defect"]  # (B, N)
        flat_flip_score = cache["flat_flip_score"]  # (B, N)
        site_round = cache["site_round"]  # (N,)
        anchor_logits = cache["anchor_logits"]  # (B, N)
        valid_mask = cache["valid_mask"]  # (1, N)
        n_rounds = int(cache["rounds"])
        n_sites = int(cache["n_sites"])
        bsz = flat_keys.shape[0]
        device = flat_keys.device
        m_max = (
            target_flat.shape[2]
            if target_flat is not None
            else self.max_cluster_size
        )
        temp = max(0.05, float(temperature))
        scale = 1.0 / math.sqrt(float(self.d_key))

        # Step 0: 选择锚点位点 (B, K)
        scaled_anchor = anchor_logits / temp
        log_probs_0 = F.log_softmax(scaled_anchor, dim=-1)
        if target_flat is not None:
            idx_0 = target_flat[:, :, 0].clamp(min=0)
        elif deterministic:
            # 取 top-K 不重复高分锚点
            k_eff = min(num_clusters, n_sites)
            idx_0 = torch.topk(scaled_anchor, k=k_eff, dim=-1).indices
            if k_eff < num_clusters:
                idx_0 = idx_0.repeat(1, (num_clusters + k_eff - 1) // k_eff)[:, :num_clusters]
        else:
            probs_0 = torch.exp(log_probs_0)
            idx_0 = torch.multinomial(probs_0, num_samples=num_clusters, replacement=True)

        total_log_prob = torch.gather(log_probs_0, 1, idx_0)  # (B, K)

        chosen_seq = [idx_0]
        # 追踪已经选过的比特防止同一团簇内自我抵消（全程非原地更新保证反传版本兼容）
        chosen_mask = ~valid_mask.unsqueeze(1).expand(bsz, num_clusters, n_sites).clone()
        chosen_mask = chosen_mask.scatter(2, idx_0.unsqueeze(-1), True)

        # 提取锚点向量初始化 GRU 隐状态: (B*K, d_key)
        idx_0_exp = idx_0.unsqueeze(-1).expand(bsz, num_clusters, self.d_key)
        h_gru = torch.gather(flat_keys, 1, idx_0_exp).reshape(bsz * num_clusters, self.d_key)
        alive = torch.ones((bsz, num_clusters), dtype=torch.bool, device=device)

        # 计算以锚点轮次为中心的 ARX 局部与跨轮滞后 {-2,-7,-15,-16} 物理偏置
        t_anchor = site_round[idx_0].unsqueeze(-1)  # (B, K, 1)
        round_diff = (t_anchor - site_round.view(1, 1, n_sites)).abs()
        is_local = (round_diff <= 1).to(torch.float32)
        is_lag = (
            (round_diff == 2) | (round_diff == 7) | (round_diff == 15) | (round_diff == 16)
        ).to(torch.float32)
        partner_phys_bias = (
            self.partner_defect_weight * flat_defect.unsqueeze(1)
            + 0.5 * self.flip_logit_weight * flat_flip_score.unsqueeze(1)
            + self.partner_local_round_bias * is_local
            + self.partner_lag_bias * is_lag
        )

        for step in range(1, m_max):
            q = self.query_proj(self.query_norm(h_gru)).reshape(bsz, num_clusters, self.d_key)
            # (B, K, N_sites)
            partner_logits = torch.einsum("bkd,bnd->bkn", q, flat_keys) * scale + partner_phys_bias
            partner_logits = partner_logits.masked_fill(chosen_mask, -1e9)

            if step >= 2:
                # 仅在 m >= 2 后允许选择 STOP（索引 n_sites），强制所有团簇至少 2-bit 协同
                stop_logit = self.stop_proj(h_gru).reshape(bsz, num_clusters, 1)
                step_logits = torch.cat([partner_logits, stop_logit], dim=-1) / temp
            else:
                step_logits = partner_logits / temp

            step_log_probs = F.log_softmax(step_logits, dim=-1)
            n_choices = step_logits.shape[-1]

            if target_flat is not None:
                tgt = target_flat[:, :, step]
                is_pad = tgt < 0
                # 若之前还活着但此时 tgt==-1，等价于选择了 STOP
                step_action = torch.where(is_pad, torch.full_like(tgt, n_sites), tgt)
                if step < 2:
                    step_action = step_action.clamp(0, n_sites - 1)
            elif deterministic:
                step_action = torch.argmax(step_logits, dim=-1)
            else:
                probs_s = torch.exp(step_log_probs).reshape(bsz * num_clusters, n_choices)
                step_action = torch.multinomial(probs_s, num_samples=1).reshape(bsz, num_clusters)

            picked_log_p = torch.gather(step_log_probs, 2, step_action.unsqueeze(-1)).squeeze(-1)
            total_log_prob = total_log_prob + torch.where(alive, picked_log_p, torch.zeros_like(picked_log_p))

            if step >= 2:
                chose_stop = step_action >= n_sites
                alive_next = alive & (~chose_stop)
            else:
                alive_next = alive

            # 写入输出（已停止或本步选 STOP 的槽位写 -1 填充）
            out_idx = torch.where(alive_next, step_action.clamp(0, n_sites - 1), torch.full_like(step_action, -1))
            chosen_seq.append(out_idx)

            # 更新掩码与 GRU 隐状态（对于仍存活且非 STOP 的位置）
            safe_idx = step_action.clamp(0, n_sites - 1)
            step_mask_upd = alive_next.unsqueeze(-1)
            scatter_src = torch.where(step_mask_upd, torch.ones_like(step_mask_upd), torch.zeros_like(step_mask_upd))
            # 仅置 True，不覆盖既有 True
            cur_existing = torch.gather(chosen_mask, 2, safe_idx.unsqueeze(-1))
            chosen_mask = chosen_mask.scatter(2, safe_idx.unsqueeze(-1), cur_existing | scatter_src)

            picked_keys = torch.gather(
                flat_keys, 1, safe_idx.unsqueeze(-1).expand(bsz, num_clusters, self.d_key)
            ).reshape(bsz * num_clusters, self.d_key)
            h_gru_new = self.step_gru(picked_keys, h_gru)
            alive_flat = alive_next.reshape(bsz * num_clusters, 1)
            h_gru = torch.where(alive_flat, h_gru_new, h_gru)
            alive = alive_next

        clusters_flat = torch.stack(chosen_seq, dim=-1)  # (B, K, M_max)
        clusters_coord = flat_site_to_coord(clusters_flat, rounds=n_rounds)
        cluster_sizes = (clusters_flat >= 0).sum(dim=-1)
        return {
            "clusters_coord": clusters_coord,
            "clusters_flat": clusters_flat,
            "log_prob": total_log_prob,
            "cluster_sizes": cluster_sizes,
        }


class CDCLBranchPhaseScorer(nn.Module):
    """
    Head 2：CDCL 瓶颈变量坍缩得分（替代 VSIDS）与条件相位极性概率 P(x=1 | crystal)。
    """

    def __init__(self, d_node: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_node)
        self.hidden = nn.Linear(d_node, d_node)
        self.collapse_head = nn.Linear(d_node, PLANE_COUNT)
        self.phase_head = nn.Linear(d_node, PLANE_COUNT)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = F.silu(self.hidden(self.norm(h)))  # (B, R, 32, D)
        collapse_score = F.softplus(self.collapse_head(z)).permute(0, 1, 3, 2).contiguous()
        phase_logits = self.phase_head(z).permute(0, 1, 3, 2).contiguous()
        return collapse_score, phase_logits


class CrystalEnergyCritic(nn.Module):
    """
    Head 3：全局晶体残余势能价值头，联合节点信念 H 与跨轮纠缠键 J_time 预测 log1p(E)。
    """

    def __init__(self, d_node: int, bond_dim: int = 4, extra_scalar_dim: int = 5) -> None:
        super().__init__()
        self.bond_dim = bond_dim
        self.node_norm = nn.LayerNorm(d_node)
        self.attn_gate = nn.Linear(d_node, 1)
        j_feat_dim = 16
        self.j_time_proj = nn.Linear(bond_dim * bond_dim, j_feat_dim)
        in_dim = d_node * 2 + j_feat_dim + extra_scalar_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_node),
            nn.SiLU(),
            nn.Linear(d_node, max(32, d_node // 2)),
            nn.SiLU(),
            nn.Linear(max(32, d_node // 2), 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        j_time: torch.Tensor,
        scalar_stats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: (B, R, 32, D)
            j_time: (B, R-1, 32, chi, chi)
            scalar_stats: (B, 5) 归一化全局势能与纠缠熵统计特征
        Returns:
            pred_log_energy: (B,)
        """
        bsz = h.shape[0]
        hn = self.node_norm(h).reshape(bsz, -1, h.shape[-1])  # (B, R*32, D)
        mean_pool = hn.mean(dim=1)
        attn_w = torch.softmax(self.attn_gate(hn), dim=1)
        attn_pool = torch.sum(attn_w * hn, dim=1)
        jt_summary = j_time.reshape(bsz, -1, self.bond_dim * self.bond_dim).mean(dim=1)
        jt_feat = F.silu(self.j_time_proj(jt_summary))
        feat = torch.cat([mean_pool, attn_pool, jt_feat, scalar_stats.to(torch.float32)], dim=-1)
        return F.softplus(self.mlp(feat).squeeze(-1))
