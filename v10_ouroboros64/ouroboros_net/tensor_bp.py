r"""
Phase-SHA V10「衔尾蛇-64」神经阻尼张量置信传播层（Neural Damped Tensor-BP）。

核心物理突破：
  不再假定单比特独立边缘分布 P(x) = \prod_i p_i(x_i)（秩-1 平均场退相干），
  而是在进位相邻边 (b, b+1) 与跨轮压缩边 (t, t+1) 上显式维护低秩成对纠缠键张量
  J_carry ∈ R^{χ×χ} 与 J_time ∈ R^{χ×χ}，执行：
    1) 三轴显式稀疏物理边消息聚合（进位扫描 + ROTR/SHR + 时空滞后）
    2) 经由 J_carry / J_time 收缩的双边纠缠消息传递
    3) 规范 Frobenius 归一化与可学习阻尼系数 λ ∈ (0, 1) 更新
  并导出纠缠熵诊断 S_ent(J) ∈ [0, ln χ]，直接监测多比特协同相关强度。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .arx_layers import (
    ARXRotationEdgeBlock,
    BidirectionalCarryBlock,
    TemporalLagEdgeBlock,
)


def canonical_bond_normalize(j_tensor: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """对成对键张量 (..., chi, chi) 执行规范 Frobenius 归一化，使 sum(sigma_k^2) = 1。"""
    sq_norm = torch.sum(j_tensor * j_tensor, dim=(-2, -1), keepdim=True).clamp(min=eps)
    return j_tensor * torch.rsqrt(sq_norm)


def compute_bond_entropy(j_tensor: torch.Tensor, return_mean: bool = True) -> torch.Tensor:
    """
    计算规范化键张量 J ∈ R^{..., chi, chi} 的冯·诺依曼纠缠熵：
      sigma = svdvals(J), p_k = sigma_k^2 / sum_m sigma_m^2,
      S_ent = -sum_k p_k log(p_k) ∈ [0, ln(chi)].
    """
    chi = j_tensor.shape[-1]
    max_ent = math.log(max(2, chi)) if chi > 1 else 0.0
    flat = j_tensor.reshape(-1, chi, chi).to(torch.float32)
    # svdvals 仅需奇异值，GPU/CPU FP32 下速度极快且稳定
    sv = torch.linalg.svdvals(flat)
    p = sv * sv
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    ent = -torch.sum(p * torch.log(p.clamp(min=1e-12)), dim=-1)
    ent = ent.clamp(min=0.0, max=max_ent)
    if return_mean:
        return ent.mean()
    return ent.reshape(j_tensor.shape[:-2])


class BondInitializer(nn.Module):
    """从初始晶体一元隐状态 H_0 生成规范化的低秩相邻纠缠键张量 (J_carry, J_time)。"""

    def __init__(self, d_node: int, bond_dim: int) -> None:
        super().__init__()
        self.bond_dim = bond_dim
        self.norm = nn.LayerNorm(d_node)
        self.carry_left = nn.Linear(d_node, bond_dim * 2)
        self.carry_right = nn.Linear(d_node, bond_dim * 2)
        self.time_left = nn.Linear(d_node, bond_dim * 2)
        self.time_right = nn.Linear(d_node, bond_dim * 2)
        self.register_buffer("eye_anchor", torch.eye(bond_dim, dtype=torch.float32), persistent=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, R, 32, D)
        Returns:
            j_carry: (B, R, 31, chi, chi)
            j_time:  (B, R-1, 32, chi, chi)
        """
        bsz, n_rounds, n_bits, _ = h.shape
        chi = self.bond_dim
        hn = self.norm(h)
        scale = 1.0 / math.sqrt(2.0 * chi)

        # 进位相邻边 (b, b+1)
        cl = self.carry_left(hn[:, :, :-1, :]).reshape(bsz, n_rounds, n_bits - 1, chi, 2)
        cr = self.carry_right(hn[:, :, 1:, :]).reshape(bsz, n_rounds, n_bits - 1, chi, 2)
        j_carry_raw = torch.einsum("ntbik,ntbjk->ntbij", cl, cr) * scale
        j_carry = canonical_bond_normalize(j_carry_raw + 0.35 * self.eye_anchor)

        # 跨轮时间边 (t, t+1)
        tl = self.time_left(hn[:, :-1, :, :]).reshape(bsz, n_rounds - 1, n_bits, chi, 2)
        tr = self.time_right(hn[:, 1:, :, :]).reshape(bsz, n_rounds - 1, n_bits, chi, 2)
        j_time_raw = torch.einsum("ntbik,ntbjk->ntbij", tl, tr) * scale
        j_time = canonical_bond_normalize(j_time_raw + 0.35 * self.eye_anchor)
        return j_carry, j_time


class NeuralDampedTensorBPLayer(nn.Module):
    """
    单层神经阻尼 Tensor-BP 算子。
    联合演化节点信念 H (B, R, 32, D) 与两类纠缠键张量 J_carry, J_time。
    """

    def __init__(
        self,
        d_node: int,
        bond_dim: int = 4,
        dropout: float = 0.0,
        init_damping: float = 0.65,
    ) -> None:
        super().__init__()
        self.d_node = d_node
        self.bond_dim = bond_dim

        # 三轴显式稀疏物理消息算子
        self.carry_edge = BidirectionalCarryBlock(d_node, dropout=dropout)
        self.rot_edge = ARXRotationEdgeBlock(d_node, dropout=dropout)
        self.time_edge = TemporalLagEdgeBlock(d_node, dropout=dropout)

        # 键张量收缩通信：node <-> chi 双向投影
        self.node_norm = nn.LayerNorm(d_node)
        self.to_bond_q = nn.Linear(d_node, bond_dim)
        self.from_bond_msg = nn.Linear(bond_dim * 4, d_node)

        # 一元消息融合投影
        self.msg_fuse = nn.Sequential(
            nn.LayerNorm(d_node * 4),
            nn.Linear(d_node * 4, d_node * 2),
            nn.SiLU(),
            nn.Linear(d_node * 2, d_node),
        )

        # 键张量演化门控更新（秩-2 外积残差 + 对角缩放）
        rank_u = max(2, min(4, bond_dim))
        self.bond_rank = rank_u
        self.j_carry_u = nn.Linear(d_node, bond_dim * rank_u)
        self.j_carry_v = nn.Linear(d_node, bond_dim * rank_u)
        self.j_time_u = nn.Linear(d_node, bond_dim * rank_u)
        self.j_time_v = nn.Linear(d_node, bond_dim * rank_u)

        # 可学习阻尼系数 lambda = sigmoid(logit) ∈ (0, 1)
        init_logit = math.log(init_damping / (1.0 - init_damping))
        self.damp_h_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        self.damp_j_logit = nn.Parameter(torch.tensor(init_logit - 0.3, dtype=torch.float32))

        self.post_norm = nn.LayerNorm(d_node)

    def forward(
        self,
        h: torch.Tensor,
        j_carry: torch.Tensor,
        j_time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n_rounds, n_bits, d_node = h.shape
        chi = self.bond_dim

        # 1. 三类稀疏 ARX 物理消息
        m_carry = self.carry_edge(h)
        m_rot = self.rot_edge(h)
        m_time = self.time_edge(h)

        # 2. 经由纠缠键张量 (J_carry, J_time) 收缩传递多体联合消息
        hn = self.node_norm(h)
        q = torch.tanh(self.to_bond_q(hn))  # (B, R, 32, chi)

        msg_c_right = torch.zeros_like(q)
        msg_c_left = torch.zeros_like(q)
        # b -> b+1: J_carry[:, :, b, i, j] * q[:, :, b, i]
        msg_c_right[:, :, 1:, :] = torch.einsum("ntbij,ntbi->ntbj", j_carry, q[:, :, :-1, :])
        # b+1 -> b: J_carry[:, :, b, i, j] * q[:, :, b+1, j]
        msg_c_left[:, :, :-1, :] = torch.einsum("ntbij,ntbj->ntbi", j_carry, q[:, :, 1:, :])

        msg_t_fwd = torch.zeros_like(q)
        msg_t_bwd = torch.zeros_like(q)
        msg_t_fwd[:, 1:, :, :] = torch.einsum("ntbij,ntbi->ntbj", j_time, q[:, :-1, :, :])
        msg_t_bwd[:, :-1, :, :] = torch.einsum("ntbij,ntbj->ntbi", j_time, q[:, 1:, :, :])

        m_bond = F.silu(
            self.from_bond_msg(torch.cat([msg_c_right, msg_c_left, msg_t_fwd, msg_t_bwd], dim=-1))
        )

        # 3. 阻尼一元信念更新
        m_total = self.msg_fuse(torch.cat([m_carry, m_rot, m_time, m_bond], dim=-1))
        lam_h = torch.sigmoid(self.damp_h_logit)
        h_next = self.post_norm((1.0 - lam_h) * h + lam_h * (h + m_total))

        # 4. 阻尼成对纠缠键张量更新 + 规范归一化
        hn_next = self.node_norm(h_next)
        r = self.bond_rank
        scale_j = 1.0 / math.sqrt(float(r))

        cu = torch.tanh(self.j_carry_u(hn_next[:, :, :-1, :])).reshape(bsz, n_rounds, n_bits - 1, chi, r)
        cv = torch.tanh(self.j_carry_v(hn_next[:, :, 1:, :])).reshape(bsz, n_rounds, n_bits - 1, chi, r)
        j_carry_prop = canonical_bond_normalize(torch.einsum("ntbir,ntbjr->ntbij", cu, cv) * scale_j)

        tu = torch.tanh(self.j_time_u(hn_next[:, :-1, :, :])).reshape(bsz, n_rounds - 1, n_bits, chi, r)
        tv = torch.tanh(self.j_time_v(hn_next[:, 1:, :, :])).reshape(bsz, n_rounds - 1, n_bits, chi, r)
        j_time_prop = canonical_bond_normalize(torch.einsum("ntbir,ntbjr->ntbij", tu, tv) * scale_j)

        lam_j = torch.sigmoid(self.damp_j_logit)
        j_carry_next = canonical_bond_normalize((1.0 - lam_j) * j_carry + lam_j * j_carry_prop)
        j_time_next = canonical_bond_normalize((1.0 - lam_j) * j_time + lam_j * j_time_prop)

        return h_next, j_carry_next, j_time_next


class TensorBPStack(nn.Module):
    """多层堆叠神经阻尼张量置信传播骨干网络（可选每层梯度检查点节省显存）。"""

    def __init__(
        self,
        d_node: int,
        n_layers: int = 4,
        bond_dim: int = 4,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.d_node = d_node
        self.n_layers = n_layers
        self.bond_dim = bond_dim
        self.use_checkpoint = use_checkpoint

        self.bond_init = BondInitializer(d_node, bond_dim)
        self.layers = nn.ModuleList(
            [
                NeuralDampedTensorBPLayer(
                    d_node=d_node,
                    bond_dim=bond_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        h0: torch.Tensor,
        return_entropy: bool = False,
    ) -> Dict[str, torch.Tensor]:
        h = h0
        j_carry, j_time = self.bond_init(h)

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                h, j_carry, j_time = checkpoint(
                    layer, h, j_carry, j_time, use_reentrant=False
                )
            else:
                h, j_carry, j_time = layer(h, j_carry, j_time)

        out: Dict[str, torch.Tensor] = {
            "H": h,
            "J_carry": j_carry,
            "J_time": j_time,
        }
        if return_entropy:
            # 纠缠熵诊断作为可观测监控量，不额外引入 svdvals 反传开销
            with torch.no_grad():
                out["S_ent_carry"] = compute_bond_entropy(j_carry, return_mean=True)
                out["S_ent_time"] = compute_bond_entropy(j_time, return_mean=True)
        return out
