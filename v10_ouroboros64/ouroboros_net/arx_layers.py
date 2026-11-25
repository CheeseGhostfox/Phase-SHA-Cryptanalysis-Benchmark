"""
Phase-SHA V10「衔尾蛇-64」3D ARX 显式稀疏物理边算子（纯粹精确几何偏置，纯 FP32）。

所有比特轴 b ∈ [0, 31] 严格采用 LSB-first 约定：
  - b = 0 为最低有效位 (LSB)
  - b = 31 为最高有效位 (MSB)
  - 模加进位沿 b=0 -> 31 正向扫描，终端焊死边界约束沿 b=31 -> 0 逆向回传
  - ROTR(x, n)[b] = x[(b + n) % 32]
  - SHR(x, n)[b]  = x[b + n] if b + n < 32 else 0
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# SHA-256 精确 ARX 循环移位与右移偏移集合
SIGMA0_ROTS = (2, 13, 22)
SIGMA1_ROTS = (6, 11, 25)
LOWER_SIGMA0_ROTS = (7, 18)
LOWER_SIGMA0_SHR = 3
LOWER_SIGMA1_ROTS = (17, 19)
LOWER_SIGMA1_SHR = 10

# 消息扩展滞后步长：W[t] <- (-2, -7, -15, -16)
MSG_LAG_OFFSETS = (2, 7, 15, 16)


class CircularConv1d(nn.Module):
    """沿 32-bit 比特轴执行环形零退相干卷积，对齐模 2^32 循环移位边界。"""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size 必须为奇数以保证对称环形填充，收到 {kernel_size}")
        self.pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (..., 32, C)
        Returns:
            shape (..., 32, C)
        """
        orig_shape = x.shape
        d = orig_shape[-1]
        y = x.reshape(-1, 32, d).transpose(1, 2)  # (N, C, 32)
        if self.pad > 0:
            y = F.pad(y, (self.pad, self.pad), mode="circular")
        y = self.conv(y).transpose(1, 2).reshape(orig_shape)
        return y


class GatedProductXORBlock(nn.Module):
    """双线性乘积交互块：在 {-1, +1} 双极空间原生表达精确异或 / 奇偶非线性。"""

    def __init__(self, d_model: int, expansion: float = 1.5, dropout: float = 0.0) -> None:
        super().__init__()
        d_hidden = max(16, int(round(d_model * expansion)))
        self.norm = nn.LayerNorm(d_model)
        self.proj_u = nn.Linear(d_model, d_hidden)
        self.proj_v = nn.Linear(d_model, d_hidden)
        self.norm_u = nn.LayerNorm(d_hidden)
        self.norm_v = nn.LayerNorm(d_hidden)
        self.proj_out = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        # 显式双极乘积交互 (tanh_u * tanh_v) + 残差门控，捕捉多比特 XOR 相关
        u = torch.tanh(self.norm_u(self.proj_u(h)))
        v = torch.tanh(self.norm_v(self.proj_v(h)))
        z = u * v
        return x + self.dropout(self.proj_out(z))


class BidirectionalCarryBlock(nn.Module):
    """
    轴 1：模 2^32 进位轴（Intra-Word Carry Axis, LSB 0 <-> MSB 31）。
    结合双向 GRU 顺序进位扫描（FP32）与局部环形双线性奇偶块。
    """

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        d_gru = max(16, d_model // 2)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model)
        # bidirectional=True 自动沿 b=0->31 (LSB->MSB) 与 b=31->0 (MSB->LSB) 双向传递进位状态
        self.carry_gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_gru,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.out_proj = nn.Linear(d_gru * 2, d_model)
        self.circ_conv = CircularConv1d(d_model, kernel_size=3, dilation=1)
        self.xor_block = GatedProductXORBlock(d_model, expansion=1.5, dropout=dropout)
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, R, 32, D)
        Returns:
            delta_carry: (B, R, 32, D)
        """
        bsz, n_rounds, n_bits, d_model = x.shape
        h = self.norm(x)
        h_in = F.silu(self.in_proj(h)).reshape(bsz * n_rounds, n_bits, d_model)
        # torch.nn.GRU 在 allow_tf32=False 下严格执行纯 FP32
        gru_out, _ = self.carry_gru(h_in)
        gru_out = self.out_proj(gru_out).reshape(bsz, n_rounds, n_bits, d_model)
        circ_out = self.circ_conv(h)
        combined = torch.tanh(self.gate(torch.cat([gru_out, circ_out], dim=-1)))
        mixed = self.xor_block(x + combined) - x
        return mixed


class ARXRotationEdgeBlock(nn.Module):
    """
    轴 2：SHA-256 循环移位/右移稀疏 Gather-Scatter 物理连边块。
    显式构建全部 12 组前向+反向 ARX 稀疏边：
      - Sigma0 (ROTR 2, 13, 22)
      - Sigma1 (ROTR 6, 11, 25)
      - sigma0 (ROTR 7, 18, SHR 3)
      - sigma1 (ROTR 17, 19, SHR 10)
    全部前向与逆向索引预编译为 buffer，在 GPU 上零 Python 循环向量化 gather。
    """

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        # 构建精确旋转 + 右移边列表（共 10 个旋转 + 2 个右移；每条边含前向与反向视图）
        rot_offsets = (
            list(SIGMA0_ROTS)
            + list(SIGMA1_ROTS)
            + list(LOWER_SIGMA0_ROTS)
            + list(LOWER_SIGMA1_ROTS)
        )
        shr_offsets = [LOWER_SIGMA0_SHR, LOWER_SIGMA1_SHR]

        fwd_indices = []
        bwd_indices = []
        valid_fwd_masks = []
        valid_bwd_masks = []
        bit_pos = torch.arange(32, dtype=torch.long)

        for n in rot_offsets:
            fwd_indices.append((bit_pos + n) % 32)
            bwd_indices.append((bit_pos - n) % 32)
            valid_fwd_masks.append(torch.ones(32, dtype=torch.float32))
            valid_bwd_masks.append(torch.ones(32, dtype=torch.float32))

        for n in shr_offsets:
            idx_f = (bit_pos + n).clamp(max=31)
            mask_f = ((bit_pos + n) < 32).to(torch.float32)
            idx_b = (bit_pos - n).clamp(min=0)
            mask_b = ((bit_pos - n) >= 0).to(torch.float32)
            fwd_indices.append(idx_f)
            bwd_indices.append(idx_b)
            valid_fwd_masks.append(mask_f)
            valid_bwd_masks.append(mask_b)

        self.num_edges = len(fwd_indices)  # 12
        # shape: (E, 32)
        self.register_buffer("fwd_idx", torch.stack(fwd_indices, dim=0), persistent=False)
        self.register_buffer("bwd_idx", torch.stack(bwd_indices, dim=0), persistent=False)
        self.register_buffer("fwd_mask", torch.stack(valid_fwd_masks, dim=0), persistent=False)
        self.register_buffer("bwd_mask", torch.stack(valid_bwd_masks, dim=0), persistent=False)

        # 用低秩瓶颈投影处理 2 * num_edges 条移位邻居，避免参数随边数线性爆炸
        d_edge = max(8, d_model // 4)
        self.norm = nn.LayerNorm(d_model)
        self.node_compress = nn.Linear(d_model, d_edge)
        # 2 * num_edges 个通道组融合（12前向 + 12反向）
        self.edge_fuse = nn.Linear(d_edge * self.num_edges * 2, d_model)
        self.circ_conv = CircularConv1d(d_model, kernel_size=5, dilation=1)
        self.xor_block = GatedProductXORBlock(d_model, expansion=1.5, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, R, 32, D)
        Returns:
            delta_rot: (B, R, 32, D)
        """
        bsz, n_rounds, n_bits, _ = x.shape
        h = self.norm(x)
        comp = self.node_compress(h)  # (B, R, 32, d_edge)
        d_edge = comp.shape[-1]

        # comp[:, :, fwd_idx, :] -> shape (B, R, E, 32, d_edge)
        fwd_gather = comp[:, :, self.fwd_idx, :] * self.fwd_mask.view(1, 1, self.num_edges, n_bits, 1)
        bwd_gather = comp[:, :, self.bwd_idx, :] * self.bwd_mask.view(1, 1, self.num_edges, n_bits, 1)

        # (B, R, 32, 2*E*d_edge)
        fwd_flat = fwd_gather.permute(0, 1, 3, 2, 4).reshape(bsz, n_rounds, n_bits, self.num_edges * d_edge)
        bwd_flat = bwd_gather.permute(0, 1, 3, 2, 4).reshape(bsz, n_rounds, n_bits, self.num_edges * d_edge)

        edge_msg = F.silu(self.edge_fuse(torch.cat([fwd_flat, bwd_flat], dim=-1)))
        circ_msg = self.circ_conv(h)
        out = self.xor_block(x + edge_msg + 0.5 * circ_msg) - x
        return out


class TemporalLagEdgeBlock(nn.Module):
    """
    轴 3：时空跨轮物理边（相邻压缩步 t±1 与消息扩展精确滞后 t±{2, 7, 15, 16}）。
    支持双向消息传递，让起点 S0=IV 与终点 SR=H-IV 的约束沿时间链和消息跳跃链无障碍交汇。
    """

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        d_edge = max(8, d_model // 3)
        self.norm = nn.LayerNorm(d_model)
        self.proj_adj = nn.Linear(d_model, d_edge)
        self.proj_lag = nn.Linear(d_model, d_edge)

        # 邻接边：t-1, t+1（2 条）; 滞后边：t-{2,7,15,16} 及 t+{2,7,15,16}（8 条）
        n_streams = 2 + 2 * len(MSG_LAG_OFFSETS)
        self.fuse = nn.Linear(d_edge * n_streams, d_model)
        self.ffn = GatedProductXORBlock(d_model, expansion=1.5, dropout=dropout)

    @staticmethod
    def _shift_round(tensor: torch.Tensor, offset: int) -> torch.Tensor:
        """
        沿轮数轴 t（dim=1）平移：out[:, t] = tensor[:, t + offset]（越界填 0）。
        offset < 0 表示从过去 t-|offset| 传入当前轮 t；offset > 0 表示从未来传入当前轮 t。
        """
        bsz, n_rounds, n_bits, d = tensor.shape
        if abs(offset) >= n_rounds:
            return torch.zeros_like(tensor)
        out = torch.zeros_like(tensor)
        if offset < 0:
            k = -offset
            out[:, k:, :, :] = tensor[:, : n_rounds - k, :, :]
        elif offset > 0:
            k = offset
            out[:, : n_rounds - k, :, :] = tensor[:, k:, :, :]
        else:
            out.copy_(tensor)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, R, 32, D)
        Returns:
            delta_time: (B, R, 32, D)
        """
        h = self.norm(x)
        adj_feat = self.proj_adj(h)
        lag_feat = self.proj_lag(h)

        streams = [
            self._shift_round(adj_feat, -1),
            self._shift_round(adj_feat, +1),
        ]
        for lag in MSG_LAG_OFFSETS:
            streams.append(self._shift_round(lag_feat, -lag))
            streams.append(self._shift_round(lag_feat, +lag))

        fused = F.silu(self.fuse(torch.cat(streams, dim=-1)))
        return self.ffn(x + fused) - x
