"""
Phase-SHA V10「衔尾蛇-64」主干模型封装：Ouroboros64Net。

严格遵守双精度隔离与硬焊边界协议：
  - 全网络强制纯 FP32（enforce_fp32_determinism()），绝不开 TF32/FP16/BF16；
  - 精确物理残差场由 TorchCrystalEngine 仅在 uint/int64 位运算下于 GPU 计算；
  - 输入特征同时编码 S[t] 与 S[t+1]，确保 t=0 的 S[0]=IV 与 t=R-1 的 S[R]=H-IV 双端焊死边界全量可见；
  - 集成 TensorBPStack、DefectHealingHead、WolffClusterProposer、CDCLBranchPhaseScorer、CrystalEnergyCritic。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..ouroboros_core.engine import (
    C_PLANES,
    K_WORDS,
    MAX_ROUNDS,
    TorchCrystalEngine,
    enforce_fp32_determinism,
    words_to_bits_lsb,
)
from .heads import (
    PLANE_COUNT,
    CDCLBranchPhaseScorer,
    CrystalEnergyCritic,
    DefectHealingHead,
    WolffClusterProposer,
)
from .tensor_bp import TensorBPStack


INPUT_FEATURE_DIM = 104  # 见 build_crystal_features 详细通道对齐表


def get_model_preset(name: str = "smoke") -> Dict[str, Any]:
    """返回标准硬件架构预设。"""
    presets = {
        "smoke": {
            "d_node": 64,
            "n_layers": 4,
            "bond_dim": 4,
            "d_key": 24,
            "max_cluster_size": 6,
            "dropout": 0.0,
            "use_checkpoint": False,
        },
        "a100": {
            "d_node": 160,
            "n_layers": 8,
            "bond_dim": 8,
            "d_key": 48,
            "max_cluster_size": 8,
            "dropout": 0.05,
            "use_checkpoint": False,
        },
    }
    if name not in presets:
        raise ValueError(f"未知预设 {name!r}，可选: {sorted(presets)}")
    return dict(presets[name])


class Ouroboros64Net(nn.Module):
    """
    Phase-SHA V10 边界焊死型 64 轮三维时空晶体神经-符号张量求解器网络。
    """

    def __init__(
        self,
        d_node: int = 64,
        n_layers: int = 4,
        bond_dim: int = 4,
        d_key: int = 24,
        max_cluster_size: int = 6,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        enforce_fp32_determinism()
        self.d_node = int(d_node)
        self.n_layers = int(n_layers)
        self.bond_dim = int(bond_dim)
        self.d_key = int(d_key)
        self.max_cluster_size = int(max_cluster_size)

        # 预注册 SHA-256 K_WORDS[0..63] 与双极常数比特缓冲
        k_tensor = torch.tensor(list(K_WORDS), dtype=torch.int64)
        k_bits = words_to_bits_lsb(k_tensor) * 2.0 - 1.0  # (64, 32)
        self.register_buffer("k_bits_bipolar", k_bits, persistent=False)

        # 输入特征编码投影
        self.input_proj = nn.Sequential(
            nn.Linear(INPUT_FEATURE_DIM, d_node),
            nn.LayerNorm(d_node),
            nn.SiLU(),
            nn.Linear(d_node, d_node),
            nn.LayerNorm(d_node),
        )

        # 神经阻尼 Tensor-BP 骨干网络
        self.backbone = TensorBPStack(
            d_node=d_node,
            n_layers=n_layers,
            bond_dim=bond_dim,
            dropout=dropout,
            use_checkpoint=use_checkpoint,
        )

        # 四大输出头（缺陷愈合 + 进位联合分布 + Wolff 团簇 + CDCL 坍缩/相位 + 全局势能 Critic）
        self.healing_head = DefectHealingHead(d_node=d_node, bond_dim=bond_dim)
        self.wolff_head = WolffClusterProposer(
            d_node=d_node,
            d_key=d_key,
            max_cluster_size=max_cluster_size,
        )
        self.cdcl_head = CDCLBranchPhaseScorer(d_node=d_node)
        self.critic_head = CrystalEnergyCritic(d_node=d_node, bond_dim=bond_dim, extra_scalar_dim=5)

    @staticmethod
    def _to_bipolar(bits_01: torch.Tensor, unknown_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bipolar = bits_01 * 2.0 - 1.0
        if unknown_mask is not None:
            bipolar = bipolar * (1.0 - unknown_mask.to(torch.float32))
        return bipolar

    @torch.no_grad()
    def _build_pos_features(self, rounds: int, device: torch.device) -> torch.Tensor:
        """构建 (1, R, 32, 11) 时空边界位置编码与比特周期坐标。"""
        r_denom = float(max(1, rounds - 1))
        t = torch.arange(rounds, device=device, dtype=torch.float32)
        t_fwd = (t / r_denom).view(1, rounds, 1, 1).expand(1, rounds, 32, 1)
        t_bwd = ((rounds - 1 - t) / r_denom).view(1, rounds, 1, 1).expand(1, rounds, 32, 1)
        is_t0 = (t == 0).to(torch.float32).view(1, rounds, 1, 1).expand(1, rounds, 32, 1)
        is_tRm1 = (t == rounds - 1).to(torch.float32).view(1, rounds, 1, 1).expand(1, rounds, 32, 1)

        b = torch.arange(32, device=device, dtype=torch.float32)
        b_norm = (b / 31.0).view(1, 1, 32, 1).expand(1, rounds, 32, 1)
        harmonics = []
        for freq in (1.0, 2.0, 4.0):
            ang = 2.0 * math.pi * freq * (b / 32.0)
            harmonics.append(torch.sin(ang).view(1, 1, 32, 1).expand(1, rounds, 32, 1))
            harmonics.append(torch.cos(ang).view(1, 1, 32, 1).expand(1, rounds, 32, 1))

        return torch.cat([t_fwd, t_bwd, is_t0, is_tRm1, b_norm, *harmonics], dim=-1)

    def build_crystal_features(
        self,
        H_target: torch.Tensor,
        W: torch.Tensor,
        S: torch.Tensor,
        C: torch.Tensor,
        rounds: int = MAX_ROUNDS,
        unknown_mask: Optional[torch.Tensor] = None,
        conflict_heatmap: Optional[torch.Tensor] = None,
        zero_defects: bool = False,
        crystal_engine: Optional[TorchCrystalEngine] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        将晶体状态与精确 GPU 缺陷能量场组装为 (B, R, 32, 104) 纯 FP32 节点张量。
        同时返回 19 平面局部缺陷信号 (B, R, 19, 32) 与全局势能特征。
        """
        bsz = W.shape[0]
        device = W.device
        eng = crystal_engine or TorchCrystalEngine(device=device)

        with torch.no_grad():
            if zero_defects:
                delta_w = torch.zeros((bsz, rounds), dtype=torch.float32, device=device)
                delta_s = torch.zeros((bsz, rounds + 2, 8), dtype=torch.float32, device=device)
                delta_c = torch.zeros((bsz, rounds, C_PLANES), dtype=torch.float32, device=device)
                e_total = torch.zeros((bsz,), dtype=torch.float32, device=device)
                d_s = torch.zeros((bsz,), dtype=torch.float32, device=device)
                d_w = torch.zeros((bsz,), dtype=torch.float32, device=device)
                d_c = torch.zeros((bsz,), dtype=torch.float32, device=device)
            else:
                eval_res = eng.eval_crystal_batch(H_target, W, S, C, rounds=rounds)
                delta_w = eval_res["delta_W"]
                delta_s = eval_res["delta_S"]
                delta_c = eval_res["delta_C"]
                e_total = eval_res["E"].to(torch.float32)
                d_s = eval_res["dS"].to(torch.float32)
                d_w = eval_res["dW"].to(torch.float32)
                d_c = eval_res["dC"].to(torch.float32)

            # 精确比特展平: (B, R, 32) 等，LSB-first (dim=-1 为 b=0..31)
            w_bits = words_to_bits_lsb(W[:, :rounds])  # (B, R, 32)
            s_cur_bits = words_to_bits_lsb(S[:, :rounds, :])  # (B, R, 8, 32)
            s_next_bits = words_to_bits_lsb(S[:, 1 : rounds + 1, :])  # (B, R, 8, 32) 含 S[R]=H-IV!
            c_bits = words_to_bits_lsb(C[:, :rounds, :])  # (B, R, 10, 32)

            if unknown_mask is None:
                unk = torch.zeros((bsz, rounds, PLANE_COUNT, 32), dtype=torch.float32, device=device)
            else:
                unk = unknown_mask.to(device=device, dtype=torch.float32)

            if conflict_heatmap is None:
                conf = torch.zeros((bsz, rounds, PLANE_COUNT, 32), dtype=torch.float32, device=device)
            else:
                conf = conflict_heatmap.to(device=device, dtype=torch.float32)

            # 对已知位做 {-1, +1} 双极化，对未知位 '?' 严格置 0 并在 unk 通道标记 1
            w_bip = self._to_bipolar(w_bits, unk[:, :, 0, :]).unsqueeze(2)  # (B, R, 1, 32)
            s_cur_bip = self._to_bipolar(s_cur_bits, unk[:, :, 1:9, :])     # (B, R, 8, 32)
            # S_next 在 t < R-1 处对应 S[t+1]；在 t == R-1 处对应焊死 S[R]=H-IV（永不未知）
            unk_next = torch.zeros_like(s_next_bits)
            if rounds > 1:
                unk_next[:, :-1, :, :] = unk[:, 1:, 1:9, :]
            s_next_bip = self._to_bipolar(s_next_bits, unk_next)            # (B, R, 8, 32)
            c_bip = self._to_bipolar(c_bits, unk[:, :, 9:19, :])            # (B, R, 10, 32)

            # 残差位平面：同时传入 delta_S_in (S[t]) 与 delta_S_out (S[t+1]，含末端 S[R] 违约!)
            dw_bits = words_to_bits_lsb(delta_w) if not zero_defects else torch.zeros_like(w_bits)
            ds_in_bits = (
                words_to_bits_lsb(delta_s[:, :rounds, :])
                if not zero_defects
                else torch.zeros_like(s_cur_bits)
            )
            ds_out_bits = (
                words_to_bits_lsb(delta_s[:, 1 : rounds + 1, :])
                if not zero_defects
                else torch.zeros_like(s_next_bits)
            )
            dc_bits = (
                words_to_bits_lsb(delta_c[:, :rounds, :])
                if not zero_defects
                else torch.zeros_like(c_bits)
            )

            # 构建对齐 19 平面的局部缺陷强度信号 (B, R, 19, 32)
            # W[t] 直接参与轮步 T1 压缩，因此同步耦合 delta_W[t] 与 delta_S_out[t, a/e]
            round_defect = (
                dw_bits
                + 0.25 * ds_out_bits.mean(dim=2)
                + 0.15 * dc_bits.mean(dim=2)
            ).clamp(max=1.0)
            w_def = torch.maximum(dw_bits, torch.maximum(ds_out_bits[:, :, 0, :], ds_out_bits[:, :, 4, :]))
            w_def = torch.maximum(w_def, 0.35 * round_defect)
            s_def = torch.maximum(ds_in_bits, 0.35 * round_defect.unsqueeze(2))
            c_def = torch.maximum(dc_bits, 0.25 * round_defect.unsqueeze(2))
            site_defect_signal = torch.cat([w_def.unsqueeze(2), s_def, c_def], dim=2)  # (B, R, 19, 32)

            k_feat = self.k_bits_bipolar[:rounds].view(1, rounds, 1, 32).expand(bsz, rounds, 1, 32)
            pos_feat = self._build_pos_features(rounds, device).expand(bsz, rounds, 32, 11)

            # 将 (B, R, C, 32) 转置拼接为 (B, R, 32, 104)
            plane_feats = torch.cat(
                [
                    w_bip,
                    s_cur_bip,
                    s_next_bip,
                    c_bip,
                    unk,
                    dw_bits.unsqueeze(2),
                    ds_in_bits,
                    ds_out_bits,
                    dc_bits,
                    conf,
                    k_feat,
                ],
                dim=2,
            )  # (B, R, 93, 32)
            node_feat = torch.cat([plane_feats.permute(0, 1, 3, 2), pos_feat], dim=-1)  # (B, R, 32, 104)

        return {
            "node_feat": node_feat,
            "site_defect_signal": site_defect_signal,
            "E": e_total,
            "dS": d_s,
            "dW": d_w,
            "dC": d_c,
        }

    def forward(
        self,
        H_target: torch.Tensor,
        W: torch.Tensor,
        S: torch.Tensor,
        C: torch.Tensor,
        rounds: int = MAX_ROUNDS,
        unknown_mask: Optional[torch.Tensor] = None,
        conflict_heatmap: Optional[torch.Tensor] = None,
        zero_defects: bool = False,
        propose_clusters: bool = False,
        num_clusters: int = 16,
        temperature: float = 1.0,
        deterministic_clusters: bool = False,
        return_entropy: bool = True,
        crystal_engine: Optional[TorchCrystalEngine] = None,
    ) -> Dict[str, Any]:
        """
        前向推理：执行 3D Tensor-BP 并导出缺陷修复头、CDCL 评分头、Critic 价值及（可选）Wolff 团簇提议。
        """
        prep = self.build_crystal_features(
            H_target=H_target,
            W=W,
            S=S,
            C=C,
            rounds=rounds,
            unknown_mask=unknown_mask,
            conflict_heatmap=conflict_heatmap,
            zero_defects=zero_defects,
            crystal_engine=crystal_engine,
        )

        h0 = self.input_proj(prep["node_feat"])
        bp_out = self.backbone(h0, return_entropy=return_entropy)
        h_final = bp_out["H"]
        j_carry = bp_out["J_carry"]
        j_time = bp_out["J_time"]

        flip_logits, pair_logits = self.healing_head(h_final, j_carry)
        collapse_score, phase_logits = self.cdcl_head(h_final)

        s_ent_c = bp_out.get(
            "S_ent_carry",
            torch.zeros((), dtype=torch.float32, device=h_final.device),
        )
        s_ent_t = bp_out.get(
            "S_ent_time",
            torch.zeros((), dtype=torch.float32, device=h_final.device),
        )
        bsz = h_final.shape[0]
        scalar_stats = torch.stack(
            [
                torch.log1p(prep["E"]) / 10.0,
                torch.log1p(prep["dS"]) / 10.0,
                torch.log1p(prep["dW"]) / 10.0,
                s_ent_c.expand(bsz),
                s_ent_t.expand(bsz),
            ],
            dim=-1,
        )
        pred_log_energy = self.critic_head(h_final, j_time, scalar_stats)

        out: Dict[str, Any] = {
            "H": h_final,
            "J_carry": j_carry,
            "J_time": j_time,
            "flip_logits": flip_logits,
            "pair_logits": pair_logits,
            "collapse_score": collapse_score,
            "phase_logits": phase_logits,
            "pred_log_energy": pred_log_energy,
            "site_defect_signal": prep["site_defect_signal"],
            "E_true": prep["E"],
            "dS": prep["dS"],
            "dW": prep["dW"],
            "dC": prep["dC"],
            "S_ent_carry": s_ent_c,
            "S_ent_time": s_ent_t,
        }

        if propose_clusters:
            wolff = self.wolff_head.propose_clusters(
                h=h_final,
                site_defect_signal=prep["site_defect_signal"],
                num_clusters=num_clusters,
                temperature=temperature,
                deterministic=deterministic_clusters,
                flip_logits=flip_logits,
            )
            out["clusters"] = wolff

        return out
