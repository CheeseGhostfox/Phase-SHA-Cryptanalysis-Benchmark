"""
Phase-SHA V10 Milestone 2 单元测试：
验证纯 FP32 锁死、R=64 零 NaN 前后向、Tensor-BP 纠缠熵范围、
WolffClusterProposer 焊死边界零越界、C++17 协同评估闭环与 6 符号 MCTS 环境。
"""

import math
import numpy as np
import pytest
import torch

from v10_ouroboros64.ouroboros_core.engine import (
    C_PLANES,
    SYM_0,
    SYM_1,
    SYM_DASH,
    SYM_N,
    SYM_Q,
    SYM_U,
    TorchCrystalEngine,
    get_core,
    words_to_bits_np,
)
from v10_ouroboros64.ouroboros_mcts.symbol_env import GeneralizedSymbolEnv
from v10_ouroboros64.ouroboros_net.heads import (
    coord_to_flat_site,
    flat_site_to_coord,
)
from v10_ouroboros64.ouroboros_net.model import Ouroboros64Net, get_model_preset


def test_fp32_lock_and_r64_forward_backward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = get_core()
    torch_eng = TorchCrystalEngine(device=device)
    cfg = get_model_preset("smoke")
    model = Ouroboros64Net(**cfg).to(device=device, dtype=torch.float32)

    # 1. 验证纯 FP32 锁定
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    for p in model.parameters():
        assert p.dtype == torch.float32

    # 2. 构造原生 R=64 晶体
    rng = np.random.default_rng(12345)
    w0 = rng.integers(0, 2**32, size=(2, 16), dtype=np.uint64).astype(np.uint32)
    cr = core.compress_exact(w0, rounds=64)

    H_t = torch.from_numpy(cr.H_target.astype(np.int64)).to(device)
    W_t = torch.from_numpy(cr.W.astype(np.int64)).to(device)
    S_t = torch.from_numpy(cr.S.astype(np.int64)).to(device)
    C_t = torch.from_numpy(cr.C.astype(np.int64)).to(device)

    out = model(
        H_t,
        W_t,
        S_t,
        C_t,
        rounds=64,
        propose_clusters=True,
        num_clusters=6,
        return_entropy=True,
        crystal_engine=torch_eng,
    )

    assert out["flip_logits"].shape == (2, 64, 19, 32)
    assert out["pair_logits"].shape == (2, 64, 31, C_PLANES, 4)
    assert out["collapse_score"].shape == (2, 64, 19, 32)
    assert out["phase_logits"].shape == (2, 64, 19, 32)
    assert out["pred_log_energy"].shape == (2,)
    assert out["clusters"]["clusters_coord"].shape == (2, 6, cfg["max_cluster_size"], 4)
    assert out["clusters"]["log_prob"].shape == (2, 6)

    # 纠缠熵严格落在 [0, ln(chi)]
    max_ent = math.log(cfg["bond_dim"]) + 1e-4
    s_c = float(out["S_ent_carry"].item())
    s_t = float(out["S_ent_time"].item())
    assert 0.0 <= s_c <= max_ent
    assert 0.0 <= s_t <= max_ent

    # 全头联立反向传播零 NaN
    loss = (
        out["flip_logits"].pow(2).mean()
        + out["pair_logits"].pow(2).mean()
        + out["collapse_score"].mean()
        + out["phase_logits"].pow(2).mean()
        + out["pred_log_energy"].mean()
        - out["clusters"]["log_prob"].mean()
    )
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"参数 {name} 无梯度回传"
        assert torch.isfinite(p.grad).all(), f"参数 {name} 出现 NaN/Inf 梯度"


def test_wolff_boundary_mask_bijection_and_cpp_bridge():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = get_core()
    torch_eng = TorchCrystalEngine(device=device)
    cfg = get_model_preset("smoke")
    model = Ouroboros64Net(**cfg).to(device=device, dtype=torch.float32)

    rng = np.random.default_rng(777)
    w0 = rng.integers(0, 2**32, size=(2, 16), dtype=np.uint64).astype(np.uint32)
    cr = core.compress_exact(w0, rounds=64)
    # 注入内部一个比特翻转制造活跃缺陷信号
    W_d = cr.W.copy()
    W_d[:, 3] ^= np.uint32(1 << 5)

    out = model(
        torch.from_numpy(cr.H_target.astype(np.int64)).to(device),
        torch.from_numpy(W_d.astype(np.int64)).to(device),
        torch.from_numpy(cr.S.astype(np.int64)).to(device),
        torch.from_numpy(cr.C.astype(np.int64)).to(device),
        rounds=64,
        propose_clusters=True,
        num_clusters=24,
        return_entropy=False,
        crystal_engine=torch_eng,
    )
    coords = out["clusters"]["clusters_coord"]  # (2, 24, M, 4)
    flat = out["clusters"]["clusters_flat"]

    # 坐标 <-> 扁平索引 严格双射校验
    roundtrip_flat = coord_to_flat_site(coords)
    assert torch.equal(roundtrip_flat, flat)
    roundtrip_coord = flat_site_to_coord(roundtrip_flat, rounds=64)
    assert torch.equal(roundtrip_coord, coords)

    # 校验所有提议团簇：1) m >= 2；2) 团簇内无重复；3) 绝不触碰 S[0]/S[64] 与常量进位位
    coords_np = coords.cpu().numpy()
    for b_i in range(2):
        for k_i in range(24):
            slots = []
            for sl in range(coords_np.shape[2]):
                kk, tt, ww, bb = [int(v) for v in coords_np[b_i, k_i, sl]]
                if kk < 0:
                    continue
                if kk == 1:
                    assert 1 <= tt < 64, f"非法篡改硬焊死边界 S[{tt}]"
                if kk == 2:
                    assert bb != 0, "非法提议进位常量 bit 0"
                    if tt < 16:
                        assert ww >= 3, f"非法提议未使用消息进位 C[{tt},{ww}]"
                slots.append((kk, tt, ww, bb))
            assert len(slots) >= 2
            assert len(set(slots)) == len(slots), "团簇内出现重复比特"

    # 验证 score_given_clusters 计算的条件对数似然与采样时完全一致
    re_logp = model.wolff_head.score_given_clusters(
        out["H"], out["site_defect_signal"], coords, flip_logits=out["flip_logits"]
    )
    assert torch.allclose(re_logp, out["clusters"]["log_prob"], atol=1e-4, rtol=1e-4)

    # 验证 C++17 eval_cluster_flips_batch 无缝接收提议坐标
    k_total = 2 * 24
    delta_e, new_e = core.eval_cluster_flips_batch(
        np.repeat(cr.H_target, 24, axis=0),
        np.repeat(W_d, 24, axis=0),
        np.repeat(cr.S, 24, axis=0),
        np.repeat(cr.C, 24, axis=0),
        coords_np.reshape(k_total, coords_np.shape[2], 4),
        rounds=64,
    )
    assert delta_e.shape == (k_total,)
    assert new_e.shape == (k_total,)


def test_generalized_symbol_env_track_a():
    core = get_core()
    rng = np.random.default_rng(2026)
    w0 = rng.integers(0, 2**32, size=(1, 16), dtype=np.uint64).astype(np.uint32)
    cr = core.compress_exact(w0, rounds=64)

    env = GeneralizedSymbolEnv(H_target=cr.H_target[0], rounds=64, splice_round=32, core=core)
    stats0 = env.get_stats()
    assert stats0["splice_freedom_bits"] == 256
    assert stats0["active_carry_gates"] > 0

    # 硬焊边界保护：尝试写入 S[0] 或 S[64] 必须触发 PermissionError
    with pytest.raises(PermissionError):
        env.set_symbol(1, 0, 0, 3, SYM_0)
    with pytest.raises(PermissionError):
        env.set_symbol(1, 64, 0, 3, SYM_1)
    with pytest.raises(PermissionError):
        env.set_symbol(2, 10, 4, 0, SYM_1)

    # 写入轨 A 差分符号 '-', 'u', 'n' 并保留中性比特掩码
    res_diff = env.set_symbol(0, 20, 0, 5, SYM_DASH, is_neutral=True, run_propagate=False)
    assert res_diff.conflict is False
    env.set_symbol(0, 21, 0, 6, SYM_U, is_neutral=False, run_propagate=False)
    env.set_symbol(1, 32, 0, 7, SYM_N, is_neutral=False, run_propagate=False)
    stats_mid = env.get_stats()
    assert stats_mid["neutral_bits"] == 1
    assert stats_mid["symbol_counts"]["-"] == 1
    assert stats_mid["symbol_counts"]["u"] == 1
    assert stats_mid["symbol_counts"]["n"] == 1

    # 填入完整合法原像 W[0..15]，检查双平面零退相干传播无冲突闭合全部 W[0..63] 与 S[32]
    w_bits = words_to_bits_np(cr.W[0, :16])
    for wi in range(16):
        for bb in range(32):
            env.set_symbol(0, wi, 0, bb, SYM_1 if w_bits[wi, bb] else SYM_0, run_propagate=False)
    res_prop = env.propagate(use_gf2=True, max_iters=16)
    assert res_prop.conflict is False
    assert env.splice_freedom_bits() == 0
