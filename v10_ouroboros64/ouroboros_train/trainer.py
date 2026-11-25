"""
Phase-SHA V10「衔尾蛇-64」原生 64 轮自监督晶体缺陷愈合 + GRPO 训练流与绝对隔离评测套件。

三大严格铁律：
  1. 精度铁律：物理计算全整数 uint32/int64，神经训练/推理死锁纯 FP32（allow_tf32=False）；
  2. 评测隔离：defect_healing_eval 与 blind_inversion_eval 物理分轨；所有盲求逆 SAT 强制
     经过 OuroborosCore.verify_preimage() 独立全量验真，绝不混淆汇报；
  3. 无缓冲实时遥测：所有 print 均 flush=True，遥测 JSON 原子替换写入供 8789 大屏实时读取。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ouroboros_core.cnf_cdcl import CaDiCaLCDCLSolver, SHA256CNFEncoder
from ..ouroboros_core.engine import (
    C_PLANES,
    IV_WORDS,
    MASK32,
    MAX_ROUNDS,
    SYM_0,
    SYM_1,
    SYM_Q,
    OuroborosCore,
    TorchCrystalEngine,
    bits_to_words_np,
    enforce_fp32_determinism,
    get_core,
    words_to_bits_lsb,
    words_to_bits_np,
)
from ..ouroboros_net.heads import (
    PLANE_COUNT,
    build_valid_site_mask,
    coord_to_flat_site,
    flat_site_to_coord,
)
from ..ouroboros_net.model import Ouroboros64Net, get_model_preset


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _sample_random_words(rng: np.random.Generator, batch_size: int, n_words: int = 16) -> np.ndarray:
    return rng.integers(0, 2**32, size=(batch_size, n_words), dtype=np.uint64).astype(np.uint32)


def _corrupt_crystals_np(
    rng: np.random.Generator,
    W_clean: np.ndarray,
    S_clean: np.ndarray,
    C_clean: np.ndarray,
    rounds: int,
    mode: str = "mixed",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    在完全保持边界 S[0]=IV 与 S[R]=H-IV 硬焊死的前提下，向 64 轮晶体内部注入多尺度缺陷：
      - small: 1..8 个孤立或成对位缺陷
      - medium: 12..48 个跨轮缺陷
      - cluster: 模拟多比特协同（连续进位+状态/消息耦合）物理团簇缺陷
    Returns:
      W_dirty, S_dirty, C_dirty, flip_target (B, R, 19, 32) float32
    """
    bsz = W_clean.shape[0]
    W_d = W_clean.copy()
    S_d = S_clean.copy()
    C_d = C_clean.copy()

    # 构建 (R, 19, 32) 合法可变掩码并提取合法扁平索引
    valid_mask_np = build_valid_site_mask(rounds, device="cpu").numpy()
    valid_flat = np.flatnonzero(valid_mask_np.reshape(-1))
    n_valid = len(valid_flat)

    # W / internal S / active C 子集用于构造结构化局部团簇缺陷
    for b_idx in range(bsz):
        sub_mode = mode
        if mode == "mixed":
            p = float(rng.random())
            sub_mode = "small" if p < 0.35 else ("medium" if p < 0.70 else "cluster")

        chosen_flats: List[int] = []
        if sub_mode == "small":
            k = int(rng.integers(2, 10))
            picks = rng.choice(n_valid, size=k, replace=False)
            chosen_flats.extend(valid_flat[picks].tolist())
        elif sub_mode == "medium":
            k = int(rng.integers(14, 52))
            picks = rng.choice(n_valid, size=k, replace=False)
            chosen_flats.extend(valid_flat[picks].tolist())
        else:
            # 注入多个 2..4 bit 时空协同团簇（如 W[t,b] + 同轮/相邻轮进位或状态）
            n_clusters = int(rng.integers(2, 8))
            for _ in range(n_clusters):
                t_c = int(rng.integers(1, max(2, rounds - 1)))
                bit_c = int(rng.integers(1, 31))
                # W[t, bit_c], S[t, a/e, bit_c], C[t, carry, bit_c/bit_c+1]
                cand_coords = [
                    (0, t_c, 0, bit_c),
                    (1, t_c, int(rng.choice([0, 4])), bit_c),
                    (2, t_c, int(rng.integers(3, C_PLANES)), bit_c),
                    (2, t_c, int(rng.integers(3, C_PLANES)), min(31, bit_c + 1)),
                ]
                if t_c + 1 < rounds:
                    cand_coords.append((1, t_c + 1, int(rng.choice([0, 4])), (bit_c + 2) % 32))
                for kind, tt, ww, bb in cand_coords:
                    plane = 0 if kind == 0 else (1 + ww if kind == 1 else 9 + ww)
                    if valid_mask_np[tt, plane, bb]:
                        chosen_flats.append(((tt * PLANE_COUNT) + plane) * 32 + bb)

        for flat in set(chosen_flats):
            bb = flat % 32
            rem = flat // 32
            pp = rem % PLANE_COUNT
            tt = rem // PLANE_COUNT
            bit_mask = np.uint32(1 << bb)
            if pp == 0:
                W_d[b_idx, tt] ^= bit_mask
            elif pp < 9:
                # 双重守护：绝不允许写 S[0] 或 S[R]
                if 1 <= tt < rounds:
                    S_d[b_idx, tt, pp - 1] ^= bit_mask
            else:
                C_d[b_idx, tt, pp - 9] ^= bit_mask

    # 严格生成 19 平面翻转真值标签 (B, R, 19, 32)
    w_diff = words_to_bits_np(W_d ^ W_clean)  # (B, R, 32)
    s_diff = words_to_bits_np(S_d[:, :rounds, :] ^ S_clean[:, :rounds, :])  # (B, R, 8, 32)
    c_diff = words_to_bits_np(C_d ^ C_clean)  # (B, R, 10, 32)
    flip_target = np.concatenate(
        [w_diff[:, :, None, :], s_diff, c_diff],
        axis=2,
    ).astype(np.float32)

    # 断言边界一丝不动
    assert np.array_equal(S_d[:, 0, :], S_clean[:, 0, :]), "致命越界：S[0] 被篡改！"
    assert np.array_equal(S_d[:, rounds, :], S_clean[:, rounds, :]), "致命越界：S[R] 被篡改！"
    return W_d, S_d, C_d, flip_target


def _compute_carry_pair_targets(C_clean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算净态进位平面相邻对 (c_{k,b}, c_{k,b+1}) ∈ {0,1,2,3} 真实标签及活跃掩码。
    返回:
      pair_tgt: (B, R, 31, 10) int64
      pair_active_mask: (1, R, 31, 10) bool （屏蔽 b=0 以及 t<16 的 W 进位）
    """
    c_bits = words_to_bits_lsb(C_clean).to(torch.int64)  # (B, R, 10, 32)
    left = c_bits[:, :, :, :-1]   # (B, R, 10, 31)
    right = c_bits[:, :, :, 1:]   # (B, R, 10, 31)
    pair_idx = (left * 2 + right).permute(0, 1, 3, 2).contiguous()  # (B, R, 31, 10)

    rounds = C_clean.shape[1]
    active = torch.ones((1, rounds, 31, C_PLANES), dtype=torch.bool, device=C_clean.device)
    active[:, :, 0, :] = False  # b=0 恒 0
    if rounds > 0:
        active[:, : min(16, rounds), :, 0:3] = False
    return pair_idx, active


def _find_positive_wolff_clusters(
    core: OuroborosCore,
    H_np: np.ndarray,
    W_np: np.ndarray,
    S_np: np.ndarray,
    C_np: np.ndarray,
    proposed_coords: np.ndarray,
    rounds: int,
    oracle_flip_target: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    混合使用神经网络提议团簇 + 结构化缺陷局部双/三比特组合，经 C++17 OpenMP 精确批量评测，
    筛选 ΔE < 0（且优于单比特最优）的高价值正样本团簇供 Stage A 蒸馏。
    Returns:
      best_pos_coords: (B, K_pos, M_max, 4) int32
      has_pos_mask:    (B, K_pos) bool
      all_delta_e:     (B, K_prop) int64
      hit_rate:        float
    """
    bsz, k_prop, m_max, _ = proposed_coords.shape
    flat_cands = proposed_coords.reshape(bsz * k_prop, m_max, 4)
    H_rep = np.repeat(H_np, k_prop, axis=0)
    W_rep = np.repeat(W_np, k_prop, axis=0)
    S_rep = np.repeat(S_np, k_prop, axis=0)
    C_rep = np.repeat(C_np, k_prop, axis=0)

    delta_e_prop, _ = core.eval_cluster_flips_batch(
        H_rep, W_rep, S_rep, C_rep, flat_cands, rounds=rounds
    )
    delta_e_mat = delta_e_prop.reshape(bsz, k_prop)
    hit_rate = float(np.mean(np.min(delta_e_mat, axis=1) < 0))

    # 同步检查结构化锚点配对团簇，确保蒸馏期始终能捕获跨进位/跨轮降能样例
    eval_cur = core.eval_crystal_batch(H_np, W_np, S_np, C_np, rounds=rounds, return_masks=True)
    dw_mask = eval_cur["delta_W"]
    ds_mask = eval_cur["delta_S"]

    k_pos = 4
    out_coords = np.full((bsz, k_pos, m_max, 4), -1, dtype=np.int32)
    out_coords[:, :, :, 0] = -1
    out_coords[:, :, :, 1:] = 0
    has_pos = np.zeros((bsz, k_pos), dtype=bool)
    valid_mask_np = build_valid_site_mask(rounds, device="cpu").numpy()

    extra_list: List[np.ndarray] = []
    extra_owner: List[int] = []
    for b_idx in range(bsz):
        # 寻找本晶体内最活跃的缺陷轮与比特作为锚点构造物理局部团簇
        bad_rounds_w = np.flatnonzero(dw_mask[b_idx] != 0)
        bad_rounds_s = np.flatnonzero(np.any(ds_mask[b_idx, 1 : rounds + 1] != 0, axis=1))
        cand_rounds = list(dict.fromkeys(list(bad_rounds_w[:3]) + [max(1, min(rounds - 1, int(r))) for r in bad_rounds_s[:3]]))
        if not cand_rounds:
            cand_rounds = [rounds // 2]
        for tt in cand_rounds[:3]:
            if not (1 <= tt < rounds):
                tt = max(1, min(rounds - 1, tt))
            # 识别受损位（或低位进位活跃点）
            w_val = int(dw_mask[b_idx, tt])
            bit = int(np.argmax([(w_val >> bb) & 1 for bb in range(32)])) if w_val != 0 else 1
            bit = max(1, min(30, bit))
            # 候选协同对/三元组：W[t, b] + S[t, 0/4, b] + C[t, c, b+1]
            c1 = np.full((m_max, 4), -1, dtype=np.int32)
            c1[:, 1:] = 0
            pts = [(0, tt, 0, bit), (1, tt, 0, bit)]
            for idx_p, (kk, tr, ww, bb) in enumerate(pts):
                pl = 0 if kk == 0 else (1 + ww if kk == 1 else 9 + ww)
                if valid_mask_np[tr, pl, bb]:
                    c1[idx_p] = (kk, tr, ww, bb)
            if c1[0, 0] >= 0 and c1[1, 0] >= 0:
                extra_list.append(c1)
                extra_owner.append(b_idx)

            c2 = np.full((m_max, 4), -1, dtype=np.int32)
            c2[:, 1:] = 0
            pts2 = [(2, tt, 8, bit), (2, tt, 8, bit + 1), (1, tt, 0, bit)]
            slot = 0
            for kk, tr, ww, bb in pts2:
                pl = 0 if kk == 0 else (1 + ww if kk == 1 else 9 + ww)
                if valid_mask_np[tr, pl, bb] and slot < m_max:
                    c2[slot] = (kk, tr, ww, bb)
                    slot += 1
            if slot >= 2:
                extra_list.append(c2)
                extra_owner.append(b_idx)

        # 仅限 Stage A 自监督蒸馏：从同轮/邻轮真实注毒缺陷集合构造 2..4-bit 协同团簇候选（仍须通过 C++ ΔE < 0 验真）
        if oracle_flip_target is not None:
            pts_bad = np.argwhere(oracle_flip_target[b_idx] > 0.5)
            if len(pts_bad) >= 2:
                by_round: Dict[int, List[Tuple[int, int, int, int]]] = {}
                for tt_i, pl_i, bb_i in pts_bad:
                    tt_i, pl_i, bb_i = int(tt_i), int(pl_i), int(bb_i)
                    if not valid_mask_np[tt_i, pl_i, bb_i]:
                        continue
                    kk = 0 if pl_i == 0 else (1 if pl_i < 9 else 2)
                    ww = 0 if pl_i == 0 else (pl_i - 1 if pl_i < 9 else pl_i - 9)
                    by_round.setdefault(tt_i, []).append((kk, tt_i, ww, bb_i))
                sorted_r = sorted(by_round.items(), key=lambda kv: len(kv[1]), reverse=True)
                n_added_or = 0
                for _, coord_group in sorted_r:
                    if len(coord_group) >= 2 and n_added_or < 4:
                        c_or = np.full((m_max, 4), -1, dtype=np.int32)
                        c_or[:, 1:] = 0
                        for sl_i, item_c in enumerate(coord_group[: min(4, m_max)]):
                            c_or[sl_i] = item_c
                        extra_list.append(c_or)
                        extra_owner.append(b_idx)
                        n_added_or += 1
                # 若跨相邻轮也组一份多比特团簇
                flat_all = [c for _, grp in sorted_r for c in grp]
                if len(flat_all) >= 2 and n_added_or < 4:
                    c_all = np.full((m_max, 4), -1, dtype=np.int32)
                    c_all[:, 1:] = 0
                    for sl_i, item_c in enumerate(flat_all[: min(4, m_max)]):
                        c_all[sl_i] = item_c
                    extra_list.append(c_all)
                    extra_owner.append(b_idx)

    extra_delta = np.zeros((0,), dtype=np.int64)
    if extra_list:
        ex_arr = np.stack(extra_list, axis=0)
        ex_owners = np.asarray(extra_owner, dtype=np.int64)
        extra_delta, _ = core.eval_cluster_flips_batch(
            H_np[ex_owners],
            W_np[ex_owners],
            S_np[ex_owners],
            C_np[ex_owners],
            ex_arr,
            rounds=rounds,
        )

    for b_idx in range(bsz):
        pool: List[Tuple[int, np.ndarray]] = []
        for k_i in range(k_prop):
            de = int(delta_e_mat[b_idx, k_i])
            if de < 0:
                pool.append((de, proposed_coords[b_idx, k_i]))
        for ex_i, own in enumerate(extra_owner):
            if own == b_idx and int(extra_delta[ex_i]) < 0:
                pool.append((int(extra_delta[ex_i]), extra_list[ex_i]))
        # 严格校验不重复且至少 2-bit 非填充才能送入蒸馏
        pool.sort(key=lambda item: item[0])
        added = 0
        for _, arr in pool:
            if added >= k_pos:
                break
            valid_slots = [tuple(arr[s].tolist()) for s in range(m_max) if arr[s, 0] >= 0]
            if len(valid_slots) >= 2 and len(set(valid_slots)) == len(valid_slots):
                out_coords[b_idx, added] = arr
                has_pos[b_idx, added] = True
                added += 1

    return out_coords, has_pos, delta_e_mat, hit_rate


def _random_1bit_candidates(
    rng: np.random.Generator,
    bsz: int,
    k_cands: int,
    rounds: int,
    valid_flat: np.ndarray,
    defect_probs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """为 1-bit WalkSAT 基线生成与神经网络完全等额预算 K 的单比特候选坐标 (B*K, 1, 4)。"""
    n_valid = len(valid_flat)
    out_flat = np.empty((bsz, k_cands), dtype=np.int64)
    for b_idx in range(bsz):
        if defect_probs is not None and float(defect_probs[b_idx].sum()) > 1e-6:
            p = defect_probs[b_idx][valid_flat]
            p = p + 1e-3
            p /= p.sum()
            picks = rng.choice(n_valid, size=k_cands, replace=False, p=p)
        else:
            picks = rng.choice(n_valid, size=k_cands, replace=False)
        out_flat[b_idx] = valid_flat[picks]

    flat_t = torch.from_numpy(out_flat.reshape(-1, 1))
    return flat_site_to_coord(flat_t, rounds=rounds).numpy()


def _apply_cluster_inplace(
    W: np.ndarray,
    S: np.ndarray,
    C: np.ndarray,
    chosen_cluster: np.ndarray,
    rounds: int,
) -> None:
    """对批量晶体逐个应用选定翻转团簇（仅内部位，严守边界焊死）。"""
    bsz, m_max, _ = chosen_cluster.shape
    for b_idx in range(bsz):
        seen = set()
        for slot in range(m_max):
            kind = int(chosen_cluster[b_idx, slot, 0])
            if kind < 0:
                continue
            t = int(chosen_cluster[b_idx, slot, 1])
            w = int(chosen_cluster[b_idx, slot, 2])
            bit = int(chosen_cluster[b_idx, slot, 3])
            key = (kind, t, w, bit)
            if key in seen:
                continue
            seen.add(key)
            mask = np.uint32(1 << bit)
            if kind == 0:
                W[b_idx, t] ^= mask
            elif kind == 1:
                if 1 <= t < rounds:
                    S[b_idx, t, w] ^= mask
            elif kind == 2:
                C[b_idx, t, w] ^= mask


@torch.no_grad()
def evaluate_defect_healing(
    model: Ouroboros64Net,
    core: OuroborosCore,
    torch_eng: TorchCrystalEngine,
    device: torch.device,
    rounds: int = 64,
    n_crystals: int = 4,
    n_steps: int = 20,
    k_budget: int = 24,
    seed: int = 99991,
) -> Dict[str, Any]:
    """
    评测域 1（严格隔离）：晶体缺陷愈合能力（Defect-Healing Eval）。
    已知内部多尺度缺陷，在等 C++ eval 预算 (K = k_budget/步) 下，直接对比：
      - 纯 1-bit WalkSAT 基线
      - V10 Neural Wolff 多比特协同团簇求解器
    """
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)

    w0_15 = _sample_random_words(rng, n_crystals, 16)
    cr = core.compress_exact(w0_15, rounds=rounds)
    W_d0, S_d0, C_d0, _ = _corrupt_crystals_np(
        rng, cr.W, cr.S, cr.C, rounds=rounds, mode="cluster"
    )
    H_np = cr.H_target.copy()
    init_eval = core.eval_crystal_batch(H_np, W_d0, S_d0, C_d0, rounds=rounds, return_masks=False)
    e0_mean = float(np.mean(init_eval["E"]))

    valid_mask_np = build_valid_site_mask(rounds, device="cpu").numpy()
    valid_flat = np.flatnonzero(valid_mask_np.reshape(-1))

    # --- 1) 1-bit WalkSAT 基线 ---
    W_ws, S_ws, C_ws = W_d0.copy(), S_d0.copy(), C_d0.copy()
    ws_curve: List[float] = [e0_mean]
    ws_evals = 0
    for _step in range(n_steps):
        cand_1bit = _random_1bit_candidates(rng, n_crystals, k_budget, rounds, valid_flat)
        ws_evals += n_crystals * k_budget
        de, _ = core.eval_cluster_flips_batch(
            np.repeat(H_np, k_budget, axis=0),
            np.repeat(W_ws, k_budget, axis=0),
            np.repeat(S_ws, k_budget, axis=0),
            np.repeat(C_ws, k_budget, axis=0),
            cand_1bit,
            rounds=rounds,
        )
        de = de.reshape(n_crystals, k_budget)
        cand_mat = cand_1bit.reshape(n_crystals, k_budget, 1, 4)
        chosen = np.full((n_crystals, 1, 4), -1, dtype=np.int32)
        for b_i in range(n_crystals):
            best_k = int(np.argmin(de[b_i]))
            if int(de[b_i, best_k]) <= 0 or float(rng.random()) < 0.08:
                pick_k = best_k if int(de[b_i, best_k]) <= 0 else int(rng.integers(0, k_budget))
                chosen[b_i] = cand_mat[b_i, pick_k]
        _apply_cluster_inplace(W_ws, S_ws, C_ws, chosen, rounds=rounds)
        e_now = core.eval_crystal_batch(H_np, W_ws, S_ws, C_ws, rounds=rounds, return_masks=False)["E"]
        ws_curve.append(float(np.mean(e_now)))

    # --- 2) V10 Neural Wolff 多比特协同团簇 ---
    W_wf, S_wf, C_wf = W_d0.copy(), S_d0.copy(), C_d0.copy()
    H_t = torch.from_numpy(H_np.astype(np.int64)).to(device)
    wf_curve: List[float] = [e0_mean]
    wf_evals = 0
    tunnel_events = 0
    accept_events = 0

    for _step in range(n_steps):
        W_t = torch.from_numpy(W_wf.astype(np.int64)).to(device)
        S_t = torch.from_numpy(S_wf.astype(np.int64)).to(device)
        C_t = torch.from_numpy(C_wf.astype(np.int64)).to(device)
        out = model(
            H_t,
            W_t,
            S_t,
            C_t,
            rounds=rounds,
            propose_clusters=True,
            num_clusters=k_budget,
            temperature=0.85,
            deterministic_clusters=False,
            return_entropy=False,
            crystal_engine=torch_eng,
        )
        clusters_np = out["clusters"]["clusters_coord"].cpu().numpy()  # (B, K, M_max, 4)
        m_max = clusters_np.shape[2]
        wf_evals += n_crystals * k_budget
        de_c, _ = core.eval_cluster_flips_batch(
            np.repeat(H_np, k_budget, axis=0),
            np.repeat(W_wf, k_budget, axis=0),
            np.repeat(S_wf, k_budget, axis=0),
            np.repeat(C_wf, k_budget, axis=0),
            clusters_np.reshape(n_crystals * k_budget, m_max, 4),
            rounds=rounds,
        )
        de_c = de_c.reshape(n_crystals, k_budget)

        # 检查是否发生了“单比特全非负而多比特团簇严格降能”的隧穿事件
        chosen_wf = np.full((n_crystals, m_max, 4), -1, dtype=np.int32)
        for b_i in range(n_crystals):
            best_k = int(np.argmin(de_c[b_i]))
            best_de = int(de_c[b_i, best_k])
            if best_de < 0:
                chosen_wf[b_i] = clusters_np[b_i, best_k]
                accept_events += 1
                # 验证其组成单比特翻转中最优者是否均 >= best_de
                cl = clusters_np[b_i, best_k]
                singles = []
                for sl in range(m_max):
                    if cl[sl, 0] >= 0:
                        s1 = np.full((1, 4), -1, dtype=np.int32)
                        s1[0] = cl[sl]
                        singles.append(s1)
                if singles:
                    s_arr = np.stack(singles, axis=0)
                    de_s, _ = core.eval_cluster_flips_batch(
                        np.repeat(H_np[b_i : b_i + 1], len(singles), axis=0),
                        np.repeat(W_wf[b_i : b_i + 1], len(singles), axis=0),
                        np.repeat(S_wf[b_i : b_i + 1], len(singles), axis=0),
                        np.repeat(C_wf[b_i : b_i + 1], len(singles), axis=0),
                        s_arr,
                        rounds=rounds,
                    )
                    if best_de < int(np.min(de_s)):
                        tunnel_events += 1
            elif best_de == 0 and float(rng.random()) < 0.2:
                chosen_wf[b_i] = clusters_np[b_i, best_k]

        _apply_cluster_inplace(W_wf, S_wf, C_wf, chosen_wf, rounds=rounds)
        e_now_wf = core.eval_crystal_batch(H_np, W_wf, S_wf, C_wf, rounds=rounds, return_masks=False)["E"]
        wf_curve.append(float(np.mean(e_now_wf)))

    e_ws_end_arr = core.eval_crystal_batch(H_np, W_ws, S_ws, C_ws, rounds=rounds, return_masks=False)["E"]
    e_wf_end_arr = core.eval_crystal_batch(H_np, W_wf, S_wf, C_wf, rounds=rounds, return_masks=False)["E"]
    e_ws_end = float(np.mean(e_ws_end_arr))
    e_wf_end = float(np.mean(e_wf_end_arr))

    if was_training:
        model.train()

    return {
        "eval_type": "DEFECT_HEALING_ONLY_ISOLATED",
        "rounds": rounds,
        "n_crystals": n_crystals,
        "n_steps": n_steps,
        "k_budget": k_budget,
        "initial_energy_mean": e0_mean,
        "walksat_1bit": {
            "final_energy_mean": e_ws_end,
            "reduction_pct": float(100.0 * (e0_mean - e_ws_end) / max(1e-6, e0_mean)),
            "exact_healed_rate": float(np.mean(e_ws_end_arr == 0)),
            "cpp_evals": ws_evals,
            "energy_curve": ws_curve,
        },
        "neural_wolff": {
            "final_energy_mean": e_wf_end,
            "reduction_pct": float(100.0 * (e0_mean - e_wf_end) / max(1e-6, e0_mean)),
            "exact_healed_rate": float(np.mean(e_wf_end_arr == 0)),
            "tunneling_events": tunnel_events,
            "accept_events": accept_events,
            "tunneling_ratio": float(tunnel_events / max(1, accept_events)),
            "cpp_evals": wf_evals,
            "energy_curve": wf_curve,
        },
    }


@torch.no_grad()
def evaluate_blind_inversion(
    model: Ouroboros64Net,
    core: OuroborosCore,
    torch_eng: TorchCrystalEngine,
    device: torch.device,
    rounds_list: Tuple[Tuple[Any, ...], ...] = (
        (16, None, "end"),
        (20, 3, "end"),
        (24, 1, "end"),
        (24, 2, "end"),
    ),
    n_cases: int = 2,
    sat_time_limit: float = 4.0,
    seed: int = 88887,
    run_r64_blind_trace: bool = True,
) -> Dict[str, Any]:
    """
    评测域 2（绝对零泄漏盲求逆）：Blind Inversion Eval。
    输入仅包含硬焊死边界 S[0]=IV 与 S[R]=(H-IV) mod 2^32（外加指定的非自由字），
    严禁传入任何真实内部 W/S/C 比特！所有 SAT 宣称解均强制经过 verify_preimage 核验。
    """
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    solver = CaDiCaLCDCLSolver("cadical195")

    cdcl_benchmarks: List[Dict[str, Any]] = []
    for entry in rounds_list:
        if len(entry) == 2:
            r_bench, free_words = int(entry[0]), entry[1]
            pos_mode = "end"
        else:
            r_bench, free_words, pos_mode = int(entry[0]), entry[1], str(entry[2])

        w0_15 = _sample_random_words(rng, n_cases, 16)
        cr = core.compress_exact(w0_15, rounds=r_bench)

        pure_confs: List[int] = []
        pure_times: List[float] = []
        pure_sats = 0

        guided_confs: List[int] = []
        guided_times: List[float] = []
        guided_sats = 0

        for idx in range(n_cases):
            H_single = cr.H_target[idx]
            fixed_words = None
            if free_words is not None:
                if pos_mode == "end":
                    # 释放后端 W[16-free_words .. 15]，固定前端 W[0 .. 15-free_words]
                    fixed_words = {wi: int(w0_15[idx, wi]) for wi in range(16 - int(free_words))}
                else:
                    fixed_words = {wi: int(w0_15[idx, wi]) for wi in range(int(free_words), 16)}

            # 1) 纯 CaDiCaL-195 基准
            res_pure = solver.solve(
                H_single,
                rounds=r_bench,
                fixed_words=fixed_words,
                conflict_budget=250000,
                time_limit_sec=sat_time_limit,
                phase_lits=None,
            )
            if res_pure["status"] == "SAT":
                assert res_pure["valid_preimage"], f"纯 SAT 未通过真值验算 R={r_bench}"
                pure_sats += 1
            pure_confs.append(int(res_pure["conflicts"]))
            pure_times.append(float(res_pure["elapsed_sec"]))

            # 2) V10 神经引导相位 CaDiCaL（绝对零内部真值泄漏！）
            # 构建盲晶体张量：仅填入 S[0]=IV, S[R]=H-IV 及已知 fixed_words，其余全置 0 且 unk=1
            W_blind = np.zeros((1, r_bench), dtype=np.uint32)
            S_blind = np.zeros((1, r_bench + 1, 8), dtype=np.uint32)
            C_blind = np.zeros((1, r_bench, C_PLANES), dtype=np.uint32)
            s0_np = np.asarray(IV_WORDS, dtype=np.uint32)
            sr_np = ((H_single.astype(np.uint64) - s0_np.astype(np.uint64)) & MASK32).astype(np.uint32)
            S_blind[0, 0] = s0_np
            S_blind[0, r_bench] = sr_np

            unk_mask = torch.ones((1, r_bench, PLANE_COUNT, 32), dtype=torch.float32, device=device)
            unk_mask[:, 0, 1:9, :] = 0.0  # S[0]=IV 已知
            if fixed_words is not None:
                for wi, val in fixed_words.items():
                    W_blind[0, wi] = np.uint32(val)
                    unk_mask[:, wi, 0, :] = 0.0

            H_t = torch.from_numpy(H_single[None, :].astype(np.int64)).to(device)
            W_t = torch.from_numpy(W_blind.astype(np.int64)).to(device)
            S_t = torch.from_numpy(S_blind.astype(np.int64)).to(device)
            C_t = torch.from_numpy(C_blind.astype(np.int64)).to(device)

            out = model(
                H_t,
                W_t,
                S_t,
                C_t,
                rounds=r_bench,
                unknown_mask=unk_mask,
                zero_defects=True,
                propose_clusters=False,
                return_entropy=False,
                crystal_engine=torch_eng,
            )
            phase_w = out["phase_logits"][0, :16, 0, :].cpu().numpy()       # (16, 32)
            coll_w = out["collapse_score"][0, :16, 0, :].cpu().numpy()      # (16, 32)

            enc = SHA256CNFEncoder(rounds=r_bench).build(H_single, fixed_words=fixed_words)
            scored_lits: List[Tuple[float, int]] = []
            for wi in range(16):
                if fixed_words is not None and wi in fixed_words:
                    continue
                for b_i in range(32):
                    lit_v = enc.W_vars[wi][b_i]
                    if not isinstance(lit_v, bool) and isinstance(lit_v, int):
                        logit = float(phase_w[wi, b_i])
                        score = float(coll_w[wi, b_i]) * abs(logit)
                        signed = lit_v if logit >= 0.0 else -lit_v
                        scored_lits.append((score, signed))
            scored_lits.sort(key=lambda x: x[0], reverse=True)
            phase_lits = [lit for _, lit in scored_lits[:256]]

            res_guided = solver.solve(
                H_single,
                rounds=r_bench,
                fixed_words=fixed_words,
                conflict_budget=250000,
                time_limit_sec=sat_time_limit,
                phase_lits=phase_lits,
            )
            if res_guided["status"] == "SAT":
                assert res_guided["valid_preimage"], f"引导 SAT 未通过真值验算 R={r_bench}"
                guided_sats += 1
            guided_confs.append(int(res_guided["conflicts"]))
            guided_times.append(float(res_guided["elapsed_sec"]))

        cdcl_benchmarks.append(
            {
                "rounds": r_bench,
                "free_bits": int((free_words if free_words is not None else 16) * 32),
                "pos_mode": pos_mode,
                "cases": n_cases,
                "pure_cadical": {
                    "sat": pure_sats,
                    "mean_conflicts": float(np.mean(pure_confs)),
                    "mean_sec": float(np.mean(pure_times)),
                },
                "v10_guided_cadical": {
                    "sat": guided_sats,
                    "mean_conflicts": float(np.mean(guided_confs)),
                    "mean_sec": float(np.mean(guided_times)),
                },
            }
        )

    # 3) R=64 原生全盲双端焊死求解轨迹（零内部泄漏：内部 W/S/C 全随机猜值初始化）
    r64_blind: Dict[str, Any] = {}
    if run_r64_blind_trace:
        n_b = 2
        w_secret = _sample_random_words(rng, n_b, 16)
        cr_secret = core.compress_exact(w_secret, rounds=64)
        H_blind = cr_secret.H_target.copy()

        # 构造 100% 盲随机晶体：绝不碰 cr_secret.W / cr_secret.S / cr_secret.C！
        W_guess = _sample_random_words(rng, n_b, 64)
        S_guess = _sample_random_words(rng, n_b * 65, 8).reshape(n_b, 65, 8)
        C_guess = _sample_random_words(rng, n_b * 64, C_PLANES).reshape(n_b, 64, C_PLANES)
        C_guess[:, :, :] &= np.uint32(0xFFFFFFFE)  # 已知 b=0 进位为 0
        C_guess[:, :16, 0:3] = 0
        s0_arr = np.asarray(IV_WORDS, dtype=np.uint32)
        sr_arr = ((H_blind.astype(np.uint64) - s0_arr.astype(np.uint64)) & MASK32).astype(np.uint32)
        S_guess[:, 0, :] = s0_arr[None, :]
        S_guess[:, 64, :] = sr_arr

        e_init = core.eval_crystal_batch(H_blind, W_guess, S_guess, C_guess, rounds=64, return_masks=False)["E"]

        # 等预算 1-bit WalkSAT vs Neural Wolff 盲下降对比
        valid_mask_np = build_valid_site_mask(64, device="cpu").numpy()
        valid_flat = np.flatnonzero(valid_mask_np.reshape(-1))
        k_b = 24
        n_b_steps = 12

        W_ws, S_ws, C_ws = W_guess.copy(), S_guess.copy(), C_guess.copy()
        W_wf, S_wf, C_wf = W_guess.copy(), S_guess.copy(), C_guess.copy()
        H_t = torch.from_numpy(H_blind.astype(np.int64)).to(device)

        for _ in range(n_b_steps):
            c_1b = _random_1bit_candidates(rng, n_b, k_b, 64, valid_flat)
            de_1b, _ = core.eval_cluster_flips_batch(
                np.repeat(H_blind, k_b, axis=0),
                np.repeat(W_ws, k_b, axis=0),
                np.repeat(S_ws, k_b, axis=0),
                np.repeat(C_ws, k_b, axis=0),
                c_1b,
                rounds=64,
            )
            de_1b = de_1b.reshape(n_b, k_b)
            c_1b_m = c_1b.reshape(n_b, k_b, 1, 4)
            pick_1b = np.full((n_b, 1, 4), -1, dtype=np.int32)
            for i in range(n_b):
                bk = int(np.argmin(de_1b[i]))
                if int(de_1b[i, bk]) <= 0:
                    pick_1b[i] = c_1b_m[i, bk]
            _apply_cluster_inplace(W_ws, S_ws, C_ws, pick_1b, rounds=64)

            out_b = model(
                H_t,
                torch.from_numpy(W_wf.astype(np.int64)).to(device),
                torch.from_numpy(S_wf.astype(np.int64)).to(device),
                torch.from_numpy(C_wf.astype(np.int64)).to(device),
                rounds=64,
                propose_clusters=True,
                num_clusters=k_b,
                temperature=0.9,
                return_entropy=False,
                crystal_engine=torch_eng,
            )
            cl_b = out_b["clusters"]["clusters_coord"].cpu().numpy()
            de_cb, _ = core.eval_cluster_flips_batch(
                np.repeat(H_blind, k_b, axis=0),
                np.repeat(W_wf, k_b, axis=0),
                np.repeat(S_wf, k_b, axis=0),
                np.repeat(C_wf, k_b, axis=0),
                cl_b.reshape(n_b * k_b, cl_b.shape[2], 4),
                rounds=64,
            )
            de_cb = de_cb.reshape(n_b, k_b)
            pick_cb = np.full((n_b, cl_b.shape[2], 4), -1, dtype=np.int32)
            for i in range(n_b):
                bk = int(np.argmin(de_cb[i]))
                if int(de_cb[i, bk]) <= 0:
                    pick_cb[i] = cl_b[i, bk]
            _apply_cluster_inplace(W_wf, S_wf, C_wf, pick_cb, rounds=64)

        e_ws_f = core.eval_crystal_batch(H_blind, W_ws, S_ws, C_ws, rounds=64, return_masks=False)["E"]
        e_wf_f = core.eval_crystal_batch(H_blind, W_wf, S_wf, C_wf, rounds=64, return_masks=False)["E"]
        sat_ws = bool(np.any(core.verify_preimage(W_ws[:, :16], H_blind, rounds=64)))
        sat_wf = bool(np.any(core.verify_preimage(W_wf[:, :16], H_blind, rounds=64)))

        r64_blind = {
            "rounds": 64,
            "free_bits": 512,
            "n_steps": n_b_steps,
            "k_budget": k_b,
            "initial_energy_mean": float(np.mean(e_init)),
            "walksat_1bit_final_energy": float(np.mean(e_ws_f)),
            "neural_wolff_final_energy": float(np.mean(e_wf_f)),
            "walksat_exact_sat_rate": 1.0 if sat_ws else 0.0,
            "neural_wolff_exact_sat_rate": 1.0 if sat_wf else 0.0,
        }

    if was_training:
        model.train()

    return {
        "eval_type": "BLIND_INVERSION_ZERO_LEAKAGE",
        "cdcl_benchmarks": cdcl_benchmarks,
        "r64_full_blind": r64_blind,
    }


def _extract_64x9_heatmap(
    core: OuroborosCore,
    rounds: int = 64,
    model: Optional[Ouroboros64Net] = None,
    torch_eng: Optional[TorchCrystalEngine] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """生成 64x9 时空晶体缺陷热力图快照供 8789 看板渲染（前 1 列 W，后 8 列 S[0..7]）。"""
    rng = np.random.default_rng(20261124)
    w0 = _sample_random_words(rng, 1, 16)
    cr = core.compress_exact(w0, rounds=rounds)
    W_d, S_d, C_d, _ = _corrupt_crystals_np(rng, cr.W, cr.S, cr.C, rounds=rounds, mode="cluster")
    if model is not None and torch_eng is not None and device is not None:
        # 展示经过 4 步神经网络 Wolff 愈合后的残余缺陷热力图
        model_was_train = model.training
        model.eval()
        H_t = torch.from_numpy(cr.H_target.astype(np.int64)).to(device)
        with torch.no_grad():
            for _ in range(4):
                out = model(
                    H_t,
                    torch.from_numpy(W_d.astype(np.int64)).to(device),
                    torch.from_numpy(S_d.astype(np.int64)).to(device),
                    torch.from_numpy(C_d.astype(np.int64)).to(device),
                    rounds=rounds,
                    propose_clusters=True,
                    num_clusters=16,
                    temperature=0.8,
                    return_entropy=False,
                    crystal_engine=torch_eng,
                )
                cl = out["clusters"]["clusters_coord"].cpu().numpy()  # (1, K, M, 4)
                de, _ = core.eval_cluster_flips_batch(
                    np.repeat(cr.H_target, cl.shape[1], axis=0),
                    np.repeat(W_d, cl.shape[1], axis=0),
                    np.repeat(S_d, cl.shape[1], axis=0),
                    np.repeat(C_d, cl.shape[1], axis=0),
                    cl[0],
                    rounds=rounds,
                )
                bk = int(np.argmin(de))
                if int(de[bk]) < 0:
                    _apply_cluster_inplace(W_d, S_d, C_d, cl[:, bk], rounds=rounds)
        if model_was_train:
            model.train()

    ev = core.eval_crystal_batch(cr.H_target, W_d, S_d, C_d, rounds=rounds, return_masks=True)
    # matrix[t][0] = popcnt(delta_W[t]), matrix[t][1..8] = popcnt(delta_S[t+1, 0..7])
    mat = np.zeros((rounds, 9), dtype=np.int32)
    for t in range(rounds):
        mat[t, 0] = int(bin(int(ev["delta_W"][0, t])).count("1"))
        for w in range(8):
            mat[t, 1 + w] = int(bin(int(ev["delta_S"][0, t + 1, w])).count("1"))

    return {
        "rounds": rounds,
        "E_total": int(ev["E"][0]),
        "welded_S0_ok": bool(np.array_equal(S_d[0, 0], cr.S[0, 0])),
        "welded_SR_ok": bool(np.array_equal(S_d[0, rounds], cr.S[0, rounds])),
        "columns": ["W", "a", "b", "c", "d", "e", "f", "g", "h"],
        "matrix": mat.tolist(),
    }


def _evaluate_pair_entanglement_diagnostic(
    model: Ouroboros64Net,
    core: OuroborosCore,
    torch_eng: TorchCrystalEngine,
    device: torch.device,
    rounds: int = 64,
    seed: int = 54321,
) -> Dict[str, float]:
    """
    墙 3 诊断：检验 pair_logits 是否实质优于秩-1 独立边缘乘积 p(c_b) * p(c_{b+1})。
    """
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    w0 = _sample_random_words(rng, 4, 16)
    cr = core.compress_exact(w0, rounds=rounds)
    W_d, S_d, C_d, _ = _corrupt_crystals_np(rng, cr.W, cr.S, cr.C, rounds=rounds, mode="medium")

    with torch.no_grad():
        H_t = torch.from_numpy(cr.H_target.astype(np.int64)).to(device)
        W_t = torch.from_numpy(W_d.astype(np.int64)).to(device)
        S_t = torch.from_numpy(S_d.astype(np.int64)).to(device)
        C_t = torch.from_numpy(C_d.astype(np.int64)).to(device)
        C_clean_t = torch.from_numpy(cr.C.astype(np.int64)).to(device)

        out = model(H_t, W_t, S_t, C_t, rounds=rounds, return_entropy=True, crystal_engine=torch_eng)
        pair_logits = out["pair_logits"]  # (B, R, 31, 10, 4)
        pair_probs = torch.softmax(pair_logits, dim=-1).clamp(min=1e-7)

        # 由同一联合分布抽取其边缘乘积 P_rank1(a, b) = P(a) * P(b)
        p_left_1 = pair_probs[..., 2] + pair_probs[..., 3]
        p_left_0 = 1.0 - p_left_1
        p_right_1 = pair_probs[..., 1] + pair_probs[..., 3]
        p_right_0 = 1.0 - p_right_1
        rank1_probs = torch.stack(
            [p_left_0 * p_right_0, p_left_0 * p_right_1, p_left_1 * p_right_0, p_left_1 * p_right_1],
            dim=-1,
        ).clamp(min=1e-7)

        pair_tgt, active_mask = _compute_carry_pair_targets(C_clean_t)
        mask_flat = active_mask.expand_as(pair_tgt).reshape(-1)

        nll_joint = F.nll_loss(torch.log(pair_probs).reshape(-1, 4)[mask_flat], pair_tgt.reshape(-1)[mask_flat])
        nll_rank1 = F.nll_loss(torch.log(rank1_probs).reshape(-1, 4)[mask_flat], pair_tgt.reshape(-1)[mask_flat])
        # 非秩-1 协同互信息耦合 KL(P_joint || P_left * P_right)
        kl_non_rank1 = torch.sum(pair_probs * (torch.log(pair_probs) - torch.log(rank1_probs)), dim=-1)
        kl_mean = float(kl_non_rank1[active_mask.expand_as(kl_non_rank1)].mean().item())
        acc_joint = float((torch.argmax(pair_probs, dim=-1).reshape(-1)[mask_flat] == pair_tgt.reshape(-1)[mask_flat]).float().mean().item())

    if was_training:
        model.train()

    return {
        "pair_joint_nll": float(nll_joint.item()),
        "pair_rank1_marginal_nll": float(nll_rank1.item()),
        "nll_gain_over_rank1": float((nll_rank1 - nll_joint).item()),
        "non_rank1_mutual_info_kl": kl_mean,
        "pair_joint_acc": acc_joint,
        "s_ent_carry": float(out["S_ent_carry"].item()),
        "s_ent_time": float(out["S_ent_time"].item()),
    }


def train_ouroboros(args: argparse.Namespace) -> Dict[str, Any]:
    """执行 Stage A -> Stage B -> Stage C 三阶段完整训练并在全流程隔离评测与记录遥测。"""
    enforce_fp32_determinism()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    core = get_core()
    torch_eng = TorchCrystalEngine(device=device)

    preset_cfg = get_model_preset(args.preset)
    if args.use_checkpoint == "true":
        preset_cfg["use_checkpoint"] = True
    elif args.use_checkpoint == "false":
        preset_cfg["use_checkpoint"] = False
    model = Ouroboros64Net(**preset_cfg).to(device=device, dtype=torch.float32)

    n_params = sum(p.numel() for p in model.parameters())
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = out_dir / "telemetry_v10.json"

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.stage_a_steps + args.stage_b_steps + args.stage_c_steps
    warmup_steps = max(1, min(100, total_steps // 10))

    def _lr_lambda(cur_step: int) -> float:
        if cur_step < warmup_steps:
            return float(cur_step + 1) / float(warmup_steps)
        denom = max(1, total_steps - warmup_steps)
        prog = min(1.0, float(cur_step - warmup_steps) / float(denom))
        return max(0.04, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    best_heal_pct = -1e9

    def save_checkpoint(filename: str, g_step: int) -> Path:
        p_ckpt = ckpt_dir / filename
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "preset": args.preset,
                "preset_cfg": preset_cfg,
                "rounds": args.rounds,
                "step": g_step,
            },
            p_ckpt,
        )
        return p_ckpt

    rng = np.random.default_rng(args.seed)
    valid_mask_t = build_valid_site_mask(args.rounds, device=device)  # (R, 19, 32)
    _, pair_active_mask_t = _compute_carry_pair_targets(
        torch.zeros((1, args.rounds, C_PLANES), dtype=torch.int64, device=device)
    )

    t_start = time.perf_counter()
    log_tail: List[str] = []
    history: List[Dict[str, Any]] = []
    heal_history: List[Dict[str, Any]] = []
    blind_history: List[Dict[str, Any]] = []

    def log_msg(msg: str) -> None:
        stamp = f"[{time.perf_counter() - t_start:7.1f}s]"
        line = f"{stamp} {msg}"
        print(line, flush=True)
        log_tail.append(line)
        if len(log_tail) > 120:
            del log_tail[:-120]

    def flush_telemetry(phase: str, g_step: int, status: str = "running") -> None:
        vram_alloc = float(torch.cuda.memory_allocated(device) / 1e6) if device.type == "cuda" else 0.0
        vram_peak = float(torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else 0.0
        payload = {
            "meta": {
                "project": "Phase-SHA V10 Ouroboros-64",
                "preset": args.preset,
                "rounds": args.rounds,
                "device": str(device),
                "gpu_name": gpu_name,
                "n_params": n_params,
                "fp32_locked": (not torch.backends.cuda.matmul.allow_tf32),
                "phase": phase,
                "status": status,
                "step": g_step,
                "total_steps": total_steps,
                "elapsed_sec": round(time.perf_counter() - t_start, 2),
                "vram_alloc_mb": round(vram_alloc, 1),
                "vram_peak_mb": round(vram_peak, 1),
            },
            "history": history[-300:],
            "defect_healing_eval": {
                "latest": heal_history[-1] if heal_history else None,
                "history": heal_history,
            },
            "blind_inversion_eval": {
                "latest": blind_history[-1] if blind_history else None,
                "history": blind_history,
            },
            "crystal_heatmap": _extract_64x9_heatmap(
                core, rounds=args.rounds, model=model, torch_eng=torch_eng, device=device
            ),
            "log_tail": log_tail[-60:],
        }
        _atomic_write_json(telemetry_path, payload)

    log_msg(
        f"启动 Ouroboros-64 训练 | preset={args.preset} | params={n_params:,} | "
        f"device={gpu_name} | R={args.rounds} | allow_tf32={torch.backends.cuda.matmul.allow_tf32}"
    )

    # 初始基线对比（Step 0 Before Training）
    ent_diag_init = _evaluate_pair_entanglement_diagnostic(model, core, torch_eng, device, rounds=args.rounds)
    heal_init = evaluate_defect_healing(
        model, core, torch_eng, device, rounds=args.rounds,
        n_crystals=3 if args.smoke else 8,
        n_steps=16 if args.smoke else 32,
        k_budget=16 if args.smoke else 32,
    )
    heal_init["step"] = 0
    heal_history.append(heal_init)
    blind_init = evaluate_blind_inversion(
        model, core, torch_eng, device,
        n_cases=2 if args.smoke else 4,
        sat_time_limit=2.5 if args.smoke else 5.0,
        run_r64_blind_trace=True,
    )
    blind_init["step"] = 0
    blind_history.append(blind_init)
    log_msg(
        f"[Step 0 基线] WalkSAT均能={heal_init['walksat_1bit']['final_energy_mean']:.1f} | "
        f"未训Wolff均能={heal_init['neural_wolff']['final_energy_mean']:.1f} | "
        f"PairAcc={ent_diag_init['pair_joint_acc']:.4f} | "
        f"S_ent_carry={ent_diag_init['s_ent_carry']:.3f}"
    )
    flush_telemetry("INIT", 0)

    def run_periodic_eval(stage_label: str, g_step: int) -> None:
        nonlocal best_heal_pct
        ent_now = _evaluate_pair_entanglement_diagnostic(model, core, torch_eng, device, rounds=args.rounds)
        heal_now = evaluate_defect_healing(
            model, core, torch_eng, device, rounds=args.rounds,
            n_crystals=3 if args.smoke else 8,
            n_steps=16 if args.smoke else 32,
            k_budget=16 if args.smoke else 32,
        )
        heal_now["step"] = g_step
        heal_now["pair_joint_acc"] = round(ent_now["pair_joint_acc"], 4)
        heal_now["nll_gain_over_rank1"] = round(ent_now["nll_gain_over_rank1"], 4)
        heal_history.append(heal_now)
        blind_now = evaluate_blind_inversion(
            model, core, torch_eng, device,
            n_cases=2 if args.smoke else 4,
            sat_time_limit=2.5 if args.smoke else 5.0,
            run_r64_blind_trace=True,
        )
        blind_now["step"] = g_step
        blind_history.append(blind_now)
        red_pct = float(heal_now["neural_wolff"]["reduction_pct"])
        save_checkpoint("latest.pt", g_step)
        if red_pct >= best_heal_pct:
            best_heal_pct = red_pct
            save_checkpoint("best_healing.pt", g_step)
        log_msg(
            f"[Eval@{g_step} | {stage_label}] Wolff降能={red_pct:.1f}% (WalkSAT={heal_now['walksat_1bit']['reduction_pct']:.1f}%) | "
            f"PairAcc={ent_now['pair_joint_acc']:.4f} | KL_gain={ent_now['nll_gain_over_rank1']:+.4f} | "
            f"S_ent_c={ent_now['s_ent_carry']:.3f}"
        )
        flush_telemetry(stage_label, g_step)

    global_step = 0
    bsz_a = args.batch_size
    k_wolff_train = 8 if args.smoke else 16

    # =========================================================================
    # Stage A: 全 64 轮自监督晶体缺陷愈合 + 进位纠缠联合学习 + Wolff 正团簇蒸馏
    # =========================================================================
    model.train()
    stage_a_losses: List[float] = []
    stage_a_wolff_hits: List[float] = []

    for step_a in range(1, args.stage_a_steps + 1):
        global_step += 1
        w0 = _sample_random_words(rng, bsz_a, 16)
        cr = core.compress_exact(w0, rounds=args.rounds)
        W_d, S_d, C_d, flip_tgt_np = _corrupt_crystals_np(
            rng, cr.W, cr.S, cr.C, rounds=args.rounds, mode="mixed"
        )

        H_t = torch.from_numpy(cr.H_target.astype(np.int64)).to(device)
        W_t = torch.from_numpy(W_d.astype(np.int64)).to(device)
        S_t = torch.from_numpy(S_d.astype(np.int64)).to(device)
        C_t = torch.from_numpy(C_d.astype(np.int64)).to(device)
        C_clean_t = torch.from_numpy(cr.C.astype(np.int64)).to(device)
        flip_tgt_t = torch.from_numpy(flip_tgt_np).to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(
            H_t, W_t, S_t, C_t,
            rounds=args.rounds,
            propose_clusters=True,
            num_clusters=k_wolff_train,
            temperature=1.0,
            return_entropy=True,
            crystal_engine=torch_eng,
        )

        # 1) 缺陷逐比特掩码动态平衡 BCE 损失（收敛上限 35，抑制假阳性泛滥）
        m_valid = valid_mask_t.unsqueeze(0).expand_as(flip_tgt_t)
        f_pred = out["flip_logits"][m_valid]
        f_true = flip_tgt_t[m_valid]
        pos_cnt = float(f_true.sum().item())
        neg_cnt = float(f_true.numel() - pos_cnt)
        dyn_pos_w = min(35.0, max(8.0, neg_cnt / max(1.0, pos_cnt)))
        pos_w = torch.tensor(dyn_pos_w, dtype=torch.float32, device=device)
        loss_flip = F.binary_cross_entropy_with_logits(f_pred, f_true, pos_weight=pos_w)

        with torch.no_grad():
            pred_pos = f_pred > 0.0
            tp = float((pred_pos & (f_true > 0.5)).sum().item())
            fp = float((pred_pos & (f_true <= 0.5)).sum().item())
            fn = float((~pred_pos & (f_true > 0.5)).sum().item())
            defect_recall = tp / max(1.0, tp + fn)
            defect_precision = tp / max(1.0, tp + fp)
            defect_f1 = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)
            k_top = max(1, min(int(f_pred.numel()), int(round(2.0 * max(1.0, pos_cnt)))))
            topk_idx = torch.topk(f_pred, k=k_top).indices
            topk_recall = float(f_true[topk_idx].sum().item()) / max(1.0, pos_cnt)

        # 2) 相邻进位对 4 态联合分布监督（强迫 J_carry 捕捉真实非秩-1 纠缠）
        pair_tgt, _ = _compute_carry_pair_targets(C_clean_t)
        p_mask = pair_active_mask_t.expand_as(pair_tgt)
        loss_pair = F.cross_entropy(
            out["pair_logits"][p_mask], pair_tgt[p_mask]
        )

        # 3) Critic 全局势能拟合
        e_tgt = torch.log1p(out["E_true"])
        loss_critic = F.smooth_l1_loss(out["pred_log_energy"], e_tgt)

        # 4) C++17 正向 Wolff 团簇似然蒸馏（筛选真实 ΔE < 0 协同团簇；仅 Stage A 训练使用自监督缺陷团簇）
        with torch.no_grad():
            prop_np = out["clusters"]["clusters_coord"].cpu().numpy()
            pos_coords_np, has_pos_np, _, hit_rate = _find_positive_wolff_clusters(
                core, cr.H_target, W_d, S_d, C_d, prop_np, rounds=args.rounds,
                oracle_flip_target=flip_tgt_np,
            )
        if np.any(has_pos_np):
            pos_coords_t = torch.from_numpy(pos_coords_np).to(device)
            has_pos_t = torch.from_numpy(has_pos_np).to(device)
            pos_logp = model.wolff_head.score_given_clusters(
                out["H"],
                out["site_defect_signal"],
                pos_coords_t,
                flip_logits=out["flip_logits"],
            )
            # 提升正团簇蒸馏权重以克服 38,912 位点初期软最大化稀释
            loss_wolff = -(pos_logp[has_pos_t].clamp(min=-30.0)).mean() * 0.25
        else:
            loss_wolff = torch.zeros((), dtype=torch.float32, device=device)

        total_loss = loss_flip + 0.8 * loss_pair + 0.3 * loss_critic + loss_wolff
        total_loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 2.0).item())
        optimizer.step()
        scheduler.step()

        stage_a_losses.append(float(total_loss.item()))
        stage_a_wolff_hits.append(hit_rate)
        history.append(
            {
                "stage": "A_HEAL",
                "step": global_step,
                "total_loss": round(float(total_loss.item()), 4),
                "flip_loss": round(float(loss_flip.item()), 4),
                "pair_loss": round(float(loss_pair.item()), 4),
                "wolff_loss": round(float(loss_wolff.item()), 4),
                "critic_loss": round(float(loss_critic.item()), 4),
                "wolff_hit_rate": round(hit_rate, 4),
                "grpo_reward": None,
                "tunnel_rate": None,
                "s_ent_carry": round(float(out["S_ent_carry"].item()), 4),
                "s_ent_time": round(float(out["S_ent_time"].item()), 4),
                "defect_recall": round(defect_recall, 4),
                "defect_f1": round(defect_f1, 4),
                "topk_recall": round(topk_recall, 4),
                "grad_norm": round(grad_norm, 3),
            }
        )
        log_interval_a = max(1, min(50, args.stage_a_steps // 6))
        if step_a % log_interval_a == 0 or step_a == args.stage_a_steps:
            log_msg(
                f"[Stage A {step_a}/{args.stage_a_steps}] loss={total_loss.item():.4f} "
                f"(flip={loss_flip.item():.3f}, f1={defect_f1:.3f}, topk_rec={topk_recall:.3f}, pair={loss_pair.item():.3f}, wolff={loss_wolff.item():.3f}) | "
                f"wolff_hit={hit_rate*100:.1f}% | S_ent_c={out['S_ent_carry'].item():.3f}"
            )
            flush_telemetry("Stage A: 缺陷愈合", global_step)
        if args.eval_every > 0 and step_a % args.eval_every == 0 and step_a < args.stage_a_steps:
            run_periodic_eval("Stage A: 阶段评测", global_step)

    if args.stage_a_steps > 0:
        run_periodic_eval("Stage A->B 切换评测", global_step)

    # =========================================================================
    # Stage B: R∈{16,20,24}->64 跨轮课程 50%->93% 强掩蔽传播坍缩探测
    # =========================================================================
    stage_b_losses: List[float] = []
    stage_b_corrs: List[float] = []
    n_probe_crystals = min(2, bsz_a) if args.smoke else min(4, bsz_a)
    n_probes_per_crystal = 6 if args.smoke else 10
    warmup_b_steps = int(0.45 * args.stage_b_steps)

    for step_b in range(1, args.stage_b_steps + 1):
        global_step += 1
        frac = float(step_b) / max(1, args.stage_b_steps)
        reveal_prob = 0.50 - 0.43 * frac  # 从 50% 已知平滑降至 7% 已知（93% 未知！）
        cur_r = (16, 20, 24)[step_b % 3] if step_b <= warmup_b_steps else args.rounds

        w0 = _sample_random_words(rng, n_probe_crystals, 16)
        cr = core.compress_exact(w0, rounds=cur_r)

        W_vis = np.zeros((n_probe_crystals, cur_r), dtype=np.uint32)
        S_vis = np.zeros((n_probe_crystals, cur_r + 1, 8), dtype=np.uint32)
        C_vis = np.zeros((n_probe_crystals, cur_r, C_PLANES), dtype=np.uint32)
        unk_np = np.ones((n_probe_crystals, cur_r, PLANE_COUNT, 32), dtype=np.float32)

        probe_coords: List[Tuple[int, int, int, float, float]] = []  # (b_i, wi, bit, true_bit, target_log_collapse)
        for b_i in range(n_probe_crystals):
            W_sym = np.full((cur_r, 32), SYM_Q, dtype=np.uint8)
            S_sym = np.full((cur_r + 1, 8, 32), SYM_Q, dtype=np.uint8)
            C_sym = np.full((cur_r, C_PLANES, 32), SYM_Q, dtype=np.uint8)
            # 焊死双端边界
            S_sym[0] = words_to_bits_np(cr.S[b_i, 0])
            S_sym[cur_r] = words_to_bits_np(cr.S[b_i, cur_r])
            S_vis[b_i, 0] = cr.S[b_i, 0]
            S_vis[b_i, cur_r] = cr.S[b_i, cur_r]
            unk_np[b_i, 0, 1:9, :] = 0.0

            w_clean_bits = words_to_bits_np(cr.W[b_i, :16])  # (16, 32)
            rev_mask = rng.random((16, 32)) < reveal_prob
            for wi in range(16):
                for bb in range(32):
                    if rev_mask[wi, bb]:
                        W_sym[wi, bb] = SYM_1 if w_clean_bits[wi, bb] else SYM_0

            # 先经平面 B ARX 双向传播闭合既有已知约束（毫秒级，避免重复构建 RREF 阻塞 GPU）
            prop0 = core.propagate_symbols(cr.H_target[b_i], W_sym, S_sym, C_sym, rounds=cur_r, use_gf2=False, max_iters=6)
            W_s0, S_s0, C_s0 = prop0["W_sym"], prop0["S_sym"], prop0["C_sym"]

            # 写入可视掩码晶体
            W_known = np.where((W_s0 == SYM_0) | (W_s0 == SYM_1), W_s0 == SYM_1, 0).astype(np.uint8)
            S_known = np.where((S_s0[: cur_r] == SYM_0) | (S_s0[: cur_r] == SYM_1), S_s0[: cur_r] == SYM_1, 0).astype(np.uint8)
            C_known = np.where((C_s0 == SYM_0) | (C_s0 == SYM_1), C_s0 == SYM_1, 0).astype(np.uint8)
            W_vis[b_i] = bits_to_words_np(W_known)
            S_vis[b_i, : cur_r] = bits_to_words_np(S_known)
            C_vis[b_i] = bits_to_words_np(C_known)

            unk_np[b_i, :, 0, :] = ((W_s0 != SYM_0) & (W_s0 != SYM_1)).astype(np.float32)
            unk_np[b_i, :, 1:9, :] = ((S_s0[: cur_r] != SYM_0) & (S_s0[: cur_r] != SYM_1)).astype(np.float32)
            unk_np[b_i, :, 9:19, :] = ((C_s0 != SYM_0) & (C_s0 != SYM_1)).astype(np.float32)
            unk_np[b_i, 0, 1:9, :] = 0.0

            # 在靠近 S[0]=IV (W[0..3]) 与远端 (W[4..15]) 之间分层抽样未知比特，探测真实坍缩收益方差
            unk_w_pairs = np.argwhere(W_s0[:16] == SYM_Q)
            if len(unk_w_pairs) > 0:
                near_pairs = unk_w_pairs[unk_w_pairs[:, 0] < 4]
                far_pairs = unk_w_pairs[unk_w_pairs[:, 0] >= 4]
                selected_rows: List[np.ndarray] = []
                n_near = min(len(near_pairs), max(1, n_probes_per_crystal // 2))
                if n_near > 0:
                    idx_n = rng.choice(len(near_pairs), size=n_near, replace=False)
                    selected_rows.extend(near_pairs[idx_n])
                n_far = min(len(far_pairs), n_probes_per_crystal - len(selected_rows))
                if n_far > 0:
                    idx_f = rng.choice(len(far_pairs), size=n_far, replace=False)
                    selected_rows.extend(far_pairs[idx_f])
                for p_row in selected_rows:
                    wi_p, bb_p = int(p_row[0]), int(p_row[1])
                    true_val = int(w_clean_bits[wi_p, bb_p])
                    W_probe = W_s0.copy()
                    W_probe[wi_p, bb_p] = SYM_1 if true_val else SYM_0
                    pr = core.propagate_symbols(
                        cr.H_target[b_i], W_probe, S_s0.copy(), C_s0.copy(),
                        rounds=cur_r, use_gf2=False, max_iters=3
                    )
                    gain = int(pr["unit_prop_deductions"])
                    probe_coords.append((b_i, wi_p, bb_p, float(true_val), float(math.log1p(max(0, gain)))))

        H_t = torch.from_numpy(cr.H_target.astype(np.int64)).to(device)
        W_t = torch.from_numpy(W_vis.astype(np.int64)).to(device)
        S_t = torch.from_numpy(S_vis.astype(np.int64)).to(device)
        C_t = torch.from_numpy(C_vis.astype(np.int64)).to(device)
        unk_t = torch.from_numpy(unk_np).to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(
            H_t, W_t, S_t, C_t,
            rounds=cur_r,
            unknown_mask=unk_t,
            zero_defects=True,
            propose_clusters=False,
            return_entropy=True,
            crystal_engine=torch_eng,
        )

        # 全 W[0..15] 相位真值辅助监督 + 探针稀疏坍缩得分精确回归
        w_phase_logits = out["phase_logits"][:, :16, 0, :]
        w_phase_tgt = words_to_bits_lsb(torch.from_numpy(cr.W[:, :16].astype(np.int64)).to(device))
        w_unk_mask = unk_t[:, :16, 0, :] > 0.5
        if torch.any(w_unk_mask):
            loss_phase = F.binary_cross_entropy_with_logits(w_phase_logits[w_unk_mask], w_phase_tgt[w_unk_mask])
        else:
            loss_phase = torch.zeros((), dtype=torch.float32, device=device)

        corr_val = 0.0
        if probe_coords:
            b_idx_t = torch.tensor([x[0] for x in probe_coords], dtype=torch.long, device=device)
            w_idx_t = torch.tensor([x[1] for x in probe_coords], dtype=torch.long, device=device)
            bit_idx_t = torch.tensor([x[2] for x in probe_coords], dtype=torch.long, device=device)
            tgt_col_t = torch.tensor([x[4] for x in probe_coords], dtype=torch.float32, device=device)
            pred_col_t = out["collapse_score"][b_idx_t, w_idx_t, 0, bit_idx_t]
            loss_collapse = F.smooth_l1_loss(pred_col_t, tgt_col_t)
            if pred_col_t.numel() >= 4 and float(tgt_col_t.std().item()) > 1e-4 and float(pred_col_t.std().item()) > 1e-4:
                vx = pred_col_t - pred_col_t.mean()
                vy = tgt_col_t - tgt_col_t.mean()
                corr_val = float((torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx * vx) * torch.sum(vy * vy)) + 1e-8)).item())
        else:
            loss_collapse = torch.zeros((), dtype=torch.float32, device=device)

        loss_b = loss_phase + 0.8 * loss_collapse
        loss_b.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 2.0).item())
        optimizer.step()
        scheduler.step()

        stage_b_losses.append(float(loss_b.item()))
        stage_b_corrs.append(corr_val)
        history.append(
            {
                "stage": "B_COLLAPSE",
                "step": global_step,
                "rounds_cur": cur_r,
                "total_loss": round(float(loss_b.item()), 4),
                "flip_loss": round(float(loss_phase.item()), 4),
                "pair_loss": round(float(loss_collapse.item()), 4),
                "wolff_loss": None,
                "critic_loss": None,
                "wolff_hit_rate": None,
                "grpo_reward": None,
                "tunnel_rate": round(corr_val, 4),
                "s_ent_carry": round(float(out["S_ent_carry"].item()), 4),
                "s_ent_time": round(float(out["S_ent_time"].item()), 4),
                "grad_norm": round(grad_norm, 3),
            }
        )
        log_interval_b = max(1, min(40, args.stage_b_steps // 4))
        if step_b % log_interval_b == 0 or step_b == args.stage_b_steps:
            log_msg(
                f"[Stage B {step_b}/{args.stage_b_steps} R={cur_r}] mask={100*(1-reveal_prob):.0f}% | "
                f"loss={loss_b.item():.4f} (phase={loss_phase.item():.3f}, collapse={loss_collapse.item():.3f}) | "
                f"corr={corr_val:+.3f}"
            )
            flush_telemetry("Stage B: 坍缩探针", global_step)

    if args.stage_b_steps > 0:
        run_periodic_eval("Stage B->C 切换评测", global_step)

    # =========================================================================
    # Stage C: Solver-in-the-Loop GRPO 在线强化学习 + 20% 纠缠稳态回放
    # =========================================================================
    bsz_c = max(2, args.batch_size // 2)
    g_rollouts = 8 if args.smoke else 12
    stage_c_rewards: List[float] = []
    stage_c_tunnel_rates: List[float] = []

    for step_c in range(1, args.stage_c_steps + 1):
        global_step += 1
        w0 = _sample_random_words(rng, bsz_c, 16)
        cr = core.compress_exact(w0, rounds=args.rounds)
        W_d, S_d, C_d, flip_tgt_c_np = _corrupt_crystals_np(
            rng, cr.W, cr.S, cr.C, rounds=args.rounds, mode="cluster"
        )
        e_init_np = core.eval_crystal_batch(cr.H_target, W_d, S_d, C_d, rounds=args.rounds, return_masks=False)["E"]

        H_t = torch.from_numpy(cr.H_target.astype(np.int64)).to(device)
        W_t = torch.from_numpy(W_d.astype(np.int64)).to(device)
        S_t = torch.from_numpy(S_d.astype(np.int64)).to(device)
        C_t = torch.from_numpy(C_d.astype(np.int64)).to(device)
        C_clean_c_t = torch.from_numpy(cr.C.astype(np.int64)).to(device)
        flip_tgt_c_t = torch.from_numpy(flip_tgt_c_np).to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(
            H_t, W_t, S_t, C_t,
            rounds=args.rounds,
            propose_clusters=True,
            num_clusters=g_rollouts,
            temperature=0.95,
            deterministic_clusters=False,
            return_entropy=True,
            crystal_engine=torch_eng,
        )
        log_prob = out["clusters"]["log_prob"]  # (B, G)

        with torch.no_grad():
            cl_np = out["clusters"]["clusters_coord"].cpu().numpy()  # (B, G, M_max, 4)
            m_max = cl_np.shape[2]
            de_c, new_e_c = core.eval_cluster_flips_batch(
                np.repeat(cr.H_target, g_rollouts, axis=0),
                np.repeat(W_d, g_rollouts, axis=0),
                np.repeat(S_d, g_rollouts, axis=0),
                np.repeat(C_d, g_rollouts, axis=0),
                cl_np.reshape(bsz_c * g_rollouts, m_max, 4),
                rounds=args.rounds,
            )
            de_mat = de_c.reshape(bsz_c, g_rollouts)
            new_e_mat = new_e_c.reshape(bsz_c, g_rollouts)

            # 计算单比特组成最优，用于精确认定多比特协同跨垒 R_tunnel
            rewards_np = np.zeros((bsz_c, g_rollouts), dtype=np.float32)
            tunnel_hits = 0
            for b_i in range(bsz_c):
                e0 = max(1.0, float(e_init_np[b_i]) * 0.04)
                # 收集该批量内所有活跃成员单比特坐标，统一批量评估节省调用开销
                s_coords = []
                s_map: List[Tuple[int, int]] = []
                for g_i in range(g_rollouts):
                    for sl in range(m_max):
                        if cl_np[b_i, g_i, sl, 0] >= 0:
                            arr1 = np.full((1, 4), -1, dtype=np.int32)
                            arr1[0] = cl_np[b_i, g_i, sl]
                            s_coords.append(arr1)
                            s_map.append((g_i, sl))
                best_single_per_g = np.full((g_rollouts,), 10**9, dtype=np.int64)
                if s_coords:
                    s_arr = np.stack(s_coords, axis=0)
                    de_s, _ = core.eval_cluster_flips_batch(
                        np.repeat(cr.H_target[b_i : b_i + 1], len(s_coords), axis=0),
                        np.repeat(W_d[b_i : b_i + 1], len(s_coords), axis=0),
                        np.repeat(S_d[b_i : b_i + 1], len(s_coords), axis=0),
                        np.repeat(C_d[b_i : b_i + 1], len(s_coords), axis=0),
                        s_arr,
                        rounds=args.rounds,
                    )
                    for idx_s, (gi, _) in enumerate(s_map):
                        if int(de_s[idx_s]) < int(best_single_per_g[gi]):
                            best_single_per_g[gi] = int(de_s[idx_s])

                for g_i in range(g_rollouts):
                    de_val = int(de_mat[b_i, g_i])
                    # 取消绝对上限封顶，改用对数尺度压降与严格跨垒增量，根治组内满奖同分导致 adv=0
                    r_heal = float(math.copysign(math.log1p(abs(de_val)), -de_val))
                    r_tunnel = 0.0
                    bs_val = int(best_single_per_g[g_i])
                    if de_val < 0 and de_val < bs_val:
                        tunnel_hits += 1
                        r_tunnel = float(1.5 + math.log1p(bs_val - de_val))
                    elif de_val < 0 and de_val == bs_val:
                        r_tunnel = 0.6
                    r_solve = 12.0 if int(new_e_mat[b_i, g_i]) == 0 else 0.0
                    rewards_np[b_i, g_i] = r_heal + r_tunnel + r_solve

            r_mean = rewards_np.mean(axis=1, keepdims=True)
            r_std = rewards_np.std(axis=1, keepdims=True) + 1e-5
            adv_np = (rewards_np - r_mean) / r_std

        adv_t = torch.from_numpy(adv_np).to(device)
        loss_grpo = -(adv_t * log_prob.clamp(min=-40.0)).mean()
        loss_critic_c = F.smooth_l1_loss(out["pred_log_energy"], torch.log1p(out["E_true"]))

        # 混入 20% Stage A 进位纠缠+定位回放，防止纯 RL 下 S_ent_carry 塌缩
        pair_tgt_c, _ = _compute_carry_pair_targets(C_clean_c_t)
        p_mask_c = pair_active_mask_t.expand_as(pair_tgt_c)
        loss_pair_c = F.cross_entropy(out["pair_logits"][p_mask_c], pair_tgt_c[p_mask_c])
        m_valid_c = valid_mask_t.unsqueeze(0).expand_as(flip_tgt_c_t)
        f_pred_c = out["flip_logits"][m_valid_c]
        f_true_c = flip_tgt_c_t[m_valid_c]
        pw_c = min(35.0, max(8.0, float((f_true_c.numel() - f_true_c.sum().item()) / max(1.0, float(f_true_c.sum().item())))))
        loss_flip_c = F.binary_cross_entropy_with_logits(
            f_pred_c, f_true_c, pos_weight=torch.tensor(pw_c, dtype=torch.float32, device=device)
        )
        loss_replay = 0.20 * (loss_pair_c + 0.5 * loss_flip_c)

        loss_c = loss_grpo + 0.15 * loss_critic_c + loss_replay
        loss_c.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 2.0).item())
        optimizer.step()
        scheduler.step()

        mean_r = float(rewards_np.mean())
        t_rate = float(tunnel_hits / max(1, bsz_c * g_rollouts))
        stage_c_rewards.append(mean_r)
        stage_c_tunnel_rates.append(t_rate)

        history.append(
            {
                "stage": "C_GRPO",
                "step": global_step,
                "total_loss": round(float(loss_c.item()), 4),
                "flip_loss": round(float(loss_flip_c.item()), 4),
                "pair_loss": round(float(loss_pair_c.item()), 4),
                "wolff_loss": round(float(loss_grpo.item()), 4),
                "critic_loss": round(float(loss_critic_c.item()), 4),
                "wolff_hit_rate": round(float(np.mean(np.min(de_mat, axis=1) < 0)), 4),
                "grpo_reward": round(mean_r, 4),
                "tunnel_rate": round(t_rate, 4),
                "s_ent_carry": round(float(out["S_ent_carry"].item()), 4),
                "s_ent_time": round(float(out["S_ent_time"].item()), 4),
                "grad_norm": round(grad_norm, 3),
            }
        )
        log_interval_c = max(1, min(40, args.stage_c_steps // 4))
        if step_c % log_interval_c == 0 or step_c == args.stage_c_steps:
            log_msg(
                f"[Stage C {step_c}/{args.stage_c_steps}] reward={mean_r:+.3f} | "
                f"tunnel_rate={t_rate*100:.1f}% | wolff_hit={float(np.mean(np.min(de_mat, axis=1)<0))*100:.1f}% | "
                f"pair_c={loss_pair_c.item():.3f} | S_ent_c={out['S_ent_carry'].item():.3f}"
            )
            flush_telemetry("Stage C: GRPO强化", global_step)
        if args.eval_every > 0 and step_c % args.eval_every == 0 and step_c < args.stage_c_steps:
            run_periodic_eval("Stage C: 阶段评测", global_step)

    # =========================================================================
    # 训练完成最终双域严格隔离评测与墙 3 诊断对比
    # =========================================================================
    ent_diag_final = _evaluate_pair_entanglement_diagnostic(model, core, torch_eng, device, rounds=args.rounds)
    heal_final = evaluate_defect_healing(
        model, core, torch_eng, device, rounds=args.rounds,
        n_crystals=3 if args.smoke else 8,
        n_steps=16 if args.smoke else 32,
        k_budget=16 if args.smoke else 32,
    )
    heal_final["step"] = global_step
    heal_history.append(heal_final)

    blind_final = evaluate_blind_inversion(
        model, core, torch_eng, device,
        n_cases=2 if args.smoke else 4,
        sat_time_limit=2.5 if args.smoke else 5.0,
        run_r64_blind_trace=True,
    )
    blind_final["step"] = global_step
    blind_history.append(blind_final)

    ckpt_path = save_checkpoint(f"ouroboros_{args.preset}_step{global_step}.pt", global_step)
    save_checkpoint("latest.pt", global_step)
    if float(heal_final["neural_wolff"]["reduction_pct"]) >= best_heal_pct:
        best_heal_pct = float(heal_final["neural_wolff"]["reduction_pct"])
        save_checkpoint("best_healing.pt", global_step)

    elapsed_total = time.perf_counter() - t_start
    vram_peak_mb = float(torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else 0.0

    # 构建深度定量诊断分析报告（用于定量分析冒烟效果并外推 A100）
    analysis_summary = {
        "meta": {
            "preset": args.preset,
            "rounds": args.rounds,
            "n_params": n_params,
            "device": str(device),
            "gpu_name": gpu_name,
            "elapsed_sec": round(elapsed_total, 2),
            "steps_per_sec": round(total_steps / max(1e-6, elapsed_total), 3),
            "vram_peak_mb": round(vram_peak_mb, 1),
            "checkpoint_path": str(ckpt_path),
        },
        "stage_a": {
            "initial_loss": round(stage_a_losses[0], 4) if stage_a_losses else None,
            "final_loss": round(stage_a_losses[-1], 4) if stage_a_losses else None,
            "mean_wolff_hit_early": round(float(np.mean(stage_a_wolff_hits[: max(1, len(stage_a_wolff_hits)//4)])), 4) if stage_a_wolff_hits else None,
            "mean_wolff_hit_late": round(float(np.mean(stage_a_wolff_hits[-max(1, len(stage_a_wolff_hits)//4):])), 4) if stage_a_wolff_hits else None,
        },
        "wall3_entanglement": {
            "before": ent_diag_init,
            "after": ent_diag_final,
        },
        "stage_b": {
            "initial_loss": round(stage_b_losses[0], 4) if stage_b_losses else None,
            "final_loss": round(stage_b_losses[-1], 4) if stage_b_losses else None,
            "mean_collapse_corr_late": round(float(np.mean(stage_b_corrs[-max(1, len(stage_b_corrs)//4):])), 4) if stage_b_corrs else None,
        },
        "stage_c": {
            "mean_reward_early": round(float(np.mean(stage_c_rewards[: max(1, len(stage_c_rewards)//3)])), 4) if stage_c_rewards else None,
            "mean_reward_late": round(float(np.mean(stage_c_rewards[-max(1, len(stage_c_rewards)//3):])), 4) if stage_c_rewards else None,
            "mean_tunnel_rate_early": round(float(np.mean(stage_c_tunnel_rates[: max(1, len(stage_c_tunnel_rates)//3)])), 4) if stage_c_tunnel_rates else None,
            "mean_tunnel_rate_late": round(float(np.mean(stage_c_tunnel_rates[-max(1, len(stage_c_tunnel_rates)//3):])), 4) if stage_c_tunnel_rates else None,
        },
        "defect_healing_before": heal_init,
        "defect_healing_after": heal_final,
        "blind_inversion_before": blind_init,
        "blind_inversion_after": blind_final,
    }
    _atomic_write_json(out_dir / "smoke_analysis.json", analysis_summary)
    log_msg(
        f"全阶段训练完成 | 耗时 {elapsed_total:.1f}s | 峰值显存 {vram_peak_mb:.1f} MB | "
        f"愈合前后均能: {heal_init['neural_wolff']['final_energy_mean']:.1f} -> "
        f"{heal_final['neural_wolff']['final_energy_mean']:.1f} "
        f"(WalkSAT基线={heal_final['walksat_1bit']['final_energy_mean']:.1f})"
    )
    flush_telemetry("COMPLETED", global_step, status="completed")
    return analysis_summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-SHA V10 Ouroboros-64 Trainer")
    p.add_argument("--out_dir", type=str, default="v10_ouroboros64/runs/local_smoke")
    p.add_argument("--preset", type=str, default="smoke", choices=["smoke", "a100"])
    p.add_argument("--rounds", type=int, default=64)
    p.add_argument("--stage_a_steps", type=int, default=60)
    p.add_argument("--stage_b_steps", type=int, default=20)
    p.add_argument("--stage_c_steps", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=400)
    p.add_argument("--use_checkpoint", type=str, default="auto", choices=["auto", "true", "false"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train_ouroboros(args)
