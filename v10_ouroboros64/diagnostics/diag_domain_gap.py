"""
R2 只读诊断脚本 1：域 A（已知内部缺陷愈合）vs 域 B（严格零泄漏盲求逆）结构性与条件分布差异诊断。

功能覆盖（只读加载 best_m3v2.pt 与 ouroboros_a100_step4000.pt，不修改任何权重或存档）：
  1. 显式调用 enforce_fp32_determinism() 并校验 allow_tf32 == False；
  2. input_proj[0].weight (160, 104) 通道组 L2 范数剖析；
  3. 域 A vs 域 B 104 通道与 site_defect_signal 非零率及统计量对比；
  4. phase_logits 二元熵 / BCE / 置信度 / |logit|>=0.08 比例 / collapse_score 分布及与真实 CaDiCaL 前瞻的相关性；
  5. PairAcc 与「拷贝输入 C_d」哑基线对比（含未翻转对 vs 真实翻转对互斥子集拆分，及域 B 对比）；
  6. 单/双向消融实验：
     (a) 域 A 剥离 zero_defects=True 对 flip_logits F1/TopK 召回与 Wolff 降能命中率的影响；
     (b) 域 B Stage B 随机独立掩蔽 vs make_end_free_constraints 末尾连续未知掩蔽分布错位对比。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pysat.solvers import Solver

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v10_ouroboros64.ouroboros_core.cnf_cdcl import (
    SHA256CNFEncoder,
    make_end_free_constraints,
)
from v10_ouroboros64.ouroboros_core.engine import (
    C_PLANES,
    IV_WORDS,
    MASK32,
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
from v10_ouroboros64.ouroboros_mcts.dynamic_cdcl import DynamicNeuralCDCLSolver
from v10_ouroboros64.ouroboros_mcts.hybrid_solver import load_v10_checkpoint
from v10_ouroboros64.ouroboros_net.heads import PLANE_COUNT, build_valid_site_mask
from v10_ouroboros64.ouroboros_net.model import Ouroboros64Net
from v10_ouroboros64.ouroboros_train.m3_trainer import _build_blind_crystal_inputs
from v10_ouroboros64.ouroboros_train.trainer import (
    _compute_carry_pair_targets,
    _corrupt_crystals_np,
    _sample_random_words,
)


DEFAULT_M3V2_CKPT = "v10_ouroboros64/runs/a100_m3v2/checkpoints/best_m3v2.pt"
DEFAULT_M2_CKPT = "v10_ouroboros64/runs/a100_m2/checkpoints/ouroboros_a100_step4000.pt"
DEFAULT_OUT_JSON = "v10_ouroboros64/diagnostics/domain_gap_report.json"


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _check_fp32_locked() -> Dict[str, Any]:
    enforce_fp32_determinism()
    cuda_avail = torch.cuda.is_available()
    matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32) if cuda_avail else False
    cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32) if cuda_avail else False
    assert not matmul_tf32 and not cudnn_tf32, (
        f"TF32 未禁用! matmul={matmul_tf32}, cudnn={cudnn_tf32}"
    )
    return {
        "cuda_available": cuda_avail,
        "allow_tf32_matmul": matmul_tf32,
        "allow_tf32_cudnn": cudnn_tf32,
        "fp32_locked": (not matmul_tf32) and (not cudnn_tf32),
    }


def _analyze_input_proj_norms(model: Ouroboros64Net) -> Dict[str, Any]:
    w = model.input_proj[0].weight.detach().cpu().to(torch.float32)  # (d_node, 104)
    col_norms = torch.linalg.vector_norm(w, ord=2, dim=0).numpy()  # (104,)
    fine_groups = {
        "w_bip_[0]": (0, 1),
        "s_cur_bip_[1..8]": (1, 9),
        "s_next_bip_[9..16]": (9, 17),
        "c_bip_[17..26]": (17, 27),
        "unk_[27..45]": (27, 46),
        "dw_[46]": (46, 47),
        "ds_in_[47..54]": (47, 55),
        "ds_out_[55..62]": (55, 63),
        "dc_[63..72]": (63, 73),
        "conf_W_[73]": (73, 74),
        "conf_SC_[74..91]": (74, 92),
        "k_feat_[92]": (92, 93),
        "pos_feat_[93..103]": (93, 104),
    }
    coarse_groups = {
        "bipolar_[0..26]": (0, 27),
        "unk_[27..45]": (27, 46),
        "delta_[46..72]": (46, 73),
        "conf_W_[73]": (73, 74),
        "conf_SC_[74..91]": (74, 92),
        "k_pos_[92..103]": (92, 104),
    }

    def _summarize(groups: Dict[str, Tuple[int, int]]) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        for name, (lo, hi) in groups.items():
            sl = col_norms[lo:hi]
            res[name] = {
                "dims": hi - lo,
                "l2_mean": round(float(np.mean(sl)), 6),
                "l2_std": round(float(np.std(sl)), 6),
                "l2_max": round(float(np.max(sl)), 6),
                "l2_min": round(float(np.min(sl)), 6),
            }
        return res

    return {
        "weight_shape": list(w.shape),
        "overall_col_l2_mean": round(float(np.mean(col_norms)), 6),
        "coarse_groups": _summarize(coarse_groups),
        "fine_groups": _summarize(fine_groups),
    }


def _summarize_104_channels(
    node_feat: torch.Tensor,
    site_defect_signal: torch.Tensor,
) -> Dict[str, Any]:
    nf = node_feat.detach().cpu().numpy()  # (B, R, 32, 104)
    sd = site_defect_signal.detach().cpu().numpy()  # (B, R, 19, 32)
    groups = {
        "w_bip_[0]": (0, 1),
        "s_cur_bip_[1..8]": (1, 9),
        "s_next_bip_[9..16]": (9, 17),
        "c_bip_[17..26]": (17, 27),
        "unk_[27..45]": (27, 46),
        "dw_[46]": (46, 47),
        "ds_in_[47..54]": (47, 55),
        "ds_out_[55..62]": (55, 63),
        "dc_[63..72]": (63, 73),
        "conf_W_[73]": (73, 74),
        "conf_SC_[74..91]": (74, 92),
        "k_feat_[92]": (92, 93),
        "pos_feat_[93..103]": (93, 104),
    }
    ch_stats: Dict[str, Any] = {}
    for gname, (lo, hi) in groups.items():
        sub = nf[..., lo:hi]
        ch_stats[gname] = {
            "nonzero_ratio": round(float(np.mean(np.abs(sub) > 1e-8)), 6),
            "mean": round(float(np.mean(sub)), 6),
            "std": round(float(np.std(sub)), 6),
        }
    return {
        "shape": list(nf.shape),
        "channels": ch_stats,
        "site_defect_signal_nonzero_ratio": round(float(np.mean(np.abs(sd) > 1e-8)), 6),
        "site_defect_signal_mean": round(float(np.mean(sd)), 6),
    }


def _build_dyn_cdcl_partial_inputs(
    H_single: np.ndarray,
    rounds: int,
    fixed_words: Optional[Dict[int, int]],
    fixed_bits: Optional[Dict[Tuple[int, int], int]],
    device: torch.device,
    stamp_heat_demo: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], np.ndarray]:
    """完全复刻 DynamicNeuralCDCLSolver._neural_score_partial 的输入构造（无内部泄漏）。"""
    W_b = np.zeros((1, rounds), dtype=np.uint32)
    S_b = np.zeros((1, rounds + 1, 8), dtype=np.uint32)
    C_b = np.zeros((1, rounds, C_PLANES), dtype=np.uint32)
    s0 = np.asarray(IV_WORDS, dtype=np.uint32)
    sr = ((H_single.astype(np.uint64) - s0.astype(np.uint64)) & MASK32).astype(np.uint32)
    S_b[0, 0] = s0
    S_b[0, rounds] = sr

    unk = torch.ones((1, rounds, PLANE_COUNT, 32), dtype=torch.float32, device=device)
    unk[:, 0, 1:9, :] = 0.0
    free_mask_w = np.ones((16, 32), dtype=bool)
    for wi, val in (fixed_words or {}).items():
        u = int(val) & MASK32
        W_b[0, wi] = np.uint32(u)
        unk[0, wi, 0, :] = 0.0
        if wi < 16:
            free_mask_w[wi, :] = False
    for (wi, bb), bval in (fixed_bits or {}).items():
        if int(bval) & 1:
            W_b[0, wi] |= np.uint32(1 << int(bb))
        else:
            W_b[0, wi] &= np.uint32(~(1 << int(bb)) & MASK32)
        unk[0, wi, 0, int(bb)] = 0.0
        if wi < 16:
            free_mask_w[wi, int(bb)] = False

    conf_t: Optional[torch.Tensor] = None
    if stamp_heat_demo:
        heatmap = np.zeros((rounds, PLANE_COUNT, 32), dtype=np.float32)
        free_coords = np.argwhere(free_mask_w)
        for wi_c, bb_c in free_coords[:8]:
            DynamicNeuralCDCLSolver._stamp_heat(heatmap, int(wi_c), int(bb_c), rounds, weight=1.0)
        conf_t = torch.from_numpy(np.log1p(np.maximum(0.0, heatmap))[None, ...]).to(device)

    return (
        torch.from_numpy(H_single[None, :].astype(np.int64)).to(device),
        torch.from_numpy(W_b.astype(np.int64)).to(device),
        torch.from_numpy(S_b.astype(np.int64)).to(device),
        torch.from_numpy(C_b.astype(np.int64)).to(device),
        unk,
        conf_t,
        free_mask_w,
    )


@torch.no_grad()
def _measure_channel_stats(
    model: Ouroboros64Net,
    core: OuroborosCore,
    torch_eng: TorchCrystalEngine,
    device: torch.device,
    n_crystals: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    results: Dict[str, Any] = {}

    # 域 A (R=64): cluster & medium
    w64 = _sample_random_words(rng, n_crystals, 16)
    cr64 = core.compress_exact(w64, rounds=64)
    H64_t = torch.from_numpy(cr64.H_target.astype(np.int64)).to(device)
    for mode in ("cluster", "medium"):
        W_d, S_d, C_d, _ = _corrupt_crystals_np(rng, cr64.W, cr64.S, cr64.C, rounds=64, mode=mode)
        prep = model.build_crystal_features(
            H64_t,
            torch.from_numpy(W_d.astype(np.int64)).to(device),
            torch.from_numpy(S_d.astype(np.int64)).to(device),
            torch.from_numpy(C_d.astype(np.int64)).to(device),
            rounds=64,
            unknown_mask=None,
            conflict_heatmap=None,
            zero_defects=False,
            crystal_engine=torch_eng,
        )
        results[f"domain_A_R64_{mode}"] = _summarize_104_channels(
            prep["node_feat"], prep["site_defect_signal"]
        )

    # 域 B: ARX 预处理（_build_blind_crystal_inputs）以及 DynCDCL 模拟（Episode 1 vs Episode>=2 含热力图）
    for r_b, fb in ((24, 40), (25, 24), (64, 512)):
        w_b = _sample_random_words(rng, n_crystals, 16)
        cr_b = core.compress_exact(w_b, rounds=r_b)
        nf_arx_list, sd_arx_list = [], []
        nf_ep1_list, sd_ep1_list = [], []
        nf_ep2_list, sd_ep2_list = [], []
        for idx in range(n_crystals):
            fw, fbits = (None, None) if fb >= 512 else make_end_free_constraints(w_b[idx], fb)
            H_t, W_t, S_t, C_t, unk_t, _, _ = _build_blind_crystal_inputs(
                core, cr_b.H_target[idx], r_b, fw, fbits, device
            )
            prep_arx = model.build_crystal_features(
                H_t, W_t, S_t, C_t, rounds=r_b, unknown_mask=unk_t,
                conflict_heatmap=None, zero_defects=True, crystal_engine=torch_eng,
            )
            nf_arx_list.append(prep_arx["node_feat"])
            sd_arx_list.append(prep_arx["site_defect_signal"])

            H1, W1, S1, C1, u1, _, _ = _build_dyn_cdcl_partial_inputs(
                cr_b.H_target[idx], r_b, fw, fbits, device, stamp_heat_demo=False
            )
            prep_ep1 = model.build_crystal_features(
                H1, W1, S1, C1, rounds=r_b, unknown_mask=u1,
                conflict_heatmap=None, zero_defects=True, crystal_engine=torch_eng,
            )
            nf_ep1_list.append(prep_ep1["node_feat"])
            sd_ep1_list.append(prep_ep1["site_defect_signal"])

            H2, W2, S2, C2, u2, c2, _ = _build_dyn_cdcl_partial_inputs(
                cr_b.H_target[idx], r_b, fw, fbits, device, stamp_heat_demo=True
            )
            prep_ep2 = model.build_crystal_features(
                H2, W2, S2, C2, rounds=r_b, unknown_mask=u2,
                conflict_heatmap=c2, zero_defects=True, crystal_engine=torch_eng,
            )
            nf_ep2_list.append(prep_ep2["node_feat"])
            sd_ep2_list.append(prep_ep2["site_defect_signal"])

        tag = f"domain_B_R{r_b}_{fb}b"
        results[f"{tag}_arx_blind"] = _summarize_104_channels(
            torch.cat(nf_arx_list, dim=0), torch.cat(sd_arx_list, dim=0)
        )
        results[f"{tag}_dyn_ep1_zero_heat"] = _summarize_104_channels(
            torch.cat(nf_ep1_list, dim=0), torch.cat(sd_ep1_list, dim=0)
        )
        results[f"{tag}_dyn_ep2_stamped_heat"] = _summarize_104_channels(
            torch.cat(nf_ep2_list, dim=0), torch.cat(sd_ep2_list, dim=0)
        )

    return results


def _phase_metrics_from_logits(
    logits_1d: np.ndarray,
    targets_1d: np.ndarray,
    collapse_1d: np.ndarray,
    near_mask_1d: np.ndarray,
) -> Dict[str, Any]:
    l_clamped = np.clip(logits_1d.astype(np.float64), -30.0, 30.0)
    p = 1.0 / (1.0 + np.exp(-l_clamped))
    p_safe = np.clip(p, 1e-7, 1.0 - 1e-7)
    entropy_bits = -(p_safe * np.log2(p_safe) + (1.0 - p_safe) * np.log2(1.0 - p_safe))
    bce = -(targets_1d * np.log(p_safe) + (1.0 - targets_1d) * np.log(1.0 - p_safe))
    conf_dist = np.abs(p - 0.5)
    acc = float(np.mean((l_clamped >= 0.0) == (targets_1d > 0.5)))
    ge_008 = float(np.mean(np.abs(logits_1d) >= 0.08))

    near_vals = collapse_1d[near_mask_1d] if np.any(near_mask_1d) else np.array([], dtype=np.float64)
    far_vals = collapse_1d[~near_mask_1d] if np.any(~near_mask_1d) else np.array([], dtype=np.float64)
    near_mean = float(np.mean(near_vals)) if near_vals.size else None
    far_mean = float(np.mean(far_vals)) if far_vals.size else None
    near_far_ratio = (
        round(near_mean / max(1e-8, far_mean), 4)
        if (near_mean is not None and far_mean is not None)
        else None
    )

    return {
        "n_unknown_w_bits": int(len(logits_1d)),
        "entropy_bits_mean": round(float(np.mean(entropy_bits)), 6),
        "bce_loss_mean": round(float(np.mean(bce)), 6),
        "ln2_baseline": round(math.log(2.0), 6),
        "confidence_abs_p_minus_05_mean": round(float(np.mean(conf_dist)), 6),
        "confidence_abs_p_minus_05_p90": round(float(np.percentile(conf_dist, 90)), 6),
        "confidence_abs_p_minus_05_max": round(float(np.max(conf_dist)), 6),
        "frac_abs_logit_ge_008": round(ge_008, 6),
        "phase_accuracy": round(acc, 6),
        "collapse_score_mean": round(float(np.mean(collapse_1d)), 6),
        "collapse_score_std": round(float(np.std(collapse_1d)), 6),
        "collapse_near_w0_3_mean": round(near_mean, 6) if near_mean is not None else None,
        "collapse_far_w4_15_mean": round(far_mean, 6) if far_mean is not None else None,
        "collapse_near_to_far_ratio": near_far_ratio,
    }


def _pearson_corr(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 4:
        return None
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if float(np.std(xa)) < 1e-8 or float(np.std(ya)) < 1e-8:
        return None
    vx = xa - np.mean(xa)
    vy = ya - np.mean(ya)
    return round(float(np.sum(vx * vy) / (math.sqrt(float(np.sum(vx * vx) * np.sum(vy * vy))) + 1e-12)), 6)


@torch.no_grad()
def _measure_phase_and_collapse(
    model: Ouroboros64Net,
    core: OuroborosCore,
    torch_eng: TorchCrystalEngine,
    device: torch.device,
    n_crystals: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 101)
    out_dict: Dict[str, Any] = {}

    # 1) 域 A 假定掩蔽位置对照（zero_defects=False 揭示真实残差时）
    w64 = _sample_random_words(rng, n_crystals, 16)
    cr64 = core.compress_exact(w64, rounds=64)
    W_d, S_d, C_d, _ = _corrupt_crystals_np(rng, cr64.W, cr64.S, cr64.C, rounds=64, mode="cluster")
    out_a = model(
        torch.from_numpy(cr64.H_target.astype(np.int64)).to(device),
        torch.from_numpy(W_d.astype(np.int64)).to(device),
        torch.from_numpy(S_d.astype(np.int64)).to(device),
        torch.from_numpy(C_d.astype(np.int64)).to(device),
        rounds=64,
        unknown_mask=None,
        zero_defects=False,
        propose_clusters=False,
        return_entropy=False,
        crystal_engine=torch_eng,
    )
    w_true_64 = words_to_bits_np(cr64.W[:, :16]).astype(np.float64)
    ph_a = out_a["phase_logits"][:, :16, 0, :].cpu().numpy()
    co_a = out_a["collapse_score"][:, :16, 0, :].cpu().numpy()
    near_a = np.zeros((n_crystals, 16, 32), dtype=bool)
    near_a[:, :4, :] = True
    out_dict["domain_A_R64_cluster_all_W0_15"] = _phase_metrics_from_logits(
        ph_a.reshape(-1), w_true_64.reshape(-1), co_a.reshape(-1), near_a.reshape(-1)
    )

    # 2) 域 B 多档位（含动态输入模式与全盲模式）
    tiers = ((24, 32), (24, 40), (25, 24), (64, 512))
    for r_b, fb in tiers:
        w_b = _sample_random_words(rng, n_crystals, 16)
        cr_b = core.compress_exact(w_b, rounds=r_b)
        logits_all: List[float] = []
        tgts_all: List[float] = []
        coll_all: List[float] = []
        near_all: List[bool] = []

        # 为 R=24 40b 与 R=25 24b 同步统计真实 CaDiCaL 单步前瞻级联相关性
        la_preds: List[float] = []
        la_tgts: List[float] = []

        for idx in range(n_crystals):
            fw, fbits = (None, None) if fb >= 512 else make_end_free_constraints(w_b[idx], fb)
            H_t, W_t, S_t, C_t, unk_t, _, free_mask_w = _build_dyn_cdcl_partial_inputs(
                cr_b.H_target[idx], r_b, fw, fbits, device, stamp_heat_demo=False
            )
            out_b = model(
                H_t, W_t, S_t, C_t,
                rounds=r_b,
                unknown_mask=unk_t,
                conflict_heatmap=None,
                zero_defects=True,
                propose_clusters=False,
                return_entropy=False,
                crystal_engine=torch_eng,
            )
            ph_b = out_b["phase_logits"][0, :16, 0, :].cpu().numpy()
            co_b = out_b["collapse_score"][0, :16, 0, :].cpu().numpy()
            w_true_b = words_to_bits_np(cr_b.W[idx : idx + 1, :16])[0].astype(np.float64)

            coords = np.argwhere(free_mask_w)
            for wi, bb in coords:
                logits_all.append(float(ph_b[wi, bb]))
                tgts_all.append(float(w_true_b[wi, bb]))
                coll_all.append(float(co_b[wi, bb]))
                near_all.append(int(wi) < 4)

            if (r_b, fb) in ((24, 40), (25, 24)) and idx < min(4, n_crystals):
                enc = SHA256CNFEncoder(rounds=r_b).build(
                    H_target=cr_b.H_target[idx], fixed_words=fw, fixed_bits=fbits
                )
                with Solver(name="cadical195", bootstrap_with=enc.clauses) as ps:
                    ok_r, lbase = ps.propagate()
                    n_base = len(lbase)
                    if ok_r:
                        pick_cnt = min(16, len(coords))
                        sel = rng.choice(len(coords), size=pick_cnt, replace=False)
                        for si in sel:
                            wi_p, bb_p = int(coords[si][0]), int(coords[si][1])
                            lit_p = enc.W_vars[wi_p][bb_p]
                            if isinstance(lit_p, bool) or not isinstance(lit_p, int):
                                continue
                            ok1, l1 = ps.propagate(assumptions=[lit_p])
                            ok0, l0 = ps.propagate(assumptions=[-lit_p])
                            if (not ok1) or (not ok0):
                                tgt_val = 6.0
                            else:
                                d1 = max(0, len(l1) - n_base)
                                d0 = max(0, len(l0) - n_base)
                                tgt_val = math.log1p(0.5 * (d1 + d0))
                            la_preds.append(float(co_b[wi_p, bb_p]))
                            la_tgts.append(float(tgt_val))

        metrics = _phase_metrics_from_logits(
            np.asarray(logits_all, dtype=np.float64),
            np.asarray(tgts_all, dtype=np.float64),
            np.asarray(coll_all, dtype=np.float64),
            np.asarray(near_all, dtype=bool),
        )
        if la_preds:
            metrics["cadical_lookahead_probe_n"] = len(la_preds)
            metrics["collapse_vs_cadical_lookahead_pearson"] = _pearson_corr(la_preds, la_tgts)
            metrics["cadical_lookahead_target_mean"] = round(float(np.mean(la_tgts)), 6)
        out_dict[f"domain_B_R{r_b}_{fb}b"] = metrics

    return out_dict


@torch.no_grad()
def _measure_pair_copy_baseline(
    model: Ouroboros64Net,
    core: OuroborosCore,
    torch_eng: TorchCrystalEngine,
    device: torch.device,
    n_crystals: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 202)
    w64 = _sample_random_words(rng, n_crystals, 16)
    cr64 = core.compress_exact(w64, rounds=64)
    H_t = torch.from_numpy(cr64.H_target.astype(np.int64)).to(device)
    C_clean_t = torch.from_numpy(cr64.C.astype(np.int64)).to(device)
    pair_tgt, active_mask = _compute_carry_pair_targets(C_clean_t)  # (B, 64, 31, 10), (1, 64, 31, 10)
    active_full = active_mask.expand_as(pair_tgt)

    results: Dict[str, Any] = {}
    for mode in ("medium", "cluster"):
        W_d, S_d, C_d, _ = _corrupt_crystals_np(rng, cr64.W, cr64.S, cr64.C, rounds=64, mode=mode)
        C_dirty_t = torch.from_numpy(C_d.astype(np.int64)).to(device)
        copy_pair_idx, _ = _compute_carry_pair_targets(C_dirty_t)

        out = model(
            H_t,
            torch.from_numpy(W_d.astype(np.int64)).to(device),
            torch.from_numpy(S_d.astype(np.int64)).to(device),
            C_dirty_t,
            rounds=64,
            return_entropy=True,
            crystal_engine=torch_eng,
        )
        pair_probs = torch.softmax(out["pair_logits"], dim=-1).clamp(min=1e-7)
        p_l1 = pair_probs[..., 2] + pair_probs[..., 3]
        p_r1 = pair_probs[..., 1] + pair_probs[..., 3]
        rank1_probs = torch.stack(
            [(1 - p_l1) * (1 - p_r1), (1 - p_l1) * p_r1, p_l1 * (1 - p_r1), p_l1 * p_r1], dim=-1
        ).clamp(min=1e-7)

        mask_flat = active_full.reshape(-1)
        tgt_flat = pair_tgt.reshape(-1)[mask_flat]
        pred_flat = torch.argmax(pair_probs, dim=-1).reshape(-1)[mask_flat]
        copy_flat = copy_pair_idx.reshape(-1)[mask_flat]

        nll_joint = float(F.nll_loss(torch.log(pair_probs).reshape(-1, 4)[mask_flat], tgt_flat).item())
        nll_rank1 = float(F.nll_loss(torch.log(rank1_probs).reshape(-1, 4)[mask_flat], tgt_flat).item())
        kl = torch.sum(pair_probs * (torch.log(pair_probs) - torch.log(rank1_probs)), dim=-1)
        kl_mean = float(kl[active_full].mean().item())

        # 互斥子集拆分：未翻转进位对 vs 真实缺陷翻转进位对
        uncorrupted = (copy_flat == tgt_flat)
        corrupted = ~uncorrupted

        def _subset_acc(sub_mask: torch.Tensor) -> Dict[str, Any]:
            cnt = int(sub_mask.sum().item())
            if cnt == 0:
                return {"count": 0, "model_acc": None, "copy_cd_acc": None}
            return {
                "count": cnt,
                "frac_of_active": round(cnt / int(mask_flat.sum().item()), 6),
                "model_acc": round(float((pred_flat[sub_mask] == tgt_flat[sub_mask]).float().mean().item()), 6),
                "copy_cd_acc": round(float((copy_flat[sub_mask] == tgt_flat[sub_mask]).float().mean().item()), 6),
            }

        results[f"domain_A_R64_{mode}"] = {
            "active_pair_count": int(mask_flat.sum().item()),
            "pair_joint_nll": round(nll_joint, 6),
            "pair_rank1_marginal_nll": round(nll_rank1, 6),
            "nll_gain_over_rank1": round(nll_rank1 - nll_joint, 6),
            "non_rank1_mutual_info_kl": round(kl_mean, 6),
            "overall_model_pair_acc": round(float((pred_flat == tgt_flat).float().mean().item()), 6),
            "overall_copy_cd_dummy_acc": round(float((copy_flat == tgt_flat).float().mean().item()), 6),
            "model_vs_copy_agreement": round(float((pred_flat == copy_flat).float().mean().item()), 6),
            "uncorrupted_pairs": _subset_acc(uncorrupted),
            "corrupted_flipped_pairs": _subset_acc(corrupted),
        }

    # 域 B 全盲对照（C_bip=0, zero_defects=True）
    W_zero = torch.zeros((n_crystals, 64), dtype=torch.int64, device=device)
    S_zero = torch.zeros((n_crystals, 65, 8), dtype=torch.int64, device=device)
    S_zero[:, 0, :] = torch.tensor(IV_WORDS, dtype=torch.int64, device=device)
    S_zero[:, 64, :] = (H_t - S_zero[:, 0, :]) & MASK32
    C_zero = torch.zeros((n_crystals, 64, C_PLANES), dtype=torch.int64, device=device)
    unk_b = torch.ones((n_crystals, 64, PLANE_COUNT, 32), dtype=torch.float32, device=device)
    unk_b[:, 0, 1:9, :] = 0.0
    out_b = model(
        H_t, W_zero, S_zero, C_zero,
        rounds=64, unknown_mask=unk_b, zero_defects=True,
        propose_clusters=False, return_entropy=True, crystal_engine=torch_eng,
    )
    pb = torch.softmax(out_b["pair_logits"], dim=-1).clamp(min=1e-7)
    mask_flat_b = active_full.reshape(-1)
    tgt_flat_b = pair_tgt.reshape(-1)[mask_flat_b]
    pred_flat_b = torch.argmax(pb, dim=-1).reshape(-1)[mask_flat_b]
    nll_b = float(F.nll_loss(torch.log(pb).reshape(-1, 4)[mask_flat_b], tgt_flat_b).item())
    results["domain_B_R64_512b_blind"] = {
        "active_pair_count": int(mask_flat_b.sum().item()),
        "pair_joint_nll": round(nll_b, 6),
        "pair_joint_acc": round(float((pred_flat_b == tgt_flat_b).float().mean().item()), 6),
        "uniform_4state_baseline_acc": 0.25,
    }
    return results


@torch.no_grad()
def _run_ablation_domain_a_zero_defects(
    model: Ouroboros64Net,
    core: OuroborosCore,
    torch_eng: TorchCrystalEngine,
    device: torch.device,
    n_crystals: int,
    seed: int,
) -> Dict[str, Any]:
    """消融 1：同一批 R=64 缺陷晶体，对比 zero_defects=False vs zero_defects=True。"""
    rng = np.random.default_rng(seed + 303)
    w64 = _sample_random_words(rng, n_crystals, 16)
    cr64 = core.compress_exact(w64, rounds=64)
    W_d, S_d, C_d, flip_tgt = _corrupt_crystals_np(rng, cr64.W, cr64.S, cr64.C, rounds=64, mode="cluster")

    H_t = torch.from_numpy(cr64.H_target.astype(np.int64)).to(device)
    W_t = torch.from_numpy(W_d.astype(np.int64)).to(device)
    S_t = torch.from_numpy(S_d.astype(np.int64)).to(device)
    C_t = torch.from_numpy(C_d.astype(np.int64)).to(device)
    flip_t = torch.from_numpy(flip_tgt).to(device)
    valid_mask_t = build_valid_site_mask(64, device=device).unsqueeze(0).expand_as(flip_t)

    k_budget = 16
    res: Dict[str, Any] = {}
    for zd in (False, True):
        torch.manual_seed(seed + 303)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 303)
        out = model(
            H_t, W_t, S_t, C_t,
            rounds=64, unknown_mask=None, zero_defects=zd,
            propose_clusters=True, num_clusters=k_budget,
            temperature=0.85, deterministic_clusters=False,
            return_entropy=False, crystal_engine=torch_eng,
        )
        f_pred = out["flip_logits"][valid_mask_t]
        f_true = flip_t[valid_mask_t]
        pred_pos = f_pred >= 0.0
        true_pos = f_true > 0.5
        tp = int((pred_pos & true_pos).sum().item())
        fp = int((pred_pos & (~true_pos)).sum().item())
        fn = int(((~pred_pos) & true_pos).sum().item())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2.0 * prec * rec / max(1e-8, prec + rec)

        # per-crystal Top-64 召回率
        topk_recalls: List[float] = []
        f_logits_all = out["flip_logits"]
        v_single = build_valid_site_mask(64, device=device)
        for bi in range(n_crystals):
            sc_i = f_logits_all[bi][v_single]
            gt_i = flip_t[bi][v_single] > 0.5
            n_gt = int(gt_i.sum().item())
            if n_gt > 0:
                tk = torch.topk(sc_i, k=min(64, sc_i.numel())).indices
                topk_recalls.append(float(gt_i[tk].sum().item() / n_gt))

        cl_np = out["clusters"]["clusters_coord"].cpu().numpy()  # (B, K, M, 4)
        m_max = cl_np.shape[2]
        de_c, _ = core.eval_cluster_flips_batch(
            np.repeat(cr64.H_target, k_budget, axis=0),
            np.repeat(W_d, k_budget, axis=0),
            np.repeat(S_d, k_budget, axis=0),
            np.repeat(C_d, k_budget, axis=0),
            cl_np.reshape(n_crystals * k_budget, m_max, 4),
            rounds=64,
        )
        de_mat = de_c.reshape(n_crystals, k_budget)
        min_de_per_cryst = np.min(de_mat, axis=1)

        key = "zero_defects_true_ablated" if zd else "zero_defects_false_intact"
        res[key] = {
            "flip_precision": round(prec, 6),
            "flip_recall": round(rec, 6),
            "flip_f1": round(f1, 6),
            "flip_top64_recall": round(float(np.mean(topk_recalls)), 6),
            "wolff_any_reducing_hit_rate": round(float(np.mean(min_de_per_cryst < 0)), 6),
            "wolff_best_delta_e_mean": round(float(np.mean(min_de_per_cryst)), 4),
            "wolff_candidate_reducing_frac": round(float(np.mean(de_mat < 0)), 6),
            "site_defect_signal_nonzero": round(
                float((out["site_defect_signal"].abs() > 1e-8).float().mean().item()), 6
            ),
        }
    return res


@torch.no_grad()
def _run_ablation_stage_b_mask_mismatch(
    model: Ouroboros64Net,
    core: OuroborosCore,
    torch_eng: TorchCrystalEngine,
    device: torch.device,
    n_crystals: int,
    seed: int,
) -> Dict[str, Any]:
    """消融 2：R=24 下对比 Stage B 随机掩蔽（reveal=50%, 25%, 7%）vs make_end_free_constraints（fb=32, 40）。"""
    rng = np.random.default_rng(seed + 404)
    r_b = 24
    w0 = _sample_random_words(rng, n_crystals, 16)
    cr = core.compress_exact(w0, rounds=r_b)

    def _eval_mask_regime(regime_name: str, mode: str, param: float) -> Dict[str, Any]:
        logits_all: List[float] = []
        tgts_all: List[float] = []
        coll_all: List[float] = []
        near_all: List[bool] = []
        arx_ded_list: List[int] = []
        la_preds: List[float] = []
        la_tgts: List[float] = []

        for bi in range(n_crystals):
            W_sym = np.full((r_b, 32), SYM_Q, dtype=np.uint8)
            S_sym = np.full((r_b + 1, 8, 32), SYM_Q, dtype=np.uint8)
            C_sym = np.full((r_b, C_PLANES, 32), SYM_Q, dtype=np.uint8)
            S_sym[0] = words_to_bits_np(cr.S[bi, 0])
            S_sym[r_b] = words_to_bits_np(cr.S[bi, r_b])
            w_clean_bits = words_to_bits_np(cr.W[bi, :16])

            if mode == "stage_b_random":
                rev = rng.random((16, 32)) < float(param)
                fw_d: Optional[Dict[int, int]] = None
                fbits_d: Dict[Tuple[int, int], int] = {}
                for wi in range(16):
                    for bb in range(32):
                        if rev[wi, bb]:
                            val = int(w_clean_bits[wi, bb])
                            W_sym[wi, bb] = SYM_1 if val else SYM_0
                            fbits_d[(wi, bb)] = val
            else:
                fw_d, fbits_d = make_end_free_constraints(w0[bi], int(param))
                for wi, wv in fw_d.items():
                    for bb in range(32):
                        W_sym[wi, bb] = SYM_1 if ((wv >> bb) & 1) else SYM_0
                for (wi, bb), bv in fbits_d.items():
                    W_sym[wi, bb] = SYM_1 if (bv & 1) else SYM_0

            prop0 = core.propagate_symbols(
                cr.H_target[bi], W_sym, S_sym, C_sym, rounds=r_b, use_gf2=False, max_iters=6
            )
            arx_ded_list.append(int(prop0["unit_prop_deductions"]))
            W_s0, S_s0, C_s0 = prop0["W_sym"], prop0["S_sym"], prop0["C_sym"]

            W_k = np.where((W_s0 == SYM_0) | (W_s0 == SYM_1), W_s0 == SYM_1, 0).astype(np.uint8)
            S_k = np.where((S_s0[:r_b] == SYM_0) | (S_s0[:r_b] == SYM_1), S_s0[:r_b] == SYM_1, 0).astype(np.uint8)
            C_k = np.where((C_s0 == SYM_0) | (C_s0 == SYM_1), C_s0 == SYM_1, 0).astype(np.uint8)

            S_words = np.zeros((1, r_b + 1, 8), dtype=np.uint32)
            S_words[0, :r_b] = bits_to_words_np(S_k)
            S_words[0, 0] = cr.S[bi, 0]
            S_words[0, r_b] = cr.S[bi, r_b]
            W_words = bits_to_words_np(W_k)[None, :]
            C_words = bits_to_words_np(C_k)[None, :]

            unk_np = np.ones((1, r_b, PLANE_COUNT, 32), dtype=np.float32)
            unk_np[0, :, 0, :] = ((W_s0 != SYM_0) & (W_s0 != SYM_1)).astype(np.float32)
            unk_np[0, :, 1:9, :] = ((S_s0[:r_b] != SYM_0) & (S_s0[:r_b] != SYM_1)).astype(np.float32)
            unk_np[0, :, 9:19, :] = ((C_s0 != SYM_0) & (C_s0 != SYM_1)).astype(np.float32)
            unk_np[0, 0, 1:9, :] = 0.0

            out = model(
                torch.from_numpy(cr.H_target[bi : bi + 1].astype(np.int64)).to(device),
                torch.from_numpy(W_words.astype(np.int64)).to(device),
                torch.from_numpy(S_words.astype(np.int64)).to(device),
                torch.from_numpy(C_words.astype(np.int64)).to(device),
                rounds=r_b,
                unknown_mask=torch.from_numpy(unk_np).to(device),
                zero_defects=True,
                propose_clusters=False,
                return_entropy=False,
                crystal_engine=torch_eng,
            )
            ph = out["phase_logits"][0, :16, 0, :].cpu().numpy()
            co = out["collapse_score"][0, :16, 0, :].cpu().numpy()

            unk_coords = np.argwhere(W_s0[:16] == SYM_Q)
            for wi, bb in unk_coords:
                logits_all.append(float(ph[wi, bb]))
                tgts_all.append(float(w_clean_bits[wi, bb]))
                coll_all.append(float(co[wi, bb]))
                near_all.append(int(wi) < 4)

            if bi < min(4, n_crystals) and len(unk_coords) > 0:
                enc = SHA256CNFEncoder(rounds=r_b).build(
                    H_target=cr.H_target[bi], fixed_words=fw_d, fixed_bits=fbits_d
                )
                with Solver(name="cadical195", bootstrap_with=enc.clauses) as ps:
                    ok_r, lbase = ps.propagate()
                    n_base = len(lbase)
                    if ok_r:
                        sel = rng.choice(len(unk_coords), size=min(12, len(unk_coords)), replace=False)
                        for si in sel:
                            wi_p, bb_p = int(unk_coords[si][0]), int(unk_coords[si][1])
                            lit_p = enc.W_vars[wi_p][bb_p]
                            if isinstance(lit_p, bool) or not isinstance(lit_p, int):
                                continue
                            ok1, l1 = ps.propagate(assumptions=[lit_p])
                            ok0, l0 = ps.propagate(assumptions=[-lit_p])
                            if (not ok1) or (not ok0):
                                tgt_v = 6.0
                            else:
                                tgt_v = math.log1p(0.5 * (max(0, len(l1) - n_base) + max(0, len(l0) - n_base)))
                            la_preds.append(float(co[wi_p, bb_p]))
                            la_tgts.append(float(tgt_v))

        m = _phase_metrics_from_logits(
            np.asarray(logits_all, dtype=np.float64),
            np.asarray(tgts_all, dtype=np.float64),
            np.asarray(coll_all, dtype=np.float64),
            np.asarray(near_all, dtype=bool),
        )
        m["regime"] = regime_name
        m["arx_unit_deductions_mean"] = round(float(np.mean(arx_ded_list)), 2)
        m["collapse_vs_cadical_lookahead_pearson"] = _pearson_corr(la_preds, la_tgts)
        return m

    return {
        "stage_b_random_reveal_50pct": _eval_mask_regime("stage_b_random_50pct", "stage_b_random", 0.50),
        "stage_b_random_reveal_25pct": _eval_mask_regime("stage_b_random_25pct", "stage_b_random", 0.25),
        "stage_b_random_reveal_07pct": _eval_mask_regime("stage_b_random_07pct", "stage_b_random", 0.07),
        "eval_end_free_32b": _eval_mask_regime("end_free_32b", "end_free", 32),
        "eval_end_free_40b": _eval_mask_regime("end_free_40b", "end_free", 40),
    }


def _evaluate_checkpoint(
    label: str,
    ckpt_path: str,
    core: OuroborosCore,
    torch_eng: TorchCrystalEngine,
    device: torch.device,
    n_crystals: int,
    seed: int,
    full_detail: bool = True,
) -> Dict[str, Any]:
    print(f"[diag_domain_gap] 加载检查点 {label}: {ckpt_path}", flush=True)
    model, payload = load_v10_checkpoint(ckpt_path, device=device)
    step = int(payload.get("step", -1))

    res: Dict[str, Any] = {
        "label": label,
        "ckpt_path": ckpt_path,
        "step": step,
        "input_proj_norms": _analyze_input_proj_norms(model),
    }
    print(f"[diag_domain_gap] ({label}) 测量 phase_logits / collapse_score 分布...", flush=True)
    res["phase_and_collapse"] = _measure_phase_and_collapse(
        model, core, torch_eng, device, n_crystals=n_crystals, seed=seed
    )
    print(f"[diag_domain_gap] ({label}) 测量 PairAcc vs 拷贝 C_d 哑基线...", flush=True)
    res["pair_entanglement_vs_copy_dummy"] = _measure_pair_copy_baseline(
        model, core, torch_eng, device, n_crystals=n_crystals, seed=seed
    )
    if full_detail:
        print(f"[diag_domain_gap] ({label}) 测量域 A vs 域 B 104 通道非零率...", flush=True)
        res["channel_activation_104d"] = _measure_channel_stats(
            model, core, torch_eng, device, n_crystals=n_crystals, seed=seed
        )
        print(f"[diag_domain_gap] ({label}) 运行消融 1 (域 A zero_defects=True 剥离)...", flush=True)
        res["ablation_domain_a_zero_defects"] = _run_ablation_domain_a_zero_defects(
            model, core, torch_eng, device, n_crystals=n_crystals, seed=seed
        )
        print(f"[diag_domain_gap] ({label}) 运行消融 2 (Stage B 随机掩蔽 vs end-free 错位)...", flush=True)
        res["ablation_stage_b_mask_mismatch"] = _run_ablation_stage_b_mask_mismatch(
            model, core, torch_eng, device, n_crystals=n_crystals, seed=seed
        )
    return res


def run_diag_domain_gap(
    m3_ckpt: str = DEFAULT_M3V2_CKPT,
    m2_ckpt: str = DEFAULT_M2_CKPT,
    n_crystals: int = 8,
    seed: int = 20261124,
    fast: bool = False,
    cpu: bool = False,
    out_json: str = DEFAULT_OUT_JSON,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    fp32_info = _check_fp32_locked()
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    core = get_core()
    torch_eng = TorchCrystalEngine(device=device)
    n_eff = min(4, n_crystals) if fast else int(n_crystals)

    print(
        f"[diag_domain_gap] 启动 | device={device} | fp32_locked={fp32_info['fp32_locked']} | "
        f"n_crystals={n_eff} | fast={fast}",
        flush=True,
    )

    models_report: Dict[str, Any] = {}
    models_report["best_m3v2"] = _evaluate_checkpoint(
        "best_m3v2", m3_ckpt, core, torch_eng, device, n_eff, seed, full_detail=True
    )
    models_report["m2_step4000"] = _evaluate_checkpoint(
        "m2_step4000", m2_ckpt, core, torch_eng, device, n_eff, seed, full_detail=(not fast)
    )

    elapsed = round(time.perf_counter() - t0, 3)
    report: Dict[str, Any] = {
        "diagnostic": "domain_gap_readonly",
        "fp32_check": fp32_info,
        "device": str(device),
        "n_crystals": n_eff,
        "seed": int(seed),
        "fast_mode": bool(fast),
        "elapsed_sec": elapsed,
        "models": models_report,
    }
    _atomic_write_json(Path(out_json), report)
    print(f"[diag_domain_gap] 完成，耗时 {elapsed:.2f}s，结果已写入 {out_json}", flush=True)
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="R2 只读诊断 1：域 A vs 域 B 结构与条件分布鸿沟")
    p.add_argument("--m3_ckpt", type=str, default=DEFAULT_M3V2_CKPT)
    p.add_argument("--m2_ckpt", type=str, default=DEFAULT_M2_CKPT)
    p.add_argument("--n_crystals", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261124)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--quick", action="store_true", help="--fast 的别名")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_diag_domain_gap(
        m3_ckpt=args.m3_ckpt,
        m2_ckpt=args.m2_ckpt,
        n_crystals=args.n_crystals,
        seed=args.seed,
        fast=bool(args.fast or args.quick),
        cpu=args.cpu,
        out_json=args.out_json,
    )
