"""
R4-3 脚本 3：PyTorch FP32 只读边界核验与 C++17 局部窗口 GF(2) 复杂度实测。

严格只读与精度约束：
  - 显式调用 enforce_fp32_determinism() 并断言 allow_tf32_matmul == False 且 allow_tf32_cudnn == False；
  - 只读加载 best_m3v2.pt 与 ouroboros_a100_step4000.pt，不修改任何权重与存档；
  - 核验项：
      B1: PairAcc 恒等拷贝支配（model_acc < copy_dummy_acc，flipped_acc < 0.35，混合恒等式）
      B2: 盲态 BCE > ln 2 与 M3-V2 置信度膨胀倍率（>= 3.5x，frac(|logit|>=0.08) >= 0.95）
      B3: zero_defects=True 剥离退化（Wolff 降能命中率归 0）
      B4: 盲态随机初态期望能量 E0 统计（近 N_checks / 2 = 18136）
      B5: 调用 core.solve_gf2_equations 实测局部轮窗 (w=8) vs 全局 R=64 的变量缩减比与加速比
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v10_ouroboros64.formal_checks import DEFAULT_M2_CKPT, DEFAULT_M3V2_CKPT
from v10_ouroboros64.ouroboros_core.cnf_cdcl import make_end_free_constraints
from v10_ouroboros64.ouroboros_core.engine import (
    C_PLANES,
    IV_WORDS,
    K_WORDS,
    MASK32,
    OuroborosCore,
    TorchCrystalEngine,
    enforce_fp32_determinism,
    get_core,
)
from v10_ouroboros64.ouroboros_mcts.hybrid_solver import load_v10_checkpoint
from v10_ouroboros64.ouroboros_net.heads import PLANE_COUNT, build_valid_site_mask
from v10_ouroboros64.ouroboros_net.model import Ouroboros64Net
from v10_ouroboros64.ouroboros_train.m3_trainer import _build_blind_crystal_inputs
from v10_ouroboros64.ouroboros_train.trainer import (
    _compute_carry_pair_targets,
    _corrupt_crystals_np,
    _sample_random_words,
)


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
    assert (not matmul_tf32) and (not cudnn_tf32), "TF32 必须禁用"
    return {
        "cuda_available": cuda_avail,
        "allow_tf32_matmul": matmul_tf32,
        "allow_tf32_cudnn": cudnn_tf32,
        "fp32_locked": (not matmul_tf32) and (not cudnn_tf32),
    }


@torch.no_grad()
def verify_b1_pair_copy_dominance(
    model: Ouroboros64Net,
    core: OuroborosCore,
    torch_eng: TorchCrystalEngine,
    device: torch.device,
    n_crystals: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 11)
    w64 = _sample_random_words(rng, n_crystals, 16)
    cr64 = core.compress_exact(w64, rounds=64)
    H_t = torch.from_numpy(cr64.H_target.astype(np.int64)).to(device)
    C_clean_t = torch.from_numpy(cr64.C.astype(np.int64)).to(device)
    pair_tgt, active_mask = _compute_carry_pair_targets(C_clean_t)
    active_full = active_mask.expand_as(pair_tgt)
    mask_flat = active_full.reshape(-1)
    tgt_flat = pair_tgt.reshape(-1)[mask_flat]

    res: Dict[str, Any] = {}
    for mode in ("medium", "cluster"):
        W_d, S_d, C_d, _ = _corrupt_crystals_np(rng, cr64.W, cr64.S, cr64.C, rounds=64, mode=mode)
        C_dirty_t = torch.from_numpy(C_d.astype(np.int64)).to(device)
        copy_idx, _ = _compute_carry_pair_targets(C_dirty_t)
        out = model(
            H_t,
            torch.from_numpy(W_d.astype(np.int64)).to(device),
            torch.from_numpy(S_d.astype(np.int64)).to(device),
            C_dirty_t,
            rounds=64,
            return_entropy=False,
            crystal_engine=torch_eng,
        )
        probs = torch.softmax(out["pair_logits"], dim=-1).clamp(min=1e-7)
        p_l1 = probs[..., 2] + probs[..., 3]
        p_r1 = probs[..., 1] + probs[..., 3]
        rank1_probs = torch.stack(
            [(1 - p_l1) * (1 - p_r1), (1 - p_l1) * p_r1, p_l1 * (1 - p_r1), p_l1 * p_r1], dim=-1
        ).clamp(min=1e-7)

        pred_flat = torch.argmax(probs, dim=-1).reshape(-1)[mask_flat]
        copy_flat = copy_idx.reshape(-1)[mask_flat]
        nll_joint = float(F.nll_loss(torch.log(probs).reshape(-1, 4)[mask_flat], tgt_flat).item())
        nll_rank1 = float(F.nll_loss(torch.log(rank1_probs).reshape(-1, 4)[mask_flat], tgt_flat).item())
        kl = torch.sum(probs * (torch.log(probs) - torch.log(rank1_probs)), dim=-1)
        kl_mean = float(kl[active_full].mean().item())

        model_acc = float((pred_flat == tgt_flat).float().mean().item())
        copy_acc = float((copy_flat == tgt_flat).float().mean().item())
        uncorr = copy_flat == tgt_flat
        corr = ~uncorr
        uncorr_acc = float((pred_flat[uncorr] == tgt_flat[uncorr]).float().mean().item())
        corr_acc = float((pred_flat[corr] == tgt_flat[corr]).float().mean().item())
        eps = float(corr.float().mean().item())
        recon = (1.0 - eps) * uncorr_acc + eps * corr_acc

        assert abs(model_acc - recon) < 1e-5
        assert model_acc < copy_acc, f"{mode}: model_acc={model_acc} >= copy_acc={copy_acc}"
        assert corr_acc < 0.38, f"{mode}: flipped_acc={corr_acc} unexpectedly high"
        assert (nll_rank1 - nll_joint) < 0.005
        assert kl_mean >= 0.0

        res[mode] = {
            "active_pair_count": int(mask_flat.sum().item()),
            "model_acc": round(model_acc, 6),
            "copy_dummy_acc": round(copy_acc, 6),
            "uncorrupted_acc": round(uncorr_acc, 6),
            "flipped_acc": round(corr_acc, 6),
            "flipped_frac": round(eps, 6),
            "nll_gain_over_rank1": round(nll_rank1 - nll_joint, 6),
            "kl_mean": round(kl_mean, 6),
        }
    return {"passed": True, "modes": res}


@torch.no_grad()
def verify_b2_blind_bce_and_inflation(
    m3v2_model: Ouroboros64Net,
    m2_model: Ouroboros64Net,
    core: OuroborosCore,
    torch_eng: TorchCrystalEngine,
    device: torch.device,
    n_crystals: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 22)
    ln2 = math.log(2.0)
    tiers_res: Dict[str, Any] = {}

    for r_b, fb in ((24, 40), (25, 24)):
        w_b = _sample_random_words(rng, n_crystals, 16)
        cr_b = core.compress_exact(w_b, rounds=r_b)
        stats_by_model: Dict[str, Dict[str, float]] = {}

        for mname, mod in (("best_m3v2", m3v2_model), ("m2_step4000", m2_model)):
            logits_list: List[float] = []
            tgts_list: List[float] = []
            for idx in range(n_crystals):
                fw, fbits = make_end_free_constraints(w_b[idx], fb)
                H_t, W_t, S_t, C_t, unk_t, _, free_mask_w = _build_blind_crystal_inputs(
                    core, cr_b.H_target[idx], r_b, fw, fbits, device
                )
                out = mod(
                    H_t, W_t, S_t, C_t,
                    rounds=r_b, unknown_mask=unk_t, conflict_heatmap=None,
                    zero_defects=True, propose_clusters=False, return_entropy=False,
                    crystal_engine=torch_eng,
                )
                ph = out["phase_logits"][0, :16, 0, :].cpu().numpy()
                w0_bits = ((w_b[idx, :16, None] >> np.arange(32, dtype=np.uint32)[None, :]) & 1).astype(np.float64)
                for wi, bb in np.argwhere(free_mask_w):
                    logits_list.append(float(ph[wi, bb]))
                    tgts_list.append(float(w0_bits[wi, bb]))

            la = np.clip(np.asarray(logits_list, dtype=np.float64), -30.0, 30.0)
            ta = np.asarray(tgts_list, dtype=np.float64)
            pa = np.clip(1.0 / (1.0 + np.exp(-la)), 1e-7, 1.0 - 1e-7)
            bce = float(np.mean(-(ta * np.log(pa) + (1.0 - ta) * np.log(1.0 - pa))))
            acc = float(np.mean((la >= 0.0) == (ta > 0.5)))
            conf = float(np.mean(np.abs(pa - 0.5)))
            frac_008 = float(np.mean(np.abs(la) >= 0.08))
            stats_by_model[mname] = {
                "n_bits": int(len(la)),
                "bce_loss": round(bce, 6),
                "phase_acc": round(acc, 6),
                "conf_abs_p_minus_05": round(conf, 6),
                "frac_abs_logit_ge_008": round(frac_008, 6),
            }

        m3_s = stats_by_model["best_m3v2"]
        m2_s = stats_by_model["m2_step4000"]
        inflation = m3_s["conf_abs_p_minus_05"] / max(1e-8, m2_s["conf_abs_p_minus_05"])
        assert m3_s["bce_loss"] > ln2, f"R={r_b} {fb}b bce={m3_s['bce_loss']} <= ln2"
        assert 0.43 <= m3_s["phase_acc"] <= 0.58
        assert inflation >= 3.5, f"置信度膨胀倍率 {inflation} < 3.5"
        assert m3_s["frac_abs_logit_ge_008"] >= 0.95

        tiers_res[f"R{r_b}_{fb}b"] = {
            "ln2": round(ln2, 6),
            "confidence_inflation_ratio": round(inflation, 4),
            "models": stats_by_model,
        }
    return {"passed": True, "tiers": tiers_res}


@torch.no_grad()
def verify_b3_zero_defects_collapse(
    model: Ouroboros64Net,
    core: OuroborosCore,
    torch_eng: TorchCrystalEngine,
    device: torch.device,
    n_crystals: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 33)
    w64 = _sample_random_words(rng, n_crystals, 16)
    cr64 = core.compress_exact(w64, rounds=64)
    W_d, S_d, C_d, flip_tgt = _corrupt_crystals_np(rng, cr64.W, cr64.S, cr64.C, rounds=64, mode="cluster")

    H_t = torch.from_numpy(cr64.H_target.astype(np.int64)).to(device)
    W_t = torch.from_numpy(W_d.astype(np.int64)).to(device)
    S_t = torch.from_numpy(S_d.astype(np.int64)).to(device)
    C_t = torch.from_numpy(C_d.astype(np.int64)).to(device)
    flip_t = torch.from_numpy(flip_tgt).to(device)
    valid_mask = build_valid_site_mask(64, device=device).unsqueeze(0).expand_as(flip_t)

    k_budget = 16
    out_map: Dict[str, Any] = {}
    for zd in (False, True):
        torch.manual_seed(seed + 33)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 33)
        out = model(
            H_t, W_t, S_t, C_t,
            rounds=64, unknown_mask=None, zero_defects=zd,
            propose_clusters=True, num_clusters=k_budget,
            temperature=0.85, deterministic_clusters=False,
            return_entropy=False, crystal_engine=torch_eng,
        )
        fp = out["flip_logits"][valid_mask] >= 0.0
        ft = flip_t[valid_mask] > 0.5
        tp = int((fp & ft).sum().item())
        f1 = (2.0 * tp) / max(1, int(fp.sum().item()) + int(ft.sum().item()))

        cl_np = out["clusters"]["clusters_coord"].cpu().numpy()
        de_c, _ = core.eval_cluster_flips_batch(
            np.repeat(cr64.H_target, k_budget, axis=0),
            np.repeat(W_d, k_budget, axis=0),
            np.repeat(S_d, k_budget, axis=0),
            np.repeat(C_d, k_budget, axis=0),
            cl_np.reshape(n_crystals * k_budget, cl_np.shape[2], 4),
            rounds=64,
        )
        min_de = np.min(de_c.reshape(n_crystals, k_budget), axis=1)
        hit_rate = float(np.mean(min_de < 0))
        out_map[f"zero_defects_{str(zd).lower()}"] = {
            "flip_f1": round(f1, 6),
            "wolff_hit_rate": round(hit_rate, 6),
            "best_delta_e_mean": round(float(np.mean(min_de)), 4),
        }

    assert out_map["zero_defects_false"]["wolff_hit_rate"] >= 0.5
    assert out_map["zero_defects_true"]["wolff_hit_rate"] == 0.0
    assert out_map["zero_defects_true"]["flip_f1"] == 0.0
    return {"passed": True, "ablation": out_map}


def verify_b4_blind_high_entropy_energy(
    core: OuroborosCore,
    n_crystals: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 44)
    w_secret = _sample_random_words(rng, n_crystals, 16)
    H_blind = core.compress_exact(w_secret, rounds=64).H_target

    W_guess = _sample_random_words(rng, n_crystals, 64)
    S_guess = _sample_random_words(rng, n_crystals * 65, 8).reshape(n_crystals, 65, 8)
    C_guess = _sample_random_words(rng, n_crystals * 64, C_PLANES).reshape(n_crystals, 64, C_PLANES)
    C_guess[:, :, :] &= np.uint32(0xFFFFFFFE)
    C_guess[:, :16, 0:3] = 0
    s0_arr = np.asarray(IV_WORDS, dtype=np.uint32)
    sr_arr = ((H_blind.astype(np.uint64) - s0_arr.astype(np.uint64)) & MASK32).astype(np.uint32)
    S_guess[:, 0, :] = s0_arr[None, :]
    S_guess[:, 64, :] = sr_arr

    e_init = core.eval_crystal_batch(H_blind, W_guess, S_guess, C_guess, rounds=64, return_masks=False)["E"]
    e0_mean = float(np.mean(e_init))
    expected = 18136.0
    rel_err = abs(e0_mean - expected) / expected
    assert rel_err < 0.05, f"盲初态均值能量 {e0_mean} 偏离理论值 {expected} 超过 5%"
    return {
        "passed": True,
        "n_crystals": int(n_crystals),
        "theoretical_expected_e0": expected,
        "empirical_e0_mean": round(e0_mean, 4),
        "empirical_e0_min": int(np.min(e_init)),
        "empirical_e0_max": int(np.max(e_init)),
        "rel_error": round(rel_err, 6),
    }


def _build_window_gf2_system(t_start: int, t_end: int) -> Tuple[int, List[Tuple[List[int], int]]]:
    """构建 [t_start, t_end) 轮基础线性 GF(2) 方程组，调用 C++ core.solve_gf2_equations。"""
    R_len = t_end - t_start
    # 变量布局: W[t_min..t_end-1], a[t_start-3..t_end], e[t_start-3..t_end], C[t_start..t_end-1], Ch, Maj
    w_min = max(0, t_start - 16)
    n_w = t_end - w_min
    n_ae = R_len + 4  # offset -3..R_len

    off_W = 0
    off_a = off_W + n_w * 32
    off_e = off_a + n_ae * 32
    off_C = off_e + n_ae * 32
    off_Ch = off_C + R_len * C_PLANES * 32
    off_Maj = off_Ch + R_len * 32
    num_vars = off_Maj + R_len * 32

    def cW(t: int, b: int) -> int:
        return off_W + (t - w_min) * 32 + b

    def ca(t: int, b: int) -> int:
        return off_a + (t - (t_start - 3)) * 32 + b

    def ce(t: int, b: int) -> int:
        return off_e + (t - (t_start - 3)) * 32 + b

    def cC(t: int, k: int, b: int) -> int:
        return off_C + ((t - t_start) * C_PLANES + k) * 32 + b

    def cCh(t: int, b: int) -> int:
        return off_Ch + (t - t_start) * 32 + b

    def cMaj(t: int, b: int) -> int:
        return off_Maj + (t - t_start) * 32 + b

    eqs: List[Tuple[List[int], int]] = []
    for t in range(max(16, t_start), t_end):
        for b in range(32):
            cols = [
                cW(t, b),
                cW(t - 2, (b + 17) & 31),
                cW(t - 2, (b + 19) & 31),
                cW(t - 7, b),
                cW(t - 15, (b + 7) & 31),
                cW(t - 15, (b + 18) & 31),
                cW(t - 16, b),
                cC(t, 0, b),
                cC(t, 1, b),
                cC(t, 2, b),
            ]
            if b + 10 < 32:
                cols.append(cW(t - 2, b + 10))
            if b + 3 < 32:
                cols.append(cW(t - 15, b + 3))
            eqs.append((cols, 0))

    for t in range(t_start, t_end):
        kw = int(K_WORDS[t])
        for b in range(32):
            kbit = (kw >> b) & 1
            eqs.append((
                [
                    ce(t + 1, b),
                    ca(t - 3, b),
                    ce(t - 3, b),
                    ce(t, (b + 6) & 31),
                    ce(t, (b + 11) & 31),
                    ce(t, (b + 25) & 31),
                    cCh(t, b),
                    cW(t, b),
                    cC(t, 3, b),
                    cC(t, 4, b),
                    cC(t, 5, b),
                    cC(t, 6, b),
                    cC(t, 9, b),
                ],
                kbit,
            ))
            eqs.append((
                [
                    ca(t + 1, b),
                    ce(t + 1, b),
                    ca(t - 3, b),
                    ca(t, (b + 2) & 31),
                    ca(t, (b + 13) & 31),
                    ca(t, (b + 22) & 31),
                    cMaj(t, b),
                    cC(t, 7, b),
                    cC(t, 8, b),
                    cC(t, 9, b),
                ],
                0,
            ))
    return num_vars, eqs


def verify_b5_local_window_gf2_scaling(core: OuroborosCore) -> Dict[str, Any]:
    """使用 C++ core.solve_gf2_equations 实测 w=8 局部轮窗 vs 全局 R=64 的变量缩减与加速比。"""
    n_vars_g, eqs_g = _build_window_gf2_system(0, 64)
    assert n_vars_g == 30976
    t0 = time.perf_counter()
    res_g = core.solve_gf2_equations(n_vars_g, eqs_g)
    dt_g_ms = (time.perf_counter() - t0) * 1000.0

    n_vars_w, eqs_w = _build_window_gf2_system(28, 36)  # w=8 轮窗 (S[32]/W[32] 拼接中心)
    t1 = time.perf_counter()
    res_w = core.solve_gf2_equations(n_vars_w, eqs_w)
    dt_w_ms = (time.perf_counter() - t1) * 1000.0

    var_reduction = n_vars_g / float(n_vars_w)
    speedup = dt_g_ms / max(1e-4, dt_w_ms)
    assert not res_g["conflict"] and not res_w["conflict"]
    assert var_reduction >= 6.0, f"变量缩减比 {var_reduction} < 6.0"
    assert speedup >= 5.0, f"局部窗口加速比 {speedup} < 5.0"

    return {
        "passed": True,
        "global_r64": {
            "num_vars": n_vars_g,
            "num_eqs": len(eqs_g),
            "rank": int(res_g["rank"]),
            "elapsed_ms": round(dt_g_ms, 3),
        },
        "local_w8": {
            "window": [28, 36],
            "num_vars": n_vars_w,
            "num_eqs": len(eqs_w),
            "rank": int(res_w["rank"]),
            "elapsed_ms": round(dt_w_ms, 3),
        },
        "var_reduction_ratio": round(var_reduction, 4),
        "empirical_speedup": round(speedup, 2),
    }


def run_all_torch_checks(fast: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    fp32_info = _check_fp32_locked()
    device = torch.device("cuda" if fp32_info["cuda_available"] else "cpu")
    core = get_core()
    torch_eng = TorchCrystalEngine(device=device)

    repo_root = Path(__file__).resolve().parents[2]
    m3v2_model, _ = load_v10_checkpoint(repo_root / DEFAULT_M3V2_CKPT, device=device)
    m2_model, _ = load_v10_checkpoint(repo_root / DEFAULT_M2_CKPT, device=device)

    n_cryst = 4 if fast else 8
    seed = 20261124

    b1 = verify_b1_pair_copy_dominance(m3v2_model, core, torch_eng, device, n_cryst, seed)
    b2 = verify_b2_blind_bce_and_inflation(m3v2_model, m2_model, core, torch_eng, device, n_cryst, seed)
    b3 = verify_b3_zero_defects_collapse(m3v2_model, core, torch_eng, device, n_cryst, seed)
    b4 = verify_b4_blind_high_entropy_energy(core, max(8, n_cryst * 2), seed)
    b5 = verify_b5_local_window_gf2_scaling(core)

    dt = round(time.perf_counter() - t0, 3)
    all_ok = all(x["passed"] for x in (b1, b2, b3, b4, b5))
    assert all_ok
    return {
        "tool": "pytorch_and_cpp_core",
        "torch_version": torch.__version__,
        "device": str(device),
        "fp32_check": fp32_info,
        "fast_mode": bool(fast),
        "all_passed": all_ok,
        "elapsed_sec": dt,
        "B1_pair_copy_dominance": b1,
        "B2_blind_bce_and_inflation": b2,
        "B3_zero_defects_collapse": b3,
        "B4_blind_high_entropy_energy": b4,
        "B5_local_window_gf2_scaling": b5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="R4-3 PyTorch FP32 只读边界核验")
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    print(f"[verify_torch_bounds] 启动 | torch={torch.__version__} | fast={args.fast}", flush=True)
    res = run_all_torch_checks(fast=bool(args.fast))
    print(
        f"[verify_torch_bounds] B1-B5 全部通过! "
        f"allow_tf32=({res['fp32_check']['allow_tf32_matmul']}, {res['fp32_check']['allow_tf32_cudnn']}), "
        f"e0_mean={res['B4_blind_high_entropy_energy']['empirical_e0_mean']}, "
        f"gf2_w8_speedup={res['B5_local_window_gf2_scaling']['empirical_speedup']}x, "
        f"elapsed={res['elapsed_sec']}s",
        flush=True,
    )
    if args.out_json:
        _atomic_write_json(Path(args.out_json), res)
        print(f"[verify_torch_bounds] 结果已写入 {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
