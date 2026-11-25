"""
Phase-SHA V10「衔尾蛇-64」只读旁路探针：开放边界 D-bit 泄压标度斜率 α_hat 实测脚本。

严守零污染、零泄漏与 <= 5 分钟耗时铁律：
  - 加载冻结 Stage C 毕业权重（torch.no_grad()，纯 FP32）；
  - 维持 R=64 原生全晶体拓扑不变，仅将 S[64][7] 低 D 比特焊死至目标尾零约束 (-IV[7]) mod 2^D，
    其余 (256 - D) 比特保持开放自由边界 '?'（缺陷能量残差严格置 0）；
  - 独立调用 C++17 OuroborosCore.compress_exact 对每一次命中进行 SHA256_64 尾零逐比特验算；
  - 统计各档位中位触发评估步数 T(D)、相对 2^-D 算力增益倍率 G(D) 与标度斜率 α_hat = Δlog2 T(D) / ΔD；
  - 可选对比标准 BTC 80 字节区块头末块 320-bit 零填充模板 (W[4]=0x80000000, W[5..14]=0, W[15]=640)。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..ouroboros_core.engine import (
    IV_WORDS,
    MASK32,
    TorchCrystalEngine,
    enforce_fp32_determinism,
    get_core,
)
from ..ouroboros_mcts.hybrid_solver import load_v10_checkpoint
from ..ouroboros_net.model import Ouroboros64Net, get_model_preset


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _trailing_zeros_u32(val: int, max_bits: int = 32) -> int:
    v = int(val) & 0xFFFFFFFF
    if v == 0:
        return max_bits
    return int((v & -v).bit_length() - 1)


def _linear_slope(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    vx = x - x.mean()
    denom = float(np.sum(vx * vx))
    if denom < 1e-12:
        return None
    return float(np.sum(vx * (y - y.mean())) / denom)


def _init_w_batch(
    rng: np.random.Generator,
    bsz: int,
    btc_padding: bool,
) -> Tuple[np.ndarray, int]:
    """初始化 W[0..15]；BTC 模板固定 W[4]=0x80000000, W[5..14]=0 (320-bit 零填充), W[15]=640。"""
    w0 = rng.integers(0, 2**32, size=(bsz, 16), dtype=np.uint64).astype(np.uint32)
    free_limit = 16
    if btc_padding:
        w0[:, 4] = np.uint32(0x80000000)
        w0[:, 5:15] = np.uint32(0)
        w0[:, 15] = np.uint32(640)
        free_limit = 4
    return w0, free_limit


@torch.no_grad()
def probe_single_d_tier(
    model: Ouroboros64Net,
    core: Any,
    torch_eng: TorchCrystalEngine,
    device: torch.device,
    d_bits: int,
    btc_padding: bool = False,
    trials: int = 5,
    batch_cands: int = 64,
    max_steps_per_trial: int = 160,
    tier_time_budget_sec: float = 30.0,
    seed: int = 202611,
) -> Dict[str, Any]:
    """
    在完全冻结模型与开放边界 (256-D 比特设为 '?') 下探测单档尾零难度 D 命中行为。
    """
    if not (1 <= d_bits <= 32):
        raise ValueError(f"d_bits 必须处于 [1, 32]，收到 {d_bits}")

    rng = np.random.default_rng(seed + d_bits * 101 + (999 if btc_padding else 0))
    mask_d = (1 << d_bits) - 1
    mask_d_t = torch.tensor(mask_d, dtype=torch.int64, device=device)
    t0 = time.perf_counter()

    hit_evals_list: List[int] = []
    hit_steps_list: List[int] = []
    total_evals = 0
    best_tz_seen = 0
    hits = 0
    trials_attempted = 0

    for _tr in range(trials):
        if (time.perf_counter() - t0) >= tier_time_budget_sec:
            break
        trials_attempted += 1

        w_cur_np, free_words = _init_w_batch(rng, 1, btc_padding=btc_padding)
        trial_evals = 0

        # 验证初始随机点是否直接碰撞命中
        cr_init = core.compress_exact(w_cur_np, rounds=64)
        trial_evals += 1
        total_evals += 1
        best_tz_seen = max(best_tz_seen, _trailing_zeros_u32(int(cr_init.H_target[0, 7])))
        if (int(cr_init.H_target[0, 7]) & mask_d) == 0:
            hits += 1
            hit_evals_list.append(trial_evals)
            hit_steps_list.append(0)
            continue

        w_cur_t = torch.from_numpy(w_cur_np.astype(np.int64)).to(device)

        for st in range(1, max_steps_per_trial + 1):
            if (time.perf_counter() - t0) >= tier_time_budget_sec:
                break

            cr_t = torch_eng.compress_exact(w_cur_t, rounds=64)
            # 构建开放边界目标 H_open：仅将 word 7 低 D 比特归零，其余 (256-D) 比特保留 cr_t['H']
            # 这使 eval_crystal_batch 在其余 (256-D) 位上残差严格为 0，仅保留 D 比特泄压张力！
            H_open = cr_t["H"].clone()
            H_open[:, 7] = H_open[:, 7] & (~mask_d_t)

            out = model(
                H_open,
                cr_t["W"],
                cr_t["S"],
                cr_t["C"],
                rounds=64,
                propose_clusters=False,
                return_entropy=False,
                crystal_engine=torch_eng,
            )
            # 抽取网络对自由输入 W[0..free_words-1] 的条件翻转强度，并叠加轻微均匀保底避免单点锁死
            w_scores = out["flip_logits"][0, :free_words, 0, :].reshape(-1).clamp(-6.0, 6.0)
            n_free_bits = free_words * 32
            probs = 0.85 * torch.softmax(w_scores / 0.9, dim=0) + (0.15 / float(n_free_bits))

            # 按神经网络注意力加权抽样生成 batch_cands 组 1..4-bit 协同原像扰动
            cands_w = w_cur_t.repeat(batch_cands, 1)  # (K, 16)
            for k_i in range(batch_cands):
                n_flips = 1 if k_i < batch_cands // 4 else int((k_i % 3) + 2)
                idx_flips = torch.multinomial(probs, num_samples=min(n_flips, n_free_bits), replacement=False)
                for flat_b in idx_flips.tolist():
                    wi = flat_b // 32
                    bb = flat_b % 32
                    cands_w[k_i, wi] ^= (1 << bb)

            # GPU 批量精确计算 R=64 SHA-256 摘要并核验尾零约束
            cr_cands = torch_eng.compress_exact(cands_w, rounds=64)
            h7 = cr_cands["H"][:, 7]
            tail_err = h7 & mask_d_t
            trial_evals += batch_cands
            total_evals += batch_cands

            # 统计本批最低尾部违约 Hamming 距，贪婪跃迁至最低尾部张力状态继续泄压
            bit_pop = torch.zeros_like(tail_err)
            for b_i in range(d_bits):
                bit_pop += (tail_err >> b_i) & 1
            best_k = int(torch.argmin(bit_pop).item())
            h7_best = int(h7[best_k].item())
            best_tz_seen = max(best_tz_seen, _trailing_zeros_u32(h7_best))

            hit_indices = torch.nonzero(tail_err == 0, as_tuple=False).reshape(-1)
            if hit_indices.numel() > 0:
                hi = int(hit_indices[0].item())
                w_hit_np = cands_w[hi].cpu().numpy().astype(np.uint32)
                # 双重独立物理验真：严禁只信 PyTorch，必须再经 C++17 core.compress_exact 独立全量验算！
                cpp_h = core.compress_exact(w_hit_np[None, :], rounds=64).H_target[0]
                assert (int(cpp_h[7]) & mask_d) == 0, f"D={d_bits} 独立 C++ SHA256_64 验算失败！"
                hits += 1
                hit_evals_list.append(trial_evals)
                hit_steps_list.append(st)
                break

            # 按模拟退火概率接受尾部违约降幅最优或等价邻域，保持探索流动性
            cur_err_cnt = int(torch.sum(torch.tensor([(int(cr_t["H"][0, 7].item()) >> b) & 1 for b in range(d_bits)])).item())
            cand_err_cnt = int(bit_pop[best_k].item())
            if cand_err_cnt <= cur_err_cnt or float(rng.random()) < 0.25:
                w_cur_t = cands_w[best_k : best_k + 1].clone()

    dt = time.perf_counter() - t0
    p_obs = float(hits) / float(max(1, total_evals))
    p_rand = float(2.0 ** (-d_bits))
    gain = p_obs / p_rand if hits > 0 else 0.0
    med_evals = float(np.median(hit_evals_list)) if hit_evals_list else None
    med_steps = float(np.median(hit_steps_list)) if hit_steps_list else None

    return {
        "D": int(d_bits),
        "btc_padding": bool(btc_padding),
        "trials_attempted": trials_attempted,
        "hits": int(hits),
        "total_evals": int(total_evals),
        "median_evals_T_D": med_evals,
        "median_steps": med_steps,
        "observed_hit_prob": p_obs,
        "uniform_random_prob": p_rand,
        "gain_G_D": round(gain, 4),
        "max_trailing_zeros_seen": int(best_tz_seen),
        "elapsed_sec": round(dt, 3),
    }


def run_open_boundary_probe(args: argparse.Namespace) -> Dict[str, Any]:
    """执行开放边界 D-bit 梯度扫描、BTC 对照与标度斜率 α_hat 拟合（总耗时死锁 <= max_total_sec）。"""
    enforce_fp32_determinism()
    t_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    core = get_core()
    torch_eng = TorchCrystalEngine(device=device)

    ckpt_used = None
    if args.ckpt_path and Path(args.ckpt_path).exists():
        model, payload = load_v10_checkpoint(args.ckpt_path, device=device)
        ckpt_used = {"path": str(args.ckpt_path), "step": int(payload.get("step", -1))}
    else:
        cfg = get_model_preset(args.preset)
        cfg["use_checkpoint"] = False
        model = Ouroboros64Net(**cfg).to(device=device, dtype=torch.float32)
        model.eval()

    if args.open_boundary_D is not None:
        d_tiers = [int(args.open_boundary_D)]
    elif args.quick:
        d_tiers = [8, 12]
    else:
        d_tiers = [int(x.strip()) for x in args.d_list.split(",") if x.strip()]

    btc_tiers = (
        [int(x.strip()) for x in args.btc_d_list.split(",") if x.strip()]
        if (args.btc_d_list and not args.open_boundary_D and not args.quick)
        else []
    )

    total_slots = len(d_tiers) + len(btc_tiers)
    results_free: List[Dict[str, Any]] = []
    results_btc: List[Dict[str, Any]] = []

    print(
        f"[开放边界 α 探针] 启动 | device={device} | ckpt={ckpt_used} | D={d_tiers} | BTC={btc_tiers} | 硬上限={args.max_total_sec}s",
        flush=True,
    )

    for idx_t, d_bits in enumerate(d_tiers):
        rem_sec = float(args.max_total_sec) - (time.perf_counter() - t_start)
        if rem_sec <= 2.0:
            break
        rem_slots = max(1, total_slots - idx_t)
        slot_budget = max(5.0, min(42.0, rem_sec / rem_slots))
        row = probe_single_d_tier(
            model=model,
            core=core,
            torch_eng=torch_eng,
            device=device,
            d_bits=d_bits,
            btc_padding=False,
            trials=2 if args.quick else args.trials_per_d,
            batch_cands=args.batch_cands,
            max_steps_per_trial=args.max_steps_per_trial,
            tier_time_budget_sec=slot_budget,
            seed=args.seed,
        )
        results_free.append(row)
        print(
            f"  [Free W | D={d_bits:2d}] hits={row['hits']} | evals={row['total_evals']:,} | "
            f"T(D)={row['median_evals_T_D']} | G(D)={row['gain_G_D']:.3f}x | max_tz={row['max_trailing_zeros_seen']} | {row['elapsed_sec']}s",
            flush=True,
        )

    for idx_b, d_bits in enumerate(btc_tiers):
        rem_sec = float(args.max_total_sec) - (time.perf_counter() - t_start)
        if rem_sec <= 2.0:
            break
        rem_slots = max(1, len(btc_tiers) - idx_b)
        slot_budget = max(5.0, min(30.0, rem_sec / rem_slots))
        row_b = probe_single_d_tier(
            model=model,
            core=core,
            torch_eng=torch_eng,
            device=device,
            d_bits=d_bits,
            btc_padding=True,
            trials=args.trials_per_d,
            batch_cands=args.batch_cands,
            max_steps_per_trial=args.max_steps_per_trial,
            tier_time_budget_sec=slot_budget,
            seed=args.seed + 5000,
        )
        results_btc.append(row_b)
        print(
            f"  [BTC-320Z | D={d_bits:2d}] hits={row_b['hits']} | evals={row_b['total_evals']:,} | "
            f"T(D)={row_b['median_evals_T_D']} | G(D)={row_b['gain_G_D']:.3f}x | max_tz={row_b['max_trailing_zeros_seen']} | {row_b['elapsed_sec']}s",
            flush=True,
        )

    def _fit_alpha(rows: List[Dict[str, Any]]) -> Optional[float]:
        xs = [float(r["D"]) for r in rows if r["median_evals_T_D"] is not None and r["median_evals_T_D"] > 0]
        ys = [math.log2(float(r["median_evals_T_D"])) for r in rows if r["median_evals_T_D"] is not None and r["median_evals_T_D"] > 0]
        sl = _linear_slope(xs, ys)
        return round(sl, 4) if sl is not None else None

    alpha_free = _fit_alpha(results_free)
    alpha_btc = _fit_alpha(results_btc)
    elapsed_all = round(time.perf_counter() - t_start, 2)

    report = {
        "probe_type": "OPEN_BOUNDARY_ALPHA_SCALING_READONLY",
        "checkpoint": ckpt_used,
        "device": str(device),
        "elapsed_sec": elapsed_all,
        "alpha_hat_free": alpha_free,
        "alpha_hat_btc_320zero": alpha_btc,
        "free_boundary_scan": results_free,
        "btc_padding_scan": results_btc,
    }
    out_dir = Path(args.out_dir)
    _atomic_write_json(out_dir / "open_boundary_alpha_report.json", report)
    print(
        f"[开放边界 α 探针完成] 总耗时={elapsed_all}s | α_hat(Free)={alpha_free} | α_hat(BTC-320Z)={alpha_btc} | 已写入 {out_dir / 'open_boundary_alpha_report.json'}",
        flush=True,
    )
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-SHA V10 Open-Boundary Alpha Probe")
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--out_dir", type=str, default="v10_ouroboros64/runs/a100_m3")
    p.add_argument("--preset", type=str, default="a100", choices=["smoke", "a100"])
    p.add_argument("--open_boundary_D", type=int, default=None, help="单档尾零比特数 D (1..32)")
    p.add_argument("--d_list", type=str, default="12,16,20,24,28,32")
    p.add_argument("--btc_d_list", type=str, default="12,16,20,24")
    p.add_argument("--trials_per_d", type=int, default=5)
    p.add_argument("--batch_cands", type=int, default=128)
    p.add_argument("--max_steps_per_trial", type=int, default=200)
    p.add_argument("--max_total_sec", type=float, default=275.0)
    p.add_argument("--seed", type=int, default=202611)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    run_open_boundary_probe(build_argparser().parse_args())
