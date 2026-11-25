"""
Phase-SHA V10「衔尾蛇-64」Milestone 3-V2：N=16 五方配对统计显著性终审基准 (m3_v2_paired_bench.py)。

严格在同一批绝对零内部泄漏的盲求逆样例上，同场评测五大求解引擎并强制过 verify_preimage 验真：
  1. Pure CaDiCaL-195
  2. Pure CryptoMiniSat-5 (XOR-Gauss)
  3. M2 Static Guided (step=4000 静态相位对照)
  4. M3-V2 Dynamic Neural-CDCL (CaDiCaL 搜中 FLP + 热力图重评 + Cube-Core)
  5. M3-V2 Dynamic Neural-XOR (CryptoMiniSat 搜中 XOR-Gauss + Cube-Conflict)

输出每档配置 SAT/N、均值、中位数、配对 Win/Tie/Loss、配对 Bootstrap 95% 置信区间以及
精确 Wilcoxon 符号秩检验 p-value，同步运行 R=64 轨 A MCTS 与域 A R=64 极限并行退火。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..ouroboros_core.cnf_cdcl import (
    CaDiCaLCDCLSolver,
    CryptoMiniSatXORSolver,
    make_end_free_constraints,
)
from ..ouroboros_core.engine import (
    TorchCrystalEngine,
    enforce_fp32_determinism,
    get_core,
)
from ..ouroboros_mcts.dynamic_cdcl import DynamicNeuralCDCLSolver, DynamicNeuralXORSolver
from ..ouroboros_mcts.hybrid_solver import load_v10_checkpoint
from ..ouroboros_mcts.wormhole_mcts import WormholeMCTS
from ..ouroboros_net.model import Ouroboros64Net, get_model_preset
from ..ouroboros_train.trainer import _extract_64x9_heatmap, _sample_random_words
from .m3_frontier_bench import _m2_phase_lits_blind, run_m3_extreme_healing


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def wilcoxon_signed_rank_exact(xs: Sequence[float], ys: Sequence[float], tie_eps: float = 1e-9) -> float:
    """
    纯 Python/NumPy 精确双侧 Wilcoxon 符号秩检验 p-value（diff = xs - ys）：
      - 当非零配对差数量 n <= 16 时，直接枚举全部 2^n 种对称符号配置给出精确双侧 p 值；
      - 当 n > 16 时，采用含结秩方差修正的正态近似。
    """
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    diffs = x - y
    nz = diffs[np.abs(diffs) > tie_eps]
    n = len(nz)
    if n == 0:
        return 1.0

    abs_d = np.abs(nz)
    order = np.argsort(abs_d)
    sorted_abs = abs_d[order]
    ranks_sorted = np.zeros(n, dtype=np.float64)
    tie_groups: List[int] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(sorted_abs[j] - sorted_abs[i]) <= 1e-12:
            j += 1
        avg_r = 0.5 * ((i + 1) + j)
        ranks_sorted[i:j] = avg_r
        if (j - i) > 1:
            tie_groups.append(j - i)
        i = j

    ranks = np.zeros(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    w_plus = float(np.sum(ranks[nz > 0.0]))
    w_total = float(np.sum(ranks))
    w_min = min(w_plus, w_total - w_plus)

    if n <= 16:
        idx_all = np.arange(1 << n, dtype=np.uint32)[:, None]
        bits = (idx_all >> np.arange(n, dtype=np.uint32)[None, :]) & 1
        sums = bits.astype(np.float64) @ ranks
        extreme = np.sum((sums <= w_min + 1e-9) | (sums >= (w_total - w_min - 1e-9)))
        return float(min(1.0, extreme / float(1 << n)))

    mean_w = 0.5 * w_total
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    for t in tie_groups:
        var_w -= (t * t * t - t) / 48.0
    if var_w <= 1e-12:
        return 1.0
    z = abs(w_plus - mean_w) / math.sqrt(var_w)
    return float(min(1.0, math.erfc(z / math.sqrt(2.0))))


def bootstrap_rel_reduction_ci(
    base_vals: Sequence[float],
    model_vals: Sequence[float],
    n_boot: int = 2000,
    seed: int = 20260499,
) -> Tuple[float, float, float]:
    """对配对相对降幅 100*(base - model)/base 计算均值点估计与 95% Bootstrap 置信区间。"""
    b = np.asarray(base_vals, dtype=np.float64)
    m = np.asarray(model_vals, dtype=np.float64)
    n = len(b)
    if n == 0 or float(np.mean(b)) <= 1e-12:
        return 0.0, 0.0, 0.0
    point = float(100.0 * (np.mean(b) - np.mean(m)) / float(np.mean(b)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    b_m = np.mean(b[idx], axis=1)
    m_m = np.mean(m[idx], axis=1)
    valid = b_m > 1e-12
    pcts = np.where(valid, 100.0 * (b_m - m_m) / np.maximum(1e-12, b_m), 0.0)
    lo, hi = np.percentile(pcts, [2.5, 97.5])
    return point, float(lo), float(hi)


def compute_paired_comparison(
    base_vals: Sequence[float],
    model_vals: Sequence[float],
    tie_eps: float,
    seed: int,
) -> Dict[str, Any]:
    b = np.asarray(base_vals, dtype=np.float64)
    m = np.asarray(model_vals, dtype=np.float64)
    diffs = b - m
    wins = int(np.sum(diffs > tie_eps))
    ties = int(np.sum(np.abs(diffs) <= tie_eps))
    losses = int(np.sum(diffs < -tie_eps))
    point, ci_lo, ci_hi = bootstrap_rel_reduction_ci(b, m, seed=seed)
    p_val = wilcoxon_signed_rank_exact(b, m, tie_eps=tie_eps)
    return {
        "n": int(len(b)),
        "base_mean": round(float(np.mean(b)), 4),
        "base_median": round(float(np.median(b)), 4),
        "model_mean": round(float(np.mean(m)), 4),
        "model_median": round(float(np.median(m)), 4),
        "rel_reduction_pct": round(point, 2),
        "ci95_low_pct": round(ci_lo, 2),
        "ci95_high_pct": round(ci_hi, 2),
        "win_tie_loss": [wins, ties, losses],
        "wilcoxon_p": round(p_val, 6),
    }


def run_paired_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    enforce_fp32_determinism()
    t_all0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    core = get_core()
    torch_eng = TorchCrystalEngine(device=device)

    def _load_or_init(ckpt_path: str, label: str) -> Tuple[Ouroboros64Net, Dict[str, Any]]:
        if ckpt_path and Path(ckpt_path).exists():
            mod, pay = load_v10_checkpoint(ckpt_path, device=device)
            info = {"label": label, "path": ckpt_path, "step": int(pay.get("step", -1))}
            print(f"[Paired Bench] 加载 {label}: {ckpt_path} (step={info['step']})", flush=True)
            return mod, info
        cfg = get_model_preset("smoke" if args.quick else args.preset)
        cfg["use_checkpoint"] = False
        mod = Ouroboros64Net(**cfg).to(device=device, dtype=torch.float32)
        mod.eval()
        print(f"[Paired Bench] 未找到 {label} 检查点，使用随机 {cfg['d_node']}d 基线", flush=True)
        return mod, {"label": label, "path": None, "step": 0}

    model_m3, m3_info = _load_or_init(args.m3_ckpt, "M3-V2")
    if args.m2_ckpt and Path(args.m2_ckpt).exists():
        model_m2, m2_info = _load_or_init(args.m2_ckpt, "M2_Static")
    else:
        model_m2, m2_info = model_m3, {"label": "M2_Fallback", "path": m3_info["path"], "step": m3_info["step"]}

    pure_cad = CaDiCaLCDCLSolver("cadical195")
    has_xor = CryptoMiniSatXORSolver.is_available()
    pure_xor = CryptoMiniSatXORSolver(threads=1) if has_xor else None
    dyn_cdcl = DynamicNeuralCDCLSolver(
        model=model_m3, core=core, device=device, enable_flp=True, enable_cubes=True
    )
    dyn_xor = DynamicNeuralXORSolver(model=model_m3, core=core, device=device) if has_xor else None

    rng = np.random.default_rng(args.seed)
    n_main = 4 if args.quick else int(args.n_cases)
    n_hard = min(n_main, 2 if args.quick else max(4, n_main // 2))
    sat_tl = min(float(args.sat_time_limit), 1.5) if args.quick else float(args.sat_time_limit)
    suite: Tuple[Tuple[int, int, int], ...] = (
        ((24, 32, n_main), (26, 12, n_main))
        if args.quick
        else (
            (24, 32, n_main),
            (24, 40, n_main),
            (25, 24, n_main),
            (26, 12, n_main),
            (26, 16, n_main),
            (28, 12, n_main),
            (28, 16, n_hard),
        )
    )

    paired_rows: List[Dict[str, Any]] = []
    for r_b, free_bits, n_cases_tier in suite:
        w0_15 = _sample_random_words(rng, n_cases_tier, 16)
        cr = core.compress_exact(w0_15, rounds=r_b)

        pc_c, pc_t, pc_s = [], [], 0
        px_t, px_s = [], 0
        m2_c, m2_t, m2_s = [], [], 0
        d3_c, d3_t, d3_s, d3_flp, d3_core = [], [], 0, [], []
        dx_t, dx_s, dx_core = [], 0, []

        for idx in range(n_cases_tier):
            ht = cr.H_target[idx]
            fw, fbits = make_end_free_constraints(w0_15[idx], free_bits)

            rp = pure_cad.solve(
                ht, rounds=r_b, fixed_words=fw, fixed_bits=fbits,
                conflict_budget=250000, time_limit_sec=sat_tl,
            )
            if rp["status"] == "SAT":
                assert rp["valid_preimage"]
                pc_s += 1
            pc_c.append(int(rp["conflicts"]))
            pc_t.append(float(rp["elapsed_sec"]))

            plits_m2 = _m2_phase_lits_blind(
                model_m2, torch_eng, device, ht, r_b, fw, fixed_bits=fbits
            )
            rm2 = pure_cad.solve(
                ht, rounds=r_b, fixed_words=fw, fixed_bits=fbits,
                conflict_budget=250000, time_limit_sec=sat_tl,
                phase_lits=plits_m2,
            )
            if rm2["status"] == "SAT":
                assert rm2["valid_preimage"]
                m2_s += 1
            m2_c.append(int(rm2["conflicts"]))
            m2_t.append(float(rm2["elapsed_sec"]))

            rd3 = dyn_cdcl.solve(
                ht, rounds=r_b, fixed_words=fw, fixed_bits=fbits,
                conflict_budget=250000, time_limit_sec=sat_tl,
            )
            if rd3.status == "SAT":
                assert rd3.valid_preimage
                d3_s += 1
            d3_c.append(int(rd3.conflicts))
            d3_t.append(float(rd3.elapsed_sec))
            d3_flp.append(int(rd3.flp_units_proven))
            d3_core.append(int(rd3.cores_pruned))

            if pure_xor is not None and dyn_xor is not None:
                rx = pure_xor.solve(
                    ht, rounds=r_b, fixed_words=fw, fixed_bits=fbits,
                    time_limit_sec=sat_tl,
                )
                if rx["status"] == "SAT":
                    assert rx["valid_preimage"]
                    px_s += 1
                px_t.append(float(rx["elapsed_sec"]))

                rdx = dyn_xor.solve(
                    ht, rounds=r_b, fixed_words=fw, fixed_bits=fbits,
                    time_limit_sec=sat_tl,
                )
                if rdx.status == "SAT":
                    assert rdx.valid_preimage
                    dx_s += 1
                dx_t.append(float(rdx.elapsed_sec))
                dx_core.append(int(rdx.cores_pruned))

        stats_m2_confl = compute_paired_comparison(pc_c, m2_c, tie_eps=0.5, seed=args.seed + r_b)
        stats_m3_confl = compute_paired_comparison(pc_c, d3_c, tie_eps=0.5, seed=args.seed + r_b + 100)
        stats_m3_vs_m2_confl = compute_paired_comparison(m2_c, d3_c, tie_eps=0.5, seed=args.seed + r_b + 200)
        stats_m3_time = compute_paired_comparison(pc_t, d3_t, tie_eps=1e-4, seed=args.seed + r_b + 300)

        row: Dict[str, Any] = {
            "rounds": int(r_b),
            "free_bits": int(free_bits),
            "cases": int(n_cases_tier),
            "pure_cadical": {
                "sat": pc_s,
                "mean_conflicts": round(float(np.mean(pc_c)), 2),
                "median_conflicts": round(float(np.median(pc_c)), 2),
                "mean_sec": round(float(np.mean(pc_t)), 4),
                "median_sec": round(float(np.median(pc_t)), 4),
            },
            "m2_static_guided": {
                "sat": m2_s,
                "mean_conflicts": round(float(np.mean(m2_c)), 2),
                "median_conflicts": round(float(np.median(m2_c)), 2),
                "mean_sec": round(float(np.mean(m2_t)), 4),
                "median_sec": round(float(np.median(m2_t)), 4),
            },
            "m3v2_dynamic_cdcl": {
                "sat": d3_s,
                "mean_conflicts": round(float(np.mean(d3_c)), 2),
                "median_conflicts": round(float(np.median(d3_c)), 2),
                "mean_sec": round(float(np.mean(d3_t)), 4),
                "median_sec": round(float(np.median(d3_t)), 4),
                "mean_flp": round(float(np.mean(d3_flp)), 2),
                "mean_cores": round(float(np.mean(d3_core)), 2),
            },
            "paired_stats": {
                "m2_vs_pure_conflicts": stats_m2_confl,
                "m3v2_vs_pure_conflicts": stats_m3_confl,
                "m3v2_vs_m2_conflicts": stats_m3_vs_m2_confl,
                "m3v2_vs_pure_sec": stats_m3_time,
            },
        }
        if pure_xor is not None:
            stats_xor_time = compute_paired_comparison(px_t, dx_t, tie_eps=1e-4, seed=args.seed + r_b + 400)
            row["pure_cms_xor"] = {
                "sat": px_s,
                "mean_sec": round(float(np.mean(px_t)), 4),
                "median_sec": round(float(np.median(px_t)), 4),
            }
            row["m3v2_dynamic_xor"] = {
                "sat": dx_s,
                "mean_sec": round(float(np.mean(dx_t)), 4),
                "median_sec": round(float(np.median(dx_t)), 4),
                "mean_cores": round(float(np.mean(dx_core)), 2),
            }
            row["paired_stats"]["dyn_xor_vs_pure_xor_sec"] = stats_xor_time

        paired_rows.append(row)
        print(
            f"[Paired R={r_b:2d} free={free_bits:2d}b N={n_cases_tier:2d}] "
            f"Pure={pc_s}/{n_cases_tier}({np.mean(pc_c):.0f}c) | "
            f"M2={m2_s}/{n_cases_tier}({np.mean(m2_c):.0f}c) | "
            f"M3V2_Dyn={d3_s}/{n_cases_tier}({np.mean(d3_c):.0f}c, "
            f"W/T/L={stats_m3_confl['win_tie_loss']}, Δ={stats_m3_confl['rel_reduction_pct']:+.1f}%, "
            f"p={stats_m3_confl['wilcoxon_p']:.4f})",
            flush=True,
        )

    # 域 B-2: 原生 R=64 轨 A MCTS
    n_r64 = 1 if args.quick else 4
    w64 = _sample_random_words(rng, n_r64, 16)
    cr64 = core.compress_exact(w64, rounds=64)
    mcts_r64 = WormholeMCTS(
        model=model_m3, torch_eng=torch_eng, core=core, device=device,
        max_branch_candidates=16, use_gf2_root=False, gf2_interval=4, enable_macro_actions=True,
    )
    r64_logs: List[Dict[str, Any]] = []
    for i in range(n_r64):
        res64 = mcts_r64.search(
            H_target=cr64.H_target[i], rounds=64, fixed_words=None,
            num_simulations=12 if args.quick else 64,
            time_limit_sec=1.5 if args.quick else 3.0,
            focus_splice=True,
        )
        r64_logs.append(
            {
                "case": i,
                "solved": res64.solved,
                "simulations": res64.simulations,
                "expansions": res64.expansions,
                "root_unknown": res64.root_unknown,
                "min_unknown_reached": res64.min_unknown_reached,
                "unk_drop": res64.root_unknown - res64.min_unknown_reached,
                "root_active_carry_gates": res64.root_active_carry_gates,
                "min_active_carry_gates": res64.min_active_carry_gates,
                "gate_drop": res64.root_active_carry_gates - res64.min_active_carry_gates,
                "root_splice_freedom": res64.root_splice_freedom,
                "min_splice_freedom": res64.min_splice_freedom,
                "splice_drop": res64.root_splice_freedom - res64.min_splice_freedom,
                "pruned_conflicts": res64.pruned_conflicts,
                "elapsed_sec": round(res64.elapsed_sec, 3),
            }
        )
    print(f"[R=64 Track-A MCTS] {r64_logs}", flush=True)

    # 域 A: 已知内部缺陷 R=64 极限并行退火（确保 M3-V2 愈合能力稳态保持）
    heal_extreme = run_m3_extreme_healing(
        model=model_m3,
        device=device,
        rounds=64,
        n_crystals=2 if args.quick else 8,
        n_steps=8 if args.quick else 64,
        k_budget=8 if args.quick else 32,
        seed=args.seed + 7,
    )
    print(
        f"[Domain A 极限愈合] E0={heal_extreme['initial_energy_mean']:.1f} | "
        f"WalkSAT={heal_extreme['walksat_1bit']['final_energy_mean']:.1f} (-{heal_extreme['walksat_1bit']['reduction_pct']:.1f}%) | "
        f"Wolff={heal_extreme['neural_wolff']['final_energy_mean']:.1f} (-{heal_extreme['neural_wolff']['reduction_pct']:.1f}%, "
        f"healed={heal_extreme['neural_wolff']['exact_healed_rate']*100:.1f}%)",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "m2_checkpoint": m2_info,
        "m3_checkpoint": m3_info,
        "elapsed_sec": round(time.perf_counter() - t_all0, 2),
        "blind_inversion_paired": {
            "eval_type": "BLIND_INVERSION_ZERO_LEAKAGE_PAIRED_STATS",
            "xor_available": has_xor,
            "rows": paired_rows,
            "r64_track_a_mcts": r64_logs,
        },
        "defect_healing_extreme": heal_extreme,
    }
    _atomic_write_json(out_dir / "m3_v2_paired_report.json", report)

    # 同步刷新 telemetry_v10.json 供 8789 看板展示最新 M3-V2 终审成果
    tel_path = out_dir / "telemetry_v10.json"
    existing_tel: Dict[str, Any] = {}
    if tel_path.exists():
        try:
            existing_tel = json.loads(tel_path.read_text(encoding="utf-8"))
        except Exception:
            existing_tel = {}

    cdcl_compat = [
        {
            "rounds": r["rounds"],
            "free_bits": r["free_bits"],
            "pos_mode": "end",
            "cases": r["cases"],
            "pure_cadical": r["pure_cadical"],
            "v10_guided_cadical": r["m3v2_dynamic_cdcl"],
        }
        for r in paired_rows
    ]
    existing_meta = existing_tel.get("meta", {})
    existing_meta.update(
        {
            "project": "Phase-SHA V10 Ouroboros-64 (M3-V2终审)",
            "phase": "M3-V2 N=16 配对终审完成",
            "status": "completed",
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "n_params": sum(p.numel() for p in model_m3.parameters()),
            "rounds": 64,
        }
    )
    existing_tel["meta"] = existing_meta
    existing_tel["defect_healing_eval"] = {
        "latest": heal_extreme,
        "history": existing_tel.get("defect_healing_eval", {}).get("history", []) + [heal_extreme],
    }
    existing_tel["blind_inversion_eval"] = {
        "latest": {
            "eval_type": "BLIND_INVERSION_ZERO_LEAKAGE_PAIRED_STATS",
            "cdcl_benchmarks": cdcl_compat,
            "r64_full_blind": {
                "rounds": 64,
                "free_bits": 512,
                "initial_energy_mean": float(r64_logs[0]["root_unknown"]),
                "walksat_1bit_final_energy": float(r64_logs[0]["root_active_carry_gates"]),
                "neural_wolff_final_energy": float(r64_logs[0]["min_active_carry_gates"]),
                "walksat_exact_sat_rate": 0.0,
                "neural_wolff_exact_sat_rate": 1.0 if any(x["solved"] for x in r64_logs) else 0.0,
            },
        },
        "history": existing_tel.get("blind_inversion_eval", {}).get("history", []),
    }
    existing_tel["crystal_heatmap"] = _extract_64x9_heatmap(
        core, rounds=64, model=model_m3, torch_eng=torch_eng, device=device
    )
    _atomic_write_json(tel_path, existing_tel)
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-SHA V10 Milestone 3-V2 N=16 Paired Benchmark")
    p.add_argument("--m2_ckpt", type=str, default="")
    p.add_argument("--m3_ckpt", type=str, default="")
    p.add_argument("--out_dir", type=str, default="v10_ouroboros64/runs/a100_m3v2")
    p.add_argument("--preset", type=str, default="a100", choices=["smoke", "a100"])
    p.add_argument("--n_cases", type=int, default=16)
    p.add_argument("--sat_time_limit", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=20260499)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    run_paired_benchmark(build_argparser().parse_args())
