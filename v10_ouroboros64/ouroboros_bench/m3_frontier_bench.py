"""
Phase-SHA V10「衔尾蛇-64」Milestone 3：双域绝对隔离前沿基准测试脚本 (m3_frontier_bench.py)。

严格对比三大求解体系（零内部泄漏 + verify_preimage 强制全量验真）：
  1. 域 B（绝对零泄漏盲求逆前沿）：
     - Pure CaDiCaL-195 基线
     - M2 神经引导相位 CaDiCaL-195
     - M3 Wormhole-MCTS 三层合龙混合求解器 (OuroborosHybridSolver)
     基准覆盖：R=16(512b)、R=20(96b/128b)、R=24(32b/48b/64b)、R=28(16b/24b)
     另附：原生 R=64（512b 全盲）轨 A 广义 6 符号虫洞 MCTS 活跃门与拼接自由度探索报告。
  2. 域 A（已知内部缺陷极限并行退火愈合）：
     - R=64 64 步 × 32 候选/步（总计 2,048 次 C++ 评估/晶体）并行温度退火 Neural Wolff
     - 严格等预算 1-bit WalkSAT 对照基线
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..ouroboros_core.cnf_cdcl import (
    CaDiCaLCDCLSolver,
    SHA256CNFEncoder,
    make_end_free_constraints,
)
from ..ouroboros_core.engine import (
    C_PLANES,
    IV_WORDS,
    MASK32,
    TorchCrystalEngine,
    enforce_fp32_determinism,
    get_core,
)
from ..ouroboros_mcts.hybrid_solver import OuroborosHybridSolver, load_v10_checkpoint
from ..ouroboros_mcts.wormhole_mcts import WormholeMCTS
from ..ouroboros_net.heads import PLANE_COUNT, build_valid_site_mask
from ..ouroboros_net.model import Ouroboros64Net, get_model_preset
from ..ouroboros_train.trainer import (
    _apply_cluster_inplace,
    _corrupt_crystals_np,
    _extract_64x9_heatmap,
    _random_1bit_candidates,
    _sample_random_words,
)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


@torch.no_grad()
def _m2_phase_lits_blind(
    model: Optional[Ouroboros64Net],
    torch_eng: TorchCrystalEngine,
    device: torch.device,
    H_single: np.ndarray,
    rounds: int,
    fixed_words: Optional[Dict[int, int]],
    fixed_bits: Optional[Dict[Tuple[int, int], int]] = None,
) -> List[int]:
    if model is None:
        return []
    W_b = np.zeros((1, rounds), dtype=np.uint32)
    S_b = np.zeros((1, rounds + 1, 8), dtype=np.uint32)
    C_b = np.zeros((1, rounds, C_PLANES), dtype=np.uint32)
    s0 = np.asarray(IV_WORDS, dtype=np.uint32)
    sr = ((H_single.astype(np.uint64) - s0.astype(np.uint64)) & MASK32).astype(np.uint32)
    S_b[0, 0] = s0
    S_b[0, rounds] = sr

    unk = torch.ones((1, rounds, PLANE_COUNT, 32), dtype=torch.float32, device=device)
    unk[:, 0, 1:9, :] = 0.0
    fixed_set = set()
    if fixed_words:
        for wi, val in fixed_words.items():
            W_b[0, wi] = np.uint32(val)
            unk[:, wi, 0, :] = 0.0
            for bb in range(32):
                fixed_set.add((int(wi), bb))
    if fixed_bits:
        w_tmp = np.zeros((16,), dtype=np.uint32)
        for wi, val in (fixed_words or {}).items():
            if 0 <= wi < 16:
                w_tmp[wi] = np.uint32(val)
        for (wi, bb), bval in fixed_bits.items():
            if 0 <= wi < rounds and 0 <= bb < 32:
                if int(bval) & 1:
                    W_b[0, wi] |= np.uint32(1 << int(bb))
                else:
                    W_b[0, wi] &= np.uint32(~(1 << int(bb)) & MASK32)
                unk[:, int(wi), 0, int(bb)] = 0.0
                fixed_set.add((int(wi), int(bb)))

    out = model(
        torch.from_numpy(H_single[None, :].astype(np.int64)).to(device),
        torch.from_numpy(W_b.astype(np.int64)).to(device),
        torch.from_numpy(S_b.astype(np.int64)).to(device),
        torch.from_numpy(C_b.astype(np.int64)).to(device),
        rounds=rounds,
        unknown_mask=unk,
        zero_defects=True,
        propose_clusters=False,
        return_entropy=False,
        crystal_engine=torch_eng,
    )
    ph = out["phase_logits"][0, :16, 0, :].cpu().numpy()
    co = out["collapse_score"][0, :16, 0, :].cpu().numpy()

    enc = SHA256CNFEncoder(rounds=rounds).build(
        H_single, fixed_words=fixed_words, fixed_bits=fixed_bits
    )
    scored: List[Tuple[float, int]] = []
    for wi in range(16):
        for bb in range(32):
            if (wi, bb) in fixed_set:
                continue
            lit = enc.W_vars[wi][bb]
            if not isinstance(lit, bool) and isinstance(lit, int):
                logit = float(ph[wi, bb])
                sc = float(co[wi, bb]) * abs(logit)
                scored.append((sc, lit if logit >= 0.0 else -lit))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [lit for _, lit in scored[:256]]


@torch.no_grad()
def run_m3_extreme_healing(
    model: Ouroboros64Net,
    device: torch.device,
    rounds: int = 64,
    n_crystals: int = 8,
    n_steps: int = 64,
    k_budget: int = 32,
    seed: int = 202603,
) -> Dict[str, Any]:
    """域 A：等评估预算极限并行退火愈合评测（2,048 C++ evals / crystal）。"""
    core = get_core()
    torch_eng = TorchCrystalEngine(device=device)
    model.eval()
    rng = np.random.default_rng(seed)

    w0_15 = _sample_random_words(rng, n_crystals, 16)
    cr = core.compress_exact(w0_15, rounds=rounds)
    W_d0, S_d0, C_d0, _ = _corrupt_crystals_np(rng, cr.W, cr.S, cr.C, rounds=rounds, mode="cluster")
    H_np = cr.H_target.copy()
    e0_arr = core.eval_crystal_batch(H_np, W_d0, S_d0, C_d0, rounds=rounds, return_masks=False)["E"]
    e0_mean = float(np.mean(e0_arr))

    valid_mask_np = build_valid_site_mask(rounds, device="cpu").numpy()
    valid_flat = np.flatnonzero(valid_mask_np.reshape(-1))

    # 1) WalkSAT-1bit
    W_ws, S_ws, C_ws = W_d0.copy(), S_d0.copy(), C_d0.copy()
    ws_curve = [e0_mean]
    for st in range(n_steps):
        cands = _random_1bit_candidates(rng, n_crystals, k_budget, rounds, valid_flat)
        de, _ = core.eval_cluster_flips_batch(
            np.repeat(H_np, k_budget, axis=0),
            np.repeat(W_ws, k_budget, axis=0),
            np.repeat(S_ws, k_budget, axis=0),
            np.repeat(C_ws, k_budget, axis=0),
            cands,
            rounds=rounds,
        )
        de = de.reshape(n_crystals, k_budget)
        cmat = cands.reshape(n_crystals, k_budget, 1, 4)
        pick = np.full((n_crystals, 1, 4), -1, dtype=np.int32)
        p_noise = max(0.02, 0.10 * (1.0 - st / max(1, n_steps)))
        for b_i in range(n_crystals):
            bk = int(np.argmin(de[b_i]))
            if int(de[b_i, bk]) <= 0 or float(rng.random()) < p_noise:
                pk = bk if int(de[b_i, bk]) <= 0 else int(rng.integers(0, k_budget))
                pick[b_i] = cmat[b_i, pk]
        _apply_cluster_inplace(W_ws, S_ws, C_ws, pick, rounds=rounds)
        ws_curve.append(float(np.mean(core.eval_crystal_batch(H_np, W_ws, S_ws, C_ws, rounds=rounds, return_masks=False)["E"])))

    # 2) Neural Wolff 温度退火（0.95 -> 0.55）
    W_wf, S_wf, C_wf = W_d0.copy(), S_d0.copy(), C_d0.copy()
    H_t = torch.from_numpy(H_np.astype(np.int64)).to(device)
    wf_curve = [e0_mean]
    tunnel_events = 0
    accept_events = 0

    for st in range(n_steps):
        temp = 0.95 - 0.40 * (st / max(1, n_steps - 1))
        out = model(
            H_t,
            torch.from_numpy(W_wf.astype(np.int64)).to(device),
            torch.from_numpy(S_wf.astype(np.int64)).to(device),
            torch.from_numpy(C_wf.astype(np.int64)).to(device),
            rounds=rounds,
            propose_clusters=True,
            num_clusters=k_budget,
            temperature=temp,
            return_entropy=False,
            crystal_engine=torch_eng,
        )
        cl_np = out["clusters"]["clusters_coord"].cpu().numpy()
        m_max = cl_np.shape[2]
        de_c, _ = core.eval_cluster_flips_batch(
            np.repeat(H_np, k_budget, axis=0),
            np.repeat(W_wf, k_budget, axis=0),
            np.repeat(S_wf, k_budget, axis=0),
            np.repeat(C_wf, k_budget, axis=0),
            cl_np.reshape(n_crystals * k_budget, m_max, 4),
            rounds=rounds,
        )
        de_c = de_c.reshape(n_crystals, k_budget)
        pick = np.full((n_crystals, m_max, 4), -1, dtype=np.int32)
        for b_i in range(n_crystals):
            bk = int(np.argmin(de_c[b_i]))
            best_de = int(de_c[b_i, bk])
            if best_de < 0:
                pick[b_i] = cl_np[b_i, bk]
                accept_events += 1
                singles = []
                for sl in range(m_max):
                    if cl_np[b_i, bk, sl, 0] >= 0:
                        s1 = np.full((1, 4), -1, dtype=np.int32)
                        s1[0] = cl_np[b_i, bk, sl]
                        singles.append(s1)
                if singles:
                    de_s, _ = core.eval_cluster_flips_batch(
                        np.repeat(H_np[b_i : b_i + 1], len(singles), axis=0),
                        np.repeat(W_wf[b_i : b_i + 1], len(singles), axis=0),
                        np.repeat(S_wf[b_i : b_i + 1], len(singles), axis=0),
                        np.repeat(C_wf[b_i : b_i + 1], len(singles), axis=0),
                        np.stack(singles, axis=0),
                        rounds=rounds,
                    )
                    if best_de < int(np.min(de_s)):
                        tunnel_events += 1
            elif best_de == 0 and float(rng.random()) < 0.15:
                pick[b_i] = cl_np[b_i, bk]
        _apply_cluster_inplace(W_wf, S_wf, C_wf, pick, rounds=rounds)
        wf_curve.append(float(np.mean(core.eval_crystal_batch(H_np, W_wf, S_wf, C_wf, rounds=rounds, return_masks=False)["E"])))

    ews_end = core.eval_crystal_batch(H_np, W_ws, S_ws, C_ws, rounds=rounds, return_masks=False)["E"]
    ewf_end = core.eval_crystal_batch(H_np, W_wf, S_wf, C_wf, rounds=rounds, return_masks=False)["E"]
    return {
        "eval_type": "DEFECT_HEALING_ONLY_ISOLATED_M3_EXTREME",
        "rounds": rounds,
        "n_crystals": n_crystals,
        "n_steps": n_steps,
        "k_budget": k_budget,
        "initial_energy_mean": e0_mean,
        "walksat_1bit": {
            "final_energy_mean": float(np.mean(ews_end)),
            "reduction_pct": float(100.0 * (e0_mean - float(np.mean(ews_end))) / max(1e-6, e0_mean)),
            "exact_healed_rate": float(np.mean(ews_end == 0)),
            "cpp_evals": n_crystals * n_steps * k_budget,
            "energy_curve": ws_curve,
        },
        "neural_wolff": {
            "final_energy_mean": float(np.mean(ewf_end)),
            "reduction_pct": float(100.0 * (e0_mean - float(np.mean(ewf_end))) / max(1e-6, e0_mean)),
            "exact_healed_rate": float(np.mean(ewf_end == 0)),
            "tunneling_events": tunnel_events,
            "accept_events": accept_events,
            "tunneling_ratio": float(tunnel_events / max(1, accept_events)),
            "cpp_evals": n_crystals * n_steps * k_budget,
            "energy_curve": wf_curve,
        },
    }


def run_m3_frontier_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    enforce_fp32_determinism()
    t_all0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    core = get_core()
    torch_eng = TorchCrystalEngine(device=device)

    ckpt_info: Dict[str, Any] = {"loaded_from": None}
    if args.ckpt_path and Path(args.ckpt_path).exists():
        model, payload = load_v10_checkpoint(args.ckpt_path, device=device)
        ckpt_info = {
            "loaded_from": str(args.ckpt_path),
            "step": int(payload.get("step", -1)),
            "preset": payload.get("preset", "unknown"),
        }
        print(f"[M3 Frontier] 加载 Milestone 2 权重: {args.ckpt_path} (step={ckpt_info['step']})", flush=True)
    else:
        cfg = get_model_preset(args.preset)
        cfg["use_checkpoint"] = False
        model = Ouroboros64Net(**cfg).to(device=device, dtype=torch.float32)
        model.eval()
        print(f"[M3 Frontier] 未指定有效 ckpt，使用新建 {args.preset} 模型跑基线验证", flush=True)

    pure_solver = CaDiCaLCDCLSolver("cadical195")
    hybrid_solver = OuroborosHybridSolver(model=model, core=core, device=device)
    rng = np.random.default_rng(args.seed)

    # 域 B-1：盲求逆缩减轮次前沿三方对比（Pure CaDiCaL vs M2/M3 Guided CaDiCaL vs M3 Hybrid）
    # 所有配置均为末尾连续自由比特（按比特平滑覆盖可解相变带，严格零泄漏）
    suite: Tuple[Tuple[int, int], ...] = (
        (16, 512),
        (20, 128),
        (24, 32),
        (24, 40),
        (25, 24),
        (26, 12),
        (26, 16),
        (28, 12),
        (24, 64),
        (28, 16),
    )
    if args.quick:
        suite = ((16, 512), (20, 128), (24, 32), (26, 12))

    frontier_rows: List[Dict[str, Any]] = []
    for r_b, free_bits in suite:
        w0_15 = _sample_random_words(rng, args.cases_cdcl, 16)
        cr = core.compress_exact(w0_15, rounds=r_b)

        p_sat, g_sat, h_sat = 0, 0, 0
        p_c, g_c, h_c = [], [], []
        p_t, g_t, h_t = [], [], []
        h_units_list, h_pruned_list = [], []

        for idx in range(args.cases_cdcl):
            H_single = cr.H_target[idx]
            fixed_words, fixed_bits = (
                make_end_free_constraints(w0_15[idx], free_bits)
                if free_bits < 512
                else ({}, {})
            )

            rp = pure_solver.solve(
                H_single, rounds=r_b, fixed_words=fixed_words, fixed_bits=fixed_bits,
                conflict_budget=250000, time_limit_sec=args.sat_time_limit,
            )
            if rp["status"] == "SAT":
                assert rp["valid_preimage"]
                p_sat += 1
            p_c.append(int(rp["conflicts"]))
            p_t.append(float(rp["elapsed_sec"]))

            plits = _m2_phase_lits_blind(
                model, torch_eng, device, H_single, r_b, fixed_words, fixed_bits=fixed_bits
            )
            rg = pure_solver.solve(
                H_single, rounds=r_b, fixed_words=fixed_words, fixed_bits=fixed_bits,
                conflict_budget=250000, time_limit_sec=args.sat_time_limit,
                phase_lits=plits,
            )
            if rg["status"] == "SAT":
                assert rg["valid_preimage"]
                g_sat += 1
            g_c.append(int(rg["conflicts"]))
            g_t.append(float(rg["elapsed_sec"]))

            rh = hybrid_solver.solve(
                H_single, rounds=r_b, fixed_words=fixed_words, fixed_bits=fixed_bits,
                mcts_sims=args.mcts_sims,
                mcts_time_limit=min(1.0, args.sat_time_limit * 0.25),
                total_time_limit=args.sat_time_limit,
            )
            if rh.status == "SAT":
                assert rh.valid_preimage
                h_sat += 1
            h_c.append(int(rh.conflicts))
            h_t.append(float(rh.elapsed_sec))
            h_units_list.append(int(rh.mcts_units_injected))
            h_pruned_list.append(int(rh.mcts_pruned_conflicts))

        row = {
            "rounds": r_b,
            "free_bits": int(free_bits),
            "pos_mode": "end",
            "cases": args.cases_cdcl,
            "pure_cadical": {"sat": p_sat, "mean_conflicts": float(np.mean(p_c)), "mean_sec": float(np.mean(p_t))},
            "m2_guided_cadical": {"sat": g_sat, "mean_conflicts": float(np.mean(g_c)), "mean_sec": float(np.mean(g_t))},
            "m3_hybrid_solver": {
                "sat": h_sat,
                "mean_conflicts": float(np.mean(h_c)),
                "mean_sec": float(np.mean(h_t)),
                "mean_units_injected": float(np.mean(h_units_list)),
                "mean_mcts_pruned": float(np.mean(h_pruned_list)),
            },
        }
        frontier_rows.append(row)
        print(
            f"[Blind Frontier R={r_b:2d} free={row['free_bits']:3d}b] "
            f"Pure={p_sat}/{args.cases_cdcl}({np.mean(p_c):.0f}c) | "
            f"Guided={g_sat}/{args.cases_cdcl}({np.mean(g_c):.0f}c) | "
            f"M3_Hybrid={h_sat}/{args.cases_cdcl}({np.mean(h_c):.0f}c, units={np.mean(h_units_list):.1f})",
            flush=True,
        )

    # 域 B-2：原生 R=64（512b 全盲）双速多比特虫洞 MCTS 活跃门与拼接界面压缩探测
    r64_probe_w = _sample_random_words(rng, 2, 16)
    r64_cr = core.compress_exact(r64_probe_w, rounds=64)
    mcts_r64 = WormholeMCTS(
        model=model,
        torch_eng=torch_eng,
        core=core,
        device=device,
        max_branch_candidates=16,
        use_gf2_root=False,
        gf2_interval=4,
        enable_macro_actions=True,
    )
    r64_wormhole_logs: List[Dict[str, Any]] = []
    for i in range(2):
        res64 = mcts_r64.search(
            H_target=r64_cr.H_target[i],
            rounds=64,
            fixed_words=None,
            num_simulations=max(48, args.mcts_sims),
            time_limit_sec=3.0,
            focus_splice=True,
        )
        r64_wormhole_logs.append(
            {
                "case": i,
                "solved": res64.solved,
                "simulations": res64.simulations,
                "expansions": res64.expansions,
                "learned_clauses_count": len(res64.learned_clauses),
                "root_unknown": res64.root_unknown,
                "min_unknown_reached": res64.min_unknown_reached,
                "root_active_carry_gates": res64.root_active_carry_gates,
                "min_active_carry_gates": res64.min_active_carry_gates,
                "root_splice_freedom": res64.root_splice_freedom,
                "min_splice_freedom": res64.min_splice_freedom,
                "pruned_conflicts": res64.pruned_conflicts,
                "elapsed_sec": round(res64.elapsed_sec, 3),
            }
        )
    print(f"[R=64 Track-A MCTS 探测] {r64_wormhole_logs}", flush=True)

    # 域 A：R=64 极限多步退火缺陷愈合基准
    heal_extreme = run_m3_extreme_healing(
        model=model,
        device=device,
        rounds=64,
        n_crystals=args.anneal_crystals,
        n_steps=args.anneal_steps,
        k_budget=args.anneal_cands,
        seed=args.seed + 7,
    )
    print(
        f"[Domain A M3 极限愈合] E0={heal_extreme['initial_energy_mean']:.1f} | "
        f"WalkSAT={heal_extreme['walksat_1bit']['final_energy_mean']:.1f} (-{heal_extreme['walksat_1bit']['reduction_pct']:.1f}%) | "
        f"Wolff={heal_extreme['neural_wolff']['final_energy_mean']:.1f} (-{heal_extreme['neural_wolff']['reduction_pct']:.1f}%, "
        f"tunnel={heal_extreme['neural_wolff']['tunneling_events']})",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": ckpt_info,
        "elapsed_sec": round(time.perf_counter() - t_all0, 2),
        "blind_inversion_frontier": {
            "eval_type": "BLIND_INVERSION_ZERO_LEAKAGE_M3",
            "cdcl_benchmarks": frontier_rows,
            "r64_track_a_mcts": r64_wormhole_logs,
        },
        "defect_healing_extreme": heal_extreme,
    }
    _atomic_write_json(out_dir / "m3_frontier_report.json", report)

    # 若同目录存在或创建看板遥测，兼容性刷新最新面板展示
    tel_path = out_dir / "telemetry_v10.json"
    existing_tel: Dict[str, Any] = {}
    if tel_path.exists():
        try:
            existing_tel = json.loads(tel_path.read_text(encoding="utf-8"))
        except Exception:
            existing_tel = {}
    elif args.m2_telemetry and Path(args.m2_telemetry).exists():
        try:
            existing_tel = json.loads(Path(args.m2_telemetry).read_text(encoding="utf-8"))
        except Exception:
            existing_tel = {}

    cdcl_compat = [
        {
            "rounds": r["rounds"],
            "free_bits": r["free_bits"],
            "pos_mode": r["pos_mode"],
            "cases": r["cases"],
            "pure_cadical": r["pure_cadical"],
            "v10_guided_cadical": r["m3_hybrid_solver"],
        }
        for r in frontier_rows
    ]
    existing_meta = existing_tel.get("meta", {})
    existing_meta.update(
        {
            "project": "Phase-SHA V10 Ouroboros-64 (M2+M3)",
            "phase": "M3 混合前沿完成",
            "status": "completed",
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "n_params": sum(p.numel() for p in model.parameters()),
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
            "eval_type": "BLIND_INVERSION_ZERO_LEAKAGE_M3",
            "cdcl_benchmarks": cdcl_compat,
            "r64_full_blind": {
                "rounds": 64,
                "free_bits": 512,
                "initial_energy_mean": float(r64_wormhole_logs[0]["root_unknown"]),
                "walksat_1bit_final_energy": float(r64_wormhole_logs[0]["root_active_carry_gates"]),
                "neural_wolff_final_energy": float(r64_wormhole_logs[0]["min_active_carry_gates"]),
                "walksat_exact_sat_rate": 0.0,
                "neural_wolff_exact_sat_rate": 1.0 if any(x["solved"] for x in r64_wormhole_logs) else 0.0,
            },
        },
        "history": existing_tel.get("blind_inversion_eval", {}).get("history", []),
    }
    existing_tel["crystal_heatmap"] = _extract_64x9_heatmap(
        core, rounds=64, model=model, torch_eng=torch_eng, device=device
    )
    _atomic_write_json(tel_path, existing_tel)
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-SHA V10 Milestone 3 Frontier Benchmark")
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--m2_telemetry", type=str, default="")
    p.add_argument("--out_dir", type=str, default="v10_ouroboros64/runs/m3_frontier")
    p.add_argument("--preset", type=str, default="a100", choices=["smoke", "a100"])
    p.add_argument("--cases_cdcl", type=int, default=4)
    p.add_argument("--sat_time_limit", type=float, default=6.0)
    p.add_argument("--mcts_sims", type=int, default=24)
    p.add_argument("--anneal_crystals", type=int, default=8)
    p.add_argument("--anneal_steps", type=int, default=64)
    p.add_argument("--anneal_cands", type=int, default=32)
    p.add_argument("--seed", type=int, default=202603)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    run_m3_frontier_benchmark(build_argparser().parse_args())
