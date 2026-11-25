"""
Phase-SHA V10「衔尾蛇-64」M3 同条件配对实测基准 (`run_prototype_paired_bench.py`)。

严格隔离、只读加载 `v10_ouroboros64/runs/a100_m3v2/checkpoints/best_m3v2.pt`（不修改既有核心源码与存档），
锁定纯 IEEE-754 FP32（`allow_tf32=False`），以与 `m3_v2_paired_bench.py` 完全相同的
`rng = np.random.default_rng(seed)` 顺序抽样流执行顺序单实例配对实测：
  1. 域 B 7 档配对对比（Pure CaDiCaL / Baseline M3-V2 Dyn CDCL / Baseline Pure CMS-XOR [SafeSubprocess] /
     Prototype P1+P2 CleanDynCDCL / Prototype P3 CarryLockHybridRouter）；
  2. R=64 轨 A MCTS 配对对比（Baseline WormholeMCTS vs Prototype P4 LocalWindowWormholeMCTS）
     + R=64 free_bits in {4, 6} 严格零泄漏过约束可解验证探针；
  3. 域 A R=64 物理隔离复核（run_m3_extreme_healing, N=8, 64步, K=32, seed+7）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v10_ouroboros64.ouroboros_bench.m3_frontier_bench import run_m3_extreme_healing
from v10_ouroboros64.ouroboros_bench.m3_v2_paired_bench import (
    _atomic_write_json,
    compute_paired_comparison,
)
from v10_ouroboros64.ouroboros_core.cnf_cdcl import (
    CaDiCaLCDCLSolver,
    CryptoMiniSatXORSolver,
    SHA256XORCNFEncoder,
    extract_w0_15_from_cryptosat,
    make_end_free_constraints,
)
from v10_ouroboros64.ouroboros_core.engine import (
    TorchCrystalEngine,
    enforce_fp32_determinism,
    get_core,
)
from v10_ouroboros64.ouroboros_mcts.dynamic_cdcl import DynamicNeuralCDCLSolver
from v10_ouroboros64.ouroboros_mcts.hybrid_solver import load_v10_checkpoint
from v10_ouroboros64.ouroboros_mcts.wormhole_mcts import WormholeMCTS
from v10_ouroboros64.ouroboros_train.trainer import _sample_random_words
from v10_ouroboros64.prototypes import (
    CarryLockHybridRouter,
    CleanDynamicNeuralCDCLSolver,
    LocalWindowWormholeMCTS,
    SafeCMSSubprocessRunner,
)

DEFAULT_CKPT = "v10_ouroboros64/runs/a100_m3v2/checkpoints/best_m3v2.pt"
DEFAULT_OUT_JSON = "v10_ouroboros64/prototypes/prototype_bench_report.json"


def _check_fp32_locked() -> Dict[str, Any]:
    enforce_fp32_determinism()
    cuda_avail = torch.cuda.is_available()
    matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32) if cuda_avail else False
    cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32) if cuda_avail else False
    assert (not matmul_tf32) and (not cudnn_tf32), "TF32 必须严格禁用"
    return {
        "cuda_available": cuda_avail,
        "device_name": torch.cuda.get_device_name(0) if cuda_avail else "CPU",
        "allow_tf32_matmul": matmul_tf32,
        "allow_tf32_cudnn": cudnn_tf32,
        "fp32_locked": (not matmul_tf32) and (not cudnn_tf32),
    }


def _summarize_solver_tier(
    sat_cnt: int,
    n_cases: int,
    confl_list: List[int],
    sec_list: List[float],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "sat": int(sat_cnt),
        "cases": int(n_cases),
        "sat_rate": round(float(sat_cnt / max(1, n_cases)), 4),
        "mean_conflicts": round(float(np.mean(confl_list)), 2) if confl_list else 0.0,
        "median_conflicts": round(float(np.median(confl_list)), 2) if confl_list else 0.0,
        "mean_sec": round(float(np.mean(sec_list)), 4) if sec_list else 0.0,
        "median_sec": round(float(np.median(sec_list)), 4) if sec_list else 0.0,
    }
    if extra:
        d.update(extra)
    return d


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    t_all0 = time.perf_counter()
    fp32_info = _check_fp32_locked()
    device = torch.device("cuda" if fp32_info["cuda_available"] and not args.cpu else "cpu")
    core = get_core()
    torch_eng = TorchCrystalEngine(device=device)

    ckpt_path = Path(args.ckpt_path)
    assert ckpt_path.exists(), f"未找到只读检查点: {ckpt_path}"
    model_m3, pay = load_v10_checkpoint(ckpt_path, device=device)
    m3_info = {
        "path": str(ckpt_path),
        "step": int(pay.get("step", -1)),
        "n_params": sum(p.numel() for p in model_m3.parameters()),
    }
    print(
        f"[ProtoBench] FP32 锁定={fp32_info['fp32_locked']} | 设备={device} ({fp32_info['device_name']}) | "
        f"只读权重={ckpt_path} (step={m3_info['step']}, params={m3_info['n_params']:,})",
        flush=True,
    )

    out_path = Path(args.out_json)
    pure_cad = CaDiCaLCDCLSolver("cadical195")
    base_dyn = DynamicNeuralCDCLSolver(
        model=model_m3, core=core, device=device, enable_flp=True, enable_cubes=True
    )
    has_xor = CryptoMiniSatXORSolver.is_available()
    cms_runner = SafeCMSSubprocessRunner(threads=1) if has_xor else None
    p1_p2_solver = CleanDynamicNeuralCDCLSolver(
        model=model_m3, core=core, device=device,
        phase_cal_temp=4.5, phase_conf_thresh=0.08, flp_asym_thresh=16,
        enable_flp=True, flp_top_k=8, long_chunk_conflicts=120000,
        enable_zero_waste_probe=True, probe_n_max_1bit=12,
    )
    p3_router = CarryLockHybridRouter(
        model=model_m3, core=core, device=device,
        cms_runner=cms_runner, cdcl_solver=p1_p2_solver,
    )

    verify_total = 0
    verify_passed = 0

    def _record_verify(w_rec: Optional[np.ndarray], ht: np.ndarray, rounds: int, tag: str) -> None:
        nonlocal verify_total, verify_passed
        assert w_rec is not None, f"{tag} 返回 SAT 但 w0_15 为 None"
        verify_total += 1
        ok = bool(core.verify_preimage(w_rec, ht, rounds=rounds)[0])
        assert ok, f"{tag} 未通过 verify_preimage 校验 (R={rounds})"
        verify_passed += 1

    rng = np.random.default_rng(int(args.seed))
    n_main = 2 if args.fast else 16
    n_hard = 2 if args.fast else 8
    sat_tl = min(float(args.sat_time_limit), 2.0) if args.fast else float(args.sat_time_limit)
    confl_budget = int(args.conflict_budget)

    suite: Tuple[Tuple[int, int, int], ...] = (
        (
            (24, 32, n_main),
            (24, 40, n_main),
            (25, 24, n_main),
            (26, 12, n_main),
            (26, 16, n_main),
            (28, 12, n_main),
            (28, 16, n_hard),
        )
    )

    report: Dict[str, Any] = {
        "benchmark": "M3_PROTOTYPE_PAIRED_BENCHMARK",
        "fp32_check": fp32_info,
        "checkpoint": m3_info,
        "config": {
            "seed": int(args.seed),
            "sat_time_limit": sat_tl,
            "conflict_budget": confl_budget,
            "fast_mode": bool(args.fast),
            "xor_available": has_xor,
            "sequential_single_instance": True,
        },
        "domain_b_tiers": [],
        "r64_track_a_mcts": {},
        "r64_overconstrained_probe": {},
        "domain_a_isolation_check": {},
        "verify_preimage_summary": {},
        "elapsed_sec": 0.0,
    }

    try:
        # =====================================================================
        # Part 1: 域 B 7 档顺序单实例同条件配对实测
        # =====================================================================
        for tier_idx, (r_b, free_bits, n_cases_tier) in enumerate(suite, start=1):
            t_tier0 = time.perf_counter()
            w0_15 = _sample_random_words(rng, n_cases_tier, 16)
            cr = core.compress_exact(w0_15, rounds=r_b)

            pc_c, pc_t, pc_s = [], [], 0
            b3_c, b3_t, b3_s = [], [], 0
            px_t, px_s, px_crashes = [], 0, 0
            p1_c, p1_t, p1_s, p1_flp, p1_bin, p1_gate_pass = [], [], 0, [], [], []
            p3_c, p3_t, p3_s, p3_zones, p3_rlock, p3_dof, p3_equiv = [], [], 0, {}, [], [], []

            print(
                f"\n[Tier {tier_idx}/7] R={r_b} free={free_bits}b N={n_cases_tier} tl={sat_tl}s ...",
                flush=True,
            )

            for idx in range(n_cases_tier):
                ht = cr.H_target[idx]
                fw, fbits = make_end_free_constraints(w0_15[idx], free_bits)

                # 1) Baseline Pure CaDiCaL
                rp = pure_cad.solve(
                    ht, rounds=r_b, fixed_words=fw, fixed_bits=fbits,
                    conflict_budget=confl_budget, time_limit_sec=sat_tl,
                )
                if rp["status"] == "SAT":
                    _record_verify(np.asarray(rp["w0_15"], dtype=np.uint32), ht, r_b, "PureCaDiCaL")
                    pc_s += 1
                pc_c.append(int(rp["conflicts"]))
                pc_t.append(float(rp["elapsed_sec"]))

                # 2) Baseline M3-V2 Dyn CDCL
                rb = base_dyn.solve(
                    ht, rounds=r_b, fixed_words=fw, fixed_bits=fbits,
                    conflict_budget=confl_budget, time_limit_sec=sat_tl,
                )
                if rb.status == "SAT":
                    _record_verify(rb.w0_15, ht, r_b, "BaselineM3V2DynCDCL")
                    b3_s += 1
                b3_c.append(int(rb.conflicts))
                b3_t.append(float(rb.elapsed_sec))

                # 3) Baseline Pure CMS-XOR (经 SafeCMSSubprocessRunner 防崩包装)
                if cms_runner is not None:
                    t_xor0 = time.perf_counter()
                    enc_xor = SHA256XORCNFEncoder(rounds=r_b).build(
                        H_target=ht, fixed_words=fw, fixed_bits=fbits
                    )
                    rx = cms_runner.solve_clauses(
                        clauses=enc_xor.clauses,
                        xor_clauses=enc_xor.xor_clauses,
                        time_limit_sec=sat_tl,
                    )
                    dt_xor = time.perf_counter() - t_xor0
                    if rx["crashed_or_timeout"]:
                        px_crashes += 1
                    elif rx["status"] == "SAT" and rx["solution"] is not None:
                        w_xor = extract_w0_15_from_cryptosat(enc_xor, rx["solution"])
                        _record_verify(w_xor, ht, r_b, "BaselinePureCMSXOR")
                        px_s += 1
                    px_t.append(float(dt_xor))

                # 4) Prototype P1+P2 CleanDynCDCL
                rp1 = p1_p2_solver.solve(
                    ht, rounds=r_b, fixed_words=fw, fixed_bits=fbits,
                    conflict_budget=confl_budget, time_limit_sec=sat_tl,
                )
                if rp1.status == "SAT":
                    _record_verify(rp1.w0_15, ht, r_b, "PrototypeP1P2")
                    p1_s += 1
                p1_c.append(int(rp1.conflicts))
                p1_t.append(float(rp1.elapsed_sec))
                p1_flp.append(int(rp1.flp_units_proven))
                p1_bin.append(int(p1_p2_solver.last_diagnostics.get("probed_binary_clauses", 0)))
                p1_gate_pass.append(int(p1_p2_solver.last_diagnostics.get("gate_passed_total", 0)))

                # 5) Prototype P3 CarryLockHybridRouter
                rp3 = p3_router.solve(
                    ht, rounds=r_b, fixed_words=fw, fixed_bits=fbits,
                    conflict_budget=confl_budget, time_limit_sec=sat_tl,
                )
                if rp3.status == "SAT":
                    _record_verify(rp3.w0_15, ht, r_b, "PrototypeP3Router")
                    p3_s += 1
                p3_c.append(int(rp3.conflicts))
                p3_t.append(float(rp3.elapsed_sec))
                d3 = p3_router.last_diagnostics
                zn = str(d3.get("zone", "unknown"))
                p3_zones[zn] = p3_zones.get(zn, 0) + 1
                p3_rlock.append(float(d3.get("carry_lock_rate", 0.0)))
                p3_dof.append(int(d3.get("gf2_residual_dof", 0)))
                p3_equiv.append(int(d3.get("n_equiv_2lit", 0)) // 2)

                if (idx + 1) % 4 == 0 or (idx + 1) == n_cases_tier:
                    print(
                        f"  [R={r_b} {free_bits}b case {idx+1}/{n_cases_tier}] "
                        f"Pure={pc_c[-1]}c({rp['status'][:3]}) | "
                        f"BaseDyn={b3_c[-1]}c({rb.status[:3]}) | "
                        f"P1+P2={p1_c[-1]}c({rp1.status[:3]}) | "
                        f"P3={zn}:{p3_t[-1]:.2f}s({rp3.status[:3]})",
                        flush=True,
                    )

            paired_stats = {
                "base_dyn_vs_pure_conflicts": compute_paired_comparison(pc_c, b3_c, tie_eps=0.5, seed=args.seed + r_b + 10),
                "p1p2_vs_pure_conflicts": compute_paired_comparison(pc_c, p1_c, tie_eps=0.5, seed=args.seed + r_b + 20),
                "p1p2_vs_base_dyn_conflicts": compute_paired_comparison(b3_c, p1_c, tie_eps=0.5, seed=args.seed + r_b + 30),
                "p1p2_vs_pure_sec": compute_paired_comparison(pc_t, p1_t, tie_eps=1e-4, seed=args.seed + r_b + 40),
                "p1p2_vs_base_dyn_sec": compute_paired_comparison(b3_t, p1_t, tie_eps=1e-4, seed=args.seed + r_b + 50),
                "p3_vs_pure_sec": compute_paired_comparison(pc_t, p3_t, tie_eps=1e-4, seed=args.seed + r_b + 60),
                "p3_vs_base_dyn_sec": compute_paired_comparison(b3_t, p3_t, tie_eps=1e-4, seed=args.seed + r_b + 70),
            }
            if cms_runner is not None:
                paired_stats["p3_vs_pure_cms_xor_sec"] = compute_paired_comparison(
                    px_t, p3_t, tie_eps=1e-4, seed=args.seed + r_b + 80
                )

            tier_row: Dict[str, Any] = {
                "rounds": int(r_b),
                "free_bits": int(free_bits),
                "cases": int(n_cases_tier),
                "tier_elapsed_sec": round(time.perf_counter() - t_tier0, 2),
                "pure_cadical": _summarize_solver_tier(pc_s, n_cases_tier, pc_c, pc_t),
                "baseline_m3v2_dyn_cdcl": _summarize_solver_tier(b3_s, n_cases_tier, b3_c, b3_t),
                "baseline_pure_cms_xor": (
                    {
                        "sat": px_s,
                        "cases": n_cases_tier,
                        "sat_rate": round(px_s / max(1, n_cases_tier), 4),
                        "mean_sec": round(float(np.mean(px_t)), 4),
                        "median_sec": round(float(np.median(px_t)), 4),
                        "subprocess_crashes_or_timeouts": px_crashes,
                    }
                    if cms_runner is not None
                    else None
                ),
                "prototype_p1_p2_clean_cdcl": _summarize_solver_tier(
                    p1_s, n_cases_tier, p1_c, p1_t,
                    extra={
                        "mean_flp_units": round(float(np.mean(p1_flp)), 2),
                        "mean_probed_2lit": round(float(np.mean(p1_bin)), 2),
                        "mean_gate_passed": round(float(np.mean(p1_gate_pass)), 3),
                    },
                ),
                "prototype_p3_hybrid_router": _summarize_solver_tier(
                    p3_s, n_cases_tier, p3_c, p3_t,
                    extra={
                        "zone_counts": p3_zones,
                        "mean_carry_lock_rate": round(float(np.mean(p3_rlock)), 4),
                        "mean_gf2_residual_dof": round(float(np.mean(p3_dof)), 1),
                        "mean_equiv_2lit_classes": round(float(np.mean(p3_equiv)), 1),
                    },
                ),
                "paired_stats": paired_stats,
            }
            report["domain_b_tiers"].append(tier_row)
            report["elapsed_sec"] = round(time.perf_counter() - t_all0, 2)
            _atomic_write_json(out_path, report)

            st_p1_pure = paired_stats["p1p2_vs_pure_conflicts"]
            st_p1_base = paired_stats["p1p2_vs_base_dyn_conflicts"]
            st_p3_xor = paired_stats.get("p3_vs_pure_cms_xor_sec", {})
            print(
                f"  => [Tier {tier_idx} Summary] Pure={pc_s}/{n_cases_tier}({np.mean(pc_c):.1f}c) | "
                f"BaseDyn={b3_s}/{n_cases_tier}({np.mean(b3_c):.1f}c) | "
                f"P1+P2={p1_s}/{n_cases_tier}({np.mean(p1_c):.1f}c, Δ_Pure={st_p1_pure['rel_reduction_pct']:+.1f}% p={st_p1_pure['wilcoxon_p']:.4f}, "
                f"Δ_Base={st_p1_base['rel_reduction_pct']:+.1f}% p={st_p1_base['wilcoxon_p']:.4f}) | "
                f"CMS-XOR={px_s}/{n_cases_tier}({np.mean(px_t):.2f}s) | "
                f"P3={p3_s}/{n_cases_tier}({np.mean(p3_t):.2f}s, zones={p3_zones}, Δt_XOR={st_p3_xor.get('rel_reduction_pct', 0):+.1f}%)",
                flush=True,
            )

        # =====================================================================
        # Part 2A: R=64 轨 A MCTS 配对实测 (严格沿用主 rng 流)
        # =====================================================================
        n_r64 = 1 if args.fast else 4
        w64 = _sample_random_words(rng, n_r64, 16)
        cr64 = core.compress_exact(w64, rounds=64)

        mcts_base = WormholeMCTS(
            model=model_m3, torch_eng=torch_eng, core=core, device=device,
            max_branch_candidates=16, use_gf2_root=False, gf2_interval=4, enable_macro_actions=True,
        )
        mcts_p4 = LocalWindowWormholeMCTS(
            model=model_m3, torch_eng=torch_eng, core=core, device=device,
            max_branch_candidates=8, half_win=4, use_local_gf2=True, skip_leaf_neural_eval=True,
        )

        r64_base_logs: List[Dict[str, Any]] = []
        r64_p4_logs: List[Dict[str, Any]] = []
        print(f"\n[R=64 Track-A MCTS] Running N={n_r64}, time_limit=3.0s ...", flush=True)
        for i in range(n_r64):
            rb64 = mcts_base.search(
                H_target=cr64.H_target[i], rounds=64, fixed_words=None,
                num_simulations=16 if args.fast else 64,
                time_limit_sec=1.5 if args.fast else 3.0,
                focus_splice=True,
            )
            if rb64.solved:
                _record_verify(rb64.w0_15, cr64.H_target[i], 64, "BaselineWormholeMCTS")
            r64_base_logs.append({
                "case": i,
                "solved": rb64.solved,
                "simulations": rb64.simulations,
                "expansions": rb64.expansions,
                "unk_drop": rb64.root_unknown - rb64.min_unknown_reached,
                "gate_drop": rb64.root_active_carry_gates - rb64.min_active_carry_gates,
                "splice_drop": rb64.root_splice_freedom - rb64.min_splice_freedom,
                "pruned_conflicts": rb64.pruned_conflicts,
                "elapsed_sec": round(rb64.elapsed_sec, 3),
            })

            rp4 = mcts_p4.search(
                H_target=cr64.H_target[i], rounds=64, fixed_words=None,
                num_simulations=600,
                time_limit_sec=1.5 if args.fast else 3.0,
                focus_splice=True,
            )
            if rp4.solved:
                _record_verify(rp4.w0_15, cr64.H_target[i], 64, "PrototypeP4MCTS")
            d4 = mcts_p4.last_search_diagnostics
            r64_p4_logs.append({
                "case": i,
                "solved": rp4.solved,
                "simulations": rp4.simulations,
                "expansions": rp4.expansions,
                "max_depth": d4.get("max_depth_reached", 0),
                "unk_drop": rp4.root_unknown - rp4.min_unknown_reached,
                "gate_drop": rp4.root_active_carry_gates - rp4.min_active_carry_gates,
                "splice_drop": rp4.root_splice_freedom - rp4.min_splice_freedom,
                "pruned_conflicts": rp4.pruned_conflicts,
                "local_gf2_units": d4.get("local_gf2_units", 0),
                "local_gf2_calls": d4.get("local_gf2_calls", 0),
                "local_gf2_mean_ms": d4.get("local_gf2_mean_ms", 0.0),
                "neural_eval_calls": d4.get("neural_eval_calls", 0),
                "elapsed_sec": round(rp4.elapsed_sec, 3),
            })
            print(
                f"  [R=64 case {i}] Base(sims={rb64.simulations}, drop={r64_base_logs[-1]['unk_drop']}/{r64_base_logs[-1]['gate_drop']}/{r64_base_logs[-1]['splice_drop']}) | "
                f"P4(sims={rp4.simulations}, depth={d4.get('max_depth_reached')}, drop={r64_p4_logs[-1]['unk_drop']}/{r64_p4_logs[-1]['gate_drop']}/{r64_p4_logs[-1]['splice_drop']}, "
                f"gf2_units={d4.get('local_gf2_units')}, gf2_ms={d4.get('local_gf2_mean_ms')})",
                flush=True,
            )

        report["r64_track_a_mcts"] = {
            "n_cases": n_r64,
            "baseline_wormhole_mcts": r64_base_logs,
            "prototype_p4_local_window_mcts": r64_p4_logs,
            "p4_unique_unk_drop_count": len({x["unk_drop"] for x in r64_p4_logs}),
            "p4_unique_gate_drop_count": len({x["gate_drop"] for x in r64_p4_logs}),
        }
        _atomic_write_json(out_path, report)

        # =====================================================================
        # Part 2B: R=64 free_bits in {4, 6} 严格零泄漏过约束可解验证探针
        # =====================================================================
        probe_rng = np.random.default_rng(int(args.seed) + 1001)
        n_probe = 1 if args.fast else 2
        probe_rows: List[Dict[str, Any]] = []
        print(f"\n[R=64 Overconstrained Probe] fb in (4, 6), N={n_probe}/tier ...", flush=True)
        for fb in (4, 6):
            wp = _sample_random_words(probe_rng, n_probe, 16)
            crp = core.compress_exact(wp, rounds=64)
            for idx_p in range(n_probe):
                fw_p, fbits_p = make_end_free_constraints(wp[idx_p], fb)
                rb_p = mcts_base.search(
                    H_target=crp.H_target[idx_p], rounds=64, fixed_words=fw_p, fixed_bits=fbits_p,
                    num_simulations=256, time_limit_sec=3.0, focus_splice=False,
                )
                if rb_p.solved:
                    _record_verify(rb_p.w0_15, crp.H_target[idx_p], 64, f"BaseMCTS_Probe_fb{fb}")

                rp4_p = mcts_p4.search(
                    H_target=crp.H_target[idx_p], rounds=64, fixed_words=fw_p, fixed_bits=fbits_p,
                    num_simulations=256, time_limit_sec=3.0, focus_splice=False,
                )
                if rp4_p.solved:
                    _record_verify(rp4_p.w0_15, crp.H_target[idx_p], 64, f"P4MCTS_Probe_fb{fb}")

                probe_rows.append({
                    "free_bits": fb,
                    "case": idx_p,
                    "baseline": {
                        "solved": rb_p.solved,
                        "simulations": rb_p.simulations,
                        "pruned_conflicts": rb_p.pruned_conflicts,
                        "elapsed_sec": round(rb_p.elapsed_sec, 4),
                    },
                    "prototype_p4": {
                        "solved": rp4_p.solved,
                        "simulations": rp4_p.simulations,
                        "pruned_conflicts": rp4_p.pruned_conflicts,
                        "elapsed_sec": round(rp4_p.elapsed_sec, 4),
                    },
                })
                print(
                    f"  [Probe R=64 fb={fb} case {idx_p}] Base(solved={rb_p.solved}, sims={rb_p.simulations}, pruned={rb_p.pruned_conflicts}) | "
                    f"P4(solved={rp4_p.solved}, sims={rp4_p.simulations}, pruned={rp4_p.pruned_conflicts}, sec={rp4_p.elapsed_sec:.3f}s)",
                    flush=True,
                )
        report["r64_overconstrained_probe"] = {
            "n_cases_per_fb": n_probe,
            "rows": probe_rows,
            "p4_sat_rate": round(float(np.mean([1.0 if r["prototype_p4"]["solved"] else 0.0 for r in probe_rows])), 4),
            "base_sat_rate": round(float(np.mean([1.0 if r["baseline"]["solved"] else 0.0 for r in probe_rows])), 4),
        }
        _atomic_write_json(out_path, report)

        # =====================================================================
        # Part 3: 域 A 物理隔离复核 (run_m3_extreme_healing, N=8, 64步, K=32, seed+7)
        # =====================================================================
        print("\n[Domain A Isolation Check] Running run_m3_extreme_healing ...", flush=True)
        heal_extreme = run_m3_extreme_healing(
            model=model_m3,
            device=device,
            rounds=64,
            n_crystals=2 if args.fast else 8,
            n_steps=8 if args.fast else 64,
            k_budget=8 if args.fast else 32,
            seed=int(args.seed) + 7,
        )
        print(
            f"  [Domain A] E0={heal_extreme['initial_energy_mean']:.3f} | "
            f"WalkSAT={heal_extreme['walksat_1bit']['final_energy_mean']:.3f} (-{heal_extreme['walksat_1bit']['reduction_pct']:.2f}%) | "
            f"Wolff={heal_extreme['neural_wolff']['final_energy_mean']:.3f} (-{heal_extreme['neural_wolff']['reduction_pct']:.2f}%, "
            f"healed={heal_extreme['neural_wolff']['exact_healed_rate']*100:.1f}%, tunnel={heal_extreme['neural_wolff']['tunneling_ratio']*100:.2f}%)",
            flush=True,
        )
        report["domain_a_isolation_check"] = heal_extreme

        report["verify_preimage_summary"] = {
            "total_sat_checked": verify_total,
            "passed": verify_passed,
            "pass_rate": round(float(verify_passed / max(1, verify_total)), 6),
        }
        report["elapsed_sec"] = round(time.perf_counter() - t_all0, 2)
        _atomic_write_json(out_path, report)
        print(
            f"\n[ProtoBench] 全部实测完成！耗时 {report['elapsed_sec']:.1f}s | "
            f"verify_preimage={verify_passed}/{verify_total} (100%) | 报告已写至 {out_path}",
            flush=True,
        )
    finally:
        if cms_runner is not None:
            cms_runner.close()

    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-SHA V10 M3 Prototype Paired Benchmark")
    p.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT)
    p.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    p.add_argument("--sat_time_limit", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=20260499)
    p.add_argument("--conflict_budget", type=int, default=250000)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    run_benchmark(build_argparser().parse_args())
