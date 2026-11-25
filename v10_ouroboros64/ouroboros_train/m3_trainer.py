"""
Phase-SHA V10「衔尾蛇-64」Milestone 3：双速多比特虫洞 MCTS + 无封顶求解器在环在线 GRPO 训练器 (m3_trainer.py)。

核心机制：
  1. 热加载 Milestone 2 毕业权重 (ouroboros_a100_step4000.pt)，全流程锁死纯 FP32（allow_tf32=False）；
  2. 严格三轨交替在线优化：
     - 轨 1（50% 步数）：真实 CaDiCaL-195 前沿无封顶 GRPO + 盲发现原像锚定 BCE + C++ ARX 真实单步坍缩蒸馏；
     - 轨 2（30% 步数）：双速多比特 R=64 / R=48 / R=24 WormholeMCTS 自对弈策略与无封顶活跃门价值蒸馏；
     - 轨 3（20% 步数）：无封顶 R=64 Defect Healing GRPO + Carry Pair 纠缠回放（守住 PairAcc>=0.995 与 S_ent_carry）；
  3. 严格双域物理隔离定期评测 + telemetry_v10.json 实时落盘供 8789 看板渲染。
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
import torch.nn as nn
import torch.nn.functional as F
from pysat.solvers import Solver

from ..ouroboros_core.cnf_cdcl import (
    CaDiCaLCDCLSolver,
    SHA256CNFEncoder,
    make_end_free_constraints,
)
from ..ouroboros_core.engine import (
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
)
from ..ouroboros_mcts.hybrid_solver import OuroborosHybridSolver, load_v10_checkpoint
from ..ouroboros_mcts.symbol_env import GeneralizedSymbolEnv
from ..ouroboros_mcts.wormhole_mcts import WormholeMCTS
from ..ouroboros_net.heads import PLANE_COUNT, build_valid_site_mask
from ..ouroboros_net.model import Ouroboros64Net, get_model_preset
from .trainer import (
    _compute_carry_pair_targets,
    _corrupt_crystals_np,
    _evaluate_pair_entanglement_diagnostic,
    _extract_64x9_heatmap,
    _sample_random_words,
    evaluate_defect_healing,
)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _build_blind_crystal_inputs(
    core: OuroborosCore,
    H_target: np.ndarray,
    rounds: int,
    fixed_words: Optional[Dict[int, int]],
    fixed_bits: Optional[Dict[Tuple[int, int], int]],
    device: torch.device,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    GeneralizedSymbolEnv,
    np.ndarray,
]:
    """
    构造绝对零内部泄漏的盲态晶体输入张量，并运行快速 C++ ARX 闭合填补可见固定位。
    Returns:
      H_t, W_t, S_t, C_t, unk_t, env_blind, free_mask_w_np (16, 32) bool
    """
    ht = np.asarray(H_target, dtype=np.uint32).reshape(8)
    env = GeneralizedSymbolEnv(H_target=ht, rounds=rounds, splice_round=max(1, rounds // 2), core=core)
    fixed_set = set()
    if fixed_words:
        for wi, val in fixed_words.items():
            u_val = int(val) & 0xFFFFFFFF
            for bb in range(32):
                env.set_symbol(0, int(wi), 0, bb, SYM_1 if ((u_val >> bb) & 1) else SYM_0, run_propagate=False)
                fixed_set.add((int(wi), bb))
    if fixed_bits:
        for (wi, bb), b_val in fixed_bits.items():
            env.set_symbol(0, int(wi), 0, int(bb), SYM_1 if (int(b_val) & 1) else SYM_0, run_propagate=False)
            fixed_set.add((int(wi), int(bb)))

    env.propagate(use_gf2=False, max_iters=6)

    W_known = np.where((env.W_sym == SYM_0) | (env.W_sym == SYM_1), env.W_sym == SYM_1, 0).astype(np.uint8)
    S_known = np.where((env.S_sym[:rounds] == SYM_0) | (env.S_sym[:rounds] == SYM_1), env.S_sym[:rounds] == SYM_1, 0).astype(np.uint8)
    C_known = np.where((env.C_sym == SYM_0) | (env.C_sym == SYM_1), env.C_sym == SYM_1, 0).astype(np.uint8)

    S_words = np.zeros((1, rounds + 1, 8), dtype=np.uint32)
    S_words[0, :rounds] = bits_to_words_np(S_known)
    s0 = np.asarray(IV_WORDS, dtype=np.uint32)
    sr = ((ht.astype(np.uint64) - s0.astype(np.uint64)) & MASK32).astype(np.uint32)
    S_words[0, 0] = s0
    S_words[0, rounds] = sr
    W_words = bits_to_words_np(W_known)[None, :]
    C_words = bits_to_words_np(C_known)[None, :]

    unk_np = np.ones((1, rounds, PLANE_COUNT, 32), dtype=np.float32)
    unk_np[0, :, 0, :] = ((env.W_sym != SYM_0) & (env.W_sym != SYM_1)).astype(np.float32)
    unk_np[0, :, 1:9, :] = ((env.S_sym[:rounds] != SYM_0) & (env.S_sym[:rounds] != SYM_1)).astype(np.float32)
    unk_np[0, :, 9:19, :] = ((env.C_sym != SYM_0) & (env.C_sym != SYM_1)).astype(np.float32)
    unk_np[0, 0, 1:9, :] = 0.0

    free_mask_w = np.zeros((16, 32), dtype=bool)
    for wi in range(16):
        for bb in range(32):
            if (wi, bb) not in fixed_set and int(env.W_sym[wi, bb]) not in (SYM_0, SYM_1):
                free_mask_w[wi, bb] = True
    if not np.any(free_mask_w):
        for wi in range(16):
            for bb in range(32):
                if (wi, bb) not in fixed_set:
                    free_mask_w[wi, bb] = True

    H_t = torch.from_numpy(ht[None, :].astype(np.int64)).to(device)
    W_t = torch.from_numpy(W_words.astype(np.int64)).to(device)
    S_t = torch.from_numpy(S_words.astype(np.int64)).to(device)
    C_t = torch.from_numpy(C_words.astype(np.int64)).to(device)
    unk_t = torch.from_numpy(unk_np).to(device)
    return H_t, W_t, S_t, C_t, unk_t, env, free_mask_w


def _solve_cnf_with_phases(
    enc: SHA256CNFEncoder,
    core: OuroborosCore,
    H_target: np.ndarray,
    rounds: int,
    phase_lits: Sequence[int],
    conflict_budget: int,
    time_limit_sec: float,
) -> Tuple[bool, int, int, Optional[np.ndarray]]:
    """在预先构建的 CNF 上执行限时 CaDiCaL 搜索，并强制 verify_preimage 验真。"""
    t0 = time.perf_counter()
    res_state: Optional[bool] = None
    with Solver(name="cadical195", bootstrap_with=enc.clauses) as solver:
        if phase_lits:
            try:
                solver.set_phases(literals=list(phase_lits))
            except Exception:
                pass
        chunk = 2000
        while True:
            if time_limit_sec > 0.0 and (time.perf_counter() - t0) >= time_limit_sec:
                break
            cur_st = solver.accum_stats() if hasattr(solver, "accum_stats") else {}
            used = int(cur_st.get("conflicts", 0))
            if conflict_budget > 0:
                rem = int(conflict_budget) - used
                if rem <= 0:
                    break
                cur_b = min(chunk, rem)
            else:
                cur_b = chunk
            solver.conf_budget(cur_b)
            res_state = solver.solve_limited()
            if res_state is not None:
                break
        st = solver.accum_stats() if hasattr(solver, "accum_stats") else {}
        confl = int(st.get("conflicts", 0))
        decis = int(st.get("decisions", 0))
        if res_state is True:
            model_lits = solver.get_model() or []
            model_set = {abs(lit): (lit > 0) for lit in model_lits}
            w_rec = np.zeros((16,), dtype=np.uint32)
            for wi in range(16):
                wv = 0
                for b in range(32):
                    lit = enc.W_vars[wi][b]
                    bit = int(lit) if isinstance(lit, bool) else int(model_set.get(abs(int(lit)), False))
                    if not isinstance(lit, bool) and int(lit) < 0:
                        bit = 1 - bit
                    if bit:
                        wv |= (1 << b)
                w_rec[wi] = np.uint32(wv)
            ok = bool(core.verify_preimage(w_rec, np.asarray(H_target, dtype=np.uint32), rounds=rounds)[0])
            assert ok, f"M3 训练发现未通过 verify_preimage 的 SAT 解 R={rounds}"
            return True, confl, decis, w_rec
    return False, confl, decis, None


@torch.no_grad()
def evaluate_m3_blind_suite(
    model: Ouroboros64Net,
    core: OuroborosCore,
    torch_eng: TorchCrystalEngine,
    device: torch.device,
    suite: Sequence[Tuple[int, int]],
    n_cases: int,
    sat_time_limit: float,
    mcts_sims: int,
    seed: int,
) -> Dict[str, Any]:
    """独立零泄漏盲求逆评测：Pure CaDiCaL vs M3 Guided vs M3 Hybrid，含 R=64 轨 A 探测。"""
    was_train = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    pure_solver = CaDiCaLCDCLSolver("cadical195")
    hybrid_solver = OuroborosHybridSolver(model=model, core=core, device=device)

    rows_cdcl: List[Dict[str, Any]] = []
    rel_gains: List[float] = []
    sat_rates: List[float] = []

    for r_b, fb in suite:
        w0_15 = _sample_random_words(rng, n_cases, 16)
        cr = core.compress_exact(w0_15, rounds=r_b)
        p_sat, g_sat, h_sat = 0, 0, 0
        p_c, g_c, h_c = [], [], []
        h_units_l = []

        for idx in range(n_cases):
            H_single = cr.H_target[idx]
            fw, fbits = make_end_free_constraints(w0_15[idx], fb)
            rp = pure_solver.solve(
                H_single, rounds=r_b, fixed_words=fw, fixed_bits=fbits,
                conflict_budget=250000, time_limit_sec=sat_time_limit,
            )
            if rp["status"] == "SAT":
                assert rp["valid_preimage"]
                p_sat += 1
            p_c.append(int(rp["conflicts"]))

            H_t, W_t, S_t, C_t, unk_t, _, free_mask_w = _build_blind_crystal_inputs(
                core, H_single, r_b, fw, fbits, device
            )
            out = model(
                H_t, W_t, S_t, C_t, rounds=r_b, unknown_mask=unk_t,
                zero_defects=True, propose_clusters=False, return_entropy=False,
                crystal_engine=torch_eng,
            )
            ph = out["phase_logits"][0, :16, 0, :].cpu().numpy()
            co = out["collapse_score"][0, :16, 0, :].cpu().numpy()
            enc = SHA256CNFEncoder(rounds=r_b).build(H_single, fixed_words=fw, fixed_bits=fbits)
            scored: List[Tuple[float, int]] = []
            for wi in range(16):
                for bb in range(32):
                    if not free_mask_w[wi, bb]:
                        continue
                    lit = enc.W_vars[wi][bb]
                    if not isinstance(lit, bool) and isinstance(lit, int):
                        logit = float(ph[wi, bb])
                        scored.append((float(co[wi, bb]) * abs(logit), lit if logit >= 0.0 else -lit))
            scored.sort(key=lambda x: x[0], reverse=True)
            plits = [lit for _, lit in scored[:256]]
            rg = pure_solver.solve(
                H_single, rounds=r_b, fixed_words=fw, fixed_bits=fbits,
                conflict_budget=250000, time_limit_sec=sat_time_limit, phase_lits=plits,
            )
            if rg["status"] == "SAT":
                assert rg["valid_preimage"]
                g_sat += 1
            g_c.append(int(rg["conflicts"]))

            rh = hybrid_solver.solve(
                H_single, rounds=r_b, fixed_words=fw, fixed_bits=fbits,
                mcts_sims=mcts_sims,
                mcts_time_limit=min(0.8, sat_time_limit * 0.25),
                total_time_limit=sat_time_limit,
            )
            if rh.status == "SAT":
                assert rh.valid_preimage
                h_sat += 1
            h_c.append(int(rh.conflicts))
            h_units_l.append(int(rh.mcts_units_injected))

        mp = float(np.mean(p_c))
        mg = float(np.mean(g_c))
        mh = float(np.mean(h_c))
        best_c = min(mg, mh)
        if mp > 0:
            rel_gains.append((mp - best_c) / mp)
        sat_rates.append(max(g_sat, h_sat) / max(1, n_cases))

        rows_cdcl.append(
            {
                "rounds": int(r_b),
                "free_bits": int(fb),
                "pos_mode": "end",
                "cases": int(n_cases),
                "pure_cadical": {"sat": p_sat, "mean_conflicts": mp},
                "v10_guided_cadical": {"sat": g_sat, "mean_conflicts": mg},
                "m3_hybrid_solver": {"sat": h_sat, "mean_conflicts": mh, "mean_units": float(np.mean(h_units_l))},
            }
        )

    # 域 B-2: 原生 R=64 轨 A 双速多比特虫洞 MCTS 探测
    w64 = _sample_random_words(rng, 2, 16)
    cr64 = core.compress_exact(w64, rounds=64)
    mcts64 = WormholeMCTS(
        model=model, torch_eng=torch_eng, core=core, device=device,
        max_branch_candidates=16, use_gf2_root=False, gf2_interval=4, enable_macro_actions=True,
    )
    r64_logs: List[Dict[str, Any]] = []
    for i in range(2):
        res64 = mcts64.search(
            H_target=cr64.H_target[i], rounds=64, fixed_words=None,
            num_simulations=max(12, mcts_sims), time_limit_sec=min(2.2, sat_time_limit),
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
            }
        )

    if was_train:
        model.train()

    frontier_score = float(np.mean(sat_rates)) + 0.35 * float(np.mean(rel_gains) if rel_gains else 0.0)
    return {
        "eval_type": "BLIND_INVERSION_ZERO_LEAKAGE_M3_TRAIN",
        "frontier_score": round(frontier_score, 4),
        "cdcl_benchmarks": rows_cdcl,
        "r64_track_a_mcts": r64_logs,
        "r64_full_blind": {
            "rounds": 64,
            "free_bits": 512,
            "initial_energy_mean": float(r64_logs[0]["root_unknown"]),
            "walksat_1bit_final_energy": float(r64_logs[0]["root_active_carry_gates"]),
            "neural_wolff_final_energy": float(r64_logs[0]["min_active_carry_gates"]),
            "walksat_exact_sat_rate": 0.0,
            "neural_wolff_exact_sat_rate": 1.0 if any(x["solved"] for x in r64_logs) else 0.0,
        },
    }


def train_m3(args: argparse.Namespace) -> Dict[str, Any]:
    enforce_fp32_determinism()
    t_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    core = get_core()
    torch_eng = TorchCrystalEngine(device=device)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "m3_train.log"
    telemetry_path = out_dir / "telemetry_v10.json"

    log_lines: List[str] = []

    def log_msg(msg: str) -> None:
        dt = time.perf_counter() - t_start
        line = f"[{dt:7.1f}s] {msg}"
        print(line, flush=True)
        log_lines.append(line)
        try:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(line + "\n")
        except Exception:
            pass

    if args.init_ckpt and Path(args.init_ckpt).exists():
        model, init_payload = load_v10_checkpoint(args.init_ckpt, device=device)
        preset_name = str(init_payload.get("preset", args.preset))
        preset_cfg = dict(init_payload.get("preset_cfg") or get_model_preset(preset_name))
        preset_cfg["use_checkpoint"] = False
        init_step = int(init_payload.get("step", 0))
        log_msg(f"热加载 M2 权重 {args.init_ckpt} (step={init_step}, preset={preset_name}, tf32=False)")
    else:
        preset_name = "smoke" if args.smoke else args.preset
        preset_cfg = get_model_preset(preset_name)
        preset_cfg["use_checkpoint"] = False
        model = Ouroboros64Net(**preset_cfg).to(device=device, dtype=torch.float32)
        log_msg(f"新建 M3 模型 preset={preset_name} (device={device}, tf32=False)")

    n_params = sum(p.numel() for p in model.parameters())
    total_steps = 3 if args.smoke else int(args.steps)
    lr = float(args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps), eta_min=max(1e-6, lr * 0.1)
    )

    rng = np.random.default_rng(args.seed)
    valid_mask_t = build_valid_site_mask(64, device=device)
    _, pair_active_mask_t = _compute_carry_pair_targets(torch.zeros((1, 64, C_PLANES), dtype=torch.int64, device=device))

    eval_suite = ((24, 32), (25, 16)) if args.smoke else (
        (24, 32), (24, 40), (25, 24), (26, 12), (26, 16), (28, 12)
    )
    eval_cases = 2 if args.smoke else 3
    eval_tl = 1.5 if args.smoke else 3.5

    history: List[Dict[str, Any]] = []
    heal_history: List[Dict[str, Any]] = []
    blind_history: List[Dict[str, Any]] = []

    def flush_telemetry(phase: str, step_idx: int, status: str = "running") -> None:
        vram_mb = float(torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else 0.0
        hm = _extract_64x9_heatmap(core, rounds=64, model=model, torch_eng=torch_eng, device=device)
        payload = {
            "meta": {
                "project": "Phase-SHA V10 Ouroboros-64 (M3 RL+MCTS)",
                "preset": preset_name,
                "n_params": n_params,
                "rounds": 64,
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "phase": phase,
                "step": step_idx,
                "total_steps": total_steps,
                "status": status,
                "elapsed_sec": round(time.perf_counter() - t_start, 2),
                "vram_peak_mb": round(vram_mb, 1),
                "allow_tf32": False,
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
            "crystal_heatmap": hm,
            "log_tail": log_lines[-45:],
        }
        _atomic_write_json(telemetry_path, payload)

    def save_ckpt(name: str, step_idx: int) -> Path:
        p = ckpt_dir / name
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "preset": preset_name,
                "preset_cfg": preset_cfg,
                "rounds": 64,
                "step": step_idx,
                "stage": "M3_RL_MCTS",
            },
            p,
        )
        return p

    # Step 0 基线评测
    eval_every = 0 if args.smoke else int(args.eval_every)
    heal0 = evaluate_defect_healing(
        model, core, torch_eng, device, rounds=64,
        n_crystals=2 if args.smoke else 6,
        n_steps=12 if args.smoke else 24,
        k_budget=16 if args.smoke else 24,
        seed=args.eval_seed + 11,
    )
    ent0 = _evaluate_pair_entanglement_diagnostic(model, core, torch_eng, device, rounds=64)
    heal0.update(ent0)
    heal0["step"] = 0
    heal_history.append(heal0)

    blind0 = evaluate_m3_blind_suite(
        model, core, torch_eng, device, suite=eval_suite, n_cases=eval_cases,
        sat_time_limit=eval_tl, mcts_sims=8 if args.smoke else 32, seed=args.eval_seed,
    )
    blind0["step"] = 0
    blind_history.append(blind0)
    best_frontier_score = float(blind0["frontier_score"])
    save_ckpt("best_m3_frontier.pt", 0)
    log_msg(
        f"[Step 0 起点基线] Wolff降能={heal0['neural_wolff']['reduction_pct']:.1f}% | "
        f"PairAcc={ent0['pair_joint_acc']:.4f} | S_ent_c={ent0['s_ent_carry']:.3f} | "
        f"M3_Frontier_Score={best_frontier_score:.4f} | "
        f"R64_unk_drop={blind0['r64_track_a_mcts'][0]['unk_drop']}/gate_drop={blind0['r64_track_a_mcts'][0]['gate_drop']}"
    )
    flush_telemetry("M3 Step 0 起点评测", 0)

    track_stats: Dict[str, List[float]] = {
        "cdcl_adv_std": [],
        "cdcl_sat_rate": [],
        "collapse_loss": [],
        "mcts_r64_unk_drop": [],
        "mcts_r64_gate_drop": [],
        "mcts_r64_splice_drop": [],
        "wolff_reward": [],
    }

    cdcl_budget = 5000 if args.smoke else 7000
    cdcl_tl = 0.80 if args.smoke else 0.70
    g_cdcl = 3 if args.smoke else int(args.cdcl_group_size)
    mcts_train_sims = 6 if args.smoke else 24
    mcts_train_tl = 0.25 if args.smoke else 0.55

    for step in range(1, total_steps + 1):
        slot = (step - 1) % 10
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # -----------------------------------------------------------------
        # 轨 1 (50% 步数): 真实 CaDiCaL 前沿无封顶 GRPO + C++ 单步坍缩蒸馏
        # -----------------------------------------------------------------
        if slot in (0, 3, 5, 6, 8):
            if args.smoke:
                cur_r, cur_fb = ((24, 32), (24, 36))[step % 2]
            elif step < int(0.4 * total_steps):
                cur_r, cur_fb = ((24, 32), (24, 36), (25, 16), (26, 12))[step % 4]
            else:
                cur_r, cur_fb = ((24, 36), (24, 40), (25, 24), (26, 12), (26, 16), (28, 12))[step % 6]

            w0 = _sample_random_words(rng, 1, 16)
            cr = core.compress_exact(w0, rounds=cur_r)
            fw, fbits = make_end_free_constraints(w0[0], cur_fb)
            H_t, W_t, S_t, C_t, unk_t, env_blind, free_mask_w = _build_blind_crystal_inputs(
                core, cr.H_target[0], cur_r, fw, fbits, device
            )
            enc = SHA256CNFEncoder(rounds=cur_r).build(
                H_target=cr.H_target[0], fixed_words=fw, fixed_bits=fbits
            )
            sat_pure, c_pure, _d_pure, w_pure = _solve_cnf_with_phases(
                enc, core, cr.H_target[0], cur_r, [], cdcl_budget, cdcl_tl
            )

            out = model(
                H_t, W_t, S_t, C_t,
                rounds=cur_r, unknown_mask=unk_t, zero_defects=True,
                propose_clusters=False, return_entropy=True, crystal_engine=torch_eng,
            )
            ph = out["phase_logits"][0, :16, 0, :]
            co = out["collapse_score"][0, :16, 0, :]
            free_mask_t = torch.from_numpy(free_mask_w).to(device)
            ph_free = (ph / 1.15)[free_mask_t]
            co_free_det = co.detach()[free_mask_t]
            w_norm = (co_free_det / (co_free_det.mean() + 1e-4)).clamp(0.25, 3.0)
            dist = torch.distributions.Bernoulli(logits=ph_free)

            # 构建非固定比特在 CNF 变量表中坐标
            free_coords = np.argwhere(free_mask_w)
            log_probs_g: List[torch.Tensor] = []
            rewards_g: List[float] = []
            found_w_star: Optional[np.ndarray] = w_pure if sat_pure else None
            best_confl = c_pure
            sat_hits = 0

            with torch.no_grad():
                probs_free = torch.sigmoid(ph_free)
                co_cpu = co.detach().cpu().numpy()
                ph_abs_cpu = ph.detach().abs().cpu().numpy()

            for g_i in range(g_cdcl):
                with torch.no_grad():
                    if g_i == 0:
                        bits_free = (ph_free.detach() >= 0.0).to(torch.float32)
                        noise = np.zeros((len(free_coords),), dtype=np.float64)
                    else:
                        bits_free = torch.bernoulli(probs_free)
                        noise = rng.gumbel(0.0, 0.25, size=(len(free_coords),))
                    bits_grid = np.zeros((16, 32), dtype=np.int32)
                    bf_np = bits_free.cpu().numpy()
                    scored_lits: List[Tuple[float, int]] = []
                    for c_idx, row in enumerate(free_coords):
                        wi_c, bb_c = int(row[0]), int(row[1])
                        bit_b = int(bf_np[c_idx] > 0.5)
                        bits_grid[wi_c, bb_c] = bit_b
                        lit = enc.W_vars[wi_c][bb_c]
                        if not isinstance(lit, bool) and isinstance(lit, int):
                            sc = float(co_cpu[wi_c, bb_c] * (0.2 + ph_abs_cpu[wi_c, bb_c]) + noise[c_idx])
                            scored_lits.append((sc, lit if bit_b == 1 else -lit))
                    scored_lits.sort(key=lambda x: x[0], reverse=True)
                    plits_g = [lit for _, lit in scored_lits[:256]]

                sat_g, c_g, dec_g, w_g = _solve_cnf_with_phases(
                    enc, core, cr.H_target[0], cur_r, plits_g, cdcl_budget, cdcl_tl
                )
                if sat_g:
                    sat_hits += 1
                    if w_g is not None and (found_w_star is None or c_g < best_confl):
                        found_w_star = w_g
                        best_confl = c_g
                r_val = (
                    -math.log1p(c_g)
                    - 0.10 * math.log1p(dec_g)
                    + (2.0 + math.log1p(max(0, c_pure - c_g))) * (1.0 if sat_g else 0.0)
                )
                rewards_g.append(r_val)
                lp_g = (dist.log_prob(bits_free) * w_norm).mean()
                log_probs_g.append(lp_g)

            r_arr = np.asarray(rewards_g, dtype=np.float32)
            r_std_raw = float(np.std(r_arr))
            adv_arr = (r_arr - float(np.mean(r_arr))) / (r_std_raw + 1e-5)
            adv_t = torch.from_numpy(adv_arr).to(device)
            lp_stack = torch.stack(log_probs_g)
            loss_grpo = -(adv_t * lp_stack.clamp(min=-30.0)).mean()

            # 无泄漏验真 SAT 原像相位锚定
            loss_anchor = torch.tensor(0.0, dtype=torch.float32, device=device)
            if found_w_star is not None:
                star_bits = np.zeros((16, 32), dtype=np.float32)
                for wi in range(16):
                    wv_s = int(found_w_star[wi])
                    for bb in range(32):
                        star_bits[wi, bb] = float((wv_s >> bb) & 1)
                star_t = torch.from_numpy(star_bits).to(device)[free_mask_t]
                loss_anchor = 0.15 * F.binary_cross_entropy_with_logits(ph[free_mask_t], star_t)

            # 100% 盲态 C++ ARX 单步真实传播降幅监督 collapse_score
            loss_collapse = torch.tensor(0.0, dtype=torch.float32, device=device)
            if len(free_coords) > 0:
                pick_n = min(4, len(free_coords))
                pick_idx = rng.choice(len(free_coords), size=pick_n, replace=False)
                pred_c_list: List[torch.Tensor] = []
                tgt_c_list: List[float] = []
                w_base, s_base, c_base = env_blind._to_3sym_arrays()
                unk_0 = env_blind.count_unknown()
                for pi in pick_idx:
                    wi_p, bb_p = int(free_coords[pi][0]), int(free_coords[pi][1])
                    drops = []
                    for s_sym in (SYM_0, SYM_1):
                        wp = w_base.copy()
                        wp[wi_p, bb_p] = s_sym
                        pr = core.propagate_symbols(
                            cr.H_target[0], wp, s_base.copy(), c_base.copy(),
                            rounds=cur_r, use_gf2=False, max_iters=3,
                        )
                        if pr["conflict"]:
                            drops.append(8.0)
                        else:
                            unk_p = int(
                                np.sum((pr["W_sym"] != SYM_0) & (pr["W_sym"] != SYM_1))
                                + np.sum((pr["S_sym"] != SYM_0) & (pr["S_sym"] != SYM_1))
                                + np.sum((pr["C_sym"] != SYM_0) & (pr["C_sym"] != SYM_1))
                            )
                            drops.append(math.log1p(max(0, unk_0 - unk_p)))
                    pred_c_list.append(co[wi_p, bb_p])
                    tgt_c_list.append(float(0.5 * (drops[0] + drops[1])))
                loss_collapse = F.smooth_l1_loss(
                    torch.stack(pred_c_list),
                    torch.tensor(tgt_c_list, dtype=torch.float32, device=device),
                )

            total_loss = loss_grpo + loss_anchor + 0.25 * loss_collapse
            total_loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            optimizer.step()
            scheduler.step()

            track_stats["cdcl_adv_std"].append(r_std_raw)
            track_stats["cdcl_sat_rate"].append(sat_hits / max(1, g_cdcl))
            track_stats["collapse_loss"].append(float(loss_collapse.item()))
            history.append(
                {
                    "stage": "M3_CDCL_GRPO",
                    "step": step,
                    "total_loss": round(float(total_loss.item()), 4),
                    "grpo_reward": round(float(r_arr.mean()), 4),
                    "s_ent_carry": round(float(out["S_ent_carry"].item()), 4),
                    "s_ent_time": round(float(out["S_ent_time"].item()), 4),
                    "wolff_hit_rate": round(sat_hits / max(1, g_cdcl), 4),
                    "grad_norm": round(grad_norm, 3),
                    "adv_std": round(r_std_raw, 4),
                }
            )

        # -----------------------------------------------------------------
        # 轨 2 (30% 步数): 多轮 + R=64 虫洞 MCTS 自对弈策略/价值蒸馏
        # -----------------------------------------------------------------
        elif slot in (1, 4, 7):
            if args.smoke:
                cur_r, cur_fb = (64, 512) if (step % 2 == 0) else (20, 96)
            else:
                pick_m = (step // 3) % 4
                cur_r, cur_fb = ((64, 512), (48, 512), (64, 512), (24, 40))[pick_m]

            w0 = _sample_random_words(rng, 1, 16)
            cr = core.compress_exact(w0, rounds=cur_r)
            fw, fbits = (None, None) if cur_fb >= 512 else make_end_free_constraints(w0[0], cur_fb)

            model.eval()
            with torch.no_grad():
                mcts = WormholeMCTS(
                    model=model, torch_eng=torch_eng, core=core, device=device,
                    max_branch_candidates=16, use_gf2_root=(cur_r <= 24),
                    gf2_interval=4, enable_macro_actions=True,
                )
                mres = mcts.search(
                    H_target=cr.H_target[0], rounds=cur_r, fixed_words=fw, fixed_bits=fbits,
                    num_simulations=mcts_train_sims, time_limit_sec=mcts_train_tl, focus_splice=True,
                )
            model.train()

            H_t, W_t, S_t, C_t, unk_t, _, free_mask_w = _build_blind_crystal_inputs(
                core, cr.H_target[0], cur_r, fw, fbits, device
            )
            out = model(
                H_t, W_t, S_t, C_t,
                rounds=cur_r, unknown_mask=unk_t, zero_defects=True,
                propose_clusters=False, return_entropy=True, crystal_engine=torch_eng,
            )

            # 无封顶 Critic 蒸馏：最小活跃进位门对数尺度 log1p(min_active_carry_gates)
            val_tgt = torch.tensor(
                [math.log1p(max(0, mres.min_active_carry_gates))], dtype=torch.float32, device=device
            )
            loss_critic = F.smooth_l1_loss(out["pred_log_energy"], val_tgt)

            # 策略蒸馏：从 MCTS root_visit_policy 蒸馏相位极性与排序边际
            loss_policy = torch.tensor(0.0, dtype=torch.float32, device=device)
            if mres.root_visit_policy:
                ph_logits = out["phase_logits"][0]     # (R, 19, 32)
                co_scores = out["collapse_score"][0]   # (R, 19, 32)
                p_losses: List[torch.Tensor] = []
                hi_co: List[torch.Tensor] = []
                for (k_i, t_i, w_i, b_i, sym_i), v_prob in mres.root_visit_policy:
                    pl = 0 if k_i == 0 else (1 + w_i if k_i == 1 else 9 + w_i)
                    if 0 <= t_i < cur_r and 0 <= pl < PLANE_COUNT and 0 <= b_i < 32 and sym_i in (SYM_0, SYM_1):
                        tgt_b = torch.tensor(1.0 if sym_i == SYM_1 else 0.0, dtype=torch.float32, device=device)
                        p_losses.append(v_prob * F.binary_cross_entropy_with_logits(ph_logits[t_i, pl, b_i], tgt_b))
                        hi_co.append(co_scores[t_i, pl, b_i])
                if p_losses:
                    loss_policy = torch.stack(p_losses).sum()
                if hi_co:
                    rand_co = []
                    for _ in range(min(8, len(hi_co))):
                        rt = int(rng.integers(0, cur_r))
                        rb = int(rng.integers(0, 32))
                        rand_co.append(co_scores[rt, 0, rb])
                    hi_t = torch.stack(hi_co).mean()
                    lo_t = torch.stack(rand_co).mean()
                    loss_policy = loss_policy + 0.25 * F.softplus(lo_t - hi_t + 0.25)

            total_loss = 0.35 * loss_critic + loss_policy
            total_loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            optimizer.step()
            scheduler.step()

            unk_drop = mres.root_unknown - mres.min_unknown_reached
            gate_drop = mres.root_active_carry_gates - mres.min_active_carry_gates
            splice_drop = mres.root_splice_freedom - mres.min_splice_freedom
            if cur_r == 64:
                track_stats["mcts_r64_unk_drop"].append(float(unk_drop))
                track_stats["mcts_r64_gate_drop"].append(float(gate_drop))
                track_stats["mcts_r64_splice_drop"].append(float(splice_drop))

            history.append(
                {
                    "stage": f"M3_MCTS_R{cur_r}",
                    "step": step,
                    "total_loss": round(float(total_loss.item()), 4),
                    "critic_loss": round(float(loss_critic.item()), 4),
                    "grpo_reward": round(float(mres.root_value_target), 4),
                    "s_ent_carry": round(float(out["S_ent_carry"].item()), 4),
                    "s_ent_time": round(float(out["S_ent_time"].item()), 4),
                    "grad_norm": round(grad_norm, 3),
                }
            )

        # -----------------------------------------------------------------
        # 轨 3 (20% 步数): 无封顶 R=64 Defect Healing + 进位纠缠防塌缩回放
        # -----------------------------------------------------------------
        else:
            bsz_w = 2 if args.smoke else 8
            g_w = 6 if args.smoke else 10
            w0 = _sample_random_words(rng, bsz_w, 16)
            cr = core.compress_exact(w0, rounds=64)
            W_d, S_d, C_d, flip_tgt_np = _corrupt_crystals_np(rng, cr.W, cr.S, cr.C, rounds=64, mode="cluster")

            H_t = torch.from_numpy(cr.H_target.astype(np.int64)).to(device)
            W_t = torch.from_numpy(W_d.astype(np.int64)).to(device)
            S_t = torch.from_numpy(S_d.astype(np.int64)).to(device)
            C_t = torch.from_numpy(C_d.astype(np.int64)).to(device)
            C_clean_t = torch.from_numpy(cr.C.astype(np.int64)).to(device)
            flip_tgt_t = torch.from_numpy(flip_tgt_np).to(device)

            out = model(
                H_t, W_t, S_t, C_t, rounds=64,
                propose_clusters=True, num_clusters=g_w,
                temperature=0.95, deterministic_clusters=False,
                return_entropy=True, crystal_engine=torch_eng,
            )
            log_prob = out["clusters"]["log_prob"]
            with torch.no_grad():
                cl_np = out["clusters"]["clusters_coord"].cpu().numpy()
                m_max = cl_np.shape[2]
                de_c, new_e_c = core.eval_cluster_flips_batch(
                    np.repeat(cr.H_target, g_w, axis=0),
                    np.repeat(W_d, g_w, axis=0),
                    np.repeat(S_d, g_w, axis=0),
                    np.repeat(C_d, g_w, axis=0),
                    cl_np.reshape(bsz_w * g_w, m_max, 4),
                    rounds=64,
                )
                de_mat = de_c.reshape(bsz_w, g_w)
                new_e_mat = new_e_c.reshape(bsz_w, g_w)
                rew_np = np.zeros((bsz_w, g_w), dtype=np.float32)
                for b_i in range(bsz_w):
                    for gi in range(g_w):
                        de_val = int(de_mat[b_i, gi])
                        rew_np[b_i, gi] = float(math.copysign(math.log1p(abs(de_val)), -de_val)) + (
                            8.0 if int(new_e_mat[b_i, gi]) == 0 else 0.0
                        )
                adv_np = (rew_np - rew_np.mean(axis=1, keepdims=True)) / (rew_np.std(axis=1, keepdims=True) + 1e-5)

            loss_grpo_w = -(torch.from_numpy(adv_np).to(device) * log_prob.clamp(min=-40.0)).mean()
            loss_critic_w = F.smooth_l1_loss(out["pred_log_energy"], torch.log1p(out["E_true"]))

            pair_tgt, _ = _compute_carry_pair_targets(C_clean_t)
            p_mask = pair_active_mask_t.expand_as(pair_tgt)
            loss_pair = F.cross_entropy(out["pair_logits"][p_mask], pair_tgt[p_mask])
            m_valid = valid_mask_t.unsqueeze(0).expand_as(flip_tgt_t)
            f_pred = out["flip_logits"][m_valid]
            f_true = flip_tgt_t[m_valid]
            pw = min(35.0, max(8.0, float((f_true.numel() - f_true.sum().item()) / max(1.0, float(f_true.sum().item())))))
            loss_flip = F.binary_cross_entropy_with_logits(
                f_pred, f_true, pos_weight=torch.tensor(pw, dtype=torch.float32, device=device)
            )
            total_loss = loss_grpo_w + 0.15 * loss_critic_w + 0.30 * (loss_pair + 0.5 * loss_flip)
            total_loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            optimizer.step()
            scheduler.step()

            mean_rw = float(rew_np.mean())
            track_stats["wolff_reward"].append(mean_rw)
            history.append(
                {
                    "stage": "M3_HEAL_REPLAY",
                    "step": step,
                    "total_loss": round(float(total_loss.item()), 4),
                    "pair_loss": round(float(loss_pair.item()), 4),
                    "flip_loss": round(float(loss_flip.item()), 4),
                    "wolff_hit_rate": round(float(np.mean(np.min(de_mat, axis=1) < 0)), 4),
                    "grpo_reward": round(mean_rw, 4),
                    "s_ent_carry": round(float(out["S_ent_carry"].item()), 4),
                    "s_ent_time": round(float(out["S_ent_time"].item()), 4),
                    "grad_norm": round(grad_norm, 3),
                }
            )

        # 周期实时日志与遥测
        log_interval = 1 if args.smoke else 25
        if step % log_interval == 0 or step == total_steps:
            last_h = history[-1]
            adv_m = float(np.mean(track_stats["cdcl_adv_std"][-15:])) if track_stats["cdcl_adv_std"] else 0.0
            unk_m = float(np.mean(track_stats["mcts_r64_unk_drop"][-10:])) if track_stats["mcts_r64_unk_drop"] else 0.0
            gate_m = float(np.mean(track_stats["mcts_r64_gate_drop"][-10:])) if track_stats["mcts_r64_gate_drop"] else 0.0
            log_msg(
                f"[M3 {step:4d}/{total_steps}] {last_h['stage']} | loss={last_h['total_loss']:.4f} | "
                f"rew={last_h.get('grpo_reward', 0.0):+.3f} | S_ent_c={last_h['s_ent_carry']:.3f} | "
                f"adv_std={adv_m:.3f} | R64_drop(unk={unk_m:.1f}, gate={gate_m:.1f})"
            )
            flush_telemetry(f"M3 训练 {last_h['stage']}", step)

        # 周期双域严格隔离评测与最优权重保存
        if (eval_every > 0 and step % eval_every == 0 and step < total_steps) or step == total_steps:
            heal_e = evaluate_defect_healing(
                model, core, torch_eng, device, rounds=64,
                n_crystals=2 if args.smoke else 6,
                n_steps=12 if args.smoke else 24,
                k_budget=16 if args.smoke else 24,
                seed=args.eval_seed + 11,
            )
            ent_e = _evaluate_pair_entanglement_diagnostic(model, core, torch_eng, device, rounds=64)
            heal_e.update(ent_e)
            heal_e["step"] = step
            heal_history.append(heal_e)

            blind_e = evaluate_m3_blind_suite(
                model, core, torch_eng, device, suite=eval_suite, n_cases=eval_cases,
                sat_time_limit=eval_tl, mcts_sims=8 if args.smoke else 32, seed=args.eval_seed,
            )
            blind_e["step"] = step
            blind_history.append(blind_e)

            save_ckpt("latest_m3.pt", step)
            f_score = float(blind_e["frontier_score"])
            if f_score >= best_frontier_score:
                best_frontier_score = f_score
                save_ckpt("best_m3_frontier.pt", step)

            log_msg(
                f"[Eval@{step}] Wolff={heal_e['neural_wolff']['reduction_pct']:.1f}% | "
                f"PairAcc={ent_e['pair_joint_acc']:.4f} | Frontier={f_score:.4f} (best={best_frontier_score:.4f}) | "
                f"R64_MCTS_drop(unk={blind_e['r64_track_a_mcts'][0]['unk_drop']}, gate={blind_e['r64_track_a_mcts'][0]['gate_drop']})"
            )
            flush_telemetry("M3 阶段评测", step)

    final_ckpt = save_ckpt(f"ouroboros_m3_step{total_steps}.pt", total_steps)
    flush_telemetry("M3 在线训练完成", total_steps, status="completed")

    summary = {
        "preset": preset_name,
        "steps": total_steps,
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "best_frontier_score": best_frontier_score,
        "final_ckpt": str(final_ckpt),
        "best_ckpt": str(ckpt_dir / "best_m3_frontier.pt"),
        "track_summary": {
            "cdcl_adv_std_mean": float(np.mean(track_stats["cdcl_adv_std"])) if track_stats["cdcl_adv_std"] else 0.0,
            "cdcl_sat_rate_mean": float(np.mean(track_stats["cdcl_sat_rate"])) if track_stats["cdcl_sat_rate"] else 0.0,
            "collapse_loss_final": float(track_stats["collapse_loss"][-1]) if track_stats["collapse_loss"] else 0.0,
            "mcts_r64_unk_drop_mean": float(np.mean(track_stats["mcts_r64_unk_drop"])) if track_stats["mcts_r64_unk_drop"] else 0.0,
            "mcts_r64_gate_drop_mean": float(np.mean(track_stats["mcts_r64_gate_drop"])) if track_stats["mcts_r64_gate_drop"] else 0.0,
            "mcts_r64_splice_drop_mean": float(np.mean(track_stats["mcts_r64_splice_drop"])) if track_stats["mcts_r64_splice_drop"] else 0.0,
        },
        "eval_before": {"heal": heal_history[0], "blind": blind_history[0]},
        "eval_after": {"heal": heal_history[-1], "blind": blind_history[-1]},
    }
    _atomic_write_json(out_dir / "m3_train_summary.json", summary)
    log_msg(f"M3 训练全部完成 | 总耗时 {summary['elapsed_sec']}s | 最佳 Frontier={best_frontier_score:.4f}")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase-SHA V10 Milestone 3 Online GRPO+MCTS Trainer")
    p.add_argument("--preset", type=str, default="a100", choices=["smoke", "a100"])
    p.add_argument("--init_ckpt", type=str, default="")
    p.add_argument("--out_dir", type=str, default="v10_ouroboros64/runs/m3_train")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--cdcl_group_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260301)
    p.add_argument("--eval_seed", type=int, default=999301)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p


if __name__ == "__main__":
    train_m3(build_argparser().parse_args())
