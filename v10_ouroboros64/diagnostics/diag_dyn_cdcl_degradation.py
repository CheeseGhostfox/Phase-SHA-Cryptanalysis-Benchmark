"""
R2 只读诊断脚本 2：M3-V2 DynamicNeuralCDCLSolver 冲突数反升 6 组单因素消融反例实验（含 no_any_set_phases）。

严格零侵入（不修改 dynamic_cdcl.py 或任何已有源码与存档）：
  - 通过只读继承 DynamicNeuralCDCLSolver 实现精确受控的单因素消融变体：
      1. full: 原版 DynamicNeuralCDCLSolver (enable_flp=True, enable_cubes=True)
      2. disable_cube: 关闭 cube_depth=2 探测 (enable_cubes=False)
      3. no_midsearch_set_phases: 保留 Episode 1 set_phases，Episode>=2 跳过 set_phases（避免覆写 Phase Saving）
      4. zero_heatmap: 将传入模型的 conflict_heatmap 强制清零（剥离零训练暴露 OOD 通道 [73]）
      5. no_any_set_phases: 关闭全阶段 set_phases（剥离 Episode 1 虚假高置信相位注入）
      6. pure_cadical: 纯 CaDiCaL-195 基线
  - 在 R=24 40b、R=25 24b（同源随机流 seed=20260499，默认 N=8，--fast 时 N=4）与
    R=26 16b（同源流前 N=4 例即 case 0..3，--fast 时 N=2，对齐 m3_v2_paired_bench 6.0s 时限）上运行；
  - 全部 SAT 解强制经 core.verify_preimage(...) 逐比特验真并断言 100% 通过。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from pysat.solvers import Solver

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v10_ouroboros64.ouroboros_bench.m3_v2_paired_bench import compute_paired_comparison
from v10_ouroboros64.ouroboros_core.cnf_cdcl import (
    CaDiCaLCDCLSolver,
    SHA256CNFEncoder,
    make_end_free_constraints,
)
from v10_ouroboros64.ouroboros_core.engine import (
    OuroborosCore,
    enforce_fp32_determinism,
    get_core,
)
from v10_ouroboros64.ouroboros_mcts.dynamic_cdcl import (
    DynamicNeuralCDCLSolver,
    DynamicSolveResult,
    _build_var_map_and_known_mask,
    _extract_cadical_w0_15,
)
from v10_ouroboros64.ouroboros_mcts.hybrid_solver import load_v10_checkpoint
from v10_ouroboros64.ouroboros_net.heads import PLANE_COUNT
from v10_ouroboros64.ouroboros_train.trainer import _sample_random_words


DEFAULT_M3V2_CKPT = "v10_ouroboros64/runs/a100_m3v2/checkpoints/best_m3v2.pt"
DEFAULT_OUT_JSON = "v10_ouroboros64/diagnostics/dyn_cdcl_degradation_report.json"


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
    assert not matmul_tf32 and not cudnn_tf32, "TF32 未禁用"
    return {
        "cuda_available": cuda_avail,
        "allow_tf32_matmul": matmul_tf32,
        "allow_tf32_cudnn": cudnn_tf32,
        "fp32_locked": (not matmul_tf32) and (not cudnn_tf32),
    }


class InstrumentedAblationDynCDCLSolver(DynamicNeuralCDCLSolver):
    """
    只读继承 DynamicNeuralCDCLSolver：
      - 当 zero_heatmap=False 且 skip_midsearch_set_phases=False 时，执行与基类完全一致的逻辑，
        并额外统计 flp_propagations、cube_attempts、cube_unsolved_wasted；
      - 当 zero_heatmap=True 时，屏蔽传入 _neural_score_partial 的 conflict_heatmap；
      - 当 skip_midsearch_set_phases=True 时，仅在 episodes == 1 调用 set_phases，episodes >= 2 跳过。
    """

    def __init__(
        self,
        *args: Any,
        zero_heatmap: bool = False,
        skip_midsearch_set_phases: bool = False,
        skip_all_set_phases: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.zero_heatmap = bool(zero_heatmap)
        self.skip_midsearch_set_phases = bool(skip_midsearch_set_phases)
        self.skip_all_set_phases = bool(skip_all_set_phases)
        self.last_diag_extra: Dict[str, int] = {}

    @torch.no_grad()
    def _neural_score_partial(
        self,
        H_single: np.ndarray,
        rounds: int,
        assigned_w_bits: Dict[Tuple[int, int], int],
        conflict_heatmap: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        hm = np.zeros_like(conflict_heatmap) if self.zero_heatmap else conflict_heatmap
        return super()._neural_score_partial(H_single, rounds, assigned_w_bits, hm)

    def solve(
        self,
        H_target: Sequence[int],
        rounds: int = 24,
        fixed_words: Optional[Dict[int, int]] = None,
        fixed_bits: Optional[Dict[Tuple[int, int], int]] = None,
        extra_clauses: Optional[Sequence[Sequence[int]]] = None,
        conflict_budget: int = 250000,
        time_limit_sec: float = 6.0,
        initial_chunk: int = 3500,
        cube_depth: int = 2,
        phase_temperature: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> DynamicSolveResult:
        t0 = time.perf_counter()
        ht_np = np.asarray(H_target, dtype=np.uint32).reshape(8)
        enc = SHA256CNFEncoder(rounds=rounds).build(
            H_target=ht_np, fixed_words=fixed_words, fixed_bits=fixed_bits
        )
        clauses = list(enc.clauses)
        if extra_clauses:
            clauses.extend(list(c) for c in extra_clauses if c)

        var_to_coord, const_fixed = _build_var_map_and_known_mask(enc)
        base_assigned: Dict[Tuple[int, int], int] = {}
        for wi, wv in (fixed_words or {}).items():
            u = int(wv) & 0xFFFFFFFF
            for bb in range(32):
                base_assigned[(int(wi), bb)] = (u >> bb) & 1
        for (wi, bb), bv in (fixed_bits or {}).items():
            base_assigned[(int(wi), int(bb))] = int(bv) & 1

        conflict_heatmap = np.zeros((rounds, PLANE_COUNT, 32), dtype=np.float32)
        episodes = 0
        flp_units_proven = 0
        flp_propagations = 0
        cores_pruned = 0
        min_core_len = 0
        cube_attempts = 0
        cube_unsolved_wasted = 0
        set_phases_calls = 0
        res_state: Optional[bool] = None
        w0_15: Optional[np.ndarray] = None
        valid_preimage = False

        with Solver(name=self.solver_name, bootstrap_with=clauses) as solver:
            chunk = int(initial_chunk)
            tried_cubes = set()
            tried_pairs = set()

            while True:
                elapsed = time.perf_counter() - t0
                if time_limit_sec > 0.0 and elapsed >= time_limit_sec:
                    break
                cur_stats = solver.accum_stats() if hasattr(solver, "accum_stats") else {}
                used_confl = int(cur_stats.get("conflicts", 0))
                if conflict_budget > 0 and used_confl >= conflict_budget:
                    break

                ok_root, prop_lits = solver.propagate()
                if not ok_root:
                    res_state = False
                    break

                cur_assigned = dict(base_assigned)
                assigned_vars = set()
                for lit in prop_lits:
                    av = abs(int(lit))
                    assigned_vars.add(av)
                    if av in var_to_coord:
                        wi_m, bb_m, s_in_w = var_to_coord[av]
                        lit_true = 1 if int(lit) > 0 else 0
                        cur_assigned[(wi_m, bb_m)] = lit_true if s_in_w > 0 else (1 - lit_true)

                episodes += 1
                coll_w, phase_w = self._neural_score_partial(
                    ht_np, rounds, cur_assigned, conflict_heatmap
                )

                free_candidates: List[Tuple[float, int, int, int, float]] = []
                for wi in range(16):
                    for bb in range(32):
                        if const_fixed[wi, bb] or (wi, bb) in cur_assigned:
                            continue
                        lit_v = enc.W_vars[wi][bb]
                        if isinstance(lit_v, bool) or not isinstance(lit_v, int):
                            continue
                        if abs(lit_v) in assigned_vars:
                            continue
                        logit = float(phase_w[wi, bb])
                        if phase_temperature > 0.0 and rng is not None:
                            logit += float(rng.normal(0.0, phase_temperature))
                        sc = float(coll_w[wi, bb]) * (0.25 + abs(logit))
                        free_candidates.append((sc, lit_v, wi, bb, logit))

                free_candidates.sort(key=lambda x: x[0], reverse=True)

                rescored: List[Tuple[float, int, int, int, float]] = []
                if self.enable_flp and free_candidates:
                    n_base = len(prop_lits)
                    k_1bit = max(self.flp_top_k, 12 if episodes >= 2 else self.flp_top_k)
                    surviving_top: List[Tuple[int, int, int]] = []
                    for idx_c, (sc, lit_v, wi, bb, logit) in enumerate(free_candidates):
                        if idx_c < k_1bit:
                            ok_p, lp = solver.propagate(assumptions=[lit_v])
                            ok_n, ln = solver.propagate(assumptions=[-lit_v])
                            flp_propagations += 2
                            if (not ok_p) and (not ok_n):
                                res_state = False
                                break
                            if not ok_p:
                                solver.add_clause([-lit_v])
                                flp_units_proven += 1
                                self._stamp_heat(conflict_heatmap, wi, bb, rounds, weight=1.0)
                                continue
                            if not ok_n:
                                solver.add_clause([lit_v])
                                flp_units_proven += 1
                                self._stamp_heat(conflict_heatmap, wi, bb, rounds, weight=1.0)
                                continue
                            dp = max(0, len(lp) - n_base)
                            dn = max(0, len(ln) - n_base)
                            la_bonus = 0.25 * min(dp, dn) + 0.05 * abs(dp - dn)
                            rescored.append((sc + la_bonus, lit_v, wi, bb, logit))
                            if len(surviving_top) < 5:
                                surviving_top.append((lit_v, wi, bb))
                        else:
                            rescored.append((sc, lit_v, wi, bb, logit))
                    if res_state is False:
                        break

                    if episodes >= 2 and len(surviving_top) >= 2:
                        for i_p in range(len(surviving_top)):
                            l1, w1, b1 = surviving_top[i_p]
                            for j_p in range(i_p + 1, len(surviving_top)):
                                l2, w2, b2 = surviving_top[j_p]
                                for s1 in (l1, -l1):
                                    for s2 in (l2, -l2):
                                        pkey = (s1, s2) if abs(s1) < abs(s2) else (s2, s1)
                                        if pkey in tried_pairs:
                                            continue
                                        tried_pairs.add(pkey)
                                        ok_2, _ = solver.propagate(assumptions=[s1, s2])
                                        flp_propagations += 1
                                        if not ok_2:
                                            solver.add_clause([-s1, -s2])
                                            cores_pruned += 1
                                            min_core_len = 2 if min_core_len == 0 else min(min_core_len, 2)
                                            self._stamp_heat(conflict_heatmap, w1, b1, rounds, weight=0.6)
                                            self._stamp_heat(conflict_heatmap, w2, b2, rounds, weight=0.6)

                    rescored.sort(key=lambda x: x[0], reverse=True)
                else:
                    rescored = free_candidates

                if not rescored:
                    solver.conf_budget(max(100, chunk))
                    res_state = solver.solve_limited()
                    break

                # 受控消融点：skip_all_set_phases 或 skip_midsearch_set_phases
                do_set_phases = (
                    (not self.skip_all_set_phases)
                    and ((episodes == 1) or (not self.skip_midsearch_set_phases))
                )
                if do_set_phases:
                    max_inject = 64 if episodes == 1 else 16
                    fresh_phases = [
                        (lit_v if logit >= 0.0 else -lit_v)
                        for _, lit_v, _, _, logit in rescored
                        if abs(logit) >= 0.08
                    ][:max_inject]
                    if episodes == 1 and not fresh_phases:
                        fresh_phases = [
                            (lit_v if logit >= 0.0 else -lit_v)
                            for _, lit_v, _, _, logit in rescored[:32]
                        ]
                    if fresh_phases:
                        try:
                            solver.set_phases(literals=fresh_phases)
                            set_phases_calls += 1
                        except Exception:
                            pass

                run_cube = (
                    self.enable_cubes
                    and used_confl >= 7000
                    and (episodes % 2 == 0)
                    and len(rescored) >= max(6, cube_depth * 3)
                )
                if run_cube:
                    cube = []
                    for _, lit_v, wi, bb, logit in rescored[:cube_depth]:
                        signed_l = lit_v if logit >= 0.0 else -lit_v
                        cube.append(signed_l)
                    ckey = tuple(cube)
                    if ckey in tried_cubes and cube:
                        cube[-1] = -cube[-1]
                        ckey = tuple(cube)

                    if ckey not in tried_cubes:
                        tried_cubes.add(ckey)
                        cube_attempts += 1
                        cube_budget = min(1500, max(500, chunk // 2))
                        if conflict_budget > 0:
                            rem_b = int(conflict_budget) - used_confl
                            if rem_b <= 0:
                                break
                            cube_budget = min(cube_budget, rem_b)
                        solver.conf_budget(cube_budget)
                        rc = solver.solve_limited(assumptions=cube)
                        if rc is True:
                            res_state = True
                            break
                        if rc is False:
                            uc = solver.get_core()
                            if uc is None or len(uc) == 0:
                                res_state = False
                                break
                            cores_pruned += 1
                            cl_len = len(uc)
                            min_core_len = cl_len if min_core_len == 0 else min(min_core_len, cl_len)
                            solver.add_clause([-int(x) for x in uc])
                            for clit in uc:
                                av = abs(int(clit))
                                if av in var_to_coord:
                                    wi_c, bb_c, _ = var_to_coord[av]
                                    self._stamp_heat(conflict_heatmap, wi_c, bb_c, rounds, weight=1.5)
                            continue
                        else:
                            cube_unsolved_wasted += 1
                            _, _, wi0, bb0, _ = rescored[0]
                            self._stamp_heat(conflict_heatmap, wi0, bb0, rounds, weight=0.6)

                cur_budget = chunk
                if conflict_budget > 0:
                    rem = int(conflict_budget) - int((solver.accum_stats() or {}).get("conflicts", 0))
                    if rem <= 0:
                        break
                    cur_budget = min(cur_budget, rem)
                solver.conf_budget(max(100, cur_budget))
                res_state = solver.solve_limited()
                if res_state is not None:
                    break
                chunk = min(8000, int(chunk * 1.4))

            stats = solver.accum_stats() if hasattr(solver, "accum_stats") else {}
            if res_state is True:
                model_lits = solver.get_model() or []
                w0_15 = _extract_cadical_w0_15(enc, model_lits)
                valid_preimage = bool(self.core.verify_preimage(w0_15, ht_np, rounds=rounds)[0])
                assert valid_preimage, f"InstrumentedAblationDynCDCLSolver 未通过 verify_preimage R={rounds}"

        dt = time.perf_counter() - t0
        self.last_diag_extra = {
            "flp_propagations": flp_propagations,
            "cube_attempts": cube_attempts,
            "cube_unsolved_wasted": cube_unsolved_wasted,
            "set_phases_calls": set_phases_calls,
        }
        status_str = "SAT" if res_state is True else ("UNSAT" if res_state is False else "TIMEOUT_OR_BUDGET")
        return DynamicSolveResult(
            rounds=int(rounds),
            solver_type="dynamic_neural_cdcl",
            status=status_str,
            valid_preimage=bool(valid_preimage),
            conflicts=int(stats.get("conflicts", 0)),
            decisions=int(stats.get("decisions", 0)),
            elapsed_sec=float(dt),
            episodes=episodes,
            flp_units_proven=flp_units_proven,
            cores_pruned=cores_pruned,
            min_core_len=min_core_len,
            w0_15=w0_15,
        )


def run_diag_dyn_cdcl_degradation(
    ckpt_path: str = DEFAULT_M3V2_CKPT,
    n_cases: int = 8,
    n_cases_r26: int = 4,
    time_limit_sec: float = 5.0,
    conflict_budget: int = 250000,
    seed: int = 20260499,
    fast: bool = False,
    cpu: bool = False,
    out_json: str = DEFAULT_OUT_JSON,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    fp32_info = _check_fp32_locked()
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    core = get_core()
    n_eff = min(4, n_cases) if fast else int(n_cases)
    n_r26_eff = min(2, n_cases_r26) if fast else int(n_cases_r26)

    print(
        f"[diag_dyn_cdcl] 启动 | device={device} | fp32_locked={fp32_info['fp32_locked']} | "
        f"ckpt={ckpt_path} | n_cases={n_eff} | n_cases_r26={n_r26_eff} | tl={time_limit_sec}s | fast={fast}",
        flush=True,
    )
    model, payload = load_v10_checkpoint(ckpt_path, device=device)
    ckpt_step = int(payload.get("step", -1))

    # 实例化 6 组求解器（含 no_any_set_phases）+ 1 个原生基类用于等价性自检
    orig_full = DynamicNeuralCDCLSolver(
        model=model, core=core, device=device, enable_flp=True, enable_cubes=True
    )
    solvers = {
        "full": InstrumentedAblationDynCDCLSolver(
            model=model, core=core, device=device, enable_flp=True, enable_cubes=True,
            zero_heatmap=False, skip_midsearch_set_phases=False,
        ),
        "disable_cube": InstrumentedAblationDynCDCLSolver(
            model=model, core=core, device=device, enable_flp=True, enable_cubes=False,
            zero_heatmap=False, skip_midsearch_set_phases=False,
        ),
        "no_midsearch_set_phases": InstrumentedAblationDynCDCLSolver(
            model=model, core=core, device=device, enable_flp=True, enable_cubes=True,
            zero_heatmap=False, skip_midsearch_set_phases=True,
        ),
        "zero_heatmap": InstrumentedAblationDynCDCLSolver(
            model=model, core=core, device=device, enable_flp=True, enable_cubes=True,
            zero_heatmap=True, skip_midsearch_set_phases=False,
        ),
        "no_any_set_phases": InstrumentedAblationDynCDCLSolver(
            model=model, core=core, device=device, enable_flp=True, enable_cubes=True,
            zero_heatmap=False, skip_midsearch_set_phases=False, skip_all_set_phases=True,
        ),
        "pure_cadical": CaDiCaLCDCLSolver("cadical195"),
    }

    # 与 m3_v2_paired_bench.py (seed=20260499, n_main=16) 完全同源对齐：
    # 跳过 (24, 32)，采样 (24, 40) 与 (25, 24)，跳过 (26, 12)，再采样 (26, 16) 前 n_r26_eff 例
    rng = np.random.default_rng(seed)
    _ = _sample_random_words(rng, 16, 16)  # 跳过 (24, 32)
    w_24_40 = _sample_random_words(rng, 16, 16)[:n_eff]
    w_25_24 = _sample_random_words(rng, 16, 16)[:n_eff]
    _ = _sample_random_words(rng, 16, 16)  # 跳过 (26, 12)
    w_26_16 = _sample_random_words(rng, 16, 16)[:n_r26_eff]
    tier_samples: List[Tuple[int, int, np.ndarray]] = [
        (24, 40, w_24_40),
        (25, 24, w_25_24),
        (26, 16, w_26_16),
    ]

    # 等价性自检：确认 InstrumentedAblationDynCDCLSolver(full) 与原版 DynamicNeuralCDCLSolver 在同样例上一致
    w_chk = tier_samples[0][2][0:1]
    cr_chk = core.compress_exact(w_chk, rounds=24)
    fw_chk, fbits_chk = make_end_free_constraints(w_chk[0], 40)
    r_orig = orig_full.solve(
        cr_chk.H_target[0], rounds=24, fixed_words=fw_chk, fixed_bits=fbits_chk,
        conflict_budget=conflict_budget, time_limit_sec=time_limit_sec,
    )
    r_inst = solvers["full"].solve(
        cr_chk.H_target[0], rounds=24, fixed_words=fw_chk, fixed_bits=fbits_chk,
        conflict_budget=conflict_budget, time_limit_sec=time_limit_sec,
    )
    assert (
        r_orig.status == r_inst.status
        and r_orig.conflicts == r_inst.conflicts
        and r_orig.episodes == r_inst.episodes
        and r_orig.flp_units_proven == r_inst.flp_units_proven
        and r_orig.cores_pruned == r_inst.cores_pruned
    ), f"自检不一致: orig={r_orig.to_metrics_dict()} vs inst={r_inst.to_metrics_dict()}"
    print(
        f"[diag_dyn_cdcl] 基类等价性自检通过 (R=24 40b case0: conflicts={r_orig.conflicts}, "
        f"episodes={r_orig.episodes}, flp={r_orig.flp_units_proven}, cores={r_orig.cores_pruned})",
        flush=True,
    )

    total_sat_checked = 0
    total_sat_verified = 0
    tier_reports: List[Dict[str, Any]] = []

    for r_b, fb, w_mat in tier_samples:
        n_tier = int(w_mat.shape[0])
        tl_tier = max(float(time_limit_sec), 6.0) if r_b >= 26 else float(time_limit_sec)
        cr = core.compress_exact(w_mat, rounds=r_b)
        per_group_records: Dict[str, List[Dict[str, Any]]] = {g: [] for g in solvers}

        for idx in range(n_tier):
            ht = cr.H_target[idx]
            fw, fbits = make_end_free_constraints(w_mat[idx], fb)

            for gname, slv in solvers.items():
                if gname == "pure_cadical":
                    rp = slv.solve(
                        ht, rounds=r_b, fixed_words=fw, fixed_bits=fbits,
                        conflict_budget=conflict_budget, time_limit_sec=tl_tier,
                    )
                    is_sat = (rp["status"] == "SAT")
                    valid_ok = False
                    if is_sat:
                        assert rp["w0_15"] is not None
                        v_arr = core.verify_preimage(np.asarray(rp["w0_15"], dtype=np.uint32), ht, rounds=r_b)
                        valid_ok = bool(v_arr[0]) and bool(rp["valid_preimage"])
                        assert valid_ok, f"pure_cadical verify_preimage 失败 R={r_b} case={idx}"
                        total_sat_checked += 1
                        total_sat_verified += 1
                    rec = {
                        "case": idx,
                        "status": rp["status"],
                        "conflicts": int(rp["conflicts"]),
                        "decisions": int(rp["decisions"]),
                        "time_sec": float(rp["elapsed_sec"]),
                        "episodes": 1,
                        "flp_units": 0,
                        "flp_propagations": 0,
                        "cube_cores": 0,
                        "cube_attempts": 0,
                        "cube_unsolved_wasted": 0,
                        "set_phases_calls": 0,
                        "valid_preimage": valid_ok,
                    }
                else:
                    rd = slv.solve(
                        ht, rounds=r_b, fixed_words=fw, fixed_bits=fbits,
                        conflict_budget=conflict_budget, time_limit_sec=tl_tier,
                    )
                    is_sat = (rd.status == "SAT")
                    valid_ok = False
                    if is_sat:
                        assert rd.w0_15 is not None
                        v_arr = core.verify_preimage(rd.w0_15, ht, rounds=r_b)
                        valid_ok = bool(v_arr[0]) and bool(rd.valid_preimage)
                        assert valid_ok, f"{gname} verify_preimage 失败 R={r_b} case={idx}"
                        total_sat_checked += 1
                        total_sat_verified += 1
                    ex = slv.last_diag_extra
                    rec = {
                        "case": idx,
                        "status": rd.status,
                        "conflicts": int(rd.conflicts),
                        "decisions": int(rd.decisions),
                        "time_sec": float(rd.elapsed_sec),
                        "episodes": int(rd.episodes),
                        "flp_units": int(rd.flp_units_proven),
                        "flp_propagations": int(ex.get("flp_propagations", 0)),
                        "cube_cores": int(rd.cores_pruned),
                        "cube_attempts": int(ex.get("cube_attempts", 0)),
                        "cube_unsolved_wasted": int(ex.get("cube_unsolved_wasted", 0)),
                        "set_phases_calls": int(ex.get("set_phases_calls", 0)),
                        "valid_preimage": valid_ok,
                    }
                per_group_records[gname].append(rec)

            print(
                f"  [R={r_b} {fb}b case {idx+1}/{n_tier}] "
                + " | ".join(
                    f"{g}={per_group_records[g][-1]['conflicts']}c({per_group_records[g][-1]['status'][:3]})"
                    for g in ("pure_cadical", "full", "disable_cube", "no_midsearch_set_phases", "zero_heatmap", "no_any_set_phases")
                ),
                flush=True,
            )

        pure_confls = [x["conflicts"] for x in per_group_records["pure_cadical"]]
        full_confls = [x["conflicts"] for x in per_group_records["full"]]

        summary_groups: Dict[str, Any] = {}
        paired_vs_pure: Dict[str, Any] = {}
        paired_vs_full: Dict[str, Any] = {}

        for gname, recs in per_group_records.items():
            confls = [x["conflicts"] for x in recs]
            times = [x["time_sec"] for x in recs]
            sat_cnt = sum(1 for x in recs if x["status"] == "SAT")
            val_cnt = sum(1 for x in recs if x["valid_preimage"])
            assert val_cnt == sat_cnt
            summary_groups[gname] = {
                "sat": sat_cnt,
                "cases": n_tier,
                "valid_preimage_count": val_cnt,
                "mean_conflicts": round(float(np.mean(confls)), 2),
                "median_conflicts": round(float(np.median(confls)), 2),
                "mean_time_sec": round(float(np.mean(times)), 4),
                "median_time_sec": round(float(np.median(times)), 4),
                "mean_episodes": round(float(np.mean([x["episodes"] for x in recs])), 2),
                "mean_flp_units": round(float(np.mean([x["flp_units"] for x in recs])), 2),
                "mean_flp_propagations": round(float(np.mean([x["flp_propagations"] for x in recs])), 2),
                "mean_cube_cores": round(float(np.mean([x["cube_cores"] for x in recs])), 2),
                "mean_cube_attempts": round(float(np.mean([x["cube_attempts"] for x in recs])), 2),
                "mean_cube_unsolved_wasted": round(float(np.mean([x["cube_unsolved_wasted"] for x in recs])), 2),
                "mean_set_phases_calls": round(float(np.mean([x["set_phases_calls"] for x in recs])), 2),
            }
            if gname != "pure_cadical":
                paired_vs_pure[f"{gname}_vs_pure_cadical"] = compute_paired_comparison(
                    pure_confls, confls, tie_eps=0.5, seed=seed + r_b
                )
            if gname != "full":
                # 正值代表相比 full 减少的冲突百分比
                paired_vs_full[f"{gname}_vs_full"] = compute_paired_comparison(
                    full_confls, confls, tie_eps=0.5, seed=seed + r_b + 500
                )

        tier_reports.append(
            {
                "rounds": r_b,
                "free_bits": fb,
                "cases": n_tier,
                "time_limit_sec": tl_tier,
                "summary": summary_groups,
                "paired_vs_pure_cadical": paired_vs_pure,
                "paired_vs_full": paired_vs_full,
                "per_case_records": per_group_records,
            }
        )

    assert total_sat_checked == total_sat_verified and total_sat_checked > 0
    elapsed = round(time.perf_counter() - t0, 3)
    report: Dict[str, Any] = {
        "diagnostic": "dyn_cdcl_degradation_ablation_readonly",
        "fp32_check": fp32_info,
        "device": str(device),
        "ckpt_path": ckpt_path,
        "ckpt_step": ckpt_step,
        "seed": int(seed),
        "n_cases": n_eff,
        "n_cases_r26": n_r26_eff,
        "conflict_budget": int(conflict_budget),
        "time_limit_sec": float(time_limit_sec),
        "fast_mode": bool(fast),
        "self_check_orig_vs_instrumented_passed": True,
        "total_sat_verified": total_sat_verified,
        "verify_pass_rate": round(total_sat_verified / max(1, total_sat_checked), 6),
        "elapsed_sec": elapsed,
        "tiers": tier_reports,
    }
    _atomic_write_json(Path(out_json), report)
    print(
        f"[diag_dyn_cdcl] 完成，耗时 {elapsed:.2f}s | verify_pass_rate={report['verify_pass_rate']:.4f} "
        f"({total_sat_verified}/{total_sat_checked}) | 写入 {out_json}",
        flush=True,
    )
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="R2 只读诊断 2：Dyn CDCL 冲突反升 6 组消融实验（含 no_any_set_phases 与 R=26 16b 探针）"
    )
    p.add_argument("--ckpt_path", type=str, default=DEFAULT_M3V2_CKPT)
    p.add_argument("--n_cases", type=int, default=8)
    p.add_argument("--n_cases_r26", type=int, default=4)
    p.add_argument("--time_limit_sec", type=float, default=5.0)
    p.add_argument("--conflict_budget", type=int, default=250000)
    p.add_argument("--seed", type=int, default=20260499)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_diag_dyn_cdcl_degradation(
        ckpt_path=args.ckpt_path,
        n_cases=args.n_cases,
        n_cases_r26=args.n_cases_r26,
        time_limit_sec=args.time_limit_sec,
        conflict_budget=args.conflict_budget,
        seed=args.seed,
        fast=bool(args.fast or args.quick),
        cpu=args.cpu,
        out_json=args.out_json,
    )
