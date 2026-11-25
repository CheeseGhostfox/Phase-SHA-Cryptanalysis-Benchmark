"""
R2 只读诊断脚本 4：原生 R=64 轨 A 虫洞 MCTS 停滞根因逐模拟追踪与反例验证。

严格零侵入（不修改 wormhole_mcts.py 或任何已有源码与存档）：
  1. 显式调用 enforce_fp32_determinism() 并校验 allow_tf32 == False；
  2. 只读加载 best_m3v2.pt（并对比 ouroboros_a100_step4000.pt），在 R=64（同源测试集 N=4, sims=51）
     精确追踪前 51 次扩展，核验：
       - 根节点 28 个动作结构；
       - splice_block[:8] 仅含 S[32, {0,4,3,7}, 0..1] 8 bit 而截断丢弃全部 W[32]；
       - 4 例均确定收敛至 unk_drop=94, gate_drop=26, splice_drop=8, pruned_conflicts=0（对比 M2 step4000 的 80/16/16）；
  3. 只读反例实验：
       (a) 绕过 rounds<=28 硬编码限制强开 R=64 根与子节点 GF(2)（time_limit_sec=3.0）；
       (b) 关闭固定 splice_block 与 enable_macro_actions=False（sims=51），对比深度与坍缩变化。
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

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v10_ouroboros64.ouroboros_core.engine import (
    SYM_0,
    SYM_1,
    TorchCrystalEngine,
    enforce_fp32_determinism,
    get_core,
)
from v10_ouroboros64.ouroboros_mcts.hybrid_solver import load_v10_checkpoint
from v10_ouroboros64.ouroboros_mcts.symbol_env import GeneralizedSymbolEnv
from v10_ouroboros64.ouroboros_mcts.wormhole_mcts import (
    MCTSAction,
    MCTSNode,
    MCTSSearchResult,
    WormholeMCTS,
)
from v10_ouroboros64.ouroboros_net.heads import PLANE_COUNT
from v10_ouroboros64.ouroboros_train.trainer import _sample_random_words


DEFAULT_M3V2_CKPT = "v10_ouroboros64/runs/a100_m3v2/checkpoints/best_m3v2.pt"
DEFAULT_M2_CKPT = "v10_ouroboros64/runs/a100_m2/checkpoints/ouroboros_a100_step4000.pt"
DEFAULT_OUT_JSON = "v10_ouroboros64/diagnostics/diag_r64_wormhole_stall_report.json"


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


def _describe_action(act: MCTSAction, idx: int) -> Dict[str, Any]:
    assigns = (
        list(act.extra_assignments)
        if act.extra_assignments
        else [(act.kind, act.t, act.word, act.bit, act.sym)]
    )
    return {
        "index": idx,
        "is_macro": bool(act.is_macro),
        "n_assigns": len(assigns),
        "kind": int(act.kind),
        "t": int(act.t),
        "word": int(act.word),
        "bit": int(act.bit),
        "sym": int(act.sym),
        "prior": round(float(act.prior), 6),
        "collapse_score": round(float(act.collapse_score), 6),
        "assigns": [list(a) for a in assigns],
    }


class InstrumentedWormholeMCTS(WormholeMCTS):
    """
    只读继承 WormholeMCTS：
      - 记录根节点候选动作列表与逐模拟 1..K 扩展轨迹；
      - 支持反例开关：force_gf2_child（绕过 rounds<=28）、disable_splice_block_only（保留 W 宏动作仅移除固定 S[32] splice_block）。
    """

    def __init__(
        self,
        *args: Any,
        force_gf2_child: bool = False,
        disable_splice_block_only: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.force_gf2_child = bool(force_gf2_child)
        self.disable_splice_block_only = bool(disable_splice_block_only)
        self.last_trace: Dict[str, Any] = {}

    def _build_candidate_actions(
        self,
        env: GeneralizedSymbolEnv,
        focus_splice: bool = False,
        conflict_heatmap: Optional[np.ndarray] = None,
    ) -> Tuple[List[MCTSAction], np.ndarray, np.ndarray, float]:
        actions, coll_w, phase_w, pred_le = super()._build_candidate_actions(
            env, focus_splice=focus_splice, conflict_heatmap=conflict_heatmap
        )
        if self.disable_splice_block_only:
            # 仅移除固定在 S[splice_round] 的 splice_block / splice_block_alt，保留 W 宏动作与单比特动作
            sr = env.splice_round
            filtered: List[MCTSAction] = []
            for act in actions:
                if act.is_macro and all(a[0] == 1 and a[1] == sr for a in act.extra_assignments):
                    continue
                filtered.append(act)
            if filtered:
                s_p = sum(a.prior for a in filtered) + 1e-12
                for a in filtered:
                    a.prior = float(a.prior / s_p)
            actions = filtered
        return actions, coll_w, phase_w, pred_le

    def search(
        self,
        H_target: np.ndarray,
        rounds: int = 64,
        fixed_words: Optional[Dict[int, int]] = None,
        num_simulations: int = 51,
        time_limit_sec: float = 10.0,
        focus_splice: bool = True,
        fixed_bits: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> MCTSSearchResult:
        t0 = time.perf_counter()
        env_root = GeneralizedSymbolEnv(
            H_target=H_target,
            rounds=rounds,
            splice_round=max(1, rounds // 2),
            core=self.core,
        )
        fixed_bit_set = set()
        if fixed_words:
            for wi, word_val in fixed_words.items():
                u_val = int(word_val) & 0xFFFFFFFF
                for bb in range(32):
                    bit_sym = SYM_1 if ((u_val >> bb) & 1) else SYM_0
                    env_root.set_symbol(0, int(wi), 0, bb, bit_sym, run_propagate=False)
                    fixed_bit_set.add((int(wi), bb))
        if fixed_bits:
            for (wi, bb), bit_val in fixed_bits.items():
                w_i, b_i = int(wi), int(bb)
                bit_sym = SYM_1 if (int(bit_val) & 1) else SYM_0
                env_root.set_symbol(0, w_i, 0, b_i, bit_sym, run_propagate=False)
                fixed_bit_set.add((w_i, b_i))

        use_gf2_root = (rounds <= 28) if self.use_gf2_root is None else bool(self.use_gf2_root)
        t_root_prop0 = time.perf_counter()
        root_step = env_root.propagate(use_gf2=use_gf2_root, max_iters=self.prop_max_iters)
        root_prop_sec = time.perf_counter() - t_root_prop0

        learned_units: List[Tuple[int, int, int, int, int]] = []
        seen_units = set()
        learned_clauses: List[List[Tuple[int, int, int, int, int]]] = []
        seen_clauses = set()
        for wi in range(16):
            for bb in range(32):
                if (wi, bb) in fixed_bit_set:
                    continue
                s_val = int(env_root.W_sym[wi, bb])
                if s_val in (SYM_0, SYM_1):
                    key = (0, wi, 0, bb, s_val)
                    if key not in seen_units:
                        seen_units.add(key)
                        learned_units.append(key)

        root = MCTSNode(env=env_root, prior=1.0)
        root_splice = root.splice_freedom
        best_env = env_root
        min_unk = root.unknown_count
        min_gates = root.active_carry_gates
        min_splice = root.splice_freedom

        solved_w: Optional[np.ndarray] = None
        if not root.is_conflict:
            solved_w = self._try_extract_preimage(env_root)
            if solved_w is not None:
                root.is_solved = True

        conflict_heatmap = np.zeros((rounds, PLANE_COUNT, 32), dtype=np.float32)
        actions_r, coll_r, phase_r, root_pred_le = self._build_candidate_actions(
            env_root, focus_splice=focus_splice, conflict_heatmap=conflict_heatmap
        )
        root.untried_actions = actions_r
        root_actions_desc = [_describe_action(a, i) for i, a in enumerate(actions_r)]

        phase_preferences: Dict[Tuple[int, int], Tuple[float, int]] = {}
        for wi in range(16):
            for bb in range(32):
                if (wi, bb) in fixed_bit_set:
                    continue
                logit = float(phase_r[wi, bb])
                conf = float(coll_r[wi, bb]) * abs(logit)
                phase_preferences[(wi, bb)] = (conf, 1 if logit >= 0.0 else 0)

        sims = 0
        expansions = 0
        pruned_conflicts = 0
        gf2_child_calls = 0
        gf2_child_deductions_total = 0
        max_depth_reached = 0
        peak_unk_drop_at_exp = 0
        peak_unk_drop_path: List[str] = []
        sim_trace: List[Dict[str, Any]] = []

        if solved_w is None and not root.is_conflict:
            for _ in range(num_simulations):
                if time_limit_sec > 0.0 and (time.perf_counter() - t0) >= time_limit_sec:
                    break
                sims += 1
                node = root
                path = [node]

                while node.expanded and not node.untried_actions and node.children and not node.is_conflict and not node.is_solved:
                    best_child = max(node.children.values(), key=lambda ch: ch.ucb_score(self.c_puct))
                    if best_child.is_conflict:
                        break
                    node = best_child
                    path.append(node)

                leaf_val = 0.0
                if not node.is_conflict and not node.is_solved:
                    if not node.expanded and not node.untried_actions:
                        acts, _, _, _ = self._build_candidate_actions(
                            node.env, focus_splice=focus_splice, conflict_heatmap=conflict_heatmap
                        )
                        node.untried_actions = acts
                        node.expanded = True

                    if node.untried_actions:
                        act = node.untried_actions.pop(0)
                        node.expanded = True
                        child_env = node.env.clone()
                        depth = len(path)
                        max_depth_reached = max(max_depth_reached, depth)
                        try:
                            step_res = self._apply_action_to_env(child_env, act, use_gf2=False)
                            rem_w = int(np.sum((child_env.W_sym[:16] != SYM_0) & (child_env.W_sym[:16] != SYM_1)))
                            allow_gf2 = self.force_gf2_child or (rounds <= 28)
                            if (
                                (not step_res.conflict)
                                and allow_gf2
                                and (
                                    (depth % self.gf2_interval == 0)
                                    or (step_res.unit_prop_deductions >= 12)
                                    or (rem_w <= 8)
                                )
                            ):
                                step_res = child_env.propagate(use_gf2=True, max_iters=4)
                                gf2_child_calls += 1
                                gf2_child_deductions_total += int(step_res.gf2_deductions)
                        except PermissionError:
                            from v10_ouroboros64.ouroboros_mcts.symbol_env import SymbolStepResult
                            step_res = SymbolStepResult(
                                conflict=True, unit_prop_deductions=0, gf2_deductions=0,
                                gf2_rank=0, gf2_conflicts=1, unknown_before=node.unknown_count,
                                unknown_after=node.unknown_count, active_carry_gates=node.active_carry_gates,
                                splice_freedom_bits=node.splice_freedom,
                            )

                        child = MCTSNode(env=child_env, parent=node, action=act, prior=act.prior)
                        child.is_conflict = bool(step_res.conflict)
                        node.children[act.action_key()] = child
                        path.append(child)
                        expansions += 1

                        if child.is_conflict:
                            pruned_conflicts += 1
                            leaf_val = -10.0
                            self._record_conflict_heat(conflict_heatmap, act, rounds)
                        else:
                            if child.unknown_count < min_unk:
                                min_unk = child.unknown_count
                                best_env = child_env
                                peak_unk_drop_at_exp = expansions
                                peak_unk_drop_path = [
                                    (
                                        f"macro(n={len(nd.action.extra_assignments)},k={nd.action.kind},t={nd.action.t},w={nd.action.word},b={nd.action.bit})"
                                        if nd.action.is_macro
                                        else f"bit(k={nd.action.kind},t={nd.action.t},w={nd.action.word},b={nd.action.bit},s={nd.action.sym})"
                                    )
                                    for nd in path[1:] if nd.action is not None
                                ]
                            if child.active_carry_gates < min_gates:
                                min_gates = child.active_carry_gates
                            if child.splice_freedom < min_splice:
                                min_splice = child.splice_freedom

                            v_free = math.log1p(max(0, root.unknown_count - child.unknown_count))
                            v_gate = math.log1p(max(0, root.active_carry_gates - child.active_carry_gates))
                            v_splice = math.log1p(max(0, root_splice - child.splice_freedom))
                            _, _, child_pred_le = self._neural_score_env(
                                child_env, conflict_heatmap=conflict_heatmap
                            )
                            v_energy = math.log1p(max(0.0, root_pred_le - child_pred_le))
                            uncapped_val = 0.45 * v_free + 0.35 * v_gate + 0.25 * v_splice + 0.20 * v_energy
                            cand_w = self._try_extract_preimage(child_env)
                            if cand_w is not None:
                                child.is_solved = True
                                solved_w = cand_w
                                leaf_val = 50.0 + uncapped_val
                            else:
                                leaf_val = uncapped_val

                        sim_trace.append(
                            {
                                "sim": sims,
                                "exp": expansions,
                                "depth": depth,
                                "is_macro": bool(act.is_macro),
                                "action": f"k={act.kind},t={act.t},w={act.word},b={act.bit},s={act.sym},n={len(act.extra_assignments) if act.extra_assignments else 1}",
                                "cum_unk_drop": int(root.unknown_count - child.unknown_count),
                                "cum_gate_drop": int(root.active_carry_gates - child.active_carry_gates),
                                "cum_splice_drop": int(root_splice - child.splice_freedom),
                                "unit_prop_ded": int(step_res.unit_prop_deductions),
                                "gf2_ded": int(step_res.gf2_deductions),
                                "is_conflict": bool(child.is_conflict),
                                "leaf_val": round(float(leaf_val), 4),
                            }
                        )

                for nd in path:
                    nd.visit_count += 1
                    nd.value_sum += leaf_val
                if solved_w is not None:
                    break

        self.last_trace = {
            "root_prop_sec": round(root_prop_sec, 4),
            "root_gf2_rank": int(root_step.gf2_rank),
            "root_gf2_deductions": int(root_step.gf2_deductions),
            "root_unit_prop_deductions": int(root_step.unit_prop_deductions),
            "root_actions_count": len(root_actions_desc),
            "root_actions": root_actions_desc,
            "max_depth_reached": max_depth_reached,
            "peak_unk_drop_at_exp": peak_unk_drop_at_exp,
            "peak_unk_drop_path": peak_unk_drop_path,
            "gf2_child_calls": gf2_child_calls,
            "gf2_child_deductions_total": gf2_child_deductions_total,
            "sim_trace": sim_trace,
        }

        return MCTSSearchResult(
            rounds=rounds,
            solved=(solved_w is not None),
            w0_15=solved_w,
            simulations=sims,
            expansions=expansions,
            pruned_conflicts=pruned_conflicts,
            root_unknown=root.unknown_count,
            min_unknown_reached=min_unk,
            root_active_carry_gates=root.active_carry_gates,
            min_active_carry_gates=min_gates,
            root_splice_freedom=root_splice,
            min_splice_freedom=min_splice,
            learned_units=learned_units,
            phase_preferences=phase_preferences,
            elapsed_sec=time.perf_counter() - t0,
            best_env_stats=best_env.get_stats(),
            learned_clauses=learned_clauses,
            root_visit_policy=[],
            root_value_target=0.0,
        )


def _sample_aligned_r64_cases(seed: int, n_cases: int) -> np.ndarray:
    """精确复现 m3_v2_paired_bench.py (seed=20260499) 中 R=64 轨 A MCTS 的 4 个原像实例。"""
    rng = np.random.default_rng(seed)
    for n_tier in (16, 16, 16, 16, 16, 16, 8):
        _ = _sample_random_words(rng, n_tier, 16)
    return _sample_random_words(rng, 4, 16)[:n_cases]


def _verify_root_actions_structure(root_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """核验根节点 28 个候选动作的精确结构及 splice_block[:8] 截断丢失 W[32] 现象。"""
    assert len(root_actions) == 28, f"预期根节点 28 个动作，实际 {len(root_actions)}"
    a0 = root_actions[0]
    a1 = root_actions[1]
    assert a0["is_macro"] and a0["n_assigns"] == 8
    assert a1["is_macro"] and a1["n_assigns"] == 8
    # 检查前 2 个宏动作只含 kind=1 (S), t=32, word in {0,4,3,7}, bit in {0,1}，完全没有 kind=0 (W[32])
    kinds_a0 = {x[0] for x in a0["assigns"]}
    rounds_a0 = {x[1] for x in a0["assigns"]}
    words_a0 = [x[2] for x in a0["assigns"]]
    bits_a0 = [x[3] for x in a0["assigns"]]
    has_w32_in_splice_block = any(x[0] == 0 and x[1] == 32 for x in a0["assigns"])
    assert kinds_a0 == {1} and rounds_a0 == {32} and not has_w32_in_splice_block

    w_single = [a for a in root_actions[2:18] if not a["is_macro"] and a["kind"] == 0]
    s_single = [a for a in root_actions[18:26] if not a["is_macro"] and a["kind"] == 1 and a["t"] == 32]
    w_macro = [a for a in root_actions[26:28] if a["is_macro"] and a["kind"] == 0]
    assert len(w_single) == 16 and len(s_single) == 8 and len(w_macro) == 2

    return {
        "total_root_actions": len(root_actions),
        "splice_block_0_assigns": a0["assigns"],
        "splice_block_0_words": words_a0,
        "splice_block_0_bits": bits_a0,
        "has_w32_in_splice_block": has_w32_in_splice_block,
        "w_single_branch_actions_count": len(w_single),
        "s32_single_branch_actions_count": len(s_single),
        "w_macro_actions_count": len(w_macro),
        "w_macro_target_word": w_macro[0]["t"],
        "w_macro_assigns": w_macro[0]["assigns"],
    }


def run_diag_r64_wormhole_stall(
    ckpt_path: str = DEFAULT_M3V2_CKPT,
    m2_ckpt_path: str = DEFAULT_M2_CKPT,
    n_cases: int = 4,
    sims: int = 51,
    time_limit_sec: float = 12.0,
    seed: int = 20260499,
    fast: bool = False,
    cpu: bool = False,
    out_json: str = DEFAULT_OUT_JSON,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    fp32_info = _check_fp32_locked()
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    core = get_core()
    torch_eng = TorchCrystalEngine(device=device)
    n_eff = min(2, n_cases) if fast else int(n_cases)

    print(
        f"[diag_r64_wormhole] 启动 | device={device} | fp32_locked={fp32_info['fp32_locked']} | "
        f"n_cases={n_eff} | sims={sims} | fast={fast}",
        flush=True,
    )

    model_m3, p3 = load_v10_checkpoint(ckpt_path, device=device)
    model_m2, p2 = load_v10_checkpoint(m2_ckpt_path, device=device)
    w64 = _sample_aligned_r64_cases(seed, n_eff)
    cr64 = core.compress_exact(w64, rounds=64)

    # A. 基线复现 (best_m3v2.pt, N=n_eff, sims=51)
    print(f"[diag_r64_wormhole] (A) 基线逐模拟追踪 (best_m3v2.pt, N={n_eff}, sims={sims})...", flush=True)
    mcts_base = InstrumentedWormholeMCTS(
        model=model_m3, torch_eng=torch_eng, core=core, device=device,
        max_branch_candidates=16, use_gf2_root=False, gf2_interval=4, enable_macro_actions=True,
    )
    baseline_cases: List[Dict[str, Any]] = []
    first_case_trace: Optional[Dict[str, Any]] = None
    first_root_verify: Optional[Dict[str, Any]] = None

    for idx in range(n_eff):
        res = mcts_base.search(
            H_target=cr64.H_target[idx], rounds=64, fixed_words=None,
            num_simulations=sims, time_limit_sec=time_limit_sec, focus_splice=True,
        )
        tr = mcts_base.last_trace
        rv = _verify_root_actions_structure(tr["root_actions"])
        if idx == 0:
            first_root_verify = rv
            first_case_trace = {
                "peak_unk_drop_at_exp": tr["peak_unk_drop_at_exp"],
                "peak_unk_drop_path": tr["peak_unk_drop_path"],
                "max_depth_reached": tr["max_depth_reached"],
                "sim_trace": tr["sim_trace"],
            }
        unk_drop = int(res.root_unknown - res.min_unknown_reached)
        gate_drop = int(res.root_active_carry_gates - res.min_active_carry_gates)
        splice_drop = int(res.root_splice_freedom - res.min_splice_freedom)
        assert unk_drop == 94 and gate_drop == 26 and splice_drop == 8 and res.pruned_conflicts == 0, (
            f"case {idx} 指标偏离预期 94/26/8/0: {unk_drop}/{gate_drop}/{splice_drop}/{res.pruned_conflicts}"
        )
        baseline_cases.append(
            {
                "case": idx,
                "solved": bool(res.solved),
                "simulations": int(res.simulations),
                "expansions": int(res.expansions),
                "root_unknown": int(res.root_unknown),
                "min_unknown_reached": int(res.min_unknown_reached),
                "unk_drop": unk_drop,
                "root_active_carry_gates": int(res.root_active_carry_gates),
                "min_active_carry_gates": int(res.min_active_carry_gates),
                "gate_drop": gate_drop,
                "root_splice_freedom": int(res.root_splice_freedom),
                "min_splice_freedom": int(res.min_splice_freedom),
                "splice_drop": splice_drop,
                "pruned_conflicts": int(res.pruned_conflicts),
                "max_depth_reached": int(tr["max_depth_reached"]),
                "peak_unk_drop_at_exp": int(tr["peak_unk_drop_at_exp"]),
                "peak_unk_drop_path": tr["peak_unk_drop_path"],
                "w_macro_target_word": rv["w_macro_target_word"],
                "elapsed_sec": round(float(res.elapsed_sec), 3),
            }
        )
        print(
            f"  [best_m3v2 case {idx}] root_unk={res.root_unknown} -> drop={unk_drop}/{gate_drop}/{splice_drop} | "
            f"pruned={res.pruned_conflicts} | peak@exp={tr['peak_unk_drop_at_exp']} | "
            f"w_macro=W[{rv['w_macro_target_word']}] | {res.elapsed_sec:.2f}s",
            flush=True,
        )

    # A2. M2 step4000 权重对照（解释 m3v2 step0 80/16/16 vs best 94/26/8）
    print("[diag_r64_wormhole] (A2) M2 step4000 权重对照 (case 0)...", flush=True)
    mcts_m2 = InstrumentedWormholeMCTS(
        model=model_m2, torch_eng=torch_eng, core=core, device=device,
        max_branch_candidates=16, use_gf2_root=False, gf2_interval=4, enable_macro_actions=True,
    )
    res_m2 = mcts_m2.search(
        H_target=cr64.H_target[0], rounds=64, fixed_words=None,
        num_simulations=sims, time_limit_sec=time_limit_sec, focus_splice=True,
    )
    tr_m2 = mcts_m2.last_trace
    rv_m2 = _verify_root_actions_structure(tr_m2["root_actions"])
    m2_comp = {
        "ckpt_step": int(p2.get("step", -1)),
        "root_unknown": int(res_m2.root_unknown),
        "unk_drop": int(res_m2.root_unknown - res_m2.min_unknown_reached),
        "gate_drop": int(res_m2.root_active_carry_gates - res_m2.min_active_carry_gates),
        "splice_drop": int(res_m2.root_splice_freedom - res_m2.min_splice_freedom),
        "pruned_conflicts": int(res_m2.pruned_conflicts),
        "w_macro_target_word": rv_m2["w_macro_target_word"],
        "peak_unk_drop_at_exp": int(tr_m2["peak_unk_drop_at_exp"]),
        "peak_unk_drop_path": tr_m2["peak_unk_drop_path"],
    }
    print(
        f"  [M2 step4000 case 0] drop={m2_comp['unk_drop']}/{m2_comp['gate_drop']}/{m2_comp['splice_drop']} | "
        f"w_macro=W[{m2_comp['w_macro_target_word']}] | path={m2_comp['peak_unk_drop_path']}",
        flush=True,
    )

    # B. 反例 1：绕过 rounds<=28 硬编码限制强开 R=64 GF(2)
    print("[diag_r64_wormhole] (B) 反例 1: 强开 R=64 根/子 GF(2) (time_limit=3.0s)...", flush=True)
    env_gf2_root = GeneralizedSymbolEnv(H_target=cr64.H_target[0], rounds=64, splice_round=32, core=core)
    t_g0 = time.perf_counter()
    step_g = env_gf2_root.propagate(use_gf2=True, max_iters=8)
    dt_g = time.perf_counter() - t_g0

    mcts_gf2 = InstrumentedWormholeMCTS(
        model=model_m3, torch_eng=torch_eng, core=core, device=device,
        max_branch_candidates=16, use_gf2_root=True, gf2_interval=4,
        enable_macro_actions=True, force_gf2_child=True,
    )
    res_gf2 = mcts_gf2.search(
        H_target=cr64.H_target[0], rounds=64, fixed_words=None,
        num_simulations=sims, time_limit_sec=3.0, focus_splice=True,
    )
    tr_gf2 = mcts_gf2.last_trace
    counterexample_force_gf2 = {
        "root_single_call": {
            "elapsed_sec": round(dt_g, 4),
            "gf2_rank": int(step_g.gf2_rank),
            "unit_prop_deductions": int(step_g.unit_prop_deductions),
            "gf2_deductions": int(step_g.gf2_deductions),
            "unknown_after": int(step_g.unknown_after),
            "conflict": bool(step_g.conflict),
        },
        "mcts_3s_budget": {
            "simulations": int(res_gf2.simulations),
            "expansions": int(res_gf2.expansions),
            "unk_drop": int(res_gf2.root_unknown - res_gf2.min_unknown_reached),
            "gate_drop": int(res_gf2.root_active_carry_gates - res_gf2.min_active_carry_gates),
            "splice_drop": int(res_gf2.root_splice_freedom - res_gf2.min_splice_freedom),
            "pruned_conflicts": int(res_gf2.pruned_conflicts),
            "gf2_child_calls": int(tr_gf2["gf2_child_calls"]),
            "gf2_child_deductions_total": int(tr_gf2["gf2_child_deductions_total"]),
            "max_depth_reached": int(tr_gf2["max_depth_reached"]),
            "elapsed_sec": round(float(res_gf2.elapsed_sec), 3),
        },
    }
    print(
        f"  [强开 GF2] root_gf2 dt={dt_g:.3f}s, rank={step_g.gf2_rank}, gf2_ded={step_g.gf2_deductions} | "
        f"3s_MCTS sims={res_gf2.simulations}, unk_drop={counterexample_force_gf2['mcts_3s_budget']['unk_drop']}, "
        f"pruned={res_gf2.pruned_conflicts}",
        flush=True,
    )

    # C. 反例 2：关闭固定 splice_block 与全关 enable_macro_actions=False (sims=51)
    print("[diag_r64_wormhole] (C) 反例 2: 剥离固定 splice_block 与关闭 enable_macro_actions...", flush=True)
    mcts_no_sb = InstrumentedWormholeMCTS(
        model=model_m3, torch_eng=torch_eng, core=core, device=device,
        max_branch_candidates=16, use_gf2_root=False, gf2_interval=4,
        enable_macro_actions=True, disable_splice_block_only=True,
    )
    res_no_sb = mcts_no_sb.search(
        H_target=cr64.H_target[0], rounds=64, fixed_words=None,
        num_simulations=sims, time_limit_sec=time_limit_sec, focus_splice=True,
    )
    tr_no_sb = mcts_no_sb.last_trace

    mcts_no_macro = InstrumentedWormholeMCTS(
        model=model_m3, torch_eng=torch_eng, core=core, device=device,
        max_branch_candidates=16, use_gf2_root=False, gf2_interval=4,
        enable_macro_actions=False,
    )
    res_no_macro = mcts_no_macro.search(
        H_target=cr64.H_target[0], rounds=64, fixed_words=None,
        num_simulations=sims, time_limit_sec=time_limit_sec, focus_splice=True,
    )
    tr_no_macro = mcts_no_macro.last_trace

    counterexample_macro = {
        "disable_splice_block_only": {
            "root_actions_count": int(tr_no_sb["root_actions_count"]),
            "simulations": int(res_no_sb.simulations),
            "expansions": int(res_no_sb.expansions),
            "unk_drop": int(res_no_sb.root_unknown - res_no_sb.min_unknown_reached),
            "gate_drop": int(res_no_sb.root_active_carry_gates - res_no_sb.min_active_carry_gates),
            "splice_drop": int(res_no_sb.root_splice_freedom - res_no_sb.min_splice_freedom),
            "pruned_conflicts": int(res_no_sb.pruned_conflicts),
            "max_depth_reached": int(tr_no_sb["max_depth_reached"]),
            "peak_unk_drop_at_exp": int(tr_no_sb["peak_unk_drop_at_exp"]),
            "peak_unk_drop_path": tr_no_sb["peak_unk_drop_path"],
        },
        "disable_all_macro_actions": {
            "root_actions_count": int(tr_no_macro["root_actions_count"]),
            "simulations": int(res_no_macro.simulations),
            "expansions": int(res_no_macro.expansions),
            "unk_drop": int(res_no_macro.root_unknown - res_no_macro.min_unknown_reached),
            "gate_drop": int(res_no_macro.root_active_carry_gates - res_no_macro.min_active_carry_gates),
            "splice_drop": int(res_no_macro.root_splice_freedom - res_no_macro.min_splice_freedom),
            "pruned_conflicts": int(res_no_macro.pruned_conflicts),
            "max_depth_reached": int(tr_no_macro["max_depth_reached"]),
            "peak_unk_drop_at_exp": int(tr_no_macro["peak_unk_drop_at_exp"]),
            "peak_unk_drop_path": tr_no_macro["peak_unk_drop_path"],
        },
    }
    print(
        f"  [No splice_block] drop={counterexample_macro['disable_splice_block_only']['unk_drop']}/"
        f"{counterexample_macro['disable_splice_block_only']['gate_drop']}/"
        f"{counterexample_macro['disable_splice_block_only']['splice_drop']}, "
        f"depth={tr_no_sb['max_depth_reached']} | "
        f"[No macro] drop={counterexample_macro['disable_all_macro_actions']['unk_drop']}/"
        f"{counterexample_macro['disable_all_macro_actions']['gate_drop']}/"
        f"{counterexample_macro['disable_all_macro_actions']['splice_drop']}, "
        f"depth={tr_no_macro['max_depth_reached']}",
        flush=True,
    )

    elapsed = round(time.perf_counter() - t0, 3)
    report: Dict[str, Any] = {
        "diagnostic": "r64_wormhole_stall_readonly",
        "fp32_check": fp32_info,
        "device": str(device),
        "m3_ckpt": ckpt_path,
        "m3_ckpt_step": int(p3.get("step", -1)),
        "m2_ckpt": m2_ckpt_path,
        "seed": int(seed),
        "n_cases": n_eff,
        "sims": int(sims),
        "fast_mode": bool(fast),
        "elapsed_sec": elapsed,
        "root_actions_verification": first_root_verify,
        "baseline_best_m3v2_cases": baseline_cases,
        "case0_51_sim_trace": first_case_trace,
        "m2_step4000_comparison": m2_comp,
        "counterexample_force_gf2": counterexample_force_gf2,
        "counterexample_disable_macro_and_splice_block": counterexample_macro,
    }
    _atomic_write_json(Path(out_json), report)
    print(f"[diag_r64_wormhole] 完成，耗时 {elapsed:.2f}s | 写入 {out_json}", flush=True)
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="R2 只读诊断 4：原生 R=64 虫洞 MCTS 停滞根因与反例")
    p.add_argument("--ckpt_path", type=str, default=DEFAULT_M3V2_CKPT)
    p.add_argument("--m2_ckpt_path", type=str, default=DEFAULT_M2_CKPT)
    p.add_argument("--n_cases", type=int, default=4)
    p.add_argument("--sims", type=int, default=51)
    p.add_argument("--time_limit_sec", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=20260499)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_diag_r64_wormhole_stall(
        ckpt_path=args.ckpt_path,
        m2_ckpt_path=args.m2_ckpt_path,
        n_cases=args.n_cases,
        sims=args.sims,
        time_limit_sec=args.time_limit_sec,
        seed=args.seed,
        fast=bool(args.fast or args.quick),
        cpu=args.cpu,
        out_json=args.out_json,
    )
