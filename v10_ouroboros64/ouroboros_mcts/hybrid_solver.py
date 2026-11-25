"""
Phase-SHA V10「衔尾蛇-64」Milestone 3：神经-符号三层合龙混合求解器 (OuroborosHybridSolver)。

求解流水线：
  层 1（双端焊死 + 双平面精确闭包）：ARX 单元传播 + GF(2) 增量高斯消元；
  层 2（神经引导虫洞 MCTS）：加载 M2 Ouroboros64Net 权重，经 PUCT 发现高坍缩路径、
        精确剪枝矛盾分支并导出根节点可证明强制单子句 (learned_units) 与相位偏好 (phase_lits)；
  层 3（精确 CDCL 终局仲裁）：将 MCTS 单子句与神经相位注入 CaDiCaL-195 执行精确求解，
        每例 SAT 强制经 OuroborosCore.verify_preimage 逐比特全量验真（零伪解、零泄漏）。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pysat.solvers import Solver

from ..ouroboros_core.cnf_cdcl import SHA256CNFEncoder
from ..ouroboros_core.engine import (
    SYM_1,
    OuroborosCore,
    TorchCrystalEngine,
    enforce_fp32_determinism,
    get_core,
)
from ..ouroboros_net.model import Ouroboros64Net, get_model_preset
from .wormhole_mcts import MCTSSearchResult, WormholeMCTS


def load_v10_checkpoint(
    ckpt_path: str | Path,
    device: Optional[torch.device | str] = None,
) -> Tuple[Ouroboros64Net, Dict[str, Any]]:
    """从 Milestone 2 检查点加载纯 FP32 Ouroboros64Net 权重。"""
    enforce_fp32_determinism()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    cfg = payload.get("preset_cfg") or get_model_preset(payload.get("preset", "a100"))
    cfg = dict(cfg)
    cfg["use_checkpoint"] = False  # 推理期间关闭重算检查点
    model = Ouroboros64Net(**cfg).to(device=dev, dtype=torch.float32)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


@dataclass
class HybridSolveResult:
    """混合求解器单样例评测结果（绝对零内部泄漏 + 独立 verify_preimage 核验）。"""

    rounds: int
    status: str
    valid_preimage: bool
    method: str
    conflicts: int
    decisions: int
    elapsed_sec: float
    mcts_time_sec: float
    sat_time_sec: float
    mcts_units_injected: int
    mcts_pruned_conflicts: int
    w0_15: Optional[np.ndarray] = None

    def to_metrics_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("w0_15", None)
        return d


class OuroborosHybridSolver:
    """
    Phase-SHA V10 神经-符号三层混合盲求逆求解器。
    """

    def __init__(
        self,
        model: Optional[Ouroboros64Net] = None,
        core: Optional[OuroborosCore] = None,
        device: Optional[torch.device | str] = None,
        solver_name: str = "cadical195",
        c_puct: float = 1.35,
    ) -> None:
        self.core = core or get_core()
        self.model = model
        self.device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters()).device if model is not None else torch.device("cpu"))
        )
        self.torch_eng = TorchCrystalEngine(device=self.device) if model is not None else None
        self.solver_name = solver_name
        self.mcts = WormholeMCTS(
            model=self.model,
            torch_eng=self.torch_eng,
            core=self.core,
            device=self.device,
            c_puct=c_puct,
            max_branch_candidates=24,
            prop_max_iters=8,
        )

    def solve(
        self,
        H_target: np.ndarray,
        rounds: int = 20,
        fixed_words: Optional[Dict[int, int]] = None,
        mcts_sims: int = 24,
        mcts_time_limit: float = 0.8,
        total_time_limit: float = 5.0,
        conflict_budget: int = 250000,
        fixed_bits: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> HybridSolveResult:
        """
        对给定目标摘要 H_target 执行严格零泄漏混合求逆。
        """
        t_all = time.perf_counter()
        ht_np = np.asarray(H_target, dtype=np.uint32).reshape(8)

        # 第一/二层：神经引导虫洞 MCTS + 双速双平面精确传播
        mcts_res: MCTSSearchResult = self.mcts.search(
            H_target=ht_np,
            rounds=rounds,
            fixed_words=fixed_words,
            num_simulations=mcts_sims,
            time_limit_sec=min(mcts_time_limit, total_time_limit * 0.4),
            focus_splice=False,
            fixed_bits=fixed_bits,
        )

        if mcts_res.solved and mcts_res.w0_15 is not None:
            ok_v = bool(self.core.verify_preimage(mcts_res.w0_15, ht_np, rounds=rounds)[0])
            if ok_v:
                return HybridSolveResult(
                    rounds=rounds,
                    status="SAT",
                    valid_preimage=True,
                    method="MCTS_DIRECT",
                    conflicts=0,
                    decisions=mcts_res.expansions,
                    elapsed_sec=time.perf_counter() - t_all,
                    mcts_time_sec=mcts_res.elapsed_sec,
                    sat_time_sec=0.0,
                    mcts_units_injected=len(mcts_res.learned_units),
                    mcts_pruned_conflicts=mcts_res.pruned_conflicts,
                    w0_15=mcts_res.w0_15,
                )

        # 第三层：构造精确 CNF，注入 MCTS 单子句、二元冲突子句与神经条件相位交由 CaDiCaL-195 仲裁
        t_sat0 = time.perf_counter()
        enc = SHA256CNFEncoder(rounds=rounds).build(
            H_target=ht_np,
            fixed_words=fixed_words,
            fixed_bits=fixed_bits,
        )
        clauses = list(enc.clauses)

        injected_cnt = 0
        for kind, t_wi, _w, bb, sym in mcts_res.learned_units:
            if kind == 0 and 0 <= t_wi < 16:
                lit_v = enc.W_vars[t_wi][bb]
                if not isinstance(lit_v, bool) and isinstance(lit_v, int):
                    signed = lit_v if sym == SYM_1 else -lit_v
                    clauses.append([signed])
                    injected_cnt += 1

        # 注入 MCTS 深度-2 触发数学矛盾导出之严密二元子句
        for cl_items in getattr(mcts_res, "learned_clauses", ()):
            cnf_lits: List[int] = []
            tautology = False
            for kind, t_wi, _w, bb, sym in cl_items:
                if kind != 0 or not (0 <= t_wi < 16):
                    tautology = True
                    break
                lit_v = enc.W_vars[t_wi][bb]
                if isinstance(lit_v, bool):
                    if (lit_v and sym == SYM_1) or ((not lit_v) and sym != SYM_1):
                        tautology = True
                        break
                elif isinstance(lit_v, int):
                    cnf_lits.append(lit_v if sym == SYM_1 else -lit_v)
            if (not tautology) and cnf_lits:
                clauses.append(cnf_lits)

        # 构造排好序的高置信相位偏好列表
        scored_lits: List[Tuple[float, int]] = []
        for (wi, bb), (conf, pref_bit) in mcts_res.phase_preferences.items():
            lit_v = enc.W_vars[wi][bb]
            if not isinstance(lit_v, bool) and isinstance(lit_v, int):
                signed = lit_v if pref_bit == 1 else -lit_v
                scored_lits.append((conf, signed))
        scored_lits.sort(key=lambda x: x[0], reverse=True)
        phase_lits = [lit for _, lit in scored_lits[:256]]

        rem_time = max(0.2, float(total_time_limit) - (time.perf_counter() - t_all))
        res_state: Optional[bool] = None
        w0_15: Optional[np.ndarray] = None
        valid_preimage = False
        stats: Dict[str, Any] = {}

        with Solver(name=self.solver_name, bootstrap_with=clauses) as solver:
            if phase_lits:
                try:
                    solver.set_phases(literals=phase_lits)
                except Exception:
                    pass

            chunk_confl = 4000
            while True:
                if (time.perf_counter() - t_sat0) >= rem_time:
                    break
                cur_stats = solver.accum_stats() if hasattr(solver, "accum_stats") else {}
                used_confl = int(cur_stats.get("conflicts", 0))
                if conflict_budget > 0:
                    rem = int(conflict_budget) - used_confl
                    if rem <= 0:
                        break
                    cur_budget = min(chunk_confl, rem)
                else:
                    cur_budget = chunk_confl

                solver.conf_budget(cur_budget)
                res_state = solver.solve_limited()
                if res_state is not None:
                    break
                if (time.perf_counter() - t_sat0) >= rem_time:
                    break

            stats = solver.accum_stats() if hasattr(solver, "accum_stats") else {}
            if res_state is True:
                model_lits = solver.get_model() or []
                model_set = {abs(lit): (lit > 0) for lit in model_lits}
                w_rec = np.zeros((16,), dtype=np.uint32)
                for wi in range(16):
                    word_val = 0
                    for b in range(32):
                        lit = enc.W_vars[wi][b]
                        bit = int(lit) if isinstance(lit, bool) else int(model_set.get(abs(int(lit)), False))
                        if not isinstance(lit, bool) and int(lit) < 0:
                            bit = 1 - bit
                        if bit:
                            word_val |= (1 << b)
                    w_rec[wi] = np.uint32(word_val)
                w0_15 = w_rec
                valid_preimage = bool(self.core.verify_preimage(w0_15, ht_np, rounds=rounds)[0])
                assert valid_preimage, f"OuroborosHybridSolver 发现未通过 verify_preimage 的解 R={rounds}"

        dt_sat = time.perf_counter() - t_sat0
        dt_all = time.perf_counter() - t_all
        status_str = "SAT" if res_state is True else ("UNSAT" if res_state is False else "TIMEOUT_OR_BUDGET")
        return HybridSolveResult(
            rounds=rounds,
            status=status_str,
            valid_preimage=valid_preimage,
            method="MCTS+CDCL_HYBRID",
            conflicts=int(stats.get("conflicts", 0)),
            decisions=int(stats.get("decisions", 0)),
            elapsed_sec=dt_all,
            mcts_time_sec=mcts_res.elapsed_sec,
            sat_time_sec=dt_sat,
            mcts_units_injected=injected_cnt,
            mcts_pruned_conflicts=mcts_res.pruned_conflicts,
            w0_15=w0_15,
        )
