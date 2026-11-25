"""
Phase-SHA V10「衔尾蛇-64」Milestone 3-V2：搜中动态神经介入引擎 (dynamic_cdcl.py)。

彻底摆脱 M3-V1「仅启动前注入一次静态 phase_lits」的局限，构建两套搜中闭环求解器：
  1. DynamicNeuralCDCLSolver（持久化 CaDiCaL-195 + 搜中 FLP 前瞻单子句 + 动态重启重评 + get_core 子空间剪枝）：
     - 全程复用单个增量 Solver 实例（跨重启 100% 保留 CDCL 学习子句）；
     - 搜中暂停调用 propagate() 回收已证明变量，与 conflict_heatmap 一并回灌 Ouroboros64Net 重算相位/瓶颈排序；
     - 对 Top 瓶颈变量执行毫秒级前瞻 Failed-Literal Probing，确证矛盾即刻写入永久单位子句；
     - 在超过基线阈值的难例尾端触发浅层 Cube-and-Conquer + get_core() 极小矛盾核剪枝。
  2. DynamicNeuralXORSolver（CryptoMiniSat-5 原生 XOR-Gauss + 搜中神经动态 Cube 与冲突学习）。

所有返回 SAT 解强制经 OuroborosCore.verify_preimage 逐比特验真，严格零内部泄漏。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from pysat.solvers import Solver

from ..ouroboros_core.cnf_cdcl import (
    CryptoMiniSatXORSolver,
    SHA256CNFEncoder,
    SHA256XORCNFEncoder,
    extract_w0_15_from_cryptosat,
)
from ..ouroboros_core.engine import (
    C_PLANES,
    IV_WORDS,
    MASK32,
    OuroborosCore,
    TorchCrystalEngine,
    get_core,
)
from ..ouroboros_net.heads import PLANE_COUNT
from ..ouroboros_net.model import Ouroboros64Net


@dataclass
class DynamicSolveResult:
    """搜中动态神经-符号求解器单样例结果。"""

    rounds: int
    solver_type: str
    status: str
    valid_preimage: bool
    conflicts: int
    decisions: int
    elapsed_sec: float
    episodes: int
    flp_units_proven: int
    cores_pruned: int
    min_core_len: int
    w0_15: Optional[np.ndarray] = None

    def to_metrics_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("w0_15", None)
        return d


def _build_var_map_and_known_mask(
    enc: SHA256CNFEncoder,
) -> Tuple[Dict[int, Tuple[int, int, int]], np.ndarray]:
    """
    从 CNF 变量表构建 DIMACS var_id -> (wi, bb, sign_in_w) 反向映射，
    以及已被常量折叠固定的比特掩码 (16, 32)。
    """
    var_to_coord: Dict[int, Tuple[int, int, int]] = {}
    const_fixed = np.zeros((16, 32), dtype=bool)
    const_vals = np.zeros((16, 32), dtype=np.uint8)
    for wi in range(16):
        for bb in range(32):
            lit = enc.W_vars[wi][bb]
            if isinstance(lit, bool):
                const_fixed[wi, bb] = True
                const_vals[wi, bb] = 1 if lit else 0
            else:
                v = abs(int(lit))
                s_in_w = 1 if int(lit) > 0 else -1
                var_to_coord[v] = (wi, bb, s_in_w)
    return var_to_coord, const_fixed


def _extract_cadical_w0_15(enc: SHA256CNFEncoder, model_lits: Sequence[int]) -> np.ndarray:
    model_set = {abs(int(lit)): (int(lit) > 0) for lit in model_lits}
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
    return w_rec


class DynamicNeuralCDCLSolver:
    """
    持久化 CaDiCaL-195 + 搜中神经动态重评 + Failed-Literal Probing + Cube-Core 剪枝。
    """

    def __init__(
        self,
        model: Optional[Ouroboros64Net] = None,
        core: Optional[OuroborosCore] = None,
        device: Optional[torch.device | str] = None,
        solver_name: str = "cadical195",
        enable_flp: bool = True,
        enable_cubes: bool = True,
        flp_top_k: int = 8,
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
        self.enable_flp = bool(enable_flp)
        self.enable_cubes = bool(enable_cubes)
        self.flp_top_k = int(flp_top_k)

    @torch.no_grad()
    def _neural_score_partial(
        self,
        H_single: np.ndarray,
        rounds: int,
        assigned_w_bits: Dict[Tuple[int, int], int],
        conflict_heatmap: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """根据搜中时刻已确定的部分 W 位与动态 conflict_heatmap 获取瓶颈度与相位概率。"""
        if self.model is None or self.torch_eng is None:
            coll = np.zeros((16, 32), dtype=np.float32)
            for wi in range(16):
                coll[wi, :] = 1.0 / (1.0 + 0.15 * wi)
            return coll, np.zeros((16, 32), dtype=np.float32)

        was_train = self.model.training
        self.model.eval()
        W_b = np.zeros((1, rounds), dtype=np.uint32)
        S_b = np.zeros((1, rounds + 1, 8), dtype=np.uint32)
        C_b = np.zeros((1, rounds, C_PLANES), dtype=np.uint32)
        s0 = np.asarray(IV_WORDS, dtype=np.uint32)
        sr = ((H_single.astype(np.uint64) - s0.astype(np.uint64)) & MASK32).astype(np.uint32)
        S_b[0, 0] = s0
        S_b[0, rounds] = sr

        unk = torch.ones((1, rounds, PLANE_COUNT, 32), dtype=torch.float32, device=self.device)
        unk[:, 0, 1:9, :] = 0.0
        for (wi, bb), bval in assigned_w_bits.items():
            if 0 <= wi < rounds and 0 <= bb < 32:
                if bval & 1:
                    W_b[0, wi] |= np.uint32(1 << bb)
                else:
                    W_b[0, wi] &= np.uint32(~(1 << bb) & MASK32)
                unk[0, wi, 0, bb] = 0.0

        conf_t = torch.from_numpy(
            np.log1p(np.maximum(0.0, conflict_heatmap[:rounds]))[None, ...].astype(np.float32)
        ).to(self.device)

        out = self.model(
            torch.from_numpy(H_single[None, :].astype(np.int64)).to(self.device),
            torch.from_numpy(W_b.astype(np.int64)).to(self.device),
            torch.from_numpy(S_b.astype(np.int64)).to(self.device),
            torch.from_numpy(C_b.astype(np.int64)).to(self.device),
            rounds=rounds,
            unknown_mask=unk,
            conflict_heatmap=conf_t,
            zero_defects=True,
            propose_clusters=False,
            return_entropy=False,
            crystal_engine=self.torch_eng,
        )
        coll_w = out["collapse_score"][0, :16, 0, :].cpu().numpy()
        phase_w = out["phase_logits"][0, :16, 0, :].cpu().numpy()
        if was_train:
            self.model.train()
        return coll_w, phase_w

    @staticmethod
    def _stamp_heat(
        heatmap: np.ndarray,
        wi: int,
        bb: int,
        rounds: int,
        weight: float = 1.0,
    ) -> None:
        if 0 <= wi < rounds and 0 <= bb < 32:
            heatmap[wi, 0, bb] += float(weight)
            for db in (-1, 1):
                nb = bb + db
                if 0 <= nb < 32:
                    heatmap[wi, 0, nb] += 0.35 * float(weight)

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
        """
        对目标摘要 H_target 执行绝对零泄漏、搜中动态神经介入 CDCL 求解：
          - Episode 1 执行 1-bit FLP + 高置信前 64 比特初始化（不盲目干扰 CaDiCaL 其余相位置换）；
          - 若超过 initial_chunk 仍未解出，每轮搜中同步 propagate() 新推导单子句、回灌 conflict_heatmap
            重跑神经骨干、执行 1-bit + Top-Pair 2-bit 前瞻 FLP 学习精确二元子句、仅微调 Top-16 瓶颈变量极性；
          - 在 conflicts >= 7000 的难例尾端再启用浅层 Cube-and-Conquer + get_core 剪枝，杜绝浅猜浪费冲突预算。
        """
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
            u = int(wv) & MASK32
            for bb in range(32):
                base_assigned[(int(wi), bb)] = (u >> bb) & 1
        for (wi, bb), bv in (fixed_bits or {}).items():
            base_assigned[(int(wi), int(bb))] = int(bv) & 1

        conflict_heatmap = np.zeros((rounds, PLANE_COUNT, 32), dtype=np.float32)
        episodes = 0
        flp_units_proven = 0
        cores_pruned = 0
        min_core_len = 0
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

                # 1) 同步搜中级 0 层已推导单位文字
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

                # 2) 动态神经前向重评（融合最新推导 + 动态 conflict_heatmap）
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

                # 3) 搜中毫秒级 1-bit + 2-bit Pairwise Failed-Literal Probing + Lookahead
                rescored: List[Tuple[float, int, int, int, float]] = []
                if self.enable_flp and free_candidates:
                    n_base = len(prop_lits)
                    k_1bit = max(self.flp_top_k, 12 if episodes >= 2 else self.flp_top_k)
                    surviving_top: List[Tuple[int, int, int]] = []
                    for idx_c, (sc, lit_v, wi, bb, logit) in enumerate(free_candidates):
                        if idx_c < k_1bit:
                            ok_p, lp = solver.propagate(assumptions=[lit_v])
                            ok_n, ln = solver.propagate(assumptions=[-lit_v])
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

                    # 在难例中场 (episodes >= 2) 对 Top-5 瓶颈变量配对做 2-bit FLP，免费学入精确二元阻断子句！
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

                # 4) 非破坏性选择性极性注入：Episode 1 注入高置信前置序列；Episode>=2 仅更新 Top-16 瓶颈变量极性（绝不冲刷 CaDiCaL 已保存相位）
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
                    except Exception:
                        pass

                # 5) 仅在 conflicts >= 7000 的顽固难例尾端触发浅层 Cube-and-Conquer + get_core 剪枝
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
                            _, _, wi0, bb0, _ = rescored[0]
                            self._stamp_heat(conflict_heatmap, wi0, bb0, rounds, weight=0.6)

                # 6) 无假设全空间 CDCL 冲刺一个动态窗口（保留全部历史子句 + FLP 单位子句 + 2-bit/core 阻断子句）
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
                assert valid_preimage, f"DynamicNeuralCDCLSolver 发现未通过 verify_preimage 的解 R={rounds}"

        dt = time.perf_counter() - t0
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


class DynamicNeuralXORSolver:
    """
    CryptoMiniSat-5 原生 XOR-Gauss + 搜中神经动态 Cube / get_conflict() 阻断子句闭环求解器。
    """

    def __init__(
        self,
        model: Optional[Ouroboros64Net] = None,
        core: Optional[OuroborosCore] = None,
        device: Optional[torch.device | str] = None,
        threads: int = 1,
    ) -> None:
        self.core = core or get_core()
        self.model = model
        self.device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters()).device if model is not None else torch.device("cpu"))
        )
        self.torch_eng = TorchCrystalEngine(device=self.device) if model is not None else None
        self.xor_solver = CryptoMiniSatXORSolver(threads=threads)

    @torch.no_grad()
    def _neural_rank(
        self,
        H_single: np.ndarray,
        rounds: int,
        base_assigned: Dict[Tuple[int, int], int],
        conflict_heatmap: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.torch_eng is None:
            coll = np.zeros((16, 32), dtype=np.float32)
            for wi in range(16):
                coll[wi, :] = 1.0 / (1.0 + 0.15 * wi)
            return coll, np.zeros((16, 32), dtype=np.float32)

        was_train = self.model.training
        self.model.eval()
        W_b = np.zeros((1, rounds), dtype=np.uint32)
        S_b = np.zeros((1, rounds + 1, 8), dtype=np.uint32)
        C_b = np.zeros((1, rounds, C_PLANES), dtype=np.uint32)
        s0 = np.asarray(IV_WORDS, dtype=np.uint32)
        sr = ((H_single.astype(np.uint64) - s0.astype(np.uint64)) & MASK32).astype(np.uint32)
        S_b[0, 0] = s0
        S_b[0, rounds] = sr

        unk = torch.ones((1, rounds, PLANE_COUNT, 32), dtype=torch.float32, device=self.device)
        unk[:, 0, 1:9, :] = 0.0
        for (wi, bb), bval in base_assigned.items():
            if 0 <= wi < rounds and 0 <= bb < 32:
                if bval & 1:
                    W_b[0, wi] |= np.uint32(1 << bb)
                else:
                    W_b[0, wi] &= np.uint32(~(1 << bb) & MASK32)
                unk[0, wi, 0, bb] = 0.0

        conf_t = torch.from_numpy(
            np.log1p(np.maximum(0.0, conflict_heatmap[:rounds]))[None, ...].astype(np.float32)
        ).to(self.device)
        out = self.model(
            torch.from_numpy(H_single[None, :].astype(np.int64)).to(self.device),
            torch.from_numpy(W_b.astype(np.int64)).to(self.device),
            torch.from_numpy(S_b.astype(np.int64)).to(self.device),
            torch.from_numpy(C_b.astype(np.int64)).to(self.device),
            rounds=rounds,
            unknown_mask=unk,
            conflict_heatmap=conf_t,
            zero_defects=True,
            propose_clusters=False,
            return_entropy=False,
            crystal_engine=self.torch_eng,
        )
        coll_w = out["collapse_score"][0, :16, 0, :].cpu().numpy()
        phase_w = out["phase_logits"][0, :16, 0, :].cpu().numpy()
        if was_train:
            self.model.train()
        return coll_w, phase_w

    def solve(
        self,
        H_target: Sequence[int],
        rounds: int = 24,
        fixed_words: Optional[Dict[int, int]] = None,
        fixed_bits: Optional[Dict[Tuple[int, int], int]] = None,
        extra_clauses: Optional[Sequence[Sequence[int]]] = None,
        time_limit_sec: float = 6.0,
        cube_depth: int = 2,
    ) -> DynamicSolveResult:
        t0 = time.perf_counter()
        ht_np = np.asarray(H_target, dtype=np.uint32).reshape(8)
        enc = SHA256XORCNFEncoder(rounds=rounds).build(
            H_target=ht_np, fixed_words=fixed_words, fixed_bits=fixed_bits
        )
        solver = self.xor_solver.new_solver(enc, extra_clauses=extra_clauses)
        var_to_coord, const_fixed = _build_var_map_and_known_mask(enc)
        base_assigned: Dict[Tuple[int, int], int] = {}
        for wi, wv in (fixed_words or {}).items():
            u = int(wv) & MASK32
            for bb in range(32):
                base_assigned[(int(wi), bb)] = (u >> bb) & 1
        for (wi, bb), bv in (fixed_bits or {}).items():
            base_assigned[(int(wi), int(bb))] = int(bv) & 1

        episodes = 0
        cores_pruned = 0
        min_core_len = 0
        sat_final: Optional[bool] = None
        sol_final: Optional[Sequence[Optional[bool]]] = None

        # 首回合：先执行短时间无假设原生 XOR-Gauss 求解，不拖累可秒解案例
        init_tl = min(0.6, max(0.15, time_limit_sec * 0.25))
        sat0, sol0 = solver.solve(confl_limit=2000, time_limit=init_tl)
        if sat0 is not None:
            sat_final, sol_final = sat0, sol0
            episodes = 1
        else:
            conflict_heatmap = np.zeros((rounds, PLANE_COUNT, 32), dtype=np.float32)
            tried_cubes = set()
            burst_confl = 3000

            while (time.perf_counter() - t0) < time_limit_sec:
                episodes += 1
                coll_w, phase_w = self._neural_rank(ht_np, rounds, base_assigned, conflict_heatmap)
                cands: List[Tuple[float, int, int, int, float]] = []
                for wi in range(16):
                    for bb in range(32):
                        if const_fixed[wi, bb] or (wi, bb) in base_assigned:
                            continue
                        lit_v = enc.W_vars[wi][bb]
                        if isinstance(lit_v, bool) or not isinstance(lit_v, int):
                            continue
                        logit = float(phase_w[wi, bb])
                        sc = float(coll_w[wi, bb]) * (0.25 + abs(logit)) - 0.5 * float(conflict_heatmap[wi, 0, bb])
                        cands.append((sc, lit_v, wi, bb, logit))
                cands.sort(key=lambda x: x[0], reverse=True)

                rem = max(0.05, time_limit_sec - (time.perf_counter() - t0))
                if len(cands) >= cube_depth and rem > 0.25:
                    cube = [
                        (lit_v if logit >= 0.0 else -lit_v)
                        for _, lit_v, _, _, logit in cands[:cube_depth]
                    ]
                    if tuple(cube) in tried_cubes and cube:
                        cube[-1] = -cube[-1]
                    if tuple(cube) not in tried_cubes:
                        tried_cubes.add(tuple(cube))
                        sc_r, sol_r = solver.solve(
                            assumptions=cube,
                            confl_limit=1500,
                            time_limit=min(rem * 0.35, 0.8),
                        )
                        if sc_r is True:
                            sat_final, sol_final = True, sol_r
                            break
                        if sc_r is False:
                            conf = solver.get_conflict() or []
                            if len(conf) == 0:
                                sat_final = False
                                break
                            cores_pruned += 1
                            cl_len = len(conf)
                            min_core_len = cl_len if min_core_len == 0 else min(min_core_len, cl_len)
                            # pycryptosat.get_conflict() 已返回否定后的可直接注入阻断子句
                            solver.add_clause([int(x) for x in conf])
                            for clit in conf:
                                av = abs(int(clit))
                                if av in var_to_coord:
                                    wi_c, bb_c, _ = var_to_coord[av]
                                    conflict_heatmap[wi_c, 0, bb_c] += 1.2
                            continue
                        else:
                            _, _, wi0, bb0, _ = cands[0]
                            conflict_heatmap[wi0, 0, bb0] += 0.6

                rem2 = max(0.05, time_limit_sec - (time.perf_counter() - t0))
                sg, sol_g = solver.solve(confl_limit=burst_confl, time_limit=rem2)
                if sg is not None:
                    sat_final, sol_final = sg, sol_g
                    break
                burst_confl = min(16000, int(burst_confl * 1.6))

        dt = time.perf_counter() - t0
        w0_15: Optional[np.ndarray] = None
        valid_preimage = False
        if sat_final is True and sol_final is not None:
            w0_15 = extract_w0_15_from_cryptosat(enc, sol_final)
            valid_preimage = bool(self.core.verify_preimage(w0_15, ht_np, rounds=rounds)[0])
            assert valid_preimage, f"DynamicNeuralXORSolver 发现未通过 verify_preimage 的解 R={rounds}"

        status_str = "SAT" if sat_final is True else ("UNSAT" if sat_final is False else "TIMEOUT_OR_BUDGET")
        return DynamicSolveResult(
            rounds=int(rounds),
            solver_type="dynamic_neural_xor",
            status=status_str,
            valid_preimage=bool(valid_preimage),
            conflicts=-1,
            decisions=-1,
            elapsed_sec=float(dt),
            episodes=episodes,
            flp_units_proven=0,
            cores_pruned=cores_pruned,
            min_core_len=min_core_len,
            w0_15=w0_15,
        )
