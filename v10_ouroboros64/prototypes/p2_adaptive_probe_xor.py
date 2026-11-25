"""
Phase-SHA V10「衔尾蛇-64」M3 原型 P2：零虚耗边界探测与状态同步 XOR-Cube (`p2_adaptive_probe_xor.py`)。

核心组件：
  1. ZeroWasteBoundaryProber：
     - 彻底关闭消耗冲突预算的有损 Cube（B_c = 0）；
     - 仅调用 PySAT Solver.propagate(assumptions=[±x]) 执行 0 冲突单比特边界探测；
     - 提取公共蕴涵交集 (L_p ∩ L_n) \\ L_base 免费写入永久单位子句（Stålmarck/BCP-Intersection）；
     - 在耦合邻近窗口 {-2, -7, -15, -16} 或同字低位执行 2-bit 零冲突探测，直接写入二元阻断子句。
  2. SyncedDynamicXORSolver：
     - 维护伴随 PySAT CaDiCaL 影子传播器 shadow_bcp（绝不调用 solve()，仅作 0 层 BCP 状态追踪）；
     - 双向同步单位与二元子句，每轮动态刷新全晶体 W/S/C 未知掩码后净态调用 Ouroboros64Net，
       彻底消除原生 DynamicNeuralXORSolver 循环恒传静止 base_assigned 的空转病灶。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from pysat.solvers import Solver

from ..ouroboros_core.cnf_cdcl import (
    CryptoMiniSatXORSolver,
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
from ..ouroboros_mcts.dynamic_cdcl import DynamicSolveResult
from ..ouroboros_net.heads import PLANE_COUNT
from ..ouroboros_net.model import Ouroboros64Net


class ZeroWasteBoundaryProber:
    """
    零冲突预算边界探测器（B_c = 0）。
    严格保证：不消耗任何 CDCL 冲突预算，只提取全局等价的 0 层单位与二元阻断子句。
    """

    @staticmethod
    def build_coupled_pairs(
        candidates: Sequence[Tuple[float, int, int, int, float]],
        max_pairs: int = 4,
    ) -> List[Tuple[int, int]]:
        """
        从候选列表 [(score, lit_v, wi, bb, logit), ...] 中按同字近邻或消息扩展滞后
        {-2, -7, -15, -16} 筛选高耦合变量对。
        """
        top = list(candidates[:12])
        if len(top) < 2:
            return []
        lags = {0, 2, 7, 15, 16}
        pairs: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()
        for i in range(len(top)):
            _, l1, w1, b1, _ = top[i]
            for j in range(i + 1, len(top)):
                _, l2, w2, b2, _ = top[j]
                if l1 == l2:
                    continue
                dw = abs(w1 - w2)
                if dw in lags and (dw != 0 or abs(b1 - b2) <= 4):
                    key = (min(abs(l1), abs(l2)), max(abs(l1), abs(l2)))
                    if key not in seen:
                        seen.add(key)
                        pairs.append((int(l1), int(l2)))
                        if len(pairs) >= max_pairs:
                            return pairs
        # 若不足 max_pairs，从最高分邻近补充
        for i in range(len(top)):
            _, l1, _, _, _ = top[i]
            for j in range(i + 1, len(top)):
                _, l2, _, _, _ = top[j]
                key = (min(abs(l1), abs(l2)), max(abs(l1), abs(l2)))
                if key not in seen and abs(l1) != abs(l2):
                    seen.add(key)
                    pairs.append((int(l1), int(l2)))
                    if len(pairs) >= max_pairs:
                        return pairs
        return pairs

    @staticmethod
    def probe(
        solver: Solver,
        cands: Sequence[int],
        n_max_1bit: int = 12,
        coupled_pairs: Optional[Sequence[Tuple[int, int]]] = None,
        max_2bit_pairs: int = 4,
        initial_base_lits: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """
        对候选变量执行 0 冲突单比特与耦合 2-bit BCP 探测：
          - 单向失败 -> 写入永久单位子句；
          - 双向均可行 -> 抽取公共蕴涵交集 (L_p ∩ L_n) \\ L_base；
          - 耦合 2-bit 失败 -> 写入永久二元阻断子句 [-s1, -s2]。
        """
        ok_base, base_lits = solver.propagate()
        if not ok_base:
            return {
                "unsat": True,
                "units": [],
                "binary_clauses": [],
                "deduced_vars": 0,
                "dp_dn": {},
            }

        base_set: Set[int] = {int(x) for x in base_lits if not isinstance(x, bool)}
        if initial_base_lits:
            base_set.update(int(x) for x in initial_base_lits if not isinstance(x, bool))
        units: List[int] = []
        seen_units: Set[int] = set()
        binary_clauses: List[List[int]] = []
        seen_bins: Set[Tuple[int, int]] = set()
        dp_dn: Dict[int, Tuple[int, int]] = {}

        def _commit_unit(u: int) -> bool:
            u = int(u)
            if -u in base_set or -u in seen_units:
                return False
            if u in base_set or u in seen_units:
                return True
            solver.add_clause([u])
            units.append(u)
            seen_units.add(u)
            base_set.add(u)
            return True

        def _refresh_base() -> bool:
            ok_b, bl = solver.propagate()
            if not ok_b:
                return False
            base_set.update(int(x) for x in bl if not isinstance(x, bool))
            return True

        for raw_lit in list(cands)[: max(0, int(n_max_1bit))]:
            if isinstance(raw_lit, bool):
                continue
            v = int(raw_lit)
            if v == 0 or v in base_set or -v in base_set:
                continue

            ok_p, lp = solver.propagate(assumptions=[v])
            ok_n, ln = solver.propagate(assumptions=[-v])

            if (not ok_p) and (not ok_n):
                return {
                    "unsat": True,
                    "units": units,
                    "binary_clauses": binary_clauses,
                    "deduced_vars": len(units),
                    "dp_dn": dp_dn,
                }
            if not ok_p:
                if not _commit_unit(-v) or not _refresh_base():
                    return {
                        "unsat": True,
                        "units": units,
                        "binary_clauses": binary_clauses,
                        "deduced_vars": len(units),
                        "dp_dn": dp_dn,
                    }
                continue
            if not ok_n:
                if not _commit_unit(v) or not _refresh_base():
                    return {
                        "unsat": True,
                        "units": units,
                        "binary_clauses": binary_clauses,
                        "deduced_vars": len(units),
                        "dp_dn": dp_dn,
                    }
                continue

            lp_set = {int(x) for x in lp if not isinstance(x, bool)}
            ln_set = {int(x) for x in ln if not isinstance(x, bool)}
            dp = len(lp_set - base_set)
            dn = len(ln_set - base_set)
            dp_dn[v] = (dp, dn)

            # 公共蕴涵交集: (x -> y) ∧ (¬x -> y) ⊨ y
            common = (lp_set & ln_set) - base_set
            if common:
                for y in sorted(common, key=lambda z: (abs(z), z)):
                    if not _commit_unit(y):
                        return {
                            "unsat": True,
                            "units": units,
                            "binary_clauses": binary_clauses,
                            "deduced_vars": len(units),
                            "dp_dn": dp_dn,
                        }
                if not _refresh_base():
                    return {
                        "unsat": True,
                        "units": units,
                        "binary_clauses": binary_clauses,
                        "deduced_vars": len(units),
                        "dp_dn": dp_dn,
                    }

        if coupled_pairs and max_2bit_pairs > 0:
            probed_pairs = 0
            for raw_l1, raw_l2 in coupled_pairs:
                if probed_pairs >= max_2bit_pairs:
                    break
                if isinstance(raw_l1, bool) or isinstance(raw_l2, bool):
                    continue
                l1, l2 = abs(int(raw_l1)), abs(int(raw_l2))
                if l1 == 0 or l2 == 0 or l1 == l2:
                    continue
                if l1 in base_set or -l1 in base_set or l2 in base_set or -l2 in base_set:
                    continue
                probed_pairs += 1
                added_here = False
                for s1 in (l1, -l1):
                    for s2 in (l2, -l2):
                        if (
                            s1 in base_set
                            or -s1 in base_set
                            or s2 in base_set
                            or -s2 in base_set
                        ):
                            continue
                        ok_2, _ = solver.propagate(assumptions=[s1, s2])
                        if not ok_2:
                            c0, c1 = -s1, -s2
                            bkey = (c0, c1) if abs(c0) <= abs(c1) else (c1, c0)
                            if bkey not in seen_bins:
                                seen_bins.add(bkey)
                                cl = [c0, c1]
                                solver.add_clause(cl)
                                binary_clauses.append(cl)
                                added_here = True
                if added_here and not _refresh_base():
                    return {
                        "unsat": True,
                        "units": units,
                        "binary_clauses": binary_clauses,
                        "deduced_vars": len(units),
                        "dp_dn": dp_dn,
                    }

        return {
            "unsat": False,
            "units": units,
            "binary_clauses": binary_clauses,
            "deduced_vars": len(units),
            "dp_dn": dp_dn,
        }


class SyncedDynamicXORSolver:
    """
    伴随影子 BCP 同步型 DynamicNeuralXORSolver：
      - 维护伴随 PySAT shadow_bcp（仅 0 层 propagate，绝不调用 shadow_bcp.solve()）；
      - 将 2-literal XOR 与 CNF 单位/二元子句同时同步至 cms 与 shadow_bcp；
      - 每次进入神经排序前以 shadow_bcp.propagate() 同步最新 W/S/C 未知掩码并传 conflict_heatmap=None, zero_defects=True。
    """

    def __init__(
        self,
        model: Optional[Ouroboros64Net] = None,
        core: Optional[OuroborosCore] = None,
        device: Optional[torch.device | str] = None,
        threads: int = 1,
        shadow_solver_name: str = "cadical195",
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
        self.shadow_solver_name = shadow_solver_name
        self.last_diagnostics: Dict[str, Any] = {}

    @torch.no_grad()
    def _neural_rank_clean(
        self,
        H_single: np.ndarray,
        rounds: int,
        W_b: np.ndarray,
        S_b: np.ndarray,
        C_b: np.ndarray,
        unk: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None or self.torch_eng is None:
            coll = np.zeros((16, 32), dtype=np.float32)
            for wi in range(16):
                coll[wi, :] = 1.0 / (1.0 + 0.15 * wi)
            return coll, np.zeros((16, 32), dtype=np.float32)

        was_train = self.model.training
        self.model.eval()
        out = self.model(
            torch.from_numpy(H_single[None, :].astype(np.int64)).to(self.device),
            torch.from_numpy(W_b.astype(np.int64)).to(self.device),
            torch.from_numpy(S_b.astype(np.int64)).to(self.device),
            torch.from_numpy(C_b.astype(np.int64)).to(self.device),
            rounds=rounds,
            unknown_mask=unk,
            conflict_heatmap=None,
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
    def _xor_to_shadow_clauses(xor_clauses: Sequence[Tuple[Sequence[int], bool]]) -> List[List[int]]:
        """将 2-literal XOR (a ^ b == parity) 转为等价 2-CNF 注入 shadow_bcp。"""
        out: List[List[int]] = []
        for x_vars, parity in xor_clauses:
            if len(x_vars) == 2:
                a, b = int(x_vars[0]), int(x_vars[1])
                if parity:
                    out.append([a, b])
                    out.append([-a, -b])
                else:
                    out.append([-a, b])
                    out.append([a, -b])
        return out

    def solve(
        self,
        H_target: Sequence[int],
        rounds: int = 24,
        fixed_words: Optional[Dict[int, int]] = None,
        fixed_bits: Optional[Dict[Tuple[int, int], int]] = None,
        extra_clauses: Optional[Sequence[Sequence[int]]] = None,
        xor_clauses: Optional[Sequence[Tuple[Sequence[int], bool]]] = None,
        time_limit_sec: float = 6.0,
        cube_depth: int = 2,
        init_tl: Optional[float] = None,
    ) -> DynamicSolveResult:
        from .p1_clean_cdcl import _build_full_crystal_var_map, _run_0level_bcp, _sync_crystal_state

        t0 = time.perf_counter()
        ht_np = np.asarray(H_target, dtype=np.uint32).reshape(8)
        enc = SHA256XORCNFEncoder(rounds=rounds).build(
            H_target=ht_np, fixed_words=fixed_words, fixed_bits=fixed_bits
        )
        all_extra = [list(c) for c in (extra_clauses or []) if c]
        all_xors = list(enc.xor_clauses)
        if xor_clauses:
            for xv, xp in xor_clauses:
                all_xors.append((list(xv), bool(xp)))

        cms = self.xor_solver.new_solver(enc, extra_clauses=all_extra)
        if xor_clauses:
            for xv, xp in xor_clauses:
                cms.add_xor_clause(list(xv), bool(xp))

        shadow_clauses = list(enc.clauses) + all_extra + self._xor_to_shadow_clauses(all_xors)
        crystal_map = _build_full_crystal_var_map(enc)

        episodes = 0
        flp_units_proven = 0
        cores_pruned = 0
        min_core_len = 0
        sat_final: Optional[bool] = None
        sol_final: Optional[Sequence[Optional[bool]]] = None

        diag_assigned_counts: List[int] = []
        diag_cand_var_sets: List[Set[int]] = []
        diag_shadow_prop_calls = 0
        diag_sync_events = 0

        with Solver(name=self.shadow_solver_name, bootstrap_with=shadow_clauses) as shadow_bcp:
            # 初始化后做一次 0 层传播并与 cms 同步单位/二元
            ok_s, _ = shadow_bcp.propagate()
            diag_shadow_prop_calls += 1
            prop_lits: List[int] = []
            if ok_s:
                ok_s, prop_lits = _run_0level_bcp(shadow_clauses, enc.next_var - 1)
            if not ok_s:
                sat_final = False
            else:
                probe0 = ZeroWasteBoundaryProber.probe(
                    shadow_bcp,
                    cands=[],
                    n_max_1bit=0,
                    initial_base_lits=prop_lits,
                )
                if probe0["unsat"]:
                    sat_final = False

            if sat_final is None:
                eff_init_tl = (
                    float(init_tl)
                    if init_tl is not None
                    else min(0.6, max(0.15, time_limit_sec * 0.25))
                )
                sat0, sol0 = cms.solve(confl_limit=2000, time_limit=eff_init_tl)
                if sat0 is not None:
                    sat_final, sol_final = sat0, sol0
                    episodes = 1
                else:
                    tried_cubes: Set[Tuple[int, ...]] = set()
                    burst_confl = 3000
                    shadow_dirty = False

                    while (time.perf_counter() - t0) < time_limit_sec:
                        episodes += 1
                        ok_s, _ = shadow_bcp.propagate()
                        diag_shadow_prop_calls += 1
                        if not ok_s:
                            sat_final = False
                            break
                        if shadow_dirty:
                            ok_s, prop_lits = _run_0level_bcp(shadow_clauses, enc.next_var - 1)
                            shadow_dirty = False
                            if not ok_s:
                                sat_final = False
                                break

                        W_b, S_b, C_b, unk, assigned_vars = _sync_crystal_state(
                            H_single=ht_np,
                            rounds=rounds,
                            crystal_map=crystal_map,
                            fixed_words=fixed_words,
                            fixed_bits=fixed_bits,
                            prop_lits=prop_lits,
                            device=self.device,
                        )
                        diag_assigned_counts.append(len(assigned_vars))

                        coll_w, phase_w = self._neural_rank_clean(
                            ht_np, rounds, W_b, S_b, C_b, unk
                        )
                        cands: List[Tuple[float, int, int, int, float]] = []
                        for wi in range(16):
                            for bb in range(32):
                                if crystal_map.const_fixed_w16[wi, bb]:
                                    continue
                                if float(unk[0, wi, 0, bb].item()) < 0.5:
                                    continue
                                lit_v = enc.W_vars[wi][bb]
                                if isinstance(lit_v, bool) or not isinstance(lit_v, int):
                                    continue
                                if abs(lit_v) in assigned_vars:
                                    continue
                                logit = float(phase_w[wi, bb])
                                sc = float(coll_w[wi, bb]) * (0.25 + abs(logit))
                                cands.append((sc, lit_v, wi, bb, logit))
                        cands.sort(key=lambda x: x[0], reverse=True)
                        diag_cand_var_sets.append({abs(x[1]) for x in cands})

                        # 零冲突边界探测（仅在 shadow_bcp 上，B_c = 0），并将新单位/二元双向同步至 cms
                        if cands:
                            cand_lits = [x[1] for x in cands[:8]]
                            coupled = ZeroWasteBoundaryProber.build_coupled_pairs(cands, max_pairs=2)
                            pr = ZeroWasteBoundaryProber.probe(
                                shadow_bcp,
                                cand_lits,
                                n_max_1bit=6,
                                coupled_pairs=coupled,
                                max_2bit_pairs=2,
                                initial_base_lits=prop_lits,
                            )
                            if pr["unsat"]:
                                sat_final = False
                                break
                            if pr["units"] or pr["binary_clauses"]:
                                diag_sync_events += 1
                                for u in pr["units"]:
                                    cms.add_clause([int(u)])
                                    shadow_clauses.append([int(u)])
                                    flp_units_proven += 1
                                for bcl in pr["binary_clauses"]:
                                    cl_b = [int(x) for x in bcl]
                                    cms.add_clause(cl_b)
                                    shadow_clauses.append(cl_b)
                                    cores_pruned += 1
                                    min_core_len = 2 if min_core_len == 0 else min(min_core_len, 2)
                                ok_s2, prop_lits = _run_0level_bcp(shadow_clauses, enc.next_var - 1)
                                diag_shadow_prop_calls += 1
                                if not ok_s2:
                                    sat_final = False
                                    break
                                new_assigned = {abs(int(x)) for x in prop_lits if not isinstance(x, bool)}
                                diag_assigned_counts[-1] = len(new_assigned)
                                cands = [x for x in cands if abs(x[1]) not in new_assigned]

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
                                sc_r, sol_r = cms.solve(
                                    assumptions=cube,
                                    confl_limit=1500,
                                    time_limit=min(rem * 0.35, 0.8),
                                )
                                if sc_r is True:
                                    sat_final, sol_final = True, sol_r
                                    break
                                if sc_r is False:
                                    conf = cms.get_conflict() or []
                                    if len(conf) == 0:
                                        sat_final = False
                                        break
                                    cores_pruned += 1
                                    cl_len = len(conf)
                                    min_core_len = cl_len if min_core_len == 0 else min(min_core_len, cl_len)
                                    conf_cl = [int(x) for x in conf]
                                    cms.add_clause(conf_cl)
                                    shadow_bcp.add_clause(conf_cl)
                                    shadow_clauses.append(conf_cl)
                                    shadow_dirty = True
                                    diag_sync_events += 1
                                    continue

                        rem2 = max(0.05, time_limit_sec - (time.perf_counter() - t0))
                        sg, sol_g = cms.solve(confl_limit=burst_confl, time_limit=rem2)
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
            assert valid_preimage, f"SyncedDynamicXORSolver 发现未通过 verify_preimage 的解 R={rounds}"

        self.last_diagnostics = {
            "assigned_counts": diag_assigned_counts,
            "cand_var_sets": diag_cand_var_sets,
            "shadow_prop_calls": diag_shadow_prop_calls,
            "sync_events": diag_sync_events,
            "zero_defects_arg": True,
            "conflict_heatmap_arg": None,
        }

        status_str = "SAT" if sat_final is True else ("UNSAT" if sat_final is False else "TIMEOUT_OR_BUDGET")
        return DynamicSolveResult(
            rounds=int(rounds),
            solver_type="synced_dynamic_xor",
            status=status_str,
            valid_preimage=bool(valid_preimage),
            conflicts=-1,
            decisions=-1,
            elapsed_sec=float(dt),
            episodes=episodes,
            flp_units_proven=flp_units_proven,
            cores_pruned=cores_pruned,
            min_core_len=min_core_len,
            w0_15=w0_15,
        )
