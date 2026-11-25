"""
Phase-SHA V10「衔尾蛇-64」M3 原型 P1：动态 CDCL 净态修复、置信校准门控与长窗口连续预算 (`p1_clean_cdcl.py`)。

核心修正（严格只读消费既有权重，不修改核心源码）：
  1. `_build_full_crystal_var_map(enc)`：
     - 前置 `if isinstance(lit, bool):` 拦截，严禁把 Python `False/True` 误当 DIMACS 变量 `0/1`；
     - 使用 `Dict[int, List[Tuple[int, int, int, int, int]]]` 完整映射 `W_vars`、`a_canon`/`e_canon`
       四重移位别名至 `S_b`、`C_vars`（含 `C[:,:,0]=False`）。
  2. 净态前向：
     - 在 Episode 0 `solver.propagate()` 后将全部 0 层真值回填至 `W_b/S_b/C_b` 与 `unk`；
     - 显式传入 `conflict_heatmap=None, zero_defects=True`，彻底清除 OOD `[73..91]` 热力图与残差通道污染。
  3. 校准相位门控（Calibrated Phase Gate）：
     - `p_cal = sigmoid(logit / phase_cal_temp)`，仅当 `|p_cal - 0.5| >= phase_conf_thresh` 或
       `|dp - dn| >= flp_asym_thresh` 时于开局 Episode 0 注入一次；全拦时**不调用** `set_phases`；
     - 严禁搜中覆写 CaDiCaL 原生 Phase Saving。
  4. 零虚耗边界探测（P2）+ 长窗口连续预算（`long_chunk_conflicts=120000`）+ `threading.Timer` 精准硬中断。
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from pysat.solvers import Cadical103, Cadical153, Cadical195, Solver

from ..ouroboros_core.cnf_cdcl import SHA256CNFEncoder
from ..ouroboros_core.engine import (
    C_PLANES,
    IV_WORDS,
    MASK32,
    OuroborosCore,
    TorchCrystalEngine,
    get_core,
)
from ..ouroboros_mcts.dynamic_cdcl import DynamicSolveResult, _extract_cadical_w0_15
from ..ouroboros_net.heads import PLANE_COUNT
from ..ouroboros_net.model import Ouroboros64Net
from .p2_adaptive_probe_xor import ZeroWasteBoundaryProber


def _bcp_enqueue(lit: Any, n: int, assign: List[int], queue: deque[Tuple[int, int]]) -> bool:
    if isinstance(lit, bool):
        return False
    raw = int(lit)
    v = abs(raw)
    if v <= 0 or v > n:
        return False
    val = 1 if raw > 0 else 0
    cur = assign[v]
    if cur != -1:
        return cur != val
    assign[v] = val
    queue.append((v, val))
    return False


def _run_0level_bcp(clauses: Sequence[Sequence[int]], num_vars: int) -> Tuple[bool, List[int]]:
    """
    纯 Python 0 层 CNF 单位传播（BCP）。
    注：PySAT `solver.propagate()` 空参调用不返回 0 层单位文字（仅返回由 assumptions 触发的增量推导），
    因此必须使用显式 0 层 BCP 提取全部已推导单位文字并同步至时空晶体。
    """
    n = max(1, int(num_vars))
    assign = [-1] * (n + 1)
    occ_pos: List[List[int]] = [[] for _ in range(n + 1)]
    occ_neg: List[List[int]] = [[] for _ in range(n + 1)]
    cl_list: List[List[int]] = []
    satisfied: List[bool] = []
    unassigned_cnt: List[int] = []
    queue: deque[Tuple[int, int]] = deque()
    conflict = False

    # 第一遍：先吸纳所有单子句
    multi_clauses: List[Sequence[int]] = []
    for raw_c in clauses:
        if not raw_c:
            return False, []
        if len(raw_c) == 1:
            if _bcp_enqueue(raw_c[0], n, assign, queue):
                return False, []
        else:
            multi_clauses.append(raw_c)

    # 第二遍：跳过已被单子句满足的子句，仅索引剩余未满足多文字子句
    for raw_c in multi_clauses:
        kept: List[int] = []
        is_sat = False
        seen_in_c: Set[int] = set()
        for raw_l in raw_c:
            if isinstance(raw_l, bool):
                if raw_l:
                    is_sat = True
                    break
                continue
            l = int(raw_l)
            if -l in seen_in_c:
                is_sat = True
                break
            if l in seen_in_c:
                continue
            seen_in_c.add(l)
            av = abs(l)
            if av <= n and assign[av] != -1:
                if assign[av] == (1 if l > 0 else 0):
                    is_sat = True
                    break
                continue  # 该文字为假，跳过
            kept.append(l)
        if is_sat:
            continue
        if not kept:
            return False, []
        if len(kept) == 1:
            if _bcp_enqueue(kept[0], n, assign, queue):
                return False, []
            continue
        cid = len(cl_list)
        cl_list.append(kept)
        satisfied.append(False)
        unassigned_cnt.append(len(kept))
        for l in kept:
            v = abs(l)
            if v <= n:
                if l > 0:
                    occ_pos[v].append(cid)
                else:
                    occ_neg[v].append(cid)

    while queue and not conflict:
        v, val = queue.popleft()
        true_clauses = occ_pos[v] if val == 1 else occ_neg[v]
        false_clauses = occ_neg[v] if val == 1 else occ_pos[v]

        for cid in true_clauses:
            satisfied[cid] = True

        for cid in false_clauses:
            if satisfied[cid]:
                continue
            unassigned_cnt[cid] -= 1
            rem = unassigned_cnt[cid]
            if rem == 0:
                if not any(assign[abs(l)] == (1 if l > 0 else 0) for l in cl_list[cid]):
                    conflict = True
                    break
            elif rem == 1:
                found_true = False
                unit_lit = 0
                for l in cl_list[cid]:
                    av = assign[abs(l)]
                    if av == -1:
                        unit_lit = l
                    elif av == (1 if l > 0 else 0):
                        found_true = True
                        break
                if found_true:
                    satisfied[cid] = True
                elif unit_lit != 0:
                    if _bcp_enqueue(unit_lit, n, assign, queue):
                        conflict = True
                        break

    if conflict:
        return False, []
    prop_lits = [v if assign[v] == 1 else -v for v in range(1, n + 1) if assign[v] != -1]
    return True, prop_lits


def _get_round_timing_profile(rounds: int) -> Tuple[float, float, float, int, int, int]:
    """
    按 SHA-256 轮数分带的不可中断 CaDiCaL 控时参数表：
      返回 (prior_rate, rate_floor, rate_hard_cap, init_cap_long, init_cap_short, max_chunk_cap)
    """
    r = int(rounds)
    if r >= 56:
        return (220.0, 120.0, 450.0, 64, 64, 1024)
    if r >= 40:
        return (800.0, 200.0, 950.0, 64, 64, 1500)
    if r >= 29:
        return (2500.0, 800.0, 4000.0, 256, 256, 4000)
    if r >= 27:
        return (7000.0, 2500.0, 14000.0, 4000, 1000, 32000)
    return (8000.0, 3000.0, 22000.0, 4000, 1000, 48000)


@dataclass
class FullCrystalVarMap:
    """全时空晶体 DIMACS 变量双向/多别名映射结构。"""

    rounds: int
    var_to_coords: Dict[int, List[Tuple[int, int, int, int, int]]]
    const_w: Dict[Tuple[int, int], int]
    const_s: Dict[Tuple[int, int, int], int]
    const_c: Dict[Tuple[int, int, int], int]
    w_var_to_coord: Dict[int, Tuple[int, int, int]]
    const_fixed_w16: np.ndarray


def _build_full_crystal_var_map(enc: SHA256CNFEncoder) -> FullCrystalVarMap:
    """
    构建完整 W[0..R-1]、S[0..R, 0..7]（含 a_canon/e_canon 4 重移位别名）、C[0..R-1, 0..9] 变量映射。
    所有扫描均严格前置 `if isinstance(lit, bool):` 拦截，防止 False/True 混淆 DIMACS 0/1。
    """
    rounds = int(enc.rounds)
    var_to_coords: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
    const_w: Dict[Tuple[int, int], int] = {}
    const_s: Dict[Tuple[int, int, int], int] = {}
    const_c: Dict[Tuple[int, int, int], int] = {}
    w_var_to_coord: Dict[int, Tuple[int, int, int]] = {}
    const_fixed_w16 = np.zeros((16, 32), dtype=bool)

    def _add_coord(lit_val: Any, kind: int, t: int, word: int, bit: int) -> Optional[int]:
        if isinstance(lit_val, bool):
            return 1 if lit_val else 0
        raw = int(lit_val)
        v = abs(raw)
        if v <= 0:
            raise ValueError(f"非法 DIMACS 变量编号: {lit_val}")
        sign = 1 if raw > 0 else -1
        var_to_coords.setdefault(v, []).append((kind, t, word, bit, sign))
        return None

    # 1. W_vars[0..R-1][0..31]
    for t in range(rounds):
        for b in range(32):
            lit = enc.W_vars[t][b]
            cval = _add_coord(lit, 0, t, 0, b)
            if cval is not None:
                const_w[(t, b)] = cval
                if t < 16:
                    const_fixed_w16[t, b] = True
            elif t < 16:
                raw = int(lit)
                w_var_to_coord[abs(raw)] = (t, b, 1 if raw > 0 else -1)

    # 2. S[0..R][0..7][0..31] <- a_canon[t - w] (w<4) / e_canon[t - (w - 4)] (w>=4)
    for t in range(rounds + 1):
        for word in range(8):
            if word < 4:
                canon_row = enc.a_canon.get(t - word)
            else:
                canon_row = enc.e_canon.get(t - (word - 4))
            if canon_row is None:
                continue
            for b in range(32):
                lit = canon_row[b]
                cval = _add_coord(lit, 1, t, word, b)
                if cval is not None:
                    const_s[(t, word, b)] = cval

    # 3. C_vars[0..R-1][0..9][0..31]
    for t in range(rounds):
        for k in range(C_PLANES):
            for b in range(32):
                lit = enc.C_vars[t][k][b]
                cval = _add_coord(lit, 2, t, k, b)
                if cval is not None:
                    const_c[(t, k, b)] = cval

    return FullCrystalVarMap(
        rounds=rounds,
        var_to_coords=var_to_coords,
        const_w=const_w,
        const_s=const_s,
        const_c=const_c,
        w_var_to_coord=w_var_to_coord,
        const_fixed_w16=const_fixed_w16,
    )


def _set_bit_u32(arr: np.ndarray, idx: Tuple[int, ...], bit: int, val: int) -> None:
    if val & 1:
        arr[idx] |= np.uint32(1 << bit)
    else:
        arr[idx] &= np.uint32(~(1 << bit) & MASK32)


def _sync_crystal_state(
    H_single: np.ndarray,
    rounds: int,
    crystal_map: FullCrystalVarMap,
    fixed_words: Optional[Dict[int, int]],
    fixed_bits: Optional[Dict[Tuple[int, int], int]],
    prop_lits: Sequence[int],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor, Set[int]]:
    """
    将常量折叠位、外部已知边界以及 solver.propagate() 全部 0 层文字同步写入全晶体张量与 unk 掩码。
    """
    W_b = np.zeros((1, rounds), dtype=np.uint32)
    S_b = np.zeros((1, rounds + 1, 8), dtype=np.uint32)
    C_b = np.zeros((1, rounds, C_PLANES), dtype=np.uint32)
    unk_np = np.ones((1, rounds, PLANE_COUNT, 32), dtype=np.float32)

    # 双端硬焊边界 S[0]=IV, S[R]=(H-IV) mod 2^32
    s0 = np.asarray(IV_WORDS, dtype=np.uint32)
    sr = ((H_single.astype(np.uint64) - s0.astype(np.uint64)) & MASK32).astype(np.uint32)
    S_b[0, 0] = s0
    S_b[0, rounds] = sr
    unk_np[0, 0, 1:9, :] = 0.0

    # 常量折叠 bool 位 (W, S, C)
    for (t, b), val in crystal_map.const_w.items():
        if 0 <= t < rounds:
            _set_bit_u32(W_b, (0, t), b, val)
            unk_np[0, t, 0, b] = 0.0
    for (t, w, b), val in crystal_map.const_s.items():
        if 0 <= t <= rounds:
            _set_bit_u32(S_b, (0, t, w), b, val)
            if t < rounds:
                unk_np[0, t, 1 + w, b] = 0.0
    for (t, k, b), val in crystal_map.const_c.items():
        if 0 <= t < rounds:
            _set_bit_u32(C_b, (0, t, k), b, val)
            unk_np[0, t, 9 + k, b] = 0.0

    # 外部固定约束 (仅限 W[0..15] 指定已知位)
    for wi, wv in (fixed_words or {}).items():
        wi_i = int(wi)
        if 0 <= wi_i < rounds:
            W_b[0, wi_i] = np.uint32(int(wv) & MASK32)
            unk_np[0, wi_i, 0, :] = 0.0
    for (wi, bb), bv in (fixed_bits or {}).items():
        wi_i, bb_i = int(wi), int(bb)
        if 0 <= wi_i < rounds and 0 <= bb_i < 32:
            _set_bit_u32(W_b, (0, wi_i), bb_i, int(bv) & 1)
            unk_np[0, wi_i, 0, bb_i] = 0.0

    # 0 层传播文字回填至全晶体所有别名坐标
    assigned_vars: Set[int] = set()
    var_to_coords = crystal_map.var_to_coords
    for lit in prop_lits:
        if isinstance(lit, bool):
            continue
        raw = int(lit)
        av = abs(raw)
        assigned_vars.add(av)
        coords = var_to_coords.get(av)
        if not coords:
            continue
        lit_true = 1 if raw > 0 else 0
        for kind, t, word, bit, sign in coords:
            val = lit_true if sign > 0 else (1 - lit_true)
            if kind == 0:
                if 0 <= t < rounds:
                    _set_bit_u32(W_b, (0, t), bit, val)
                    unk_np[0, t, 0, bit] = 0.0
            elif kind == 1:
                if 0 <= t <= rounds:
                    _set_bit_u32(S_b, (0, t, word), bit, val)
                    if t < rounds:
                        unk_np[0, t, 1 + word, bit] = 0.0
            elif kind == 2:
                if 0 <= t < rounds:
                    _set_bit_u32(C_b, (0, t, word), bit, val)
                    unk_np[0, t, 9 + word, bit] = 0.0

    unk = torch.from_numpy(unk_np).to(device=device)
    return W_b, S_b, C_b, unk, assigned_vars


class CleanDynamicNeuralCDCLSolver:
    """
    P1+P2 净态动态神经 CDCL 求解器：
      - 全晶体 0 层回填 + 净态前向 (conflict_heatmap=None, zero_defects=True)；
      - 校准相位门控（仅开局 Episode 0，默认 T_cal=4.5, tau=0.08, flp_asym=16）；
      - 零冲突边界探测 (ZeroWasteBoundaryProber, B_c=0) 取代有损 Cube；
      - 长窗口连续预算 (long_chunk_conflicts=120000) + Timer 硬中断避免分段重入混沌。
    """

    def __init__(
        self,
        model: Optional[Ouroboros64Net] = None,
        core: Optional[OuroborosCore] = None,
        device: Optional[torch.device | str] = None,
        solver_name: str = "cadical195",
        phase_cal_temp: float = 4.5,
        phase_conf_thresh: float = 0.08,
        flp_asym_thresh: int = 16,
        enable_flp: bool = True,
        flp_top_k: int = 8,
        long_chunk_conflicts: int = 120000,
        enable_zero_waste_probe: bool = True,
        probe_n_max_1bit: int = 12,
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
        self.phase_cal_temp = float(phase_cal_temp)
        self.phase_conf_thresh = float(phase_conf_thresh)
        self.flp_asym_thresh = int(flp_asym_thresh)
        self.enable_flp = bool(enable_flp)
        self.flp_top_k = int(flp_top_k)
        self.long_chunk_conflicts = max(1000, int(long_chunk_conflicts))
        self.enable_zero_waste_probe = bool(enable_zero_waste_probe)
        self.probe_n_max_1bit = int(probe_n_max_1bit)
        self.last_diagnostics: Dict[str, Any] = {}

    @torch.no_grad()
    def _neural_score_full_crystal(
        self,
        H_single: np.ndarray,
        rounds: int,
        W_b: np.ndarray,
        S_b: np.ndarray,
        C_b: np.ndarray,
        unk: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """净态前向：显式传 conflict_heatmap=None, zero_defects=True。"""
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

    def solve(
        self,
        H_target: Sequence[int],
        rounds: int = 24,
        fixed_words: Optional[Dict[int, int]] = None,
        fixed_bits: Optional[Dict[Tuple[int, int], int]] = None,
        extra_clauses: Optional[Sequence[Sequence[int]]] = None,
        conflict_budget: int = 250000,
        time_limit_sec: float = 6.0,
    ) -> DynamicSolveResult:
        t0 = time.perf_counter()

        def _time_exhausted(margin: float = 0.005) -> bool:
            return time_limit_sec > 0.0 and (time.perf_counter() - t0) >= (time_limit_sec - margin)

        ht_np = np.asarray(H_target, dtype=np.uint32).reshape(8)
        enc = SHA256CNFEncoder(rounds=rounds).build(
            H_target=ht_np, fixed_words=fixed_words, fixed_bits=fixed_bits
        )
        clauses = list(enc.clauses)
        if extra_clauses:
            clauses.extend(list(c) for c in extra_clauses if c)

        crystal_map = _build_full_crystal_var_map(enc)

        episodes = 0
        flp_units_proven = 0
        cores_pruned = 0
        min_core_len = 0
        res_state: Optional[bool] = None
        w0_15: Optional[np.ndarray] = None
        valid_preimage = False
        stats: Dict[str, Any] = {}

        diag: Dict[str, Any] = {
            "conflict_heatmap_arg": None,
            "zero_defects_arg": True,
            "max_abs_logit": 0.0,
            "gate_total_candidates": 0,
            "gate_passed_phase_conf": 0,
            "gate_passed_flp_asym": 0,
            "gate_passed_total": 0,
            "set_phases_called": False,
            "probed_units": 0,
            "probed_binary_clauses": 0,
            "unknown_ratio_after_root_prop": 1.0,
            "supports_interrupt": False,
            "supports_propagate": True,
            "rate_est_final": 0.0,
        }

        if not _time_exhausted():
            with Solver(name=self.solver_name, bootstrap_with=clauses) as solver:
                # 1) Episode 0: 0 层传播 (PySAT propagate 仅验一致性，需配合 _run_0level_bcp 导出全量 0 层单位；
                #    若底层如 Cadical103 不支持 propagate()，则优雅回落至纯 Python 0 层 BCP 并跳过 FLP 探测)
                supports_propagate = True
                try:
                    ok_root, _ = solver.propagate()
                except NotImplementedError:
                    supports_propagate = False
                    diag["supports_propagate"] = False
                    ok_root = True
                if ok_root and not _time_exhausted():
                    ok_root, prop_lits = _run_0level_bcp(clauses, enc.next_var - 1)
                else:
                    prop_lits = []
                if not ok_root:
                    res_state = False
                elif not _time_exhausted():
                    episodes = 1
                    W_b, S_b, C_b, unk, assigned_vars = _sync_crystal_state(
                        H_single=ht_np,
                        rounds=rounds,
                        crystal_map=crystal_map,
                        fixed_words=fixed_words,
                        fixed_bits=fixed_bits,
                        prop_lits=prop_lits,
                        device=self.device,
                    )
                    diag["unknown_ratio_after_root_prop"] = float(unk.mean().item())

                    # 2) 单次净态全晶体前向 (conflict_heatmap=None, zero_defects=True)
                    free_candidates: List[Tuple[float, int, int, int, float]] = []
                    if not _time_exhausted():
                        coll_w, phase_w = self._neural_score_full_crystal(
                            ht_np, rounds, W_b, S_b, C_b, unk
                        )
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
                                free_candidates.append((sc, lit_v, wi, bb, logit))

                        free_candidates.sort(key=lambda x: x[0], reverse=True)
                        if free_candidates:
                            diag["max_abs_logit"] = float(max(abs(x[4]) for x in free_candidates))
                        diag["gate_total_candidates"] = len(free_candidates)

                    # 3) P2 零冲突边界探测 / Top-K 1-bit FLP (B_c = 0)
                    dp_dn_map: Dict[int, Tuple[int, int]] = {}
                    if self.enable_flp and supports_propagate and free_candidates and not _time_exhausted(margin=0.03):
                        n_1bit = max(
                            self.flp_top_k,
                            self.probe_n_max_1bit if self.enable_zero_waste_probe else self.flp_top_k,
                        )
                        cand_lits = [x[1] for x in free_candidates[:n_1bit]]
                        coupled = (
                            ZeroWasteBoundaryProber.build_coupled_pairs(free_candidates, max_pairs=4)
                            if self.enable_zero_waste_probe
                            else None
                        )
                        pr = ZeroWasteBoundaryProber.probe(
                            solver=solver,
                            cands=cand_lits,
                            n_max_1bit=n_1bit,
                            coupled_pairs=coupled,
                            max_2bit_pairs=4 if self.enable_zero_waste_probe else 0,
                            initial_base_lits=prop_lits,
                        )
                        flp_units_proven += int(pr["deduced_vars"])
                        n_bins = len(pr["binary_clauses"])
                        if n_bins > 0:
                            cores_pruned += n_bins
                            min_core_len = 2 if min_core_len == 0 else min(min_core_len, 2)
                        diag["probed_units"] = int(pr["deduced_vars"])
                        diag["probed_binary_clauses"] = int(n_bins)
                        dp_dn_map = pr["dp_dn"]

                        if pr["unsat"]:
                            res_state = False
                        else:
                            ok_after = True
                            if supports_propagate:
                                try:
                                    ok_after, _ = solver.propagate()
                                except NotImplementedError:
                                    supports_propagate = False
                                    diag["supports_propagate"] = False
                                    ok_after = True
                            if not ok_after:
                                res_state = False
                            else:
                                for u in pr["units"]:
                                    assigned_vars.add(abs(int(u)))

                    # 4) 校准极性门控 (仅在 Episode 0 门控；全拦则不调用 set_phases)
                    if res_state is None and free_candidates and not _time_exhausted():
                        surviving: List[Tuple[float, int, float, int, int]] = []
                        for sc, lit_v, wi, bb, logit in free_candidates:
                            if abs(lit_v) in assigned_vars:
                                continue
                            dp, dn = dp_dn_map.get(lit_v, (0, 0))
                            la_bonus = 0.25 * min(dp, dn) + 0.05 * abs(dp - dn)
                            surviving.append((sc + la_bonus, lit_v, logit, dp, dn))
                        surviving.sort(key=lambda x: x[0], reverse=True)

                        fresh_phases: List[int] = []
                        for _, lit_v, logit, dp, dn in surviving:
                            scaled = max(-20.0, min(20.0, logit / max(1e-6, self.phase_cal_temp)))
                            p_cal = 1.0 / (1.0 + math.exp(-scaled))
                            pass_conf = abs(p_cal - 0.5) >= self.phase_conf_thresh
                            pass_asym = abs(dp - dn) >= self.flp_asym_thresh
                            if pass_conf:
                                diag["gate_passed_phase_conf"] += 1
                            if pass_asym:
                                diag["gate_passed_flp_asym"] += 1
                            if pass_conf or pass_asym:
                                if pass_conf:
                                    signed_l = lit_v if logit >= 0.0 else -lit_v
                                else:
                                    signed_l = lit_v if dp >= dn else -lit_v
                                fresh_phases.append(signed_l)
                                if len(fresh_phases) >= 64:
                                    break

                        diag["gate_passed_total"] = len(fresh_phases)
                        if fresh_phases:
                            try:
                                solver.set_phases(literals=fresh_phases)
                                diag["set_phases_called"] = True
                            except Exception:
                                diag["set_phases_called"] = False

                    # 5) 长窗口连续预算：兼容 CaDiCaL 全系列别名（cd103/cd153/cd195/cadical* 等）与首段速率微标定
                    sname = str(self.solver_name).lower().strip()
                    no_int_names = {
                        "cd", "cd103", "cd15", "cd153", "cd19", "cd195",
                        "cdl", "cdl103", "cdl15", "cdl153", "cdl19", "cdl195",
                        "cadical", "cadical103", "cadical153", "cadical195",
                    }
                    is_cadical_alias = (
                        sname in no_int_names
                        or sname.startswith("cadical")
                        or sname.startswith("cd1")
                        or sname.startswith("cdl")
                        or isinstance(getattr(solver, "solver", solver), (Cadical103, Cadical153, Cadical195))
                    )
                    supports_interrupt = not is_cadical_alias
                    if supports_interrupt:
                        try:
                            solver.interrupt()
                            if hasattr(solver, "clear_interrupt"):
                                solver.clear_interrupt()
                        except NotImplementedError:
                            supports_interrupt = False
                        except Exception:
                            pass
                    diag["supports_interrupt"] = supports_interrupt

                    prior_rate, rate_floor, rate_hard_cap, init_cap_long, init_cap_short, max_chunk_cap = (
                        _get_round_timing_profile(rounds)
                    )
                    rate_est = prior_rate
                    prev_budget = 0
                    chunk_idx = 0
                    while res_state is None:
                        elapsed = time.perf_counter() - t0
                        rem_sec = (time_limit_sec - elapsed) if time_limit_sec > 0.0 else 0.0
                        if time_limit_sec > 0.0 and rem_sec <= 0.01:
                            break

                        cur_stats = solver.accum_stats() if hasattr(solver, "accum_stats") else {}
                        used_confl = int(cur_stats.get("conflicts", 0))
                        if conflict_budget > 0:
                            rem_c = int(conflict_budget) - used_confl
                            if rem_c <= 0:
                                break
                            max_allowed = min(self.long_chunk_conflicts, rem_c)
                        else:
                            max_allowed = self.long_chunk_conflicts

                        if time_limit_sec > 0.0 and not supports_interrupt:
                            if chunk_idx == 0:
                                base_init = init_cap_long if time_limit_sec >= 3.0 else init_cap_short
                                time_cap = max(8, int(rem_sec * 0.45 * prior_rate))
                                cur_budget = min(max_allowed, base_init, time_cap)
                            else:
                                if chunk_idx == 1:
                                    frac = 0.35 if rounds <= 28 else 0.30
                                elif chunk_idx == 2:
                                    frac = 0.40 if rounds <= 28 else 0.38
                                else:
                                    frac = 0.45
                                eff_rate = min(rate_hard_cap, max(rate_floor, rate_est))
                                time_capped = max(8, int(rem_sec * frac * eff_rate))
                                step_mult = 2.5 if rounds <= 28 else 2.0
                                step_capped = int(max(init_cap_short, prev_budget * step_mult))
                                cur_budget = min(max_allowed, max_chunk_cap, step_capped, time_capped)
                                if cur_budget < 8:
                                    cur_budget = min(max_allowed, max(1, time_capped))
                        else:
                            cur_budget = max_allowed

                        prev_budget = cur_budget
                        solver.conf_budget(max(1, cur_budget))
                        t_solve0 = time.perf_counter()
                        if supports_interrupt and time_limit_sec > 0.0:
                            interrupt_unsupported = False

                            def _safe_interrupt() -> None:
                                nonlocal interrupt_unsupported
                                try:
                                    solver.interrupt()
                                except NotImplementedError:
                                    interrupt_unsupported = True
                                except Exception:
                                    pass

                            timer = threading.Timer(max(0.001, rem_sec), _safe_interrupt)
                            timer.daemon = True
                            timer.start()
                            try:
                                res_state = solver.solve_limited(expect_interrupt=True)
                            except NotImplementedError:
                                supports_interrupt = False
                                diag["supports_interrupt"] = False
                                res_state = None
                            finally:
                                timer.cancel()
                                if hasattr(solver, "clear_interrupt"):
                                    try:
                                        solver.clear_interrupt()
                                    except NotImplementedError:
                                        supports_interrupt = False
                                        diag["supports_interrupt"] = False
                            if interrupt_unsupported:
                                supports_interrupt = False
                                diag["supports_interrupt"] = False
                        else:
                            res_state = solver.solve_limited()
                        dt_solve = max(1e-4, time.perf_counter() - t_solve0)

                        post_stats = solver.accum_stats() if hasattr(solver, "accum_stats") else {}
                        post_confl = int(post_stats.get("conflicts", 0))
                        d_confl = max(0, post_confl - used_confl)

                        if d_confl >= (8 if chunk_idx == 0 else 16):
                            rate_inst = float(d_confl) / dt_solve
                            if rate_inst > rate_est:
                                # 上行阻尼：防止浅层爆发短窗 (dt < 0.08s) 拉爆速率估计
                                max_up = 1.35 if dt_solve < 0.08 else 1.60
                                alpha_up = 0.25 if dt_solve < 0.08 else 0.45
                                obs_clamped = min(rate_inst, rate_est * max_up)
                                rate_est = (1.0 - alpha_up) * rate_est + alpha_up * obs_clamped
                            else:
                                # 下行快追：一旦跌入深层慢区立即快速下调
                                alpha_down = 0.50 if dt_solve < 0.05 else 0.75
                                rate_est = (1.0 - alpha_down) * rate_est + alpha_down * rate_inst
                            rate_est = min(rate_hard_cap, max(rate_floor, rate_est))
                        elif dt_solve > 0.05:
                            rate_inst = float(max(1, d_confl)) / dt_solve
                            rate_est = min(rate_hard_cap, max(rate_floor, 0.25 * rate_est + 0.75 * rate_inst))

                        diag["rate_est_final"] = round(rate_est, 2)
                        chunk_idx += 1

                        if res_state is not None:
                            break
                        if time_limit_sec > 0.0 and (time.perf_counter() - t0) >= time_limit_sec:
                            break

                stats = solver.accum_stats() if hasattr(solver, "accum_stats") else {}
                if res_state is True:
                    model_lits = solver.get_model() or []
                    w0_15 = _extract_cadical_w0_15(enc, model_lits)
                    valid_preimage = bool(self.core.verify_preimage(w0_15, ht_np, rounds=rounds)[0])
                    assert valid_preimage, f"CleanDynamicNeuralCDCLSolver 发现未通过 verify_preimage 的解 R={rounds}"

        dt = time.perf_counter() - t0
        self.last_diagnostics = diag
        status_str = "SAT" if res_state is True else ("UNSAT" if res_state is False else "TIMEOUT_OR_BUDGET")
        return DynamicSolveResult(
            rounds=int(rounds),
            solver_type="p1_clean_dynamic_cdcl",
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
