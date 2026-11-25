"""
Phase-SHA V10「衔尾蛇-64」M3 原型 P4：`R=64` 虫洞 MCTS 局部窗口 GF(2) + `W[32]` 耦合修复 + 轻量化评估
(`p4_local_wormhole_mcts.py`)。

核心改进：
  1. 局部轮窗 GF(2) 精确消元 `local_window_gf2_propagate(env, center_round, half_win=4)`：
     - 精确合并 `a_canon`/`e_canon` 在 `env.S_sym` 中的移位别名并立即检出冲突；
     - 构建局部窗口消息调度、压缩线性方程与补全后的 `Ch(5) + Maj(6) = 11` 条条件线性化规则；
     - 代入 `env` 已知位并压缩重标号至 `0 .. V_compact - 1`，调用既有 C++17 `core.solve_gf2_equations`；
     - 安全回写新推导单位（跳过硬焊边界 `S[0]/S[R]` 与 `C[:,:,0]/C[:16,:3,:]`，仅做只读校对防 PermissionError），
       并补跑 4 步轻量 `env.propagate(use_gf2=False, max_iters=4)`。
  2. 修复 `splice_block[:12]` 完整跨字耦合（8 bit `S[32]` + 4 bit `W[32]`）与紧凑三区配额：
     - 当未知 `W[0..15] <= 16` 时优先全覆以支持过约束可解探针（如 `R=64, free_bits in {4, 6}`）；
     - 满盲态下紧凑分配至末端摘要锚定区（`W[60..63]/S[63]`）、中间拼接区（`S[32]/W[32]`）与起始注入区（`W[0..15]`）。
  3. 单次前向与纯符号物理叶评估（消除叶节点第二次冗余前向，`_neural_score_env` 显式传 `conflict_heatmap=None`）。
  4. 严格继承 `WormholeMCTS`，返回标准 `MCTSSearchResult`，严守 `R=64` 密码学诚实边界。
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from ..ouroboros_core.engine import (
    C_PLANES,
    IV_WORDS,
    K_WORDS,
    MASK32,
    SYM_0,
    SYM_1,
    OuroborosCore,
    TorchCrystalEngine,
    get_core,
)
from ..ouroboros_mcts.symbol_env import GeneralizedSymbolEnv, SymbolStepResult
from ..ouroboros_mcts.wormhole_mcts import (
    MCTSAction,
    MCTSNode,
    MCTSSearchResult,
    WormholeMCTS,
)
from ..ouroboros_net.model import Ouroboros64Net


def compute_window_theory_var_count(t_start: int, t_end: int) -> int:
    """计算半开轮窗 [t_start, t_end) 在全未知状态下的理论 GF(2) 规范变量总数。"""
    w = max(0, int(t_end) - int(t_start))
    if w == 0:
        return 0
    w_min = max(0, t_start - 16) if t_end > 16 else t_start
    n_w = t_end - w_min
    n_ae = w + 4
    return int((n_w + 2 * n_ae + w * C_PLANES + 2 * w) * 32)


def local_window_gf2_propagate(
    env: GeneralizedSymbolEnv,
    center_round: int,
    half_win: int = 4,
    core: Optional[OuroborosCore] = None,
) -> Dict[str, Any]:
    """
    在局部轮窗 [t_L, t_R] = [max(0, center - half_win), min(R-1, center + half_win)] 内
    执行含已知位代入、Ch(5)+Maj(6) 条件线性化与紧凑重标号的精确 GF(2) RREF 传播。
    """
    t0 = time.perf_counter()
    c_core = core or env.core
    R = int(env.rounds)
    t_L = max(0, int(center_round) - int(half_win))
    t_R = min(R - 1, int(center_round) + int(half_win))
    t_start = t_L
    t_end = t_R + 1
    if t_end <= t_start:
        return {
            "conflict": bool(env.last_conflict),
            "units_applied": 0,
            "rank": 0,
            "window": (t_L, t_R),
            "num_vars": 0,
            "elapsed_ms": 0.0,
        }

    w_min = max(0, t_start - 16) if t_end > 16 else t_start
    known_map: Dict[int, int] = {}
    w_arr = env.W_sym.tolist()
    s_arr = env.S_sym.tolist()
    c_arr = env.C_sym.tolist()
    h_arr = [int(x) for x in env.H_target]

    # 紧凑整数键编码: (tag << 20) | ... 避免热循环海量 tuple 分配
    def _kw(t_i: int, b_i: int) -> int:
        return (t_i << 5) | b_i

    def _ka(idx_i: int, b_i: int) -> int:
        return (1 << 20) | ((idx_i + 4) << 5) | b_i

    def _ke(idx_i: int, b_i: int) -> int:
        return (2 << 20) | ((idx_i + 4) << 5) | b_i

    def _kc(t_i: int, k_i: int, b_i: int) -> int:
        return (3 << 20) | (t_i << 9) | (k_i << 5) | b_i

    def _kch(t_i: int, b_i: int) -> int:
        return (4 << 20) | (t_i << 5) | b_i

    def _kmaj(t_i: int, b_i: int) -> int:
        return (5 << 20) | (t_i << 5) | b_i

    def _fail() -> Dict[str, Any]:
        env.last_conflict = True
        return {
            "conflict": True,
            "units_applied": 0,
            "rank": 0,
            "window": (t_L, t_R),
            "num_vars": 0,
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }

    # 1. 收集已知变量并检查 a_canon / e_canon 在 S_sym 四重移位别名上的一致性
    for t in range(w_min, t_R + 1):
        w_row = w_arr[t]
        base_kw = t << 5
        for b in range(32):
            sv = w_row[b]
            if sv == SYM_0:
                known_map[base_kw | b] = 0
            elif sv == SYM_1:
                known_map[base_kw | b] = 1

    sr_words = [((h_arr[j] - IV_WORDS[j]) & MASK32) for j in range(8)]

    for idx in range(t_L - 3, t_R + 2):
        is_iv = -3 <= idx <= 0
        is_sr = R - 3 <= idx <= R
        iv_a = IV_WORDS[-idx] if is_iv else 0
        iv_e = IV_WORDS[4 + (-idx)] if is_iv else 0
        sr_a = sr_words[R - idx] if is_sr else 0
        sr_e = sr_words[4 + (R - idx)] if is_sr else 0
        valid_offsets = [w_off for w_off in range(4) if 0 <= idx + w_off <= R]
        base_ka = (1 << 20) | ((idx + 4) << 5)
        base_ke = (2 << 20) | ((idx + 4) << 5)

        for b in range(32):
            a_val: Optional[int] = None
            if is_iv:
                a_val = (iv_a >> b) & 1
            elif is_sr:
                a_val = (sr_a >> b) & 1
            for w_off in valid_offsets:
                sv = s_arr[idx + w_off][w_off][b]
                if sv == SYM_0 or sv == SYM_1:
                    v = 1 if sv == SYM_1 else 0
                    if a_val is not None and a_val != v:
                        return _fail()
                    a_val = v
            if a_val is not None:
                known_map[base_ka | b] = a_val

            e_val: Optional[int] = None
            if is_iv:
                e_val = (iv_e >> b) & 1
            elif is_sr:
                e_val = (sr_e >> b) & 1
            for w_off in valid_offsets:
                sv = s_arr[idx + w_off][4 + w_off][b]
                if sv == SYM_0 or sv == SYM_1:
                    v = 1 if sv == SYM_1 else 0
                    if e_val is not None and e_val != v:
                        return _fail()
                    e_val = v
            if e_val is not None:
                known_map[base_ke | b] = e_val

    for t in range(t_L, t_R + 1):
        c_t = c_arr[t]
        t_lt_16 = t < 16
        base_t = (3 << 20) | (t << 9)
        for k in range(C_PLANES):
            c_tk = c_t[k]
            base_tk = base_t | (k << 5)
            fixed_zero_plane = t_lt_16 and k < 3
            if c_tk[0] == SYM_1:
                return _fail()
            known_map[base_tk] = 0
            if fixed_zero_plane:
                for b in range(1, 32):
                    if c_tk[b] == SYM_1:
                        return _fail()
                    known_map[base_tk | b] = 0
            else:
                for b in range(1, 32):
                    sv = c_tk[b]
                    if sv == SYM_0:
                        known_map[base_tk | b] = 0
                    elif sv == SYM_1:
                        known_map[base_tk | b] = 1

    # 2. 构建局部原始方程并同步完成已知位代入与奇偶消去
    col_map: Dict[int, int] = {}
    col_to_coord: List[int] = []
    compact_eqs: List[Tuple[List[int], int]] = []
    km_get = known_map.get
    cm_get = col_map.get

    def _get_col(key: int) -> int:
        cid = cm_get(key)
        if cid is None:
            cid = len(col_to_coord)
            col_map[key] = cid
            col_to_coord.append(key)
        return cid

    def _push_eq(keys: Sequence[int], rhs0: int) -> bool:
        rhs = int(rhs0) & 1
        unk_set: Set[int] = set()
        for ky in keys:
            kv = km_get(ky)
            if kv is not None:
                rhs ^= kv
            elif ky in unk_set:
                unk_set.remove(ky)
            else:
                unk_set.add(ky)
        if not unk_set:
            if rhs == 1:
                return False  # 0 == 1 精确冲突
            return True
        cols = [col_map[ky] if ky in col_map else _get_col(ky) for ky in unk_set]
        compact_eqs.append((cols, rhs))
        return True

    # (a) 消息调度线性方程 t >= 16
    for t in range(max(16, t_L), t_R + 1):
        for b in range(32):
            terms = [
                _kw(t, b),
                _kw(t - 2, (b + 17) & 31),
                _kw(t - 2, (b + 19) & 31),
                _kw(t - 7, b),
                _kw(t - 15, (b + 7) & 31),
                _kw(t - 15, (b + 18) & 31),
                _kw(t - 16, b),
                _kc(t, 0, b),
                _kc(t, 1, b),
                _kc(t, 2, b),
            ]
            if b + 10 < 32:
                terms.append(_kw(t - 2, b + 10))
            if b + 3 < 32:
                terms.append(_kw(t - 15, b + 3))
            if not _push_eq(terms, 0):
                return _fail()

    # (b) 压缩线性方程 + (c) 补全 Ch(5) + Maj(6) = 11 条规则
    for t in range(t_L, t_R + 1):
        kw = int(K_WORDS[t])
        for b in range(32):
            kbit = (kw >> b) & 1
            ch_k = _kch(t, b)
            e_t_k = _ke(t, b)
            e_tm1_k = _ke(t - 1, b)
            e_tm2_k = _ke(t - 2, b)
            e_eq = [
                _ke(t + 1, b),
                _ka(t - 3, b),
                _ke(t - 3, b),
                _ke(t, (b + 6) & 31),
                _ke(t, (b + 11) & 31),
                _ke(t, (b + 25) & 31),
                ch_k,
                _kw(t, b),
                _kc(t, 3, b),
                _kc(t, 4, b),
                _kc(t, 5, b),
                _kc(t, 6, b),
                _kc(t, 9, b),
            ]
            if not _push_eq(e_eq, kbit):
                return _fail()

            maj_k = _kmaj(t, b)
            a_t_k = _ka(t, b)
            a_tm1_k = _ka(t - 1, b)
            a_tm2_k = _ka(t - 2, b)
            a_diff_eq = [
                _ka(t + 1, b),
                _ke(t + 1, b),
                _ka(t - 3, b),
                _ka(t, (b + 2) & 31),
                _ka(t, (b + 13) & 31),
                _ka(t, (b + 22) & 31),
                maj_k,
                _kc(t, 7, b),
                _kc(t, 8, b),
                _kc(t, 9, b),
            ]
            if not _push_eq(a_diff_eq, 0):
                return _fail()

            # Ch(e, f, g) 5 条规则
            ev = km_get(e_t_k)
            fv = km_get(e_tm1_k)
            gv = km_get(e_tm2_k)
            if ev == 1:
                if not _push_eq([ch_k, e_tm1_k], 0):
                    return _fail()
            elif ev == 0:
                if not _push_eq([ch_k, e_tm2_k], 0):
                    return _fail()
            if fv is not None and gv is not None:
                if fv == gv:
                    if not _push_eq([ch_k], fv):
                        return _fail()
                elif fv == 1 and gv == 0:
                    if not _push_eq([ch_k, e_t_k], 0):
                        return _fail()
                else:
                    if not _push_eq([ch_k, e_t_k], 1):
                        return _fail()

            # Maj(a, b, c) 补全 6 条规则
            av = km_get(a_t_k)
            bv = km_get(a_tm1_k)
            cv = km_get(a_tm2_k)
            if av is not None and bv is not None:
                if av == bv:
                    if not _push_eq([maj_k], av):
                        return _fail()
                else:
                    if not _push_eq([maj_k, a_tm2_k], 0):
                        return _fail()
            if av is not None and cv is not None:
                if av == cv:
                    if not _push_eq([maj_k], av):
                        return _fail()
                else:
                    if not _push_eq([maj_k, a_tm1_k], 0):
                        return _fail()
            if bv is not None and cv is not None:
                if bv == cv:
                    if not _push_eq([maj_k], bv):
                        return _fail()
                else:
                    if not _push_eq([maj_k, a_t_k], 0):
                        return _fail()

    v_compact = len(col_to_coord)
    if v_compact == 0 or not compact_eqs:
        return {
            "conflict": False,
            "units_applied": 0,
            "rank": 0,
            "window": (t_L, t_R),
            "num_vars": v_compact,
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }

    # 3. 调用既有 C++17 core.solve_gf2_equations
    res = c_core.solve_gf2_equations(v_compact, compact_eqs)
    if res["conflict"]:
        env.last_conflict = True
        return {
            "conflict": True,
            "units_applied": 0,
            "rank": int(res["rank"]),
            "window": (t_L, t_R),
            "num_vars": v_compact,
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        }

    # 4. 安全回写新增单位（绝不触碰不可写硬边界 S[0]/S[R]/C[:,:,0]/C[:16,:3,:]）
    units_applied = 0
    for col, val in res["units"].items():
        ky = col_to_coord[int(col)]
        target_sym = SYM_1 if int(val) == 1 else SYM_0
        kind_tag = ky >> 20
        if kind_tag == 0:
            t_i = (ky >> 5) & 0x3F
            b_i = ky & 31
            cur = int(env.W_sym[t_i, b_i])
            if cur in (SYM_0, SYM_1):
                if cur != target_sym:
                    return _fail()
            else:
                env.W_sym[t_i, b_i] = target_sym
                units_applied += 1
        elif kind_tag in (1, 2):
            idx_i = ((ky >> 5) & 0x7F) - 4
            b_i = ky & 31
            w_base = 0 if kind_tag == 1 else 4
            wrote_any = False
            for w_off in range(4):
                t_s = idx_i + w_off
                if not (0 <= t_s <= R):
                    continue
                word = w_base + w_off
                cur = int(env.S_sym[t_s, word, b_i])
                if t_s == 0 or t_s == R:
                    if cur != target_sym:
                        return _fail()
                    continue
                if cur in (SYM_0, SYM_1):
                    if cur != target_sym:
                        return _fail()
                else:
                    env.S_sym[t_s, word, b_i] = target_sym
                    wrote_any = True
            if wrote_any:
                units_applied += 1
        elif kind_tag == 3:
            t_i = (ky >> 9) & 0x3F
            k_i = (ky >> 5) & 0xF
            b_i = ky & 31
            if b_i == 0 or (t_i < 16 and k_i < 3):
                if int(val) != 0:
                    return _fail()
                continue
            cur = int(env.C_sym[t_i, k_i, b_i])
            if cur in (SYM_0, SYM_1):
                if cur != target_sym:
                    return _fail()
            else:
                env.C_sym[t_i, k_i, b_i] = target_sym
                units_applied += 1
        # ("Ch", ...) 与 ("Maj", ...) 为局部辅助消元桥，不写回 env

    if units_applied > 0:
        env.last_gf2_deductions = int(env.last_gf2_deductions) + units_applied
        step_res = env.propagate(use_gf2=False, max_iters=4)
        if step_res.conflict:
            return _fail()

    return {
        "conflict": False,
        "units_applied": units_applied,
        "rank": int(res["rank"]),
        "window": (t_L, t_R),
        "num_vars": v_compact,
        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
    }


class LocalWindowWormholeMCTS(WormholeMCTS):
    """
    P4 局部轮窗 GF(2) + splice_block[:12] 修复 + 紧凑三区动作 + 轻量化物理叶评估虫洞 MCTS。
    严格返回标准 MCTSSearchResult。
    """

    def __init__(
        self,
        model: Optional[Ouroboros64Net] = None,
        torch_eng: Optional[TorchCrystalEngine] = None,
        core: Optional[OuroborosCore] = None,
        device: Optional[torch.device | str] = None,
        c_puct: float = 1.35,
        max_branch_candidates: int = 8,
        prop_max_iters: int = 8,
        use_gf2_root: Optional[bool] = None,
        gf2_interval: int = 4,
        enable_macro_actions: bool = True,
        half_win: int = 4,
        use_local_gf2: bool = True,
        skip_leaf_neural_eval: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            torch_eng=torch_eng,
            core=core,
            device=device,
            c_puct=c_puct,
            max_branch_candidates=max_branch_candidates,
            prop_max_iters=prop_max_iters,
            use_gf2_root=use_gf2_root,
            gf2_interval=gf2_interval,
            enable_macro_actions=enable_macro_actions,
        )
        self.half_win = int(half_win)
        self.use_local_gf2 = bool(use_local_gf2)
        self.skip_leaf_neural_eval = bool(skip_leaf_neural_eval)
        self.last_search_diagnostics: Dict[str, Any] = {}
        self._diag_neural_calls = 0
        self._diag_heatmap_nonnull = 0
        self._cached_root_scores: Optional[Tuple[np.ndarray, np.ndarray, float]] = None

    @torch.no_grad()
    def _neural_score_env(
        self,
        env: GeneralizedSymbolEnv,
        conflict_heatmap: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        self._diag_neural_calls += 1
        if conflict_heatmap is not None:
            self._diag_heatmap_nonnull += 1
        # 强制净态前向：绝不把未训练的 conflict_heatmap 注入通道 [73..91]
        return super()._neural_score_env(env, conflict_heatmap=None)

    def _build_candidate_actions(
        self,
        env: GeneralizedSymbolEnv,
        focus_splice: bool = False,
        conflict_heatmap: Optional[np.ndarray] = None,
    ) -> Tuple[List[MCTSAction], np.ndarray, np.ndarray, float]:
        unk_w = np.argwhere((env.W_sym[:16] != SYM_0) & (env.W_sym[:16] != SYM_1))
        if 0 < len(unk_w) <= 16 and self._cached_root_scores is not None:
            coll_w, phase_w, pred_le = self._cached_root_scores
        else:
            coll_w, phase_w, pred_le = self._neural_score_env(env, conflict_heatmap=None)

        scored_vars: List[Tuple[float, int, int, float, float]] = []
        for row in unk_w:
            wi, bb = int(row[0]), int(row[1])
            c_sc = float(coll_w[wi, bb])
            ph = float(phase_w[wi, bb])
            total_sc = c_sc + 0.35 * abs(ph) + (0.15 if wi < 4 else 0.0)
            scored_vars.append((total_sc, wi, bb, c_sc, ph))
        scored_vars.sort(key=lambda x: x[0], reverse=True)

        actions: List[MCTSAction] = []
        raw_logits: List[float] = []

        # 过约束或接近闭合时（未知 W[0..15] <= 16 bit）：单变量 LSB 优先二分展开，避免 k! 排列爆炸
        if 0 < len(scored_vars) <= 16:
            sc, wi, bb, c_sc, ph = min(scored_vars, key=lambda x: (x[1], x[2]))
            p1 = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, ph))))
            first_sym, second_sym = (SYM_1, SYM_0) if p1 >= 0.5 else (SYM_0, SYM_1)
            p_first = max(0.51, p1 if first_sym == SYM_1 else (1.0 - p1))
            p_second = max(0.05, 1.0 - p_first)
            actions.append(MCTSAction(0, wi, 0, bb, first_sym, p_first, c_sc))
            raw_logits.append(sc + math.log(p_first + 1e-4))
            actions.append(MCTSAction(0, wi, 0, bb, second_sym, p_second, c_sc))
            raw_logits.append(sc + math.log(p_second + 1e-4))
        else:
            # 紧凑三区配额（大未知空间下每节点仅生成 ~8..12 个动作，快速突破 depth=1）
            r = env.rounds
            k_start = max(1, min(3, self.max_branch_candidates // 4))
            k_anchor = max(1, min(3, self.max_branch_candidates // 4))

            # Zone B (优先): splice_block[:12] 完整跨字耦合（8 bit S[32] + 4 bit W[32]）
            if self.enable_macro_actions and focus_splice:
                sr = env.splice_round
                splice_block: List[Tuple[int, int, int, int, int]] = []
                splice_block_alt: List[Tuple[int, int, int, int, int]] = []
                for ww in (0, 4, 3, 7):
                    unk_bits = [
                        b for b in range(32)
                        if int(env.S_sym[sr, ww, b]) not in (SYM_0, SYM_1)
                    ]
                    for b in unk_bits[:2]:
                        splice_block.append((1, sr, ww, b, SYM_0))
                        splice_block_alt.append((1, sr, ww, b, SYM_1))
                unk_wsr = [
                    b for b in range(32)
                    if int(env.W_sym[sr, b]) not in (SYM_0, SYM_1)
                ]
                for b in unk_wsr[:4]:
                    splice_block.append((0, sr, 0, b, SYM_0))
                    splice_block_alt.append((0, sr, 0, b, SYM_1))
                if len(splice_block) >= 4:
                    blk0 = tuple(splice_block[:12])
                    blk1 = tuple(splice_block_alt[:12])
                    k0, t0_s, w0_s, b0_s, sym0_s = blk0[0]
                    actions.append(
                        MCTSAction(k0, t0_s, w0_s, b0_s, sym0_s, 0.7, 2.5, extra_assignments=blk0)
                    )
                    raw_logits.append(2.5)
                    k1, t1_s, w1_s, b1_s, sym1_s = blk1[0]
                    actions.append(
                        MCTSAction(k1, t1_s, w1_s, b1_s, sym1_s, 0.3, 1.8, extra_assignments=blk1)
                    )
                    raw_logits.append(1.8)

            # Zone A: 末端摘要锚定区 W[R-4..R-1] / S[R-3..R-1]，直连 S[R] = H - IV
            anchor_cands: List[Tuple[float, int, int, int, int, int]] = []
            sr_last = r - 1
            for ww in (0, 4, 3, 7):
                for b in range(32):
                    if int(env.S_sym[sr_last, ww, b]) not in (SYM_0, SYM_1):
                        # 依据末端 S[R] = H - IV 的真实确定比特偏置排序，确保不同 H_target 产生真实分化
                        h_bit_a = int(env.S_sym[r, 0, b] == SYM_1)
                        h_bit_e = int(env.S_sym[r, 4, b] == SYM_1)
                        # 检查同轮 Maj(b,c) / Ch(f,g) 已被 H 锁定的耦合信号
                        maj_locked = 1 if int(env.S_sym[sr_last, 1, b]) == int(env.S_sym[sr_last, 2, b]) else 0
                        ch_locked = 1 if int(env.S_sym[sr_last, 5, b]) == int(env.S_sym[sr_last, 6, b]) else 0
                        # 优先低位进位归零 b=0 及 H 决定的耦合强位
                        a_sc = (
                            1.8
                            + (0.5 if b == 0 else 0.0)
                            + 0.25 * maj_locked
                            + 0.25 * ch_locked
                            + 0.15 * (h_bit_a ^ h_bit_e)
                            + 0.02 * (31 - b) / 31.0
                        )
                        pref = SYM_1 if ((h_bit_a ^ h_bit_e ^ (ww == 4)) & 1) else SYM_0
                        anchor_cands.append((a_sc, 1, sr_last, ww, b, pref))
            for wt in range(max(16, r - 4), r):
                for b in range(32):
                    if int(env.W_sym[wt, b]) not in (SYM_0, SYM_1):
                        h_bias = int(env.S_sym[r, wt - (r - 4), b] == SYM_1) if wt >= r - 4 else 0
                        w_sc = 1.6 + (0.45 if wt == r - 1 else 0.0) + (0.35 if b == 0 else 0.0) + 0.12 * h_bias
                        pref = SYM_1 if h_bias == 1 else SYM_0
                        anchor_cands.append((w_sc, 0, wt, 0, b, pref))
            anchor_cands.sort(key=lambda x: x[0], reverse=True)
            for a_sc, kd, tt, ww, bb, pref in anchor_cands[:k_anchor]:
                opp = SYM_0 if pref == SYM_1 else SYM_1
                actions.append(MCTSAction(kd, tt, ww, bb, pref, 0.58, a_sc))
                raw_logits.append(a_sc + 0.15)
                actions.append(MCTSAction(kd, tt, ww, bb, opp, 0.42, a_sc))
                raw_logits.append(a_sc - 0.15)

            # Zone C: 起始注入区 W[0..15]
            for sc, wi, bb, c_sc, ph in scored_vars[:k_start]:
                p1 = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, ph))))
                first_sym, second_sym = (SYM_1, SYM_0) if p1 >= 0.5 else (SYM_0, SYM_1)
                p_first = max(0.51, p1 if first_sym == SYM_1 else (1.0 - p1))
                p_second = max(0.05, 1.0 - p_first)
                actions.append(MCTSAction(0, wi, 0, bb, first_sym, p_first, c_sc))
                raw_logits.append(sc + math.log(p_first + 1e-4))
                actions.append(MCTSAction(0, wi, 0, bb, second_sym, p_second, c_sc))
                raw_logits.append(sc + math.log(p_second + 1e-4))

            # Zone B 单比特补充 (1 bit)
            if focus_splice:
                sr = env.splice_round
                unk_s = np.argwhere((env.S_sym[sr] != SYM_0) & (env.S_sym[sr] != SYM_1))
                if len(unk_s) > 0:
                    ww_s, bb_s = int(unk_s[0, 0]), int(unk_s[0, 1])
                    actions.append(MCTSAction(1, sr, ww_s, bb_s, SYM_1, 0.5, 0.8))
                    raw_logits.append(0.8)
                    actions.append(MCTSAction(1, sr, ww_s, bb_s, SYM_0, 0.5, 0.8))
                    raw_logits.append(0.8)

        if raw_logits:
            mx = max(raw_logits)
            exps = [math.exp(max(-20.0, v - mx)) for v in raw_logits]
            s_exp = sum(exps) + 1e-12
            for act, ex in zip(actions, exps):
                act.prior = float(ex / s_exp)

        return actions, coll_w, phase_w, pred_le

    def search(
        self,
        H_target: np.ndarray,
        rounds: int = 20,
        fixed_words: Optional[Dict[int, int]] = None,
        num_simulations: int = 48,
        time_limit_sec: float = 3.0,
        focus_splice: bool = False,
        fixed_bits: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> MCTSSearchResult:
        t0 = time.perf_counter()
        self._diag_neural_calls = 0
        self._diag_heatmap_nonnull = 0
        self._cached_root_scores = None
        local_gf2_calls = 0
        local_gf2_units = 0
        local_gf2_conflicts = 0
        local_gf2_ms_list: List[float] = []
        max_depth_reached = 0

        env_root = GeneralizedSymbolEnv(
            H_target=H_target,
            rounds=rounds,
            splice_round=max(1, rounds // 2),
            core=self.core,
        )
        fixed_bit_set: Set[Tuple[int, int]] = set()
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
        env_root.propagate(use_gf2=use_gf2_root, max_iters=self.prop_max_iters)

        # R > 28 根节点执行两窗毫秒级局部 GF(2) (末端摘要窗 + 中间拼接窗)
        if (not env_root.last_conflict) and (rounds > 28) and self.use_local_gf2:
            for ctr in (rounds - 1, env_root.splice_round):
                loc_r = local_window_gf2_propagate(
                    env_root, center_round=ctr, half_win=self.half_win, core=self.core
                )
                local_gf2_calls += 1
                local_gf2_ms_list.append(float(loc_r["elapsed_ms"]))
                local_gf2_units += int(loc_r["units_applied"])
                if loc_r["conflict"]:
                    local_gf2_conflicts += 1
                    break

        learned_units: List[Tuple[int, int, int, int, int]] = []
        seen_units: Set[Tuple[int, int, int, int, int]] = set()
        learned_clauses: List[List[Tuple[int, int, int, int, int]]] = []
        seen_clauses: Set[Tuple[Tuple[int, int, int, int, int], ...]] = set()
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

        actions_r, coll_r, phase_r, _root_pred_le = self._build_candidate_actions(
            env_root, focus_splice=focus_splice, conflict_heatmap=None
        )
        self._cached_root_scores = (coll_r, phase_r, _root_pred_le)
        root.untried_actions = actions_r
        root_num_actions = len(actions_r)
        splice_macro_lengths = [len(a.extra_assignments) for a in actions_r if a.is_macro]

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

        if solved_w is None and not root.is_conflict:
            for _ in range(num_simulations):
                if time_limit_sec > 0.0 and (time.perf_counter() - t0) >= time_limit_sec:
                    break
                sims += 1
                node = root
                path = [node]

                # 1. Selection
                while node.expanded and not node.untried_actions and node.children and not node.is_conflict and not node.is_solved:
                    best_child = max(node.children.values(), key=lambda ch: ch.ucb_score(self.c_puct))
                    if best_child.is_conflict:
                        break
                    node = best_child
                    path.append(node)

                # 2. Expansion + 局部窗口 GF(2)
                leaf_val = 0.0
                if not node.is_conflict and not node.is_solved:
                    if not node.expanded and not node.untried_actions:
                        acts, _, _, _ = self._build_candidate_actions(
                            node.env, focus_splice=focus_splice, conflict_heatmap=None
                        )
                        node.untried_actions = acts
                        node.expanded = True

                    if node.untried_actions:
                        act = node.untried_actions.pop(0)
                        node.expanded = True
                        child_env = node.env.clone()
                        depth = len(path)
                        max_depth_reached = max(max_depth_reached, depth)
                        gf2_ded_here = 0
                        try:
                            step_res = self._apply_action_to_env(child_env, act, use_gf2=False)
                            rem_w = int(np.sum((child_env.W_sym[:16] != SYM_0) & (child_env.W_sym[:16] != SYM_1)))
                            if not step_res.conflict:
                                if rounds <= 28:
                                    if (
                                        (depth % self.gf2_interval == 0)
                                        or (step_res.unit_prop_deductions >= 12)
                                        or (rem_w <= 8)
                                    ):
                                        step_res = child_env.propagate(use_gf2=True, max_iters=4)
                                        gf2_ded_here = int(step_res.gf2_deductions)
                                elif self.use_local_gf2:
                                    ctr = act.extra_assignments[-1][1] if act.extra_assignments else act.t
                                    should_loc = (
                                        act.is_macro
                                        or (depth % 3 == 0)
                                        or (step_res.unit_prop_deductions >= 2)
                                        or (rem_w <= 16)
                                    )
                                    if should_loc:
                                        loc_res = local_window_gf2_propagate(
                                            child_env,
                                            center_round=ctr,
                                            half_win=self.half_win,
                                            core=self.core,
                                        )
                                        local_gf2_calls += 1
                                        local_gf2_ms_list.append(float(loc_res["elapsed_ms"]))
                                        gf2_ded_here = int(loc_res["units_applied"])
                                        local_gf2_units += gf2_ded_here
                                        if loc_res["conflict"]:
                                            local_gf2_conflicts += 1
                                            step_res = SymbolStepResult(
                                                conflict=True,
                                                unit_prop_deductions=step_res.unit_prop_deductions,
                                                gf2_deductions=gf2_ded_here,
                                                gf2_rank=int(loc_res["rank"]),
                                                gf2_conflicts=1,
                                                unknown_before=step_res.unknown_before,
                                                unknown_after=child_env.count_unknown(),
                                                active_carry_gates=child_env.count_active_carry_gates(),
                                                splice_freedom_bits=child_env.splice_freedom_bits(),
                                            )
                        except PermissionError:
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
                            if node is root and (not act.is_macro) and act.kind == 0 and act.sym in (SYM_0, SYM_1):
                                opp = SYM_1 if act.sym == SYM_0 else SYM_0
                                k_opp = (act.kind, act.t, act.word, act.bit, opp)
                                if k_opp not in seen_units:
                                    seen_units.add(k_opp)
                                    learned_units.append(k_opp)
                            elif (
                                len(path) == 3
                                and node.action is not None
                                and (not node.action.is_macro)
                                and (not act.is_macro)
                                and node.action.kind == 0
                                and act.kind == 0
                                and node.action.sym in (SYM_0, SYM_1)
                                and act.sym in (SYM_0, SYM_1)
                            ):
                                opp1 = SYM_1 if node.action.sym == SYM_0 else SYM_0
                                opp2 = SYM_1 if act.sym == SYM_0 else SYM_0
                                cl = [
                                    (0, node.action.t, 0, node.action.bit, opp1),
                                    (0, act.t, 0, act.bit, opp2),
                                ]
                                ck = tuple(sorted(cl))
                                if ck not in seen_clauses:
                                    seen_clauses.add(ck)
                                    learned_clauses.append(cl)
                        else:
                            if child.unknown_count < min_unk:
                                min_unk = child.unknown_count
                                best_env = child_env
                            if child.active_carry_gates < min_gates:
                                min_gates = child.active_carry_gates
                            if child.splice_freedom < min_splice:
                                min_splice = child.splice_freedom

                            v_free = math.log1p(max(0, root.unknown_count - child.unknown_count))
                            v_gate = math.log1p(max(0, root.active_carry_gates - child.active_carry_gates))
                            v_splice = math.log1p(max(0, root_splice - child.splice_freedom))
                            if self.skip_leaf_neural_eval:
                                v_extra = math.log1p(max(0, gf2_ded_here))
                                uncapped_val = 0.45 * v_free + 0.35 * v_gate + 0.25 * v_splice + 0.15 * v_extra
                            else:
                                _, _, child_pred_le = self._neural_score_env(child_env, conflict_heatmap=None)
                                v_energy = math.log1p(max(0.0, _root_pred_le - child_pred_le))
                                uncapped_val = 0.45 * v_free + 0.35 * v_gate + 0.25 * v_splice + 0.20 * v_energy

                            cand_w = self._try_extract_preimage(child_env)
                            if cand_w is not None:
                                child.is_solved = True
                                solved_w = cand_w
                                leaf_val = 50.0 + uncapped_val
                            else:
                                leaf_val = uncapped_val

                # 3. Backpropagation + 死胡同子树向上闭合
                for nd in reversed(path):
                    nd.visit_count += 1
                    nd.value_sum += leaf_val
                    if (
                        nd.expanded
                        and not nd.untried_actions
                        and nd.children
                        and all(ch.is_conflict for ch in nd.children.values())
                    ):
                        nd.is_conflict = True

                if solved_w is not None or root.is_conflict:
                    break

        # 收尾刷新 phase_preferences（同样显式传 conflict_heatmap=None）
        if expansions > 0 and solved_w is None:
            coll_b, phase_b, _ = self._neural_score_env(best_env, conflict_heatmap=None)
            q_bias: Dict[Tuple[int, int], List[float]] = {}
            for ch in root.children.values():
                if ch.action is not None and (not ch.action.is_macro) and ch.action.kind == 0 and ch.visit_count >= 2:
                    coord = (ch.action.t, ch.action.bit)
                    pair_q = q_bias.setdefault(coord, [0.0, 0.0])
                    if ch.action.sym == SYM_1:
                        pair_q[1] = max(pair_q[1], ch.q_value)
                    else:
                        pair_q[0] = max(pair_q[0], ch.q_value)
            for (wi, bb) in list(phase_preferences.keys()):
                logit = float(phase_b[wi, bb])
                if (wi, bb) in q_bias:
                    logit += 0.35 * (q_bias[(wi, bb)][1] - q_bias[(wi, bb)][0])
                conf = float(coll_b[wi, bb]) * abs(logit)
                phase_preferences[(wi, bb)] = (conf, 1 if logit >= 0.0 else 0)

        total_root_visits = sum(ch.visit_count for ch in root.children.values() if ch.visit_count > 0)
        root_visit_policy: List[Tuple[Tuple[int, int, int, int, int], float]] = []
        if total_root_visits > 0:
            for ch in root.children.values():
                if ch.action is not None and (not ch.action.is_macro) and ch.visit_count > 0:
                    root_visit_policy.append(
                        (
                            (ch.action.kind, ch.action.t, ch.action.word, ch.action.bit, ch.action.sym),
                            float(ch.visit_count / total_root_visits),
                        )
                    )
        root_value_target = max(
            (ch.q_value for ch in root.children.values() if ch.visit_count > 0 and not ch.is_conflict),
            default=0.0,
        )

        self.last_search_diagnostics = {
            "max_depth_reached": max_depth_reached,
            "local_gf2_calls": local_gf2_calls,
            "local_gf2_units": local_gf2_units,
            "local_gf2_conflicts": local_gf2_conflicts,
            "local_gf2_mean_ms": round(float(np.mean(local_gf2_ms_list)), 3) if local_gf2_ms_list else 0.0,
            "neural_eval_calls": self._diag_neural_calls,
            "conflict_heatmap_was_always_none": (self._diag_heatmap_nonnull == 0),
            "root_num_actions": root_num_actions,
            "splice_macro_lengths": splice_macro_lengths,
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
            root_visit_policy=root_visit_policy,
            root_value_target=float(root_value_target),
        )
