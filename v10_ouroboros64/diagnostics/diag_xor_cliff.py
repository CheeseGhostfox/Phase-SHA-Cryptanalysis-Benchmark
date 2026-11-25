"""
R2 只读诊断脚本 3：XOR-CDCL 自 R=26 12b 起基本归零相变断崖诊断。

严格只读，覆盖：
  1. R in [20, 24, 25, 26, 28]（默认 8 档，--fast 4 档）的 SHA256XORCNFEncoder 静态门分解：
     - 压缩进位门 217*R、消息扩展进位门 93*max(0, R-16)、Ch 32*R、Maj 32*R；
  2. 纯 Python CNF 0 层单位传播 BCP（绝不误用空参 pysat.propagate()）推导能力统计；
  3. GeneralizedSymbolEnv.propagate(use_gf2=True, max_iters=16) 双平面闭合统计：
     - 活跃未定进位门 / Ch 门 / Maj 门（含未定输出 & 非线性化未满足门）比例；
     - GF(2) 秩、消元自由度（调用 core.solve_gf2_equations(..., return_degrees=True)）、未知 W/S 轮次；
  4. CNF-XOR 变量耦合度（|V_cnf ∩ V_xor| / |V_cnf ∪ V_xor| 及平均度数）；
  5. CryptoMiniSatXORSolver vs CaDiCaLCDCLSolver 实测对比（全部 SAT 强制 verify_preimage 100% 校验）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v10_ouroboros64.ouroboros_core.cnf_cdcl import (
    CaDiCaLCDCLSolver,
    CryptoMiniSatXORSolver,
    SHA256XORCNFEncoder,
    make_end_free_constraints,
)
from v10_ouroboros64.ouroboros_core.engine import (
    C_PLANES,
    IV_WORDS,
    K_WORDS,
    MASK32,
    SYM_0,
    SYM_1,
    OuroborosCore,
    enforce_fp32_determinism,
    get_core,
)
from v10_ouroboros64.ouroboros_mcts.symbol_env import GeneralizedSymbolEnv
from v10_ouroboros64.ouroboros_train.trainer import _sample_random_words


DEFAULT_TIERS: Tuple[Tuple[int, int], ...] = (
    (20, 12),
    (24, 12),
    (24, 32),
    (25, 12),
    (25, 24),
    (26, 12),
    (26, 16),
    (28, 12),
)
FAST_TIERS: Tuple[Tuple[int, int], ...] = (
    (24, 12),
    (25, 12),
    (26, 12),
    (28, 12),
)
DEFAULT_OUT_JSON = "v10_ouroboros64/diagnostics/diag_xor_cliff_report.json"


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


def _run_cnf_bcp(clauses: Sequence[Sequence[int]], num_vars: int) -> Dict[str, Any]:
    """
    纯 Python 0 层单位子句传播（BCP），不依赖 pysat.propagate() 的空参限制。
    Returns:
      conflict: bool
      assigned: Dict[int, bool]  (var_id -> True/False)
      deduced_count: int
    """
    # assign[v] in (-1=unassigned, 0=False, 1=True)
    assign = [-1] * (num_vars + 1)
    # occ_pos[v] / occ_neg[v]: list of clause indices
    occ_pos: List[List[int]] = [[] for _ in range(num_vars + 1)]
    occ_neg: List[List[int]] = [[] for _ in range(num_vars + 1)]
    cl_list: List[List[int]] = []
    satisfied: List[bool] = []
    unassigned_cnt: List[int] = []
    queue: deque[Tuple[int, int]] = deque()
    conflict = False

    def _enqueue(lit: int) -> None:
        nonlocal conflict
        v = abs(lit)
        val = 1 if lit > 0 else 0
        cur = assign[v]
        if cur != -1:
            if cur != val:
                conflict = True
            return
        assign[v] = val
        queue.append((v, val))

    for raw_c in clauses:
        if not raw_c:
            conflict = True
            break
        # 去重并检查重言式
        lits_set = set(int(x) for x in raw_c)
        if any(-x in lits_set for x in lits_set):
            continue
        c = list(lits_set)
        if len(c) == 1:
            _enqueue(c[0])
            if conflict:
                break
            continue
        cid = len(cl_list)
        cl_list.append(c)
        satisfied.append(False)
        unassigned_cnt.append(len(c))
        for lit in c:
            v = abs(lit)
            if lit > 0:
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
                # 再核实是否因同时存在已满足文字
                if not any(
                    assign[abs(l)] == (1 if l > 0 else 0)
                    for l in cl_list[cid]
                ):
                    conflict = True
                    break
            elif rem == 1:
                # 找到唯一未赋值文字（或确认已有真文字）
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
                    _enqueue(unit_lit)
                    if conflict:
                        break

    assigned_map = {v: bool(assign[v]) for v in range(1, num_vars + 1) if assign[v] != -1}
    return {
        "conflict": conflict,
        "assigned": assigned_map,
        "deduced_count": len(assigned_map),
    }


def _count_active_ch_maj_and_gf2_degrees(
    core: OuroborosCore,
    env: GeneralizedSymbolEnv,
) -> Dict[str, Any]:
    """
    统计传播后未确定的 Ch/Maj 门数量，以及使用 core.solve_gf2_equations 构建与 C++ Plane A 等价的
    GF(2) 线性系统测算自由度与变量出现度数分布。
    """
    r = env.rounds
    s_sym = env.S_sym  # (R+1, 8, 32)
    w_sym = env.W_sym  # (R, 32)
    c_sym = env.C_sym  # (R, 10, 32)

    ch_undetermined = 0
    ch_nonlinearized = 0
    maj_undetermined = 0
    maj_nonlinearized = 0
    total_ch = r * 32
    total_maj = r * 32

    for t in range(r):
        for b in range(32):
            e = int(s_sym[t, 4, b])
            f = int(s_sym[t, 5, b])
            g = int(s_sym[t, 6, b])
            e_det = e in (SYM_0, SYM_1)
            f_det = f in (SYM_0, SYM_1)
            g_det = g in (SYM_0, SYM_1)
            # Ch 输出可确定条件: (e==1 and f_det) or (e==0 and g_det) or (f_det and g_det and f==g)
            ch_det = (
                (e == SYM_1 and f_det)
                or (e == SYM_0 and g_det)
                or (f_det and g_det and f == g)
            )
            if not ch_det:
                ch_undetermined += 1
            # GF(2) 线性化条件 (对应 crystal_core.hpp:1151-1169): e_det or (f_det and g_det)
            if not (e_det or (f_det and g_det)):
                ch_nonlinearized += 1

            a = int(s_sym[t, 0, b])
            bb = int(s_sym[t, 1, b])
            c = int(s_sym[t, 2, b])
            a_det = a in (SYM_0, SYM_1)
            b_det = bb in (SYM_0, SYM_1)
            c_det = c in (SYM_0, SYM_1)
            maj_det = (
                (a_det and b_det and a == bb)
                or (a_det and c_det and a == c)
                or (b_det and c_det and bb == c)
            )
            if not maj_det:
                maj_undetermined += 1
            # GF(2) 线性化条件 (crystal_core.hpp:1174-1187): 任意两输入确定
            if (int(a_det) + int(b_det) + int(c_det)) < 2:
                maj_nonlinearized += 1

    # 镜像构建与 crystal_core.hpp:1084-1257 匹配的 GF(2) 线性方程系统并测自由度
    # 列编号布局:
    #   W: 0 .. R*32-1
    #   S[1..R-1, 0..7]: R*32 .. R*32 + (R-1)*256 - 1  (仅中间层 a/e 映射或全 S[t,w,b])
    #   为与 crystal_core.hpp 一致，用动态 col_id 分配器仅给未知或相关信号分配列:
    #   W[0..R-1, b], a[1..R-1, b], e[1..R-1, b], C[0..R-1, 0..9, b], Ch[0..R-1, b], Maj[0..R-1, b]
    #   (已知常量 S[0]/S[R]/a[-3..0]/e[-3..0]/C[:,:,0] 直接折叠进 RHS)
    col_map: Dict[Tuple[str, int, int, int], int] = {}

    def _get_col(kind: str, t_i: int, w_i: int, b_i: int) -> int:
        key = (kind, t_i, w_i, b_i)
        cid = col_map.get(key)
        if cid is None:
            cid = len(col_map)
            col_map[key] = cid
        return cid

    def _s_canon_a(t_i: int, b_i: int) -> Tuple[Optional[int], int]:
        if -3 <= t_i <= 0:
            val = (IV_WORDS[-t_i] >> b_i) & 1
            return None, val
        if r - 3 <= t_i <= r:
            sr = (int(env.H_target[r - t_i]) - IV_WORDS[r - t_i]) & MASK32
            return None, (sr >> b_i) & 1
        # S[t, 0, b]
        sv = int(s_sym[t_i, 0, b_i])
        if sv in (SYM_0, SYM_1):
            return None, 1 if sv == SYM_1 else 0
        return _get_col("a", t_i, 0, b_i), 0

    def _s_canon_e(t_i: int, b_i: int) -> Tuple[Optional[int], int]:
        if -3 <= t_i <= 0:
            val = (IV_WORDS[4 + (-t_i)] >> b_i) & 1
            return None, val
        if r - 3 <= t_i <= r:
            sr = (int(env.H_target[4 + (r - t_i)]) - IV_WORDS[4 + (r - t_i)]) & MASK32
            return None, (sr >> b_i) & 1
        sv = int(s_sym[t_i, 4, b_i])
        if sv in (SYM_0, SYM_1):
            return None, 1 if sv == SYM_1 else 0
        return _get_col("e", t_i, 4, b_i), 0

    def _w_term(t_i: int, b_i: int) -> Tuple[Optional[int], int]:
        wv = int(w_sym[t_i, b_i])
        if wv in (SYM_0, SYM_1):
            return None, 1 if wv == SYM_1 else 0
        return _get_col("W", t_i, 0, b_i), 0

    def _c_term(t_i: int, k_i: int, b_i: int) -> Tuple[Optional[int], int]:
        if b_i == 0 or (t_i < 16 and k_i < 3):
            return None, 0
        cv = int(c_sym[t_i, k_i, b_i])
        if cv in (SYM_0, SYM_1):
            return None, 1 if cv == SYM_1 else 0
        return _get_col("C", t_i, k_i, b_i), 0

    eqs: List[Tuple[List[int], int]] = []

    def _push_terms(terms: List[Tuple[Optional[int], int]], rhs0: int) -> None:
        rhs = rhs0 & 1
        counts: Dict[int, int] = {}
        for cid, cval in terms:
            if cid is None:
                rhs ^= (cval & 1)
            else:
                counts[cid] = (counts.get(cid, 0) + 1) & 1
        cols = sorted(k for k, v in counts.items() if v == 1)
        if cols or rhs:
            eqs.append((cols, rhs))

    # 消息扩展线性方程 t >= 16
    for t in range(16, r):
        for b in range(32):
            terms = [
                _w_term(t, b),
                _w_term(t - 2, (b + 17) & 31),
                _w_term(t - 2, (b + 19) & 31),
                _w_term(t - 7, b),
                _w_term(t - 15, (b + 7) & 31),
                _w_term(t - 15, (b + 18) & 31),
                _w_term(t - 16, b),
                _c_term(t, 0, b),
                _c_term(t, 1, b),
                _c_term(t, 2, b),
            ]
            if b + 10 < 32:
                terms.append(_w_term(t - 2, b + 10))
            if b + 3 < 32:
                terms.append(_w_term(t - 15, b + 3))
            _push_terms(terms, 0)

    # 压缩轮方程 t = 0..R-1 + 条件线性化 Ch/Maj
    for t in range(r):
        for b in range(32):
            e_t = int(s_sym[t, 4, b])
            f_t = int(s_sym[t, 5, b])
            g_t = int(s_sym[t, 6, b])
            ch_col = _get_col("Ch", t, 0, b)
            ch_term: Tuple[Optional[int], int] = (ch_col, 0)
            if e_t == SYM_1:
                _push_terms([ch_term, _s_canon_e(t - 1, b)], 0)
            elif e_t == SYM_0:
                _push_terms([ch_term, _s_canon_e(t - 2, b)], 0)
            elif f_t in (SYM_0, SYM_1) and g_t in (SYM_0, SYM_1):
                if f_t == g_t:
                    ch_term = (None, 1 if f_t == SYM_1 else 0)
                elif f_t == SYM_1 and g_t == SYM_0:
                    _push_terms([ch_term, _s_canon_e(t, b)], 0)
                else:
                    _push_terms([ch_term, _s_canon_e(t, b)], 1)

            a_t = int(s_sym[t, 0, b])
            b_t = int(s_sym[t, 1, b])
            c_t = int(s_sym[t, 2, b])
            mj_col = _get_col("Maj", t, 0, b)
            mj_term: Tuple[Optional[int], int] = (mj_col, 0)
            if a_t in (SYM_0, SYM_1) and b_t in (SYM_0, SYM_1):
                if a_t == b_t:
                    mj_term = (None, 1 if a_t == SYM_1 else 0)
                else:
                    _push_terms([mj_term, _s_canon_a(t - 2, b)], 0)
            elif a_t in (SYM_0, SYM_1) and c_t in (SYM_0, SYM_1) and a_t != c_t:
                _push_terms([mj_term, _s_canon_a(t - 1, b)], 0)
            elif b_t in (SYM_0, SYM_1) and c_t in (SYM_0, SYM_1) and b_t != c_t:
                _push_terms([mj_term, _s_canon_a(t, b)], 0)

            kbit = (K_WORDS[t] >> b) & 1
            e_terms = [
                _s_canon_e(t + 1, b),
                _s_canon_a(t - 3, b),
                _s_canon_e(t - 3, b),
                _s_canon_e(t, (b + 6) & 31),
                _s_canon_e(t, (b + 11) & 31),
                _s_canon_e(t, (b + 25) & 31),
                ch_term,
                _w_term(t, b),
                _c_term(t, 3, b),
                _c_term(t, 4, b),
                _c_term(t, 5, b),
                _c_term(t, 6, b),
                _c_term(t, 9, b),
            ]
            _push_terms(e_terms, kbit)

            a_terms = [
                _s_canon_a(t + 1, b),
                _s_canon_e(t + 1, b),
                _s_canon_a(t - 3, b),
                _s_canon_a(t, (b + 2) & 31),
                _s_canon_a(t, (b + 13) & 31),
                _s_canon_a(t, (b + 22) & 31),
                mj_term,
                _c_term(t, 7, b),
                _c_term(t, 8, b),
                _c_term(t, 9, b),
            ]
            _push_terms(a_terms, 0)

    n_cols = len(col_map)
    gf2_res = core.solve_gf2_equations(
        num_vars=max(1, n_cols), equations=eqs, return_degrees=True
    ) if eqs else {"conflict": False, "rank": 0, "units": {}, "degrees": np.zeros((max(1, n_cols),), dtype=np.uint32)}
    rank = int(gf2_res["rank"])
    degs = gf2_res["degrees"][:n_cols] if n_cols > 0 else np.zeros((0,), dtype=np.uint32)
    nonzero_degs = degs[degs > 0]
    dof = max(0, n_cols - rank)

    return {
        "total_ch_gates": total_ch,
        "ch_undetermined": ch_undetermined,
        "ch_undetermined_ratio": round(ch_undetermined / max(1, total_ch), 6),
        "ch_nonlinearized": ch_nonlinearized,
        "ch_nonlinearized_ratio": round(ch_nonlinearized / max(1, total_ch), 6),
        "total_maj_gates": total_maj,
        "maj_undetermined": maj_undetermined,
        "maj_undetermined_ratio": round(maj_undetermined / max(1, total_maj), 6),
        "maj_nonlinearized": maj_nonlinearized,
        "maj_nonlinearized_ratio": round(maj_nonlinearized / max(1, total_maj), 6),
        "gf2_residual_vars": n_cols,
        "gf2_residual_equations": len(eqs),
        "gf2_residual_rank": rank,
        "gf2_residual_dof": dof,
        "gf2_residual_units": len(gf2_res["units"]),
        "gf2_var_degree_mean": round(float(np.mean(nonzero_degs)), 3) if nonzero_degs.size else 0.0,
        "gf2_var_degree_max": int(np.max(nonzero_degs)) if nonzero_degs.size else 0,
    }


def _compute_cnf_xor_coupling(enc: SHA256XORCNFEncoder) -> Dict[str, Any]:
    cnf_vars: Set[int] = set()
    xor_vars: Set[int] = set()
    cnf_occ: Dict[int, int] = {}
    xor_occ: Dict[int, int] = {}
    cnf_lits = 0
    xor_lits = 0

    for cl in enc.clauses:
        cnf_lits += len(cl)
        for lit in cl:
            v = abs(int(lit))
            cnf_vars.add(v)
            cnf_occ[v] = cnf_occ.get(v, 0) + 1

    for xv, _rhs in enc.xor_clauses:
        xor_lits += len(xv)
        for v_raw in xv:
            v = abs(int(v_raw))
            xor_vars.add(v)
            xor_occ[v] = xor_occ.get(v, 0) + 1

    inter = cnf_vars & xor_vars
    union = cnf_vars | xor_vars
    return {
        "n_cnf_vars": len(cnf_vars),
        "n_xor_vars": len(xor_vars),
        "n_shared_vars": len(inter),
        "n_union_vars": len(union),
        "jaccard_coupling_ratio": round(len(inter) / max(1, len(union)), 6),
        "shared_over_xor_ratio": round(len(inter) / max(1, len(xor_vars)), 6),
        "cnf_total_literals": cnf_lits,
        "xor_total_literals": xor_lits,
        "cnf_mean_deg_on_shared": round(
            float(np.mean([cnf_occ[v] for v in inter])) if inter else 0.0, 4
        ),
        "xor_mean_deg_on_shared": round(
            float(np.mean([xor_occ[v] for v in inter])) if inter else 0.0, 4
        ),
    }


def run_diag_xor_cliff(
    tiers: Optional[Sequence[Tuple[int, int]]] = None,
    n_cases: int = 4,
    time_limit_sec: float = 3.0,
    seed: int = 20260499,
    fast: bool = False,
    out_json: str = DEFAULT_OUT_JSON,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    fp32_info = _check_fp32_locked()
    core = get_core()
    has_xor = CryptoMiniSatXORSolver.is_available()
    assert has_xor, "CryptoMiniSat (pycryptosat) 不可用！"

    tier_list = list(tiers) if tiers is not None else list(FAST_TIERS if fast else DEFAULT_TIERS)
    n_eff = min(2, n_cases) if fast else int(n_cases)
    tl_eff = min(2.0, time_limit_sec) if fast else float(time_limit_sec)

    print(
        f"[diag_xor_cliff] 启动 | tiers={tier_list} | n_cases={n_eff} | tl={tl_eff}s | fast={fast}",
        flush=True,
    )

    xor_solver = CryptoMiniSatXORSolver(threads=1)
    cad_solver = CaDiCaLCDCLSolver("cadical195")
    rng = np.random.default_rng(seed)

    total_sat_checked = 0
    total_sat_verified = 0
    tier_results: List[Dict[str, Any]] = []
    map_for_assert: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for r_b, fb in tier_list:
        w_mat = _sample_random_words(rng, n_eff, 16)
        cr = core.compress_exact(w_mat, rounds=r_b)

        # 静态门数分解
        carry_comp = 7 * 31 * r_b
        carry_msg = 3 * 31 * max(0, r_b - 16)
        total_carry_gates = carry_comp + carry_msg
        total_ch_maj = 64 * r_b
        total_nonlinear_gates = total_carry_gates + total_ch_maj

        enc_vars_list, cnf_cl_list, xor_cl_list = [], [], []
        bcp_deduced_list, bcp_w_deduced_list, bcp_w_rem_list = [], [], []
        dual_unk_list, dual_carry_list, gf2_rank_list, gf2_ded_list = [], [], [], []
        ch_undet_list, ch_nonlin_list, maj_undet_list, maj_nonlin_list = [], [], [], []
        gf2_dof_list = []
        unk_w_rounds_union: Set[int] = set()
        unk_s_rounds_union: Set[int] = set()
        coupling_list: List[Dict[str, Any]] = []

        cms_sat, cms_times = 0, []
        cad_sat, cad_times, cad_confs = 0, [], []

        for idx in range(n_eff):
            ht = cr.H_target[idx]
            fw, fbits = make_end_free_constraints(w_mat[idx], fb)
            enc = SHA256XORCNFEncoder(rounds=r_b).build(
                H_target=ht, fixed_words=fw, fixed_bits=fbits
            )
            num_vars = int(enc.next_var - 1)
            enc_vars_list.append(num_vars)
            cnf_cl_list.append(len(enc.clauses))
            xor_cl_list.append(len(enc.xor_clauses))
            coupling_list.append(_compute_cnf_xor_coupling(enc))

            # 1) 纯 Python CNF BCP（无 xor_clauses）
            bcp = _run_cnf_bcp(enc.clauses, num_vars)
            assert not bcp["conflict"], f"CNF BCP 发现矛盾 R={r_b}, fb={fb}, idx={idx}"
            bcp_deduced_list.append(int(bcp["deduced_count"]))
            w_det_bcp = 0
            for wi in range(16):
                for bb in range(32):
                    lit_v = enc.W_vars[wi][bb]
                    if isinstance(lit_v, bool) or (isinstance(lit_v, int) and abs(lit_v) in bcp["assigned"]):
                        w_det_bcp += 1
            bcp_w_deduced_list.append(w_det_bcp)
            bcp_w_rem_list.append(512 - w_det_bcp)

            # 2) GeneralizedSymbolEnv 双平面传播 (ARX + GF2)
            env = GeneralizedSymbolEnv(H_target=ht, rounds=r_b, core=core)
            for wi, wv in fw.items():
                u = int(wv) & MASK32
                for bb in range(32):
                    env.set_symbol(0, wi, 0, bb, SYM_1 if ((u >> bb) & 1) else SYM_0, run_propagate=False)
            for (wi, bb), bv in fbits.items():
                env.set_symbol(0, wi, 0, bb, SYM_1 if (bv & 1) else SYM_0, run_propagate=False)
            s_res = env.propagate(use_gf2=True, max_iters=16)
            assert not s_res.conflict, f"DualProp 发现矛盾 R={r_b}, fb={fb}, idx={idx}"

            dual_unk_list.append(int(s_res.unknown_after))
            dual_carry_list.append(int(s_res.active_carry_gates))
            gf2_rank_list.append(int(s_res.gf2_rank))
            gf2_ded_list.append(int(s_res.gf2_deductions))

            for tw in range(r_b):
                if np.any((env.W_sym[tw] != SYM_0) & (env.W_sym[tw] != SYM_1)):
                    unk_w_rounds_union.add(tw)
            for ts in range(r_b + 1):
                if np.any((env.S_sym[ts] != SYM_0) & (env.S_sym[ts] != SYM_1)):
                    unk_s_rounds_union.add(ts)

            cm_stats = _count_active_ch_maj_and_gf2_degrees(core, env)
            ch_undet_list.append(cm_stats["ch_undetermined_ratio"])
            ch_nonlin_list.append(cm_stats["ch_nonlinearized_ratio"])
            maj_undet_list.append(cm_stats["maj_undetermined_ratio"])
            maj_nonlin_list.append(cm_stats["maj_nonlinearized_ratio"])
            gf2_dof_list.append(cm_stats["gf2_residual_dof"])

            # 3) 实测 CryptoMiniSat-XOR vs CaDiCaL-195
            rx = xor_solver.solve(
                ht, rounds=r_b, fixed_words=fw, fixed_bits=fbits, time_limit_sec=tl_eff
            )
            if rx["status"] == "SAT":
                assert rx["valid_preimage"], f"CMS-XOR verify_preimage 失败 R={r_b} fb={fb} idx={idx}"
                cms_sat += 1
                total_sat_checked += 1
                total_sat_verified += 1
            cms_times.append(float(rx["elapsed_sec"]))

            rc = cad_solver.solve(
                ht, rounds=r_b, fixed_words=fw, fixed_bits=fbits,
                conflict_budget=250000, time_limit_sec=tl_eff,
            )
            if rc["status"] == "SAT":
                assert rc["valid_preimage"], f"CaDiCaL verify_preimage 失败 R={r_b} fb={fb} idx={idx}"
                cad_sat += 1
                total_sat_checked += 1
                total_sat_verified += 1
            cad_times.append(float(rc["elapsed_sec"]))
            cad_confs.append(int(rc["conflicts"]))

        mean_active_carry = float(np.mean(dual_carry_list))
        row: Dict[str, Any] = {
            "rounds": r_b,
            "free_bits": fb,
            "cases": n_eff,
            "static_gates": {
                "carry_compress_gates": carry_comp,
                "carry_msg_sched_gates": carry_msg,
                "total_carry_gates": total_carry_gates,
                "ch_gates": 32 * r_b,
                "maj_gates": 32 * r_b,
                "total_nonlinear_gates": total_nonlinear_gates,
            },
            "xor_cnf_encoding": {
                "num_vars": int(np.mean(enc_vars_list)),
                "num_cnf_clauses_mean": round(float(np.mean(cnf_cl_list)), 1),
                "num_xor_clauses_mean": round(float(np.mean(xor_cl_list)), 1),
            },
            "pure_cnf_bcp_only": {
                "deduced_vars_mean": round(float(np.mean(bcp_deduced_list)), 2),
                "w0_15_determined_bits_mean": round(float(np.mean(bcp_w_deduced_list)), 2),
                "w0_15_remaining_free_bits_mean": round(float(np.mean(bcp_w_rem_list)), 2),
            },
            "dual_plane_after_prop": {
                "unknown_after_mean": round(float(np.mean(dual_unk_list)), 2),
                "active_carry_gates_mean": round(mean_active_carry, 2),
                "active_carry_ratio": round(mean_active_carry / max(1, total_carry_gates), 6),
                "ch_undetermined_ratio_mean": round(float(np.mean(ch_undet_list)), 6),
                "ch_nonlinearized_ratio_mean": round(float(np.mean(ch_nonlin_list)), 6),
                "maj_undetermined_ratio_mean": round(float(np.mean(maj_undet_list)), 6),
                "maj_nonlinearized_ratio_mean": round(float(np.mean(maj_nonlin_list)), 6),
                "gf2_rank_mean": round(float(np.mean(gf2_rank_list)), 2),
                "gf2_deductions_mean": round(float(np.mean(gf2_ded_list)), 2),
                "gf2_residual_dof_mean": round(float(np.mean(gf2_dof_list)), 2),
                "unknown_W_rounds": sorted(unk_w_rounds_union),
                "unknown_S_rounds": sorted(unk_s_rounds_union),
            },
            "cnf_xor_coupling": {
                "shared_vars_mean": round(float(np.mean([x["n_shared_vars"] for x in coupling_list])), 1),
                "jaccard_coupling_ratio_mean": round(float(np.mean([x["jaccard_coupling_ratio"] for x in coupling_list])), 6),
                "shared_over_xor_ratio_mean": round(float(np.mean([x["shared_over_xor_ratio"] for x in coupling_list])), 6),
                "cnf_mean_deg_on_shared": round(float(np.mean([x["cnf_mean_deg_on_shared"] for x in coupling_list])), 4),
                "xor_mean_deg_on_shared": round(float(np.mean([x["xor_mean_deg_on_shared"] for x in coupling_list])), 4),
            },
            "solver_cliff_empirical": {
                "cms_xor_sat": cms_sat,
                "cms_xor_sat_rate": round(cms_sat / max(1, n_eff), 4),
                "cms_xor_mean_sec": round(float(np.mean(cms_times)), 4),
                "cms_xor_median_sec": round(float(np.median(cms_times)), 4),
                "cadical_sat": cad_sat,
                "cadical_sat_rate": round(cad_sat / max(1, n_eff), 4),
                "cadical_mean_conflicts": round(float(np.mean(cad_confs)), 2),
                "cadical_mean_sec": round(float(np.mean(cad_times)), 4),
            },
        }
        tier_results.append(row)
        map_for_assert[(r_b, fb)] = row
        print(
            f"  [R={r_b:2d} {fb:2d}b] unk={row['dual_plane_after_prop']['unknown_after_mean']:.0f} | "
            f"active_carry={mean_active_carry:.0f}/{total_carry_gates} "
            f"({row['dual_plane_after_prop']['active_carry_ratio']*100:.1f}%) | "
            f"gf2_dof={row['dual_plane_after_prop']['gf2_residual_dof_mean']:.0f} | "
            f"CMS-XOR={cms_sat}/{n_eff} ({row['solver_cliff_empirical']['cms_xor_mean_sec']:.3f}s) | "
            f"CaDiCaL={cad_sat}/{n_eff} ({row['solver_cliff_empirical']['cadical_mean_conflicts']:.0f}c)",
            flush=True,
        )

    # 核心断崖断言
    if (24, 12) in map_for_assert:
        assert map_for_assert[(24, 12)]["dual_plane_after_prop"]["active_carry_gates_mean"] == 0.0, (
            "预期 R=24 12b 双平面初始传播后活跃进位门为 0"
        )
    if (26, 12) in map_for_assert:
        assert map_for_assert[(26, 12)]["dual_plane_after_prop"]["active_carry_gates_mean"] > 0.0, (
            "预期 R=26 12b 双平面初始传播后活跃进位门 > 0"
        )
    assert total_sat_checked == total_sat_verified

    elapsed = round(time.perf_counter() - t0, 3)
    report: Dict[str, Any] = {
        "diagnostic": "xor_cliff_readonly",
        "fp32_check": fp32_info,
        "seed": int(seed),
        "n_cases_per_tier": n_eff,
        "time_limit_sec": tl_eff,
        "fast_mode": bool(fast),
        "total_sat_verified": total_sat_verified,
        "verify_pass_rate": 1.0 if total_sat_checked > 0 else None,
        "elapsed_sec": elapsed,
        "tiers": tier_results,
    }
    _atomic_write_json(Path(out_json), report)
    print(f"[diag_xor_cliff] 完成，耗时 {elapsed:.2f}s | 写入 {out_json}", flush=True)
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="R2 只读诊断 3：XOR-CDCL 自 R=26 起归零相变断崖")
    p.add_argument("--n_cases", type=int, default=4)
    p.add_argument("--time_limit_sec", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=20260499)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out_json", type=str, default=DEFAULT_OUT_JSON)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_diag_xor_cliff(
        n_cases=args.n_cases,
        time_limit_sec=args.time_limit_sec,
        seed=args.seed,
        fast=bool(args.fast or args.quick),
        out_json=args.out_json,
    )
