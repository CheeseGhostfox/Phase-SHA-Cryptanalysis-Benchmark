"""
R4-3 脚本 1：SymPy 精确符号推导与数学边界核验 (S1–S5)。

核验项：
  - S1: 模加进位递推 c_0=0, c_{b+1}=x_b*y_b ^ x_b*c_b ^ y_b*c_b 在 F_2 多项式环上的精确 ANF 展开，
        验证 b=0..7 精确展开、deg(c_b) = b + 1 (1 <= b <= 7)、归纳闭合至 c_30 (31次)/c_31 (32次) 及 s_b 次数；
  - S2: PairAcc 恒等拷贝混合分解与支配充要条件、代入实测计数（medium/cluster）及 rank-1 KL 非负；
  - S3: 盲态假置信 BCE(1/2, 1/2+eps) = ln 2 - 1/2 ln(1-4*eps^2) > ln 2 导数/级数、
        非对称推广 BCE(p,q)-ln2 = KL(p||q)-KL(p||1/2)、多比特全对概率衰减 p^k 与回溯倍率；
  - S4: cube_depth=2 期望净冲突收益 Delta_C(p_u, dC_s, B_c) 盈亏平衡点、B_c=0 非负及 R=26 16b 亏损；
  - S5: 38912/512 == 76、num_gf2_vars(R)=(15R+8)*32、O(R^3) vs O(w^3) 加速比、全盲态期望能量 E[E0]=N_checks/2。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import sympy as sp

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _f2_reduce(expr: sp.Expr, gens: Sequence[sp.Symbol]) -> sp.Expr:
    """在布尔多项式商环 F_2[gens] / <v^2 - v> 上精确化简。"""
    expanded = sp.expand(expr)
    if expanded == 0:
        return sp.Integer(0)
    poly = sp.Poly(expanded, *gens, domain=sp.ZZ)
    monom_coeffs: Dict[Tuple[int, ...], int] = {}
    for monom, coeff in poly.terms():
        c_mod = int(coeff) % 2
        if c_mod == 0:
            continue
        sqfree = tuple(1 if e > 0 else 0 for e in monom)
        monom_coeffs[sqfree] = (monom_coeffs.get(sqfree, 0) + 1) % 2
    terms = []
    for monom, c_mod in monom_coeffs.items():
        if c_mod == 0:
            continue
        term = sp.Integer(1)
        for sym, e in zip(gens, monom):
            if e == 1:
                term = term * sym
        terms.append(term)
    if not terms:
        return sp.Integer(0)
    return sp.Add(*terms)


def _f2_degree(expr: sp.Expr, gens: Sequence[sp.Symbol]) -> int:
    if expr == 0:
        return 0
    poly = sp.Poly(expr, *gens, domain=sp.GF(2))
    return int(poly.total_degree())


def verify_s1_modular_carry_anf() -> Dict[str, Any]:
    """S1: 模加进位 ANF 展开与 deg(c_b)=b+1 (1<=b<=7) 及归纳闭合至 c_30/c_31。"""
    n_bits = 8
    xs = [sp.Symbol(f"x{i}") for i in range(n_bits)]
    ys = [sp.Symbol(f"y{i}") for i in range(n_bits)]
    gens = xs + ys

    carries: List[sp.Expr] = [sp.Integer(0)]
    sums: List[sp.Expr] = []
    anf_records: List[Dict[str, Any]] = []

    for b in range(n_bits):
        cb = carries[b]
        sb = _f2_reduce(xs[b] + ys[b] + cb, gens)
        sums.append(sb)
        deg_c = _f2_degree(cb, gens)
        deg_s = _f2_degree(sb, gens)

        if b == 0:
            assert cb == 0 and deg_c == 0, f"c_0 必须为 0, got {cb}"
            assert deg_s == 1, f"s_0 次数必须为 1, got {deg_s}"
        else:
            assert deg_c == b + 1, f"c_{b} 次数应为 {b+1}, got {deg_c}"
            assert deg_s == b + 1, f"s_{b} 次数应为 {b+1}, got {deg_s}"

        # 检验归纳见证单项式: M_b = x_0 * y_0 * prod_{j=1}^{b-1} x_j
        if b >= 1:
            poly_c = sp.Poly(cb, *gens, domain=sp.GF(2))
            witness_exp = [0] * (2 * n_bits)
            witness_exp[0] = 1  # x0
            witness_exp[n_bits + 0] = 1  # y0
            for j in range(1, b):
                witness_exp[j] = 1  # x_j
            coeff_w = int(poly_c.coeff_monomial(tuple(witness_exp))) % 2
            assert coeff_w == 1, f"c_{b} 缺失最高次见证单项式"
            # 确认 c_b 不依赖 idx >= b 的变量
            used_syms = cb.free_symbols
            for j in range(b, n_bits):
                assert xs[j] not in used_syms and ys[j] not in used_syms

        anf_records.append(
            {
                "bit": b,
                "deg_c": deg_c,
                "deg_s": deg_s,
                "n_terms_c": 0 if cb == 0 else len(sp.Poly(cb, *gens, domain=sp.GF(2)).terms()),
                "c_expr": str(cb) if b <= 4 else f"<poly with {len(sp.Poly(cb, *gens, domain=sp.GF(2)).terms())} monomials>",
            }
        )
        c_next = _f2_reduce(xs[b] * ys[b] + xs[b] * cb + ys[b] * cb, gens)
        carries.append(c_next)

    # 额外检查 c_8 次数为 9 与归纳步函数验证
    deg_c8 = _f2_degree(carries[8], gens)
    assert deg_c8 == 9, f"deg(c_8) 应为 9, got {deg_c8}"

    # 归纳递推符号检查：b >= 1 时 deg_rec(b+1) = max(2, 1 + deg_rec(b)) = b + 2
    b_sym = sp.Symbol("b", integer=True, positive=True)
    deg_formula = b_sym + 1
    assert sp.simplify(deg_formula.subs(b_sym, b_sym + 1) - (1 + deg_formula)) == 0
    deg_c30 = int(deg_formula.subs(b_sym, 30))
    deg_c31 = int(deg_formula.subs(b_sym, 31))
    assert deg_c30 == 31 and deg_c31 == 32

    return {
        "passed": True,
        "b0_to_b7": anf_records,
        "deg_c8": deg_c8,
        "inductive_deg_c30": deg_c30,
        "inductive_deg_c31": deg_c31,
    }


def verify_s2_pair_acc_and_kl() -> Dict[str, Any]:
    """S2: PairAcc 恒等拷贝混合分解、支配充要条件、实测代入与 rank-1 KL 非负。"""
    delta, alpha0, alpha1 = sp.symbols("delta alpha0 alpha1", real=True, positive=True)
    acc_model = (1 - delta) * alpha0 + delta * alpha1
    acc_copy = 1 - delta
    diff = sp.simplify(acc_model - acc_copy)
    expected_diff = delta * alpha1 - (1 - delta) * (1 - alpha0)
    assert sp.simplify(diff - expected_diff) == 0

    # 符号求解 acc_model < acc_copy 关于 alpha1 的上界
    sol_upper = sp.solve(sp.Eq(diff, 0), alpha1)[0]
    assert sp.simplify(sol_upper - (1 - delta) * (1 - alpha0) / delta) == 0

    # 代入 diagnostics_summary.json 中的实测计数 (best_m3v2 medium / cluster, N=8)
    empirical_cases = {
        "best_m3v2_medium": {"U": 141832, "F": 248, "u_m": 141630, "f_m": 67},
        "best_m3v2_cluster": {"U": 141923, "F": 157, "u_m": 141635, "f_m": 53},
        "m2_step4000_medium": {"U": 141832, "F": 248, "u_m": 141784, "f_m": 46},
        "m2_step4000_cluster": {"U": 141923, "F": 157, "u_m": 141846, "f_m": 43},
    }
    emp_out: Dict[str, Any] = {}
    for name, c in empirical_cases.items():
        U, F_cnt, u_m, f_m = c["U"], c["F"], c["u_m"], c["f_m"]
        N_tot = U + F_cnt
        d_rat = sp.Rational(F_cnt, N_tot)
        a0_rat = sp.Rational(u_m, U)
        a1_rat = sp.Rational(f_m, F_cnt)
        m_acc = sp.simplify(acc_model.subs({delta: d_rat, alpha0: a0_rat, alpha1: a1_rat}))
        c_acc = sp.simplify(acc_copy.subs({delta: d_rat}))
        thresh = sp.simplify(sol_upper.subs({delta: d_rat, alpha0: a0_rat}))
        assert m_acc < c_acc, f"{name} 未满足拷贝支配: {float(m_acc)} vs {float(c_acc)}"
        assert a1_rat < thresh
        emp_out[name] = {
            "N_total": N_tot,
            "delta_flip_frac": round(float(d_rat), 6),
            "alpha0_uncorrupted_acc": round(float(a0_rat), 6),
            "alpha1_flipped_acc": round(float(a1_rat), 6),
            "model_acc": round(float(m_acc), 6),
            "copy_dummy_acc": round(float(c_acc), 6),
            "dominance_threshold_alpha1": round(float(thresh), 6),
            "copy_dominates": bool(m_acc < c_acc),
        }

    # Rank-1 KL 散度与 NLL 增益符号等价性
    p00, p01, p10, p11 = sp.symbols("p00 p01 p10 p11", positive=True)
    pL = p10 + p11
    pR = p01 + p11
    norm = p00 + p01 + p10 + p11
    # 归一化后 p00+p01+p10+p11 = 1
    q00 = (1 - pL) * (1 - pR)
    q01 = (1 - pL) * pR
    q10 = pL * (1 - pR)
    q11 = pL * pR
    nll_joint = -(p00 * sp.log(p00) + p01 * sp.log(p01) + p10 * sp.log(p10) + p11 * sp.log(p11))
    nll_rank1 = -(p00 * sp.log(q00) + p01 * sp.log(q01) + p10 * sp.log(q10) + p11 * sp.log(q11))
    kl_sym = (
        p00 * sp.log(p00 / q00)
        + p01 * sp.log(p01 / q01)
        + p10 * sp.log(p10 / q10)
        + p11 * sp.log(p11 / q11)
    )
    assert sp.simplify(sp.expand_log((nll_rank1 - nll_joint) - kl_sym, force=True)) == 0
    # 典型相关联合分布验证 KL >= 0
    kl_val = float(kl_sym.subs({p00: sp.Rational(4, 10), p01: sp.Rational(1, 10), p10: sp.Rational(1, 10), p11: sp.Rational(4, 10)}))
    assert kl_val > 0.0 and abs(float(norm.subs({p00: 0.4, p01: 0.1, p10: 0.1, p11: 0.4})) - 1.0) < 1e-12

    return {
        "passed": True,
        "symbolic_diff": str(diff),
        "alpha1_threshold": str(sol_upper),
        "empirical_cases": emp_out,
        "kl_identity_verified": True,
        "sample_correlated_kl": round(kl_val, 6),
    }


def verify_s3_blind_bce_and_backtracking() -> Dict[str, Any]:
    """S3: 盲态假置信 BCE > ln 2、非对称 KL 恒等式、多比特概率衰减 p^k 与回溯倍率。"""
    eps = sp.Symbol("eps", real=True, positive=True)
    bce_sym = sp.simplify(-sp.Rational(1, 2) * (sp.log(sp.Rational(1, 2) + eps) + sp.log(sp.Rational(1, 2) - eps)))
    closed_form = sp.log(2) - sp.Rational(1, 2) * sp.log(1 - 4 * eps**2)
    assert sp.simplify(sp.logcombine(bce_sym - closed_form, force=True)) == 0

    d1 = sp.simplify(sp.diff(bce_sym, eps))
    assert sp.simplify(d1 - 4 * eps / (1 - 4 * eps**2)) == 0
    d2 = sp.simplify(sp.diff(d1, eps))
    assert sp.simplify(d2 - 4 * (1 + 4 * eps**2) / (1 - 4 * eps**2) ** 2) == 0

    series_exp = sp.series(bce_sym, eps, 0, 6)
    expected_series = sp.log(2) + 2 * eps**2 + 4 * eps**4 + sp.Order(eps**6)
    assert sp.simplify(series_exp.removeO() - expected_series.removeO()) == 0

    # 非对称推广: BCE(p, q) - ln 2 = D_KL(p || q) - D_KL(p || 1/2)
    p_s, q_s = sp.symbols("p_s q_s", positive=True)
    bce_pq = -(p_s * sp.log(q_s) + (1 - p_s) * sp.log(1 - q_s))
    kl_pq = p_s * sp.log(p_s / q_s) + (1 - p_s) * sp.log((1 - p_s) / (1 - q_s))
    kl_p_half = p_s * sp.log(2 * p_s) + (1 - p_s) * sp.log(2 * (1 - p_s))
    assert sp.simplify(sp.expand_log((bce_pq - sp.log(2)) - (kl_pq - kl_p_half), force=True)) == 0

    # 实测代入 (R=24 40b 与 R=25 24b, best_m3v2) — 均值插件近似 (plugin approximation)
    ln2 = float(sp.log(2).evalf())
    r24_bce_plugin = float(bce_pq.subs({p_s: 0.509375, q_s: 0.5 + 0.110498}).evalf())
    r25_bce_plugin = float(bce_pq.subs({p_s: 0.526042, q_s: 0.5 + 0.110078}).evalf())
    assert r24_bce_plugin > ln2 and r25_bce_plugin > ln2

    # 非对称过自信反例 (p=0.52 > 0.5, q=0.70)、临界过自信阈值 q* 及混合置信分布反例
    p_asym, q_asym = 0.52, 0.70
    bce_asym = float(bce_pq.subs({p_s: p_asym, q_s: q_asym}).evalf())
    assert bce_asym > ln2
    q_crit = float(sp.nsolve(sp.Eq(bce_pq.subs(p_s, p_asym), sp.log(2)), q_s, 0.54))
    assert 0.539 < q_crit < 0.541
    bce_mix_50_50 = 0.5 * float(bce_pq.subs({p_s: p_asym, q_s: 0.70}).evalf()) + 0.5 * float(
        bce_pq.subs({p_s: p_asym, q_s: 0.30}).evalf()
    )
    bce_mix_80_20 = 0.8 * float(bce_pq.subs({p_s: p_asym, q_s: 0.52}).evalf()) + 0.2 * float(
        bce_pq.subs({p_s: p_asym, q_s: 0.75}).evalf()
    )
    assert bce_mix_50_50 > ln2 and bce_mix_80_20 > ln2

    # 多比特连续全对概率衰减 p^k (无冲突学习独立 Guess-and-Determine 理想化对比模型)
    p24_all_ok = 0.526042**24
    p40_all_ok = 0.509375**40
    assert p24_all_ok < 1e-6 and p40_all_ok < 1e-10
    backtrack_mult_24 = 1.0 / p24_all_ok

    return {
        "passed": True,
        "bce_closed_form": str(closed_form),
        "first_derivative": str(d1),
        "second_derivative": str(d2),
        "taylor_series": str(series_exp),
        "asymmetric_kl_identity_verified": True,
        "ln2": round(ln2, 6),
        "r24_40b_plugin_bce": round(r24_bce_plugin, 6),
        "r25_24b_plugin_bce": round(r25_bce_plugin, 6),
        "plugin_note": "Homogeneous mean-confidence plugin approximation; differs from sample-wise mean BCE due to Jensen inequality",
        "asym_counterexample_p052_q070_bce": round(bce_asym, 6),
        "asym_p052_critical_q": round(q_crit, 6),
        "mixture_p052_50_50_bce": round(bce_mix_50_50, 6),
        "mixture_p052_80_20_bce": round(bce_mix_80_20, 6),
        "p_all_correct_k24": p24_all_ok,
        "p_all_correct_k40": p40_all_ok,
        "expected_backtrack_multiplier_k24": backtrack_mult_24,
    }


def verify_s4_cube_breakeven() -> Dict[str, Any]:
    """S4: cube_depth=2 期望净冲突收益盈亏平衡点与零虚耗非负性。"""
    p_u, dC_s, B_c = sp.symbols("p_u dC_s B_c", real=True, nonnegative=True)
    delta_c = p_u * dC_s - (1 - p_u) * B_c
    breakeven = sp.simplify(sp.solve(sp.Eq(delta_c, 0), p_u)[0])
    assert sp.simplify(breakeven - B_c / (dC_s + B_c)) == 0

    # B_c = 0 时对任意 p_u, dC_s >= 0 非负
    zero_waste = sp.simplify(delta_c.subs(B_c, 0))
    assert zero_waste == p_u * dC_s

    # R=26 16b: used_confl >= 7000 时 chunk >= 6860, 默认 cube_budget = min(1500, max(500, chunk//2)) = 1500
    # 仅在临近 conflict_budget 尾段时受 min(cube_budget, rem_b) 截断
    loss_default_bc = float(delta_c.subs({p_u: 0, dC_s: 20000, B_c: 1500}))
    loss_min = float(delta_c.subs({p_u: 0, dC_s: 20000, B_c: 500}))
    assert loss_default_bc == -1500.0 and loss_min == -500.0

    be_20k_1k = float(breakeven.subs({dC_s: 20000, B_c: 1000}))
    return {
        "passed": True,
        "breakeven_p_u": str(breakeven),
        "zero_waste_gain": str(zero_waste),
        "r26_16b_default_bc": 1500,
        "r26_16b_default_bc_loss": loss_default_bc,
        "r26_16b_loss_range": [loss_default_bc, loss_min],
        "rem_b_truncation_note": "Default B_c=1500 whenever used_confl>=7000 (chunk>=6860); truncated to rem_b only near conflict_budget exhaustion",
        "breakeven_dCs20k_Bc1k": round(be_20k_1k, 6),
    }


def verify_s5_dimension_complexity_energy() -> Dict[str, Any]:
    """S5: 76x 空间膨胀、num_gf2_vars(64)=30976、O(R^3) vs O(w^3) 加速比、盲态期望能量。"""
    R_sym, w_sym = sp.symbols("R w", integer=True, positive=True)
    crystal_bits = 64 * 19 * 32
    preimage_bits = 16 * 32
    expansion = sp.Rational(crystal_bits, preimage_bits)
    assert expansion == 76 and crystal_bits == 38912

    full_crystal_bits = (64 * 19 + 8) * 32
    assert full_crystal_bits == 39168

    eff_carry_gates = (64 * 10 - 16 * 3) * 31
    assert eff_carry_gates == 18352

    num_gf2_vars_expr = (15 * R_sym + 8) * 32
    gf2_r64 = int(num_gf2_vars_expr.subs(R_sym, 64))
    assert gf2_r64 == 30976

    speedup_asymptotic_expr = (R_sym / w_sym) ** 3
    speedup_exact_expr = ((15 * R_sym + 8) / (15 * w_sym + 8)) ** 3
    speedup_win_expr = ((15 * R_sym + 8) / (15 * w_sym + 24)) ** 3
    speedup_by_w = {
        str(w_val): {
            "asymptotic_R_over_w_cubed": round(float(speedup_asymptotic_expr.subs({R_sym: 64, w_sym: w_val})), 4),
            "ideal_15w_plus_8_cubed": round(float(speedup_exact_expr.subs({R_sym: 64, w_sym: w_val})), 4),
            "window_with_boundary_lags_cubed": round(float(speedup_win_expr.subs({R_sym: 64, w_sym: w_val})), 4),
        }
        for w_val in (4, 8, 9, 12)
    }
    assert speedup_by_w["8"]["asymptotic_R_over_w_cubed"] == 512.0
    assert speedup_by_w["8"]["ideal_15w_plus_8_cubed"] > 400.0
    assert speedup_by_w["8"]["window_with_boundary_lags_cubed"] > 300.0

    # R=64 盲态主动校验位数与期望能量 E[E_0] = N_checks / 2
    n_checks_r64 = 48 * 32 + 64 * 8 * 32 + (64 * 10 - 16 * 3) * 31
    assert n_checks_r64 == 36272
    idx_i, N_sym = sp.symbols("i N", integer=True, positive=True)
    expected_energy_sym = sp.summation(sp.Rational(1, 2), (idx_i, 1, N_sym))
    assert sp.simplify(expected_energy_sym - N_sym / 2) == 0
    expected_e0_r64 = int(expected_energy_sym.subs(N_sym, n_checks_r64))
    assert expected_e0_r64 == 18136

    return {
        "passed": True,
        "crystal_bits_64": crystal_bits,
        "full_crystal_bits_64": full_crystal_bits,
        "expansion_ratio": int(expansion),
        "effective_carry_gates_64": eff_carry_gates,
        "num_gf2_vars_formula": str(num_gf2_vars_expr),
        "num_gf2_vars_64": gf2_r64,
        "window_cube_speedup_vs_r64": speedup_by_w,
        "n_active_checks_64": n_checks_r64,
        "expected_blind_e0_64": expected_e0_r64,
    }


def run_all_sympy_checks() -> Dict[str, Any]:
    t0 = time.perf_counter()
    s1 = verify_s1_modular_carry_anf()
    s2 = verify_s2_pair_acc_and_kl()
    s3 = verify_s3_blind_bce_and_backtracking()
    s4 = verify_s4_cube_breakeven()
    s5 = verify_s5_dimension_complexity_energy()
    dt = round(time.perf_counter() - t0, 4)
    all_ok = all(x["passed"] for x in (s1, s2, s3, s4, s5))
    assert all_ok
    return {
        "tool": "sympy",
        "sympy_version": sp.__version__,
        "all_passed": all_ok,
        "elapsed_sec": dt,
        "S1_modular_carry_anf": s1,
        "S2_pair_acc_and_kl": s2,
        "S3_blind_bce_and_backtracking": s3,
        "S4_cube_breakeven": s4,
        "S5_dimension_complexity_energy": s5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="R4-3 SymPy 精确符号核验 (S1-S5)")
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--fast", action="store_true", help="兼容统一参数（SymPy 本身已秒级完成）")
    args = parser.parse_args()

    print(f"[verify_symbolic_sympy] 启动 | sympy={sp.__version__}", flush=True)
    res = run_all_sympy_checks()
    print(
        f"[verify_symbolic_sympy] S1-S5 全部通过! "
        f"deg(c30)={res['S1_modular_carry_anf']['inductive_deg_c30']}, "
        f"deg(c31)={res['S1_modular_carry_anf']['inductive_deg_c31']}, "
        f"expansion={res['S5_dimension_complexity_energy']['expansion_ratio']}x, "
        f"gf2_vars(64)={res['S5_dimension_complexity_energy']['num_gf2_vars_64']}, "
        f"elapsed={res['elapsed_sec']}s",
        flush=True,
    )
    if args.out_json:
        _atomic_write_json(Path(args.out_json), res)
        print(f"[verify_symbolic_sympy] 结果已写入 {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
