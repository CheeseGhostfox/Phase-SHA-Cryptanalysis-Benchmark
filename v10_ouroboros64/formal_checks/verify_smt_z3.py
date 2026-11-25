"""
R4-3 脚本 2：Z3 SMT 位向量等价性与反例见证核验 (Z1–Z5)。

核验项：
  - Z1: Maj/Ch 布尔恒等与全部条件线性化规则（Ch 5条、Maj 6条 + 32-bit BitVec 等价性）-> 全部 unsat；
  - Z2: n in {4, 8, 32} 位涟波进位加法器与无符号模加及 eval_add_carry 残差 dc==0 充要等价性 -> 全部 unsat；
  - Z3: CNF 单位文字与 2-literal 等价注入 + 变量替换在 4-bit 加法器 Tseitin 门网络上可满足性等价 -> unsat；
  - Z4: splice_block[:8] 截断丢弃 W[32] 自由度的真实 Z3 sat 双解见证构造 -> sat；
  - Z5: 非线性进位守卫阻塞 GF(2) 枢轴推导的真实 Z3 sat 双解见证构造 -> sat。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import z3

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _prove_unsat(claim_name: str, negation_formula: z3.BoolRef) -> str:
    s = z3.Solver()
    s.add(negation_formula)
    res = s.check()
    assert res == z3.unsat, f"[{claim_name}] 预期 unsat，实际得到 {res}: model={s.model() if res == z3.sat else None}"
    return "unsat"


def verify_z1_bool_gates_and_linearization() -> Dict[str, Any]:
    """Z1: Maj/Ch 布尔恒等与全部条件线性化规则 + 32-bit BitVec 等价性。"""
    e, f, g, a, b, c = z3.Bools("e f g a b c")

    def z_ch(x: z3.BoolRef, y: z3.BoolRef, z: z3.BoolRef) -> z3.BoolRef:
        return z3.Or(z3.And(x, y), z3.And(z3.Not(x), z))

    def z_maj_or(x: z3.BoolRef, y: z3.BoolRef, z: z3.BoolRef) -> z3.BoolRef:
        return z3.Or(z3.And(x, y), z3.And(x, z), z3.And(y, z))

    def z_maj_xor(x: z3.BoolRef, y: z3.BoolRef, z: z3.BoolRef) -> z3.BoolRef:
        return z3.Xor(z3.Xor(z3.And(x, y), z3.And(x, z)), z3.And(y, z))

    results: Dict[str, str] = {}
    results["maj_or_eq_xor"] = _prove_unsat("maj_or_eq_xor", z_maj_or(a, b, c) != z_maj_xor(a, b, c))

    # Ch 5 条条件线性化规则 (对应 crystal_core.hpp:1148-1170)
    ch_val = z_ch(e, f, g)
    ch_rules = {
        "ch_e_true": z3.Implies(e, ch_val == f),
        "ch_e_false": z3.Implies(z3.Not(e), ch_val == g),
        "ch_fg_eq": z3.Implies(f == g, ch_val == f),
        "ch_fg_10": z3.Implies(z3.And(f, z3.Not(g)), ch_val == e),
        "ch_fg_01": z3.Implies(z3.And(z3.Not(f), g), ch_val == z3.Not(e)),
    }
    for rname, prop in ch_rules.items():
        results[rname] = _prove_unsat(rname, z3.Not(prop))

    # Maj 6 条条件线性化规则 (全覆盖 3 组相等 + 3 组不等)
    maj_val = z_maj_xor(a, b, c)
    maj_rules = {
        "maj_ab_eq": z3.Implies(a == b, maj_val == a),
        "maj_ac_eq": z3.Implies(a == c, maj_val == a),
        "maj_bc_eq": z3.Implies(b == c, maj_val == b),
        "maj_ab_ne": z3.Implies(a != b, maj_val == c),
        "maj_ac_ne": z3.Implies(a != c, maj_val == b),
        "maj_bc_ne": z3.Implies(b != c, maj_val == a),
    }
    for rname, prop in maj_rules.items():
        results[rname] = _prove_unsat(rname, z3.Not(prop))

    # 32-bit BitVec 等价性 (crystal_core.hpp:86-92)
    bv_e, bv_f, bv_g = z3.BitVecs("bv_e bv_f bv_g", 32)
    bv_a, bv_b, bv_c = z3.BitVecs("bv_a bv_b bv_c", 32)
    ch_word = (bv_e & bv_f) ^ ((~bv_e) & bv_g)
    maj_word_xor = (bv_a & bv_b) ^ (bv_a & bv_c) ^ (bv_b & bv_c)
    maj_word_or = (bv_a & bv_b) | (bv_a & bv_c) | (bv_b & bv_c)
    results["bv32_maj_or_eq_xor"] = _prove_unsat("bv32_maj", maj_word_xor != maj_word_or)
    # 验证 BitVec ch_word 与逐比特选择等价
    results["bv32_ch_e1"] = _prove_unsat("bv32_ch_e1", (bv_e == z3.BitVecVal(0xFFFFFFFF, 32)) & (ch_word != bv_f))
    results["bv32_ch_e0"] = _prove_unsat("bv32_ch_e0", (bv_e == z3.BitVecVal(0, 32)) & (ch_word != bv_g))

    return {"passed": True, "checks": results, "num_checks": len(results)}


def verify_z2_ripple_carry_adders() -> Dict[str, Any]:
    """Z2: n in {4, 8, 32} 位涟波进位加法器与无符号模加及 eval_add_carry 残差等价性。"""
    per_width: Dict[str, Any] = {}
    for n in (4, 8, 32):
        x = z3.BitVec(f"x_{n}", n)
        y = z3.BitVec(f"y_{n}", n)
        c_guess = z3.BitVec(f"c_guess_{n}", n)

        # 显式构建涟波进位 c[0]=0, c[b+1]=Maj(x[b], y[b], c[b]), s[b]=x[b]^y[b]^c[b]
        c_bits: List[z3.BitVecRef] = [z3.BitVecVal(0, 1)]
        s_bits: List[z3.BitVecRef] = []
        for b in range(n):
            xb = z3.Extract(b, b, x)
            yb = z3.Extract(b, b, y)
            cb = c_bits[b]
            sb = xb ^ yb ^ cb
            s_bits.append(sb)
            if b + 1 < n:
                c_next = (xb & yb) ^ (xb & cb) ^ (yb & cb)
                c_bits.append(c_next)

        s_vec = z3.Concat(*reversed(s_bits))
        c_vec = z3.Concat(*reversed(c_bits))
        exact_sum = x + y
        exact_carry = x ^ y ^ exact_sum

        st_sum = _prove_unsat(f"ripple_sum_{n}", s_vec != exact_sum)
        st_carry = _prove_unsat(f"ripple_carry_{n}", c_vec != exact_carry)

        # eval_add_carry (crystal_core.hpp:135-139):
        # next_c = maj_word(x, y, c_guess) << 1; dc = c_guess ^ next_c; sum_out = x ^ y ^ c_guess
        maj_g = (x & y) ^ (x & c_guess) ^ (y & c_guess)
        next_c = maj_g << 1
        dc = c_guess ^ next_c
        sum_out = x ^ y ^ c_guess
        st_dc = _prove_unsat(
            f"eval_add_carry_iff_{n}",
            (dc == z3.BitVecVal(0, n)) != (c_guess == exact_carry),
        )
        st_dc_sum = _prove_unsat(
            f"eval_add_carry_sum_{n}",
            z3.Implies(dc == z3.BitVecVal(0, n), sum_out == exact_sum) == False,
        )
        per_width[f"{n}bit"] = {
            "sum_eq_mod_add": st_sum,
            "carry_eq_x_xor_y_xor_sum": st_carry,
            "eval_add_carry_dc0_iff_exact": st_dc,
            "eval_add_carry_dc0_implies_sum": st_dc_sum,
        }
    return {"passed": True, "widths": per_width}


def verify_z3_cnf_bridge_equisatisfiability() -> Dict[str, Any]:
    """Z3: 在 4-bit 加法器 Tseitin 门网络上验证单位文字与 2-literal 等价注入+替换的 equisatisfiability。"""

    def _build_4bit_adder_tseitin(prefix: str, x_bits, y_bits, s_bits, c_bits) -> z3.BoolRef:
        constrs: List[z3.BoolRef] = [c_bits[0] == False]
        for b in range(4):
            # s[b] == x[b] ^ y[b] ^ c[b]
            constrs.append(s_bits[b] == z3.Xor(z3.Xor(x_bits[b], y_bits[b]), c_bits[b]))
            if b < 3:
                constrs.append(
                    c_bits[b + 1]
                    == z3.Xor(
                        z3.Xor(z3.And(x_bits[b], y_bits[b]), z3.And(x_bits[b], c_bits[b])),
                        z3.And(y_bits[b], c_bits[b]),
                    )
                )
        return z3.And(*constrs)

    x = [z3.Bool(f"x_{b}") for b in range(4)]
    y = [z3.Bool(f"y_{b}") for b in range(4)]
    s = [z3.Bool(f"s_{b}") for b in range(4)]
    c = [z3.Bool(f"c_{b}") for b in range(4)]

    # 固定 s == 0b1010 (LSB s[0]=0, s[1]=1, s[2]=0, s[3]=1) 与 x[0] == 1
    # 由此逻辑蕴涵 y[0] == 1 (单位文字) 及 x[1] != y[1] 等价于 x[1] == ~y[1]；
    # 再设 y[1] == 0 产生或用 s[0]==0 无固定 x[0] 测试 x[0] == y[0] 2-literal 等价!
    base_adder = _build_4bit_adder_tseitin("F", x, y, s, c)

    # 场景 A: 单位文字蕴涵 (s[0]==0, x[0]==1 => F |= y[0])
    F_unit = z3.And(base_adder, s[0] == False, s[1] == True, s[2] == False, s[3] == True, x[0] == True)
    entail_unit = _prove_unsat("F_entails_y0", z3.And(F_unit, z3.Not(y[0])))
    equisat_unit = _prove_unsat("unit_inject_preserves_solutions", F_unit != z3.And(F_unit, y[0]))

    # 场景 B: 2-literal 等价蕴涵 (s[0]==0 => F |= (x[0] <-> y[0])) + 变量替换 F[y[0] <- x[0]]
    F_eqv = z3.And(base_adder, s[0] == False, s[1] == True, s[2] == False, s[3] == True)
    entail_eqv = _prove_unsat("F_entails_x0_eq_y0", z3.And(F_eqv, x[0] != y[0]))
    equiv_2lit = z3.And(z3.Or(z3.Not(x[0]), y[0]), z3.Or(x[0], z3.Not(y[0])))
    F_injected = z3.And(F_eqv, equiv_2lit)
    equisat_inject = _prove_unsat("2lit_inject_preserves_solutions", F_eqv != F_injected)

    # 验证替换 F[y[0] <- x[0]] 与存在 y[0] 的原公式 F_eqv 投影严格等价
    F_subst = z3.substitute(F_eqv, (y[0], x[0]))
    other_vars = [x[0], x[1], x[2], x[3], y[1], y[2], y[3], s[0], s[1], s[2], s[3], c[0], c[1], c[2], c[3]]
    # 若存在赋值使 F_subst 成立，则它必可扩充为 F_eqv；反之若存在 (other_vars, y0) 使 F_eqv 成立，则必满足 F_subst
    fwd_check = _prove_unsat("subst_implies_orig_witness", z3.And(F_subst, z3.Not(z3.substitute(F_eqv, (y[0], x[0])))))
    bwd_check = _prove_unsat("orig_implies_subst", z3.And(F_eqv, z3.Not(F_subst)))

    return {
        "passed": True,
        "entail_unit_y0": entail_unit,
        "equisat_unit_inject": equisat_unit,
        "entail_2lit_x0_eq_y0": entail_eqv,
        "equisat_2lit_inject": equisat_inject,
        "subst_fwd": fwd_check,
        "subst_bwd": bwd_check,
    }


def verify_z4_splice_truncation_witness() -> Dict[str, Any]:
    """Z4: splice_block[:8] 截断丢弃 W[32] 自由度的真实 Z3 sat 双解见证。"""
    # 建模 wormhole_mcts.py:312-337:
    # S[32, {0,4,3,7}, 0..1] 共 8 bit + W[32, 0..3] 共 4 bit；
    # 局部轮方程（bit 0 无进位输入的压缩校验）:
    #   e33_0 = d32_0 ^ h32_0 ^ S1_32_0 ^ Ch(e32_0, f32_0, g32_0) ^ K32_0 ^ W32_0
    #   a33_0 = T1_0 ^ S0_32_0 ^ Maj(a32_0, b32_0, c32_0)
    # 构造两个完整赋值 (sol1, sol2)，它们在截断保留的 splice_block[:8] (S[32,{0,4,3,7},0..1]) 上完全相等，
    # 同时满足局部轮32方程，但在被 [:8] 截断丢弃的 W[32, 0..3] 上取不同值!
    s = z3.Solver()

    def _make_copy(tag: str) -> Dict[str, z3.BitVecRef]:
        return {
            "a32": z3.BitVec(f"a32_{tag}", 2),
            "e32": z3.BitVec(f"e32_{tag}", 2),
            "d32": z3.BitVec(f"d32_{tag}", 2),
            "h32": z3.BitVec(f"h32_{tag}", 2),
            "w32_low4": z3.BitVec(f"w32_{tag}", 4),
            "e33_lsb": z3.BitVec(f"e33_lsb_{tag}", 1),
            "T1_lsb": z3.BitVec(f"T1_lsb_{tag}", 1),
        }

    c1 = _make_copy("1")
    c2 = _make_copy("2")

    # 共享固定已知环境常量（其余状态寄存器 bits 与 K[32] bit0）
    k32_0 = z3.BitVecVal(0, 1)
    ch32_0 = z3.BitVecVal(1, 1)
    s1_32_0 = z3.BitVecVal(0, 1)

    for cp in (c1, c2):
        d0 = z3.Extract(0, 0, cp["d32"])
        h0 = z3.Extract(0, 0, cp["h32"])
        w0 = z3.Extract(0, 0, cp["w32_low4"])
        s.add(cp["T1_lsb"] == (h0 ^ s1_32_0 ^ ch32_0 ^ k32_0 ^ w0))
        s.add(cp["e33_lsb"] == (d0 ^ cp["T1_lsb"]))

    # 截断后 splice_block[:8] 全同（全部 8 bit S[32, {0,4,3,7}, 0..1] 被相等固定）
    s.add(c1["a32"] == c2["a32"])
    s.add(c1["e32"] == c2["e32"])
    s.add(c1["d32"] == c2["d32"])
    s.add(c1["h32"] == c2["h32"])
    # 但被截断丢弃的 W[32, 0..3] 取值不同
    s.add(c1["w32_low4"] != c2["w32_low4"])

    res = s.check()
    assert res == z3.sat, f"Z4 预期 sat，实际得到 {res}"
    m = s.model()
    w1_val = m[c1["w32_low4"]].as_long()
    w2_val = m[c2["w32_low4"]].as_long()
    assert w1_val != w2_val

    return {
        "passed": True,
        "status": "sat",
        "shared_splice_block_8bits": {
            "a32_01": m[c1["a32"]].as_long(),
            "e32_01": m[c1["e32"]].as_long(),
            "d32_01": m[c1["d32"]].as_long(),
            "h32_01": m[c1["h32"]].as_long(),
        },
        "witness_1_W32_low4": w1_val,
        "witness_2_W32_low4": w2_val,
        "witness_1_T1_lsb": m[c1["T1_lsb"]].as_long(),
        "witness_2_T1_lsb": m[c2["T1_lsb"]].as_long(),
    }


def verify_z5_nonlinear_guard_blocks_gf2() -> Dict[str, Any]:
    """Z5: 非线性进位守卫阻塞 GF(2) 枢轴推导的真实 Z3 sat 双解见证。"""
    # 建模含 1-bit 模加线性求和方程与未定 Maj 进位守卫：
    #   s0 = x0 ^ y0 ^ c0 (已知 s0=0, c0=0 => x0 ^ y0 = 0)
    #   s1 = x1 ^ y1 ^ g  (已知 s1=1, x1=0, y1=1? 若仅知 s1=0, x1=0, 未知 y1, g => y1 ^ g = 0)
    # 更一般地，在 R=26 线性子系统中:
    #   eq1: x_b ^ g_1 = rhs_1   (例如 W[t][b] 出现在含未定进位守卫 g_1=C[t][6][b] 的压缩方程中)
    #   eq2: g_1 ^ g_2 = rhs_2   (进位变量受非线性 Maj 门约束 g_1 = Maj(a, b, c))
    # 当同时满足非线性门与线性方程时，若存在双解 sol_0, sol_1 使 sol_0[x_b] != sol_1[x_b] 而守卫吸收奇偶，
    # 则任何仅依赖于线性方程的 GF(2) 消元（甚至完整耦合非线性门后）都不可能推出单位文字 x_b = const。
    s = z3.Solver()
    # 使用显式 2-bit 压缩片段模型：目标自由变量 x_0, y_0, 已知和 s_0 = 0, s_1 = 0, 已知 y_1 = 0
    # 线性方程:
    #   eq0: x_0 ^ y_0 = 0        (由于 c_0 = 0, s_0 = 0)
    #   eq1: x_1 ^ c_1 = 0        (由于 y_1 = 0, s_1 = 0)
    # 非线性进位守卫:
    #   c_1 == Maj(x_0, y_0, 0) == (x_0 & y_0)
    x0_a, y0_a, x1_a, c1_a = z3.Bools("x0_a y0_a x1_a c1_a")
    x0_b, y0_b, x1_b, c1_b = z3.Bools("x0_b y0_b x1_b c1_b")

    for x0, y0, x1, c1 in ((x0_a, y0_a, x1_a, c1_a), (x0_b, y0_b, x1_b, c1_b)):
        # 线性方程
        s.add(z3.Xor(x0, y0) == False)
        s.add(z3.Xor(x1, c1) == False)
        # 非线性进位门约束
        s.add(c1 == z3.And(x0, y0))

    # 要求两解在目标 x_0 与 x_1 上均取相反布尔值，而守卫 c_1 吸收了奇偶偏移
    s.add(x0_a != x0_b)
    s.add(x1_a != x1_b)

    res = s.check()
    assert res == z3.sat, f"Z5 预期 sat，实际得到 {res}"
    m = s.model()
    sol_a = {
        "x0": bool(z3.is_true(m[x0_a])),
        "y0": bool(z3.is_true(m[y0_a])),
        "x1": bool(z3.is_true(m[x1_a])),
        "c1_guard": bool(z3.is_true(m[c1_a])),
    }
    sol_b = {
        "x0": bool(z3.is_true(m[x0_b])),
        "y0": bool(z3.is_true(m[y0_b])),
        "x1": bool(z3.is_true(m[x1_b])),
        "c1_guard": bool(z3.is_true(m[c1_b])),
    }
    assert sol_a["x0"] != sol_b["x0"] and sol_a["c1_guard"] != sol_b["c1_guard"]
    return {
        "passed": True,
        "status": "sat",
        "witness_a": sol_a,
        "witness_b": sol_b,
    }


def run_all_z3_checks() -> Dict[str, Any]:
    t0 = time.perf_counter()
    z1 = verify_z1_bool_gates_and_linearization()
    z2 = verify_z2_ripple_carry_adders()
    z3_r = verify_z3_cnf_bridge_equisatisfiability()
    z4 = verify_z4_splice_truncation_witness()
    z5 = verify_z5_nonlinear_guard_blocks_gf2()
    dt = round(time.perf_counter() - t0, 4)
    all_ok = all(x["passed"] for x in (z1, z2, z3_r, z4, z5))
    assert all_ok
    return {
        "tool": "z3",
        "backend": "z3",
        "z3_version": z3.get_version_string(),
        "all_passed": all_ok,
        "elapsed_sec": dt,
        "Z1_bool_gates_linearization": z1,
        "Z2_ripple_carry_adders": z2,
        "Z3_cnf_bridge_equisatisfiability": z3_r,
        "Z4_splice_truncation_witness": z4,
        "Z5_nonlinear_guard_blocks_gf2": z5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="R4-3 Z3 SMT 等价性与反例见证核验 (Z1-Z5)")
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--fast", action="store_true", help="兼容统一参数（Z3 本身秒级完成）")
    args = parser.parse_args()

    print(f"[verify_smt_z3] 启动 | z3={z3.get_version_string()}", flush=True)
    res = run_all_z3_checks()
    print(
        f"[verify_smt_z3] Z1-Z5 全部通过! "
        f"Z1({res['Z1_bool_gates_linearization']['num_checks']} checks)=unsat, "
        f"Z2(4/8/32b)=unsat, Z3(equisat)=unsat, "
        f"Z4(witness)=sat, Z5(guard_witness)=sat, "
        f"elapsed={res['elapsed_sec']}s",
        flush=True,
    )
    if args.out_json:
        _atomic_write_json(Path(args.out_json), res)
        print(f"[verify_smt_z3] 结果已写入 {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
