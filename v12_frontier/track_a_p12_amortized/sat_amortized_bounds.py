"""
P12 A3 精确 SAT / Z3 最大摊还界求解器与谱隙对齐对照实验。

严格遵循 g:\\SHA-LM\\v12_frontier\\track_a_p12_amortized\\SPEC.md §3.1 & §3.2：
1. 使用 Z3 (BitVecSort(32)) 与 PySAT (CaDiCaL195) 精确求解：
   - B1 场景 1：同 midstate1，仅允许 W1[3] 或 W1[0..3] 差分（固定 BTC 填充 W1[4..15]），
     求解维持全状态零差（及部分字零差）的最大轮数 R；
   - B1 场景 2：允许 midstate1 与 W1[0..3] 联合差分补偿（Delta Header != 0），
     求解维持全状态零差 S_R == S'_R 的最大轮数 R，以及进入 W1[16..] 消息扩展后保持零扩展差分的界；
   - B2 (Hash2 界面与最小活跃门下界)：
     * 求解给定 Delta W2[0..7] != 0 时，使 Hash2 消息扩展 W2[16..20] 保持零差的最少非零字与最小汉明重；
     * 在 R in {4, 6, 8, 10, 12} 窗口上使用 PySAT (CaDiCaL195) 求解 Chunk2 与 Hash2 的最小活跃比特/字门数下界（UNSAT 证明不可能低于该下界）。
2. 在 MAIN_SHA256 与 3 组谱隙对齐随机对照 CTRL_RAND_ROTR_K 上以完全相同预算同步运行。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pysat
import z3
from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Cadical195

from btc_double_sha_engine import (
    MASK32,
    SHA256Variant,
    make_main_variant,
    make_spectral_aligned_ctrl_variant,
)


# ---------------------------------------------------------------------------
# Z3 BitVec(32) 精确编码
# ---------------------------------------------------------------------------

def _z3_rotr(x: z3.BitVecRef, n: int) -> z3.BitVecRef:
    return x if n == 0 else z3.RotateRight(x, n)


def _z3_shr(x: z3.BitVecRef, n: int) -> z3.BitVecRef:
    return z3.LShR(x, n)


def _z3_ch(e: z3.BitVecRef, f: z3.BitVecRef, g: z3.BitVecRef) -> z3.BitVecRef:
    return (e & f) ^ ((~e) & g)


def _z3_maj(a: z3.BitVecRef, b: z3.BitVecRef, c: z3.BitVecRef) -> z3.BitVecRef:
    return (a & b) ^ (a & c) ^ (b & c)


def _z3_S0(x: z3.BitVecRef, v: SHA256Variant) -> z3.BitVecRef:
    return _z3_rotr(x, v.S0_rot[0]) ^ _z3_rotr(x, v.S0_rot[1]) ^ _z3_rotr(x, v.S0_rot[2])


def _z3_S1(x: z3.BitVecRef, v: SHA256Variant) -> z3.BitVecRef:
    return _z3_rotr(x, v.S1_rot[0]) ^ _z3_rotr(x, v.S1_rot[1]) ^ _z3_rotr(x, v.S1_rot[2])


def _z3_s0(x: z3.BitVecRef, v: SHA256Variant) -> z3.BitVecRef:
    return _z3_rotr(x, v.s0_rot[0]) ^ _z3_rotr(x, v.s0_rot[1]) ^ _z3_shr(x, v.s0_shr)


def _z3_s1(x: z3.BitVecRef, v: SHA256Variant) -> z3.BitVecRef:
    return _z3_rotr(x, v.s1_rot[0]) ^ _z3_rotr(x, v.s1_rot[1]) ^ _z3_shr(x, v.s1_shr)


def encode_z3_compress(
    solver: z3.Solver,
    prefix: str,
    s0: List[z3.BitVecRef],
    w16: List[z3.BitVecRef],
    rounds: int,
    variant: SHA256Variant,
) -> Tuple[List[z3.BitVecRef], List[z3.BitVecRef], List[z3.BitVecRef]]:
    """在 Z3 中编码单块 rounds 轮精确压缩，返回 (A_cols, E_cols, W_cols)。"""
    w = list(w16[:min(16, rounds)])
    for t in range(16, rounds):
        wt = z3.BitVec(f"{prefix}_W_{t}", 32)
        solver.add(wt == _z3_s1(w[t - 2], variant) + w[t - 7] + _z3_s0(w[t - 15], variant) + w[t - 16])
        w.append(wt)

    # A_cols: [-3, -2, -1, 0, 1..R], E_cols: [-3..R]
    a = [s0[3], s0[2], s0[1], s0[0]]
    e = [s0[7], s0[6], s0[5], s0[4]]
    for t in range(rounds):
        kt = z3.BitVecVal(variant.k[t] & MASK32, 32)
        t1 = e[t] + _z3_S1(e[t + 3], variant) + _z3_ch(e[t + 3], e[t + 2], e[t + 1]) + kt + w[t]
        t2 = _z3_S0(a[t + 3], variant) + _z3_maj(a[t + 3], a[t + 2], a[t + 1])
        a_new = z3.BitVec(f"{prefix}_A_{t+1}", 32)
        e_new = z3.BitVec(f"{prefix}_E_{t+1}", 32)
        solver.add(e_new == a[t] + t1)
        solver.add(a_new == t1 + t2)
        a.append(a_new)
        e.append(e_new)
    return a, e, w


def solve_z3_b1_bounds(variant: SHA256Variant, timeout_ms: int = 15000) -> Dict[str, Any]:
    """
    求解 SPEC §3.1 界限问题 B1（Chunk2 内部及带 midstate1 补偿最大零差轮数）：
    - Scenario 1a: 同 midstate1, 仅 W1[3] != W1'[3], 固定 W1[0..2] 与 BTC 填充 -> 测试 R=4..12；
    - Scenario 1b: 同 midstate1, W1[0..3] != W1'[0..3], 固定 BTC 填充 -> 测试 R=4..12；
    - Scenario 2: midstate1 != midstate1' 或 W1[0..3] != W1'[0..3] 联合补偿 -> 测试 S_R == S'_R 及 W1[16..R-1] == W1'[16..R-1]！
    """
    results: Dict[str, Any] = {}

    def _run_case(case_name: str, same_midstate: bool, only_w3_diff: bool, rounds: int, require_w_exp_zero: bool = False) -> Dict[str, Any]:
        t0 = time.time()
        slv = z3.Solver()
        slv.set("timeout", timeout_ms)

        ms_a = [z3.BitVec(f"msa_{i}", 32) for i in range(8)]
        ms_b = list(ms_a) if same_midstate else [z3.BitVec(f"msb_{i}", 32) for i in range(8)]

        w_a = [z3.BitVec(f"wa_{i}", 32) for i in range(4)]
        if only_w3_diff:
            w_b = [w_a[0], w_a[1], w_a[2], z3.BitVec("wb_3", 32)]
        else:
            w_b = [z3.BitVec(f"wb_{i}", 32) for i in range(4)]

        # 固定合法 BTC Chunk2 320-bit 填充
        pad = [z3.BitVecVal(0x80000000, 32)] + [z3.BitVecVal(0, 32)] * 10 + [z3.BitVecVal(640, 32)]
        w16_a = w_a + pad
        w16_b = w_b + pad

        a1, e1, w1 = encode_z3_compress(slv, "T1", ms_a, w16_a, rounds, variant)
        a2, e2, w2 = encode_z3_compress(slv, "T2", ms_b, w16_b, rounds, variant)

        # 非平凡输入条件
        diff_inputs = [w_a[i] != w_b[i] for i in range(4)]
        if not same_midstate:
            diff_inputs.extend(ms_a[i] != ms_b[i] for i in range(8))
        slv.add(z3.Or(*diff_inputs))

        # 目标：在第 R 轮末输出状态 S_R 全零差 (A_{R-3..R}, E_{R-3..R} 全部相等)
        for idx in range(rounds, rounds + 4):
            slv.add(a1[idx] == a2[idx])
            slv.add(e1[idx] == e2[idx])

        if require_w_exp_zero and rounds > 16:
            for t in range(16, rounds):
                slv.add(w1[t] == w2[t])

        chk = slv.check()
        return {
            "case": case_name,
            "rounds": rounds,
            "status": str(chk).upper(),
            "solve_seconds": round(time.time() - t0, 3),
        }

    # B1 Scenario 1a: 同 midstate, 仅 W1[3] 变 (注入 t=3 -> R=4 必不等于)
    s1a = [_run_case("B1_S1a_sameMS_onlyW3", True, True, r) for r in (4, 5, 6)]
    # B1 Scenario 1b: 同 midstate, W1[0..3] 自由补偿
    s1b = [_run_case("B1_S1b_sameMS_W0123", True, False, r) for r in (4, 5, 6, 8, 10, 12)]
    # B1 Scenario 2: midstate + W1[0..3] 补偿（若在 R=16 仍碰撞，跨入 t>=16 能否保持 W1[16..19] 零差？）
    s2 = [_run_case("B1_S2_varMS_W0123", False, False, r, require_w_exp_zero=(r > 16)) for r in (4, 8, 12, 16, 18, 20)]

    results["B1_Scenario1a_same_midstate_only_W3"] = s1a
    results["B1_Scenario1b_same_midstate_W0123"] = s1b
    results["B1_Scenario2_var_midstate_plus_W0123"] = s2
    return results


def solve_z3_b2_hash2_interface(variant: SHA256Variant, timeout_ms: int = 15000) -> Dict[str, Any]:
    """
    求解 SPEC §3.1 界限问题 B2（Hash2 接口 Delta W2[0..7] != 0 下的消息扩展零差最小约束）：
    若要使 W2[16..16+k-1] 全部零差，Delta W2[0..7] 至少需多少非零字 / 多少比特汉明重。
    """
    records: List[Dict[str, Any]] = []
    for k_exp in (1, 2, 3, 4, 5, 6, 7, 8):  # 对应 W2[16..16+k-1]
        t0 = time.time()
        opt = z3.Optimize()
        opt.set("timeout", timeout_ms)

        w_a = [z3.BitVec(f"w2a_{i}", 32) for i in range(8)]
        w_b = [z3.BitVec(f"w2b_{i}", 32) for i in range(8)]
        pad = [z3.BitVecVal(0x80000000, 32)] + [z3.BitVecVal(0, 32)] * 6 + [z3.BitVecVal(256, 32)]
        wa_full = w_a + pad
        wb_full = w_b + pad

        opt.add(z3.Or(*[w_a[i] != w_b[i] for i in range(8)]))

        for t in range(16, 16 + k_exp):
            wt_a = _z3_s1(wa_full[t - 2], variant) + wa_full[t - 7] + _z3_s0(wa_full[t - 15], variant) + wa_full[t - 16]
            wt_b = _z3_s1(wb_full[t - 2], variant) + wb_full[t - 7] + _z3_s0(wb_full[t - 15], variant) + wb_full[t - 16]
            wa_full.append(wt_a)
            wb_full.append(wt_b)
            opt.add(wt_a == wt_b)

        nz_words = z3.Sum([z3.If(w_a[i] != w_b[i], 1, 0) for i in range(8)])
        opt.minimize(nz_words)
        chk = opt.check()
        min_nz = int(opt.model().eval(nz_words).as_long()) if chk == z3.sat else -1
        records.append(
            {
                "hash2_zero_exp_steps": f"W2[16..{15+k_exp}]",
                "status": str(chk).upper(),
                "min_nonzero_input_words_in_W2_0_7": min_nz,
                "solve_seconds": round(time.time() - t0, 3),
            }
        )
    # 额外检查：强制 Delta W2[0] != 0 且 W2[16..23] 全零差 (即必须取消前驱影响)
    t0 = time.time()
    slv_w0 = z3.Solver()
    slv_w0.set("timeout", timeout_ms)
    wa0 = [z3.BitVec(f"w2z0a_{i}", 32) for i in range(8)]
    wb0 = [z3.BitVec(f"w2z0b_{i}", 32) for i in range(8)]
    pad0 = [z3.BitVecVal(0x80000000, 32)] + [z3.BitVecVal(0, 32)] * 6 + [z3.BitVecVal(256, 32)]
    waf, wbf = wa0 + pad0, wb0 + pad0
    slv_w0.add(wa0[0] != wb0[0])
    for t in range(16, 24):
        wta = _z3_s1(waf[t - 2], variant) + waf[t - 7] + _z3_s0(waf[t - 15], variant) + waf[t - 16]
        wtb = _z3_s1(wbf[t - 2], variant) + wbf[t - 7] + _z3_s0(wbf[t - 15], variant) + wbf[t - 16]
        waf.append(wta)
        wbf.append(wtb)
        slv_w0.add(wta == wtb)
    chk0 = slv_w0.check()
    records.append(
        {
            "hash2_zero_exp_steps": "W2[16..23]_with_DeltaW2_0_nz",
            "status": str(chk0).upper(),
            "min_nonzero_input_words_in_W2_0_7": 8 if chk0 == z3.sat else -1,
            "solve_seconds": round(time.time() - t0, 3),
        }
    )
    return {"hash2_expander_zero_diff_bounds": records}


# ---------------------------------------------------------------------------
# PySAT (CaDiCaL195) 比特级 Tseitin 编码与最小活跃门下界求解
# ---------------------------------------------------------------------------

class SHA256SATPairBuilder:
    """将双轨 SHA-256（含差分活跃标志位）精确编译为 CNF 并用 CaDiCaL195 求解。"""

    def __init__(self) -> None:
        self.vpool = IDPool()
        self.clauses: List[List[int]] = []

    def new_var(self, name: Optional[str] = None) -> int:
        return self.vpool.id(name) if name else self.vpool._next()

    def new_vec(self, prefix: str) -> List[int]:
        return [self.new_var(f"{prefix}_{b}") for b in range(32)]

    def const_vec(self, val: int) -> List[int]:
        out: List[int] = []
        for b in range(32):
            v = self.new_var()
            if (val >> b) & 1:
                self.clauses.append([v])
            else:
                self.clauses.append([-v])
            out.append(v)
        return out

    def add_xor2(self, a: int, b: int, out: int) -> None:
        self.clauses.extend([
            [-a, -b, -out],
            [a, b, -out],
            [a, -b, out],
            [-a, b, out],
        ])

    def add_xor3(self, a: int, b: int, c: int, out: int) -> None:
        t = self.new_var()
        self.add_xor2(a, b, t)
        self.add_xor2(t, c, out)

    def add_and2(self, a: int, b: int, out: int) -> None:
        self.clauses.extend([
            [-out, a],
            [-out, b],
            [out, -a, -b],
        ])

    def add_or2(self, a: int, b: int, out: int) -> None:
        self.clauses.extend([
            [out, -a],
            [out, -b],
            [-out, a, b],
        ])

    def add_ch_bit(self, e: int, f: int, g: int, out: int) -> None:
        # out = (e & f) | (~e & g)
        self.clauses.extend([
            [-e, -f, out],
            [-e, f, -out],
            [e, -g, out],
            [e, g, -out],
        ])

    def add_maj_bit(self, a: int, b: int, c: int, out: int) -> None:
        self.clauses.extend([
            [-a, -b, out],
            [-a, -c, out],
            [-b, -c, out],
            [a, b, -out],
            [a, c, -out],
            [b, c, -out],
        ])

    def add_fa(self, a: int, b: int, cin: int, s: int, cout: int) -> None:
        self.add_xor3(a, b, cin, s)
        self.add_maj_bit(a, b, cin, cout)

    def add_ha(self, a: int, b: int, s: int, cout: int) -> None:
        self.add_xor2(a, b, s)
        self.add_and2(a, b, cout)

    def add_add32(self, x: List[int], y: List[int]) -> List[int]:
        out = [self.new_var() for _ in range(32)]
        c_in = self.new_var()
        self.add_ha(x[0], y[0], out[0], c_in)
        for b in range(1, 31):
            c_out = self.new_var()
            self.add_fa(x[b], y[b], c_in, out[b], c_out)
            c_in = c_out
        self.add_xor3(x[31], y[31], c_in, out[31])
        return out

    def rotr(self, x: List[int], n: int) -> List[int]:
        if n == 0:
            return list(x)
        return [x[(b + n) % 32] for b in range(32)]

    def shr(self, x: List[int], n: int, zero_bit: int) -> List[int]:
        return [x[b + n] if (b + n) < 32 else zero_bit for b in range(32)]

    def sigma0_u(self, x: List[int], v: SHA256Variant) -> List[int]:
        r0, r1, r2 = self.rotr(x, v.S0_rot[0]), self.rotr(x, v.S0_rot[1]), self.rotr(x, v.S0_rot[2])
        out = [self.new_var() for _ in range(32)]
        for b in range(32):
            self.add_xor3(r0[b], r1[b], r2[b], out[b])
        return out

    def sigma1_u(self, x: List[int], v: SHA256Variant) -> List[int]:
        r0, r1, r2 = self.rotr(x, v.S1_rot[0]), self.rotr(x, v.S1_rot[1]), self.rotr(x, v.S1_rot[2])
        out = [self.new_var() for _ in range(32)]
        for b in range(32):
            self.add_xor3(r0[b], r1[b], r2[b], out[b])
        return out

    def sigma0_l(self, x: List[int], v: SHA256Variant, zero_bit: int) -> List[int]:
        r0, r1, sh = self.rotr(x, v.s0_rot[0]), self.rotr(x, v.s0_rot[1]), self.shr(x, v.s0_shr, zero_bit)
        out = [self.new_var() for _ in range(32)]
        for b in range(32):
            self.add_xor3(r0[b], r1[b], sh[b], out[b])
        return out

    def sigma1_l(self, x: List[int], v: SHA256Variant, zero_bit: int) -> List[int]:
        r0, r1, sh = self.rotr(x, v.s1_rot[0]), self.rotr(x, v.s1_rot[1]), self.shr(x, v.s1_shr, zero_bit)
        out = [self.new_var() for _ in range(32)]
        for b in range(32):
            self.add_xor3(r0[b], r1[b], sh[b], out[b])
        return out

    def ch32_vec(self, e: List[int], f: List[int], g: List[int]) -> List[int]:
        out = [self.new_var() for _ in range(32)]
        for b in range(32):
            self.add_ch_bit(e[b], f[b], g[b], out[b])
        return out

    def maj32_vec(self, a: List[int], b_vec: List[int], c: List[int]) -> List[int]:
        out = [self.new_var() for _ in range(32)]
        for b in range(32):
            self.add_maj_bit(a[b], b_vec[b], c[b], out[b])
        return out

    def encode_compress(
        self,
        prefix: str,
        s0: List[List[int]],
        w16: List[List[int]],
        rounds: int,
        variant: SHA256Variant,
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        zero_bit = self.new_var(f"{prefix}_ZERO")
        self.clauses.append([-zero_bit])

        w = list(w16[:min(16, rounds)])
        for t in range(16, rounds):
            s1_w = self.sigma1_l(w[t - 2], variant, zero_bit)
            s0_w = self.sigma0_l(w[t - 15], variant, zero_bit)
            sum1 = self.add_add32(s1_w, w[t - 7])
            sum2 = self.add_add32(s0_w, w[t - 16])
            wt = self.add_add32(sum1, sum2)
            w.append(wt)

        a = [s0[3], s0[2], s0[1], s0[0]]
        e = [s0[7], s0[6], s0[5], s0[4]]
        for t in range(rounds):
            kt = self.const_vec(variant.k[t])
            s1 = self.sigma1_u(e[t + 3], variant)
            ch = self.ch32_vec(e[t + 3], e[t + 2], e[t + 1])
            p1 = self.add_add32(e[t], s1)
            p2 = self.add_add32(p1, ch)
            p3 = self.add_add32(p2, kt)
            t1 = self.add_add32(p3, w[t])

            s0_u = self.sigma0_u(a[t + 3], variant)
            mj = self.maj32_vec(a[t + 3], a[t + 2], a[t + 1])
            t2 = self.add_add32(s0_u, mj)

            e_new = self.add_add32(a[t], t1)
            a_new = self.add_add32(t1, t2)
            a.append(a_new)
            e.append(e_new)
        return a, e, w


def solve_pysat_min_active_bounds(
    variant: SHA256Variant,
    rounds_list: Sequence[int] = (4, 6, 8, 10, 12),
    conf_budget: int = 1_000_000,
) -> List[Dict[str, Any]]:
    """
    使用 PySAT Cadical195 求解在 Chunk2 (仅改 W1[3]) 与 Hash2 (改 W2[0..7] / W2[0], 固定 IV) 上，
    在指定轮数窗口 R 内维持非平凡输入差时，活跃状态字数 (active_words in {A_t, E_t}) 的精确下界。
    带有冲突数上限 conf_budget 的安全求解，绝不无限挂起。
    """
    records: List[Dict[str, Any]] = []
    for stage in ("Chunk2_onlyW3", "Hash2_W0_inj0", "Hash2_W0to7"):
        for r in rounds_list:
            t0 = time.time()
            bld = SHA256SATPairBuilder()

            if stage == "Chunk2_onlyW3":
                # R <= 15 时 t=0..2 在双轨完全一致，直接从 t=3 注入点编码（等价且 CNF 减半）
                s3 = [bld.new_vec(f"s3_{i}") for i in range(8)]
                w3_a = bld.new_vec("w3_a")
                w3_b = bld.new_vec("w3_b")
                pad = [bld.const_vec(0x80000000)] + [bld.const_vec(0) for _ in range(10)] + [bld.const_vec(640)]
                w_tail_a = [w3_a] + pad[: max(0, r - 4)]
                w_tail_b = [w3_b] + pad[: max(0, r - 4)]
                # 使用自定义偏移轮 t=3..r-1
                a1 = [s3[3], s3[2], s3[1], s3[0]]
                e1 = [s3[7], s3[6], s3[5], s3[4]]
                a2 = list(a1)
                e2 = list(e1)
                for idx_r in range(r - 3):
                    t_abs = 3 + idx_r
                    kt = bld.const_vec(variant.k[t_abs])
                    wt_a = w_tail_a[idx_r]
                    wt_b = w_tail_b[idx_r]
                    for a_arr, e_arr, wt in ((a1, e1, wt_a), (a2, e2, wt_b)):
                        base = idx_r
                        s1 = bld.sigma1_u(e_arr[base + 3], variant)
                        ch = bld.ch32_vec(e_arr[base + 3], e_arr[base + 2], e_arr[base + 1])
                        p1 = bld.add_add32(e_arr[base], s1)
                        p2 = bld.add_add32(p1, ch)
                        p3 = bld.add_add32(p2, kt)
                        t1 = bld.add_add32(p3, wt)
                        s0_u = bld.sigma0_u(a_arr[base + 3], variant)
                        mj = bld.maj32_vec(a_arr[base + 3], a_arr[base + 2], a_arr[base + 1])
                        t2 = bld.add_add32(s0_u, mj)
                        e_arr.append(bld.add_add32(a_arr[base], t1))
                        a_arr.append(bld.add_add32(t1, t2))
                diff_in = [bld.new_var() for _ in range(32)]
                for b in range(32):
                    bld.add_xor2(w3_a[b], w3_b[b], diff_in[b])
                bld.clauses.append(diff_in)
                inj_round = 3
                post_pairs = [
                    (f"A_{3 + idx_r + 1}", a1[4 + idx_r], a2[4 + idx_r], f"E_{3 + idx_r + 1}", e1[4 + idx_r], e2[4 + idx_r])
                    for idx_r in range(r - 3)
                ]
            else:
                iv = [bld.const_vec(x) for x in variant.iv]
                if stage == "Hash2_W0_inj0":
                    w0_a = bld.new_vec("w2_0a")
                    w0_b = bld.new_vec("w2_0b")
                    w1to7 = [bld.new_vec(f"w2_eq_{i}") for i in range(1, 8)]
                    pad = [bld.const_vec(0x80000000)] + [bld.const_vec(0) for _ in range(6)] + [bld.const_vec(256)]
                    w16_a = [w0_a] + w1to7 + pad
                    w16_b = [w0_b] + w1to7 + pad
                    diff_in = [bld.new_var() for _ in range(32)]
                    for b in range(32):
                        bld.add_xor2(w0_a[b], w0_b[b], diff_in[b])
                    bld.clauses.append(diff_in)
                else:
                    w_a = [bld.new_vec(f"w2a_{i}") for i in range(8)]
                    w_b = [bld.new_vec(f"w2b_{i}") for i in range(8)]
                    pad = [bld.const_vec(0x80000000)] + [bld.const_vec(0) for _ in range(6)] + [bld.const_vec(256)]
                    w16_a = w_a + pad
                    w16_b = w_b + pad
                    diff_in_all: List[int] = []
                    for wi in range(8):
                        for b in range(32):
                            dv = bld.new_var()
                            bld.add_xor2(w_a[wi][b], w_b[wi][b], dv)
                            diff_in_all.append(dv)
                    bld.clauses.append(diff_in_all)
                a1, e1, _ = bld.encode_compress("P", iv, w16_a, r, variant)
                a2, e2, _ = bld.encode_compress("Q", iv, w16_b, r, variant)
                inj_round = 0
                post_pairs = [
                    (f"A_{t+1}", a1[4 + t], a2[4 + t], f"E_{t+1}", e1[4 + t], e2[4 + t])
                    for t in range(r)
                ]

            active_word_vars: List[int] = []
            for tag_a, va1, va2, tag_e, ve1, ve2 in post_pairs:
                for tag, v1, v2 in ((tag_a, va1, va2), (tag_e, ve1, ve2)):
                    act = bld.new_var(f"act_{tag}")
                    bit_diffs: List[int] = []
                    for b in range(32):
                        bd = bld.new_var()
                        bld.add_xor2(v1[b], v2[b], bd)
                        bld.clauses.append([-bd, act])
                        bit_diffs.append(bd)
                    bld.clauses.append([-act] + bit_diffs)
                    active_word_vars.append(act)

            # 对每个 k = 0..max_k 带 conf_budget 独立求解
            proven_unsat_up_to = -1
            min_sat_k: Optional[int] = None
            solver_status = "EXACT_MIN_PROVEN"
            max_k = len(active_word_vars)
            for k in range(0, max_k + 1):
                if k == 0:
                    extra_clauses = [[-v] for v in active_word_vars]
                else:
                    card_pool = IDPool(start_from=bld.vpool.top + 1)
                    card = CardEnc.atmost(lits=active_word_vars, bound=k, vpool=card_pool, encoding=EncType.seqcounter)
                    extra_clauses = card.clauses
                with Cadical195(bootstrap_with=bld.clauses + extra_clauses) as slv:
                    slv.conf_budget(conf_budget)
                    res = slv.solve_limited()
                    if res is True:
                        min_sat_k = k
                        break
                    elif res is False:
                        proven_unsat_up_to = k
                    else:
                        solver_status = "BUDGET_LOWER_BOUND"
                        break

            records.append(
                {
                    "stage": stage,
                    "rounds": r,
                    "inj_round": inj_round,
                    "total_post_inj_state_words": len(active_word_vars),
                    "min_active_state_words": min_sat_k if min_sat_k is not None else (proven_unsat_up_to + 1),
                    "unsat_below_bound": proven_unsat_up_to,
                    "solver_status": solver_status,
                    "solve_seconds": round(time.time() - t0, 3),
                }
            )
            print(
                f"[A3-PySAT] {variant.name} {stage} R={r:2d}: status={solver_status} "
                f"min_active={records[-1]['min_active_state_words']}/{len(active_word_vars)} "
                f"(UNSAT<={proven_unsat_up_to}) in {records[-1]['solve_seconds']}s",
                flush=True,
            )
    return records


def run_full_a3(is_smoke: bool = False, out_path: Optional[Path] = None) -> Dict[str, Any]:
    t0 = time.time()
    main_var = make_main_variant()
    ctrl_vars = [
        make_spectral_aligned_ctrl_variant(seed=20261125 + s, min_gap=4, name=f"CTRL_RAND_ROTR_K_S{s}")
        for s in range(3)
    ]
    variants = [main_var] + ctrl_vars

    rounds_sat = (4, 6) if is_smoke else (4, 6, 8, 10, 12)
    z3_timeout = 2500 if is_smoke else 8000
    conf_budget = 150_000 if is_smoke else 600_000

    bounds_by_variant: Dict[str, Any] = {}
    for var in variants:
        tv = time.time()
        print(f"[A3] Solving Z3 + Cadical195 exact bounds for {var.name} ...", flush=True)
        b1_z3 = solve_z3_b1_bounds(var, timeout_ms=z3_timeout)
        b2_z3 = solve_z3_b2_hash2_interface(var, timeout_ms=z3_timeout)
        pysat_bounds = solve_pysat_min_active_bounds(var, rounds_list=rounds_sat, conf_budget=conf_budget)
        bounds_by_variant[var.name] = {
            "Z3_B1_chunk2": b1_z3,
            "Z3_B2_hash2_interface": b2_z3,
            "PySAT_min_active_words": pysat_bounds,
            "elapsed_seconds": round(time.time() - tv, 2),
        }

    payload = {
        "solver_info": {
            "z3_version": z3.get_version_string(),
            "pysat_version": getattr(pysat, "__version__", "unknown"),
            "sat_backend": "CaDiCaL195",
            "is_smoke": is_smoke,
            "total_elapsed_seconds": round(time.time() - t0, 2),
        },
        "bounds": bounds_by_variant,
        "new_identities": [],
    }
    if out_path is None:
        out_path = Path(__file__).resolve().parent / "results" / "a3_sat_bounds.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[A3] Saved -> {out_path}", flush=True)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    out_p = Path(args.out) if args.out else None
    run_full_a3(is_smoke=args.smoke, out_path=out_p)


if __name__ == "__main__":
    main()
