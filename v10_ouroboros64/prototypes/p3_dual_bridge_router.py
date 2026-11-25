"""
Phase-SHA V10「衔尾蛇-64」M3 原型 P3：双平面预处理等价桥接与 Carry-Lock 感知混合路由 (`p3_dual_bridge_router.py`)。

核心组件：
  1. BridgeableSHA256XORCNFEncoder：
     - 完整镜像 SHA256CNFEncoder.build()（严禁 hook _emit_maj），显式保存 self.Ch_vars 与 self.Maj_vars；
     - 保证 num_vars (next_var)、clauses、xor_clauses 与原生 SHA256XORCNFEncoder 1:1 完全等价。
  2. SignedUnionFind & DualPlaneEquivalenceBridge：
     - 调用 GeneralizedSymbolEnv.propagate(use_gf2=True, max_iters=16) 完成 C++ 双平面闭合；
     - 前置 `isinstance(lit, bool)` 拦截映射单位文字并检测常数冲突；
     - 基于 Python 任意精度 int 位向量 RREF（涵盖 Ch 5 条 + Maj 6 条共 11 条规则）抽取补充单位与 2-literal 等价类；
     - SignedUnionFind 检测奇校验环与常数冲突（early_unsat=True），导出等价 2-CNF 子句；
     - 精确计算 carry_lock_rate 与 gf2_residual_dof。
  3. SafeCMSSubprocessRunner：
     - 常驻单 Worker 子进程隔离运行 pycryptosat（长度前缀 pickle 帧 + stderr=DEVNULL + 队列超时强杀 + 崩溃自动重启），
       杜绝 Windows 0xC0000005 拖垮主进程并规避重复 spawn 启动延迟。
  4. CarryLockHybridRouter：
     - 三区动态路由（Zone 1 -> CMS-XOR；Zone 2 -> 短探针 min(0.6s, 0.20*tl) 回落 P1+P2；Zone 3 -> 直入 P1+P2）；
     - 统一返回 DynamicSolveResult，SAT 100% 过 verify_preimage。
"""

from __future__ import annotations

import os
import pickle
import queue
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from ..ouroboros_core.cnf_cdcl import (
    Lit,
    SHA256XORCNFEncoder,
    _rotr_lits,
    _shr_lits,
    _word_const_lits,
    extract_w0_15_from_cryptosat,
)
from ..ouroboros_core.engine import (
    C_PLANES,
    IV_WORDS,
    K_WORDS,
    MASK32,
    SYM_0,
    SYM_1,
    OuroborosCore,
    get_core,
)
from ..ouroboros_mcts.dynamic_cdcl import DynamicSolveResult
from ..ouroboros_mcts.symbol_env import GeneralizedSymbolEnv
from ..ouroboros_net.model import Ouroboros64Net
from .p1_clean_cdcl import CleanDynamicNeuralCDCLSolver


class BridgeableSHA256XORCNFEncoder(SHA256XORCNFEncoder):
    """
    可桥接 XOR-CNF 编码器：
    通过完整镜像 SHA256CNFEncoder.build() 并在 Ch/Maj 分配点记录 self.Ch_vars 与 self.Maj_vars，
    确保变量分配顺序、clauses 与 xor_clauses 与原生 SHA256XORCNFEncoder 1:1 完全一致。
    """

    def __init__(self, rounds: int = 64) -> None:
        super().__init__(rounds=rounds)
        self.Ch_vars: List[List[int]] = []
        self.Maj_vars: List[List[int]] = []

    def build(
        self,
        H_target: Sequence[int],
        fixed_words: Optional[Dict[int, int]] = None,
        fixed_bits: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> "BridgeableSHA256XORCNFEncoder":
        self.xor_clauses.clear()
        self.clauses.clear()
        self.Ch_vars.clear()
        self.Maj_vars.clear()
        fixed = fixed_words or {}

        # Weld boundary S[0] == IV -> a[-3..0] = IV[3,2,1,0], e[-3..0] = IV[7,6,5,4]
        for w_idx in range(4):
            self.a_canon[-w_idx] = _word_const_lits(IV_WORDS[w_idx])
            self.e_canon[-w_idx] = _word_const_lits(IV_WORDS[4 + w_idx])

        # Weld target boundary S[R] == (H_target - IV) mod 2^32
        sr = [((int(H_target[j]) - IV_WORDS[j]) & MASK32) for j in range(8)]
        for w_idx in range(4):
            self.a_canon[self.rounds - w_idx] = _word_const_lits(sr[w_idx])
            self.e_canon[self.rounds - w_idx] = _word_const_lits(sr[4 + w_idx])

        # Allocate any non-boundary canonical a[t], e[t] for t in 1..R
        for t in range(1, self.rounds + 1):
            if t not in self.a_canon:
                self.a_canon[t] = [self._fresh() for _ in range(32)]
            if t not in self.e_canon:
                self.e_canon[t] = [self._fresh() for _ in range(32)]

        # Unit-fix known message words
        for w_idx, w_val in fixed.items():
            if 0 <= w_idx < self.rounds:
                bits = _word_const_lits(int(w_val) & MASK32)
                for b in range(32):
                    self._emit_eq(self.W_vars[w_idx][b], bits[b])

        # Unit-fix bit-level message constraints
        for (w_idx, b_idx), bit_val in (fixed_bits or {}).items():
            wi, bi = int(w_idx), int(b_idx)
            if 0 <= wi < self.rounds and 0 <= bi < 32:
                self._emit_eq(self.W_vars[wi][bi], bool(int(bit_val) & 1))

        # Force C[0..15][0..2] == 0
        for t in range(min(16, self.rounds)):
            for k in range(3):
                for b in range(1, 32):
                    self._emit_eq(self.C_vars[t][k][b], False)

        # Message schedule equations for t >= 16
        for t in range(16, self.rounds):
            w2 = self.W_vars[t - 2]
            w15 = self.W_vars[t - 15]
            s1 = self._xor3_word(_rotr_lits(w2, 17), _rotr_lits(w2, 19), _shr_lits(w2, 10))
            s0 = self._xor3_word(_rotr_lits(w15, 7), _rotr_lits(w15, 18), _shr_lits(w15, 3))
            u0 = self._add32_word(s1, self.W_vars[t - 7], self.C_vars[t][0])
            u1 = self._add32_word(u0, s0, self.C_vars[t][1])
            wt_rhs = self._add32_word(u1, self.W_vars[t - 16], self.C_vars[t][2])
            for b in range(32):
                self._emit_eq(self.W_vars[t][b], wt_rhs[b])

        # Compression rounds t = 0..R-1 (镜像记录 Ch_vars 与 Maj_vars)
        for t in range(self.rounds):
            a_t = self.a_canon[t]
            b_t = self.a_canon[t - 1]
            c_t = self.a_canon[t - 2]
            d_t = self.a_canon[t - 3]
            e_t = self.e_canon[t]
            f_t = self.e_canon[t - 1]
            g_t = self.e_canon[t - 2]
            h_t = self.e_canon[t - 3]

            S1 = self._xor3_word(_rotr_lits(e_t, 6), _rotr_lits(e_t, 11), _rotr_lits(e_t, 25))
            ch = [self._fresh() for _ in range(32)]
            self.Ch_vars.append(list(ch))
            for b in range(32):
                self._emit_ch(e_t[b], f_t[b], g_t[b], ch[b])

            p1 = self._add32_word(h_t, S1, self.C_vars[t][3])
            p2 = self._add32_word(p1, ch, self.C_vars[t][4])
            p3 = self._add32_word(p2, _word_const_lits(K_WORDS[t]), self.C_vars[t][5])
            T1 = self._add32_word(p3, self.W_vars[t], self.C_vars[t][6])

            S0 = self._xor3_word(_rotr_lits(a_t, 2), _rotr_lits(a_t, 13), _rotr_lits(a_t, 22))
            mj = [self._fresh() for _ in range(32)]
            self.Maj_vars.append(list(mj))
            for b in range(32):
                self._emit_maj(a_t[b], b_t[b], c_t[b], mj[b])
            T2 = self._add32_word(S0, mj, self.C_vars[t][7])

            a_next_rhs = self._add32_word(T1, T2, self.C_vars[t][8])
            e_next_rhs = self._add32_word(d_t, T1, self.C_vars[t][9])

            a_nxt = self.a_canon[t + 1]
            e_nxt = self.e_canon[t + 1]
            for b in range(32):
                self._emit_eq(a_nxt[b], a_next_rhs[b])
                self._emit_eq(e_nxt[b], e_next_rhs[b])

        return self


class SignedUnionFind:
    """
    带符号并查集（维护 v ^ root == parity_to_root mod 2）：
      - 检出奇校验环 (x = ¬x)；
      - 结合 known_units 检出等价链常数冲突，并把已固定根的取值传导至整个等价类；
      - 导出不违背既定常数的 2-literal 等价子句。
    """

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}
        self.parity: Dict[int, int] = {}

    def find(self, x: int) -> Tuple[int, int]:
        if x not in self.parent:
            self.parent[x] = x
            self.parity[x] = 0
            return x, 0
        # 迭代路径压缩，避免极端链深度触发递归栈限制
        path: List[Tuple[int, int]] = []
        cur = x
        while self.parent[cur] != cur:
            path.append((cur, self.parity[cur]))
            cur = self.parent[cur]
        root = cur
        acc = 0
        for node, p in reversed(path):
            acc ^= p
            self.parent[node] = root
            self.parity[node] = acc
        return root, acc

    def union(self, u: int, v: int, diff_parity: int) -> bool:
        """合并 u ^ v == diff_parity；遇奇校验环返回 False。"""
        ru, pu = self.find(int(u))
        rv, pv = self.find(int(v))
        req = (pu ^ pv ^ (int(diff_parity) & 1)) & 1
        if ru == rv:
            return req == 0
        if ru > rv:
            ru, rv = rv, ru
        self.parent[rv] = ru
        self.parity[rv] = req
        return True

    def reconcile_units_and_export_2lit(
        self,
        known_units: Dict[int, int],
    ) -> Tuple[bool, Dict[int, int], List[List[int]]]:
        """
        检验等价类与 known_units 是否存在常数冲突；一致时把固定根传播给全成员，
        并为未固定代表元的等价类生成等价 2-CNF 子句。
        Returns: (consistent, updated_units, equiv_2lit_clauses)
        """
        updated_units = dict(known_units)
        root_fixed: Dict[int, int] = {}

        for v, val in updated_units.items():
            if v not in self.parent:
                continue
            r, p = self.find(v)
            r_val = (int(val) & 1) ^ p
            if r in root_fixed:
                if root_fixed[r] != r_val:
                    return False, updated_units, []
            else:
                root_fixed[r] = r_val

        # 若根已固定，推导并写入全部等价类成员（同时二次检查无冲突）
        for v in list(self.parent.keys()):
            r, p = self.find(v)
            if r in root_fixed:
                implied = root_fixed[r] ^ p
                if v in updated_units and updated_units[v] != implied:
                    return False, updated_units, []
                updated_units[v] = implied

        # 导出未固定等价类的 2-literal 等价子句
        equiv_clauses: List[List[int]] = []
        for v in sorted(self.parent.keys()):
            r, p = self.find(v)
            if r == v or r in root_fixed:
                continue
            if p == 0:
                equiv_clauses.append([-v, r])
                equiv_clauses.append([v, -r])
            else:
                equiv_clauses.append([v, r])
                equiv_clauses.append([-v, -r])

        return True, updated_units, equiv_clauses


@dataclass
class BridgeExtractionResult:
    """双平面预处理与 Bitset RREF 抽取结果。"""

    rounds: int
    early_unsat: bool
    encoder: BridgeableSHA256XORCNFEncoder
    unit_clauses: List[List[int]]
    equiv_2lit_clauses: List[List[int]]
    simplified_xors: List[Tuple[List[int], bool]]
    known_units: Dict[int, int]
    carry_lock_rate: float
    gf2_residual_dof: int
    active_carry_gates: int
    dual_prop_unknown: int
    elapsed_ms: float


class DualPlaneEquivalenceBridge:
    """
    双平面闭包 + 条件 Ch(5)/Maj(6) 线性化 + Python Bitset RREF + SignedUnionFind 桥接器。
    """

    def __init__(self, core: Optional[OuroborosCore] = None) -> None:
        self.core = core or get_core()

    @staticmethod
    def _bitset_rref_and_extract(
        equations: Sequence[Tuple[int, int]],
        known_units: Dict[int, int],
        uf: Optional[SignedUnionFind] = None,
    ) -> Tuple[bool, Dict[int, int], List[Tuple[int, int]]]:
        """
        对 0-indexed 变量掩码方程 [(mask, rhs), ...] 执行完整 GF(2) RREF（前向消元 + 稀疏反向回代）。
        Returns: (ok, new_units_var_id_to_bit, surviving_pivots_list)
        """
        pivots: Dict[int, Tuple[int, int]] = {}
        for mask, rhs in equations:
            m = int(mask)
            r = int(rhs) & 1
            while m:
                lb = (m & -m).bit_length() - 1
                piv = pivots.get(lb)
                if piv is not None:
                    pm, pr = piv
                    m ^= pm
                    r ^= pr
                else:
                    pivots[lb] = (m, r)
                    break
            else:
                if r:
                    return False, {}, []

        # 稀疏反向回代（自大到小处理 pivot，保证 pivots[b] 无其他 pivot 位）
        sorted_leads = sorted(pivots.keys(), reverse=True)
        pivots_mask = 0
        for k in sorted_leads:
            pivots_mask |= 1 << k

        for k in sorted_leads:
            m_k, r_k = pivots[k]
            hits = (m_k & pivots_mask) ^ (1 << k)
            while hits:
                lsb = hits & -hits
                b = lsb.bit_length() - 1
                pm, pr = pivots[b]
                m_k ^= pm
                r_k ^= pr
                hits ^= lsb
            pivots[k] = (m_k, r_k)

        new_units: Dict[int, int] = {}
        uf_obj = uf if uf is not None else SignedUnionFind()
        surviving: List[Tuple[int, int]] = []

        for k in sorted(pivots.keys()):
            m_k, r_k = pivots[k]
            wt = m_k.bit_count()
            if wt == 0:
                if r_k:
                    return False, {}, []
                continue
            if wt == 1:
                v = (m_k & -m_k).bit_length()  # bit index (v-1) -> var_id v
                if v in known_units and known_units[v] != r_k:
                    return False, {}, []
                if v in new_units and new_units[v] != r_k:
                    return False, {}, []
                new_units[v] = r_k
            elif wt == 2:
                lsb1 = m_k & -m_k
                b1 = lsb1.bit_length() - 1
                rem = m_k ^ lsb1
                b2 = (rem & -rem).bit_length() - 1
                v1, v2 = b1 + 1, b2 + 1
                if not uf_obj.union(v1, v2, r_k):
                    return False, {}, []
                surviving.append((m_k, r_k))
            else:
                surviving.append((m_k, r_k))

        return True, new_units, surviving

    def extract(
        self,
        H_target: Sequence[int],
        rounds: int,
        fixed_words: Optional[Dict[int, int]] = None,
        fixed_bits: Optional[Dict[Tuple[int, int], int]] = None,
        run_bitset_rref: bool = True,
    ) -> BridgeExtractionResult:
        t0 = time.perf_counter()
        ht_np = np.asarray(H_target, dtype=np.uint32).reshape(8)
        enc = BridgeableSHA256XORCNFEncoder(rounds=rounds).build(
            H_target=ht_np, fixed_words=fixed_words, fixed_bits=fixed_bits
        )

        # Step 1: C++ 双平面闭合
        env = GeneralizedSymbolEnv(H_target=ht_np, rounds=rounds, core=self.core)
        early_unsat = False
        try:
            for wi, wv in (fixed_words or {}).items():
                u = int(wv) & MASK32
                for bb in range(32):
                    env.set_symbol(0, int(wi), 0, bb, SYM_1 if ((u >> bb) & 1) else SYM_0, run_propagate=False)
            for (wi, bb), bv in (fixed_bits or {}).items():
                env.set_symbol(0, int(wi), 0, int(bb), SYM_1 if (int(bv) & 1) else SYM_0, run_propagate=False)
        except PermissionError:
            early_unsat = True

        if not early_unsat:
            step_res = env.propagate(use_gf2=True, max_iters=16)
            if step_res.conflict or env.last_conflict:
                early_unsat = True

        active_carries = env.count_active_carry_gates()
        denom_carries = max(1, (rounds * 10 - min(16, rounds) * 3) * 31)
        carry_lock_rate = float(active_carries / denom_carries)

        if early_unsat:
            return BridgeExtractionResult(
                rounds=rounds,
                early_unsat=True,
                encoder=enc,
                unit_clauses=[[1], [-1]],
                equiv_2lit_clauses=[],
                simplified_xors=[],
                known_units={},
                carry_lock_rate=carry_lock_rate,
                gf2_residual_dof=0,
                active_carry_gates=active_carries,
                dual_prop_unknown=env.count_unknown(),
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Step 2: 单位映射（严密 bool 常量拦截 + 常数冲突检验）
        known_units: Dict[int, int] = {}

        def _bind_lit(lit: Lit, expected_bit: int) -> bool:
            if isinstance(lit, bool):
                return (1 if lit else 0) == (expected_bit & 1)
            raw = int(lit)
            v = abs(raw)
            if v <= 0:
                return False
            var_val = (expected_bit & 1) if raw > 0 else (1 - (expected_bit & 1))
            prev = known_units.get(v)
            if prev is not None and prev != var_val:
                return False
            known_units[v] = var_val
            return True

        # 先消费编码器已有单子句（含可能折叠出的 [1], [-1]）
        for cl in enc.clauses:
            if len(cl) == 0:
                early_unsat = True
                break
            if len(cl) == 1:
                l0 = cl[0]
                if not _bind_lit(abs(int(l0)), 1 if int(l0) > 0 else 0):
                    early_unsat = True
                    break

        if not early_unsat:
            # 绑定 W_sym
            for t in range(rounds):
                for b in range(32):
                    sv = int(env.W_sym[t, b])
                    if sv in (SYM_0, SYM_1):
                        if not _bind_lit(enc.W_vars[t][b], 1 if sv == SYM_1 else 0):
                            early_unsat = True
                            break
                if early_unsat:
                    break

        if not early_unsat:
            # 绑定 S_sym -> a_canon / e_canon
            for t in range(rounds + 1):
                for w in range(8):
                    canon_row = enc.a_canon.get(t - w) if w < 4 else enc.e_canon.get(t - (w - 4))
                    if canon_row is None:
                        continue
                    for b in range(32):
                        sv = int(env.S_sym[t, w, b])
                        if sv in (SYM_0, SYM_1):
                            if not _bind_lit(canon_row[b], 1 if sv == SYM_1 else 0):
                                early_unsat = True
                                break
                    if early_unsat:
                        break
                if early_unsat:
                    break

        if not early_unsat:
            # 绑定 C_sym -> C_vars
            for t in range(rounds):
                for k in range(C_PLANES):
                    for b in range(32):
                        sv = int(env.C_sym[t, k, b])
                        if sv in (SYM_0, SYM_1):
                            if not _bind_lit(enc.C_vars[t][k][b], 1 if sv == SYM_1 else 0):
                                early_unsat = True
                                break
                    if early_unsat:
                        break
                if early_unsat:
                    break

        if early_unsat:
            return BridgeExtractionResult(
                rounds=rounds,
                early_unsat=True,
                encoder=enc,
                unit_clauses=[[1], [-1]],
                equiv_2lit_clauses=[],
                simplified_xors=[],
                known_units=known_units,
                carry_lock_rate=carry_lock_rate,
                gf2_residual_dof=0,
                active_carry_gates=active_carries,
                dual_prop_unknown=env.count_unknown(),
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

        if not run_bitset_rref:
            unit_clauses = [[v if val == 1 else -v] for v, val in sorted(known_units.items())]
            return BridgeExtractionResult(
                rounds=rounds,
                early_unsat=False,
                encoder=enc,
                unit_clauses=unit_clauses,
                equiv_2lit_clauses=[],
                simplified_xors=list(enc.xor_clauses),
                known_units=known_units,
                carry_lock_rate=carry_lock_rate,
                gf2_residual_dof=max(0, env.count_unknown()),
                active_carry_gates=active_carries,
                dual_prop_unknown=env.count_unknown(),
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Step 3: 构建 Python Bitset 方程组 (enc.xor_clauses + Ch 5 规则 + Maj 6 规则)
        def _resolve_lit(lit: Lit) -> Tuple[Optional[int], int]:
            if isinstance(lit, bool):
                return None, (1 if lit else 0)
            raw = int(lit)
            v = abs(raw)
            if v in known_units:
                kv = known_units[v]
                return None, (kv if raw > 0 else (1 - kv))
            # raw < 0 等价于 v ^ 1
            return v, (0 if raw > 0 else 1)

        raw_eqs: List[Tuple[int, int]] = []
        active_vars_mask = 0

        def _push_lits_eq(lits_seq: Sequence[Lit], rhs_bit: int) -> None:
            nonlocal active_vars_mask
            mask = 0
            rhs = int(rhs_bit) & 1
            for lt in lits_seq:
                v, cbit = _resolve_lit(lt)
                rhs ^= cbit
                if v is not None:
                    mask ^= 1 << (v - 1)
            if mask == 0 and rhs == 0:
                return
            active_vars_mask |= mask
            raw_eqs.append((mask, rhs))

        for x_vars, parity in enc.xor_clauses:
            _push_lits_eq(x_vars, 1 if parity else 0)

        # Ch (5 规则) + Maj (6 规则) 条件线性化
        def _lit_known_val(lit: Lit) -> Optional[int]:
            v, cbit = _resolve_lit(lit)
            return cbit if v is None else None

        for t in range(rounds):
            for b in range(32):
                ch_v = enc.Ch_vars[t][b]
                e_lit = enc.e_canon[t][b]
                f_lit = enc.e_canon[t - 1][b]
                g_lit = enc.e_canon[t - 2][b]
                ev = _lit_known_val(e_lit)
                fv = _lit_known_val(f_lit)
                gv = _lit_known_val(g_lit)
                if ev == 1:
                    _push_lits_eq([ch_v, f_lit], 0)
                elif ev == 0:
                    _push_lits_eq([ch_v, g_lit], 0)
                if fv is not None and gv is not None:
                    if fv == gv:
                        _push_lits_eq([ch_v], fv)
                    elif fv == 1 and gv == 0:
                        _push_lits_eq([ch_v, e_lit], 0)
                    else:
                        _push_lits_eq([ch_v, e_lit], 1)

                mj_v = enc.Maj_vars[t][b]
                a_lit = enc.a_canon[t][b]
                b_lit = enc.a_canon[t - 1][b]
                c_lit = enc.a_canon[t - 2][b]
                av = _lit_known_val(a_lit)
                bv = _lit_known_val(b_lit)
                cv = _lit_known_val(c_lit)
                if av is not None and bv is not None:
                    if av == bv:
                        _push_lits_eq([mj_v], av)
                    else:
                        _push_lits_eq([mj_v, c_lit], 0)
                if av is not None and cv is not None:
                    if av == cv:
                        _push_lits_eq([mj_v], av)
                    else:
                        _push_lits_eq([mj_v, b_lit], 0)
                if bv is not None and cv is not None:
                    if bv == cv:
                        _push_lits_eq([mj_v], bv)
                    else:
                        _push_lits_eq([mj_v, a_lit], 0)

        uf = SignedUnionFind()
        ok_rref, rref_units, surviving = self._bitset_rref_and_extract(raw_eqs, known_units, uf=uf)
        if not ok_rref:
            early_unsat = True
        else:
            for v, val in rref_units.items():
                if v in known_units and known_units[v] != val:
                    early_unsat = True
                    break
                known_units[v] = val

        if not early_unsat:
            ok_uf, known_units, equiv_2lit_clauses = uf.reconcile_units_and_export_2lit(known_units)
            if not ok_uf:
                early_unsat = True
        else:
            equiv_2lit_clauses = []

        if early_unsat:
            return BridgeExtractionResult(
                rounds=rounds,
                early_unsat=True,
                encoder=enc,
                unit_clauses=[[1], [-1]],
                equiv_2lit_clauses=[],
                simplified_xors=[],
                known_units=known_units,
                carry_lock_rate=carry_lock_rate,
                gf2_residual_dof=0,
                active_carry_gates=active_carries,
                dual_prop_unknown=env.count_unknown(),
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 计算精确 GF(2) 残余自由度（与 diag_xor_cliff.py 口径一致：当前线性子系统活跃未知变元数 - RREF 秩）
        # 注：surviving 包含全部非空且未塌缩为单/双变量的枢轴行 + rref_units(秩1) + uf_equivs
        # 直接统计原始 raw_eqs 对应线性空间秩: 等于 pivots 数量 = len(rref_units) + len(surviving) + n_uf_edges
        # 严格按 raw_eqs 在 surviving/rref 中的枢轴计数:
        # 由于 _bitset_rref_and_extract 每个非矛盾 pivot 均落入 wt==1, wt==2 或 wt>=3 三者之一，
        # 这里为精确统计 rank 与活跃未知变元数，直接计数:
        # 每条 raw_eqs 中涉及的非已知变量共 active_vars_mask.bit_count() 个；
        # 但在 BridgeableSHA256XORCNFEncoder 中，若 Ch/Maj 未触发规则且也出现在 e_eq/a_diff_eq，则已被纳入 active_vars_mask。
        # 且每个未消除的 Tseitin 辅变量引入 1 变量 + 1 独立方程（两者在 vars - rank 中抵消！）。
        # 但需减去“已经从 raw_eqs 推出但在 RREF 前已知”的空缺？无需，因为 known_units 已在代入阶段移除。
        # 计算 exact_rank:
        n_active_vars = active_vars_mask.bit_count()
        # exact_rank = total_pivots（注意 weight-2 也对应 surviving 中的一部分，每个独立等价边贡献 1 秩）
        # 为防 weight-2 合并冗余，直接在 uf 中 count edges = sum(1 for v in parent if parent[v] != v)
        n_uf_edges = sum(1 for v, p in uf.parent.items() if p != v)
        n_wt3_plus = sum(1 for m, _ in surviving if m.bit_count() >= 3)
        exact_rank = len(rref_units) + n_uf_edges + n_wt3_plus
        gf2_residual_dof = max(0, n_active_vars - exact_rank)

        simplified_xors: List[Tuple[List[int], bool]] = []
        for m, r in surviving:
            if m.bit_count() >= 3:
                vs: List[int] = []
                mm = m
                while mm:
                    lsb = mm & -mm
                    vs.append(lsb.bit_length())  # (bit_idx + 1)
                    mm ^= lsb
                simplified_xors.append((vs, bool(r)))

        unit_clauses = [[v if val == 1 else -v] for v, val in sorted(known_units.items())]
        return BridgeExtractionResult(
            rounds=rounds,
            early_unsat=False,
            encoder=enc,
            unit_clauses=unit_clauses,
            equiv_2lit_clauses=equiv_2lit_clauses,
            simplified_xors=simplified_xors,
            known_units=known_units,
            carry_lock_rate=carry_lock_rate,
            gf2_residual_dof=gf2_residual_dof,
            active_carry_gates=active_carries,
            dual_prop_unknown=env.count_unknown(),
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )


def _cms_worker_loop() -> None:
    """
    隔离在独立子进程中的常驻 CryptoMiniSat-5 Worker 循环。
    通过长度前缀 pickle 帧接收求解请求并返回结果，避免 Windows 0xC0000005 拖垮主进程。
    """
    import pycryptosat

    stdin_buf = sys.stdin.buffer
    stdout_buf = sys.stdout.buffer

    while True:
        hdr = stdin_buf.read(4)
        if not hdr or len(hdr) < 4:
            break
        (length,) = struct.unpack("<I", hdr)
        payload = bytearray()
        while len(payload) < length:
            chunk = stdin_buf.read(length - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) < length:
            break

        req = pickle.loads(bytes(payload))
        cmd = req.get("cmd", "solve")
        if cmd == "exit":
            break

        fault = req.get("simulate_fault")
        if fault == "crash":
            os._exit(11)
        if fault == "timeout":
            time.sleep(float(req.get("fault_sleep_sec", 5.0)))

        t0 = time.perf_counter()
        threads = int(req.get("threads", 1))
        clauses = req["clauses"]
        xor_clauses = req["xor_clauses"]
        extra_clauses = req.get("extra_clauses") or []
        time_limit_sec = float(req.get("time_limit_sec", 5.0))
        conflict_budget = int(req.get("conflict_budget", -1))
        assumptions = req.get("assumptions")

        solver = pycryptosat.Solver(threads=threads)
        if clauses:
            solver.add_clauses(clauses)
        for x_vars, rhs in xor_clauses:
            solver.add_xor_clause(x_vars, bool(rhs))
        if extra_clauses:
            solver.add_clauses(extra_clauses)

        solve_kwargs: Dict[str, Any] = {}
        if assumptions:
            solve_kwargs["assumptions"] = list(assumptions)
        if time_limit_sec > 0.0:
            solve_kwargs["time_limit"] = time_limit_sec
        if conflict_budget > 0:
            solve_kwargs["confl_limit"] = conflict_budget

        sat, solution = solver.solve(**solve_kwargs)
        dt = time.perf_counter() - t0
        resp = {
            "status": "SAT" if sat is True else ("UNSAT" if sat is False else "TIMEOUT_OR_BUDGET"),
            "solution": list(solution) if (sat is True and solution is not None) else None,
            "elapsed_sec": float(dt),
        }
        out_bytes = pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL)
        stdout_buf.write(struct.pack("<I", len(out_bytes)) + out_bytes)
        stdout_buf.flush()


class SafeCMSSubprocessRunner:
    """
    安全隔离 CryptoMiniSat 子进程运行器（常驻单 Worker 复用 + 超时强杀 + 崩溃重启）。
    """

    def __init__(self, python_exe: Optional[str] = None, threads: int = 1) -> None:
        self.python_exe = python_exe or sys.executable
        self.threads = int(threads)
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._resp_q: Optional[queue.Queue] = None
        self._lock = threading.Lock()

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._cleanup_proc()
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        self._proc = subprocess.Popen(
            [self.python_exe, "-u", "-m", "v10_ouroboros64.prototypes.p3_dual_bridge_router", "--cms_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        self._resp_q = queue.Queue()
        proc_ref = self._proc
        q_ref = self._resp_q

        def _reader() -> None:
            try:
                assert proc_ref.stdout is not None
                out_buf = proc_ref.stdout
                while True:
                    hdr = out_buf.read(4)
                    if not hdr or len(hdr) < 4:
                        q_ref.put({"_eof": True})
                        break
                    (length,) = struct.unpack("<I", hdr)
                    buf = bytearray()
                    while len(buf) < length:
                        chunk = out_buf.read(length - len(buf))
                        if not chunk:
                            break
                        buf.extend(chunk)
                    if len(buf) < length:
                        q_ref.put({"_eof": True})
                        break
                    q_ref.put(pickle.loads(bytes(buf)))
            except Exception as exc:
                q_ref.put({"_eof": True, "_exc": repr(exc)})

        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()

    def _cleanup_proc(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.poll() is None:
                    self._proc.kill()
                    self._proc.wait(timeout=2.0)
            except Exception:
                pass
            for s in (self._proc.stdin, self._proc.stdout):
                try:
                    if s is not None:
                        s.close()
                except Exception:
                    pass
            self._proc = None
            self._resp_q = None
            self._reader_thread = None

    def solve_clauses(
        self,
        clauses: Sequence[Sequence[int]],
        xor_clauses: Sequence[Tuple[Sequence[int], bool]],
        extra_clauses: Optional[Sequence[Sequence[int]]] = None,
        time_limit_sec: float = 5.0,
        conflict_budget: int = -1,
        assumptions: Optional[Sequence[int]] = None,
        hard_timeout_slack: float = 1.0,
        simulate_fault: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            t0 = time.perf_counter()
            try:
                self._ensure_worker()
                assert self._proc is not None and self._proc.stdin is not None and self._resp_q is not None
                req = {
                    "cmd": "solve",
                    "threads": self.threads,
                    "clauses": [list(c) for c in clauses],
                    "xor_clauses": [(list(xv), bool(xp)) for xv, xp in xor_clauses],
                    "extra_clauses": [list(c) for c in (extra_clauses or []) if c],
                    "time_limit_sec": float(time_limit_sec),
                    "conflict_budget": int(conflict_budget),
                    "assumptions": list(assumptions) if assumptions else None,
                    "simulate_fault": simulate_fault,
                    "fault_sleep_sec": max(2.0, float(time_limit_sec) + float(hard_timeout_slack) + 2.0),
                }
                raw = pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL)
                self._proc.stdin.write(struct.pack("<I", len(raw)) + raw)
                self._proc.stdin.flush()

                wait_sec = max(0.3, float(time_limit_sec) + float(hard_timeout_slack))
                resp = self._resp_q.get(timeout=wait_sec)
                if resp.get("_eof"):
                    rc = self._proc.poll()
                    if rc is None:
                        try:
                            rc = self._proc.wait(timeout=1.0)
                        except Exception:
                            rc = -1
                    self._cleanup_proc()
                    return {
                        "crashed_or_timeout": True,
                        "reason": f"worker_exit_{rc}",
                        "status": "CRASH_FALLBACK",
                        "solution": None,
                        "elapsed_sec": time.perf_counter() - t0,
                    }
                return {
                    "crashed_or_timeout": False,
                    "reason": "ok",
                    "status": resp["status"],
                    "solution": resp["solution"],
                    "elapsed_sec": float(resp.get("elapsed_sec", time.perf_counter() - t0)),
                }
            except queue.Empty:
                self._cleanup_proc()
                return {
                    "crashed_or_timeout": True,
                    "reason": "hard_timeout_killed",
                    "status": "TIMEOUT_FALLBACK",
                    "solution": None,
                    "elapsed_sec": time.perf_counter() - t0,
                }
            except Exception as exc:
                self._cleanup_proc()
                return {
                    "crashed_or_timeout": True,
                    "reason": f"ipc_error:{exc!r}",
                    "status": "CRASH_FALLBACK",
                    "solution": None,
                    "elapsed_sec": time.perf_counter() - t0,
                }

    def close(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    assert self._proc.stdin is not None
                    raw = pickle.dumps({"cmd": "exit"}, protocol=pickle.HIGHEST_PROTOCOL)
                    self._proc.stdin.write(struct.pack("<I", len(raw)) + raw)
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=1.5)
                except Exception:
                    pass
            self._cleanup_proc()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class CarryLockHybridRouter:
    """
    P3 Carry-Lock 感知三区混合路由器：
      - Zone 1 (r_lock < 0.29 and D_gf2 <= 2150): 桥接增强 CMS-XOR（带子进程崩溃/超时回落）；
      - Zone 2 (0.29 <= r_lock < 0.325 and D_gf2 <= 2450): 短探针 min(0.6s, 0.20*tl) -> 回落 P1+P2；
      - Zone 3 (r_lock >= 0.325 or D_gf2 > 2450 或 R >= 48): 绕过 CMS-XOR 直入桥接增强 P1+P2。
    """

    def __init__(
        self,
        model: Optional[Ouroboros64Net] = None,
        core: Optional[OuroborosCore] = None,
        device: Optional[torch.device | str] = None,
        solver_name: str = "cadical195",
        cms_runner: Optional[SafeCMSSubprocessRunner] = None,
        cdcl_solver: Optional[CleanDynamicNeuralCDCLSolver] = None,
    ) -> None:
        self.core = core or get_core()
        self.model = model
        self.device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters()).device if model is not None else torch.device("cpu"))
        )
        self.bridge = DualPlaneEquivalenceBridge(core=self.core)
        self._owns_cms_runner = cms_runner is None
        self.cms_runner = cms_runner or SafeCMSSubprocessRunner(threads=1)
        self.cdcl_solver = cdcl_solver or CleanDynamicNeuralCDCLSolver(
            model=model,
            core=self.core,
            device=self.device,
            solver_name=solver_name,
        )
        self.last_diagnostics: Dict[str, Any] = {}

    def close(self) -> None:
        if self._owns_cms_runner and self.cms_runner is not None:
            self.cms_runner.close()

    def solve(
        self,
        H_target: Sequence[int],
        rounds: int = 26,
        fixed_words: Optional[Dict[int, int]] = None,
        fixed_bits: Optional[Dict[Tuple[int, int], int]] = None,
        extra_clauses: Optional[Sequence[Sequence[int]]] = None,
        conflict_budget: int = 250000,
        time_limit_sec: float = 5.0,
        safe_cms_subprocess: bool = True,
        simulate_cms_fault: Optional[str] = None,
    ) -> DynamicSolveResult:
        t0 = time.perf_counter()
        ht_np = np.asarray(H_target, dtype=np.uint32).reshape(8)
        user_extra = [list(c) for c in (extra_clauses or []) if c]

        if rounds >= 48:
            # 高轮边界关闭全局 RREF，直走 Zone 3 P1+P2
            res = self.cdcl_solver.solve(
                H_target=ht_np,
                rounds=rounds,
                fixed_words=fixed_words,
                fixed_bits=fixed_bits,
                extra_clauses=user_extra,
                conflict_budget=conflict_budget,
                time_limit_sec=time_limit_sec,
            )
            self.last_diagnostics = {
                "zone": "Zone3_HighRounds",
                "carry_lock_rate": 1.0,
                "gf2_residual_dof": -1,
                "early_unsat": False,
                "n_units": 0,
                "n_equiv_2lit": 0,
                "cms_attempted": False,
                "cms_fallback_triggered": False,
                "bridge_ms": 0.0,
            }
            return res

        # 1) 双平面预处理 + Bitset RREF 抽取单位与 2-literal 等价类
        br = self.bridge.extract(
            H_target=ht_np,
            rounds=rounds,
            fixed_words=fixed_words,
            fixed_bits=fixed_bits,
            run_bitset_rref=True,
        )
        diag: Dict[str, Any] = {
            "zone": "unknown",
            "carry_lock_rate": br.carry_lock_rate,
            "gf2_residual_dof": br.gf2_residual_dof,
            "early_unsat": br.early_unsat,
            "n_units": len(br.unit_clauses),
            "n_equiv_2lit": len(br.equiv_2lit_clauses),
            "cms_attempted": False,
            "cms_fallback_triggered": False,
            "cms_reason": "",
            "bridge_ms": br.elapsed_ms,
        }

        if br.early_unsat:
            diag["zone"] = "EarlyUNSAT"
            self.last_diagnostics = diag
            return DynamicSolveResult(
                rounds=int(rounds),
                solver_type="p3_carry_lock_router",
                status="UNSAT",
                valid_preimage=False,
                conflicts=0,
                decisions=0,
                elapsed_sec=time.perf_counter() - t0,
                episodes=0,
                flp_units_proven=len(br.unit_clauses),
                cores_pruned=0,
                min_core_len=0,
                w0_15=None,
            )

        bridge_extra = list(br.unit_clauses) + list(br.equiv_2lit_clauses) + user_extra
        r_lock = br.carry_lock_rate
        d_gf2 = br.gf2_residual_dof

        # 2) Carry-Lock 三区判定
        if (r_lock < 0.29) and (d_gf2 <= 2150 or rounds <= 20):
            zone = "Zone1_Linear"
        elif (r_lock < 0.325) and (d_gf2 <= 2450):
            zone = "Zone2_EdgeProbe"
        else:
            zone = "Zone3_CarryLocked"
        diag["zone"] = zone

        def _run_cms(tl_sec: float) -> Optional[DynamicSolveResult]:
            diag["cms_attempted"] = True
            if safe_cms_subprocess:
                cms_res = self.cms_runner.solve_clauses(
                    clauses=br.encoder.clauses,
                    xor_clauses=br.encoder.xor_clauses,
                    extra_clauses=bridge_extra,
                    time_limit_sec=max(0.05, tl_sec),
                    conflict_budget=-1,
                    hard_timeout_slack=0.6 if simulate_cms_fault is None else 0.4,
                    simulate_fault=simulate_cms_fault,
                )
                diag["cms_reason"] = cms_res["reason"]
                if cms_res["crashed_or_timeout"]:
                    diag["cms_fallback_triggered"] = True
                    return None
                st = cms_res["status"]
                if st == "SAT" and cms_res["solution"] is not None:
                    w_rec = extract_w0_15_from_cryptosat(br.encoder, cms_res["solution"])
                    valid = bool(self.core.verify_preimage(w_rec, ht_np, rounds=rounds)[0])
                    assert valid, f"P3 CMS 子进程返回未通过 verify_preimage 的解 R={rounds}"
                    return DynamicSolveResult(
                        rounds=int(rounds),
                        solver_type="p3_carry_lock_router_cms",
                        status="SAT",
                        valid_preimage=True,
                        conflicts=0,
                        decisions=0,
                        elapsed_sec=time.perf_counter() - t0,
                        episodes=1,
                        flp_units_proven=len(br.unit_clauses),
                        cores_pruned=len(br.equiv_2lit_clauses) // 2,
                        min_core_len=2 if br.equiv_2lit_clauses else 0,
                        w0_15=w_rec,
                    )
                if st == "UNSAT":
                    return DynamicSolveResult(
                        rounds=int(rounds),
                        solver_type="p3_carry_lock_router_cms",
                        status="UNSAT",
                        valid_preimage=False,
                        conflicts=0,
                        decisions=0,
                        elapsed_sec=time.perf_counter() - t0,
                        episodes=1,
                        flp_units_proven=len(br.unit_clauses),
                        cores_pruned=len(br.equiv_2lit_clauses) // 2,
                        min_core_len=2 if br.equiv_2lit_clauses else 0,
                        w0_15=None,
                    )
                diag["cms_fallback_triggered"] = True
                return None
            else:
                import pycryptosat

                s = pycryptosat.Solver(threads=1)
                if br.encoder.clauses:
                    s.add_clauses(br.encoder.clauses)
                for xv, xp in br.encoder.xor_clauses:
                    s.add_xor_clause(xv, bool(xp))
                if bridge_extra:
                    s.add_clauses(bridge_extra)
                sat, sol = s.solve(time_limit=max(0.05, tl_sec))
                if sat is True and sol is not None:
                    w_rec = extract_w0_15_from_cryptosat(br.encoder, sol)
                    valid = bool(self.core.verify_preimage(w_rec, ht_np, rounds=rounds)[0])
                    assert valid
                    return DynamicSolveResult(
                        rounds=int(rounds),
                        solver_type="p3_carry_lock_router_cms",
                        status="SAT",
                        valid_preimage=True,
                        conflicts=0,
                        decisions=0,
                        elapsed_sec=time.perf_counter() - t0,
                        episodes=1,
                        flp_units_proven=len(br.unit_clauses),
                        cores_pruned=len(br.equiv_2lit_clauses) // 2,
                        min_core_len=2 if br.equiv_2lit_clauses else 0,
                        w0_15=w_rec,
                    )
                if sat is False:
                    return DynamicSolveResult(
                        rounds=int(rounds),
                        solver_type="p3_carry_lock_router_cms",
                        status="UNSAT",
                        valid_preimage=False,
                        conflicts=0,
                        decisions=0,
                        elapsed_sec=time.perf_counter() - t0,
                        episodes=1,
                        flp_units_proven=len(br.unit_clauses),
                        cores_pruned=len(br.equiv_2lit_clauses) // 2,
                        min_core_len=2 if br.equiv_2lit_clauses else 0,
                        w0_15=None,
                    )
                diag["cms_fallback_triggered"] = True
                return None

        # Zone 1 / Zone 2 路由执行
        if zone == "Zone1_Linear":
            rem_tl = max(0.05, time_limit_sec - (time.perf_counter() - t0)) if time_limit_sec > 0 else 5.0
            cms_budget = min(2.0, rem_tl * 0.50) if time_limit_sec > 0 else 5.0
            cms_out = _run_cms(cms_budget)
            if cms_out is not None:
                self.last_diagnostics = diag
                return cms_out
        elif zone == "Zone2_EdgeProbe":
            probe_tl = min(0.6, 0.20 * time_limit_sec) if time_limit_sec > 0 else 0.6
            cms_out = _run_cms(probe_tl)
            if cms_out is not None:
                self.last_diagnostics = diag
                return cms_out

        # Zone 3 或 CMS 超时/崩溃回落 -> 携带全量桥接单位+2-lit 等价子句转入 P1+P2 CleanDynamicNeuralCDCLSolver
        rem_cdcl_tl = (
            max(0.05, time_limit_sec - (time.perf_counter() - t0))
            if time_limit_sec > 0
            else 0.0
        )
        cdcl_res = self.cdcl_solver.solve(
            H_target=ht_np,
            rounds=rounds,
            fixed_words=fixed_words,
            fixed_bits=fixed_bits,
            extra_clauses=bridge_extra,
            conflict_budget=conflict_budget,
            time_limit_sec=rem_cdcl_tl,
        )
        cdcl_res.solver_type = f"p3_{zone.lower()}_cdcl"
        cdcl_res.elapsed_sec = float(time.perf_counter() - t0)
        self.last_diagnostics = diag
        return cdcl_res


if __name__ == "__main__":
    if "--cms_worker" in sys.argv:
        _cms_worker_loop()
