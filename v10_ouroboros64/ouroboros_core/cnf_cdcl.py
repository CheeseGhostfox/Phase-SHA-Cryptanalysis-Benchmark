"""Zero-approximation Tseitin CNF encoder & CaDiCaL CDCL solver bridge for Ouroboros-64."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pysat.solvers import Solver

from .engine import C_PLANES, IV_WORDS, K_WORDS, MASK32, get_core

Lit = Union[int, bool]  # positive/negative DIMACS variable int, or folded constant bool


def _neg(lit: Lit) -> Lit:
    if isinstance(lit, bool):
        return not lit
    return -lit


def _rotr_lits(word: Sequence[Lit], n: int) -> List[Lit]:
    return [word[(b + n) & 31] for b in range(32)]


def _shr_lits(word: Sequence[Lit], n: int) -> List[Lit]:
    return [word[b + n] if (b + n < 32) else False for b in range(32)]


def _word_const_lits(val: int) -> List[Lit]:
    return [bool((val >> b) & 1) for b in range(32)]


def make_end_free_constraints(
    w0_15_single: Sequence[int],
    free_bits: int,
) -> Tuple[Dict[int, int], Dict[Tuple[int, int], int]]:
    """
    构造 W[0..15] 末尾连续 free_bits 个自由比特的零泄漏约束：
    - 末尾 [512 - free_bits, 512) 保持完全未知，严禁任何内部真值泄漏；
    - 其余前置固定位拆分为整字 fixed_words 与半字固定位 fixed_bits。
    """
    fb = int(free_bits)
    if fb < 0 or fb > 512:
        raise ValueError(f"free_bits 必须位于 [0, 512]，收到 {fb}")
    total_fixed = 512 - fb
    n_full_words = total_fixed // 32
    rem_bits = total_fixed % 32
    fixed_words: Dict[int, int] = {
        wi: (int(w0_15_single[wi]) & MASK32) for wi in range(n_full_words)
    }
    fixed_bits: Dict[Tuple[int, int], int] = {}
    if rem_bits > 0 and n_full_words < 16:
        w_val = int(w0_15_single[n_full_words]) & MASK32
        for b in range(rem_bits):
            fixed_bits[(n_full_words, b)] = (w_val >> b) & 1
    return fixed_words, fixed_bits


class SHA256CNFEncoder:
    """Exact Tseitin CNF builder for R-round SHA-256 with welded boundaries S[0]=IV, S[R]=H-IV."""

    def __init__(self, rounds: int = 64) -> None:
        if rounds < 1 or rounds > 64:
            raise ValueError(f"rounds must be in [1, 64], got {rounds}")
        self.rounds = int(rounds)
        self.next_var = 1
        self.clauses: List[List[int]] = []

        # Allocate crystal variables matching Ouroboros variable layout
        self.W_vars: List[List[Lit]] = [[self._fresh() for _ in range(32)] for _ in range(self.rounds)]
        self.a_canon: Dict[int, List[Lit]] = {}
        self.e_canon: Dict[int, List[Lit]] = {}
        self.C_vars: List[List[List[Lit]]] = [
            [[False] + [self._fresh() for _ in range(31)] for _ in range(C_PLANES)]
            for _ in range(self.rounds)
        ]

    def _fresh(self) -> int:
        v = self.next_var
        self.next_var += 1
        return v

    def _emit_clause(self, *lits: Lit) -> None:
        kept: List[int] = []
        for x in lits:
            if isinstance(x, bool):
                if x:
                    return  # tautology
            else:
                kept.append(x)
        if not kept:
            # UNSAT contradiction marker
            self.clauses.append([1])
            self.clauses.append([-1])
            return
        self.clauses.append(kept)

    def _emit_eq(self, x: Lit, y: Lit) -> None:
        self._emit_clause(_neg(x), y)
        self._emit_clause(x, _neg(y))

    def _emit_xor2(self, a: Lit, b: Lit, out: Lit) -> None:
        self._emit_clause(_neg(a), _neg(b), _neg(out))
        self._emit_clause(a, b, _neg(out))
        self._emit_clause(a, _neg(b), out)
        self._emit_clause(_neg(a), b, out)

    def _emit_xor3(self, a: Lit, b: Lit, c: Lit, out: Lit) -> None:
        for va in (0, 1):
            for vb in (0, 1):
                for vc in (0, 1):
                    xor_val = va ^ vb ^ vc
                    la = a if va == 0 else _neg(a)
                    lb = b if vb == 0 else _neg(b)
                    lc = c if vc == 0 else _neg(c)
                    lo = out if xor_val == 1 else _neg(out)
                    self._emit_clause(la, lb, lc, lo)

    def _emit_maj(self, a: Lit, b: Lit, c: Lit, out: Lit) -> None:
        self._emit_clause(_neg(a), _neg(b), out)
        self._emit_clause(_neg(a), _neg(c), out)
        self._emit_clause(_neg(b), _neg(c), out)
        self._emit_clause(a, b, _neg(out))
        self._emit_clause(a, c, _neg(out))
        self._emit_clause(b, c, _neg(out))

    def _emit_ch(self, e: Lit, f: Lit, g: Lit, out: Lit) -> None:
        self._emit_clause(_neg(e), _neg(f), out)
        self._emit_clause(_neg(e), f, _neg(out))
        self._emit_clause(e, _neg(g), out)
        self._emit_clause(e, g, _neg(out))
        self._emit_clause(_neg(f), _neg(g), out)
        self._emit_clause(f, g, _neg(out))

    def _xor3_word(self, x: Sequence[Lit], y: Sequence[Lit], z: Sequence[Lit]) -> List[Lit]:
        out: List[Lit] = []
        for b in range(32):
            v = self._fresh()
            self._emit_xor3(x[b], y[b], z[b], v)
            out.append(v)
        return out

    def _add32_word(
        self,
        x: Sequence[Lit],
        y: Sequence[Lit],
        c_word: Optional[Sequence[Lit]] = None,
    ) -> List[Lit]:
        sum_word: List[Lit] = []
        cin: Lit = False
        for b in range(32):
            sb = self._fresh()
            self._emit_xor3(x[b], y[b], cin, sb)
            sum_word.append(sb)
            if b < 31:
                cout = c_word[b + 1] if c_word is not None else self._fresh()
                self._emit_maj(x[b], y[b], cin, cout)
                cin = cout
        return sum_word

    def build(
        self,
        H_target: Sequence[int],
        fixed_words: Optional[Dict[int, int]] = None,
        fixed_bits: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> "SHA256CNFEncoder":
        """Build exact CNF encoding with S[0]=IV, S[R]=H-IV, optional fixed W words and bit constraints."""
        self.clauses.clear()
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

        # Unit-fix any known message words (e.g. reduced-DoF blind benchmarks or padding)
        for w_idx, w_val in fixed.items():
            if 0 <= w_idx < self.rounds:
                bits = _word_const_lits(int(w_val) & MASK32)
                for b in range(32):
                    self._emit_eq(self.W_vars[w_idx][b], bits[b])

        # Unit-fix any bit-level message constraints (non-free bits)
        for (w_idx, b_idx), bit_val in (fixed_bits or {}).items():
            wi, bi = int(w_idx), int(b_idx)
            if 0 <= wi < self.rounds and 0 <= bi < 32:
                self._emit_eq(self.W_vars[wi][bi], bool(int(bit_val) & 1))

        # Force C[0..15][0..2] == 0 for unused schedule carry planes
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

        # Compression rounds t = 0..R-1
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
            for b in range(32):
                self._emit_ch(e_t[b], f_t[b], g_t[b], ch[b])

            p1 = self._add32_word(h_t, S1, self.C_vars[t][3])
            p2 = self._add32_word(p1, ch, self.C_vars[t][4])
            p3 = self._add32_word(p2, _word_const_lits(K_WORDS[t]), self.C_vars[t][5])
            T1 = self._add32_word(p3, self.W_vars[t], self.C_vars[t][6])

            S0 = self._xor3_word(_rotr_lits(a_t, 2), _rotr_lits(a_t, 13), _rotr_lits(a_t, 22))
            mj = [self._fresh() for _ in range(32)]
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


class CaDiCaLCDCLSolver:
    """Industrial CaDiCaL-195 CDCL wrapper with independent preimage verification."""

    def __init__(self, solver_name: str = "cadical195") -> None:
        self.solver_name = solver_name
        self.core = get_core()

    def solve(
        self,
        H_target: Sequence[int],
        rounds: int = 16,
        fixed_words: Optional[Dict[int, int]] = None,
        conflict_budget: int = -1,
        time_limit_sec: float = 15.0,
        phase_lits: Optional[Sequence[int]] = None,
        fixed_bits: Optional[Dict[Tuple[int, int], int]] = None,
        extra_clauses: Optional[Sequence[Sequence[int]]] = None,
    ) -> Dict[str, Any]:
        """Run blind CDCL search against target digest H and strictly verify SHA256_R(W[0..15])==H."""
        enc = SHA256CNFEncoder(rounds=rounds).build(
            H_target=H_target,
            fixed_words=fixed_words,
            fixed_bits=fixed_bits,
        )
        clauses = list(enc.clauses)
        if extra_clauses:
            clauses.extend(list(c) for c in extra_clauses if c)
        t0 = time.perf_counter()

        res_state: Optional[bool] = None
        w0_15: Optional[np.ndarray] = None
        valid_preimage = False
        stats: Dict[str, Any] = {}

        with Solver(name=self.solver_name, bootstrap_with=clauses) as solver:
            if phase_lits:
                try:
                    solver.set_phases(literals=list(phase_lits))
                except Exception:
                    pass

            chunk_confl = 4000
            while True:
                if time_limit_sec > 0.0 and (time.perf_counter() - t0) >= time_limit_sec:
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
                if time_limit_sec > 0.0 and (time.perf_counter() - t0) >= time_limit_sec:
                    break
                if conflict_budget > 0:
                    post_stats = solver.accum_stats() if hasattr(solver, "accum_stats") else {}
                    if int(post_stats.get("conflicts", 0)) >= conflict_budget:
                        break

            stats = solver.accum_stats() if hasattr(solver, "accum_stats") else {}
            if res_state is True:
                model = solver.get_model() or []
                model_set = {abs(lit): (lit > 0) for lit in model}
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
                ht_np = np.asarray(H_target, dtype=np.uint32)
                valid_preimage = bool(self.core.verify_preimage(w0_15, ht_np, rounds=rounds)[0])

        dt = time.perf_counter() - t0
        return {
            "rounds": int(rounds),
            "solver": self.solver_name,
            "num_vars": int(enc.next_var - 1),
            "num_clauses": int(len(enc.clauses)),
            "status": "SAT" if res_state is True else ("UNSAT" if res_state is False else "TIMEOUT_OR_BUDGET"),
            "valid_preimage": bool(valid_preimage),
            "conflicts": int(stats.get("conflicts", 0)),
            "decisions": int(stats.get("decisions", 0)),
            "propagations": int(stats.get("propagations", 0)),
            "elapsed_sec": float(dt),
            "w0_15": w0_15.tolist() if w0_15 is not None else None,
        }


try:
    import pycryptosat as _pycryptosat
except ImportError:
    _pycryptosat = None


class SHA256XORCNFEncoder(SHA256CNFEncoder):
    """
    面向原生 XOR-Gauss 求解器（CryptoMiniSat）的 SHA-256 混合编码器：
    - 将全部线性旋转异或（Σ0, Σ1, σ0, σ1）及 32-bit 模加求和位分解为紧凑奇偶校验 xor_clauses；
    - 保留 Maj / Ch 非线性门及常量单元约束为精确 CNF clauses（严守双端边界焊死）。
    """

    def __init__(self, rounds: int = 64) -> None:
        super().__init__(rounds=rounds)
        self.xor_clauses: List[Tuple[List[int], bool]] = []

    def _emit_xor_lin(self, terms: Sequence[Lit], rhs: bool = False) -> None:
        parity = bool(rhs)
        counts: Dict[int, int] = {}
        for item in terms:
            if isinstance(item, bool):
                if item:
                    parity = not parity
            else:
                v = int(item)
                if v < 0:
                    parity = not parity
                    v = -v
                counts[v] = (counts.get(v, 0) + 1) & 1
        vars_kept = sorted(v for v, c in counts.items() if c == 1)
        if not vars_kept:
            if parity:
                # 0 == 1 精确矛盾
                self.clauses.append([1])
                self.clauses.append([-1])
            return
        if len(vars_kept) == 1:
            v = vars_kept[0]
            self.clauses.append([v if parity else -v])
            return
        self.xor_clauses.append((vars_kept, parity))

    def _emit_eq(self, x: Lit, y: Lit) -> None:
        if isinstance(x, bool) or isinstance(y, bool):
            super()._emit_eq(x, y)
            return
        self._emit_xor_lin([x, y], rhs=False)

    def _emit_xor2(self, a: Lit, b: Lit, out: Lit) -> None:
        self._emit_xor_lin([a, b, out], rhs=False)

    def _emit_xor3(self, a: Lit, b: Lit, c: Lit, out: Lit) -> None:
        self._emit_xor_lin([a, b, c, out], rhs=False)

    def build(
        self,
        H_target: Sequence[int],
        fixed_words: Optional[Dict[int, int]] = None,
        fixed_bits: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> "SHA256XORCNFEncoder":
        self.xor_clauses.clear()
        super().build(H_target=H_target, fixed_words=fixed_words, fixed_bits=fixed_bits)
        return self


def extract_w0_15_from_cryptosat(
    enc: SHA256CNFEncoder,
    solution: Sequence[Optional[bool]],
) -> np.ndarray:
    """从 pycryptosat 返回的模型元组 (None, v1, v2, ...) 提取 W[0..15] uint32。"""
    sol_len = len(solution)
    w_rec = np.zeros((16,), dtype=np.uint32)
    for wi in range(16):
        word_val = 0
        for b in range(32):
            lit = enc.W_vars[wi][b]
            if isinstance(lit, bool):
                bit = 1 if lit else 0
            else:
                idx = abs(int(lit))
                raw_b = bool(solution[idx]) if (0 < idx < sol_len and solution[idx] is not None) else False
                bit = (1 - int(raw_b)) if int(lit) < 0 else int(raw_b)
            if bit:
                word_val |= (1 << b)
        w_rec[wi] = np.uint32(word_val)
    return w_rec


class CryptoMiniSatXORSolver:
    """原生 XOR-Gauss CDCL 求解封装（CryptoMiniSat-5 + 强制 verify_preimage 验真）。"""

    def __init__(self, threads: int = 1) -> None:
        self.threads = int(threads)
        self.core = get_core()

    @staticmethod
    def is_available() -> bool:
        return _pycryptosat is not None

    def new_solver(
        self,
        enc: SHA256XORCNFEncoder,
        extra_clauses: Optional[Sequence[Sequence[int]]] = None,
    ) -> Any:
        if _pycryptosat is None:
            raise ImportError("未安装 pycryptosat，无法启用原生 XOR-Gauss 求解器")
        solver = _pycryptosat.Solver(threads=self.threads)
        if enc.clauses:
            solver.add_clauses(enc.clauses)
        for x_vars, rhs in enc.xor_clauses:
            solver.add_xor_clause(x_vars, bool(rhs))
        if extra_clauses:
            solver.add_clauses([list(c) for c in extra_clauses if c])
        return solver

    def solve(
        self,
        H_target: Sequence[int],
        rounds: int = 16,
        fixed_words: Optional[Dict[int, int]] = None,
        fixed_bits: Optional[Dict[Tuple[int, int], int]] = None,
        conflict_budget: int = -1,
        time_limit_sec: float = 15.0,
        assumptions: Optional[Sequence[int]] = None,
        extra_clauses: Optional[Sequence[Sequence[int]]] = None,
    ) -> Dict[str, Any]:
        enc = SHA256XORCNFEncoder(rounds=rounds).build(
            H_target=H_target,
            fixed_words=fixed_words,
            fixed_bits=fixed_bits,
        )
        t0 = time.perf_counter()
        solver = self.new_solver(enc, extra_clauses=extra_clauses)
        solve_kwargs: Dict[str, Any] = {}
        if assumptions:
            solve_kwargs["assumptions"] = list(assumptions)
        if time_limit_sec > 0.0:
            solve_kwargs["time_limit"] = float(time_limit_sec)
        if conflict_budget > 0:
            solve_kwargs["confl_limit"] = int(conflict_budget)

        sat, solution = solver.solve(**solve_kwargs)
        dt = time.perf_counter() - t0
        w0_15: Optional[np.ndarray] = None
        valid_preimage = False
        if sat is True and solution is not None:
            w0_15 = extract_w0_15_from_cryptosat(enc, solution)
            ht_np = np.asarray(H_target, dtype=np.uint32)
            valid_preimage = bool(self.core.verify_preimage(w0_15, ht_np, rounds=rounds)[0])
        return {
            "rounds": int(rounds),
            "solver": "cryptominisat5_xor",
            "num_vars": int(enc.next_var - 1),
            "num_clauses": int(len(enc.clauses)),
            "num_xor_clauses": int(len(enc.xor_clauses)),
            "status": "SAT" if sat is True else ("UNSAT" if sat is False else "TIMEOUT_OR_BUDGET"),
            "valid_preimage": bool(valid_preimage),
            "conflicts": -1,  # pycryptosat C API 未导出冲突计数，统一用 elapsed_sec 与 SAT 率对齐比较
            "elapsed_sec": float(dt),
            "w0_15": w0_15.tolist() if w0_15 is not None else None,
        }
