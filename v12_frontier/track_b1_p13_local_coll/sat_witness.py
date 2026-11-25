"""Exact Tseitin-encoded PySAT (CaDiCaL195) SHA-256 witness & differential solver.

Supports:
  - Single-trace and dual-trace (M, M') differential SAT encoding (1..64 rounds)
  - Arbitrary 16-state Mendel generalized bit conditions on W, A, E, and digest H
  - Semi-Free-Start (custom IV), Free-Start (unconstrained IV), and standard IV
  - Truncated leading-zero output differential constraints: clz(H[0]) >= k & clz(H'[0]) >= k
  - Bitcoin Double-SHA256 Block 2 fixed-padding boundary & multi-witness enumeration
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from pysat.solvers import Solver

_THIS_DIR = Path(__file__).resolve().parent
_CANDIDATE_ROOTS = [_THIS_DIR.parents[1], _THIS_DIR.parent, _THIS_DIR]
for _root in _CANDIDATE_ROOTS:
    if (_root / "src" / "sha256_ref.py").exists():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break

from src.sha256_ref import MASK32, SHA256_IV, SHA256_K  # noqa: E402


class CNFBuilder:
    """Minimal, fast Tseitin CNF builder over PySAT CaDiCaL195."""

    def __init__(self, solver_name: str = "cadical195") -> None:
        self.solver = Solver(name=solver_name)
        self.next_var = 1
        self.num_clauses = 0

    def new_var(self) -> int:
        v = self.next_var
        self.next_var += 1
        return v

    def new_word(self) -> List[int]:
        """Allocate 32 LSB-first Boolean variables."""
        v0 = self.next_var
        self.next_var += 32
        return list(range(v0, v0 + 32))

    def add_clause(self, lits: Sequence[int]) -> None:
        self.solver.add_clause(list(lits))
        self.num_clauses += 1

    def fix_bit(self, var: int, val: int) -> None:
        self.add_clause([var if val else -var])

    def fix_word(self, word_vars: Sequence[int], val: int) -> None:
        u = val & MASK32
        for b in range(32):
            self.fix_bit(word_vars[b], (u >> b) & 1)

    def eq_bit(self, a: int, b: int) -> None:
        self.add_clause([-a, b])
        self.add_clause([a, -b])

    def neq_bit(self, a: int, b: int) -> None:
        self.add_clause([a, b])
        self.add_clause([-a, -b])

    def xor2(self, a: int, b: int) -> int:
        z = self.new_var()
        self.add_clause([-a, b, z])
        self.add_clause([a, -b, z])
        self.add_clause([a, b, -z])
        self.add_clause([-a, -b, -z])
        return z

    def xor3(self, a: int, b: int, c: int) -> int:
        z = self.new_var()
        self.add_clause([-a, -b, -c, z])
        self.add_clause([-a, b, c, z])
        self.add_clause([a, -b, c, z])
        self.add_clause([a, b, -c, z])
        self.add_clause([a, b, c, -z])
        self.add_clause([a, -b, -c, -z])
        self.add_clause([-a, b, -c, -z])
        self.add_clause([-a, -b, c, -z])
        return z

    def ch_bit(self, e: int, f: int, g: int) -> int:
        z = self.new_var()
        self.add_clause([-e, -f, z])
        self.add_clause([-e, f, -z])
        self.add_clause([e, -g, z])
        self.add_clause([e, g, -z])
        self.add_clause([-f, -g, z])
        self.add_clause([f, g, -z])
        return z

    def maj_bit(self, a: int, b: int, c: int) -> int:
        z = self.new_var()
        self.add_clause([-a, -b, z])
        self.add_clause([-a, -c, z])
        self.add_clause([-b, -c, z])
        self.add_clause([a, b, -z])
        self.add_clause([a, c, -z])
        self.add_clause([b, c, -z])
        return z

    def add2_words(self, x: Sequence[int], y: Sequence[int]) -> List[int]:
        """Exact 32-bit ripple-carry modular addition."""
        out = self.new_word()
        # Bit 0: half-adder
        self.add_clause([-x[0], y[0], out[0]])
        self.add_clause([x[0], -y[0], out[0]])
        self.add_clause([x[0], y[0], -out[0]])
        self.add_clause([-x[0], -y[0], -out[0]])
        # Carry out at bit 0: cin_1 = x[0] & y[0]
        cin = self.new_var()
        self.add_clause([-x[0], -y[0], cin])
        self.add_clause([x[0], -cin])
        self.add_clause([y[0], -cin])

        for b in range(1, 32):
            s = out[b]
            xb, yb = x[b], y[b]
            self.add_clause([-xb, -yb, -cin, s])
            self.add_clause([-xb, yb, cin, s])
            self.add_clause([xb, -yb, cin, s])
            self.add_clause([xb, yb, -cin, s])
            self.add_clause([xb, yb, cin, -s])
            self.add_clause([xb, -yb, -cin, -s])
            self.add_clause([-xb, yb, -cin, -s])
            self.add_clause([-xb, -yb, cin, -s])
            if b < 31:
                cout = self.new_var()
                self.add_clause([-xb, -yb, cout])
                self.add_clause([-xb, -cin, cout])
                self.add_clause([-yb, -cin, cout])
                self.add_clause([xb, yb, -cout])
                self.add_clause([xb, cin, -cout])
                self.add_clause([yb, cin, -cout])
                cin = cout
        return out

    def add_const_word(self, x: Sequence[int], c: int) -> List[int]:
        """Exact 32-bit modular addition with a constant uint32 word."""
        cu = c & MASK32
        if cu == 0:
            return list(x)
        out = self.new_word()
        cin: Optional[int] = None
        for b in range(32):
            cb = (cu >> b) & 1
            xb = x[b]
            sb = out[b]
            if cin is None:
                if cb == 0:
                    self.eq_bit(sb, xb)
                else:
                    # sb = ~xb, cout = xb
                    self.neq_bit(sb, xb)
                    if b < 31:
                        cin = xb
            else:
                if cb == 0:
                    # sb = xb ^ cin, cout = xb & cin
                    self.add_clause([-xb, cin, sb])
                    self.add_clause([xb, -cin, sb])
                    self.add_clause([xb, cin, -sb])
                    self.add_clause([-xb, -cin, -sb])
                    if b < 31:
                        cout = self.new_var()
                        self.add_clause([-xb, -cin, cout])
                        self.add_clause([xb, -cout])
                        self.add_clause([cin, -cout])
                        cin = cout
                else:
                    # sb = ~(xb ^ cin), cout = xb | cin
                    self.add_clause([-xb, -cin, sb])
                    self.add_clause([xb, cin, sb])
                    self.add_clause([xb, -cin, -sb])
                    self.add_clause([-xb, cin, -sb])
                    if b < 31:
                        cout = self.new_var()
                        self.add_clause([-xb, cout])
                        self.add_clause([-cin, cout])
                        self.add_clause([xb, cin, -cout])
                        cin = cout
        return out

    def add_many_words(self, words: Sequence[Sequence[int]], const: int = 0) -> List[int]:
        acc = list(words[0])
        for w in words[1:]:
            acc = self.add2_words(acc, w)
        if const & MASK32:
            acc = self.add_const_word(acc, const)
        return acc

    def rotr_word(self, x: Sequence[int], n: int) -> List[int]:
        if n == 0:
            return list(x)
        return [x[(b + n) % 32] for b in range(32)]

    def shr_word(self, x: Sequence[int], n: int, zero_var: int) -> List[int]:
        return [x[b + n] if b + n < 32 else zero_var for b in range(32)]

    def sigma0_upper(self, x: Sequence[int]) -> List[int]:
        r2, r13, r22 = self.rotr_word(x, 2), self.rotr_word(x, 13), self.rotr_word(x, 22)
        return [self.xor3(r2[b], r13[b], r22[b]) for b in range(32)]

    def sigma1_upper(self, x: Sequence[int]) -> List[int]:
        r6, r11, r25 = self.rotr_word(x, 6), self.rotr_word(x, 11), self.rotr_word(x, 25)
        return [self.xor3(r6[b], r11[b], r25[b]) for b in range(32)]

    def sigma0_lower(self, x: Sequence[int], zero_var: int) -> List[int]:
        r7, r18, s3 = self.rotr_word(x, 7), self.rotr_word(x, 18), self.shr_word(x, 3, zero_var)
        return [self.xor3(r7[b], r18[b], s3[b]) for b in range(32)]

    def sigma1_lower(self, x: Sequence[int], zero_var: int) -> List[int]:
        r17, r19, s10 = self.rotr_word(x, 17), self.rotr_word(x, 19), self.shr_word(x, 10, zero_var)
        return [self.xor3(r17[b], r19[b], s10[b]) for b in range(32)]

    def ch_word(self, e: Sequence[int], f: Sequence[int], g: Sequence[int]) -> List[int]:
        return [self.ch_bit(e[b], f[b], g[b]) for b in range(32)]

    def maj_word(self, a: Sequence[int], b: Sequence[int], c: Sequence[int]) -> List[int]:
        return [self.maj_bit(a[b_idx], b[b_idx], c[b_idx]) for b_idx in range(32)]

    def constrain_gdiff_bit(self, x_var: int, xp_var: int, state_mask: int) -> None:
        """Enforce 4-bit generalized differential state mask on (x_var, xp_var)."""
        m = state_mask & 0xF
        if m == 0xF:
            return
        if m == 0x0:
            self.add_clause([])
            return
        if (m & 0x1) == 0:  # forbid (0, 0)
            self.add_clause([x_var, xp_var])
        if (m & 0x2) == 0:  # forbid (0, 1)
            self.add_clause([x_var, -xp_var])
        if (m & 0x4) == 0:  # forbid (1, 0)
            self.add_clause([-x_var, xp_var])
        if (m & 0x8) == 0:  # forbid (1, 1)
            self.add_clause([-x_var, -xp_var])

    def constrain_gdiff_word(
        self, w_vars: Sequence[int], wp_vars: Sequence[int], masks32: Sequence[int]
    ) -> None:
        for b in range(32):
            self.constrain_gdiff_bit(w_vars[b], wp_vars[b], int(masks32[b]))

    def decode_word(self, model_set: set[int], word_vars: Sequence[int]) -> int:
        val = 0
        for b, v in enumerate(word_vars):
            if v in model_set:
                val |= 1 << b
        return val & MASK32

    def delete(self) -> None:
        self.solver.delete()


@dataclass
class SHA256SATInstance:
    """SAT variables for one SHA-256 execution trace."""

    iv_vars: List[List[int]]
    w16_vars: List[List[int]]
    w_vars: List[List[int]]
    a_vars: List[List[int]]  # length R + 4
    e_vars: List[List[int]]  # length R + 4
    digest_vars: List[List[int]]  # length 8


def encode_sha256_trace(
    cnf: CNFBuilder,
    rounds: int,
    zero_var: int,
    iv_vars: Optional[List[List[int]]] = None,
    w16_vars: Optional[List[List[int]]] = None,
) -> SHA256SATInstance:
    """Encode R-round SHA-256 compression + feed-forward digest into `cnf`."""
    if iv_vars is None:
        iv_vars = [cnf.new_word() for _ in range(8)]
    if w16_vars is None:
        w16_vars = [cnf.new_word() for _ in range(16)]

    w_vars: List[List[int]] = list(w16_vars[: min(rounds, 16)])
    for t in range(16, rounds):
        s1 = cnf.sigma1_lower(w_vars[t - 2], zero_var)
        s0 = cnf.sigma0_lower(w_vars[t - 15], zero_var)
        wt = cnf.add_many_words([s1, w_vars[t - 7], s0, w_vars[t - 16]])
        w_vars.append(wt)

    a_vars = [iv_vars[3], iv_vars[2], iv_vars[1], iv_vars[0]]
    e_vars = [iv_vars[7], iv_vars[6], iv_vars[5], iv_vars[4]]

    for t in range(rounds):
        s1 = cnf.sigma1_upper(e_vars[t + 3])
        ch = cnf.ch_word(e_vars[t + 3], e_vars[t + 2], e_vars[t + 1])
        t1 = cnf.add_many_words([e_vars[t], s1, ch, w_vars[t]], const=SHA256_K[t])
        s0 = cnf.sigma0_upper(a_vars[t + 3])
        maj = cnf.maj_word(a_vars[t + 3], a_vars[t + 2], a_vars[t + 1])
        t2 = cnf.add2_words(s0, maj)
        e_tp1 = cnf.add2_words(a_vars[t], t1)
        a_tp1 = cnf.add2_words(t1, t2)
        e_vars.append(e_tp1)
        a_vars.append(a_tp1)

    s_r = [
        a_vars[rounds + 3],
        a_vars[rounds + 2],
        a_vars[rounds + 1],
        a_vars[rounds],
        e_vars[rounds + 3],
        e_vars[rounds + 2],
        e_vars[rounds + 1],
        e_vars[rounds],
    ]
    digest_vars = [cnf.add2_words(s_r[i], iv_vars[i]) for i in range(8)]
    return SHA256SATInstance(
        iv_vars=iv_vars,
        w16_vars=w16_vars,
        w_vars=w_vars,
        a_vars=a_vars,
        e_vars=e_vars,
        digest_vars=digest_vars,
    )


@dataclass
class WitnessResult:
    """Result of solving for a concrete SHA-256 differential witness pair (M, M')."""

    sat: bool
    solve_seconds: float
    num_vars: int
    num_clauses: int
    w16: Optional[List[int]] = None
    w16_prime: Optional[List[int]] = None
    iv: Optional[List[int]] = None
    iv_prime: Optional[List[int]] = None
    digest: Optional[List[int]] = None
    digest_prime: Optional[List[int]] = None


def solve_differential_witness(
    rounds: int,
    iv_mode: str = "standard",  # "standard", "semi_free", "free"
    w_gdiff: Optional[Dict[int, Sequence[int]]] = None,
    a_gdiff: Optional[Dict[int, Sequence[int]]] = None,
    e_gdiff: Optional[Dict[int, Sequence[int]]] = None,
    digest_xor_zero_words: Optional[Sequence[int]] = None,
    leading_zero_bits: Optional[int] = None,
    require_nonzero_msg_diff: bool = True,
    btc_block2_padding: bool = False,
    fixed_w_indices: Optional[Dict[int, int]] = None,
    max_solutions: int = 1,
) -> List[WitnessResult]:
    """Build dual-trace SAT instance and solve for exact message pair(s) (M, M').

    Args:
        rounds: number of SHA-256 rounds (1..64).
        iv_mode:
          - "standard": both M and M' use SHA256_IV.
          - "semi_free": IV == IV', but IV is free to vary.
          - "free": IV and IV' are independent free variables.
        w_gdiff / a_gdiff / e_gdiff: dicts mapping word/round index -> 32-element 16-state masks.
          Note: for a_gdiff/e_gdiff, key is algebraic index k in [-3, rounds], stored at k+3.
        digest_xor_zero_words: digest word indices (0..7) forced to have zero XOR diff.
        leading_zero_bits: if set (1..32), force top k bits of H[0] AND H'[0] to be 0.
        require_nonzero_msg_diff: forbid M == M' (at least one message word bit differs).
        btc_block2_padding: if True, lock W[4]=W'[4]=0x80000000, W[5..14]=0, W[15]=640.
        fixed_w_indices: optional {idx: uint32_val} fixed on M (and M' unless gdiff overrides).
        max_solutions: number of distinct (M, M') witnesses to enumerate.
    """
    t0 = time.perf_counter()
    cnf = CNFBuilder("cadical195")
    zero_var = cnf.new_var()
    cnf.fix_bit(zero_var, 0)

    tr1 = encode_sha256_trace(cnf, rounds=rounds, zero_var=zero_var)
    tr2 = encode_sha256_trace(cnf, rounds=rounds, zero_var=zero_var)

    # IV mode
    if iv_mode == "standard":
        for i in range(8):
            cnf.fix_word(tr1.iv_vars[i], SHA256_IV[i])
            cnf.fix_word(tr2.iv_vars[i], SHA256_IV[i])
    elif iv_mode == "semi_free":
        for i in range(8):
            for b in range(32):
                cnf.eq_bit(tr1.iv_vars[i][b], tr2.iv_vars[i][b])
    elif iv_mode == "free":
        pass
    else:
        cnf.delete()
        raise ValueError(f"Unsupported iv_mode: {iv_mode}")

    # BTC Block 2 fixed padding
    if btc_block2_padding:
        for tr in (tr1, tr2):
            cnf.fix_word(tr.w16_vars[4], 0x80000000)
            for idx in range(5, 15):
                cnf.fix_word(tr.w16_vars[idx], 0)
            cnf.fix_word(tr.w16_vars[15], 640)

    if fixed_w_indices:
        for idx, val in fixed_w_indices.items():
            w1 = tr1.w16_vars[idx] if idx < 16 else tr1.w_vars[idx]
            w2 = tr2.w16_vars[idx] if idx < 16 else tr2.w_vars[idx]
            cnf.fix_word(w1, val)
            if w_gdiff is None or idx not in w_gdiff:
                cnf.fix_word(w2, val)

    # Apply generalized differential constraints
    if w_gdiff:
        for idx, masks in w_gdiff.items():
            w1 = tr1.w16_vars[idx] if idx < 16 else tr1.w_vars[idx]
            w2 = tr2.w16_vars[idx] if idx < 16 else tr2.w_vars[idx]
            cnf.constrain_gdiff_word(w1, w2, masks)
    if a_gdiff:
        for k, masks in a_gdiff.items():
            cnf.constrain_gdiff_word(tr1.a_vars[k + 3], tr2.a_vars[k + 3], masks)
    if e_gdiff:
        for k, masks in e_gdiff.items():
            cnf.constrain_gdiff_word(tr1.e_vars[k + 3], tr2.e_vars[k + 3], masks)

    # Digest collision / truncated constraints
    if digest_xor_zero_words:
        for idx in digest_xor_zero_words:
            for b in range(32):
                cnf.eq_bit(tr1.digest_vars[idx][b], tr2.digest_vars[idx][b])

    if leading_zero_bits is not None and leading_zero_bits > 0:
        if leading_zero_bits > 32:
            cnf.delete()
            raise ValueError("leading_zero_bits must be <= 32")
        for b in range(32 - leading_zero_bits, 32):
            cnf.fix_bit(tr1.digest_vars[0][b], 0)
            cnf.fix_bit(tr2.digest_vars[0][b], 0)

    if require_nonzero_msg_diff:
        diff_flags = []
        for idx in range(min(rounds, 16)):
            for b in range(32):
                d = cnf.xor2(tr1.w16_vars[idx][b], tr2.w16_vars[idx][b])
                diff_flags.append(d)
        cnf.add_clause(diff_flags)

    results: List[WitnessResult] = []
    try:
        for _ in range(max_solutions):
            sat = cnf.solver.solve()
            elapsed = time.perf_counter() - t0
            if not sat:
                if not results:
                    results.append(
                        WitnessResult(
                            sat=False,
                            solve_seconds=elapsed,
                            num_vars=cnf.next_var - 1,
                            num_clauses=cnf.num_clauses,
                        )
                    )
                break
            model = cnf.solver.get_model()
            mset = {lit for lit in model if lit > 0}
            w16 = [cnf.decode_word(mset, tr1.w16_vars[i]) for i in range(16)]
            w16p = [cnf.decode_word(mset, tr2.w16_vars[i]) for i in range(16)]
            iv = [cnf.decode_word(mset, tr1.iv_vars[i]) for i in range(8)]
            ivp = [cnf.decode_word(mset, tr2.iv_vars[i]) for i in range(8)]
            dig = [cnf.decode_word(mset, tr1.digest_vars[i]) for i in range(8)]
            digp = [cnf.decode_word(mset, tr2.digest_vars[i]) for i in range(8)]
            results.append(
                WitnessResult(
                    sat=True,
                    solve_seconds=elapsed,
                    num_vars=cnf.next_var - 1,
                    num_clauses=cnf.num_clauses,
                    w16=w16,
                    w16_prime=w16p,
                    iv=iv,
                    iv_prime=ivp,
                    digest=dig,
                    digest_prime=digp,
                )
            )
            # Block this (w16, w16_prime) pair for next enumeration
            if len(results) < max_solutions:
                block_lits = []
                for i in range(min(rounds, 16)):
                    for b in range(32):
                        v1 = tr1.w16_vars[i][b]
                        v2 = tr2.w16_vars[i][b]
                        block_lits.append(-v1 if v1 in mset else v1)
                        block_lits.append(-v2 if v2 in mset else v2)
                cnf.add_clause(block_lits)
    finally:
        cnf.delete()

    return results
