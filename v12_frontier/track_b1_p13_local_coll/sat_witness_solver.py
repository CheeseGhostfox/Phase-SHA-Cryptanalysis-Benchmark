"""Exact Tseitin PySAT (CaDiCaL195) differential witness & parity solver for P13.

Aligned with SPEC.md:
  - Encodes single-block (S1), BTC Block 2 fixed-padding (S2a/S2b), and Double-SHA256 (S3)
  - Supports custom constants & rotation tuples for G4 CTRL_ROTR_K control ciphers
  - Applies 16-state Mendel generalized conditions (idx = x + 2*x') directly to CNF
  - Includes 10,000-sample exact bit-level parity check against src/sha256_ref.py (xor_diff == 0)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from pysat.solvers import Solver

_THIS_DIR = Path(__file__).resolve().parent
for _root in [_THIS_DIR, *_THIS_DIR.parents]:
    if (_root / "src" / "sha256_ref.py").exists():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break

from src.sha256_ref import MASK32, SHA256_IV, SHA256_K, sha256_digest  # noqa: E402


class CNFBuilder:
    """Minimal, fast Tseitin CNF builder backed by PySAT CaDiCaL195."""

    def __init__(self, solver_name: str = "cadical195") -> None:
        self.solver = Solver(name=solver_name)
        self.next_var = 1
        self.num_clauses = 0

    def new_var(self) -> int:
        v = self.next_var
        self.next_var += 1
        return v

    def new_word(self) -> List[int]:
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
        out = self.new_word()
        self.add_clause([-x[0], y[0], out[0]])
        self.add_clause([x[0], -y[0], out[0]])
        self.add_clause([x[0], y[0], -out[0]])
        self.add_clause([-x[0], -y[0], -out[0]])
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
                    self.neq_bit(sb, xb)
                    if b < 31:
                        cin = xb
            else:
                if cb == 0:
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

    def sigma_upper(self, x: Sequence[int], shifts: Tuple[int, int, int]) -> List[int]:
        r0, r1, r2 = self.rotr_word(x, shifts[0]), self.rotr_word(x, shifts[1]), self.rotr_word(x, shifts[2])
        return [self.xor3(r0[b], r1[b], r2[b]) for b in range(32)]

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
        """Enforce SPEC 2.1 4-bit state mask (idx = x + 2*x') on (x_var, xp_var)."""
        m = state_mask & 0xF
        if m == 0xF:
            return
        if m == 0x0:
            self.add_clause([])
            return
        if (m & 0x1) == 0:  # forbid (0, 0)
            self.add_clause([x_var, xp_var])
        if (m & 0x2) == 0:  # forbid (1, 0) -> 'u'
            self.add_clause([-x_var, xp_var])
        if (m & 0x4) == 0:  # forbid (0, 1) -> 'n'
            self.add_clause([x_var, -xp_var])
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
    iv_vars: List[List[int]]
    w16_vars: List[List[int]]
    w_vars: List[List[int]]
    a_vars: List[List[int]]
    e_vars: List[List[int]]
    digest_vars: List[List[int]]


def encode_sha256_trace(
    cnf: CNFBuilder,
    rounds: int,
    zero_var: int,
    iv_vars: Optional[List[List[int]]] = None,
    w16_vars: Optional[List[List[int]]] = None,
    k_consts: Sequence[int] = SHA256_K,
    sigma0_shifts: Tuple[int, int, int] = (2, 13, 22),
    sigma1_shifts: Tuple[int, int, int] = (6, 11, 25),
) -> SHA256SATInstance:
    """Encode R-round SHA-256 (or isomorphic CTRL_ROTR_K variant) into `cnf`."""
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
        s1 = cnf.sigma_upper(e_vars[t + 3], sigma1_shifts)
        ch = cnf.ch_word(e_vars[t + 3], e_vars[t + 2], e_vars[t + 1])
        t1 = cnf.add_many_words([e_vars[t], s1, ch, w_vars[t]], const=k_consts[t])
        s0 = cnf.sigma_upper(a_vars[t + 3], sigma0_shifts)
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
    iv_mode: str = "standard",
    custom_iv: Optional[Sequence[int]] = None,
    w_gdiff: Optional[Dict[int, Sequence[int]]] = None,
    a_gdiff: Optional[Dict[int, Sequence[int]]] = None,
    e_gdiff: Optional[Dict[int, Sequence[int]]] = None,
    digest_xor_zero_words: Optional[Sequence[int]] = None,
    leading_zero_bits: Optional[int] = None,
    require_nonzero_msg_diff: bool = True,
    btc_block2_padding: bool = False,
    fixed_w_indices: Optional[Dict[int, int]] = None,
    max_solutions: int = 1,
    conflict_budget: int = 0,
    k_consts: Sequence[int] = SHA256_K,
    sigma0_shifts: Tuple[int, int, int] = (2, 13, 22),
    sigma1_shifts: Tuple[int, int, int] = (6, 11, 25),
) -> List[WitnessResult]:
    """Solve for concrete (M, M') witnesses under generalized differential constraints."""
    t0 = time.perf_counter()
    cnf = CNFBuilder("cadical195")
    zero_var = cnf.new_var()
    cnf.fix_bit(zero_var, 0)

    tr1 = encode_sha256_trace(
        cnf, rounds=rounds, zero_var=zero_var,
        k_consts=k_consts, sigma0_shifts=sigma0_shifts, sigma1_shifts=sigma1_shifts,
    )
    tr2 = encode_sha256_trace(
        cnf, rounds=rounds, zero_var=zero_var,
        k_consts=k_consts, sigma0_shifts=sigma0_shifts, sigma1_shifts=sigma1_shifts,
    )

    iv_ref = custom_iv if custom_iv is not None else SHA256_IV
    if iv_mode == "standard":
        for i in range(8):
            cnf.fix_word(tr1.iv_vars[i], iv_ref[i])
            cnf.fix_word(tr2.iv_vars[i], iv_ref[i])
    elif iv_mode == "semi_free":
        for i in range(8):
            for b in range(32):
                cnf.eq_bit(tr1.iv_vars[i][b], tr2.iv_vars[i][b])
    elif iv_mode == "free":
        pass
    else:
        cnf.delete()
        raise ValueError(f"Unsupported iv_mode: {iv_mode}")

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

    if digest_xor_zero_words:
        for idx in digest_xor_zero_words:
            for b in range(32):
                cnf.eq_bit(tr1.digest_vars[idx][b], tr2.digest_vars[idx][b])

    if leading_zero_bits is not None and leading_zero_bits > 0:
        for b in range(32 - leading_zero_bits, 32):
            cnf.fix_bit(tr1.digest_vars[0][b], 0)
            cnf.fix_bit(tr2.digest_vars[0][b], 0)

    if require_nonzero_msg_diff:
        diff_flags = []
        for idx in range(min(rounds, 16)):
            for b in range(32):
                diff_flags.append(cnf.xor2(tr1.w16_vars[idx][b], tr2.w16_vars[idx][b]))
        cnf.add_clause(diff_flags)

    results: List[WitnessResult] = []
    try:
        if conflict_budget > 0:
            cnf.solver.conf_budget(conflict_budget)
        for _ in range(max_solutions):
            sat = cnf.solver.solve_limited() if conflict_budget > 0 else cnf.solver.solve()
            elapsed = time.perf_counter() - t0
            if sat is not True:
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
            if len(results) < max_solutions:
                block_lits = []
                for i in range(16):
                    for b in range(32):
                        v1 = tr1.w16_vars[i][b]
                        v2 = tr2.w16_vars[i][b]
                        block_lits.append(-v1 if v1 in mset else v1)
                        block_lits.append(-v2 if v2 in mset else v2)
                cnf.add_clause(block_lits)
    finally:
        cnf.delete()
    return results


def solve_double_sha256_witness(
    r1: int,
    r2: int,
    leading_zero_bits: int,
    w_gdiff_b1: Optional[Dict[int, Sequence[int]]] = None,
    btc_block2_padding: bool = True,
    require_nonzero_msg_diff: bool = True,
    conflict_budget: int = 0,
    max_solutions: int = 1,
) -> List[WitnessResult]:
    """Solve S3 Double-SHA256 truncated leading-zero differential witness (R1 + R2)."""
    t0 = time.perf_counter()
    cnf = CNFBuilder("cadical195")
    zero_var = cnf.new_var()
    cnf.fix_bit(zero_var, 0)

    # First compression (Block 1 / BTC Block 2)
    b1_a = encode_sha256_trace(cnf, rounds=r1, zero_var=zero_var)
    b1_b = encode_sha256_trace(cnf, rounds=r1, zero_var=zero_var)
    for i in range(8):
        cnf.fix_word(b1_a.iv_vars[i], SHA256_IV[i])
        cnf.fix_word(b1_b.iv_vars[i], SHA256_IV[i])

    if btc_block2_padding:
        for tr in (b1_a, b1_b):
            cnf.fix_word(tr.w16_vars[4], 0x80000000)
            for idx in range(5, 15):
                cnf.fix_word(tr.w16_vars[idx], 0)
            cnf.fix_word(tr.w16_vars[15], 640)

    if w_gdiff_b1:
        for idx, masks in w_gdiff_b1.items():
            w1 = b1_a.w16_vars[idx] if idx < 16 else b1_a.w_vars[idx]
            w2 = b1_b.w16_vars[idx] if idx < 16 else b1_b.w_vars[idx]
            cnf.constrain_gdiff_word(w1, w2, masks)

    # Second compression: W''[0..7] = H^(1), W''[8]=0x80000000, W''[9..14]=0, W''[15]=256
    def _build_b2_w16(h1_vars: List[List[int]]) -> List[List[int]]:
        w16 = list(h1_vars)
        w8 = cnf.new_word()
        cnf.fix_word(w8, 0x80000000)
        w16.append(w8)
        for _ in range(9, 15):
            wz = cnf.new_word()
            cnf.fix_word(wz, 0)
            w16.append(wz)
        w15 = cnf.new_word()
        cnf.fix_word(w15, 256)
        w16.append(w15)
        return w16

    b2_a = encode_sha256_trace(cnf, rounds=r2, zero_var=zero_var, w16_vars=_build_b2_w16(b1_a.digest_vars))
    b2_b = encode_sha256_trace(cnf, rounds=r2, zero_var=zero_var, w16_vars=_build_b2_w16(b1_b.digest_vars))
    for i in range(8):
        cnf.fix_word(b2_a.iv_vars[i], SHA256_IV[i])
        cnf.fix_word(b2_b.iv_vars[i], SHA256_IV[i])

    for b in range(32 - leading_zero_bits, 32):
        cnf.fix_bit(b2_a.digest_vars[0][b], 0)
        cnf.fix_bit(b2_b.digest_vars[0][b], 0)

    if require_nonzero_msg_diff:
        diff_flags = []
        for idx in range(4 if btc_block2_padding else 16):
            for b in range(32):
                diff_flags.append(cnf.xor2(b1_a.w16_vars[idx][b], b1_b.w16_vars[idx][b]))
        cnf.add_clause(diff_flags)

    results: List[WitnessResult] = []
    try:
        if conflict_budget > 0:
            cnf.solver.conf_budget(conflict_budget)
        for _ in range(max_solutions):
            sat = cnf.solver.solve_limited() if conflict_budget > 0 else cnf.solver.solve()
            elapsed = time.perf_counter() - t0
            if sat is not True:
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
            w16 = [cnf.decode_word(mset, b1_a.w16_vars[i]) for i in range(16)]
            w16p = [cnf.decode_word(mset, b1_b.w16_vars[i]) for i in range(16)]
            dig = [cnf.decode_word(mset, b2_a.digest_vars[i]) for i in range(8)]
            digp = [cnf.decode_word(mset, b2_b.digest_vars[i]) for i in range(8)]
            results.append(
                WitnessResult(
                    sat=True,
                    solve_seconds=elapsed,
                    num_vars=cnf.next_var - 1,
                    num_clauses=cnf.num_clauses,
                    w16=w16,
                    w16_prime=w16p,
                    iv=list(SHA256_IV),
                    iv_prime=list(SHA256_IV),
                    digest=dig,
                    digest_prime=digp,
                )
            )
            if len(results) < max_solutions:
                block_lits = []
                for i in range(4 if btc_block2_padding else 16):
                    for b in range(32):
                        v1 = b1_a.w16_vars[i][b]
                        v2 = b1_b.w16_vars[i][b]
                        block_lits.append(-v1 if v1 in mset else v1)
                        block_lits.append(-v2 if v2 in mset else v2)
                cnf.add_clause(block_lits)
    finally:
        cnf.delete()
    return results


def run_sat_parity_check(n_samples: int = 10_000, rounds: int = 64, seed: int = 42) -> Dict[str, object]:
    """SPEC 2.4 item 3: verify CaDiCaL195 Tseitin encoding vs sha256_ref on 10^4 random messages."""
    t0 = time.perf_counter()
    gen = torch.Generator().manual_seed(seed)
    w_all = torch.randint(0, 1 << 32, (n_samples, 16), dtype=torch.int64, generator=gen)
    ref_digests = sha256_digest(w_all, rounds=rounds)

    cnf = CNFBuilder("cadical195")
    zero_var = cnf.new_var()
    cnf.fix_bit(zero_var, 0)
    tr = encode_sha256_trace(cnf, rounds=rounds, zero_var=zero_var)
    for i in range(8):
        cnf.fix_word(tr.iv_vars[i], SHA256_IV[i])

    xor_mismatches = 0
    try:
        for s_idx in range(n_samples):
            assump = []
            w_row = w_all[s_idx].tolist()
            for i in range(16):
                u = w_row[i] & MASK32
                for b in range(32):
                    v = tr.w16_vars[i][b]
                    assump.append(v if ((u >> b) & 1) else -v)
            assert cnf.solver.solve(assumptions=assump) is True
            mset = {lit for lit in cnf.solver.get_model() if lit > 0}
            got = [cnf.decode_word(mset, tr.digest_vars[i]) for i in range(8)]
            exp = ref_digests[s_idx].tolist()
            if any((g ^ e) != 0 for g, e in zip(got, exp)):
                xor_mismatches += 1
            if (s_idx + 1) % 2500 == 0 or s_idx + 1 == n_samples:
                print(f"[sat_parity] checked {s_idx + 1}/{n_samples}, xor_mismatches={xor_mismatches}", flush=True)
    finally:
        cnf.delete()

    elapsed = time.perf_counter() - t0
    report = {
        "n_samples": n_samples,
        "rounds": rounds,
        "xor_mismatches": xor_mismatches,
        "elapsed_seconds": round(elapsed, 3),
        "passed": (xor_mismatches == 0),
    }
    print(f"[sat_parity] complete: {json.dumps(report)}", flush=True)
    return report


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    rep = run_sat_parity_check(n_samples=10_000, rounds=64)
    out_dir = _THIS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sat_parity_10k.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    if not rep["passed"]:
        sys.exit(1)
