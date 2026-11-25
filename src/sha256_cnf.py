"""Bit-exact SHA-256 Tseitin CNF encoder and zero-leakage R=18 algebraic reduction helpers."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src.boundary_clamp import build_clamped_boundary
from src.sha256_ref import (
    MASK32,
    ch32,
    get_iv_tensor,
    get_k_tensor,
    maj32,
    sigma0_lower,
    sigma0_upper,
    sigma1_lower,
    sigma1_upper,
)


def _int_to_unit_clauses(
    vars_32: Sequence[int], val: int, mask_bits: Optional[Sequence[float]] = None
) -> List[List[int]]:
    out: List[List[int]] = []
    ival = int(val) & MASK32
    for b in range(32):
        if mask_bits is not None and float(mask_bits[b]) < 0.5:
            continue
        bit = (ival >> b) & 1
        out.append([vars_32[b] if bit == 1 else -vars_32[b]])
    return out


class SHA256CNFEncoder:
    """Bit-exact Tseitin CNF encoder for R-round single-block SHA-256 (LSB-first)."""

    def __init__(self, rounds: int = 18) -> None:
        if not (8 <= rounds <= 64):
            raise ValueError(f"rounds must be in [8, 64], got {rounds}")
        self.rounds = rounds
        self._next_var = 1
        self.base_clauses: List[List[int]] = []

        # Constant literals (enforced at level 0)
        self.const0 = self.new_var()
        self.const1 = self.new_var()
        self.base_clauses.append([-self.const0])
        self.base_clauses.append([self.const1])

        # Public SHA-256 node variables matching Phase-SHA layout (A_{-3..R}, E_{-3..R}, W_{0..R-1})
        self.A_vars: List[List[int]] = [[self.new_var() for _ in range(32)] for _ in range(rounds + 4)]
        self.E_vars: List[List[int]] = [[self.new_var() for _ in range(32)] for _ in range(rounds + 4)]
        self.W_vars: List[List[int]] = [[self.new_var() for _ in range(32)] for _ in range(rounds)]

        self._encode_base_network()

    def new_var(self) -> int:
        v = self._next_var
        self._next_var += 1
        return v

    def new_word(self) -> List[int]:
        return [self.new_var() for _ in range(32)]

    @property
    def num_vars(self) -> int:
        return self._next_var - 1

    def node_var(self, t: int, ch: int, b: int) -> int:
        """Map (round t in [0..R-1], ch in {0:A, 1:E, 2:W}, bit b in [0..31]) -> DIMACS var."""
        if ch == 0:
            return self.A_vars[t + 3][b]
        if ch == 1:
            return self.E_vars[t + 3][b]
        if ch == 2:
            return self.W_vars[t][b]
        raise ValueError(f"Invalid channel {ch}")

    def _add_and2(self, a: int, b: int, z: int) -> None:
        self.base_clauses.append([-a, -b, z])
        self.base_clauses.append([a, -z])
        self.base_clauses.append([b, -z])

    def _add_xor2(self, a: int, b: int, z: int) -> None:
        self.base_clauses.append([a, b, -z])
        self.base_clauses.append([a, -b, z])
        self.base_clauses.append([-a, b, z])
        self.base_clauses.append([-a, -b, -z])

    def _add_xor3(self, a: int, b: int, c: int, z: int) -> None:
        self.base_clauses.append([a, b, c, -z])
        self.base_clauses.append([a, b, -c, z])
        self.base_clauses.append([a, -b, c, z])
        self.base_clauses.append([-a, b, c, z])
        self.base_clauses.append([a, -b, -c, -z])
        self.base_clauses.append([-a, b, -c, -z])
        self.base_clauses.append([-a, -b, c, -z])
        self.base_clauses.append([-a, -b, -c, z])

    def _add_ch_bit(self, e: int, f: int, g: int, z: int) -> None:
        self.base_clauses.append([-e, -f, z])
        self.base_clauses.append([-e, f, -z])
        self.base_clauses.append([e, -g, z])
        self.base_clauses.append([e, g, -z])
        self.base_clauses.append([-f, -g, z])
        self.base_clauses.append([f, g, -z])

    def _add_maj_bit(self, a: int, b: int, c: int, z: int) -> None:
        self.base_clauses.append([-a, -b, z])
        self.base_clauses.append([-a, -c, z])
        self.base_clauses.append([-b, -c, z])
        self.base_clauses.append([a, b, -z])
        self.base_clauses.append([a, c, -z])
        self.base_clauses.append([b, c, -z])

    def _encode_xor3_word(
        self, a: Sequence[int], b: Sequence[int], c: Sequence[int]
    ) -> List[int]:
        out = self.new_word()
        for k in range(32):
            self._add_xor3(a[k], b[k], c[k], out[k])
        return out

    def _encode_ch_word(
        self, e: Sequence[int], f: Sequence[int], g: Sequence[int]
    ) -> List[int]:
        out = self.new_word()
        for k in range(32):
            self._add_ch_bit(e[k], f[k], g[k], out[k])
        return out

    def _encode_maj_word(
        self, a: Sequence[int], b: Sequence[int], c: Sequence[int]
    ) -> List[int]:
        out = self.new_word()
        for k in range(32):
            self._add_maj_bit(a[k], b[k], c[k], out[k])
        return out

    def _encode_add2(
        self,
        a: Sequence[int],
        b: Sequence[int],
        out: Optional[List[int]] = None,
    ) -> List[int]:
        res = self.new_word() if out is None else out
        self._add_xor2(a[0], b[0], res[0])
        c = self.new_var()
        self._add_and2(a[0], b[0], c)
        for k in range(1, 32):
            self._add_xor3(a[k], b[k], c, res[k])
            if k < 31:
                c_next = self.new_var()
                self._add_maj_bit(a[k], b[k], c, c_next)
                c = c_next
        return res

    @staticmethod
    def _rotr(word: Sequence[int], n: int) -> List[int]:
        return [word[(k + n) % 32] for k in range(32)]

    def _shr(self, word: Sequence[int], n: int) -> List[int]:
        return [word[k + n] if (k + n) < 32 else self.const0 for k in range(32)]

    def _const_word(self, val: int) -> List[int]:
        ival = int(val) & MASK32
        return [self.const1 if ((ival >> k) & 1) else self.const0 for k in range(32)]

    def _encode_base_network(self) -> None:
        k_vec = get_k_tensor(self.rounds, device="cpu").tolist()

        # 1. Message schedule expansion t = 16 .. R-1
        for t in range(16, self.rounds):
            w_tm2 = self.W_vars[t - 2]
            w_tm7 = self.W_vars[t - 7]
            w_tm15 = self.W_vars[t - 15]
            w_tm16 = self.W_vars[t - 16]
            s1 = self._encode_xor3_word(
                self._rotr(w_tm2, 17), self._rotr(w_tm2, 19), self._shr(w_tm2, 10)
            )
            s0 = self._encode_xor3_word(
                self._rotr(w_tm15, 7), self._rotr(w_tm15, 18), self._shr(w_tm15, 3)
            )
            u = self._encode_add2(s1, w_tm7)
            v = self._encode_add2(s0, w_tm16)
            self._encode_add2(u, v, out=self.W_vars[t])

        # 2. Compression rounds t = 0 .. R-1
        for t in range(self.rounds):
            a_t = self.A_vars[t + 3]
            a_tm1 = self.A_vars[t + 2]
            a_tm2 = self.A_vars[t + 1]
            a_tm3 = self.A_vars[t]

            e_t = self.E_vars[t + 3]
            e_tm1 = self.E_vars[t + 2]
            e_tm2 = self.E_vars[t + 1]
            e_tm3 = self.E_vars[t]

            big_s0 = self._encode_xor3_word(
                self._rotr(a_t, 2), self._rotr(a_t, 13), self._rotr(a_t, 22)
            )
            big_s1 = self._encode_xor3_word(
                self._rotr(e_t, 6), self._rotr(e_t, 11), self._rotr(e_t, 25)
            )
            ch_w = self._encode_ch_word(e_t, e_tm1, e_tm2)
            maj_w = self._encode_maj_word(a_t, a_tm1, a_tm2)

            k_const = self._const_word(int(k_vec[t]))
            sum_kw = self._encode_add2(k_const, self.W_vars[t])
            sum_sc = self._encode_add2(big_s1, ch_w)
            sum_e_sc = self._encode_add2(e_tm3, sum_sc)
            t1 = self._encode_add2(sum_e_sc, sum_kw)
            t2 = self._encode_add2(big_s0, maj_w)

            self._encode_add2(a_tm3, t1, out=self.E_vars[t + 4])
            self._encode_add2(t1, t2, out=self.A_vars[t + 4])

    def build_instance_unit_clauses(
        self,
        digest: torch.Tensor,
        domain_mode: str = "int_range",
        num_bits: int = 32,
        msg_bytes_len: int = 4,
        include_algebraic: bool = False,
    ) -> Tuple[List[List[int]], Dict[Tuple[int, int], int]]:
        """Build boundary unit clauses for one digest (0 leakage; purely digest + public IV/padding).

        Returns:
            (unit_clauses, fixed_node_dict) where fixed_node_dict maps (t, ch) -> uint32 value
            for all clamped/algebraically-derived internal words in [0..R-1].
        """
        dig_cpu = digest.detach().cpu().to(torch.int64)
        if dig_cpu.ndim == 2:
            dig_cpu = dig_cpu[0]
        iv = get_iv_tensor(device="cpu")
        boundary = build_clamped_boundary(
            digest=dig_cpu.unsqueeze(0),
            rounds=self.rounds,
            domain_mode=domain_mode,
            num_bits=num_bits,
            msg_bytes_len=msg_bytes_len,
        )

        clauses: List[List[int]] = []
        fixed_nodes: Dict[Tuple[int, int], int] = {}

        # Left boundary IV: A_{-3..0} = IV[3..0], E_{-3..0} = IV[7..4]
        iv_list = [int(v) for v in iv.tolist()]
        for idx, iv_val in enumerate((iv_list[3], iv_list[2], iv_list[1], iv_list[0])):
            clauses.extend(_int_to_unit_clauses(self.A_vars[idx], iv_val))
        for idx, iv_val in enumerate((iv_list[7], iv_list[6], iv_list[5], iv_list[4])):
            clauses.extend(_int_to_unit_clauses(self.E_vars[idx], iv_val))
        fixed_nodes[(0, 0)] = iv_list[0]
        fixed_nodes[(0, 1)] = iv_list[4]

        # Right boundary digest: S_R = (digest - IV) mod 2^32 at A_{R-3..R}, E_{R-3..R}
        s_r = ((dig_cpu - iv) & MASK32).tolist()
        for offset, val in enumerate((s_r[3], s_r[2], s_r[1], s_r[0])):
            idx = self.rounds + offset
            clauses.extend(_int_to_unit_clauses(self.A_vars[idx], int(val)))
            if idx - 3 < self.rounds:
                fixed_nodes[(idx - 3, 0)] = int(val)
        for offset, val in enumerate((s_r[7], s_r[6], s_r[5], s_r[4])):
            idx = self.rounds + offset
            clauses.extend(_int_to_unit_clauses(self.E_vars[idx], int(val)))
            if idx - 3 < self.rounds:
                fixed_nodes[(idx - 3, 1)] = int(val)

        # Clamped padding / domain mask bits in W_{0..min(R,16)-1}
        k_lim = min(self.rounds, 16)
        w_words = boundary.W16_clamped_words[0].tolist()
        w_mask = boundary.W16_mask_bits[0].tolist()
        for t in range(k_lim):
            clauses.extend(_int_to_unit_clauses(self.W_vars[t], int(w_words[t]), w_mask[t]))
            if all(float(m) > 0.5 for m in w_mask[t]):
                fixed_nodes[(t, 2)] = int(w_words[t]) & MASK32

        if include_algebraic:
            # 1. 4-round analytical back-solve A_{R-7..R-4} from boundary_clamp
            a_clamped = boundary.A_words_clamped[0].tolist()
            for t in range(self.rounds - 7, self.rounds - 3):
                val = int(a_clamped[t + 3]) & MASK32
                clauses.extend(_int_to_unit_clauses(self.A_vars[t + 3], val))
                fixed_nodes[(t, 0)] = val

            # 2. Pure algebraic derivation for R >= 18 when W_{1..15} are known constants (W17, E14, A10)
            alg = compute_algebraic_known_words_r18(
                dig_cpu, domain_mode=domain_mode, num_bits=num_bits, msg_bytes_len=msg_bytes_len
            )
            for (t, ch), val in alg.items():
                if (t, ch) not in fixed_nodes and t < self.rounds:
                    if ch == 0:
                        clauses.extend(_int_to_unit_clauses(self.A_vars[t + 3], val))
                    elif ch == 1:
                        clauses.extend(_int_to_unit_clauses(self.E_vars[t + 3], val))
                    elif ch == 2:
                        clauses.extend(_int_to_unit_clauses(self.W_vars[t], val))
                    fixed_nodes[(t, ch)] = val

        return clauses, fixed_nodes

    @staticmethod
    def bits_to_word(vars_32: Sequence[int], model_set: set[int]) -> int:
        val = 0
        for b, v in enumerate(vars_32):
            if v in model_set:
                val |= 1 << b
        return val

    def extract_w0_from_model(self, model: Sequence[int]) -> int:
        model_set = set(int(lit) for lit in model)
        return self.bits_to_word(self.W_vars[0], model_set)


def compute_algebraic_known_words_r18(
    digest: torch.Tensor,
    domain_mode: str = "int_range",
    num_bits: int = 32,
    msg_bytes_len: int = 4,
) -> Dict[Tuple[int, int], int]:
    """Zero-leakage algebraic derivation of {A_{11..14}, W_{17}, E_{14}, A_{10}} from digest only."""
    dig_cpu = digest.detach().cpu().to(torch.int64)
    if dig_cpu.ndim == 2:
        dig_cpu = dig_cpu[0]
    rounds = 18
    boundary = build_clamped_boundary(
        digest=dig_cpu.unsqueeze(0),
        rounds=rounds,
        domain_mode=domain_mode,
        num_bits=num_bits,
        msg_bytes_len=msg_bytes_len,
    )
    out: Dict[Tuple[int, int], int] = {}

    # A_{11..14} from 4-round analytical back-solve
    for t in (11, 12, 13, 14):
        out[(t, 0)] = int(boundary.A_words_clamped[0, t + 3].item()) & MASK32

    w_mask = boundary.W16_mask_bits[0]
    w_words = boundary.W16_clamped_words[0]
    need_w17 = (
        (w_mask[1] > 0.5).all()
        and (w_mask[2] > 0.5).all()
        and (w_mask[10] > 0.5).all()
        and (w_mask[15] > 0.5).all()
    )
    if bool(need_w17):
        w1 = w_words[1]
        w2 = w_words[2]
        w10 = w_words[10]
        w15 = w_words[15]
        w17_t = (sigma1_lower(w15) + w10 + sigma0_lower(w2) + w1) & MASK32
        out[(17, 2)] = int(w17_t.item()) & MASK32

        # Eq W(17): E_{18} = A_{14} + E_{14} + Sigma1(E_{17}) + Ch(E_{17}, E_{16}, E_{15}) + K_{17} + W_{17}
        k_vec = get_k_tensor(18, device="cpu")
        e_cl = boundary.E_words_clamped[0]
        a_cl = boundary.A_words_clamped[0]
        e18 = e_cl[21]
        e17 = e_cl[20]
        e16 = e_cl[19]
        e15 = e_cl[18]
        a14 = a_cl[17]
        e14_t = (
            e18 - a14 - sigma1_upper(e17) - ch32(e17, e16, e15) - k_vec[17] - w17_t
        ) & MASK32
        out[(14, 1)] = int(e14_t.item()) & MASK32

        # Eq E(13): A_{10} = E_{14} - A_{14} + Sigma0(A_{13}) + Maj(A_{13}, A_{12}, A_{11})
        a13 = a_cl[16]
        a12 = a_cl[15]
        a11 = a_cl[14]
        a10_t = (e14_t - a14 + sigma0_upper(a13) + maj32(a13, a12, a11)) & MASK32
        out[(10, 0)] = int(a10_t.item()) & MASK32

    return out
