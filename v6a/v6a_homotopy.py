"""Phase-SHA V6A Layer 1: Exact F_2 Linear Skeleton, Carry-Slack Circuit, Peel-Back & 2D Homotopy.

Implements all approved design revisions cleanly without touching V3/V4/V5 files:
  - Supports arbitrary rounds R >= 4 (calibration R = 4, 5, 6, 8, 10, 12 and attack R = 18, 20).
  - Supports `domain_mode` in {'int_range', 'free_msg', 'fixed_len_bytes'} (`free_msg` + `allow_peel_back=False`
    prevents R <= 16 algebraic peel-back cheating during Phase-0 calibration and R=4/5 verification).
  - Represents every ADD32 gate with explicit 31-bit carry slack variables so ALL F_2 linear constraints
    (Sigma0/Sigma1/sigma0/sigma1 and bitwise a_k ^ b_k ^ c_{k-1} ^ s_k = 0) hold identically for all alpha in [0, 1]
    and remain 100% consistent under full m=256 digest clamping at alpha=0.
  - Exact sparse bitmask GF(2) Gaussian elimination produces exact zero-residual alpha=0 starting states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from src.sha256_ref import (
    MASK32,
    ch32,
    expand_message_schedule,
    get_iv_tensor,
    get_k_tensor,
    maj32,
    sha256_trace,
    sigma0_lower,
    sigma0_upper,
    sigma1_lower,
    sigma1_upper,
    words_to_bits,
    words_to_spins,
)


def _rotr_bit_idx(word_bit_fn, w: int, rot: int, b: int) -> int:
    return word_bit_fn(w, (b + rot) % 32)


def _shr_bit_idx(word_bit_fn, w: int, sh: int, b: int, const0_idx: int) -> Tuple[int, int]:
    """Return (var_bit_idx, is_valid_nonconst)."""
    if b + sh < 32:
        return word_bit_fn(w, b + sh), 1
    return word_bit_fn(const0_idx, 0), 0


@dataclass
class V6ACircuitSpec:
    """Exact variable, F_2 parity-check, and nonlinear gate graph for R-round SHA-256."""

    rounds: int
    n_var_words: int
    n_var_bits: int
    a_start: int
    e_start: int
    w_start: int
    k_start: int
    iv_start: int
    h_start: int
    const0_idx: int
    aux_start: int
    carry_start: int

    # Per-round aux word mappings and sched aux word mappings
    aux_round: List[Dict[str, int]]
    aux_sched: List[Dict[str, int]]

    # Exact F_2 linear parity checks (N_chk, 4) where sum(bits[valid]) % 2 == 0
    chk_vars: torch.Tensor  # (N_chk, 4) int64
    chk_valid: torch.Tensor  # (N_chk, 4) float32 (0/1)
    var_to_chk_idx: torch.Tensor  # (E_lin,) int64
    var_to_chk_slot: torch.Tensor  # (E_lin,) int64
    var_to_chk_var: torch.Tensor  # (E_lin,) int64

    # Exact nonlinear gate tables
    ch_quads: torch.Tensor  # (N_ch, 4) bit indices [e0, e1, e2, out]
    maj_quads: torch.Tensor  # (N_maj, 4) bit indices [a0, a1, a2, out]
    add_word_quads: torch.Tensor  # (N_add, 4) word indices [a_w, b_w, s_w, carry_w]

    # Nonlinear gate bit-occurrence replica map -> variable bit index for Set-A consensus
    nl_rep_to_var_bit: torch.Tensor  # (N_nl_rep,) int64
    var_total_degree: torch.Tensor  # (n_var_bits,) float32 (>= 1.0)


_SPEC_CACHE: Dict[int, V6ACircuitSpec] = {}


def build_v6a_circuit_spec(rounds: int = 18) -> V6ACircuitSpec:
    """Build exact R-round SHA-256 circuit graph with explicit 31-bit carry slack words."""
    r = int(rounds)
    if r < 4 or r > 64:
        raise ValueError(f"V6A supports rounds in [4, 64], got {r}")
    cached = _SPEC_CACHE.get(r)
    if cached is not None:
        return cached

    a_start = 0
    e_start = r + 4
    w_start = e_start + r + 4
    k_start = w_start + r
    iv_start = k_start + r
    h_start = iv_start + 8
    const0_idx = h_start + 8
    cur = const0_idx + 1
    aux_start = cur

    aux_round: List[Dict[str, int]] = []
    for _ in range(r):
        d = {
            name: cur + idx
            for idx, name in enumerate(("S1", "Ch", "S0", "Maj", "u1", "u2", "u3", "T1", "T2"))
        }
        aux_round.append(d)
        cur += 9

    aux_sched: List[Dict[str, int]] = []
    for _ in range(max(0, r - 16)):
        d = {name: cur + idx for idx, name in enumerate(("s1", "s0", "v1", "v2"))}
        aux_sched.append(d)
        cur += 4

    # Enumerate ALL ADD32 triples: (wa, wb, ws)
    add_triples: List[Tuple[int, int, int]] = []
    for t in range(r):
        at = a_start + t + 3
        atm3 = a_start + t
        etm3 = e_start + t
        atp1 = a_start + t + 4
        etp1 = e_start + t + 4
        wt = w_start + t
        kt = k_start + t
        aux = aux_round[t]
        add_triples.append((etm3, aux["S1"], aux["u1"]))
        add_triples.append((aux["u1"], aux["Ch"], aux["u2"]))
        add_triples.append((aux["u2"], kt, aux["u3"]))
        add_triples.append((aux["u3"], wt, aux["T1"]))
        add_triples.append((aux["S0"], aux["Maj"], aux["T2"]))
        add_triples.append((atm3, aux["T1"], etp1))
        add_triples.append((aux["T1"], aux["T2"], atp1))
        del at

    for idx_s, t in enumerate(range(16, r)):
        aux_s = aux_sched[idx_s]
        wm7 = w_start + t - 7
        wm16 = w_start + t - 16
        wt = w_start + t
        add_triples.append((aux_s["s1"], wm7, aux_s["v1"]))
        add_triples.append((aux_s["s0"], wm16, aux_s["v2"]))
        add_triples.append((aux_s["v1"], aux_s["v2"], wt))

    # Feedforward H_i = S_{R,i} + IV_i (mod 2^32)
    sr_words = [
        a_start + r + 3,
        a_start + r + 2,
        a_start + r + 1,
        a_start + r,
        e_start + r + 3,
        e_start + r + 2,
        e_start + r + 1,
        e_start + r,
    ]
    for i in range(8):
        add_triples.append((sr_words[i], iv_start + i, h_start + i))

    carry_start = cur
    add_word_quads_list: List[Tuple[int, int, int, int]] = []
    for wa, wb, ws in add_triples:
        wc = cur
        cur += 1
        add_word_quads_list.append((wa, wb, ws, wc))

    n_var_words = cur
    n_var_bits = n_var_words * 32

    def vb(w: int, b: int) -> int:
        return int(w) * 32 + int(b)

    chk_vars_list: List[Tuple[int, int, int, int]] = []
    chk_valid_list: List[Tuple[float, float, float, float]] = []
    ch_quads_list: List[Tuple[int, int, int, int]] = []
    maj_quads_list: List[Tuple[int, int, int, int]] = []

    for t in range(r):
        at = a_start + t + 3
        atm1 = a_start + t + 2
        atm2 = a_start + t + 1
        et = e_start + t + 3
        etm1 = e_start + t + 2
        etm2 = e_start + t + 1
        aux = aux_round[t]
        for b in range(32):
            chk_vars_list.append(
                (
                    _rotr_bit_idx(vb, et, 6, b),
                    _rotr_bit_idx(vb, et, 11, b),
                    _rotr_bit_idx(vb, et, 25, b),
                    vb(aux["S1"], b),
                )
            )
            chk_valid_list.append((1.0, 1.0, 1.0, 1.0))

            chk_vars_list.append(
                (
                    _rotr_bit_idx(vb, at, 2, b),
                    _rotr_bit_idx(vb, at, 13, b),
                    _rotr_bit_idx(vb, at, 22, b),
                    vb(aux["S0"], b),
                )
            )
            chk_valid_list.append((1.0, 1.0, 1.0, 1.0))

            ch_quads_list.append((vb(et, b), vb(etm1, b), vb(etm2, b), vb(aux["Ch"], b)))
            maj_quads_list.append((vb(at, b), vb(atm1, b), vb(atm2, b), vb(aux["Maj"], b)))

    for idx_s, t in enumerate(range(16, r)):
        aux_s = aux_sched[idx_s]
        wm2 = w_start + t - 2
        wm15 = w_start + t - 15
        for b in range(32):
            v_sh10, ok10 = _shr_bit_idx(vb, wm2, 10, b, const0_idx)
            chk_vars_list.append(
                (
                    _rotr_bit_idx(vb, wm2, 17, b),
                    _rotr_bit_idx(vb, wm2, 19, b),
                    v_sh10,
                    vb(aux_s["s1"], b),
                )
            )
            chk_valid_list.append((1.0, 1.0, float(ok10), 1.0))

            v_sh3, ok3 = _shr_bit_idx(vb, wm15, 3, b, const0_idx)
            chk_vars_list.append(
                (
                    _rotr_bit_idx(vb, wm15, 7, b),
                    _rotr_bit_idx(vb, wm15, 18, b),
                    v_sh3,
                    vb(aux_s["s0"], b),
                )
            )
            chk_valid_list.append((1.0, 1.0, float(ok3), 1.0))

    # Bitwise linear parity equations for ALL ADD32 gates (exact sum mod 2 with carry-in)
    c0_bit = vb(const0_idx, 0)
    for wa, wb, ws, wc in add_word_quads_list:
        chk_vars_list.append((vb(wa, 0), vb(wb, 0), c0_bit, vb(ws, 0)))
        chk_valid_list.append((1.0, 1.0, 0.0, 1.0))
        for b in range(1, 32):
            chk_vars_list.append((vb(wa, b), vb(wb, b), vb(wc, b - 1), vb(ws, b)))
            chk_valid_list.append((1.0, 1.0, 1.0, 1.0))

    chk_vars = torch.tensor(chk_vars_list, dtype=torch.int64)
    chk_valid = torch.tensor(chk_valid_list, dtype=torch.float32)

    valid_mask = chk_valid > 0.5
    chk_idx_grid = torch.arange(chk_vars.shape[0], dtype=torch.int64).unsqueeze(1).expand_as(chk_vars)
    slot_grid = torch.arange(4, dtype=torch.int64).unsqueeze(0).expand_as(chk_vars)
    var_to_chk_idx = chk_idx_grid[valid_mask].contiguous()
    var_to_chk_slot = slot_grid[valid_mask].contiguous()
    var_to_chk_var = chk_vars[valid_mask].contiguous()

    ch_quads = torch.tensor(ch_quads_list, dtype=torch.int64)
    maj_quads = torch.tensor(maj_quads_list, dtype=torch.int64)
    add_word_quads = torch.tensor(add_word_quads_list, dtype=torch.int64)

    # Nonlinear replica variable IDs layout: [Ch (N_ch, 4) | Maj (N_maj, 4) | ADD32 (N_add, 4, 32)]
    nl_parts: List[torch.Tensor] = [ch_quads.reshape(-1), maj_quads.reshape(-1)]
    bit_offsets = torch.arange(32, dtype=torch.int64)
    add_bit_ids = (add_word_quads.unsqueeze(-1) * 32 + bit_offsets.view(1, 1, 32)).reshape(-1)
    nl_parts.append(add_bit_ids)
    nl_rep_to_var_bit = torch.cat(nl_parts, dim=0).contiguous()

    # Consensus Set-A has 1 global linear copy per variable bit + nl_rep occurrences
    nl_deg = torch.bincount(nl_rep_to_var_bit, minlength=n_var_bits).to(torch.float32)
    var_total_degree = (1.0 + nl_deg).clamp_min(1.0)

    spec = V6ACircuitSpec(
        rounds=r,
        n_var_words=n_var_words,
        n_var_bits=n_var_bits,
        a_start=a_start,
        e_start=e_start,
        w_start=w_start,
        k_start=k_start,
        iv_start=iv_start,
        h_start=h_start,
        const0_idx=const0_idx,
        aux_start=aux_start,
        carry_start=carry_start,
        aux_round=aux_round,
        aux_sched=aux_sched,
        chk_vars=chk_vars,
        chk_valid=chk_valid,
        var_to_chk_idx=var_to_chk_idx,
        var_to_chk_slot=var_to_chk_slot,
        var_to_chk_var=var_to_chk_var,
        ch_quads=ch_quads,
        maj_quads=maj_quads,
        add_word_quads=add_word_quads,
        nl_rep_to_var_bit=nl_rep_to_var_bit,
        var_total_degree=var_total_degree,
    )
    _SPEC_CACHE[r] = spec
    return spec


@torch.no_grad()
def _compute_add32_carry_words(a_words: torch.Tensor, b_words: torch.Tensor) -> torch.Tensor:
    """Compute exact 31-bit carry-out word `c` where bit `k` is carry into bit `k+1` (bit 31 = 0)."""
    # Exact bitwise identity: a + b = (a ^ b) + 2 * carries => c_into[0..31] = ((a + b) ^ a ^ b)
    # Therefore carry_out[k] = c_into[k + 1] for k in 0..30 => shift right by 1!
    sum_full = a_words.to(torch.int64) + b_words.to(torch.int64)
    c_into = (sum_full ^ a_words ^ b_words) & ((1 << 32) - 1)
    return (c_into >> 1) & 0x7FFFFFFF


@torch.no_grad()
def embed_ground_truth_vars(
    w16: torch.Tensor,
    rounds: int = 18,
    w_full_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Embed exact discrete ground-truth trace (spins in {-1, +1}) across ALL `n_var_bits`."""
    spec = build_v6a_circuit_spec(rounds)
    bsz = w16.shape[0]
    dev = w16.device
    tr = sha256_trace(w16, rounds=rounds)
    w_full = tr.W_full if w_full_override is None else w_full_override.to(device=dev, dtype=torch.int64)
    k_vec = get_k_tensor(rounds, device=dev)
    iv = get_iv_tensor(device=dev)

    words = torch.zeros((bsz, spec.n_var_words), dtype=torch.int64, device=dev)
    words[:, spec.a_start : spec.a_start + rounds + 4] = tr.A_full
    words[:, spec.e_start : spec.e_start + rounds + 4] = tr.E_full
    words[:, spec.w_start : spec.w_start + rounds] = w_full
    words[:, spec.k_start : spec.k_start + rounds] = k_vec.unsqueeze(0)
    words[:, spec.iv_start : spec.iv_start + 8] = iv.unsqueeze(0)
    words[:, spec.h_start : spec.h_start + 8] = tr.digest
    words[:, spec.const0_idx] = 0

    for t in range(rounds):
        at = tr.A_full[:, t + 3]
        atm1 = tr.A_full[:, t + 2]
        atm2 = tr.A_full[:, t + 1]
        et = tr.E_full[:, t + 3]
        etm1 = tr.E_full[:, t + 2]
        etm2 = tr.E_full[:, t + 1]
        etm3 = tr.E_full[:, t]
        s1 = sigma1_upper(et)
        ch = ch32(et, etm1, etm2)
        s0 = sigma0_upper(at)
        maj = maj32(at, atm1, atm2)
        u1 = (etm3 + s1) & MASK32
        u2 = (u1 + ch) & MASK32
        u3 = (u2 + k_vec[t]) & MASK32
        t1 = (u3 + w_full[:, t]) & MASK32
        t2 = (s0 + maj) & MASK32
        aux = spec.aux_round[t]
        for nm, val in (
            ("S1", s1), ("Ch", ch), ("S0", s0), ("Maj", maj),
            ("u1", u1), ("u2", u2), ("u3", u3), ("T1", t1), ("T2", t2),
        ):
            words[:, aux[nm]] = val

    for idx_s, t in enumerate(range(16, rounds)):
        s1_l = sigma1_lower(w_full[:, t - 2])
        s0_l = sigma0_lower(w_full[:, t - 15])
        v1 = (s1_l + w_full[:, t - 7]) & MASK32
        v2 = (s0_l + w_full[:, t - 16]) & MASK32
        aux_s = spec.aux_sched[idx_s]
        for nm, val in (("s1", s1_l), ("s0", s0_l), ("v1", v1), ("v2", v2)):
            words[:, aux_s[nm]] = val

    # Compute exact 31-bit carries for all ADD32 gates
    quads = spec.add_word_quads.to(dev)
    a_w = words[:, quads[:, 0]]
    b_w = words[:, quads[:, 1]]
    words[:, quads[:, 3]] = _compute_add32_carry_words(a_w, b_w)

    return words_to_spins(words, dtype=torch.float32).reshape(bsz, spec.n_var_bits)


@dataclass
class V6ABoundary:
    """Clamped boundary template for V6A supporting R >= 4, domain modes, peel-back & aperture m."""

    rounds: int
    domain_mode: str
    mask_m: int
    allow_peel_back: bool
    var_mask: torch.Tensor  # (B, n_var_bits) float32 in {0, 1}
    var_vals: torch.Tensor  # (B, n_var_bits) float32 in {-1, 0, +1}
    A_full_clamped: torch.Tensor  # (B, R+4, 32)
    A_full_mask: torch.Tensor  # (B, R+4, 32)
    E_full_clamped: torch.Tensor  # (B, R+4, 32)
    E_full_mask: torch.Tensor  # (B, R+4, 32)
    W_full_clamped: torch.Tensor  # (B, R, 32)
    W_full_mask: torch.Tensor  # (B, R, 32)
    target_digests: torch.Tensor  # (B, 8) int64
    W16_clamped_words: torch.Tensor  # (B, 16) int64
    W16_mask_bits: torch.Tensor  # (B, 16, 32) float32
    digest_target_bits: Optional[torch.Tensor] = None  # (B, 8, 32) float32
    digest_target_mask: Optional[torch.Tensor] = None  # (B, 8, 32) float32
    is_full_digest: bool = True

    @property
    def batch_size(self) -> int:
        return int(self.var_mask.shape[0])

    @property
    def device(self) -> torch.device:
        return self.var_mask.device

    @property
    def clamp_mask_nodes(self) -> torch.Tensor:
        r = self.rounds
        return torch.stack(
            [
                self.A_full_mask[:, 3 : 3 + r],
                self.E_full_mask[:, 3 : 3 + r],
                self.W_full_mask[:, :r],
            ],
            dim=2,
        )

    @property
    def x_clamped_nodes(self) -> torch.Tensor:
        r = self.rounds
        return torch.stack(
            [
                self.A_full_clamped[:, 3 : 3 + r],
                self.E_full_clamped[:, 3 : 3 + r],
                self.W_full_clamped[:, :r],
            ],
            dim=2,
        )

    def repeat_for_candidates(self, k: int) -> "V6ABoundary":
        def rep(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if t is None else t.repeat_interleave(k, dim=0)

        return V6ABoundary(
            rounds=self.rounds,
            domain_mode=self.domain_mode,
            mask_m=self.mask_m,
            allow_peel_back=self.allow_peel_back,
            var_mask=rep(self.var_mask),  # type: ignore[arg-type]
            var_vals=rep(self.var_vals),  # type: ignore[arg-type]
            A_full_clamped=rep(self.A_full_clamped),  # type: ignore[arg-type]
            A_full_mask=rep(self.A_full_mask),  # type: ignore[arg-type]
            E_full_clamped=rep(self.E_full_clamped),  # type: ignore[arg-type]
            E_full_mask=rep(self.E_full_mask),  # type: ignore[arg-type]
            W_full_clamped=rep(self.W_full_clamped),  # type: ignore[arg-type]
            W_full_mask=rep(self.W_full_mask),  # type: ignore[arg-type]
            target_digests=rep(self.target_digests),  # type: ignore[arg-type]
            W16_clamped_words=rep(self.W16_clamped_words),  # type: ignore[arg-type]
            W16_mask_bits=rep(self.W16_mask_bits),  # type: ignore[arg-type]
            digest_target_bits=rep(self.digest_target_bits),
            digest_target_mask=rep(self.digest_target_mask),
            is_full_digest=self.is_full_digest,
        )


def _aperture_lsb_bit_mask(mask_m: int, device: torch.device) -> torch.Tensor:
    """Return (8, 32) binary LSB-major aperture mask locking `mask_m` bits across the 8 digest words."""
    m = max(1, min(256, int(mask_m)))
    out = torch.zeros((8, 32), dtype=torch.float32, device=device)
    base, rem = divmod(m, 8)
    for i in range(8):
        k = base + (1 if i < rem else 0)
        if k > 0:
            out[i, :k] = 1.0
    return out


def _domain_w16_template(
    domain_mode: str, num_bits: int, msg_bytes_len: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    w16_words = torch.zeros(16, dtype=torch.int64, device=device)
    w16_mask_bits = torch.zeros((16, 32), dtype=torch.float32, device=device)
    if domain_mode == "free_msg":
        return w16_words, w16_mask_bits
    if domain_mode == "int_range":
        if num_bits < 32:
            w16_mask_bits[0, num_bits:] = 1.0
        w16_words[1] = 0x80000000
        w16_words[15] = 32
        w16_mask_bits[1:16, :] = 1.0
        return w16_words, w16_mask_bits
    if domain_mode == "fixed_len_bytes":
        from src.boundary_clamp import build_w16_domain_clamp

        return build_w16_domain_clamp("fixed_len_bytes", num_bits=num_bits, msg_bytes_len=msg_bytes_len, device=device)
    raise ValueError(f"Unsupported domain_mode '{domain_mode}'")


@torch.no_grad()
def build_v6a_boundary(
    digest: torch.Tensor,
    rounds: int = 18,
    domain_mode: str = "int_range",
    num_bits: int = 32,
    msg_bytes_len: int = 4,
    mask_m: int = 256,
    allow_peel_back: bool = True,
    digest_target_mask: Optional[torch.Tensor] = None,
) -> V6ABoundary:
    """Build exact V6A boundary with algebraic peel-back (when enabled) and aperture clamping."""
    r = int(rounds)
    spec = build_v6a_circuit_spec(r)
    if digest.ndim == 1:
        digest = digest.unsqueeze(0)
    bsz = digest.shape[0]
    dev = digest.device
    dig = digest.to(torch.int64) & MASK32

    iv = get_iv_tensor(dev)
    k_vec = get_k_tensor(r, device=dev)

    word_vals = torch.zeros((bsz, spec.n_var_words), dtype=torch.int64, device=dev)
    word_mask = torch.zeros((bsz, spec.n_var_words, 32), dtype=torch.float32, device=dev)
    word_fully_known = torch.zeros((spec.n_var_words,), dtype=torch.bool, device=dev)

    # 1. Fixed SHA constants: IV, K, CONST_ZERO
    word_vals[:, spec.iv_start : spec.iv_start + 8] = iv.unsqueeze(0)
    word_mask[:, spec.iv_start : spec.iv_start + 8, :] = 1.0
    word_fully_known[spec.iv_start : spec.iv_start + 8] = True

    word_vals[:, spec.k_start : spec.k_start + r] = k_vec.unsqueeze(0)
    word_mask[:, spec.k_start : spec.k_start + r, :] = 1.0
    word_fully_known[spec.k_start : spec.k_start + r] = True

    word_vals[:, spec.const0_idx] = 0
    word_mask[:, spec.const0_idx, :] = 1.0
    word_fully_known[spec.const0_idx] = True

    # Carry bit 31 is structurally 0 (no carry into bit 32 stored in variable bit 31)
    word_mask[:, spec.carry_start : spec.n_var_words, 31] = 1.0

    # 2. Left IV states A_{-3..0}, E_{-3..0}
    for i, idx in enumerate((3, 2, 1, 0)):
        word_vals[:, spec.a_start + i] = iv[idx]
        word_vals[:, spec.e_start + i] = iv[4 + idx]
    word_mask[:, spec.a_start : spec.a_start + 4, :] = 1.0
    word_mask[:, spec.e_start : spec.e_start + 4, :] = 1.0
    word_fully_known[spec.a_start : spec.a_start + 4] = True
    word_fully_known[spec.e_start : spec.e_start + 4] = True

    # 3. Right digest / S_R clamping (honoring aperture m OR explicit V4E partial digest mask)
    s_r_words = (dig - iv.unsqueeze(0)) & MASK32
    sr_slot_indices = [
        spec.a_start + r + 3,
        spec.a_start + r + 2,
        spec.a_start + r + 1,
        spec.a_start + r,
        spec.e_start + r + 3,
        spec.e_start + r + 2,
        spec.e_start + r + 1,
        spec.e_start + r,
    ]

    if digest_target_mask is None:
        is_full_dig = int(mask_m) >= 256
        h_bit_mask = _aperture_lsb_bit_mask(mask_m, dev)  # (8, 32)
        h_bits = words_to_bits(dig).to(torch.float32)  # (B, 8, 32)
        for i in range(8):
            w_h = spec.h_start + i
            w_sr = sr_slot_indices[i]
            word_vals[:, w_h] = dig[:, i]
            word_mask[:, w_h, :] = h_bit_mask[i].unsqueeze(0)
            # Crucially: LSB-contiguous mask on H_i + full IV_i uniquely fixes LSBs of S_{R,i} = H_i - IV_i
            word_vals[:, w_sr] = s_r_words[:, i]
            word_mask[:, w_sr, :] = h_bit_mask[i].unsqueeze(0)
            if bool((h_bit_mask[i] > 0.5).all().item()):
                word_fully_known[w_h] = True
                word_fully_known[w_sr] = True
        dig_t_mask = h_bit_mask.unsqueeze(0).expand(bsz, 8, 32).clone()
        dig_t_bits = h_bits
    else:
        is_full_dig = False
        dig_t_mask = digest_target_mask.to(device=dev, dtype=torch.float32)
        dig_t_bits = words_to_bits(dig).to(torch.float32)
        for i in range(8):
            w_h = spec.h_start + i
            word_vals[:, w_h] = dig[:, i]
            word_mask[:, w_h, :] = dig_t_mask[:, i, :]

    # 4. Domain W clamping
    w16_1d, w16_m2d = _domain_w16_template(domain_mode, num_bits, msg_bytes_len, dev)
    w16_words = w16_1d.unsqueeze(0).expand(bsz, 16).clone()
    w16_mask_bits = w16_m2d.unsqueeze(0).expand(bsz, 16, 32).clone()
    k_copy = min(r, 16)
    word_vals[:, spec.w_start : spec.w_start + k_copy] = w16_words[:, :k_copy]
    word_mask[:, spec.w_start : spec.w_start + k_copy, :] = w16_mask_bits[:, :k_copy, :]
    for t in range(k_copy):
        if bool((w16_m2d[t] > 0.5).all().item()):
            word_fully_known[spec.w_start + t] = True

    # 5. Deterministic algebraic peel-back ONLY when explicitly enabled (R >= 8 & full words known)
    if allow_peel_back and r >= 8:
        changed = True
        while changed:
            changed = False
            # 5a. Message schedule expansion where all 4 inputs are fully known
            for idx_s, t in enumerate(range(16, r)):
                wt = spec.w_start + t
                wm2, wm7, wm15, wm16 = (
                    spec.w_start + t - 2,
                    spec.w_start + t - 7,
                    spec.w_start + t - 15,
                    spec.w_start + t - 16,
                )
                if not word_fully_known[wt] and (
                    word_fully_known[wm2]
                    and word_fully_known[wm7]
                    and word_fully_known[wm15]
                    and word_fully_known[wm16]
                ):
                    val = (
                        sigma1_lower(word_vals[:, wm2])
                        + word_vals[:, wm7]
                        + sigma0_lower(word_vals[:, wm15])
                        + word_vals[:, wm16]
                    ) & MASK32
                    word_vals[:, wt] = val
                    word_mask[:, wt, :] = 1.0
                    word_fully_known[wt] = True
                    changed = True

            # 5b. Backward A-peel: A_{t-3} = E_{t+1} - A_{t+1} + Sigma0(A_t) + Maj(A_t, A_{t-1}, A_{t-2})
            for t in range(r - 1, -1, -1):
                a_tm3 = spec.a_start + t
                if word_fully_known[a_tm3]:
                    continue
                if (
                    word_fully_known[spec.e_start + t + 4]
                    and word_fully_known[spec.a_start + t + 4]
                    and word_fully_known[spec.a_start + t + 3]
                    and word_fully_known[spec.a_start + t + 2]
                    and word_fully_known[spec.a_start + t + 1]
                ):
                    val = (
                        word_vals[:, spec.e_start + t + 4]
                        - word_vals[:, spec.a_start + t + 4]
                        + sigma0_upper(word_vals[:, spec.a_start + t + 3])
                        + maj32(
                            word_vals[:, spec.a_start + t + 3],
                            word_vals[:, spec.a_start + t + 2],
                            word_vals[:, spec.a_start + t + 1],
                        )
                    ) & MASK32
                    word_vals[:, a_tm3] = val
                    word_mask[:, a_tm3, :] = 1.0
                    word_fully_known[a_tm3] = True
                    changed = True

            # 5c. Backward E-peel when W_t and A_{t-3}, E_{t-2..t+1} are known (peels r17 on R=18)
            for t in range(r - 1, -1, -1):
                e_tm3 = spec.e_start + t
                if word_fully_known[e_tm3]:
                    continue
                if (
                    word_fully_known[spec.e_start + t + 4]
                    and word_fully_known[spec.a_start + t]
                    and word_fully_known[spec.e_start + t + 3]
                    and word_fully_known[spec.e_start + t + 2]
                    and word_fully_known[spec.e_start + t + 1]
                    and word_fully_known[spec.w_start + t]
                ):
                    val = (
                        word_vals[:, spec.e_start + t + 4]
                        - word_vals[:, spec.a_start + t]
                        - sigma1_upper(word_vals[:, spec.e_start + t + 3])
                        - ch32(
                            word_vals[:, spec.e_start + t + 3],
                            word_vals[:, spec.e_start + t + 2],
                            word_vals[:, spec.e_start + t + 1],
                        )
                        - k_vec[t]
                        - word_vals[:, spec.w_start + t]
                    ) & MASK32
                    word_vals[:, e_tm3] = val
                    word_mask[:, e_tm3, :] = 1.0
                    word_fully_known[e_tm3] = True
                    changed = True

        # 5d. Propagate fully known deterministic aux & carry words in peeled zones
        for t in range(r):
            aux = spec.aux_round[t]
            et, etm1, etm2, etm3 = spec.e_start + t + 3, spec.e_start + t + 2, spec.e_start + t + 1, spec.e_start + t
            at, atm1, atm2 = spec.a_start + t + 3, spec.a_start + t + 2, spec.a_start + t + 1
            if word_fully_known[et]:
                word_vals[:, aux["S1"]] = sigma1_upper(word_vals[:, et])
                word_mask[:, aux["S1"], :] = 1.0
                word_fully_known[aux["S1"]] = True
            if word_fully_known[et] and word_fully_known[etm1] and word_fully_known[etm2]:
                word_vals[:, aux["Ch"]] = ch32(word_vals[:, et], word_vals[:, etm1], word_vals[:, etm2])
                word_mask[:, aux["Ch"], :] = 1.0
                word_fully_known[aux["Ch"]] = True
            if word_fully_known[at]:
                word_vals[:, aux["S0"]] = sigma0_upper(word_vals[:, at])
                word_mask[:, aux["S0"], :] = 1.0
                word_fully_known[aux["S0"]] = True
            if word_fully_known[at] and word_fully_known[atm1] and word_fully_known[atm2]:
                word_vals[:, aux["Maj"]] = maj32(word_vals[:, at], word_vals[:, atm1], word_vals[:, atm2])
                word_mask[:, aux["Maj"], :] = 1.0
                word_fully_known[aux["Maj"]] = True
            if word_fully_known[etm3] and word_fully_known[aux["S1"]]:
                word_vals[:, aux["u1"]] = (word_vals[:, etm3] + word_vals[:, aux["S1"]]) & MASK32
                word_mask[:, aux["u1"], :] = 1.0
                word_fully_known[aux["u1"]] = True
            if word_fully_known[aux["u1"]] and word_fully_known[aux["Ch"]]:
                word_vals[:, aux["u2"]] = (word_vals[:, aux["u1"]] + word_vals[:, aux["Ch"]]) & MASK32
                word_mask[:, aux["u2"], :] = 1.0
                word_fully_known[aux["u2"]] = True
            if word_fully_known[aux["u2"]]:
                word_vals[:, aux["u3"]] = (word_vals[:, aux["u2"]] + k_vec[t]) & MASK32
                word_mask[:, aux["u3"], :] = 1.0
                word_fully_known[aux["u3"]] = True
            if word_fully_known[aux["u3"]] and word_fully_known[spec.w_start + t]:
                word_vals[:, aux["T1"]] = (word_vals[:, aux["u3"]] + word_vals[:, spec.w_start + t]) & MASK32
                word_mask[:, aux["T1"], :] = 1.0
                word_fully_known[aux["T1"]] = True
            if word_fully_known[aux["S0"]] and word_fully_known[aux["Maj"]]:
                word_vals[:, aux["T2"]] = (word_vals[:, aux["S0"]] + word_vals[:, aux["Maj"]]) & MASK32
                word_mask[:, aux["T2"], :] = 1.0
                word_fully_known[aux["T2"]] = True

        for idx_s, t in enumerate(range(16, r)):
            aux_s = spec.aux_sched[idx_s]
            wm2, wm7, wm15, wm16 = spec.w_start + t - 2, spec.w_start + t - 7, spec.w_start + t - 15, spec.w_start + t - 16
            if word_fully_known[wm2]:
                word_vals[:, aux_s["s1"]] = sigma1_lower(word_vals[:, wm2])
                word_mask[:, aux_s["s1"], :] = 1.0
                word_fully_known[aux_s["s1"]] = True
            if word_fully_known[wm15]:
                word_vals[:, aux_s["s0"]] = sigma0_lower(word_vals[:, wm15])
                word_mask[:, aux_s["s0"], :] = 1.0
                word_fully_known[aux_s["s0"]] = True
            if word_fully_known[aux_s["s1"]] and word_fully_known[wm7]:
                word_vals[:, aux_s["v1"]] = (word_vals[:, aux_s["s1"]] + word_vals[:, wm7]) & MASK32
                word_mask[:, aux_s["v1"], :] = 1.0
                word_fully_known[aux_s["v1"]] = True
            if word_fully_known[aux_s["s0"]] and word_fully_known[wm16]:
                word_vals[:, aux_s["v2"]] = (word_vals[:, aux_s["s0"]] + word_vals[:, wm16]) & MASK32
                word_mask[:, aux_s["v2"], :] = 1.0
                word_fully_known[aux_s["v2"]] = True

        for g_idx in range(spec.add_word_quads.shape[0]):
            wa, wb, _, wc = [int(x) for x in spec.add_word_quads[g_idx].tolist()]
            if word_fully_known[wa] and word_fully_known[wb]:
                word_vals[:, wc] = _compute_add32_carry_words(word_vals[:, wa], word_vals[:, wb])
                word_mask[:, wc, :] = 1.0
                word_fully_known[wc] = True

    word_spins = words_to_spins(word_vals, dtype=torch.float32) * word_mask
    var_mask = word_mask.reshape(bsz, spec.n_var_bits)
    var_vals = word_spins.reshape(bsz, spec.n_var_bits)

    a_clamped = word_spins[:, spec.a_start : spec.a_start + r + 4]
    a_m = word_mask[:, spec.a_start : spec.a_start + r + 4]
    e_clamped = word_spins[:, spec.e_start : spec.e_start + r + 4]
    e_m = word_mask[:, spec.e_start : spec.e_start + r + 4]
    w_clamped = word_spins[:, spec.w_start : spec.w_start + r]
    w_m = word_mask[:, spec.w_start : spec.w_start + r]

    return V6ABoundary(
        rounds=r,
        domain_mode=domain_mode,
        mask_m=int(mask_m),
        allow_peel_back=bool(allow_peel_back),
        var_mask=var_mask,
        var_vals=var_vals,
        A_full_clamped=a_clamped,
        A_full_mask=a_m,
        E_full_clamped=e_clamped,
        E_full_mask=e_m,
        W_full_clamped=w_clamped,
        W_full_mask=w_m,
        target_digests=dig,
        W16_clamped_words=w16_words,
        W16_mask_bits=w16_mask_bits,
        digest_target_bits=dig_t_bits,
        digest_target_mask=dig_t_mask,
        is_full_digest=is_full_dig,
    )


@torch.no_grad()
def solve_alpha0_gf2_linear_skeleton(
    spec: V6ACircuitSpec,
    var_clamp_mask: torch.Tensor,
    var_clamp_vals: torch.Tensor,
    num_samples: int = 1,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, int, int]:
    """Solve H_lin u = 0 (mod 2) subject to clamped bits using sparse unit-prop + exact Python-int RREF.

    Guaranteed 100% consistent even under full 256-bit digest clamping due to explicit carry & nonlinear slacks.
    Returns (spins (num_samples, n_var_bits) in {-1, +1}, lin_violations, free_nullity).
    """
    dev = var_clamp_mask.device
    m_cpu = (var_clamp_mask.detach().to("cpu") > 0.5).tolist()
    b_cpu = (var_clamp_vals.detach().to("cpu") < 0.0).tolist()  # +1 -> 0, -1 -> 1

    # Start fixed array from clamped variables
    fixed: Dict[int, int] = {i: (1 if b_cpu[i] else 0) for i, flag in enumerate(m_cpu) if flag}

    chk_v = spec.chk_vars.tolist()
    chk_ok = spec.chk_valid.tolist()

    # Build lightweight row list: ([var_ids], rhs_const)
    eqs: List[Tuple[List[int], int]] = []
    var_to_eqs: Dict[int, List[int]] = {}
    for row_id in range(len(chk_v)):
        vs = [chk_v[row_id][s] for s in range(4) if chk_ok[row_id][s] > 0.5]
        eqs.append((vs, 0))
        for v in vs:
            var_to_eqs.setdefault(v, []).append(row_id)

    # 1. Iterative exact unit-propagation (eliminates thousands of variables in milliseconds)
    queue: List[int] = list(fixed.keys())
    in_queue = set(queue)
    while queue:
        v_known = queue.pop()
        bit_val = fixed[v_known]
        for eq_id in var_to_eqs.get(v_known, ()):
            vs, rhs = eqs[eq_id]
            new_vs = []
            cnt_removed = 0
            for x in vs:
                if x == v_known:
                    cnt_removed += 1
                elif x in fixed:
                    rhs ^= fixed[x]
                else:
                    new_vs.append(x)
            if cnt_removed & 1:
                rhs ^= bit_val
            # Deduplicate pairs modulo 2
            if len(new_vs) > 1:
                counts: Dict[int, int] = {}
                for x in new_vs:
                    counts[x] = counts.get(x, 0) ^ 1
                new_vs = [x for x, c in counts.items() if c]
            eqs[eq_id] = (new_vs, rhs)
            if len(new_vs) == 1:
                u = new_vs[0]
                if u in fixed:
                    if fixed[u] != rhs:
                        raise RuntimeError(f"GF(2) unit-propagation conflict at var {u}!")
                else:
                    fixed[u] = rhs
                    if u not in in_queue:
                        queue.append(u)
                        in_queue.add(u)
            elif len(new_vs) == 0 and rhs != 0:
                raise RuntimeError("GF(2) inconsistency after unit propagation!")

    # 2. Compact remaining free variables for bitmask Gaussian elimination
    remaining_vars_set = set()
    rem_eqs: List[Tuple[List[int], int]] = []
    for vs, rhs in eqs:
        if not vs:
            continue
        rem_eqs.append((vs, rhs))
        for v in vs:
            remaining_vars_set.add(v)

    unconstrained_vars = [i for i in range(spec.n_var_bits) if i not in fixed and i not in remaining_vars_set]
    rem_vars = sorted(remaining_vars_set)
    n_rem = len(rem_vars)
    var_to_loc = {v: idx for idx, v in enumerate(rem_vars)}
    rhs_shift = n_rem
    rhs_bit = 1 << rhs_shift

    pivots: Dict[int, int] = {}
    pivot_cols_desc: List[int] = []

    for vs, rhs in rem_eqs:
        row_mask = rhs_bit if rhs else 0
        for v in vs:
            row_mask ^= 1 << var_to_loc[v]
        while True:
            vars_only = row_mask & (rhs_bit - 1)
            if vars_only == 0:
                if (row_mask >> rhs_shift) & 1:
                    raise RuntimeError("Linear skeleton system H_lin u = b_lin is inconsistent!")
                break
            piv = (vars_only & -vars_only).bit_length() - 1
            if piv in pivots:
                row_mask ^= pivots[piv]
            else:
                pivots[piv] = row_mask
                pivot_cols_desc.append(piv)
                break

    pivot_set = set(pivots.keys())
    free_local = [idx for idx in range(n_rem) if idx not in pivot_set]
    # Back-substitute in ascending pivot order (note: pivots chosen by lowest bit, so higher indices are later!)
    pivots_ordered = sorted(pivots.keys(), reverse=True)
    nullity = len(free_local) + len(unconstrained_vars)

    # Sample `num_samples` distinct exact GF(2) solutions cleanly
    k_samp = max(1, int(num_samples))
    rand_free = torch.randint(0, 2, (k_samp, len(free_local)), dtype=torch.int64, generator=generator, device="cpu").tolist()
    rand_unc = torch.randint(0, 2, (k_samp, len(unconstrained_vars)), dtype=torch.int64, generator=generator, device="cpu").tolist()

    sol_bits_rows: List[List[int]] = []
    base_vec = [fixed.get(i, 0) for i in range(spec.n_var_bits)]
    for s_idx in range(k_samp):
        vec = list(base_vec)
        loc_vals = [0] * n_rem
        loc_mask = 0
        for j, fl in enumerate(free_local):
            b_val = int(rand_free[s_idx][j])
            loc_vals[fl] = b_val
            if b_val:
                loc_mask |= 1 << fl
        for pcol in pivots_ordered:
            prow = pivots[pcol]
            val = (prow >> rhs_shift) & 1
            # XOR parity of upper bits already assigned in loc_mask
            upper = (prow & (rhs_bit - 1)) & loc_mask
            val ^= upper.bit_count() & 1
            loc_vals[pcol] = val
            if val:
                loc_mask |= 1 << pcol
        for loc_i, v_glob in enumerate(rem_vars):
            vec[v_glob] = loc_vals[loc_i]
        for j, v_unc in enumerate(unconstrained_vars):
            vec[v_unc] = int(rand_unc[s_idx][j])
        sol_bits_rows.append(vec)

    bits_t = torch.tensor(sol_bits_rows, dtype=torch.float32, device=dev)  # (K, n_var_bits)
    spins_t = 1.0 - 2.0 * bits_t

    # Exact verification across ALL parity checks
    cv = spec.chk_vars.to(dev)
    cm = spec.chk_valid.to(dev)
    gathered_bits = bits_t[:, cv]  # (K, N_chk, 4)
    sums_mod2 = (gathered_bits.to(torch.int64) * cm.to(torch.int64).unsqueeze(0)).sum(dim=-1) & 1
    violations = int((sums_mod2 != 0).sum().item())
    return spins_t, violations, nullity


class HomotopySchedule2D:
    """Adaptive 2D interleaved continuous homotopy controller over (alpha in [0,1], mask_m in [64..256])."""

    def __init__(
        self,
        start_alpha: float = 0.0,
        start_mask_m: int = 64,
        target_mask_m: int = 256,
        mask_levels: Tuple[int, ...] = (64, 96, 128, 160, 192, 224, 256),
        step_alpha: float = 0.05,
        min_step_alpha: float = 0.01,
        step_m: int = 32,
        min_step_m: int = 8,
        eps_advance: float = 0.018,
        spike_factor: float = 1.20,
        spike_abs_margin: float = 0.003,
        max_ticks_per_alpha: int = 12,
        eps_gap: float = 0.14,
        baseline_window: int = 3,
        interleaved_2d: bool = True,
    ) -> None:
        self.alpha = max(0.0, min(1.0, float(start_alpha)))
        self.mask_m = max(1, min(256, int(start_mask_m)))
        self.target_mask_m = max(self.mask_m, min(256, int(target_mask_m)))
        levels = sorted({int(x) for x in mask_levels if self.mask_m <= int(x) <= self.target_mask_m} | {self.mask_m, self.target_mask_m})
        self.mask_levels: Tuple[int, ...] = tuple(levels)
        self.step_alpha = float(step_alpha)
        self.base_step_alpha = float(step_alpha)
        self.min_step_alpha = float(min_step_alpha)
        self.step_m = int(step_m)
        self.min_step_m = int(min_step_m)
        self.eps_advance = float(eps_advance)
        self.spike_factor = float(spike_factor)
        self.spike_abs_margin = float(spike_abs_margin)
        self.max_ticks_per_alpha = max(1, int(max_ticks_per_alpha))
        self.eps_gap = float(eps_gap)
        self.baseline_window = max(1, int(baseline_window))
        self.interleaved_2d = bool(interleaved_2d)

        self.bifurc_count = 0
        self.alpha_c: Optional[float] = None
        self.prev_inst_e_hom: Optional[float] = None
        self.last_step_succeeded = False
        self.ticks_at_level = 0
        self.e_history: List[float] = []
        self.pre_jump_baseline: Optional[float] = None
        self.pre_jump_alpha: float = self.alpha
        self.last_advance_kind: str = ""
        self.last_alpha_delta: float = 0.0
        self.alpha_steps_since_mask: int = 0
        self.bifurc_cooldown: int = 0
        self.bifurc_cooldown_advances: int = 0
        self.smooth_advances: int = 0
        self.mask_history: List[int] = [self.mask_m]

    def state_dict(self) -> Dict[str, object]:
        return {
            "alpha": self.alpha,
            "mask_m": self.mask_m,
            "target_mask_m": self.target_mask_m,
            "mask_levels": list(self.mask_levels),
            "step_alpha": self.step_alpha,
            "base_step_alpha": self.base_step_alpha,
            "step_m": self.step_m,
            "bifurc_count": self.bifurc_count,
            "alpha_c": self.alpha_c,
            "prev_inst_e_hom": self.prev_inst_e_hom,
            "last_step_succeeded": self.last_step_succeeded,
            "ticks_at_level": self.ticks_at_level,
            "e_history": list(self.e_history),
            "pre_jump_baseline": self.pre_jump_baseline,
            "pre_jump_alpha": self.pre_jump_alpha,
            "last_advance_kind": self.last_advance_kind,
            "last_alpha_delta": self.last_alpha_delta,
            "alpha_steps_since_mask": self.alpha_steps_since_mask,
            "bifurc_cooldown": self.bifurc_cooldown,
            "bifurc_cooldown_advances": self.bifurc_cooldown_advances,
            "smooth_advances": self.smooth_advances,
            "mask_history": list(self.mask_history),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.alpha = float(state.get("alpha", self.alpha))
        self.mask_m = int(state.get("mask_m", self.mask_m))
        self.target_mask_m = int(state.get("target_mask_m", self.target_mask_m))
        if "mask_levels" in state:
            self.mask_levels = tuple(int(x) for x in state["mask_levels"])  # type: ignore[union-attr]
        self.step_alpha = float(state.get("step_alpha", self.step_alpha))
        self.base_step_alpha = float(state.get("base_step_alpha", self.base_step_alpha))
        self.step_m = int(state.get("step_m", self.step_m))
        self.bifurc_count = int(state.get("bifurc_count", 0))
        ac = state.get("alpha_c")
        self.alpha_c = None if ac is None else float(ac)
        pe = state.get("prev_inst_e_hom")
        self.prev_inst_e_hom = None if pe is None else float(pe)
        self.last_step_succeeded = bool(state.get("last_step_succeeded", False))
        self.ticks_at_level = int(state.get("ticks_at_level", 0))
        self.e_history = [float(x) for x in state.get("e_history", [])]  # type: ignore[union-attr]
        pjb = state.get("pre_jump_baseline")
        self.pre_jump_baseline = None if pjb is None else float(pjb)
        self.pre_jump_alpha = float(state.get("pre_jump_alpha", self.alpha))
        self.last_advance_kind = str(state.get("last_advance_kind", ""))
        self.last_alpha_delta = float(state.get("last_alpha_delta", 0.0))
        self.alpha_steps_since_mask = int(state.get("alpha_steps_since_mask", 0))
        self.bifurc_cooldown = int(state.get("bifurc_cooldown", 0))
        self.bifurc_cooldown_advances = int(state.get("bifurc_cooldown_advances", 0))
        self.smooth_advances = int(state.get("smooth_advances", 0))
        if "mask_history" in state:
            self.mask_history = [int(x) for x in state["mask_history"]]  # type: ignore[union-attr]

    def _next_mask_level(self) -> int:
        for lvl in self.mask_levels:
            if lvl > self.mask_m:
                return min(self.target_mask_m, lvl)
        return min(self.target_mask_m, self.mask_m + self.step_m)

    def record_stagnation_bifurcation(self, inst_e_hom: float) -> None:
        """Record a stagnation-triggered bifurcation without rolling back full alpha=1.0 coupling."""
        self.bifurc_count += 1
        if self.alpha_c is None and self.alpha > 0.0:
            self.alpha_c = float(self.alpha)
        self.bifurc_cooldown = 2
        self.smooth_advances = 0
        self.ticks_at_level = 0
        self.e_history = [float(inst_e_hom)]
        self.last_step_succeeded = False

    def update(
        self, inst_e_hom: float, gl_gap: Optional[float] = None, dt_ticks: int = 1
    ) -> Tuple[bool, bool, bool]:
        """Evaluate current instantaneous homotopy energy and decide (advanced, bifurcated, boundary_changed).

        Never uses running historical minimums!
        """
        del gl_gap  # Convergence strictly governed by inst_e_hom < eps_advance or max_ticks_per_alpha
        e_cur = float(inst_e_hom)
        advanced = False
        bifurcated = False
        boundary_changed = False
        self.ticks_at_level += max(1, int(dt_ticks))
        if self.bifurc_cooldown > 0:
            self.bifurc_cooldown -= 1

        cooldown_skip_adv = False
        if self.last_step_succeeded and self.bifurc_cooldown_advances > 0:
            self.bifurc_cooldown_advances -= 1
            cooldown_skip_adv = True

        # 1. Direct previous-plateau baseline bifurcation check (skips initial alpha=0 -> step_alpha transition)
        can_check_bifurc = (
            self.last_step_succeeded
            and not cooldown_skip_adv
            and self.bifurc_cooldown == 0
            and self.bifurc_cooldown_advances == 0
            and self.pre_jump_baseline is not None
            and (self.pre_jump_alpha > 0.0 or self.last_advance_kind == "mask")
        )
        if can_check_bifurc:
            assert self.pre_jump_baseline is not None
            e_ref = float(self.pre_jump_baseline)
            thresh = e_ref * self.spike_factor
            if e_cur > thresh and (e_cur - e_ref) > self.spike_abs_margin:
                bifurcated = True
                self.bifurc_count += 1
                if self.alpha_c is None and self.alpha > 0.0:
                    self.alpha_c = float(self.alpha)
                self.step_alpha = max(self.min_step_alpha, round(0.5 * self.step_alpha, 6))
                if self.last_advance_kind == "alpha" and self.last_alpha_delta > self.min_step_alpha + 1e-9:
                    self.alpha = max(0.0, round(self.alpha - 0.5 * self.last_alpha_delta, 6))
                self.bifurc_cooldown = 2
                self.bifurc_cooldown_advances = 1
                self.smooth_advances = 0
                self.ticks_at_level = 0
                self.e_history = [e_cur]
                self.last_step_succeeded = False
            else:
                self.smooth_advances += 1
                if self.smooth_advances >= 2 and self.step_alpha < self.base_step_alpha:
                    self.step_alpha = min(self.base_step_alpha, round(self.step_alpha * 1.5, 6))
                    self.smooth_advances = 0

        if not bifurcated:
            self.e_history = (self.e_history + [e_cur])[-self.baseline_window:]
            self.last_step_succeeded = False
            has_room = (self.mask_m < self.target_mask_m) or (self.alpha < 1.0)
            relaxed = (e_cur < self.eps_advance) or (self.ticks_at_level >= self.max_ticks_per_alpha)
            if has_room and relaxed:
                self.pre_jump_baseline = sum(self.e_history) / float(len(self.e_history))
                self.pre_jump_alpha = self.alpha
                should_step_mask = (
                    self.alpha > 0.0
                    and self.mask_m < self.target_mask_m
                    and (
                        self.alpha >= 1.0
                        or (
                            self.interleaved_2d
                            and (
                                self.alpha_steps_since_mask >= 3
                                or (e_cur < self.eps_advance and self.alpha_steps_since_mask >= 2)
                            )
                        )
                    )
                )
                if should_step_mask:
                    nxt_m = self._next_mask_level()
                    if nxt_m > self.mask_m:
                        self.mask_m = nxt_m
                        self.mask_history.append(self.mask_m)
                        boundary_changed = True
                        advanced = True
                        self.last_advance_kind = "mask"
                        self.alpha_steps_since_mask = 0
                elif self.alpha < 1.0:
                    old_a = self.alpha
                    self.alpha = min(1.0, round(self.alpha + self.step_alpha, 6))
                    self.last_alpha_delta = max(0.0, round(self.alpha - old_a, 6))
                    self.alpha_steps_since_mask += 1
                    self.last_advance_kind = "alpha"
                    advanced = True
                elif self.mask_m < self.target_mask_m:
                    nxt_m = self._next_mask_level()
                    if nxt_m > self.mask_m:
                        self.mask_m = nxt_m
                        self.mask_history.append(self.mask_m)
                        boundary_changed = True
                        advanced = True
                        self.last_advance_kind = "mask"
                        self.alpha_steps_since_mask = 0

                if advanced:
                    self.ticks_at_level = 0
                    self.last_step_succeeded = True
                    self.e_history = []

        self.prev_inst_e_hom = e_cur
        return advanced, bifurcated, boundary_changed
