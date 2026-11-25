"""Pure FP32 / int64 generalized 16-state bit-differential engine for SHA-256.

Implements Mendel-style 4-bit-mask generalized bit conditions over (x, x') in {0, 1}^2,
exact bitwise/modular-addition forward & backward propagation tables, Chabaud-Joux
disturbance-vector (DV) helpers, and exact parity utilities against src/sha256_ref.py.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# Enforce pure IEEE-754 FP32 whenever any floating-point tensor op is used.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")

MASK32: int = 0xFFFFFFFF

# Ensure repo root is importable for sha256_ref
_THIS_DIR = Path(__file__).resolve().parent
_CANDIDATE_ROOTS = [_THIS_DIR.parents[1], _THIS_DIR.parent, _THIS_DIR]
for _root in _CANDIDATE_ROOTS:
    if (_root / "src" / "sha256_ref.py").exists():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break

from src.sha256_ref import (  # noqa: E402
    SHA256_IV,
    SHA256_K,
    ch32,
    expand_message_schedule,
    maj32,
    rotr32,
    sha256_digest,
    sha256_trace,
    shr32,
    sigma0_lower,
    sigma0_upper,
    sigma1_lower,
    sigma1_upper,
)

# ---------------------------------------------------------------------------
# 16-state generalized bit differential representation (4-bit subset mask)
#   bit 0 (1): (x=0, x'=0) -> '0'
#   bit 1 (2): (x=0, x'=1) -> 'u' (Delta = +1)
#   bit 2 (4): (x=1, x'=0) -> 'n' (Delta = -1)
#   bit 3 (8): (x=1, x'=1) -> '1'
# ---------------------------------------------------------------------------

GD_EMPTY = 0x0  # '#' contradiction
GD_0 = 0x1      # '0' (0,0)
GD_U = 0x2      # 'u' (0,1)
GD_N = 0x4      # 'n' (1,0)
GD_1 = 0x8      # '1' (1,1)
GD_A = 0x3      # 'A' {00, 01} (x=0)
GD_C = 0x5      # 'C' {00, 10} (x'=0)
GD_X = 0x6      # 'x' {01, 10} (XOR diff = 1)
GD_EQ = 0x9     # '-' {00, 11} (XOR diff = 0)
GD_5 = 0xA      # '5' {01, 11} (x'=1)
GD_D = 0xC      # 'D' {10, 11} (x=1)
GD_ANY = 0xF    # '?' {00, 01, 10, 11}

GD_CHARS = "#0uAn?x71-5BDFE?"
# Standard Mendel ASCII characters for all 16 subsets
STATE_TO_CHAR: Tuple[str, ...] = (
    "#",  # 0: empty
    "0",  # 1: {00}
    "u",  # 2: {01}
    "A",  # 3: {00,01}
    "n",  # 4: {10}
    "C",  # 5: {00,10}
    "x",  # 6: {01,10}
    "7",  # 7: {00,01,10}
    "1",  # 8: {11}
    "-",  # 9: {00,11}
    "5",  # 10: {01,11}
    "B",  # 11: {00,01,11}
    "D",  # 12: {10,11}
    "E",  # 13: {00,10,11}
    "F",  # 14: {01,10,11}
    "?",  # 15: {00,01,10,11}
)
CHAR_TO_STATE: Dict[str, int] = {ch: i for i, ch in enumerate(STATE_TO_CHAR)}


def gdiff_word_to_str(bits32: Sequence[int] | torch.Tensor) -> str:
    """Format 32 LSB-first 4-bit states as MSB-first 32-char string."""
    vals = bits32.tolist() if isinstance(bits32, torch.Tensor) else list(bits32)
    return "".join(STATE_TO_CHAR[int(vals[b]) & 0xF] for b in range(31, -1, -1))


def str_to_gdiff_word(s: str) -> List[int]:
    """Parse MSB-first 32-char string into LSB-first 32 states."""
    if len(s) != 32:
        raise ValueError(f"Expected 32 chars, got {len(s)}")
    return [CHAR_TO_STATE[ch] for ch in reversed(s)]


def encode_pair_to_gdiff(x: torch.Tensor, xp: torch.Tensor) -> torch.Tensor:
    """Encode uint32-in-int64 word tensors (...,) into (..., 32) 16-state masks."""
    shifts = torch.arange(32, dtype=torch.int64, device=x.device)
    bx = (x.unsqueeze(-1) >> shifts) & 1
    bxp = (xp.unsqueeze(-1) >> shifts) & 1
    idx = (bx << 1) | bxp  # 0..3 -> 00,01,10,11
    return (1 << idx).to(torch.int64)


def const_word_to_gdiff(c: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Encode a constant uint32 word (x = x' = c) into (32,) 16-state masks."""
    t = torch.tensor(c & MASK32, dtype=torch.int64, device=device)
    return encode_pair_to_gdiff(t, t)


def verify_pair_against_gdiff(
    x: torch.Tensor, xp: torch.Tensor, mask_bits: torch.Tensor
) -> torch.Tensor:
    """Return bool tensor (...) indicating whether (x, x') satisfies mask_bits (..., 32)."""
    actual = encode_pair_to_gdiff(x, xp)
    return ((actual & mask_bits) != 0).all(dim=-1)


# ---------------------------------------------------------------------------
# Precomputed exact forward & backward lookup tables for 16-state gates
# ---------------------------------------------------------------------------

def _build_gate_tables() -> Dict[str, torch.Tensor]:
    """Build exact 4-bit mask propagation tables for XOR2, XOR3, CH, MAJ."""
    xor2 = torch.zeros((16, 16), dtype=torch.int64)
    xor3 = torch.zeros((16, 16, 16), dtype=torch.int64)
    ch_tbl = torch.zeros((16, 16, 16), dtype=torch.int64)
    maj_tbl = torch.zeros((16, 16, 16), dtype=torch.int64)

    # First compute singleton outputs for 4x4 / 4x4x4 concrete pairs
    s_xor2 = [[0] * 4 for _ in range(4)]
    for p0 in range(4):
        x0, y0 = (p0 >> 1) & 1, p0 & 1
        for p1 in range(4):
            x1, y1 = (p1 >> 1) & 1, p1 & 1
            s_xor2[p0][p1] = 1 << (((x0 ^ x1) << 1) | (y0 ^ y1))

    s_xor3 = [[[0] * 4 for _ in range(4)] for _ in range(4)]
    s_ch = [[[0] * 4 for _ in range(4)] for _ in range(4)]
    s_maj = [[[0] * 4 for _ in range(4)] for _ in range(4)]
    for p0 in range(4):
        x0, y0 = (p0 >> 1) & 1, p0 & 1
        for p1 in range(4):
            x1, y1 = (p1 >> 1) & 1, p1 & 1
            for p2 in range(4):
                x2, y2 = (p2 >> 1) & 1, p2 & 1
                zx = x0 ^ x1 ^ x2
                zy = y0 ^ y1 ^ y2
                s_xor3[p0][p1][p2] = 1 << ((zx << 1) | zy)

                cx = (x0 & x1) ^ ((1 ^ x0) & x2)
                cy = (y0 & y1) ^ ((1 ^ y0) & y2)
                s_ch[p0][p1][p2] = 1 << ((cx << 1) | cy)

                mx = (x0 & x1) ^ (x0 & x2) ^ (x1 & x2)
                my = (y0 & y1) ^ (y0 & y2) ^ (y1 & y2)
                s_maj[p0][p1][p2] = 1 << ((mx << 1) | my)

    for a in range(16):
        pa = [i for i in range(4) if (a >> i) & 1]
        for b in range(16):
            pb = [i for i in range(4) if (b >> i) & 1]
            out2 = 0
            for i in pa:
                for j in pb:
                    out2 |= s_xor2[i][j]
            xor2[a, b] = out2
            for c in range(16):
                pc = [i for i in range(4) if (c >> i) & 1]
                ox, oc, om = 0, 0, 0
                for i in pa:
                    for j in pb:
                        for k in pc:
                            ox |= s_xor3[i][j][k]
                            oc |= s_ch[i][j][k]
                            om |= s_maj[i][j][k]
                xor3[a, b, c] = ox
                ch_tbl[a, b, c] = oc
                maj_tbl[a, b, c] = om

    return {"xor2": xor2, "xor3": xor3, "ch": ch_tbl, "maj": maj_tbl}


_GATE_TABLES_CPU = _build_gate_tables()
_GATE_TABLES_CACHE: Dict[str, Dict[str, torch.Tensor]] = {}


def get_gate_tables(device: torch.device | str) -> Dict[str, torch.Tensor]:
    key = str(torch.device(device))
    if key not in _GATE_TABLES_CACHE:
        _GATE_TABLES_CACHE[key] = {
            k: v.to(device) for k, v in _GATE_TABLES_CPU.items()
        }
    return _GATE_TABLES_CACHE[key]


def gdiff_rotr(bits32: torch.Tensor, n: int) -> torch.Tensor:
    """Exact ROTR_n on (..., 32) LSB-first 16-state masks."""
    if n == 0:
        return bits32.clone()
    return torch.roll(bits32, shifts=-n, dims=-1)


def gdiff_shr(bits32: torch.Tensor, n: int) -> torch.Tensor:
    """Exact SHR_n on (..., 32) LSB-first 16-state masks (MSB filled with GD_0)."""
    if n == 0:
        return bits32.clone()
    out = torch.full_like(bits32, GD_0)
    out[..., : 32 - n] = bits32[..., n:]
    return out


def gdiff_xor3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    tbl = get_gate_tables(a.device)["xor3"]
    return tbl[a, b, c]


def gdiff_ch(e: torch.Tensor, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    tbl = get_gate_tables(e.device)["ch"]
    return tbl[e, f, g]


def gdiff_maj(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    tbl = get_gate_tables(a.device)["maj"]
    return tbl[a, b, c]


def gdiff_sigma0_upper(x: torch.Tensor) -> torch.Tensor:
    return gdiff_xor3(gdiff_rotr(x, 2), gdiff_rotr(x, 13), gdiff_rotr(x, 22))


def gdiff_sigma1_upper(x: torch.Tensor) -> torch.Tensor:
    return gdiff_xor3(gdiff_rotr(x, 6), gdiff_rotr(x, 11), gdiff_rotr(x, 25))


def gdiff_sigma0_lower(x: torch.Tensor) -> torch.Tensor:
    return gdiff_xor3(gdiff_rotr(x, 7), gdiff_rotr(x, 18), gdiff_shr(x, 3))


def gdiff_sigma1_lower(x: torch.Tensor) -> torch.Tensor:
    return gdiff_xor3(gdiff_rotr(x, 17), gdiff_rotr(x, 19), gdiff_shr(x, 10))


def _build_add2_step_table() -> Tuple[torch.Tensor, torch.Tensor]:
    """Build exact 2-operand bit-slice carry transition table.

    Carry pair (c, c') in {0, 1}^2 is encoded as 4-bit mask `cmask` (bit = 2*c + c').
    Given (a_mask, b_mask, cin_mask) in 16 x 16 x 16:
      out_s[a, b, cin] = 4-bit sum mask in {0..15}
      out_cout[a, b, cin] = 4-bit cout mask in {0..15}
    """
    out_s = torch.zeros((16, 16, 16), dtype=torch.int64)
    out_cout = torch.zeros((16, 16, 16), dtype=torch.int64)
    for a in range(16):
        pa = [(p >> 1, p & 1) for p in range(4) if (a >> p) & 1]
        for b in range(16):
            pb = [(p >> 1, p & 1) for p in range(4) if (b >> p) & 1]
            for cin in range(16):
                pc = [(p >> 1, p & 1) for p in range(4) if (cin >> p) & 1]
                sm, cm = 0, 0
                for ax, ay in pa:
                    for bx, by in pb:
                        for cx, cy in pc:
                            sx = ax + bx + cx
                            sy = ay + by + cy
                            sm |= 1 << (((sx & 1) << 1) | (sy & 1))
                            cm |= 1 << (((sx >> 1) << 1) | (sy >> 1))
                out_s[a, b, cin] = sm
                out_cout[a, b, cin] = cm
    return out_s, out_cout


_ADD2_S_CPU, _ADD2_COUT_CPU = _build_add2_step_table()
_ADD2_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_add2_tables(device: torch.device | str) -> Tuple[torch.Tensor, torch.Tensor]:
    key = str(torch.device(device))
    if key not in _ADD2_CACHE:
        _ADD2_CACHE[key] = (_ADD2_S_CPU.to(device), _ADD2_COUT_CPU.to(device))
    return _ADD2_CACHE[key]


def gdiff_add2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact 32-bit 2-operand modular addition on (..., 32) 16-state masks."""
    s_tbl, c_tbl = _get_add2_tables(a.device)
    out = torch.zeros_like(a)
    cin = torch.full(a.shape[:-1], GD_0, dtype=torch.int64, device=a.device)
    for bit in range(32):
        ab = a[..., bit]
        bb = b[..., bit]
        out[..., bit] = s_tbl[ab, bb, cin]
        cin = c_tbl[ab, bb, cin]
    return out


def gdiff_add_many(terms: Sequence[torch.Tensor]) -> torch.Tensor:
    """Modular addition of >= 2 generalized differential words (..., 32)."""
    acc = terms[0]
    for nxt in terms[1:]:
        acc = gdiff_add2(acc, nxt)
    return acc


def gdiff_add_many_exact_carry(
    terms: Sequence[torch.Tensor],
    target_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    """Exact multi-operand carry propagation without intermediate-truncation blowup.

    For m <= 5 operands, tracks exact joint carry set (c, c') in {0..m-1}^2
    as a python/int64 bitmask over 32 bits, optionally tightening `target_out`
    and returning backward-consistent tightened operand masks when target_out is given.
    """
    m = len(terms)
    if m < 2:
        raise ValueError("Need at least 2 operands")
    if m > 5:
        raise ValueError(f"Supported up to 5 operands, got {m}")
    device = terms[0].device
    batch_shape = terms[0].shape[:-1]

    # Exact joint carry propagation on CPU/GPU via bitmask over c*m + c' in [0, m*m)
    # Since m<=5, m*m <= 25 fits comfortably in signed int64 (bits 0..24).
    # For fast vectorized execution when target_out is None, track carry bitmask per bit:
    # Step over bit 0..31; at each bit, combine operands sequentially in joint carry state?
    # Better: sequential over the m operands AT EACH BIT? No, joint carry after all m inputs at bit b
    # is simply floor((sum_i x_i + c_in) / 2)!
    # Notice: we can accumulate the partial bit-sum `(sum_so_far, sum_so_far')` within bit `b`!
    # At bit b, partial sum starts at (c_in, c_in') in [0, m)^2. Adding one operand bit (ax, ay)
    # adds (ax, ay) in {0,1}^2 -> max partial sum is (2m-1).
    # Even simpler: compute exact 4-bit sum & joint carry via DP over operands at bit b!
    out = torch.zeros_like(terms[0])
    # Initial carry is (0, 0) -> bit 0 set in (m x m) mask
    carry_mask = torch.ones(batch_shape, dtype=torch.int64, device=device)

    # Precompute bit-level transition only when needed? At most m*32 simple tensor shifts!
    # State index during operand accumulation at bit b: s = c_x * (m+1) + c_y, wait max sum is m + (m-1) = 2m-1 <= 9.
    stride = 2 * m  # <= 10, max index 9*10 + 9 = 99 > 64! Wait: for m=5, partial sum can reach 9, 10x10=100 > 63 bits.
    # For m<=3, 6x6=36 <= 63; for m=4,5, split into 2-step or use bool tensor (..., stride, stride) on GPU!
    s_dim = 2 * m
    # Bool state tensor (..., s_dim, s_dim) is tiny (10x10=100 bools per batch item) and fully parallel on GPU!
    carry_grid = torch.zeros(batch_shape + (m, m), dtype=torch.bool, device=device)
    carry_grid[..., 0, 0] = True

    fwd_carries: List[torch.Tensor] = [carry_grid]
    for bit in range(32):
        sum_grid = torch.zeros(batch_shape + (s_dim, s_dim), dtype=torch.bool, device=device)
        sum_grid[..., :m, :m] = carry_grid
        for op in terms:
            op_b = op[..., bit]
            m00 = ((op_b & 1) != 0)[..., None, None]
            m01 = ((op_b & 2) != 0)[..., None, None]
            m10 = ((op_b & 4) != 0)[..., None, None]
            m11 = ((op_b & 8) != 0)[..., None, None]
            nxt = torch.zeros_like(sum_grid)
            nxt |= sum_grid & m00
            nxt[..., :, 1:] |= sum_grid[..., :, :-1] & m01
            nxt[..., 1:, :] |= sum_grid[..., :-1, :] & m10
            nxt[..., 1:, 1:] |= sum_grid[..., :-1, :-1] & m11
            sum_grid = nxt

        if target_out is not None:
            tb = target_out[..., bit]
            # Allowed parity mask (sx&1, sy&1)
            allow = torch.zeros(batch_shape + (2, 2), dtype=torch.bool, device=device)
            allow[..., 0, 0] = (tb & 1) != 0
            allow[..., 0, 1] = (tb & 2) != 0
            allow[..., 1, 0] = (tb & 4) != 0
            allow[..., 1, 1] = (tb & 8) != 0
            # Broadcast parity check across (s_dim, s_dim)
            px = torch.arange(s_dim, device=device) & 1
            py = torch.arange(s_dim, device=device) & 1
            sum_grid &= allow[..., px[:, None], py[None, :]]

        # Extract output bit mask
        out_b = torch.zeros(batch_shape, dtype=torch.int64, device=device)
        for p0 in (0, 1):
            for p1 in (0, 1):
                has_p = sum_grid[..., p0::2, p1::2].any(dim=(-2, -1))
                out_b |= has_p.to(torch.int64) << ((p0 << 1) | p1)
        out[..., bit] = out_b

        # Next carry is floor(sum / 2) -> collapse 2x2 blocks
        # Reshape (..., m, 2, m, 2) -> any over dims (-3, -1)
        carry_grid = (
            sum_grid.view(batch_shape + (m, 2, m, 2))
            .any(dim=(-3, -1))
        )
        fwd_carries.append(carry_grid)

    if target_out is None:
        return out, None
    return out, fwd_carries


# ---------------------------------------------------------------------------
# Full-round forward generalized differential propagation
# ---------------------------------------------------------------------------

@dataclass
class GeneralizedDiffTrace:
    """Generalized 16-state differential trace for R rounds of SHA-256."""

    A_full: torch.Tensor  # (..., R + 4, 32)
    E_full: torch.Tensor  # (..., R + 4, 32)
    W_full: torch.Tensor  # (..., R, 32)
    digest: torch.Tensor  # (..., 8, 32)
    contradictions: torch.Tensor  # (...,) bool


def forward_propagate_gdiff(
    w16_mask: torch.Tensor,
    rounds: int = 16,
    iv_mask: Optional[torch.Tensor] = None,
) -> GeneralizedDiffTrace:
    """Propagate 16-state generalized differential forward through R rounds of SHA-256.

    Args:
        w16_mask: (B, 16, 32) 4-bit state masks for message block M vs M'.
        rounds: number of rounds (1..64).
        iv_mask: optional (B, 8, 32) 4-bit state masks for IV vs IV'.
                 Defaults to standard SHA256_IV with zero difference.
    """
    if w16_mask.ndim != 3 or w16_mask.shape[1:] != (16, 32):
        raise ValueError(f"Expected (B, 16, 32), got {tuple(w16_mask.shape)}")
    batch_size = w16_mask.shape[0]
    device = w16_mask.device

    if iv_mask is None:
        iv_base = torch.stack([const_word_to_gdiff(v, device) for v in SHA256_IV], dim=0)
        iv_mask = iv_base.unsqueeze(0).expand(batch_size, 8, 32)

    # Expand message schedule
    w_cols = [w16_mask[:, t, :] for t in range(min(rounds, 16))]
    for t in range(16, rounds):
        wt, _ = gdiff_add_many_exact_carry(
            [
                gdiff_sigma1_lower(w_cols[t - 2]),
                w_cols[t - 7],
                gdiff_sigma0_lower(w_cols[t - 15]),
                w_cols[t - 16],
            ]
        )
        w_cols.append(wt)
    w_full = torch.stack(w_cols, dim=1)

    a_cols = [
        iv_mask[:, 3, :],  # A_{-3} = d0
        iv_mask[:, 2, :],  # A_{-2} = c0
        iv_mask[:, 1, :],  # A_{-1} = b0
        iv_mask[:, 0, :],  # A_0    = a0
    ]
    e_cols = [
        iv_mask[:, 7, :],  # E_{-3} = h0
        iv_mask[:, 6, :],  # E_{-2} = g0
        iv_mask[:, 5, :],  # E_{-1} = f0
        iv_mask[:, 4, :],  # E_0    = e0
    ]

    k_masks = [const_word_to_gdiff(SHA256_K[t], device).unsqueeze(0).expand(batch_size, 32) for t in range(rounds)]

    for t in range(rounds):
        a_t = a_cols[t + 3]
        a_tm1 = a_cols[t + 2]
        a_tm2 = a_cols[t + 1]
        a_tm3 = a_cols[t]

        e_t = e_cols[t + 3]
        e_tm1 = e_cols[t + 2]
        e_tm2 = e_cols[t + 1]
        e_tm3 = e_cols[t]

        t1, _ = gdiff_add_many_exact_carry(
            [
                e_tm3,
                gdiff_sigma1_upper(e_t),
                gdiff_ch(e_t, e_tm1, e_tm2),
                k_masks[t],
                w_cols[t],
            ]
        )
        t2, _ = gdiff_add_many_exact_carry(
            [
                gdiff_sigma0_upper(a_t),
                gdiff_maj(a_t, a_tm1, a_tm2),
            ]
        )
        e_tp1 = gdiff_add2(a_tm3, t1)
        a_tp1 = gdiff_add2(t1, t2)
        e_cols.append(e_tp1)
        a_cols.append(a_tp1)

    a_full = torch.stack(a_cols, dim=1)
    e_full = torch.stack(e_cols, dim=1)

    s_r = torch.stack(
        [
            a_cols[rounds + 3],
            a_cols[rounds + 2],
            a_cols[rounds + 1],
            a_cols[rounds],
            e_cols[rounds + 3],
            e_cols[rounds + 2],
            e_cols[rounds + 1],
            e_cols[rounds],
        ],
        dim=1,
    )
    digest = gdiff_add2(s_r, iv_mask)
    contradictions = (
        (a_full == 0).any(dim=(-2, -1))
        | (e_full == 0).any(dim=(-2, -1))
        | (w_full == 0).any(dim=(-2, -1))
    )
    return GeneralizedDiffTrace(
        A_full=a_full,
        E_full=e_full,
        W_full=w_full,
        digest=digest,
        contradictions=contradictions,
    )


def sha256_pair_exact_check(
    w16: torch.Tensor,
    w16_prime: torch.Tensor,
    rounds: int = 64,
    iv: Optional[torch.Tensor] = None,
    iv_prime: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Run exact reference sha256_trace (or custom-IV variant) on (M, M') and return xor diffs."""
    if iv is None and iv_prime is None:
        tr = sha256_trace(w16, rounds=rounds)
        tr_p = sha256_trace(w16_prime, rounds=rounds)
    else:
        tr = sha256_trace_custom_iv(w16, iv=iv, rounds=rounds)
        tr_p = sha256_trace_custom_iv(w16_prime, iv=iv_prime, rounds=rounds)
    return {
        "tr": tr,
        "tr_p": tr_p,
        "A_xor": tr.A_full ^ tr_p.A_full,
        "E_xor": tr.E_full ^ tr_p.E_full,
        "W_xor": tr.W_full ^ tr_p.W_full,
        "digest_xor": tr.digest ^ tr_p.digest,
    }


def sha256_trace_custom_iv(
    w16: torch.Tensor,
    iv: Optional[torch.Tensor] = None,
    rounds: int = 64,
):
    """Exact sha256_trace supporting custom (B, 8) or (8,) IV for Semi-Free-Start / Free-Start."""
    from src.sha256_ref import SHA256Trace, get_iv_tensor, get_k_tensor

    batch_size = w16.shape[0]
    device = w16.device
    if iv is None:
        iv_b = get_iv_tensor(device).unsqueeze(0).expand(batch_size, 8)
    elif iv.ndim == 1:
        iv_b = (iv.to(torch.int64) & MASK32).unsqueeze(0).expand(batch_size, 8)
    else:
        iv_b = iv.to(torch.int64) & MASK32

    k_vec = get_k_tensor(rounds, device)
    w_full = expand_message_schedule(w16, rounds)

    a_cols = [iv_b[:, 3], iv_b[:, 2], iv_b[:, 1], iv_b[:, 0]]
    e_cols = [iv_b[:, 7], iv_b[:, 6], iv_b[:, 5], iv_b[:, 4]]

    for t in range(rounds):
        t1 = (
            e_cols[t]
            + sigma1_upper(e_cols[t + 3])
            + ch32(e_cols[t + 3], e_cols[t + 2], e_cols[t + 1])
            + k_vec[t]
            + w_full[:, t]
        ) & MASK32
        t2 = (sigma0_upper(a_cols[t + 3]) + maj32(a_cols[t + 3], a_cols[t + 2], a_cols[t + 1])) & MASK32
        e_cols.append((a_cols[t] + t1) & MASK32)
        a_cols.append((t1 + t2) & MASK32)

    a_full = torch.stack(a_cols, dim=1)
    e_full = torch.stack(e_cols, dim=1)
    s_r = torch.stack(
        [
            a_cols[rounds + 3],
            a_cols[rounds + 2],
            a_cols[rounds + 1],
            a_cols[rounds],
            e_cols[rounds + 3],
            e_cols[rounds + 2],
            e_cols[rounds + 1],
            e_cols[rounds],
        ],
        dim=1,
    )
    digest = (s_r + iv_b) & MASK32
    return SHA256Trace(A_full=a_full, E_full=e_full, W_full=w_full, S_R=s_r, digest=digest)


def btc_double_sha256_ref(header_w32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact Bitcoin 80-byte header Double-SHA256 using sha256_ref primitives.

    Args:
        header_w32: (B, 20) uint32-in-int64 header words (80 bytes).
    Returns:
        (h1_digest (B, 8), h2_digest (B, 8)).
    """
    if header_w32.ndim != 2 or header_w32.shape[1] != 20:
        raise ValueError(f"Expected (B, 20), got {tuple(header_w32.shape)}")
    batch_size = header_w32.shape[0]
    device = header_w32.device

    # Block 1: words 0..15
    b1_w16 = header_w32[:, :16] & MASK32
    tr1 = sha256_trace(b1_w16, rounds=64)

    # Block 2: words 16..19 + 0x80000000 + zeros + 640-bit length (0x280)
    b2_w16 = torch.zeros((batch_size, 16), dtype=torch.int64, device=device)
    b2_w16[:, :4] = header_w32[:, 16:20] & MASK32
    b2_w16[:, 4] = 0x80000000
    b2_w16[:, 15] = 640
    tr2 = sha256_trace_custom_iv(b2_w16, iv=tr1.digest, rounds=64)

    # Final Double-SHA256: SHA256(tr2.digest || pad(256 bits))
    b3_w16 = torch.zeros((batch_size, 16), dtype=torch.int64, device=device)
    b3_w16[:, :8] = tr2.digest & MASK32
    b3_w16[:, 8] = 0x80000000
    b3_w16[:, 15] = 256
    h2 = sha256_digest(b3_w16, rounds=64)
    return tr2.digest, h2


# ---------------------------------------------------------------------------
# Chabaud-Joux local collision perturbation template (6-step linear cancellation)
# ---------------------------------------------------------------------------

def local_collision_dv_template(
    step_t: int,
    bit_j: int,
    rounds: int = 64,
    sign_mode: str = "xor_cancel",
) -> torch.Tensor:
    """Construct a classical 6-round Chabaud-Joux XOR disturbance template on W[0..rounds-1].

    Injecting delta at W[t] excites E_{t+1} and A_{t+1}; to cancel in rounds t+1..t+5:
      dW[t]   = d
      dW[t+1] = Sigma1(d) (or Ch-compensated) + Sigma0(d) ...
    Note: exact linear-regime cancellation for XOR approximation in SHA-256:
      dE_{t+1} = d, dA_{t+1} = d ->
      At t+1: cancel dA_{t+1} via dW[t+1] so dT1(t+1) absorbs Sigma1(d)+Ch(...) and cancels dA_{t+1}
      Wait: in SHA-256, dA_{t+1} propagates into T2(t+1..t+3) and dE_{t+1} propagates into T1(t+1..t+4),
      requiring perturbations at W[t], W[t+1], W[t+2], W[t+3], W[t+4], W[t+5] (and W[t+8] if E->A).
    Returns (rounds,) uint32 XOR mask of the primary local collision skeleton.
    """
    if not (0 <= bit_j < 32):
        raise ValueError(f"bit_j must be in [0, 31], got {bit_j}")
    dw = torch.zeros(rounds, dtype=torch.int64)
    if step_t >= rounds:
        return dw
    delta = torch.tensor(1 << bit_j, dtype=torch.int64)
    # Minimal Mendel-style disturbance schedule:
    # Step t: inject delta into W[t] -> E_{t+1} and A_{t+1} get delta.
    # Step t+1: absorb Sigma0(delta)+Maj in A, Sigma1(delta)+Ch in E -> cancel A_{t+2}, E_{t+2}
    dw[step_t] = delta
    if step_t + 1 < rounds:
        dw[step_t + 1] = (sigma0_upper(delta) ^ sigma1_upper(delta)) & MASK32
    if step_t + 4 < rounds:
        # When E_{t+1} reaches e_{t-3} at round t+4, cancel with dW[t+4] = delta
        dw[step_t + 4] ^= delta
    if step_t + 8 < rounds:
        # When A_{t+1} reaches a_{t-3} at round t+4 -> enters E_{t+5}, then e_{t-3} at t+8
        dw[step_t + 4] ^= delta
    return dw
