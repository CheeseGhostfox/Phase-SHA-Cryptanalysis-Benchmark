"""Exact 16-state Mendel generalized differential state machine for SHA-256.

Aligned 100% with SPEC.md Section 2:
  - 4-bit state mask encoding: idx(x, x') = x + 2 * x' in {0..3}
      0x0 '#': empty (contradiction)
      0x1 '0': {(0,0)}
      0x2 'u': {(1,0)}  (x=1, x'=0, Delta = +1)
      0x4 'n': {(0,1)}  (x=0, x'=1, Delta = -1)
      0x8 '1': {(1,1)}
      0x6 'x': {(1,0),(0,1)} (XOR diff = 1)
      0x9 '-': {(0,0),(1,1)} (XOR diff = 0)
      0xF '?': unconstrained
  - Exact 4-way arc-consistency LUTs (16^4 = 65,536) for Ch, Maj, XOR3
  - Exact 5-way arc-consistency LUT (16^5 = 1,048,576) for atomic 2-operand ADD32 full-adder
  - Bidirectional forward/backward fixed-point propagation over R-round SHA-256
  - Mandatory N >= 1,000,000 randomized soundness & exactness self-test vs src/sha256_ref.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Enforce pure IEEE-754 FP32 (no TF32)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")

MASK32: int = 0xFFFFFFFF

_THIS_DIR = Path(__file__).resolve().parent
for _root in [_THIS_DIR, *_THIS_DIR.parents]:
    if (_root / "src" / "sha256_ref.py").exists():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break

from src.sha256_ref import (  # noqa: E402
    SHA256_IV,
    SHA256_K,
    ch32,
    maj32,
    rotr32,
    sha256_trace,
    shr32,
    sigma0_lower,
    sigma0_upper,
    sigma1_lower,
    sigma1_upper,
)

# 16-state mask constants (idx = x + 2*x')
GD_EMPTY: int = 0x0  # '#'
GD_0: int = 0x1      # '0' (0,0)
GD_U: int = 0x2      # 'u' (1,0)
GD_3: int = 0x3      # '3' {(0,0),(1,0)}
GD_N: int = 0x4      # 'n' (0,1)
GD_5: int = 0x5      # '5' {(0,0),(0,1)}
GD_X: int = 0x6      # 'x' {(1,0),(0,1)}
GD_7: int = 0x7      # '7'
GD_1: int = 0x8      # '1' (1,1)
GD_EQ: int = 0x9     # '-' {(0,0),(1,1)}
GD_A: int = 0xA      # 'A' {(1,0),(1,1)}
GD_B: int = 0xB      # 'B'
GD_C: int = 0xC      # 'C' {(0,1),(1,1)}
GD_D: int = 0xD      # 'D'
GD_E: int = 0xE      # 'E'
GD_ANY: int = 0xF    # '?'

STATE_TO_CHAR: Tuple[str, ...] = (
    "#", "0", "u", "3", "n", "5", "x", "7",
    "1", "-", "A", "B", "C", "D", "E", "?",
)
CHAR_TO_STATE: Dict[str, int] = {ch: i for i, ch in enumerate(STATE_TO_CHAR)}

POPCNT_LUT = torch.tensor([bin(i).count("1") for i in range(16)], dtype=torch.int64)


def gdiff_word_to_str(bits32: Sequence[int] | torch.Tensor) -> str:
    """Format 32 LSB-first 4-bit states as MSB-first 32-char Mendel string."""
    vals = bits32.tolist() if isinstance(bits32, torch.Tensor) else list(bits32)
    return "".join(STATE_TO_CHAR[int(vals[b]) & 0xF] for b in range(31, -1, -1))


def str_to_gdiff_word(s: str) -> List[int]:
    """Parse MSB-first 32-char Mendel string into LSB-first 32 states."""
    if len(s) != 32:
        raise ValueError(f"Expected 32 chars, got {len(s)}")
    return [CHAR_TO_STATE[ch] for ch in reversed(s)]


def encode_pair_to_gdiff(x: torch.Tensor, xp: torch.Tensor) -> torch.Tensor:
    """Encode uint32-in-int64 tensors (...,) into (..., 32) 16-state masks (idx = x + 2*x')."""
    shifts = torch.arange(32, dtype=torch.int64, device=x.device)
    bx = (x.unsqueeze(-1) >> shifts) & 1
    bxp = (xp.unsqueeze(-1) >> shifts) & 1
    idx = bx + (bxp << 1)
    return (1 << idx).to(torch.int64)


def const_word_to_gdiff(c: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Encode constant uint32 word (x = x' = c) into (32,) 16-state masks."""
    t = torch.tensor(c & MASK32, dtype=torch.int64, device=device)
    return encode_pair_to_gdiff(t, t)


def verify_pair_against_gdiff(
    x: torch.Tensor, xp: torch.Tensor, mask_bits: torch.Tensor
) -> torch.Tensor:
    """Return bool tensor (...) checking whether (x, x') falls inside mask_bits (..., 32)."""
    actual = encode_pair_to_gdiff(x, xp)
    return ((actual & mask_bits) != 0).all(dim=-1)


# ---------------------------------------------------------------------------
# Vectorized precomputation of exact 4-way and 5-way arc-consistency LUTs
# ---------------------------------------------------------------------------

def _build_all_luts() -> Dict[str, torch.Tensor]:
    """Build exact 4-way (16^4) LUTs for Ch/Maj/XOR3 and 5-way (16^5) LUT for FullAdder."""
    k = np.arange(64, dtype=np.uint64)
    u = (k >> 0) & 1
    up = (k >> 1) & 1
    v = (k >> 2) & 1
    vp = (k >> 3) & 1
    w = (k >> 4) & 1
    wp = (k >> 5) & 1

    idx_u = u + 2 * up
    idx_v = v + 2 * vp
    idx_w = w + 2 * wp

    def _var_bitmasks(idx_arr: np.ndarray) -> np.ndarray:
        """Return (16,) uint64 bitmasks over k=0..63 for each 4-bit mask m in 0..15."""
        # For each concrete pair p in 0..3, bitmask of k where idx_arr[k] == p
        p_masks = np.array(
            [np.sum(np.uint64(1) << k[idx_arr == p]) for p in range(4)],
            dtype=np.uint64,
        )
        out = np.zeros(16, dtype=np.uint64)
        for m in range(16):
            acc = np.uint64(0)
            for p in range(4):
                if (m >> p) & 1:
                    acc |= p_masks[p]
            out[m] = acc
        return out, p_masks

    mu_64, pu_64 = _var_bitmasks(idx_u)
    mv_64, pv_64 = _var_bitmasks(idx_v)
    mw_64, pw_64 = _var_bitmasks(idx_w)

    def _build_3in_1out_lut(fn) -> torch.Tensor:
        z = fn(u, v, w)
        zp = fn(up, vp, wp)
        idx_z = z + 2 * zp
        mz_64, pz_64 = _var_bitmasks(idx_z)

        # (16, 16, 16) -> 4096 uint64
        uvw = (mu_64[:, None, None] & mv_64[None, :, None] & mw_64[None, None, :]).reshape(4096)
        valid = uvw[:, None] & mz_64[None, :]  # (4096, 16)

        def _proj(p_masks: np.ndarray) -> np.ndarray:
            res = np.zeros(valid.shape, dtype=np.int32)
            for p in range(4):
                res |= ((valid & p_masks[p]) != 0).astype(np.int32) << p
            return res

        nu = _proj(pu_64)
        nv = _proj(pv_64)
        nw = _proj(pw_64)
        nz = _proj(pz_64)
        packed = (nu | (nv << 4) | (nw << 8) | (nz << 12)).reshape(65536)
        return torch.from_numpy(packed)

    ch_lut = _build_3in_1out_lut(lambda a, b, c: (a & b) ^ ((1 ^ a) & c))
    maj_lut = _build_3in_1out_lut(lambda a, b, c: (a & b) ^ (a & c) ^ (b & c))
    xor3_lut = _build_3in_1out_lut(lambda a, b, c: a ^ b ^ c)

    # 5-way FullAdder LUT: (x=u, y=v, cin=w) -> (z = x^y^cin, cout = Maj(x,y,cin))
    z_fa = u ^ v ^ w
    zp_fa = up ^ vp ^ wp
    c_fa = (u & v) | (u & w) | (v & w)
    cp_fa = (up & vp) | (up & wp) | (vp & wp)
    mz_64, pz_64 = _var_bitmasks(z_fa + 2 * zp_fa)
    mc_64, pc_64 = _var_bitmasks(c_fa + 2 * cp_fa)

    xyc = (mu_64[:, None, None] & mv_64[None, :, None] & mw_64[None, None, :]).reshape(4096)
    zc = (mz_64[:, None] & mc_64[None, :]).reshape(256)
    valid5 = xyc[:, None] & zc[None, :]  # (4096, 256) = 1,048,576 entries

    def _proj5(p_masks: np.ndarray) -> np.ndarray:
        res = np.zeros(valid5.shape, dtype=np.int32)
        for p in range(4):
            res |= ((valid5 & p_masks[p]) != 0).astype(np.int32) << p
        return res

    nx = _proj5(pu_64)
    ny = _proj5(pv_64)
    ncin = _proj5(pw_64)
    nz5 = _proj5(pz_64)
    ncout = _proj5(pc_64)
    fa_packed = (nx | (ny << 4) | (ncin << 8) | (nz5 << 12) | (ncout << 16)).reshape(1048576)
    fa_lut = torch.from_numpy(fa_packed)

    return {"ch": ch_lut, "maj": maj_lut, "xor3": xor3_lut, "fa": fa_lut}


_LUTS_CPU = _build_all_luts()
_LUTS_CACHE: Dict[str, Dict[str, torch.Tensor]] = {}


def get_luts(device: torch.device | str) -> Dict[str, torch.Tensor]:
    key = str(torch.device(device))
    if key not in _LUTS_CACHE:
        _LUTS_CACHE[key] = {k: v.to(device) for k, v in _LUTS_CPU.items()}
    return _LUTS_CACHE[key]


def propagate_3in_1out(
    lut: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    z: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact 4-way arc-consistency update on (...,) 16-state masks."""
    idx = (u << 12) | (v << 8) | (w << 4) | z
    packed = lut[idx]
    return (
        packed & 0xF,
        (packed >> 4) & 0xF,
        (packed >> 8) & 0xF,
        (packed >> 12) & 0xF,
    )


def propagate_add32_bidir(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    carries: Optional[torch.Tensor] = None,
    sweeps: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact bidirectional arc-consistency propagation for 32-bit z = (x + y) mod 2^32.

    Args:
        x, y, z: (..., 32) int64 tensors of 4-bit state masks.
        carries: optional (..., 33) int64 carry masks (c_0 fixed to GD_0, c_32 to GD_ANY).
    """
    fa_lut = get_luts(x.device)["fa"]
    x = x.clone()
    y = y.clone()
    z = z.clone()
    if carries is None:
        c = torch.full(x.shape[:-1] + (33,), GD_ANY, dtype=torch.int64, device=x.device)
        c[..., 0] = GD_0
    else:
        c = carries.clone()
        c[..., 0] &= GD_0

    for _ in range(sweeps):
        # Forward b = 0 -> 31
        for b in range(32):
            idx = (
                (x[..., b] << 16)
                | (y[..., b] << 12)
                | (c[..., b] << 8)
                | (z[..., b] << 4)
                | c[..., b + 1]
            )
            p = fa_lut[idx]
            x[..., b] = p & 0xF
            y[..., b] = (p >> 4) & 0xF
            c[..., b] = (p >> 8) & 0xF
            z[..., b] = (p >> 12) & 0xF
            c[..., b + 1] = (p >> 16) & 0xF
        # Backward b = 31 -> 0
        for b in range(31, -1, -1):
            idx = (
                (x[..., b] << 16)
                | (y[..., b] << 12)
                | (c[..., b] << 8)
                | (z[..., b] << 4)
                | c[..., b + 1]
            )
            p = fa_lut[idx]
            x[..., b] = p & 0xF
            y[..., b] = (p >> 4) & 0xF
            c[..., b] = (p >> 8) & 0xF
            z[..., b] = (p >> 12) & 0xF
            c[..., b + 1] = (p >> 16) & 0xF

    return x, y, z, c


def propagate_rotr_xor3_op(
    x: torch.Tensor,
    z: torch.Tensor,
    shifts: Tuple[int, int, int],
    is_shr_last: bool = False,
    sweeps: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bidirectional arc-consistency for Sigma0/Sigma1/sigma0/sigma1 (XOR3 of rotated/shifted x)."""
    xor3_lut = get_luts(x.device)["xor3"]
    x = x.clone()
    z = z.clone()
    s0, s1, s2 = shifts
    for _ in range(sweeps):
        u = torch.roll(x, shifts=-s0, dims=-1)
        v = torch.roll(x, shifts=-s1, dims=-1)
        if is_shr_last:
            w = torch.full_like(x, GD_0)
            w[..., : 32 - s2] = x[..., s2:]
        else:
            w = torch.roll(x, shifts=-s2, dims=-1)

        nu, nv, nw, nz = propagate_3in_1out(xor3_lut, u, v, w, z)
        z &= nz
        # Intersect tightened projections back into x
        x &= torch.roll(nu, shifts=s0, dims=-1)
        x &= torch.roll(nv, shifts=s1, dims=-1)
        if is_shr_last:
            x[..., s2:] &= nw[..., : 32 - s2]
        else:
            x &= torch.roll(nw, shifts=s2, dims=-1)
    return x, z


# ---------------------------------------------------------------------------
# Full R-round SHA-256 atomic Mendel graph + bidirectional fixed-point solver
# ---------------------------------------------------------------------------

@dataclass
class MendelGraphState:
    """Full atomic generalized differential state for R-round SHA-256."""

    rounds: int
    iv: torch.Tensor       # (B, 8, 32)
    W: torch.Tensor        # (B, R, 32)
    A: torch.Tensor        # (B, R + 4, 32)  index k+3 for k=-3..R
    E: torch.Tensor        # (B, R + 4, 32)
    digest: torch.Tensor   # (B, 8, 32)
    contradiction: torch.Tensor  # (B,) bool


def create_blank_mendel_state(
    batch_size: int,
    rounds: int,
    iv_mode: str = "standard",
    device: torch.device | str = "cpu",
    btc_block2_padding: bool = False,
) -> MendelGraphState:
    """Create initial Mendel state tensor for R-round SHA-256."""
    if iv_mode == "standard":
        iv_base = torch.stack([const_word_to_gdiff(v, device) for v in SHA256_IV], dim=0)
        iv = iv_base.unsqueeze(0).expand(batch_size, 8, 32).clone()
    elif iv_mode == "semi_free":
        iv = torch.full((batch_size, 8, 32), GD_EQ, dtype=torch.int64, device=device)
    elif iv_mode == "free":
        iv = torch.full((batch_size, 8, 32), GD_ANY, dtype=torch.int64, device=device)
    else:
        raise ValueError(f"Unsupported iv_mode: {iv_mode}")

    W = torch.full((batch_size, rounds, 32), GD_ANY, dtype=torch.int64, device=device)
    if btc_block2_padding:
        if rounds > 4:
            W[:, 4, :] = const_word_to_gdiff(0x80000000, device)
        if rounds > 5:
            W[:, 5 : min(rounds, 15), :] = GD_0
        if rounds > 15:
            W[:, 15, :] = const_word_to_gdiff(640, device)

    A = torch.full((batch_size, rounds + 4, 32), GD_ANY, dtype=torch.int64, device=device)
    E = torch.full((batch_size, rounds + 4, 32), GD_ANY, dtype=torch.int64, device=device)
    # Sync initial A_{-3..0}, E_{-3..0} with IV
    A[:, 3, :] = iv[:, 0, :]
    A[:, 2, :] = iv[:, 1, :]
    A[:, 1, :] = iv[:, 2, :]
    A[:, 0, :] = iv[:, 3, :]
    E[:, 3, :] = iv[:, 4, :]
    E[:, 2, :] = iv[:, 5, :]
    E[:, 1, :] = iv[:, 6, :]
    E[:, 0, :] = iv[:, 7, :]

    digest = torch.full((batch_size, 8, 32), GD_ANY, dtype=torch.int64, device=device)
    contradiction = torch.zeros(batch_size, dtype=torch.bool, device=device)
    return MendelGraphState(
        rounds=rounds, iv=iv, W=W, A=A, E=E, digest=digest, contradiction=contradiction
    )


def propagate_mendel_fixpoint(
    state: MendelGraphState,
    max_iters: int = 6,
    custom_k: Optional[Sequence[int]] = None,
    sigma0_shifts: Tuple[int, int, int] = (2, 13, 22),
    sigma1_shifts: Tuple[int, int, int] = (6, 11, 25),
) -> MendelGraphState:
    """Run bidirectional forward/backward Mendel fixed-point propagation (SPEC 2.2).

    Decomposes every multi-operand modular addition into atomic 2-operand ADD32 nodes
    and runs alternating forward & backward sweeps until convergence or max_iters.
    """
    R = state.rounds
    B = state.W.shape[0]
    device = state.W.device
    luts = get_luts(device)
    ch_lut = luts["ch"]
    maj_lut = luts["maj"]

    k_list = custom_k if custom_k is not None else SHA256_K
    k_masks = torch.stack(
        [const_word_to_gdiff(k_list[t], device) for t in range(R)], dim=0
    ).unsqueeze(0).expand(B, R, 32)

    iv = state.iv.clone()
    W = state.W.clone()
    A = state.A.clone()
    E = state.E.clone()
    digest = state.digest.clone()

    # Sync IV <-> A_0..-3, E_0..-3
    A[:, 3, :] &= iv[:, 0, :]; iv[:, 0, :] = A[:, 3, :]
    A[:, 2, :] &= iv[:, 1, :]; iv[:, 1, :] = A[:, 2, :]
    A[:, 1, :] &= iv[:, 2, :]; iv[:, 2, :] = A[:, 1, :]
    A[:, 0, :] &= iv[:, 3, :]; iv[:, 3, :] = A[:, 0, :]
    E[:, 3, :] &= iv[:, 4, :]; iv[:, 4, :] = E[:, 3, :]
    E[:, 2, :] &= iv[:, 5, :]; iv[:, 5, :] = E[:, 2, :]
    E[:, 1, :] &= iv[:, 6, :]; iv[:, 6, :] = E[:, 1, :]
    E[:, 0, :] &= iv[:, 7, :]; iv[:, 7, :] = E[:, 0, :]

    # Persistent intermediate nodes across fixed-point iterations
    s1_e = torch.full((B, R, 32), GD_ANY, dtype=torch.int64, device=device)
    ch_e = torch.full((B, R, 32), GD_ANY, dtype=torch.int64, device=device)
    p1 = torch.full((B, R, 32), GD_ANY, dtype=torch.int64, device=device)
    p2 = torch.full((B, R, 32), GD_ANY, dtype=torch.int64, device=device)
    p3 = torch.full((B, R, 32), GD_ANY, dtype=torch.int64, device=device)
    T1 = torch.full((B, R, 32), GD_ANY, dtype=torch.int64, device=device)
    s0_a = torch.full((B, R, 32), GD_ANY, dtype=torch.int64, device=device)
    maj_a = torch.full((B, R, 32), GD_ANY, dtype=torch.int64, device=device)
    T2 = torch.full((B, R, 32), GD_ANY, dtype=torch.int64, device=device)

    n_exp = max(0, R - 16)
    s1_w = torch.full((B, n_exp, 32), GD_ANY, dtype=torch.int64, device=device)
    s0_w = torch.full((B, n_exp, 32), GD_ANY, dtype=torch.int64, device=device)
    q1 = torch.full((B, n_exp, 32), GD_ANY, dtype=torch.int64, device=device)
    q2 = torch.full((B, n_exp, 32), GD_ANY, dtype=torch.int64, device=device)

    def _sched_step(t: int) -> None:
        i = t - 16
        W[:, t - 2], s1_w[:, i] = propagate_rotr_xor3_op(W[:, t - 2], s1_w[:, i], (17, 19, 10), is_shr_last=True, sweeps=1)
        W[:, t - 15], s0_w[:, i] = propagate_rotr_xor3_op(W[:, t - 15], s0_w[:, i], (7, 18, 3), is_shr_last=True, sweeps=1)
        s1_w[:, i], W[:, t - 7], q1[:, i], _ = propagate_add32_bidir(s1_w[:, i], W[:, t - 7], q1[:, i], sweeps=1)
        q1[:, i], s0_w[:, i], q2[:, i], _ = propagate_add32_bidir(q1[:, i], s0_w[:, i], q2[:, i], sweeps=1)
        q2[:, i], W[:, t - 16], W[:, t], _ = propagate_add32_bidir(q2[:, i], W[:, t - 16], W[:, t], sweeps=1)

    def _round_step(t: int) -> None:
        e_t, e_tm1, e_tm2, e_tm3 = E[:, t + 3], E[:, t + 2], E[:, t + 1], E[:, t]
        a_t, a_tm1, a_tm2, a_tm3 = A[:, t + 3], A[:, t + 2], A[:, t + 1], A[:, t]

        e_t, s1_e[:, t] = propagate_rotr_xor3_op(e_t, s1_e[:, t], sigma1_shifts, is_shr_last=False, sweeps=1)
        ne, nef, neg, nch = propagate_3in_1out(ch_lut, e_t, e_tm1, e_tm2, ch_e[:, t])
        e_t &= ne; e_tm1 &= nef; e_tm2 &= neg; ch_e[:, t] &= nch

        e_tm3, s1_e[:, t], p1[:, t], _ = propagate_add32_bidir(e_tm3, s1_e[:, t], p1[:, t], sweeps=1)
        p1[:, t], ch_e[:, t], p2[:, t], _ = propagate_add32_bidir(p1[:, t], ch_e[:, t], p2[:, t], sweeps=1)
        p2[:, t], _, p3[:, t], _ = propagate_add32_bidir(p2[:, t], k_masks[:, t], p3[:, t], sweeps=1)
        p3[:, t], W[:, t], T1[:, t], _ = propagate_add32_bidir(p3[:, t], W[:, t], T1[:, t], sweeps=1)

        a_t, s0_a[:, t] = propagate_rotr_xor3_op(a_t, s0_a[:, t], sigma0_shifts, is_shr_last=False, sweeps=1)
        na, nab, nac, nmaj = propagate_3in_1out(maj_lut, a_t, a_tm1, a_tm2, maj_a[:, t])
        a_t &= na; a_tm1 &= nab; a_tm2 &= nac; maj_a[:, t] &= nmaj
        s0_a[:, t], maj_a[:, t], T2[:, t], _ = propagate_add32_bidir(s0_a[:, t], maj_a[:, t], T2[:, t], sweeps=1)

        a_tm3, T1[:, t], E[:, t + 4], _ = propagate_add32_bidir(a_tm3, T1[:, t], E[:, t + 4], sweeps=1)
        T1[:, t], T2[:, t], A[:, t + 4], _ = propagate_add32_bidir(T1[:, t], T2[:, t], A[:, t + 4], sweeps=1)

        E[:, t + 3] = e_t; E[:, t + 2] = e_tm1; E[:, t + 1] = e_tm2; E[:, t] = e_tm3
        A[:, t + 3] = a_t; A[:, t + 2] = a_tm1; A[:, t + 1] = a_tm2; A[:, t] = a_tm3

    def _feedforward_step() -> None:
        s_map = [
            (A, R + 3, 0), (A, R + 2, 1), (A, R + 1, 2), (A, R, 3),
            (E, R + 3, 4), (E, R + 2, 5), (E, R + 1, 6), (E, R, 7),
        ]
        for arr, idx, j in s_map:
            arr[:, idx], iv[:, j], digest[:, j], _ = propagate_add32_bidir(
                arr[:, idx], iv[:, j], digest[:, j], sweeps=1
            )

    for _ in range(max_iters):
        prev_hash = (W.sum() + A.sum() + E.sum() + digest.sum()).item()
        # Forward sweep
        for t in range(16, R):
            _sched_step(t)
        for t in range(R):
            _round_step(t)
        _feedforward_step()
        # Backward sweep
        _feedforward_step()
        for t in range(R - 1, -1, -1):
            _round_step(t)
        for t in range(R - 1, 15, -1):
            _sched_step(t)

        # Sync IV <-> A_0..-3, E_0..-3
        A[:, 3, :] &= iv[:, 0, :]; iv[:, 0, :] = A[:, 3, :]
        A[:, 2, :] &= iv[:, 1, :]; iv[:, 1, :] = A[:, 2, :]
        A[:, 1, :] &= iv[:, 2, :]; iv[:, 2, :] = A[:, 1, :]
        A[:, 0, :] &= iv[:, 3, :]; iv[:, 3, :] = A[:, 0, :]
        E[:, 3, :] &= iv[:, 4, :]; iv[:, 4, :] = E[:, 3, :]
        E[:, 2, :] &= iv[:, 5, :]; iv[:, 5, :] = E[:, 2, :]
        E[:, 1, :] &= iv[:, 6, :]; iv[:, 6, :] = E[:, 1, :]
        E[:, 0, :] &= iv[:, 7, :]; iv[:, 7, :] = E[:, 0, :]

        cur_hash = (W.sum() + A.sum() + E.sum() + digest.sum()).item()
        contradiction = (
            (iv == 0).any(dim=(-2, -1))
            | (W == 0).any(dim=(-2, -1))
            | (A == 0).any(dim=(-2, -1))
            | (E == 0).any(dim=(-2, -1))
            | (digest == 0).any(dim=(-2, -1))
        )
        if cur_hash == prev_hash:
            break

    return MendelGraphState(
        rounds=R, iv=iv, W=W, A=A, E=E, digest=digest, contradiction=contradiction
    )


def build_mendel_dv_skeleton(
    step_t: int,
    bit_j: int,
    rounds: int,
    iv_mode: str = "semi_free",
    btc_block2_padding: bool = False,
) -> MendelGraphState:
    """Create a Mendel state seeded with a localized Chabaud-Joux disturbance window."""
    st = create_blank_mendel_state(1, rounds=rounds, iv_mode=iv_mode, btc_block2_padding=btc_block2_padding)
    # Lock W outside active window [step_t .. min(R-1, step_t+8)] to '-'
    for t in range(rounds):
        if t < step_t or t > step_t + 8:
            if not (btc_block2_padding and 4 <= t <= 15):
                st.W[:, t, :] = GD_EQ
    # Inject 1-bit disturbance at W[step_t][bit_j]
    if 0 <= step_t < rounds:
        st.W[:, step_t, :] = GD_EQ
        st.W[:, step_t, bit_j] = GD_X
    # Lock state A, E outside disturbance window to '-'
    for k in range(-3, rounds + 1):
        if k <= step_t or k >= step_t + 9:
            st.A[:, k + 3, :] = GD_EQ
            st.E[:, k + 3, :] = GD_EQ
    return propagate_mendel_fixpoint(st, max_iters=6)


# ---------------------------------------------------------------------------
# B1-1 Mandatory >= 1,000,000 randomized soundness & exactness self-test
# ---------------------------------------------------------------------------

def run_mendel_million_selftest(
    n_soundness: int = 1_000_000,
    n_exactness: int = 100_000,
    batch_size: int = 50_000,
    seed: int = 20261125,
) -> Dict[str, object]:
    """Run B1-1 SPEC 2.4 self-test on CPU against src/sha256_ref.py."""
    t0 = time.perf_counter()
    gen = torch.Generator().manual_seed(seed)
    luts = get_luts("cpu")

    soundness_violations = 0
    # 1. Soundness across N = 1,000,000 random (x, x', y, y', z, z') & superset masks
    n_batches = (n_soundness + batch_size - 1) // batch_size
    checked = 0
    for b_idx in range(n_batches):
        bs = min(batch_size, n_soundness - checked)
        a = torch.randint(0, 1 << 32, (bs,), dtype=torch.int64, generator=gen)
        ap = torch.randint(0, 1 << 32, (bs,), dtype=torch.int64, generator=gen)
        b = torch.randint(0, 1 << 32, (bs,), dtype=torch.int64, generator=gen)
        bp = torch.randint(0, 1 << 32, (bs,), dtype=torch.int64, generator=gen)
        c = torch.randint(0, 1 << 32, (bs,), dtype=torch.int64, generator=gen)
        cp = torch.randint(0, 1 << 32, (bs,), dtype=torch.int64, generator=gen)

        # Concrete 1-element masks OR-ed with random relaxation noise
        ma = encode_pair_to_gdiff(a, ap) | torch.randint(0, 16, (bs, 32), dtype=torch.int64, generator=gen)
        mb = encode_pair_to_gdiff(b, bp) | torch.randint(0, 16, (bs, 32), dtype=torch.int64, generator=gen)
        mc = encode_pair_to_gdiff(c, cp) | torch.randint(0, 16, (bs, 32), dtype=torch.int64, generator=gen)

        # Include explicit structured '-' / 'x' / 'u' / 'n' bits on half the batch
        eq_sel = torch.randint(0, 4, (bs, 32), dtype=torch.int64, generator=gen) == 0
        ap = torch.where(eq_sel.any(dim=-1), a ^ ((ap ^ a) & ~((eq_sel.to(torch.int64) << torch.arange(32)).sum(dim=-1))), ap)
        ma = encode_pair_to_gdiff(a, ap) | (torch.randint(0, 2, (bs, 32), dtype=torch.int64, generator=gen) * GD_EQ)

        # Ch, Maj, XOR3 forward with output = '?'
        q_any = torch.full_like(ma, GD_ANY)
        _, _, _, out_ch = propagate_3in_1out(luts["ch"], ma, mb, mc, q_any)
        _, _, _, out_maj = propagate_3in_1out(luts["maj"], ma, mb, mc, q_any)
        _, _, _, out_xor = propagate_3in_1out(luts["xor3"], ma, mb, mc, q_any)

        # Sigma0 / Sigma1 / sigma0 / sigma1
        _, out_s0u = propagate_rotr_xor3_op(ma, q_any, (2, 13, 22), is_shr_last=False, sweeps=1)
        _, out_s1u = propagate_rotr_xor3_op(ma, q_any, (6, 11, 25), is_shr_last=False, sweeps=1)
        _, out_s0l = propagate_rotr_xor3_op(ma, q_any, (7, 18, 3), is_shr_last=True, sweeps=1)
        _, out_s1l = propagate_rotr_xor3_op(ma, q_any, (17, 19, 10), is_shr_last=True, sweeps=1)

        # ADD32 bidirectional with constrained output superset
        add_ref = (a + b) & MASK32
        addp_ref = (ap + bp) & MASK32
        mz = encode_pair_to_gdiff(add_ref, addp_ref) | torch.randint(0, 16, (bs, 32), dtype=torch.int64, generator=gen)
        nx, ny, nz, _ = propagate_add32_bidir(ma, mb, mz, sweeps=2)

        ok = (
            verify_pair_against_gdiff(ch32(a, b, c), ch32(ap, bp, cp), out_ch)
            & verify_pair_against_gdiff(maj32(a, b, c), maj32(ap, bp, cp), out_maj)
            & verify_pair_against_gdiff(a ^ b ^ c, ap ^ bp ^ cp, out_xor)
            & verify_pair_against_gdiff(sigma0_upper(a), sigma0_upper(ap), out_s0u)
            & verify_pair_against_gdiff(sigma1_upper(a), sigma1_upper(ap), out_s1u)
            & verify_pair_against_gdiff(sigma0_lower(a), sigma0_lower(ap), out_s0l)
            & verify_pair_against_gdiff(sigma1_lower(a), sigma1_lower(ap), out_s1l)
            & verify_pair_against_gdiff(a, ap, nx)
            & verify_pair_against_gdiff(b, bp, ny)
            & verify_pair_against_gdiff(add_ref, addp_ref, nz)
        )
        soundness_violations += int((~ok).sum().item())
        checked += bs
        if (b_idx + 1) % 5 == 0 or checked == n_soundness:
            print(f"[selftest] soundness checked {checked}/{n_soundness}, violations={soundness_violations}", flush=True)

    # 2. Exactness check on N = 100,000 random 4-bit partial constraints for Ch / Maj / XOR3 / FA
    exactness_mismatches = 0
    # Precompute 64 concrete assignments for brute-force projection check
    k_np = np.arange(64, dtype=np.int64)
    u_k = k_np & 1
    up_k = (k_np >> 1) & 1
    v_k = (k_np >> 2) & 1
    vp_k = (k_np >> 3) & 1
    w_k = (k_np >> 4) & 1
    wp_k = (k_np >> 5) & 1
    iu = torch.tensor(u_k + 2 * up_k, dtype=torch.int64)
    iv_t = torch.tensor(v_k + 2 * vp_k, dtype=torch.int64)
    iw = torch.tensor(w_k + 2 * wp_k, dtype=torch.int64)
    iz_ch = torch.tensor(((u_k & v_k) ^ ((1 ^ u_k) & w_k)) + 2 * ((up_k & vp_k) ^ ((1 ^ up_k) & wp_k)), dtype=torch.int64)
    iz_maj = torch.tensor(((u_k & v_k) ^ (u_k & w_k) ^ (v_k & w_k)) + 2 * ((up_k & vp_k) ^ (up_k & wp_k) ^ (vp_k & wp_k)), dtype=torch.int64)
    ic_fa = torch.tensor(((u_k & v_k) | (u_k & w_k) | (v_k & w_k)) + 2 * ((up_k & vp_k) | (up_k & wp_k) | (vp_k & wp_k)), dtype=torch.int64)
    iz_xor = torch.tensor((u_k ^ v_k ^ w_k) + 2 * (up_k ^ vp_k ^ wp_k), dtype=torch.int64)

    mu = torch.randint(1, 16, (n_exactness,), dtype=torch.int64, generator=gen)
    mv = torch.randint(1, 16, (n_exactness,), dtype=torch.int64, generator=gen)
    mw = torch.randint(1, 16, (n_exactness,), dtype=torch.int64, generator=gen)
    mz = torch.randint(1, 16, (n_exactness,), dtype=torch.int64, generator=gen)
    mc = torch.randint(1, 16, (n_exactness,), dtype=torch.int64, generator=gen)

    # Check in chunks of 25,000
    for start in range(0, n_exactness, 25_000):
        end = min(n_exactness, start + 25_000)
        cu, cv, cw, cz, cc = mu[start:end], mv[start:end], mw[start:end], mz[start:end], mc[start:end]

        for name, iz_t, lut_key in (("ch", iz_ch, "ch"), ("maj", iz_maj, "maj"), ("xor3", iz_xor, "xor3")):
            valid_k = (
                (((cu.unsqueeze(1) >> iu.unsqueeze(0)) & 1) != 0)
                & (((cv.unsqueeze(1) >> iv_t.unsqueeze(0)) & 1) != 0)
                & (((cw.unsqueeze(1) >> iw.unsqueeze(0)) & 1) != 0)
                & (((cz.unsqueeze(1) >> iz_t.unsqueeze(0)) & 1) != 0)
            )
            bf_u = torch.zeros_like(cu)
            bf_v = torch.zeros_like(cv)
            bf_w = torch.zeros_like(cw)
            bf_z = torch.zeros_like(cz)
            for p in range(4):
                bf_u |= ((valid_k & (iu == p).unsqueeze(0)).any(dim=1).to(torch.int64)) << p
                bf_v |= ((valid_k & (iv_t == p).unsqueeze(0)).any(dim=1).to(torch.int64)) << p
                bf_w |= ((valid_k & (iw == p).unsqueeze(0)).any(dim=1).to(torch.int64)) << p
                bf_z |= ((valid_k & (iz_t == p).unsqueeze(0)).any(dim=1).to(torch.int64)) << p
            lu, lv, lw, lz = propagate_3in_1out(luts[lut_key], cu, cv, cw, cz)
            exactness_mismatches += int(((bf_u != lu) | (bf_v != lv) | (bf_w != lw) | (bf_z != lz)).sum().item())

        # Exactness of 5-way FullAdder
        valid_fa = (
            (((cu.unsqueeze(1) >> iu.unsqueeze(0)) & 1) != 0)
            & (((cv.unsqueeze(1) >> iv_t.unsqueeze(0)) & 1) != 0)
            & (((cw.unsqueeze(1) >> iw.unsqueeze(0)) & 1) != 0)
            & (((cz.unsqueeze(1) >> iz_xor.unsqueeze(0)) & 1) != 0)
            & (((cc.unsqueeze(1) >> ic_fa.unsqueeze(0)) & 1) != 0)
        )
        bf_u = torch.zeros_like(cu); bf_v = torch.zeros_like(cv); bf_w = torch.zeros_like(cw)
        bf_z = torch.zeros_like(cz); bf_c = torch.zeros_like(cc)
        for p in range(4):
            bf_u |= ((valid_fa & (iu == p).unsqueeze(0)).any(dim=1).to(torch.int64)) << p
            bf_v |= ((valid_fa & (iv_t == p).unsqueeze(0)).any(dim=1).to(torch.int64)) << p
            bf_w |= ((valid_fa & (iw == p).unsqueeze(0)).any(dim=1).to(torch.int64)) << p
            bf_z |= ((valid_fa & (iz_xor == p).unsqueeze(0)).any(dim=1).to(torch.int64)) << p
            bf_c |= ((valid_fa & (ic_fa == p).unsqueeze(0)).any(dim=1).to(torch.int64)) << p
        idx5 = (cu << 16) | (cv << 12) | (cw << 8) | (cz << 4) | cc
        pk = luts["fa"][idx5]
        lu, lv, lw, lz, lc = pk & 0xF, (pk >> 4) & 0xF, (pk >> 8) & 0xF, (pk >> 12) & 0xF, (pk >> 16) & 0xF
        exactness_mismatches += int(
            ((bf_u != lu) | (bf_v != lv) | (bf_w != lw) | (bf_z != lz) | (bf_c != lc)).sum().item()
        )

    # 3. Full R=64 end-to-end forward/backward containment test on 256 random pairs
    w1 = torch.randint(0, 1 << 32, (256, 16), dtype=torch.int64, generator=gen)
    w2 = w1.clone()
    w2[:, 0] ^= 0x80000001
    w2[:, 3] ^= 0x00000010
    for rounds in (8, 16, 32, 64):
        tr1 = sha256_trace(w1, rounds=rounds)
        tr2 = sha256_trace(w2, rounds=rounds)
        st = create_blank_mendel_state(256, rounds=rounds, iv_mode="standard", device="cpu")
        n_w = min(rounds, 16)
        st.W[:, :n_w, :] = encode_pair_to_gdiff(w1[:, :n_w], w2[:, :n_w])
        # Relax W[1] bits 0..7 to '?' and constrain digest[0] bits 24..31 to exact observed gdiff
        st.W[:, 1, :8] = GD_ANY
        st.digest[:, 0, 24:] = encode_pair_to_gdiff(tr1.digest[:, 0], tr2.digest[:, 0])[:, 24:]
        res = propagate_mendel_fixpoint(st, max_iters=4)
        ok_end = (
            (~res.contradiction).all()
            & verify_pair_against_gdiff(tr1.digest, tr2.digest, res.digest).all()
            & verify_pair_against_gdiff(tr1.A_full, tr2.A_full, res.A).all()
            & verify_pair_against_gdiff(tr1.E_full, tr2.E_full, res.E).all()
        )
        if not bool(ok_end):
            soundness_violations += 1

    elapsed = time.perf_counter() - t0
    report = {
        "n_soundness": n_soundness,
        "n_exactness": n_exactness,
        "soundness_violations": soundness_violations,
        "exactness_mismatches": exactness_mismatches,
        "fp32_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "elapsed_seconds": round(elapsed, 3),
        "passed": (soundness_violations == 0 and exactness_mismatches == 0),
    }
    print(f"[selftest] complete: {json.dumps(report)}", flush=True)
    return report


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    res = run_mendel_million_selftest()
    out_dir = _THIS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mendel_selftest.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    if not res["passed"]:
        sys.exit(1)
