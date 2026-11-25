"""Pure FP32 vectorized ARX primitives, SHA-256 single-round step (with control), and human analytic bases."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from src.sha256_ref import MASK32, SHA256_IV, SHA256_K


def enforce_fp32_lock() -> Dict[str, bool]:
    """Enforce strict IEEE-754 FP32 with TF32 disabled for both matmul and cuDNN."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }


def mask_n(n_bits: int) -> int:
    return (1 << int(n_bits)) - 1


def rotr_amount_for_bits(n_bits: int) -> int:
    """Scale rotation offset proportionally (~7/32, matching sigma0's ROTR 7 on 32-bit)."""
    r = max(1, (7 * int(n_bits)) // 32)
    return r % int(n_bits) if int(n_bits) > 1 else 0


def op_xor(x: torch.Tensor, y: torch.Tensor, n_bits: int) -> torch.Tensor:
    m = mask_n(n_bits)
    return (x ^ y) & m


def op_add(x: torch.Tensor, y: torch.Tensor, n_bits: int) -> torch.Tensor:
    m = mask_n(n_bits)
    return (x + y) & m


def op_rotr(x: torch.Tensor, n_bits: int, r: int | None = None) -> torch.Tensor:
    m = mask_n(n_bits)
    rot = rotr_amount_for_bits(n_bits) if r is None else (int(r) % int(n_bits))
    xu = x & m
    if rot == 0:
        return xu
    return ((xu >> rot) | ((xu & ((1 << rot) - 1)) << (int(n_bits) - rot))) & m


def int_to_bits(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Convert (...) int64 tensor to (..., n_bits) LSB-first float32 bits in {0.0, 1.0}."""
    shifts = torch.arange(int(n_bits), dtype=torch.int64, device=x.device)
    return ((x.unsqueeze(-1) >> shifts) & 1).to(torch.float32)


def bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    """Convert (..., n_bits) {0,1} tensor to (...) int64."""
    n_bits = bits.shape[-1]
    shifts = torch.arange(n_bits, dtype=torch.int64, device=bits.device)
    b = (bits > 0.5).to(torch.int64)
    return (b << shifts).sum(dim=-1) & mask_n(n_bits)


def state_words_to_bits256(s8: torch.Tensor) -> torch.Tensor:
    """Convert (B, 8) uint32-in-int64 SHA-256 state to (B, 256) float32 LSB-first bits."""
    shifts = torch.arange(32, dtype=torch.int64, device=s8.device)
    bits = ((s8.unsqueeze(-1) >> shifts) & 1).to(torch.float32)
    return bits.reshape(s8.shape[0], 256)


def bits256_to_state_words(b256: torch.Tensor) -> torch.Tensor:
    """Convert (B, 256) bit tensor to (B, 8) uint32-in-int64 SHA-256 state."""
    b = b256.reshape(b256.shape[0], 8, 32)
    return bits_to_int(b)


def _rotr32_vec(x: torch.Tensor, r: int) -> torch.Tensor:
    xu = x & MASK32
    if r % 32 == 0:
        return xu
    rot = r % 32
    return ((xu >> rot) | ((xu & ((1 << rot) - 1)) << (32 - rot))) & MASK32


def make_cipher_profile(ctrl: bool = False, seed: int = 20261125) -> Dict[str, Tuple[int, ...]]:
    """Standard SHA-256 profile vs isomorphic random rotation/constant clone."""
    if not ctrl:
        return {
            "s0": (2, 13, 22),
            "s1": (6, 11, 25),
            "k": tuple(SHA256_K),
        }
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    r0 = torch.randperm(31, generator=g)[:3] + 1
    r1 = torch.randperm(31, generator=g)[:3] + 1
    k_ctrl = torch.randint(0, 1 << 32, (64,), dtype=torch.int64, generator=g).tolist()
    return {
        "s0": tuple(int(v) for v in r0.tolist()),
        "s1": tuple(int(v) for v in r1.tolist()),
        "k": tuple(int(v) for v in k_ctrl),
    }


def sha256_round_step(
    s8: torch.Tensor,
    w_t: torch.Tensor,
    k_t: int | torch.Tensor,
    s0_rots: Tuple[int, int, int] = (2, 13, 22),
    s1_rots: Tuple[int, int, int] = (6, 11, 25),
) -> torch.Tensor:
    """Exact 1-round SHA-256 state transition: (B, 8) -> (B, 8)."""
    a, b, c, d, e, f, g, h = [s8[:, i] & MASK32 for i in range(8)]
    sig1 = _rotr32_vec(e, s1_rots[0]) ^ _rotr32_vec(e, s1_rots[1]) ^ _rotr32_vec(e, s1_rots[2])
    ch = ((e & f) ^ ((~e & MASK32) & g)) & MASK32
    sig0 = _rotr32_vec(a, s0_rots[0]) ^ _rotr32_vec(a, s0_rots[1]) ^ _rotr32_vec(a, s0_rots[2])
    maj = ((a & b) ^ (a & c) ^ (b & c)) & MASK32

    k_val = k_t if isinstance(k_t, torch.Tensor) else int(k_t)
    t1 = (h + sig1 + ch + k_val + (w_t & MASK32)) & MASK32
    t2 = (sig0 + maj) & MASK32
    new_e = (d + t1) & MASK32
    new_a = (t1 + t2) & MASK32
    return torch.stack([new_a, a, b, c, new_e, e, f, g], dim=1)


def sha256_k_rounds_from_state(
    s8: torch.Tensor,
    w_seq: torch.Tensor,
    k_tuple: Tuple[int, ...],
    start_round: int = 0,
    num_rounds: int = 1,
    s0_rots: Tuple[int, int, int] = (2, 13, 22),
    s1_rots: Tuple[int, int, int] = (6, 11, 25),
) -> torch.Tensor:
    """Roll SHA-256 state forward for num_rounds starting at start_round. Returns (B, num_rounds+1, 8)."""
    traj = [s8 & MASK32]
    cur = s8 & MASK32
    for step in range(num_rounds):
        r_idx = (start_round + step) % 64
        cur = sha256_round_step(
            cur,
            w_seq[:, step],
            k_tuple[r_idx],
            s0_rots=s0_rots,
            s1_rots=s1_rots,
        )
        traj.append(cur)
    return torch.stack(traj, dim=1)


def encode_human_hybrid_basis(
    x: torch.Tensor,
    n_bits: int,
    dim_out: int,
    proj_mat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Human named coordinate system: Walsh bipolar spins + 2-adic phase tower (cos/sin) + fixed expansion."""
    bits = int_to_bits(x, n_bits)  # (B, n)
    spins = 1.0 - 2.0 * bits       # (B, n), exact for XOR & ROTR

    # 2-adic circle tower phases: exp(2*pi*i * x / 2^m), m = 1..n_bits -> 2*n dims, exact for ADD
    xf = (x & mask_n(n_bits)).to(torch.float64)
    ang_list = []
    for m in range(1, n_bits + 1):
        mod_val = float(1 << m)
        theta = (2.0 * math.pi) * torch.remainder(xf, mod_val) / mod_val
        ang_list.append(torch.cos(theta).to(torch.float32))
        ang_list.append(torch.sin(theta).to(torch.float32))
    two_adic = torch.stack(ang_list, dim=-1)  # (B, 2*n)

    # Neighbor XOR & carry-free bit pairs (natural human quadratic features)
    pair_xor = spins * torch.roll(spins, shifts=-1, dims=-1)  # (B, n)
    base = torch.cat([spins, two_adic, pair_xor], dim=-1)      # (B, 4*n)

    if base.shape[-1] >= dim_out:
        return base[:, :dim_out]
    if proj_mat is None:
        raise ValueError("proj_mat required when dim_out > 4*n_bits")
    extra = torch.cos(base @ proj_mat)  # Random Kitchen Sinks / Fourier expansion of human hybrid basis
    return torch.cat([base, extra], dim=-1)
