"""GPU-vectorized int64 (& 0xFFFFFFFF) reference SHA-256 and trace generator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch

MASK32: int = 0xFFFFFFFF


def _first_n_primes(n: int) -> Tuple[int, ...]:
    primes = []
    candidate = 2
    while len(primes) < n:
        limit = math.isqrt(candidate)
        if all(candidate % p != 0 for p in primes if p <= limit):
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


def _icbrt(n: int) -> int:
    """Exact integer floor cube root floor(n ** (1/3)) for non-negative int."""
    if n <= 0:
        return 0
    x = 1 << ((n.bit_length() + 2) // 3)
    while True:
        y = (2 * x + n // (x * x)) // 3
        if y >= x:
            return x
        x = y


def _derive_sha256_constants() -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    primes = _first_n_primes(64)
    iv = tuple((math.isqrt(p << 64)) & MASK32 for p in primes[:8])
    k = tuple((_icbrt(p << 96)) & MASK32 for p in primes)
    return iv, k


SHA256_IV, SHA256_K = _derive_sha256_constants()


def configure_cuda_memory(fraction: float = 0.80, device: int = 0) -> None:
    """Cap per-process CUDA memory to prevent silent Windows shared GPU spillover."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, device=device)


def get_iv_tensor(device: torch.device | str | None = None) -> torch.Tensor:
    """Return standard 8-word SHA-256 IV as int64 tensor on target device."""
    return torch.tensor(SHA256_IV, dtype=torch.int64, device=device)


def get_k_tensor(
    rounds: int = 64, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return first `rounds` SHA-256 round constants as int64 tensor."""
    if not (1 <= rounds <= 64):
        raise ValueError(f"rounds must be in [1, 64], got {rounds}")
    return torch.tensor(SHA256_K[:rounds], dtype=torch.int64, device=device)


def rotr32(x: torch.Tensor, n: int) -> torch.Tensor:
    """Exact 32-bit circular right rotation on int64 tensors."""
    if n == 0:
        return x & MASK32
    xu = x & MASK32
    return ((xu >> n) | ((xu & ((1 << n) - 1)) << (32 - n))) & MASK32


def shr32(x: torch.Tensor, n: int) -> torch.Tensor:
    """Exact 32-bit logical right shift on int64 tensors."""
    return (x & MASK32) >> n


def ch32(e: torch.Tensor, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Exact 32-bit SHA-256 Ch(e, f, g) on int64 tensors."""
    eu = e & MASK32
    fu = f & MASK32
    gu = g & MASK32
    return ((eu & fu) ^ ((~eu & MASK32) & gu)) & MASK32


def maj32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Exact 32-bit SHA-256 Maj(a, b, c) on int64 tensors."""
    au = a & MASK32
    bu = b & MASK32
    cu = c & MASK32
    return ((au & bu) ^ (au & cu) ^ (bu & cu)) & MASK32


def sigma0_upper(x: torch.Tensor) -> torch.Tensor:
    """SHA-256 Sigma0(x) = ROTR(x,2) ^ ROTR(x,13) ^ ROTR(x,22)."""
    return (rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22)) & MASK32


def sigma1_upper(x: torch.Tensor) -> torch.Tensor:
    """SHA-256 Sigma1(x) = ROTR(x,6) ^ ROTR(x,11) ^ ROTR(x,25)."""
    return (rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25)) & MASK32


def sigma0_lower(x: torch.Tensor) -> torch.Tensor:
    """SHA-256 sigma0(x) = ROTR(x,7) ^ ROTR(x,18) ^ SHR(x,3)."""
    return (rotr32(x, 7) ^ rotr32(x, 18) ^ shr32(x, 3)) & MASK32


def sigma1_lower(x: torch.Tensor) -> torch.Tensor:
    """SHA-256 sigma1(x) = ROTR(x,17) ^ ROTR(x,19) ^ SHR(x,10)."""
    return (rotr32(x, 17) ^ rotr32(x, 19) ^ shr32(x, 10)) & MASK32


def words_to_bits(words: torch.Tensor) -> torch.Tensor:
    """Convert uint32-in-int64 words (...) to LSB-first bits (..., 32)."""
    shifts = torch.arange(32, dtype=torch.int64, device=words.device)
    return ((words.to(torch.int64).unsqueeze(-1) >> shifts) & 1).to(torch.int64)


def words_to_spins(
    words: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Convert words (...) to LSB-first bipolar spins (..., 32): 0 -> +1, 1 -> -1."""
    bits = words_to_bits(words).to(dtype)
    return 1.0 - 2.0 * bits


def bits_to_words(bits: torch.Tensor) -> torch.Tensor:
    """Convert LSB-first bits (..., 32) to uint32-in-int64 words (...)."""
    shifts = torch.arange(32, dtype=torch.int64, device=bits.device)
    return ((bits.to(torch.int64) << shifts).sum(dim=-1)) & MASK32


def spins_to_words(spins: torch.Tensor) -> torch.Tensor:
    """Threshold continuous spins (..., 32) via (x < 0) into uint32-in-int64 words."""
    bits = (spins < 0).to(torch.int64)
    return bits_to_words(bits)


def pad_bytes_to_w16(
    msg: bytes, device: torch.device | str | None = None
) -> torch.Tensor:
    """Pad a single-block message (0 <= len <= 55 bytes) into 16 big-endian words."""
    if len(msg) > 55:
        raise ValueError(f"Single-block SHA-256 supports <= 55 bytes, got {len(msg)}")
    bit_len = len(msg) * 8
    padded = bytearray(msg)
    padded.append(0x80)
    while len(padded) < 56:
        padded.append(0)
    padded.extend(bit_len.to_bytes(8, byteorder="big"))
    words = [int.from_bytes(padded[4 * i : 4 * (i + 1)], "big") for i in range(16)]
    return torch.tensor(words, dtype=torch.int64, device=device)


def pack_int_messages_to_w16(w0: torch.Tensor) -> torch.Tensor:
    """Pack batch (B,) of 32-bit int64 words into single-block padded (B, 16) blocks."""
    batch_size = w0.shape[0]
    w16 = torch.zeros((batch_size, 16), dtype=torch.int64, device=w0.device)
    w16[:, 0] = w0.to(torch.int64) & MASK32
    w16[:, 1] = 0x80000000
    w16[:, 15] = 32
    return w16


def sample_random_w16(
    batch_size: int,
    mode: str = "int_range",
    num_bits: int = 32,
    msg_bytes_len: int = 4,
    device: torch.device | str = "cuda",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate random padded single-block message words (B, 16) directly on device."""
    w16 = torch.zeros((batch_size, 16), dtype=torch.int64, device=device)
    if mode == "int_range":
        if not (1 <= num_bits <= 32):
            raise ValueError(f"num_bits must be in [1, 32], got {num_bits}")
        high = 1 << num_bits
        w16[:, 0] = torch.randint(
            0, high, (batch_size,), dtype=torch.int64, device=device, generator=generator
        )
        w16[:, 1] = 0x80000000
        w16[:, 15] = 32
        return w16

    if mode == "fixed_len_bytes":
        if not (0 <= msg_bytes_len <= 55):
            raise ValueError(
                f"msg_bytes_len must be in [0, 55], got {msg_bytes_len}"
            )
        rand_bytes = torch.randint(
            0,
            256,
            (batch_size, 64),
            dtype=torch.int64,
            device=device,
            generator=generator,
        )
        # Zero out padding region and write deterministic 0x80 and length
        if msg_bytes_len < 56:
            rand_bytes[:, msg_bytes_len] = 0x80
        if msg_bytes_len + 1 < 56:
            rand_bytes[:, msg_bytes_len + 1 : 56] = 0
        rand_bytes[:, 56:64] = 0
        bit_len = msg_bytes_len * 8
        for idx_byte in range(8):
            shift = 8 * (7 - idx_byte)
            rand_bytes[:, 56 + idx_byte] = (bit_len >> shift) & 0xFF
        bytes_view = rand_bytes.view(batch_size, 16, 4)
        w16 = (
            (bytes_view[:, :, 0] << 24)
            | (bytes_view[:, :, 1] << 16)
            | (bytes_view[:, :, 2] << 8)
            | bytes_view[:, :, 3]
        ) & MASK32
        return w16

    raise ValueError(f"Unsupported domain mode: {mode}")


def expand_message_schedule(w16: torch.Tensor, rounds: int) -> torch.Tensor:
    """Expand (B, 16) message block into (B, rounds) schedule words on device."""
    if not (1 <= rounds <= 64):
        raise ValueError(f"rounds must be in [1, 64], got {rounds}")
    if rounds <= 16:
        return (w16[:, :rounds] & MASK32).clone()

    batch_size = w16.shape[0]
    w_cols = [w16[:, t] & MASK32 for t in range(16)]
    for t in range(16, rounds):
        wt = (
            sigma1_lower(w_cols[t - 2])
            + w_cols[t - 7]
            + sigma0_lower(w_cols[t - 15])
            + w_cols[t - 16]
        ) & MASK32
        w_cols.append(wt)
    return torch.stack(w_cols, dim=1).reshape(batch_size, rounds)


@dataclass
class SHA256Trace:
    """Full two-track SHA-256 execution trace in uint32-in-int64 representation.

    Algebraic index k in [-3, R] maps to array index (k + 3).
    """

    A_full: torch.Tensor  # (B, R + 4)
    E_full: torch.Tensor  # (B, R + 4)
    W_full: torch.Tensor  # (B, R)
    S_R: torch.Tensor  # (B, 8)
    digest: torch.Tensor  # (B, 8)


def sha256_trace(w16: torch.Tensor, rounds: int = 64) -> SHA256Trace:
    """Run vectorized reduced- or full-round SHA-256 and record full (A, E, W) trace."""
    if w16.ndim != 2 or w16.shape[1] != 16:
        raise ValueError(f"Expected w16 shape (B, 16), got {tuple(w16.shape)}")
    batch_size = w16.shape[0]
    device = w16.device

    iv = get_iv_tensor(device)
    k_vec = get_k_tensor(rounds, device)
    w_full = expand_message_schedule(w16, rounds)

    # A_cols[k + 3] = A_k for k = -3 .. R
    a_cols = [
        iv[3].expand(batch_size),  # A_{-3} = d_0
        iv[2].expand(batch_size),  # A_{-2} = c_0
        iv[1].expand(batch_size),  # A_{-1} = b_0
        iv[0].expand(batch_size),  # A_0    = a_0
    ]
    e_cols = [
        iv[7].expand(batch_size),  # E_{-3} = h_0
        iv[6].expand(batch_size),  # E_{-2} = g_0
        iv[5].expand(batch_size),  # E_{-1} = f_0
        iv[4].expand(batch_size),  # E_0    = e_0
    ]

    for t in range(rounds):
        a_t = a_cols[t + 3]
        a_tm1 = a_cols[t + 2]
        a_tm2 = a_cols[t + 1]
        a_tm3 = a_cols[t]

        e_t = e_cols[t + 3]
        e_tm1 = e_cols[t + 2]
        e_tm2 = e_cols[t + 1]
        e_tm3 = e_cols[t]

        t1 = (
            e_tm3
            + sigma1_upper(e_t)
            + ch32(e_t, e_tm1, e_tm2)
            + k_vec[t]
            + w_full[:, t]
        ) & MASK32
        t2 = (sigma0_upper(a_t) + maj32(a_t, a_tm1, a_tm2)) & MASK32

        e_tp1 = (a_tm3 + t1) & MASK32
        a_tp1 = (t1 + t2) & MASK32
        e_cols.append(e_tp1)
        a_cols.append(a_tp1)

    a_full = torch.stack(a_cols, dim=1)
    e_full = torch.stack(e_cols, dim=1)

    s_r = torch.stack(
        [
            a_cols[rounds + 3],  # A_R
            a_cols[rounds + 2],  # A_{R-1}
            a_cols[rounds + 1],  # A_{R-2}
            a_cols[rounds],  # A_{R-3}
            e_cols[rounds + 3],  # E_R
            e_cols[rounds + 2],  # E_{R-1}
            e_cols[rounds + 1],  # E_{R-2}
            e_cols[rounds],  # E_{R-3}
        ],
        dim=1,
    )
    digest = (s_r + iv.unsqueeze(0)) & MASK32
    return SHA256Trace(
        A_full=a_full, E_full=e_full, W_full=w_full, S_R=s_r, digest=digest
    )


def sha256_digest(w16: torch.Tensor, rounds: int = 64) -> torch.Tensor:
    """Return only (B, 8) uint32-in-int64 digest words."""
    return sha256_trace(w16, rounds=rounds).digest


def digest_words_to_bytes(digest_words: torch.Tensor) -> bytes:
    """Convert a single (8,) uint32-in-int64 digest tensor into 32 raw bytes."""
    if digest_words.shape != (8,):
        raise ValueError(f"Expected shape (8,), got {tuple(digest_words.shape)}")
    vals = (digest_words.detach().cpu() & MASK32).tolist()
    return b"".join(int(v).to_bytes(4, byteorder="big") for v in vals)
