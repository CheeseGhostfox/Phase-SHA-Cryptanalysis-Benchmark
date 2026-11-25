"""Exact R-round SHA-256 and isomorphic control cipher operators for P7."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

# Ensure repo root is importable both locally and when uploaded to Colab
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if (_REPO_ROOT / "src" / "sha256_ref.py").exists() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
elif str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from src.sha256_ref import (
    MASK32,
    SHA256_IV,
    SHA256_K,
    ch32,
    maj32,
    rotr32,
    shr32,
)


def enforce_fp32_lock() -> None:
    """Lock pure real FP32 execution and disable TF32."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, "preferred_linalg_library"):
        torch.backends.cuda.preferred_linalg_library("cusolver")


def assert_fp32_locked() -> None:
    """Assert TF32 is disabled and default dtype is float32."""
    assert torch.get_default_dtype() == torch.float32, (
        f"Expected default dtype float32, got {torch.get_default_dtype()}"
    )
    assert not torch.backends.cuda.matmul.allow_tf32, "matmul allow_tf32 must be False"
    assert not torch.backends.cudnn.allow_tf32, "cudnn allow_tf32 must be False"


@dataclass(frozen=True)
class CipherSpec:
    """Specification of a SHA-256 or isomorphic ROTR/SHR/K control variant."""

    name: str
    iv: Tuple[int, ...]
    k_consts: Tuple[int, ...]
    sig0_upper: Tuple[int, int, int]  # (r1, r2, r3)
    sig1_upper: Tuple[int, int, int]  # (r1, r2, r3)
    sig0_lower: Tuple[int, int, int]  # (r1, r2, s3)
    sig1_lower: Tuple[int, int, int]  # (r1, r2, s3)


def make_main_spec() -> CipherSpec:
    """Standard SHA-256 specification."""
    return CipherSpec(
        name="main",
        iv=SHA256_IV,
        k_consts=SHA256_K,
        sig0_upper=(2, 13, 22),
        sig1_upper=(6, 11, 25),
        sig0_lower=(7, 18, 3),
        sig1_lower=(17, 19, 10),
    )


def make_ctrl_rotr_k_spec(seed: int = 20261125) -> CipherSpec:
    """Isomorphic SHA-256 clone with resampled ROTR/SHR shifts and round constants."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    k_rand = torch.randint(0, 1 << 32, (64,), dtype=torch.int64, generator=gen)
    k_tuple = tuple(int(v) & MASK32 for v in k_rand.tolist())

    def _sample_triple(allow_shr: bool) -> Tuple[int, int, int]:
        perm = torch.randperm(31, generator=gen) + 1  # in [1, 31]
        a, b, c = int(perm[0]), int(perm[1]), int(perm[2])
        if not allow_shr:
            vals = sorted((a, b, c))
            return (vals[0], vals[1], vals[2])
        r_vals = sorted((a, b))
        return (r_vals[0], r_vals[1], c)

    return CipherSpec(
        name="ctrl_rotr_k",
        iv=SHA256_IV,
        k_consts=k_tuple,
        sig0_upper=_sample_triple(False),
        sig1_upper=_sample_triple(False),
        sig0_lower=_sample_triple(True),
        sig1_lower=_sample_triple(True),
    )


def _sigma_upper(x: torch.Tensor, shifts: Tuple[int, int, int]) -> torch.Tensor:
    r1, r2, r3 = shifts
    return (rotr32(x, r1) ^ rotr32(x, r2) ^ rotr32(x, r3)) & MASK32


def _sigma_lower(x: torch.Tensor, shifts: Tuple[int, int, int]) -> torch.Tensor:
    r1, r2, s3 = shifts
    return (rotr32(x, r1) ^ rotr32(x, r2) ^ shr32(x, s3)) & MASK32


def expand_schedule_spec(
    w16: torch.Tensor, rounds: int, spec: CipherSpec
) -> torch.Tensor:
    """Expand (B, 16) message block into (B, rounds) words under CipherSpec."""
    if rounds <= 16:
        return (w16[:, :rounds] & MASK32).clone()
    w_cols = [w16[:, t] & MASK32 for t in range(16)]
    for t in range(16, rounds):
        wt = (
            _sigma_lower(w_cols[t - 2], spec.sig1_lower)
            + w_cols[t - 7]
            + _sigma_lower(w_cols[t - 15], spec.sig0_lower)
            + w_cols[t - 16]
        ) & MASK32
        w_cols.append(wt)
    return torch.stack(w_cols, dim=1)


def run_rounds_ae(
    w16: torch.Tensor,
    rounds: int,
    spec: CipherSpec,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run exact R-round compression and return (A_R, E_R) as (B,) int64 & MASK32."""
    if w16.ndim != 2 or w16.shape[1] != 16:
        raise ValueError(f"Expected w16 shape (B, 16), got {tuple(w16.shape)}")
    batch_size = w16.shape[0]
    device = w16.device

    iv = torch.tensor(spec.iv, dtype=torch.int64, device=device)
    k_vec = torch.tensor(spec.k_consts[:rounds], dtype=torch.int64, device=device)
    w_full = expand_schedule_spec(w16, rounds, spec)

    a_cols = [
        iv[3].expand(batch_size),
        iv[2].expand(batch_size),
        iv[1].expand(batch_size),
        iv[0].expand(batch_size),
    ]
    e_cols = [
        iv[7].expand(batch_size),
        iv[6].expand(batch_size),
        iv[5].expand(batch_size),
        iv[4].expand(batch_size),
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
            + _sigma_upper(e_t, spec.sig1_upper)
            + ch32(e_t, e_tm1, e_tm2)
            + k_vec[t]
            + w_full[:, t]
        ) & MASK32
        t2 = (_sigma_upper(a_t, spec.sig0_upper) + maj32(a_t, a_tm1, a_tm2)) & MASK32

        e_cols.append((a_tm3 + t1) & MASK32)
        a_cols.append((t1 + t2) & MASK32)

    return a_cols[rounds + 3], e_cols[rounds + 3]


def feistel32_strong_bijection(w0: torch.Tensor, seed: int = 918273645) -> torch.Tensor:
    """Deterministic 4-round 32-bit ARX-Feistel bijection on uint32-in-int64 tensor."""
    x = w0.to(torch.int64) & MASK32
    lo = x & 0xFFFF
    hi = (x >> 16) & 0xFFFF

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    keys = torch.randint(0, 1 << 16, (4,), dtype=torch.int64, generator=gen).tolist()
    mults = (0x9E37, 0x7F4A, 0x85EB, 0xC2B2)

    for r in range(4):
        f_val = (((lo ^ keys[r]) * mults[r]) + (lo >> 5) + ((lo << 7) & 0xFFFF)) & 0xFFFF
        f_val = (f_val ^ (f_val >> 7)) & 0xFFFF
        new_hi = (hi ^ f_val) & 0xFFFF
        hi = lo
        lo = new_hi

    return ((hi << 16) | lo) & MASK32


def sample_keys(
    num_keys: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample independent background keys: w0_mask (K,) and w_rest (K, 15)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    w0_mask = torch.randint(0, 1 << 32, (num_keys,), dtype=torch.int64, generator=gen)
    w_rest = torch.randint(
        0, 1 << 32, (num_keys, 15), dtype=torch.int64, generator=gen
    )
    return w0_mask.to(device), w_rest.to(device)


def sample_perms_256(num_keys: int, seed: int, device: torch.device) -> torch.Tensor:
    """Sample num_keys independent uniform random bijections in S_256."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    perms = [torch.randperm(256, dtype=torch.int64, generator=gen) for _ in range(num_keys)]
    return torch.stack(perms, dim=0).to(device)
