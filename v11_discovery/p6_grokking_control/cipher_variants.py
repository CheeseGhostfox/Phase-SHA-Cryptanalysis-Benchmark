"""Isomorphic SHA-256 and random ROTR/SHR/K control cipher variants."""

from __future__ import annotations

import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
for _cand in (str(_THIS_DIR), str(_REPO_ROOT)):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    from src.sha256_ref import (
        MASK32,
        SHA256_IV,
        SHA256_K,
        ch32,
        get_iv_tensor,
        maj32,
        rotr32,
        sha256_trace,
        shr32,
    )
except ImportError:
    from sha256_ref import (  # type: ignore
        MASK32,
        SHA256_IV,
        SHA256_K,
        ch32,
        get_iv_tensor,
        maj32,
        rotr32,
        sha256_trace,
        shr32,
    )


SEED_CTRL_1: int = 1000003
SEED_CTRL_2: int = 2000003


@dataclass(frozen=True)
class CipherVariant:
    """Parameterized SHA-256-isomorphic ARX cipher specification."""

    name: str
    k_seed: int | None
    k_tuple: Tuple[int, ...]
    sigma0_upper_rotr: Tuple[int, int, int]
    sigma1_upper_rotr: Tuple[int, int, int]
    sigma0_lower_rotr: Tuple[int, int]
    sigma0_lower_shr: int
    sigma1_lower_rotr: Tuple[int, int]
    sigma1_lower_shr: int

    def offset_signature(self) -> Tuple[Tuple[int, ...], ...]:
        return (
            tuple(sorted(self.sigma0_upper_rotr)),
            tuple(sorted(self.sigma1_upper_rotr)),
            tuple(sorted(self.sigma0_lower_rotr)) + (self.sigma0_lower_shr,),
            tuple(sorted(self.sigma1_lower_rotr)) + (self.sigma1_lower_shr,),
        )

    def to_meta_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "k_seed": self.k_seed,
            "k_head4_hex": [f"0x{v:08x}" for v in self.k_tuple[:4]],
            "rotr_offsets": {
                "Sigma0": list(self.sigma0_upper_rotr),
                "Sigma1": list(self.sigma1_upper_rotr),
                "sigma0": list(self.sigma0_lower_rotr),
                "sigma1": list(self.sigma1_lower_rotr),
            },
            "shr_offsets": {
                "sigma0": self.sigma0_lower_shr,
                "sigma1": self.sigma1_lower_shr,
            },
        }


def _sample_random_variant(
    name: str,
    seed: int,
    forbidden_signatures: set[Tuple[Tuple[int, ...], ...]],
) -> CipherVariant:
    rng = random.Random(seed)
    k_tuple = tuple(rng.randrange(0, 1 << 32) for _ in range(64))
    rotr_pool = list(range(1, 32))
    shr_pool = list(range(1, 16))
    while True:
        s0_u = tuple(rng.sample(rotr_pool, 3))
        s1_u = tuple(rng.sample(rotr_pool, 3))
        s0_l = tuple(rng.sample(rotr_pool, 2))
        s0_shr = rng.choice(shr_pool)
        s1_l = tuple(rng.sample(rotr_pool, 2))
        s1_shr = rng.choice(shr_pool)
        cand = CipherVariant(
            name=name,
            k_seed=seed,
            k_tuple=k_tuple,
            sigma0_upper_rotr=(s0_u[0], s0_u[1], s0_u[2]),
            sigma1_upper_rotr=(s1_u[0], s1_u[1], s1_u[2]),
            sigma0_lower_rotr=(s0_l[0], s0_l[1]),
            sigma0_lower_shr=s0_shr,
            sigma1_lower_rotr=(s1_l[0], s1_l[1]),
            sigma1_lower_shr=s1_shr,
        )
        if cand.offset_signature() not in forbidden_signatures:
            return cand


def build_variants() -> Dict[str, CipherVariant]:
    """Construct REAL_SHA256 and two distinct isomorphic random controls."""
    real = CipherVariant(
        name="REAL_SHA256",
        k_seed=None,
        k_tuple=SHA256_K,
        sigma0_upper_rotr=(2, 13, 22),
        sigma1_upper_rotr=(6, 11, 25),
        sigma0_lower_rotr=(7, 18),
        sigma0_lower_shr=3,
        sigma1_lower_rotr=(17, 19),
        sigma1_lower_shr=10,
    )
    forbidden = {real.offset_signature()}
    ctrl1 = _sample_random_variant("CTRL_RAND_1", SEED_CTRL_1, forbidden)
    forbidden.add(ctrl1.offset_signature())
    ctrl2 = _sample_random_variant("CTRL_RAND_2", SEED_CTRL_2, forbidden)
    return {
        "REAL_SHA256": real,
        "CTRL_RAND_1": ctrl1,
        "CTRL_RAND_2": ctrl2,
    }


def _rotr3_xor(x: torch.Tensor, r: Tuple[int, int, int]) -> torch.Tensor:
    return (rotr32(x, r[0]) ^ rotr32(x, r[1]) ^ rotr32(x, r[2])) & MASK32


def _rotr2_shr_xor(
    x: torch.Tensor, r: Tuple[int, int], s: int
) -> torch.Tensor:
    return (rotr32(x, r[0]) ^ rotr32(x, r[1]) ^ shr32(x, s)) & MASK32


def variant_digest_and_state(
    w16: torch.Tensor,
    rounds: int,
    variant: CipherVariant,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run vectorized reduced-round cipher and return (S_R, H_R) as (B, 8) int64."""
    if w16.ndim != 2 or w16.shape[1] != 16:
        raise ValueError(f"Expected w16 shape (B, 16), got {tuple(w16.shape)}")
    if not (1 <= rounds <= 64):
        raise ValueError(f"rounds must be in [1, 64], got {rounds}")

    batch_size = w16.shape[0]
    device = w16.device
    iv = get_iv_tensor(device)
    k_vec = torch.tensor(variant.k_tuple[:rounds], dtype=torch.int64, device=device)

    # Message schedule expansion
    if rounds <= 16:
        w_full = (w16[:, :rounds] & MASK32).clone()
    else:
        w_cols = [w16[:, t] & MASK32 for t in range(16)]
        for t in range(16, rounds):
            s1 = _rotr2_shr_xor(
                w_cols[t - 2], variant.sigma1_lower_rotr, variant.sigma1_lower_shr
            )
            s0 = _rotr2_shr_xor(
                w_cols[t - 15], variant.sigma0_lower_rotr, variant.sigma0_lower_shr
            )
            wt = (s1 + w_cols[t - 7] + s0 + w_cols[t - 16]) & MASK32
            w_cols.append(wt)
        w_full = torch.stack(w_cols, dim=1).reshape(batch_size, rounds)

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

        sig1 = _rotr3_xor(e_t, variant.sigma1_upper_rotr)
        sig0 = _rotr3_xor(a_t, variant.sigma0_upper_rotr)
        t1 = (e_tm3 + sig1 + ch32(e_t, e_tm1, e_tm2) + k_vec[t] + w_full[:, t]) & MASK32
        t2 = (sig0 + maj32(a_t, a_tm1, a_tm2)) & MASK32

        e_cols.append((a_tm3 + t1) & MASK32)
        a_cols.append((t1 + t2) & MASK32)

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
    h_r = (s_r + iv.unsqueeze(0)) & MASK32
    return s_r, h_r


def verify_real_sha256_parity(device: torch.device | str = "cpu") -> None:
    """Assert REAL_SHA256 matches src.sha256_ref.sha256_trace bit-for-bit."""
    variants = build_variants()
    real = variants["REAL_SHA256"]
    gen = torch.Generator(device="cpu").manual_seed(42)
    w0 = torch.randint(0, 1 << 32, (64,), dtype=torch.int64, generator=gen).to(device)
    w16 = torch.zeros((64, 16), dtype=torch.int64, device=device)
    w16[:, 0] = w0
    w16[:, 1] = 0x80000000
    w16[:, 15] = 32
    for r in (4, 6, 8, 10, 12, 20):
        ref = sha256_trace(w16, rounds=r)
        s_r, h_r = variant_digest_and_state(w16, rounds=r, variant=real)
        if not torch.equal(s_r, ref.S_R):
            raise AssertionError(f"S_R mismatch at R={r}")
        if not torch.equal(h_r, ref.digest):
            raise AssertionError(f"H_R mismatch at R={r}")
