"""Pure-FP32 SHA-256 & Random-Clone Trace Engine + Multi-Scale 2-adic/Bit/Nibble Features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

MASK32: int = 0xFFFFFFFF


def enforce_fp32_determinism() -> None:
    """Lock PyTorch execution to pure IEEE-754 FP32 (TF32 disabled)."""
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _first_n_primes(n: int) -> Tuple[int, ...]:
    primes: List[int] = []
    candidate = 2
    while len(primes) < n:
        limit = math.isqrt(candidate)
        if all(candidate % p != 0 for p in primes if p <= limit):
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


def _icbrt(n: int) -> int:
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

STD_SIGMA0_UPPER = (2, 13, 22)
STD_SIGMA1_UPPER = (6, 11, 25)
STD_SIGMA0_LOWER = (7, 18, 3)   # (rotr1, rotr2, shr)
STD_SIGMA1_LOWER = (17, 19, 10) # (rotr1, rotr2, shr)


@dataclass(frozen=True)
class ControlConfig:
    """Configuration for isomorphic random-ROTR + random-K control SHA-256."""

    seed: int
    sigma0_upper: Tuple[int, int, int]
    sigma1_upper: Tuple[int, int, int]
    sigma0_lower: Tuple[int, int, int]  # (rotr1, rotr2, shr)
    sigma1_lower: Tuple[int, int, int]  # (rotr1, rotr2, shr)
    k_ctrl: Tuple[int, ...]

    def to_meta_dict(self) -> Dict[str, object]:
        return {
            "k_seed": self.seed,
            "rotr_offsets": {
                "Sigma0": list(self.sigma0_upper),
                "Sigma1": list(self.sigma1_upper),
                "sigma0": {"rotr": list(self.sigma0_lower[:2]), "shr": self.sigma0_lower[2]},
                "sigma1": {"rotr": list(self.sigma1_lower[:2]), "shr": self.sigma1_lower[2]},
            },
            "k_ctrl_first4_hex": [f"0x{v:08x}" for v in self.k_ctrl[:4]],
        }


def make_control_config(seed: int = 20261125) -> ControlConfig:
    """Deterministically sample random ROTR/SHR offsets and 64 uniform K constants."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    def _sample_distinct(low: int, high_incl: int, count: int, forbidden: Sequence[Tuple[int, ...]]) -> Tuple[int, ...]:
        pool = list(range(low, high_incl + 1))
        for _ in range(128):
            perm = torch.randperm(len(pool), generator=gen).tolist()
            cand = tuple(sorted(pool[i] for i in perm[:count]))
            if cand not in forbidden:
                return cand
        raise RuntimeError("Failed to sample distinct offsets")

    s0_up = _sample_distinct(1, 31, 3, [STD_SIGMA0_UPPER, STD_SIGMA1_UPPER])
    s1_up = _sample_distinct(1, 31, 3, [STD_SIGMA0_UPPER, STD_SIGMA1_UPPER, s0_up])

    def _sample_lower(forbidden_rotr: Sequence[Tuple[int, int]], forbidden_shr: Sequence[int]) -> Tuple[int, int, int]:
        r_pair = _sample_distinct(1, 31, 2, forbidden_rotr)
        shr_pool = list(range(1, 16))
        perm = torch.randperm(len(shr_pool), generator=gen).tolist()
        shr_val = next(shr_pool[i] for i in perm if shr_pool[i] not in forbidden_shr)
        return (r_pair[0], r_pair[1], shr_val)

    s0_lo = _sample_lower([(7, 18), (17, 19)], [3, 10])
    s1_lo = _sample_lower([(7, 18), (17, 19), s0_lo[:2]], [3, 10, s0_lo[2]])

    k_rand = torch.randint(0, 1 << 32, (64,), dtype=torch.int64, generator=gen).tolist()
    return ControlConfig(
        seed=seed,
        sigma0_upper=s0_up,
        sigma1_upper=s1_up,
        sigma0_lower=s0_lo,
        sigma1_lower=s1_lo,
        k_ctrl=tuple(int(v) & MASK32 for v in k_rand),
    )


def rotr32(x: torch.Tensor, n: int) -> torch.Tensor:
    if n == 0:
        return x & MASK32
    xu = x & MASK32
    return ((xu >> n) | ((xu & ((1 << n) - 1)) << (32 - n))) & MASK32


def shr32(x: torch.Tensor, n: int) -> torch.Tensor:
    return (x & MASK32) >> n


def ch32(e: torch.Tensor, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    eu = e & MASK32
    fu = f & MASK32
    gu = g & MASK32
    return ((eu & fu) ^ ((~eu & MASK32) & gu)) & MASK32


def maj32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    au = a & MASK32
    bu = b & MASK32
    cu = c & MASK32
    return ((au & bu) ^ (au & cu) ^ (bu & cu)) & MASK32


def _sigma_upper(x: torch.Tensor, offsets: Tuple[int, int, int]) -> torch.Tensor:
    return (rotr32(x, offsets[0]) ^ rotr32(x, offsets[1]) ^ rotr32(x, offsets[2])) & MASK32


def _sigma_lower(x: torch.Tensor, offsets: Tuple[int, int, int]) -> torch.Tensor:
    return (rotr32(x, offsets[0]) ^ rotr32(x, offsets[1]) ^ shr32(x, offsets[2])) & MASK32


def _sha256_compress_sr(
    w16: torch.Tensor,
    rounds: int,
    k_vec: torch.Tensor,
    iv: torch.Tensor,
    s0_up: Tuple[int, int, int],
    s1_up: Tuple[int, int, int],
    s0_lo: Tuple[int, int, int],
    s1_lo: Tuple[int, int, int],
) -> torch.Tensor:
    """Compute R-round terminal internal state S_R (B, 8) in uint32-in-int64."""
    if w16.ndim != 2 or w16.shape[1] != 16:
        raise ValueError(f"Expected w16 shape (B, 16), got {tuple(w16.shape)}")
    if not (1 <= rounds <= 64):
        raise ValueError(f"rounds must be in [1, 64], got {rounds}")

    batch_size = w16.shape[0]
    w_cols = [w16[:, t] & MASK32 for t in range(min(16, rounds))]
    for t in range(16, rounds):
        wt = (
            _sigma_lower(w_cols[t - 2], s1_lo)
            + w_cols[t - 7]
            + _sigma_lower(w_cols[t - 15], s0_lo)
            + w_cols[t - 16]
        ) & MASK32
        w_cols.append(wt)

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
        e_t = e_cols[t + 3]
        e_tm1 = e_cols[t + 2]
        e_tm2 = e_cols[t + 1]
        e_tm3 = e_cols[t]

        t1 = (
            e_tm3
            + _sigma_upper(e_t, s1_up)
            + ch32(e_t, e_tm1, e_tm2)
            + k_vec[t]
            + w_cols[t]
        ) & MASK32
        t2 = (_sigma_upper(a_t, s0_up) + maj32(a_t, a_tm1, a_tm2)) & MASK32

        e_cols.append((a_cols[t] + t1) & MASK32)
        a_cols.append((t1 + t2) & MASK32)

    return torch.stack(
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


def sha256_sr_main(w16: torch.Tensor, rounds: int) -> torch.Tensor:
    """Standard SHA-256 R-round internal state S_R (B, 8)."""
    device = w16.device
    iv = torch.tensor(SHA256_IV, dtype=torch.int64, device=device)
    k_vec = torch.tensor(SHA256_K[:rounds], dtype=torch.int64, device=device)
    return _sha256_compress_sr(
        w16, rounds, k_vec, iv,
        STD_SIGMA0_UPPER, STD_SIGMA1_UPPER,
        STD_SIGMA0_LOWER, STD_SIGMA1_LOWER,
    )


def sha256_sr_control(w16: torch.Tensor, rounds: int, cfg: ControlConfig) -> torch.Tensor:
    """Isomorphic random-ROTR + random-K control R-round internal state S_R (B, 8)."""
    device = w16.device
    iv = torch.tensor(SHA256_IV, dtype=torch.int64, device=device)
    k_vec = torch.tensor(cfg.k_ctrl[:rounds], dtype=torch.int64, device=device)
    return _sha256_compress_sr(
        w16, rounds, k_vec, iv,
        cfg.sigma0_upper, cfg.sigma1_upper,
        cfg.sigma0_lower, cfg.sigma1_lower,
    )


def sample_uniform_w16(
    batch_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample S_0 = W_{0..15} i.i.d. uniform in (Z_{2^32})^16 (512-bit full entropy)."""
    return torch.randint(
        0, 1 << 32, (batch_size, 16), dtype=torch.int64, device=device, generator=generator
    )


TWO_ADIC_SCALES: Tuple[int, ...] = (4, 8, 16, 32)


def encode_multiscale_features(words: torch.Tensor) -> torch.Tensor:
    """Encode (B, M) uint32-in-int64 words into deterministic FP32 features (B, 56*M).

    Feature layout per word block:
      1. Bit bipolar spins (32*M): LSB-first, 0 -> +1.0, 1 -> -1.0
      2. 2-adic unit-circle phases (8*M): for p in (4, 8, 16, 32), [cos(theta_p), sin(theta_p)]
      3. Nibble local parity & centered Hamming weight (16*M): 8 nibbles * [parity_spin, (hw-2)/2]
    """
    if words.ndim != 2:
        raise ValueError(f"Expected 2D (B, M) tensor, got {tuple(words.shape)}")
    bsz, n_words = words.shape
    device = words.device
    w_u32 = words & MASK32

    # 1. Bit-level bipolar spins (B, M*32)
    shifts = torch.arange(32, dtype=torch.int64, device=device)
    bits = (w_u32.unsqueeze(-1) >> shifts) & 1  # (B, M, 32)
    phi_bit = (1.0 - 2.0 * bits.to(torch.float32)).reshape(bsz, n_words * 32)

    # 2. 2-adic multi-scale phases (B, M*8)
    # Use float64 before trig to avoid 24-bit mantissa truncation for p=32.
    phase_cols: List[torch.Tensor] = []
    two_pi = 2.0 * math.pi
    for p in TWO_ADIC_SCALES:
        mask = (1 << p) - 1
        denom = float(1 << p)
        theta = ((w_u32 & mask).to(torch.float64) / denom) * two_pi
        phase_cols.append(torch.cos(theta).to(torch.float32))
        phase_cols.append(torch.sin(theta).to(torch.float32))
    # Stack as (B, M, 8) -> (B, M*8) with order per word: p4_cos, p4_sin, p8_cos, ...
    phi_2adic = torch.stack(phase_cols, dim=-1).reshape(bsz, n_words * 8)

    # 3. Nibble parity & density (B, M*16)
    nib_bits = bits.reshape(bsz, n_words, 8, 4)
    hw = nib_bits.sum(dim=-1)  # (B, M, 8) in {0..4}
    parity = hw & 1
    parity_spin = 1.0 - 2.0 * parity.to(torch.float32)
    hw_centered = (hw.to(torch.float32) - 2.0) * 0.5
    phi_nib = torch.stack([parity_spin, hw_centered], dim=-1).reshape(bsz, n_words * 16)

    return torch.cat([phi_bit, phi_2adic, phi_nib], dim=1)


def build_feature_names(word_prefixes: Sequence[str]) -> List[str]:
    """Return exact feature names matching `encode_multiscale_features` column order."""
    names: List[str] = []
    # 1. Bit spins
    for wp in word_prefixes:
        for b in range(32):
            names.append(f"{wp}_b{b}")
    # 2. 2-adic phases
    for wp in word_prefixes:
        for p in TWO_ADIC_SCALES:
            names.append(f"{wp}_2adic_p{p}_cos")
            names.append(f"{wp}_2adic_p{p}_sin")
    # 3. Nibbles
    for wp in word_prefixes:
        for n in range(8):
            names.append(f"{wp}_nib{n}_parity")
            names.append(f"{wp}_nib{n}_hw")
    return names


IN_WORD_PREFIXES: Tuple[str, ...] = tuple(f"W{i}" for i in range(16))
OUT_WORD_PREFIXES: Tuple[str, ...] = (
    "A_R", "A_Rm1", "A_Rm2", "A_Rm3",
    "E_R", "E_Rm1", "E_Rm2", "E_Rm3",
)
IN_FEATURE_NAMES: List[str] = build_feature_names(IN_WORD_PREFIXES)
OUT_FEATURE_NAMES: List[str] = build_feature_names(OUT_WORD_PREFIXES)
