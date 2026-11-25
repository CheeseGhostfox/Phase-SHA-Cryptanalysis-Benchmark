"""Exact SHA-256 / Isomorphic Control Tuple Generator & 3D Crystal Feature Encoder."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import torch

MASK32: int = 0xFFFFFFFF

FAMILIES: Tuple[str, ...] = (
    "F1_xor2",
    "F2_mod2",
    "F3_rx2",
    "F4_boom4",
    "F5_cube8",
)

FAMILY_K: Dict[str, int] = {
    "F1_xor2": 2,
    "F2_mod2": 2,
    "F3_rx2": 2,
    "F4_boom4": 4,
    "F5_cube8": 8,
}

FAMILY_C_IN: Dict[str, int] = {
    "F1_xor2": 6,
    "F2_mod2": 6,
    "F3_rx2": 6,
    "F4_boom4": 8,
    "F5_cube8": 16,
}

CHANNEL_NAMES: Dict[int, Tuple[str, ...]] = {
    6: (
        "spin_m0",
        "spin_m1",
        "xor_diff",
        "mod_diff",
        "mod_borrow_carry",
        "rx1_diff",
    ),
    8: (
        "spin_m0",
        "spin_m1",
        "spin_m2",
        "spin_m3",
        "xor_edge_a",
        "xor_edge_b",
        "xor_2nd_deriv",
        "mod_2nd_alt",
    ),
    16: (
        "spin_m0",
        "spin_m1",
        "spin_m2",
        "spin_m3",
        "spin_m4",
        "spin_m5",
        "spin_m6",
        "spin_m7",
        "xor_edge_01",
        "xor_edge_02",
        "xor_edge_04",
        "xor_face_0123",
        "xor_face_0145",
        "xor_face_0246",
        "xor_3rd_deriv",
        "mod_3rd_alt",
    ),
}

WORD_LABELS: Tuple[str, ...] = (
    "A_R",
    "A_Rm1",
    "A_Rm2",
    "A_Rm3",
    "E_R",
    "E_Rm1",
    "E_Rm2",
    "E_Rm3",
)


def enforce_fp32_determinism() -> None:
    """Enforce pure FP32 precision and disable TF32 everywhere."""
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass


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


def build_control_config(seed: int = 20261125) -> Dict[str, Any]:
    """Build isomorphic random ROTR/SHR/K control clone configuration."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    def _sample_distinct(pool: List[int], k: int, forbidden: Tuple[int, ...]) -> Tuple[int, ...]:
        forbidden_set = set(forbidden)
        while True:
            perm = torch.randperm(len(pool), generator=g).tolist()
            picked = tuple(pool[i] for i in perm[:k])
            if set(picked) != forbidden_set:
                return picked

    rotr_pool = list(range(1, 32))
    shr_pool = list(range(1, 16))

    sig0_upper = _sample_distinct(rotr_pool, 3, (2, 13, 22))
    sig1_upper = _sample_distinct(rotr_pool, 3, (6, 11, 25))
    sig0_lower_rotr = _sample_distinct(rotr_pool, 2, (7, 18))
    sig0_lower_shr = _sample_distinct(shr_pool, 1, (3,))[0]
    sig1_lower_rotr = _sample_distinct(rotr_pool, 2, (17, 19))
    sig1_lower_shr = _sample_distinct(shr_pool, 1, (10,))[0]

    k_ctrl = tuple(
        int(v)
        for v in torch.randint(0, 1 << 32, (64,), dtype=torch.int64, generator=g).tolist()
    )
    return {
        "k_seed": int(seed),
        "rotr_offsets": {
            "Sigma0": list(sig0_upper),
            "Sigma1": list(sig1_upper),
            "sigma0_rotr": list(sig0_lower_rotr),
            "sigma0_shr": int(sig0_lower_shr),
            "sigma1_rotr": list(sig1_lower_rotr),
            "sigma1_shr": int(sig1_lower_shr),
        },
        "k_ctrl": k_ctrl,
    }


DEFAULT_CONTROL_CONFIG = build_control_config(20261125)


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


def _sigma_triple_rotr(x: torch.Tensor, offs: Sequence[int]) -> torch.Tensor:
    xu = x & MASK32
    r0 = ((xu >> offs[0]) | ((xu & ((1 << offs[0]) - 1)) << (32 - offs[0]))) & MASK32
    r1 = ((xu >> offs[1]) | ((xu & ((1 << offs[1]) - 1)) << (32 - offs[1]))) & MASK32
    r2 = ((xu >> offs[2]) | ((xu & ((1 << offs[2]) - 1)) << (32 - offs[2]))) & MASK32
    return r0 ^ r1 ^ r2


def _sigma_two_rotr_shr(x: torch.Tensor, rotrs: Sequence[int], shr_n: int) -> torch.Tensor:
    xu = x & MASK32
    r0 = ((xu >> rotrs[0]) | ((xu & ((1 << rotrs[0]) - 1)) << (32 - rotrs[0]))) & MASK32
    r1 = ((xu >> rotrs[1]) | ((xu & ((1 << rotrs[1]) - 1)) << (32 - rotrs[1]))) & MASK32
    s0 = xu >> shr_n
    return r0 ^ r1 ^ s0


class SHA256CoreCache:
    """Per-device cached constants for fast vectorized SHA-256 and Control Clone."""

    def __init__(self, device: torch.device, control_cfg: Dict[str, Any] | None = None) -> None:
        self.device = device
        self.iv = torch.tensor(SHA256_IV, dtype=torch.int64, device=device)
        self.k_main = torch.tensor(SHA256_K, dtype=torch.int64, device=device)
        cfg = control_cfg or DEFAULT_CONTROL_CONFIG
        self.control_cfg = cfg
        self.k_ctrl = torch.tensor(cfg["k_ctrl"], dtype=torch.int64, device=device)
        offs = cfg["rotr_offsets"]
        self.ctrl_sig0 = tuple(offs["Sigma0"])
        self.ctrl_sig1 = tuple(offs["Sigma1"])
        self.ctrl_s0_rotr = tuple(offs["sigma0_rotr"])
        self.ctrl_s0_shr = int(offs["sigma0_shr"])
        self.ctrl_s1_rotr = tuple(offs["sigma1_rotr"])
        self.ctrl_s1_shr = int(offs["sigma1_shr"])
        self.shifts32 = torch.arange(32, dtype=torch.int64, device=device)


_CORE_CACHE: Dict[Tuple[str, int], SHA256CoreCache] = {}


def get_core_cache(device: torch.device, control_cfg: Dict[str, Any] | None = None) -> SHA256CoreCache:
    seed = (control_cfg or DEFAULT_CONTROL_CONFIG)["k_seed"]
    key = (str(device), int(seed))
    if key not in _CORE_CACHE:
        _CORE_CACHE[key] = SHA256CoreCache(device, control_cfg)
    return _CORE_CACHE[key]


def sha256_state_sr(
    w16: torch.Tensor,
    rounds: int,
    control: bool = False,
    control_cfg: Dict[str, Any] | None = None,
) -> torch.Tensor:
    """Run exact R-round SHA-256 (or isomorphic control) and return S_R (B, 8)."""
    if not (1 <= rounds <= 64):
        raise ValueError(f"rounds must be in [1, 64], got {rounds}")
    cache = get_core_cache(w16.device, control_cfg)
    bsz = w16.shape[0]
    iv = cache.iv

    if control:
        k_vec = cache.k_ctrl
        sig0_offs = cache.ctrl_sig0
        sig1_offs = cache.ctrl_sig1
        s0_rotr = cache.ctrl_s0_rotr
        s0_shr = cache.ctrl_s0_shr
        s1_rotr = cache.ctrl_s1_rotr
        s1_shr = cache.ctrl_s1_shr
    else:
        k_vec = cache.k_main
        sig0_offs = (2, 13, 22)
        sig1_offs = (6, 11, 25)
        s0_rotr = (7, 18)
        s0_shr = 3
        s1_rotr = (17, 19)
        s1_shr = 10

    w_cols = [w16[:, t] & MASK32 for t in range(min(16, rounds))]
    for t in range(16, rounds):
        wt = (
            _sigma_two_rotr_shr(w_cols[t - 2], s1_rotr, s1_shr)
            + w_cols[t - 7]
            + _sigma_two_rotr_shr(w_cols[t - 15], s0_rotr, s0_shr)
            + w_cols[t - 16]
        ) & MASK32
        w_cols.append(wt)

    a_cols = [
        iv[3].expand(bsz),
        iv[2].expand(bsz),
        iv[1].expand(bsz),
        iv[0].expand(bsz),
    ]
    e_cols = [
        iv[7].expand(bsz),
        iv[6].expand(bsz),
        iv[5].expand(bsz),
        iv[4].expand(bsz),
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
            + _sigma_triple_rotr(e_t, sig1_offs)
            + ch32(e_t, e_tm1, e_tm2)
            + k_vec[t]
            + w_cols[t]
        ) & MASK32
        t2 = (_sigma_triple_rotr(a_t, sig0_offs) + maj32(a_t, a_tm1, a_tm2)) & MASK32

        e_cols.append((a_tm3 + t1) & MASK32)
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


def sr_to_crystal_words(s_r: torch.Tensor) -> torch.Tensor:
    """Map (..., 8) [A_R..A_{R-3}, E_R..E_{R-3}] -> (..., 4, 2) C_R[t_rel, track]."""
    track_a = torch.stack(
        [s_r[..., 3], s_r[..., 2], s_r[..., 1], s_r[..., 0]], dim=-1
    )
    track_e = torch.stack(
        [s_r[..., 7], s_r[..., 6], s_r[..., 5], s_r[..., 4]], dim=-1
    )
    return torch.stack([track_a, track_e], dim=-1)


def words_to_bits_lsb(words: torch.Tensor) -> torch.Tensor:
    """Convert uint32-in-int64 words (...) to LSB-first bits (..., 32) in int64."""
    cache = get_core_cache(words.device)
    return (words.unsqueeze(-1) >> cache.shifts32) & 1


def words_to_spins_fp32(words: torch.Tensor) -> torch.Tensor:
    """Convert uint32-in-int64 words (...) to bipolar FP32 spins (..., 32): 0->+1, 1->-1."""
    bits = words_to_bits_lsb(words).to(torch.float32)
    return 1.0 - 2.0 * bits


@torch.no_grad()
def generate_tuple_batch(
    family: str,
    batch_size: int,
    rounds: int,
    device: torch.device,
    generator: torch.Generator,
    control: bool = False,
    control_cfg: Dict[str, Any] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate balanced on-the-fly batch of structured tuples (Y=1) and independent tuples (Y=0).

    Returns:
        s_r_tuples: (B, k, 8) uint32-in-int64 S_R states
        c_tuples:   (B, k, 4, 2) uint32-in-int64 3D crystal words
        labels:     (B,) float32 in {1.0, 0.0}
    """
    if family not in FAMILY_K:
        raise ValueError(f"Unknown family {family!r}")
    k = FAMILY_K[family]
    n_pos = batch_size // 2
    n_neg = batch_size - n_pos

    w_base = torch.randint(0, 1 << 32, (n_pos, 16), dtype=torch.int64, device=device, generator=generator)

    if family == "F1_xor2":
        w_alt = w_base.clone()
        w_alt[:, 0] ^= 0x80000000
        w_pos = torch.stack([w_base, w_alt], dim=1)
    elif family == "F2_mod2":
        w_alt = w_base.clone()
        w_alt[:, 0] = (w_alt[:, 0] + 1) & MASK32
        w_pos = torch.stack([w_base, w_alt], dim=1)
    elif family == "F3_rx2":
        w_alt = rotr32(w_base, 1)
        w_pos = torch.stack([w_base, w_alt], dim=1)
    elif family == "F4_boom4":
        w0 = w_base
        w1 = w_base.clone()
        w1[:, 0] ^= 0x80000000
        w2 = w_base.clone()
        w2[:, 0] ^= 0x00008000
        w3 = w1.clone()
        w3[:, 0] ^= 0x00008000
        w_pos = torch.stack([w0, w1, w2, w3], dim=1)
    elif family == "F5_cube8":
        deltas = (0x80000000, 0x00008000, 0x00000080)
        members = []
        for m in range(8):
            wm = w_base.clone()
            d = 0
            if m & 1:
                d ^= deltas[0]
            if m & 2:
                d ^= deltas[1]
            if m & 4:
                d ^= deltas[2]
            if d:
                wm[:, 0] ^= d
            members.append(wm)
        w_pos = torch.stack(members, dim=1)
    else:
        raise ValueError(family)

    w_neg = torch.randint(
        0, 1 << 32, (n_neg, k, 16), dtype=torch.int64, device=device, generator=generator
    )
    w_all = torch.cat([w_pos, w_neg], dim=0).reshape(batch_size * k, 16)
    sr_all = sha256_state_sr(w_all, rounds=rounds, control=control, control_cfg=control_cfg)
    sr_tuples = sr_all.reshape(batch_size, k, 8)
    c_tuples = sr_to_crystal_words(sr_tuples)

    labels = torch.empty((batch_size,), dtype=torch.float32, device=device)
    labels[:n_pos] = 1.0
    labels[n_pos:] = 0.0
    return sr_tuples, c_tuples, labels


@torch.no_grad()
def encode_crystal_features(c_tuples: torch.Tensor, family: str) -> torch.Tensor:
    """
    Encode (B, k, 4, 2) uint32 crystal tuples into (B, C_in, 4, 2, 32) pure FP32 tensor Psi_3D.
    """
    k = FAMILY_K[family]
    if c_tuples.shape[1] != k:
        raise ValueError(f"Expected k={k} for {family}, got {c_tuples.shape[1]}")

    if k == 2:
        c0 = c_tuples[:, 0]
        c1 = c_tuples[:, 1]
        diff_mod = (c1 - c0) & MASK32
        borrow = (diff_mod ^ c1 ^ c0) & MASK32
        rx1 = (c1 ^ rotr32(c0, 1)) & MASK32
        planes = torch.stack([c0, c1, c0 ^ c1, diff_mod, borrow, rx1], dim=1)
    elif k == 4:
        c0, c1, c2, c3 = c_tuples[:, 0], c_tuples[:, 1], c_tuples[:, 2], c_tuples[:, 3]
        e01 = c0 ^ c1
        e02 = c0 ^ c2
        d2_xor = e01 ^ c2 ^ c3
        d2_mod = (c0 - c1 - c2 + c3) & MASK32
        planes = torch.stack([c0, c1, c2, c3, e01, e02, d2_xor, d2_mod], dim=1)
    elif k == 8:
        c = [c_tuples[:, m] for m in range(8)]
        e01 = c[0] ^ c[1]
        e02 = c[0] ^ c[2]
        e04 = c[0] ^ c[4]
        f0123 = e01 ^ c[2] ^ c[3]
        f0145 = e01 ^ c[4] ^ c[5]
        f0246 = e02 ^ c[4] ^ c[6]
        d3_xor = f0123 ^ c[4] ^ c[5] ^ c[6] ^ c[7]
        d3_mod = (c[0] - c[1] - c[2] + c[3] - c[4] + c[5] + c[6] - c[7]) & MASK32
        planes = torch.stack(
            [*c, e01, e02, e04, f0123, f0145, f0246, d3_xor, d3_mod], dim=1
        )
    else:
        raise ValueError(f"Unsupported k={k}")

    return words_to_spins_fp32(planes)


def highest_order_diff_sr(sr_tuples: torch.Tensor, family: str) -> torch.Tensor:
    """Return highest-order differential word tensor (B, 8) uint32-in-int64 for baselines & Walsh."""
    if family == "F1_xor2":
        return sr_tuples[:, 0] ^ sr_tuples[:, 1]
    if family == "F2_mod2":
        return (sr_tuples[:, 1] - sr_tuples[:, 0]) & MASK32
    if family == "F3_rx2":
        return sr_tuples[:, 1] ^ rotr32(sr_tuples[:, 0], 1)
    if family == "F4_boom4":
        return sr_tuples[:, 0] ^ sr_tuples[:, 1] ^ sr_tuples[:, 2] ^ sr_tuples[:, 3]
    if family == "F5_cube8":
        out = sr_tuples[:, 0].clone()
        for m in range(1, 8):
            out ^= sr_tuples[:, m]
        return out
    raise ValueError(f"Unknown family {family!r}")


def structured_mod_residual_sr(sr_tuples: torch.Tensor, family: str) -> torch.Tensor:
    """Return structured 2-adic modular/rotation residual d_w (B, 8) uint32-in-int64."""
    if family in ("F1_xor2", "F2_mod2"):
        return (sr_tuples[:, 1] - sr_tuples[:, 0]) & MASK32
    if family == "F3_rx2":
        return (sr_tuples[:, 1] - rotr32(sr_tuples[:, 0], 1)) & MASK32
    if family == "F4_boom4":
        return (sr_tuples[:, 0] - sr_tuples[:, 1] - sr_tuples[:, 2] + sr_tuples[:, 3]) & MASK32
    if family == "F5_cube8":
        return (
            sr_tuples[:, 0]
            - sr_tuples[:, 1]
            - sr_tuples[:, 2]
            + sr_tuples[:, 3]
            - sr_tuples[:, 4]
            + sr_tuples[:, 5]
            + sr_tuples[:, 6]
            - sr_tuples[:, 7]
        ) & MASK32
    raise ValueError(f"Unknown family {family!r}")
