"""
P5「BTC 320-bit 固定填充 × 弱 Midstate 共振探测」工程实现。

严格遵循 g:\\SHA-LM\\v11_discovery\\p5_btc_weak_midstate\\SPEC.md 契约：
- 精确 SHA-256 / Double-SHA256 算术全程使用 int64 & 0xFFFFFFFF；
- 可微松弛搜索严格锁定纯 FP32（allow_tf32=False, torch.float32）；
- 严格 Nonce 两阶段物理隔离：训练池 [0, 2^30) vs 留出池 [2^30, 2^32)；
- 同步运行 Main、BASE_RAND_MIDSTATE 与同构随机 ROTR/K 克隆对照组 CTRL_RAND_ROTR_K；
- 扫频 R in {8, 12, 16, 20, 24, 32, 64, double64}。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 若命令行请求 --cpu_only，必须在 import torch 前屏蔽 CUDA 设备
if "--cpu_only" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch

MASK32: int = 0xFFFFFFFF

SEED_TRAIN: int = 20261105
SEED_HOLDOUT: int = 998244353
SEED_HOLDOUT2: int = 104729017
SEED_CTRL: int = 135792468
SEED_BASE: int = 777654321

SWEEP_KEYS: Tuple[str, ...] = ("8", "12", "16", "20", "24", "32", "64", "double64")


def _first_n_primes(n: int) -> Tuple[int, ...]:
    primes: List[int] = []
    cand = 2
    while len(primes) < n:
        lim = math.isqrt(cand)
        if all(cand % p != 0 for p in primes if p <= lim):
            primes.append(cand)
        cand += 1
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


def enforce_fp32_strict() -> Dict[str, Any]:
    """锁定纯 IEEE-754 FP32，禁止 TF32 截断。"""
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(torch.float32)
    return {
        "default_dtype": str(torch.get_default_dtype()),
        "matmul_allow_tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
    }


@dataclass(frozen=True)
class SHA256Variant:
    """SHA-256 标准算子或同构随机 ROTR/K 克隆对照算子定义。"""

    name: str
    iv: Tuple[int, ...]
    k: Tuple[int, ...]
    S0_rot: Tuple[int, int, int]
    S1_rot: Tuple[int, int, int]
    s0_rot: Tuple[int, int]
    s0_shr: int
    s1_rot: Tuple[int, int]
    s1_shr: int


def make_main_variant() -> SHA256Variant:
    return SHA256Variant(
        name="SHA256_MAIN",
        iv=SHA256_IV,
        k=SHA256_K,
        S0_rot=(2, 13, 22),
        S1_rot=(6, 11, 25),
        s0_rot=(7, 18),
        s0_shr=3,
        s1_rot=(17, 19),
        s1_shr=10,
    )


def make_ctrl_variant(seed: int = SEED_CTRL) -> SHA256Variant:
    """按 SPEC 4.2 使用固定种子重新采样全部 64 个 K 与 12 个互异位移参数。"""
    rng = np.random.default_rng(seed)

    def _sample_rot3(forbid: Tuple[int, int, int]) -> Tuple[int, int, int]:
        while True:
            cand = tuple(sorted(int(x) for x in rng.choice(np.arange(1, 32), size=3, replace=False)))
            if cand != forbid:
                return cand  # type: ignore[return-value]

    def _sample_rot2_shr(forbid_rot: Tuple[int, int], forbid_shr: int) -> Tuple[Tuple[int, int], int]:
        while True:
            rots = tuple(sorted(int(x) for x in rng.choice(np.arange(1, 32), size=2, replace=False)))
            shr = int(rng.integers(1, 32))
            if rots != forbid_rot and shr != forbid_shr:
                return rots, shr  # type: ignore[return-value]

    S0_rot = _sample_rot3((2, 13, 22))
    S1_rot = _sample_rot3((6, 11, 25))
    s0_rot, s0_shr = _sample_rot2_shr((7, 18), 3)
    s1_rot, s1_shr = _sample_rot2_shr((17, 19), 10)
    k_ctrl = tuple(int(x) for x in rng.integers(0, 2**32, size=64, dtype=np.uint64))
    return SHA256Variant(
        name="CTRL_RAND_ROTR_K",
        iv=SHA256_IV,
        k=k_ctrl,
        S0_rot=S0_rot,
        S1_rot=S1_rot,
        s0_rot=s0_rot,
        s0_shr=s0_shr,
        s1_rot=s1_rot,
        s1_shr=s1_shr,
    )


# ---------------------------------------------------------------------------
# 精确 uint32-in-int64 向量化原语
# ---------------------------------------------------------------------------

def rotr32(x: torch.Tensor, n: int) -> torch.Tensor:
    xu = x & MASK32
    if n == 0:
        return xu
    return ((xu >> n) | ((xu & ((1 << n) - 1)) << (32 - n))) & MASK32


def shr32(x: torch.Tensor, n: int) -> torch.Tensor:
    return (x & MASK32) >> n


def ch32(e: torch.Tensor, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    eu, fu, gu = e & MASK32, f & MASK32, g & MASK32
    return ((eu & fu) ^ ((~eu & MASK32) & gu)) & MASK32


def maj32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    au, bu, cu = a & MASK32, b & MASK32, c & MASK32
    return ((au & bu) ^ (au & cu) ^ (bu & cu)) & MASK32


def popcnt32(x: torch.Tensor) -> torch.Tensor:
    """精确 32-bit SWAR popcount（输入任意形状 int64，仅看低 32 bits）。"""
    v = x & MASK32
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    return ((v * 0x01010101) & MASK32) >> 24


def clz32(x: torch.Tensor) -> torch.Tensor:
    """精确 32-bit 无符号前导零位数。"""
    v = x & MASK32
    v = v | (v >> 1)
    v = v | (v >> 2)
    v = v | (v >> 4)
    v = v | (v >> 8)
    v = v | (v >> 16)
    return 32 - popcnt32(v)


def bswap32(x: torch.Tensor) -> torch.Tensor:
    v = x & MASK32
    return (
        ((v & 0x000000FF) << 24)
        | ((v & 0x0000FF00) << 8)
        | ((v & 0x00FF0000) >> 8)
        | ((v & 0xFF000000) >> 24)
    ) & MASK32


def clz256(h_words: torch.Tensor) -> torch.Tensor:
    """(B, 8) uint32-in-int64 摘要的大端 256-bit 前导零位数。"""
    bsz = h_words.shape[0]
    device = h_words.device
    clz = torch.zeros(bsz, dtype=torch.int64, device=device)
    active = torch.ones(bsz, dtype=torch.bool, device=device)
    for j in range(8):
        wj = h_words[:, j] & MASK32
        clz = clz + torch.where(active, clz32(wj), 0)
        active = active & (wj == 0)
    return clz


def ctz_btc256(h_words: torch.Tensor) -> torch.Tensor:
    """(B, 8) 摘要按 BTC 小端 256-bit 整数解释的高位连续零位数。"""
    le_words = torch.stack([bswap32(h_words[:, 7 - j]) for j in range(8)], dim=1)
    return clz256(le_words)


def _sigma0_upper(x: torch.Tensor, v: SHA256Variant) -> torch.Tensor:
    r0, r1, r2 = v.S0_rot
    return (rotr32(x, r0) ^ rotr32(x, r1) ^ rotr32(x, r2)) & MASK32


def _sigma1_upper(x: torch.Tensor, v: SHA256Variant) -> torch.Tensor:
    r0, r1, r2 = v.S1_rot
    return (rotr32(x, r0) ^ rotr32(x, r1) ^ rotr32(x, r2)) & MASK32


def _sigma0_lower(x: torch.Tensor, v: SHA256Variant) -> torch.Tensor:
    r0, r1 = v.s0_rot
    return (rotr32(x, r0) ^ rotr32(x, r1) ^ shr32(x, v.s0_shr)) & MASK32


def _sigma1_lower(x: torch.Tensor, v: SHA256Variant) -> torch.Tensor:
    r0, r1 = v.s1_rot
    return (rotr32(x, r0) ^ rotr32(x, r1) ^ shr32(x, v.s1_shr)) & MASK32


def pack_btc_block2(w012: torch.Tensor, nonces: torch.Tensor) -> torch.Tensor:
    """
    构造 BTC Block 2 固定 320-bit 尾部填充消息块 (B, 16)。
    w012: (B, 3) 或 (3,)
    nonces: (B,)
    """
    bsz = nonces.shape[0]
    device = nonces.device
    w16 = torch.zeros((bsz, 16), dtype=torch.int64, device=device)
    if w012.ndim == 1:
        w16[:, 0] = w012[0] & MASK32
        w16[:, 1] = w012[1] & MASK32
        w16[:, 2] = w012[2] & MASK32
    else:
        w16[:, 0:3] = w012 & MASK32
    w16[:, 3] = nonces & MASK32
    w16[:, 4] = 0x80000000
    # w16[:, 5:15] = 0
    w16[:, 15] = 640
    return w16


def pack_double_sha_second_block(h1: torch.Tensor) -> torch.Tensor:
    """构造 Double-SHA256 第二重 512-bit 单块填充输入 (B, 16)。"""
    bsz = h1.shape[0]
    w16 = torch.zeros((bsz, 16), dtype=torch.int64, device=h1.device)
    w16[:, 0:8] = h1 & MASK32
    w16[:, 8] = 0x80000000
    w16[:, 15] = 256
    return w16


def sha256_compress_exact(
    init_state: torch.Tensor,
    w16: torch.Tensor,
    rounds: int,
    variant: SHA256Variant,
    collect_stats: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    向量化精确 SHA-256 压缩：
    init_state: (B, 8) 或 (8,) uint32-in-int64
    w16: (B, 16) uint32-in-int64
    返回 H (B, 8), carry_all_bits (B,), carry_post4_bits (B,), w_post18_bits (B,)
    """
    bsz = w16.shape[0]
    device = w16.device
    if init_state.ndim == 1:
        s0 = init_state.unsqueeze(0).expand(bsz, 8)
    else:
        s0 = init_state
    s0 = s0 & MASK32

    k_vec = torch.tensor(variant.k[:rounds], dtype=torch.int64, device=device)

    w_cols = [w16[:, t] & MASK32 for t in range(16)]
    w_post18_bits = torch.zeros(bsz, dtype=torch.int64, device=device)
    for t in range(16, rounds):
        wt = (
            _sigma1_lower(w_cols[t - 2], variant)
            + w_cols[t - 7]
            + _sigma0_lower(w_cols[t - 15], variant)
            + w_cols[t - 16]
        ) & MASK32
        w_cols.append(wt)
        if collect_stats and t >= 18:
            w_post18_bits = w_post18_bits + popcnt32(wt)

    a, b, c, d, e, f, g, h = [s0[:, j] for j in range(8)]
    carry_all_bits = torch.zeros(bsz, dtype=torch.int64, device=device)
    carry_post4_bits = torch.zeros(bsz, dtype=torch.int64, device=device)

    for t in range(rounds):
        S1 = _sigma1_upper(e, variant)
        ch = ch32(e, f, g)
        kt = k_vec[t]
        wt = w_cols[t]

        p1 = (h + S1) & MASK32
        p2 = (p1 + ch) & MASK32
        p3 = (p2 + kt) & MASK32
        t1 = (p3 + wt) & MASK32

        S0 = _sigma0_upper(a, variant)
        mj = maj32(a, b, c)
        t2 = (S0 + mj) & MASK32

        a_new = (t1 + t2) & MASK32
        e_new = (d + t1) & MASK32

        if collect_stats:
            c3 = h ^ S1 ^ p1
            c4 = p1 ^ ch ^ p2
            c5 = p2 ^ kt ^ p3
            c6 = p3 ^ wt ^ t1
            c7 = S0 ^ mj ^ t2
            c8 = t1 ^ t2 ^ a_new
            c9 = d ^ t1 ^ e_new
            c_round = (
                popcnt32(c3)
                + popcnt32(c4)
                + popcnt32(c5)
                + popcnt32(c6)
                + popcnt32(c7)
                + popcnt32(c8)
                + popcnt32(c9)
            )
            carry_all_bits = carry_all_bits + c_round
            if t >= 4:
                carry_post4_bits = carry_post4_bits + c_round

        h, g, f, e, d, c, b, a = g, f, e, e_new, c, b, a, a_new

    s_r = torch.stack([a, b, c, d, e, f, g, h], dim=1)
    h_out = (s_r + s0) & MASK32
    return {
        "H": h_out,
        "carry_all_bits": carry_all_bits,
        "carry_post4_bits": carry_post4_bits,
        "w_post18_bits": w_post18_bits,
    }


def btc_forward_exact(
    templates: torch.Tensor,
    nonces: torch.Tensor,
    sweep_key: str,
    variant: SHA256Variant,
    collect_stats: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    对 1 个模板 (11,) 和一批 Nonce (B,)，或广播对 (B, 11) 模板与 (B,) Nonce，
    执行精确 BTC Block 2 单层 R 轮或 Double-SHA256 前向。
    """
    if templates.ndim == 1:
        s0 = templates[:8]
        w012 = templates[8:11]
    else:
        s0 = templates[:, :8]
        w012 = templates[:, 8:11]

    w16 = pack_btc_block2(w012, nonces)
    if sweep_key == "double64":
        out1 = sha256_compress_exact(s0, w16, rounds=64, variant=variant, collect_stats=collect_stats)
        iv_t = torch.tensor(variant.iv, dtype=torch.int64, device=nonces.device)
        w_dbl = pack_double_sha_second_block(out1["H"])
        out2 = sha256_compress_exact(iv_t, w_dbl, rounds=64, variant=variant, collect_stats=False)
        return {
            "H": out2["H"],
            "carry_all_bits": out1["carry_all_bits"],
            "carry_post4_bits": out1["carry_post4_bits"],
            "w_post18_bits": out1["w_post18_bits"],
        }

    rounds = int(sweep_key)
    return sha256_compress_exact(s0, w16, rounds=rounds, variant=variant, collect_stats=collect_stats)


def sweep_rounds_count(sweep_key: str) -> int:
    return 64 if sweep_key == "double64" else int(sweep_key)


def generate_valid_block1_midstates(
    n: int,
    seed: int,
    variant: SHA256Variant,
    device: torch.device,
) -> torch.Tensor:
    """从随机 64-byte Block 1 经完整 64 轮压缩生成物理合法 BTC Midstate (n, 8)。"""
    rng = np.random.default_rng(seed)
    blk1 = rng.integers(0, 2**32, size=(n, 16), dtype=np.uint64).astype(np.int64)
    blk1_t = torch.from_numpy(blk1).to(device=device, dtype=torch.int64)
    iv_t = torch.tensor(variant.iv, dtype=torch.int64, device=device)
    with torch.no_grad():
        return sha256_compress_exact(iv_t, blk1_t, rounds=64, variant=variant, collect_stats=False)["H"]


# ---------------------------------------------------------------------------
# 严格 Nonce 空间互斥隔离
# ---------------------------------------------------------------------------

def sample_train_nonces(n: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    """严格限定在 [0, 2^30) 的训练 Nonce 池 N_train。"""
    arr = rng.integers(0, 1 << 30, size=n, dtype=np.int64)
    return torch.from_numpy(arr).to(device=device, dtype=torch.int64)


def build_holdout_nonces(
    n_holdout: int,
    seed: int,
    exclude: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    在 [2^30, 2^32) 独立采样无重复留出 Nonce 池 N_holdout。
    """
    rng = np.random.default_rng(seed)
    low, high = 1 << 30, 1 << 32
    chunks: List[np.ndarray] = []
    collected = 0
    seen: Optional[np.ndarray] = np.sort(exclude) if exclude is not None else None
    while collected < n_holdout:
        need = int((n_holdout - collected) * 1.05) + 1024
        cand = np.unique(rng.integers(low, high, size=need, dtype=np.int64))
        if seen is not None and seen.size > 0:
            mask = ~np.isin(cand, seen, assume_unique=True)
            cand = cand[mask]
        chunks.append(cand)
        merged = np.unique(np.concatenate(chunks))
        chunks = [merged]
        collected = int(merged.size)
    res = chunks[0][:n_holdout]
    rng.shuffle(res)
    assert int(res.min()) >= low and int(res.max()) < high
    assert np.unique(res).size == n_holdout
    return res


# ---------------------------------------------------------------------------
# 精确留出统计评估（单模板大样本分块）
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_template_holdout(
    template: torch.Tensor,
    holdout_nonces: torch.Tensor,
    sweep_key: str,
    variant: SHA256Variant,
    chunk_size: int = 65536,
) -> Dict[str, Any]:
    """
    对单个冻结模板 (11,) 在 holdout_nonces 上执行纯整数精确前向与全维统计。
    """
    n_total = int(holdout_nonces.shape[0])
    device = holdout_nonces.device
    rounds = sweep_rounds_count(sweep_key)
    shifts = torch.arange(32, dtype=torch.int64, device=device)

    hw_sum = 0.0
    hw_sq_sum = 0.0
    clz_sum = 0.0
    clz_sq_sum = 0.0
    btc_lz_sum = 0.0
    btc_lz_sq_sum = 0.0
    clz_ge8 = 0
    btc_lz_ge8 = 0
    carry_all_sum = 0.0
    carry_post4_sum = 0.0
    carry_post4_sq_sum = 0.0
    w_post18_sum = 0.0
    bit_ones = torch.zeros((8, 32), dtype=torch.int64, device=device)

    denom_carry_all = float(224 * rounds)
    denom_carry_post4 = float(224 * max(1, rounds - 4))
    has_post18 = rounds > 18
    denom_w_post18 = float(32 * max(1, rounds - 18))

    for start in range(0, n_total, chunk_size):
        nonces_c = holdout_nonces[start : start + chunk_size]
        out = btc_forward_exact(template, nonces_c, sweep_key, variant, collect_stats=True)
        h = out["H"]  # (B, 8)

        hw = popcnt32(h).sum(dim=1).to(torch.float64)
        clz = clz256(h).to(torch.float64)
        btc_lz = ctz_btc256(h).to(torch.float64)

        hw_sum += float(hw.sum().item())
        hw_sq_sum += float((hw * hw).sum().item())
        clz_sum += float(clz.sum().item())
        clz_sq_sum += float((clz * clz).sum().item())
        btc_lz_sum += float(btc_lz.sum().item())
        btc_lz_sq_sum += float((btc_lz * btc_lz).sum().item())
        clz_ge8 += int((clz >= 8.0).sum().item())
        btc_lz_ge8 += int((btc_lz >= 8.0).sum().item())

        c_all = out["carry_all_bits"].to(torch.float64) / denom_carry_all
        c_p4 = out["carry_post4_bits"].to(torch.float64) / denom_carry_post4
        carry_all_sum += float(c_all.sum().item())
        carry_post4_sum += float(c_p4.sum().item())
        carry_post4_sq_sum += float((c_p4 * c_p4).sum().item())
        if has_post18:
            w_p18 = out["w_post18_bits"].to(torch.float64) / denom_w_post18
            w_post18_sum += float(w_p18.sum().item())

        bits = (h.unsqueeze(-1) >> shifts) & 1  # (B, 8, 32)
        bit_ones += bits.sum(dim=0)

    inv_n = 1.0 / float(n_total)
    sqrt_n = math.sqrt(float(n_total))

    hw_mean = hw_sum * inv_n
    hw_var = max(0.0, hw_sq_sum * inv_n - hw_mean * hw_mean)
    hw_se = math.sqrt(hw_var * inv_n)
    z_hw = (hw_mean - 128.0) / (8.0 / sqrt_n)

    clz_mean = clz_sum * inv_n
    clz_var = max(0.0, clz_sq_sum * inv_n - clz_mean * clz_mean)
    clz_se = math.sqrt(clz_var * inv_n)
    z_clz = (clz_mean - 1.0) / math.sqrt(2.0 * inv_n)

    btc_lz_mean = btc_lz_sum * inv_n
    btc_lz_var = max(0.0, btc_lz_sq_sum * inv_n - btc_lz_mean * btc_lz_mean)
    btc_lz_se = math.sqrt(btc_lz_var * inv_n)
    z_btc_lz = (btc_lz_mean - 1.0) / math.sqrt(2.0 * inv_n)

    p_bits = bit_ones.to(torch.float64).reshape(256) * inv_n
    z_bits = 2.0 * sqrt_n * (p_bits - 0.5)
    abs_z_bits = z_bits.abs()
    best_bit_idx = int(torch.argmax(abs_z_bits).item())
    z_bit_max = float(abs_z_bits[best_bit_idx].item())
    max_bit_bias = float((p_bits[best_bit_idx] - 0.5).abs().item())
    p_best_bit = float(p_bits[best_bit_idx].item())
    se_best_bit = math.sqrt(max(1e-12, p_best_bit * (1.0 - p_best_bit) * inv_n))

    carry_all_mean = carry_all_sum * inv_n
    carry_post4_mean = carry_post4_sum * inv_n
    c_p4_var = max(0.0, carry_post4_sq_sum * inv_n - carry_post4_mean * carry_post4_mean)
    carry_post4_se = math.sqrt(c_p4_var * inv_n)
    w_density_post18 = (w_post18_sum * inv_n) if has_post18 else None

    max_abs_z_output = max(abs(z_hw), abs(z_clz), abs(z_btc_lz), z_bit_max)

    return {
        "N_holdout": n_total,
        "template_hex": [f"0x{int(x) & MASK32:08x}" for x in template.detach().cpu().tolist()],
        "hw_mean": hw_mean,
        "hw_se": hw_se,
        "z_hw": z_hw,
        "clz_mean": clz_mean,
        "clz_se": clz_se,
        "z_clz": z_clz,
        "p_clz_ge_8": clz_ge8 * inv_n,
        "btc_lz_mean": btc_lz_mean,
        "btc_lz_se": btc_lz_se,
        "z_btc_lz": z_btc_lz,
        "p_btc_lz_ge_8": btc_lz_ge8 * inv_n,
        "max_bit_bias": max_bit_bias,
        "z_bit_max": z_bit_max,
        "best_bit_idx": best_bit_idx,
        "se_best_bit": se_best_bit,
        "carry_all": carry_all_mean,
        "carry_post4": carry_post4_mean,
        "carry_post4_se": carry_post4_se,
        "w_density_post18": w_density_post18,
        "max_abs_z_output": max_abs_z_output,
    }


# ---------------------------------------------------------------------------
# 路径 B：纯 FP32 可微松弛 SHA-256（Kogge-Stone 并行前缀加法 + STE）
# ---------------------------------------------------------------------------

def _soft_xor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y - 2.0 * x * y


def _soft_or(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y - x * y


def _soft_ch(e: torch.Tensor, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return e * f + (1.0 - e) * g


def _soft_maj(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return a * b + a * c + b * c - 2.0 * a * b * c


def _soft_rotr(x: torch.Tensor, n: int) -> torch.Tensor:
    if n == 0:
        return x
    return torch.roll(x, shifts=-n, dims=-1)


def _soft_shr(x: torch.Tensor, n: int) -> torch.Tensor:
    if n == 0:
        return x
    zeros = torch.zeros((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
    return torch.cat([x[..., n:], zeros], dim=-1)


def _soft_shift_left(x: torch.Tensor, n: int) -> torch.Tensor:
    zeros = torch.zeros((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
    return torch.cat([zeros, x[..., :-n]], dim=-1)


def soft_add32(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    纯 FP32 Kogge-Stone 5 步并行前缀 32-bit 软加法器。
    输入 x, y 形状 (..., 32) LSB-first，取值 [0, 1]。
    返回 (sum_bits (..., 32), carry_in_bits (..., 32))。
    """
    p0 = _soft_xor(x, y)
    p = p0
    g = x * y
    for dist in (1, 2, 4, 8, 16):
        g_shift = _soft_shift_left(g, dist)
        p_shift = _soft_shift_left(p, dist)
        g = _soft_or(g, p * g_shift)
        p = p * p_shift
    c_in = _soft_shift_left(g, 1)
    return _soft_xor(p0, c_in), c_in


def _words_to_soft_bits(words: torch.Tensor, device: torch.device) -> torch.Tensor:
    shifts = torch.arange(32, dtype=torch.int64, device=device)
    return ((words.to(torch.int64).unsqueeze(-1) >> shifts) & 1).to(torch.float32)


def _soft_bits_to_words(bits: torch.Tensor) -> torch.Tensor:
    b = (bits >= 0.5).to(torch.int64)
    shifts = torch.arange(32, dtype=torch.int64, device=bits.device)
    return ((b << shifts).sum(dim=-1)) & MASK32


def soft_btc_forward(
    theta_bits: torch.Tensor,
    nonces: torch.Tensor,
    rounds: int,
    variant: SHA256Variant,
) -> Dict[str, torch.Tensor]:
    """
    纯 FP32 可微 BTC Block 2 单层前向。
    theta_bits: (M, 11, 32) 软比特 [0, 1]
    nonces: (B,) int64
    输出展平批次 (M * B, ...) 的软输出比特与进位能量。
    """
    device = theta_bits.device
    m_restarts = theta_bits.shape[0]
    bsz = nonces.shape[0]

    nonce_bits = _words_to_soft_bits(nonces, device)  # (B, 32)
    # 扩展到 (M, B, 32)
    s0 = theta_bits[:, :8, :].unsqueeze(1).expand(m_restarts, bsz, 8, 32)
    w012 = theta_bits[:, 8:11, :].unsqueeze(1).expand(m_restarts, bsz, 3, 32)
    w3 = nonce_bits.unsqueeze(0).expand(m_restarts, bsz, 32)

    zero_w = torch.zeros((m_restarts, bsz, 32), dtype=torch.float32, device=device)
    w4 = _words_to_soft_bits(torch.tensor([0x80000000], dtype=torch.int64, device=device), device).view(1, 1, 32).expand(m_restarts, bsz, 32)
    w15 = _words_to_soft_bits(torch.tensor([640], dtype=torch.int64, device=device), device).view(1, 1, 32).expand(m_restarts, bsz, 32)

    w_cols: List[torch.Tensor] = [w012[:, :, 0], w012[:, :, 1], w012[:, :, 2], w3, w4]
    for _ in range(5, 15):
        w_cols.append(zero_w)
    w_cols.append(w15)

    r0_s0, r1_s0 = variant.s0_rot
    r0_s1, r1_s1 = variant.s1_rot
    for t in range(16, rounds):
        w_m2 = w_cols[t - 2]
        w_m15 = w_cols[t - 15]
        s1 = _soft_xor(_soft_xor(_soft_rotr(w_m2, r0_s1), _soft_rotr(w_m2, r1_s1)), _soft_shr(w_m2, variant.s1_shr))
        s0_l = _soft_xor(_soft_xor(_soft_rotr(w_m15, r0_s0), _soft_rotr(w_m15, r1_s0)), _soft_shr(w_m15, variant.s0_shr))
        u0, _ = soft_add32(s1, w_cols[t - 7])
        u1, _ = soft_add32(u0, s0_l)
        wt, _ = soft_add32(u1, w_cols[t - 16])
        w_cols.append(wt)

    k_bits = _words_to_soft_bits(torch.tensor(variant.k[:rounds], dtype=torch.int64, device=device), device)  # (R, 32)
    a, b, c, d, e, f, g, h = [s0[:, :, j, :] for j in range(8)]

    R0_S0, R1_S0, R2_S0 = variant.S0_rot
    R0_S1, R1_S1, R2_S1 = variant.S1_rot

    carry_post4_energy = torch.zeros((m_restarts, bsz), dtype=torch.float32, device=device)
    for t in range(rounds):
        S1 = _soft_xor(_soft_xor(_soft_rotr(e, R0_S1), _soft_rotr(e, R1_S1)), _soft_rotr(e, R2_S1))
        ch = _soft_ch(e, f, g)
        kt = k_bits[t].view(1, 1, 32)
        wt = w_cols[t]

        p1, c3 = soft_add32(h, S1)
        p2, c4 = soft_add32(p1, ch)
        p3, c5 = soft_add32(p2, kt)
        t1, c6 = soft_add32(p3, wt)

        S0 = _soft_xor(_soft_xor(_soft_rotr(a, R0_S0), _soft_rotr(a, R1_S0)), _soft_rotr(a, R2_S0))
        mj = _soft_maj(a, b, c)
        t2, c7 = soft_add32(S0, mj)

        a_new, c8 = soft_add32(t1, t2)
        e_new, c9 = soft_add32(d, t1)

        if t >= 4:
            carry_post4_energy = carry_post4_energy + (c3 + c4 + c5 + c6 + c7 + c8 + c9).mean(dim=-1) / 7.0

        h, g, f, e, d, c, b, a = g, f, e, e_new, c, b, a, a_new

    out_words = []
    for j, st_w in enumerate((a, b, c, d, e, f, g, h)):
        hw_j, _ = soft_add32(st_w, s0[:, :, j, :])
        out_words.append(hw_j)
    h_out_bits = torch.stack(out_words, dim=2)  # (M, B, 8, 32)
    carry_post4_mean = carry_post4_energy / float(max(1, rounds - 4))
    w_post18_mean = (
        torch.stack(w_cols[18:rounds], dim=2).mean(dim=(2, 3))
        if rounds > 18
        else torch.zeros((m_restarts, bsz), dtype=torch.float32, device=device)
    )
    return {
        "H_bits": h_out_bits,
        "carry_post4": carry_post4_mean,
        "w_post18": w_post18_mean,
    }


def run_path_b_gradient_search(
    sweep_key: str,
    variant: SHA256Variant,
    train_nonces_pool: torch.Tensor,
    rng: np.random.Generator,
    device: torch.device,
    steps: int = 36,
    restarts: int = 4,
    batch_size: int = 64,
    init_templates: Optional[torch.Tensor] = None,
) -> List[Tuple[Tuple[int, ...], Dict[str, float]]]:
    """
    路径 B：纯 FP32 可微 STE 弱模板搜索。
    返回候选离散模板及其在训练批上的多维指标列表。
    """
    enforce_fp32_strict()
    rounds = 32 if sweep_key == "double64" else int(sweep_key)
    soft_rounds = min(rounds, 24)  # 对 R=32/64 在软搜索端控制图深以防梯消散并保证速度，离散评估仍用精确全轮

    alpha = torch.randn((restarts, 11, 32), dtype=torch.float32, device=device) * 0.8
    if init_templates is not None and init_templates.shape[0] > 0:
        n_copy = min(restarts // 2, int(init_templates.shape[0]))
        if n_copy > 0:
            bits_init = _words_to_soft_bits(init_templates[:n_copy], device)
            alpha[:n_copy] = (bits_init * 2.0 - 1.0) * 1.5
    alpha.requires_grad_(True)
    opt = torch.optim.Adam([alpha], lr=0.22)

    candidates: Dict[Tuple[int, ...], Dict[str, float]] = {}
    pool_n = int(train_nonces_pool.shape[0])

    for step in range(steps):
        tau = max(0.35, 1.0 - 0.65 * (step / max(1, steps - 1)))
        idx = torch.randint(0, pool_n, (batch_size,), device=device)
        batch_nonces = train_nonces_pool[idx]

        p = torch.sigmoid(alpha / tau)
        hard = (p >= 0.5).to(torch.float32)
        theta_ste = 0.30 * p + 0.70 * (hard + (p - p.detach()))

        out = soft_btc_forward(theta_ste, batch_nonces, rounds=soft_rounds, variant=variant)
        h_bits = out["H_bits"]  # (M, B, 8, 32)
        p_bit_mean = h_bits.mean(dim=1)  # (M, 8, 32)
        hw_mean = h_bits.sum(dim=(2, 3)).mean(dim=1)  # (M,)

        # 软前导零对数/乘积目标（H[0] 最高 12 bits -> 0）
        msb_bits = torch.flip(h_bits[:, :, 0, 20:32], dims=[-1])  # (M, B, 12)
        p_zero_prefix = torch.cumprod(1.0 - msb_bits + 1e-6, dim=-1).sum(dim=-1).mean(dim=1)

        # 多目标分配给不同 restart：0=低进位, 1=低W+低进位, 2=单比特偏置/HW, 3=前导零
        loss_vec = torch.zeros(restarts, dtype=torch.float32, device=device)
        for m in range(restarts):
            mode = m % 4
            if mode == 0:
                loss_vec[m] = out["carry_post4"][m].mean()
            elif mode == 1:
                loss_vec[m] = out["carry_post4"][m].mean() + 0.5 * out["w_post18"][m].mean()
            elif mode == 2:
                max_bias = (p_bit_mean[m] - 0.5).abs().max()
                loss_vec[m] = -max_bias - 0.02 * (hw_mean[m] - 128.0).abs()
            else:
                loss_vec[m] = -p_zero_prefix[m] + 0.2 * out["carry_post4"][m].mean()

        opt.zero_grad(set_to_none=True)
        loss_vec.sum().backward()
        opt.step()

        if step % 6 == 0 or step == steps - 1:
            with torch.no_grad():
                disc = _soft_bits_to_words(torch.sigmoid(alpha))  # (M, 11)
                eval_nonces = train_nonces_pool[: min(1024, pool_n)]
                for m in range(restarts):
                    tpl = disc[m]
                    key = tuple(int(x) for x in tpl.detach().cpu().tolist())
                    if key not in candidates:
                        candidates[key] = score_template_train(tpl, eval_nonces, sweep_key, variant)

    return list(candidates.items())


# ---------------------------------------------------------------------------
# 路径 A：代数稀疏锚定 + 多目标演化搜索
# ---------------------------------------------------------------------------

def _make_algebraic_seed_templates(
    count: int,
    rng: np.random.Generator,
    variant: SHA256Variant,
    device: torch.device,
) -> torch.Tensor:
    """
    利用 Block 2 解析结构构造初始种子：
    W[16] = W[17] = 0 且对消第 0 轮 T1/T2 进位链。
    """
    def _rotr_py(x: int, n: int) -> int:
        x &= MASK32
        return ((x >> n) | ((x & ((1 << n) - 1)) << (32 - n))) & MASK32

    def _s0_l(x: int) -> int:
        r0, r1 = variant.s0_rot
        return (_rotr_py(x, r0) ^ _rotr_py(x, r1) ^ ((x & MASK32) >> variant.s0_shr)) & MASK32

    def _s1_l(x: int) -> int:
        r0, r1 = variant.s1_rot
        return (_rotr_py(x, r0) ^ _rotr_py(x, r1) ^ ((x & MASK32) >> variant.s1_shr)) & MASK32

    def _S1_u(x: int) -> int:
        r0, r1, r2 = variant.S1_rot
        return (_rotr_py(x, r0) ^ _rotr_py(x, r1) ^ _rotr_py(x, r2)) & MASK32

    rows: List[List[int]] = []
    s1_640 = _s1_l(640)
    for i in range(count):
        if i % 2 == 0:
            w2 = int(rng.integers(0, 2**32, dtype=np.uint64)) & MASK32
            w1 = (-s1_640 - _s0_l(w2)) & MASK32
            w0 = (-_s0_l(w1)) & MASK32
            a0 = b0 = c0 = d0 = 0
            e0 = int(rng.integers(0, 2**32, dtype=np.uint64)) & MASK32
            f0 = int(rng.integers(0, 2**32, dtype=np.uint64)) & MASK32
            g0 = int(rng.integers(0, 2**32, dtype=np.uint64)) & MASK32
            ch0 = ((e0 & f0) ^ ((~e0 & MASK32) & g0)) & MASK32
            h0 = (-_S1_u(e0) - ch0 - variant.k[0] - w0) & MASK32
            rows.append([a0, b0, c0, d0, e0, f0, g0, h0, w0, w1, w2])
        else:
            # 稀疏 + 轮常数对消种子
            w0 = (-variant.k[0]) & MASK32
            w1 = (-variant.k[1]) & MASK32
            w2 = (-variant.k[2]) & MASK32
            rows.append([0, 0, 0, 0, 0, 0, 0, 0, w0, w1, w2])
    return torch.tensor(rows, dtype=torch.int64, device=device)


@torch.no_grad()
def score_population_train(
    templates: torch.Tensor,
    train_nonces: torch.Tensor,
    sweep_key: str,
    variant: SHA256Variant,
) -> List[Dict[str, float]]:
    """批量向量化在训练 Nonce 批上精确评分 (P, 11) x (B,)。"""
    if templates.ndim == 1:
        templates = templates.unsqueeze(0)
    p_cnt = int(templates.shape[0])
    b_cnt = int(train_nonces.shape[0])
    n = float(b_cnt)
    sqrt_n = math.sqrt(n)
    rounds = sweep_rounds_count(sweep_key)
    shifts = torch.arange(32, dtype=torch.int64, device=templates.device)

    tpl_exp = templates.unsqueeze(1).expand(p_cnt, b_cnt, 11).reshape(p_cnt * b_cnt, 11)
    nonce_exp = train_nonces.unsqueeze(0).expand(p_cnt, b_cnt).reshape(p_cnt * b_cnt)
    out = btc_forward_exact(tpl_exp, nonce_exp, sweep_key, variant, collect_stats=True)

    h = out["H"].view(p_cnt, b_cnt, 8)
    hw_mean = popcnt32(h).sum(dim=2).to(torch.float32).mean(dim=1)
    clz_mean = clz256(h.reshape(p_cnt * b_cnt, 8)).view(p_cnt, b_cnt).to(torch.float32).mean(dim=1)
    btc_lz_mean = ctz_btc256(h.reshape(p_cnt * b_cnt, 8)).view(p_cnt, b_cnt).to(torch.float32).mean(dim=1)

    bits = (h.unsqueeze(-1) >> shifts) & 1  # (P, B, 8, 32)
    p_bits = bits.to(torch.float32).mean(dim=1)  # (P, 8, 32)
    z_bit_max = (2.0 * sqrt_n * (p_bits - 0.5).abs()).amax(dim=(1, 2))

    carry_post4 = out["carry_post4_bits"].view(p_cnt, b_cnt).to(torch.float32).mean(dim=1) / float(
        224 * max(1, rounds - 4)
    )
    if rounds > 18:
        w_post18 = out["w_post18_bits"].view(p_cnt, b_cnt).to(torch.float32).mean(dim=1) / float(
            32 * max(1, rounds - 18)
        )
    else:
        w_post18 = torch.full((p_cnt,), 0.5, dtype=torch.float32, device=templates.device)

    results: List[Dict[str, float]] = []
    hw_l = hw_mean.tolist()
    clz_l = clz_mean.tolist()
    btc_l = btc_lz_mean.tolist()
    zb_l = z_bit_max.tolist()
    cp_l = carry_post4.tolist()
    wp_l = w_post18.tolist()
    denom_hw = 8.0 / sqrt_n
    denom_lz = math.sqrt(2.0 / n)
    for i in range(p_cnt):
        z_hw_abs = abs(hw_l[i] - 128.0) / denom_hw
        z_clz = (clz_l[i] - 1.0) / denom_lz
        z_btc = (btc_l[i] - 1.0) / denom_lz
        results.append(
            {
                "carry_post4": cp_l[i],
                "w_post18": wp_l[i],
                "z_bit_max": zb_l[i],
                "z_hw_abs": z_hw_abs,
                "z_clz": z_clz,
                "z_btc_lz": z_btc,
                "output_score": max(zb_l[i], z_hw_abs, abs(z_clz), abs(z_btc)),
            }
        )
    return results


@torch.no_grad()
def score_template_train(
    template: torch.Tensor,
    train_nonces: torch.Tensor,
    sweep_key: str,
    variant: SHA256Variant,
) -> Dict[str, float]:
    """单模板训练批评分（委托批量实现）。"""
    return score_population_train(template.unsqueeze(0), train_nonces, sweep_key, variant)[0]


def _mutate_templates(
    parents: torch.Tensor,
    rng: np.random.Generator,
    variant: SHA256Variant,
) -> torch.Tensor:
    """多尺度变异：单比特翻转、稀疏多比特、字级交叉与半字节扰动。"""
    p_np = parents.detach().cpu().numpy().copy()
    n = p_np.shape[0]
    for i in range(n):
        mode = int(rng.integers(0, 4))
        if mode == 0:
            # 单比特翻转
            w_idx = int(rng.integers(0, 11))
            b_idx = int(rng.integers(0, 32))
            p_np[i, w_idx] ^= 1 << b_idx
        elif mode == 1:
            # 2-4 比特翻转
            k_flips = int(rng.integers(2, 5))
            for _ in range(k_flips):
                w_idx = int(rng.integers(0, 11))
                b_idx = int(rng.integers(0, 32))
                p_np[i, w_idx] ^= 1 << b_idx
        elif mode == 2:
            # 字级交叉
            mate = int(rng.integers(0, n))
            mask = rng.random(11) < 0.35
            p_np[i, mask] = p_np[mate, mask]
        else:
            # 局部字加减微扰或整字替换
            w_idx = int(rng.integers(0, 11))
            delta = int(rng.integers(-64, 65))
            p_np[i, w_idx] = (int(p_np[i, w_idx]) + delta) & MASK32
    return torch.from_numpy(p_np).to(device=parents.device, dtype=torch.int64)


def run_weak_template_search(
    sweep_key: str,
    variant: SHA256Variant,
    train_nonces_pool: torch.Tensor,
    seed: int,
    device: torch.device,
    pop_size: int = 48,
    generations: int = 18,
    train_batch: int = 1024,
    grad_steps: int = 32,
    grad_batch: int = 64,
    top_k: int = 16,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    在严格隔离的 N_train 上联合执行路径 A（演化）+ 路径 B（可微 STE），输出 Top-K 离散模板。
    """
    rng = np.random.default_rng(seed)
    n_alg = pop_size // 2
    n_rand = pop_size - n_alg
    n_b1 = n_rand // 2
    n_unif = n_rand - n_b1

    alg_pop = _make_algebraic_seed_templates(n_alg, rng, variant, device)
    b1_mids = generate_valid_block1_midstates(n_b1, seed + 17, variant, device)
    b1_w012 = torch.from_numpy(rng.integers(0, 2**32, size=(n_b1, 3), dtype=np.uint64).astype(np.int64)).to(device)
    b1_pop = torch.cat([b1_mids, b1_w012], dim=1)
    unif_pop = torch.from_numpy(rng.integers(0, 2**32, size=(n_unif, 11), dtype=np.uint64).astype(np.int64)).to(device)

    pop = torch.cat([alg_pop, b1_pop, unif_pop], dim=0)
    archive: Dict[Tuple[int, ...], Dict[str, float]] = {}

    pool_n = int(train_nonces_pool.shape[0])
    bsz = min(train_batch, pool_n)

    def _obj_val(sc: Dict[str, float], sub_id: int) -> float:
        # 越大越优
        if sub_id == 0:
            return -sc["carry_post4"]
        if sub_id == 1:
            return -(sc["carry_post4"] + 0.5 * sc["w_post18"])
        if sub_id == 2:
            return sc["output_score"]
        return sc["z_clz"] + sc["z_btc_lz"]

    sub_size = pop_size // 4
    for gen in range(generations):
        idx = torch.randint(0, pool_n, (bsz,), device=device)
        batch_nonces = train_nonces_pool[idx]
        scores = score_population_train(pop, batch_nonces, sweep_key, variant)
        pop_list = pop.detach().cpu().tolist()
        for i in range(pop_size):
            archive[tuple(int(x) for x in pop_list[i])] = scores[i]

        next_parts: List[torch.Tensor] = []
        for sub_id in range(4):
            s_idx = sub_id * sub_size
            e_idx = pop_size if sub_id == 3 else (sub_id + 1) * sub_size
            sub_indices = list(range(s_idx, e_idx))
            sub_indices.sort(key=lambda idx_i: _obj_val(scores[idx_i], sub_id), reverse=True)
            elites = pop[sub_indices[: max(2, len(sub_indices) // 3)]]
            n_children = (e_idx - s_idx) - elites.shape[0]
            parents_idx = rng.integers(0, elites.shape[0], size=n_children)
            children = _mutate_templates(elites[parents_idx], rng, variant)
            next_parts.append(torch.cat([elites, children], dim=0))
        pop = torch.cat(next_parts, dim=0)

    # 路径 B：以最佳演化种子启动可微搜索
    best_carry_key = min(archive.keys(), key=lambda k: archive[k]["carry_post4"])
    best_out_key = max(archive.keys(), key=lambda k: archive[k]["output_score"])
    seed_for_grad = torch.tensor([list(best_carry_key), list(best_out_key)], dtype=torch.int64, device=device)
    grad_cands = run_path_b_gradient_search(
        sweep_key=sweep_key,
        variant=variant,
        train_nonces_pool=train_nonces_pool,
        rng=rng,
        device=device,
        steps=grad_steps,
        restarts=4,
        batch_size=min(grad_batch, pool_n),
        init_templates=seed_for_grad,
    )
    for k, sc in grad_cands:
        archive[k] = sc

    # 统一在较大训练复评批上选出平衡的 Top-K（覆盖低进位、低 W、高输出偏置与高 LZ）
    reeval_nonces = train_nonces_pool[: min(2048, pool_n)]
    pool_keys = list(archive.keys())
    # 先初筛前 48 个候选再精确复评，兼顾速度与精度
    shortlist = set()
    shortlist.update(sorted(pool_keys, key=lambda k: archive[k]["carry_post4"])[:12])
    shortlist.update(sorted(pool_keys, key=lambda k: archive[k]["carry_post4"] + 0.5 * archive[k]["w_post18"])[:12])
    shortlist.update(sorted(pool_keys, key=lambda k: archive[k]["output_score"], reverse=True)[:16])
    shortlist.update(sorted(pool_keys, key=lambda k: archive[k]["z_clz"] + archive[k]["z_btc_lz"], reverse=True)[:12])

    shortlist_list = list(shortlist)
    sl_tensor = torch.tensor([list(k) for k in shortlist_list], dtype=torch.int64, device=device)
    sl_scores = score_population_train(sl_tensor, reeval_nonces, sweep_key, variant)
    reeval_scores: Dict[Tuple[int, ...], Dict[str, float]] = dict(zip(shortlist_list, sl_scores))

    selected: List[Tuple[int, ...]] = []
    # 每类各取最优若干，确保不漏任何弱方向并去重
    buckets = [
        sorted(reeval_scores.keys(), key=lambda k: reeval_scores[k]["output_score"], reverse=True),
        sorted(reeval_scores.keys(), key=lambda k: reeval_scores[k]["z_clz"] + reeval_scores[k]["z_btc_lz"], reverse=True),
        sorted(reeval_scores.keys(), key=lambda k: reeval_scores[k]["carry_post4"]),
        sorted(reeval_scores.keys(), key=lambda k: reeval_scores[k]["carry_post4"] + 0.5 * reeval_scores[k]["w_post18"]),
    ]
    ptr = 0
    while len(selected) < top_k and ptr < len(reeval_scores):
        added = False
        for b in buckets:
            if ptr < len(b) and b[ptr] not in selected:
                selected.append(b[ptr])
                added = True
                if len(selected) >= top_k:
                    break
        if not added:
            break
        ptr += 1

    while len(selected) < top_k:
        # 若唯一模板不足 top_k，补随机模板
        extra = tuple(int(x) for x in rng.integers(0, 2**32, size=11, dtype=np.uint64))
        if extra not in selected:
            selected.append(extra)

    top_templates = torch.tensor([list(k) for k in selected[:top_k]], dtype=torch.int64, device=device)
    meta = {
        "archive_size": len(archive),
        "shortlist_size": len(shortlist),
        "best_train_carry_post4": min(s["carry_post4"] for s in reeval_scores.values()),
        "best_train_output_z": max(s["output_score"] for s in reeval_scores.values()),
    }
    return top_templates, meta


def _build_base_rand_templates(
    variant: SHA256Variant,
    device: torch.device,
    seed: int = SEED_BASE,
    k: int = 16,
) -> torch.Tensor:
    """SPEC 4.2：8 个均匀随机 Midstate + 8 个合法 Block 1 压缩 Midstate。"""
    rng = np.random.default_rng(seed)
    n_unif = k // 2
    n_b1 = k - n_unif
    unif = torch.from_numpy(rng.integers(0, 2**32, size=(n_unif, 11), dtype=np.uint64).astype(np.int64)).to(device)
    b1_mids = generate_valid_block1_midstates(n_b1, seed + 99, variant, device)
    b1_w012 = torch.from_numpy(rng.integers(0, 2**32, size=(n_b1, 3), dtype=np.uint64).astype(np.int64)).to(device)
    b1 = torch.cat([b1_mids, b1_w012], dim=1)
    return torch.cat([unif, b1], dim=0)


def _summarize_group_holdout(
    templates: torch.Tensor,
    holdout_nonces: torch.Tensor,
    sweep_key: str,
    variant: SHA256Variant,
    chunk_size: int = 65536,
) -> Dict[str, Any]:
    per_tpl: List[Dict[str, Any]] = []
    for i in range(int(templates.shape[0])):
        per_tpl.append(evaluate_template_holdout(templates[i], holdout_nonces, sweep_key, variant, chunk_size=chunk_size))

    best_out = max(per_tpl, key=lambda r: r["max_abs_z_output"])
    best_hw = max(per_tpl, key=lambda r: abs(r["z_hw"]))
    best_clz = max(per_tpl, key=lambda r: abs(r["z_clz"]))
    best_btc_lz = max(per_tpl, key=lambda r: abs(r["z_btc_lz"]))
    best_bit = max(per_tpl, key=lambda r: r["z_bit_max"])
    best_carry = min(per_tpl, key=lambda r: r["carry_post4"])

    w_vals = [r["w_density_post18"] for r in per_tpl if r["w_density_post18"] is not None]
    best_w = min(w_vals) if w_vals else None

    return {
        "best_of_16": {
            "hw_mean": best_hw["hw_mean"],
            "hw_se": best_hw["hw_se"],
            "z_hw": best_hw["z_hw"],
            "clz_mean": best_clz["clz_mean"],
            "clz_se": best_clz["clz_se"],
            "z_clz": best_clz["z_clz"],
            "p_clz_ge_8": best_clz["p_clz_ge_8"],
            "btc_lz_mean": best_btc_lz["btc_lz_mean"],
            "btc_lz_se": best_btc_lz["btc_lz_se"],
            "z_btc_lz": best_btc_lz["z_btc_lz"],
            "p_btc_lz_ge_8": best_btc_lz["p_btc_lz_ge_8"],
            "max_bit_bias": best_bit["max_bit_bias"],
            "z_bit_max": best_bit["z_bit_max"],
            "se_best_bit": best_bit["se_best_bit"],
            "carry_all": best_carry["carry_all"],
            "carry_post4": best_carry["carry_post4"],
            "carry_post4_se": best_carry["carry_post4_se"],
            "w_density_post18": best_w,
            "max_abs_z_output": best_out["max_abs_z_output"],
            "best_out_template_hex": best_out["template_hex"],
            "best_carry_template_hex": best_carry["template_hex"],
        },
        "group_mean": {
            "hw_mean": float(np.mean([r["hw_mean"] for r in per_tpl])),
            "clz_mean": float(np.mean([r["clz_mean"] for r in per_tpl])),
            "btc_lz_mean": float(np.mean([r["btc_lz_mean"] for r in per_tpl])),
            "z_bit_max_mean": float(np.mean([r["z_bit_max"] for r in per_tpl])),
            "carry_all_mean": float(np.mean([r["carry_all"] for r in per_tpl])),
            "carry_post4_mean": float(np.mean([r["carry_post4"] for r in per_tpl])),
            "w_density_post18_mean": float(np.mean(w_vals)) if w_vals else None,
        },
        "top_16_templates": per_tpl,
    }


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def run_full_experiment(
    out_dir: Path,
    sweep_keys: Sequence[str] = SWEEP_KEYS,
    n_train_pool: int = 8192,
    n_holdout: int = 262144,
    n_holdout2: int = 1048576,
    pop_size: int = 48,
    generations: int = 18,
    grad_steps: int = 32,
    top_k: int = 16,
    chunk_size: int = 65536,
    cpu_only: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    t_start = time.time()
    fp32_info = enforce_fp32_strict()
    device = torch.device("cpu" if cpu_only or not torch.cuda.is_available() else "cuda:0")
    hw_name = "CPU" if device.type == "cpu" else torch.cuda.get_device_name(device)
    print(f"[P5] Start on device={device} ({hw_name}), fp32={fp32_info}", flush=True)

    main_var = make_main_variant()
    ctrl_var = make_ctrl_variant(SEED_CTRL)

    # 严格隔离生成训练池与留出池
    rng_train = np.random.default_rng(SEED_TRAIN)
    train_pool = sample_train_nonces(n_train_pool, rng_train, device)
    assert int(train_pool.min().item()) >= 0 and int(train_pool.max().item()) < (1 << 30)

    holdout_np = build_holdout_nonces(n_holdout, SEED_HOLDOUT)
    holdout_t = torch.from_numpy(holdout_np).to(device=device, dtype=torch.int64)
    holdout2_t: Optional[torch.Tensor] = None

    base_templates = _build_base_rand_templates(main_var, device, seed=SEED_BASE, k=top_k)

    sweep_results: Dict[str, Any] = {}
    for idx_k, skey in enumerate(sweep_keys):
        t_sk = time.time()
        print(f"\n[P5] === Sweep R={skey} ({idx_k+1}/{len(sweep_keys)}) ===", flush=True)

        main_tpls, main_search_meta = run_weak_template_search(
            sweep_key=skey,
            variant=main_var,
            train_nonces_pool=train_pool,
            seed=SEED_TRAIN + idx_k * 1009,
            device=device,
            pop_size=pop_size,
            generations=generations,
            grad_steps=grad_steps,
            top_k=top_k,
        )
        ctrl_tpls, ctrl_search_meta = run_weak_template_search(
            sweep_key=skey,
            variant=ctrl_var,
            train_nonces_pool=train_pool,
            seed=SEED_TRAIN + idx_k * 1009,
            device=device,
            pop_size=pop_size,
            generations=generations,
            grad_steps=grad_steps,
            top_k=top_k,
        )

        main_eval = _summarize_group_holdout(main_tpls, holdout_t, skey, main_var, chunk_size=chunk_size)
        base_eval = _summarize_group_holdout(base_templates, holdout_t, skey, main_var, chunk_size=chunk_size)
        ctrl_eval = _summarize_group_holdout(ctrl_tpls, holdout_t, skey, ctrl_var, chunk_size=chunk_size)

        mb = main_eval["best_of_16"]
        bb = base_eval["best_of_16"]
        cb = ctrl_eval["best_of_16"]

        # 计算与 Base 的 carry_post4 z-score 以及与 Ctrl 的差异 z-score
        se_carry_pool = math.sqrt(max(1e-16, mb["carry_post4_se"] ** 2 + bb["carry_post4_se"] ** 2))
        z_carry_post4_vs_base = (mb["carry_post4"] - bb["carry_post4"]) / se_carry_pool

        se_carry_ctrl = math.sqrt(max(1e-16, mb["carry_post4_se"] ** 2 + cb["carry_post4_se"] ** 2))
        z_carry_post4_vs_ctrl = (mb["carry_post4"] - cb["carry_post4"]) / se_carry_ctrl

        z_hw_vs_ctrl = (abs(mb["hw_mean"] - 128.0) - abs(cb["hw_mean"] - 128.0)) / math.sqrt(
            max(1e-12, mb["hw_se"] ** 2 + cb["hw_se"] ** 2)
        )
        z_clz_vs_ctrl = (mb["clz_mean"] - cb["clz_mean"]) / math.sqrt(
            max(1e-12, mb["clz_se"] ** 2 + cb["clz_se"] ** 2)
        )
        z_btc_lz_vs_ctrl = (mb["btc_lz_mean"] - cb["btc_lz_mean"]) / math.sqrt(
            max(1e-12, mb["btc_lz_se"] ** 2 + cb["btc_lz_se"] ** 2)
        )
        z_bit_vs_ctrl = (mb["max_bit_bias"] - cb["max_bit_bias"]) / math.sqrt(
            max(1e-12, mb["se_best_bit"] ** 2 + cb["se_best_bit"] ** 2)
        )

        # 对齐输出最优统计量的 z_vs_ctrl
        if abs(mb["z_hw"]) >= max(abs(mb["z_clz"]), abs(mb["z_btc_lz"]), mb["z_bit_max"]):
            z_vs_ctrl_primary = z_hw_vs_ctrl
        elif abs(mb["z_clz"]) >= max(abs(mb["z_btc_lz"]), mb["z_bit_max"]):
            z_vs_ctrl_primary = z_clz_vs_ctrl
        elif abs(mb["z_btc_lz"]) >= mb["z_bit_max"]:
            z_vs_ctrl_primary = z_btc_lz_vs_ctrl
        else:
            z_vs_ctrl_primary = z_bit_vs_ctrl

        # 若任意模板 |z| >= 4.5，自动追加 N_holdout2 独立二次验证
        holdout2_check = None
        if mb["max_abs_z_output"] >= 4.5 and n_holdout2 > 0:
            if holdout2_t is None:
                h2_np = build_holdout_nonces(n_holdout2, SEED_HOLDOUT2, exclude=holdout_np)
                holdout2_t = torch.from_numpy(h2_np).to(device=device, dtype=torch.int64)
            best_idx = int(
                max(range(top_k), key=lambda i: main_eval["top_16_templates"][i]["max_abs_z_output"])
            )
            holdout2_check = evaluate_template_holdout(
                main_tpls[best_idx], holdout2_t, skey, main_var, chunk_size=chunk_size
            )
            print(
                f"  [Holdout2 R={skey}] max_abs_z={holdout2_check['max_abs_z_output']:.3f} "
                f"(primary={mb['max_abs_z_output']:.3f})",
                flush=True,
            )

        elapsed_sk = time.time() - t_sk
        print(
            f"  [R={skey}] Main best z_out={mb['max_abs_z_output']:.3f} "
            f"(z_hw={mb['z_hw']:.2f}, z_clz={mb['z_clz']:.2f}, z_btc={mb['z_btc_lz']:.2f}, z_bit={mb['z_bit_max']:.2f}), "
            f"carry_post4={mb['carry_post4']:.5f} (base={bb['carry_post4']:.5f}, z_c={z_carry_post4_vs_base:.2f}), "
            f"Ctrl z_out={cb['max_abs_z_output']:.3f}, z_vs_ctrl={z_vs_ctrl_primary:.2f} | {elapsed_sk:.1f}s",
            flush=True,
        )

        sweep_results[skey] = {
            "rounds": sweep_rounds_count(skey),
            "is_double_sha256": skey == "double64",
            "search_meta": {"main": main_search_meta, "ctrl": ctrl_search_meta},
            "main": main_eval,
            "base_rand_midstate": base_eval,
            "ctrl_rand_rotr_k": ctrl_eval,
            "comparisons": {
                "z_carry_post4_vs_base": z_carry_post4_vs_base,
                "z_carry_post4_vs_ctrl": z_carry_post4_vs_ctrl,
                "z_hw_vs_ctrl": z_hw_vs_ctrl,
                "z_clz_vs_ctrl": z_clz_vs_ctrl,
                "z_btc_lz_vs_ctrl": z_btc_lz_vs_ctrl,
                "z_bit_vs_ctrl": z_bit_vs_ctrl,
                "z_vs_ctrl": z_vs_ctrl_primary,
            },
            "holdout2_verification": holdout2_check,
            "elapsed_sec": elapsed_sk,
        }

        # 每个扫频点落盘一次，断点安全
        _atomic_write_json(out_dir / "p5_sweep_summary.json", {"sweeps": sweep_results})

    total_elapsed = time.time() - t_start
    run_meta = {
        "experiment": "P5_BTC_320bit_Weak_Midstate",
        "device": str(device),
        "hardware_name": hw_name,
        "fp32_lock": fp32_info,
        "seeds": {
            "SEED_TRAIN": SEED_TRAIN,
            "SEED_HOLDOUT": SEED_HOLDOUT,
            "SEED_HOLDOUT2": SEED_HOLDOUT2,
            "SEED_CTRL": SEED_CTRL,
            "SEED_BASE": SEED_BASE,
        },
        "nonce_isolation": {
            "train_range": "[0, 2^30)",
            "holdout_range": "[2^30, 2^32)",
            "n_train_pool": n_train_pool,
            "n_holdout": n_holdout,
            "n_holdout2": n_holdout2,
            "overlap": 0,
        },
        "ctrl_variant": {
            "S0_rot": list(ctrl_var.S0_rot),
            "S1_rot": list(ctrl_var.S1_rot),
            "s0_rot": list(ctrl_var.s0_rot),
            "s0_shr": ctrl_var.s0_shr,
            "s1_rot": list(ctrl_var.s1_rot),
            "s1_shr": ctrl_var.s1_shr,
            "k_first8_hex": [f"0x{x:08x}" for x in ctrl_var.k[:8]],
        },
        "search_config": {
            "pop_size": pop_size,
            "generations": generations,
            "grad_steps": grad_steps,
            "top_k": top_k,
        },
        "total_elapsed_sec": total_elapsed,
    }
    summary = {"sweeps": sweep_results, "meta": run_meta}
    _atomic_write_json(out_dir / "p5_sweep_summary.json", summary)
    _atomic_write_json(out_dir / "p5_run_meta.json", run_meta)
    print(f"\n[P5] All sweeps finished in {total_elapsed:.1f}s -> {out_dir}", flush=True)
    return summary, run_meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P5 BTC Weak Midstate Resonance Probe")
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--sweeps", type=str, default=",".join(SWEEP_KEYS))
    p.add_argument("--n_train_pool", type=int, default=8192)
    p.add_argument("--n_holdout", type=int, default=262144)
    p.add_argument("--n_holdout2", type=int, default=1048576)
    p.add_argument("--pop_size", type=int, default=48)
    p.add_argument("--generations", type=int, default=18)
    p.add_argument("--grad_steps", type=int, default=32)
    p.add_argument("--top_k", type=int, default=16)
    p.add_argument("--chunk_size", type=int, default=65536)
    p.add_argument("--cpu_only", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    keys = [x.strip() for x in args.sweeps.split(",") if x.strip()]
    run_full_experiment(
        out_dir=Path(args.out_dir),
        sweep_keys=keys,
        n_train_pool=args.n_train_pool,
        n_holdout=args.n_holdout,
        n_holdout2=args.n_holdout2,
        pop_size=args.pop_size,
        generations=args.generations,
        grad_steps=args.grad_steps,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        cpu_only=args.cpu_only,
    )
