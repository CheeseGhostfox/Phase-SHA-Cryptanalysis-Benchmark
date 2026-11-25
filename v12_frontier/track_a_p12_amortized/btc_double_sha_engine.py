"""
P12 精确 BTC Double-SHA256 向量化引擎与逐门/逐比特摊还计费器。

严格遵循 g:\\SHA-LM\\v12_frontier\\track_a_p12_amortized\\SPEC.md 契约：
1. 浮点张量强制纯 IEEE-754 FP32（torch.float32, allow_tf32=False），
   精确 SHA-256 与 Double-SHA256 算术全程使用 uint32-in-int64（& 0xFFFFFFFF）。
2. 精确建模真实比特币 80-byte Block Header 三段式 Double-SHA256：
   - Stage 0: Hash1-Chunk1 (64B: nVersion, hashPrevBlock, hashMerkleRoot[0..27])
   - Stage 1: Hash1-Chunk2 (64B: hashMerkleRoot[28..31], nTime, nBits, nNonce,
              W1[4]=0x80000000, W1[5..14]=0, W1[15]=640)
   - Stage 2: Hash2 (64B: W2[0..7]=hash1[0..7], W2[8]=0x80000000, W2[9..14]=0, W2[15]=256)
3. 比特级 GE（Gate Equivalent）与算子级双重摊还计费器：
   - 严格采用 SPEC.md §1.2(B) 固定 GE 权重表（FA=11, HA=5, ADD32=346, Ch=256, Maj=384, Sigma=192）；
   - 严格执行静态零常量折叠（W1[5..14]=0, W2[9..14]=0 等零加数/零移位折算 0 GE）；
   - 结合“输入状态元组唯一数 u_inputs”与“比特位可变掩码 bit_active”，同时精确捕捉
     跨输入多路字复用（如 AsicBoost）与比特级静默门（进位吸收/掩码锁定），严禁重复计费。
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

if sys.platform == "win32" or "--cpu_only" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch

MASK32: int = 0xFFFFFFFF


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
    """锁定纯 IEEE-754 FP32，严禁 TF32 截断。"""
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(torch.float32)
    matmul_tf32 = False
    cudnn_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    if hasattr(torch.backends, "cudnn"):
        cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    return {
        "default_dtype": str(torch.get_default_dtype()),
        "matmul_allow_tf32": matmul_tf32,
        "cudnn_allow_tf32": cudnn_tf32,
    }


@dataclass(frozen=True)
class SHA256Variant:
    """SHA-256 标准算子或谱隙对齐随机 ROTR/K 对照算子。"""

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


def _cyclic_min_gap(rots: Sequence[int]) -> int:
    s = sorted(int(r) % 32 for r in rots)
    gaps = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    gaps.append(32 - s[-1] + s[0])
    return min(gaps)


def make_spectral_aligned_ctrl_variant(
    seed: int = 20261125,
    min_gap: int = 4,
    name: str = "CTRL_RAND_ROTR_K",
) -> SHA256Variant:
    """
    按 SPEC §3.2 构造谱隙对齐的同构随机 ROTR/K/IV 对照密码：
    保持 3-抽头互异且最小环形间距 >= min_gap，随机采样 K_ctrl[0..63] 与 IV_ctrl[0..7]。
    """
    rng = np.random.default_rng(seed)

    def _sample_rot3(forbid: Tuple[int, int, int]) -> Tuple[int, int, int]:
        while True:
            cand = tuple(sorted(int(x) for x in rng.choice(np.arange(1, 32), size=3, replace=False)))
            if cand != forbid and _cyclic_min_gap(cand) >= min_gap:
                return cand  # type: ignore[return-value]

    def _sample_rot2_shr(forbid_rot: Tuple[int, int], forbid_shr: int) -> Tuple[Tuple[int, int], int]:
        while True:
            rots = tuple(sorted(int(x) for x in rng.choice(np.arange(1, 32), size=2, replace=False)))
            shr = int(rng.integers(2, 24))
            if rots != forbid_rot and shr != forbid_shr and _cyclic_min_gap(rots) >= min_gap:
                return rots, shr  # type: ignore[return-value]

    S0_rot = _sample_rot3((2, 13, 22))
    S1_rot = _sample_rot3((6, 11, 25))
    s0_rot, s0_shr = _sample_rot2_shr((7, 18), 3)
    s1_rot, s1_shr = _sample_rot2_shr((17, 19), 10)
    iv_ctrl = tuple(int(x) for x in rng.integers(0, 2**32, size=8, dtype=np.uint64))
    k_ctrl = tuple(int(x) for x in rng.integers(0, 2**32, size=64, dtype=np.uint64))
    return SHA256Variant(
        name=name,
        iv=iv_ctrl,
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
    v = x & MASK32
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    return ((v * 0x01010101) & MASK32) >> 24


def sigma0_upper(x: torch.Tensor, v: SHA256Variant) -> torch.Tensor:
    r0, r1, r2 = v.S0_rot
    return (rotr32(x, r0) ^ rotr32(x, r1) ^ rotr32(x, r2)) & MASK32


def sigma1_upper(x: torch.Tensor, v: SHA256Variant) -> torch.Tensor:
    r0, r1, r2 = v.S1_rot
    return (rotr32(x, r0) ^ rotr32(x, r1) ^ rotr32(x, r2)) & MASK32


def sigma0_lower(x: torch.Tensor, v: SHA256Variant) -> torch.Tensor:
    r0, r1 = v.s0_rot
    return (rotr32(x, r0) ^ rotr32(x, r1) ^ shr32(x, v.s0_shr)) & MASK32


def sigma1_lower(x: torch.Tensor, v: SHA256Variant) -> torch.Tensor:
    r0, r1 = v.s1_rot
    return (rotr32(x, r0) ^ rotr32(x, r1) ^ shr32(x, v.s1_shr)) & MASK32


# ---------------------------------------------------------------------------
# 真实 BTC 80-byte Header 打包与拆包
# ---------------------------------------------------------------------------

def btc_header_bytes_to_blocks(
    headers: bytes | Sequence[bytes],
    device: torch.device | str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将 80-byte BTC 区块头转换为 (B, 16) Chunk1 与 Chunk2（含 320-bit 固定填充）。"""
    if isinstance(headers, (bytes, bytearray)):
        header_list = [bytes(headers)]
    else:
        header_list = [bytes(h) for h in headers]

    b = len(header_list)
    blk1 = torch.zeros((b, 16), dtype=torch.int64, device=device)
    blk2 = torch.zeros((b, 16), dtype=torch.int64, device=device)
    for i, hdr in enumerate(header_list):
        if len(hdr) != 80:
            raise ValueError(f"BTC header must be exactly 80 bytes, got {len(hdr)}")
        w1 = [int.from_bytes(hdr[4 * j : 4 * (j + 1)], "big") for j in range(16)]
        w2_head = [int.from_bytes(hdr[64 + 4 * j : 64 + 4 * (j + 1)], "big") for j in range(4)]
        blk1[i] = torch.tensor(w1, dtype=torch.int64, device=device)
        blk2[i, 0:4] = torch.tensor(w2_head, dtype=torch.int64, device=device)
    blk2[:, 4] = 0x80000000
    blk2[:, 15] = 640
    return blk1, blk2


def blocks_to_btc_header_bytes(blk1_w16: torch.Tensor, blk2_w16: torch.Tensor) -> List[bytes]:
    """将 (B, 16) Chunk1 与 Chunk2 还原为 80-byte 区块头。"""
    b1 = (blk1_w16.detach().cpu() & MASK32).tolist()
    b2 = (blk2_w16.detach().cpu() & MASK32).tolist()
    out: List[bytes] = []
    for row1, row2 in zip(b1, b2):
        chunks = [int(w).to_bytes(4, "big") for w in row1]
        chunks.extend(int(w).to_bytes(4, "big") for w in row2[:4])
        out.append(b"".join(chunks))
    return out


def pack_btc_chunk2_from_words(w012: torch.Tensor, nonces: torch.Tensor) -> torch.Tensor:
    """由 Chunk2 前三字 (merkle_tail, nTime, nBits) 与 nonce 构造 Chunk2 (B, 16)。"""
    bsz = nonces.shape[0]
    w16 = torch.zeros((bsz, 16), dtype=torch.int64, device=nonces.device)
    if w012.ndim == 1:
        w16[:, 0:3] = (w012[:3] & MASK32).unsqueeze(0).expand(bsz, 3)
    else:
        w16[:, 0:3] = w012[:, :3] & MASK32
    w16[:, 3] = nonces & MASK32
    w16[:, 4] = 0x80000000
    w16[:, 15] = 640
    return w16


def pack_outer_block_from_h_inner(h_inner: torch.Tensor) -> torch.Tensor:
    """构造 Double-SHA256 外层 Hash2 256-bit 固定填充输入块 (B, 16)。"""
    bsz = h_inner.shape[0]
    w16 = torch.zeros((bsz, 16), dtype=torch.int64, device=h_inner.device)
    w16[:, 0:8] = h_inner & MASK32
    w16[:, 8] = 0x80000000
    w16[:, 15] = 256
    return w16


def digest_words_to_bytes(digest_words: torch.Tensor) -> bytes:
    """将单个 (8,) uint32-in-int64 摘要转为 32 字节大端哈希。"""
    if digest_words.shape != (8,):
        raise ValueError(f"Expected (8,), got {tuple(digest_words.shape)}")
    vals = (digest_words.detach().cpu() & MASK32).tolist()
    return b"".join(int(v).to_bytes(4, "big") for v in vals)


# ---------------------------------------------------------------------------
# 精确单块压缩与完整 Double-SHA256 轨迹引擎
# ---------------------------------------------------------------------------

@dataclass
class CompressionTrace:
    """单块 R 轮精确压缩轨迹（全部张量均为 uint32-in-int64）。"""

    A_full: torch.Tensor      # (B, R + 4)
    E_full: torch.Tensor      # (B, R + 4)
    W_full: torch.Tensor      # (B, R)
    S1_hist: torch.Tensor     # (B, R)
    Ch_hist: torch.Tensor     # (B, R)
    S0_hist: torch.Tensor     # (B, R)
    Maj_hist: torch.Tensor    # (B, R)
    P1_hist: torch.Tensor     # (B, R) = h + S1
    P2_hist: torch.Tensor     # (B, R) = p1 + Ch
    P3_hist: torch.Tensor     # (B, R) = p2 + K[t]
    T1_hist: torch.Tensor     # (B, R) = p3 + W[t]
    T2_hist: torch.Tensor     # (B, R) = S0 + Maj
    S_R: torch.Tensor         # (B, 8)
    digest: torch.Tensor      # (B, 8)


@dataclass
class BTCDoubleSHA256Trace:
    """BTC 80-byte Header 三段式 Double-SHA256 精确轨迹。"""

    blk1: CompressionTrace    # Hash1-Chunk1
    blk2: CompressionTrace    # Hash1-Chunk2
    outer: CompressionTrace   # Hash2
    midstate: torch.Tensor    # (B, 8)
    inner_hash: torch.Tensor  # (B, 8)
    final_hash: torch.Tensor  # (B, 8)


def sha256_compress_trace(
    init_state: torch.Tensor,
    w16: torch.Tensor,
    rounds: int = 64,
    variant: Optional[SHA256Variant] = None,
    record_gates: bool = True,
) -> CompressionTrace:
    """向量化精确 SHA-256 压缩（支持 1..64 轮与任意初始态/变体）。"""
    if variant is None:
        variant = make_main_variant()
    if w16.ndim != 2 or w16.shape[1] != 16:
        raise ValueError(f"Expected w16 shape (B, 16), got {tuple(w16.shape)}")
    if not (1 <= rounds <= 64):
        raise ValueError(f"rounds must be in [1, 64], got {rounds}")

    bsz = w16.shape[0]
    device = w16.device
    s0 = init_state.unsqueeze(0).expand(bsz, 8) if init_state.ndim == 1 else init_state
    s0 = s0 & MASK32
    k_vec = torch.tensor(variant.k[:rounds], dtype=torch.int64, device=device)

    w_cols = [w16[:, t] & MASK32 for t in range(min(16, rounds))]
    for t in range(16, rounds):
        wt = (
            sigma1_lower(w_cols[t - 2], variant)
            + w_cols[t - 7]
            + sigma0_lower(w_cols[t - 15], variant)
            + w_cols[t - 16]
        ) & MASK32
        w_cols.append(wt)
    w_full = torch.stack(w_cols, dim=1)

    a_cols = [s0[:, 3], s0[:, 2], s0[:, 1], s0[:, 0]]
    e_cols = [s0[:, 7], s0[:, 6], s0[:, 5], s0[:, 4]]

    s1_list: List[torch.Tensor] = []
    ch_list: List[torch.Tensor] = []
    s0_list: List[torch.Tensor] = []
    maj_list: List[torch.Tensor] = []
    p1_list: List[torch.Tensor] = []
    p2_list: List[torch.Tensor] = []
    p3_list: List[torch.Tensor] = []
    t1_list: List[torch.Tensor] = []
    t2_list: List[torch.Tensor] = []

    for t in range(rounds):
        a_t, a_tm1, a_tm2, a_tm3 = a_cols[t + 3], a_cols[t + 2], a_cols[t + 1], a_cols[t]
        e_t, e_tm1, e_tm2, e_tm3 = e_cols[t + 3], e_cols[t + 2], e_cols[t + 1], e_cols[t]

        s1 = sigma1_upper(e_t, variant)
        ch = ch32(e_t, e_tm1, e_tm2)
        p1 = (e_tm3 + s1) & MASK32
        p2 = (p1 + ch) & MASK32
        p3 = (p2 + k_vec[t]) & MASK32
        t1 = (p3 + w_cols[t]) & MASK32

        s0_u = sigma0_upper(a_t, variant)
        mj = maj32(a_t, a_tm1, a_tm2)
        t2 = (s0_u + mj) & MASK32

        e_cols.append((a_tm3 + t1) & MASK32)
        a_cols.append((t1 + t2) & MASK32)

        if record_gates:
            s1_list.append(s1)
            ch_list.append(ch)
            s0_list.append(s0_u)
            maj_list.append(mj)
            p1_list.append(p1)
            p2_list.append(p2)
            p3_list.append(p3)
            t1_list.append(t1)
            t2_list.append(t2)

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
    digest = (s_r + s0) & MASK32

    if record_gates:
        s1_hist = torch.stack(s1_list, dim=1)
        ch_hist = torch.stack(ch_list, dim=1)
        s0_hist = torch.stack(s0_list, dim=1)
        maj_hist = torch.stack(maj_list, dim=1)
        p1_hist = torch.stack(p1_list, dim=1)
        p2_hist = torch.stack(p2_list, dim=1)
        p3_hist = torch.stack(p3_list, dim=1)
        t1_hist = torch.stack(t1_list, dim=1)
        t2_hist = torch.stack(t2_list, dim=1)
    else:
        empty = torch.empty((bsz, 0), dtype=torch.int64, device=device)
        s1_hist = ch_hist = s0_hist = maj_hist = empty
        p1_hist = p2_hist = p3_hist = t1_hist = t2_hist = empty

    return CompressionTrace(
        A_full=a_full,
        E_full=e_full,
        W_full=w_full,
        S1_hist=s1_hist,
        Ch_hist=ch_hist,
        S0_hist=s0_hist,
        Maj_hist=maj_hist,
        P1_hist=p1_hist,
        P2_hist=p2_hist,
        P3_hist=p3_hist,
        T1_hist=t1_hist,
        T2_hist=t2_hist,
        S_R=s_r,
        digest=digest,
    )


def btc_double_sha256_trace(
    blk1_w16: torch.Tensor,
    blk2_w16: torch.Tensor,
    rounds_blk1: int = 64,
    rounds_blk2: int = 64,
    rounds_outer: int = 64,
    variant: Optional[SHA256Variant] = None,
    record_gates: bool = True,
) -> BTCDoubleSHA256Trace:
    """执行完整 BTC 三段式 Double-SHA256（Chunk1 -> Chunk2 -> Hash2）并返回精确轨迹。"""
    if variant is None:
        variant = make_main_variant()
    iv_t = torch.tensor(variant.iv, dtype=torch.int64, device=blk1_w16.device)
    t1 = sha256_compress_trace(iv_t, blk1_w16, rounds=rounds_blk1, variant=variant, record_gates=record_gates)
    t2 = sha256_compress_trace(t1.digest, blk2_w16, rounds=rounds_blk2, variant=variant, record_gates=record_gates)
    w_outer = pack_outer_block_from_h_inner(t2.digest)
    t3 = sha256_compress_trace(iv_t, w_outer, rounds=rounds_outer, variant=variant, record_gates=record_gates)
    return BTCDoubleSHA256Trace(
        blk1=t1,
        blk2=t2,
        outer=t3,
        midstate=t1.digest,
        inner_hash=t2.digest,
        final_hash=t3.digest,
    )


def btc_double_sha256_hashlib_check(header_80b: bytes) -> torch.Tensor:
    """用 Python 标准库 hashlib 计算真实 BTC Double-SHA256 摘要字（零容忍对照）。"""
    d = hashlib.sha256(hashlib.sha256(header_80b).digest()).digest()
    words = [int.from_bytes(d[4 * i : 4 * (i + 1)], "big") for i in range(8)]
    return torch.tensor(words, dtype=torch.int64)


# ---------------------------------------------------------------------------
# 逐门/逐比特 GE 权重与静态常量折叠计费器（SPEC §1.2 精确实现）
# ---------------------------------------------------------------------------

GE_WEIGHTS_DICT: Dict[str, int] = {
    "XOR2": 3,
    "AND2": 2,
    "OR2": 2,
    "NOT": 1,
    "FA": 11,
    "HA": 5,
    "ADD32": 346,          # 31 * FA(11) + HA(5)
    "CH32": 256,           # 32 * 8
    "MAJ32": 384,          # 32 * 12
    "SIGMA_UPPER": 192,    # 64 * XOR2(3)
    "SIGMA_LOWER": 192,    # 64 * XOR2(3)
}


def constant_add32_ge(const_val: int) -> int:
    """
    计算 ADD32(x, C) 在编译期常量 C 已知时的折叠门数：
    - C == 0 -> 0 GE
    - bit < ctz(C) -> 0 GE (直接透传)
    - bit == ctz(C) -> 1 GE (NOT)
    - bit > ctz(C) -> 5 GE (HA: 与常量 0/1 相加退化为半加器)
    """
    c = int(const_val) & MASK32
    if c == 0:
        return 0
    lsb = (c & -c).bit_length() - 1
    return 1 + (31 - lsb) * 5


def _bit_active_mask_32(x: torch.Tensor) -> int:
    """返回张量 x (B,) 在 batch 内发生过取反的 32-bit 掩码整数。"""
    if x.shape[0] <= 1:
        return 0
    diff = (x ^ x[0:1]) & MASK32
    # 按列 bitwise-or：利用 numpy uint32 bitwise_or.reduce 极速且精确
    arr = diff.detach().cpu().numpy().astype(np.uint32)
    return int(np.bitwise_or.reduce(arr))


def _add32_active_bit_ge(mask_x: int, mask_y: int) -> float:
    """
    已知操作数 x, y 的变化掩码 mask_x, mask_y，计算 32-bit 行波加法器的动态 vs 静态门数：
    若某比特位 b 的输入 x_b, y_b 与进位 c_in_b 均不变，则该位加法器仅计 1/M（由外层缩放），
    返回 (active_ge_sum, static_ge_sum)。
    """
    active = 0
    static = 0
    carry_active = 0
    for b in range(32):
        xb = (mask_x >> b) & 1
        yb = (mask_y >> b) & 1
        ge_b = 5 if b == 0 else 11
        if xb or yb or carry_active:
            active += ge_b
            carry_active = 1
        else:
            static += ge_b
    return float(active), float(static)


def _sigma_active_bit_ge(mask_x: int, rots: Sequence[int], shr: int | None = None) -> Tuple[float, float]:
    """计算 Sigma/sigma 的 32-bit 各位置 2-XOR (6 GE/bit) 动态与静态门数。"""
    out_mask = 0
    for r in rots:
        out_mask |= ((mask_x >> r) | ((mask_x & ((1 << r) - 1)) << (32 - r))) & MASK32
    if shr is not None:
        out_mask |= (mask_x >> shr) & MASK32
    n_active_bits = int(out_mask).bit_count()
    return float(n_active_bits * 6), float((32 - n_active_bits) * 6)


def _ch_active_bit_ge(mask_e: int, mask_f: int, mask_g: int) -> Tuple[float, float]:
    act_mask = (mask_e | mask_f | mask_g) & MASK32
    n_act = int(act_mask).bit_count()
    return float(n_act * 8), float((32 - n_act) * 8)


def _maj_active_bit_ge(mask_a: int, mask_b: int, mask_c: int) -> Tuple[float, float]:
    act_mask = (mask_a | mask_b | mask_c) & MASK32
    n_act = int(act_mask).bit_count()
    return float(n_act * 12), float((32 - n_act) * 12)


def _const_add32_active_bit_ge(mask_x: int, const_val: int) -> Tuple[float, float]:
    c = int(const_val) & MASK32
    if c == 0:
        return 0.0, 0.0
    lsb = (c & -c).bit_length() - 1
    active = 0
    static = 0
    carry_active = 0
    for b in range(lsb, 32):
        xb = (mask_x >> b) & 1
        ge_b = 1 if b == lsb else 5
        if xb or carry_active:
            active += ge_b
            carry_active = 1
        else:
            static += ge_b
    return float(active), float(static)


@dataclass
class StageCostBreakdown:
    stage_name: str
    expander_ge: float
    compression_ge: float
    feedforward_ge: float
    total_ge: float
    active_gate_ratio: float


@dataclass
class AmortizedCostReport:
    """SPEC §1.2 & §4.2 严格批量摊还报告。"""

    label: str
    batch_size: int
    total_ge: float
    amortized_ge_per_hash: float
    naive_ge_per_hash: float
    saving_pct_vs_noab: float
    chunk1_ge_per_hash: float
    chunk2_exp_ge_per_hash: float
    chunk2_comp_ge_per_hash: float
    hash2_exp_ge_per_hash: float
    hash2_comp_ge_per_hash: float
    p_valid: float = 1.0
    setup_ge_per_hash: float = 0.0


def _static_padding_word_info(stage: str) -> List[Optional[int]]:
    """
    返回固定填充阶段 W[0..15] 中编译期恒定字（None 表示非常量，int 表示固定常量值）。
    - Chunk2: W[4]=0x80000000, W[5..14]=0, W[15]=640
    - Hash2:  W[8]=0x80000000, W[9..14]=0, W[15]=256
    """
    info: List[Optional[int]] = [None] * 16
    if stage == "chunk2":
        info[4] = 0x80000000
        for j in range(5, 15):
            info[j] = 0
        info[15] = 640
    elif stage == "hash2":
        info[8] = 0x80000000
        for j in range(9, 15):
            info[j] = 0
        info[15] = 256
    return info


def _propagate_static_zero_schedule(stage: str, rounds: int) -> List[Optional[int]]:
    """跟踪经静态折叠后哪些 W[t] 仍为全零/编译期常量（严格只折算 100% 确定性结构常量）。"""
    const_w = _static_padding_word_info(stage)
    for _ in range(16, rounds):
        # 一旦依赖任何非常量字即标为 None
        t = len(const_w)
        c_m2 = const_w[t - 2]
        c_m7 = const_w[t - 7]
        c_m15 = const_w[t - 15]
        c_m16 = const_w[t - 16]
        if c_m2 is not None and c_m7 is not None and c_m15 is not None and c_m16 is not None:
            # 标准常数即可编译期求值，这里仅在 0 时折叠
            if c_m2 == 0 and c_m7 == 0 and c_m15 == 0 and c_m16 == 0:
                const_w.append(0)
            else:
                const_w.append(None)
        else:
            const_w.append(None)
    return const_w


def _count_unique_cols(cols_np: Sequence[np.ndarray]) -> int:
    """精确统计若干等长 (B,) uint32/int64 数组按行拼接后的唯一元组个数（零崩溃且高速）。"""
    if not cols_np:
        return 1
    if len(cols_np) == 1:
        c = cols_np[0]
        if c.size <= 1 or bool(np.all(c == c[0])):
            return 1
        return len(set(c.tolist()))
    if all(c.size <= 1 or bool(np.all(c == c[0])) for c in cols_np):
        return 1
    return len(set(zip(*(c.tolist() for c in cols_np))))


def _bit_active_mask_np(col: np.ndarray) -> int:
    """返回 (B,) uint32/int64 数组在 batch 内发生过取反的 32-bit 掩码。"""
    if col.size <= 1:
        return 0
    diff = (col ^ col[0]).astype(np.uint32)
    return int(np.bitwise_or.reduce(diff))


def _score_stage_amortized_ge(
    tr: CompressionTrace,
    stage: str,
    variant: SHA256Variant,
    skip_first_rounds_when_single_midstate: int = 0,
    skip_tail_rounds_for_h7: bool = False,
) -> StageCostBreakdown:
    """
    在给定精确轨迹 (B, R) 上按 SPEC §1.2(B) 计算该阶段全部哈希的合计 GE（再由外层除以 B）。
    """
    bsz = tr.W_full.shape[0]
    rounds = tr.W_full.shape[1]
    effective_rounds = 61 if (skip_tail_rounds_for_h7 and rounds == 64) else rounds
    const_w = _propagate_static_zero_schedule(stage, rounds)

    # 一次性搬运至 NumPy 避免逐轮 GPU->CPU 同步与 Windows torch.unique 内存访问异常
    w_np = tr.W_full.detach().cpu().numpy()
    a_np = tr.A_full.detach().cpu().numpy()
    e_np = tr.E_full.detach().cpu().numpy()
    s1_np = tr.S1_hist.detach().cpu().numpy()
    ch_np = tr.Ch_hist.detach().cpu().numpy()
    s0_np = tr.S0_hist.detach().cpu().numpy()
    mj_np = tr.Maj_hist.detach().cpu().numpy()
    p1_np = tr.P1_hist.detach().cpu().numpy()
    p2_np = tr.P2_hist.detach().cpu().numpy()
    p3_np = tr.P3_hist.detach().cpu().numpy()
    t1_np = tr.T1_hist.detach().cpu().numpy()
    t2_np = tr.T2_hist.detach().cpu().numpy()
    sr_np = tr.S_R.detach().cpu().numpy()
    dig_np = tr.digest.detach().cpu().numpy()

    # 1. 消息扩展 Expander (t = 16 .. effective_rounds - 1)
    exp_ge = 0.0
    exp_full_ge = 0.0
    for t in range(16, effective_rounds):
        c_m2 = const_w[t - 2]
        c_m7 = const_w[t - 7]
        c_m15 = const_w[t - 15]
        c_m16 = const_w[t - 16]

        dep_cols: List[np.ndarray] = []
        if c_m2 != 0:
            dep_cols.append(w_np[:, t - 2])
        if c_m7 != 0:
            dep_cols.append(w_np[:, t - 7])
        if c_m15 != 0:
            dep_cols.append(w_np[:, t - 15])
        if c_m16 != 0:
            dep_cols.append(w_np[:, t - 16])

        u_step = _count_unique_cols(dep_cols)

        m2_mask = 0 if c_m2 is not None else _bit_active_mask_np(w_np[:, t - 2])
        m7_mask = 0 if c_m7 is not None else _bit_active_mask_np(w_np[:, t - 7])
        m15_mask = 0 if c_m15 is not None else _bit_active_mask_np(w_np[:, t - 15])
        m16_mask = 0 if c_m16 is not None else _bit_active_mask_np(w_np[:, t - 16])

        step_active = 0.0
        step_static = 0.0

        if c_m2 != 0 and c_m2 is None:
            a, s = _sigma_active_bit_ge(m2_mask, variant.s1_rot, variant.s1_shr)
            step_active += a
            step_static += s
        if c_m15 != 0 and c_m15 is None:
            a, s = _sigma_active_bit_ge(m15_mask, variant.s0_rot, variant.s0_shr)
            step_active += a
            step_static += s

        terms: List[Tuple[Optional[int], int]] = []
        if c_m2 != 0:
            if c_m2 is not None:
                val = ((c_m2 >> variant.s1_rot[0]) | ((c_m2 & ((1 << variant.s1_rot[0]) - 1)) << (32 - variant.s1_rot[0]))) ^ (
                    (c_m2 >> variant.s1_rot[1]) | ((c_m2 & ((1 << variant.s1_rot[1]) - 1)) << (32 - variant.s1_rot[1]))
                ) ^ (c_m2 >> variant.s1_shr)
                terms.append((val & MASK32, 0))
            else:
                terms.append((None, m2_mask if m2_mask == 0 else MASK32))
        if c_m7 != 0:
            terms.append((c_m7, m7_mask))
        if c_m15 != 0:
            if c_m15 is not None:
                val = ((c_m15 >> variant.s0_rot[0]) | ((c_m15 & ((1 << variant.s0_rot[0]) - 1)) << (32 - variant.s0_rot[0]))) ^ (
                    (c_m15 >> variant.s0_rot[1]) | ((c_m15 & ((1 << variant.s0_rot[1]) - 1)) << (32 - variant.s0_rot[1]))
                ) ^ (c_m15 >> variant.s0_shr)
                terms.append((val & MASK32, 0))
            else:
                terms.append((None, m15_mask if m15_mask == 0 else MASK32))
        if c_m16 != 0:
            terms.append((c_m16, m16_mask))

        if len(terms) >= 2:
            cur_const, cur_mask = terms[0]
            for nxt_const, nxt_mask in terms[1:]:
                if cur_const is not None and nxt_const is not None:
                    cur_const = (cur_const + nxt_const) & MASK32
                elif cur_const is not None:
                    a, s = _const_add32_active_bit_ge(nxt_mask, cur_const)
                    step_active += a
                    step_static += s
                    cur_const = None
                    cur_mask = nxt_mask if nxt_mask == 0 else MASK32
                elif nxt_const is not None:
                    a, s = _const_add32_active_bit_ge(cur_mask, nxt_const)
                    step_active += a
                    step_static += s
                    cur_mask = cur_mask if cur_mask == 0 else MASK32
                else:
                    a, s = _add32_active_bit_ge(cur_mask, nxt_mask)
                    step_active += a
                    step_static += s
                    cur_mask = (cur_mask | nxt_mask) if (cur_mask == 0 and nxt_mask == 0) else MASK32

        exp_ge += u_step * step_active + 1.0 * step_static
        exp_full_ge += bsz * (step_active + step_static)

    # 2. 压缩轮 Compression (t = 0 .. effective_rounds - 1)
    comp_ge = 0.0
    comp_full_ge = 0.0
    for t in range(effective_rounds):
        e_t = e_np[:, t + 3]
        e_tm1 = e_np[:, t + 2]
        e_tm2 = e_np[:, t + 1]
        e_tm3 = e_np[:, t]
        a_t = a_np[:, t + 3]
        a_tm1 = a_np[:, t + 2]
        a_tm2 = a_np[:, t + 1]
        a_tm3 = a_np[:, t]
        w_t = w_np[:, t]
        k_t = variant.k[t]

        m_et = _bit_active_mask_np(e_t)
        m_et1 = _bit_active_mask_np(e_tm1)
        m_et2 = _bit_active_mask_np(e_tm2)
        m_et3 = _bit_active_mask_np(e_tm3)
        m_at = _bit_active_mask_np(a_t)
        m_at1 = _bit_active_mask_np(a_tm1)
        m_at2 = _bit_active_mask_np(a_tm2)
        m_at3 = _bit_active_mask_np(a_tm3)
        m_wt = _bit_active_mask_np(w_t)

        u_et = _count_unique_cols([e_t])
        a_s1, s_s1 = _sigma_active_bit_ge(m_et, variant.S1_rot)

        u_ch = _count_unique_cols([e_t, e_tm1, e_tm2])
        a_ch, s_ch = _ch_active_bit_ge(m_et, m_et1, m_et2)

        u_p1 = _count_unique_cols([e_tm3, e_t])
        m_s1 = _bit_active_mask_np(s1_np[:, t])
        a_p1, s_p1 = _add32_active_bit_ge(m_et3, m_s1)

        u_p2 = _count_unique_cols([e_tm3, e_tm2, e_tm1, e_t])
        m_p1 = _bit_active_mask_np(p1_np[:, t])
        m_ch = _bit_active_mask_np(ch_np[:, t])
        a_p2, s_p2 = _add32_active_bit_ge(m_p1, m_ch)

        m_p2 = _bit_active_mask_np(p2_np[:, t])
        a_p3, s_p3 = _const_add32_active_bit_ge(m_p2, k_t)

        if const_w[t] == 0:
            u_t1 = u_p2
            a_t1, s_t1 = 0.0, 0.0
        else:
            u_t1 = _count_unique_cols([e_tm3, e_tm2, e_tm1, e_t, w_t])
            m_p3 = _bit_active_mask_np(p3_np[:, t])
            a_t1, s_t1 = _add32_active_bit_ge(m_p3, m_wt)

        u_at = _count_unique_cols([a_t])
        a_s0, s_s0 = _sigma_active_bit_ge(m_at, variant.S0_rot)

        u_maj = _count_unique_cols([a_t, a_tm1, a_tm2])
        a_maj, s_maj = _maj_active_bit_ge(m_at, m_at1, m_at2)

        m_s0 = _bit_active_mask_np(s0_np[:, t])
        m_mj = _bit_active_mask_np(mj_np[:, t])
        a_t2, s_t2 = _add32_active_bit_ge(m_s0, m_mj)

        m_t1 = _bit_active_mask_np(t1_np[:, t])
        m_t2 = _bit_active_mask_np(t2_np[:, t])
        u_anew = _count_unique_cols([t1_np[:, t], t2_np[:, t]])
        a_anew, s_anew = _add32_active_bit_ge(m_t1, m_t2)

        u_enew = _count_unique_cols([a_tm3, t1_np[:, t]])
        a_enew, s_enew = _add32_active_bit_ge(m_at3, m_t1)

        round_ge = (
            u_et * a_s1 + s_s1
            + u_ch * a_ch + s_ch
            + u_p1 * a_p1 + s_p1
            + u_p2 * a_p2 + s_p2
            + u_p2 * a_p3 + s_p3
            + u_t1 * a_t1 + s_t1
            + u_at * a_s0 + s_s0
            + u_maj * a_maj + s_maj
            + u_maj * a_t2 + s_t2
            + u_anew * a_anew + s_anew
            + u_enew * a_enew + s_enew
        )
        round_full = (
            a_s1 + s_s1 + a_ch + s_ch + a_p1 + s_p1 + a_p2 + s_p2 + a_p3 + s_p3 + a_t1 + s_t1
            + a_s0 + s_s0 + a_maj + s_maj + a_t2 + s_t2 + a_anew + s_anew + a_enew + s_enew
        )
        _ = skip_first_rounds_when_single_midstate
        comp_ge += round_ge
        comp_full_ge += bsz * round_full

    # 3. Feed-forward (8 words)
    ff_words = 1 if (skip_tail_rounds_for_h7 and rounds == 64) else 8
    ff_ge = 0.0
    ff_full_ge = float(bsz * ff_words * 346)
    word_indices = [7] if ff_words == 1 else list(range(8))
    for wi in word_indices:
        sr_col = sr_np[:, wi]
        init_col = (dig_np[:, wi] - sr_col) & MASK32
        u_ff = _count_unique_cols([sr_col, init_col])
        a_ff, s_ff = _add32_active_bit_ge(_bit_active_mask_np(sr_col), _bit_active_mask_np(init_col))
        ff_ge += u_ff * a_ff + s_ff

    total = exp_ge + comp_ge + ff_ge
    full_total = exp_full_ge + comp_full_ge + ff_full_ge
    active_ratio = total / max(1.0, full_total)
    return StageCostBreakdown(
        stage_name=stage,
        expander_ge=exp_ge,
        compression_ge=comp_ge,
        feedforward_ge=ff_ge,
        total_ge=total,
        active_gate_ratio=active_ratio,
    )


def compute_exact_btc_amortized_report(
    traces: BTCDoubleSHA256Trace,
    label: str,
    variant: SHA256Variant,
    naive_baseline_ge_per_hash: Optional[float] = None,
    amortize_chunk1_over_nonces: int = 1 << 32,
    skip_outer_tail_h7: bool = False,
    p_valid: float = 1.0,
    setup_ge_total: float = 0.0,
) -> AmortizedCostReport:
    """
    严格按 SPEC §1.2 有效单哈希净成本公式计费：
    Cost_eff = (1 / p_valid) * (setup_ge_total / B + b1_eff + b2_total + b3_total) / B
    """
    bsz = traces.blk2.W_full.shape[0]
    b1 = _score_stage_amortized_ge(traces.blk1, "chunk1", variant)
    b2 = _score_stage_amortized_ge(traces.blk2, "chunk2", variant)
    b3 = _score_stage_amortized_ge(traces.outer, "hash2", variant, skip_tail_rounds_for_h7=skip_outer_tail_h7)

    # Chunk1 在矿机中每不同 midstate 按各自扫过的 nonce 数均摊
    chunk1_eff_total = b1.total_ge / float(max(1, amortize_chunk1_over_nonces))
    raw_total = (setup_ge_total / float(max(1, amortize_chunk1_over_nonces))) + chunk1_eff_total + b2.total_ge + b3.total_ge
    eff_total = raw_total / float(max(1e-12, p_valid))
    eff_per_hash = eff_total / float(bsz)

    if naive_baseline_ge_per_hash is None:
        naive_baseline_ge_per_hash = eff_per_hash
    saving_pct = (1.0 - eff_per_hash / float(naive_baseline_ge_per_hash)) * 100.0

    return AmortizedCostReport(
        label=label,
        batch_size=bsz,
        total_ge=eff_total,
        amortized_ge_per_hash=eff_per_hash,
        naive_ge_per_hash=float(naive_baseline_ge_per_hash),
        saving_pct_vs_noab=saving_pct,
        chunk1_ge_per_hash=(chunk1_eff_total / float(bsz)),
        chunk2_exp_ge_per_hash=(b2.expander_ge / float(bsz)),
        chunk2_comp_ge_per_hash=((b2.compression_ge + b2.feedforward_ge) / float(bsz)),
        hash2_exp_ge_per_hash=(b3.expander_ge / float(bsz)),
        hash2_comp_ge_per_hash=((b3.compression_ge + b3.feedforward_ge) / float(bsz)),
        p_valid=p_valid,
        setup_ge_per_hash=(setup_ge_total / float(max(1, amortize_chunk1_over_nonces) * bsz)),
    )
