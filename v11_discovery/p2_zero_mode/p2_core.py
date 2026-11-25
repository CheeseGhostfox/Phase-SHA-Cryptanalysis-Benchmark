"""P2「时空晶体等零面零模波包」核心引擎、对照组构建、晶体流形采样与统计检验模块。

严格只读复用 v10_ouroboros64/ouroboros_core/engine.py 的纯 FP32 与精确 uint32/int64 约定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from v10_ouroboros64.ouroboros_core.engine import (
    C_PLANES,
    IV_WORDS,
    K_WORDS,
    MASK32,
    enforce_fp32_determinism,
    popcnt32_torch,
    words_to_bits_lsb,
)
from v10_ouroboros64.ouroboros_net.arx_layers import (
    LOWER_SIGMA0_ROTS,
    LOWER_SIGMA0_SHR,
    LOWER_SIGMA1_ROTS,
    LOWER_SIGMA1_SHR,
    SIGMA0_ROTS,
    SIGMA1_ROTS,
    ARXRotationEdgeBlock,
)


ROUNDS_GRID: Tuple[int, ...] = (4, 6, 8, 10, 12, 16, 20)
K_GRID: Tuple[int, ...] = tuple(range(1, 13))


def _rotr32_t(x: torch.Tensor, n: int) -> torch.Tensor:
    return ((x >> n) | (x << (32 - n))) & MASK32


def _ch_t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return (x & y) ^ ((~x & MASK32) & z)


def _maj_t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return (x & y) ^ (x & z) ^ (y & z)


@dataclass(frozen=True)
class CipherConfig:
    """参数化 SHA-256 压缩拓扑配置（标准 SHA-256 或随机 ROTR/K 克隆对照）。"""

    name: str
    iv_words: Tuple[int, ...]
    k_words: Tuple[int, ...]
    sigma0_rots: Tuple[int, int, int]
    sigma1_rots: Tuple[int, int, int]
    lower_sigma0_rots: Tuple[int, int]
    lower_sigma0_shr: int
    lower_sigma1_rots: Tuple[int, int]
    lower_sigma1_shr: int
    is_control: bool = False


def make_standard_cipher_config() -> CipherConfig:
    """标准 SHA-256 精确配置。"""
    return CipherConfig(
        name="SHA256_STD",
        iv_words=tuple(int(x) for x in IV_WORDS),
        k_words=tuple(int(x) for x in K_WORDS),
        sigma0_rots=tuple(SIGMA0_ROTS),
        sigma1_rots=tuple(SIGMA1_ROTS),
        lower_sigma0_rots=tuple(LOWER_SIGMA0_ROTS),
        lower_sigma0_shr=int(LOWER_SIGMA0_SHR),
        lower_sigma1_rots=tuple(LOWER_SIGMA1_ROTS),
        lower_sigma1_shr=int(LOWER_SIGMA1_SHR),
        is_control=False,
    )


def make_random_control_cipher_config(seed: int = 20261125) -> CipherConfig:
    """构建 SPEC 2.2 随机 ROTR/K 同构克隆对照组（每组 3 个偏移量互异，随机 K[0..63]）。"""
    rng = np.random.default_rng(seed)

    def _sample_distinct_triple() -> Tuple[int, int, int]:
        vals = rng.choice(np.arange(1, 32), size=3, replace=False)
        return int(vals[0]), int(vals[1]), int(vals[2])

    s0 = _sample_distinct_triple()
    s1 = _sample_distinct_triple()
    ls0_r0, ls0_r1, ls0_shr = _sample_distinct_triple()
    ls1_r0, ls1_r1, ls1_shr = _sample_distinct_triple()
    k_rand = tuple(int(x) for x in rng.integers(0, 2**32, size=64, dtype=np.uint64))

    return CipherConfig(
        name=f"CTRL_RAND_SHA_seed{seed}",
        iv_words=tuple(int(x) for x in IV_WORDS),
        k_words=k_rand,
        sigma0_rots=s0,
        sigma1_rots=s1,
        lower_sigma0_rots=(ls0_r0, ls0_r1),
        lower_sigma0_shr=ls0_shr,
        lower_sigma1_rots=(ls1_r0, ls1_r1),
        lower_sigma1_shr=ls1_shr,
        is_control=True,
    )


class ParametricCrystalEngine:
    """支持标准 SHA-256 与随机 ROTR/K 克隆对照的纯 int64 位运算 GPU/CPU 时空晶体引擎。"""

    def __init__(
        self,
        cipher_cfg: Optional[CipherConfig] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cipher_cfg or make_standard_cipher_config()
        self.device = torch.device(
            device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        self.iv = torch.tensor(self.cfg.iv_words, device=self.device, dtype=torch.int64)
        self.k = torch.tensor(self.cfg.k_words, device=self.device, dtype=torch.int64)

    @torch.no_grad()
    def compress_exact(
        self,
        w0_15: torch.Tensor,
        rounds: int = 64,
        hash_only: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """精确前向压缩（与 v10 TorchCrystalEngine 1:1 对齐）。"""
        w_in = w0_15.to(device=self.device, dtype=torch.int64) & MASK32
        if w_in.ndim == 1:
            w_in = w_in.unsqueeze(0)
        bsz = int(w_in.shape[0])

        s0_r0, s0_r1, s0_r2 = self.cfg.sigma0_rots
        s1_r0, s1_r1, s1_r2 = self.cfg.sigma1_rots
        ls0_r0, ls0_r1 = self.cfg.lower_sigma0_rots
        ls0_shr = self.cfg.lower_sigma0_shr
        ls1_r0, ls1_r1 = self.cfg.lower_sigma1_rots
        ls1_shr = self.cfg.lower_sigma1_shr

        w_all: List[torch.Tensor] = [w_in[:, t] for t in range(min(16, rounds))]
        if rounds > 16:
            w_all = [w_in[:, t] for t in range(16)]
            if not hash_only:
                c_list: List[List[torch.Tensor]] = [
                    [torch.zeros(bsz, device=self.device, dtype=torch.int64) for _ in range(C_PLANES)]
                    for _ in range(rounds)
                ]
            for t in range(16, rounds):
                w2 = w_all[t - 2]
                w15 = w_all[t - 15]
                sig1 = _rotr32_t(w2, ls1_r0) ^ _rotr32_t(w2, ls1_r1) ^ (w2 >> ls1_shr)
                sig0 = _rotr32_t(w15, ls0_r0) ^ _rotr32_t(w15, ls0_r1) ^ (w15 >> ls0_shr)

                u0 = (sig1 + w_all[t - 7]) & MASK32
                u1 = (u0 + sig0) & MASK32
                wt = (u1 + w_all[t - 16]) & MASK32
                if not hash_only:
                    c_list[t][0] = sig1 ^ w_all[t - 7] ^ u0
                    c_list[t][1] = u0 ^ sig0 ^ u1
                    c_list[t][2] = u1 ^ w_all[t - 16] ^ wt
                w_all.append(wt)
        else:
            if not hash_only:
                c_list = [
                    [torch.zeros(bsz, device=self.device, dtype=torch.int64) for _ in range(C_PLANES)]
                    for _ in range(rounds)
                ]

        st = self.iv.unsqueeze(0).expand(bsz, 8).clone()
        a, b, c, d, e, f, g, h = [st[:, j] for j in range(8)]
        if not hash_only:
            s_list: List[torch.Tensor] = [st]

        for t in range(rounds):
            big_s1 = _rotr32_t(e, s1_r0) ^ _rotr32_t(e, s1_r1) ^ _rotr32_t(e, s1_r2)
            ch = _ch_t(e, f, g)
            kt = self.k[t]
            wt = w_all[t]

            p1 = (h + big_s1) & MASK32
            p2 = (p1 + ch) & MASK32
            p3 = (p2 + kt) & MASK32
            t1 = (p3 + wt) & MASK32

            big_s0 = _rotr32_t(a, s0_r0) ^ _rotr32_t(a, s0_r1) ^ _rotr32_t(a, s0_r2)
            mj = _maj_t(a, b, c)
            t2 = (big_s0 + mj) & MASK32

            a_new = (t1 + t2) & MASK32
            e_new = (d + t1) & MASK32

            if not hash_only:
                c_list[t][3] = h ^ big_s1 ^ p1
                c_list[t][4] = p1 ^ ch ^ p2
                c_list[t][5] = p2 ^ kt ^ p3
                c_list[t][6] = p3 ^ wt ^ t1
                c_list[t][7] = big_s0 ^ mj ^ t2
                c_list[t][8] = t1 ^ t2 ^ a_new
                c_list[t][9] = d ^ t1 ^ e_new

            h, g, f, e, d, c, b, a = g, f, e, e_new, c, b, a, a_new
            if not hash_only:
                s_list.append(torch.stack([a, b, c, d, e, f, g, h], dim=1))

        h_out = (torch.stack([a, b, c, d, e, f, g, h], dim=1) + self.iv.unsqueeze(0)) & MASK32
        if hash_only:
            return {"H": h_out}

        W = torch.stack(w_all[:rounds], dim=1)
        S = torch.stack(s_list, dim=1)
        C = torch.stack([torch.stack(row, dim=1) for row in c_list], dim=1)
        return {"H": h_out, "W": W, "S": S, "C": C}

    @torch.no_grad()
    def eval_crystal_batch(
        self,
        H_target: torch.Tensor,
        W: torch.Tensor,
        S: torch.Tensor,
        C: torch.Tensor,
        rounds: int = 64,
        alpha: int = 1,
        beta: int = 1,
        gamma: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """兼容 TorchCrystalEngine.eval_crystal_batch 接口（支持随机 ROTR/K 克隆版）。"""
        ht = H_target.to(device=self.device, dtype=torch.int64) & MASK32
        wt = W.to(device=self.device, dtype=torch.int64) & MASK32
        st = S.to(device=self.device, dtype=torch.int64) & MASK32
        ct = C.to(device=self.device, dtype=torch.int64) & MASK32
        bsz = wt.shape[0]

        def _add_err(x: torch.Tensor, y: torch.Tensor, c_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            next_c = (_maj_t(x, y, c_in) << 1) & MASK32
            return (x ^ y ^ c_in), (c_in ^ next_c)

        bnd0 = st[:, 0, :] ^ self.iv.unsqueeze(0)
        bndR = st[:, rounds, :] ^ ((ht - self.iv.unsqueeze(0)) & MASK32)

        dW = torch.zeros((bsz, rounds), device=self.device, dtype=torch.int64)
        dC = torch.zeros((bsz, rounds, C_PLANES), device=self.device, dtype=torch.int64)
        dS_mid = torch.zeros((bsz, rounds, 8), device=self.device, dtype=torch.int64)

        s0_r0, s0_r1, s0_r2 = self.cfg.sigma0_rots
        s1_r0, s1_r1, s1_r2 = self.cfg.sigma1_rots
        ls0_r0, ls0_r1 = self.cfg.lower_sigma0_rots
        ls0_shr = self.cfg.lower_sigma0_shr
        ls1_r0, ls1_r1 = self.cfg.lower_sigma1_rots
        ls1_shr = self.cfg.lower_sigma1_shr

        dC[:, : min(16, rounds), :3] = ct[:, : min(16, rounds), :3]
        if rounds > 16:
            w2 = wt[:, 14 : rounds - 2]
            w7 = wt[:, 9 : rounds - 7]
            w15 = wt[:, 1 : rounds - 15]
            w16 = wt[:, 0 : rounds - 16]
            sig1 = _rotr32_t(w2, ls1_r0) ^ _rotr32_t(w2, ls1_r1) ^ (w2 >> ls1_shr)
            sig0 = _rotr32_t(w15, ls0_r0) ^ _rotr32_t(w15, ls0_r1) ^ (w15 >> ls0_shr)
            u0, dc0 = _add_err(sig1, w7, ct[:, 16:rounds, 0])
            u1, dc1 = _add_err(u0, sig0, ct[:, 16:rounds, 1])
            wr, dc2 = _add_err(u1, w16, ct[:, 16:rounds, 2])
            dW[:, 16:rounds] = wt[:, 16:rounds] ^ wr
            dC[:, 16:rounds, 0] = dc0
            dC[:, 16:rounds, 1] = dc1
            dC[:, 16:rounds, 2] = dc2

        a = st[:, :rounds, 0]
        b = st[:, :rounds, 1]
        c = st[:, :rounds, 2]
        d = st[:, :rounds, 3]
        e = st[:, :rounds, 4]
        f = st[:, :rounds, 5]
        g = st[:, :rounds, 6]
        h = st[:, :rounds, 7]

        big_s1 = _rotr32_t(e, s1_r0) ^ _rotr32_t(e, s1_r1) ^ _rotr32_t(e, s1_r2)
        ch = _ch_t(e, f, g)
        kt = self.k[:rounds].unsqueeze(0)

        p1, dc3 = _add_err(h, big_s1, ct[:, :, 3])
        p2, dc4 = _add_err(p1, ch, ct[:, :, 4])
        p3, dc5 = _add_err(p2, kt, ct[:, :, 5])
        t1, dc6 = _add_err(p3, wt, ct[:, :, 6])

        big_s0 = _rotr32_t(a, s0_r0) ^ _rotr32_t(a, s0_r1) ^ _rotr32_t(a, s0_r2)
        mj = _maj_t(a, b, c)
        t2, dc7 = _add_err(big_s0, mj, ct[:, :, 7])

        a_rhs, dc8 = _add_err(t1, t2, ct[:, :, 8])
        e_rhs, dc9 = _add_err(d, t1, ct[:, :, 9])

        dC[:, :, 3] = dc3
        dC[:, :, 4] = dc4
        dC[:, :, 5] = dc5
        dC[:, :, 6] = dc6
        dC[:, :, 7] = dc7
        dC[:, :, 8] = dc8
        dC[:, :, 9] = dc9

        dS_mid[:, :, 0] = st[:, 1 : rounds + 1, 0] ^ a_rhs
        dS_mid[:, :, 1] = st[:, 1 : rounds + 1, 1] ^ a
        dS_mid[:, :, 2] = st[:, 1 : rounds + 1, 2] ^ b
        dS_mid[:, :, 3] = st[:, 1 : rounds + 1, 3] ^ c
        dS_mid[:, :, 4] = st[:, 1 : rounds + 1, 4] ^ e_rhs
        dS_mid[:, :, 5] = st[:, 1 : rounds + 1, 5] ^ e
        dS_mid[:, :, 6] = st[:, 1 : rounds + 1, 6] ^ f
        dS_mid[:, :, 7] = st[:, 1 : rounds + 1, 7] ^ g

        dS = torch.cat([bnd0.unsqueeze(1), dS_mid, bndR.unsqueeze(1)], dim=1)
        cnt_dS = popcnt32_torch(dS).sum(dim=(1, 2))
        cnt_dW = popcnt32_torch(dW).sum(dim=1)
        cnt_dC = popcnt32_torch(dC).sum(dim=(1, 2))
        E = alpha * cnt_dS + beta * cnt_dW + gamma * cnt_dC
        return {
            "E": E,
            "dS": cnt_dS,
            "dW": cnt_dW,
            "dC": cnt_dC,
            "delta_S": dS,
            "delta_W": dW,
            "delta_C": dC,
        }


def patch_model_for_cipher(model: torch.nn.Module, cfg: CipherConfig) -> None:
    """将 Ouroboros64Net 的 k_bits_bipolar 与 ARXRotationEdgeBlock 旋转/右移索引同步至目标 CipherConfig。"""
    device = model.k_bits_bipolar.device
    k_tensor = torch.tensor(cfg.k_words, dtype=torch.int64, device=device)
    k_bits = words_to_bits_lsb(k_tensor) * 2.0 - 1.0
    model.k_bits_bipolar.copy_(k_bits)

    rot_offsets = (
        list(cfg.sigma0_rots)
        + list(cfg.sigma1_rots)
        + list(cfg.lower_sigma0_rots)
        + list(cfg.lower_sigma1_rots)
    )
    shr_offsets = [cfg.lower_sigma0_shr, cfg.lower_sigma1_shr]

    bit_pos = torch.arange(32, dtype=torch.long, device=device)
    fwd_indices = []
    bwd_indices = []
    valid_fwd_masks = []
    valid_bwd_masks = []

    for n in rot_offsets:
        fwd_indices.append((bit_pos + n) % 32)
        bwd_indices.append((bit_pos - n) % 32)
        valid_fwd_masks.append(torch.ones(32, dtype=torch.float32, device=device))
        valid_bwd_masks.append(torch.ones(32, dtype=torch.float32, device=device))

    for n in shr_offsets:
        fwd_indices.append((bit_pos + n).clamp(max=31))
        valid_fwd_masks.append(((bit_pos + n) < 32).to(torch.float32))
        bwd_indices.append((bit_pos - n).clamp(min=0))
        valid_bwd_masks.append(((bit_pos - n) >= 0).to(torch.float32))

    fwd_idx = torch.stack(fwd_indices, dim=0)
    bwd_idx = torch.stack(bwd_indices, dim=0)
    fwd_mask = torch.stack(valid_fwd_masks, dim=0)
    bwd_mask = torch.stack(valid_bwd_masks, dim=0)

    for mod in model.modules():
        if isinstance(mod, ARXRotationEdgeBlock):
            mod.fwd_idx.copy_(fwd_idx)
            mod.bwd_idx.copy_(bwd_idx)
            mod.fwd_mask.copy_(fwd_mask)
            mod.bwd_mask.copy_(bwd_mask)


def verify_sha256_rounds_np(
    w0_15: np.ndarray,
    rounds: int,
    cipher_cfg: Optional[CipherConfig] = None,
) -> np.ndarray:
    """独立纯 NumPy uint32 前向压缩核验（不依赖 PyTorch），返回 H (B, 8) uint32。"""
    cfg = cipher_cfg or make_standard_cipher_config()
    w_in = np.asarray(w0_15, dtype=np.uint32)
    if w_in.ndim == 1:
        w_in = w_in[None, :]
    bsz = w_in.shape[0]

    def _rotr(x: np.ndarray, n: int) -> np.ndarray:
        return ((x >> np.uint32(n)) | (x << np.uint32(32 - n))) & np.uint32(0xFFFFFFFF)

    w_all = [w_in[:, t].copy() for t in range(16)]
    ls0_r0, ls0_r1 = cfg.lower_sigma0_rots
    ls1_r0, ls1_r1 = cfg.lower_sigma1_rots
    for t in range(16, rounds):
        w2 = w_all[t - 2]
        w15 = w_all[t - 15]
        s1 = _rotr(w2, ls1_r0) ^ _rotr(w2, ls1_r1) ^ (w2 >> np.uint32(cfg.lower_sigma1_shr))
        s0 = _rotr(w15, ls0_r0) ^ _rotr(w15, ls0_r1) ^ (w15 >> np.uint32(cfg.lower_sigma0_shr))
        wt = (s1 + w_all[t - 7] + s0 + w_all[t - 16]) & np.uint32(0xFFFFFFFF)
        w_all.append(wt)

    iv = np.asarray(cfg.iv_words, dtype=np.uint32)
    state = np.broadcast_to(iv[None, :], (bsz, 8)).copy()
    a, b, c, d, e, f, g, h = [state[:, j].copy() for j in range(8)]
    s0_r0, s0_r1, s0_r2 = cfg.sigma0_rots
    s1_r0, s1_r1, s1_r2 = cfg.sigma1_rots

    for t in range(rounds):
        big_s1 = _rotr(e, s1_r0) ^ _rotr(e, s1_r1) ^ _rotr(e, s1_r2)
        ch = (e & f) ^ ((~e) & g)
        t1 = (h + big_s1 + ch + np.uint32(cfg.k_words[t]) + w_all[t]) & np.uint32(0xFFFFFFFF)
        big_s0 = _rotr(a, s0_r0) ^ _rotr(a, s0_r1) ^ _rotr(a, s0_r2)
        mj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (big_s0 + mj) & np.uint32(0xFFFFFFFF)
        h = g
        g = f
        f = e
        e = (d + t1) & np.uint32(0xFFFFFFFF)
        d = c
        c = b
        b = a
        a = (t1 + t2) & np.uint32(0xFFFFFFFF)

    s_end = np.stack([a, b, c, d, e, f, g, h], axis=1)
    return (s_end + iv[None, :]) & np.uint32(0xFFFFFFFF)


def free_word_count(rounds: int, carrier: str) -> int:
    """SPEC 1.2 合法自由输入微扰域长度 w_free(R, carrier)。"""
    if carrier == "STD":
        return min(16, int(rounds))
    if carrier == "BTC320":
        return 4
    raise ValueError(f"未知载体类型: {carrier!r}")


def sample_carrier_w0_np(
    rng: np.random.Generator,
    bsz: int,
    carrier: str,
) -> np.ndarray:
    """按 SPEC 1.2 载体规范生成 (B, 16) uint32 输入块。"""
    w0 = rng.integers(0, 2**32, size=(bsz, 16), dtype=np.uint64).astype(np.uint32)
    if carrier == "BTC320":
        w0[:, 4] = np.uint32(0x80000000)
        w0[:, 5:15] = np.uint32(0)
        w0[:, 15] = np.uint32(640)
    elif carrier != "STD":
        raise ValueError(f"未知载体类型: {carrier!r}")
    return w0


def leading_zeros_u32_np(x: np.ndarray) -> np.ndarray:
    """计算 uint32 数组的大端前导零个数 lcz(x) ∈ [0, 32]。"""
    arr = np.asarray(x, dtype=np.uint32)
    out = np.full(arr.shape, 32, dtype=np.int32)
    nz = arr != 0
    if np.any(nz):
        # bit_length on uint32 -> 1..32; leading zeros = 32 - bit_length
        vals = arr[nz].astype(np.uint64)
        bl = np.floor(np.log2(vals)).astype(np.int32) + 1
        # 修正浮点 log2 边界误差（例如 2^k - 1）
        bl = np.where((np.uint64(1) << bl.astype(np.uint64)) <= vals, bl + 1, bl)
        out[nz] = 32 - bl
    return out


def leading_zeros_u32_torch(x: torch.Tensor) -> torch.Tensor:
    """向量化 GPU/CPU 计算 uint32-in-int64 张量的前导零数 lcz(x) ∈ [0, 32]。"""
    v = (x & MASK32).to(torch.int64)
    lcz = torch.zeros_like(v, dtype=torch.int64)
    all_zero = v == 0
    for b in range(31, -1, -1):
        # 只要当前前缀尚未遇到 1 且该位仍为 0，则计数 +1
        still_zero = (v >> b) == 0
        lcz = lcz + still_zero.to(torch.int64)
    return torch.where(all_zero, torch.full_like(lcz, 32), lcz)


def collect_base_crystals_for_grid(
    eng: ParametricCrystalEngine,
    rounds: int,
    carrier: str,
    counts_by_k: Dict[int, int],
    seed: int,
    batch_size: int = 131072,
    max_draws: int = 6_000_000,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    通过批量前向采样一次性为 k=1..12 收集属于等零面基底流形 M_{R,k}
    （y_0..y_{k-1}=0 且 y_k=1）的输入块 W0 (N_k, 16) 与 H0 (N_k, 8)。
    """
    rng = np.random.default_rng(seed)
    collected_w: Dict[int, List[np.ndarray]] = {k: [] for k in counts_by_k}
    collected_h: Dict[int, List[np.ndarray]] = {k: [] for k in counts_by_k}
    total_counts: Dict[int, int] = {k: 0 for k in counts_by_k}
    draws = 0

    while draws < max_draws:
        pending = [k for k, req in counts_by_k.items() if total_counts[k] < req]
        if not pending:
            break
        cur_b = min(batch_size, max_draws - draws)
        w0_np = sample_carrier_w0_np(rng, cur_b, carrier=carrier)
        w0_t = torch.from_numpy(w0_np.astype(np.int64))
        res = eng.compress_exact(w0_t, rounds=rounds, hash_only=True)
        h_np = res["H"].cpu().numpy().astype(np.uint32)
        lcz = leading_zeros_u32_np(h_np[:, 0])
        draws += cur_b

        for k in pending:
            need = counts_by_k[k] - total_counts[k]
            if need <= 0:
                continue
            idx = np.flatnonzero(lcz == k)
            if idx.size > 0:
                take = idx[:need]
                collected_w[k].append(w0_np[take])
                collected_h[k].append(h_np[take])
                total_counts[k] += int(take.size)

    bank: Dict[int, Dict[str, torch.Tensor]] = {}
    for k, req in counts_by_k.items():
        if total_counts[k] == 0:
            raise RuntimeError(f"未收集到 R={rounds}, k={k}, carrier={carrier} 的任何基底晶体")
        w_cat = np.concatenate(collected_w[k], axis=0)[:req]
        h_cat = np.concatenate(collected_h[k], axis=0)[:req]
        bank[k] = {
            "W0": torch.from_numpy(w_cat.astype(np.int64)),
            "H0": torch.from_numpy(h_cat.astype(np.int64)),
        }
    return bank


def binomial_z_score(hits: int, n_eval: int, p0: float) -> float:
    """SPEC 4.3 精确二项 z-score（对比理论基线 p0）。"""
    if n_eval <= 0 or p0 <= 0.0 or p0 >= 1.0:
        return 0.0
    denom = math.sqrt(float(n_eval) * p0 * (1.0 - p0))
    if denom <= 0.0:
        return 0.0
    return (float(hits) - float(n_eval) * p0) / denom


def exact_binomial_pvalue_greater(hits: int, n_eval: int, p0: float) -> float:
    """单边精确二项检验 p-value: P[Binomial(N, p0) >= hits]。"""
    if hits <= 0:
        return 1.0
    if hits > n_eval:
        return 0.0
    try:
        from scipy.stats import binomtest

        return float(binomtest(hits, n_eval, p0, alternative="greater").pvalue)
    except Exception:
        # 纯标准库对数伽马近似兜底（仅当 scipy 不可用时）
        z = binomial_z_score(hits, n_eval, p0)
        return 0.5 * math.erfc(z / math.sqrt(2.0))


def wilson_ci95(hits: int, n_eval: int) -> Tuple[float, float]:
    """Wilson score 95% 置信区间。"""
    if n_eval <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p_hat = float(hits) / float(n_eval)
    z2 = z * z
    denom = 1.0 + z2 / float(n_eval)
    center = (p_hat + z2 / (2.0 * float(n_eval))) / denom
    half = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat)) / float(n_eval) + z2 / (4.0 * float(n_eval) * float(n_eval)))
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def two_proportion_z(hits1: int, n1: int, hits2: int, n2: int) -> float:
    """两样本比例差异 z-score（用于相对随机 ROTR/K 对照组与随机游走基线检验）。"""
    if n1 <= 0 or n2 <= 0:
        return 0.0
    p1 = float(hits1) / float(n1)
    p2 = float(hits2) / float(n2)
    p_pool = float(hits1 + hits2) / float(n1 + n2)
    var = p_pool * (1.0 - p_pool) * (1.0 / float(n1) + 1.0 / float(n2))
    if var <= 1e-18:
        return 0.0
    return (p1 - p2) / math.sqrt(var)


def compute_wavepacket_kl_and_bias(heatmap: np.ndarray) -> Dict[str, Any]:
    """
    计算 3D 热力图 Delta_bar (R, 19, 32) 相对 0.5 的平均 KL 散度与高偏向位点比例 eta_bias。
    """
    q = np.clip(heatmap.astype(np.float64), 1e-9, 1.0 - 1e-9)
    kl_bits = q * np.log2(q / 0.5) + (1.0 - q) * np.log2((1.0 - q) / 0.5)
    kl_nats = q * np.log(q / 0.5) + (1.0 - q) * np.log((1.0 - q) / 0.5)
    bias_mask = np.abs(heatmap - 0.5) > 0.15

    rho_w = heatmap[:, 0, :].mean(axis=1).tolist()
    rho_s = heatmap[:, 1:9, :].mean(axis=(1, 2)).tolist()
    rho_c = heatmap[:, 9:19, :].mean(axis=(1, 2)).tolist()

    return {
        "kl_div_bits": float(np.mean(kl_bits)),
        "kl_div_nats": float(np.mean(kl_nats)),
        "eta_bias_all": float(np.mean(bias_mask)),
        "eta_bias_W": float(np.mean(bias_mask[:, 0, :])),
        "eta_bias_S": float(np.mean(bias_mask[:, 1:9, :])),
        "eta_bias_C": float(np.mean(bias_mask[:, 9:19, :])),
        "rho_W": [float(v) for v in rho_w],
        "rho_S": [float(v) for v in rho_s],
        "rho_C": [float(v) for v in rho_c],
    }
