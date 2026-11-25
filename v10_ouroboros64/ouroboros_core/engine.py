"""Phase-SHA V10 Ouroboros-64 Core Engine (C++17 OpenMP + PyTorch Exact Bitwise Backend)."""

from __future__ import annotations

import ctypes
import decimal
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .build_cpp import add_mingw_runtime_dirs, build_capi

MASK32 = 0xFFFFFFFF
C_PLANES = 10
MAX_ROUNDS = 64
SUPPORTED_ROUNDS = (16, 20, 24, 32, 48, 64)

# 6 generalized crystal bit states
SYM_0 = 0
SYM_1 = 1
SYM_Q = 2
SYM_DASH = 3
SYM_U = 4
SYM_N = 5

_BIT_SHIFTS_LSB = list(range(32))  # bit 0 = LSB, bit 31 = MSB


def derive_sha256_constants() -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Derive standard IV[8] and K[64] words via prime fractional roots."""
    primes = (
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
        59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
        137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
        227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
    )
    ctx = decimal.Context(prec=64)
    scale = decimal.Decimal(1 << 32)
    iv = tuple(int((ctx.sqrt(decimal.Decimal(p)) % 1) * scale) & MASK32 for p in primes[:8])
    three = decimal.Decimal(3)
    k = tuple(
        int(((ctx.exp(ctx.ln(decimal.Decimal(p)) / three)) % 1) * scale) & MASK32
        for p in primes
    )
    return iv, k


IV_WORDS, K_WORDS = derive_sha256_constants()


def enforce_fp32_determinism() -> None:
    """Lock neural execution to pure IEEE-754 FP32 (TF32 disabled)."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def words_to_bits_np(words: np.ndarray) -> np.ndarray:
    """Convert uint32/int64 array (...) -> uint8 LSB-first bits (..., 32)."""
    w = np.asarray(words, dtype=np.uint32)
    shifts = np.arange(32, dtype=np.uint32)
    return ((w[..., None] >> shifts) & 1).astype(np.uint8)


def bits_to_words_np(bits: np.ndarray) -> np.ndarray:
    """Convert uint8 LSB-first bits (..., 32) -> uint32 words (...)."""
    b = (np.asarray(bits) > 0).astype(np.uint32)
    shifts = np.arange(32, dtype=np.uint32)
    return np.sum(b << shifts, axis=-1, dtype=np.uint32)


def words_to_bits_lsb(words: torch.Tensor) -> torch.Tensor:
    """Convert int64 word tensor (...) -> float32 LSB-first bits (..., 32)."""
    shifts = torch.arange(32, device=words.device, dtype=torch.int64)
    return ((words.unsqueeze(-1) >> shifts) & 1).to(torch.float32)


def bits_to_words_lsb(bits: torch.Tensor) -> torch.Tensor:
    """Convert binary tensor (..., 32) LSB-first -> uint32-masked int64 words (...)."""
    b = (bits >= 0.5).to(torch.int64) if bits.is_floating_point() else (bits & 1).to(torch.int64)
    shifts = torch.arange(32, device=bits.device, dtype=torch.int64)
    return (b << shifts).sum(dim=-1) & MASK32


def popcnt32_torch(x: torch.Tensor) -> torch.Tensor:
    """Exact 32-bit SWAR population count in torch.int64 across CPU/CUDA."""
    v = x & MASK32
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    return ((v * 0x01010101) & MASK32) >> 24


def _rotr32_t(x: torch.Tensor, n: int) -> torch.Tensor:
    return ((x >> n) | (x << (32 - n))) & MASK32


def _ch_t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return (x & y) ^ ((~x & MASK32) & z)


def _maj_t(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return (x & y) ^ (x & z) ^ (y & z)


@dataclass
class SpacetimeCrystal:
    """CPU NumPy container for a single or batched R-round spacetime crystal."""

    rounds: int
    H_target: np.ndarray  # (B, 8) uint32
    W: np.ndarray         # (B, R) uint32
    S: np.ndarray         # (B, R+1, 8) uint32
    C: np.ndarray         # (B, R, 10) uint32

    def weld_boundaries(self) -> None:
        """Hard-weld S[:, 0] = IV and S[:, R] = (H_target - IV) mod 2^32."""
        iv = np.asarray(IV_WORDS, dtype=np.uint32)
        self.S[:, 0, :] = iv[None, :]
        self.S[:, self.rounds, :] = (self.H_target.astype(np.uint64) - iv[None, :].astype(np.uint64)) & MASK32


class OuroborosCore:
    """Singleton-friendly Python wrapper around the C++17 OpenMP crystal core."""

    def __init__(self, rebuild: bool = False) -> None:
        add_mingw_runtime_dirs()
        lib_path = build_capi(force=rebuild)
        self._lib = ctypes.CDLL(str(lib_path))
        self._bind_signatures()

    def _bind_signatures(self) -> None:
        u32_p = ctypes.POINTER(ctypes.c_uint32)
        u64_p = ctypes.POINTER(ctypes.c_uint64)
        i32_p = ctypes.POINTER(ctypes.c_int32)
        i64_p = ctypes.POINTER(ctypes.c_int64)
        u8_p = ctypes.POINTER(ctypes.c_uint8)

        self._lib.ouro_max_threads.argtypes = []
        self._lib.ouro_max_threads.restype = ctypes.c_int

        self._lib.ouro_sha256_compress_batch.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            u32_p, u32_p, u32_p, u32_p, u32_p,
        ]
        self._lib.ouro_sha256_compress_batch.restype = None

        self._lib.ouro_eval_crystal_batch.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            u32_p, u32_p, u32_p, u32_p,
            u32_p, u32_p, u32_p,
            u64_p, u64_p, u64_p, u64_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ]
        self._lib.ouro_eval_crystal_batch.restype = None

        self._lib.ouro_eval_cluster_flips_batch.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            u32_p, u32_p, u32_p, u32_p,
            i32_p, i64_p, u64_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ]
        self._lib.ouro_eval_cluster_flips_batch.restype = None

        self._lib.ouro_propagate_symbols.argtypes = [
            ctypes.c_uint32, u32_p, u8_p, u8_p, u8_p,
            ctypes.c_uint32, ctypes.c_uint32, u32_p,
        ]
        self._lib.ouro_propagate_symbols.restype = None

        self._lib.ouro_gf2_solve_equations.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32,
            u32_p, u32_p, u8_p, u32_p, u8_p, u32_p, u32_p,
        ]
        self._lib.ouro_gf2_solve_equations.restype = None

    @property
    def max_threads(self) -> int:
        return int(self._lib.ouro_max_threads())

    def compress_exact(
        self,
        w0_15: np.ndarray,
        rounds: int = 64,
    ) -> SpacetimeCrystal:
        """Compute exact (H, W, S, C) crystal from (B, 16) uint32 preimage blocks."""
        w_in = np.ascontiguousarray(w0_15, dtype=np.uint32)
        if w_in.ndim == 1:
            w_in = w_in[None, :]
        bsz = int(w_in.shape[0])
        assert w_in.shape == (bsz, 16), f"Expected (B, 16), got {w_in.shape}"

        h_out = np.zeros((bsz, 8), dtype=np.uint32)
        w_out = np.zeros((bsz, rounds), dtype=np.uint32)
        s_out = np.zeros((bsz, rounds + 1, 8), dtype=np.uint32)
        c_out = np.zeros((bsz, rounds, C_PLANES), dtype=np.uint32)

        self._lib.ouro_sha256_compress_batch(
            ctypes.c_uint32(bsz),
            ctypes.c_uint32(rounds),
            w_in.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            h_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            w_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            s_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            c_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        )
        return SpacetimeCrystal(rounds=rounds, H_target=h_out, W=w_out, S=s_out, C=c_out)

    def verify_preimage(self, w0_15: np.ndarray, h_target: np.ndarray, rounds: int = 64) -> np.ndarray:
        """Exact classical verification SHA256_R(W[0..15]) == H."""
        cr = self.compress_exact(w0_15, rounds=rounds)
        ht = np.ascontiguousarray(h_target, dtype=np.uint32)
        if ht.ndim == 1:
            ht = ht[None, :]
        return np.all(cr.H_target == ht, axis=1)

    def eval_crystal_batch(
        self,
        H_target: np.ndarray,
        W: np.ndarray,
        S: np.ndarray,
        C: np.ndarray,
        rounds: int = 64,
        alpha: int = 1,
        beta: int = 1,
        gamma: int = 1,
        return_masks: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Evaluate exact crystal defect energy E_defect = alpha*dS + beta*dW + gamma*dC."""
        hb = np.ascontiguousarray(H_target, dtype=np.uint32)
        wb = np.ascontiguousarray(W, dtype=np.uint32)
        sb = np.ascontiguousarray(S, dtype=np.uint32)
        cb = np.ascontiguousarray(C, dtype=np.uint32)
        bsz = int(wb.shape[0])

        out_e = np.zeros((bsz,), dtype=np.uint64)
        out_ds = np.zeros((bsz,), dtype=np.uint64)
        out_dw = np.zeros((bsz,), dtype=np.uint64)
        out_dc = np.zeros((bsz,), dtype=np.uint64)

        dS = np.zeros((bsz, rounds + 2, 8), dtype=np.uint32) if return_masks else None
        dW = np.zeros((bsz, rounds), dtype=np.uint32) if return_masks else None
        dC = np.zeros((bsz, rounds, C_PLANES), dtype=np.uint32) if return_masks else None

        u32_null = ctypes.POINTER(ctypes.c_uint32)()
        self._lib.ouro_eval_crystal_batch(
            ctypes.c_uint32(bsz),
            ctypes.c_uint32(rounds),
            hb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            wb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            sb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            cb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            dS.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)) if dS is not None else u32_null,
            dW.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)) if dW is not None else u32_null,
            dC.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)) if dC is not None else u32_null,
            out_e.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            out_ds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            out_dw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            out_dc.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_uint32(alpha),
            ctypes.c_uint32(beta),
            ctypes.c_uint32(gamma),
        )
        res: Dict[str, np.ndarray] = {"E": out_e, "dS": out_ds, "dW": out_dw, "dC": out_dc}
        if return_masks and dS is not None and dW is not None and dC is not None:
            res["delta_S"] = dS
            res["delta_W"] = dW
            res["delta_C"] = dC
        return res

    def eval_cluster_flips_batch(
        self,
        H_target: np.ndarray,
        W_base: np.ndarray,
        S_base: np.ndarray,
        C_base: np.ndarray,
        cluster_idx: np.ndarray,
        rounds: int = 64,
        alpha: int = 1,
        beta: int = 1,
        gamma: int = 1,
        allow_boundary_flip: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Incremental L1-cached OpenMP batch evaluation of candidate multi-bit cluster flips.

        Args:
            cluster_idx: shape (B_cryst * K_cands, M_max, 4) int32 of (kind, t, word, bit)
        Returns:
            delta_E (B_cryst * K_cands,) int64, new_E (B_cryst * K_cands,) uint64
        """
        hb = np.ascontiguousarray(H_target, dtype=np.uint32)
        wb = np.ascontiguousarray(W_base, dtype=np.uint32)
        sb = np.ascontiguousarray(S_base, dtype=np.uint32)
        cb = np.ascontiguousarray(C_base, dtype=np.uint32)
        cl = np.ascontiguousarray(cluster_idx, dtype=np.int32)

        b_cryst = int(wb.shape[0])
        assert cl.ndim == 3 and cl.shape[2] == 4 and cl.shape[0] % b_cryst == 0
        k_cands = int(cl.shape[0] // b_cryst)
        m_max = int(cl.shape[1])

        out_de = np.zeros((b_cryst * k_cands,), dtype=np.int64)
        out_ne = np.zeros((b_cryst * k_cands,), dtype=np.uint64)

        self._lib.ouro_eval_cluster_flips_batch(
            ctypes.c_uint32(b_cryst),
            ctypes.c_uint32(k_cands),
            ctypes.c_uint32(m_max),
            ctypes.c_uint32(rounds),
            hb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            wb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            sb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            cb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            cl.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            out_de.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            out_ne.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_uint32(alpha),
            ctypes.c_uint32(beta),
            ctypes.c_uint32(gamma),
            ctypes.c_uint32(1 if allow_boundary_flip else 0),
        )
        return out_de, out_ne

    def propagate_symbols(
        self,
        H_target: np.ndarray,
        W_sym: np.ndarray,
        S_sym: np.ndarray,
        C_sym: np.ndarray,
        rounds: int = 64,
        use_gf2: bool = True,
        max_iters: int = 16,
    ) -> Dict[str, Any]:
        """Run coupled Plane A (GF2 RREF) + Plane B (bidirectional ARX unit prop) to fixpoint."""
        ht = np.ascontiguousarray(H_target, dtype=np.uint32).reshape(8)
        ws = np.ascontiguousarray(W_sym, dtype=np.uint8).copy()
        ss = np.ascontiguousarray(S_sym, dtype=np.uint8).copy()
        cs = np.ascontiguousarray(C_sym, dtype=np.uint8).copy()
        stats = np.zeros((6,), dtype=np.uint32)

        self._lib.ouro_propagate_symbols(
            ctypes.c_uint32(rounds),
            ht.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ws.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ss.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            cs.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_uint32(1 if use_gf2 else 0),
            ctypes.c_uint32(max_iters),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        )
        return {
            "W_sym": ws,
            "S_sym": ss,
            "C_sym": cs,
            "conflict": bool(stats[0] != 0),
            "unit_prop_deductions": int(stats[1]),
            "gf2_deductions": int(stats[2]),
            "gf2_rank": int(stats[3]),
            "gf2_conflicts": int(stats[4]),
            "unknown_after": int(stats[5]),
        }

    def solve_gf2_equations(
        self,
        num_vars: int,
        equations: Sequence[Tuple[Sequence[int], int]],
        return_degrees: bool = False,
    ) -> Dict[str, Any]:
        """Direct Plane A GF(2) RREF solver on arbitrary XOR equations (cols, rhs_bit)."""
        offsets = [0]
        flat_cols: List[int] = []
        rhs_list: List[int] = []
        for cols, rhs in equations:
            flat_cols.extend(int(c) for c in cols)
            offsets.append(len(flat_cols))
            rhs_list.append(int(rhs) & 1)

        off_np = np.asarray(offsets, dtype=np.uint32)
        col_np = np.asarray(flat_cols if flat_cols else [0], dtype=np.uint32)
        rhs_np = np.asarray(rhs_list if rhs_list else [0], dtype=np.uint8)
        ucols = np.zeros((max(1, num_vars),), dtype=np.uint32)
        uvals = np.zeros((max(1, num_vars),), dtype=np.uint8)
        deg = np.zeros((max(1, num_vars),), dtype=np.uint32) if return_degrees else None
        meta = np.zeros((3,), dtype=np.uint32)

        self._lib.ouro_gf2_solve_equations(
            ctypes.c_uint32(num_vars),
            ctypes.c_uint32(len(equations)),
            off_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            col_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            rhs_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ucols.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            uvals.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            deg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)) if deg is not None else ctypes.POINTER(ctypes.c_uint32)(),
            meta.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        )
        n_units = int(meta[2])
        res: Dict[str, Any] = {
            "conflict": bool(meta[0] != 0),
            "rank": int(meta[1]),
            "units": {int(ucols[i]): int(uvals[i]) for i in range(n_units)},
        }
        if deg is not None:
            res["degrees"] = deg[:num_vars]
        return res


class TorchCrystalEngine:
    """Vectorized GPU/CPU exact integer crystal engine mirroring OuroborosCore 1:1."""

    def __init__(self, device: torch.device | str = "cuda") -> None:
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.iv = torch.tensor(IV_WORDS, device=self.device, dtype=torch.int64)
        self.k = torch.tensor(K_WORDS, device=self.device, dtype=torch.int64)

    @torch.no_grad()
    def compress_exact(self, w0_15: torch.Tensor, rounds: int = 64) -> Dict[str, torch.Tensor]:
        """Exact uint32-in-int64 crystal construction on GPU/CPU."""
        w_in = (w0_15.to(device=self.device, dtype=torch.int64) & MASK32)
        if w_in.ndim == 1:
            w_in = w_in.unsqueeze(0)
        bsz = w_in.shape[0]

        w_all: List[torch.Tensor] = [w_in[:, t] for t in range(16)]
        c_list: List[List[torch.Tensor]] = [[torch.zeros(bsz, device=self.device, dtype=torch.int64) for _ in range(C_PLANES)] for _ in range(rounds)]

        for t in range(16, rounds):
            w2 = w_all[t - 2]
            w15 = w_all[t - 15]
            s1 = _rotr32_t(w2, 17) ^ _rotr32_t(w2, 19) ^ (w2 >> 10)
            s0 = _rotr32_t(w15, 7) ^ _rotr32_t(w15, 18) ^ (w15 >> 3)

            u0 = (s1 + w_all[t - 7]) & MASK32
            c_list[t][0] = s1 ^ w_all[t - 7] ^ u0
            u1 = (u0 + s0) & MASK32
            c_list[t][1] = u0 ^ s0 ^ u1
            wt = (u1 + w_all[t - 16]) & MASK32
            c_list[t][2] = u1 ^ w_all[t - 16] ^ wt
            w_all.append(wt)

        W = torch.stack(w_all[:rounds], dim=1)  # (B, R)

        st = self.iv.unsqueeze(0).expand(bsz, 8).clone()
        a, b, c, d, e, f, g, h = [st[:, j] for j in range(8)]
        s_list: List[torch.Tensor] = [st]

        for t in range(rounds):
            S1 = _rotr32_t(e, 6) ^ _rotr32_t(e, 11) ^ _rotr32_t(e, 25)
            ch = _ch_t(e, f, g)
            kt = self.k[t]
            wt = W[:, t]

            p1 = (h + S1) & MASK32
            c_list[t][3] = h ^ S1 ^ p1
            p2 = (p1 + ch) & MASK32
            c_list[t][4] = p1 ^ ch ^ p2
            p3 = (p2 + kt) & MASK32
            c_list[t][5] = p2 ^ kt ^ p3
            T1 = (p3 + wt) & MASK32
            c_list[t][6] = p3 ^ wt ^ T1

            S0 = _rotr32_t(a, 2) ^ _rotr32_t(a, 13) ^ _rotr32_t(a, 22)
            mj = _maj_t(a, b, c)
            T2 = (S0 + mj) & MASK32
            c_list[t][7] = S0 ^ mj ^ T2

            a_new = (T1 + T2) & MASK32
            c_list[t][8] = T1 ^ T2 ^ a_new
            e_new = (d + T1) & MASK32
            c_list[t][9] = d ^ T1 ^ e_new

            h, g, f, e, d, c, b, a = g, f, e, e_new, c, b, a, a_new
            s_list.append(torch.stack([a, b, c, d, e, f, g, h], dim=1))

        S = torch.stack(s_list, dim=1)  # (B, R+1, 8)
        C = torch.stack([torch.stack(row, dim=1) for row in c_list], dim=1)  # (B, R, 10)
        H = (S[:, rounds, :] + self.iv.unsqueeze(0)) & MASK32
        return {"H": H, "W": W, "S": S, "C": C}

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
        """Vectorized GPU/CPU evaluation of crystal defect energy & residual masks."""
        ht = (H_target.to(device=self.device, dtype=torch.int64) & MASK32)
        wt = (W.to(device=self.device, dtype=torch.int64) & MASK32)
        st = (S.to(device=self.device, dtype=torch.int64) & MASK32)
        ct = (C.to(device=self.device, dtype=torch.int64) & MASK32)
        bsz = wt.shape[0]

        def _add_err(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            next_c = (_maj_t(x, y, c) << 1) & MASK32
            return (x ^ y ^ c), (c ^ next_c)

        bnd0 = st[:, 0, :] ^ self.iv.unsqueeze(0)
        bndR = st[:, rounds, :] ^ ((ht - self.iv.unsqueeze(0)) & MASK32)

        dW = torch.zeros((bsz, rounds), device=self.device, dtype=torch.int64)
        dC = torch.zeros((bsz, rounds, C_PLANES), device=self.device, dtype=torch.int64)
        dS_mid = torch.zeros((bsz, rounds, 8), device=self.device, dtype=torch.int64)

        dC[:, : min(16, rounds), :3] = ct[:, : min(16, rounds), :3]
        if rounds > 16:
            w2 = wt[:, 14 : rounds - 2]
            w7 = wt[:, 9 : rounds - 7]
            w15 = wt[:, 1 : rounds - 15]
            w16 = wt[:, 0 : rounds - 16]
            s1 = _rotr32_t(w2, 17) ^ _rotr32_t(w2, 19) ^ (w2 >> 10)
            s0 = _rotr32_t(w15, 7) ^ _rotr32_t(w15, 18) ^ (w15 >> 3)
            u0, dc0 = _add_err(s1, w7, ct[:, 16:rounds, 0])
            u1, dc1 = _add_err(u0, s0, ct[:, 16:rounds, 1])
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

        S1 = _rotr32_t(e, 6) ^ _rotr32_t(e, 11) ^ _rotr32_t(e, 25)
        ch = _ch_t(e, f, g)
        kt = self.k[:rounds].unsqueeze(0)

        p1, dc3 = _add_err(h, S1, ct[:, :, 3])
        p2, dc4 = _add_err(p1, ch, ct[:, :, 4])
        p3, dc5 = _add_err(p2, kt, ct[:, :, 5])
        T1, dc6 = _add_err(p3, wt, ct[:, :, 6])

        S0 = _rotr32_t(a, 2) ^ _rotr32_t(a, 13) ^ _rotr32_t(a, 22)
        mj = _maj_t(a, b, c)
        T2, dc7 = _add_err(S0, mj, ct[:, :, 7])

        a_rhs, dc8 = _add_err(T1, T2, ct[:, :, 8])
        e_rhs, dc9 = _add_err(d, T1, ct[:, :, 9])

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

        dS = torch.cat([bnd0.unsqueeze(1), dS_mid, bndR.unsqueeze(1)], dim=1)  # (B, R+2, 8)
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


_DEFAULT_CORE: Optional[OuroborosCore] = None


def get_core() -> OuroborosCore:
    global _DEFAULT_CORE
    if _DEFAULT_CORE is None:
        _DEFAULT_CORE = OuroborosCore(rebuild=False)
    return _DEFAULT_CORE
