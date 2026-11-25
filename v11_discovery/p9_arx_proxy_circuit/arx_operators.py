"""Exact vectorized ARX operators for P9 (Families A1, A2, B, C, and Random-ARX controls).

All bit indices follow LSB-first convention (b=0 is LSB, b=w-1 is MSB), strictly matching
g:\\SHA-LM\\src\\sha256_ref.py and g:\\SHA-LM\\v10_ouroboros64\\ouroboros_core\\cnf_cdcl.py.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src.sha256_ref import SHA256_IV, SHA256_K, MASK32


@dataclass(frozen=True)
class ARXParams:
    """Parameters defining a concrete SHA-256 or Random-ARX operator instance."""

    word_bits: int
    sigma0_upper_rot: Tuple[int, int, int]
    sigma1_upper_rot: Tuple[int, int, int]
    sigma0_lower_op: Tuple[int, int, int]  # (rot1, rot2, shr3)
    sigma1_lower_op: Tuple[int, int, int]  # (rot1, rot2, shr3)
    k_consts: Tuple[int, ...]
    iv_consts: Tuple[int, ...]
    label: str = "sha256_real"


def get_real_sha256_params(rounds: int = 64) -> ARXParams:
    return ARXParams(
        word_bits=32,
        sigma0_upper_rot=(2, 13, 22),
        sigma1_upper_rot=(6, 11, 25),
        sigma0_lower_op=(7, 18, 3),
        sigma1_lower_op=(17, 19, 10),
        k_consts=tuple(SHA256_K[: max(rounds, 24)]),
        iv_consts=tuple(SHA256_IV),
        label="sha256_real",
    )


def get_mini_arx_params(word_bits: int, rounds: int = 8) -> ARXParams:
    """Isomorphic w-bit Mini-ARX parameters for Family C (w in {4, 8})."""
    if word_bits == 4:
        mask = 0xF
        return ARXParams(
            word_bits=4,
            sigma0_upper_rot=(1, 2, 3),
            sigma1_upper_rot=(1, 3, 2),
            sigma0_lower_op=(1, 3, 1),
            sigma1_lower_op=(2, 3, 1),
            k_consts=tuple(k & mask for k in SHA256_K[: max(rounds, 16)]),
            iv_consts=tuple(v & mask for v in SHA256_IV),
            label="mini_arx_w4",
        )
    if word_bits == 8:
        mask = 0xFF
        return ARXParams(
            word_bits=8,
            sigma0_upper_rot=(1, 3, 5),
            sigma1_upper_rot=(2, 4, 7),
            sigma0_lower_op=(1, 5, 2),
            sigma1_lower_op=(3, 6, 3),
            k_consts=tuple(k & mask for k in SHA256_K[: max(rounds, 16)]),
            iv_consts=tuple(v & mask for v in SHA256_IV),
            label="mini_arx_w8",
        )
    raise ValueError(f"Unsupported Family C word_bits={word_bits}; expected 4 or 8.")


def get_random_arx_params(seed: int, word_bits: int = 32, rounds: int = 24) -> ARXParams:
    """Random-ARX structural control with identical topology but randomized shifts and K."""
    rng = random.Random(10_000 + seed * 997 + word_bits)
    rot_pool = list(range(1, word_bits))
    shr_max = max(2, word_bits // 2 - 1)
    shr_pool = list(range(1, shr_max + 1))
    mask = (1 << word_bits) - 1

    s0_u = tuple(sorted(rng.sample(rot_pool, 3)))
    s1_u = tuple(sorted(rng.sample(rot_pool, 3)))
    # Ensure distinct from real SHA-256 if word_bits==32
    while word_bits == 32 and s0_u == (2, 13, 22):
        s0_u = tuple(sorted(rng.sample(rot_pool, 3)))
    while word_bits == 32 and (s1_u == (6, 11, 25) or s1_u == s0_u):
        s1_u = tuple(sorted(rng.sample(rot_pool, 3)))

    s0_l = (*sorted(rng.sample(rot_pool, 2)), rng.choice(shr_pool))
    s1_l = (*sorted(rng.sample(rot_pool, 2)), rng.choice(shr_pool))
    while word_bits == 32 and s0_l == (7, 18, 3):
        s0_l = (*sorted(rng.sample(rot_pool, 2)), rng.choice(shr_pool))
    while word_bits == 32 and (s1_l == (17, 19, 10) or s1_l == s0_l):
        s1_l = (*sorted(rng.sample(rot_pool, 2)), rng.choice(shr_pool))

    k_consts = tuple(rng.getrandbits(word_bits) & mask for _ in range(max(rounds, 24)))
    iv_consts = tuple(SHA256_IV[i] & mask for i in range(8))
    return ARXParams(
        word_bits=word_bits,
        sigma0_upper_rot=s0_u,
        sigma1_upper_rot=s1_u,
        sigma0_lower_op=s0_l,
        sigma1_lower_op=s1_l,
        k_consts=k_consts,
        iv_consts=iv_consts,
        label=f"random_arx_s{seed}",
    )


def rotr_w(x: torch.Tensor, n: int, w: int) -> torch.Tensor:
    mask = (1 << w) - 1
    n = n % w
    if n == 0:
        return x & mask
    xu = x & mask
    return ((xu >> n) | ((xu & ((1 << n) - 1)) << (w - n))) & mask


def shr_w(x: torch.Tensor, n: int, w: int) -> torch.Tensor:
    mask = (1 << w) - 1
    return (x & mask) >> n


def ch_w(e: torch.Tensor, f: torch.Tensor, g: torch.Tensor, w: int) -> torch.Tensor:
    mask = (1 << w) - 1
    eu, fu, gu = e & mask, f & mask, g & mask
    return ((eu & fu) ^ ((~eu & mask) & gu)) & mask


def maj_w(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, w: int) -> torch.Tensor:
    mask = (1 << w) - 1
    au, bu, cu = a & mask, b & mask, c & mask
    return ((au & bu) ^ (au & cu) ^ (bu & cu)) & mask


def sigma_rot3(x: torch.Tensor, rots: Tuple[int, int, int], w: int) -> torch.Tensor:
    mask = (1 << w) - 1
    return (rotr_w(x, rots[0], w) ^ rotr_w(x, rots[1], w) ^ rotr_w(x, rots[2], w)) & mask


def sigma_rot2_shr1(x: torch.Tensor, ops: Tuple[int, int, int], w: int) -> torch.Tensor:
    mask = (1 << w) - 1
    return (rotr_w(x, ops[0], w) ^ rotr_w(x, ops[1], w) ^ shr_w(x, ops[2], w)) & mask


@dataclass(frozen=True)
class OperatorSpec:
    """Describes one target configuration (Family, R, params)."""

    family: str  # "A1", "A2", "B", "B_joint", "C4", "C8"
    rounds: int
    params: ARXParams

    @property
    def word_bits(self) -> int:
        return self.params.word_bits

    @property
    def input_words_count(self) -> int:
        if self.family == "A1":
            return 8 + self.rounds  # S_0 (8 words) + W[0..R-1]
        if self.family == "A2":
            return self.rounds  # W[0..R-1] only (S_0 = IV)
        if self.family == "B":
            return 16  # W[0..15]
        if self.family == "B_joint":
            return 24  # S_16 (8 words) + W[0..15]
        if self.family in ("C4", "C8"):
            return 8 + self.rounds  # S_0 (8 words) + W[0..R-1]
        raise ValueError(f"Unknown family {self.family}")

    @property
    def n_inputs(self) -> int:
        return self.input_words_count * self.word_bits

    @property
    def exact_and_gates(self) -> int:
        w = self.word_bits
        # Per compression round: 2*(w) for Ch, Maj + 7*(w-1) for 7 Add_w
        comp_per_round = 2 * w + 7 * (w - 1)
        # Per schedule step: 3*(w-1) for 3 Add_w
        sched_per_step = 3 * (w - 1)
        if self.family in ("A1", "A2", "C4", "C8"):
            return self.rounds * comp_per_round
        if self.family == "B":
            return self.rounds * sched_per_step
        if self.family == "B_joint":
            return self.rounds * (comp_per_round + sched_per_step)
        raise ValueError(f"Unknown family {self.family}")

    def input_var_names(self) -> List[str]:
        w = self.word_bits
        names: List[str] = []
        if self.family in ("A1", "C4", "C8"):
            state_labels = ["a0", "am1", "am2", "am3", "e0", "em1", "em2", "em3"]
            for lbl in state_labels:
                for b in range(w):
                    names.append(f"{lbl}[{b}]")
            for t in range(self.rounds):
                for b in range(w):
                    names.append(f"W{t}[{b}]")
        elif self.family == "A2":
            for t in range(self.rounds):
                for b in range(w):
                    names.append(f"W{t}[{b}]")
        elif self.family == "B":
            for t in range(16):
                for b in range(w):
                    names.append(f"W{t}[{b}]")
        elif self.family == "B_joint":
            state_labels = ["a16", "a15", "a14", "a13", "e16", "e15", "e14", "e13"]
            for lbl in state_labels:
                for b in range(w):
                    names.append(f"{lbl}[{b}]")
            for t in range(16):
                for b in range(w):
                    names.append(f"W{t}[{b}]")
        return names

    def target_list(self) -> List[str]:
        w = self.word_bits
        if w == 32:
            bits = [0, 1, 7, 15, 23, 31]
        elif w == 8:
            bits = [0, 1, 4, 7]
        else:
            bits = [0, 1, 3]
        if self.family == "B":
            target_w = 15 + self.rounds
            return [f"W{target_w}[{b}]" for b in bits]
        if self.family == "B_joint":
            r = 16 + self.rounds
            return [f"a{r}[{b}]" for b in bits] + [f"e{r}[{b}]" for b in bits]
        r = self.rounds
        return [f"a{r}[{b}]" for b in bits] + [f"e{r}[{b}]" for b in bits]


def sample_input_words(
    spec: OperatorSpec,
    n_samples: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample uniformly distributed input words of shape (n_samples, input_words_count) int64."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    high = 1 << spec.word_bits
    return torch.randint(
        0,
        high,
        (n_samples, spec.input_words_count),
        dtype=torch.int64,
        device=device,
        generator=gen,
    )


def words_to_packed_inputs(words: torch.Tensor, word_bits: int, chunk_words: int = 1024) -> torch.Tensor:
    """Convert (N, K) uint-in-int64 words (N % 64 == 0) into (K * word_bits, N // 64) bit-packed int64.

    Processes in chunks of `chunk_words * 64` samples to keep peak GPU/CPU memory tiny even for N=2^22.
    """
    n_samples, n_words = words.shape
    if n_samples % 64 != 0:
        raise ValueError(f"n_samples={n_samples} must be a multiple of 64")
    device = words.device
    total_packed = n_samples // 64
    n_inputs = n_words * word_bits
    out = torch.empty((n_inputs, total_packed), dtype=torch.int64, device=device)

    bit_shifts = torch.arange(word_bits, dtype=torch.int64, device=device)
    pack_shifts = torch.arange(64, dtype=torch.int64, device=device)
    for start_w in range(0, total_packed, chunk_words):
        end_w = min(total_packed, start_w + chunk_words)
        s0 = start_w * 64
        s1 = end_w * 64
        w_chunk = words[s0:s1]
        bits = ((w_chunk.unsqueeze(-1) >> bit_shifts) & 1).permute(1, 2, 0).reshape(n_inputs, end_w - start_w, 64)
        out[:, start_w:end_w] = (bits << pack_shifts.view(1, 1, 64)).sum(dim=-1)
    return out


def _run_compression_rounds(
    s0_words: torch.Tensor,  # (N, 8): a0, a-1, a-2, a-3, e0, e-1, e-2, e-3
    w_sched: torch.Tensor,   # (N, R)
    params: ARXParams,
    k_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run R rounds of generalized ARX compression, returning (a_R, e_R)."""
    w = params.word_bits
    mask = (1 << w) - 1
    rounds = w_sched.shape[1]

    a_cols = [s0_words[:, 3], s0_words[:, 2], s0_words[:, 1], s0_words[:, 0]]
    e_cols = [s0_words[:, 7], s0_words[:, 6], s0_words[:, 5], s0_words[:, 4]]

    for t in range(rounds):
        a_t, a_m1, a_m2, a_m3 = a_cols[t + 3], a_cols[t + 2], a_cols[t + 1], a_cols[t]
        e_t, e_m1, e_m2, e_m3 = e_cols[t + 3], e_cols[t + 2], e_cols[t + 1], e_cols[t]

        s1 = sigma_rot3(e_t, params.sigma1_upper_rot, w)
        ch = ch_w(e_t, e_m1, e_m2, w)
        k_t = params.k_consts[k_offset + t] & mask
        t1 = (e_m3 + s1 + ch + k_t + w_sched[:, t]) & mask

        s0 = sigma_rot3(a_t, params.sigma0_upper_rot, w)
        mj = maj_w(a_t, a_m1, a_m2, w)
        t2 = (s0 + mj) & mask

        e_tp1 = (a_m3 + t1) & mask
        a_tp1 = (t1 + t2) & mask
        e_cols.append(e_tp1)
        a_cols.append(a_tp1)

    return a_cols[-1], e_cols[-1]


def _run_message_schedule(
    w16: torch.Tensor,  # (N, 16)
    total_len: int,
    params: ARXParams,
) -> torch.Tensor:
    """Expand (N, 16) words to (N, total_len) schedule words."""
    w = params.word_bits
    mask = (1 << w) - 1
    cols = [w16[:, t] & mask for t in range(16)]
    for t in range(16, total_len):
        s1 = sigma_rot2_shr1(cols[t - 2], params.sigma1_lower_op, w)
        s0 = sigma_rot2_shr1(cols[t - 15], params.sigma0_lower_op, w)
        wt = (s1 + cols[t - 7] + s0 + cols[t - 16]) & mask
        cols.append(wt)
    return torch.stack(cols, dim=1)


def evaluate_operator_targets(
    spec: OperatorSpec,
    input_words: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute exact target bit vectors (shape (N,) int64 0/1) for all targets in spec.target_list()."""
    n = input_words.shape[0]
    w = spec.word_bits
    device = input_words.device
    params = spec.params

    out_words: Dict[str, torch.Tensor] = {}
    if spec.family in ("A1", "C4", "C8"):
        s0 = input_words[:, :8]
        w_sched = input_words[:, 8:]
        a_r, e_r = _run_compression_rounds(s0, w_sched, params, k_offset=0)
        out_words[f"a{spec.rounds}"] = a_r
        out_words[f"e{spec.rounds}"] = e_r
    elif spec.family == "A2":
        iv = torch.tensor(params.iv_consts, dtype=torch.int64, device=device).unsqueeze(0).expand(n, 8)
        w_sched = input_words
        a_r, e_r = _run_compression_rounds(iv, w_sched, params, k_offset=0)
        out_words[f"a{spec.rounds}"] = a_r
        out_words[f"e{spec.rounds}"] = e_r
    elif spec.family == "B":
        target_w_idx = 15 + spec.rounds
        w_all = _run_message_schedule(input_words, target_w_idx + 1, params)
        out_words[f"W{target_w_idx}"] = w_all[:, target_w_idx]
    elif spec.family == "B_joint":
        s16 = input_words[:, :8]
        w16 = input_words[:, 8:]
        w_all = _run_message_schedule(w16, 16 + spec.rounds, params)
        a_r, e_r = _run_compression_rounds(s16, w_all[:, 16:], params, k_offset=16)
        r = 16 + spec.rounds
        out_words[f"a{r}"] = a_r
        out_words[f"e{r}"] = e_r
    else:
        raise ValueError(f"Unknown family {spec.family}")

    targets: Dict[str, torch.Tensor] = {}
    for tname in spec.target_list():
        word_name, bit_str = tname[:-1].split("[")
        b = int(bit_str)
        targets[tname] = (out_words[word_name] >> b) & 1
    return targets


def parse_target_name(tname: str) -> Tuple[str, int]:
    word_name, bit_str = tname[:-1].split("[")
    return word_name, int(bit_str)
