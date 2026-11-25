"""SHA-256 & BTC Double SHA-256 symbolic worlds for Tracks A1, A2, B1, B2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import z3

from ast_spec import (
    MASK32,
    np_ch32,
    np_maj32,
    np_Sigma0,
    np_Sigma1,
    np_sigma0,
    np_sigma1,
    z3_ch32,
    z3_maj32,
    z3_Sigma0,
    z3_Sigma1,
    z3_sigma0,
    z3_sigma1,
)

STATE_FIELDS = ("a", "b", "c", "d", "e", "f", "g", "h")


def _first_n_primes(n: int) -> List[int]:
    primes: List[int] = []
    cand = 2
    while len(primes) < n:
        lim = math.isqrt(cand)
        if all(cand % p != 0 for p in primes if p <= lim):
            primes.append(cand)
        cand += 1
    return primes


def _icbrt(n: int) -> int:
    if n <= 0:
        return 0
    x = 1 << ((n.bit_length() + 2) // 3)
    while True:
        y = (2 * x + n // (x * x)) // 3
        if y >= x:
            return x
        x = y


def _derive_constants() -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    primes = _first_n_primes(64)
    iv = tuple((math.isqrt(p << 64)) & MASK32 for p in primes[:8])
    k = tuple((_icbrt(p << 96)) & MASK32 for p in primes)
    return iv, k


SHA256_IV, SHA256_K = _derive_constants()


@dataclass
class RoundsConfig:
    track: str  # "A1", "A2", "B1", "B2"
    round_start: int = 0
    num_rounds: int = 5
    use_msg_schedule: bool = False
    btc_splice: bool = False
    block1_rounds: int = 0
    block2_rounds: int = 0
    padded_mode: bool = False  # e.g. single-block fixed padding in A2 or B2
    padded_free_words: int = 16

    def to_dict(self) -> Dict[str, object]:
        return {
            "round_start": self.round_start,
            "num_rounds": self.num_rounds,
            "use_msg_schedule": self.use_msg_schedule,
            "btc_splice": self.btc_splice,
            "block1_rounds": self.block1_rounds,
            "block2_rounds": self.block2_rounds,
        }


@dataclass
class WorldSpec:
    config: RoundsConfig
    base_vars: List[str] = field(default_factory=list)
    state_vars: List[str] = field(default_factory=list)
    msg_vars: List[str] = field(default_factory=list)
    round_indices: List[int] = field(default_factory=list)  # global round IDs for V6

    def all_env_vars(self) -> List[str]:
        return self.base_vars + self.state_vars + self.msg_vars


def _np_round(
    s: Dict[str, np.ndarray], k_val: int, w: np.ndarray, batch_size: int
) -> Dict[str, np.ndarray]:
    k_arr = np.full((batch_size,), np.uint32(k_val), dtype=np.uint32)
    t1 = (s["h"] + np_Sigma1(s["e"]) + np_ch32(s["e"], s["f"], s["g"]) + k_arr + w).astype(np.uint32)
    t2 = (np_Sigma0(s["a"]) + np_maj32(s["a"], s["b"], s["c"])).astype(np.uint32)
    return {
        "a": (t1 + t2).astype(np.uint32),
        "b": s["a"],
        "c": s["b"],
        "d": s["c"],
        "e": (s["d"] + t1).astype(np.uint32),
        "f": s["e"],
        "g": s["f"],
        "h": s["g"],
    }


def _z3_round(
    s: Dict[str, z3.BitVecRef],
    k_val: int,
    w: z3.BitVecRef,
    broken_ae: Optional[Tuple[z3.BitVecRef, z3.BitVecRef]] = None,
) -> Dict[str, z3.BitVecRef]:
    if broken_ae is not None:
        a_next, e_next = broken_ae
    else:
        k_bv = z3.BitVecVal(int(k_val) & MASK32, 32)
        t1 = s["h"] + z3_Sigma1(s["e"]) + z3_ch32(s["e"], s["f"], s["g"]) + k_bv + w
        t2 = z3_Sigma0(s["a"]) + z3_maj32(s["a"], s["b"], s["c"])
        a_next = t1 + t2
        e_next = s["d"] + t1
    return {
        "a": a_next,
        "b": s["a"],
        "c": s["b"],
        "d": s["c"],
        "e": e_next,
        "f": s["e"],
        "g": s["f"],
        "h": s["g"],
    }


def _np_sched(w_m2: np.ndarray, w_m7: np.ndarray, w_m15: np.ndarray, w_m16: np.ndarray) -> np.ndarray:
    return (np_sigma1(w_m2) + w_m7 + np_sigma0(w_m15) + w_m16).astype(np.uint32)


def _z3_sched(w_m2: z3.BitVecRef, w_m7: z3.BitVecRef, w_m15: z3.BitVecRef, w_m16: z3.BitVecRef) -> z3.BitVecRef:
    return z3_sigma1(w_m2) + w_m7 + z3_sigma0(w_m15) + w_m16


def build_world_spec(cfg: RoundsConfig) -> WorldSpec:
    spec = WorldSpec(config=cfg)
    if not cfg.btc_splice:
        # Track A1 or A2
        for f in STATE_FIELDS:
            spec.base_vars.append(f"s0.{f}")
        if not cfg.use_msg_schedule:
            for i in range(cfg.num_rounds):
                spec.base_vars.append(f"w{cfg.round_start + i}")
                spec.msg_vars.append(f"w{cfg.round_start + i}")
        else:
            # Message schedule mode:
            # If round_start + num_rounds <= 16: w[0..15] free (or padded)
            # If round_start >= 16 or crosses 16: w_init[0..15] are base_vars (unless padded)
            end_r = cfg.round_start + cfg.num_rounds
            if cfg.padded_mode:
                for i in range(cfg.padded_free_words):
                    spec.base_vars.append(f"w{i}")
            else:
                for i in range(max(16, cfg.round_start)):
                    if i < 16:
                        spec.base_vars.append(f"w{i}")
            for t in range(end_r):
                spec.msg_vars.append(f"w{t}")
        for r in range(cfg.num_rounds + 1):
            for f in STATE_FIELDS:
                v = f"s{r}.{f}"
                if v not in spec.state_vars:
                    spec.state_vars.append(v)
        spec.round_indices = list(range(cfg.round_start, cfg.round_start + cfg.num_rounds))
    else:
        # Track B1 or B2: BTC double SHA-256
        # Block 1: r1 rounds from b1_s0 -> b1_s[r1], using b1_w[0..r1-1]
        for f in STATE_FIELDS:
            spec.base_vars.append(f"b1_s0.{f}")
        for i in range(cfg.block1_rounds):
            spec.base_vars.append(f"b1_w{i}")
            spec.msg_vars.append(f"b1_w{i}")
        for r in range(cfg.block1_rounds + 1):
            for f in STATE_FIELDS:
                spec.state_vars.append(f"b1_s{r}.{f}")
        # Block 2: starts at round_start (default 0 -> b2_s0 = IV, or window start >= 1 -> b2_s0 free)
        if cfg.round_start > 0:
            for f in STATE_FIELDS:
                spec.base_vars.append(f"b2_s{cfg.round_start}.{f}")
        for r in range(cfg.round_start, cfg.round_start + cfg.block2_rounds + 1):
            for f in STATE_FIELDS:
                spec.state_vars.append(f"b2_s{r}.{f}")
        for t in range(max(16, cfg.round_start + cfg.block2_rounds)):
            spec.msg_vars.append(f"b2_w{t}")
        # Global round indices: Block 1 rounds 0..r1-1, Block 2 rounds 100+round_start..100+round_start+r2-1
        spec.round_indices = list(range(cfg.block1_rounds)) + [
            100 + cfg.round_start + i for i in range(cfg.block2_rounds)
        ]
    return spec


def eval_world_np(
    cfg: RoundsConfig, base_env: Dict[str, np.ndarray], batch_size: int
) -> Dict[str, np.ndarray]:
    env: Dict[str, np.ndarray] = dict(base_env)
    if not cfg.btc_splice:
        s = {f: env[f"s0.{f}"] for f in STATE_FIELDS}
        for f in STATE_FIELDS:
            env[f"s0.{f}"] = s[f]
        if cfg.use_msg_schedule:
            w_arr: Dict[int, np.ndarray] = {}
            if cfg.padded_mode:
                for i in range(cfg.padded_free_words):
                    w_arr[i] = env[f"w{i}"]
                w_arr[cfg.padded_free_words] = np.full((batch_size,), np.uint32(0x80000000), dtype=np.uint32)
                for i in range(cfg.padded_free_words + 1, 15):
                    w_arr[i] = np.zeros((batch_size,), dtype=np.uint32)
                w_arr[15] = np.full((batch_size,), np.uint32(cfg.padded_free_words * 32), dtype=np.uint32)
            else:
                for i in range(16):
                    w_arr[i] = env[f"w{i}"]
            end_r = cfg.round_start + cfg.num_rounds
            for t in range(16, end_r):
                w_arr[t] = _np_sched(w_arr[t - 2], w_arr[t - 7], w_arr[t - 15], w_arr[t - 16])
            for t in range(end_r):
                env[f"w{t}"] = w_arr[t]
        for r in range(cfg.num_rounds):
            t = cfg.round_start + r
            w_t = env[f"w{t}"]
            s = _np_round(s, SHA256_K[t & 63], w_t, batch_size)
            for f in STATE_FIELDS:
                env[f"s{r+1}.{f}"] = s[f]
    else:
        # Block 1
        s1 = {f: env[f"b1_s0.{f}"] for f in STATE_FIELDS}
        for f in STATE_FIELDS:
            env[f"b1_s0.{f}"] = s1[f]
        for r in range(cfg.block1_rounds):
            k_idx = (64 - cfg.block1_rounds + r) & 63
            s1 = _np_round(s1, SHA256_K[k_idx], env[f"b1_w{r}"], batch_size)
            for f in STATE_FIELDS:
                env[f"b1_s{r+1}.{f}"] = s1[f]
        # Splice: w2[0..7] = IV + b1_s_end
        w2: Dict[int, np.ndarray] = {}
        for idx, f in enumerate(STATE_FIELDS):
            iv_arr = np.full((batch_size,), np.uint32(SHA256_IV[idx]), dtype=np.uint32)
            w2[idx] = (iv_arr + s1[f]).astype(np.uint32)
        w2[8] = np.full((batch_size,), np.uint32(0x80000000), dtype=np.uint32)
        for idx in range(9, 15):
            w2[idx] = np.zeros((batch_size,), dtype=np.uint32)
        w2[15] = np.full((batch_size,), np.uint32(256), dtype=np.uint32)
        end_b2 = cfg.round_start + cfg.block2_rounds
        for t in range(16, max(16, end_b2)):
            w2[t] = _np_sched(w2[t - 2], w2[t - 7], w2[t - 15], w2[t - 16])
        for t, arr in w2.items():
            env[f"b2_w{t}"] = arr
        if cfg.round_start == 0:
            s2 = {
                f: np.full((batch_size,), np.uint32(SHA256_IV[i]), dtype=np.uint32)
                for i, f in enumerate(STATE_FIELDS)
            }
        else:
            s2 = {f: env[f"b2_s{cfg.round_start}.{f}"] for f in STATE_FIELDS}
        for f in STATE_FIELDS:
            env[f"b2_s{cfg.round_start}.{f}"] = s2[f]
        for i in range(cfg.block2_rounds):
            t = cfg.round_start + i
            s2 = _np_round(s2, SHA256_K[t & 63], w2[t], batch_size)
            for f in STATE_FIELDS:
                env[f"b2_s{t+1}.{f}"] = s2[f]
    return env


def eval_world_z3(
    cfg: RoundsConfig,
    base_env: Dict[str, z3.BitVecRef],
    broken_rounds: Optional[Set[int]] = None,
    break_splice: bool = False,
    break_sched: bool = False,
    suffix: str = "",
) -> Dict[str, z3.BitVecRef]:
    """Compile SHA-256 world to Z3 BitVec32, optionally breaking specific rounds/splice/schedule."""
    broken_rounds = broken_rounds or set()
    env: Dict[str, z3.BitVecRef] = dict(base_env)

    def _fresh(name: str) -> z3.BitVecRef:
        return z3.BitVec(f"{name}{suffix}", 32)

    if not cfg.btc_splice:
        s = {f: env[f"s0.{f}"] for f in STATE_FIELDS}
        if cfg.use_msg_schedule:
            w_arr: Dict[int, z3.BitVecRef] = {}
            if cfg.padded_mode:
                for i in range(cfg.padded_free_words):
                    w_arr[i] = env[f"w{i}"]
                w_arr[cfg.padded_free_words] = z3.BitVecVal(0x80000000, 32)
                for i in range(cfg.padded_free_words + 1, 15):
                    w_arr[i] = z3.BitVecVal(0, 32)
                w_arr[15] = z3.BitVecVal(cfg.padded_free_words * 32, 32)
            else:
                for i in range(16):
                    w_arr[i] = env[f"w{i}"]
            end_r = cfg.round_start + cfg.num_rounds
            for t in range(16, end_r):
                if break_sched:
                    w_arr[t] = _fresh(f"broken_sched_w{t}")
                else:
                    w_arr[t] = _z3_sched(w_arr[t - 2], w_arr[t - 7], w_arr[t - 15], w_arr[t - 16])
            for t in range(end_r):
                env[f"w{t}"] = w_arr[t]
        for r in range(cfg.num_rounds):
            t = cfg.round_start + r
            broken_ae = None
            if t in broken_rounds:
                broken_ae = (_fresh(f"brk_a_{t}"), _fresh(f"brk_e_{t}"))
            s = _z3_round(s, SHA256_K[t & 63], env[f"w{t}"], broken_ae=broken_ae)
            for f in STATE_FIELDS:
                env[f"s{r+1}.{f}"] = s[f]
    else:
        s1 = {f: env[f"b1_s0.{f}"] for f in STATE_FIELDS}
        for r in range(cfg.block1_rounds):
            k_idx = (64 - cfg.block1_rounds + r) & 63
            broken_ae = None
            if r in broken_rounds:
                broken_ae = (_fresh(f"brk_b1_a_{r}"), _fresh(f"brk_b1_e_{r}"))
            s1 = _z3_round(s1, SHA256_K[k_idx], env[f"b1_w{r}"], broken_ae=broken_ae)
            for f in STATE_FIELDS:
                env[f"b1_s{r+1}.{f}"] = s1[f]
        w2: Dict[int, z3.BitVecRef] = {}
        for idx, f in enumerate(STATE_FIELDS):
            if break_splice:
                w2[idx] = _fresh(f"brk_splice_w{idx}")
            else:
                w2[idx] = z3.BitVecVal(SHA256_IV[idx], 32) + s1[f]
        w2[8] = z3.BitVecVal(0x80000000, 32)
        for idx in range(9, 15):
            w2[idx] = z3.BitVecVal(0, 32)
        w2[15] = z3.BitVecVal(256, 32)
        end_b2 = cfg.round_start + cfg.block2_rounds
        for t in range(16, max(16, end_b2)):
            if break_sched:
                w2[t] = _fresh(f"brk_b2_sched_w{t}")
            else:
                w2[t] = _z3_sched(w2[t - 2], w2[t - 7], w2[t - 15], w2[t - 16])
        for t, bv in w2.items():
            env[f"b2_w{t}"] = bv
        if cfg.round_start == 0:
            s2 = {f: z3.BitVecVal(SHA256_IV[i], 32) for i, f in enumerate(STATE_FIELDS)}
        else:
            s2 = {f: env[f"b2_s{cfg.round_start}.{f}"] for f in STATE_FIELDS}
        for f in STATE_FIELDS:
            env[f"b2_s{cfg.round_start}.{f}"] = s2[f]
        for i in range(cfg.block2_rounds):
            t = cfg.round_start + i
            gid = 100 + t
            broken_ae = None
            if gid in broken_rounds:
                broken_ae = (_fresh(f"brk_b2_a_{t}"), _fresh(f"brk_b2_e_{t}"))
            s2 = _z3_round(s2, SHA256_K[t & 63], w2[t], broken_ae=broken_ae)
            for f in STATE_FIELDS:
                env[f"b2_s{t+1}.{f}"] = s2[f]
    return env


def make_z3_base_env(spec: WorldSpec, suffix: str = "") -> Dict[str, z3.BitVecRef]:
    return {v: z3.BitVec(f"{v}{suffix}", 32) for v in spec.base_vars}
