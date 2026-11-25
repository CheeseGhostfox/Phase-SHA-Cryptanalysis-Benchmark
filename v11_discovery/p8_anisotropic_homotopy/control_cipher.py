"""Isomorphic random ROTR/K control cipher matching V6A SHA-256 circuit topology."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from src.dual_basis import spin_rotr, spin_shr, spin_Ch, spin_Maj
from src.sha256_ref import (
    MASK32,
    SHA256Trace,
    ch32,
    get_iv_tensor,
    maj32,
    rotr32,
    shr32,
    words_to_bits,
    words_to_spins,
)
from v6a.v6a_homotopy import (
    V6ABoundary,
    V6ACircuitSpec,
    _aperture_lsb_bit_mask,
    _compute_add32_carry_words,
    _domain_w16_template,
    _rotr_bit_idx,
    _shr_bit_idx,
)

STD_SIGMA0_UPPER = (2, 13, 22)
STD_SIGMA1_UPPER = (6, 11, 25)
STD_SIGMA0_LOWER = (7, 18, 3)   # (rotr1, rotr2, shr)
STD_SIGMA1_LOWER = (17, 19, 10)  # (rotr1, rotr2, shr)


@dataclass(frozen=True)
class ControlCipherParams:
    """Parameters of an isomorphic random ROTR/K clone of SHA-256."""

    seed: int
    k_words: Tuple[int, ...]          # length 64
    sigma0_upper: Tuple[int, int, int]
    sigma1_upper: Tuple[int, int, int]
    sigma0_lower: Tuple[int, int, int]  # (rotr1, rotr2, shr)
    sigma1_lower: Tuple[int, int, int]  # (rotr1, rotr2, shr)

    def to_dict(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "sigma0_upper": list(self.sigma0_upper),
            "sigma1_upper": list(self.sigma1_upper),
            "sigma0_lower": list(self.sigma0_lower),
            "sigma1_lower": list(self.sigma1_lower),
            "k_words_head8": [hex(x) for x in self.k_words[:8]],
        }


def make_control_cipher_params(seed: int = 20261125) -> ControlCipherParams:
    """Sample deterministic isomorphic random ROTR/SHR/K parameters excluding SHA-256 values."""
    rng = random.Random(seed)
    k_words = tuple(rng.getrandbits(32) for _ in range(64))

    def _sample_upper(forbidden: Tuple[int, int, int]) -> Tuple[int, int, int]:
        while True:
            rots = tuple(sorted(rng.sample(range(1, 32), 3)))
            if rots != tuple(sorted(forbidden)):
                return rots  # type: ignore[return-value]

    def _sample_lower(forbidden: Tuple[int, int, int], used_sh: set[int]) -> Tuple[int, int, int]:
        while True:
            r1, r2 = sorted(rng.sample(range(1, 32), 2))
            sh = rng.choice([s for s in range(1, 16) if s not in used_sh])
            cand = (r1, r2, sh)
            if (r1, r2) != tuple(sorted(forbidden[:2])) and sh != forbidden[2]:
                used_sh.add(sh)
                return cand

    s0_u = _sample_upper(STD_SIGMA0_UPPER)
    s1_u = _sample_upper(STD_SIGMA1_UPPER)
    used_sh: set[int] = set()
    s0_l = _sample_lower(STD_SIGMA0_LOWER, used_sh)
    s1_l = _sample_lower(STD_SIGMA1_LOWER, used_sh)
    return ControlCipherParams(
        seed=seed,
        k_words=k_words,
        sigma0_upper=s0_u,
        sigma1_upper=s1_u,
        sigma0_lower=s0_l,
        sigma1_lower=s1_l,
    )


def ctrl_sigma0_upper(x: torch.Tensor, p: ControlCipherParams) -> torch.Tensor:
    r0, r1, r2 = p.sigma0_upper
    return (rotr32(x, r0) ^ rotr32(x, r1) ^ rotr32(x, r2)) & MASK32


def ctrl_sigma1_upper(x: torch.Tensor, p: ControlCipherParams) -> torch.Tensor:
    r0, r1, r2 = p.sigma1_upper
    return (rotr32(x, r0) ^ rotr32(x, r1) ^ rotr32(x, r2)) & MASK32


def ctrl_sigma0_lower(x: torch.Tensor, p: ControlCipherParams) -> torch.Tensor:
    r0, r1, sh = p.sigma0_lower
    return (rotr32(x, r0) ^ rotr32(x, r1) ^ shr32(x, sh)) & MASK32


def ctrl_sigma1_lower(x: torch.Tensor, p: ControlCipherParams) -> torch.Tensor:
    r0, r1, sh = p.sigma1_lower
    return (rotr32(x, r0) ^ rotr32(x, r1) ^ shr32(x, sh)) & MASK32


def spin_ctrl_sigma0_upper(x: torch.Tensor, p: ControlCipherParams) -> torch.Tensor:
    r0, r1, r2 = p.sigma0_upper
    return spin_rotr(x, r0) * spin_rotr(x, r1) * spin_rotr(x, r2)


def spin_ctrl_sigma1_upper(x: torch.Tensor, p: ControlCipherParams) -> torch.Tensor:
    r0, r1, r2 = p.sigma1_upper
    return spin_rotr(x, r0) * spin_rotr(x, r1) * spin_rotr(x, r2)


def spin_ctrl_sigma0_lower(x: torch.Tensor, p: ControlCipherParams) -> torch.Tensor:
    r0, r1, sh = p.sigma0_lower
    return spin_rotr(x, r0) * spin_rotr(x, r1) * spin_shr(x, sh)


def spin_ctrl_sigma1_lower(x: torch.Tensor, p: ControlCipherParams) -> torch.Tensor:
    r0, r1, sh = p.sigma1_lower
    return spin_rotr(x, r0) * spin_rotr(x, r1) * spin_shr(x, sh)


def get_ctrl_k_tensor(
    rounds: int, params: ControlCipherParams, device: torch.device | str | None = None
) -> torch.Tensor:
    return torch.tensor(params.k_words[:rounds], dtype=torch.int64, device=device)


def expand_control_schedule(
    w16: torch.Tensor, rounds: int, params: ControlCipherParams
) -> torch.Tensor:
    if rounds <= 16:
        return (w16[:, :rounds] & MASK32).clone()
    bsz = w16.shape[0]
    w_cols = [w16[:, t] & MASK32 for t in range(16)]
    for t in range(16, rounds):
        wt = (
            ctrl_sigma1_lower(w_cols[t - 2], params)
            + w_cols[t - 7]
            + ctrl_sigma0_lower(w_cols[t - 15], params)
            + w_cols[t - 16]
        ) & MASK32
        w_cols.append(wt)
    return torch.stack(w_cols, dim=1).reshape(bsz, rounds)


@torch.no_grad()
def control_trace(
    w16: torch.Tensor, rounds: int, params: ControlCipherParams
) -> SHA256Trace:
    """Run vectorized isomorphic control cipher trace."""
    bsz = w16.shape[0]
    dev = w16.device
    iv = get_iv_tensor(dev)
    k_vec = get_ctrl_k_tensor(rounds, params, device=dev)
    w_full = expand_control_schedule(w16, rounds, params)

    a_cols = [iv[3].expand(bsz), iv[2].expand(bsz), iv[1].expand(bsz), iv[0].expand(bsz)]
    e_cols = [iv[7].expand(bsz), iv[6].expand(bsz), iv[5].expand(bsz), iv[4].expand(bsz)]
    for t in range(rounds):
        a_t, a_tm1, a_tm2, a_tm3 = a_cols[t + 3], a_cols[t + 2], a_cols[t + 1], a_cols[t]
        e_t, e_tm1, e_tm2, e_tm3 = e_cols[t + 3], e_cols[t + 2], e_cols[t + 1], e_cols[t]
        t1 = (
            e_tm3
            + ctrl_sigma1_upper(e_t, params)
            + ch32(e_t, e_tm1, e_tm2)
            + k_vec[t]
            + w_full[:, t]
        ) & MASK32
        t2 = (ctrl_sigma0_upper(a_t, params) + maj32(a_t, a_tm1, a_tm2)) & MASK32
        e_cols.append((a_tm3 + t1) & MASK32)
        a_cols.append((t1 + t2) & MASK32)

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
    digest = (s_r + iv.unsqueeze(0)) & MASK32
    return SHA256Trace(A_full=a_full, E_full=e_full, W_full=w_full, S_R=s_r, digest=digest)


@torch.no_grad()
def control_digest(
    w16: torch.Tensor, rounds: int, params: ControlCipherParams
) -> torch.Tensor:
    return control_trace(w16, rounds, params).digest


_CTRL_SPEC_CACHE: Dict[Tuple[int, int], V6ACircuitSpec] = {}


def build_control_circuit_spec(
    rounds: int, params: ControlCipherParams
) -> V6ACircuitSpec:
    """Build exact V6ACircuitSpec for isomorphic control cipher (never touches v6a cache)."""
    r = int(rounds)
    key = (r, params.seed)
    cached = _CTRL_SPEC_CACHE.get(key)
    if cached is not None:
        return cached

    a_start = 0
    e_start = r + 4
    w_start = e_start + r + 4
    k_start = w_start + r
    iv_start = k_start + r
    h_start = iv_start + 8
    const0_idx = h_start + 8
    cur = const0_idx + 1
    aux_start = cur

    aux_round: List[Dict[str, int]] = []
    for _ in range(r):
        d = {
            name: cur + idx
            for idx, name in enumerate(("S1", "Ch", "S0", "Maj", "u1", "u2", "u3", "T1", "T2"))
        }
        aux_round.append(d)
        cur += 9

    aux_sched: List[Dict[str, int]] = []
    for _ in range(max(0, r - 16)):
        d = {name: cur + idx for idx, name in enumerate(("s1", "s0", "v1", "v2"))}
        aux_sched.append(d)
        cur += 4

    add_triples: List[Tuple[int, int, int]] = []
    for t in range(r):
        atm3 = a_start + t
        etm3 = e_start + t
        atp1 = a_start + t + 4
        etp1 = e_start + t + 4
        wt = w_start + t
        kt = k_start + t
        aux = aux_round[t]
        add_triples.append((etm3, aux["S1"], aux["u1"]))
        add_triples.append((aux["u1"], aux["Ch"], aux["u2"]))
        add_triples.append((aux["u2"], kt, aux["u3"]))
        add_triples.append((aux["u3"], wt, aux["T1"]))
        add_triples.append((aux["S0"], aux["Maj"], aux["T2"]))
        add_triples.append((atm3, aux["T1"], etp1))
        add_triples.append((aux["T1"], aux["T2"], atp1))

    for idx_s, t in enumerate(range(16, r)):
        aux_s = aux_sched[idx_s]
        add_triples.append((aux_s["s1"], w_start + t - 7, aux_s["v1"]))
        add_triples.append((aux_s["s0"], w_start + t - 16, aux_s["v2"]))
        add_triples.append((aux_s["v1"], aux_s["v2"], w_start + t))

    sr_words = [
        a_start + r + 3,
        a_start + r + 2,
        a_start + r + 1,
        a_start + r,
        e_start + r + 3,
        e_start + r + 2,
        e_start + r + 1,
        e_start + r,
    ]
    for i in range(8):
        add_triples.append((sr_words[i], iv_start + i, h_start + i))

    carry_start = cur
    add_word_quads_list: List[Tuple[int, int, int, int]] = []
    for wa, wb, ws in add_triples:
        add_word_quads_list.append((wa, wb, ws, cur))
        cur += 1

    n_var_words = cur
    n_var_bits = n_var_words * 32

    def vb(w: int, b: int) -> int:
        return int(w) * 32 + int(b)

    chk_vars_list: List[Tuple[int, int, int, int]] = []
    chk_valid_list: List[Tuple[float, float, float, float]] = []
    ch_quads_list: List[Tuple[int, int, int, int]] = []
    maj_quads_list: List[Tuple[int, int, int, int]] = []

    s1_r0, s1_r1, s1_r2 = params.sigma1_upper
    s0_r0, s0_r1, s0_r2 = params.sigma0_upper
    sl1_r0, sl1_r1, sl1_sh = params.sigma1_lower
    sl0_r0, sl0_r1, sl0_sh = params.sigma0_lower

    for t in range(r):
        at, atm1, atm2 = a_start + t + 3, a_start + t + 2, a_start + t + 1
        et, etm1, etm2 = e_start + t + 3, e_start + t + 2, e_start + t + 1
        aux = aux_round[t]
        for b in range(32):
            chk_vars_list.append(
                (
                    _rotr_bit_idx(vb, et, s1_r0, b),
                    _rotr_bit_idx(vb, et, s1_r1, b),
                    _rotr_bit_idx(vb, et, s1_r2, b),
                    vb(aux["S1"], b),
                )
            )
            chk_valid_list.append((1.0, 1.0, 1.0, 1.0))

            chk_vars_list.append(
                (
                    _rotr_bit_idx(vb, at, s0_r0, b),
                    _rotr_bit_idx(vb, at, s0_r1, b),
                    _rotr_bit_idx(vb, at, s0_r2, b),
                    vb(aux["S0"], b),
                )
            )
            chk_valid_list.append((1.0, 1.0, 1.0, 1.0))

            ch_quads_list.append((vb(et, b), vb(etm1, b), vb(etm2, b), vb(aux["Ch"], b)))
            maj_quads_list.append((vb(at, b), vb(atm1, b), vb(atm2, b), vb(aux["Maj"], b)))

    for idx_s, t in enumerate(range(16, r)):
        aux_s = aux_sched[idx_s]
        wm2, wm15 = w_start + t - 2, w_start + t - 15
        for b in range(32):
            v_sh1, ok1 = _shr_bit_idx(vb, wm2, sl1_sh, b, const0_idx)
            chk_vars_list.append(
                (
                    _rotr_bit_idx(vb, wm2, sl1_r0, b),
                    _rotr_bit_idx(vb, wm2, sl1_r1, b),
                    v_sh1,
                    vb(aux_s["s1"], b),
                )
            )
            chk_valid_list.append((1.0, 1.0, float(ok1), 1.0))

            v_sh0, ok0 = _shr_bit_idx(vb, wm15, sl0_sh, b, const0_idx)
            chk_vars_list.append(
                (
                    _rotr_bit_idx(vb, wm15, sl0_r0, b),
                    _rotr_bit_idx(vb, wm15, sl0_r1, b),
                    v_sh0,
                    vb(aux_s["s0"], b),
                )
            )
            chk_valid_list.append((1.0, 1.0, float(ok0), 1.0))

    c0_bit = vb(const0_idx, 0)
    for wa, wb, ws, wc in add_word_quads_list:
        chk_vars_list.append((vb(wa, 0), vb(wb, 0), c0_bit, vb(ws, 0)))
        chk_valid_list.append((1.0, 1.0, 0.0, 1.0))
        for b in range(1, 32):
            chk_vars_list.append((vb(wa, b), vb(wb, b), vb(wc, b - 1), vb(ws, b)))
            chk_valid_list.append((1.0, 1.0, 1.0, 1.0))

    chk_vars = torch.tensor(chk_vars_list, dtype=torch.int64)
    chk_valid = torch.tensor(chk_valid_list, dtype=torch.float32)

    valid_mask = chk_valid > 0.5
    chk_idx_grid = torch.arange(chk_vars.shape[0], dtype=torch.int64).unsqueeze(1).expand_as(chk_vars)
    slot_grid = torch.arange(4, dtype=torch.int64).unsqueeze(0).expand_as(chk_vars)
    var_to_chk_idx = chk_idx_grid[valid_mask].contiguous()
    var_to_chk_slot = slot_grid[valid_mask].contiguous()
    var_to_chk_var = chk_vars[valid_mask].contiguous()

    ch_quads = torch.tensor(ch_quads_list, dtype=torch.int64)
    maj_quads = torch.tensor(maj_quads_list, dtype=torch.int64)
    add_word_quads = torch.tensor(add_word_quads_list, dtype=torch.int64)

    bit_offsets = torch.arange(32, dtype=torch.int64)
    add_bit_ids = (add_word_quads.unsqueeze(-1) * 32 + bit_offsets.view(1, 1, 32)).reshape(-1)
    nl_rep_to_var_bit = torch.cat([ch_quads.reshape(-1), maj_quads.reshape(-1), add_bit_ids], dim=0).contiguous()
    nl_deg = torch.bincount(nl_rep_to_var_bit, minlength=n_var_bits).to(torch.float32)
    var_total_degree = (1.0 + nl_deg).clamp_min(1.0)

    spec = V6ACircuitSpec(
        rounds=r,
        n_var_words=n_var_words,
        n_var_bits=n_var_bits,
        a_start=a_start,
        e_start=e_start,
        w_start=w_start,
        k_start=k_start,
        iv_start=iv_start,
        h_start=h_start,
        const0_idx=const0_idx,
        aux_start=aux_start,
        carry_start=carry_start,
        aux_round=aux_round,
        aux_sched=aux_sched,
        chk_vars=chk_vars,
        chk_valid=chk_valid,
        var_to_chk_idx=var_to_chk_idx,
        var_to_chk_slot=var_to_chk_slot,
        var_to_chk_var=var_to_chk_var,
        ch_quads=ch_quads,
        maj_quads=maj_quads,
        add_word_quads=add_word_quads,
        nl_rep_to_var_bit=nl_rep_to_var_bit,
        var_total_degree=var_total_degree,
    )
    _CTRL_SPEC_CACHE[key] = spec
    return spec


@torch.no_grad()
def embed_control_ground_truth_vars(
    w16: torch.Tensor,
    rounds: int,
    params: ControlCipherParams,
) -> torch.Tensor:
    """Embed exact discrete ground-truth trace (spins in {-1, +1}) for control cipher."""
    spec = build_control_circuit_spec(rounds, params)
    bsz = w16.shape[0]
    dev = w16.device
    tr = control_trace(w16, rounds, params)
    w_full = tr.W_full
    k_vec = get_ctrl_k_tensor(rounds, params, device=dev)
    iv = get_iv_tensor(device=dev)

    words = torch.zeros((bsz, spec.n_var_words), dtype=torch.int64, device=dev)
    words[:, spec.a_start : spec.a_start + rounds + 4] = tr.A_full
    words[:, spec.e_start : spec.e_start + rounds + 4] = tr.E_full
    words[:, spec.w_start : spec.w_start + rounds] = w_full
    words[:, spec.k_start : spec.k_start + rounds] = k_vec.unsqueeze(0)
    words[:, spec.iv_start : spec.iv_start + 8] = iv.unsqueeze(0)
    words[:, spec.h_start : spec.h_start + 8] = tr.digest
    words[:, spec.const0_idx] = 0

    for t in range(rounds):
        at, atm1, atm2 = tr.A_full[:, t + 3], tr.A_full[:, t + 2], tr.A_full[:, t + 1]
        et, etm1, etm2, etm3 = tr.E_full[:, t + 3], tr.E_full[:, t + 2], tr.E_full[:, t + 1], tr.E_full[:, t]
        s1 = ctrl_sigma1_upper(et, params)
        ch = ch32(et, etm1, etm2)
        s0 = ctrl_sigma0_upper(at, params)
        maj = maj32(at, atm1, atm2)
        u1 = (etm3 + s1) & MASK32
        u2 = (u1 + ch) & MASK32
        u3 = (u2 + k_vec[t]) & MASK32
        t1 = (u3 + w_full[:, t]) & MASK32
        t2 = (s0 + maj) & MASK32
        aux = spec.aux_round[t]
        for nm, val in (
            ("S1", s1), ("Ch", ch), ("S0", s0), ("Maj", maj),
            ("u1", u1), ("u2", u2), ("u3", u3), ("T1", t1), ("T2", t2),
        ):
            words[:, aux[nm]] = val

    for idx_s, t in enumerate(range(16, rounds)):
        s1_l = ctrl_sigma1_lower(w_full[:, t - 2], params)
        s0_l = ctrl_sigma0_lower(w_full[:, t - 15], params)
        v1 = (s1_l + w_full[:, t - 7]) & MASK32
        v2 = (s0_l + w_full[:, t - 16]) & MASK32
        aux_s = spec.aux_sched[idx_s]
        for nm, val in (("s1", s1_l), ("s0", s0_l), ("v1", v1), ("v2", v2)):
            words[:, aux_s[nm]] = val

    quads = spec.add_word_quads.to(dev)
    words[:, quads[:, 3]] = _compute_add32_carry_words(words[:, quads[:, 0]], words[:, quads[:, 1]])
    return words_to_spins(words, dtype=torch.float32).reshape(bsz, spec.n_var_bits)


@torch.no_grad()
def build_control_boundary(
    digest: torch.Tensor,
    rounds: int,
    params: ControlCipherParams,
    domain_mode: str = "free_msg",
    num_bits: int = 32,
    msg_bytes_len: int = 4,
    mask_m: int = 256,
) -> V6ABoundary:
    """Build exact V6ABoundary for control cipher (with allow_peel_back=False)."""
    r = int(rounds)
    spec = build_control_circuit_spec(r, params)
    if digest.ndim == 1:
        digest = digest.unsqueeze(0)
    bsz = digest.shape[0]
    dev = digest.device
    dig = digest.to(torch.int64) & MASK32

    iv = get_iv_tensor(dev)
    k_vec = get_ctrl_k_tensor(r, params, device=dev)

    word_vals = torch.zeros((bsz, spec.n_var_words), dtype=torch.int64, device=dev)
    word_mask = torch.zeros((bsz, spec.n_var_words, 32), dtype=torch.float32, device=dev)

    word_vals[:, spec.iv_start : spec.iv_start + 8] = iv.unsqueeze(0)
    word_mask[:, spec.iv_start : spec.iv_start + 8, :] = 1.0
    word_vals[:, spec.k_start : spec.k_start + r] = k_vec.unsqueeze(0)
    word_mask[:, spec.k_start : spec.k_start + r, :] = 1.0
    word_vals[:, spec.const0_idx] = 0
    word_mask[:, spec.const0_idx, :] = 1.0
    word_mask[:, spec.carry_start : spec.n_var_words, 31] = 1.0

    for i, idx in enumerate((3, 2, 1, 0)):
        word_vals[:, spec.a_start + i] = iv[idx]
        word_vals[:, spec.e_start + i] = iv[4 + idx]
    word_mask[:, spec.a_start : spec.a_start + 4, :] = 1.0
    word_mask[:, spec.e_start : spec.e_start + 4, :] = 1.0

    s_r_words = (dig - iv.unsqueeze(0)) & MASK32
    sr_slot_indices = [
        spec.a_start + r + 3, spec.a_start + r + 2, spec.a_start + r + 1, spec.a_start + r,
        spec.e_start + r + 3, spec.e_start + r + 2, spec.e_start + r + 1, spec.e_start + r,
    ]
    is_full_dig = int(mask_m) >= 256
    h_bit_mask = _aperture_lsb_bit_mask(mask_m, dev)
    h_bits = words_to_bits(dig).to(torch.float32)
    for i in range(8):
        w_h = spec.h_start + i
        w_sr = sr_slot_indices[i]
        word_vals[:, w_h] = dig[:, i]
        word_mask[:, w_h, :] = h_bit_mask[i].unsqueeze(0)
        word_vals[:, w_sr] = s_r_words[:, i]
        word_mask[:, w_sr, :] = h_bit_mask[i].unsqueeze(0)
    dig_t_mask = h_bit_mask.unsqueeze(0).expand(bsz, 8, 32).clone()

    w16_1d, w16_m2d = _domain_w16_template(domain_mode, num_bits, msg_bytes_len, dev)
    w16_words = w16_1d.unsqueeze(0).expand(bsz, 16).clone()
    w16_mask_bits = w16_m2d.unsqueeze(0).expand(bsz, 16, 32).clone()
    k_copy = min(r, 16)
    word_vals[:, spec.w_start : spec.w_start + k_copy] = w16_words[:, :k_copy]
    word_mask[:, spec.w_start : spec.w_start + k_copy, :] = w16_mask_bits[:, :k_copy, :]

    word_spins = words_to_spins(word_vals, dtype=torch.float32) * word_mask
    return V6ABoundary(
        rounds=r,
        domain_mode=domain_mode,
        mask_m=int(mask_m),
        allow_peel_back=False,
        var_mask=word_mask.reshape(bsz, spec.n_var_bits),
        var_vals=word_spins.reshape(bsz, spec.n_var_bits),
        A_full_clamped=word_spins[:, spec.a_start : spec.a_start + r + 4],
        A_full_mask=word_mask[:, spec.a_start : spec.a_start + r + 4],
        E_full_clamped=word_spins[:, spec.e_start : spec.e_start + r + 4],
        E_full_mask=word_mask[:, spec.e_start : spec.e_start + r + 4],
        W_full_clamped=word_spins[:, spec.w_start : spec.w_start + r],
        W_full_mask=word_mask[:, spec.w_start : spec.w_start + r],
        target_digests=dig,
        W16_clamped_words=w16_words,
        W16_mask_bits=w16_mask_bits,
        digest_target_bits=h_bits,
        digest_target_mask=dig_t_mask,
        is_full_digest=is_full_dig,
    )


@torch.no_grad()
def compute_control_2adic_energy(
    A_full: torch.Tensor,
    E_full: torch.Tensor,
    W_full: torch.Tensor,
    p_t_fp64: torch.Tensor,
    theta_k_fp64: torch.Tensor,
    rounds: int,
    params: ControlCipherParams,
    include_tail_e: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact 32-scale 2-adic energy for control cipher in FP64->FP32."""
    r = int(rounds)
    a_f = A_full.to(torch.float32)
    e_f = E_full.to(torch.float32)
    w_f = W_full.to(torch.float32)
    bsz = a_f.shape[0]
    dev = a_f.device

    def _to_phase(spins: torch.Tensor) -> torch.Tensor:
        return torch.matmul(0.5 * (1.0 - spins.to(torch.float64)), p_t_fp64)

    def _wrap(d_fp64: torch.Tensor) -> torch.Tensor:
        two_pi = 2.0 * math.pi
        return (torch.remainder(d_fp64 + math.pi, two_pi) - math.pi).to(torch.float32)

    theta_a = _to_phase(a_f)
    theta_e = _to_phase(e_f)
    theta_w = _to_phase(w_f)
    per_scale = torch.zeros((bsz, 32), dtype=torch.float32, device=dev)

    len_e = r if (include_tail_e or r == 4) else (r - 4)
    if len_e > 0:
        sig0_a = spin_ctrl_sigma0_upper(a_f[:, 3 : 3 + len_e], params)
        maj_a = spin_Maj(a_f[:, 3 : 3 + len_e], a_f[:, 2 : 2 + len_e], a_f[:, 1 : 1 + len_e])
        delta_e = _wrap(
            theta_e[:, 4 : 4 + len_e]
            + _to_phase(sig0_a)
            + _to_phase(maj_a)
            - theta_a[:, 4 : 4 + len_e]
            - theta_a[:, 0:len_e]
        )
        per_scale = per_scale + (2.0 * torch.sin(0.5 * delta_e).square()).sum(dim=1)

    sig1_e = spin_ctrl_sigma1_upper(e_f[:, 3 : 3 + r], params)
    ch_e = spin_Ch(e_f[:, 3 : 3 + r], e_f[:, 2 : 2 + r], e_f[:, 1 : 1 + r])
    delta_w = _wrap(
        theta_w
        + theta_k_fp64.unsqueeze(0)
        + _to_phase(sig1_e)
        + _to_phase(ch_e)
        + theta_e[:, 0:r]
        + theta_a[:, 0:r]
        - theta_e[:, 4 : 4 + r]
    )
    per_scale = per_scale + (2.0 * torch.sin(0.5 * delta_w).square()).sum(dim=1)

    if r > 16:
        sig1_w = spin_ctrl_sigma1_lower(w_f[:, 14 : r - 2], params)
        sig0_w = spin_ctrl_sigma0_lower(w_full_slice := w_f[:, 1 : r - 15], params)
        del w_full_slice
        delta_s = _wrap(
            theta_w[:, 16:r]
            - _to_phase(sig1_w)
            - theta_w[:, 9 : r - 7]
            - _to_phase(sig0_w)
            - theta_w[:, 0 : r - 16]
        )
        per_scale = per_scale + (2.0 * torch.sin(0.5 * delta_s).square()).sum(dim=1)

    return per_scale[:, 0], per_scale[:, 31], per_scale.sum(dim=-1)
