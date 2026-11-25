"""Phase-SHA V5D: Dual-Frequency Split (WHT + 32-bit Viterbi) & Neural-Guided RRR Reflection Engine.

Strictly aligned with Lean 4 Module 6D (formal_proofs/PhaseSHA/V5/DualDomainRRR.lean):
  - Enforces strict pure FP32 (allow_tf32=False, zero complex64).
  - Exact ProjectionPair (A: concur consensus + clamped boundary; B: exact Boolean WHT/vertex + exact
    32-step GPU-batched Viterbi modular 2^32 addition carry DP).
  - Enforces all 3 Lean 4 caveats:
      1) Shadow readout strictly from P_A(x_t), never raw x_t;
      2) Strict non-zero step clamp beta_t in [0.35, 1.15];
      3) Linear drift slope radar (drift_flag) monitoring ||x_t||_2 divergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dual_basis import DualBasisEngine
from src.sha256_ref import (
    MASK32,
    ch32,
    get_iv_tensor,
    get_k_tensor,
    maj32,
    sha256_digest,
    sha256_trace,
    sigma0_lower,
    sigma0_upper,
    sigma1_lower,
    sigma1_upper,
    spins_to_words,
    words_to_spins,
)
from src.v4_boundary import ManifoldBoundary
from src.v5_interfaces import (
    DynamicsOperator,
    OuterSearchShell,
    SolveResultV5,
    compute_calibrated_probs_from_candidates,
    compute_calibrated_probs_from_logits,
    enforce_v5_fp32,
)


def fast_wht_fp32(f: torch.Tensor) -> torch.Tensor:
    """Exact unnormalized iterative butterfly Walsh-Hadamard transform in pure FP32.

    Matches Lean 4 `wht f u = sum_x (-1)^{<u,x>_2} f(x)`:
      fast_wht_fp32(fast_wht_fp32(f)) == N * f.
    """
    out = f.to(torch.float32)
    n_dim = out.shape[-1]
    if n_dim == 0 or (n_dim & (n_dim - 1)) != 0:
        raise ValueError(f"WHT last dimension must be power of 2, got {n_dim}")
    orig_shape = out.shape
    h = 1
    while h < n_dim:
        y = out.view(*orig_shape[:-1], n_dim // (2 * h), 2, h)
        u = y[..., 0, :]
        v = y[..., 1, :]
        out = torch.stack([u + v, u - v], dim=-2).reshape(orig_shape)
        h *= 2
    return out


def wht_xor_conv_fp32(f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Exact Boolean-group XOR convolution via WHT spectral diagonalization."""
    n_dim = float(f.shape[-1])
    return fast_wht_fp32(fast_wht_fp32(f) * fast_wht_fp32(g)) / n_dim


def _build_bool_gate_vertices() -> Dict[str, torch.Tensor]:
    """Precompute exact {-1, +1} spin truth-table vertices (0 -> +1, 1 -> -1)."""
    xor2 = []
    for a in (0, 1):
        for b in (0, 1):
            y = a ^ b
            xor2.append([1.0 - 2.0 * a, 1.0 - 2.0 * b, 1.0 - 2.0 * y])

    xor3 = []
    ch_v = []
    maj_v = []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                y_xor = a ^ b ^ c
                y_ch = (a & b) ^ ((1 - a) & c)
                y_maj = (a & b) ^ (a & c) ^ (b & c)
                sa, sb, sc = 1.0 - 2.0 * a, 1.0 - 2.0 * b, 1.0 - 2.0 * c
                xor3.append([sa, sb, sc, 1.0 - 2.0 * y_xor])
                ch_v.append([sa, sb, sc, 1.0 - 2.0 * y_ch])
                maj_v.append([sa, sb, sc, 1.0 - 2.0 * y_maj])

    return {
        "xor2": torch.tensor(xor2, dtype=torch.float32),
        "xor3": torch.tensor(xor3, dtype=torch.float32),
        "ch": torch.tensor(ch_v, dtype=torch.float32),
        "maj": torch.tensor(maj_v, dtype=torch.float32),
    }


def _build_add8_table() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build 8-vertex spin table V8 (8,3) for (a,b,c) in {0,1}^3 and transition index groups."""
    triples: List[List[float]] = []
    idx_map: Dict[Tuple[int, int, int], int] = {}
    for ba in (0, 1):
        for bb in (0, 1):
            for bc in (0, 1):
                idx_map[(ba, bb, bc)] = len(triples)
                triples.append([1.0 - 2.0 * ba, 1.0 - 2.0 * bb, 1.0 - 2.0 * bc])
    v8 = torch.tensor(triples, dtype=torch.float32)  # (8, 3)

    # Carry transitions:
    # c_in=0 -> c_out=0: (0,0,0), (0,1,1), (1,0,1)
    # c_in=0 -> c_out=1: (1,1,0)
    # c_in=1 -> c_out=0: (0,0,1)
    # c_in=1 -> c_out=1: (0,1,0), (1,0,0), (1,1,1)
    t00_choices = torch.tensor(
        [idx_map[(0, 0, 0)], idx_map[(0, 1, 1)], idx_map[(1, 0, 1)]], dtype=torch.int64
    )
    t11_choices = torch.tensor(
        [idx_map[(0, 1, 0)], idx_map[(1, 0, 0)], idx_map[(1, 1, 1)]], dtype=torch.int64
    )
    t01_single = idx_map[(1, 1, 0)]
    t10_single = idx_map[(0, 0, 1)]
    return v8, t00_choices, t11_choices, torch.tensor([t01_single, t10_single], dtype=torch.int64)


class ExactLocalGateProjectors(nn.Module):
    """Exact Euclidean projectors onto Boolean truth tables & modular 2^32 addition."""

    def __init__(self) -> None:
        super().__init__()
        verts = _build_bool_gate_vertices()
        for k, t in verts.items():
            self.register_buffer(f"V_{k}", t, persistent=False)
        v8, t00_c, t11_c, t_sing = _build_add8_table()
        self.register_buffer("V8_add", v8, persistent=False)
        self.register_buffer("t00_choices", t00_c, persistent=False)
        self.register_buffer("t11_choices", t11_c, persistent=False)
        self.register_buffer("t_sing", t_sing, persistent=False)

    @torch.amp.autocast("cuda", enabled=False)
    def project_gate_vertices(
        self,
        x: torch.Tensor,
        vertices: torch.Tensor,
        ste_temp: float = 0.0,
    ) -> torch.Tensor:
        """Project (..., M_arity) onto exact discrete truth-table vertices in pure FP32.

        When ste_temp > 0 and torch.is_grad_enabled(), forwards the EXACT hard discrete vertex
        while supplying a smooth softmax-weighted straight-through Jacobian.
        """
        xf = x.to(torch.float32)
        scores = torch.matmul(xf, vertices.t())  # (..., K_vert)
        idx = torch.argmax(scores, dim=-1)
        hard = vertices[idx]
        if ste_temp > 0.0 and torch.is_grad_enabled() and xf.requires_grad:
            soft_w = torch.softmax(scores * float(ste_temp), dim=-1)
            soft = torch.matmul(soft_w, vertices)
            return soft + (hard - soft).detach()
        return hard

    def project_xor2(self, x: torch.Tensor, ste_temp: float = 0.0) -> torch.Tensor:
        return self.project_gate_vertices(x, self.V_xor2, ste_temp=ste_temp)

    def project_xor3(self, x: torch.Tensor, ste_temp: float = 0.0) -> torch.Tensor:
        return self.project_gate_vertices(x, self.V_xor3, ste_temp=ste_temp)

    def project_ch(self, x: torch.Tensor, ste_temp: float = 0.0) -> torch.Tensor:
        return self.project_gate_vertices(x, self.V_ch, ste_temp=ste_temp)

    def project_maj(self, x: torch.Tensor, ste_temp: float = 0.0) -> torch.Tensor:
        return self.project_gate_vertices(x, self.V_maj, ste_temp=ste_temp)

    @torch.no_grad()
    def viterbi_add32_project(
        self, a_s: torch.Tensor, b_s: torch.Tensor, c_s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact GPU-batched 32-step carry Viterbi DP for (a + b) mod 2^32 == c.

        Inputs: a_s, b_s, c_s of shape (..., 32) LSB-first bipolar spins (float32).
        Returns exact discrete spin tensors (a_p, b_p, c_p) minimizing total L2^2 distance
        subject to spins_to_words(a_p) + spins_to_words(b_p) == spins_to_words(c_p) (mod 2^32).
        """
        orig_shape = a_s.shape[:-1]
        device = a_s.device
        x_cat = torch.stack(
            [a_s.to(torch.float32), b_s.to(torch.float32), c_s.to(torch.float32)], dim=-1
        ).reshape(-1, 32, 3)  # (M, 32, 3)
        m_cnt = x_cat.shape[0]
        if m_cnt == 0:
            z = a_s.to(torch.float32)
            return z, z.clone(), z.clone()

        # Max <x_k, v_k> is strictly equivalent to min ||x_k - v_k||_2^2 because ||v_k||_2^2 == 3
        scores8 = torch.matmul(x_cat, self.V8_add.t())  # (M, 32, 8)
        t00_sub = scores8[:, :, self.t00_choices]  # (M, 32, 3)
        t00_max, t00_rel = t00_sub.max(dim=-1)
        t00_vidx = self.t00_choices[t00_rel]  # (M, 32)

        t11_sub = scores8[:, :, self.t11_choices]  # (M, 32, 3)
        t11_max, t11_rel = t11_sub.max(dim=-1)
        t11_vidx = self.t11_choices[t11_rel]  # (M, 32)

        t01_idx = int(self.t_sing[0].item())
        t10_idx = int(self.t_sing[1].item())
        t01_score = scores8[:, :, t01_idx]  # (M, 32)
        t10_score = scores8[:, :, t10_idx]  # (M, 32)

        # Forward DP (bit 0 has initial carry strictly 0)
        dp0 = t00_max[:, 0]
        dp1 = t01_score[:, 0]
        take1_from0 = [None] * 32
        take1_from1 = [None] * 32
        false_mask = torch.zeros(m_cnt, dtype=torch.bool, device=device)
        take1_from0[0] = false_mask
        take1_from1[0] = false_mask

        for k in range(1, 32):
            c0_0 = dp0 + t00_max[:, k]
            c0_1 = dp1 + t10_score[:, k]
            sel0 = c0_1 > c0_0
            new_dp0 = torch.where(sel0, c0_1, c0_0)

            c1_0 = dp0 + t01_score[:, k]
            c1_1 = dp1 + t11_max[:, k]
            sel1 = c1_1 > c1_0
            new_dp1 = torch.where(sel1, c1_1, c1_0)

            take1_from0[k] = sel0
            take1_from1[k] = sel1
            dp0, dp1 = new_dp0, new_dp1

        # Modular 2^32 final carry bit 32 is dropped -> choose best of dp0 and dp1
        cur_c = dp1 > dp0  # (M,) bool: 1 if carry out of bit 31 is True
        chosen_idx = [None] * 32
        for k in range(31, -1, -1):
            prev_c = torch.where(cur_c, take1_from1[k], take1_from0[k])
            v00 = t00_vidx[:, k]
            v11 = t11_vidx[:, k]
            vk = torch.where(
                ~prev_c & ~cur_c,
                v00,
                torch.where(
                    ~prev_c & cur_c,
                    torch.full_like(v00, t01_idx),
                    torch.where(prev_c & ~cur_c, torch.full_like(v00, t10_idx), v11),
                ),
            )
            chosen_idx[k] = vk
            cur_c = prev_c

        best_idx = torch.stack(chosen_idx, dim=1)  # (M, 32)
        proj = self.V8_add[best_idx]  # (M, 32, 3)
        a_p = proj[:, :, 0].reshape(*orig_shape, 32)
        b_p = proj[:, :, 1].reshape(*orig_shape, 32)
        c_p = proj[:, :, 2].reshape(*orig_shape, 32)
        return a_p, b_p, c_p

    @torch.amp.autocast("cuda", enabled=False)
    def project_add32(
        self, a_s: torch.Tensor, b_s: torch.Tensor, c_s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact forward Viterbi DP projection (detached from Viterbi discrete argmax)."""
        a_p, b_p, c_p = self.viterbi_add32_project(a_s.detach(), b_s.detach(), c_s.detach())
        return a_p, b_p, c_p


@dataclass
class ReplicaGraphSpec:
    """Static variable-gate factor-graph index tables for R-round SHA-256."""

    rounds: int
    n_var_words: int
    n_rep_bits: int
    # Bit-level index ranges inside flat replica tensor (..., n_rep_bits)
    xor3_offset: int
    xor3_count: int
    ch_offset: int
    ch_count: int
    maj_offset: int
    maj_count: int
    add_offset: int
    add_word_count: int
    # Variable bit index per replica position [0..n_rep_bits) -> [0..n_var_bits)
    rep_to_var_bit: torch.Tensor  # (n_rep_bits,) int64
    var_bit_degree: torch.Tensor  # (n_var_bits,) float32
    # Word index blocks inside var_words (..., n_var_words, 32)
    a_start: int
    e_start: int
    w_start: int
    k_start: int
    iv_start: int
    h_start: int
    const0_idx: int


def _build_sha256_replica_graph(rounds: int) -> ReplicaGraphSpec:
    """Compile SHA-256 R-round Boolean + modular-addition factor graph."""
    if not (8 <= rounds <= 64):
        raise ValueError(f"rounds must be in [8, 64], got {rounds}")

    # Allocate word IDs
    cur_w = 0
    a_start = cur_w
    cur_w += rounds + 4  # A_{-3..R} at a_start + (k+3)
    e_start = cur_w
    cur_w += rounds + 4  # E_{-3..R} at e_start + (k+3)
    w_start = cur_w
    cur_w += rounds  # W_{0..R-1}
    k_start = cur_w
    cur_w += rounds  # K_{0..R-1}
    iv_start = cur_w
    cur_w += 8  # IV_{0..7}
    h_start = cur_w
    cur_w += 8  # H_{0..7} (feedforward S_R + IV = H)
    const0_idx = cur_w
    cur_w += 1  # All-zero constant word for SHR padding

    # Aux words per round: S1, Ch, S0, Maj, u1, u2, u3, T1, T2
    aux_round: List[Dict[str, int]] = []
    for _t in range(rounds):
        d = {}
        for name in ("S1", "Ch", "S0", "Maj", "u1", "u2", "u3", "T1", "T2"):
            d[name] = cur_w
            cur_w += 1
        aux_round.append(d)

    # Expanded schedule aux words for t = 16..rounds-1: s1, s0, v1, v2
    aux_sched: List[Dict[str, int]] = []
    for _t in range(16, rounds):
        d = {}
        for name in ("s1", "s0", "v1", "v2"):
            d[name] = cur_w
            cur_w += 1
        aux_sched.append(d)

    n_var_words = cur_w
    n_var_bits = n_var_words * 32

    def vb(word_id: int, bit_idx: int) -> int:
        return word_id * 32 + bit_idx

    def rotr_vb(word_id: int, rot: int, bit_idx: int) -> int:
        return vb(word_id, (bit_idx + rot) % 32)

    def shr_vb(word_id: int, sh: int, bit_idx: int) -> int:
        if bit_idx + sh < 32:
            return vb(word_id, bit_idx + sh)
        return vb(const0_idx, 0)

    xor3_tuples: List[Tuple[int, int, int, int]] = []
    ch_tuples: List[Tuple[int, int, int, int]] = []
    maj_tuples: List[Tuple[int, int, int, int]] = []
    add_word_triples: List[Tuple[int, int, int]] = []

    for t in range(rounds):
        at = a_start + t + 3
        atm1 = a_start + t + 2
        atm2 = a_start + t + 1
        atm3 = a_start + t
        atp1 = a_start + t + 4

        et = e_start + t + 3
        etm1 = e_start + t + 2
        etm2 = e_start + t + 1
        etm3 = e_start + t
        etp1 = e_start + t + 4

        wt = w_start + t
        kt = k_start + t
        aux = aux_round[t]

        # Bit gates: Sigma1(E_t)=S1, Ch(E_t,E_{t-1},E_{t-2})=Ch, Sigma0(A_t)=S0, Maj=Maj
        for bit in range(32):
            xor3_tuples.append(
                (rotr_vb(et, 6, bit), rotr_vb(et, 11, bit), rotr_vb(et, 25, bit), vb(aux["S1"], bit))
            )
            xor3_tuples.append(
                (rotr_vb(at, 2, bit), rotr_vb(at, 13, bit), rotr_vb(at, 22, bit), vb(aux["S0"], bit))
            )
            ch_tuples.append((vb(et, bit), vb(etm1, bit), vb(etm2, bit), vb(aux["Ch"], bit)))
            maj_tuples.append((vb(at, bit), vb(atm1, bit), vb(atm2, bit), vb(aux["Maj"], bit)))

        # Modular addition chain
        add_word_triples.append((etm3, aux["S1"], aux["u1"]))
        add_word_triples.append((aux["u1"], aux["Ch"], aux["u2"]))
        add_word_triples.append((aux["u2"], kt, aux["u3"]))
        add_word_triples.append((aux["u3"], wt, aux["T1"]))
        add_word_triples.append((aux["S0"], aux["Maj"], aux["T2"]))
        add_word_triples.append((atm3, aux["T1"], etp1))
        add_word_triples.append((aux["T1"], aux["T2"], atp1))

    for idx_s, t in enumerate(range(16, rounds)):
        aux_s = aux_sched[idx_s]
        wm2 = w_start + t - 2
        wm7 = w_start + t - 7
        wm15 = w_start + t - 15
        wm16 = w_start + t - 16
        wt = w_start + t
        for bit in range(32):
            xor3_tuples.append(
                (rotr_vb(wm2, 17, bit), rotr_vb(wm2, 19, bit), shr_vb(wm2, 10, bit), vb(aux_s["s1"], bit))
            )
            xor3_tuples.append(
                (rotr_vb(wm15, 7, bit), rotr_vb(wm15, 18, bit), shr_vb(wm15, 3, bit), vb(aux_s["s0"], bit))
            )
        add_word_triples.append((aux_s["s1"], wm7, aux_s["v1"]))
        add_word_triples.append((aux_s["s0"], wm16, aux_s["v2"]))
        add_word_triples.append((aux_s["v1"], aux_s["v2"], wt))

    # Feedforward H_i = S_{R,i} + IV_i
    sr_words = [
        a_start + rounds + 3,
        a_start + rounds + 2,
        a_start + rounds + 1,
        a_start + rounds,
        e_start + rounds + 3,
        e_start + rounds + 2,
        e_start + rounds + 1,
        e_start + rounds,
    ]
    for i in range(8):
        add_word_triples.append((sr_words[i], iv_start + i, h_start + i))

    # Pack flat replica mapping in contiguous gate blocks: [XOR3 | Ch | Maj | ADD32]
    rep_ids: List[int] = []
    xor3_offset = len(rep_ids)
    for a_b, b_b, c_b, y_b in xor3_tuples:
        rep_ids.extend((a_b, b_b, c_b, y_b))
    xor3_count = len(xor3_tuples)

    ch_offset = len(rep_ids)
    for a_b, b_b, c_b, y_b in ch_tuples:
        rep_ids.extend((a_b, b_b, c_b, y_b))
    ch_count = len(ch_tuples)

    maj_offset = len(rep_ids)
    for a_b, b_b, c_b, y_b in maj_tuples:
        rep_ids.extend((a_b, b_b, c_b, y_b))
    maj_count = len(maj_tuples)

    add_offset = len(rep_ids)
    for wa, wb, wc in add_word_triples:
        for bit in range(32):
            rep_ids.append(vb(wa, bit))
        for bit in range(32):
            rep_ids.append(vb(wb, bit))
        for bit in range(32):
            rep_ids.append(vb(wc, bit))
    add_word_count = len(add_word_triples)

    rep_to_var_bit = torch.tensor(rep_ids, dtype=torch.int64)
    var_bit_degree = torch.bincount(rep_to_var_bit, minlength=n_var_bits).to(torch.float32).clamp_min(1.0)
    return ReplicaGraphSpec(
        rounds=rounds,
        n_var_words=n_var_words,
        n_rep_bits=len(rep_ids),
        xor3_offset=xor3_offset,
        xor3_count=xor3_count,
        ch_offset=ch_offset,
        ch_count=ch_count,
        maj_offset=maj_offset,
        maj_count=maj_count,
        add_offset=add_offset,
        add_word_count=add_word_count,
        rep_to_var_bit=rep_to_var_bit,
        var_bit_degree=var_bit_degree,
        a_start=a_start,
        e_start=e_start,
        w_start=w_start,
        k_start=k_start,
        iv_start=iv_start,
        h_start=h_start,
        const0_idx=const0_idx,
    )


class SHA256ReplicaGraph(nn.Module):
    """Exact R-round SHA-256 Variable-Gate Replica Graph supporting PA and PB projections."""

    def __init__(self, rounds: int = 18) -> None:
        super().__init__()
        self.rounds = int(rounds)
        self.spec = _build_sha256_replica_graph(self.rounds)
        self.projectors = ExactLocalGateProjectors()
        self.register_buffer("rep_to_var_bit", self.spec.rep_to_var_bit, persistent=False)
        self.register_buffer("var_bit_degree", self.spec.var_bit_degree, persistent=False)

        k_words = get_k_tensor(self.rounds, device=None)
        iv_words = get_iv_tensor(device=None)
        self.register_buffer("K_spins", words_to_spins(k_words, dtype=torch.float32), persistent=False)
        self.register_buffer("IV_spins", words_to_spins(iv_words, dtype=torch.float32), persistent=False)
        self.register_buffer("CONST_ZERO_spins", torch.ones((1, 32), dtype=torch.float32), persistent=False)

    @property
    def n_var_words(self) -> int:
        return self.spec.n_var_words

    @property
    def n_var_bits(self) -> int:
        return self.spec.n_var_words * 32

    @property
    def n_rep_bits(self) -> int:
        return self.spec.n_rep_bits

    @torch.no_grad()
    def build_boundary_clamps(
        self, boundary: ManifoldBoundary
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert ManifoldBoundary (full digest or V4E arbitrary mask) into variable-bit (mask, target)."""
        bsz = boundary.batch_size
        r = self.rounds
        dev = boundary.device
        s = self.spec

        var_mask = torch.zeros((bsz, s.n_var_words, 32), dtype=torch.float32, device=dev)
        var_vals = torch.zeros((bsz, s.n_var_words, 32), dtype=torch.float32, device=dev)

        # Primary A, E, W boundary clamps
        var_mask[:, s.a_start : s.a_start + r + 4] = boundary.A_full_mask.to(torch.float32)
        var_vals[:, s.a_start : s.a_start + r + 4] = boundary.A_full_clamped.to(torch.float32)
        var_mask[:, s.e_start : s.e_start + r + 4] = boundary.E_full_mask.to(torch.float32)
        var_vals[:, s.e_start : s.e_start + r + 4] = boundary.E_full_clamped.to(torch.float32)
        var_mask[:, s.w_start : s.w_start + r] = boundary.W_full_mask.to(torch.float32)
        var_vals[:, s.w_start : s.w_start + r] = boundary.W_full_clamped.to(torch.float32)

        # Fixed SHA-256 constants K, IV, CONST_ZERO
        var_mask[:, s.k_start : s.k_start + r] = 1.0
        var_vals[:, s.k_start : s.k_start + r] = self.K_spins.unsqueeze(0).expand(bsz, r, 32)
        var_mask[:, s.iv_start : s.iv_start + 8] = 1.0
        var_vals[:, s.iv_start : s.iv_start + 8] = self.IV_spins.unsqueeze(0).expand(bsz, 8, 32)
        var_mask[:, s.const0_idx : s.const0_idx + 1] = 1.0
        var_vals[:, s.const0_idx : s.const0_idx + 1] = 1.0

        # Feedforward digest H clamping (explicit V4E prefix or derived from clamped S_R)
        if boundary.digest_target_mask is not None and boundary.digest_target_bits is not None:
            d_mask = boundary.digest_target_mask.to(torch.float32).to(dev)
            d_spins = 1.0 - 2.0 * boundary.digest_target_bits.to(torch.float32).to(dev)
            var_mask[:, s.h_start : s.h_start + 8] = d_mask
            var_vals[:, s.h_start : s.h_start + 8] = d_spins * d_mask

        return var_mask.reshape(bsz, self.n_var_bits), var_vals.reshape(bsz, self.n_var_bits)

    @torch.no_grad()
    def embed_ground_truth_replicas(self, w16: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute exact discrete trace spins over ALL variables & replicas for verification/tests."""
        tr = sha256_trace(w16, rounds=self.rounds)
        bsz = w16.shape[0]
        dev = w16.device
        s = self.spec
        k_vec = get_k_tensor(self.rounds, device=dev)
        iv = get_iv_tensor(device=dev)

        words = torch.zeros((bsz, s.n_var_words), dtype=torch.int64, device=dev)
        words[:, s.a_start : s.a_start + self.rounds + 4] = tr.A_full
        words[:, s.e_start : s.e_start + self.rounds + 4] = tr.E_full
        words[:, s.w_start : s.w_start + self.rounds] = tr.W_full
        words[:, s.k_start : s.k_start + self.rounds] = k_vec.unsqueeze(0)
        words[:, s.iv_start : s.iv_start + 8] = iv.unsqueeze(0)
        words[:, s.h_start : s.h_start + 8] = tr.digest
        words[:, s.const0_idx] = 0

        cur = s.const0_idx + 1
        for t in range(self.rounds):
            at = tr.A_full[:, t + 3]
            atm1 = tr.A_full[:, t + 2]
            atm2 = tr.A_full[:, t + 1]
            et = tr.E_full[:, t + 3]
            etm1 = tr.E_full[:, t + 2]
            etm2 = tr.E_full[:, t + 1]
            etm3 = tr.E_full[:, t]
            s1 = sigma1_upper(et)
            ch = ch32(et, etm1, etm2)
            s0 = sigma0_upper(at)
            maj = maj32(at, atm1, atm2)
            u1 = (etm3 + s1) & MASK32
            u2 = (u1 + ch) & MASK32
            u3 = (u2 + k_vec[t]) & MASK32
            t1 = (u3 + tr.W_full[:, t]) & MASK32
            t2 = (s0 + maj) & MASK32
            for val in (s1, ch, s0, maj, u1, u2, u3, t1, t2):
                words[:, cur] = val
                cur += 1
        for t in range(16, self.rounds):
            s1_l = sigma1_lower(tr.W_full[:, t - 2])
            s0_l = sigma0_lower(tr.W_full[:, t - 15])
            v1 = (s1_l + tr.W_full[:, t - 7]) & MASK32
            v2 = (s0_l + tr.W_full[:, t - 16]) & MASK32
            for val in (s1_l, s0_l, v1, v2):
                words[:, cur] = val
                cur += 1

        var_spins = words_to_spins(words, dtype=torch.float32).reshape(bsz, self.n_var_bits)
        rep_spins = var_spins[:, self.rep_to_var_bit]
        return var_spins, rep_spins

    @torch.amp.autocast("cuda", enabled=False)
    def project_A_vars(
        self,
        x_rep: torch.Tensor,
        var_clamp_mask: torch.Tensor,
        var_clamp_vals: torch.Tensor,
        rep_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Exact Set-A Concur consensus projection returning variable-space vector (B, n_var_bits)."""
        xf = x_rep.to(torch.float32)
        bsz = xf.shape[0]
        idx = self.rep_to_var_bit.unsqueeze(0).expand(bsz, -1)
        if rep_weights is None:
            summed = torch.zeros((bsz, self.n_var_bits), dtype=torch.float32, device=xf.device)
            summed.scatter_add_(1, idx, xf)
            consensus = summed / self.var_bit_degree.unsqueeze(0)
        else:
            wf = rep_weights.to(torch.float32).clamp_min(1e-3)
            w_sum = torch.zeros((bsz, self.n_var_bits), dtype=torch.float32, device=xf.device)
            w_sum.scatter_add_(1, idx, wf)
            xw_sum = torch.zeros((bsz, self.n_var_bits), dtype=torch.float32, device=xf.device)
            xw_sum.scatter_add_(1, idx, xf * wf)
            consensus = xw_sum / w_sum.clamp_min(1e-6)

        m = var_clamp_mask.to(torch.float32)
        v = var_clamp_vals.to(torch.float32)
        return m * v + (1.0 - m) * consensus

    @torch.amp.autocast("cuda", enabled=False)
    def project_A(
        self,
        x_rep: torch.Tensor,
        var_clamp_mask: torch.Tensor,
        var_clamp_vals: torch.Tensor,
        rep_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Exact Set-A projection mapped back to replica tensor (B, n_rep_bits)."""
        v_a = self.project_A_vars(x_rep, var_clamp_mask, var_clamp_vals, rep_weights=rep_weights)
        return v_a[:, self.rep_to_var_bit]

    @torch.amp.autocast("cuda", enabled=False)
    def project_B(
        self,
        r_rep: torch.Tensor,
        ste_temp: float = 0.0,
    ) -> torch.Tensor:
        """Exact Set-B Divide gate projection onto local Boolean truth tables & 32-bit adders."""
        rf = r_rep.to(torch.float32)
        bsz = rf.shape[0]
        s = self.spec

        parts: List[torch.Tensor] = []
        if s.xor3_count > 0:
            x3_in = rf[:, s.xor3_offset : s.xor3_offset + 4 * s.xor3_count].reshape(
                bsz, s.xor3_count, 4
            )
            parts.append(self.projectors.project_xor3(x3_in, ste_temp=ste_temp).reshape(bsz, -1))
        if s.ch_count > 0:
            ch_in = rf[:, s.ch_offset : s.ch_offset + 4 * s.ch_count].reshape(
                bsz, s.ch_count, 4
            )
            parts.append(self.projectors.project_ch(ch_in, ste_temp=ste_temp).reshape(bsz, -1))
        if s.maj_count > 0:
            maj_in = rf[:, s.maj_offset : s.maj_offset + 4 * s.maj_count].reshape(
                bsz, s.maj_count, 4
            )
            parts.append(self.projectors.project_maj(maj_in, ste_temp=ste_temp).reshape(bsz, -1))
        if s.add_word_count > 0:
            add_in = rf[:, s.add_offset : s.add_offset + s.add_word_count * 96].reshape(
                bsz * s.add_word_count, 3, 32
            )
            a_p, b_p, c_p = self.projectors.project_add32(
                add_in[:, 0, :], add_in[:, 1, :], add_in[:, 2, :]
            )
            add_out = torch.stack([a_p, b_p, c_p], dim=1).reshape(bsz, s.add_word_count * 96)
            parts.append(add_out)

        return torch.cat(parts, dim=1)

    @torch.amp.autocast("cuda", enabled=False)
    def extract_primary_nodes(self, var_spins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract (A_full, E_full, W_full, x_nodes (B, R, 3, 32)) from variable-space shadow."""
        bsz = var_spins.shape[0]
        r = self.rounds
        s = self.spec
        words = var_spins.view(bsz, s.n_var_words, 32)
        a_full = words[:, s.a_start : s.a_start + r + 4]
        e_full = words[:, s.e_start : s.e_start + r + 4]
        w_full = words[:, s.w_start : s.w_start + r]
        x_nodes = torch.stack(
            [a_full[:, 3 : 3 + r], e_full[:, 3 : 3 + r], w_full], dim=2
        )
        return a_full, e_full, w_full, x_nodes


class RRRDynamicsOperator(DynamicsOperator):
    """V6 Slot 3: Exact Elser RRR Reflection Operator x <- x + beta * (P_B(2P_A(x)-x) - P_A(x)).

    Strictly enforces all Lean 4 caveats:
      1) Shadow readout via `readout_shadow(x)` (always P_A(x), NEVER raw x);
      2) Step size beta clamped strictly to [0.35, 1.15];
      3) Tracks ||x_t||_2 linear divergence drift slope & flag.
    """

    BETA_MIN: float = 0.35
    BETA_MAX: float = 1.15

    def __init__(self, graph: SHA256ReplicaGraph, ste_temp: float = 0.0) -> None:
        self.graph = graph
        self.ste_temp = float(ste_temp)

    @staticmethod
    def clamp_beta(beta: float | torch.Tensor) -> float | torch.Tensor:
        if isinstance(beta, torch.Tensor):
            return beta.to(torch.float32).clamp(RRRDynamicsOperator.BETA_MIN, RRRDynamicsOperator.BETA_MAX)
        return max(RRRDynamicsOperator.BETA_MIN, min(RRRDynamicsOperator.BETA_MAX, float(beta)))

    @torch.amp.autocast("cuda", enabled=False)
    def step(
        self,
        state: torch.Tensor,
        var_clamp_mask: torch.Tensor,
        var_clamp_vals: torch.Tensor,
        beta: float | torch.Tensor = 0.75,
        rep_weights: Optional[torch.Tensor] = None,
        ste_temp: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Execute one RRR step returning (x_next, diagnostics with y_A, y_B, gap)."""
        beta_c = self.clamp_beta(beta)
        if isinstance(beta_c, torch.Tensor) and beta_c.ndim == 1:
            beta_c = beta_c.unsqueeze(-1)
        temp = self.ste_temp if ste_temp is None else float(ste_temp)

        xf = state.to(torch.float32)
        y_a = self.graph.project_A(xf, var_clamp_mask, var_clamp_vals, rep_weights=rep_weights)
        r_ref = 2.0 * y_a - xf
        y_b = self.graph.project_B(r_ref, ste_temp=temp)
        diff = y_b - y_a
        x_next = xf + beta_c * diff
        gap_vec = torch.linalg.vector_norm(diff, dim=-1) / math.sqrt(float(xf.shape[-1]))
        x_norm = torch.linalg.vector_norm(x_next, dim=-1) / math.sqrt(float(xf.shape[-1]))
        return x_next, {"y_A": y_a, "y_B": y_b, "gap_rms": gap_vec, "x_rms_norm": x_norm}

    @torch.amp.autocast("cuda", enabled=False)
    def readout_shadow(
        self,
        state: torch.Tensor,
        var_clamp_mask: torch.Tensor,
        var_clamp_vals: torch.Tensor,
        rep_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Strictly extract primal shadow P_A(x) -> (A_full, E_full, W_full, x_nodes)."""
        v_a = self.graph.project_A_vars(
            state, var_clamp_mask, var_clamp_vals, rep_weights=rep_weights
        )
        return self.graph.extract_primary_nodes(v_a)

    @torch.amp.autocast("cuda", enabled=False)
    def gap_norm(
        self,
        state: torch.Tensor,
        var_clamp_mask: torch.Tensor,
        var_clamp_vals: torch.Tensor,
        rep_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xf = state.to(torch.float32)
        y_a = self.graph.project_A(xf, var_clamp_mask, var_clamp_vals, rep_weights=rep_weights)
        y_b = self.graph.project_B(2.0 * y_a - xf, ste_temp=0.0)
        return torch.linalg.vector_norm(y_b - y_a, dim=-1) / math.sqrt(float(xf.shape[-1]))


class DriftMonitor:
    """Linear divergence radar tracking ||x_t||_2 slope per Lean 4 `rrr_boundary_caveats`."""

    def __init__(self, slope_thresh: float = 1e-2, r2_thresh: float = 0.85, max_history: int = 64) -> None:
        self.slope_thresh = float(slope_thresh)
        self.r2_thresh = float(r2_thresh)
        self.max_history = max(4, int(max_history))
        self.history: List[torch.Tensor] = []

    def record(self, x_rms_norms: torch.Tensor) -> None:
        self.history.append(x_rms_norms.detach().to(torch.float32))
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def summarize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-trajectory (linear_slope, drift_flag) tensors."""
        if len(self.history) < 3:
            if not self.history:
                z = torch.zeros((1,), dtype=torch.float32)
                return z, z
            z = torch.zeros_like(self.history[0])
            return z, z

        y = torch.stack(self.history, dim=0)  # (T, B)
        t_cnt = y.shape[0]
        t = torch.arange(t_cnt, dtype=torch.float32, device=y.device).unsqueeze(-1)
        t_mean = t.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
        dt = t - t_mean
        dy = y - y_mean
        ss_tt = (dt * dt).sum(dim=0).clamp_min(1e-8)
        ss_yy = (dy * dy).sum(dim=0).clamp_min(1e-12)
        ss_ty = (dt * dy).sum(dim=0)
        slope = ss_ty / ss_tt
        r2 = (ss_ty * ss_ty) / (ss_tt * ss_yy)
        flag = ((slope > self.slope_thresh) & (r2 > self.r2_thresh)).to(torch.float32)
        return slope, flag


class V5DNeuralController(nn.Module):
    """Lightweight neural metric & RRR damping controller predicting (rep_weights, beta_t, logits)."""

    def __init__(self, rounds: int = 18, hidden_dim: int = 64) -> None:
        super().__init__()
        self.rounds = int(rounds)
        self.hidden_dim = int(hidden_dim)
        self.graph = SHA256ReplicaGraph(rounds=self.rounds)
        self.dual_basis = DualBasisEngine(rounds=self.rounds)

        # Variable-level controller features: [consensus_spin, mean_abs_gap, var_rms_diff, is_clamped]
        self.var_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Positive log-weight per variable mapped to rep weights in [0.25, 4.0] (preserves PA axiom!)
        self.weight_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)

        # Trajectory damping beta_t predictor mapped smoothly onto [0.35, 1.15]
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.beta_head[-1].weight)
        nn.init.constant_(self.beta_head[-1].bias, 0.0)

        # Primary-node calibrated spin logit correction head
        self.logit_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.logit_head.weight)
        nn.init.zeros_(self.logit_head.bias)

    def float(self) -> "V5DNeuralController":
        for name, mod in self.named_children():
            if name != "dual_basis":
                mod.float()
        return self

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        x_rep: torch.Tensor,
        var_clamp_mask: torch.Tensor,
        var_clamp_vals: torch.Tensor,
        step_frac: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (rep_weights in [0.25,4.0], beta in [0.35,1.15], primary_logits (B,R,3,32), x_nodes_a)."""
        xf = x_rep.to(torch.float32)
        bsz = xf.shape[0]
        v_a = self.graph.project_A_vars(xf, var_clamp_mask, var_clamp_vals)
        y_a = v_a[:, self.graph.rep_to_var_bit]
        y_b = self.graph.project_B(2.0 * y_a - xf, ste_temp=0.0)
        rep_gap = (y_b - y_a).abs()

        # Scatter gap magnitude to variable level
        idx = self.graph.rep_to_var_bit.unsqueeze(0).expand(bsz, -1)
        gap_sum = torch.zeros((bsz, self.graph.n_var_bits), dtype=torch.float32, device=xf.device)
        gap_sum.scatter_add_(1, idx, rep_gap)
        var_gap = gap_sum / self.graph.var_bit_degree.unsqueeze(0)

        diff_sq_sum = torch.zeros((bsz, self.graph.n_var_bits), dtype=torch.float32, device=xf.device)
        diff_sq_sum.scatter_add_(1, idx, (xf - y_a).square())
        var_spread = torch.sqrt((diff_sq_sum / self.graph.var_bit_degree.unsqueeze(0)).clamp_min(1e-8))

        feat = torch.stack(
            [v_a, var_gap, var_spread, var_clamp_mask.to(torch.float32)], dim=-1
        )  # (B, n_var_bits, 4)
        h = self.var_net(feat)  # (B, n_var_bits, H)

        # Positive metric weights in [0.25, 4.0] via bounded tanh -> exp(1.386 * tanh(.))
        raw_lw = 1.38629436 * torch.tanh(self.weight_head(h).squeeze(-1))
        var_weights = torch.exp(raw_lw)  # (B, n_var_bits) in [0.25, 4.0]
        rep_weights = var_weights[:, self.graph.rep_to_var_bit]

        # Global trajectory damping beta in [0.35, 1.15]
        h_pool = h.mean(dim=1)  # (B, H)
        sf = step_frac.to(torch.float32).view(-1, 1)
        if sf.shape[0] != bsz:
            sf = sf.expand(bsz, 1)
        b_raw = self.beta_head(torch.cat([h_pool, sf], dim=-1)).squeeze(-1)
        beta = 0.35 + 0.80 * torch.sigmoid(b_raw)

        # Primary node shadow + calibrated logit output
        _, _, _, x_nodes_a = self.graph.extract_primary_nodes(v_a)
        s = self.graph.spec
        r = self.rounds
        h_words = h.view(bsz, s.n_var_words, 32, self.hidden_dim)
        h_prim = torch.stack(
            [
                h_words[:, s.a_start + 3 : s.a_start + 3 + r],
                h_words[:, s.e_start + 3 : s.e_start + 3 + r],
                h_words[:, s.w_start : s.w_start + r],
            ],
            dim=2,
        )
        logits_delta = self.logit_head(h_prim).squeeze(-1)
        # Base logit anchors on shadow P_A value
        logits_spin = 2.0 * x_nodes_a + 0.5 * logits_delta
        return rep_weights, beta, logits_spin, x_nodes_a


class V5DRRRSolver(OuterSearchShell):
    """V6 Slot 4: Batched parallel RRR Reflection Solver supporting pure (0-param) & neural modes."""

    def __init__(
        self,
        rounds: int = 18,
        mode: str = "pure",
        controller: Optional[V5DNeuralController] = None,
        dual_basis: Optional[DualBasisEngine] = None,
        num_candidates: int = 256,
        num_steps: int = 48,
        chunk_size: int = 1024,
        beta_min: float = 0.38,
        beta_max: float = 1.05,
        init_scale: float = 0.85,
    ) -> None:
        self.rounds = int(rounds)
        self.mode = str(mode).lower()
        self.controller = controller
        self.graph = (
            controller.graph if controller is not None else SHA256ReplicaGraph(rounds=self.rounds)
        )
        self.dual_basis = (
            dual_basis if dual_basis is not None else DualBasisEngine(rounds=self.rounds)
        )
        self.rrr = RRRDynamicsOperator(self.graph, ste_temp=0.0)
        self.num_candidates = int(num_candidates)
        self.num_steps = int(num_steps)
        self.chunk_size = int(chunk_size)
        self.beta_min = max(0.35, float(beta_min))
        self.beta_max = min(1.15, float(beta_max))
        self.init_scale = float(init_scale)

    def to(self, device: torch.device) -> "V5DRRRSolver":
        self.graph.to(device)
        self.dual_basis.to(device)
        if self.controller is not None:
            self.controller.to(device)
        return self

    def _init_replicas(
        self,
        var_mask: torch.Tensor,
        var_vals: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        n_traj = var_mask.shape[0]
        dev = var_mask.device
        raw = (
            torch.randn(
                (n_traj, self.graph.n_rep_bits),
                dtype=torch.float32,
                device=dev,
                generator=generator,
            )
            * self.init_scale
        )
        rep_mask = var_mask[:, self.graph.rep_to_var_bit]
        rep_vals = var_vals[:, self.graph.rep_to_var_bit]
        return rep_mask * rep_vals + (1.0 - rep_mask) * raw

    @torch.inference_mode()
    def infer_deterministic_zero(
        self,
        boundary: ManifoldBoundary,
        num_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Deterministic single-trajectory RRR probe for eval_bce / round probes / drift radar."""
        device = boundary.device
        self.to(device)
        bsz = boundary.batch_size
        n_steps = self.num_steps if num_steps is None else int(num_steps)
        var_mask, var_vals = self.graph.build_boundary_clamps(boundary)

        # Deterministic zero free-bit initialization -> exact neutral P_A(0)
        x_cur = self.graph.project_A(
            torch.zeros((bsz, self.graph.n_rep_bits), dtype=torch.float32, device=device),
            var_mask,
            var_vals,
        )
        beta_mid = 0.5 * (self.beta_min + self.beta_max)
        drift = DriftMonitor()
        last_gap = torch.zeros(bsz, dtype=torch.float32, device=device)

        logits_last: Optional[torch.Tensor] = None
        for step_idx in range(n_steps):
            rep_w = None
            beta_val: float | torch.Tensor = beta_mid
            if self.mode == "neural" and self.controller is not None:
                sf = torch.full((bsz, 1), float(step_idx) / float(max(1, n_steps - 1)), dtype=torch.float32, device=device)
                rep_w, beta_val, logits_last, _ = self.controller(x_cur, var_mask, var_vals, sf)
            x_cur, diag = self.rrr.step(
                x_cur, var_mask, var_vals, beta=beta_val, rep_weights=rep_w
            )
            drift.record(diag["x_rms_norm"])
            last_gap = diag["gap_rms"]

        # Shadow readout strictly from P_A(x) (never raw x!)
        a_full, e_full, w_full, x_nodes = self.rrr.readout_shadow(x_cur, var_mask, var_vals)
        x_disc = torch.where(x_nodes < 0.0, -torch.ones_like(x_nodes), torch.ones_like(x_nodes))
        x_disc = boundary.to_clamped_boundary().apply_clamp(x_disc)
        if logits_last is None:
            # Map shadow consensus spin in [-1, 1] smoothly to BCE-calibrated logits
            logits_last = 1.5 * x_nodes

        a_d, e_d, w_d = boundary.to_clamped_boundary().assemble_full(x_disc)
        full_lam = torch.ones(32, dtype=torch.float32, device=device)
        _, _, stats = self.dual_basis.compute_energy_and_features(a_d, e_d, w_d, lam=full_lam)

        m1 = stats["per_scale"][:, 0]
        m32 = stats["per_scale"][:, 31]
        free_m = (1.0 - boundary.clamp_mask).clamp_min(0.0)
        align_sq = (1.0 - x_nodes.clamp(-1.0, 1.0).square()) * free_m
        mean_align = align_sq.sum(dim=(1, 2, 3)) / free_m.sum(dim=(1, 2, 3)).clamp_min(1.0)
        latch_ratio = torch.ones_like(mean_align) if float(last_gap.mean().item()) < 1e-3 else torch.zeros_like(mean_align)
        comm_zero = torch.zeros_like(mean_align)
        slope, flag = drift.summarize()
        extra = {
            "rrr_gap": float(last_gap.mean().item()),
            "rrr_beta": float(beta_mid if not isinstance(beta_val, torch.Tensor) else beta_val.mean().item()),
            "drift_slope": float(slope.mean().item()),
            "drift_flag": float(flag.max().item()),
        }
        return logits_last, x_disc, m1, m32, latch_ratio, mean_align, comm_zero, extra

    @torch.inference_mode()
    def solve(
        self,
        target_digests: Optional[torch.Tensor] = None,
        boundary: Optional[ManifoldBoundary] = None,
        generator: Optional[torch.Generator] = None,
        num_candidates: Optional[int] = None,
        num_steps: Optional[int] = None,
    ) -> SolveResultV5:
        """Run massive GPU-batched parallel RRR chaos trajectories and score strictly on P_A(x)."""
        if boundary is None:
            if target_digests is None:
                raise ValueError("Either boundary or target_digests must be provided.")
            from src.v4_boundary import build_manifold_boundary

            boundary = build_manifold_boundary(target_digests, rounds=self.rounds)

        device = boundary.device
        self.to(device)
        b_cnt = boundary.batch_size
        k_cand = self.num_candidates if num_candidates is None else int(num_candidates)
        n_steps = self.num_steps if num_steps is None else int(num_steps)
        if target_digests is None:
            target_digests = torch.zeros((b_cnt, 8), dtype=torch.int64, device=device)

        total_traj = b_cnt * k_cand
        b_rep = boundary.repeat_for_candidates(k_cand)

        best_disc_chunks: List[torch.Tensor] = []
        best_e32_chunks: List[torch.Tensor] = []
        best_m1_chunks: List[torch.Tensor] = []
        best_e_chunks: List[torch.Tensor] = []
        best_gap_chunks: List[torch.Tensor] = []
        slope_chunks: List[torch.Tensor] = []
        flag_chunks: List[torch.Tensor] = []
        beta_means: List[float] = []

        lam_full = torch.ones(32, dtype=torch.float32, device=device)

        for start in range(0, total_traj, self.chunk_size):
            end = min(start + self.chunk_size, total_traj)
            m_cur = end - start
            b_sub = ManifoldBoundary(
                rounds=b_rep.rounds,
                A_full_clamped=b_rep.A_full_clamped[start:end],
                A_full_mask=b_rep.A_full_mask[start:end],
                E_full_clamped=b_rep.E_full_clamped[start:end],
                E_full_mask=b_rep.E_full_mask[start:end],
                W_full_clamped=b_rep.W_full_clamped[start:end],
                W_full_mask=b_rep.W_full_mask[start:end],
                A_words_clamped=b_rep.A_words_clamped[start:end],
                E_words_clamped=b_rep.E_words_clamped[start:end],
                W16_clamped_words=b_rep.W16_clamped_words[start:end],
                W16_mask_bits=b_rep.W16_mask_bits[start:end],
                digest_target_bits=(
                    None if b_rep.digest_target_bits is None else b_rep.digest_target_bits[start:end]
                ),
                digest_target_mask=(
                    None if b_rep.digest_target_mask is None else b_rep.digest_target_mask[start:end]
                ),
                iv_words=b_rep.iv_words,
                is_full_digest_clamped=b_rep.is_full_digest_clamped,
            )
            var_mask, var_vals = self.graph.build_boundary_clamps(b_sub)
            x_cur = self._init_replicas(var_mask, var_vals, generator=generator)
            # Diverse deterministic beta ladder per candidate across [beta_min, beta_max]
            beta_ladder = torch.linspace(self.beta_min, self.beta_max, k_cand, dtype=torch.float32, device=device)
            traj_ids = torch.arange(start, end, device=device) % k_cand
            beta_vec = beta_ladder[traj_ids]

            drift = DriftMonitor()
            cb_sub = b_sub.to_clamped_boundary()

            best_x = None
            best_e = None
            best_m1 = None
            best_m32 = None
            best_gap = torch.full((m_cur,), 1e9, dtype=torch.float32, device=device)

            # Evaluate shadow P_A at periodic checkpoints & final step
            probe_every = max(1, n_steps // 4)
            for step_idx in range(n_steps):
                rep_w = None
                beta_step: torch.Tensor = beta_vec
                if self.mode == "neural" and self.controller is not None:
                    sf = torch.full(
                        (m_cur, 1),
                        float(step_idx) / float(max(1, n_steps - 1)),
                        dtype=torch.float32,
                        device=device,
                    )
                    rep_w, beta_pred, _, _ = self.controller(x_cur, var_mask, var_vals, sf)
                    beta_step = 0.5 * beta_pred + 0.5 * beta_vec
                x_cur, diag = self.rrr.step(
                    x_cur, var_mask, var_vals, beta=beta_step, rep_weights=rep_w
                )
                drift.record(diag["x_rms_norm"])
                beta_means.append(float(beta_step.mean().item()))

                if ((step_idx + 1) % probe_every == 0) or (step_idx == n_steps - 1):
                    _, _, _, x_nodes = self.rrr.readout_shadow(
                        x_cur, var_mask, var_vals, rep_weights=rep_w
                    )
                    x_disc = torch.where(
                        x_nodes < 0.0, -torch.ones_like(x_nodes), torch.ones_like(x_nodes)
                    )
                    x_disc = cb_sub.apply_clamp(x_disc)
                    a_d, e_d, w_d = cb_sub.assemble_full(x_disc)
                    tot_e, _, stats = self.dual_basis.compute_energy_and_features(
                        a_d, e_d, w_d, lam=lam_full
                    )
                    m1 = stats["per_scale"][:, 0]
                    m32 = stats["per_scale"][:, 31]
                    if best_x is None:
                        best_x, best_e, best_m1, best_m32 = x_disc, tot_e, m1, m32
                        best_gap = diag["gap_rms"]
                    else:
                        better = m32 < best_m32
                        b4 = better.view(m_cur, 1, 1, 1)
                        best_x = torch.where(b4, x_disc, best_x)
                        best_e = torch.where(better, tot_e, best_e)
                        best_m1 = torch.where(better, m1, best_m1)
                        best_m32 = torch.where(better, m32, best_m32)
                        best_gap = torch.where(better, diag["gap_rms"], best_gap)

            slope, flag = drift.summarize()
            best_disc_chunks.append(best_x)
            best_e_chunks.append(best_e)
            best_m1_chunks.append(best_m1)
            best_e32_chunks.append(best_m32)
            best_gap_chunks.append(best_gap)
            slope_chunks.append(slope)
            flag_chunks.append(flag)

        x_disc_all = torch.cat(best_disc_chunks, dim=0).view(b_cnt, k_cand, self.rounds, 3, 32)
        e_all = torch.cat(best_e_chunks, dim=0).view(b_cnt, k_cand)
        m1_all = torch.cat(best_m1_chunks, dim=0).view(b_cnt, k_cand)
        m32_all = torch.cat(best_e32_chunks, dim=0).view(b_cnt, k_cand)
        gap_all = torch.cat(best_gap_chunks, dim=0).view(b_cnt, k_cand)

        # Verify SHA-256 hash matches & check uniqueness over all candidates
        from src.sha256_ref import words_to_bits, bits_to_words

        k_w = min(16, self.rounds)
        w_pred_bits = (x_disc_all[:, :, :k_w, 2, :] < 0.0).to(torch.int64)
        if k_w < 16:
            pad_b = torch.zeros(
                (b_cnt, k_cand, 16 - k_w, 32), dtype=torch.int64, device=device
            )
            w_pred_bits = torch.cat([w_pred_bits, pad_b], dim=2)
        w16_m_int = (boundary.W16_mask_bits > 0.5).to(torch.int64).unsqueeze(1)
        bits_clamp = words_to_bits(boundary.W16_clamped_words).unsqueeze(1)
        w16_cand = bits_to_words(torch.where(w16_m_int > 0, bits_clamp, w_pred_bits))  # (B, K, 16)

        if boundary.is_full_digest_clamped:
            dig_cand = sha256_digest(w16_cand.reshape(-1, 16), rounds=self.rounds).view(b_cnt, k_cand, 8)
            hits_mat = (dig_cand == target_digests.unsqueeze(1)).all(dim=-1)
        elif boundary.digest_target_mask is not None and boundary.digest_target_bits is not None:
            dig_cand = sha256_digest(w16_cand.reshape(-1, 16), rounds=self.rounds).view(b_cnt, k_cand, 8)
            dig_bits = words_to_bits(dig_cand).to(torch.float32)
            tgt_b = boundary.digest_target_bits.unsqueeze(1).to(torch.float32)
            tgt_m = boundary.digest_target_mask.unsqueeze(1).to(torch.float32)
            err = ((dig_bits - tgt_b).abs() * tgt_m).sum(dim=(-2, -1))
            hits_mat = err < 0.5
        else:
            hits_mat = torch.zeros((b_cnt, k_cand), dtype=torch.bool, device=device)

        num_hits = hits_mat.sum(dim=1)
        solved = num_hits > 0

        best_idx = m32_all.argmin(dim=1)
        if solved.any():
            hit_score = torch.where(hits_mat, m32_all - 1e6, m32_all)
            best_idx = hit_score.argmin(dim=1)

        b_idx = torch.arange(b_cnt, device=device)
        best_x_disc = x_disc_all[b_idx, best_idx]
        best_w16 = w16_cand[b_idx, best_idx]
        best_energy = e_all[b_idx, best_idx]
        best_m1 = m1_all[b_idx, best_idx]
        best_m32 = m32_all[b_idx, best_idx]
        min_m32 = m32_all.min(dim=1).values

        # Unique W0 ratio among K candidates
        w0_vals = w16_cand[:, :, 0].detach().cpu().tolist()
        uniq_ratios = torch.tensor(
            [float(len(set(row))) / float(max(1, k_cand)) for row in w0_vals],
            dtype=torch.float32,
            device=device,
        )
        cal_probs, p_weights = compute_calibrated_probs_from_candidates(
            x_disc_all, m32_all, temperature=1.0
        )

        slopes_all = torch.cat(slope_chunks, dim=0)
        flags_all = torch.cat(flag_chunks, dim=0)
        return SolveResultV5(
            solved=solved,
            num_hits=num_hits,
            unique_ratio=uniq_ratios,
            latch_ratio=(gap_all.min(dim=1).values < 1e-3).to(torch.float32),
            best_w16=best_w16,
            best_energy=best_energy,
            best_m1_energy=best_m1,
            best_m32_energy=best_m32,
            min_m32_energy=min_m32,
            best_x_disc=best_x_disc,
            mean_align=torch.zeros(b_cnt, dtype=torch.float32, device=device),
            mean_comm=torch.zeros(b_cnt, dtype=torch.float32, device=device),
            calibrated_probs=cal_probs,
            particle_weights=p_weights,
            extra_metrics={
                "rrr_gap": float(gap_all.min(dim=1).values.mean().item()),
                "rrr_beta": float(sum(beta_means) / max(1, len(beta_means))),
                "drift_slope": float(slopes_all.mean().item()),
                "drift_flag": float(flags_all.max().item()),
            },
        )

    def export_calibrated_probs(self, solve_result: SolveResultV5) -> torch.Tensor:
        if solve_result.calibrated_probs is not None:
            return solve_result.calibrated_probs
        return compute_calibrated_probs_from_logits(1.5 * solve_result.best_x_disc)


__all__ = [
    "DriftMonitor",
    "ExactLocalGateProjectors",
    "RRRDynamicsOperator",
    "SHA256ReplicaGraph",
    "V5DNeuralController",
    "V5DRRRSolver",
    "fast_wht_fp32",
    "wht_xor_conv_fp32",
]
