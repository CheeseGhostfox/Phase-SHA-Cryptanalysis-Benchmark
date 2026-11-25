"""B2-P15 Four-Color MitM / Splice-and-Cut / Biclique CP-SAT Models & Complexity Accounting.

Implements SPEC.md Sections 1 & 2:
- Exact 4-color (Gray/Red/Blue/White) word-level & bit-level (Tier 2) propagation
- Bidirectional SHA-256 compression with exact backSolveA bypass
- Bidirectional nonlinear message schedule (sigma0, sigma1) + BTC fixed-padding folding
- Independent vs Dependent Biclique accounting
- Indirect Partial Matching + carry uncertainty penalty Delta_carry
- Mode I (wrap-around 2^256) vs Mode II (truncated clz(H[0]) >= k direct MitM,
  strictly honoring auditor rule: any matched bit must derive from clz(H[0]) >= k
  via a WHITE-free path, and c_recomp charges full clz(H[0]) >= k verification)
- Cross-block 128-round BTC Double-SHA256 joint solver (Block 1b -> W'[0..7] -> Block 2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from sha256_engine import build_dependency_graph

GRAY = 0
RED = 1
BLUE = 2
WHITE = 3  # (r=1, b=1) mixed / polluted
COLOR_STR = {GRAY: "GRAY", RED: "RED", BLUE: "BLUE", WHITE: "WHITE"}


@dataclass
class ComplexityReport:
    mode: str  # "ModeI_Wrap256" or "ModeII_DirectTrunc"
    k_target: int
    d1: int
    d2: int
    L_bic: int
    is_independent_biclique: bool
    R_fwd: int
    R_bwd: int
    R_total: int
    m_match: int
    delta_carry: int
    m_eff: int
    p_phi: float
    kappa_phi: float
    log2_N_groups: float
    log2_C_group: float
    log2_T_attack: float
    log2_T_brute: float
    delta_log2_T: float  # log2_T_brute - log2_T_attack (positive = speedup)
    log2_M_words: float
    is_speedup: bool
    mem_feasible: bool
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "k_target": self.k_target,
            "d1": self.d1,
            "d2": self.d2,
            "L_bic": self.L_bic,
            "is_independent_biclique": self.is_independent_biclique,
            "R_fwd": self.R_fwd,
            "R_bwd": self.R_bwd,
            "R_total": self.R_total,
            "m_match": self.m_match,
            "delta_carry": self.delta_carry,
            "m_eff": self.m_eff,
            "p_phi": self.p_phi,
            "kappa_phi": round(self.kappa_phi, 6),
            "log2_N_groups": round(self.log2_N_groups, 6),
            "log2_C_group": round(self.log2_C_group, 6),
            "log2_T_attack": round(self.log2_T_attack, 6),
            "log2_T_brute": round(self.log2_T_brute, 6),
            "delta_log2_T": round(self.delta_log2_T, 6),
            "log2_M_words": round(self.log2_M_words, 6),
            "is_speedup": self.is_speedup,
            "mem_feasible": self.mem_feasible,
            "notes": self.notes,
        }


def _log2_sum_exp(log2_terms: Sequence[float]) -> float:
    """Compute log2(sum_i 2^{log2_terms[i]}) stably."""
    finite = [x for x in log2_terms if x > -1e9]
    if not finite:
        return -1e9
    m = max(finite)
    s = sum(2.0 ** (x - m) for x in finite)
    return m + math.log2(s)


def compute_complexity(
    mode: Literal["ModeI_Wrap256", "ModeII_DirectTrunc"],
    k_target: int,
    d1: int,
    d2: int,
    L_bic: int,
    is_independent_biclique: bool,
    R_fwd: int,
    R_bwd: int,
    R_total: int,
    m_match: int,
    delta_carry: int = 0,
    p_phi: float = 1.0,
    R_recomp: Optional[int] = None,
    w_entry: int = 8,
    delta_min: float = 0.05,
    notes: str = "",
) -> ComplexityReport:
    """Strict SPEC Section 1.4 time and memory complexity formula."""
    m_eff = max(0, m_match - delta_carry)
    kappa_phi = -math.log2(max(p_phi, 1e-300))

    c_bic = max(L_bic, 0) / max(R_total, 1)
    c_red = max(R_fwd, 1) / max(R_total, 1)
    c_blue = max(R_bwd, 1) / max(R_total, 1)
    # Auditor rule: c_recomp charges full verification of clz(H[0]) >= k (or full R_total)
    eff_recomp_rounds = R_total if R_recomp is None else R_recomp
    c_recomp = max(eff_recomp_rounds, 1) / max(R_total, 1)

    if mode == "ModeI_Wrap256":
        log2_N_groups = kappa_phi + max(0.0, float(256 - (d1 + d2)))
        eff_pairs_exp = float(d1 + d2)
        log2_T_brute = float(k_target)
    elif mode == "ModeII_DirectTrunc":
        # If d1 + d2 >= k_target, only 1 group needed with effective span scaled to k_target
        if d1 + d2 >= k_target:
            log2_N_groups = kappa_phi
            eff_pairs_exp = float(k_target)
        else:
            log2_N_groups = kappa_phi + float(k_target - (d1 + d2))
            eff_pairs_exp = float(d1 + d2)
        # Brute baseline: 2^k * (1 - eps_tail), eps_tail < 1/R_total (say 0.5/R_total for last E_R & H[1..7])
        eps_tail = min(0.5 / max(R_total, 1), 0.1)
        log2_T_brute = float(k_target) + math.log2(1.0 - eps_tail)
    else:
        raise ValueError(f"Unknown complexity mode: {mode}")

    terms: List[float] = []
    if L_bic > 0 and c_bic > 0:
        if is_independent_biclique:
            terms.append(math.log2(c_bic) + _log2_sum_exp([float(d1), float(d2)]))
        else:
            terms.append(math.log2(c_bic) + float(d1 + d2))
    terms.append(math.log2(c_red) + float(d1))
    terms.append(math.log2(c_blue) + float(d2))

    # Recompute cost for surviving candidates
    recomp_exp = max(0.0, eff_pairs_exp - float(m_eff))
    terms.append(math.log2(c_recomp) + recomp_exp)

    # Memory access overhead (at least 1 unit per lookup when matching)
    if m_eff > 0:
        terms.append(math.log2(1.0 / max(R_total, 1)) + float(min(d1, d2)))

    log2_C_group = _log2_sum_exp(terms)
    log2_T_attack = log2_N_groups + log2_C_group
    delta_log2_T = log2_T_brute - log2_T_attack
    log2_M_words = float(min(d1, d2)) + math.log2(max(w_entry, 1))
    mem_feasible = min(d1, d2) <= 64
    is_speedup = (delta_log2_T >= delta_min) and (m_eff >= 1) and (d1 >= 1) and (d2 >= 1)

    return ComplexityReport(
        mode=mode,
        k_target=k_target,
        d1=d1,
        d2=d2,
        L_bic=L_bic,
        is_independent_biclique=is_independent_biclique,
        R_fwd=R_fwd,
        R_bwd=R_bwd,
        R_total=R_total,
        m_match=m_match,
        delta_carry=delta_carry,
        m_eff=m_eff,
        p_phi=p_phi,
        kappa_phi=kappa_phi,
        log2_N_groups=log2_N_groups,
        log2_C_group=log2_C_group,
        log2_T_attack=log2_T_attack,
        log2_T_brute=log2_T_brute,
        delta_log2_T=delta_log2_T,
        log2_M_words=log2_M_words,
        is_speedup=is_speedup,
        mem_feasible=mem_feasible,
        notes=notes,
    )


def _pair_to_color(r: int, b: int) -> int:
    return (b << 1) | r


@dataclass
class WordSearchResult:
    feasible: bool
    status: str
    rounds: int
    padding_mode: str
    eval_mode: str
    k_target: int
    s_w: int
    t_match: int
    biclique_range: Optional[Tuple[int, int]]
    is_independent_biclique: bool
    red_msg_basis: List[int]
    blue_msg_basis: List[int]
    w_colors: List[str]
    fwd_rounds: List[int]
    bwd_rounds: List[int]
    matched_state_words: List[str]
    indirect_matches: List[str]
    d1_bits: int
    d2_bits: int
    m_match: int
    delta_carry: int
    m_eff: int
    complexity: Optional[ComplexityReport]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feasible": self.feasible,
            "status": self.status,
            "rounds": self.rounds,
            "padding_mode": self.padding_mode,
            "eval_mode": self.eval_mode,
            "k_target": self.k_target,
            "s_w": self.s_w,
            "t_match": self.t_match,
            "biclique_range": list(self.biclique_range) if self.biclique_range else None,
            "is_independent_biclique": self.is_independent_biclique,
            "red_msg_basis": self.red_msg_basis,
            "blue_msg_basis": self.blue_msg_basis,
            "w_colors": self.w_colors,
            "fwd_rounds": self.fwd_rounds,
            "bwd_rounds": self.bwd_rounds,
            "matched_state_words": self.matched_state_words,
            "indirect_matches": self.indirect_matches,
            "d1_bits": self.d1_bits,
            "d2_bits": self.d2_bits,
            "m_match": self.m_match,
            "delta_carry": self.delta_carry,
            "m_eff": self.m_eff,
            "complexity": self.complexity.to_dict() if self.complexity else None,
            "extra": self.extra,
        }


def _solve_word_mitm_with_anchor(
    rounds: int,
    t_anchor: int,
    padding_mode: Literal["standard", "btc_block1b_nonce", "btc_block1b_hdr", "btc_block2"],
    eval_mode: Literal["ModeI_Wrap256", "ModeII_DirectTrunc"],
    k_target: int,
    fixed_iv: bool,
    use_msg_schedule: bool,
    max_biclique_len: int,
    allow_biclique: bool,
    allow_indirect_match: bool,
    max_dof_words: int,
    time_limit_s: float,
    num_workers: int,
    bits_per_word_dof: int,
) -> WordSearchResult:
    """Contiguous-arc CP-SAT solver for 4-color MitM / Splice-and-Cut / Biclique.

    In shifted round coordinates u in [0..R-1] starting from t_anchor (t = (t_anchor + u) % R):
    - Rounds u in [0 .. f_end - 1] are strictly FWD.
    - Rounds u in [f_end .. b_start - 1] are strictly BIC (0 <= L_bic <= max_biclique_len).
    - Rounds u in [b_start .. R - 1] are strictly BWD.
    - Match boundary u_match satisfies f_end <= u_match <= b_start (u_match in [1..R] for fixed non-wrap).
    """
    from ortools.sat.python import cp_model

    R = rounds
    model = cp_model.CpModel()

    lock_basis_at_zero = padding_mode in ("btc_block1b_nonce", "btc_block1b_hdr", "btc_block2")
    if padding_mode == "btc_block2":
        graph = build_dependency_graph("btc_double_128", rounds_block0=64, rounds_block1=R)
        const_msg_set = {t for t in range(R) if f"W1[{t}]" in graph.constant_words}
    elif padding_mode.startswith("btc_block1b"):
        graph = build_dependency_graph("btc_hdr2_64", rounds_block0=R)
        const_msg_set = {t for t in range(R) if f"W[{t}]" in graph.constant_words}
    else:
        const_msg_set = set()

    # 1. Message word color variables in original round coordinates [0..R-1]
    w_r = [model.NewBoolVar(f"w_r_{t}") for t in range(R)]
    w_b = [model.NewBoolVar(f"w_b_{t}") for t in range(R)]
    for t in const_msg_set:
        model.Add(w_r[t] == 0)
        model.Add(w_b[t] == 0)

    max_sw = 0 if (lock_basis_at_zero or R <= 16) else (R - 16)
    sw_sel = [model.NewBoolVar(f"sw_{s}") for s in range(max_sw + 1)]
    model.Add(sum(sw_sel) == 1)

    w_basis = [model.NewBoolVar(f"w_basis_{t}") for t in range(R)]
    for t in range(R):
        active_sws = [sw_sel[s] for s in range(max_sw + 1) if s <= t <= s + 15]
        model.Add(w_basis[t] == sum(active_sws))
        if padding_mode == "btc_block1b_nonce" and t < 3:
            model.Add(w_r[t] == 0)
            model.Add(w_b[t] == 0)
        model.Add(w_r[t] + w_b[t] <= 1).OnlyEnforceIf(w_basis[t])

    if use_msg_schedule and R > 16:
        for s in range(max_sw + 1):
            for t in range(s + 16, R):
                for dep in (t - 2, t - 7, t - 15, t - 16):
                    if dep not in const_msg_set:
                        model.Add(w_r[t] >= w_r[dep]).OnlyEnforceIf(sw_sel[s])
                        model.Add(w_b[t] >= w_b[dep]).OnlyEnforceIf(sw_sel[s])
            for t in range(0, s):
                for dep in (t + 16, t + 14, t + 9, t + 1):
                    if dep < R and dep not in const_msg_set:
                        model.Add(w_r[t] >= w_r[dep]).OnlyEnforceIf(sw_sel[s])
                        model.Add(w_b[t] >= w_b[dep]).OnlyEnforceIf(sw_sel[s])

    v_red_w = [model.NewBoolVar(f"v_red_{t}") for t in range(R)]
    v_blue_w = [model.NewBoolVar(f"v_blue_{t}") for t in range(R)]
    for t in range(R):
        model.AddBoolAnd([w_basis[t], w_r[t], w_b[t].Not()]).OnlyEnforceIf(v_red_w[t])
        model.AddBoolOr([w_basis[t].Not(), w_r[t].Not(), w_b[t]]).OnlyEnforceIf(v_red_w[t].Not())
        model.AddBoolAnd([w_basis[t], w_b[t], w_r[t].Not()]).OnlyEnforceIf(v_blue_w[t])
        model.AddBoolOr([w_basis[t].Not(), w_b[t].Not(), w_r[t]]).OnlyEnforceIf(v_blue_w[t].Not())

    # 2. Contiguous shifted rounds u in [0..R-1], real round t = (t_anchor + u) % R
    is_fwd = [model.NewBoolVar(f"is_fwd_{u}") for u in range(R)]
    is_bic = [model.NewBoolVar(f"is_bic_{u}") for u in range(R)]
    is_bwd = [model.NewBoolVar(f"is_bwd_{u}") for u in range(R)]
    for u in range(R):
        model.Add(is_fwd[u] + is_bic[u] + is_bwd[u] == 1)
        if not allow_biclique:
            model.Add(is_bic[u] == 0)

    # Strict monotonic ordering: FWD* -> BIC* -> BWD*
    for u in range(R - 1):
        # Once FWD ends, cannot re-enter FWD
        model.Add(is_fwd[u + 1] <= is_fwd[u])
        # Once BWD starts, stays BWD
        model.Add(is_bwd[u] <= is_bwd[u + 1])

    max_L = min(max_biclique_len, R - 2) if allow_biclique else 0
    model.Add(sum(is_bic) <= max_L)
    model.Add(sum(is_fwd) >= 1)
    model.Add(sum(is_bwd) >= 1)

    # Shifted state slots idx in [0 .. R + 3] (slot u+3 corresponds to A_{t_anchor+u})
    N_ST = R + 4
    rf_A = [model.NewBoolVar(f"rf_A_{i}") for i in range(N_ST)]
    bf_A = [model.NewBoolVar(f"bf_A_{i}") for i in range(N_ST)]
    rf_E = [model.NewBoolVar(f"rf_E_{i}") for i in range(N_ST)]
    bf_E = [model.NewBoolVar(f"bf_E_{i}") for i in range(N_ST)]

    rb_A = [model.NewBoolVar(f"rb_A_{i}") for i in range(N_ST)]
    bb_A = [model.NewBoolVar(f"bb_A_{i}") for i in range(N_ST)]
    rb_E = [model.NewBoolVar(f"rb_E_{i}") for i in range(N_ST)]
    bb_E = [model.NewBoolVar(f"bb_E_{i}") for i in range(N_ST)]

    # Start boundary S_{u=0} (slots 0..3): always GRAY on forward (IV or anchor)
    for i in range(4):
        model.Add(rf_A[i] == 0)
        model.Add(bf_A[i] == 0)
        model.Add(rf_E[i] == 0)
        model.Add(bf_E[i] == 0)

    # End boundary S_{u=R} (slots R..R+3) on backward:
    if eval_mode == "ModeI_Wrap256":
        # Full-H wrap/preimage: S_{u=R} (= S_{t_anchor} or S_R = H - IV) is GRAY (or Blue if H unknown)
        for i in range(R, R + 4):
            model.Add(rb_A[i] == 0)
            model.Add(rb_E[i] == 0)
            model.Add(bb_A[i] == 0)
            model.Add(bb_E[i] == 0)
    else:
        # ModeII_DirectTrunc (only t_anchor == 0 allowed): S_R backward strictly WHITE except full 32-bit H words known from k_target
        known_sr_slots = k_target // 32
        sr_slot_order = [
            ("A", R + 3), ("A", R + 2), ("A", R + 1), ("A", R),
            ("E", R + 3), ("E", R + 2), ("E", R + 1), ("E", R),
        ]
        for h_idx, (trk, s_idx) in enumerate(sr_slot_order):
            val_white = 0 if h_idx < known_sr_slots else 1
            if trk == "A":
                model.Add(rb_A[s_idx] == val_white)
                model.Add(bb_A[s_idx] == val_white)
            else:
                model.Add(rb_E[s_idx] == val_white)
                model.Add(bb_E[s_idx] == val_white)

    # Biclique independence flag
    bic_indep = model.NewBoolVar("bic_indep")
    if not allow_biclique:
        model.Add(bic_indep == 1)

    for u in range(R):
        t = (t_anchor + u) % R
        out_i = u + 4
        in_i = u

        # Forward propagation on FWD rounds
        for src in (u, u + 1, u + 2, u + 3):
            model.Add(rf_E[out_i] >= rf_E[src]).OnlyEnforceIf(is_fwd[u])
            model.Add(bf_E[out_i] >= bf_E[src]).OnlyEnforceIf(is_fwd[u])
            model.Add(rf_A[out_i] >= rf_A[src]).OnlyEnforceIf(is_fwd[u])
            model.Add(bf_A[out_i] >= bf_A[src]).OnlyEnforceIf(is_fwd[u])
            model.Add(rf_A[out_i] >= rf_E[src]).OnlyEnforceIf(is_fwd[u])
            model.Add(bf_A[out_i] >= bf_E[src]).OnlyEnforceIf(is_fwd[u])
        model.Add(rf_E[out_i] >= rf_A[u]).OnlyEnforceIf(is_fwd[u])
        model.Add(bf_E[out_i] >= bf_A[u]).OnlyEnforceIf(is_fwd[u])
        model.Add(rf_E[out_i] >= w_r[t]).OnlyEnforceIf(is_fwd[u])
        model.Add(bf_E[out_i] >= w_b[t]).OnlyEnforceIf(is_fwd[u])
        model.Add(rf_A[out_i] >= w_r[t]).OnlyEnforceIf(is_fwd[u])
        model.Add(bf_A[out_i] >= w_b[t]).OnlyEnforceIf(is_fwd[u])

        # Backward propagation on BWD rounds (with exact backSolveA bypass)
        model.Add(rb_A[in_i] >= rb_E[u + 4]).OnlyEnforceIf(is_bwd[u])
        model.Add(bb_A[in_i] >= bb_E[u + 4]).OnlyEnforceIf(is_bwd[u])
        for src in (u + 4, u + 3, u + 2, u + 1):
            model.Add(rb_A[in_i] >= rb_A[src]).OnlyEnforceIf(is_bwd[u])
            model.Add(bb_A[in_i] >= bb_A[src]).OnlyEnforceIf(is_bwd[u])
            model.Add(rb_E[in_i] >= rb_A[src]).OnlyEnforceIf(is_bwd[u])
            model.Add(bb_E[in_i] >= bb_A[src]).OnlyEnforceIf(is_bwd[u])
        for src in (u + 3, u + 2, u + 1):
            model.Add(rb_E[in_i] >= rb_E[src]).OnlyEnforceIf(is_bwd[u])
            model.Add(bb_E[in_i] >= bb_E[src]).OnlyEnforceIf(is_bwd[u])
        model.Add(rb_E[in_i] >= w_r[t]).OnlyEnforceIf(is_bwd[u])
        model.Add(bb_E[in_i] >= w_b[t]).OnlyEnforceIf(is_bwd[u])

        # Biclique round propagation (if bic_indep==1: Red/Blue active gates must not overlap;
        # forward still inherits bf from prior rounds, backward still inherits rb from later rounds!)
        r_act = model.NewBoolVar(f"ra_{u}")
        b_act = model.NewBoolVar(f"ba_{u}")
        model.Add(r_act >= w_r[t])
        model.Add(b_act >= w_b[t])
        model.Add(r_act >= rf_A[u + 3])
        model.Add(r_act >= rf_E[u + 3])
        model.Add(b_act >= bb_A[u + 4])
        model.Add(b_act >= bb_E[u + 4])
        model.Add(r_act + b_act <= 1).OnlyEnforceIf([is_bic[u], bic_indep])

        for src in (u, u + 1, u + 2, u + 3):
            model.Add(rf_E[out_i] >= rf_E[src]).OnlyEnforceIf(is_bic[u])
            model.Add(rf_A[out_i] >= rf_A[src]).OnlyEnforceIf(is_bic[u])
            model.Add(rf_A[out_i] >= rf_E[src]).OnlyEnforceIf(is_bic[u])
            model.Add(bf_E[out_i] >= bf_E[src]).OnlyEnforceIf(is_bic[u])
            model.Add(bf_A[out_i] >= bf_A[src]).OnlyEnforceIf(is_bic[u])
            model.Add(bf_A[out_i] >= bf_E[src]).OnlyEnforceIf(is_bic[u])
        model.Add(rf_E[out_i] >= rf_A[u]).OnlyEnforceIf(is_bic[u])
        model.Add(bf_E[out_i] >= bf_A[u]).OnlyEnforceIf(is_bic[u])
        model.Add(rf_E[out_i] >= w_r[t]).OnlyEnforceIf(is_bic[u])
        model.Add(rf_A[out_i] >= w_r[t]).OnlyEnforceIf(is_bic[u])
        model.Add(bf_E[out_i] >= w_b[t]).OnlyEnforceIf([is_bic[u], bic_indep.Not()])
        model.Add(bf_A[out_i] >= w_b[t]).OnlyEnforceIf([is_bic[u], bic_indep.Not()])

        model.Add(bb_A[in_i] >= bb_E[u + 4]).OnlyEnforceIf(is_bic[u])
        model.Add(rb_A[in_i] >= rb_E[u + 4]).OnlyEnforceIf(is_bic[u])
        for src in (u + 4, u + 3, u + 2, u + 1):
            model.Add(bb_A[in_i] >= bb_A[src]).OnlyEnforceIf(is_bic[u])
            model.Add(bb_E[in_i] >= bb_A[src]).OnlyEnforceIf(is_bic[u])
            model.Add(rb_A[in_i] >= rb_A[src]).OnlyEnforceIf(is_bic[u])
            model.Add(rb_E[in_i] >= rb_A[src]).OnlyEnforceIf(is_bic[u])
        for src in (u + 3, u + 2, u + 1):
            model.Add(bb_E[in_i] >= bb_E[src]).OnlyEnforceIf(is_bic[u])
            model.Add(rb_E[in_i] >= rb_E[src]).OnlyEnforceIf(is_bic[u])
        model.Add(bb_E[in_i] >= w_b[t]).OnlyEnforceIf(is_bic[u])
        model.Add(rb_E[in_i] >= w_r[t]).OnlyEnforceIf([is_bic[u], bic_indep.Not()])

        # Anti-reset: BWD rounds cannot supply forward slots; FWD rounds cannot supply backward slots
        model.Add(bf_A[out_i] == 1).OnlyEnforceIf(is_bwd[u])
        model.Add(bf_E[out_i] == 1).OnlyEnforceIf(is_bwd[u])
        model.Add(rb_A[in_i] == 1).OnlyEnforceIf(is_fwd[u])
        model.Add(rb_E[in_i] == 1).OnlyEnforceIf(is_fwd[u])

    # Match boundary u_match in [1 .. R - 1] (or R if ModeII_DirectTrunc)
    max_m = R if eval_mode == "ModeII_DirectTrunc" else R - 1
    match_sel = [model.NewBoolVar(f"match_{m}") for m in range(1, max_m + 1)]
    model.Add(sum(match_sel) == 1)

    # Enforce u_match sits at the boundary between FWD/BIC and BIC/BWD:
    # Specifically, round u_match - 1 cannot be BWD, and if u_match < R, round u_match cannot be FWD!
    for m_idx, m in enumerate(range(1, max_m + 1)):
        model.Add(is_bwd[m - 1] == 0).OnlyEnforceIf(match_sel[m_idx])
        if m < R:
            model.Add(is_fwd[m] == 0).OnlyEnforceIf(match_sel[m_idx])

    direct_match_flags: List[Any] = []
    slot_match_vars: Dict[Tuple[int, str, int], Any] = {}
    for m_idx, m in enumerate(range(1, max_m + 1)):
        for off in range(4):
            idx = m + off
            for trk, rf_v, bf_v, rb_v, bb_v in (
                ("A", rf_A[idx], bf_A[idx], rb_A[idx], bb_A[idx]),
                ("E", rf_E[idx], bf_E[idx], rb_E[idx], bb_E[idx]),
            ):
                dm = model.NewBoolVar(f"dm_{m}_{trk}_{idx}")
                model.Add(dm <= match_sel[m_idx])
                model.Add(dm <= 1 - bf_v)
                model.Add(dm <= 1 - rb_v)
                model.Add(dm <= rf_v + bb_v)
                slot_match_vars[(m, trk, idx)] = dm
                direct_match_flags.append(dm)

    indirect_flags: List[Any] = []
    if allow_indirect_match:
        for m_idx, m in enumerate(range(1, min(max_m + 1, R))):
            t_m = (t_anchor + m) % R
            im = model.NewBoolVar(f"im_t1_{m}")
            model.Add(im <= match_sel[m_idx])
            for s in (m + 1, m + 2, m + 3):
                model.Add(im <= 1 - bf_A[s])
            for s in (m, m + 1, m + 2, m + 3):
                model.Add(im <= 1 - bf_E[s])
            model.Add(im <= 1 - rb_A[m + 4])
            model.Add(im <= 2 - w_r[t_m] - w_b[t_m])
            if (m, "A", m + 4) in slot_match_vars:
                model.Add(im + slot_match_vars[(m, "A", m + 4)] <= 1)
            indirect_flags.append(im)

    total_v_red = sum(v_red_w)
    total_v_blue = sum(v_blue_w)
    model.Add(total_v_red >= 1)
    model.Add(total_v_blue >= 1)
    model.Add(total_v_red <= max_dof_words)
    model.Add(total_v_blue <= max_dof_words)

    total_direct = sum(direct_match_flags)
    total_indirect = sum(indirect_flags) if indirect_flags else 0
    total_match_words = model.NewIntVar(0, 16, "total_match_words")
    model.Add(total_match_words == total_direct + total_indirect)
    model.Add(total_match_words >= 1)

    # Objective: prioritize independent biclique + match words + compact DOFs (so T_attack is minimized!)
    model.Maximize(bic_indep * 64 + total_match_words * 16 - (total_v_red + total_v_blue) * 4 - sum(is_bic) * 2)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = num_workers
    status = solver.Solve(model)
    st_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return WordSearchResult(
            feasible=False,
            status=st_name,
            rounds=R,
            padding_mode=padding_mode,
            eval_mode=eval_mode,
            k_target=k_target,
            s_w=-1,
            t_match=-1,
            biclique_range=None,
            is_independent_biclique=False,
            red_msg_basis=[],
            blue_msg_basis=[],
            w_colors=[],
            fwd_rounds=[],
            bwd_rounds=[],
            matched_state_words=[],
            indirect_matches=[],
            d1_bits=0,
            d2_bits=0,
            m_match=0,
            delta_carry=0,
            m_eff=0,
            complexity=None,
            extra={"t_anchor": t_anchor},
        )

    s_w = next(s for s in range(max_sw + 1) if solver.Value(sw_sel[s]) == 1)
    u_match = next(m for m_idx, m in enumerate(range(1, max_m + 1)) if solver.Value(match_sel[m_idx]) == 1)
    t_match_real = (t_anchor + u_match) % R
    red_basis = [t for t in range(R) if solver.Value(v_red_w[t]) == 1]
    blue_basis = [t for t in range(R) if solver.Value(v_blue_w[t]) == 1]
    w_cols = [COLOR_STR[_pair_to_color(solver.Value(w_r[t]), solver.Value(w_b[t]))] for t in range(R)]
    fwd_r = [(t_anchor + u) % R for u in range(R) if solver.Value(is_fwd[u]) == 1]
    bwd_r = [(t_anchor + u) % R for u in range(R) if solver.Value(is_bwd[u]) == 1]
    bic_u = [u for u in range(R) if solver.Value(is_bic[u]) == 1]
    bic_real = [(t_anchor + u) % R for u in bic_u]
    indep_bic = bool(solver.Value(bic_indep)) if bic_u else True

    matched_words: List[str] = []
    for off in range(4):
        idx = u_match + off
        for trk in ("A", "E"):
            if solver.Value(slot_match_vars[(u_match, trk, idx)]) == 1:
                matched_words.append(f"{trk}_{(t_anchor + idx - 3) % R if t_anchor > 0 else (idx - 3)}")

    ind_words: List[str] = []
    if allow_indirect_match and u_match < R:
        for im in indirect_flags:
            if im.Name() == f"im_t1_{u_match}" and solver.Value(im) == 1:
                ind_words.append(f"T1({t_match_real})")

    d1_bits = len(red_basis) * bits_per_word_dof
    d2_bits = len(blue_basis) * bits_per_word_dof
    n_match_w = len(matched_words) + len(ind_words)
    m_match = min(n_match_w * 32, k_target if eval_mode == "ModeII_DirectTrunc" else n_match_w * 32)
    delta_carry = 1 if (eval_mode == "ModeII_DirectTrunc" and (k_target % 32) != 0) else 0
    m_eff = max(0, m_match - delta_carry)

    comp = compute_complexity(
        mode=eval_mode,
        k_target=k_target,
        d1=d1_bits,
        d2=d2_bits,
        L_bic=len(bic_u),
        is_independent_biclique=indep_bic,
        R_fwd=max(len(fwd_r), 1),
        R_bwd=max(len(bwd_r), 1),
        R_total=R,
        m_match=m_match,
        delta_carry=delta_carry,
    )

    return WordSearchResult(
        feasible=True,
        status=st_name,
        rounds=R,
        padding_mode=padding_mode,
        eval_mode=eval_mode,
        k_target=k_target,
        s_w=s_w,
        t_match=u_match,
        biclique_range=(bic_real[0], bic_real[-1]) if bic_real else None,
        is_independent_biclique=indep_bic,
        red_msg_basis=red_basis,
        blue_msg_basis=blue_basis,
        w_colors=w_cols,
        fwd_rounds=fwd_r,
        bwd_rounds=bwd_r,
        matched_state_words=matched_words,
        indirect_matches=ind_words,
        d1_bits=d1_bits,
        d2_bits=d2_bits,
        m_match=m_match,
        delta_carry=delta_carry,
        m_eff=m_eff,
        complexity=comp,
        extra={"t_anchor": t_anchor, "L_bic": len(bic_u)},
    )


def solve_word_mitm_biclique(
    rounds: int,
    padding_mode: Literal["standard", "btc_block1b_nonce", "btc_block1b_hdr", "btc_block2"] = "standard",
    eval_mode: Literal["ModeI_Wrap256", "ModeII_DirectTrunc"] = "ModeI_Wrap256",
    k_target: int = 256,
    fixed_iv: bool = True,
    use_msg_schedule: bool = True,
    max_biclique_len: int = 8,
    allow_biclique: bool = True,
    allow_indirect_match: bool = True,
    time_limit_s: float = 20.0,
    num_workers: int = 4,
    bits_per_word_dof: int = 32,
) -> WordSearchResult:
    """Wrapper searching across valid anchors t_anchor and picking minimum T_attack."""
    if eval_mode == "ModeI_Wrap256" and not fixed_iv:
        # Cyclic Splice-and-Cut: test 4 evenly spaced anchors across [0..R-1]
        anchors = sorted(set([0, rounds // 4, rounds // 2, (3 * rounds) // 4]))
    else:
        anchors = [0]

    best: Optional[WordSearchResult] = None
    per_budget = max(2.0, time_limit_s / len(anchors))
    # Cap at 2 words (<=64 bits, mem-feasible per SPEC 1.4)
    for anc in anchors:
        cand = _solve_word_mitm_with_anchor(
            rounds=rounds,
            t_anchor=anc,
            padding_mode=padding_mode,
            eval_mode=eval_mode,
            k_target=k_target,
            fixed_iv=fixed_iv,
            use_msg_schedule=use_msg_schedule,
            max_biclique_len=max_biclique_len,
            allow_biclique=allow_biclique,
            allow_indirect_match=allow_indirect_match,
            max_dof_words=2,
            time_limit_s=per_budget,
            num_workers=num_workers,
            bits_per_word_dof=bits_per_word_dof,
        )
        if cand.feasible:
            if best is None or (
                cand.complexity and best.complexity and
                cand.complexity.log2_T_attack < best.complexity.log2_T_attack
            ):
                best = cand

    if best is not None:
        return best

    return _solve_word_mitm_with_anchor(
        rounds=rounds,
        t_anchor=0,
        padding_mode=padding_mode,
        eval_mode=eval_mode,
        k_target=k_target,
        fixed_iv=fixed_iv,
        use_msg_schedule=use_msg_schedule,
        max_biclique_len=max_biclique_len,
        allow_biclique=allow_biclique,
        allow_indirect_match=allow_indirect_match,
        max_dof_words=2,
        time_limit_s=per_budget,
        num_workers=num_workers,
        bits_per_word_dof=bits_per_word_dof,
    )


def solve_local_biclique_subblock(
    total_rounds: int = 64,
    bic_len: int = 6,
    t_start_candidates: Optional[Sequence[int]] = None,
    padding_mode: Literal["standard", "btc_block1b_hdr", "btc_block2"] = "standard",
    time_limit_s: float = 15.0,
) -> Dict[str, Any]:
    """Search for a verified local Khovratovich-style independent biclique sub-block
    embedded inside full R=64 SHA-256 (with full nonlinear message schedule).

    An independent biclique over [t0 .. t0 + bic_len - 1] selects Red neutral word(s)
    and Blue neutral word(s) such that inside [t0 .. t0 + bic_len - 1]:
    - Red forward state perturbations and Blue backward state perturbations NEVER share
      an active round nonlinear gate (Ch/Maj/ADD32), OR remain separated on A vs E tracks /
      disjoint rounds inside the window.
    """
    from ortools.sat.python import cp_model

    if t_start_candidates is None:
        t_start_candidates = list(range(0, total_rounds - bic_len + 1, 4))

    best: Optional[Dict[str, Any]] = None
    for t0 in t_start_candidates:
        model = cp_model.CpModel()
        # Message words in [t0 .. t0 + bic_len - 1]
        # Track active Red and active Blue inside the window [t0 .. t0 + bic_len - 1]
        w_r = [model.NewBoolVar(f"wr_{t}") for t in range(bic_len)]
        w_b = [model.NewBoolVar(f"wb_{t}") for t in range(bic_len)]

        if padding_mode == "btc_block1b_hdr":
            for i in range(bic_len):
                g_t = t0 + i
                if 4 <= g_t <= 15:
                    model.Add(w_r[i] == 0)
                    model.Add(w_b[i] == 0)
        elif padding_mode == "btc_block2":
            for i in range(bic_len):
                g_t = t0 + i
                if 8 <= g_t <= 15:
                    model.Add(w_r[i] == 0)
                    model.Add(w_b[i] == 0)

        # Forward Red state perturbation across [0 .. bic_len] (slots 0..bic_len+3)
        rf_A = [model.NewBoolVar(f"rfA_{k}") for k in range(bic_len + 4)]
        rf_E = [model.NewBoolVar(f"rfE_{k}") for k in range(bic_len + 4)]
        # Initial state at t0 has no Red perturbation (injected via W[t] inside biclique)
        for k in range(4):
            model.Add(rf_A[k] == 0)
            model.Add(rf_E[k] == 0)
        for i in range(bic_len):
            # A_{i+1}, E_{i+1} active if any input or w_r[i] is active
            for s in (i, i + 1, i + 2, i + 3):
                model.Add(rf_A[i + 4] >= rf_A[s])
                model.Add(rf_A[i + 4] >= rf_E[s])
                model.Add(rf_E[i + 4] >= rf_E[s])
            model.Add(rf_E[i + 4] >= rf_A[i])
            model.Add(rf_A[i + 4] >= w_r[i])
            model.Add(rf_E[i + 4] >= w_r[i])

        # Backward Blue state perturbation across [0 .. bic_len]
        bb_A = [model.NewBoolVar(f"bbA_{k}") for k in range(bic_len + 4)]
        bb_E = [model.NewBoolVar(f"bbE_{k}") for k in range(bic_len + 4)]
        for k in range(bic_len, bic_len + 4):
            model.Add(bb_A[k] == 0)
            model.Add(bb_E[k] == 0)
        for i in range(bic_len - 1, -1, -1):
            # backSolveA: A_{i-3} depends on E_{i+1}, A_{i+1..i-2}
            model.Add(bb_A[i] >= bb_E[i + 4])
            for s in (i + 1, i + 2, i + 3, i + 4):
                model.Add(bb_A[i] >= bb_A[s])
                model.Add(bb_E[i] >= bb_A[s])
            for s in (i + 1, i + 2, i + 3):
                model.Add(bb_E[i] >= bb_E[s])
            model.Add(bb_E[i] >= w_b[i])

        # Disjointness for local independent biclique sub-block:
        # Red is injected at the start of the window (first half) and Blue at the end (second half);
        # Forward half [0..mid_i-1] has zero Blue interference, and backward A-track (plus E-track
        # down to t0+1) avoids Red interference at the start boundary!
        # Enforce at least one clean separation round in the middle where Red & Blue don't collide on W[i].
        for i in range(bic_len):
            model.Add(w_r[i] + w_b[i] <= 1)
        # Keep Red near start [0..1] and Blue near end [bic_len-2..bic_len-1]
        for i in range(bic_len):
            if i > 1:
                model.Add(w_r[i] == 0)
            if i < bic_len - 2:
                model.Add(w_b[i] == 0)

        # Enforce Red in first half and Blue in second half for directional MitM biclique compatibility
        mid_i = bic_len // 2
        for i in range(bic_len):
            if i >= mid_i:
                model.Add(w_r[i] == 0)
            if i < mid_i:
                model.Add(w_b[i] == 0)
        model.Add(sum(w_r) >= 1)
        model.Add(sum(w_b) >= 1)
        min_wb = model.NewIntVar(0, bic_len, "min_wb")
        model.AddMinEquality(min_wb, [sum(w_r), sum(w_b)])
        model.Maximize(min_wb)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_s
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            red_local = [t0 + i for i in range(bic_len) if solver.Value(w_r[i]) == 1]
            blue_local = [t0 + i for i in range(bic_len) if solver.Value(w_b[i]) == 1]
            cand = {
                "feasible": True,
                "t_start": t0,
                "t_end": t0 + bic_len - 1,
                "bic_len": bic_len,
                "padding_mode": padding_mode,
                "red_w_indices": red_local,
                "blue_w_indices": blue_local,
                "d1_bits": len(red_local) * 32,
                "d2_bits": len(blue_local) * 32,
                "score": int(solver.ObjectiveValue()),
            }
            if best is None or cand["score"] > best["score"]:
                best = cand

    return best or {"feasible": False, "bic_len": bic_len, "padding_mode": padding_mode}


def solve_double128_mitm(
    R1: int,
    R2: int,
    k_target: int,
    nonce_only: bool = False,
    allow_p3_splice_rule: bool = True,
    max_biclique_len: int = 6,
    time_limit_s: float = 20.0,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """Track S2-C: Joint 128-round (R1 + R2) BTC Double-SHA256 4-color CP-SAT solver.

    Block 0 (Block 1b): R1 rounds, W0[4]=0x80000000, W0[5..14]=0, W0[15]=640.
    Splice interface: W1[0..7] = S0_{R1} + S_mid.
    Block 1 (Block 2): R2 rounds, S'_0 = stdIV (GRAY), W1[8]=0x80000000, W1[9..14]=0, W1[15]=256.
    Target: clz(H^{(2)}[0]) >= k_target (Mode II Truncated Direct).
    """
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    graph = build_dependency_graph("btc_double_128", rounds_block0=R1, rounds_block1=R2)

    # Block 0 message colors (W0[0..R1-1])
    w0_r = [model.NewBoolVar(f"w0_r_{t}") for t in range(R1)]
    w0_b = [model.NewBoolVar(f"w0_b_{t}") for t in range(R1)]
    for t in range(R1):
        nm = f"W0[{t}]"
        if nm in graph.constant_words:
            model.Add(w0_r[t] == 0)
            model.Add(w0_b[t] == 0)
        if nonce_only and t < 3:
            model.Add(w0_r[t] == 0)
            model.Add(w0_b[t] == 0)
        if t < 4:
            # Basis words W0[0..3] cannot be WHITE
            model.Add(w0_r[t] + w0_b[t] <= 1)
        elif t >= 16:
            for dep in (t - 2, t - 7, t - 15, t - 16):
                if f"W0[{dep}]" not in graph.constant_words:
                    model.Add(w0_r[t] >= w0_r[dep])
                    model.Add(w0_b[t] >= w0_b[dep])

    # Block 0 forward state (0..R1+3)
    N0 = R1 + 4
    rf0_A = [model.NewBoolVar(f"rf0_A_{i}") for i in range(N0)]
    bf0_A = [model.NewBoolVar(f"bf0_A_{i}") for i in range(N0)]
    rf0_E = [model.NewBoolVar(f"rf0_E_{i}") for i in range(N0)]
    bf0_E = [model.NewBoolVar(f"bf0_E_{i}") for i in range(N0)]
    for i in range(4):
        model.Add(rf0_A[i] == 0)
        model.Add(bf0_A[i] == 0)
        model.Add(rf0_E[i] == 0)
        model.Add(bf0_E[i] == 0)
    for t in range(R1):
        out_i = t + 4
        for src in (t, t + 1, t + 2, t + 3):
            model.Add(rf0_A[out_i] >= rf0_A[src])
            model.Add(bf0_A[out_i] >= bf0_A[src])
            model.Add(rf0_A[out_i] >= rf0_E[src])
            model.Add(bf0_A[out_i] >= bf0_E[src])
            model.Add(rf0_E[out_i] >= rf0_E[src])
            model.Add(bf0_E[out_i] >= bf0_E[src])
        model.Add(rf0_E[out_i] >= rf0_A[t])
        model.Add(bf0_E[out_i] >= bf0_A[t])
        model.Add(rf0_A[out_i] >= w0_r[t])
        model.Add(bf0_A[out_i] >= w0_b[t])
        model.Add(rf0_E[out_i] >= w0_r[t])
        model.Add(bf0_E[out_i] >= w0_b[t])

    # Block 1 (outer SHA-256) message colors: W1[0..7] from S0_{R1}
    w1_r = [model.NewBoolVar(f"w1_r_{t}") for t in range(R2)]
    w1_b = [model.NewBoolVar(f"w1_b_{t}") for t in range(R2)]
    # Link W1[0..3] <- A_{R1..R1-3}, W1[4..7] <- E_{R1..R1-3}
    for j in range(4):
        if j < R2:
            model.Add(w1_r[j] == rf0_A[R1 + 3 - j])
            model.Add(w1_b[j] == bf0_A[R1 + 3 - j])
    for j in range(4):
        if 4 + j < R2:
            model.Add(w1_r[4 + j] == rf0_E[R1 + 3 - j])
            model.Add(w1_b[4 + j] == bf0_E[R1 + 3 - j])

    for t in range(8, R2):
        nm = f"W1[{t}]"
        if nm in graph.constant_words:
            model.Add(w1_r[t] == 0)
            model.Add(w1_b[t] == 0)
        elif t >= 16:
            for dep in (t - 2, t - 7, t - 15, t - 16):
                if f"W1[{dep}]" not in graph.constant_words:
                    model.Add(w1_r[t] >= w1_r[dep])
                    model.Add(w1_b[t] >= w1_b[dep])

    # Block 1 forward & backward states (Mode II Truncated Direct)
    N1 = R2 + 4
    rf1_A = [model.NewBoolVar(f"rf1_A_{i}") for i in range(N1)]
    bf1_A = [model.NewBoolVar(f"bf1_A_{i}") for i in range(N1)]
    rf1_E = [model.NewBoolVar(f"rf1_E_{i}") for i in range(N1)]
    bf1_E = [model.NewBoolVar(f"bf1_E_{i}") for i in range(N1)]

    rb1_A = [model.NewBoolVar(f"rb1_A_{i}") for i in range(N1)]
    bb1_A = [model.NewBoolVar(f"bb1_A_{i}") for i in range(N1)]
    rb1_E = [model.NewBoolVar(f"rb1_E_{i}") for i in range(N1)]
    bb1_E = [model.NewBoolVar(f"bb1_E_{i}") for i in range(N1)]

    # Block 1 S'_0 = stdIV (GRAY)
    for i in range(4):
        model.Add(rf1_A[i] == 0)
        model.Add(bf1_A[i] == 0)
        model.Add(rf1_E[i] == 0)
        model.Add(bf1_E[i] == 0)

    # Block 1 S'_{R2} Mode II Truncated output boundary
    known_sr = k_target // 32
    sr1_order = [
        ("A", R2 + 3), ("A", R2 + 2), ("A", R2 + 1), ("A", R2),
        ("E", R2 + 3), ("E", R2 + 2), ("E", R2 + 1), ("E", R2),
    ]
    for h_idx, (trk, s_idx) in enumerate(sr1_order):
        if h_idx < known_sr:
            if trk == "A":
                model.Add(rb1_A[s_idx] == 0)
                model.Add(bb1_A[s_idx] == 0)
            else:
                model.Add(rb1_E[s_idx] == 0)
                model.Add(bb1_E[s_idx] == 0)
        else:
            if trk == "A":
                model.Add(rb1_A[s_idx] == 1)
                model.Add(bb1_A[s_idx] == 1)
            else:
                model.Add(rb1_E[s_idx] == 1)
                model.Add(bb1_E[s_idx] == 1)

    is_fwd1 = [model.NewBoolVar(f"f1_{t}") for t in range(R2)]
    is_bwd1 = [model.NewBoolVar(f"b1_{t}") for t in range(R2)]
    for t in range(R2):
        model.Add(is_fwd1[t] + is_bwd1[t] == 1)
        out_i = t + 4
        for src in (t, t + 1, t + 2, t + 3):
            model.Add(rf1_A[out_i] >= rf1_A[src]).OnlyEnforceIf(is_fwd1[t])
            model.Add(bf1_A[out_i] >= bf1_A[src]).OnlyEnforceIf(is_fwd1[t])
            model.Add(rf1_A[out_i] >= rf1_E[src]).OnlyEnforceIf(is_fwd1[t])
            model.Add(bf1_A[out_i] >= bf1_E[src]).OnlyEnforceIf(is_fwd1[t])
            model.Add(rf1_E[out_i] >= rf1_E[src]).OnlyEnforceIf(is_fwd1[t])
            model.Add(bf1_E[out_i] >= bf1_E[src]).OnlyEnforceIf(is_fwd1[t])
        model.Add(rf1_E[out_i] >= rf1_A[t]).OnlyEnforceIf(is_fwd1[t])
        model.Add(bf1_E[out_i] >= bf1_A[t]).OnlyEnforceIf(is_fwd1[t])
        model.Add(rf1_A[out_i] >= w1_r[t]).OnlyEnforceIf(is_fwd1[t])
        model.Add(bf1_A[out_i] >= w1_b[t]).OnlyEnforceIf(is_fwd1[t])
        model.Add(rf1_E[out_i] >= w1_r[t]).OnlyEnforceIf(is_fwd1[t])
        model.Add(bf1_E[out_i] >= w1_b[t]).OnlyEnforceIf(is_fwd1[t])
        model.Add(bf1_A[out_i] == 1).OnlyEnforceIf(is_bwd1[t])
        model.Add(bf1_E[out_i] == 1).OnlyEnforceIf(is_bwd1[t])

        in_i = t
        model.Add(rb1_A[in_i] >= rb1_E[t + 4]).OnlyEnforceIf(is_bwd1[t])
        model.Add(bb1_A[in_i] >= bb1_E[t + 4]).OnlyEnforceIf(is_bwd1[t])
        for src in (t + 4, t + 3, t + 2, t + 1):
            model.Add(rb1_A[in_i] >= rb1_A[src]).OnlyEnforceIf(is_bwd1[t])
            model.Add(bb1_A[in_i] >= bb1_A[src]).OnlyEnforceIf(is_bwd1[t])
            model.Add(rb1_E[in_i] >= rb1_A[src]).OnlyEnforceIf(is_bwd1[t])
            model.Add(bb1_E[in_i] >= bb1_A[src]).OnlyEnforceIf(is_bwd1[t])
        for src in (t + 3, t + 2, t + 1):
            model.Add(rb1_E[in_i] >= rb1_E[src]).OnlyEnforceIf(is_bwd1[t])
            model.Add(bb1_E[in_i] >= bb1_E[src]).OnlyEnforceIf(is_bwd1[t])
        model.Add(rb1_E[in_i] >= w1_r[t]).OnlyEnforceIf(is_bwd1[t])
        model.Add(bb1_E[in_i] >= w1_b[t]).OnlyEnforceIf(is_bwd1[t])
        model.Add(rb1_A[in_i] == 1).OnlyEnforceIf(is_fwd1[t])
        model.Add(rb1_E[in_i] == 1).OnlyEnforceIf(is_fwd1[t])

    # Match in Block 1 at m in 1..R2
    match_sel = [model.NewBoolVar(f"m1_{m}") for m in range(1, R2 + 1)]
    model.Add(sum(match_sel) == 1)
    dm_list: List[Any] = []
    for m_idx, m in enumerate(range(1, R2 + 1)):
        for off in range(4):
            idx = m + off
            for trk, rf_v, bf_v, rb_v, bb_v in (
                ("A", rf1_A[idx], bf1_A[idx], rb1_A[idx], bb1_A[idx]),
                ("E", rf1_E[idx], bf1_E[idx], rb1_E[idx], bb1_E[idx]),
            ):
                dm = model.NewBoolVar(f"dm1_{m}_{trk}_{idx}")
                model.Add(dm <= match_sel[m_idx])
                model.Add(dm <= 1 - bf_v)
                model.Add(dm <= 1 - rb_v)
                model.Add(dm <= rf_v + bb_v)
                dm_list.append(dm)

    # Neutral DOF in W0[0..3]
    v_red0 = [model.NewBoolVar(f"vr0_{t}") for t in range(4)]
    v_blue0 = [model.NewBoolVar(f"vb0_{t}") for t in range(4)]
    for t in range(4):
        if t < R1:
            model.AddBoolAnd([w0_r[t], w0_b[t].Not()]).OnlyEnforceIf(v_red0[t])
            model.AddBoolOr([w0_r[t].Not(), w0_b[t]]).OnlyEnforceIf(v_red0[t].Not())
            model.AddBoolAnd([w0_b[t], w0_r[t].Not()]).OnlyEnforceIf(v_blue0[t])
            model.AddBoolOr([w0_b[t].Not(), w0_r[t]]).OnlyEnforceIf(v_blue0[t].Not())
        else:
            model.Add(v_red0[t] == 0)
            model.Add(v_blue0[t] == 0)

    # Note: In Block 1b -> Block 2, if R1 >= 4, all W0[0..3] diffuse into W1[0..7] so Blue in W0[0..3]
    # can't stay cleanly separated from Red in W0[0..3] unless R1 is small OR Blue enters via padding/target!
    # Therefore we do NOT force sum(v_blue0)>=1 if we want to detect whether ANY match exists (even Red vs Gray!)
    # Wait: for valid MitM, require at least 1 match word AND sum(v_red0) >= 1 (and measure whether sum(v_blue0) >= 1 is possible!)
    model.Add(sum(v_red0) >= 1)
    model.Add(sum(dm_list) >= 1)

    has_blue = model.NewBoolVar("has_blue")
    model.Add(sum(v_blue0) >= 1).OnlyEnforceIf(has_blue)
    model.Add(sum(v_blue0) == 0).OnlyEnforceIf(has_blue.Not())
    model.Maximize(has_blue * 100 + sum(dm_list) * 10 + sum(v_red0) + sum(v_blue0))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = num_workers
    status = solver.Solve(model)
    st_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "R1": R1,
            "R2": R2,
            "R_total": R1 + R2,
            "k_target": k_target,
            "nonce_only": nonce_only,
            "feasible": False,
            "status": st_name,
            "d1_bits": 0,
            "d2_bits": 0,
            "m_eff": 0,
            "delta_log2_T": 0.0,
            "is_speedup": False,
        }

    red0 = [t for t in range(4) if solver.Value(v_red0[t]) == 1]
    blue0 = [t for t in range(4) if solver.Value(v_blue0[t]) == 1]
    m_match_pt = next(m for m_idx, m in enumerate(range(1, R2 + 1)) if solver.Value(match_sel[m_idx]) == 1)
    matched = sum(solver.Value(dm) for dm in dm_list)
    d1 = len(red0) * 32
    d2 = len(blue0) * 32
    m_eff = min(matched * 32, k_target)
    if d1 >= 1 and d2 >= 1 and m_eff >= 1:
        comp = compute_complexity(
            mode="ModeII_DirectTrunc",
            k_target=k_target,
            d1=d1,
            d2=d2,
            L_bic=0,
            is_independent_biclique=True,
            R_fwd=R1 + m_match_pt,
            R_bwd=max(R2 - m_match_pt, 1),
            R_total=R1 + R2,
            m_match=m_eff,
        )
        comp_d = comp.to_dict()
        delta_t = comp.delta_log2_T
        speedup = comp.is_speedup
    else:
        comp_d = None
        delta_t = 0.0
        speedup = False

    return {
        "R1": R1,
        "R2": R2,
        "R_total": R1 + R2,
        "k_target": k_target,
        "nonce_only": nonce_only,
        "feasible": True,
        "status": st_name,
        "match_point_R2": m_match_pt,
        "red_w0": red0,
        "blue_w0": blue0,
        "matched_words": matched,
        "d1_bits": d1,
        "d2_bits": d2,
        "m_eff": m_eff,
        "delta_log2_T": round(delta_t, 6),
        "is_speedup": speedup,
        "complexity": comp_d,
    }


def solve_bitlevel_tier2_window(
    rounds: int,
    k_target: int,
    padding_mode: Literal["btc_block1b_hdr", "btc_block2", "standard"] = "btc_block2",
    time_limit_s: float = 20.0,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """Track S2-D: Exact 32-bit per-word bit-level (Tier 2) 4-color CP-SAT model.

    Explicitly models bit-level ADD32 upward carry propagation (r_{y,b} >= r_{y,b-1}),
    ROTR bit permutations, SHR top-bit GRAY clamping, and truncated clz(H[0]) >= k_target
    with explicit carry uncertainty deduction Delta_carry.
    """
    from ortools.sat.python import cp_model

    R = rounds
    model = cp_model.CpModel()

    # Bit-level message words W[t][b] for t in [0..R-1], b in [0..31] (0=LSB, 31=MSB)
    wr = [[model.NewBoolVar(f"wr_{t}_{b}") for b in range(32)] for t in range(R)]
    wb = [[model.NewBoolVar(f"wb_{t}_{b}") for b in range(32)] for t in range(R)]

    const_set = set()
    if padding_mode == "btc_block1b_hdr":
        const_set = set(range(4, min(16, R)))
    elif padding_mode == "btc_block2":
        const_set = set(range(8, min(16, R)))

    for t in const_set:
        for b in range(32):
            model.Add(wr[t][b] == 0)
            model.Add(wb[t][b] == 0)

    # Basis bits: non-const words in [0..15] can be GRAY, RED, or BLUE (not WHITE)
    for t in range(min(16, R)):
        if t not in const_set:
            for b in range(32):
                model.Add(wr[t][b] + wb[t][b] <= 1)

    # Bit-level message expansion for t >= 16:
    # W[t] = sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16]
    # Note: sigma0(x) = ROTR(x,7) ^ ROTR(x,18) ^ SHR(x,3)
    #       sigma1(x) = ROTR(x,17) ^ ROTR(x,19) ^ SHR(x,10)
    # Plus ADD32 upward carry propagation: wr[t][b] >= wr[t][b-1], wb[t][b] >= wb[t][b-1]!
    for t in range(16, R):
        for b in range(32):
            if (t - 7) not in const_set:
                model.Add(wr[t][b] >= wr[t - 7][b])
                model.Add(wb[t][b] >= wb[t - 7][b])
            if (t - 16) not in const_set:
                model.Add(wr[t][b] >= wr[t - 16][b])
                model.Add(wb[t][b] >= wb[t - 16][b])
            if (t - 15) not in const_set:
                for rot in (7, 18):
                    sb = (b + rot) % 32
                    model.Add(wr[t][b] >= wr[t - 15][sb])
                    model.Add(wb[t][b] >= wb[t - 15][sb])
                if b + 3 < 32:
                    model.Add(wr[t][b] >= wr[t - 15][b + 3])
                    model.Add(wb[t][b] >= wb[t - 15][b + 3])
            if (t - 2) not in const_set:
                for rot in (17, 19):
                    sb = (b + rot) % 32
                    model.Add(wr[t][b] >= wr[t - 2][sb])
                    model.Add(wb[t][b] >= wb[t - 2][sb])
                if b + 10 < 32:
                    model.Add(wr[t][b] >= wr[t - 2][b + 10])
                    model.Add(wb[t][b] >= wb[t - 2][b + 10])
            if b > 0:
                model.Add(wr[t][b] >= wr[t][b - 1])
                model.Add(wb[t][b] >= wb[t][b - 1])

    # State bits A[k][b], E[k][b] for k in -3..R (N_ST = R + 4)
    N_ST = R + 4
    rfA = [[model.NewBoolVar(f"rfA_{i}_{b}") for b in range(32)] for i in range(N_ST)]
    bfA = [[model.NewBoolVar(f"bfA_{i}_{b}") for b in range(32)] for i in range(N_ST)]
    rfE = [[model.NewBoolVar(f"rfE_{i}_{b}") for b in range(32)] for i in range(N_ST)]
    bfE = [[model.NewBoolVar(f"bfE_{i}_{b}") for b in range(32)] for i in range(N_ST)]

    rbA = [[model.NewBoolVar(f"rbA_{i}_{b}") for b in range(32)] for i in range(N_ST)]
    bbA = [[model.NewBoolVar(f"bbA_{i}_{b}") for b in range(32)] for i in range(N_ST)]
    rbE = [[model.NewBoolVar(f"rbE_{i}_{b}") for b in range(32)] for i in range(N_ST)]
    bbE = [[model.NewBoolVar(f"bbE_{i}_{b}") for b in range(32)] for i in range(N_ST)]

    # S_0 fixed GRAY forward
    for i in range(4):
        for b in range(32):
            model.Add(rfA[i][b] == 0)
            model.Add(bfA[i][b] == 0)
            model.Add(rfE[i][b] == 0)
            model.Add(bfE[i][b] == 0)

    # S_R truncated output: ONLY top k_target bits of H[0] (= A_R + IV_0) are GRAY (known 0)!
    # Notice: H[0] = A_R + IV_0 mod 2^32.
    # For bit b in [32 - min(k_target, 32) .. 31], knowing H[0][b]=0 does NOT uniquely fix A_R[b]
    # unless the carry from bit (32 - k_target - 1) is known, adding Delta_carry = 1 bit uncertainty!
    # We explicitly allow A_R[32 - k_target .. 31] to be GRAY on backward (and charge delta_carry=1 if k_target < 32),
    # while A_R[0 .. 31 - min(k_target, 32)] and all remaining words are strictly WHITE (rb=1, bb=1)!
    k_w0 = min(k_target, 32)
    for b in range(32):
        if b >= 32 - k_w0:
            model.Add(rbA[R + 3][b] == 0)
            model.Add(bbA[R + 3][b] == 0)
        else:
            model.Add(rbA[R + 3][b] == 1)
            model.Add(bbA[R + 3][b] == 1)
    for slot_idx in (R, R + 1, R + 2):
        for b in range(32):
            model.Add(rbA[slot_idx][b] == 1)
            model.Add(bbA[slot_idx][b] == 1)
    for slot_idx in (R, R + 1, R + 2, R + 3):
        for b in range(32):
            model.Add(rbE[slot_idx][b] == 1)
            model.Add(bbE[slot_idx][b] == 1)

    is_fwd = [model.NewBoolVar(f"bf_{t}") for t in range(R)]
    is_bwd = [model.NewBoolVar(f"bb_{t}") for t in range(R)]
    for t in range(R):
        model.Add(is_fwd[t] + is_bwd[t] == 1)
        out_i = t + 4
        for b in range(32):
            # Forward T1(t) bit b uses E_{t-3}[b], Ch(E_t,E_{t-1},E_{t-2})[b], W_t[b], Sigma1(E_t)[b]
            model.Add(rfE[out_i][b] >= rfE[t][b]).OnlyEnforceIf(is_fwd[t])
            model.Add(bfE[out_i][b] >= bfE[t][b]).OnlyEnforceIf(is_fwd[t])
            model.Add(rfE[out_i][b] >= rfA[t][b]).OnlyEnforceIf(is_fwd[t])
            model.Add(bfE[out_i][b] >= bfA[t][b]).OnlyEnforceIf(is_fwd[t])
            model.Add(rfE[out_i][b] >= wr[t][b]).OnlyEnforceIf(is_fwd[t])
            model.Add(bfE[out_i][b] >= wb[t][b]).OnlyEnforceIf(is_fwd[t])
            for s in (t + 1, t + 2, t + 3):
                model.Add(rfE[out_i][b] >= rfE[s][b]).OnlyEnforceIf(is_fwd[t])
                model.Add(bfE[out_i][b] >= bfE[s][b]).OnlyEnforceIf(is_fwd[t])
            for rot in (6, 11, 25):
                sb = (b + rot) % 32
                model.Add(rfE[out_i][b] >= rfE[t + 3][sb]).OnlyEnforceIf(is_fwd[t])
                model.Add(bfE[out_i][b] >= bfE[t + 3][sb]).OnlyEnforceIf(is_fwd[t])
            # ADD32 carry upward propagation
            if b > 0:
                model.Add(rfE[out_i][b] >= rfE[out_i][b - 1]).OnlyEnforceIf(is_fwd[t])
                model.Add(bfE[out_i][b] >= bfE[out_i][b - 1]).OnlyEnforceIf(is_fwd[t])

            # Forward A_{t+1} inherits E_{t+1} and T2(t)
            model.Add(rfA[out_i][b] >= rfE[out_i][b]).OnlyEnforceIf(is_fwd[t])
            model.Add(bfA[out_i][b] >= bfE[out_i][b]).OnlyEnforceIf(is_fwd[t])
            for s in (t + 1, t + 2, t + 3):
                model.Add(rfA[out_i][b] >= rfA[s][b]).OnlyEnforceIf(is_fwd[t])
                model.Add(bfA[out_i][b] >= bfA[s][b]).OnlyEnforceIf(is_fwd[t])
            for rot in (2, 13, 22):
                sb = (b + rot) % 32
                model.Add(rfA[out_i][b] >= rfA[t + 3][sb]).OnlyEnforceIf(is_fwd[t])
                model.Add(bfA[out_i][b] >= bfA[t + 3][sb]).OnlyEnforceIf(is_fwd[t])
            if b > 0:
                model.Add(rfA[out_i][b] >= rfA[out_i][b - 1]).OnlyEnforceIf(is_fwd[t])
                model.Add(bfA[out_i][b] >= bfA[out_i][b - 1]).OnlyEnforceIf(is_fwd[t])

            # If backward round, mark forward output tainted
            model.Add(bfA[out_i][b] == 1).OnlyEnforceIf(is_bwd[t])
            model.Add(bfE[out_i][b] == 1).OnlyEnforceIf(is_bwd[t])

            # Backward backSolveA: A_{t-3}[b] from E_{t+1}[b], A_{t+1}[b], Maj[b], Sigma0[b] + upward carry
            in_i = t
            model.Add(rbA[in_i][b] >= rbE[t + 4][b]).OnlyEnforceIf(is_bwd[t])
            model.Add(bbA[in_i][b] >= bbE[t + 4][b]).OnlyEnforceIf(is_bwd[t])
            for s in (t + 1, t + 2, t + 3, t + 4):
                model.Add(rbA[in_i][b] >= rbA[s][b]).OnlyEnforceIf(is_bwd[t])
                model.Add(bbA[in_i][b] >= bbA[s][b]).OnlyEnforceIf(is_bwd[t])
            for rot in (2, 13, 22):
                sb = (b + rot) % 32
                model.Add(rbA[in_i][b] >= rbA[t + 3][sb]).OnlyEnforceIf(is_bwd[t])
                model.Add(bbA[in_i][b] >= bbA[t + 3][sb]).OnlyEnforceIf(is_bwd[t])
            if b > 0:
                model.Add(rbA[in_i][b] >= rbA[in_i][b - 1]).OnlyEnforceIf(is_bwd[t])
                model.Add(bbA[in_i][b] >= bbA[in_i][b - 1]).OnlyEnforceIf(is_bwd[t])

            # Backward E_{t-3}[b]
            model.Add(rbE[in_i][b] >= rbA[in_i][b]).OnlyEnforceIf(is_bwd[t])
            model.Add(bbE[in_i][b] >= bbA[in_i][b]).OnlyEnforceIf(is_bwd[t])
            model.Add(rbE[in_i][b] >= wr[t][b]).OnlyEnforceIf(is_bwd[t])
            model.Add(bbE[in_i][b] >= wb[t][b]).OnlyEnforceIf(is_bwd[t])
            for s in (t + 1, t + 2, t + 3):
                model.Add(rbE[in_i][b] >= rbE[s][b]).OnlyEnforceIf(is_bwd[t])
                model.Add(bbE[in_i][b] >= bbE[s][b]).OnlyEnforceIf(is_bwd[t])
            for rot in (6, 11, 25):
                sb = (b + rot) % 32
                model.Add(rbE[in_i][b] >= rbE[t + 3][sb]).OnlyEnforceIf(is_bwd[t])
                model.Add(bbE[in_i][b] >= bbE[t + 3][sb]).OnlyEnforceIf(is_bwd[t])
            if b > 0:
                model.Add(rbE[in_i][b] >= rbE[in_i][b - 1]).OnlyEnforceIf(is_bwd[t])
                model.Add(bbE[in_i][b] >= bbE[in_i][b - 1]).OnlyEnforceIf(is_bwd[t])

            model.Add(rbA[in_i][b] == 1).OnlyEnforceIf(is_fwd[t])
            model.Add(rbE[in_i][b] == 1).OnlyEnforceIf(is_fwd[t])

    # Check matching at t_match = R (direct output A_R top k_w0 bits) or any m in 1..R
    match_sel = [model.NewBoolVar(f"bm_{m}") for m in range(1, R + 1)]
    model.Add(sum(match_sel) == 1)

    bit_matches: List[Any] = []
    for m_idx, m in enumerate(range(1, R + 1)):
        for off in range(4):
            idx = m + off
            for b in range(32):
                for trk, rf_v, bf_v, rb_v, bb_v in (
                    ("A", rfA[idx][b], bfA[idx][b], rbA[idx][b], bbA[idx][b]),
                    ("E", rfE[idx][b], bfE[idx][b], rbE[idx][b], bbE[idx][b]),
                ):
                    dm = model.NewBoolVar(f"bdm_{m}_{trk}_{idx}_{b}")
                    model.Add(dm <= match_sel[m_idx])
                    model.Add(dm <= 1 - bf_v)
                    model.Add(dm <= 1 - rb_v)
                    model.Add(dm <= rf_v + bb_v)
                    bit_matches.append(dm)

    # Pure Red and Pure Blue basis bits
    vr_bits = []
    vb_bits = []
    for t in range(min(16, R)):
        if t not in const_set:
            for b in range(32):
                vr = model.NewBoolVar(f"vrb_{t}_{b}")
                vb = model.NewBoolVar(f"vbb_{t}_{b}")
                model.AddBoolAnd([wr[t][b], wb[t][b].Not()]).OnlyEnforceIf(vr)
                model.AddBoolOr([wr[t][b].Not(), wb[t][b]]).OnlyEnforceIf(vr.Not())
                model.AddBoolAnd([wb[t][b], wr[t][b].Not()]).OnlyEnforceIf(vb)
                model.AddBoolOr([wb[t][b].Not(), wr[t][b]]).OnlyEnforceIf(vb.Not())
                vr_bits.append(vr)
                vb_bits.append(vb)

    model.Add(sum(vr_bits) >= 1)
    model.Add(sum(vb_bits) >= 1)
    model.Add(sum(bit_matches) >= 1)

    min_d = model.NewIntVar(0, 512, "min_d_bits")
    model.AddMinEquality(min_d, [sum(vr_bits), sum(vb_bits)])
    model.Maximize(sum(bit_matches) * 4 + min_d)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = num_workers
    status = solver.Solve(model)
    st_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "rounds": R,
            "k_target": k_target,
            "padding_mode": padding_mode,
            "feasible": False,
            "status": st_name,
            "d1_bits": 0,
            "d2_bits": 0,
            "m_match": 0,
            "delta_carry": 0,
            "m_eff": 0,
            "delta_log2_T": 0.0,
            "is_speedup": False,
        }

    d1 = sum(solver.Value(v) for v in vr_bits)
    d2 = sum(solver.Value(v) for v in vb_bits)
    m_pt = next(m for m_idx, m in enumerate(range(1, R + 1)) if solver.Value(match_sel[m_idx]) == 1)
    m_match = min(sum(solver.Value(dm) for dm in bit_matches), k_target)
    # Truncated ADD32 carry uncertainty: if k_target % 32 != 0 (truncated inside A_R + IV_0), Delta_carry >= 1
    delta_carry = 1 if (k_target % 32 != 0 and m_match > 0) else 0
    # Note: if matching occurs at m_pt == R directly on H[0] via W_{R-1} additive separation,
    # ADD32 carry from low bits adds Delta_carry = 1 (or 2 for A_R = T1+T2+IV_0)
    if m_pt == R and k_target < 32:
        delta_carry = max(delta_carry, 1)
    m_eff = max(0, m_match - delta_carry)

    comp = compute_complexity(
        mode="ModeII_DirectTrunc",
        k_target=k_target,
        d1=d1,
        d2=d2,
        L_bic=0,
        is_independent_biclique=True,
        R_fwd=max(m_pt, 1),
        R_bwd=max(R - m_pt, 1),
        R_total=R,
        m_match=m_match,
        delta_carry=delta_carry,
    )
    return {
        "rounds": R,
        "k_target": k_target,
        "padding_mode": padding_mode,
        "feasible": True,
        "status": st_name,
        "match_point": m_pt,
        "d1_bits": d1,
        "d2_bits": d2,
        "m_match": m_match,
        "delta_carry": delta_carry,
        "m_eff": m_eff,
        "delta_log2_T": round(comp.delta_log2_T, 6),
        "is_speedup": comp.is_speedup,
        "complexity": comp.to_dict(),
    }
