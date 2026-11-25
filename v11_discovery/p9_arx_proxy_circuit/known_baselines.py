"""Known baseline library B_known (B1, B2, B3, B4) for P9 anti-triviality checks.

Implements SPEC Section 4.1:
- B1: Add_w -> XOR (carry-0 linear approximation), including IV constant folding for A2
- B2: Single-gate 1-bit & 2-bit local carry truncations
- B3: Known Ch/Maj linear and 1-AND approximations
- B4: Greedy multi-round composition of B1/B2/B3 on the exact operator DAG within
      budget (M_AND <= 16, deg <= 4, S_max <= 32, M_XOR <= 128), plus full-support
      reference linear masks if exact affine expansion exceeds S_max=32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .arx_operators import ARXParams, OperatorSpec, parse_target_name
from .circuit_slp import (
    MAX_AND_GATES,
    MAX_DEGREE,
    MAX_SUPPORT,
    MAX_XOR_GATES,
    AffineForm,
    AndGate,
    CircuitSLP,
    count_packed_matches,
)


@dataclass
class BitExpr:
    """Symbolic expression node over GF(2) for compiling known baselines with constant folding."""

    kind: str  # "CONST0", "CONST1", "VAR", "XOR", "AND"
    var_idx: int = -1
    children: Tuple["BitExpr", ...] = ()
    deg_bound: int = 0


_C0 = BitExpr("CONST0", deg_bound=0)
_C1 = BitExpr("CONST1", deg_bound=0)


class ExprBuilder:
    """Builds constant-folded XOR/AND expression DAGs and lowers them to CircuitSLP."""

    def __init__(self, n_inputs: int) -> None:
        self.n_inputs = n_inputs
        self.vars = [BitExpr("VAR", var_idx=i, deg_bound=1) for i in range(n_inputs)]

    def const_word(self, val: int, w: int) -> List[BitExpr]:
        return [_C1 if ((val >> b) & 1) else _C0 for b in range(w)]

    def var_word(self, start_idx: int, w: int) -> List[BitExpr]:
        return [self.vars[start_idx + b] for b in range(w)]

    def xor2(self, a: BitExpr, b: BitExpr) -> BitExpr:
        if a.kind == "CONST0":
            return b
        if b.kind == "CONST0":
            return a
        if a.kind == "CONST1" and b.kind == "CONST1":
            return _C0
        if a.kind == "VAR" and b.kind == "VAR" and a.var_idx == b.var_idx:
            return _C0
        return BitExpr("XOR", children=(a, b), deg_bound=max(a.deg_bound, b.deg_bound))

    def xor_many(self, items: Sequence[BitExpr]) -> BitExpr:
        # Keep XOR DAG compact in O(1) per call; compile_expr_to_slp cancels parities in O(|V|).
        const = 0
        var_counts: Dict[int, int] = {}
        others: List[BitExpr] = []
        for node in items:
            if node.kind == "CONST0":
                continue
            if node.kind == "CONST1":
                const ^= 1
            elif node.kind == "VAR":
                var_counts[node.var_idx] = var_counts.get(node.var_idx, 0) ^ 1
            else:
                others.append(node)
        terms: List[BitExpr] = []
        if const:
            terms.append(_C1)
        for v_idx in sorted(k for k, v in var_counts.items() if v == 1):
            terms.append(self.vars[v_idx])
        terms.extend(others)
        if not terms:
            return _C0
        if len(terms) == 1:
            return terms[0]
        return BitExpr("XOR", children=tuple(terms), deg_bound=max(t.deg_bound for t in terms))

    def and2(self, a: BitExpr, b: BitExpr) -> BitExpr:
        if a.kind == "CONST0" or b.kind == "CONST0":
            return _C0
        if a.kind == "CONST1":
            return b
        if b.kind == "CONST1":
            return a
        return BitExpr("AND", children=(a, b), deg_bound=a.deg_bound + b.deg_bound)

    def rotr_w(self, x: Sequence[BitExpr], n: int) -> List[BitExpr]:
        w = len(x)
        n = n % w
        if n == 0:
            return list(x)
        return [x[(b + n) % w] for b in range(w)]

    def shr_w(self, x: Sequence[BitExpr], n: int) -> List[BitExpr]:
        w = len(x)
        return [x[b + n] if (b + n < w) else _C0 for b in range(w)]

    def xor_words(self, x: Sequence[BitExpr], y: Sequence[BitExpr]) -> List[BitExpr]:
        return [self.xor2(a, b) for a, b in zip(x, y)]

    def ch_approx(
        self, e: Sequence[BitExpr], f: Sequence[BitExpr], g: Sequence[BitExpr], mode: str
    ) -> List[BitExpr]:
        if mode == "g":
            return list(g)
        if mode == "f":
            return list(f)
        if mode == "e_xor_g":
            return [self.xor2(eb, gb) for eb, gb in zip(e, g)]
        if mode == "exact":
            return [self.xor2(gb, self.and2(eb, self.xor2(fb, gb))) for eb, fb, gb in zip(e, f, g)]
        raise ValueError(f"Unknown ch_mode={mode}")

    def maj_approx(
        self, a: Sequence[BitExpr], b: Sequence[BitExpr], c: Sequence[BitExpr], mode: str
    ) -> List[BitExpr]:
        if mode == "a":
            return list(a)
        if mode == "b":
            return list(b)
        if mode == "c":
            return list(c)
        if mode == "xor3":
            return [self.xor_many((ab, bb, cb)) for ab, bb, cb in zip(a, b, c)]
        if mode == "exact":
            return [
                self.xor2(cb, self.and2(self.xor2(ab, cb), self.xor2(bb, cb)))
                for ab, bb, cb in zip(a, b, c)
            ]
        raise ValueError(f"Unknown maj_mode={mode}")

    def add_approx(
        self, x: Sequence[BitExpr], y: Sequence[BitExpr], carry_depth: int
    ) -> List[BitExpr]:
        """Add two w-bit words with bounded carry lookback depth k in {0, 1, 2}."""
        w = len(x)
        if carry_depth <= 0:
            return [self.xor2(xb, yb) for xb, yb in zip(x, y)]
        out: List[BitExpr] = []
        for b in range(w):
            # Local carry into bit b using up to `carry_depth` previous positions
            start = max(0, b - carry_depth)
            cin = _C0
            for pos in range(start, b):
                g_pos = self.and2(x[pos], y[pos])
                if cin.kind == "CONST0":
                    cin = g_pos
                else:
                    p_pos = self.xor2(x[pos], y[pos])
                    cin = self.xor2(g_pos, self.and2(p_pos, cin))
            out.append(self.xor_many((x[b], y[b], cin)))
        return out


def compile_expr_to_slp(
    expr: BitExpr,
    n_inputs: int,
    name: str,
    max_and: int = MAX_AND_GATES,
    max_xor: int = MAX_XOR_GATES,
    max_support: int = MAX_SUPPORT,
    max_degree: int = MAX_DEGREE,
) -> Tuple[Optional[CircuitSLP], Optional[AffineForm]]:
    """Compile a BitExpr into CircuitSLP in O(|V|) DAG time using memoized Python int bitmasks."""
    gates: List[AndGate] = []
    and_memo: Dict[int, int] = {}
    node_cache: Dict[int, Tuple[int, int]] = {}  # id(BitExpr) -> (const_bit, term_bitmask)

    def _mask_to_terms(mask: int) -> Tuple[int, ...]:
        terms: List[int] = []
        m = mask
        while m:
            lsb = m & -m
            terms.append(lsb.bit_length() - 1)
            m ^= lsb
        return tuple(terms)

    def lower_dag(node: BitExpr) -> Optional[Tuple[int, int]]:
        nid = id(node)
        cached = node_cache.get(nid)
        if cached is not None:
            return cached

        if node.kind == "CONST0":
            res = (0, 0)
        elif node.kind == "CONST1":
            res = (1, 0)
        elif node.kind == "VAR":
            res = (0, 1 << node.var_idx)
        elif node.kind == "XOR":
            c_acc = 0
            m_acc = 0
            for ch in node.children:
                ch_res = lower_dag(ch)
                if ch_res is None:
                    return None
                c_acc ^= ch_res[0]
                m_acc ^= ch_res[1]
            res = (c_acc, m_acc)
        elif node.kind == "AND":
            if nid not in and_memo:
                if len(gates) >= max_and:
                    return None
                l_res = lower_dag(node.children[0])
                if l_res is None:
                    return None
                r_res = lower_dag(node.children[1])
                if r_res is None:
                    return None
                lc, lm = l_res
                rc, rm = r_res
                if lm.bit_count() > max_support or rm.bit_count() > max_support:
                    return None
                laf = AffineForm(lc, _mask_to_terms(lm))
                raf = AffineForm(rc, _mask_to_terms(rm))
                gate_node_idx = n_inputs + len(gates)
                gates.append(AndGate(laf, raf))
                and_memo[nid] = gate_node_idx
            res = (0, 1 << and_memo[nid])
        else:
            raise ValueError(f"Unknown node kind {node.kind}")

        node_cache[nid] = res
        return res

    res = lower_dag(expr)
    if res is None:
        return None, None
    c_out, m_out = res
    t_out = _mask_to_terms(m_out)
    full_affine = AffineForm(c_out, t_out) if len(gates) == 0 else None
    if len(t_out) > max_support:
        return None, full_affine
    slp = CircuitSLP(
        n_inputs=n_inputs,
        gates=gates,
        output=AffineForm(c_out, t_out),
        name=name,
        method="known_baseline",
    ).prune_dead_gates()
    ok, _, _ = slp.validate(
        max_and=max_and,
        max_xor=max_xor,
        max_support=max_support,
        max_degree=max_degree,
    )
    if not ok:
        return None, full_affine
    return slp, full_affine


def _build_operator_expr(
    spec: OperatorSpec,
    target_name: str,
    ch_mode: str,
    maj_mode: str,
    add_carry_policy: Sequence[int],  # carry depth per round or step
) -> BitExpr:
    """Build symbolic BitExpr for (spec, target_name) under a given known-approximation policy."""
    eb = ExprBuilder(spec.n_inputs)
    w = spec.word_bits
    params = spec.params
    word_name, bit_idx = parse_target_name(target_name)

    def run_comp(
        s0_words: List[List[BitExpr]],
        w_sched: List[List[BitExpr]],
        k_offset: int,
        rounds: int,
        carry_per_round: Sequence[int],
    ) -> Tuple[List[BitExpr], List[BitExpr]]:
        a_cols = [s0_words[3], s0_words[2], s0_words[1], s0_words[0]]
        e_cols = [s0_words[7], s0_words[6], s0_words[5], s0_words[4]]
        for t in range(rounds):
            cd = carry_per_round[t] if t < len(carry_per_round) else 0
            a_t, a_m1, a_m2, a_m3 = a_cols[t + 3], a_cols[t + 2], a_cols[t + 1], a_cols[t]
            e_t, e_m1, e_m2, e_m3 = e_cols[t + 3], e_cols[t + 2], e_cols[t + 1], e_cols[t]

            if spec.family == "A2" and t == 0 and k_offset == 0:
                # Fold all constant terms at t=0 of IV-anchored A2 into single constants before adding W_0!
                mask_w = (1 << w) - 1
                iv = params.iv_consts
                e0_c, e1_c, e2_c, e3_c = iv[4], iv[5], iv[6], iv[7]
                a0_c, a1_c, a2_c, a3_c = iv[0], iv[1], iv[2], iv[3]

                def rotr_c(v: int, r_amt: int) -> int:
                    r_amt %= w
                    return v & mask_w if r_amt == 0 else ((v >> r_amt) | ((v & ((1 << r_amt) - 1)) << (w - r_amt))) & mask_w

                s1_c = rotr_c(e0_c, params.sigma1_upper_rot[0]) ^ rotr_c(e0_c, params.sigma1_upper_rot[1]) ^ rotr_c(e0_c, params.sigma1_upper_rot[2])
                ch_c = ((e0_c & e1_c) ^ ((~e0_c & mask_w) & e2_c)) & mask_w
                t1_const = (e3_c + s1_c + ch_c + params.k_consts[0]) & mask_w
                s0_c = rotr_c(a0_c, params.sigma0_upper_rot[0]) ^ rotr_c(a0_c, params.sigma0_upper_rot[1]) ^ rotr_c(a0_c, params.sigma0_upper_rot[2])
                mj_c = ((a0_c & a1_c) ^ (a0_c & a2_c) ^ (a1_c & a2_c)) & mask_w
                t2_const = (s0_c + mj_c) & mask_w

                c_e1 = (a3_c + t1_const) & mask_w
                c_a1 = (t1_const + t2_const) & mask_w
                e_tp1 = eb.add_approx(eb.const_word(c_e1, w), w_sched[0], cd)
                a_tp1 = eb.add_approx(eb.const_word(c_a1, w), w_sched[0], cd)
                e_cols.append(e_tp1)
                a_cols.append(a_tp1)
                continue

            s1 = [
                eb.xor_many((r0, r1, r2))
                for r0, r1, r2 in zip(
                    eb.rotr_w(e_t, params.sigma1_upper_rot[0]),
                    eb.rotr_w(e_t, params.sigma1_upper_rot[1]),
                    eb.rotr_w(e_t, params.sigma1_upper_rot[2]),
                )
            ]
            ch = eb.ch_approx(e_t, e_m1, e_m2, ch_mode)
            k_word = eb.const_word(params.k_consts[k_offset + t], w)

            t1 = eb.add_approx(e_m3, s1, cd)
            t1 = eb.add_approx(t1, ch, cd)
            t1 = eb.add_approx(t1, k_word, cd)
            t1 = eb.add_approx(t1, w_sched[t], cd)

            s0 = [
                eb.xor_many((r0, r1, r2))
                for r0, r1, r2 in zip(
                    eb.rotr_w(a_t, params.sigma0_upper_rot[0]),
                    eb.rotr_w(a_t, params.sigma0_upper_rot[1]),
                    eb.rotr_w(a_t, params.sigma0_upper_rot[2]),
                )
            ]
            mj = eb.maj_approx(a_t, a_m1, a_m2, maj_mode)
            t2 = eb.add_approx(s0, mj, cd)

            e_tp1 = eb.add_approx(a_m3, t1, cd)
            a_tp1 = eb.add_approx(t1, t2, cd)
            e_cols.append(e_tp1)
            a_cols.append(a_tp1)
        return a_cols[-1], e_cols[-1]

    def run_sched(w16: List[List[BitExpr]], total_len: int, carry_per_step: Sequence[int]) -> List[List[BitExpr]]:
        cols = list(w16)
        for idx, t in enumerate(range(16, total_len)):
            cd = carry_per_step[idx] if idx < len(carry_per_step) else 0
            s1 = [
                eb.xor_many((r0, r1, r2))
                for r0, r1, r2 in zip(
                    eb.rotr_w(cols[t - 2], params.sigma1_lower_op[0]),
                    eb.rotr_w(cols[t - 2], params.sigma1_lower_op[1]),
                    eb.shr_w(cols[t - 2], params.sigma1_lower_op[2]),
                )
            ]
            s0 = [
                eb.xor_many((r0, r1, r2))
                for r0, r1, r2 in zip(
                    eb.rotr_w(cols[t - 15], params.sigma0_lower_op[0]),
                    eb.rotr_w(cols[t - 15], params.sigma0_lower_op[1]),
                    eb.shr_w(cols[t - 15], params.sigma0_lower_op[2]),
                )
            ]
            u0 = eb.add_approx(s1, cols[t - 7], cd)
            u1 = eb.add_approx(u0, s0, cd)
            wt = eb.add_approx(u1, cols[t - 16], cd)
            cols.append(wt)
        return cols

    if spec.family in ("A1", "C4", "C8"):
        s0_w = [eb.var_word(i * w, w) for i in range(8)]
        w_sch = [eb.var_word((8 + t) * w, w) for t in range(spec.rounds)]
        a_r, e_r = run_comp(s0_w, w_sch, 0, spec.rounds, add_carry_policy)
        out_w = a_r if word_name.startswith("a") else e_r
        return out_w[bit_idx]

    if spec.family == "A2":
        s0_w = [eb.const_word(params.iv_consts[i], w) for i in range(8)]
        w_sch = [eb.var_word(t * w, w) for t in range(spec.rounds)]
        a_r, e_r = run_comp(s0_w, w_sch, 0, spec.rounds, add_carry_policy)
        out_w = a_r if word_name.startswith("a") else e_r
        return out_w[bit_idx]

    if spec.family == "B":
        w16 = [eb.var_word(t * w, w) for t in range(16)]
        target_w_idx = 15 + spec.rounds
        w_all = run_sched(w16, target_w_idx + 1, add_carry_policy)
        return w_all[target_w_idx][bit_idx]

    if spec.family == "B_joint":
        s16_w = [eb.var_word(i * w, w) for i in range(8)]
        w16 = [eb.var_word((8 + t) * w, w) for t in range(16)]
        w_all = run_sched(w16, 16 + spec.rounds, [0] * spec.rounds)
        a_r, e_r = run_comp(s16_w, w_all[16:], 16, spec.rounds, add_carry_policy)
        out_w = a_r if word_name.startswith("a") else e_r
        return out_w[bit_idx]

    raise ValueError(f"Unknown family {spec.family}")


@dataclass
class KnownBaselineResult:
    """Evaluated known baseline library B_known for one target."""

    circuits: List[CircuitSLP]
    extra_full_linear: List[Tuple[str, AffineForm]]
    best_name: str
    best_circuit: Optional[CircuitSLP]
    best_and_gates: int
    best_degree: int
    best_p: float
    best_bias: float
    best_pred_packed: torch.Tensor  # (B_words,) int64 on same device


def build_known_baseline_library(
    spec: OperatorSpec,
    target_name: str,
) -> Tuple[List[CircuitSLP], List[Tuple[str, AffineForm]]]:
    """Generate B1, B2, B3, B4 candidate circuits for (spec, target_name)."""
    r = spec.rounds
    circuits: List[CircuitSLP] = []
    extra_linear: List[Tuple[str, AffineForm]] = []
    seen_sigs = set()

    # Policies:
    # - All linear (B1/B3): carry=0 everywhere, all combinations of simple Ch/Maj linearizations
    ch_lin_modes = ["g", "f", "e_xor_g"] if spec.family != "B" else ["g"]
    maj_lin_modes = ["a", "b", "c", "xor3"] if spec.family != "B" else ["a"]

    policies: List[Tuple[str, str, str, Tuple[int, ...]]] = []
    for chm in ch_lin_modes:
        for mjm in maj_lin_modes:
            policies.append((f"B1_lin_{chm}_{mjm}", chm, mjm, (0,) * r))

    # - B2/B4 carry-1 & carry-2 on last round, or last 2 rounds, or first round (for A2 IV-folding)
    carry_patterns: List[Tuple[str, Tuple[int, ...]]] = [
        ("c_last1", (0,) * max(0, r - 1) + (1,)),
        ("c_last2", (0,) * max(0, r - 1) + (2,)),
        ("c_last2_1", (0,) * max(0, r - 2) + (1, 1)[-r:]),
        ("c_first1", (1,) + (0,) * max(0, r - 1)),
        ("c_first2", (2,) + (0,) * max(0, r - 1)),
        ("c_first3", (3,) + (0,) * max(0, r - 1)),
        ("c_first2_last1", (2,) + (0,) * max(0, r - 2) + ((1,) if r >= 2 else ())),
        ("c_all1", (1,) * r),
        ("c_all2", (2,) * r),
    ]
    if spec.family == "B":
        for pat_name, pat in carry_patterns:
            policies.append((f"B2_{pat_name}", "g", "a", pat))
    else:
        for pat_name, pat in carry_patterns:
            for chm, mjm in [("g", "a"), ("g", "xor3"), ("exact", "a"), ("g", "exact"), ("exact", "exact")]:
                policies.append((f"B4_{pat_name}_{chm}_{mjm}", chm, mjm, pat))

    for pname, chm, mjm, pat in policies:
        expr = _build_operator_expr(spec, target_name, chm, mjm, pat)
        slp, full_lin = compile_expr_to_slp(expr, spec.n_inputs, name=pname)
        if slp is not None:
            sig = (tuple((g.left.const, g.left.terms, g.right.const, g.right.terms) for g in slp.gates), slp.output.const, slp.output.terms)
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                circuits.append(slp)
        elif full_lin is not None:
            # Keep full-support linear mask as anti-triviality reference even if >32 vars
            extra_linear.append((pname + "_full_linear", full_lin))

    # Also add trivial constant baselines (0 and 1) and single-variable direct aliases
    for c_val in (0, 1):
        c_slp = CircuitSLP(
            n_inputs=spec.n_inputs,
            gates=[],
            output=AffineForm(c_val, ()),
            name=f"B0_const_{c_val}",
            method="known_baseline",
        )
        circuits.append(c_slp)

    return circuits, extra_linear


def evaluate_known_baselines(
    spec: OperatorSpec,
    target_name: str,
    input_packed: torch.Tensor,
    target_packed: torch.Tensor,
    n_samples: int,
    precompiled: Optional[Tuple[List[CircuitSLP], List[Tuple[str, AffineForm]]]] = None,
) -> KnownBaselineResult:
    """Build and evaluate B_known on packed bit tensors, returning the strongest known baseline."""
    if precompiled is not None:
        circuits, extra_linear = precompiled
    else:
        circuits, extra_linear = build_known_baseline_library(spec, target_name)
    ones_word = torch.full((input_packed.shape[1],), -1, dtype=torch.int64, device=input_packed.device)
    node_words = [input_packed[i] for i in range(spec.n_inputs)]

    best_bias = -1.0
    best_p = 0.5
    best_name = "B0_const_0"
    best_circuit: Optional[CircuitSLP] = None
    best_and = 0
    best_deg = 0
    best_pred = torch.zeros_like(target_packed)

    for c in circuits:
        pred = c.eval_packed(input_packed)
        matches = count_packed_matches(pred, target_packed)
        p = matches / float(n_samples)
        bias = abs(p - 0.5)
        if bias > best_bias:
            best_bias = bias
            best_p = p
            best_name = c.name
            best_circuit = c
            best_and = c.and_gates
            best_deg = c.exact_anf_degree() or c.structural_degree_bound()
            best_pred = pred

    for lname, laf in extra_linear:
        pred = laf.eval_packed(node_words, ones_word)
        matches = count_packed_matches(pred, target_packed)
        p = matches / float(n_samples)
        bias = abs(p - 0.5)
        if bias > best_bias:
            best_bias = bias
            best_p = p
            best_name = lname
            best_circuit = None
            best_and = 0
            best_deg = 1 if laf.terms else 0
            best_pred = pred

    return KnownBaselineResult(
        circuits=circuits,
        extra_full_linear=extra_linear,
        best_name=best_name,
        best_circuit=best_circuit,
        best_and_gates=best_and,
        best_degree=best_deg,
        best_p=best_p,
        best_bias=best_bias,
        best_pred_packed=best_pred,
    )


def get_structural_cone_vars(spec: OperatorSpec, target_name: str, max_vars: int = 64) -> List[int]:
    """Extract ordered input variables appearing in low-carry approximations of target_name."""
    ordered: List[int] = []
    seen = set()
    # Start from carry-0, then carry-1, then carry-2 across ch/maj modes
    for cd in (0, 1, 2):
        for chm, mjm in [("g", "a"), ("f", "b"), ("e_xor_g", "xor3"), ("exact", "exact")]:
            if spec.family == "B" and (chm, mjm) != ("g", "a"):
                continue
            expr = _build_operator_expr(spec, target_name, chm, mjm, (cd,) * spec.rounds)
            stack = [expr]
            visited = set()
            while stack:
                cur = stack.pop()
                oid = id(cur)
                if oid in visited:
                    continue
                visited.add(oid)
                if cur.kind == "VAR" and cur.var_idx not in seen:
                    seen.add(cur.var_idx)
                    ordered.append(cur.var_idx)
                    if len(ordered) >= max_vars:
                        return ordered
                elif cur.kind in ("XOR", "AND"):
                    stack.extend(cur.children)
    # Fallback: pad with nearby bit positions if needed
    for i in range(spec.n_inputs):
        if i not in seen:
            ordered.append(i)
            seen.add(i)
            if len(ordered) >= max_vars:
                break
    return ordered
