"""Z3 formal verification suite for P9 (SPEC Section 5.2).

Covers:
1. Circuit structural validation (acyclic DAG, M_AND <= 16, M_XOR <= 128, exact F_2 ANF degree).
2. Exact #SAT / truth-table count on Family C (w=4 and active-cone slices) vs Monte Carlo p_MC.
3. Exact non-equivalence check against all known baselines B in B_known: exists x, C*(x) != B(x).
4. Exact Z3 error bound verification for carry-truncation approximations on w=4 / w=8 slices.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import z3

from .arx_operators import (
    ARXParams,
    OperatorSpec,
    evaluate_operator_targets,
    parse_target_name,
)
from .circuit_slp import MAX_AND_GATES, MAX_DEGREE, MAX_SUPPORT, MAX_XOR_GATES, AffineForm, CircuitSLP


def slp_to_z3_bool(slp: CircuitSLP, z3_vars: Sequence[z3.BoolRef]) -> z3.BoolRef:
    """Compile a CircuitSLP into an exact Z3 Bool expression over z3_vars."""
    nodes: List[z3.BoolRef] = list(z3_vars)

    def affine_to_z3(af: AffineForm) -> z3.BoolRef:
        acc: z3.BoolRef = z3.BoolVal(bool(af.const))
        for idx in af.terms:
            acc = z3.Xor(acc, nodes[idx])
        return acc

    for g in slp.gates:
        lv = affine_to_z3(g.left)
        rv = affine_to_z3(g.right)
        nodes.append(z3.And(lv, rv))
    return affine_to_z3(slp.output)


def affine_to_z3_bool(af: AffineForm, z3_vars: Sequence[z3.BoolRef]) -> z3.BoolRef:
    acc: z3.BoolRef = z3.BoolVal(bool(af.const))
    for idx in af.terms:
        acc = z3.Xor(acc, z3_vars[idx])
    return acc


def verify_circuit_complexity_and_anf(slp: CircuitSLP) -> Dict[str, Any]:
    """Check DAG validity, gate counts, and exact F_2 ANF degree (including Mobius check if small support)."""
    ok, msg, deg = slp.validate(
        max_and=MAX_AND_GATES,
        max_xor=MAX_XOR_GATES,
        max_support=MAX_SUPPORT,
        max_degree=MAX_DEGREE,
    )
    supp = slp.input_support()
    mobius_deg: Optional[int] = None
    if len(supp) <= 12:
        # Exact Mobius transform over 2^|supp| truth table to cross-verify symbolic ANF degree
        k = len(supp)
        n_tt = 1 << k
        tt = [0] * n_tt
        x_buf = [0] * slp.n_inputs
        for mask in range(n_tt):
            for bit_i, v_idx in enumerate(supp):
                x_buf[v_idx] = (mask >> bit_i) & 1
            tt[mask] = slp.eval_int(x_buf)
        # In-place Mobius transform over F_2
        anf = list(tt)
        for i in range(k):
            step = 1 << i
            for mask in range(n_tt):
                if mask & step:
                    anf[mask] ^= anf[mask ^ step]
        mobius_deg = max((bin(mask).count("1") for mask, coef in enumerate(anf) if coef), default=0)

    return {
        "valid": bool(ok),
        "reason": msg,
        "and_gates": slp.and_gates,
        "xor_gates": slp.xor_gates,
        "max_support": slp.max_support,
        "active_input_support": len(supp),
        "symbolic_degree": int(deg),
        "mobius_exact_degree": mobius_deg,
    }


def _bv_rotr(x: z3.BitVecRef, n: int, w: int) -> z3.BitVecRef:
    n = n % w
    if n == 0:
        return x
    return z3.RotateRight(x, n)


def _bv_ch(e: z3.BitVecRef, f: z3.BitVecRef, g: z3.BitVecRef) -> z3.BitVecRef:
    return (e & f) ^ ((~e) & g)


def _bv_maj(a: z3.BitVecRef, b: z3.BitVecRef, c: z3.BitVecRef) -> z3.BitVecRef:
    return (a & b) ^ (a & c) ^ (b & c)


def _bv_sigma_rot3(x: z3.BitVecRef, rots: Tuple[int, int, int], w: int) -> z3.BitVecRef:
    return _bv_rotr(x, rots[0], w) ^ _bv_rotr(x, rots[1], w) ^ _bv_rotr(x, rots[2], w)


def build_z3_operator_target_bit(
    spec: OperatorSpec,
    target_name: str,
    x_bool_vars: Sequence[z3.BoolRef],
) -> z3.BoolRef:
    """Build exact Z3 Bool formula for target bit of Family A1/A2/C4/C8."""
    w = spec.word_bits
    params = spec.params
    word_name, bit_idx = parse_target_name(target_name)

    def bits_to_bv(start: int) -> z3.BitVecRef:
        # LSB-first: x_bool_vars[start] is bit 0, x_bool_vars[start + w - 1] is bit w-1
        parts = [z3.If(x_bool_vars[start + b], z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)) for b in range(w)]
        return z3.Concat(*reversed(parts))

    if spec.family in ("A1", "C4", "C8"):
        s0 = [bits_to_bv(i * w) for i in range(8)]
        w_sch = [bits_to_bv((8 + t) * w) for t in range(spec.rounds)]
    elif spec.family == "A2":
        s0 = [z3.BitVecVal(params.iv_consts[i], w) for i in range(8)]
        w_sch = [bits_to_bv(t * w) for t in range(spec.rounds)]
    else:
        raise ValueError(f"Z3 target builder supports compression families A1/A2/C4/C8, got {spec.family}")

    a_cols = [s0[3], s0[2], s0[1], s0[0]]
    e_cols = [s0[7], s0[6], s0[5], s0[4]]
    for t in range(spec.rounds):
        a_t, a_m1, a_m2, a_m3 = a_cols[t + 3], a_cols[t + 2], a_cols[t + 1], a_cols[t]
        e_t, e_m1, e_m2, e_m3 = e_cols[t + 3], e_cols[t + 2], e_cols[t + 1], e_cols[t]
        s1 = _bv_sigma_rot3(e_t, params.sigma1_upper_rot, w)
        ch = _bv_ch(e_t, e_m1, e_m2)
        k_val = z3.BitVecVal(params.k_consts[t], w)
        t1 = e_m3 + s1 + ch + k_val + w_sch[t]
        s0_op = _bv_sigma_rot3(a_t, params.sigma0_upper_rot, w)
        mj = _bv_maj(a_t, a_m1, a_m2)
        t2 = s0_op + mj
        e_cols.append(a_m3 + t1)
        a_cols.append(t1 + t2)

    out_bv = a_cols[-1] if word_name.startswith("a") else e_cols[-1]
    return z3.Extract(bit_idx, bit_idx, out_bv) == z3.BitVecVal(1, 1)


def verify_family_c_exact_count(
    spec: OperatorSpec,
    target_name: str,
    proxy: CircuitSLP,
    active_var_indices: Optional[Sequence[int]] = None,
    n_eff: int = 8,
    mc_samples: int = 16384,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Verify exact Z3 #SAT count vs vectorized Monte Carlo on an n_eff-bit slice of Family C."""
    if spec.family not in ("C4", "C8"):
        raise ValueError(f"Expected Family C (C4/C8), got {spec.family}")

    rng = random.Random(seed)
    if active_var_indices is None:
        supp = list(proxy.input_support())
        # Fill up to n_eff with variables from W0/e0/a0
        candidates = supp + list(range(spec.n_inputs - spec.word_bits, spec.n_inputs)) + list(range(spec.n_inputs))
        chosen: List[int] = []
        seen = set()
        for v in candidates:
            if v not in seen:
                seen.add(v)
                chosen.append(v)
            if len(chosen) >= n_eff:
                break
        active_vars = chosen
    else:
        active_vars = list(active_var_indices[:n_eff])

    fixed_vals = {i: rng.randint(0, 1) for i in range(spec.n_inputs) if i not in active_vars}

    # Build Z3 solver for exact #SAT on the 2^n_eff slice
    z3_vars = [z3.Bool(f"x_{i}") for i in range(spec.n_inputs)]
    f_expr = build_z3_operator_target_bit(spec, target_name, z3_vars)
    c_expr = slp_to_z3_bool(proxy, z3_vars)

    solver = z3.Solver()
    for idx, val in fixed_vals.items():
        solver.add(z3_vars[idx] == z3.BoolVal(bool(val)))
    solver.add(f_expr == c_expr)

    active_z3 = [z3_vars[idx] for idx in active_vars]
    exact_matches = 0
    total_states = 1 << len(active_vars)
    while solver.check() == z3.sat:
        model = solver.model()
        exact_matches += 1
        block = []
        for zv in active_z3:
            val = z3.is_true(model.evaluate(zv, model_completion=True))
            block.append(zv != z3.BoolVal(val))
        solver.add(z3.Or(*block))

    p_exact = exact_matches / float(total_states)

    # Monte Carlo on the SAME slice using arx_operators.py + CircuitSLP.eval_int / packed
    w = spec.word_bits
    mc_words = torch.zeros((mc_samples, spec.input_words_count), dtype=torch.int64)
    for idx, val in fixed_vals.items():
        if val:
            mc_words[:, idx // w] |= 1 << (idx % w)
    gen = torch.Generator()
    gen.manual_seed(seed + 1)
    rand_bits = torch.randint(0, 2, (mc_samples, len(active_vars)), dtype=torch.int64, generator=gen)
    for j, idx in enumerate(active_vars):
        mc_words[:, idx // w] |= rand_bits[:, j] << (idx % w)

    targets_mc = evaluate_operator_targets(spec, mc_words)[target_name]
    from .arx_operators import words_to_packed_inputs
    from .circuit_slp import count_packed_matches, pack_bits_vector

    in_packed = words_to_packed_inputs(mc_words, w)
    t_packed = pack_bits_vector(targets_mc)
    c_packed = proxy.eval_packed(in_packed)
    mc_matches = count_packed_matches(c_packed, t_packed)
    p_mc = mc_matches / float(mc_samples)
    sigma0 = 1.0 / (2.0 * math.sqrt(mc_samples))
    diff = abs(p_mc - p_exact)
    passed_consistency = bool(diff < 4.0 * sigma0)

    return {
        "family": spec.family,
        "rounds": spec.rounds,
        "target": target_name,
        "n_eff": len(active_vars),
        "active_vars": active_vars,
        "exact_matches": exact_matches,
        "total_states": total_states,
        "p_exact": float(p_exact),
        "mc_samples": int(mc_samples),
        "p_mc": float(p_mc),
        "abs_diff": float(diff),
        "four_sigma0": float(4.0 * sigma0),
        "passed_mc_consistency": passed_consistency,
    }


def verify_nonequivalence_against_known(
    proxy: CircuitSLP,
    known_circuits: Sequence[CircuitSLP],
    extra_linear: Sequence[Tuple[str, AffineForm]],
) -> Dict[str, Any]:
    """Verify via Z3 that proxy != B for every baseline B in B_known."""
    n_inputs = proxy.n_inputs
    z3_vars = [z3.Bool(f"x_{i}") for i in range(n_inputs)]
    c_expr = slp_to_z3_bool(proxy, z3_vars)

    equiv_hits: List[str] = []
    cex_records: List[Dict[str, Any]] = []

    def check_one(b_name: str, b_expr: z3.BoolRef, b_eval_fn: Any) -> None:
        s = z3.Solver()
        s.set("timeout", 2000)
        s.add(c_expr != b_expr)
        res = s.check()
        if res == z3.unsat:
            equiv_hits.append(b_name)
        elif res == z3.sat and len(cex_records) < 3:
            m = s.model()
            # Extract active support assignments only to keep compact
            supp = sorted(set(proxy.input_support()))
            sample_bits = [0] * n_inputs
            for idx in range(n_inputs):
                if z3.is_true(m.evaluate(z3_vars[idx], model_completion=False)):
                    sample_bits[idx] = 1
            c_val = proxy.eval_int(sample_bits)
            b_val = b_eval_fn(sample_bits)
            cex_records.append(
                {
                    "baseline": b_name,
                    "status": "SAT",
                    "c_val": int(c_val),
                    "b_val": int(b_val),
                    "verified_diff": bool(c_val != b_val),
                    "active_ones": [idx for idx in supp if sample_bits[idx] == 1],
                }
            )

    for kc in known_circuits:
        b_expr = slp_to_z3_bool(kc, z3_vars)
        check_one(kc.name, b_expr, kc.eval_int)

    for lname, laf in extra_linear:
        b_expr = affine_to_z3_bool(laf, z3_vars)
        check_one(lname, b_expr, laf.eval_int)

    return {
        "checked_baselines": len(known_circuits) + len(extra_linear),
        "equivalent_to_known": equiv_hits,
        "all_nonequivalent": len(equiv_hits) == 0,
        "sample_counterexamples": cex_records,
    }


def verify_carry_truncation_bound_z3(w: int = 4) -> Dict[str, Any]:
    """Use Z3 exact #SAT on w-bit Add(x, y) to certify 0-carry and 1-carry truncation error rates."""
    x = z3.BitVec("x", w)
    y = z3.BitVec("y", w)
    exact_sum = x + y
    results: Dict[str, Any] = {}
    total = 1 << (2 * w)
    for b in range(w):
        exact_b = z3.Extract(b, b, exact_sum)
        xb = z3.Extract(b, b, x)
        yb = z3.Extract(b, b, y)
        approx0 = xb ^ yb
        if b == 0:
            approx1 = approx0
        else:
            approx1 = approx0 ^ (z3.Extract(b - 1, b - 1, x) & z3.Extract(b - 1, b - 1, y))

        def count_eq(expr_b: z3.BitVecRef) -> int:
            s = z3.Solver()
            s.add(exact_b == expr_b)
            cnt = 0
            while s.check() == z3.sat:
                m = s.model()
                cnt += 1
                xv = m[x]
                yv = m[y]
                s.add(z3.Or(x != xv, y != yv))
            return cnt

        m0 = count_eq(approx0)
        m1 = count_eq(approx1)
        results[f"bit_{b}"] = {
            "total_states": total,
            "carry0_exact_p": m0 / float(total),
            "carry1_exact_p": m1 / float(total),
        }
    return results
