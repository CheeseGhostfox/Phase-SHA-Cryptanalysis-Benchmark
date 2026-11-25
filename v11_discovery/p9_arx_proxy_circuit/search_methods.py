"""Three complementary proxy circuit search engines for P9:

1. Method 1: Walsh-Hadamard Spectral + AlphaTensor Multilinear Residual Matching Pursuit
2. Method 2: Evolutionary / MCTS Straight-Line Program (SLP) Search
3. Method 3: Pure-FP32 Differentiable Logic Gate Relaxation (STE Bipolar Product Net)

All methods strictly search ONLY on the Search Split, track exact `n_trials` for Bonferroni
correction, enforce budgets (M_AND <= 16, M_XOR <= 128, S_max <= 32, deg <= 4), and return
hard-discretized CircuitSLP objects.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .arx_operators import OperatorSpec
from .circuit_slp import (
    MAX_AND_GATES,
    MAX_DEGREE,
    MAX_SUPPORT,
    MAX_XOR_GATES,
    AffineForm,
    AndGate,
    CircuitSLP,
    batch_count_packed_matches,
    count_packed_matches,
)
from .known_baselines import KnownBaselineResult, get_structural_cone_vars


@dataclass
class SearchOutcome:
    """Output of combined search across all 3 methods on the Search Split."""

    best_circuit: CircuitSLP
    top_circuits: List[CircuitSLP]
    n_trials: int
    method_best: Dict[str, float]


def _eval_candidates_bias(
    candidates: List[CircuitSLP],
    input_packed: torch.Tensor,
    target_packed: torch.Tensor,
    n_samples: int,
) -> List[float]:
    if not candidates:
        return []
    preds = torch.stack([c.eval_packed(input_packed) for c in candidates], dim=0)
    matches = batch_count_packed_matches(preds, target_packed)
    ps = matches.to(torch.float64) / float(n_samples)
    biases = torch.abs(ps - 0.5).tolist()
    return [float(b) for b in biases]


def _select_candidate_vars(
    spec: OperatorSpec,
    target_name: str,
    input_packed: torch.Tensor,
    target_packed: torch.Tensor,
    n_samples: int,
    max_pool: int = 48,
) -> List[int]:
    """Combine backward structural cone variables + top single-bit Walsh-correlated variables."""
    cone_vars = get_structural_cone_vars(spec, target_name, max_vars=max_pool // 2)
    matches = batch_count_packed_matches(input_packed, target_packed)
    biases = torch.abs(matches.to(torch.float64) / float(n_samples) - 0.5)
    top_idx = torch.topk(biases, k=min(max_pool, spec.n_inputs)).indices.tolist()

    pool: List[int] = []
    seen = set()
    for v in cone_vars + top_idx:
        if v not in seen:
            seen.add(v)
            pool.append(int(v))
        if len(pool) >= max_pool:
            break
    return pool


def _greedy_refine_affine_mask(
    base_packed: torch.Tensor,
    target_packed: torch.Tensor,
    input_packed: torch.Tensor,
    candidate_vars: Sequence[int],
    n_samples: int,
    init_terms: Tuple[int, ...] = (),
    max_supp: int = 24,
    steps: int = 8,
) -> Tuple[AffineForm, float, int]:
    """Coordinate ascent over sparse linear mask l(x) minimizing residual against target_packed."""
    trials = 0
    terms_set = set(init_terms)
    cur_packed = base_packed.clone()
    for v in terms_set:
        cur_packed = cur_packed ^ input_packed[v]

    cur_matches = count_packed_matches(cur_packed, target_packed)
    cur_p = cur_matches / float(n_samples)
    cur_bias = abs(cur_p - 0.5)
    trials += 1

    cand_list = list(candidate_vars)
    if not cand_list:
        const = 1 if cur_p < 0.5 else 0
        return AffineForm(const, tuple(sorted(terms_set))), cur_bias, trials

    cand_rows = input_packed[cand_list]  # (K, B_words)
    for _ in range(steps):
        # Toggle each candidate variable in parallel
        toggled = cand_rows ^ cur_packed.unsqueeze(0)
        matches = batch_count_packed_matches(toggled, target_packed)
        trials += len(cand_list)
        biases = torch.abs(matches.to(torch.float64) / float(n_samples) - 0.5)

        # Mask out additions that would exceed max_supp
        for i, v in enumerate(cand_list):
            if v not in terms_set and len(terms_set) >= max_supp:
                biases[i] = -1.0

        best_i = int(torch.argmax(biases).item())
        best_b = float(biases[best_i].item())
        if best_b > cur_bias + 1e-6:
            v_best = cand_list[best_i]
            if v_best in terms_set:
                terms_set.remove(v_best)
            else:
                terms_set.add(v_best)
            cur_packed = toggled[best_i]
            cur_bias = best_b
            cur_p = float(matches[best_i].item()) / float(n_samples)
        else:
            break

    const = 1 if cur_p < 0.5 else 0
    return AffineForm(const, tuple(sorted(terms_set))), cur_bias, trials


def walsh_alphatensor_search(
    spec: OperatorSpec,
    target_name: str,
    input_packed: torch.Tensor,
    target_packed: torch.Tensor,
    n_samples: int,
    cand_vars: Sequence[int],
    known_res: KnownBaselineResult,
    max_and_budget: int = MAX_AND_GATES,
    max_deg: int = MAX_DEGREE,
    seed: int = 42,
) -> Tuple[List[Tuple[float, CircuitSLP]], int]:
    """Method 1: Walsh-Hadamard + AlphaTensor multilinear residual decomposition."""
    rng = random.Random(seed)
    n_inputs = spec.n_inputs
    device = input_packed.device
    trials = 0
    found: List[Tuple[float, CircuitSLP]] = []

    zeros_packed = torch.zeros_like(target_packed)
    # 1) Find strong linear bases (deg <= 1) from multiple seeds (empty, top vars, known linear)
    lin_seeds: List[Tuple[int, ...]] = [()]
    for v in cand_vars[:6]:
        lin_seeds.append((v,))
    for kc in known_res.circuits[:4]:
        if kc.and_gates == 0 and kc.output.terms:
            lin_seeds.append(kc.output.terms[:24])

    lin_pool: List[Tuple[float, AffineForm, torch.Tensor]] = []
    for s_terms in lin_seeds:
        af, bias, t_cnt = _greedy_refine_affine_mask(
            zeros_packed, target_packed, input_packed, cand_vars, n_samples, init_terms=s_terms, max_supp=24, steps=8
        )
        trials += t_cnt
        ones_word = torch.full((input_packed.shape[1],), -1, dtype=torch.int64, device=device)
        node_words = [input_packed[i] for i in range(n_inputs)]
        pred = af.eval_packed(node_words, ones_word)
        lin_pool.append((bias, af, pred))
        slp = CircuitSLP(n_inputs=n_inputs, gates=[], output=af, name="m1_lin", method="walsh_alphatensor")
        ok, _, _ = slp.validate(max_and=max_and_budget, max_degree=max_deg)
        if ok:
            found.append((bias, slp))

    lin_pool.sort(key=lambda x: x[0], reverse=True)
    best_bias, best_af, best_pred = lin_pool[0]

    # Build a diverse dictionary of sparse linear forms for product term synthesis
    form_candidates: List[AffineForm] = []
    for v in cand_vars[:28]:
        form_candidates.append(AffineForm(0, (v,)))
        form_candidates.append(AffineForm(1, (v,)))
    # Pairwise XOR sums of top cone variables
    for i in range(min(12, len(cand_vars))):
        for j in range(i + 1, min(12, len(cand_vars))):
            form_candidates.append(AffineForm(0, (cand_vars[i], cand_vars[j])))
    # Include top discovered linear forms
    for _, af, _ in lin_pool[:3]:
        if af.terms:
            form_candidates.append(AffineForm(0, af.terms))
            form_candidates.append(AffineForm(1, af.terms))

    # Deduplicate and pre-evaluate linear forms on input_packed
    uniq_forms: List[AffineForm] = []
    seen_f = set()
    ones_word = torch.full((input_packed.shape[1],), -1, dtype=torch.int64, device=device)
    node_words = [input_packed[i] for i in range(n_inputs)]
    for af in form_candidates:
        key = (af.const, af.terms)
        if key not in seen_f and 1 <= af.support_size <= 24:
            seen_f.add(key)
            uniq_forms.append(af)
    if len(uniq_forms) > 64:
        uniq_forms = uniq_forms[:64]

    form_packed = torch.stack([af.eval_packed(node_words, ones_word) for af in uniq_forms], dim=0)  # (F, B_words)

    # 2) Iterative AlphaTensor residual matching pursuit: add rank-1 product terms (deg 2, 3, 4)
    # Run from two starting points: (a) best linear base, (b) empty base
    start_states: List[Tuple[List[AndGate], AffineForm, torch.Tensor]] = [
        ([], best_af, best_pred),
        ([], AffineForm(0, ()), zeros_packed),
    ]
    for kc in known_res.circuits[:2]:
        if 0 < kc.and_gates <= 4:
            start_states.append((list(kc.gates), kc.output, kc.eval_packed(input_packed)))

    for init_gates, init_out, init_pred in start_states:
        gates = list(init_gates)
        cur_out = init_out
        cur_pred = init_pred
        cur_matches = count_packed_matches(cur_pred, target_packed)
        cur_bias = abs(cur_matches / float(n_samples) - 0.5)

        # Iteratively add product terms up to max_and_budget
        while len(gates) < max_and_budget:
            remaining_and = max_and_budget - len(gates)
            residual = cur_pred ^ target_packed  # we want product term P to correlate with residual (or ~residual)

            # Evaluate all pairwise ANDs (deg 2) from top 36 forms
            m_f = min(36, len(uniq_forms))
            idx_a, idx_b = torch.triu_indices(m_f, m_f, offset=1)
            pair_prods = form_packed[idx_a] & form_packed[idx_b]  # (P, B_words)
            pair_matches = batch_count_packed_matches(pair_prods, residual)
            trials += int(pair_prods.shape[0])
            pair_biases = torch.abs(pair_matches.to(torch.float64) / float(n_samples) - 0.5)

            best_pair_idx = int(torch.argmax(pair_biases).item())
            best_deg2_bias = float(pair_biases[best_pair_idx].item())
            best_i = int(idx_a[best_pair_idx].item())
            best_j = int(idx_b[best_pair_idx].item())

            best_term_gates: List[AndGate] = [AndGate(uniq_forms[best_i], uniq_forms[best_j])]
            best_term_packed = pair_prods[best_pair_idx]
            best_term_bias = best_deg2_bias

            # Coordinate refine the 2nd factor holding the 1st factor fixed
            f1_packed = form_packed[best_i]
            f2_af = uniq_forms[best_j]
            # Try toggling each candidate var inside f2_af under the mask f1_packed
            cand_slice = list(cand_vars[:32])
            if cand_slice:
                f2_base = form_packed[best_j]
                toggled_f2 = f2_base.unsqueeze(0) ^ input_packed[cand_slice]
                refined_prods = f1_packed.unsqueeze(0) & toggled_f2
                r_matches = batch_count_packed_matches(refined_prods, residual)
                trials += len(cand_slice)
                r_biases = torch.abs(r_matches.to(torch.float64) / float(n_samples) - 0.5)
                rb_i = int(torch.argmax(r_biases).item())
                if float(r_biases[rb_i].item()) > best_term_bias + 1e-6:
                    v_add = cand_slice[rb_i]
                    new_terms = tuple(sorted(set(f2_af.terms) ^ {v_add}))
                    if len(new_terms) <= MAX_SUPPORT:
                        new_f2 = AffineForm(f2_af.const, new_terms)
                        best_term_gates = [AndGate(uniq_forms[best_i], new_f2)]
                        best_term_packed = refined_prods[rb_i]
                        best_term_bias = float(r_biases[rb_i].item())

            # Try extending to deg 3 or deg 4 if budget and max_deg allow
            if max_deg >= 3 and remaining_and >= 2:
                ext3 = best_term_packed.unsqueeze(0) & form_packed[:m_f]
                m3 = batch_count_packed_matches(ext3, residual)
                trials += m_f
                b3 = torch.abs(m3.to(torch.float64) / float(n_samples) - 0.5)
                k3 = int(torch.argmax(b3).item())
                if float(b3[k3].item()) > best_term_bias + 5e-4:
                    g_prev_idx = n_inputs + len(gates)
                    best_term_gates.append(AndGate(AffineForm(0, (g_prev_idx,)), uniq_forms[k3]))
                    best_term_packed = ext3[k3]
                    best_term_bias = float(b3[k3].item())

                    if max_deg >= 4 and remaining_and >= 3:
                        ext4 = best_term_packed.unsqueeze(0) & form_packed[:m_f]
                        m4 = batch_count_packed_matches(ext4, residual)
                        trials += m_f
                        b4 = torch.abs(m4.to(torch.float64) / float(n_samples) - 0.5)
                        k4 = int(torch.argmax(b4).item())
                        if float(b4[k4].item()) > best_term_bias + 5e-4:
                            g_prev2_idx = n_inputs + len(gates) + 1
                            best_term_gates.append(AndGate(AffineForm(0, (g_prev2_idx,)), uniq_forms[k4]))
                            best_term_packed = ext4[k4]
                            best_term_bias = float(b4[k4].item())

            new_gates = gates + best_term_gates
            new_gate_out_idx = n_inputs + len(new_gates) - 1
            if cur_out.support_size >= MAX_SUPPORT:
                break
            new_out_terms = tuple(sorted(set(cur_out.terms) ^ {new_gate_out_idx}))
            # Flip output sign if best_term correlates negatively
            cand_pred = cur_pred ^ best_term_packed
            m_new = count_packed_matches(cand_pred, target_packed)
            p_new = m_new / float(n_samples)
            new_const = cur_out.const ^ (1 if p_new < 0.5 else 0)
            new_out = AffineForm(new_const, new_out_terms)

            cand_slp = CircuitSLP(
                n_inputs=n_inputs,
                gates=new_gates,
                output=new_out,
                name=f"m1_at_g{len(new_gates)}",
                method="walsh_alphatensor",
            ).prune_dead_gates()
            ok, _, _ = cand_slp.validate(max_and=max_and_budget, max_degree=max_deg)
            if not ok:
                break

            new_bias = abs(p_new - 0.5)
            found.append((new_bias, cand_slp))
            if new_bias > cur_bias + 1e-5:
                gates = new_gates
                cur_out = new_out
                cur_pred = cand_pred if p_new >= 0.5 else (cand_pred ^ ones_word)
                cur_bias = new_bias
            else:
                # Try one random exploration jump before stopping
                ri, rj = rng.randrange(m_f), rng.randrange(m_f)
                if ri == rj:
                    break
                r_gate = AndGate(uniq_forms[ri], uniq_forms[rj])
                if len(gates) + 1 > max_and_budget or cur_out.support_size >= MAX_SUPPORT:
                    break
                gates.append(r_gate)
                cur_out = AffineForm(cur_out.const, tuple(sorted(set(cur_out.terms) ^ {n_inputs + len(gates) - 1})))
                cur_pred = cur_pred ^ (form_packed[ri] & form_packed[rj])
                cur_bias = abs(count_packed_matches(cur_pred, target_packed) / float(n_samples) - 0.5)

    return found, trials


def evolutionary_mcts_slp_search(
    spec: OperatorSpec,
    input_packed: torch.Tensor,
    target_packed: torch.Tensor,
    n_samples: int,
    cand_vars: Sequence[int],
    seed_circuits: Sequence[CircuitSLP],
    max_and_budget: int = MAX_AND_GATES,
    max_deg: int = MAX_DEGREE,
    generations: int = 12,
    pop_size: int = 24,
    seed: int = 123,
) -> Tuple[List[Tuple[float, CircuitSLP]], int]:
    """Method 2: Evolutionary / MCTS-style SLP DAG mutation & budget-stratified search."""
    rng = random.Random(seed)
    n_inputs = spec.n_inputs
    trials = 0
    cand_list = list(cand_vars) if cand_vars else list(range(min(32, n_inputs)))

    def rand_affine(max_idx: int, max_terms: int = 4) -> AffineForm:
        k = rng.randint(1, min(max_terms, max_idx))
        # Bias heavily toward cand_list variables
        chosen = []
        for _ in range(k):
            if rng.random() < 0.85 or max_idx <= n_inputs:
                chosen.append(rng.choice(cand_list))
            else:
                chosen.append(rng.randrange(n_inputs, max_idx))
        return AffineForm(rng.randint(0, 1), tuple(chosen))

    def mutate_affine(af: AffineForm, max_idx: int) -> AffineForm:
        terms = list(af.terms)
        r = rng.random()
        if r < 0.2:
            return AffineForm(af.const ^ 1, tuple(terms))
        if r < 0.6 or not terms:
            v = rng.choice(cand_list) if (rng.random() < 0.85 or max_idx <= n_inputs) else rng.randrange(n_inputs, max_idx)
            if v in terms:
                terms.remove(v)
            elif len(terms) < MAX_SUPPORT:
                terms.append(v)
        else:
            terms.pop(rng.randrange(len(terms)))
        return AffineForm(af.const, tuple(terms))

    def mutate_circuit(c: CircuitSLP) -> Optional[CircuitSLP]:
        gates = list(c.gates)
        out = c.output
        op = rng.random()
        if op < 0.35 and len(gates) < max_and_budget:
            # Add an AND gate
            max_idx = n_inputs + len(gates)
            new_g = AndGate(rand_affine(max_idx, 3), rand_affine(max_idx, 3))
            gates.append(new_g)
            new_node = n_inputs + len(gates) - 1
            if out.support_size < MAX_SUPPORT:
                out = AffineForm(out.const, tuple(sorted(set(out.terms) ^ {new_node})))
        elif op < 0.75 and gates:
            # Mutate an existing AND gate
            gi = rng.randrange(len(gates))
            max_idx = n_inputs + gi
            g = gates[gi]
            if rng.random() < 0.5:
                gates[gi] = AndGate(mutate_affine(g.left, max_idx), g.right)
            else:
                gates[gi] = AndGate(g.left, mutate_affine(g.right, max_idx))
        else:
            # Mutate output affine form
            out = mutate_affine(out, n_inputs + len(gates))

        cand = CircuitSLP(
            n_inputs=n_inputs,
            gates=gates,
            output=out,
            name=f"m2_evo_g{len(gates)}",
            method="evolutionary_mcts_slp",
        ).prune_dead_gates()
        ok, _, _ = cand.validate(max_and=max_and_budget, max_degree=max_deg)
        return cand if ok else None

    archive: List[Tuple[float, CircuitSLP]] = []
    # Seed archive
    valid_seeds: List[CircuitSLP] = []
    for sc in seed_circuits:
        ok, _, _ = sc.validate(max_and=max_and_budget, max_degree=max_deg)
        if ok:
            valid_seeds.append(sc)
    for _ in range(8):
        g_cnt = rng.choice([0, 1, 2, 4])
        gs: List[AndGate] = []
        for m in range(min(g_cnt, max_and_budget)):
            gs.append(AndGate(rand_affine(n_inputs + m, 3), rand_affine(n_inputs + m, 3)))
        out_af = AffineForm(rng.randint(0, 1), tuple(n_inputs + m for m in range(len(gs))) + (rng.choice(cand_list),))
        c = CircuitSLP(n_inputs=n_inputs, gates=gs, output=out_af, name="m2_rand", method="evolutionary_mcts_slp").prune_dead_gates()
        ok, _, _ = c.validate(max_and=max_and_budget, max_degree=max_deg)
        if ok:
            valid_seeds.append(c)

    if valid_seeds:
        biases = _eval_candidates_bias(valid_seeds, input_packed, target_packed, n_samples)
        trials += len(valid_seeds)
        archive.extend(zip(biases, valid_seeds))

    archive.sort(key=lambda x: x[0], reverse=True)
    archive = archive[:pop_size]

    for _gen in range(generations):
        if not archive:
            break
        offspring: List[CircuitSLP] = []
        # UCB-like parent selection (top elite + exploration)
        for idx in range(pop_size):
            parent_idx = idx % min(8, len(archive)) if idx < pop_size // 2 else rng.randrange(len(archive))
            child = mutate_circuit(archive[parent_idx][1])
            if child is not None:
                offspring.append(child)
        if offspring:
            biases = _eval_candidates_bias(offspring, input_packed, target_packed, n_samples)
            trials += len(offspring)
            archive.extend(zip(biases, offspring))
            # Deduplicate by algebraic string
            dedup: Dict[str, Tuple[float, CircuitSLP]] = {}
            for b, c in archive:
                key = c.format_algebraic()
                if key not in dedup or b > dedup[key][0]:
                    dedup[key] = (b, c)
            archive = sorted(dedup.values(), key=lambda x: x[0], reverse=True)[:pop_size]

    return archive, trials


class _STEBipolarProductNet(nn.Module):
    """Pure-FP32 differentiable XOR-AND / multilinear product circuit with STE binarization."""

    def __init__(self, n_sub: int, n_deg2: int, n_deg3: int, n_deg4: int) -> None:
        super().__init__()
        self.n_sub = n_sub
        self.n_deg2 = n_deg2
        self.n_deg3 = n_deg3
        self.n_deg4 = n_deg4
        # Linear base logits (negative = sparse off)
        self.lin_logits = nn.Parameter(torch.full((n_sub,), -2.0, dtype=torch.float32))
        # Product factors: each AND term takes 2, 3, or 4 affine factors
        self.d2_logits = nn.Parameter(torch.randn(n_deg2, 2, n_sub, dtype=torch.float32) * 0.5 - 2.0) if n_deg2 > 0 else None
        self.d3_logits = nn.Parameter(torch.randn(n_deg3, 3, n_sub, dtype=torch.float32) * 0.5 - 2.2) if n_deg3 > 0 else None
        self.d4_logits = nn.Parameter(torch.randn(n_deg4, 4, n_sub, dtype=torch.float32) * 0.5 - 2.4) if n_deg4 > 0 else None

        n_terms = n_deg2 + n_deg3 + n_deg4
        self.term_gate_logits = nn.Parameter(torch.full((n_terms,), -1.0, dtype=torch.float32)) if n_terms > 0 else None

    @staticmethod
    def _ste_prob(logits: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        p = torch.sigmoid(logits / temp)
        hard = (p > 0.5).to(torch.float32)
        return hard.detach() - p.detach() + p

    @staticmethod
    def _bipolar_affine(x_sub: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x_sub: (B, n_sub) in {0, 1}, mask: (..., n_sub) in [0, 1]
        # factor = 1 - 2 * mask * x_sub in [-1, 1]
        # Reshape for broadcasting
        factors = 1.0 - 2.0 * x_sub.unsqueeze(1) * mask.unsqueeze(0)  #Wait: flexible dims handled below
        return torch.prod(factors, dim=-1)

    def forward(self, x_sub: torch.Tensor, temp: float = 1.0, hard_ste: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_sub: (B, n_sub) float32 in {0.0, 1.0}
        b_size = x_sub.shape[0]
        if hard_ste:
            m_lin = self._ste_prob(self.lin_logits, temp)  # (n_sub,)
        else:
            m_lin = torch.sigmoid(self.lin_logits / temp)
        # Bipolar parity of linear base
        z_lin = torch.prod(1.0 - 2.0 * x_sub * m_lin.unsqueeze(0), dim=-1)  # (B,)

        z_terms: List[torch.Tensor] = []
        reg = m_lin.sum() * 0.01

        def eval_prod_group(logits: Optional[torch.Tensor]) -> None:
            nonlocal reg
            if logits is None:
                return
            m = self._ste_prob(logits, temp) if hard_ste else torch.sigmoid(logits / temp)  # (T, D, n_sub)
            reg = reg + m.sum() * 0.01
            # x_sub: (B, 1, 1, n_sub), m: (1, T, D, n_sub)
            f = 1.0 - 2.0 * x_sub.view(b_size, 1, 1, self.n_sub) * m.unsqueeze(0)
            z_factors = torch.prod(f, dim=-1)  # (B, T, D) in {-1, +1}
            u_factors = 0.5 * (1.0 - z_factors)  # (B, T, D) in {0, 1}
            u_prod = torch.prod(u_factors, dim=-1)  # (B, T) in {0, 1}
            z_prod = 1.0 - 2.0 * u_prod  # (B, T) in {-1, +1}
            for t in range(logits.shape[0]):
                z_terms.append(z_prod[:, t])

        eval_prod_group(self.d2_logits)
        eval_prod_group(self.d3_logits)
        eval_prod_group(self.d4_logits)

        if z_terms and self.term_gate_logits is not None:
            g_mask = self._ste_prob(self.term_gate_logits, temp) if hard_ste else torch.sigmoid(self.term_gate_logits / temp)
            z_stack = torch.stack(z_terms, dim=-1)  # (B, T_total)
            u_stack = 0.5 * (1.0 - z_stack)
            z_gated = 1.0 - 2.0 * u_stack * g_mask.unsqueeze(0)
            z_out = z_lin * torch.prod(z_gated, dim=-1)
            reg = reg + g_mask.sum() * 0.02
        else:
            z_out = z_lin

        return z_out, reg


def differentiable_ste_search(
    spec: OperatorSpec,
    input_words: torch.Tensor,
    target_bits: torch.Tensor,
    input_packed: torch.Tensor,
    target_packed: torch.Tensor,
    n_samples: int,
    cand_vars: Sequence[int],
    max_and_budget: int = MAX_AND_GATES,
    max_deg: int = MAX_DEGREE,
    steps: int = 45,
    batch_size: int = 4096,
    seed: int = 777,
) -> Tuple[List[Tuple[float, CircuitSLP]], int]:
    """Method 3: Pure-FP32 Differentiable Logic Gate Network (allow_tf32=False), hard-discretized."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    torch.manual_seed(seed)
    device = input_words.device
    sub_vars = list(cand_vars[:32])
    n_sub = len(sub_vars)
    if n_sub < 2:
        return [], 0

    # Extract sub-input 0/1 float32 matrix ONLY for a small subset of samples to stay super fast
    use_n = min(n_samples, max(batch_size * 2, 8192))
    w = spec.word_bits
    x_cols = [((input_words[:use_n, v // w] >> (v % w)) & 1).to(torch.float32) for v in sub_vars]
    x_sub_all = torch.stack(x_cols, dim=1)  # (use_n, n_sub)
    y_spin_all = 1.0 - 2.0 * target_bits[:use_n].to(torch.float32)

    # Budget allocation within max_and_budget: e.g., 4 deg2 (4 AND), 2 deg3 (4 AND), 1 deg4 (3 AND) -> 11 AND <= 16
    n_d2 = min(4, max_and_budget)
    rem = max_and_budget - n_d2
    n_d3 = min(2, rem // 2) if max_deg >= 3 else 0
    rem -= 2 * n_d3
    n_d4 = min(1, rem // 3) if max_deg >= 4 else 0

    model = _STEBipolarProductNet(n_sub=n_sub, n_deg2=n_d2, n_deg3=n_d3, n_deg4=n_d4).to(device=device, dtype=torch.float32)
    # Initialize sparse active seeds so gradients are non-zero at step 0
    with torch.no_grad():
        corrs_init = torch.abs(torch.mean((1.0 - 2.0 * x_sub_all) * y_spin_all.unsqueeze(1), dim=0))
        top_seed = torch.topk(corrs_init, k=min(2, n_sub)).indices
        model.lin_logits.fill_(-1.2)
        model.lin_logits[top_seed] = 0.8
        if model.term_gate_logits is not None:
            model.term_gate_logits.fill_(-0.2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.18, weight_decay=1e-4)

    for step in range(steps):
        idx = torch.randint(0, use_n, (min(batch_size, use_n),), device=device)
        xb = x_sub_all[idx]
        yb = y_spin_all[idx]
        temp = max(0.35, 1.0 - 0.65 * (step / max(1, steps - 1)))
        hard_ste = step >= (steps // 2)
        opt.zero_grad()
        z_pred, reg = model(xb, temp=temp, hard_ste=hard_ste)
        corr = torch.mean(z_pred * yb)
        loss = -(corr * corr) + 0.01 * reg
        loss.backward()
        opt.step()

    # Hard discretize into exact CircuitSLP
    def logits_to_affine(vec: torch.Tensor, max_supp: int = 24) -> AffineForm:
        probs = torch.sigmoid(vec)
        active = torch.nonzero(probs > 0.5, as_tuple=False).flatten()
        if active.numel() > max_supp:
            active = torch.topk(probs, k=max_supp).indices
        terms = tuple(sub_vars[int(i.item())] for i in active)
        return AffineForm(0, terms)

    lin_af = logits_to_affine(model.lin_logits.detach())
    gates: List[AndGate] = []
    out_terms = list(lin_af.terms)
    term_cursor = 0
    gate_masks = (
        (torch.sigmoid(model.term_gate_logits.detach()) > 0.5).tolist()
        if model.term_gate_logits is not None
        else []
    )

    def extract_group(logits: Optional[torch.Tensor]) -> None:
        nonlocal term_cursor
        if logits is None:
            return
        d = logits.shape[1]
        for t in range(logits.shape[0]):
            is_on = gate_masks[term_cursor] if term_cursor < len(gate_masks) else True
            term_cursor += 1
            if not is_on:
                continue
            factors = [logits_to_affine(logits[t, k].detach(), max_supp=16) for k in range(d)]
            # Skip trivial empty factors
            if any(not f.terms for f in factors):
                continue
            if len(gates) + (d - 1) > max_and_budget:
                continue
            cur_node = -1
            g_first = AndGate(factors[0], factors[1])
            gates.append(g_first)
            cur_node = spec.n_inputs + len(gates) - 1
            for k in range(2, d):
                gates.append(AndGate(AffineForm(0, (cur_node,)), factors[k]))
                cur_node = spec.n_inputs + len(gates) - 1
            out_terms.append(cur_node)

    extract_group(model.d2_logits)
    extract_group(model.d3_logits)
    extract_group(model.d4_logits)

    results: List[Tuple[float, CircuitSLP]] = []
    trials = 0
    for c_bit in (0, 1):
        out_af = AffineForm(c_bit, tuple(out_terms[:MAX_SUPPORT]))
        slp = CircuitSLP(
            n_inputs=spec.n_inputs,
            gates=gates,
            output=out_af,
            name="m3_ste",
            method="differentiable_ste",
        ).prune_dead_gates()
        ok, _, _ = slp.validate(max_and=max_and_budget, max_degree=max_deg)
        if ok:
            pred = slp.eval_packed(input_packed)
            p = count_packed_matches(pred, target_packed) / float(n_samples)
            bias = abs(p - 0.5)
            # Flip const if needed
            if p < 0.5:
                slp = CircuitSLP(
                    n_inputs=spec.n_inputs,
                    gates=slp.gates,
                    output=AffineForm(slp.output.const ^ 1, slp.output.terms),
                    name="m3_ste",
                    method="differentiable_ste",
                )
            results.append((bias, slp))
            trials += 1

    # Also return linear-only discretized SLP
    lin_slp = CircuitSLP(
        n_inputs=spec.n_inputs,
        gates=[],
        output=lin_af,
        name="m3_ste_lin",
        method="differentiable_ste",
    )
    ok_l, _, _ = lin_slp.validate(max_and=max_and_budget, max_degree=max_deg)
    if ok_l:
        p = count_packed_matches(lin_slp.eval_packed(input_packed), target_packed) / float(n_samples)
        if p < 0.5:
            lin_slp = CircuitSLP(
                n_inputs=spec.n_inputs,
                gates=[],
                output=AffineForm(lin_af.const ^ 1, lin_af.terms),
                name="m3_ste_lin",
                method="differentiable_ste",
            )
        results.append((abs(p - 0.5), lin_slp))
        trials += 1

    return results, trials + steps


def run_combined_search(
    spec: OperatorSpec,
    target_name: str,
    input_words_search: torch.Tensor,
    target_bits_search: torch.Tensor,
    input_packed_search: torch.Tensor,
    target_packed_search: torch.Tensor,
    known_res: KnownBaselineResult,
    n_search: int,
    max_and_budget: int = MAX_AND_GATES,
    max_deg: int = MAX_DEGREE,
    evo_gens: int = 12,
    ste_steps: int = 45,
    seed: int = 42,
) -> SearchOutcome:
    """Run Methods 1, 2, 3 on the Search Split and return deduplicated top circuits + total trials."""
    cand_vars = _select_candidate_vars(
        spec, target_name, input_packed_search, target_packed_search, n_search, max_pool=48
    )

    m1_list, t1 = walsh_alphatensor_search(
        spec=spec,
        target_name=target_name,
        input_packed=input_packed_search,
        target_packed=target_packed_search,
        n_samples=n_search,
        cand_vars=cand_vars,
        known_res=known_res,
        max_and_budget=max_and_budget,
        max_deg=max_deg,
        seed=seed,
    )

    m3_list, t3 = differentiable_ste_search(
        spec=spec,
        input_words=input_words_search,
        target_bits=target_bits_search,
        input_packed=input_packed_search,
        target_packed=target_packed_search,
        n_samples=n_search,
        cand_vars=cand_vars,
        max_and_budget=max_and_budget,
        max_deg=max_deg,
        steps=ste_steps,
        seed=seed + 300,
    )

    seed_circuits = [c for _, c in (m1_list + m3_list)] + list(known_res.circuits[:4])
    m2_list, t2 = evolutionary_mcts_slp_search(
        spec=spec,
        input_packed=input_packed_search,
        target_packed=target_packed_search,
        n_samples=n_search,
        cand_vars=cand_vars,
        seed_circuits=seed_circuits,
        max_and_budget=max_and_budget,
        max_deg=max_deg,
        generations=evo_gens,
        seed=seed + 100,
    )

    all_cands = m1_list + m2_list + m3_list
    dedup: Dict[str, CircuitSLP] = {}
    for _b, c in all_cands:
        key = c.format_algebraic()
        if key not in dedup:
            dedup[key] = c

    cand_list = list(dedup.values())
    if not cand_list:
        fallback = CircuitSLP(
            n_inputs=spec.n_inputs,
            gates=[],
            output=AffineForm(0, ()),
            name="empty_fallback",
            method="fallback",
        )
        cand_list = [fallback]

    # Internal search-split generalization check: split search packed words into 2 halves
    # so overfitted circuits cannot outrank true generalizing circuits.
    b_words = input_packed_search.shape[1]
    half_w = max(1, b_words // 2)
    n_h1 = half_w * 64
    n_h2 = max(64, (b_words - half_w) * 64)
    h1_biases = _eval_candidates_bias(cand_list, input_packed_search[:, :half_w], target_packed_search[:half_w], n_h1)
    if b_words > half_w:
        h2_biases = _eval_candidates_bias(cand_list, input_packed_search[:, half_w:], target_packed_search[half_w:], n_h2)
        gen_scores = [min(b1, b2) for b1, b2 in zip(h1_biases, h2_biases)]
    else:
        gen_scores = h1_biases

    ranked = sorted(zip(gen_scores, cand_list), key=lambda x: x[0], reverse=True)

    method_best = {
        "walsh_alphatensor": max((b for b, _ in m1_list), default=0.0),
        "evolutionary_mcts_slp": max((b for b, _ in m2_list), default=0.0),
        "differentiable_ste": max((b for b, _ in m3_list), default=0.0),
    }
    top_circuits = [c for _, c in ranked[:5]]
    return SearchOutcome(
        best_circuit=top_circuits[0],
        top_circuits=top_circuits,
        n_trials=t1 + t2 + t3,
        method_best=method_best,
    )
