"""7-point Z3 BitVec(32) non-triviality, anti-cheating, and deduplication auditor (V1-V7)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import z3

from ast_spec import (
    MASK32,
    Const,
    CondTrue,
    ast_stats,
    canonical_fingerprint,
    collect_vars,
    compile_cond_z3,
    compile_expr_z3,
    pretty_cond,
    pretty_expr,
)
from sha_worlds import RoundsConfig, WorldSpec, eval_world_z3, make_z3_base_env


@dataclass
class CandidateProposition:
    track: str  # "A1", "A2", "B1", "B2"
    title: str
    form: str  # "FormI_Reconstruction" | "FormII_MaskingInvariance"
    config: RoundsConfig
    eliminated_vars: List[str]
    condition_ast: Dict[str, Any]
    mask_ast: Dict[str, Any]
    lhs_ast: Dict[str, Any]
    rhs_ast: Dict[str, Any]
    seed_note: str = ""


@dataclass
class AuditResult:
    passed: bool
    reject_reason: str = ""
    proposition: Optional[CandidateProposition] = None
    observable_vars: List[str] = field(default_factory=list)
    v1_unsat: bool = False
    v2_nondegenerate: bool = False
    v3_cond_necessary: bool = False
    v4_mask_necessary: bool = False
    v5_nontautology: bool = False
    v6_active_rounds: List[int] = field(default_factory=list)
    v6_active_span: int = 0
    v6_splice_necessary: bool = False
    z3_solve_ms: float = 0.0
    mask_popcount: int = 32
    ast_nodes: int = 0
    ast_depth: int = 0
    fingerprint: str = ""

    def to_identity_dict(self, identity_id: str) -> Dict[str, Any]:
        assert self.proposition is not None
        p = self.proposition
        cond_str = pretty_cond(p.condition_ast)
        mask_str = pretty_expr(p.mask_ast)
        lhs_str = pretty_expr(p.lhs_ast)
        rhs_str = pretty_expr(p.rhs_ast)
        if p.form == "FormI_Reconstruction":
            if p.mask_ast.get("op") == "Const" and p.mask_ast.get("val") == MASK32:
                eq_str = f"{lhs_str} == {rhs_str}"
            else:
                eq_str = f"({mask_str} & {lhs_str}) == ({mask_str} & {rhs_str})"
        else:
            eq_str = f"inv_under({p.eliminated_vars}): ({mask_str} & {lhs_str}) == ({mask_str} & {lhs_str}')"
        pretty = eq_str if cond_str == "True" else f"{cond_str} => {eq_str}"
        return {
            "id": identity_id,
            "track": p.track,
            "title": p.title,
            "form": p.form,
            "rounds_config": p.config.to_dict(),
            "observable_vars": self.observable_vars,
            "eliminated_vars": list(p.eliminated_vars),
            "condition_ast": p.condition_ast,
            "mask_ast": p.mask_ast,
            "lhs_ast": p.lhs_ast,
            "rhs_ast": p.rhs_ast,
            "pretty_formula": pretty,
            "ast_nodes": self.ast_nodes,
            "ast_depth": self.ast_depth,
            "fingerprint": self.fingerprint,
            "z3_audit": {
                "v1_unsat": self.v1_unsat,
                "v2_nondegenerate": self.v2_nondegenerate,
                "v3_cond_necessary": self.v3_cond_necessary,
                "v4_mask_necessary": self.v4_mask_necessary,
                "v5_nontautology": self.v5_nontautology,
                "v6_active_rounds": self.v6_active_rounds,
                "v6_active_span": self.v6_active_span,
                "v6_splice_necessary": self.v6_splice_necessary,
                "z3_solve_ms": round(self.z3_solve_ms, 2),
                "mask_popcount": self.mask_popcount,
            },
        }


def _check_solver(formula: z3.BoolRef, timeout_ms: int) -> z3.CheckSatResult:
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    s.add(formula)
    return s.check()


def _check_solver_model(formula: z3.BoolRef, timeout_ms: int) -> Tuple[z3.CheckSatResult, Optional[z3.ModelRef]]:
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    s.add(formula)
    res = s.check()
    if res == z3.sat:
        return res, s.model()
    return res, None


def _popcount32(val: int) -> int:
    return bin(int(val) & MASK32).count("1")


def _compute_active_span(cfg: RoundsConfig, active_rounds: List[int]) -> int:
    if not active_rounds:
        return 0
    if not cfg.btc_splice:
        return max(active_rounds) - min(active_rounds) + 1
    b1_act = [r for r in active_rounds if r < 100]
    b2_act = [r - 100 for r in active_rounds if r >= 100]
    if b1_act and b2_act:
        b1_span = cfg.block1_rounds - min(b1_act)
        b2_span = max(b2_act) - cfg.round_start + 1
        return b1_span + b2_span
    if b2_act:
        return max(b2_act) - min(b2_act) + 1
    return max(b1_act) - min(b1_act) + 1


class Z3Auditor:
    """Runs V1-V7 Z3 checks and semantic deduplication."""

    def __init__(self, timeout_ms: int = 12000, require_min_active_count: int = 5):
        self.timeout_ms = timeout_ms
        self.require_min_active_count = require_min_active_count
        self.seen_fingerprints: Set[str] = set()
        self.accepted_records: List[AuditResult] = []

    def _build_pair_constraints(
        self,
        prop: CandidateProposition,
        world_spec: WorldSpec,
        broken_rounds: Optional[Set[int]] = None,
        break_splice: bool = False,
        break_sched: bool = False,
    ) -> Tuple[z3.BoolRef, z3.BitVecRef, z3.BitVecRef, z3.BitVecRef, Dict[str, z3.BitVecRef]]:
        """Build (cond, masked_lhs, masked_rhs, raw_rhs, env1) for Form I or Form II."""
        base1 = make_z3_base_env(world_spec, suffix="")
        env1 = eval_world_z3(
            prop.config,
            base1,
            broken_rounds=broken_rounds,
            break_splice=break_splice,
            break_sched=break_sched,
            suffix="_w1",
        )
        cond1 = compile_cond_z3(prop.condition_ast, env1)
        mask1 = compile_expr_z3(prop.mask_ast, env1)
        lhs1 = compile_expr_z3(prop.lhs_ast, env1)

        if prop.form == "FormI_Reconstruction":
            rhs1 = compile_expr_z3(prop.rhs_ast, env1)
            return cond1, (mask1 & lhs1), (mask1 & rhs1), rhs1, env1

        # FormII_MaskingInvariance: perturb base vars in eliminated_vars
        elim_set = set(prop.eliminated_vars)
        base2: Dict[str, z3.BitVecRef] = {}
        for v, bv in base1.items():
            if v in elim_set:
                base2[v] = z3.BitVec(f"{v}_pert", 32)
            else:
                base2[v] = bv
        env2 = eval_world_z3(
            prop.config,
            base2,
            broken_rounds=broken_rounds,
            break_splice=break_splice,
            break_sched=break_sched,
            suffix="_w2",
        )
        cond2 = compile_cond_z3(prop.condition_ast, env2)
        mask2 = compile_expr_z3(prop.mask_ast, env2)
        lhs2 = compile_expr_z3(prop.lhs_ast, env2)
        full_cond = z3.And(cond1, cond2)
        return full_cond, (mask1 & lhs1), (mask2 & lhs2), lhs1, env1

    def audit(self, prop: CandidateProposition, world_spec: WorldSpec) -> AuditResult:
        t0 = time.perf_counter()

        # Syntactic checks
        if not prop.eliminated_vars:
            return AuditResult(passed=False, reject_reason="empty_eliminated_vars")

        lhs_vars = collect_vars(prop.lhs_ast)
        rhs_vars = collect_vars(prop.rhs_ast)
        cond_vars = collect_vars(prop.condition_ast)
        mask_vars = collect_vars(prop.mask_ast)
        elim_set = set(prop.eliminated_vars)

        if prop.form == "FormI_Reconstruction":
            if lhs_vars & elim_set:
                return AuditResult(passed=False, reject_reason="lhs_leaks_elim_vars")
            if cond_vars & elim_set:
                return AuditResult(passed=False, reject_reason="cond_leaks_elim_vars")
            if mask_vars & elim_set:
                return AuditResult(passed=False, reject_reason="mask_leaks_elim_vars")
            if rhs_vars & elim_set:
                return AuditResult(passed=False, reject_reason="rhs_contains_elim_vars")
            # LHS must not syntactically equal RHS and should observe downstream vs target
            if prop.lhs_ast == prop.rhs_ast:
                return AuditResult(passed=False, reject_reason="syntactic_lhs_eq_rhs")
            obs_vars = sorted(lhs_vars)
        else:
            # Form II: verify elim_vars are base vars (or at least valid env vars)
            missing = [v for v in elim_set if v not in world_spec.base_vars]
            if missing:
                return AuditResult(passed=False, reject_reason=f"form2_nonbase_elim:{missing}")
            obs_vars = sorted(lhs_vars)

        # Early exact-fingerprint deduplication before running heavy Z3 checks
        early_fp = canonical_fingerprint(
            prop.track, prop.form, prop.condition_ast, prop.mask_ast, prop.lhs_ast, prop.rhs_ast
        )
        if early_fp in self.seen_fingerprints:
            return AuditResult(passed=False, reject_reason="v7_duplicate_fingerprint")

        # V1: Unsatisfiability of negation
        cond, mlhs, mrhs, raw_rhs, env1 = self._build_pair_constraints(prop, world_spec)
        neg_check = _check_solver(z3.And(cond, mlhs != mrhs), self.timeout_ms)
        if neg_check != z3.unsat:
            return AuditResult(passed=False, reject_reason=f"v1_failed:{neg_check}")

        # V3 tightening: if condition is unnecessary, drop it to True and re-audit
        is_cond_true = prop.condition_ast.get("op") == "True"
        if not is_cond_true:
            v3_res = _check_solver(z3.And(z3.Not(cond), mlhs != mrhs), self.timeout_ms)
            if v3_res == z3.unsat:
                prop = CandidateProposition(
                    track=prop.track,
                    title=prop.title,
                    form=prop.form,
                    config=prop.config,
                    eliminated_vars=prop.eliminated_vars,
                    condition_ast=CondTrue(),
                    mask_ast=prop.mask_ast,
                    lhs_ast=prop.lhs_ast,
                    rhs_ast=prop.rhs_ast,
                    seed_note=prop.seed_note,
                )
                return self.audit(prop, world_spec)
            if v3_res != z3.sat:
                return AuditResult(passed=False, reject_reason=f"v3_failed:{v3_res}")

        # V4 tightening: if mask != 0xFFFFFFFF and unmasked still holds, drop mask to 0xFFFFFFFF
        is_full_mask = prop.mask_ast.get("op") == "Const" and prop.mask_ast.get("val") == MASK32
        if not is_full_mask:
            mask_bv = compile_expr_z3(prop.mask_ast, env1)
            if _check_solver(z3.And(cond, mask_bv != z3.BitVecVal(0, 32)), self.timeout_ms) != z3.sat:
                return AuditResult(passed=False, reject_reason="v4_zero_mask")
            # Check unmasked
            if prop.form == "FormI_Reconstruction":
                unmasked_lhs = compile_expr_z3(prop.lhs_ast, env1)
                unmasked_rhs = compile_expr_z3(prop.rhs_ast, env1)
                v4_res = _check_solver(z3.And(cond, unmasked_lhs != unmasked_rhs), self.timeout_ms)
            else:
                base1 = make_z3_base_env(world_spec, suffix="")
                env_u1 = eval_world_z3(prop.config, base1, suffix="_u1")
                base2 = {v: (z3.BitVec(f"{v}_upert", 32) if v in elim_set else bv) for v, bv in base1.items()}
                env_u2 = eval_world_z3(prop.config, base2, suffix="_u2")
                c_u = z3.And(compile_cond_z3(prop.condition_ast, env_u1), compile_cond_z3(prop.condition_ast, env_u2))
                v4_res = _check_solver(
                    z3.And(c_u, compile_expr_z3(prop.lhs_ast, env_u1) != compile_expr_z3(prop.lhs_ast, env_u2)),
                    self.timeout_ms,
                )
            if v4_res == z3.unsat:
                prop = CandidateProposition(
                    track=prop.track,
                    title=prop.title,
                    form=prop.form,
                    config=prop.config,
                    eliminated_vars=prop.eliminated_vars,
                    condition_ast=prop.condition_ast,
                    mask_ast=Const(MASK32),
                    lhs_ast=prop.lhs_ast,
                    rhs_ast=prop.rhs_ast,
                    seed_note=prop.seed_note,
                )
                return self.audit(prop, world_spec)
            if v4_res != z3.sat:
                return AuditResult(passed=False, reject_reason=f"v4_failed:{v4_res}")

        # V2: Condition SAT, target non-constant, and every elim_var non-constant
        c_sat, model1 = _check_solver_model(cond, self.timeout_ms)
        if c_sat != z3.sat or model1 is None:
            return AuditResult(passed=False, reject_reason=f"v2_cond_unsat:{c_sat}")

        target_expr_for_v2 = mlhs if prop.form == "FormII_MaskingInvariance" else (compile_expr_z3(prop.mask_ast, env1) & raw_rhs)
        val_t1 = model1.eval(target_expr_for_v2, model_completion=True)
        if _check_solver(z3.And(cond, target_expr_for_v2 != val_t1), self.timeout_ms) != z3.sat:
            return AuditResult(passed=False, reject_reason="v2_target_constant")

        for u_name in prop.eliminated_vars:
            if u_name not in env1:
                return AuditResult(passed=False, reject_reason=f"v2_missing_elim_var:{u_name}")
            u_bv = env1[u_name]
            u_val1 = model1.eval(u_bv, model_completion=True)
            if _check_solver(z3.And(cond, u_bv != u_val1), self.timeout_ms) != z3.sat:
                return AuditResult(passed=False, reject_reason=f"v2_elim_var_locked:{u_name}")

        # V5: Non-tautology (break all rounds and message schedule)
        all_rounds = set(world_spec.round_indices)
        c_brk, mlhs_brk, mrhs_brk, _, _ = self._build_pair_constraints(
            prop, world_spec, broken_rounds=all_rounds, break_splice=prop.config.btc_splice, break_sched=True
        )
        if _check_solver(z3.And(c_brk, mlhs_brk != mrhs_brk), self.timeout_ms) != z3.sat:
            return AuditResult(passed=False, reject_reason="v5_pure_tautology")

        # V6: Round-by-round ARX link-breaking test
        active_rounds: List[int] = []
        for r_id in world_spec.round_indices:
            c_r, mlhs_r, mrhs_r, _, _ = self._build_pair_constraints(
                prop, world_spec, broken_rounds={r_id}, break_splice=False, break_sched=False
            )
            if _check_solver(z3.And(c_r, mlhs_r != mrhs_r), self.timeout_ms) == z3.sat:
                active_rounds.append(r_id)

        active_span = _compute_active_span(prop.config, active_rounds)
        if active_span < 5:
            return AuditResult(
                passed=False,
                reject_reason=f"v6_active_span_too_small:{active_span}:rounds={active_rounds}",
            )
        if len(active_rounds) < self.require_min_active_count:
            return AuditResult(
                passed=False,
                reject_reason=f"v6_active_count_too_small:{len(active_rounds)}:rounds={active_rounds}",
            )

        splice_necessary = False
        if prop.config.btc_splice:
            c_sp, mlhs_sp, mrhs_sp, _, _ = self._build_pair_constraints(
                prop, world_spec, broken_rounds=set(), break_splice=True, break_sched=False
            )
            if _check_solver(z3.And(c_sp, mlhs_sp != mrhs_sp), self.timeout_ms) != z3.sat:
                return AuditResult(passed=False, reject_reason="v6_btc_splice_not_necessary")
            splice_necessary = True

        # V7: AST fingerprint + Z3 semantic equivalence deduplication
        fp = canonical_fingerprint(
            prop.track, prop.form, prop.condition_ast, prop.mask_ast, prop.lhs_ast, prop.rhs_ast
        )
        if fp in self.seen_fingerprints:
            return AuditResult(passed=False, reject_reason="v7_duplicate_fingerprint")

        lhs_nodes, lhs_depth = ast_stats(prop.lhs_ast)
        rhs_nodes, rhs_depth = ast_stats(prop.rhs_ast)
        cond_nodes, cond_depth = ast_stats(prop.condition_ast)
        total_nodes = lhs_nodes + rhs_nodes + cond_nodes
        max_depth = max(lhs_depth, rhs_depth, cond_depth)

        # Check semantic equivalence against existing accepted identities in same world/form
        for prev in self.accepted_records:
            assert prev.proposition is not None
            pp = prev.proposition
            if (
                pp.track == prop.track
                and pp.form == prop.form
                and pp.config.to_dict() == prop.config.to_dict()
                and pp.rhs_ast == prop.rhs_ast
                and pp.condition_ast == prop.condition_ast
                and pp.mask_ast == prop.mask_ast
            ):
                # Same rhs/cond/mask -> check if lhs is semantically identical under cond
                p_cond, p_mlhs, _, _, _ = self._build_pair_constraints(pp, world_spec)
                eq_check = _check_solver(z3.And(cond, p_cond, mlhs != p_mlhs), 4000)
                if eq_check == z3.unsat:
                    return AuditResult(passed=False, reject_reason="v7_semantic_duplicate")

        self.seen_fingerprints.add(fp)
        mask_pop = (
            _popcount32(prop.mask_ast["val"])
            if prop.mask_ast.get("op") == "Const"
            else 32
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        res = AuditResult(
            passed=True,
            proposition=prop,
            observable_vars=obs_vars,
            v1_unsat=True,
            v2_nondegenerate=True,
            v3_cond_necessary=True,
            v4_mask_necessary=True,
            v5_nontautology=True,
            v6_active_rounds=active_rounds,
            v6_active_span=active_span,
            v6_splice_necessary=splice_necessary,
            z3_solve_ms=dt_ms,
            mask_popcount=mask_pop,
            ast_nodes=total_nodes,
            ast_depth=max_depth,
            fingerprint=fp,
        )
        self.accepted_records.append(res)
        return res
