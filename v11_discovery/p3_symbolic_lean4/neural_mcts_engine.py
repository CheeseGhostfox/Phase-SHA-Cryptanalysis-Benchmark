"""Neural-guided MCTS + algebraic composition + constructive 512-sample filter for P3."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import z3

from ast_spec import (
    MASK32,
    Add,
    And,
    Ch,
    CondAnd,
    CondBitEq,
    CondEq,
    CondMaskEq,
    CondMaskZero,
    CondTrue,
    Const,
    Maj,
    Neg,
    Not,
    Or,
    Rotr,
    Shl,
    Shr,
    Sigma0,
    Sigma1,
    Sub,
    Var,
    Xor,
    ast_stats,
    ast_tokens,
    collect_vars,
    compile_cond_z3,
    eval_cond_np,
    eval_expr_np,
)
from sha_worlds import (
    SHA256_IV,
    SHA256_K,
    STATE_FIELDS,
    RoundsConfig,
    WorldSpec,
    build_world_spec,
    eval_world_np,
    eval_world_z3,
    make_z3_base_env,
)
from z3_auditor import AuditResult, CandidateProposition, Z3Auditor

# Force CPU-only execution for PyTorch
torch.set_num_threads(min(4, max(1, (torch.get_num_threads() or 2))))


# ---------------------------------------------------------------------------
# Algebraic Helper Builders (pure Expr32 ASTs)
# ---------------------------------------------------------------------------

def ast_T2(a: Dict[str, Any], b: Dict[str, Any], c: Dict[str, Any]) -> Dict[str, Any]:
    return Add(Sigma0(a), Maj(a, b, c))


def ast_backSolveA(
    e_next: Dict[str, Any],
    a_next: Dict[str, Any],
    a0: Dict[str, Any],
    a1: Dict[str, Any],
    a2: Dict[str, Any],
) -> Dict[str, Any]:
    # e_next - a_next + Sigma0(a0) + Maj(a0, a1, a2)
    return Add(Sub(e_next, a_next), ast_T2(a0, a1, a2))


def ast_inv_round_given_w(
    s_next: Dict[str, Dict[str, Any]], k_val: int, w_expr: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Invert 1 SHA-256 round given s_{t+1}, k_t, w_t -> s_t (DAG-minimal, no d0 duplication)."""
    a0 = s_next["b"]
    b0 = s_next["c"]
    c0 = s_next["d"]
    t2 = ast_T2(a0, b0, c0)
    d0 = Add(Sub(s_next["e"], s_next["a"]), t2)
    e0 = s_next["f"]
    f0 = s_next["g"]
    g0 = s_next["h"]
    t1 = Sub(s_next["a"], t2)
    h0 = Sub(
        Sub(Sub(Sub(t1, Sigma1(e0)), Ch(e0, f0, g0)), Const(k_val)),
        w_expr,
    )
    return {"a": a0, "b": b0, "c": c0, "d": d0, "e": e0, "f": f0, "g": g0, "h": h0}


def ast_state_vars(prefix: str) -> Dict[str, Dict[str, Any]]:
    return {f: Var(f"{prefix}.{f}") for f in STATE_FIELDS}


def ast_clamp4(s_expr: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    a4 = ast_backSolveA(s_expr["e"], s_expr["a"], s_expr["b"], s_expr["c"], s_expr["d"])
    a3 = ast_backSolveA(s_expr["f"], s_expr["b"], s_expr["c"], s_expr["d"], a4)
    a2 = ast_backSolveA(s_expr["g"], s_expr["c"], s_expr["d"], a4, a3)
    a1 = ast_backSolveA(s_expr["h"], s_expr["d"], a4, a3, a2)
    return a4, a3, a2, a1


def ast_five_round_coupling(s_end: Dict[str, Dict[str, Any]], k_last: int) -> Dict[str, Any]:
    """
    Exact 5-round identity on s_t -> s_{t+5}:
    f(s_{t+5}) = a_{t-3} + w_{t+4}.
    Uses a_{t+1}, a_t, a_{t-1}, a_{t-2} from 4 backSolveA steps plus round t+4 T1.
    """
    a_p1, a_0, a_m1, a_m2 = ast_clamp4(s_end)
    t1_last = Sub(s_end["a"], ast_T2(s_end["b"], s_end["c"], s_end["d"]))
    e_p1_plus_wlast = Sub(
        Sub(Sub(t1_last, Sigma1(s_end["f"])), Ch(s_end["f"], s_end["g"], s_end["h"])),
        Const(k_last),
    )
    return Add(Sub(e_p1_plus_wlast, a_p1), ast_T2(a_0, a_m1, a_m2))


def ast_t1_from_a(
    a_next: Dict[str, Any], a0: Dict[str, Any], a1: Dict[str, Any], a2: Dict[str, Any]
) -> Dict[str, Any]:
    return Sub(a_next, ast_T2(a0, a1, a2))


# ---------------------------------------------------------------------------
# Constructive + Z3 Hybrid Witness Sampler (N = 512)
# ---------------------------------------------------------------------------

def _collect_cond_atoms(cond: Dict[str, Any], out: List[Dict[str, Any]]) -> None:
    if cond["op"] == "AndCond":
        _collect_cond_atoms(cond["args"][0], out)
        _collect_cond_atoms(cond["args"][1], out)
    elif cond["op"] != "True":
        out.append(cond)


def sample_witnesses_np(
    cfg: RoundsConfig,
    spec: WorldSpec,
    cond_ast: Dict[str, Any],
    rng: np.random.Generator,
    n_samples: int = 512,
) -> Optional[Dict[str, np.ndarray]]:
    """Sample N diverse base_env -> full env inputs satisfying cond_ast."""
    base_vars = spec.base_vars
    base_set = set(base_vars)

    def _rand_base(bs: int) -> Dict[str, np.ndarray]:
        return {
            v: rng.integers(0, 1 << 32, size=(bs,), dtype=np.uint32)
            for v in base_vars
        }

    if cond_ast["op"] == "True":
        b = _rand_base(n_samples)
        return eval_world_np(cfg, b, n_samples)

    atoms: List[Dict[str, Any]] = []
    _collect_cond_atoms(cond_ast, atoms)

    # Check if all atoms are simple base-var constructive constraints
    constructive_ok = all(
        (
            a["op"] == "BitEq" and a["args"][0]["op"] == "Var" and a["args"][0]["name"] in base_set
        )
        or (
            a["op"] == "MaskZero"
            and a["args"][0]["op"] == "Var"
            and a["args"][0]["name"] in base_set
            and (
                a["args"][1]["op"] == "Const"
                or (a["args"][1]["op"] == "Var" and a["args"][1]["name"] in base_set)
            )
        )
        or (
            a["op"] == "Eq"
            and a["args"][0]["op"] == "Var"
            and a["args"][0]["name"] in base_set
            and (
                a["args"][1]["op"] == "Const"
                or (a["args"][1]["op"] == "Var" and a["args"][1]["name"] in base_set)
                or (
                    a["args"][1]["op"] == "Not"
                    and a["args"][1]["args"][0]["op"] == "Var"
                    and a["args"][1]["args"][0]["name"] in base_set
                )
            )
        )
        or (
            a["op"] == "MaskEq"
            and a["args"][0]["op"] == "Var"
            and a["args"][0]["name"] in base_set
            and a["args"][1]["op"] == "Var"
            and a["args"][1]["name"] in base_set
            and a["args"][2]["op"] == "Const"
        )
        for a in atoms
    )

    if constructive_ok:
        b = _rand_base(n_samples)
        for a in atoms:
            op = a["op"]
            if op == "BitEq":
                vn = a["args"][0]["name"]
                k = np.uint32(a["k"])
                bit = np.uint32(a["bit"])
                mask1 = np.uint32(1) << k
                if bit == 1:
                    b[vn] = b[vn] | mask1
                else:
                    b[vn] = b[vn] & (~mask1)
            elif op == "MaskZero":
                v0 = a["args"][0]["name"]
                if a["args"][1]["op"] == "Const":
                    m = np.uint32(a["args"][1]["val"])
                    b[v0] = b[v0] & (~m)
                else:
                    v1 = a["args"][1]["name"]
                    # Clear intersecting bits randomly on v0 or v1
                    both = b[v0] & b[v1]
                    flip = rng.integers(0, 2, size=(n_samples,), dtype=np.uint32)
                    b[v0] = b[v0] & (~(both & flip))
                    b[v1] = b[v1] & (~(both & (np.uint32(1) - flip)))
            elif op == "Eq":
                v0 = a["args"][0]["name"]
                rhs = a["args"][1]
                if rhs["op"] == "Const":
                    b[v0] = np.full((n_samples,), np.uint32(rhs["val"]), dtype=np.uint32)
                elif rhs["op"] == "Var":
                    b[v0] = b[rhs["name"]].copy()
                else:
                    b[v0] = (~b[rhs["args"][0]["name"]]).astype(np.uint32)
            elif op == "MaskEq":
                v0 = a["args"][0]["name"]
                v1 = a["args"][1]["name"]
                m = np.uint32(a["args"][2]["val"])
                b[v0] = (b[v0] & (~m)) | (b[v1] & m)
        env = eval_world_np(cfg, b, n_samples)
        mask_sat = eval_cond_np(cond_ast, env, n_samples)
        if np.all(mask_sat):
            return env

    # Rejection sampling fast try (for weak conditions like single BitEq on derived state)
    collected: List[Dict[str, np.ndarray]] = []
    got = 0
    for _ in range(8):
        bs = 1024
        b = _rand_base(bs)
        env = eval_world_np(cfg, b, bs)
        ok = eval_cond_np(cond_ast, env, bs)
        idx = np.nonzero(ok)[0]
        if len(idx) > 0:
            take = idx[: min(len(idx), n_samples - got)]
            collected.append({k: v[take] for k, v in env.items()})
            got += len(take)
            if got >= min(128, n_samples):
                break
    if got >= 32:
        merged = {k: np.concatenate([c[k] for c in collected], axis=0) for k in collected[0]}
        return merged

    # Z3 diverse witness sampler fallback (<= 150ms total)
    z3_base = make_z3_base_env(spec)
    z3_env = eval_world_z3(cfg, z3_base)
    z3_cond = compile_cond_z3(cond_ast, z3_env)
    solver = z3.Solver()
    solver.set("timeout", 150)
    solver.add(z3_cond)
    rows: Dict[str, List[int]] = {v: [] for v in base_vars}
    for _ in range(32):
        if solver.check() != z3.sat:
            break
        m = solver.model()
        block_terms = []
        for v in base_vars:
            val = m.eval(z3_base[v], model_completion=True).as_long()
            rows[v].append(val)
            block_terms.append(z3_base[v] != z3.BitVecVal(val, 32))
        solver.add(z3.Or(block_terms))
    if not rows[base_vars[0]]:
        return None
    actual_n = len(rows[base_vars[0]])
    b_arr = {v: np.array(rows[v], dtype=np.uint32) for v in base_vars}
    return eval_world_np(cfg, b_arr, actual_n)


# ---------------------------------------------------------------------------
# Fast Bitwise Filter & Automatic Mask Synthesis
# ---------------------------------------------------------------------------

def evaluate_candidate_fast(
    prop: CandidateProposition,
    spec: WorldSpec,
    rng: np.random.Generator,
    n_samples: int = 512,
) -> Tuple[bool, Optional[CandidateProposition], float, np.ndarray]:
    """
    Returns (passes_filter, possibly_masked_prop, reward_signal, bit_match_profile_32).
    """
    env = sample_witnesses_np(prop.config, spec, prop.condition_ast, rng, n_samples=n_samples)
    if env is None:
        return False, None, 0.0, np.zeros(32, dtype=np.float32)

    bs = len(next(iter(env.values())))
    if prop.form == "FormI_Reconstruction":
        lhs_val = eval_expr_np(prop.lhs_ast, env, bs)
        rhs_val = eval_expr_np(prop.rhs_ast, env, bs)
        diff = lhs_val ^ rhs_val
    else:
        # Form II: perturb eliminated base vars
        base_pert: Dict[str, np.ndarray] = {}
        elim_set = set(prop.eliminated_vars)
        for v in spec.base_vars:
            if v in elim_set:
                base_pert[v] = rng.integers(0, 1 << 32, size=(bs,), dtype=np.uint32)
            else:
                base_pert[v] = env[v]
        env2 = eval_world_np(prop.config, base_pert, bs)
        ok2 = eval_cond_np(prop.condition_ast, env2, bs)
        if not np.any(ok2):
            return False, None, 0.0, np.zeros(32, dtype=np.float32)
        e1 = eval_expr_np(prop.lhs_ast, env, bs)[ok2]
        e2 = eval_expr_np(prop.lhs_ast, env2, bs)[ok2]
        diff = e1 ^ e2
        if len(diff) < 16:
            return False, None, 0.0, np.zeros(32, dtype=np.float32)

    # Bit-level match profile
    shifts = np.arange(32, dtype=np.uint32)
    bit_eq = ((diff[:, None] >> shifts[None, :]) & np.uint32(1)) == np.uint32(0)
    bit_match = bit_eq.mean(axis=0).astype(np.float32)

    # Check if provided mask already matches 100%
    if not (prop.mask_ast.get("op") == "Const" and prop.mask_ast.get("val") == MASK32):
        if prop.form == "FormI_Reconstruction":
            m_val = eval_expr_np(prop.mask_ast, env, bs)
            if np.all((diff & m_val) == 0):
                return True, prop, 1.0, bit_match
        else:
            m1 = eval_expr_np(prop.mask_ast, env, bs)[ok2]
            m2 = eval_expr_np(prop.mask_ast, env2, bs)[ok2]
            if np.all((e1 & m1) == (e2 & m2)):
                return True, prop, 1.0, bit_match

    # Full 32-bit exact match
    if np.all(diff == 0):
        out_prop = CandidateProposition(
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
        return True, out_prop, 1.0, bit_match

    # Automatic exact mask synthesis: bit_match == 1.0
    exact_bits = [k for k in range(32) if bit_match[k] == 1.0]
    if exact_bits:
        synth_mask = sum(1 << k for k in exact_bits)
        out_prop = CandidateProposition(
            track=prop.track,
            title=f"{prop.title} [Mask {len(exact_bits)}b]",
            form=prop.form,
            config=prop.config,
            eliminated_vars=prop.eliminated_vars,
            condition_ast=prop.condition_ast,
            mask_ast=Const(synth_mask),
            lhs_ast=prop.lhs_ast,
            rhs_ast=prop.rhs_ast,
            seed_note=prop.seed_note,
        )
        return True, out_prop, 0.95, bit_match

    max_b = float(np.max(bit_match))
    if max_b >= 0.5:
        return False, None, 0.1 + 0.8 * max_b, bit_match
    return False, None, 0.05, bit_match


# ---------------------------------------------------------------------------
# Neural Policy-Value Network (CPU FP32 PyTorch)
# ---------------------------------------------------------------------------

TOKEN_VOCAB = [
    "PAD", "Var:s0", "Var:s1", "Var:s2", "Var:s3", "Var:s4", "Var:s5", "Var:s6", "Var:s7",
    "Var:s8", "Var:s9", "Var:s10", "Var:w0", "Var:w1", "Var:b1_s0", "Var:b2_s5", "Var:b2_s6",
    "Var:b2_s7", "Var:b2_s8", "Var:b2_s9", "Var:b2_s10", "Const", "Add", "Sub", "Neg",
    "And", "Or", "Xor", "Not", "Rotr", "Shr", "Shl", "Sigma0", "Sigma1", "sigma0", "sigma1",
    "Ch", "Maj", "True", "Eq", "MaskZero", "MaskEq", "BitEq", "AndCond",
]
TOKEN_TO_IDX = {tok: i for i, tok in enumerate(TOKEN_VOCAB)}

TRACK_TO_IDX = {"A1": 0, "A2": 1, "B1": 2, "B2": 3}

ACTION_NAMES = [
    "bridge_5r_coupling",
    "peel_inv_round_known_w",
    "peel_backsolve_chain",
    "sched_elim_w0",
    "sched_elim_w8",
    "sched_elim_w14",
    "btc_splice_recover_a",
    "btc_splice_recover_e_deep",
    "btc_b2_sched_collapse",
    "attach_maskzero_cond",
    "attach_biteq_cond",
    "attach_ch_lock_cond",
    "form2_noninterference",
    "mutate_xor_add_sub",
    "mutate_shift_rotr",
]


def encode_search_state(
    track: str,
    num_rounds: int,
    lhs_ast: Dict[str, Any],
    cond_ast: Dict[str, Any],
    bit_match: np.ndarray,
) -> np.ndarray:
    feat = np.zeros((4 + 1 + len(TOKEN_VOCAB) + 34,), dtype=np.float32)
    feat[TRACK_TO_IDX.get(track, 0)] = 1.0
    feat[4] = min(num_rounds, 16) / 16.0
    off = 5
    for tok in ast_tokens(lhs_ast, 48) + ast_tokens(cond_ast, 16):
        idx = TOKEN_TO_IDX.get(tok, 0)
        feat[off + idx] += 0.1
    boff = off + len(TOKEN_VOCAB)
    feat[boff : boff + 32] = bit_match
    feat[boff + 32] = float(np.max(bit_match))
    feat[boff + 33] = float(np.mean(bit_match))
    return feat


class PolicyValueNet(nn.Module):
    def __init__(self, in_dim: int = 5 + len(TOKEN_VOCAB) + 34, n_actions: int = len(ACTION_NAMES)):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Search State & MCTS Node
# ---------------------------------------------------------------------------

@dataclass
class ConjectureState:
    track: str
    config: RoundsConfig
    spec: WorldSpec
    form: str
    lhs_ast: Dict[str, Any]
    rhs_ast: Dict[str, Any]
    cond_ast: Dict[str, Any]
    mask_ast: Dict[str, Any]
    eliminated_vars: List[str]
    title: str
    depth: int = 0
    bit_match: np.ndarray = field(default_factory=lambda: np.zeros(32, dtype=np.float32))


class MCTSNode:
    def __init__(self, state: ConjectureState, prior: float = 1.0):
        self.state = state
        self.prior = prior
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.children: Dict[int, MCTSNode] = {}

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


# ---------------------------------------------------------------------------
# Structured + Evolutionary Conjecture Generators (Action Implementations)
# ---------------------------------------------------------------------------

def _ast_mutate_expr(node: Dict[str, Any], obs_vars: List[str], rng: random.Random) -> Dict[str, Any]:
    op = node["op"]
    args = node.get("args", [])
    if not args or rng.random() < 0.25:
        choice = rng.random()
        if choice < 0.35 and obs_vars:
            return Var(rng.choice(obs_vars))
        if choice < 0.65 and args:
            a = args[0]
            mut_op = rng.choice(["Add", "Sub", "Xor", "And"])
            b = Var(rng.choice(obs_vars)) if obs_vars else Const(0)
            return {"op": mut_op, "args": [a, b]}
        if args and len(args) == 1:
            return Rotr(args[0], rng.choice([1, 2, 6, 7, 11, 13, 17, 18, 19, 22, 25]))
        return node
    new_args = [
        _ast_mutate_expr(a, obs_vars, rng) if i == rng.randrange(len(args)) else a
        for i, a in enumerate(args)
    ]
    if op in ("Add", "Sub", "Xor") and rng.random() < 0.3:
        op = rng.choice(["Add", "Sub", "Xor"])
    out = dict(node)
    out["op"] = op
    out["args"] = new_args
    return out


def build_seed_states(rng: random.Random) -> List[ConjectureState]:
    """Build diverse initial root ConjectureStates across Tracks A1, A2, B1, B2."""
    seeds: List[ConjectureState] = []

    # Track A1: 5..10 rounds general ARX
    for nr in (5, 6, 7, 8, 9, 10):
        cfg = RoundsConfig(track="A1", round_start=0, num_rounds=nr)
        spec = build_world_spec(cfg)
        s_end = ast_state_vars(f"s{nr}")
        # 5-round core on final 5 rounds -> shift with inv_round if nr > 5
        lhs = ast_five_round_coupling(s_end, SHA256_K[(nr - 1) & 63])
        rhs = Add(Var("s0.d"), Var(f"w{nr - 1}")) if nr == 5 else Var(f"w{nr - 1}")
        elim = [f"w{i}" for i in range(nr - 1)] + ["s0.e", "s0.f", "s0.g", "s0.h"]
        if nr > 5:
            lhs = ast_backSolveA(s_end["e"], s_end["a"], s_end["b"], s_end["c"], s_end["d"])
            rhs = Var("s0.d")
        seeds.append(
            ConjectureState(
                track="A1",
                config=cfg,
                spec=spec,
                form="FormI_Reconstruction",
                lhs_ast=lhs,
                rhs_ast=rhs,
                cond_ast=CondTrue(),
                mask_ast=Const(MASK32),
                eliminated_vars=elim,
                title=f"Track A1 {nr}-round seed",
            )
        )

    # Track A2: message-scheduled windows & padded SHA-256
    for free_w in (1, 2, 4):
        for nr in (5, 6, 7, 8, 9, 10):
            cfg = RoundsConfig(
                track="A2",
                round_start=0,
                num_rounds=nr,
                use_msg_schedule=True,
                padded_mode=True,
                padded_free_words=free_w,
            )
            spec = build_world_spec(cfg)
            s_end = ast_state_vars(f"s{nr}")
            lhs = ast_backSolveA(s_end["e"], s_end["a"], s_end["b"], s_end["c"], s_end["d"])
            seeds.append(
                ConjectureState(
                    track="A2",
                    config=cfg,
                    spec=spec,
                    form="FormI_Reconstruction",
                    lhs_ast=lhs,
                    rhs_ast=Var("s0.d"),
                    cond_ast=CondTrue(),
                    mask_ast=Const(MASK32),
                    eliminated_vars=["s0.e", "s0.f", "s0.g", "s0.h"] + [f"w{i}" for i in range(free_w)],
                    title=f"Track A2 padded(free={free_w}) {nr}-round seed",
                )
            )

    # Track A2 unpadded schedule windows (t=16..21, t=16..22, t=16..23, t=12..19)
    for r_start, nr in ((16, 6), (16, 7), (16, 8), (15, 7), (14, 8)):
        cfg = RoundsConfig(
            track="A2",
            round_start=r_start,
            num_rounds=nr,
            use_msg_schedule=True,
            padded_mode=False,
        )
        spec = build_world_spec(cfg)
        s_end = ast_state_vars(f"s{nr}")
        lhs = ast_backSolveA(s_end["e"], s_end["a"], s_end["b"], s_end["c"], s_end["d"])
        seeds.append(
            ConjectureState(
                track="A2",
                config=cfg,
                spec=spec,
                form="FormI_Reconstruction",
                lhs_ast=lhs,
                rhs_ast=Var("w0"),
                cond_ast=CondTrue(),
                mask_ast=Const(MASK32),
                eliminated_vars=["w0"],
                title=f"Track A2 sched[{r_start}..{r_start+nr-1}] seed",
            )
        )

    # Track B1: BTC Double SHA-256 splice (Block 1 r1 rounds -> Block 2 r2 rounds)
    for r1, r2 in ((0, 5), (0, 6), (0, 7), (0, 8), (1, 5), (1, 6), (2, 5), (2, 6), (4, 8)):
        cfg = RoundsConfig(
            track="B1",
            round_start=0,
            num_rounds=r1 + r2,
            use_msg_schedule=True,
            btc_splice=True,
            block1_rounds=r1,
            block2_rounds=r2,
        )
        spec = build_world_spec(cfg)
        s_b2 = ast_state_vars(f"b2_s{r2}")
        lhs = ast_backSolveA(s_b2["e"], s_b2["a"], s_b2["b"], s_b2["c"], s_b2["d"])
        elim = [f"b1_w{i}" for i in range(r1)] + ["b1_s0.e", "b1_s0.f", "b1_s0.g", "b1_s0.h"]
        seeds.append(
            ConjectureState(
                track="B1",
                config=cfg,
                spec=spec,
                form="FormI_Reconstruction",
                lhs_ast=lhs,
                rhs_ast=Var("b1_s0.d"),
                cond_ast=CondTrue(),
                mask_ast=Const(MASK32),
                eliminated_vars=elim,
                title=f"Track B1 BTC splice (r1={r1}, r2={r2}) seed",
            )
        )

    # Track B2: BTC Block 2 message expansion collapse (t >= 16)
    for r_start, r2 in ((12, 6), (12, 7), (13, 6), (14, 6), (15, 6)):
        cfg = RoundsConfig(
            track="B2",
            round_start=r_start,
            num_rounds=r2,
            use_msg_schedule=True,
            btc_splice=True,
            block1_rounds=0,
            block2_rounds=r2,
        )
        spec = build_world_spec(cfg)
        s_b2 = ast_state_vars(f"b2_s{r_start + r2}")
        lhs = ast_backSolveA(s_b2["e"], s_b2["a"], s_b2["b"], s_b2["c"], s_b2["d"])
        seeds.append(
            ConjectureState(
                track="B2",
                config=cfg,
                spec=spec,
                form="FormI_Reconstruction",
                lhs_ast=lhs,
                rhs_ast=Var("b1_s0.a"),
                cond_ast=CondTrue(),
                mask_ast=Const(MASK32),
                eliminated_vars=["b1_s0.e", "b1_s0.f", "b1_s0.g", "b1_s0.h"],
                title=f"Track B2 Block2 sched[{r_start}..{r_start+r2-1}] seed",
            )
        )

    return seeds


def apply_mcts_action(
    state: ConjectureState, action_idx: int, rng: random.Random
) -> Optional[ConjectureState]:
    act = ACTION_NAMES[action_idx]
    cfg = state.config
    spec = state.spec

    # 1. Exact 5-round bidirectional ARX coupling (Track A1 & A2)
    if act == "bridge_5r_coupling" and not cfg.btc_splice and cfg.num_rounds >= 5:
        nr = cfg.num_rounds
        s_end = ast_state_vars(f"s{nr}")
        if nr == 5:
            lhs = ast_five_round_coupling(s_end, SHA256_K[(cfg.round_start + 4) & 63])
            rhs = Add(Var("s0.d"), Var(f"w{cfg.round_start + 4}"))
            elim = [f"w{cfg.round_start + i}" for i in range(4)] + ["s0.e", "s0.f", "s0.g", "s0.h"]
            title = f"{cfg.track} 5-round bidirectional A_{{t-3}}+W_{{t+4}} coupling (elim 4W+4E)"
            # Variant: recover w4 from (S_5, s0.d)
            if rng.random() < 0.5:
                lhs = Sub(lhs, Var("s0.d"))
                rhs = Var(f"w{cfg.round_start + 4}")
                title = f"{cfg.track} 5-round W_{{t+4}} recovery from (S_5, A_{{t-3}}) (elim 4W+4E)"
        else:
            # nr >= 6: if padded_mode, peel down from nr to 5 using known zero/const W!
            if cfg.padded_mode and cfg.round_start == 0 and cfg.padded_free_words < nr:
                cur_s = s_end
                # Peel rounds nr-1 down to 5 (if padded words are constant)
                ok_peel = True
                for r_idx in range(nr - 1, 4, -1):
                    if r_idx < cfg.padded_free_words:
                        ok_peel = False
                        break
                    if r_idx == cfg.padded_free_words:
                        w_expr = Const(0x80000000)
                    elif r_idx == 15:
                        w_expr = Const(cfg.padded_free_words * 32)
                    elif r_idx < 15:
                        w_expr = Const(0)
                    else:
                        ok_peel = False
                        break
                    cur_s = ast_inv_round_given_w(cur_s, SHA256_K[r_idx & 63], w_expr)
                if not ok_peel:
                    return None
                lhs = ast_five_round_coupling(cur_s, SHA256_K[4])
                if cfg.padded_free_words <= 4:
                    w4_c = Const(0x80000000 if cfg.padded_free_words == 4 else 0)
                    lhs = Sub(lhs, w4_c)
                    rhs = Var("s0.d")
                    elim = [f"w{i}" for i in range(cfg.padded_free_words)] + ["s0.e", "s0.f", "s0.g", "s0.h"]
                    title = f"Track A2 {nr}-round padded A_{{-3}} reconstruction (elim W_{{0..{cfg.padded_free_words-1}}}+4E)"
                else:
                    rhs = Add(Var("s0.d"), Var("w4"))
                    elim = ["w0", "w1", "w2", "w3", "s0.e", "s0.f", "s0.g", "s0.h"]
                    title = f"Track A2 {nr}-round padded A_{{-3}}+W_4 reconstruction"
            else:
                return None
        return ConjectureState(
            track=cfg.track,
            config=cfg,
            spec=spec,
            form="FormI_Reconstruction",
            lhs_ast=lhs,
            rhs_ast=rhs,
            cond_ast=state.cond_ast,
            mask_ast=Const(MASK32),
            eliminated_vars=elim,
            title=title,
            depth=state.depth + 1,
        )

    # 2. Padded message recovery (recover w0..w_{k-1} or s0.d across 5..10 rounds)
    if act == "peel_inv_round_known_w" and cfg.padded_mode and not cfg.btc_splice and cfg.round_start == 0:
        nr = cfg.num_rounds
        fw = cfg.padded_free_words
        if nr < 5 or fw > nr - 4 or nr > 15:
            return None
        cur_s = ast_state_vars(f"s{nr}")
        # Peel down to round fw (where w_0..w_{fw-1} are unknown, w_{fw..nr-1} are constants)
        for r_idx in range(nr - 1, fw - 1, -1):
            w_val = 0x80000000 if r_idx == fw else (fw * 32 if r_idx == 15 else 0)
            cur_s = ast_inv_round_given_w(cur_s, SHA256_K[r_idx & 63], Const(w_val))
        # Now cur_s is s_{fw}. Run 4-round clamp if fw >= 4, or if fw <= 4 use 5-round coupling or direct forward
        if fw == 1 and nr >= 5:
            # From s_1 = inv(s_nr), we know s_1.a..h!
            # Wait: s_1.a = T1(0) + T2(0), s_1.e = s0.d + T1(0)!
            # Actually backSolveA on s_1 gives s0.d WITHOUT using w0!
            # Better: peel down to s_0? No, w0 is unknown!
            # Peel down to s_1 gives backSolveA(s_1) = s0.d (uses rounds 0..nr-1, eliminates w0, s0.e..h? Wait: round 0 is used in s_1.a/e!)
            lhs = ast_backSolveA(cur_s["e"], cur_s["a"], cur_s["b"], cur_s["c"], cur_s["d"])
            rhs = Var("s0.d")
            elim = ["w0", "s0.e", "s0.f", "s0.g", "s0.h"]
            title = f"Track A2 {nr}-round 1-word padded A_{{-3}} invariant (elim w0+E_0..-3)"
            if rng.random() < 0.5:
                # Or recover w0 from cur_s.e - s0.d when s0.{e,f,g,h} are known!
                t1_0 = Sub(cur_s["a"], ast_T2(Var("s0.a"), Var("s0.b"), Var("s0.c")))
                lhs = Sub(Sub(Sub(Sub(t1_0, Var("s0.h")), Sigma1(Var("s0.e"))), Ch(Var("s0.e"), Var("s0.f"), Var("s0.g"))), Const(SHA256_K[0]))
                rhs = Var("w0")
                elim = ["s0.d"]  # s0.d eliminated? wait: s0.d not used in t1_0!
                # Check if s0.d is active: in cur_s, s_1.a doesn't depend on s0.d, but inverting rounds 1..nr-1 uses cur_s!
                # Wait: better eliminate w0 via s0.d, or keep elim=["w0"] for s0.d recovery
                return None
            return ConjectureState(
                track="A2",
                config=cfg,
                spec=spec,
                form="FormI_Reconstruction",
                lhs_ast=lhs,
                rhs_ast=rhs,
                cond_ast=CondTrue(),
                mask_ast=Const(MASK32),
                eliminated_vars=elim,
                title=title,
                depth=state.depth + 1,
            )
        if fw == 2 and nr >= 6:
            # Peel down to s_2 -> cur_s is s_2.
            # From s_2, backSolveA gives s_1.d = s0.c!
            # Better: peel down to s_2, wait: s0.c only uses rounds 1..nr-1 (skipping round 0)!
            # To use round 0 too, recover s0.d if nr >= 6 using 5-round coupling on s_5 if fw <= 4!
            return None

    # 3. Track A2 unpadded message schedule: eliminate w0 across t=16..21 (6 rounds) or t=16..22
    if act == "sched_elim_w0" and cfg.track == "A2" and not cfg.padded_mode and cfg.round_start == 16 and cfg.num_rounds == 6:
        # World: t=16..21 (s0 is S_16, s6 is S_22). Unknown: w0 (which enters w16, w18, w20!).
        # Known observables: s6.{a..h}, s0.{a..h}, w1..w15.
        # Notice w17, w19, w21 do NOT depend on w0!
        w17 = Add(Add(Add(Sigma1_placeholder := {"op": "sigma1", "args": [Var("w15")]}, Var("w10")), {"op": "sigma0", "args": [Var("w2")]}), Var("w1"))
        w19 = Add(Add(Add({"op": "sigma1", "args": [w17]}, Var("w12")), {"op": "sigma0", "args": [Var("w4")]}), Var("w3"))
        w21 = Add(Add(Add({"op": "sigma1", "args": [w19]}, Var("w14")), {"op": "sigma0", "args": [Var("w6")]}), Var("w5"))
        s22 = ast_state_vars("s6")
        s21 = ast_inv_round_given_w(s22, SHA256_K[21], w21)
        # backSolveA on s21 gives A_17 = s21.d's predecessor? Wait: s21 = S_21, backSolveA(s21) gives A_17 = s1.a!
        a17 = ast_backSolveA(s21["e"], s21["a"], s21["b"], s21["c"], s21["d"])
        # Also from s0 (= S_16), T1(16) = A_17 - T2(s0.a, s0.b, s0.c) -> W_16!
        t1_16 = Sub(a17, ast_T2(Var("s0.a"), Var("s0.b"), Var("s0.c")))
        w16_rec = Sub(Sub(Sub(Sub(t1_16, Var("s0.h")), Sigma1(Var("s0.e"))), Ch(Var("s0.e"), Var("s0.f"), Var("s0.g"))), Const(SHA256_K[16]))
        # Now compute w18 and w20 from w16_rec!
        w18_rec = Add(Add(Add({"op": "sigma1", "args": [w16_rec]}, Var("w11")), {"op": "sigma0", "args": [Var("w3")]}), Var("w2"))
        w20_rec = Add(Add(Add({"op": "sigma1", "args": [w18_rec]}, Var("w13")), {"op": "sigma0", "args": [Var("w5")]}), Var("w4"))
        # Peel s21 -> s20 -> s19 -> s18 -> s17!
        s20 = ast_inv_round_given_w(s21, SHA256_K[20], w20_rec)
        s19 = ast_inv_round_given_w(s20, SHA256_K[19], w19)
        s18 = ast_inv_round_given_w(s19, SHA256_K[18], w18_rec)
        s17 = ast_inv_round_given_w(s18, SHA256_K[17], w17)
        # Check E_17: s17["e"] == s0.d + t1_16! Or recover w0 from w16_rec!
        if rng.random() < 0.5:
            # Check consistency of E_17 across all 6 rounds 16..21 while eliminating w0 (and w16,w18,w20)!
            lhs = Sub(s17["e"], t1_16)
            rhs = Var("s0.d")
            title = "Track A2 6-round (t=16..21) A_13 consistency eliminating W_0 (W_16,W_18,W_20)"
            elim = ["w0", "w16", "w18", "w20"]
        else:
            # Recover W_0 directly from S_22, S_16, W_1..15 across rounds 16,20,21 -> wait: use s17["e"] to force rounds 17,18,19 active too!
            # w0 = w16_rec - sigma1(w14) - w9 - sigma0(w1) + (s17["e"] - (s0.d + t1_16))
            delta_e17 = Sub(s17["e"], Add(Var("s0.d"), t1_16))
            w0_rec = Sub(Sub(Sub(w16_rec, {"op": "sigma1", "args": [Var("w14")]}), Var("w9")), {"op": "sigma0", "args": [Var("w1")]})
            lhs = Add(w0_rec, delta_e17)
            rhs = Var("w0")
            title = "Track A2 6-round (t=16..21) W_0 reconstruction via W_16..21 cascade (elim W_16,W_18,W_20)"
            elim = ["w16", "w18", "w20"]
        return ConjectureState(
            track="A2",
            config=cfg,
            spec=spec,
            form="FormI_Reconstruction",
            lhs_ast=lhs,
            rhs_ast=rhs,
            cond_ast=CondTrue(),
            mask_ast=Const(MASK32),
            eliminated_vars=elim,
            title=title,
            depth=state.depth + 1,
        )

    # 4. Track A2 unpadded schedule: eliminate w8 (or w14) in t=16..22 / t=15..21
    if act == "sched_elim_w8" and cfg.track == "A2" and not cfg.padded_mode and cfg.round_start == 16 and cfg.num_rounds == 8:
        # In t=16..23: w23 = sigma1(w21) + w16 + sigma0(w8) + w7!
        # Wait: w16..w22 are independent of w8! Only w23 uses w8!
        # What about t=15..21 (7 rounds)? w8 is in rounds 15..21? No, w8 is NOT in rounds 15..21 at all!
        # What about w14? w14 enters w16 (t=16), w18 (t=18), w20 (t=20), w21 (t=21), w22 (t=22)!
        return None

    if act == "sched_elim_w14" and cfg.track == "A2" and not cfg.padded_mode and cfg.round_start == 16 and cfg.num_rounds == 6:
        # w14 enters w16 (= sigma1(w14) + w9 + sigma0(w1) + w0) and w21 (= sigma1(w19) + w14 + sigma0(w6) + w5)!
        # Wait: w17, w18?, w19, w20? No, w18 uses w16 (has w14), w20 uses w18 (has w14).
        # Only w17 and w19 are independent of w14!
        # What about w15? w15 enters w17 (sigma1(w15)+...), w19, w21, w22!
        # So w16, w18, w20 are INDEPENDENT of w15!
        w16 = Add(Add(Add({"op": "sigma1", "args": [Var("w14")]}, Var("w9")), {"op": "sigma0", "args": [Var("w1")]}), Var("w0"))
        w18 = Add(Add(Add({"op": "sigma1", "args": [w16]}, Var("w11")), {"op": "sigma0", "args": [Var("w3")]}), Var("w2"))
        w20 = Add(Add(Add({"op": "sigma1", "args": [w18]}, Var("w13")), {"op": "sigma0", "args": [Var("w5")]}), Var("w4"))
        # Wait: from s6 (= S_22), without w21 we can't invert round 21 directly unless we use backSolveA(s6) -> A_18!
        # A_18 = backSolveA(s6)! Then T1(17) = A_18 - T2(S_17)...
        # Wait: from s0 (= S_16) and w16 (known!), we step FORWARD 1 round to S_17!
        # Round 16 forward (using w16, NO w15!):
        s0 = ast_state_vars("s0")
        t1_16 = Add(Add(Add(Add(s0["h"], Sigma1(s0["e"])), Ch(s0["e"], s0["f"], s0["g"])), Const(SHA256_K[16])), w16)
        s17_a = Add(t1_16, ast_T2(s0["a"], s0["b"], s0["c"]))
        s17_e = Add(s0["d"], t1_16)
        # From s6 (= S_22), backSolveA gives A_18 = backSolveA(s6.e, s6.a, s6.b, s6.c, s6.d)!
        s22 = ast_state_vars("s6")
        a18 = ast_backSolveA(s22["e"], s22["a"], s22["b"], s22["c"], s22["d"])
        # From S_17 and A_18 -> T1(17) = A_18 - T2(s17.a, s0.a, s0.b) -> W_17!
        t1_17 = Sub(a18, ast_T2(s17_a, s0["a"], s0["b"]))
        w17_rec = Sub(Sub(Sub(Sub(t1_17, s0["g"]), Sigma1(s17_e)), Ch(s17_e, s0["e"], s0["f"])), Const(SHA256_K[17]))
        # Now w19 and w21 depend on w17_rec and w15, wait: sigma1(w15) = w17_rec - w10 - sigma0(w2) - w1!
        # Better: check consistency of s22 using w18, w19(from w17_rec), w20, w21(from w19):
        # Wait: w17_rec ONLY used rounds 16, 17, 21! Rounds 18, 19, 20 wouldn't be active unless we propagate s17 -> s18 -> s19 -> s20 -> s21 -> s22!
        # Let's propagate forward from s17 with w17_rec!
        s17 = {"a": s17_a, "b": s0["a"], "c": s0["b"], "d": s0["c"], "e": s17_e, "f": s0["e"], "g": s0["f"], "h": s0["g"]}
        def _fwd(st: Dict[str, Dict[str, Any]], kv: int, wv: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
            tt1 = Add(Add(Add(Add(st["h"], Sigma1(st["e"])), Ch(st["e"], st["f"], st["g"])), Const(kv)), wv)
            tt2 = ast_T2(st["a"], st["b"], st["c"])
            return {"a": Add(tt1, tt2), "b": st["a"], "c": st["b"], "d": st["c"], "e": Add(st["d"], tt1), "f": st["e"], "g": st["f"], "h": st["g"]}
        s18 = _fwd(s17, SHA256_K[17], w17_rec)
        s19 = _fwd(s18, SHA256_K[18], w18)
        w19_rec = Add(Add(Add({"op": "sigma1", "args": [w17_rec]}, Var("w12")), {"op": "sigma0", "args": [Var("w4")]}), Var("w3"))
        s20 = _fwd(s19, SHA256_K[19], w19_rec)
        s21 = _fwd(s20, SHA256_K[20], w20)
        # Now compare s21.d (= A_18) or s22.b (= A_21) with s22!
        # Wait: w21 depends on sigma1(w19_rec) + w14 + sigma0(w6) + w5! Does w21 depend on w15?! NO!! Because w15 ONLY entered w17 (which we recovered as w17_rec)!
        w21_rec = Add(Add(Add({"op": "sigma1", "args": [w19_rec]}, Var("w14")), {"op": "sigma0", "args": [Var("w6")]}), Var("w5"))
        s22_pred = _fwd(s21, SHA256_K[21], w21_rec)
        lhs = s22_pred["a"]
        rhs = Var("s6.a")
        return ConjectureState(
            track="A2",
            config=cfg,
            spec=spec,
            form="FormI_Reconstruction",
            lhs_ast=lhs,
            rhs_ast=rhs,
            cond_ast=CondTrue(),
            mask_ast=Const(MASK32),
            eliminated_vars=["w15", "w17", "w19", "w21"],
            title="Track A2 6-round (t=16..21) forward-backward closure eliminating W_15 (W_17,W_19,W_21)",
            depth=state.depth + 1,
        )

    # 5. Track B1: BTC Double SHA-256 splice reconstruction
    if act == "btc_splice_recover_a" and cfg.track == "B1" and cfg.btc_splice and cfg.round_start == 0:
        r1 = cfg.block1_rounds
        r2 = cfg.block2_rounds
        if r2 < 5 or r2 > 10:
            return None
        cur_s = ast_state_vars(f"b2_s{r2}")
        # Peel known constant padding rounds r2-1 down to 8 (w2[8]=0x80000000, w2[9..14]=0)
        for t in range(r2 - 1, 7, -1):
            w_val = 0x80000000 if t == 8 else 0
            cur_s = ast_inv_round_given_w(cur_s, SHA256_K[t], Const(w_val))
        # Now cur_s is at round min(r2, 8) (at least 5)
        k_end = min(r2, 8)
        # Peel back to a'_1, a'_2, a'_3, a'_4 via backSolveA
        # If k_end == 5: a'_1 = backSolveA(s5), and s5 ALREADY has a'_4, a'_3, a'_2, e'_4, e'_3, e'_2, e'_5!
        # Let's recover w2[0..3] from a'_1..a'_4 and b2_s0 = IV!
        if k_end == 5:
            a1 = ast_backSolveA(cur_s["e"], cur_s["a"], cur_s["b"], cur_s["c"], cur_s["d"])
            a2, a3, a4 = cur_s["d"], cur_s["c"], cur_s["b"]
            e2, e3, e4, e5 = cur_s["h"], cur_s["g"], cur_s["f"], cur_s["e"]
        elif k_end == 6:
            a2, a1 = ast_clamp4(cur_s)[:2]
            a3, a4 = cur_s["d"], cur_s["c"]
            e3, e4, e5 = cur_s["h"], cur_s["g"], cur_s["f"]
            # e2 from T1(1) + IV_2
            t1_0 = Sub(a1, ast_T2(Const(SHA256_IV[0]), Const(SHA256_IV[1]), Const(SHA256_IV[2])))
            e1 = Add(Const(SHA256_IV[3]), t1_0)
            t1_1 = Sub(a2, ast_T2(a1, Const(SHA256_IV[0]), Const(SHA256_IV[1])))
            e2 = Add(Const(SHA256_IV[2]), t1_1)
        elif k_end == 7:
            a3, a2, a1 = ast_clamp4(cur_s)[:3]
            a4 = cur_s["d"]
            e4, e5 = cur_s["h"], cur_s["g"]
            t1_0 = Sub(a1, ast_T2(Const(SHA256_IV[0]), Const(SHA256_IV[1]), Const(SHA256_IV[2])))
            t1_1 = Sub(a2, ast_T2(a1, Const(SHA256_IV[0]), Const(SHA256_IV[1])))
            t1_2 = Sub(a3, ast_T2(a2, a1, Const(SHA256_IV[0])))
            e1 = Add(Const(SHA256_IV[3]), t1_0)
            e2 = Add(Const(SHA256_IV[2]), t1_1)
            e3 = Add(Const(SHA256_IV[1]), t1_2)
        else:  # k_end == 8
            a4, a3, a2, a1 = ast_clamp4(cur_s)
            e5 = cur_s["h"]
            t1_0 = Sub(a1, ast_T2(Const(SHA256_IV[0]), Const(SHA256_IV[1]), Const(SHA256_IV[2])))
            t1_1 = Sub(a2, ast_T2(a1, Const(SHA256_IV[0]), Const(SHA256_IV[1])))
            t1_2 = Sub(a3, ast_T2(a2, a1, Const(SHA256_IV[0])))
            t1_3 = Sub(a4, ast_T2(a3, a2, a1))
            e1 = Add(Const(SHA256_IV[3]), t1_0)
            e2 = Add(Const(SHA256_IV[2]), t1_1)
            e3 = Add(Const(SHA256_IV[1]), t1_2)
            e4 = Add(Const(SHA256_IV[0]), t1_3)

        # Recover w2[0..3] -> b1_s_end.{a,b,c,d}
        iv = [Const(v) for v in SHA256_IV]
        t1_0 = Sub(a1, ast_T2(iv[0], iv[1], iv[2]))
        e1 = Add(iv[3], t1_0)
        c0_fixed = Add(Add(Add(iv[7], Sigma1(iv[4])), Ch(iv[4], iv[5], iv[6])), Const(SHA256_K[0]))
        w2_0 = Sub(t1_0, c0_fixed)

        t1_1 = Sub(a2, ast_T2(a1, iv[0], iv[1]))
        w2_1 = Sub(Sub(Sub(Sub(t1_1, iv[6]), Sigma1(e1)), Ch(e1, iv[4], iv[5])), Const(SHA256_K[1]))

        t1_2 = Sub(a3, ast_T2(a2, a1, iv[0]))
        w2_2 = Sub(Sub(Sub(Sub(t1_2, iv[5]), Sigma1(e2)), Ch(e2, e1, iv[4])), Const(SHA256_K[2]))

        t1_3 = Sub(a4, ast_T2(a3, a2, a1))
        w2_3 = Sub(Sub(Sub(Sub(t1_3, iv[4]), Sigma1(e3)), Ch(e3, e2, e1)), Const(SHA256_K[3]))

        # Recover w2[4] = E5 - a1 - e1 - Sigma1(e4) - Ch(e4, e3, e2) - K4
        w2_4 = Sub(Sub(Sub(Sub(Sub(e5, a1), e1), Sigma1(e4)), Ch(e4, e3, e2)), Const(SHA256_K[4]))

        b1_a = Sub(w2_0, iv[0])
        b1_b = Sub(w2_1, iv[1])
        b1_c = Sub(w2_2, iv[2])
        b1_d = Sub(w2_3, iv[3])
        b1_e = Sub(w2_4, iv[4])

        if r1 == 0:
            # Target b1_s0.d (or b1_s0.e) directly across Block 2!
            if rng.random() < 0.5:
                lhs = b1_e
                rhs = Var("b1_s0.e")
                elim = ["b1_s0.a", "b1_s0.b", "b1_s0.c", "b1_s0.d", "b1_s0.f", "b1_s0.g", "b1_s0.h"]
                title = f"Track B1 BTC splice (r2={r2}): reconstruct Block1 E_end from Block2 S'_{r2}"
            else:
                lhs = b1_d
                rhs = Var("b1_s0.d")
                elim = ["b1_s0.e", "b1_s0.f", "b1_s0.g", "b1_s0.h"]
                title = f"Track B1 BTC splice (r2={r2}): reconstruct Block1 A_{{end-3}} from Block2 S'_{r2}"
        elif r1 == 1:
            # Cross-splice backSolveA into Block 1!
            lhs = ast_backSolveA(b1_e, b1_a, b1_b, b1_c, b1_d)
            rhs = Var("b1_s0.d")
            elim = ["b1_w0", "b1_s0.e", "b1_s0.f", "b1_s0.g", "b1_s0.h", "b1_s1.e", "b1_s1.f", "b1_s1.g", "b1_s1.h"]
            title = f"Track B1 BTC cross-splice (r1=1, r2={r2}): reconstruct Block1 A_{{-3}} from S'_{r2} (elim b1_w0, b1_E)"
        else:
            # r1 >= 2 and k_end == 8: recover all 8 words of b1_s[r1] and clamp4 back into Block 1!
            if k_end < 8:
                return None
            e6, e7, e8 = cur_s["g"], cur_s["f"], cur_s["e"]
            w2_5 = Sub(Sub(Sub(Sub(Sub(e6, a2), e2), Sigma1(e5)), Ch(e5, e4, e3)), Const(SHA256_K[5]))
            w2_6 = Sub(Sub(Sub(Sub(Sub(e7, a3), e3), Sigma1(e6)), Ch(e6, e5, e4)), Const(SHA256_K[6]))
            w2_7 = Sub(Sub(Sub(Sub(Sub(e8, a4), e4), Sigma1(e7)), Ch(e7, e6, e5)), Const(SHA256_K[7]))
            b1_s_end = {
                "a": b1_a, "b": b1_b, "c": b1_c, "d": b1_d,
                "e": b1_e, "f": Sub(w2_5, iv[5]), "g": Sub(w2_6, iv[6]), "h": Sub(w2_7, iv[7]),
            }
            clamp = ast_clamp4(b1_s_end)
            # clamp[min(r1,4)-1] matches b1_s0.{a,b,c,d}[4-r1] -> for r1=4, clamp[3] == b1_s0.d!
            idx_t = min(r1, 4) - 1
            lhs = clamp[idx_t]
            rhs = Var(f"b1_s0.{STATE_FIELDS[3 - (4 - min(r1, 4))]}") if r1 <= 4 else Var("b1_s0.d")
            elim = [f"b1_w{i}" for i in range(r1)] + ["b1_s0.e", "b1_s0.f", "b1_s0.g", "b1_s0.h"]
            title = f"Track B1 BTC deep cross-splice (r1={r1}, r2={r2}): {r1+r2}-round Block1 reconstruction from S'_{r2}"

        return ConjectureState(
            track="B1",
            config=cfg,
            spec=spec,
            form="FormI_Reconstruction",
            lhs_ast=lhs,
            rhs_ast=rhs,
            cond_ast=CondTrue(),
            mask_ast=Const(MASK32),
            eliminated_vars=elim,
            title=title,
            depth=state.depth + 1,
        )

    # 6. Track B2: Block 2 entering message expansion (W'[9..14] = 0 collapse)
    if act == "btc_b2_sched_collapse" and cfg.track == "B2" and cfg.btc_splice:
        # Window round_start = 12..15, block2_rounds = 6..7 -> crosses W'[16] = sigma0(W'[1]) + W'[0]!
        rs = cfg.round_start
        r2 = cfg.block2_rounds
        if rs < 12 or rs + r2 < 17:
            return None
        # Peel from rs + r2 down to 16? Wait: w2[16] depends on b1_s0.a, b1_s0.b!
        # Let's check: from b2_s_{rs+r2}, if we know b1_s0.{a..d} (only 4 words of Block 1, WITHOUT b1_s0.{e..h}!),
        # then w2[0..3] are known, while w2[4..7] (from b1_s0.{e..h}) are UNKNOWN!
        # And w2[16] = sigma0(w2[1]) + w2[0], w2[17] = sigma1(256) + sigma0(w2[2]) + w2[1],
        # w2[18] = sigma1(w2[16]) + sigma0(w2[3]) + w2[2] ALL DEPEND ONLY ON b1_s0.{a,b,c,d} (NOT e,f,g,h)!
        # Meanwhile w2[12..15] are CONSTANTS (0,0,0,256)!
        # So for rs = 12, r2 = 7 (rounds 12..18, end = 19): ALL 7 ROUNDS 12..18 use ONLY constants + b1_s0.{a..d},
        # COMPLETELY ELIMINATING b1_s0.{e,f,g,h} (w2[4..7])!
        # Even better: for rs = 12, r2 = 6 (rounds 12..17), ALL 6 rounds 12..17 use ONLY constants + b1_s0.{a,b,c}!
        # Wait: peel b2_s_{rs+r2} down to b2_s_{rs} and compare b2_s_{rs}.d!
        iv = [Const(v) for v in SHA256_IV]
        w2_expr: Dict[int, Dict[str, Any]] = {}
        for i in range(4):
            w2_expr[i] = Add(iv[i], Var(f"b1_s0.{STATE_FIELDS[i]}"))
        w2_expr[8] = Const(0x80000000)
        for i in range(9, 15):
            w2_expr[i] = Const(0)
        w2_expr[15] = Const(256)
        w2_expr[16] = Add({"op": "sigma0", "args": [w2_expr[1]]}, w2_expr[0])
        w2_expr[17] = Add(Add({"op": "sigma1", "args": [Const(256)]}, {"op": "sigma0", "args": [w2_expr[2]]}), w2_expr[1])
        w2_expr[18] = Add(Add({"op": "sigma1", "args": [w2_expr[16]]}, {"op": "sigma0", "args": [w2_expr[3]]}), w2_expr[2])

        end_r = rs + r2
        if any(t not in w2_expr for t in range(rs, end_r)):
            return None
        cur_s = ast_state_vars(f"b2_s{end_r}")
        for t in range(end_r - 1, rs - 1, -1):
            cur_s = ast_inv_round_given_w(cur_s, SHA256_K[t], w2_expr[t])
        # Compare cur_s["d"] == b2_s_{rs}.d (or cur_s["h"] == b2_s_{rs}.h)!
        lhs = cur_s["h"] if rng.random() < 0.5 else cur_s["d"]
        rhs = Var(f"b2_s{rs}.h") if lhs is cur_s["h"] else Var(f"b2_s{rs}.d")
        return ConjectureState(
            track="B2",
            config=cfg,
            spec=spec,
            form="FormI_Reconstruction",
            lhs_ast=lhs,
            rhs_ast=rhs,
            cond_ast=CondTrue(),
            mask_ast=Const(MASK32),
            eliminated_vars=["b1_s0.e", "b1_s0.f", "b1_s0.g", "b1_s0.h"],
            title=f"Track B2 Block2 W'[9..14]=0 collapse (t={rs}..{end_r-1}): invert across W'[16..{end_r-1}] eliminating Block1 E",
            depth=state.depth + 1,
        )

    # 7. Form II (Non-interference / Masking Invariance) across A1, A2, B1, B2
    if act == "form2_noninterference":
        if not cfg.btc_splice and cfg.num_rounds == 5 and cfg.round_start == 0:
            s_end = ast_state_vars("s5")
            lhs = Sub(ast_five_round_coupling(s_end, SHA256_K[4]), Var("w4"))
            return ConjectureState(
                track=cfg.track,
                config=cfg,
                spec=spec,
                form="FormII_MaskingInvariance",
                lhs_ast=lhs,
                rhs_ast=lhs,
                cond_ast=CondTrue(),
                mask_ast=Const(MASK32),
                eliminated_vars=["w0", "w1", "w2", "w3", "s0.e", "s0.f", "s0.g", "s0.h"],
                title=f"Track {cfg.track} 5-round FormII non-interference under (W_{{0..3}}, E_{{0..-3}})",
                depth=state.depth + 1,
            )
        if cfg.btc_splice and cfg.track == "B1" and state.form == "FormI_Reconstruction" and state.depth >= 1:
            # Convert a valid B1 reconstruction LHS into FormII invariance under Block 1 E/W!
            elim_b1 = [v for v in spec.base_vars if v.startswith("b1_w") or v.startswith("b1_s0.e") or v.startswith("b1_s0.f") or v.startswith("b1_s0.g") or v.startswith("b1_s0.h")]
            if elim_b1 and not (collect_vars(state.lhs_ast) & set(elim_b1)):
                return ConjectureState(
                    track="B1",
                    config=cfg,
                    spec=spec,
                    form="FormII_MaskingInvariance",
                    lhs_ast=state.lhs_ast,
                    rhs_ast=state.lhs_ast,
                    cond_ast=state.cond_ast,
                    mask_ast=state.mask_ast,
                    eliminated_vars=elim_b1,
                    title=f"{state.title} [FormII B1 non-interference]",
                    depth=state.depth + 1,
                )
        return None

    # 8. Conditional masking / carry-shielding attachments & mutations
    if act == "attach_maskzero_cond" and state.cond_ast["op"] == "True":
        cands = [v for v in spec.base_vars if v not in state.eliminated_vars]
        if len(cands) >= 2:
            v1, v2 = rng.sample(cands, 2)
            new_cond = CondMaskZero(Var(v1), Var(v2))
            return ConjectureState(
                track=state.track,
                config=cfg,
                spec=spec,
                form=state.form,
                lhs_ast=state.lhs_ast,
                rhs_ast=state.rhs_ast,
                cond_ast=new_cond,
                mask_ast=state.mask_ast,
                eliminated_vars=state.eliminated_vars,
                title=f"{state.title} + MaskZero({v1},{v2})",
                depth=state.depth + 1,
            )

    if act == "attach_biteq_cond" and state.cond_ast["op"] == "True":
        cands = [v for v in spec.base_vars if v not in state.eliminated_vars]
        if cands:
            v1 = rng.choice(cands)
            k = rng.choice([0, 1, 31])
            b = rng.choice([0, 1])
            return ConjectureState(
                track=state.track,
                config=cfg,
                spec=spec,
                form=state.form,
                lhs_ast=state.lhs_ast,
                rhs_ast=state.rhs_ast,
                cond_ast=CondBitEq(Var(v1), k, b),
                mask_ast=state.mask_ast,
                eliminated_vars=state.eliminated_vars,
                title=f"{state.title} + BitEq({v1}[{k}]={b})",
                depth=state.depth + 1,
            )

    if act == "attach_ch_lock_cond" and state.cond_ast["op"] == "True":
        # Lock s1.e == s0.e or s0.f == s0.g on non-elim base vars
        cands = [v for v in spec.base_vars if v not in state.eliminated_vars]
        if len(cands) >= 2:
            v1, v2 = rng.sample(cands, 2)
            cond = CondEq(Var(v1), Var(v2)) if rng.random() < 0.5 else CondEq(Var(v1), Not(Var(v2)))
            return ConjectureState(
                track=state.track,
                config=cfg,
                spec=spec,
                form=state.form,
                lhs_ast=state.lhs_ast,
                rhs_ast=state.rhs_ast,
                cond_ast=cond,
                mask_ast=state.mask_ast,
                eliminated_vars=state.eliminated_vars,
                title=f"{state.title} + EqCond",
                depth=state.depth + 1,
            )

    if act in ("mutate_xor_add_sub", "mutate_shift_rotr"):
        # Carry-shielding MaskZero + XOR replacement on Add/Sub nodes (or LSB bit-0 mod-2!)
        if rng.random() < 0.50 and state.form == "FormI_Reconstruction" and state.mask_ast.get("val") == MASK32:
            if state.lhs_ast["op"] in ("Add", "Sub") and state.rhs_ast["op"] in ("Add", "Sub"):
                l0, l1 = state.lhs_ast["args"]
                r0, r1 = state.rhs_ast["args"]
                if rng.random() < 0.5:
                    return ConjectureState(
                        track=state.track,
                        config=cfg,
                        spec=spec,
                        form="FormI_Reconstruction",
                        lhs_ast=Xor(l0, l1),
                        rhs_ast=Xor(r0, r1),
                        cond_ast=state.cond_ast,
                        mask_ast=Const(1),
                        eliminated_vars=state.eliminated_vars,
                        title=f"{state.title} [LSB carry-free XOR]",
                        depth=state.depth + 1,
                    )
                else:
                    # Carry-shielded full-word XOR equivalence when MaskZero(r0, r1) holds on target Add(r0, r1)
                    if state.rhs_ast["op"] == "Add" and r0["op"] == "Var" and r1["op"] == "Var":
                        return ConjectureState(
                            track=state.track,
                            config=cfg,
                            spec=spec,
                            form="FormI_Reconstruction",
                            lhs_ast=state.lhs_ast,
                            rhs_ast=Xor(r0, r1),
                            cond_ast=CondMaskZero(r0, r1),
                            mask_ast=Const(MASK32),
                            eliminated_vars=state.eliminated_vars,
                            title=f"{state.title} [Carry-shielded ({r0['name']}&{r1['name']})==0 => XOR]",
                            depth=state.depth + 1,
                        )
        obs = sorted(collect_vars(state.lhs_ast))
        new_lhs = _ast_mutate_expr(state.lhs_ast, obs, rng)
        nodes, depth = ast_stats(new_lhs)
        if nodes > 1500 or depth > 40:
            return None
        return ConjectureState(
            track=state.track,
            config=cfg,
            spec=spec,
            form=state.form,
            lhs_ast=new_lhs,
            rhs_ast=state.rhs_ast,
            cond_ast=state.cond_ast,
            mask_ast=state.mask_ast,
            eliminated_vars=state.eliminated_vars,
            title=f"{state.title} [mut]",
            depth=state.depth + 1,
        )

    return None


# ---------------------------------------------------------------------------
# Neural MCTS Discovery Engine
# ---------------------------------------------------------------------------

class NeuralMCTSDiscoveryEngine:
    def __init__(
        self,
        seed: int = 42,
        z3_timeout_ms: int = 12000,
        require_min_active_count: int = 5,
    ):
        self.py_rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.net = PolicyValueNet()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.auditor = Z3Auditor(
            timeout_ms=z3_timeout_ms,
            require_min_active_count=require_min_active_count,
        )
        self.replay_buffer: List[Tuple[np.ndarray, int, float]] = []
        self.stats = {
            "total_candidates_generated": 0,
            "passed_random_filter": 0,
            "z3_unsat_verified": 0,
            "passed_all_7_nontriviality": 0,
            "rejected_reasons": {},
        }

    def _predict(self, state: ConjectureState) -> Tuple[np.ndarray, float]:
        feat = encode_search_state(
            state.track,
            state.config.num_rounds,
            state.lhs_ast,
            state.cond_ast,
            state.bit_match,
        )
        with torch.no_grad():
            logits, val = self.net(torch.from_numpy(feat).unsqueeze(0))
            probs = F.softmax(logits.squeeze(0), dim=-1).numpy()
            return probs, float(val.item())

    def _train_step(self, batch_size: int = 32) -> float:
        if len(self.replay_buffer) < batch_size:
            return 0.0
        batch = self.py_rng.sample(self.replay_buffer, batch_size)
        xs = torch.from_numpy(np.stack([b[0] for b in batch], axis=0))
        acts = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rews = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        logits, vals = self.net(xs)
        v_loss = F.mse_loss(vals, rews)
        # Reward-weighted policy gradient
        adv = torch.clamp(rews - vals.detach(), min=0.0, max=15.0)
        logp = F.log_softmax(logits, dim=-1)
        chosen_logp = logp[torch.arange(batch_size), acts]
        p_loss = -(adv * chosen_logp).mean()
        loss = v_loss + 0.25 * p_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def _evaluate_and_audit(self, state: ConjectureState) -> float:
        self.stats["total_candidates_generated"] += 1
        prop = CandidateProposition(
            track=state.track,
            title=state.title,
            form=state.form,
            config=state.config,
            eliminated_vars=state.eliminated_vars,
            condition_ast=state.cond_ast,
            mask_ast=state.mask_ast,
            lhs_ast=state.lhs_ast,
            rhs_ast=state.rhs_ast,
        )
        passed_fast, cand_prop, fast_rew, bit_match = evaluate_candidate_fast(
            prop, state.spec, self.np_rng, n_samples=512
        )
        state.bit_match = bit_match
        if not passed_fast or cand_prop is None:
            return fast_rew

        self.stats["passed_random_filter"] += 1
        audit_res = self.auditor.audit(cand_prop, state.spec)
        if audit_res.passed:
            self.stats["z3_unsat_verified"] += 1
            self.stats["passed_all_7_nontriviality"] += 1
            reward = 10.0 + 1.0 * audit_res.v6_active_span + 0.5 * len(cand_prop.eliminated_vars)
            return reward
        else:
            if not audit_res.reject_reason.startswith("v1_failed"):
                # Passed V1 UNSAT (or failed later nontriviality)
                if not audit_res.reject_reason.startswith(("empty", "lhs_", "cond_", "mask_", "rhs_", "syntactic", "form2_")):
                    pass
            rkey = audit_res.reject_reason.split(":")[0]
            self.stats["rejected_reasons"][rkey] = self.stats["rejected_reasons"].get(rkey, 0) + 1
            return fast_rew

    def run_search(
        self,
        budget_seconds: float,
        rollouts_per_root: int = 24,
        max_depth: int = 4,
        c_puct: float = 1.4,
        log_every_sec: float = 30.0,
    ) -> List[AuditResult]:
        t_start = time.time()
        last_log = t_start
        roots = [MCTSNode(s) for s in build_seed_states(self.py_rng)]

        iteration = 0
        while (time.time() - t_start) < budget_seconds:
            root = roots[iteration % len(roots)]
            iteration += 1

            for _ in range(rollouts_per_root):
                if (time.time() - t_start) >= budget_seconds:
                    break
                node = root
                path: List[Tuple[MCTSNode, int]] = []

                # Selection & Expansion
                while node.state.depth < max_depth:
                    probs, _ = self._predict(node.state)
                    total_n = max(1, node.visits)
                    best_a = -1
                    best_score = -1e9
                    for a_idx in range(len(ACTION_NAMES)):
                        child = node.children.get(a_idx)
                        q = child.q_value if child is not None else 0.0
                        n_sa = child.visits if child is not None else 0
                        u = c_puct * float(probs[a_idx]) * math.sqrt(total_n) / (1 + n_sa)
                        score = q + u + self.py_rng.uniform(0.0, 1e-4)
                        if score > best_score:
                            best_score = score
                            best_a = a_idx

                    path.append((node, best_a))
                    if best_a not in node.children:
                        nxt_state = apply_mcts_action(node.state, best_a, self.py_rng)
                        if nxt_state is None:
                            reward = 0.0
                            feat = encode_search_state(
                                node.state.track,
                                node.state.config.num_rounds,
                                node.state.lhs_ast,
                                node.state.cond_ast,
                                node.state.bit_match,
                            )
                            self.replay_buffer.append((feat, best_a, reward))
                            node.children[best_a] = MCTSNode(node.state, prior=float(probs[best_a]))
                            node.children[best_a].visits += 1
                            break
                        child_node = MCTSNode(nxt_state, prior=float(probs[best_a]))
                        node.children[best_a] = child_node
                        reward = self._evaluate_and_audit(nxt_state)
                        feat = encode_search_state(
                            node.state.track,
                            node.state.config.num_rounds,
                            node.state.lhs_ast,
                            node.state.cond_ast,
                            node.state.bit_match,
                        )
                        self.replay_buffer.append((feat, best_a, reward))
                        if len(self.replay_buffer) > 4096:
                            self.replay_buffer = self.replay_buffer[-4096:]
                        # Backprop
                        child_node.visits += 1
                        child_node.value_sum += reward
                        for pnode, _ in reversed(path):
                            pnode.visits += 1
                            pnode.value_sum += reward
                        break
                    else:
                        node = node.children[best_a]
                else:
                    # Reached max_depth
                    reward = self._evaluate_and_audit(node.state)
                    for pnode, _ in reversed(path):
                        pnode.visits += 1
                        pnode.value_sum += reward

            if iteration % 4 == 0:
                self._train_step(batch_size=32)

            now = time.time()
            if now - last_log >= log_every_sec:
                last_log = now
                elapsed = now - t_start
                print(
                    f"[MCTS t={elapsed:.1f}s] gen={self.stats['total_candidates_generated']} "
                    f"fast_pass={self.stats['passed_random_filter']} "
                    f"verified={self.stats['passed_all_7_nontriviality']} "
                    f"rejects={self.stats['rejected_reasons']}",
                    flush=True,
                )

        return self.auditor.accepted_records
