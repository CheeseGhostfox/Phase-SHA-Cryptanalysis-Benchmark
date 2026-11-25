"""Four-color MitM / Splice-and-Cut / Biclique search skeleton.

Supports three solver backends:
- ``ortools`` (CP-SAT, intended for high-RAM Colab multi-core full sweeps)
- ``pysat`` (Glucose4/Cadical CNF encoding, available both locally and on Colab)
- ``z3`` (BitVec/Bool non-interference verification & exact SMT proof checks)

Color encoding (standard Khovratovich / Sasaki / Bao 4-color MitM model):
- 0 = GRAY   (constant / known in both forward and backward computations)
- 1 = RED    (depends on forward neutral set V_f, independent of backward V_b)
- 2 = BLUE   (depends on backward neutral set V_b, independent of forward V_f)
- 3 = PURPLE (depends on both V_f and V_b; cannot participate in matching without guessing)

To maintain strict non-interference for independent chunks:
- Forward computation (RED) may only consume GRAY or RED inputs.
- Backward computation (BLUE) may only consume GRAY or BLUE inputs.
- Splice / Initial Structure / Biclique connects forward and backward boundary states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from sha256_engine import (
    MASK32,
    SHA256_IV,
    SHA256_K,
    SHA256DependencyGraph,
    build_dependency_graph,
)

GRAY = 0
RED = 1
BLUE = 2
PURPLE = 3
COLOR_NAMES = {GRAY: "GRAY", RED: "RED", BLUE: "BLUE", PURPLE: "PURPLE"}


@dataclass
class MitMConfig:
    """Configuration for 4-color MitM / Biclique search."""

    rounds: int = 24
    start_round: int = 0
    mode: str = "sha256_64"  # "sha256_64", "btc_hdr2_64", "btc_double_128"
    use_msg_schedule: bool = True
    allow_splice_and_cut: bool = True
    biclique_rounds: int = 0  # 0 for plain MitM, >0 for Biclique
    min_red_dof: int = 1  # minimum independent forward words/bits
    min_blue_dof: int = 1  # minimum independent backward words/bits
    matching_bits: int = 32  # e.g., leading-zero bits k or partial state match
    # Bit-level or word-level relaxation for truncated clz(H[0]) >= k
    truncated_target_only: bool = False


@dataclass
class MitMSolution:
    """Serializable candidate found by the 4-color solver."""

    feasible: bool
    backend: str
    rounds: int
    start_round: int
    match_round: int
    biclique_range: Optional[Tuple[int, int]]
    red_msg_words: List[int]
    blue_msg_words: List[int]
    gray_msg_words: List[int]
    purple_msg_words: List[int]
    red_state_rounds: List[int]
    blue_state_rounds: List[int]
    matching_words: int
    matching_bits_effective: int
    details: Dict[str, object] = field(default_factory=dict)


def _check_ortools_available() -> bool:
    try:
        from ortools.sat.python import cp_model  # noqa: F401
        return True
    except ImportError:
        return False


def _check_pysat_available() -> bool:
    try:
        from pysat.solvers import Solver  # noqa: F401
        return True
    except ImportError:
        return False


def _check_z3_available() -> bool:
    try:
        import z3  # noqa: F401
        return True
    except ImportError:
        return False


def solve_four_color_mitm_ortools(
    cfg: MitMConfig,
    graph: Optional[SHA256DependencyGraph] = None,
    time_limit_s: float = 30.0,
    num_workers: int = 4,
) -> MitMSolution:
    """CP-SAT four-color MitM skeleton (word-level propagation + msg schedule)."""
    from ortools.sat.python import cp_model

    if graph is None:
        graph = build_dependency_graph(
            mode="sha256_64" if cfg.mode == "sha256_64" else "btc_hdr2_64",
            rounds_block0=cfg.rounds,
        )

    R = cfg.rounds
    model = cp_model.CpModel()

    # For each round t in [0..R-1], message word W[t] has 4 indicator booleans:
    # is_gray, is_red, is_blue, is_purple (exactly one true)
    # Using two boolean flags: has_red[t], has_blue[t]
    # GRAY=(0,0), RED=(1,0), BLUE=(0,1), PURPLE=(1,1)
    w_red = [model.NewBoolVar(f"w_red_{t}") for t in range(R)]
    w_blue = [model.NewBoolVar(f"w_blue_{t}") for t in range(R)]

    # Constant message words (from BTC padding) are forced GRAY=(0,0)
    for t in range(R):
        w_name = f"W[{t}]"
        if w_name in graph.msg_nodes and graph.msg_nodes[w_name].is_const:
            model.Add(w_red[t] == 0)
            model.Add(w_blue[t] == 0)

    # Message schedule propagation: W[t] = f(W[t-2], W[t-7], W[t-15], W[t-16])
    # In pure forward expansion (no careful neutral-word algebraic cancellation),
    # if W[t] is computed from deps, any RED dep forces has_red[t]=1 unless cancelled;
    # similarly any BLUE dep forces has_blue[t]=1.
    # Conversely, if W[t-16] is backward-solved from W[t], W[t-2], W[t-7], W[t-15],
    # one of the 5 words in the recurrence can be determined by the other 4.
    # Here we encode the general 5-term relation: for each color c in {red, blue},
    # a single term in {t, t-2, t-7, t-15, t-16} cannot be the ONLY one carrying color c
    # unless it's an independently chosen neutral word (only allowed for base basis words).
    # Skeleton: for t >= 16, choose an orientation or direct forward expansion:
    if cfg.use_msg_schedule and R > 16:
        for t in range(16, R):
            # Forward schedule dependency: W[t] inherits red/blue from its 4 predecessors
            # unless W[t] is inverted as part of a linear/nonlinear relation.
            # Skeleton uses exact "at least 0 or >=2 in relation" or direct forward:
            # For linear-in-W[t], W[t-7], W[t-16] (nonlinear in W[t-2], W[t-15]):
            # If W[t-2] or W[t-15] is RED, then W[t] must be RED (cannot invert sigma0/1 cleanly)
            # unless W[t] was fixed, in which case W[t-16] or W[t-7] absorbs the color!
            # Exact conservative constraint:
            # has_red(t) or has_red(t-7) or has_red(t-16) must balance, and specifically:
            # any color in {t, t-2, t-7, t-15, t-16} cannot appear in isolation on a single word!
            terms_r = [w_red[t], w_red[t - 2], w_red[t - 7], w_red[t - 15], w_red[t - 16]]
            terms_b = [w_blue[t], w_blue[t - 2], w_blue[t - 7], w_blue[t - 15], w_blue[t - 16]]
            for idx in range(5):
                others_r = [terms_r[j] for j in range(5) if j != idx]
                others_b = [terms_b[j] for j in range(5) if j != idx]
                model.Add(terms_r[idx] <= sum(others_r))
                model.Add(terms_b[idx] <= sum(others_b))

    # State coloring at round boundary r in [0..R]:
    # Round r produces (A_{r+1}, E_{r+1}) from state at r and W[r].
    # For independent forward chunk covering rounds [r_start .. match_r - 1]:
    # In the forward chunk, NO state word may have blue=1, and NO W[r] may have blue=1.
    # In the backward chunk covering rounds [match_r .. r_end - 1]:
    # NO state word may have red=1, and NO W[r] may have red=1.
    # For skeleton: partition rounds into:
    # - forward rounds: is_fwd[t]
    # - backward rounds: is_bwd[t]
    # - biclique / initial rounds: is_bic[t] (at most cfg.biclique_rounds)
    is_fwd = [model.NewBoolVar(f"is_fwd_{t}") for t in range(R)]
    is_bwd = [model.NewBoolVar(f"is_bwd_{t}") for t in range(R)]
    is_bic = [model.NewBoolVar(f"is_bic_{t}") for t in range(R)]
    for t in range(R):
        model.Add(is_fwd[t] + is_bwd[t] + is_bic[t] == 1)
        # Forward round cannot depend on BLUE W[t]; backward cannot depend on RED W[t]
        model.Add(w_blue[t] == 0).OnlyEnforceIf(is_fwd[t])
        model.Add(w_red[t] == 0).OnlyEnforceIf(is_bwd[t])

    model.Add(sum(is_bic) <= cfg.biclique_rounds)
    model.Add(sum(is_fwd) >= 1)
    model.Add(sum(is_bwd) >= 1)

    # Require non-trivial RED and BLUE neutral degrees of freedom
    # Pure RED words: w_red[t] == 1 and w_blue[t] == 0
    pure_red = [model.NewBoolVar(f"pure_red_{t}") for t in range(R)]
    pure_blue = [model.NewBoolVar(f"pure_blue_{t}") for t in range(R)]
    for t in range(R):
        model.AddBoolAnd([w_red[t], w_blue[t].Not()]).OnlyEnforceIf(pure_red[t])
        model.AddBoolOr([w_red[t].Not(), w_blue[t]]).OnlyEnforceIf(pure_red[t].Not())
        model.AddBoolAnd([w_blue[t], w_red[t].Not()]).OnlyEnforceIf(pure_blue[t])
        model.AddBoolOr([w_blue[t].Not(), w_red[t]]).OnlyEnforceIf(pure_blue[t].Not())

    model.Add(sum(pure_red) >= cfg.min_red_dof)
    model.Add(sum(pure_blue) >= cfg.min_blue_dof)

    # Contiguity of forward and backward segments (without splice-and-cut, or cyclic with splice)
    # Maximize min(pure_red, pure_blue) + matching margin
    obj_var = model.NewIntVar(0, R, "min_dof")
    model.AddMinEquality(obj_var, [sum(pure_red), sum(pure_blue)])
    model.Maximize(obj_var)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = num_workers
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return MitMSolution(
            feasible=False,
            backend="ortools",
            rounds=R,
            start_round=cfg.start_round,
            match_round=-1,
            biclique_range=None,
            red_msg_words=[],
            blue_msg_words=[],
            gray_msg_words=[],
            purple_msg_words=[],
            red_state_rounds=[],
            blue_state_rounds=[],
            matching_words=0,
            matching_bits_effective=0,
            details={"status": solver.StatusName(status)},
        )

    red_w, blue_w, gray_w, purp_w = [], [], [], []
    fwd_r, bwd_r, bic_r = [], [], []
    for t in range(R):
        r_val = solver.Value(w_red[t])
        b_val = solver.Value(w_blue[t])
        if r_val == 0 and b_val == 0:
            gray_w.append(t)
        elif r_val == 1 and b_val == 0:
            red_w.append(t)
        elif r_val == 0 and b_val == 1:
            blue_w.append(t)
        else:
            purp_w.append(t)
        if solver.Value(is_fwd[t]):
            fwd_r.append(t)
        elif solver.Value(is_bwd[t]):
            bwd_r.append(t)
        else:
            bic_r.append(t)

    return MitMSolution(
        feasible=True,
        backend="ortools",
        rounds=R,
        start_round=cfg.start_round,
        match_round=fwd_r[-1] + 1 if fwd_r else 0,
        biclique_range=(bic_r[0], bic_r[-1]) if bic_r else None,
        red_msg_words=red_w,
        blue_msg_words=blue_w,
        gray_msg_words=gray_w,
        purple_msg_words=purp_w,
        red_state_rounds=fwd_r,
        blue_state_rounds=bwd_r,
        matching_words=1,
        matching_bits_effective=cfg.matching_bits,
        details={"status": solver.StatusName(status), "obj": solver.ObjectiveValue()},
    )


def solve_four_color_mitm_pysat(
    cfg: MitMConfig,
    graph: Optional[SHA256DependencyGraph] = None,
) -> MitMSolution:
    """PySAT CNF four-color MitM skeleton (runs locally and on Colab)."""
    from pysat.card import CardEnc
    from pysat.formula import IDPool
    from pysat.solvers import Solver

    if graph is None:
        graph = build_dependency_graph(
            mode="sha256_64" if cfg.mode == "sha256_64" else "btc_hdr2_64",
            rounds_block0=cfg.rounds,
        )

    R = cfg.rounds
    pool = IDPool()
    clauses: List[List[int]] = []

    w_red = [pool.id(f"w_red_{t}") for t in range(R)]
    w_blue = [pool.id(f"w_blue_{t}") for t in range(R)]
    is_fwd = [pool.id(f"is_fwd_{t}") for t in range(R)]
    is_bwd = [pool.id(f"is_bwd_{t}") for t in range(R)]
    is_bic = [pool.id(f"is_bic_{t}") for t in range(R)]
    pure_red = [pool.id(f"pure_red_{t}") for t in range(R)]
    pure_blue = [pool.id(f"pure_blue_{t}") for t in range(R)]

    for t in range(R):
        # Exactly one of fwd / bwd / bic
        clauses.append([is_fwd[t], is_bwd[t], is_bic[t]])
        clauses.append([-is_fwd[t], -is_bwd[t]])
        clauses.append([-is_fwd[t], -is_bic[t]])
        clauses.append([-is_bwd[t], -is_bic[t]])

        # Forward forbids blue W[t]; backward forbids red W[t]
        clauses.append([-is_fwd[t], -w_blue[t]])
        clauses.append([-is_bwd[t], -w_red[t]])

        # pure_red <-> (w_red & -w_blue)
        clauses.append([-pure_red[t], w_red[t]])
        clauses.append([-pure_red[t], -w_blue[t]])
        clauses.append([pure_red[t], -w_red[t], w_blue[t]])

        # pure_blue <-> (w_blue & -w_red)
        clauses.append([-pure_blue[t], w_blue[t]])
        clauses.append([-pure_blue[t], -w_red[t]])
        clauses.append([pure_blue[t], -w_blue[t], w_red[t]])

        w_name = f"W[{t}]"
        if w_name in graph.msg_nodes and graph.msg_nodes[w_name].is_const:
            clauses.append([-w_red[t]])
            clauses.append([-w_blue[t]])

    if cfg.use_msg_schedule and R > 16:
        for t in range(16, R):
            terms_r = [w_red[t], w_red[t - 2], w_red[t - 7], w_red[t - 15], w_red[t - 16]]
            terms_b = [w_blue[t], w_blue[t - 2], w_blue[t - 7], w_blue[t - 15], w_blue[t - 16]]
            for idx in range(5):
                others_r = [terms_r[j] for j in range(5) if j != idx]
                others_b = [terms_b[j] for j in range(5) if j != idx]
                clauses.append([-terms_r[idx]] + others_r)
                clauses.append([-terms_b[idx]] + others_b)

    # Cardinality constraints
    if cfg.biclique_rounds < R:
        enc_bic = CardEnc.atmost(lits=is_bic, bound=cfg.biclique_rounds, vpool=pool)
        clauses.extend(enc_bic.clauses)
    enc_fwd = CardEnc.atleast(lits=is_fwd, bound=1, vpool=pool)
    enc_bwd = CardEnc.atleast(lits=is_bwd, bound=1, vpool=pool)
    enc_pr = CardEnc.atleast(lits=pure_red, bound=cfg.min_red_dof, vpool=pool)
    enc_pb = CardEnc.atleast(lits=pure_blue, bound=cfg.min_blue_dof, vpool=pool)
    clauses.extend(enc_fwd.clauses)
    clauses.extend(enc_bwd.clauses)
    clauses.extend(enc_pr.clauses)
    clauses.extend(enc_pb.clauses)

    with Solver(name="g4", bootstrap_with=clauses) as solver:
        sat = solver.solve()
        if not sat:
            return MitMSolution(
                feasible=False,
                backend="pysat-g4",
                rounds=R,
                start_round=cfg.start_round,
                match_round=-1,
                biclique_range=None,
                red_msg_words=[],
                blue_msg_words=[],
                gray_msg_words=[],
                purple_msg_words=[],
                red_state_rounds=[],
                blue_state_rounds=[],
                matching_words=0,
                matching_bits_effective=0,
            )
        model_set = set(v for v in solver.get_model() if v > 0)

    red_w, blue_w, gray_w, purp_w = [], [], [], []
    fwd_r, bwd_r, bic_r = [], [], []
    for t in range(R):
        rv = w_red[t] in model_set
        bv = w_blue[t] in model_set
        if not rv and not bv:
            gray_w.append(t)
        elif rv and not bv:
            red_w.append(t)
        elif bv and not rv:
            blue_w.append(t)
        else:
            purp_w.append(t)
        if is_fwd[t] in model_set:
            fwd_r.append(t)
        elif is_bwd[t] in model_set:
            bwd_r.append(t)
        else:
            bic_r.append(t)

    return MitMSolution(
        feasible=True,
        backend="pysat-g4",
        rounds=R,
        start_round=cfg.start_round,
        match_round=fwd_r[-1] + 1 if fwd_r else 0,
        biclique_range=(bic_r[0], bic_r[-1]) if bic_r else None,
        red_msg_words=red_w,
        blue_msg_words=blue_w,
        gray_msg_words=gray_w,
        purple_msg_words=purp_w,
        red_state_rounds=fwd_r,
        blue_state_rounds=bwd_r,
        matching_words=1,
        matching_bits_effective=cfg.matching_bits,
    )


# ---------------------------------------------------------------------------
# Z3 BitVec Non-Interference Independent-Chunk Verifier
# ---------------------------------------------------------------------------


def z3_rotr(x, n: int):
    import z3
    if n == 0:
        return x
    return z3.RotateRight(x, n)


def z3_shr(x, n: int):
    import z3
    return z3.LShR(x, n)


def z3_ch(e, f, g):
    return (e & f) ^ ((~e) & g)


def z3_maj(a, b, c):
    return (a & b) ^ (a & c) ^ (b & c)


def z3_sigma0_upper(x):
    return z3_rotr(x, 2) ^ z3_rotr(x, 13) ^ z3_rotr(x, 22)


def z3_sigma1_upper(x):
    return z3_rotr(x, 6) ^ z3_rotr(x, 11) ^ z3_rotr(x, 25)


def z3_step_forward(state, w_t, t: int):
    import z3
    a, b, c, d, e, f, g, h = state
    k_val = z3.BitVecVal(SHA256_K[t], 32)
    t1 = h + z3_sigma1_upper(e) + z3_ch(e, f, g) + k_val + w_t
    t2 = z3_sigma0_upper(a) + z3_maj(a, b, c)
    return (t1 + t2, a, b, c, d + t1, e, f, g)


def z3_step_backward(next_state, w_t, t: int):
    import z3
    na, nb, nc, nd, ne, nf, ng, nh = next_state
    a, b, c = nb, nc, nd
    e, f, g = nf, ng, nh
    k_val = z3.BitVecVal(SHA256_K[t], 32)
    t2 = z3_sigma0_upper(a) + z3_maj(a, b, c)
    t1 = na - t2
    d = ne - t1
    h = t1 - z3_sigma1_upper(e) - z3_ch(e, f, g) - k_val - w_t
    return (a, b, c, d, e, f, g, h)


def z3_verify_backward_chunk_noninterference(
    round_end: int,
    num_rounds: int,
    eliminated_w_indices: Sequence[int],
    observed_outputs: Sequence[str] = ("a", "b", "c"),
    timeout_ms: int = 10000,
) -> Tuple[bool, str]:
    """Verify via Z3 BitVec(32) that backward reconstruction of ``observed_outputs``
    over ``[round_end - num_rounds .. round_end - 1]`` is strictly invariant under
    arbitrary perturbations of ``eliminated_w_indices``.

    For example, over 4 rounds ``[t..t+3]``, reconstructing ``(A_t, A_{t-1}, A_{t-2}, A_{t-3})``
    from ``S_{t+4}`` via ``backSolveA`` is strictly independent of ``W[t..t+3]``!
    """
    import z3

    start = round_end - num_rounds
    if start < 0:
        raise ValueError("round_end - num_rounds must be >= 0")

    s = z3.Solver()
    s.set("timeout", timeout_ms)

    # Symbolic end state S_{round_end} (shared across both runs)
    end_state = tuple(z3.BitVec(f"end_{nm}", 32) for nm in "abcdefgh")
    w_base = {
        t: z3.BitVec(f"W_{t}", 32) for t in range(start, round_end)
    }
    w_pert = {
        t: (z3.BitVec(f"Wp_{t}", 32) if t in eliminated_w_indices else w_base[t])
        for t in range(start, round_end)
    }

    st1 = end_state
    st2 = end_state
    for t in range(round_end - 1, start - 1, -1):
        st1 = z3_step_backward(st1, w_base[t], t)
        st2 = z3_step_backward(st2, w_pert[t], t)

    idx_map = {nm: i for i, nm in enumerate("abcdefgh")}
    diff_clauses = [st1[idx_map[nm]] != st2[idx_map[nm]] for nm in observed_outputs]
    s.add(z3.Or(*diff_clauses))
    res = s.check()
    if res == z3.unsat:
        return True, "UNSAT (strictly non-interfering)"
    if res == z3.sat:
        return False, "SAT (counterexample exists)"
    return False, f"UNKNOWN ({s.reason_unknown()})"
