"""Boolean Straight-Line Program (SLP) and AlphaTensor-style multilinear proxy circuit representation.

Gate basis: {CONST_1, XOR, AND} on LSB-first input variables x_0 .. x_{n-1}.
Includes:
- Complexity & DAG validation (M_AND <= 16, M_XOR <= 128, S_max <= 32, deg <= 4)
- Exact symbolic F_2 ANF polynomial expansion for exact algebraic degree verification
- Bit-packed int64 vectorized evaluator (exact integer arithmetic on CPU or CUDA)
- Human-readable F_2 algebraic expression formatter & JSON serialization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

MAX_AND_GATES = 16
MAX_XOR_GATES = 128
MAX_SUPPORT = 32
MAX_DEGREE = 4


@dataclass(frozen=True)
class AffineForm:
    """Affine linear form over F_2: const ^ XOR_{idx in terms} node[idx].

    Node indices:
    - 0 <= idx < n_inputs: input variable x_{idx}
    - n_inputs <= idx < n_inputs + len(gates): internal AND gate g_{idx - n_inputs}
    """

    const: int
    terms: Tuple[int, ...]

    def __post_init__(self) -> None:
        c = int(self.const) & 1
        # Canonicalize: sort and cancel duplicate terms mod 2
        counts: Dict[int, int] = {}
        for t in self.terms:
            ti = int(t)
            counts[ti] = counts.get(ti, 0) ^ 1
        canon = tuple(sorted(k for k, v in counts.items() if v == 1))
        object.__setattr__(self, "const", c)
        object.__setattr__(self, "terms", canon)

    @property
    def xor_cost(self) -> int:
        """Number of 2-input XOR gates (constant inversion is free or counted as 0/1)."""
        return max(0, len(self.terms) - 1)

    @property
    def support_size(self) -> int:
        return len(self.terms)

    def eval_packed(self, node_words: List[torch.Tensor], ones_word: torch.Tensor) -> torch.Tensor:
        """Evaluate on bit-packed int64 words (shape (B_words,))."""
        acc = ones_word.clone() if self.const else torch.zeros_like(ones_word)
        for idx in self.terms:
            acc = acc ^ node_words[idx]
        return acc

    def eval_int(self, node_bits: List[int]) -> int:
        """Evaluate on single scalar bits (0/1)."""
        acc = self.const
        for idx in self.terms:
            acc ^= node_bits[idx]
        return acc & 1

    def to_dict(self) -> Dict[str, Any]:
        return {"const": self.const, "terms": list(self.terms)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AffineForm":
        return cls(const=int(data["const"]), terms=tuple(int(t) for t in data["terms"]))

    def format_expr(self, var_names: Sequence[str], n_inputs: int) -> str:
        parts: List[str] = []
        if self.const:
            parts.append("1")
        for idx in self.terms:
            if idx < n_inputs:
                parts.append(var_names[idx])
            else:
                parts.append(f"g{idx - n_inputs}")
        if not parts:
            return "0"
        return " ^ ".join(parts)


@dataclass(frozen=True)
class AndGate:
    """Two-input AND gate: g = left & right."""

    left: AffineForm
    right: AffineForm

    def to_dict(self) -> Dict[str, Any]:
        return {"left": self.left.to_dict(), "right": self.right.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AndGate":
        return cls(
            left=AffineForm.from_dict(data["left"]),
            right=AffineForm.from_dict(data["right"]),
        )


def _anf_xor(p: set[frozenset[int]], q: set[frozenset[int]]) -> set[frozenset[int]]:
    return p ^ q


def _anf_mul(
    p: set[frozenset[int]],
    q: set[frozenset[int]],
    max_terms_abort: int = 250_000,
) -> Optional[set[frozenset[int]]]:
    """Exact multiplication in F_2[x_0..x_{n-1}] / (x_i^2 - x_i)."""
    if not p or not q:
        return set()
    res: set[frozenset[int]] = set()
    for m1 in p:
        for m2 in q:
            m = m1 | m2
            if m in res:
                res.remove(m)
            else:
                res.add(m)
                if len(res) > max_terms_abort:
                    return None
    return res


@dataclass
class CircuitSLP:
    """XOR-AND Straight-Line Program over F_2^n -> F_2."""

    n_inputs: int
    gates: List[AndGate]
    output: AffineForm
    name: str = "proxy"
    method: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def and_gates(self) -> int:
        return len(self.gates)

    @property
    def xor_gates(self) -> int:
        total = self.output.xor_cost
        for g in self.gates:
            total += g.left.xor_cost + g.right.xor_cost
        return total

    @property
    def max_support(self) -> int:
        vals = [self.output.support_size]
        for g in self.gates:
            vals.append(g.left.support_size)
            vals.append(g.right.support_size)
        return max(vals) if vals else 0

    def input_support(self) -> Tuple[int, ...]:
        """Return sorted input indices actually reachable from the output."""
        used_nodes = set(self.output.terms)
        # Backward reachability over DAG
        for m in range(len(self.gates) - 1, -1, -1):
            node_idx = self.n_inputs + m
            if node_idx in used_nodes:
                used_nodes.update(self.gates[m].left.terms)
                used_nodes.update(self.gates[m].right.terms)
        return tuple(sorted(idx for idx in used_nodes if idx < self.n_inputs))

    def structural_degree_bound(self) -> int:
        """Compute structural upper bound on F_2 degree from the DAG."""
        degs: List[int] = [1] * self.n_inputs
        for g in self.gates:
            d_l = 0 if not g.left.terms else max(degs[idx] for idx in g.left.terms)
            d_r = 0 if not g.right.terms else max(degs[idx] for idx in g.right.terms)
            degs.append(d_l + d_r)
        if not self.output.terms:
            return 0
        return max(degs[idx] for idx in self.output.terms)

    def exact_anf_degree(self, max_terms_abort: int = 250_000) -> Optional[int]:
        """Compute exact F_2 ANF degree by symbolic polynomial expansion (returns None if too large)."""
        # Only expand gates reachable from output
        reachable_gates = set()
        queue = [idx - self.n_inputs for idx in self.output.terms if idx >= self.n_inputs]
        while queue:
            m = queue.pop()
            if m in reachable_gates:
                continue
            reachable_gates.add(m)
            for idx in self.gates[m].left.terms:
                if idx >= self.n_inputs:
                    queue.append(idx - self.n_inputs)
            for idx in self.gates[m].right.terms:
                if idx >= self.n_inputs:
                    queue.append(idx - self.n_inputs)

        polys: Dict[int, set[frozenset[int]]] = {}
        empty_m = frozenset()

        def eval_affine_poly(af: AffineForm) -> set[frozenset[int]]:
            acc: set[frozenset[int]] = {empty_m} if af.const else set()
            for idx in af.terms:
                if idx < self.n_inputs:
                    term_poly = {frozenset((idx,))}
                else:
                    term_poly = polys[idx]
                acc = _anf_xor(acc, term_poly)
            return acc

        for m in sorted(reachable_gates):
            pl = eval_affine_poly(self.gates[m].left)
            pr = eval_affine_poly(self.gates[m].right)
            prod = _anf_mul(pl, pr, max_terms_abort=max_terms_abort)
            if prod is None:
                return None
            polys[self.n_inputs + m] = prod

        out_poly = eval_affine_poly(self.output)
        if not out_poly:
            return 0
        return max(len(mono) for mono in out_poly)

    def validate(
        self,
        max_and: int = MAX_AND_GATES,
        max_xor: int = MAX_XOR_GATES,
        max_support: int = MAX_SUPPORT,
        max_degree: int = MAX_DEGREE,
        require_exact_degree: bool = False,
    ) -> Tuple[bool, str, int]:
        """Validate DAG acyclicity, gate budgets, support caps, and algebraic degree."""
        if len(self.gates) > max_and:
            return False, f"and_gates {len(self.gates)} > {max_and}", -1
        for m, g in enumerate(self.gates):
            limit = self.n_inputs + m
            for idx in g.left.terms:
                if idx < 0 or idx >= limit:
                    return False, f"cyclic/oob left ref {idx} at gate {m}", -1
            for idx in g.right.terms:
                if idx < 0 or idx >= limit:
                    return False, f"cyclic/oob right ref {idx} at gate {m}", -1
            if g.left.support_size > max_support or g.right.support_size > max_support:
                return False, f"support_size > {max_support} at gate {m}", -1
        total_limit = self.n_inputs + len(self.gates)
        for idx in self.output.terms:
            if idx < 0 or idx >= total_limit:
                return False, f"oob output ref {idx}", -1
        if self.output.support_size > max_support:
            return False, f"output support_size {self.output.support_size} > {max_support}", -1
        if self.xor_gates > max_xor:
            return False, f"xor_gates {self.xor_gates} > {max_xor}", -1

        deg_bound = self.structural_degree_bound()
        if deg_bound <= max_degree and not require_exact_degree:
            return True, "ok", deg_bound
        deg_exact = self.exact_anf_degree()
        deg = deg_exact if deg_exact is not None else deg_bound
        if deg_exact is None and require_exact_degree:
            return False, "exact ANF expansion aborted (too many monomials)", -1
        if deg > max_degree:
            return False, f"degree {deg} > {max_degree}", deg
        return True, "ok", deg

    def prune_dead_gates(self) -> "CircuitSLP":
        """Return an equivalent CircuitSLP with unreachable AND gates removed."""
        reachable = set()
        queue = [idx - self.n_inputs for idx in self.output.terms if idx >= self.n_inputs]
        while queue:
            m = queue.pop()
            if m in reachable:
                continue
            reachable.add(m)
            for idx in self.gates[m].left.terms:
                if idx >= self.n_inputs:
                    queue.append(idx - self.n_inputs)
            for idx in self.gates[m].right.terms:
                if idx >= self.n_inputs:
                    queue.append(idx - self.n_inputs)

        kept = sorted(reachable)
        remap: Dict[int, int] = {i: i for i in range(self.n_inputs)}
        for new_m, old_m in enumerate(kept):
            remap[self.n_inputs + old_m] = self.n_inputs + new_m

        def remap_af(af: AffineForm) -> AffineForm:
            return AffineForm(af.const, tuple(remap[t] for t in af.terms))

        new_gates = [
            AndGate(remap_af(self.gates[m].left), remap_af(self.gates[m].right))
            for m in kept
        ]
        new_out = remap_af(self.output)
        return CircuitSLP(
            n_inputs=self.n_inputs,
            gates=new_gates,
            output=new_out,
            name=self.name,
            method=self.method,
            metadata=dict(self.metadata),
        )

    def eval_packed(self, input_words: torch.Tensor) -> torch.Tensor:
        """Evaluate circuit on bit-packed inputs.

        Args:
            input_words: int64 tensor of shape (n_inputs, B_words) where each int64 holds 64 samples.
        Returns:
            int64 tensor of shape (B_words,) holding 64 output bits per int64.
        """
        if input_words.shape[0] != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} input rows, got {input_words.shape[0]}")
        ones_word = torch.full(
            (input_words.shape[1],), -1, dtype=torch.int64, device=input_words.device
        )
        node_words: List[torch.Tensor] = [input_words[i] for i in range(self.n_inputs)]
        for g in self.gates:
            lw = g.left.eval_packed(node_words, ones_word)
            rw = g.right.eval_packed(node_words, ones_word)
            node_words.append(lw & rw)
        return self.output.eval_packed(node_words, ones_word)

    def eval_int(self, x_bits: Sequence[int]) -> int:
        """Evaluate circuit on a single list/tuple of n_inputs 0/1 bits."""
        node_bits = [int(b) & 1 for b in x_bits]
        for g in self.gates:
            lv = g.left.eval_int(node_bits)
            rv = g.right.eval_int(node_bits)
            node_bits.append(lv & rv)
        return self.output.eval_int(node_bits)

    def format_algebraic(self, var_names: Optional[Sequence[str]] = None) -> str:
        """Format circuit as a readable sequence of F_2 equations."""
        names = list(var_names) if var_names is not None else [f"x{i}" for i in range(self.n_inputs)]
        lines: List[str] = []
        for m, g in enumerate(self.gates):
            le = g.left.format_expr(names, self.n_inputs)
            re = g.right.format_expr(names, self.n_inputs)
            lines.append(f"g{m} = ({le}) & ({re})")
        lines.append(f"y_hat = {self.output.format_expr(names, self.n_inputs)}")
        return "; ".join(lines)

    def to_dict(self, var_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        deg_exact = self.exact_anf_degree()
        deg = deg_exact if deg_exact is not None else self.structural_degree_bound()
        return {
            "name": self.name,
            "method": self.method,
            "n_inputs": self.n_inputs,
            "and_gates": self.and_gates,
            "xor_gates": self.xor_gates,
            "max_support": self.max_support,
            "degree": deg,
            "degree_is_exact": deg_exact is not None,
            "gates": [g.to_dict() for g in self.gates],
            "output": self.output.to_dict(),
            "algebraic_expr": self.format_algebraic(var_names),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitSLP":
        return cls(
            n_inputs=int(data["n_inputs"]),
            gates=[AndGate.from_dict(gd) for gd in data["gates"]],
            output=AffineForm.from_dict(data["output"]),
            name=str(data.get("name", "proxy")),
            method=str(data.get("method", "unknown")),
            metadata=dict(data.get("metadata", {})),
        )


def pack_bits_matrix(bits_mat: torch.Tensor) -> torch.Tensor:
    """Pack (n_inputs, N) 0/1 int64 tensor (N multiple of 64) into (n_inputs, N // 64) int64 words."""
    n_inputs, n_samples = bits_mat.shape
    if n_samples % 64 != 0:
        raise ValueError(f"N={n_samples} must be a multiple of 64")
    reshaped = bits_mat.view(n_inputs, n_samples // 64, 64)
    shifts = torch.arange(64, dtype=torch.int64, device=bits_mat.device)
    # Using signed int64 left shifts: bit 63 maps to sign bit cleanly in two's complement
    packed = (reshaped << shifts.view(1, 1, 64)).sum(dim=-1)
    return packed


def pack_bits_vector(bits_vec: torch.Tensor) -> torch.Tensor:
    """Pack (N,) 0/1 int64 tensor (N multiple of 64) into (N // 64,) int64 words."""
    return pack_bits_matrix(bits_vec.unsqueeze(0)).squeeze(0)


def count_packed_matches(pred_packed: torch.Tensor, target_packed: torch.Tensor) -> int:
    """Count exact bit agreements (0 == match after XOR, so count zeros) in int64 packed tensors."""
    diff = pred_packed ^ target_packed
    # Parallel SWAR popcount on int64 (using unsigned-safe masks via uint8 view!)
    # View as uint8 (B_words * 8 bytes) for exact, overflow-free LUT popcount on CPU & CUDA!
    return _popcount_packed_matches(diff)


_POPCOUNT_LUT_CACHE: Dict[str, torch.Tensor] = {}


def _get_popcount_lut(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _POPCOUNT_LUT_CACHE:
        lut = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.int64, device=device)
        _POPCOUNT_LUT_CACHE[key] = lut
    return _POPCOUNT_LUT_CACHE[key]


def _popcount_packed_matches(diff_packed: torch.Tensor) -> int:
    """Count total zero bits in diff_packed (int64 tensor of shape (B_words,)) using uint8 LUT."""
    lut = _get_popcount_lut(diff_packed.device)
    u8 = diff_packed.view(torch.uint8).to(torch.int64)
    mismatches = int(lut[u8].sum().item())
    total_bits = diff_packed.numel() * 64
    return total_bits - mismatches


def batch_count_packed_matches(preds_packed: torch.Tensor, target_packed: torch.Tensor) -> torch.Tensor:
    """Vectorized agreement counts for (C, B_words) candidate predictions against (B_words,) target."""
    lut = _get_popcount_lut(preds_packed.device)
    diff = preds_packed ^ target_packed.unsqueeze(0)
    u8 = diff.view(preds_packed.shape[0], -1).to(torch.uint8).to(torch.int64)
    mismatches = lut[u8].sum(dim=-1)
    total_bits = target_packed.numel() * 64
    return total_bits - mismatches
