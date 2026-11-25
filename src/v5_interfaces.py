"""Phase-SHA V5 orthogonal architecture interfaces, global schedules, datasets & telemetry (strict FP32).

Defines the four V6-ready orthogonal abstraction slots aligned with Lean 4 Modules 6A-6D:
  1. GeometryBackend    (Slot 1: single-bit manifold geometry — S^7 Octonion / S^{N-1} BM / S^3 Quat)
  2. EntanglementLayer  (Slot 2: multi-bit tensor network entanglement — V5C MPS)
  3. DynamicsOperator   (Slot 3: reflection / relaxation dynamics — V5D RRR vs Riemannian flow)
  4. OuterSearchShell   (Slot 4: dynamic latching, Born/SMC ensemble search & V4D/V4E adapters)
"""

from __future__ import annotations

import csv
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.sha256_ref import pack_int_messages_to_w16, sample_random_w16


def enforce_v5_fp32() -> None:
    """Strictly enforce pure FP32 precision everywhere in Phase-SHA V5 (disable TF32 / AMP)."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def assert_strict_fp32(obj: Any, name: str = "tensor") -> None:
    """Assert that a tensor or nn.Module contains strictly pure FP32 floating parameters/buffers."""
    if isinstance(obj, torch.Tensor):
        if obj.is_complex() or obj.dtype in (torch.float16, torch.bfloat16):
            raise AssertionError(f"V5 strict-FP32 violation on {name}: dtype={obj.dtype}")
        if obj.is_floating_point() and obj.dtype not in (torch.float32, torch.float64):
            raise AssertionError(f"V5 strict-FP32 unexpected float dtype on {name}: {obj.dtype}")
    elif isinstance(obj, nn.Module):
        for pname, p in obj.named_parameters():
            if p.is_complex() or p.dtype != torch.float32:
                raise AssertionError(f"V5 param {pname} must be torch.float32, got {p.dtype}")
        for bname, b in obj.named_buffers():
            if b.is_floating_point() and (b.is_complex() or b.dtype not in (torch.float32, torch.float64)):
                raise AssertionError(f"V5 buffer {bname} forbidden dtype {b.dtype}")


@dataclass(frozen=True)
class GlobalProgressiveCollapseSchedule:
    """Global-step-driven progressive annealing schedule preventing V4 continuous comfort-zone plateaus.

    Phase 1 [0%, phase1_frac]:      constant start_val (unrestricted exploration)
    Phase 2 [phase1_frac, phase2]:  cosine ramp start_val -> mid_val
    Phase 3 [phase2, 100%]:         strong cosine ramp mid_val -> end_val (mandatory collapse)
    """

    start_val: float
    mid_val: float
    end_val: float
    phase1_frac: float = 0.20
    phase2_frac: float = 0.80
    unroll_floor: float = 0.25

    def global_factor(self, global_step: int, total_steps: int) -> float:
        denom = max(1, int(total_steps))
        g = max(0.0, min(1.0, float(global_step) / float(denom)))
        p1 = float(self.phase1_frac)
        p2 = max(p1 + 1e-6, float(self.phase2_frac))
        if g <= p1:
            return float(self.start_val)
        if g <= p2:
            t = (g - p1) / (p2 - p1)
            w = 0.5 * (1.0 - math.cos(math.pi * t))
            return float(self.start_val) + (float(self.mid_val) - float(self.start_val)) * w
        t = (g - p2) / max(1e-6, 1.0 - p2)
        w = 0.5 * (1.0 - math.cos(math.pi * t))
        return float(self.mid_val) + (float(self.end_val) - float(self.mid_val)) * w

    def value(self, global_step: int, total_steps: int, unroll_frac: float = 1.0) -> float:
        base = self.global_factor(global_step, total_steps)
        u = max(0.0, min(1.0, float(unroll_frac)))
        scale = float(self.unroll_floor) + (1.0 - float(self.unroll_floor)) * u
        return base * scale


class GeometryBackend(ABC):
    """V6 Slot 1: Single-bit manifold coordinate system & algebraic gate residual operators."""

    variant: str
    dim: int
    spin_axis: int
    trans_axes: Tuple[int, ...]
    is_abelian: bool

    @abstractmethod
    def embed_spins(self, spins: torch.Tensor) -> torch.Tensor:
        """Embed discrete/continuous bipolar spins s in [-1, 1] onto unit poles s * e_spin."""

    @abstractmethod
    def embed_spins_with_equator(self, spins: torch.Tensor) -> torch.Tensor:
        """Lift bounded spins in [-1, 1] onto unit sphere along primary transverse axis."""

    @abstractmethod
    def project_spin(self, u: torch.Tensor) -> torch.Tensor:
        """Extract bipolar spin coordinate along spin_axis."""

    @abstractmethod
    def project_transverse(self, u: torch.Tensor) -> torch.Tensor:
        """Extract D-1 transverse components."""

    @abstractmethod
    def tangent_project(self, u: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Orthogonal projection of ambient delta onto T_u S^{D-1}."""

    @abstractmethod
    def retract(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        step_size: float | torch.Tensor = 1.0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Norm-preserving tangent-space retraction onto unit manifold."""

    @abstractmethod
    def align_sq(self, u: torch.Tensor) -> torch.Tensor:
        """Per-bit transverse alignment penalty ||u_perp||_2^2 (0 on discrete poles)."""

    @abstractmethod
    def xor2_residual(self, a: torch.Tensor, b: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Non-negative 2-input XOR algebraic residual (0 iff a^b=y on poles)."""

    @abstractmethod
    def xor3_spin(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """3-input XOR spin projection in [-1, 1]."""

    @abstractmethod
    def xor3_residual(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Non-negative 3-input XOR algebraic residual."""

    @abstractmethod
    def eq_s_lsb_residual(self, words9: List[torch.Tensor]) -> torch.Tensor:
        """9-operand closed-loop Eq S LSB odd-parity residual."""

    def commutator_sq(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.zeros(a.shape[:-1], dtype=torch.float32, device=a.device)

    def associator_sq(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.zeros(a.shape[:-1], dtype=torch.float32, device=a.device)


class EntanglementLayer(ABC):
    """V6 Slot 2: Multi-bit tensor-network entanglement state & conditional Born collapse."""

    bond_dim: int

    @abstractmethod
    def evolve_step(self, *args: Any, **kwargs: Any) -> Any:
        """Advance entangled wave-function by one TEBD / variational step in strict FP32."""

    @abstractmethod
    def conditional_sample(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact sequential autoregressive Born sample (spins (B, L), log_probs (B,))."""

    @abstractmethod
    def compute_entanglement_entropy(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Von Neumann entanglement entropy across the closed-loop bond cut."""

    @abstractmethod
    def compute_entangled_energy(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Expected constraint violation energy <psi | H | psi>."""

    @abstractmethod
    def marginal_logits(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Single-site reduced marginal logits for calibrated BCE & V4D adapters."""


class DynamicsOperator(ABC):
    """V6 Slot 3: Dynamical update operator (RRR chaos reflection vs Riemannian gradient flow)."""

    @abstractmethod
    def step(self, state: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Execute one dynamical step returning updated internal state and diagnostics."""

    @abstractmethod
    def readout_shadow(self, state: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Extract legitimate primal shadow coordinates (e.g., P_A(x) in RRR)."""

    @abstractmethod
    def gap_norm(self, state: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Convergence gap norm ||P_B(2 P_A(x) - x) - P_A(x)||_2 or gradient residual norm."""


class OuterSearchShell(ABC):
    """V6 Slot 4: Test-time multi-particle / replica search & V4D Neuro-SAT probability export."""

    @abstractmethod
    def solve(self, *args: Any, **kwargs: Any) -> "SolveResultV5":
        """Run multi-particle / batched search returning SolveResultV5."""

    @abstractmethod
    def export_calibrated_probs(self, solve_result: "SolveResultV5") -> torch.Tensor:
        """Export calibrated P(bit == 1) probabilities in [0, 1] for V4D SAT hybrids."""


@dataclass
class SolveResultV5:
    """Unified Phase-SHA V5 test-time solve container downward-compatible with V4."""

    solved: torch.Tensor  # (B,) bool
    num_hits: torch.Tensor  # (B,) int64
    unique_ratio: torch.Tensor  # (B,) float32
    latch_ratio: torch.Tensor  # (B,) float32
    best_w16: torch.Tensor  # (B, 16) int64
    best_energy: torch.Tensor  # (B,) float32
    best_m1_energy: torch.Tensor  # (B,) float32
    best_m32_energy: torch.Tensor  # (B,) float32
    min_m32_energy: torch.Tensor  # (B,) float32
    best_x_disc: torch.Tensor  # (B, R, 3, 32) float32 spins
    mean_align: torch.Tensor  # (B,) float32
    mean_comm: torch.Tensor  # (B,) float32
    calibrated_probs: Optional[torch.Tensor] = None  # (B, R, 3, 32) P(bit=1) in [0, 1]
    particle_weights: Optional[torch.Tensor] = None  # (B, K) normalized softmax confidences
    extra_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def solve_rate(self) -> float:
        return float(self.solved.float().mean().item())

    @property
    def mean_unique_ratio(self) -> float:
        return float(self.unique_ratio.float().mean().item())

    @property
    def mean_latch_ratio(self) -> float:
        return float(self.latch_ratio.float().mean().item())


def compute_calibrated_probs_from_logits(logits_spin: torch.Tensor) -> torch.Tensor:
    """Convert bipolar-spin logits (+ -> spin +1 = bit 0, - -> spin -1 = bit 1) to P(bit==1)."""
    # Matches F.binary_cross_entropy_with_logits(2.0 * logits, 0.5*(1+x_star)) -> P(spin=+1)=sig(2*logit)
    p_spin_pos = torch.sigmoid(2.0 * logits_spin.to(torch.float32))
    return (1.0 - p_spin_pos).clamp(0.0, 1.0)


def compute_calibrated_probs_from_candidates(
    cand_x_disc: torch.Tensor,
    cand_energies: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build energy-weighted ensemble bit marginals P(bit==1) of shape (B, R, 3, 32) for V4D."""
    temp = max(1e-3, float(temperature))
    e_f = cand_energies.to(torch.float32)  # (B, K)
    e_min = e_f.min(dim=1, keepdim=True).values
    weights = torch.softmax(-(e_f - e_min) / temp, dim=1)  # (B, K)
    bits_1 = (cand_x_disc.to(torch.float32) < 0.0).to(torch.float32)  # (B, K, R, 3, 32)
    probs = (weights.view(weights.shape[0], weights.shape[1], 1, 1, 1) * bits_1).sum(dim=1)
    return probs.clamp(0.0, 1.0), weights


def build_v5_holdout_w16(
    holdout_size: int = 2048,
    holdout_seed: int = 20261122,
    domain_mode: str = "int_range",
    num_bits: int = 32,
    msg_bytes_len: int = 4,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Construct the exact unified holdout W16 benchmark matching V4 (seed=20261122, N=2048).

    Returns (holdout_w16, train_w0_pool) where train_w0_pool is non-None only for num_bits <= 20.
    """
    holdout_rng = torch.Generator(device="cpu")
    holdout_rng.manual_seed(int(holdout_seed))
    dev = torch.device(device)

    if domain_mode == "int_range" and num_bits <= 20:
        total_space = 1 << int(num_bits)
        perm = torch.randperm(total_space, generator=holdout_rng, dtype=torch.int64)
        holdout_w0 = perm[:holdout_size]
        train_w0 = perm[holdout_size:]
        return pack_int_messages_to_w16(holdout_w0).to(dev), train_w0.to(dev)

    if domain_mode == "int_range":
        high = 1 << int(num_bits)
        unique_vals: List[int] = []
        seen = set()
        while len(unique_vals) < holdout_size:
            cand = torch.randint(
                0, high, (holdout_size * 2,), dtype=torch.int64, generator=holdout_rng
            ).tolist()
            for v in cand:
                if v not in seen:
                    seen.add(v)
                    unique_vals.append(v)
                    if len(unique_vals) == holdout_size:
                        break
        holdout_w0 = torch.tensor(unique_vals, dtype=torch.int64)
        return pack_int_messages_to_w16(holdout_w0).to(dev), None

    w16 = sample_random_w16(
        batch_size=holdout_size,
        mode=domain_mode,
        num_bits=num_bits,
        msg_bytes_len=msg_bytes_len,
        device="cpu",
        generator=holdout_rng,
    )
    return w16.to(dev), None


def build_r20_stress_holdout_w16(
    holdout_size: int = 2048,
    holdout_seed: int = 20261122,
    domain_mode: str = "int_range",
    num_bits: int = 32,
    msg_bytes_len: int = 4,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Construct unified R=20 stress-test W16 benchmark (same deterministic preimage universe)."""
    return build_v5_holdout_w16(
        holdout_size=holdout_size,
        holdout_seed=holdout_seed,
        domain_mode=domain_mode,
        num_bits=num_bits,
        msg_bytes_len=msg_bytes_len,
        device=device,
    )


def build_v5_csv_fields(rounds: int = 18) -> List[str]:
    """Build CSV column schema downward-compatible with V4 dashboard + V5 dual-axis wall_s & probes."""
    fields = [
        "step",
        "wall_s",
        "lr",
        "train_loss",
        "train_bce_loss",
        "train_bce_final",
        "train_phys_loss",
        "train_logit_reg",
        "frontier_rounds",
        "eval_bce_loss",
        "zero_w0_bit_acc",
        "zero_w0_lsb03_acc",
        "zero_w0_lsb03_zscore",
        "zero_latch_ratio",
        "zero_phys_energy_m1",
        "zero_phys_energy_m32",
        "w0_free_bit_acc",
        "anneal_latch_ratio",
        "phys_energy_m1",
        "phys_energy_m32",
        "min_phys_energy_m32",
        "best_global_m32_energy",
        "exact_preimage_rate@256",
        "unique_ratio@256",
        "peak_vram_gb",
        "step_time_ms",
    ]
    fields.extend([f"per_round_acc_{t}" for t in range(rounds)])
    fields.extend([f"zero_round_acc_{t}" for t in range(rounds)])
    fields.extend([f"bit_acc_pos_{k}" for k in range(32)])
    fields.extend([f"zero_w0_bit_acc_pos_{k}" for k in range(32)])
    fields.extend([f"zero_bit_acc_pos_{k}" for k in range(32)])
    fields.extend(
        [
            "train_xor_loss",
            "train_comm_loss",
            "train_align_loss",
            "zero_manifold_align",
            "zero_manifold_comm",
            # V5 orthogonal probes
            "rrr_gap",
            "rrr_beta",
            "drift_flag",
            "drift_slope",
            "ent_entropy",
            "ent_energy",
            "rank1_ratio",
            "eff_rank",
            "escape_count",
            "train_assoc_loss",
            "zero_assoc",
        ]
    )
    return fields


class V5MetricsLogger:
    """Append-only CSV & dashboard-compatible stdout logger tracking dual-axis steps & wall_s."""

    def __init__(self, output_dir: Path | str, rounds: int = 18) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rounds = int(rounds)
        self.csv_path = self.output_dir / "metrics.csv"
        self.fields = build_v5_csv_fields(self.rounds)

    def append_row(self, row: Dict[str, Any]) -> None:
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self.fields})

    @staticmethod
    def print_train_line(
        step: int,
        target_steps: int,
        lr: float,
        loss: float,
        phys: float,
        bce: float,
        xor_v: float,
        aux1_name: str,
        aux1_val: float,
        aux2_name: str,
        aux2_val: float,
        vram_gb: float,
        dt_ms: float,
        wall_s: float,
    ) -> None:
        print(
            f"step={step:6d}/{target_steps} lr={lr:.2e} "
            f"loss={loss:.4f} phys={phys:.4f} "
            f"bce={bce:.4f} xor={xor_v:.3f} "
            f"{aux1_name}={aux1_val:.3f} {aux2_name}={aux2_val:.3f} "
            f"vram={vram_gb:.2f}GB dt={dt_ms:.1f}ms wall={wall_s:.1f}s",
            flush=True,
        )

    @staticmethod
    def print_eval_block(
        rounds: int,
        eval_metrics: Dict[str, float],
        holdout_total: int,
        v5_probe_str: str = "",
        step: Optional[int] = None,
    ) -> None:
        tag = f" [eval step={step}]" if step is not None else "  [eval]"
        round_str = " ".join(
            f"r{t}:{eval_metrics[f'per_round_acc_{t}']:.2f}"
            for t in range(rounds)
        )
        print(
            f"{tag} bce={eval_metrics['eval_bce_loss']:.4f} "
            f"zero_w0={eval_metrics['zero_w0_bit_acc']:.4f} "
            f"w0_acc={eval_metrics['w0_free_bit_acc']:.4f} "
            f"latch_z={eval_metrics['zero_latch_ratio']:.3f} "
            f"E_m1={eval_metrics['phys_energy_m1']:.2f} "
            f"E_m32={eval_metrics['phys_energy_m32']:.2f} "
            f"pre@K={eval_metrics['exact_preimage_rate@256']:.4f} "
            f"uniq={eval_metrics['unique_ratio@256']:.3f}",
            flush=True,
        )
        r_tag = f"  [rounds step={step}]" if step is not None else "  [rounds]"
        print(f"{r_tag} {round_str}", flush=True)
        if rounds >= 18:
            w0_std = math.sqrt(0.25 / max(1.0, float(holdout_total) * 32.0))
            w0_z = (eval_metrics["zero_w0_bit_acc"] - 0.5) / w0_std
            extra = f" | {v5_probe_str}" if v5_probe_str else ""
            print(
                f"  [R={rounds} probes] "
                f"r17={eval_metrics['zero_round_acc_17']:.3f} "
                f"r16={eval_metrics['zero_round_acc_16']:.3f} "
                f"r14={eval_metrics['zero_round_acc_14']:.3f} "
                f"r13={eval_metrics['zero_round_acc_13']:.3f} "
                f"r12={eval_metrics['zero_round_acc_12']:.3f} "
                f"r10={eval_metrics['zero_round_acc_10']:.3f} "
                f"r0={eval_metrics['zero_round_acc_0']:.3f} | "
                f"W0_z={w0_z:+.2f} "
                f"lsb03={eval_metrics['zero_w0_lsb03_acc']:.4f}"
                f"(z={eval_metrics['zero_w0_lsb03_zscore']:+.2f}) "
                f"min_E32={eval_metrics['min_phys_energy_m32']:.2f} "
                f"global_min_E32={eval_metrics['best_global_m32_energy']:.2f} "
                f"align={eval_metrics['zero_manifold_align']:.4f} "
                f"comm={eval_metrics['zero_manifold_comm']:.4f}{extra}",
                flush=True,
            )


def compute_radar_detached(
    dual_basis: Any,
    x_spins: torch.Tensor,
    cb: Any,
    lam: torch.Tensor,
    grad_scale: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Safely compute detached 2-adic gradient radar & features even under torch.inference_mode()."""
    from src.boundary_clamp import ClampedBoundary

    with torch.inference_mode(False):
        with torch.enable_grad():
            x_in = x_spins.detach().clone()
            lam_in = lam.detach().clone()
            cb_in = ClampedBoundary(
                rounds=cb.rounds,
                A_full_clamped=cb.A_full_clamped.detach().clone(),
                A_full_mask=cb.A_full_mask.detach().clone(),
                E_full_clamped=cb.E_full_clamped.detach().clone(),
                E_full_mask=cb.E_full_mask.detach().clone(),
                W_full_clamped=cb.W_full_clamped.detach().clone(),
                W_full_mask=cb.W_full_mask.detach().clone(),
                A_words_clamped=cb.A_words_clamped.detach().clone(),
                E_words_clamped=cb.E_words_clamped.detach().clone(),
                W16_clamped_words=cb.W16_clamped_words.detach().clone(),
                W16_mask_bits=cb.W16_mask_bits.detach().clone(),
            )
            grad_radar, res_feat, _, _ = dual_basis.compute_radar_and_features(
                x_in, cb_in, lam=lam_in, grad_scale=grad_scale
            )
    return grad_radar.detach().to(torch.float32), res_feat.detach().to(torch.float32)


__all__ = [
    "DynamicsOperator",
    "EntanglementLayer",
    "GeometryBackend",
    "GlobalProgressiveCollapseSchedule",
    "OuterSearchShell",
    "SolveResultV5",
    "V5MetricsLogger",
    "assert_strict_fp32",
    "build_r20_stress_holdout_w16",
    "build_v5_csv_fields",
    "build_v5_holdout_w16",
    "compute_calibrated_probs_from_candidates",
    "compute_calibrated_probs_from_logits",
    "compute_radar_detached",
    "enforce_v5_fp32",
]
