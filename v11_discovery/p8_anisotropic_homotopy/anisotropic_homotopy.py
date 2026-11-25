"""Anisotropic Spatiotemporal Homotopy & Real-Matrix Counter-Diabatic GL-RRR Engine.

Strictly pure FP32 (allow_tf32=False, zero complex64). Never modifies v6a/ or src/.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.dual_basis import build_phase_projection_matrix
from src.sha256_ref import (
    MASK32,
    bits_to_words,
    get_k_tensor,
    sha256_digest,
    words_to_bits,
)
from src.v5d_rrr import DriftMonitor, ExactLocalGateProjectors
from v6a import enforce_v6a_fp32
from v6a.v6a_gl_rrr import compute_true_2adic_energy
from v6a.v6a_homotopy import (
    V6ABoundary,
    V6ACircuitSpec,
    _compute_add32_carry_words,
    build_v6a_circuit_spec,
)

from .control_cipher import (
    ControlCipherParams,
    build_control_circuit_spec,
    compute_control_2adic_energy,
    control_digest,
    get_ctrl_k_tensor,
)

PRIMARY_PATHS: Tuple[str, ...] = (
    "P0_uniform",
    "P1_bit_lsb_to_msb",
    "P2_bit_msb_to_lsb",
    "P3_round_forward",
    "P4_round_backward",
    "P5a_gate_bool_then_add",
    "P5b_gate_add_then_bool",
    "P6_cd_complex_detour",
    "P7_learned_aniso",
)

P6_ABLATION_PATHS: Tuple[str, ...] = (
    "P6_gamma02",
    "P6_gamma07",
    "P6_pure_damping",
    "P6_pure_complex",
)

ADD_SLOT_NAMES: Tuple[str, ...] = (
    "u1",
    "u2",
    "u3",
    "T1",
    "T2",
    "E_next",
    "A_next",
    "sched",
    "ff",
)


@dataclass
class AnisoCoordMap:
    """Mapping between spatiotemporal (t, g, b) gate coordinates and V6A nonlinear replicas."""

    rounds: int
    n_ch: int
    n_maj: int
    n_add: int
    n_nl_rep: int
    add_round_idx: torch.Tensor     # (N_add,) int64 in [0, R-1]
    add_slot_id: torch.Tensor       # (N_add,) int64 in [0, 8]
    rep_to_tgb_idx: torch.Tensor    # (n_nl_rep,) int64 in [0, R*3*32 - 1]
    tgb_deg: torch.Tensor           # (R, 3, 32) float32 replica multiplicity


def build_aniso_coord_map(spec: V6ACircuitSpec) -> AnisoCoordMap:
    r = spec.rounds
    n_ch = int(spec.ch_quads.shape[0])
    n_maj = int(spec.maj_quads.shape[0])
    n_add = int(spec.add_word_quads.shape[0])
    assert n_ch == 32 * r and n_maj == 32 * r

    # Ch and Maj: ordered (t=0..R-1, b=0..31)
    t_grid = torch.arange(r, dtype=torch.int64).unsqueeze(1).expand(r, 32)
    b_grid = torch.arange(32, dtype=torch.int64).unsqueeze(0).expand(r, 32)
    ch_tgb = (t_grid * 96 + 0 * 32 + b_grid).reshape(n_ch)
    maj_tgb = (t_grid * 96 + 1 * 32 + b_grid).reshape(n_maj)

    # ADD32 round and slot mapping per SPEC 1.2
    add_rounds = torch.empty(n_add, dtype=torch.int64)
    add_slots = torch.empty(n_add, dtype=torch.int64)
    n_comp = 7 * r
    n_sched = 3 * max(0, r - 16)
    for j in range(n_comp):
        add_rounds[j] = j // 7
        add_slots[j] = j % 7  # u1..A_next -> 0..6
    for idx_s in range(n_sched):
        j = n_comp + idx_s
        add_rounds[j] = 16 + (idx_s // 3)
        add_slots[j] = 7  # sched
    for i in range(8):
        j = n_comp + n_sched + i
        add_rounds[j] = r - 1
        add_slots[j] = 8  # ff

    b32 = torch.arange(32, dtype=torch.int64).unsqueeze(0)
    add_tgb = add_rounds.unsqueeze(1) * 96 + 2 * 32 + b32  # (N_add, 32)

    rep_parts = [
        ch_tgb.unsqueeze(1).expand(n_ch, 4).reshape(-1),
        maj_tgb.unsqueeze(1).expand(n_maj, 4).reshape(-1),
        add_tgb.unsqueeze(1).expand(n_add, 4, 32).reshape(-1),
    ]
    rep_to_tgb_idx = torch.cat(rep_parts, dim=0).contiguous()
    tgb_deg = torch.bincount(rep_to_tgb_idx, minlength=r * 3 * 32).to(torch.float32).view(r, 3, 32)

    return AnisoCoordMap(
        rounds=r,
        n_ch=n_ch,
        n_maj=n_maj,
        n_add=n_add,
        n_nl_rep=int(rep_to_tgb_idx.shape[0]),
        add_round_idx=add_rounds,
        add_slot_id=add_slots,
        rep_to_tgb_idx=rep_to_tgb_idx,
        tgb_deg=tgb_deg,
    )


@dataclass
class ScheduleEval:
    alpha_re_tgb: torch.Tensor  # (R, 3, 32) float32 in [0, 1]
    alpha_im_tgb: torch.Tensor  # (R, 3, 32) float32
    kappa_im: float
    lambda_cd: float
    eta_cd: float
    alpha_dot_rms: float


def _ramp(u: torch.Tensor) -> torch.Tensor:
    return u.clamp(0.0, 1.0)


def evaluate_deterministic_schedule(
    path_name: str,
    s: float,
    rounds: int,
    device: torch.device | str = "cpu",
) -> ScheduleEval:
    """Evaluate deterministic anisotropic / complex-detour schedule at homotopy s in [0, 1]."""
    sf = max(0.0, min(1.0, float(s)))
    r = int(rounds)
    t_idx = torch.arange(r, dtype=torch.float32, device=device).view(r, 1, 1)
    b_idx = torch.arange(32, dtype=torch.float32, device=device).view(1, 1, 32)

    alpha_im = torch.zeros((r, 3, 32), dtype=torch.float32, device=device)
    lambda_cd = 0.0
    eta_cd = 0.0
    alpha_dot_rms = 1.0
    kappa_im = 1.0

    if sf <= 0.0:
        return ScheduleEval(
            alpha_re_tgb=torch.zeros((r, 3, 32), dtype=torch.float32, device=device),
            alpha_im_tgb=alpha_im,
            kappa_im=0.15,
            lambda_cd=0.0,
            eta_cd=0.0,
            alpha_dot_rms=0.0,
        )
    if sf >= 1.0:
        return ScheduleEval(
            alpha_re_tgb=torch.ones((r, 3, 32), dtype=torch.float32, device=device),
            alpha_im_tgb=alpha_im,
            kappa_im=1.0,
            lambda_cd=0.0,
            eta_cd=0.0,
            alpha_dot_rms=0.0,
        )

    if path_name == "P0_uniform":
        alpha_re = torch.full((r, 3, 32), sf, dtype=torch.float32, device=device)
    elif path_name == "P1_bit_lsb_to_msb":
        alpha_re = _ramp(32.0 * sf - b_idx).expand(r, 3, 32).clone()
    elif path_name == "P2_bit_msb_to_lsb":
        alpha_re = _ramp(32.0 * sf - (31.0 - b_idx)).expand(r, 3, 32).clone()
    elif path_name == "P3_round_forward":
        alpha_re = _ramp(float(r) * sf - t_idx).expand(r, 3, 32).clone()
    elif path_name == "P4_round_backward":
        alpha_re = _ramp(float(r) * sf - (float(r - 1) - t_idx)).expand(r, 3, 32).clone()
    elif path_name == "P5a_gate_bool_then_add":
        alpha_re = torch.empty((r, 3, 32), dtype=torch.float32, device=device)
        v_bool = max(0.0, min(1.0, 2.0 * sf))
        v_add = max(0.0, min(1.0, 2.0 * sf - 1.0))
        alpha_re[:, 0:2, :] = v_bool
        alpha_re[:, 2, :] = v_add
    elif path_name == "P5b_gate_add_then_bool":
        alpha_re = torch.empty((r, 3, 32), dtype=torch.float32, device=device)
        v_add = max(0.0, min(1.0, 2.0 * sf))
        v_bool = max(0.0, min(1.0, 2.0 * sf - 1.0))
        alpha_re[:, 0:2, :] = v_bool
        alpha_re[:, 2, :] = v_add
    elif path_name in (
        "P6_cd_complex_detour",
        "P6_gamma02",
        "P6_gamma07",
        "P6_pure_damping",
        "P6_pure_complex",
    ):
        cfg = {
            "P6_cd_complex_detour": (0.40, 0.50, 0.15),
            "P6_gamma02": (0.20, 0.50, 0.15),
            "P6_gamma07": (0.70, 0.50, 0.15),
            "P6_pure_damping": (0.00, 0.50, 0.15),
            "P6_pure_complex": (0.40, 0.50, 0.00),
        }[path_name]
        gamma0, lambda_cd, eta_cd_0 = cfg
        alpha_re = torch.full((r, 3, 32), sf, dtype=torch.float32, device=device)
        im_val = gamma0 * math.sin(math.pi * sf)
        alpha_im = torch.full((r, 3, 32), im_val, dtype=torch.float32, device=device)
        kappa_im = 0.15 + 0.85 * sf
        eta_cd = eta_cd_0
        alpha_dot_rms = math.sqrt(1.0 + (gamma0 * math.pi * math.cos(math.pi * sf)) ** 2)
    else:
        raise ValueError(f"Unknown deterministic path '{path_name}'")

    return ScheduleEval(
        alpha_re_tgb=alpha_re,
        alpha_im_tgb=alpha_im,
        kappa_im=kappa_im,
        lambda_cd=lambda_cd,
        eta_cd=eta_cd,
        alpha_dot_rms=alpha_dot_rms,
    )


class LearnedAnisotropicField(nn.Module):
    """P7_learned_aniso: Endpoint-locked monotonic spline alpha_re + complex detour alpha_im."""

    K_BIN: int = 8

    def __init__(self, rounds: int, gamma_max: float = 0.50, lambda_cd: float = 0.50) -> None:
        super().__init__()
        enforce_v6a_fp32()
        self.rounds = int(rounds)
        self.gamma_max = float(gamma_max)
        self.lambda_cd = float(lambda_cd)
        # Zero init => softmax = 1/8 uniform => exact P0_uniform alpha(s)=s, alpha_im=0
        self.L_theta = nn.Parameter(torch.zeros((self.rounds, 3, 32, self.K_BIN), dtype=torch.float32))
        self.psi_theta = nn.Parameter(torch.zeros((self.rounds, 3, 32, self.K_BIN), dtype=torch.float32))

    def forward_at_s(self, s: float) -> ScheduleEval:
        sf = max(0.0, min(1.0, float(s)))
        dev = self.L_theta.device
        if sf <= 0.0:
            z = torch.zeros((self.rounds, 3, 32), dtype=torch.float32, device=dev)
            return ScheduleEval(z, z, kappa_im=0.15, lambda_cd=self.lambda_cd, eta_cd=0.0, alpha_dot_rms=0.0)
        if sf >= 1.0:
            ones = torch.ones((self.rounds, 3, 32), dtype=torch.float32, device=dev)
            zeros = torch.zeros((self.rounds, 3, 32), dtype=torch.float32, device=dev)
            return ScheduleEval(ones, zeros, kappa_im=1.0, lambda_cd=0.0, eta_cd=0.0, alpha_dot_rms=0.0)

        w = torch.softmax(self.L_theta, dim=-1)  # (R, 3, 32, 8)
        u = sf * float(self.K_BIN)
        m = min(self.K_BIN - 1, int(math.floor(u)))
        frac = u - float(m)

        # Cumulative knots A_m = sum_{k<m} w_k
        zeros_knot = torch.zeros((self.rounds, 3, 32, 1), dtype=torch.float32, device=dev)
        a_knots = torch.cat([zeros_knot, torch.cumsum(w, dim=-1)], dim=-1)
        alpha_re = ((1.0 - frac) * a_knots[..., m] + frac * a_knots[..., m + 1]).clamp(0.0, 1.0)

        psi_knots = torch.cat([zeros_knot, self.psi_theta], dim=-1)
        psi_s = (1.0 - frac) * psi_knots[..., m] + frac * psi_knots[..., m + 1]
        alpha_im = self.gamma_max * math.sin(math.pi * sf) * torch.tanh(psi_s)
        kappa_im = 0.15 + 0.85 * sf
        return ScheduleEval(
            alpha_re_tgb=alpha_re,
            alpha_im_tgb=alpha_im,
            kappa_im=kappa_im,
            lambda_cd=self.lambda_cd,
            eta_cd=0.0,
            alpha_dot_rms=1.0,
        )


@torch.no_grad()
def viterbi_add32_with_carry_project_fast(
    gate_proj: ExactLocalGateProjectors,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    s_s: torch.Tensor,
    c_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fast exact 32-step GPU Viterbi DP matching v6a_gl_rrr.viterbi_add32_with_carry_project."""
    orig_shape = a_s.shape
    n_grp = a_s.numel() // 32
    dev = a_s.device

    af = a_s.to(torch.float32).reshape(n_grp, 32)
    bf = b_s.to(torch.float32).reshape(n_grp, 32)
    sf = s_s.to(torch.float32).reshape(n_grp, 32)
    cf = c_s.to(torch.float32).reshape(n_grp, 32)

    triples = torch.stack([af, bf, sf], dim=-1)
    node_scores = torch.matmul(triples, gate_proj.V8_add.t())  # (N, 32, 8)

    t00 = gate_proj.t00_choices
    t11 = gate_proj.t11_choices
    idx01 = int(gate_proj.t_sing[0].item())
    idx10 = int(gate_proj.t_sing[1].item())

    # Precompute all bit-slices in parallel across 32 bits!
    s00_all = node_scores[:, :, t00]
    s00_best, s00_sub = s00_all.max(dim=-1)  # (N, 32)
    idx00_all = t00[s00_sub]

    s11_all = node_scores[:, :, t11]
    s11_best, s11_sub = s11_all.max(dim=-1)  # (N, 32)
    idx11_all = t11[s11_sub]

    ns_01 = node_scores[:, :, idx01]  # (N, 32)
    ns_10 = node_scores[:, :, idx10]  # (N, 32)

    # Bit 0: c_in = 0 strictly
    dp0 = s00_best[:, 0]
    dp1 = ns_01[:, 0]
    take1_to0: List[Optional[torch.Tensor]] = [None] * 32
    take1_to1: List[Optional[torch.Tensor]] = [None] * 32
    false_m = torch.zeros(n_grp, dtype=torch.bool, device=dev)
    take1_to0[0] = false_m
    take1_to1[0] = false_m

    for bit in range(1, 32):
        cb = cf[:, bit - 1]
        dp0_b = dp0 + cb
        dp1_b = dp1 - cb

        cand0_0 = dp0_b + s00_best[:, bit]
        cand0_1 = dp1_b + ns_10[:, bit]
        t1_0 = cand0_1 > cand0_0
        new_dp0 = torch.where(t1_0, cand0_1, cand0_0)

        cand1_0 = dp0_b + ns_01[:, bit]
        cand1_1 = dp1_b + s11_best[:, bit]
        t1_1 = cand1_1 > cand1_0
        new_dp1 = torch.where(t1_1, cand1_1, cand1_0)

        take1_to0[bit] = t1_0
        take1_to1[bit] = t1_1
        dp0, dp1 = new_dp0, new_dp1

    cur_is_1 = dp1 > dp0
    chosen_cols: List[Optional[torch.Tensor]] = [None] * 32
    carry_in_cols: List[Optional[torch.Tensor]] = [None] * 31
    for bit in range(31, -1, -1):
        prev_is_1 = torch.where(cur_is_1, take1_to1[bit], take1_to0[bit])  # type: ignore[arg-type]
        if bit >= 1:
            carry_in_cols[bit - 1] = prev_is_1
        c_idx = torch.where(
            cur_is_1,
            torch.where(prev_is_1, idx11_all[:, bit], idx01),
            torch.where(prev_is_1, idx10, idx00_all[:, bit]),
        )
        chosen_cols[bit] = c_idx
        cur_is_1 = prev_is_1

    chosen_idx = torch.stack(chosen_cols, dim=1)  # type: ignore[arg-type]
    proj_triples = gate_proj.V8_add[chosen_idx]
    c_out_spins = torch.ones((n_grp, 32), dtype=torch.float32, device=dev)
    c_in_1_31 = torch.stack(carry_in_cols, dim=1)  # type: ignore[arg-type]
    c_out_spins[:, :31] = torch.where(c_in_1_31, -1.0, 1.0)
    return (
        proj_triples[..., 0].reshape(orig_shape),
        proj_triples[..., 1].reshape(orig_shape),
        proj_triples[..., 2].reshape(orig_shape),
        c_out_spins.reshape(orig_shape),
    )


class AnisotropicGLRRREngine(nn.Module):
    """Pure FP32 Dual-Channel Anisotropic GL-RRR Engine supporting main SHA-256 and control cipher."""

    BETA_MIN: float = 0.65
    BETA_MAX: float = 1.15

    def __init__(
        self,
        rounds: int = 18,
        variant: str = "main",
        ctrl_params: Optional[ControlCipherParams] = None,
        bp_iters: int = 7,
        bp_damping: float = 0.72,
    ) -> None:
        super().__init__()
        enforce_v6a_fp32()
        self.rounds = int(rounds)
        self.variant = str(variant)
        self.ctrl_params = ctrl_params
        if self.variant == "main":
            self.spec = build_v6a_circuit_spec(self.rounds)
            k_words = get_k_tensor(self.rounds, device=None)
        elif self.variant == "control":
            if ctrl_params is None:
                raise ValueError("ctrl_params required when variant='control'")
            self.spec = build_control_circuit_spec(self.rounds, ctrl_params)
            k_words = get_ctrl_k_tensor(self.rounds, ctrl_params, device=None)
        else:
            raise ValueError(f"Unknown variant '{variant}'")

        self.coord_map = build_aniso_coord_map(self.spec)
        self.bp_iters = max(2, int(bp_iters))
        self.bp_damping = float(bp_damping)
        self.gate_proj = ExactLocalGateProjectors()

        self.register_buffer("P_T_fp64", build_phase_projection_matrix(), persistent=False)
        k_bits = words_to_bits(k_words).to(torch.float64)
        self.register_buffer("theta_K_fp64", torch.matmul(k_bits, self.P_T_fp64), persistent=False)

        s = self.spec
        self.register_buffer("chk_vars", s.chk_vars, persistent=False)
        self.register_buffer("chk_valid", s.chk_valid, persistent=False)
        self.register_buffer("chk_valid_i64", s.chk_valid.to(torch.int64), persistent=False)
        self.register_buffer("var_to_chk_idx", s.var_to_chk_idx, persistent=False)
        self.register_buffer("var_to_chk_slot", s.var_to_chk_slot, persistent=False)
        self.register_buffer("var_to_chk_var", s.var_to_chk_var, persistent=False)
        self.register_buffer("var_to_chk_edge_flat", s.var_to_chk_idx * 4 + s.var_to_chk_slot, persistent=False)

        self.register_buffer("ch_quads", s.ch_quads, persistent=False)
        self.register_buffer("maj_quads", s.maj_quads, persistent=False)
        self.register_buffer("add_word_quads", s.add_word_quads, persistent=False)
        self.register_buffer("nl_rep_to_var_bit", s.nl_rep_to_var_bit, persistent=False)
        self.register_buffer("var_total_degree", s.var_total_degree, persistent=False)
        self.register_buffer("rep_to_tgb_idx", self.coord_map.rep_to_tgb_idx, persistent=False)
        self.register_buffer("add_round_idx", self.coord_map.add_round_idx, persistent=False)
        self.register_buffer("add_slot_id", self.coord_map.add_slot_id, persistent=False)

        self.n_lin_rep = s.n_var_bits
        self.n_ch = self.coord_map.n_ch
        self.n_maj = self.coord_map.n_maj
        self.n_add = self.coord_map.n_add
        self.n_nl_rep = self.coord_map.n_nl_rep
        self.n_rep_bits = self.n_lin_rep + self.n_nl_rep
        self.drift_monitor = DriftMonitor(slope_thresh=1e-2, r2_thresh=0.85, max_history=64)

    def reset_drift(self) -> None:
        self.drift_monitor = DriftMonitor(slope_thresh=1e-2, r2_thresh=0.85, max_history=64)

    def forward_digest(self, w16: torch.Tensor) -> torch.Tensor:
        if self.variant == "main":
            return sha256_digest(w16, rounds=self.rounds)
        assert self.ctrl_params is not None
        return control_digest(w16, rounds=self.rounds, params=self.ctrl_params)

    def broadcast_tgb_to_rep(self, field_tgb: torch.Tensor) -> torch.Tensor:
        return field_tgb.reshape(-1)[self.rep_to_tgb_idx]

    @torch.no_grad()
    def init_replicas_from_vars(self, var_spins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vf = var_spins.to(torch.float32)
        x_re = torch.cat([vf, vf[:, self.nl_rep_to_var_bit]], dim=1)
        x_im = torch.zeros_like(x_re)
        return x_re, x_im

    def project_A_vars(
        self,
        x_re: torch.Tensor,
        var_clamp_mask: torch.Tensor,
        var_clamp_vals: torch.Tensor,
    ) -> torch.Tensor:
        xf = x_re.to(torch.float32)
        bsz = xf.shape[0]
        r_lin = xf[:, : self.n_lin_rep]
        r_nl = xf[:, self.n_lin_rep :]
        nl_idx = self.nl_rep_to_var_bit.unsqueeze(0).expand(bsz, -1)
        summed = r_lin.clone()
        summed.scatter_add_(1, nl_idx, r_nl)
        consensus = summed / self.var_total_degree.unsqueeze(0)
        m = var_clamp_mask.to(torch.float32)
        v = var_clamp_vals.to(torch.float32)
        return m * v + (1.0 - m) * consensus

    def project_A_adj_vars(
        self,
        x_im: torch.Tensor,
        var_clamp_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Adjoint Set-A consensus projection zeroing clamped boundary bits."""
        xf = x_im.to(torch.float32)
        bsz = xf.shape[0]
        r_lin = xf[:, : self.n_lin_rep]
        r_nl = xf[:, self.n_lin_rep :]
        nl_idx = self.nl_rep_to_var_bit.unsqueeze(0).expand(bsz, -1)
        summed = r_lin.clone()
        summed.scatter_add_(1, nl_idx, r_nl)
        consensus = summed / self.var_total_degree.unsqueeze(0)
        free_m = 1.0 - var_clamp_mask.to(torch.float32)
        return free_m * consensus

    def project_A_dual(
        self,
        x_re: torch.Tensor,
        x_im: torch.Tensor,
        var_clamp_mask: torch.Tensor,
        var_clamp_vals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v_re = self.project_A_vars(x_re, var_clamp_mask, var_clamp_vals)
        y_re = torch.cat([v_re, v_re[:, self.nl_rep_to_var_bit]], dim=1)
        v_im = self.project_A_adj_vars(x_im, var_clamp_mask)
        y_im = torch.cat([v_im, v_im[:, self.nl_rep_to_var_bit]], dim=1)
        return y_re, y_im

    @torch.no_grad()
    def project_linear_skeleton(
        self,
        r_lin: torch.Tensor,
        var_clamp_mask: torch.Tensor,
        var_clamp_vals: torch.Tensor,
        check_early_exit: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Global F_2 projection P_B^(lin): Zero-Syndrome Guard + Batched Damped Min-Sum BP."""
        rf = r_lin.to(torch.float32)
        bsz, _ = rf.shape
        dev = rf.device
        n_chk = int(self.chk_vars.shape[0])

        clamp_m = var_clamp_mask.to(torch.float32) > 0.5
        clamp_bits = (var_clamp_vals.to(torch.float32) < 0.0).to(torch.int64)
        hard_bits = (rf < 0.0).to(torch.int64)

        chk_bits = hard_bits[:, self.chk_vars]
        syn = ((chk_bits * self.chk_valid_i64.unsqueeze(0)).sum(dim=-1) & 1)
        n_syn = syn.sum(dim=-1)
        clamp_mismatch = (clamp_m & (hard_bits != clamp_bits)).sum(dim=-1)
        already_exact = (n_syn == 0) & (clamp_mismatch == 0)

        if check_early_exit and bool(already_exact.all().item()):
            return torch.where(hard_bits > 0, -1.0, 1.0), already_exact

        llr_ch = (2.0 * rf).clamp(-8.0, 8.0)
        llr_clamped = torch.where(clamp_m, 100.0 * var_clamp_vals.to(torch.float32), llr_ch)
        q_tot = llr_clamped.clone()
        msg_c2v = torch.zeros((bsz, n_chk, 4), dtype=torch.float32, device=dev)
        slot_arange = torch.arange(4, device=dev).view(1, 1, 4)

        chk_valid_b = self.chk_valid.unsqueeze(0) > 0.5
        edge_var_idx = self.var_to_chk_var.unsqueeze(0).expand(bsz, -1)
        edge_flat_idx = self.var_to_chk_edge_flat

        for _ in range(self.bp_iters):
            q_gather = q_tot[:, self.chk_vars]
            v2c = (q_gather - msg_c2v).clamp(-24.0, 24.0)
            v2c = torch.where(chk_valid_b, v2c, 50.0)

            sgn = torch.where(v2c >= 0.0, 1.0, -1.0)
            out_sgn = sgn.prod(dim=-1, keepdim=True) * sgn

            top2_vals, top2_idx = torch.topk(v2c.abs(), k=2, dim=-1, largest=False)
            min1 = top2_vals[..., 0:1]
            min2 = top2_vals[..., 1:2]
            out_mag = torch.where(slot_arange == top2_idx[..., 0:1], min2, min1)

            raw_c2v = 0.78 * out_sgn * out_mag * self.chk_valid.unsqueeze(0)
            msg_c2v = self.bp_damping * raw_c2v + (1.0 - self.bp_damping) * msg_c2v

            msg_edges = msg_c2v.reshape(bsz, -1)[:, edge_flat_idx]
            q_tot = llr_clamped.clone()
            q_tot.scatter_add_(1, edge_var_idx, msg_edges)

        bp_bits = (q_tot < 0.0).to(torch.int64)
        bp_bits = torch.where(clamp_m, clamp_bits, bp_bits)
        bp_syn = (((bp_bits[:, self.chk_vars] * self.chk_valid_i64.unsqueeze(0)).sum(dim=-1) & 1).sum(dim=-1))
        bp_exact = bp_syn == 0

        final_bits = torch.where(already_exact.unsqueeze(-1), hard_bits, bp_bits)
        y_lin = torch.where(final_bits > 0, -1.0, 1.0)
        return y_lin, (already_exact | bp_exact)

    @torch.no_grad()
    def project_B_nl_exact(self, r_nl: torch.Tensor) -> torch.Tensor:
        """Exact local nonlinear projections (Ch, Maj, ADD32+Carry) on r_nl."""
        rf = r_nl.to(torch.float32)
        bsz = rf.shape[0]
        c0 = 0
        c1 = c0 + self.n_ch * 4
        c2 = c1 + self.n_maj * 4

        ch_p = self.gate_proj.project_ch(rf[:, c0:c1].reshape(bsz, self.n_ch, 4)).reshape(bsz, -1)
        maj_p = self.gate_proj.project_maj(rf[:, c1:c2].reshape(bsz, self.n_maj, 4)).reshape(bsz, -1)

        add_in = rf[:, c2:].reshape(bsz, self.n_add, 4, 32)
        a_p, b_p, s_p, c_p = viterbi_add32_with_carry_project_fast(
            self.gate_proj,
            add_in[:, :, 0, :],
            add_in[:, :, 1, :],
            add_in[:, :, 2, :],
            add_in[:, :, 3, :],
        )
        add_p = torch.stack([a_p, b_p, s_p, c_p], dim=2).reshape(bsz, -1)
        return torch.cat([ch_p, maj_p, add_p], dim=1)

    def combine_B_dual(
        self,
        r_re: torch.Tensor,
        r_im: torch.Tensor,
        y_lin_re: torch.Tensor,
        p_b_nl: torch.Tensor,
        sched: ScheduleEval,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combine exact detached projections with differentiable 2x2 real-matrix complex rotation."""
        a_re = self.broadcast_tgb_to_rep(sched.alpha_re_tgb).unsqueeze(0)
        a_im = self.broadcast_tgb_to_rep(sched.alpha_im_tgb).unsqueeze(0)
        one_m_k = 1.0 - float(sched.kappa_im)
        lam = float(sched.lambda_cd)

        r_re_nl = r_re[:, self.n_lin_rep :]
        r_im_nl = r_im[:, self.n_lin_rep :]
        delta_nl = p_b_nl - r_re_nl

        # Real 2x2 matrix representation of (alpha_re + i alpha_im) * (delta_nl - i * lambda_cd * r_im_nl)
        y_nl_re = r_re_nl + a_re * delta_nl + lam * (a_im * r_im_nl)
        y_nl_im = one_m_k * r_im_nl + a_im * delta_nl - lam * (a_re * r_im_nl)

        y_lin_im = one_m_k * r_im[:, : self.n_lin_rep]
        y_b_re = torch.cat([y_lin_re, y_nl_re], dim=1)
        y_b_im = torch.cat([y_lin_im, y_nl_im], dim=1)
        return y_b_re, y_b_im

    @torch.no_grad()
    def step_dual(
        self,
        x_re: torch.Tensor,
        x_im: torch.Tensor,
        var_clamp_mask: torch.Tensor,
        var_clamp_vals: torch.Tensor,
        sched: ScheduleEval,
        beta: float = 0.85,
        x_re_prev: Optional[torch.Tensor] = None,
        record_drift: bool = True,
        check_early_exit: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Execute one dual-channel anisotropic / counter-diabatic GL-RRR step."""
        beta_c = max(self.BETA_MIN, min(self.BETA_MAX, float(beta)))
        xf_re = x_re.to(torch.float32)
        xf_im = x_im.to(torch.float32)

        y_a_re, y_a_im = self.project_A_dual(xf_re, xf_im, var_clamp_mask, var_clamp_vals)
        r_re = 2.0 * y_a_re - xf_re
        r_im = 2.0 * y_a_im - xf_im

        y_lin_re, valid_lin = self.project_linear_skeleton(
            r_re[:, : self.n_lin_rep],
            var_clamp_mask,
            var_clamp_vals,
            check_early_exit=check_early_exit,
        )
        p_b_nl = self.project_B_nl_exact(r_re[:, self.n_lin_rep :])
        y_b_re, y_b_im = self.combine_B_dual(r_re, r_im, y_lin_re, p_b_nl, sched)

        diff_re = y_b_re - y_a_re
        diff_im = y_b_im - y_a_im
        x_next_re = xf_re + beta_c * diff_re
        if sched.eta_cd > 0.0 and x_re_prev is not None:
            free_var = 1.0 - var_clamp_mask.to(torch.float32)
            free_rep = torch.cat([free_var, free_var[:, self.nl_rep_to_var_bit]], dim=1)
            vel = xf_re - x_re_prev.to(torch.float32)
            x_next_re = x_next_re - float(sched.eta_cd) * float(sched.alpha_dot_rms) * free_rep * vel

        x_next_im = xf_im + beta_c * diff_im

        inv_sqrt_d = 1.0 / math.sqrt(float(xf_re.shape[-1]))
        gl_gap = torch.linalg.vector_norm(diff_re, dim=-1) * inv_sqrt_d
        x_rms = torch.linalg.vector_norm(x_next_re, dim=-1) * inv_sqrt_d
        if record_drift:
            self.drift_monitor.record(x_rms)

        return x_next_re, x_next_im, {
            "y_a_re": y_a_re,
            "y_b_re": y_b_re,
            "y_a_im": y_a_im,
            "y_b_im": y_b_im,
            "gl_gap": gl_gap,
            "x_rms": x_rms,
            "valid_lin": valid_lin,
        }

    def step_differentiable(
        self,
        x_re: torch.Tensor,
        x_im: torch.Tensor,
        var_clamp_mask: torch.Tensor,
        var_clamp_vals: torch.Tensor,
        sched: ScheduleEval,
        beta: float = 0.85,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable STE-RRR unroll step for P7 training (pure FP32)."""
        beta_c = max(self.BETA_MIN, min(self.BETA_MAX, float(beta)))
        y_a_re, y_a_im = self.project_A_dual(x_re, x_im, var_clamp_mask, var_clamp_vals)
        r_re = 2.0 * y_a_re - x_re
        r_im = 2.0 * y_a_im - x_im

        with torch.no_grad():
            y_lin_re, _ = self.project_linear_skeleton(
                r_re[:, : self.n_lin_rep],
                var_clamp_mask,
                var_clamp_vals,
                check_early_exit=False,
            )
            p_b_nl = self.project_B_nl_exact(r_re[:, self.n_lin_rep :])

        y_b_re, y_b_im = self.combine_B_dual(r_re, r_im, y_lin_re, p_b_nl, sched)
        diff_re = y_b_re - y_a_re
        diff_im = y_b_im - y_a_im
        x_next_re = x_re + beta_c * diff_re
        x_next_im = x_im + beta_c * diff_im

        n_d = float(x_re.shape[-1])
        gl_gap_sq = (diff_re.square().sum(dim=-1) + diff_im.square().sum(dim=-1)) / n_d
        step_delta_norm = torch.linalg.vector_norm(x_next_re - x_re, dim=-1) / math.sqrt(n_d)
        return x_next_re, x_next_im, gl_gap_sq, step_delta_norm

    def compute_aniso_hom_energy_terms(
        self,
        v_disc: torch.Tensor,
        alpha_re_tgb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute exact discrete violations and differentiable alpha_re-weighted E_hom."""
        bsz = v_disc.shape[0]
        s = self.spec
        bits_disc = (v_disc < 0.0).to(torch.int64)

        lin_viol = (
            ((bits_disc[:, self.chk_vars] * self.chk_valid_i64.unsqueeze(0)).sum(dim=-1) & 1) != 0
        )
        lin_viol_rate = lin_viol.to(torch.float32).mean(dim=-1)

        ch_b = bits_disc[:, self.ch_quads]
        ch_pred = (ch_b[..., 0] & ch_b[..., 1]) ^ ((1 - ch_b[..., 0]) & ch_b[..., 2])
        ch_mis = (ch_pred != ch_b[..., 3]).to(torch.float32)  # (B, N_ch)

        maj_b = bits_disc[:, self.maj_quads]
        maj_pred = (maj_b[..., 0] & maj_b[..., 1]) ^ (maj_b[..., 0] & maj_b[..., 2]) ^ (maj_b[..., 1] & maj_b[..., 2])
        maj_mis = (maj_pred != maj_b[..., 3]).to(torch.float32)  # (B, N_maj)

        words_int = bits_to_words(bits_disc.view(bsz, s.n_var_words, 32))
        q_w = self.add_word_quads
        wa, wb, ws, wc = words_int[:, q_w[:, 0]], words_int[:, q_w[:, 1]], words_int[:, q_w[:, 2]], words_int[:, q_w[:, 3]]
        sum_err = words_to_bits(((wa + wb) & MASK32) ^ ws).to(torch.float32)  # (B, N_add, 32)
        carry_err = words_to_bits((wc ^ _compute_add32_carry_words(wa, wb)) & 0x7FFFFFFF)[..., :31].to(torch.float32)

        a_ch = alpha_re_tgb[:, 0, :].reshape(1, self.n_ch)
        a_maj = alpha_re_tgb[:, 1, :].reshape(1, self.n_maj)
        a_add = alpha_re_tgb[self.add_round_idx, 2, :]  # (N_add, 32)

        e_ch_w = (ch_mis * a_ch).mean(dim=-1)
        e_maj_w = (maj_mis * a_maj).mean(dim=-1)
        e_add_w = 0.5 * ((sum_err * a_add.unsqueeze(0)).mean(dim=(-2, -1)) + (carry_err * a_add[:, :31].unsqueeze(0)).mean(dim=(-2, -1)))
        e_hom = lin_viol_rate + (e_ch_w + e_maj_w + e_add_w) / 3.0

        ch_err = ch_mis.mean(dim=-1)
        maj_err = maj_mis.mean(dim=-1)
        add_err = 0.5 * (sum_err.mean(dim=(-2, -1)) + carry_err.mean(dim=(-2, -1)))
        nl_err_rate = (ch_err + maj_err + add_err) / 3.0
        e_full = lin_viol_rate + nl_err_rate

        disc_viol_cnt = (
            lin_viol.to(torch.int64).sum(dim=-1)
            + ch_mis.to(torch.int64).sum(dim=-1)
            + maj_mis.to(torch.int64).sum(dim=-1)
            + sum_err.to(torch.int64).sum(dim=(-2, -1))
            + carry_err.to(torch.int64).sum(dim=(-2, -1))
        )

        return {
            "lin_viol_rate": lin_viol_rate,
            "e_hom": e_hom,
            "e_full": e_full,
            "ch_err": ch_err,
            "maj_err": maj_err,
            "add_err": add_err,
            "nl_err_rate": nl_err_rate,
            "disc_viol_cnt": disc_viol_cnt,
            "ch_mis": ch_mis,
            "maj_mis": maj_mis,
            "sum_err": sum_err,
            "carry_err": carry_err,
        }

    @torch.no_grad()
    def readout_shadow_aniso(
        self,
        x_re: torch.Tensor,
        boundary: V6ABoundary,
        alpha_re_tgb: torch.Tensor,
    ) -> Dict[str, Any]:
        """Extract shadow P_A(x_re) and compute honest instantaneous metrics."""
        v_a = self.project_A_vars(x_re, boundary.var_mask, boundary.var_vals)
        cm = boundary.var_mask > 0.5
        v_disc = torch.where(cm, boundary.var_vals, torch.where(v_a < 0.0, -1.0, 1.0))

        bsz = v_disc.shape[0]
        r = self.rounds
        s = self.spec
        words_spins = v_disc.view(bsz, s.n_var_words, 32)
        a_full = words_spins[:, s.a_start : s.a_start + r + 4]
        e_full = words_spins[:, s.e_start : s.e_start + r + 4]
        w_full = words_spins[:, s.w_start : s.w_start + r]

        if self.variant == "main":
            e_m1, e_m32, e_all32 = compute_true_2adic_energy(
                a_full, e_full, w_full, self.P_T_fp64, self.theta_K_fp64,
                rounds=r, include_tail_e=True, return_all32=True,
            )
        else:
            assert self.ctrl_params is not None
            e_m1, e_m32, e_all32 = compute_control_2adic_energy(
                a_full, e_full, w_full, self.P_T_fp64, self.theta_K_fp64,
                rounds=r, params=self.ctrl_params, include_tail_e=True,
            )

        terms = self.compute_aniso_hom_energy_terms(v_disc, alpha_re_tgb)

        w_pred_bits = (w_full < 0.0).to(torch.int64)
        if r < 16:
            w16_cand = torch.zeros((bsz, 16), dtype=torch.int64, device=v_disc.device)
            w16_cand[:, :r] = bits_to_words(w_pred_bits[:, :r])
        else:
            w16_cand = bits_to_words(w_pred_bits[:, :16])
        w16_m_int = (boundary.W16_mask_bits > 0.5).to(torch.int64)
        w16_c_bits = words_to_bits(boundary.W16_clamped_words)
        w16_cand = bits_to_words(torch.where(w16_m_int > 0, w16_c_bits, words_to_bits(w16_cand)))

        dig_cand = self.forward_digest(w16_cand)
        exact_hits_mask = (dig_cand == boundary.target_digests).all(dim=-1)
        slope, flag = self.drift_monitor.summarize()

        return {
            "v_a": v_a,
            "v_disc": v_disc,
            "per_traj_e_hom": terms["e_hom"],
            "per_traj_e_full": terms["e_full"],
            "per_traj_lin_viol": terms["lin_viol_rate"],
            "per_traj_nl_err": terms["nl_err_rate"],
            "per_traj_m1": e_m1,
            "per_traj_m32": e_m32,
            "per_traj_all32": e_all32,
            "per_traj_disc_viol": terms["disc_viol_cnt"],
            "exact_hits_mask": exact_hits_mask,
            "exact_hits": int(exact_hits_mask.sum().item()),
            "per_traj_drift_slope": slope,
            "per_traj_drift_flag": flag,
            "terms": terms,
        }
