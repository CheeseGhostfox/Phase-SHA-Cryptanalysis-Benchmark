"""Colab GPU Full Experiment Runner for P8 Anisotropic & Complex-Plane Homotopy.

Executes all SPEC.md protocols in strict pure FP32:
  1. Learned anisotropic spline + complex detour (P7_learned_aniso) training per (R, variant)
  2. Full quasi-static s-sweep across R in {4, 6, 8, 12, 18, 24, 64} x {main, control} x {P0..P7} + P6 ablations
  3. 2D boundary relaxation comparison (fixed mask_m=256 vs interleaved 64->256) on R in {4, 8, 18}
  4. Single-component singularity probe + counterfactual cut-set ablation & greedy minimal cut-set search
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sha256_ref import sha256_digest  # noqa: E402
from v6a import assert_strict_fp32, enforce_v6a_fp32  # noqa: E402
from v6a.v6a_homotopy import (  # noqa: E402
    V6ABoundary,
    build_v6a_boundary,
    solve_alpha0_gf2_linear_skeleton,
)
from v11_discovery.p8_anisotropic_homotopy.anisotropic_homotopy import (  # noqa: E402
    ADD_SLOT_NAMES,
    P6_ABLATION_PATHS,
    PRIMARY_PATHS,
    AnisotropicGLRRREngine,
    LearnedAnisotropicField,
    ScheduleEval,
    evaluate_deterministic_schedule,
)
from v11_discovery.p8_anisotropic_homotopy.control_cipher import (  # noqa: E402
    ControlCipherParams,
    build_control_boundary,
    control_digest,
    make_control_cipher_params,
)

ALL_ROUNDS: Tuple[int, ...] = (4, 6, 8, 12, 18, 24, 64)
S_GRID_COUNT: int = 26  # s_k = k / 25, k = 0..25
T_RELAX: int = 8


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / float(n)
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / float(n - 1)
    return m, math.sqrt(max(0.0, var))


def build_variant_boundary(
    digests: torch.Tensor,
    rounds: int,
    variant: str,
    ctrl_params: ControlCipherParams,
    mask_m: int = 256,
) -> V6ABoundary:
    if variant == "main":
        return build_v6a_boundary(
            digests, rounds=rounds, domain_mode="free_msg", mask_m=mask_m, allow_peel_back=False
        )
    return build_control_boundary(
        digests, rounds=rounds, params=ctrl_params, domain_mode="free_msg", mask_m=mask_m
    )


def sample_cached_alpha0_batch(
    engine: AnisotropicGLRRREngine,
    rounds: int,
    variant: str,
    ctrl_params: ControlCipherParams,
    n_digests: int,
    k_traj: int,
    seed: int,
    mask_m: int = 256,
    device: torch.device = torch.device("cpu"),
) -> Tuple[V6ABoundary, torch.Tensor, torch.Tensor, int]:
    """Generate held-out digests and solve exact alpha=0 GF(2) linear skeleton once per (R, variant)."""
    gen_msg = torch.Generator(device="cpu").manual_seed(seed)
    w16_true = torch.randint(0, 1 << 30, (n_digests, 16), dtype=torch.int64, generator=gen_msg)
    if variant == "main":
        dig = sha256_digest(w16_true, rounds=rounds).to(device)
    else:
        dig = control_digest(w16_true, rounds=rounds, params=ctrl_params).to(device)

    bnd_single = build_variant_boundary(dig, rounds, variant, ctrl_params, mask_m=mask_m)
    var_spins_list: List[torch.Tensor] = []
    nullities: List[int] = []
    for d in range(n_digests):
        sp, viol, nul = solve_alpha0_gf2_linear_skeleton(
            engine.spec,
            bnd_single.var_mask[d],
            bnd_single.var_vals[d],
            num_samples=k_traj,
            generator=torch.Generator(device="cpu").manual_seed(seed + 1000 + d),
        )
        if viol != 0:
            raise RuntimeError(f"GF(2) solver failed with {viol} violations on R={rounds} {variant}")
        var_spins_list.append(sp.to(device))
        nullities.append(nul)

    var_spins = torch.cat(var_spins_list, dim=0)  # (N_digests * K_traj, n_var_bits)
    x0_re, x0_im = engine.init_replicas_from_vars(var_spins)
    bnd = bnd_single.repeat_for_candidates(k_traj)
    return bnd, x0_re, x0_im, int(nullities[0])


def train_p7_learned_field(
    engine: AnisotropicGLRRREngine,
    rounds: int,
    variant: str,
    ctrl_params: ControlCipherParams,
    device: torch.device,
    train_steps: int = 100,
) -> Tuple[LearnedAnisotropicField, Dict[str, Any]]:
    """Train P7_learned_aniso via pure FP32 unrolled STE-RRR on disjoint online digests."""
    t_start = time.time()
    n_dig = 4 if rounds == 64 else 8
    k_traj = 8
    field = LearnedAnisotropicField(rounds=rounds, gamma_max=0.50, lambda_cd=0.50).to(device)
    assert_strict_fp32(field)
    opt = torch.optim.Adam(field.parameters(), lr=3e-2)

    l_grid = 8
    t_unroll = 4
    s_levels = [float(l_idx) / float(l_grid) for l_idx in range(1, l_grid + 1)]

    # Pre-solve a pool of 4 disjoint training batches (rotated each step) so CPU RREF doesn't stall GPU training
    pool_size = 4
    train_pool: List[Tuple[V6ABoundary, torch.Tensor, torch.Tensor]] = []
    base_seed = 900000 + rounds * 1000 + (0 if variant == "main" else 500)
    for p_i in range(pool_size):
        bnd_p, x0_re_p, x0_im_p, _ = sample_cached_alpha0_batch(
            engine, rounds, variant, ctrl_params, n_dig, k_traj, seed=base_seed + p_i * 37, mask_m=256, device=device
        )
        train_pool.append((bnd_p, x0_re_p, x0_im_p))

    loss_hist: List[float] = []
    for step_i in range(train_steps):
        bnd, x_re, x_im = train_pool[step_i % pool_size]
        opt.zero_grad(set_to_none=True)

        e_hom_levels: List[torch.Tensor] = []
        gap_sq_terms: List[torch.Tensor] = []
        step_norms: List[torch.Tensor] = []

        for sv in s_levels:
            sc = field.forward_at_s(sv)
            last_gap_sq = torch.zeros(x_re.shape[0], dtype=torch.float32, device=device)
            norms_acc = []
            for _ in range(t_unroll):
                x_re, x_im, g_sq, d_norm = engine.step_differentiable(
                    x_re, x_im, bnd.var_mask, bnd.var_vals, sc, beta=0.85
                )
                last_gap_sq = g_sq
                norms_acc.append(d_norm)
            with torch.no_grad():
                v_a = engine.project_A_vars(x_re, bnd.var_mask, bnd.var_vals)
                cm = bnd.var_mask > 0.5
                v_disc = torch.where(cm, bnd.var_vals, torch.where(v_a < 0.0, -1.0, 1.0))
            terms = engine.compute_aniso_hom_energy_terms(v_disc, sc.alpha_re_tgb)
            e_hom_levels.append(terms["e_hom"].mean())
            gap_sq_terms.append(last_gap_sq.mean())
            step_norms.append(torch.stack(norms_acc, dim=0))

        all_norms = torch.cat(step_norms, dim=0)  # (L*T, B)
        var_step = all_norms.var(dim=0, unbiased=False).mean()
        loss_main = sum((g + 0.5 * e) for g, e in zip(gap_sq_terms, e_hom_levels)) / float(l_grid)
        spike_pen = torch.zeros((), dtype=torch.float32, device=device)
        for l_idx in range(l_grid - 1):
            spike_pen = spike_pen + torch.relu(e_hom_levels[l_idx + 1] - 1.05 * e_hom_levels[l_idx]).square()
        loss = loss_main + 2.0 * spike_pen + 0.25 * var_step
        loss.backward()
        opt.step()
        loss_hist.append(float(loss.detach().item()))

    # Summarize learned field structure at s=0.5
    with torch.no_grad():
        sc05 = field.forward_at_s(0.5)
        ar = sc05.alpha_re_tgb
        ai = sc05.alpha_im_tgb
        field_summary = {
            "train_steps": train_steps,
            "train_time_sec": round(time.time() - t_start, 2),
            "loss_init": loss_hist[0] if loss_hist else None,
            "loss_final": loss_hist[-1] if loss_hist else None,
            "loss_min": min(loss_hist) if loss_hist else None,
            "s05_alpha_re_mean": float(ar.mean().item()),
            "s05_alpha_re_by_gate": [float(ar[:, g, :].mean().item()) for g in range(3)],
            "s05_alpha_re_lsb8": float(ar[:, :, :8].mean().item()),
            "s05_alpha_re_msb8": float(ar[:, :, 24:].mean().item()),
            "s05_alpha_re_early_rounds": float(ar[: max(1, rounds // 4), :, :].mean().item()),
            "s05_alpha_re_late_rounds": float(ar[-max(1, rounds // 4) :, :, :].mean().item()),
            "s05_alpha_im_mean": float(ai.mean().item()),
            "s05_alpha_im_abs_mean": float(ai.abs().mean().item()),
        }
    return field, field_summary


def detect_bifurcation_from_curves(
    s_vals: List[float],
    alpha_tgb_vals: List[float],
    alpha_rep_vals: List[float],
    e_hom_vals: List[float],
    gl_gap_vals: List[float],
    drift_slope_vals: List[float],
    lin_viol_vals: List[float],
    skip_zero_baseline_spike: bool = False,
    instability_only: bool = False,
) -> Dict[str, Any]:
    """Apply SPEC 4.2 bifurcation trigger criteria & report (s_c, alpha_bar_c)."""
    for k in range(1, len(s_vals)):
        reasons: List[str] = []
        if not instability_only:
            allow_spike = (not skip_zero_baseline_spike) or (e_hom_vals[k - 1] > 1e-6 and k >= 2)
            if allow_spike and e_hom_vals[k] > 1.25 * e_hom_vals[k - 1] and (e_hom_vals[k] - e_hom_vals[k - 1]) > 0.005:
                reasons.append("energy_spike")
        if gl_gap_vals[k] > 0.14 or drift_slope_vals[k] > 1e-2:
            reasons.append("gl_gap_or_drift")
        if lin_viol_vals[k] > 0.02:
            reasons.append("linear_rupture")
        if reasons:
            return {
                "bifurcated": True,
                "s_c": s_vals[k],
                "k_c": k,
                "alpha_bar_c": alpha_tgb_vals[k],
                "alpha_bar_rep_c": alpha_rep_vals[k],
                "reasons": reasons,
            }
    return {
        "bifurcated": False,
        "s_c": None,
        "k_c": None,
        "alpha_bar_c": None,
        "alpha_bar_rep_c": None,
        "reasons": [],
    }


@torch.no_grad()
def run_single_path_sweep(
    engine: AnisotropicGLRRREngine,
    bnd: V6ABoundary,
    x0_re: torch.Tensor,
    x0_im: torch.Tensor,
    path_name: str,
    n_digests: int,
    k_traj: int,
    p7_field: Optional[LearnedAnisotropicField] = None,
    beta: float = 0.85,
) -> Dict[str, Any]:
    """Run 26-level quasi-static s-sweep for one (R, variant, path)."""
    rounds = engine.rounds
    device = x0_re.device
    engine.reset_drift()

    x_re = x0_re.clone()
    x_im = x0_im.clone()
    x_re_prev = x_re.clone()

    curve: List[Dict[str, Any]] = []
    # Per-digest arrays over k=0..25
    dig_e_hom = [[] for _ in range(n_digests)]
    dig_gl_gap = [[] for _ in range(n_digests)]
    dig_drift = [[] for _ in range(n_digests)]
    dig_lin = [[] for _ in range(n_digests)]
    s_vals: List[float] = []
    a_tgb_vals: List[float] = []
    a_rep_vals: List[float] = []

    def _get_sched(s_val: float) -> ScheduleEval:
        if path_name == "P7_learned_aniso":
            assert p7_field is not None
            return p7_field.forward_at_s(s_val)
        return evaluate_deterministic_schedule(path_name, s_val, rounds, device=device)

    for k in range(S_GRID_COUNT):
        sv = float(k) / float(S_GRID_COUNT - 1)
        sched = _get_sched(sv)
        last_gap = torch.zeros(x_re.shape[0], dtype=torch.float32, device=device)
        for _ in range(T_RELAX):
            nxt_re, nxt_im, info = engine.step_dual(
                x_re,
                x_im,
                bnd.var_mask,
                bnd.var_vals,
                sched,
                beta=beta,
                x_re_prev=x_re_prev,
                record_drift=True,
                check_early_exit=(k == 0),
            )
            x_re_prev = x_re
            x_re, x_im = nxt_re, nxt_im
            last_gap = info["gl_gap"]

        ro = engine.readout_shadow_aniso(x_re, bnd, sched.alpha_re_tgb)
        if k == 0:
            max_lin0 = float(ro["per_traj_lin_viol"].max().item())
            if max_lin0 > 1e-6:
                raise RuntimeError(f"Expected 0 linear viol at s=0, got {max_lin0}")

        a_bar_tgb = float(sched.alpha_re_tgb.mean().item())
        a_bar_rep = float(engine.broadcast_tgb_to_rep(sched.alpha_re_tgb).mean().item())
        s_vals.append(sv)
        a_tgb_vals.append(a_bar_tgb)
        a_rep_vals.append(a_bar_rep)

        e_hom_d = ro["per_traj_e_hom"].view(n_digests, k_traj).mean(dim=1).tolist()
        gl_gap_d = last_gap.view(n_digests, k_traj).mean(dim=1).tolist()
        drift_d = ro["per_traj_drift_slope"].view(n_digests, k_traj).mean(dim=1).tolist()
        lin_d = ro["per_traj_lin_viol"].view(n_digests, k_traj).mean(dim=1).tolist()
        for d in range(n_digests):
            dig_e_hom[d].append(float(e_hom_d[d]))
            dig_gl_gap[d].append(float(gl_gap_d[d]))
            dig_drift[d].append(float(drift_d[d]))
            dig_lin[d].append(float(lin_d[d]))

        terms = ro["terms"]
        curve.append(
            {
                "s": round(sv, 4),
                "alpha_bar_tgb": round(a_bar_tgb, 6),
                "alpha_bar_rep": round(a_bar_rep, 6),
                "alpha_im_abs_mean": round(float(sched.alpha_im_tgb.abs().mean().item()), 6),
                "E_hom": float(ro["per_traj_e_hom"].mean().item()),
                "E_hom_min": float(ro["per_traj_e_hom"].min().item()),
                "E_full": float(ro["per_traj_e_full"].mean().item()),
                "lin_viol_rate": float(ro["per_traj_lin_viol"].mean().item()),
                "nl_err_rate": float(ro["per_traj_nl_err"].mean().item()),
                "ch_err": float(terms["ch_err"].mean().item()),
                "maj_err": float(terms["maj_err"].mean().item()),
                "add_err": float(terms["add_err"].mean().item()),
                "gl_gap": float(last_gap.mean().item()),
                "drift_slope": float(ro["per_traj_drift_slope"].mean().item()),
                "inst_E32_true": float(ro["per_traj_m32"].min().item()),
                "inst_E32_mean": float(ro["per_traj_m32"].mean().item()),
                "inst_E_all32_min": float(ro["per_traj_all32"].min().item()),
                "disc_viol_min": int(ro["per_traj_disc_viol"].min().item()),
                "exact_hits": int(ro["exact_hits"]),
            }
        )

    # Extra final readout_shadow at s=1.0 per SPEC 4.2
    sched_end = _get_sched(1.0)
    ro_final = engine.readout_shadow_aniso(x_re, bnd, sched_end.alpha_re_tgb)

    agg_bif = detect_bifurcation_from_curves(
        s_vals,
        a_tgb_vals,
        a_rep_vals,
        [c["E_hom"] for c in curve],
        [c["gl_gap"] for c in curve],
        [c["drift_slope"] for c in curve],
        [c["lin_viol_rate"] for c in curve],
    )
    agg_bif_instab = detect_bifurcation_from_curves(
        s_vals,
        a_tgb_vals,
        a_rep_vals,
        [c["E_hom"] for c in curve],
        [c["gl_gap"] for c in curve],
        [c["drift_slope"] for c in curve],
        [c["lin_viol_rate"] for c in curve],
        instability_only=True,
    )

    per_digest_summary: List[Dict[str, Any]] = []
    e32_d_min = ro_final["per_traj_m32"].view(n_digests, k_traj).min(dim=1).values.tolist()
    e32_d_mean = ro_final["per_traj_m32"].view(n_digests, k_traj).mean(dim=1).tolist()
    disc_d_min = ro_final["per_traj_disc_viol"].view(n_digests, k_traj).min(dim=1).values.tolist()
    hits_d = ro_final["exact_hits_mask"].view(n_digests, k_traj).sum(dim=1).tolist()

    for d in range(n_digests):
        db = detect_bifurcation_from_curves(
            s_vals,
            a_tgb_vals,
            a_rep_vals,
            dig_e_hom[d],
            dig_gl_gap[d],
            dig_drift[d],
            dig_lin[d],
        )
        db_inst = detect_bifurcation_from_curves(
            s_vals,
            a_tgb_vals,
            a_rep_vals,
            dig_e_hom[d],
            dig_gl_gap[d],
            dig_drift[d],
            dig_lin[d],
            instability_only=True,
        )
        d_steps = [dig_e_hom[d][k] - dig_e_hom[d][k - 1] for k in range(1, S_GRID_COUNT)]
        per_digest_summary.append(
            {
                "digest_idx": d,
                "bifurcated": db["bifurcated"],
                "s_c": db["s_c"],
                "alpha_bar_c": db["alpha_bar_c"],
                "alpha_bar_rep_c": db["alpha_bar_rep_c"],
                "reasons": db["reasons"],
                "instab_bifurcated": db_inst["bifurcated"],
                "instab_s_c": db_inst["s_c"],
                "instab_alpha_bar_c": db_inst["alpha_bar_c"],
                "instab_reasons": db_inst["reasons"],
                "delta_E_hom": dig_e_hom[d][-1] - dig_e_hom[d][0],
                "delta_E_max": max(d_steps) if d_steps else 0.0,
                "final_E_hom": dig_e_hom[d][-1],
                "final_gl_gap": dig_gl_gap[d][-1],
                "final_drift_slope": dig_drift[d][-1],
                "final_lin_viol": dig_lin[d][-1],
                "final_E32_min": float(e32_d_min[d]),
                "final_E32_mean": float(e32_d_mean[d]),
                "final_disc_viol_min": int(disc_d_min[d]),
                "final_exact_hits": int(hits_d[d]),
            }
        )

    ac_vals = [x["alpha_bar_c"] for x in per_digest_summary if x["alpha_bar_c"] is not None]
    ac_mean, ac_std = _mean_std([float(v) for v in ac_vals])
    iac_vals = [x["instab_alpha_bar_c"] for x in per_digest_summary if x["instab_alpha_bar_c"] is not None]
    iac_mean, iac_std = _mean_std([float(v) for v in iac_vals])
    de_vals = [float(x["delta_E_hom"]) for x in per_digest_summary]
    de_mean, de_std = _mean_std(de_vals)
    e32_mean_d, e32_std_d = _mean_std([float(x["final_E32_min"]) for x in per_digest_summary])
    agg_steps = [curve[k]["E_hom"] - curve[k - 1]["E_hom"] for k in range(1, S_GRID_COUNT)]

    return {
        "path": path_name,
        "bifurcation": agg_bif,
        "bifurcation_instability": agg_bif_instab,
        "delta_E_hom": curve[-1]["E_hom"] - curve[0]["E_hom"],
        "delta_E_max": max(agg_steps) if agg_steps else 0.0,
        "final": {
            "E_hom": float(ro_final["per_traj_e_hom"].mean().item()),
            "E_full": float(ro_final["per_traj_e_full"].mean().item()),
            "lin_viol_rate": float(ro_final["per_traj_lin_viol"].mean().item()),
            "gl_gap": curve[-1]["gl_gap"],
            "drift_slope": float(ro_final["per_traj_drift_slope"].mean().item()),
            "inst_E32_min": float(ro_final["per_traj_m32"].min().item()),
            "inst_E32_mean": float(ro_final["per_traj_m32"].mean().item()),
            "inst_E_all32_min": float(ro_final["per_traj_all32"].min().item()),
            "disc_viol_min": int(ro_final["per_traj_disc_viol"].min().item()),
            "exact_hits": int(ro_final["exact_hits"]),
        },
        "digest_stats": {
            "n_digests": n_digests,
            "n_bifurcated": len(ac_vals),
            "alpha_bar_c_mean": ac_mean if ac_vals else None,
            "alpha_bar_c_std": ac_std if ac_vals else None,
            "instab_n_bifurcated": len(iac_vals),
            "instab_alpha_bar_c_mean": iac_mean if iac_vals else None,
            "instab_alpha_bar_c_std": iac_std if iac_vals else None,
            "delta_E_hom_mean": de_mean,
            "delta_E_hom_std": de_std,
            "final_E32_min_mean": e32_mean_d,
            "final_E32_min_std": e32_std_d,
        },
        "per_digest": per_digest_summary,
        "curve": curve,
    }


@torch.no_grad()
def run_2d_boundary_compare(
    rounds_list: Tuple[int, ...],
    ctrl_params: ControlCipherParams,
    device: torch.device,
) -> Dict[str, Any]:
    """Compare fixed mask_m=256 vs interleaved 2D mask_m: 64->256 on R in {4, 8, 18}."""
    target_rounds = [r for r in (4, 8, 18) if r in rounds_list]
    paths = ("P0_uniform", "P1_bit_lsb_to_msb", "P4_round_backward", "P6_cd_complex_detour")
    mask_levels = (64, 96, 128, 160, 192, 224, 256)
    mask_step_k = {4: 96, 8: 128, 12: 160, 16: 192, 20: 224, 24: 256}

    out: Dict[str, Any] = {"rounds": []}
    for r in target_rounds:
        r_entry: Dict[str, Any] = {"rounds": r, "variants": {}}
        for variant in ("main", "control"):
            engine = AnisotropicGLRRREngine(
                rounds=r,
                variant=variant,
                ctrl_params=ctrl_params if variant == "control" else None,
            ).to(device)
            n_dig, k_traj = 8, 16
            seed = 610000 + r * 100 + (0 if variant == "main" else 50)
            bnd_256, x0_256_re, x0_256_im, _ = sample_cached_alpha0_batch(
                engine, r, variant, ctrl_params, n_dig, k_traj, seed=seed, mask_m=256, device=device
            )
            bnd_64, x0_64_re, x0_64_im, _ = sample_cached_alpha0_batch(
                engine, r, variant, ctrl_params, n_dig, k_traj, seed=seed, mask_m=64, device=device
            )
            dig = bnd_256.target_digests

            v_entry: Dict[str, Any] = {"fixed_m256": {}, "interleaved_2d": {}}
            for p_name in paths:
                res_fixed = run_single_path_sweep(engine, bnd_256, x0_256_re, x0_256_im, p_name, n_dig, k_traj)
                v_entry["fixed_m256"][p_name] = {
                    "bifurcation": res_fixed["bifurcation"],
                    "delta_E_hom": res_fixed["delta_E_hom"],
                    "delta_E_max": res_fixed["delta_E_max"],
                    "final": res_fixed["final"],
                }

                # 2D interleaved mask_m: 64 -> 256
                engine.reset_drift()
                x_re, x_im = x0_64_re.clone(), x0_64_im.clone()
                x_re_prev = x_re.clone()
                cur_m = 64
                cur_bnd = bnd_64
                curve_2d: List[Dict[str, Any]] = []
                mask_jumps: List[Dict[str, Any]] = []
                prev_e = 0.0
                for k in range(S_GRID_COUNT):
                    sv = float(k) / float(S_GRID_COUNT - 1)
                    boundary_step = False
                    if k in mask_step_k and mask_step_k[k] > cur_m:
                        cur_m = mask_step_k[k]
                        cur_bnd = build_variant_boundary(dig, r, variant, ctrl_params, mask_m=cur_m)
                        boundary_step = True
                    sched = evaluate_deterministic_schedule(p_name, sv, r, device=device)
                    last_gap = torch.zeros(x_re.shape[0], dtype=torch.float32, device=device)
                    for _ in range(T_RELAX):
                        nxt_re, nxt_im, info = engine.step_dual(
                            x_re, x_im, cur_bnd.var_mask, cur_bnd.var_vals, sched,
                            beta=0.85, x_re_prev=x_re_prev, record_drift=True, check_early_exit=(k == 0),
                        )
                        x_re_prev = x_re
                        x_re, x_im = nxt_re, nxt_im
                        last_gap = info["gl_gap"]
                    ro = engine.readout_shadow_aniso(x_re, cur_bnd, sched.alpha_re_tgb)
                    e_h = float(ro["per_traj_e_hom"].mean().item())
                    if boundary_step:
                        mask_jumps.append({"s": round(sv, 4), "mask_m": cur_m, "delta_E_on_mask": e_h - prev_e})
                    prev_e = e_h
                    curve_2d.append(
                        {
                            "s": round(sv, 4),
                            "mask_m": cur_m,
                            "alpha_bar_tgb": float(sched.alpha_re_tgb.mean().item()),
                            "E_hom": e_h,
                            "gl_gap": float(last_gap.mean().item()),
                            "drift_slope": float(ro["per_traj_drift_slope"].mean().item()),
                            "lin_viol_rate": float(ro["per_traj_lin_viol"].mean().item()),
                            "inst_E32_min": float(ro["per_traj_m32"].min().item()),
                        }
                    )
                bif_2d = detect_bifurcation_from_curves(
                    [c["s"] for c in curve_2d],
                    [c["alpha_bar_tgb"] for c in curve_2d],
                    [c["alpha_bar_tgb"] for c in curve_2d],
                    [c["E_hom"] for c in curve_2d],
                    [c["gl_gap"] for c in curve_2d],
                    [c["drift_slope"] for c in curve_2d],
                    [c["lin_viol_rate"] for c in curve_2d],
                )
                steps_2d = [curve_2d[k]["E_hom"] - curve_2d[k - 1]["E_hom"] for k in range(1, S_GRID_COUNT)]
                ro_end = engine.readout_shadow_aniso(
                    x_re, cur_bnd, evaluate_deterministic_schedule(p_name, 1.0, r, device=device).alpha_re_tgb
                )
                v_entry["interleaved_2d"][p_name] = {
                    "bifurcation": bif_2d,
                    "delta_E_hom": curve_2d[-1]["E_hom"] - curve_2d[0]["E_hom"],
                    "delta_E_max": max(steps_2d) if steps_2d else 0.0,
                    "mask_levels": list(mask_levels),
                    "mask_jumps": mask_jumps,
                    "final": {
                        "E_hom": float(ro_end["per_traj_e_hom"].mean().item()),
                        "lin_viol_rate": float(ro_end["per_traj_lin_viol"].mean().item()),
                        "gl_gap": curve_2d[-1]["gl_gap"],
                        "drift_slope": float(ro_end["per_traj_drift_slope"].mean().item()),
                        "inst_E32_min": float(ro_end["per_traj_m32"].min().item()),
                        "disc_viol_min": int(ro_end["per_traj_disc_viol"].min().item()),
                        "exact_hits": int(ro_end["exact_hits"]),
                    },
                }
            r_entry["variants"][variant] = v_entry
        out["rounds"].append(r_entry)
    return out


@torch.no_grad()
def _eval_static_mask(
    engine: AnisotropicGLRRREngine,
    bnd: V6ABoundary,
    x0_re: torch.Tensor,
    x0_im: torch.Tensor,
    alpha_re_tgb: torch.Tensor,
    n_steps: int,
    n_digests: int,
    k_traj: int,
) -> Dict[str, Any]:
    engine.reset_drift()
    x_re, x_im = x0_re.clone(), x0_im.clone()
    sched = ScheduleEval(
        alpha_re_tgb=alpha_re_tgb,
        alpha_im_tgb=torch.zeros_like(alpha_re_tgb),
        kappa_im=1.0,
        lambda_cd=0.0,
        eta_cd=0.0,
        alpha_dot_rms=0.0,
    )
    last_gap = torch.zeros(x_re.shape[0], dtype=torch.float32, device=x_re.device)
    for _ in range(n_steps):
        x_re, x_im, info = engine.step_dual(
            x_re, x_im, bnd.var_mask, bnd.var_vals, sched, beta=0.85, record_drift=True
        )
        last_gap = info["gl_gap"]
    ro = engine.readout_shadow_aniso(x_re, bnd, alpha_re_tgb)
    lin_d = ro["per_traj_lin_viol"].view(n_digests, k_traj).mean(dim=1).tolist()
    gap_d = last_gap.view(n_digests, k_traj).mean(dim=1).tolist()
    drift_d = ro["per_traj_drift_slope"].view(n_digests, k_traj).mean(dim=1).tolist()
    lin_m = float(ro["per_traj_lin_viol"].mean().item())
    gap_m = float(last_gap.mean().item())
    drift_m = float(ro["per_traj_drift_slope"].mean().item())
    return {
        "lin_viol_rate": lin_m,
        "gl_gap": gap_m,
        "drift_slope": drift_m,
        "e_full": float(ro["per_traj_e_full"].mean().item()),
        "restored_linear": bool(lin_m < 0.02 and gap_m < 0.14 and drift_m < 1e-2),
        "per_digest_restored": [
            bool(lin_d[d] < 0.02 and gap_d[d] < 0.14 and drift_d[d] < 1e-2) for d in range(n_digests)
        ],
        "per_digest_lin_viol": lin_d,
        "per_digest_gl_gap": gap_d,
        "x_re": x_re,
        "ro": ro,
    }


@torch.no_grad()
def _eval_add_slot_mask(
    engine: AnisotropicGLRRREngine,
    bnd: V6ABoundary,
    x0_re: torch.Tensor,
    x0_im: torch.Tensor,
    slot_idx: int,
    n_steps: int,
) -> Dict[str, float]:
    """Activate ONLY a single ADD32 slot (alpha=1) while keeping all other gates at alpha=0."""
    r = engine.rounds
    device = x0_re.device
    a_rep = torch.zeros(engine.n_nl_rep, dtype=torch.float32, device=device)
    c2 = (engine.n_ch + engine.n_maj) * 4
    add_mask = (engine.add_slot_id == slot_idx).to(torch.float32)  # (N_add,)
    a_rep[c2:] = add_mask.view(engine.n_add, 1, 1).expand(engine.n_add, 4, 32).reshape(-1)

    engine.reset_drift()
    x_re, x_im = x0_re.clone(), x0_im.clone()
    last_gap = torch.zeros(x_re.shape[0], dtype=torch.float32, device=device)
    for _ in range(n_steps):
        y_a_re, _ = engine.project_A_dual(x_re, x_im, bnd.var_mask, bnd.var_vals)
        r_re = 2.0 * y_a_re - x_re
        y_lin, _ = engine.project_linear_skeleton(r_re[:, : engine.n_lin_rep], bnd.var_mask, bnd.var_vals)
        p_nl = engine.project_B_nl_exact(r_re[:, engine.n_lin_rep :])
        y_nl = (1.0 - a_rep.unsqueeze(0)) * r_re[:, engine.n_lin_rep :] + a_rep.unsqueeze(0) * p_nl
        y_b_re = torch.cat([y_lin, y_nl], dim=1)
        diff = y_b_re - y_a_re
        x_re = x_re + 0.85 * diff
        inv_sqrt_d = 1.0 / math.sqrt(float(x_re.shape[-1]))
        last_gap = torch.linalg.vector_norm(diff, dim=-1) * inv_sqrt_d
        engine.drift_monitor.record(torch.linalg.vector_norm(x_re, dim=-1) * inv_sqrt_d)

    ro = engine.readout_shadow_aniso(x_re, bnd, torch.ones((r, 3, 32), dtype=torch.float32, device=device))
    return {
        "lin_viol_rate": float(ro["per_traj_lin_viol"].mean().item()),
        "gl_gap": float(last_gap.mean().item()),
        "drift_slope": float(ro["per_traj_drift_slope"].mean().item()),
        "e_full": float(ro["per_traj_e_full"].mean().item()),
    }


@torch.no_grad()
def run_cutset_ablation(
    rounds_list: Tuple[int, ...],
    ctrl_params: ControlCipherParams,
    device: torch.device,
) -> Dict[str, Any]:
    """Run Single-Component Singularity Probe + Counterfactual Cut-Set & Greedy Minimal Cut-Set."""
    out: Dict[str, Any] = {"rounds": []}
    n_dig, k_traj = 4, 8

    for r in rounds_list:
        r_res: Dict[str, Any] = {"rounds": r, "variants": {}}
        for variant in ("main", "control"):
            engine = AnisotropicGLRRREngine(
                rounds=r,
                variant=variant,
                ctrl_params=ctrl_params if variant == "control" else None,
            ).to(device)
            seed = 730000 + r * 100 + (0 if variant == "main" else 50)
            bnd, x0_re, x0_im, _ = sample_cached_alpha0_batch(
                engine, r, variant, ctrl_params, n_dig, k_traj, seed=seed, mask_m=256, device=device
            )

            # A. Single-Component Singularity Probe (16 steps from alpha=0)
            gate_probes: Dict[str, Any] = {}
            for g_idx, g_name in enumerate(("only_Ch", "only_Maj", "only_Add")):
                mask = torch.zeros((r, 3, 32), dtype=torch.float32, device=device)
                mask[:, g_idx, :] = 1.0
                ev = _eval_static_mask(engine, bnd, x0_re, x0_im, mask, 16, n_dig, k_traj)
                gate_probes[g_name] = {k: ev[k] for k in ("lin_viol_rate", "gl_gap", "drift_slope", "e_full")}

            slot_probes: Dict[str, Any] = {}
            for s_idx, s_name in enumerate(ADD_SLOT_NAMES):
                if s_name == "sched" and r <= 16:
                    continue
                slot_probes[s_name] = _eval_add_slot_mask(engine, bnd, x0_re, x0_im, s_idx, 16)

            nibble_probes: List[Dict[str, Any]] = []
            for m in range(8):
                mask = torch.zeros((r, 3, 32), dtype=torch.float32, device=device)
                mask[:, :, 4 * m : 4 * m + 4] = 1.0
                ev = _eval_static_mask(engine, bnd, x0_re, x0_im, mask, 16, n_dig, k_traj)
                nibble_probes.append(
                    {"nibble": m, "bits": [4 * m, 4 * m + 3], **{k: ev[k] for k in ("lin_viol_rate", "gl_gap", "drift_slope", "e_full")}}
                )

            probe_t_list = list(range(r)) if r <= 24 else sorted(set(list(range(8)) + list(range(8, 56, 4)) + list(range(56, 64))))
            round_probes: List[Dict[str, Any]] = []
            for t in probe_t_list:
                mask = torch.zeros((r, 3, 32), dtype=torch.float32, device=device)
                mask[t, :, :] = 1.0
                ev = _eval_static_mask(engine, bnd, x0_re, x0_im, mask, 16, n_dig, k_traj)
                round_probes.append(
                    {"round": t, **{k: ev[k] for k in ("lin_viol_rate", "gl_gap", "drift_slope", "e_full")}}
                )

            # B. Counterfactual Cut-Set Ablation (from alpha=1, cut subset C -> 0, 24 steps)
            bit_cuts: Dict[str, Any] = {"stride": [], "cut_msb": [], "cut_lsb": []}
            for stride in (2, 4, 8, 16):
                mask = torch.ones((r, 3, 32), dtype=torch.float32, device=device)
                mask[:, 2, 0::stride] = 0.0
                ev = _eval_static_mask(engine, bnd, x0_re, x0_im, mask, 24, n_dig, k_traj)
                bit_cuts["stride"].append(
                    {"stride": stride, "add_cut_frac": float(1.0 / stride), **{k: ev[k] for k in ("lin_viol_rate", "gl_gap", "drift_slope", "restored_linear")}}
                )
            for k_b in (4, 8, 12, 16, 20, 24, 28, 32):
                mask_m = torch.ones((r, 3, 32), dtype=torch.float32, device=device)
                mask_m[:, :, 32 - k_b :] = 0.0
                evm = _eval_static_mask(engine, bnd, x0_re, x0_im, mask_m, 24, n_dig, k_traj)
                bit_cuts["cut_msb"].append(
                    {"k_msb": k_b, "cut_frac": k_b / 32.0, **{k: evm[k] for k in ("lin_viol_rate", "gl_gap", "drift_slope", "restored_linear")}}
                )
                mask_l = torch.ones((r, 3, 32), dtype=torch.float32, device=device)
                mask_l[:, :, :k_b] = 0.0
                evl = _eval_static_mask(engine, bnd, x0_re, x0_im, mask_l, 24, n_dig, k_traj)
                bit_cuts["cut_lsb"].append(
                    {"k_lsb": k_b, "cut_frac": k_b / 32.0, **{k: evl[k] for k in ("lin_viol_rate", "gl_gap", "drift_slope", "restored_linear")}}
                )

            # Round-depth percolation R_perc
            r_act_grid = list(range(1, min(r, 16) + 1))
            if r > 16:
                r_act_grid += sorted(set([20, 24, 32, 48, r] if r == 64 else [r]))
                r_act_grid = [x for x in r_act_grid if x <= r]
            prefix_curve: List[Dict[str, Any]] = []
            r_perc: Optional[int] = None
            for r_act in r_act_grid:
                mask = torch.zeros((r, 3, 32), dtype=torch.float32, device=device)
                mask[:r_act, :, :] = 1.0
                ev = _eval_static_mask(engine, bnd, x0_re, x0_im, mask, 24, n_dig, k_traj)
                prefix_curve.append(
                    {"r_act": r_act, **{k: ev[k] for k in ("lin_viol_rate", "gl_gap", "drift_slope", "restored_linear")}}
                )
                if r_perc is None and not ev["restored_linear"]:
                    r_perc = r_act

            tail_cuts: List[Dict[str, Any]] = []
            for w in sorted({x for x in (1, 2, 4, 6, 8, 12, 16, r // 2) if 1 <= x <= r}):
                mask = torch.ones((r, 3, 32), dtype=torch.float32, device=device)
                mask[r - w :, :, :] = 0.0
                ev = _eval_static_mask(engine, bnd, x0_re, x0_im, mask, 24, n_dig, k_traj)
                tail_cuts.append({"w_tail_cut": w, **{k: ev[k] for k in ("lin_viol_rate", "gl_gap", "drift_slope", "restored_linear")}})

            # C. Greedy Minimal Cut-Set C_min via sensitivity ranking + binary search
            ones_mask = torch.ones((r, 3, 32), dtype=torch.float32, device=device)
            ev_full = _eval_static_mask(engine, bnd, x0_re, x0_im, ones_mask, 24, n_dig, k_traj)
            terms = ev_full["ro"]["terms"]

            # Compute per-(t,g,b) sensitivity score S(t,g,b): gate mismatch + replica consensus gap
            sens_tgb = torch.zeros((r, 3, 32), dtype=torch.float32, device=device)
            sens_tgb[:, 0, :] = terms["ch_mis"].mean(dim=0).view(r, 32)
            sens_tgb[:, 1, :] = terms["maj_mis"].mean(dim=0).view(r, 32)
            add_err_j_b = 0.5 * terms["sum_err"].mean(dim=0)  # (N_add, 32)
            add_err_j_b[:, :31] += 0.5 * terms["carry_err"].mean(dim=0)
            add_sens = torch.zeros((r, 32), dtype=torch.float32, device=device)
            add_cnt = torch.zeros((r, 32), dtype=torch.float32, device=device)
            add_sens.index_add_(0, engine.add_round_idx, add_err_j_b)
            add_cnt.index_add_(0, engine.add_round_idx, torch.ones_like(add_err_j_b))
            sens_tgb[:, 2, :] = add_sens / add_cnt.clamp_min(1.0)

            # Add replica Set-A discrepancy averaged back to (t,g,b)
            y_a_re, _ = engine.project_A_dual(ev_full["x_re"], x0_im, bnd.var_mask, bnd.var_vals)
            nl_gap_rep = (ev_full["x_re"][:, engine.n_lin_rep :] - y_a_re[:, engine.n_lin_rep :]).abs().mean(dim=0)
            tgb_gap = torch.zeros(r * 3 * 32, dtype=torch.float32, device=device)
            tgb_gap.index_add_(0, engine.rep_to_tgb_idx, nl_gap_rep)
            tgb_gap = (tgb_gap.view(r, 3, 32) / engine.coord_map.tgb_deg.to(device).clamp_min(1.0))
            sens_total = sens_tgb + 0.5 * tgb_gap

            n_tgb = r * 3 * 32
            order_idx = torch.argsort(sens_total.reshape(-1), descending=True)

            # Binary search minimal prefix k_cut restoring linear manifold (from alpha=0 stability & turbulent restore)
            lo, hi = 0, n_tgb
            best_k = n_tgb
            best_ev = None
            while lo <= hi:
                mid = (lo + hi) // 2
                m_flat = torch.ones(n_tgb, dtype=torch.float32, device=device)
                if mid > 0:
                    m_flat[order_idx[:mid]] = 0.0
                ev_m = _eval_static_mask(engine, bnd, x0_re, x0_im, m_flat.view(r, 3, 32), 24, n_dig, k_traj)
                if ev_m["restored_linear"]:
                    best_k = mid
                    best_ev = ev_m
                    hi = mid - 1
                else:
                    lo = mid + 1

            # Also check per-digest restoration threshold around best_k via coarse + binary search per digest
            cut_flat = torch.zeros(n_tgb, dtype=torch.bool, device=device)
            if best_k > 0:
                cut_flat[order_idx[:best_k]] = True
            cut_tgb = cut_flat.view(r, 3, 32)
            rho_cut = float(best_k) / float(n_tgb)
            rho_cut_rep = float((1.0 - engine.broadcast_tgb_to_rep((~cut_tgb).to(torch.float32))).mean().item())

            # Per-digest rho_cut via independent binary search on cached order_idx (sliced to digest d)
            per_digest_rho: List[float] = []
            for d in range(n_dig):
                s_i, e_i = d * k_traj, (d + 1) * k_traj
                bnd_d = V6ABoundary(
                    rounds=bnd.rounds,
                    domain_mode=bnd.domain_mode,
                    mask_m=bnd.mask_m,
                    allow_peel_back=bnd.allow_peel_back,
                    var_mask=bnd.var_mask[s_i:e_i],
                    var_vals=bnd.var_vals[s_i:e_i],
                    A_full_clamped=bnd.A_full_clamped[s_i:e_i],
                    A_full_mask=bnd.A_full_mask[s_i:e_i],
                    E_full_clamped=bnd.E_full_clamped[s_i:e_i],
                    E_full_mask=bnd.E_full_mask[s_i:e_i],
                    W_full_clamped=bnd.W_full_clamped[s_i:e_i],
                    W_full_mask=bnd.W_full_mask[s_i:e_i],
                    target_digests=bnd.target_digests[s_i:e_i],
                    W16_clamped_words=bnd.W16_clamped_words[s_i:e_i],
                    W16_mask_bits=bnd.W16_mask_bits[s_i:e_i],
                    digest_target_bits=None if bnd.digest_target_bits is None else bnd.digest_target_bits[s_i:e_i],
                    digest_target_mask=None if bnd.digest_target_mask is None else bnd.digest_target_mask[s_i:e_i],
                    is_full_digest=bnd.is_full_digest,
                )
                x0_re_d = x0_re[s_i:e_i]
                x0_im_d = x0_im[s_i:e_i]
                d_lo, d_hi = 0, n_tgb
                d_best = n_tgb
                while d_lo <= d_hi:
                    mid = (d_lo + d_hi) // 2
                    m_flat = torch.ones(n_tgb, dtype=torch.float32, device=device)
                    if mid > 0:
                        m_flat[order_idx[:mid]] = 0.0
                    ev_d = _eval_static_mask(engine, bnd_d, x0_re_d, x0_im_d, m_flat.view(r, 3, 32), 24, 1, k_traj)
                    if ev_d["restored_linear"]:
                        d_best = mid
                        d_hi = mid - 1
                    else:
                        d_lo = mid + 1
                per_digest_rho.append(float(d_best) / float(n_tgb))

            # Support distribution of C_min
            n_cut_eff = max(1, best_k)
            support = {
                "frac_Ch": float(cut_tgb[:, 0, :].sum().item()) / float(n_cut_eff),
                "frac_Maj": float(cut_tgb[:, 1, :].sum().item()) / float(n_cut_eff),
                "frac_Add": float(cut_tgb[:, 2, :].sum().item()) / float(n_cut_eff),
                "mean_round_norm": float(
                    (cut_tgb.float().sum(dim=(1, 2)) * torch.arange(r, dtype=torch.float32, device=device)).sum().item()
                ) / float(n_cut_eff * max(1, r - 1)) if r > 1 else 0.0,
                "mean_bit": float(
                    (cut_tgb.float().sum(dim=(0, 1)) * torch.arange(32, dtype=torch.float32, device=device)).sum().item()
                ) / float(n_cut_eff),
            }

            # Also verify turbulent->restoration starting directly from ev_full["x_re"]
            keep_mask = (~cut_tgb).to(torch.float32)
            ev_restore_turb = _eval_static_mask(engine, bnd, ev_full["x_re"], x0_im, keep_mask, 24, n_dig, k_traj)

            rho_mean, rho_std = _mean_std(per_digest_rho)
            r_res["variants"][variant] = {
                "singularity_probe": {
                    "gates": gate_probes,
                    "add_slots": slot_probes,
                    "nibbles": nibble_probes,
                    "rounds": round_probes,
                },
                "bit_cuts": bit_cuts,
                "round_cuts": {
                    "r_perc": r_perc,
                    "prefix_curve": prefix_curve,
                    "tail_cuts": tail_cuts,
                },
                "greedy_minimal_cutset": {
                    "n_tgb_total": n_tgb,
                    "k_cut_min": best_k,
                    "rho_cut": rho_cut,
                    "rho_cut_rep": rho_cut_rep,
                    "per_digest_rho_cut": per_digest_rho,
                    "rho_cut_digest_mean": rho_mean,
                    "rho_cut_digest_std": rho_std,
                    "support": support,
                    "lin_viol_at_cut": best_ev["lin_viol_rate"] if best_ev else None,
                    "gl_gap_at_cut": best_ev["gl_gap"] if best_ev else None,
                    "turbulent_restore_check": {
                        "lin_viol_rate": ev_restore_turb["lin_viol_rate"],
                        "gl_gap": ev_restore_turb["gl_gap"],
                        "restored_linear": ev_restore_turb["restored_linear"],
                    },
                },
            }
        out["rounds"].append(r_res)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="P8 Anisotropic Homotopy Colab Experiment Runner")
    parser.add_argument("--rounds", type=int, nargs="*", default=list(ALL_ROUNDS))
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-2d", action="store_true")
    parser.add_argument("--skip-cutset", action="store_true")
    args = parser.parse_args()

    t_global = time.time()
    enforce_v6a_fp32()
    if not torch.cuda.is_available():
        raise RuntimeError("P8 full sweep must run on Colab CUDA GPU (local RTX 4060 forbidden).")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"[P8] Starting on GPU: {gpu_name}, allow_tf32={torch.backends.cuda.matmul.allow_tf32}", flush=True)

    rounds_tuple = tuple(int(r) for r in args.rounds)
    ctrl_params = make_control_cipher_params(seed=20261125)
    res_dir = Path(__file__).resolve().parent / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_sweep:
        sweep_out: Dict[str, Any] = {
            "gpu": gpu_name,
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "control_params": ctrl_params.to_dict(),
            "s_grid_count": S_GRID_COUNT,
            "t_relax": T_RELAX,
            "beta": 0.85,
            "rounds": [],
        }
        p6_abl_rounds = {4, 8, 18, 64}
        for r in rounds_tuple:
            t_r = time.time()
            n_dig = 4 if r == 64 else 8
            k_traj = 16
            r_obj: Dict[str, Any] = {"rounds": r, "n_digests": n_dig, "k_traj": k_traj, "variants": {}}
            for variant in ("main", "control"):
                print(f"[P8 Sweep] R={r} variant={variant}: building engine & solving alpha=0 GF(2)...", flush=True)
                engine = AnisotropicGLRRREngine(
                    rounds=r, variant=variant, ctrl_params=ctrl_params if variant == "control" else None
                ).to(device)
                assert_strict_fp32(engine)

                eval_seed = 500000 + r * 100 + (0 if variant == "main" else 50)
                bnd, x0_re, x0_im, nullity = sample_cached_alpha0_batch(
                    engine, r, variant, ctrl_params, n_dig, k_traj, seed=eval_seed, mask_m=256, device=device
                )

                print(f"[P8 Sweep] R={r} variant={variant}: training P7_learned_aniso ({args.train_steps} steps)...", flush=True)
                p7_field, p7_summary = train_p7_learned_field(
                    engine, r, variant, ctrl_params, device=device, train_steps=args.train_steps
                )

                path_list = list(PRIMARY_PATHS)
                if r in p6_abl_rounds:
                    path_list.extend(P6_ABLATION_PATHS)

                paths_out: Dict[str, Any] = {}
                for p_name in path_list:
                    p_res = run_single_path_sweep(
                        engine, bnd, x0_re, x0_im, p_name, n_dig, k_traj,
                        p7_field=p7_field if p_name == "P7_learned_aniso" else None,
                    )
                    if p_name == "P7_learned_aniso":
                        p_res["training_summary"] = p7_summary
                    paths_out[p_name] = p_res
                    bif = p_res["bifurcation"]
                    print(
                        f"  R={r:2d} {variant:7s} {p_name:24s} | bif={str(bif['bifurcated']):5s} "
                        f"s_c={bif['s_c']} a_bar_c={bif['alpha_bar_c']} "
                        f"dE={p_res['delta_E_hom']:.4f} E32_min={p_res['final']['inst_E32_min']:.4f} "
                        f"hits={p_res['final']['exact_hits']}",
                        flush=True,
                    )

                r_obj["variants"][variant] = {
                    "gf2_nullity": nullity,
                    "n_var_bits": engine.n_lin_rep,
                    "n_nl_rep": engine.n_nl_rep,
                    "paths": paths_out,
                }
            sweep_out["rounds"].append(r_obj)
            (res_dir / "homotopy_sweep.partial.json").write_text(json.dumps(sweep_out, indent=2), encoding="utf-8")
            print(f"[P8 Sweep] Finished R={r} in {time.time() - t_r:.1f}s", flush=True)

        sweep_out["elapsed_sec"] = round(time.time() - t_global, 2)
        (res_dir / "homotopy_sweep.json").write_text(json.dumps(sweep_out, indent=2), encoding="utf-8")
        print(f"[P8] Saved homotopy_sweep.json ({sweep_out['elapsed_sec']}s)", flush=True)

    if not args.skip_2d:
        t_2d = time.time()
        print("[P8 2D] Starting 2D boundary relaxation comparison...", flush=True)
        res_2d = run_2d_boundary_compare(rounds_tuple, ctrl_params, device)
        res_2d["elapsed_sec"] = round(time.time() - t_2d, 2)
        (res_dir / "boundary_2d_compare.json").write_text(json.dumps(res_2d, indent=2), encoding="utf-8")
        print(f"[P8 2D] Saved boundary_2d_compare.json ({res_2d['elapsed_sec']}s)", flush=True)

    if not args.skip_cutset:
        t_cut = time.time()
        print("[P8 CutSet] Starting singularity probe & minimal cut-set ablation...", flush=True)
        res_cut = run_cutset_ablation(rounds_tuple, ctrl_params, device)
        res_cut["elapsed_sec"] = round(time.time() - t_cut, 2)
        (res_dir / "cutset_ablation.json").write_text(json.dumps(res_cut, indent=2), encoding="utf-8")
        print(f"[P8 CutSet] Saved cutset_ablation.json ({res_cut['elapsed_sec']}s)", flush=True)

    print(f"[P8] ALL EXPERIMENTS COMPLETE in {time.time() - t_global:.1f}s", flush=True)


if __name__ == "__main__":
    main()
