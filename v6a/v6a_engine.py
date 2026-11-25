"""Phase-SHA V6A-v2 Orchestration Engine: Ouroboros Closed-Loop 2D Homotopy, Frontier Push & R=18 Deep Attack.

Coordinates all 5 V6A layers cleanly under strict pure FP32:
  - L1 (v6a_homotopy): Algebraic peel-back, compatible F_2 GF(2) alpha=0 init, 2D interleaved (alpha, mask_m).
  - L2 (v6a_gl_rrr): Global Damped Min-Sum BP + Top-K OSD-0 + alpha-interpolated Viterbi RRR + Endgame Polish.
  - L3 (v6a_bottleneck_mps): Targeted 64-site MGS2 Joint-NLL Born Machine + S_ent floor (>= 0.55).
  - L4 (v6a_bm_escape): Stagnation-triggered / bifurcation-triggered S^{N-1} history-basin orthogonal Cayley kicks.
  - L5 (v6a_navigator): Zero-BCE homotopy tangent predictor + VSIDS gap-focus weights (zero braking).

Strictly separates instantaneous energy metrics (`inst_E_hom`, `inst_E32_true`, `inst_mean_E`)
from cumulative historical minimum (`hist_min_E32`), logging binomial z-scores (|z| > 3.5 gate).
"""

from __future__ import annotations

import csv
import gc
import json
import math
import signal
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.sha256_ref import MASK32, sha256_digest, sha256_trace, words_to_spins
from src.v3_trainer import _compute_per_round_acc
from src.v5_interfaces import build_r20_stress_holdout_w16, build_v5_holdout_w16
from v6a import enforce_v6a_fp32
from v6a.v6a_bm_escape import BurerMonteiroEscapeLayer
from v6a.v6a_bottleneck_mps import (
    BottleneckMPSBornMachine,
    select_bottleneck_var_indices,
    select_diverse_elite_spins,
)
from v6a.v6a_gl_rrr import GlobalLinearRRREngine
from v6a.v6a_homotopy import (
    HomotopySchedule2D,
    V6ABoundary,
    build_v6a_boundary,
    solve_alpha0_gf2_linear_skeleton,
)
from v6a.v6a_navigator import ZeroBrakeNavigator


def _sample_free_msg_w16(batch_size: int, rounds: int, device: torch.device, gen: torch.Generator) -> torch.Tensor:
    """Sample unconstrained uint32 message blocks for `free_msg` calibration (no fixed padding)."""
    w16 = torch.randint(0, 1 << 32, (batch_size, 16), dtype=torch.int64, device=device, generator=gen)
    if rounds < 16:
        w16[:, rounds:] = 0
    return w16


def _binomial_zscore(acc: float, n_bits: int) -> float:
    """Exact binomial normal-approx z-score relative to unbiased p=0.5 coin flip."""
    n = max(1, int(n_bits))
    return 2.0 * (float(acc) - 0.5) * math.sqrt(float(n))


def build_v6a_csv_fields(rounds: int = 18) -> List[str]:
    """Build metrics.csv schema keeping V4/V5 dashboard fields plus V6A-v2 homotopy & z-score metrics."""
    fields = [
        "step",
        "wall_s",
        "dt",
        "step_time_ms",
        "peak_vram_gb",
        "lr",
        # Honest V6A core metrics (strictly separating instantaneous vs historical)
        "inst_E_hom",
        "inst_E32",
        "inst_E32_true",
        "inst_E_all32",
        "disc_viol_min",
        "inst_mean_E",
        "hist_min",
        "hist_min_E32",
        "best_global_m32_energy",
        "alpha",
        "mask_m",
        "alpha_c",
        "S_ent",
        "gl_gap",
        "bifurc",
        "bifurc_mps_calls",
        "endgame_polish_runs",
        "exact_hits",
        "exact_preimage_rate@256",
        "unique_ratio@256",
        "align",
        "drift_slope",
        "drift_flag",
        "esc_count",
        "pred_mse",
        "lin_viol",
        "nl_err",
        # Anti-illusion binomial statistics
        "eval_n_digests",
        "eval_free_bits",
        "w0_zscore",
        "r12_zscore",
        "r13_zscore",
        "r16_zscore",
        # Dashboard backwards-compatibility keys mapped honestly (NEVER running-min disguised!)
        "train_loss",
        "train_phys_loss",
        "train_bce_loss",
        "train_xor_loss",
        "train_comm_loss",
        "train_align_loss",
        "eval_bce_loss",
        "min_phys_energy_m32",
        "phys_energy_m32",
        "phys_energy_m1",
        "zero_w0_bit_acc",
        "w0_free_bit_acc",
        "zero_latch_ratio",
        "zero_manifold_align",
        "zero_manifold_comm",
    ]
    for t in range(rounds):
        fields.append(f"zero_round_acc_{t}")
    for t in range(rounds):
        fields.append(f"per_round_acc_{t}")
    return fields


class V6ACSVLogger:
    """Thread/resume-safe CSV logger appending clean V6A rows for dashboard telemetry."""

    def __init__(self, output_dir: Path, rounds: int = 18) -> None:
        self.csv_path = Path(output_dir) / "metrics.csv"
        self.fields = build_v6a_csv_fields(rounds=rounds)
        if not self.csv_path.exists():
            with self.csv_path.open("w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def append(self, row: Dict[str, Any]) -> None:
        clean = {k: row.get(k, "") for k in self.fields}
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(clean)


def solve_single_digest_v6a(
    digest: torch.Tensor,
    rounds: int = 4,
    domain_mode: str = "free_msg",
    allow_peel_back: bool = False,
    num_trajs: int = 64,
    max_ticks: int = 120,
    max_seconds: float = 60.0,
    start_mask_m: int = 64,
    target_mask_m: int = 256,
    step_alpha: float = 0.05,
    min_step_alpha: float = 0.01,
    max_ticks_per_alpha: int = 12,
    eps_advance: float = 0.018,
    spike_factor: float = 1.20,
    spike_abs_margin: float = 0.003,
    inner_rrr_per_tick: int = 6,
    mps_bond_dim: int = 8,
    mps_every: int = 10,
    stag_mps_cycles_trigger: int = 3,
    endgame_zoom_k: int = 12,
    endgame_zoom_k_large: int = 14,
    use_homotopy: bool = True,
    use_mps: bool = True,
    use_bm: bool = True,
    use_navigator: bool = True,
    interleaved_2d: bool = True,
    device: torch.device | str = "cuda",
    seed: int = 6404,
    tick_callback: Optional[Callable[[int, Dict[str, Any], float], None]] = None,
    log_interval: int = 25,
) -> Dict[str, Any]:
    """Solve or probe a single target digest end-to-end with V6A-v2.1 bifurcation triple & GPU zoom polish."""
    enforce_v6a_fp32()
    dev = torch.device(device)
    t0_wall = time.perf_counter()
    rng = torch.Generator(device=dev).manual_seed(int(seed))
    cpu_rng = torch.Generator(device="cpu").manual_seed(int(seed) + 17)

    r = int(rounds)
    gl_rrr = GlobalLinearRRREngine(rounds=r, bp_iters=7, osd_every=8, osd_top_k=12).to(dev)
    spec = gl_rrr.spec

    init_alpha = 0.0 if use_homotopy else 1.0
    init_m = int(start_mask_m) if use_homotopy else int(target_mask_m)
    sched = HomotopySchedule2D(
        start_alpha=init_alpha,
        start_mask_m=init_m,
        target_mask_m=int(target_mask_m),
        step_alpha=float(step_alpha),
        min_step_alpha=float(min_step_alpha),
        eps_advance=float(eps_advance),
        spike_factor=float(spike_factor),
        spike_abs_margin=float(spike_abs_margin),
        max_ticks_per_alpha=int(max_ticks_per_alpha),
        interleaved_2d=bool(interleaved_2d),
    )

    bnd = build_v6a_boundary(
        digest=digest.to(dev),
        rounds=r,
        domain_mode=domain_mode,
        mask_m=sched.mask_m,
        allow_peel_back=allow_peel_back,
    )
    k_t = max(8, int(num_trajs))
    u0, _, _ = solve_alpha0_gf2_linear_skeleton(
        spec, bnd.var_mask[0], bnd.var_vals[0], num_samples=k_t, generator=cpu_rng
    )
    state_rep = gl_rrr.init_replicas_from_vars(u0)
    b_rep = bnd.repeat_for_candidates(k_t)
    beta_vec = torch.linspace(0.68, 1.10, k_t, dtype=torch.float32, device=dev)

    var_idx, loop_pairs = select_bottleneck_var_indices(spec, bnd.var_mask[0], max_sites=64)
    lp_pairs = [(a, b) for a, b, _ in loop_pairs]
    mps = BottleneckMPSBornMachine(num_sites=len(var_idx), bond_dim=mps_bond_dim, seed=seed + 3).to(dev)
    bm = BurerMonteiroEscapeLayer(spec.n_var_bits, manifold_dim=16, window_size=6, stag_patience=8).to(dev)
    nav = ZeroBrakeNavigator(spec.n_var_words, hidden=48).to(dev)

    hist_min_e32 = float("inf")
    best_w16: Optional[torch.Tensor] = None
    hit_w16: Optional[torch.Tensor] = None
    exact_hits = 0
    bifurc_mps_calls = 0
    endgame_polish_runs = 0
    s_ent_val = float(mps.compute_entanglement_entropy().item())
    last_ro: Dict[str, Any] = {}
    diag: Dict[str, Any] = {}
    ticks_done = 0
    prev_rank_scores: Optional[torch.Tensor] = None
    prev_mps_cycle_e: Optional[float] = None
    stag_mps_cycles = 0

    for tick in range(1, int(max_ticks) + 1):
        if (time.perf_counter() - t0_wall) > float(max_seconds):
            break
        ticks_done = tick

        for s_in in range(max(1, int(inner_rrr_per_tick))):
            vsids_w = nav.get_vsids_weights(b_rep.var_mask) if use_navigator else None
            state_rep, diag = gl_rrr.step(
                state_rep,
                b_rep.var_mask,
                b_rep.var_vals,
                alpha=sched.alpha,
                beta=beta_vec,
                vsids_var_weights=vsids_w,
                run_osd=(tick % 6 == 0 and s_in == 0),
                priority_scores=prev_rank_scores,
            )
            if use_navigator:
                nav.record_gap(diag["var_gap_profile"])

        if use_bm:
            vsids_w = nav.get_vsids_weights(b_rep.var_mask) if use_navigator else None
            kicked, k_idx, _ = bm.record_and_maybe_escape(
                diag["v_A_vars"], diag["gl_gap"], b_rep.var_mask, b_rep.var_vals, vsids_var_weights=vsids_w, generator=rng
            )
            if kicked is not None and k_idx.numel() > 0:
                state_rep[k_idx] = gl_rrr.init_replicas_from_vars(kicked)

        vsids_w = nav.get_vsids_weights(b_rep.var_mask) if use_navigator else None
        last_ro = gl_rrr.readout_shadow(state_rep, b_rep, alpha=sched.alpha, vsids_var_weights=vsids_w)
        prev_rank_scores = last_ro["per_traj_rank"]

        if last_ro["inst_E32_true"] < hist_min_e32:
            hist_min_e32 = last_ro["inst_E32_true"]
            best_w16 = last_ro["best_w16"].detach().clone()

        if last_ro["exact_hits"] > 0:
            exact_hits = last_ro["exact_hits"]
            hit_idx = int(torch.nonzero(last_ro["exact_hits_mask"], as_tuple=False)[0].item())
            hit_w16 = last_ro["w16_cand"][hit_idx].detach().clone()
            best_w16 = hit_w16
            hist_min_e32 = min(hist_min_e32, 0.0)
            break

        # Endgame Polish: smash near-zero / low-residual candidates into genuine 256-bit hits via 2^K GPU zoom
        if (
            sched.alpha >= 0.999
            and sched.mask_m >= int(target_mask_m)
            and (
                last_ro["inst_E32_true"] < 0.05
                or last_ro["disc_viol_min"] <= 16
                or last_ro["inst_E_hom"] < 0.015
            )
            and (tick % 4 == 0 or (last_ro["disc_viol_min"] <= 8 and tick % 2 == 0))
        ):
            endgame_polish_runs += 1
            solved_eg, hw_eg, cnt_eg, top_eg, sub_eg = gl_rrr.endgame_polish(
                state_rep,
                b_rep,
                last_ro["v_a"],
                last_ro["per_traj_m32"],
                vsids_var_weights=vsids_w,
                top_k=6,
                per_traj_rank=last_ro["per_traj_rank"],
                zoom_k=endgame_zoom_k,
                zoom_k_large=endgame_zoom_k_large,
            )
            state_rep[top_eg] = sub_eg
            if solved_eg and hw_eg is not None:
                exact_hits = cnt_eg
                hit_w16 = hw_eg
                best_w16 = hw_eg
                hist_min_e32 = min(hist_min_e32, 0.0)
                last_ro["exact_hits"] = cnt_eg
                last_ro["inst_E32_true"] = 0.0
                break

        advanced, bifurcated, bnd_changed = (False, False, False)
        if use_homotopy:
            advanced, bifurcated, bnd_changed = sched.update(
                last_ro["inst_E_hom"], gl_gap=float(diag["gl_gap"].min().item()), dt_ticks=1
            )

        if advanced:
            if use_navigator:
                top_mask = last_ro["per_traj_hom"] <= torch.quantile(last_ro["per_traj_hom"], 0.35)
                nav.train_on_successful_jump(last_ro["v_a"], best_mask=top_mask)
            if bnd_changed:
                bnd = build_v6a_boundary(
                    digest=digest.to(dev),
                    rounds=r,
                    domain_mode=domain_mode,
                    mask_m=sched.mask_m,
                    allow_peel_back=allow_peel_back,
                )
                b_rep = bnd.repeat_for_candidates(k_t)
                v_clamped = torch.where(b_rep.var_mask > 0.5, b_rep.var_vals, last_ro["v_a"])
                state_rep = gl_rrr.init_replicas_from_vars(v_clamped)
                state_rep, diag = gl_rrr.step(
                    state_rep,
                    b_rep.var_mask,
                    b_rep.var_vals,
                    alpha=sched.alpha,
                    beta=beta_vec,
                    vsids_var_weights=vsids_w,
                    run_osd=True,
                    priority_scores=prev_rank_scores,
                )
            if use_navigator:
                v_base = diag["v_A_vars"] if bnd_changed else last_ro["v_a"]
                dx = nav.predict_tangent_jump(
                    v_base, b_rep.var_mask, sched.alpha, sched.step_alpha, float(sched.mask_m) / 256.0
                )
                half = k_t // 2
                nudged = torch.where(
                    b_rep.var_mask[half:] > 0.5,
                    b_rep.var_vals[half:],
                    (v_base[half:] + dx[half:]).clamp(-1.25, 1.25),
                )
                state_rep[half:] = gl_rrr.init_replicas_from_vars(nudged)

        # Periodic Bottleneck MPS Joint-NLL evolution + 3-cycle (<2% drop) stagnation bifurcation trigger
        if use_mps and (tick % max(1, int(mps_every)) == 0) and not bifurcated:
            elite = select_diverse_elite_spins(last_ro["v_disc"], last_ro["per_traj_rank"], var_idx, max_keep=32)
            st_mps = mps.update_on_elite_pool(elite, num_iters=1, loop_pairs=lp_pairs)
            s_ent_val = st_mps["S_ent"]
            e_cycle = last_ro["inst_E_hom"] + (0.002 * last_ro["inst_E_all32"] if sched.alpha >= 0.999 else 0.0)
            if not advanced:
                if prev_mps_cycle_e is not None:
                    rel_drop = (prev_mps_cycle_e - e_cycle) / max(prev_mps_cycle_e, 1e-6)
                    if rel_drop < 0.02 and e_cycle > 1e-5:
                        stag_mps_cycles += 1
                    else:
                        stag_mps_cycles = 0
                prev_mps_cycle_e = e_cycle
                if stag_mps_cycles >= max(1, int(stag_mps_cycles_trigger)):
                    stag_mps_cycles = 0
                    sched.record_stagnation_bifurcation(last_ro["inst_E_hom"])
                    bifurcated = True
            else:
                prev_mps_cycle_e = e_cycle

        if bifurcated:
            if use_navigator:
                nav.discard_pending_jump()
            inj_mask = torch.zeros((k_t,), dtype=torch.bool, device=dev)
            if use_mps:
                elite = select_diverse_elite_spins(last_ro["v_disc"], last_ro["per_traj_rank"], var_idx, max_keep=32)
                st_mps = mps.update_on_elite_pool(elite, num_iters=2, loop_pairs=lp_pairs)
                s_ent_val = st_mps["S_ent"]
                bifurc_mps_calls += 1
                n_inj = max(8, k_t // 4)
                worst_idx = torch.topk(diag["gl_gap"], k=n_inj, largest=True).indices
                branch_seeds, _ = mps.conditional_sample(num_samples=n_inj, generator=rng)
                v_inj = last_ro["v_a"][worst_idx].clone()
                v_inj[:, var_idx] = branch_seeds
                v_inj = torch.where(b_rep.var_mask[worst_idx] > 0.5, b_rep.var_vals[worst_idx], v_inj)
                state_rep[worst_idx] = gl_rrr.init_replicas_from_vars(v_inj)
                inj_mask[worst_idx] = True
            if use_bm:
                elite_cut = torch.quantile(last_ro["per_traj_rank"], 0.25)
                mid_mask = (~inj_mask) & (last_ro["per_traj_rank"] > elite_cut)
                kicked_b, k_idx_b, _ = bm.record_and_maybe_escape(
                    last_ro["v_a"],
                    diag["gl_gap"],
                    b_rep.var_mask,
                    b_rep.var_vals,
                    vsids_var_weights=vsids_w,
                    force_escape_mask=mid_mask if bool(mid_mask.any().item()) else None,
                    generator=rng,
                )
                if kicked_b is not None and k_idx_b.numel() > 0:
                    state_rep[k_idx_b] = gl_rrr.init_replicas_from_vars(kicked_b)
            # GL-RRR high-energy pulse burst
            pulse_beta = (beta_vec + 0.12).clamp(0.65, 1.15)
            for p_i in range(3):
                state_rep, diag = gl_rrr.step(
                    state_rep,
                    b_rep.var_mask,
                    b_rep.var_vals,
                    alpha=sched.alpha,
                    beta=pulse_beta,
                    vsids_var_weights=vsids_w,
                    run_osd=(p_i == 0),
                    priority_scores=prev_rank_scores,
                )

        if tick_callback is not None and (tick == 1 or tick % max(1, int(log_interval)) == 0 or tick == max_ticks):
            tick_info = {
                "inst_E32_true": float(last_ro["inst_E32_true"]),
                "inst_E_all32": float(last_ro["inst_E_all32"]),
                "disc_viol_min": int(last_ro["disc_viol_min"]),
                "inst_mean_E": float(last_ro["inst_mean_E"]),
                "inst_E_hom": float(last_ro["inst_E_hom"]),
                "inst_E1_true": float(last_ro["inst_E1_true"]),
                "hist_min_E32": float(hist_min_e32),
                "alpha": float(sched.alpha),
                "mask_m": int(sched.mask_m),
                "alpha_c": sched.alpha_c,
                "bifurc": int(sched.bifurc_count),
                "bifurc_mps_calls": int(bifurc_mps_calls),
                "endgame_polish_runs": int(endgame_polish_runs),
                "S_ent": float(s_ent_val),
                "gl_gap": float(diag["gl_gap"].min().item()),
                "exact_hits": int(last_ro["exact_hits"]),
                "lin_viol": float(last_ro["lin_viol_min"]),
                "nl_err": float(last_ro["nl_err_min"]),
                "align": float(last_ro["align"]),
            }
            tick_callback(tick, tick_info, time.perf_counter() - t0_wall)

    elapsed = time.perf_counter() - t0_wall
    out = {
        "rounds": r,
        "domain_mode": domain_mode,
        "exact_hits": int(exact_hits),
        "solved": bool(exact_hits > 0),
        "inst_E32_true": float(last_ro.get("inst_E32_true", float("inf"))),
        "inst_E_all32": float(last_ro.get("inst_E_all32", float("inf"))),
        "disc_viol_min": int(last_ro.get("disc_viol_min", 0)),
        "inst_mean_E": float(last_ro.get("inst_mean_E", float("inf"))),
        "inst_E_hom": float(last_ro.get("inst_E_hom", float("inf"))),
        "hist_min_E32": float(hist_min_e32),
        "alpha": float(sched.alpha),
        "mask_m": int(sched.mask_m),
        "mask_history": list(sched.mask_history),
        "alpha_c": sched.alpha_c,
        "bifurc": int(sched.bifurc_count),
        "bifurc_mps_calls": int(bifurc_mps_calls),
        "endgame_polish_runs": int(endgame_polish_runs),
        "S_ent": float(s_ent_val),
        "gl_gap": float(diag["gl_gap"].min().item()) if ticks_done > 0 else 0.0,
        "ticks": int(ticks_done),
        "wall_s": round(elapsed, 3),
        "hit_w16": None if hit_w16 is None else hit_w16.detach().cpu().tolist(),
        "best_w16": None if best_w16 is None else best_w16.detach().cpu().tolist(),
    }

    # Strict cross-round GPU memory reclamation (zero dangling engine/tensor references)
    del (
        state_rep,
        b_rep,
        bnd,
        gl_rrr,
        mps,
        bm,
        nav,
        u0,
        last_ro,
        diag,
        prev_rank_scores,
        beta_vec,
        var_idx,
        hit_w16,
        best_w16,
    )
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


def run_phase0_calibration(
    output_dir: Path | str,
    rounds_list: Tuple[int, ...] = (4, 5, 6, 8, 10, 12),
    max_sec_per_round: float = 120.0,
    gpu_tier: str = "40g",
    device: torch.device | str = "cuda",
    seed: int = 6400,
) -> Dict[str, Any]:
    """Phase-0 Critical Phase-Transition Sweep across R = 4, 5, 6, 8, 10, 12 (`free_msg`, no peel-back cheat)."""
    enforce_v6a_fp32()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    num_trajs = 96 if gpu_tier == "40g" else 192

    print(
        f"[Phase-SHA V6A Calibrate] Sweeping R={rounds_list} (free_msg, allow_peel_back=False, num_trajs={num_trajs}, max_sec={max_sec_per_round}s/round)",
        flush=True,
    )
    sweep_rows: List[Dict[str, Any]] = []
    r_c: Optional[int] = None

    for idx_r, r in enumerate(rounds_list):
        gen = torch.Generator(device=dev).manual_seed(seed + idx_r * 101)
        w16 = _sample_free_msg_w16(1, r, dev, gen)
        dig = sha256_digest(w16, rounds=r)
        res = solve_single_digest_v6a(
            digest=dig,
            rounds=r,
            domain_mode="free_msg",
            allow_peel_back=False,
            num_trajs=num_trajs,
            max_ticks=500,
            max_seconds=float(max_sec_per_round),
            start_mask_m=64,
            target_mask_m=256,
            step_alpha=0.08,
            min_step_alpha=0.01,
            max_ticks_per_alpha=8,
            eps_advance=0.018,
            inner_rrr_per_tick=5,
            device=dev,
            seed=seed + r,
        )
        if res["solved"] and res["hit_w16"] is not None:
            hw = torch.tensor([res["hit_w16"]], dtype=torch.int64, device=dev)
            chk_dig = sha256_digest(hw, rounds=r)
            assert bool((chk_dig == dig).all().item()), f"Calibration forward verification failed at R={r}!"
        if r_c is None and not res["solved"]:
            r_c = int(r)
        sweep_rows.append(res)
        print(
            f"  [Cal R={r:2d}] solved={res['solved']} hits={res['exact_hits']} inst_E32={res['inst_E32_true']:.3f} "
            f"inst_E_hom={res['inst_E_hom']:.4f} hist_min={res['hist_min_E32']:.3f} alpha={res['alpha']:.3f} "
            f"m={res['mask_m']} alpha_c={res['alpha_c']} bifurc={res['bifurc']} S_ent={res['S_ent']:.3f} wall={res['wall_s']:.2f}s",
            flush=True,
        )

    ablation_r = r_c if r_c is not None else 8
    gen_ab = torch.Generator(device=dev).manual_seed(seed + 999)
    w16_ab = _sample_free_msg_w16(1, ablation_r, dev, gen_ab)
    dig_ab = sha256_digest(w16_ab, rounds=ablation_r)
    ablation_specs = [
        ("GL-RRR Only", False, False, False, False),
        ("+Homotopy2D", True, False, False, False),
        ("+Homotopy2D+MPS", True, True, False, False),
        ("Full V6A-v2 (+BM+Nav)", True, True, True, True),
    ]
    ablation_results: List[Dict[str, Any]] = []
    ab_sec = min(30.0, 0.25 * float(max_sec_per_round))
    for nm, u_h, u_m, u_b, u_n in ablation_specs:
        ab_r = solve_single_digest_v6a(
            digest=dig_ab,
            rounds=ablation_r,
            domain_mode="free_msg",
            allow_peel_back=False,
            num_trajs=num_trajs,
            max_ticks=250,
            max_seconds=ab_sec,
            use_homotopy=u_h,
            use_mps=u_m,
            use_bm=u_b,
            use_navigator=u_n,
            device=dev,
            seed=seed + ablation_r,
        )
        ab_r["config_name"] = nm
        ablation_results.append(ab_r)

    report = {
        "rounds_list": list(rounds_list),
        "critical_round_R_c": r_c,
        "ablation_round": ablation_r,
        "sweep": sweep_rows,
        "ablation": ablation_results,
    }
    report_path = out_dir / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rs = [x["rounds"] for x in sweep_rows]
    inst_es = [x["inst_E32_true"] for x in sweep_rows]
    hist_es = [x["hist_min_E32"] for x in sweep_rows]
    als = [x["alpha"] for x in sweep_rows]
    ax1 = axes[0]
    ax1.plot(rs, inst_es, "o-", label="inst_E32_true (live)", color="#2563eb", linewidth=2)
    ax1.plot(rs, hist_es, "s--", label="hist_min_E32", color="#16a34a", linewidth=1.8)
    ax1_r = ax1.twinx()
    ax1_r.plot(rs, als, "^:", label="final alpha", color="#dc2626", alpha=0.85)
    ax1_r.set_ylabel("Final Homotopy alpha", color="#dc2626")
    ax1.set_xlabel("Rounds R (free_msg, zero peel-back cheat)")
    ax1.set_ylabel("True 2-adic m=32 Energy")
    ax1.set_title(f"Phase-0 Transition Sweep (Critical R_c={r_c})")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", fontsize=9)

    ax2 = axes[1]
    ab_names = [x["config_name"] for x in ablation_results]
    ab_vals = [x["inst_E32_true"] for x in ablation_results]
    ab_hist = [x["hist_min_E32"] for x in ablation_results]
    xpos = list(range(len(ab_names)))
    ax2.bar([p - 0.18 for p in xpos], ab_vals, width=0.35, label="inst_E32_true", color="#2563eb")
    ax2.bar([p + 0.18 for p in xpos], ab_hist, width=0.35, label="hist_min_E32", color="#16a34a")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(ab_names, rotation=15, fontsize=9)
    ax2.set_ylabel("True 2-adic m=32 Energy")
    ax2.set_title(f"Module Contribution Ablation @ R={ablation_r}")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(fontsize=9)
    fig.suptitle("Phase-SHA V6A Phase-0 Calibration & Mechanism Ablation (Pure FP32)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "phase_transition_curve.png", dpi=150)
    plt.close(fig)
    return report


def run_frontier_push(
    output_dir: Path | str,
    config: Optional[Dict[str, Any]] = None,
    rounds_list: Tuple[int, ...] = (6, 7, 8, 9, 10),
    device: torch.device | str = "cuda",
) -> Dict[str, Any]:
    """Battlefield A: B=1 x M=1024 focused free_msg frontier push across R=6,7,8,9,10 (ZERO cheat)."""
    enforce_v6a_fp32()
    cfg = dict(config or {})
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    num_trajs = int(cfg.get("num_trajs", 1024))
    max_ticks = int(cfg.get("max_ticks", 1500))
    max_sec = float(cfg.get("max_sec_per_round", 300.0))
    start_m = int(cfg.get("start_mask_m", 64))
    target_m = int(cfg.get("target_mask_m", 256))
    step_alpha = float(cfg.get("step_alpha", 0.05))
    min_step_alpha = float(cfg.get("min_step_alpha", 0.02))
    max_tpa = int(cfg.get("max_ticks_per_alpha", 8))
    eps_adv = float(cfg.get("eps_advance", 0.018))
    spike_fac = float(cfg.get("spike_factor", 1.20))
    spike_margin = float(cfg.get("spike_abs_margin", 0.003))
    inner_rrr = int(cfg.get("inner_rrr_per_tick", 6))
    mps_every = int(cfg.get("mps_every", 10))
    stag_mps_trigger = int(cfg.get("stag_mps_cycles_trigger", 3))
    endgame_zoom_k = int(cfg.get("endgame_zoom_k", 12))
    endgame_zoom_k_large = int(cfg.get("endgame_zoom_k_large", 14))
    seed = int(cfg.get("seed", 6606))
    log_every = int(cfg.get("log_every", 50))

    logger = V6ACSVLogger(out_dir, rounds=18)
    print(
        f"[Phase-SHA V6A-v2.1 Frontier Push] Rounds={rounds_list} | free_msg | allow_peel_back=False | "
        f"B=1 x M={num_trajs} | mask_m={start_m}->{target_m} | eps_adv={eps_adv} | spike=({spike_fac},{spike_margin}) | "
        f"zoom_k={endgame_zoom_k}/{endgame_zoom_k_large} | max_ticks={max_ticks} | max_sec={max_sec}s",
        flush=True,
    )

    sweep_rows: List[Dict[str, Any]] = []
    global_step = 0
    total_wall = 0.0
    r_c_new: Optional[int] = None

    for idx_r, r in enumerate(rounds_list):
        round_peak_vram_gb = 0.0
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        gen = torch.Generator(device=dev).manual_seed(seed + idx_r * 131)
        w16 = _sample_free_msg_w16(1, r, dev, gen)
        dig = sha256_digest(w16, rounds=r)

        def _on_tick(local_tick: int, info: Dict[str, Any], round_wall: float) -> None:
            nonlocal global_step, round_peak_vram_gb
            global_step += 1
            if dev.type == "cuda":
                round_peak_vram_gb = max(
                    round_peak_vram_gb, torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
                )
            dt_ms = (round_wall * 1000.0) / float(max(1, local_tick))
            row = {
                "step": global_step,
                "wall_s": round(total_wall + round_wall, 3),
                "dt": round(dt_ms, 2),
                "step_time_ms": round(dt_ms, 2),
                "peak_vram_gb": round(round_peak_vram_gb, 3),
                "lr": 0.0,
                "inst_E_hom": info["inst_E_hom"],
                "inst_E32": info["inst_E32_true"],
                "inst_E32_true": info["inst_E32_true"],
                "inst_E_all32": info.get("inst_E_all32", ""),
                "disc_viol_min": info.get("disc_viol_min", ""),
                "inst_mean_E": info["inst_mean_E"],
                "hist_min": info["hist_min_E32"],
                "hist_min_E32": info["hist_min_E32"],
                "best_global_m32_energy": info["hist_min_E32"],
                "alpha": info["alpha"],
                "mask_m": info["mask_m"],
                "alpha_c": "" if info["alpha_c"] is None else float(info["alpha_c"]),
                "S_ent": info["S_ent"],
                "gl_gap": info["gl_gap"],
                "bifurc": info["bifurc"],
                "bifurc_mps_calls": info["bifurc_mps_calls"],
                "endgame_polish_runs": info["endgame_polish_runs"],
                "exact_hits": info["exact_hits"],
                "exact_preimage_rate@256": 1.0 if info["exact_hits"] > 0 else 0.0,
                "unique_ratio@256": 1.0,
                "align": info["align"],
                "lin_viol": info["lin_viol"],
                "nl_err": info["nl_err"],
                "eval_n_digests": 1,
                "train_loss": info["inst_E_hom"],
                "train_phys_loss": info["inst_E32_true"],
                "min_phys_energy_m32": info["inst_E32_true"],
                "phys_energy_m32": info["inst_mean_E"],
                "phys_energy_m1": info["inst_E1_true"],
            }
            logger.append(row)
            ac_str = "--" if info["alpha_c"] is None else f"{float(info['alpha_c']):.3f}"
            print(
                f"  [R={r:2d} tick={local_tick:4d}/{max_ticks}] inst_E32={info['inst_E32_true']:.4f} "
                f"disc_viol={info.get('disc_viol_min', 0)} hist_min={info['hist_min_E32']:.4f} inst_E_hom={info['inst_E_hom']:.4f} "
                f"alpha={info['alpha']:.3f} mask_m={info['mask_m']:3d} bifurc={info['bifurc']} (mps={info['bifurc_mps_calls']}, a_c={ac_str}) "
                f"S_ent={info['S_ent']:.3f} polish={info['endgame_polish_runs']} hits={info['exact_hits']} "
                f"vram={round_peak_vram_gb:.2f}GB wall={round_wall:.1f}s",
                flush=True,
            )

        res = solve_single_digest_v6a(
            digest=dig,
            rounds=r,
            domain_mode="free_msg",
            allow_peel_back=False,
            num_trajs=num_trajs,
            max_ticks=max_ticks,
            max_seconds=max_sec,
            start_mask_m=start_m,
            target_mask_m=target_m,
            step_alpha=step_alpha,
            min_step_alpha=min_step_alpha,
            max_ticks_per_alpha=max_tpa,
            eps_advance=eps_adv,
            spike_factor=spike_fac,
            spike_abs_margin=spike_margin,
            inner_rrr_per_tick=inner_rrr,
            mps_every=mps_every,
            stag_mps_cycles_trigger=stag_mps_trigger,
            endgame_zoom_k=endgame_zoom_k,
            endgame_zoom_k_large=endgame_zoom_k_large,
            device=dev,
            seed=seed + r * 7,
            tick_callback=_on_tick,
            log_interval=log_every,
        )
        if dev.type == "cuda":
            round_peak_vram_gb = max(
                round_peak_vram_gb, torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
            )
        fwd_ok = False
        if res["solved"] and res["hit_w16"] is not None:
            hw = torch.tensor([res["hit_w16"]], dtype=torch.int64, device=dev)
            chk_dig = sha256_digest(hw, rounds=r)
            fwd_ok = bool((chk_dig == dig).all().item())
            assert fwd_ok, f"Frontier push forward SHA-256 verification failed at R={r}!"
            del hw, chk_dig
        res["forward_verified"] = fwd_ok
        res["round_peak_vram_gb"] = round(round_peak_vram_gb, 3)
        if r_c_new is None and not res["solved"]:
            r_c_new = int(r)
        sweep_rows.append(res)
        # Emit round-end summary row to CSV BEFORE incrementing total_wall
        _on_tick(
            res["ticks"],
            {
                "inst_E32_true": res["inst_E32_true"],
                "inst_E_all32": res.get("inst_E_all32", 0.0),
                "disc_viol_min": res.get("disc_viol_min", 0),
                "inst_mean_E": res["inst_mean_E"],
                "inst_E_hom": res["inst_E_hom"],
                "inst_E1_true": res["inst_E32_true"],
                "hist_min_E32": res["hist_min_E32"],
                "alpha": res["alpha"],
                "mask_m": res["mask_m"],
                "alpha_c": res["alpha_c"],
                "bifurc": res["bifurc"],
                "bifurc_mps_calls": res["bifurc_mps_calls"],
                "endgame_polish_runs": res["endgame_polish_runs"],
                "S_ent": res["S_ent"],
                "gl_gap": res["gl_gap"],
                "exact_hits": res["exact_hits"],
                "lin_viol": 0.0,
                "nl_err": 0.0,
                "align": 1.0,
            },
            float(res["wall_s"]),
        )
        total_wall += float(res["wall_s"])
        ac_s = "--" if res["alpha_c"] is None else f"{float(res['alpha_c']):.3f}"
        print(
            f"[Frontier Done R={r:2d}] solved={res['solved']} hits={res['exact_hits']} fwd_ok={fwd_ok} "
            f"inst_E32={res['inst_E32_true']:.5f} E_all32={res.get('inst_E_all32', 0.0):.3f} disc_viol={res.get('disc_viol_min', 0)} "
            f"hist_min={res['hist_min_E32']:.5f} inst_E_hom={res['inst_E_hom']:.4f} "
            f"alpha={res['alpha']:.3f} mask_m={res['mask_m']} (hist={res['mask_history']}) "
            f"alpha_c={ac_s} bifurc={res['bifurc']} mps_calls={res['bifurc_mps_calls']} polish={res['endgame_polish_runs']} "
            f"vram_peak={round_peak_vram_gb:.2f}GB wall={res['wall_s']:.2f}s",
            flush=True,
        )
        del _on_tick, dig, w16, gen
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            alloc_gb = torch.cuda.memory_allocated(dev) / (1024 ** 3)
            resv_gb = torch.cuda.memory_reserved(dev) / (1024 ** 3)
            print(f"  [VRAM Cleanup R={r:2d}] alloc={alloc_gb:.3f}GB reserved={resv_gb:.3f}GB", flush=True)

    report = {
        "mode": "frontier_push",
        "num_trajs": num_trajs,
        "rounds_list": list(rounds_list),
        "critical_round_R_c": r_c_new,
        "total_wall_s": round(total_wall, 3),
        "sweep": sweep_rows,
    }
    (out_dir / "frontier_push_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rs = [x["rounds"] for x in sweep_rows]
    inst_es = [x["inst_E32_true"] for x in sweep_rows]
    hist_es = [x["hist_min_E32"] for x in sweep_rows]
    bifs = [x["bifurc"] for x in sweep_rows]
    ax1 = axes[0]
    ax1.plot(rs, inst_es, "o-", label="inst_E32_true", color="#dc2626", linewidth=2)
    ax1.plot(rs, hist_es, "s--", label="hist_min_E32", color="#16a34a", linewidth=1.8)
    for r_val, row in zip(rs, sweep_rows):
        tag = "HIT" if row["solved"] else "MISS"
        ax1.annotate(f"{tag}", (r_val, row["hist_min_E32"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax1.set_xlabel("Rounds R (free_msg, zero peel-back cheat, B=1 x M=1024)")
    ax1.set_ylabel("True 2-adic m=32 Energy")
    ax1.set_title(f"Frontier Push Energy Curve (R_c={r_c_new})")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best", fontsize=9)

    ax2 = axes[1]
    ax2.bar([str(x) for x in rs], bifs, color="#7c3aed", alpha=0.85, label="bifurc count")
    ax2.set_xlabel("Rounds R")
    ax2.set_ylabel("Detected Bifurcations (bifurc)")
    ax2.set_title("2D Homotopy Bifurcation Activity per Round")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(fontsize=9)
    fig.suptitle("Phase-SHA V6A-v2 Battlefield A: Frontier Push R=6..10 (Pure FP32)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "frontier_push_curve.png", dpi=150)
    plt.close(fig)
    print(f"[Phase-SHA V6A-v2 Frontier Push] Completed all rounds! R_c={r_c_new} saved to {out_dir}", flush=True)
    return report


class OuroborosV6AEngine:
    """Full batched multi-digest V6A-v2 attack/evaluation engine with lossless --resume."""

    def __init__(self, config: Dict[str, Any], output_dir: Path | str, device: torch.device | str = "cuda") -> None:
        enforce_v6a_fp32()
        self.config = dict(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

        self.rounds = int(config.get("rounds", 18))
        self.domain_mode = str(config.get("domain_mode", "int_range"))
        self.num_bits = int(config.get("num_bits", 32))
        self.msg_bytes_len = int(config.get("msg_bytes_len", 4))
        self.allow_peel_back = bool(config.get("allow_peel_back", self.rounds >= 8 and self.domain_mode != "free_msg"))

        self.num_digests = int(config.get("num_digests", 4))
        self.trajs_per_digest = int(config.get("trajs_per_digest", 256))
        self.inner_steps_per_tick = int(config.get("inner_steps_per_tick", 6))
        self.total_steps = int(config.get("steps", 2000))
        self.holdout_size = int(config.get("holdout_size", 2048))
        self.holdout_seed = int(config.get("holdout_seed", 20261122))
        self.use_r20_stress = bool(config.get("r20_stress", False))
        self.seed = int(config.get("seed", 6418))

        self.rng = torch.Generator(device=self.device).manual_seed(self.seed)
        self.cpu_rng = torch.Generator(device="cpu").manual_seed(self.seed + 77)

        if self.use_r20_stress or self.rounds == 20:
            w16_cpu, _ = build_r20_stress_holdout_w16(
                holdout_size=max(self.num_digests, self.holdout_size),
                holdout_seed=self.holdout_seed,
                domain_mode="int_range" if self.domain_mode == "free_msg" else self.domain_mode,
                num_bits=self.num_bits,
                msg_bytes_len=self.msg_bytes_len,
                device="cpu",
            )
        elif self.domain_mode == "free_msg":
            w16_cpu = _sample_free_msg_w16(
                max(self.num_digests, min(256, self.holdout_size)), self.rounds, torch.device("cpu"), self.cpu_rng
            )
        else:
            w16_cpu, _ = build_v5_holdout_w16(
                holdout_size=max(self.num_digests, self.holdout_size),
                holdout_seed=self.holdout_seed,
                domain_mode=self.domain_mode,
                num_bits=self.num_bits,
                msg_bytes_len=self.msg_bytes_len,
                device="cpu",
            )

        self.holdout_w16 = w16_cpu
        self.n_dig = min(self.num_digests, self.holdout_w16.shape[0])
        self.k_traj = max(4, self.trajs_per_digest)
        self.w16_active = self.holdout_w16[: self.n_dig].to(self.device)
        self.target_digests = sha256_digest(self.w16_active, rounds=self.rounds)
        tr_gt = sha256_trace(self.w16_active, rounds=self.rounds)
        self.x_star_nodes = torch.stack(
            [
                tr_gt.A_full[:, 3 : 3 + self.rounds],
                tr_gt.E_full[:, 3 : 3 + self.rounds],
                tr_gt.W_full[:, : self.rounds],
            ],
            dim=2,
        )
        self.x_star_spins = words_to_spins(self.x_star_nodes, dtype=torch.float32)

        self.osd_every_ticks = max(1, int(config.get("osd_every", 12)))
        self.gl_rrr = GlobalLinearRRREngine(
            rounds=self.rounds,
            bp_iters=int(config.get("bp_iters", 7)),
            bp_damping=float(config.get("bp_damping", 0.72)),
            osd_every=self.osd_every_ticks,
            osd_top_k=int(config.get("osd_top_k", 12)),
        ).to(self.device)
        self.spec = self.gl_rrr.spec

        start_m = int(config.get("start_mask_m", 64))
        self.sched = HomotopySchedule2D(
            start_alpha=float(config.get("start_alpha", 0.0)),
            start_mask_m=start_m,
            target_mask_m=int(config.get("target_mask_m", 256)),
            step_alpha=float(config.get("step_alpha", 0.05)),
            min_step_alpha=float(config.get("min_step_alpha", 0.01)),
            step_m=int(config.get("step_m", 32)),
            eps_advance=float(config.get("eps_advance", 0.018)),
            spike_factor=float(config.get("spike_factor", 1.20)),
            spike_abs_margin=float(config.get("spike_abs_margin", 0.003)),
            max_ticks_per_alpha=int(config.get("max_ticks_per_alpha", 12)),
            baseline_window=int(config.get("baseline_window", 3)),
            interleaved_2d=bool(config.get("interleaved_2d", True)),
        )

        self.bnd_single: Optional[V6ABoundary] = None
        self.bnd_rep: Optional[V6ABoundary] = None
        self._rebuild_boundaries()

        assert self.bnd_single is not None
        self.var_idx, lp_full = select_bottleneck_var_indices(
            self.spec, self.bnd_single.var_mask[0], max_sites=int(config.get("mps_sites", 64))
        )
        self.lp_pairs = [(a, b) for a, b, _ in lp_full]
        self.mps = BottleneckMPSBornMachine(
            num_sites=len(self.var_idx),
            bond_dim=int(config.get("mps_bond_dim", 8)),
            min_entropy=float(config.get("mps_min_entropy", 0.55)),
            ent_floor_weight=float(config.get("mps_ent_weight", 0.50)),
            seed=self.seed + 11,
        ).to(self.device)
        self.bm = BurerMonteiroEscapeLayer(
            n_var_bits=self.spec.n_var_bits,
            manifold_dim=int(config.get("bm_manifold_dim", 16)),
            window_size=int(config.get("bm_window", 8)),
            stag_patience=int(config.get("bm_patience", 12)),
            cayley_t=float(config.get("bm_cayley_t", 0.42)),
        ).to(self.device)
        self.navigator = ZeroBrakeNavigator(
            n_var_words=self.spec.n_var_words,
            hidden=int(config.get("nav_hidden", 64)),
            lr=float(config.get("nav_lr", 1e-3)),
        ).to(self.device)

        self.mps_every = max(1, int(config.get("mps_every", 10)))
        self.sched_every = max(1, int(config.get("sched_every", 2)))
        self.stag_mps_cycles_trigger = max(1, int(config.get("stag_mps_cycles_trigger", 3)))
        self.endgame_zoom_k = int(config.get("endgame_zoom_k", 12))
        self.endgame_zoom_k_large = int(config.get("endgame_zoom_k_large", 14))
        self.best_ckpt_name = str(config.get("best_ckpt_name", "best.pt"))
        self.latest_ckpt_name = str(config.get("latest_ckpt_name", "latest.pt"))

        self.state_rep: Optional[torch.Tensor] = None
        self.beta_vec = torch.linspace(0.68, 1.10, self.k_traj, dtype=torch.float32, device=self.device).repeat(self.n_dig)
        self.hist_best_m32 = torch.full((self.n_dig,), 1e9, dtype=torch.float32, device=self.device)
        self.hist_best_global = 1e9
        self.hist_exact_hits = 0
        self.bifurc_mps_calls = 0
        self.endgame_polish_runs = 0
        self.step = 0
        self.wall_s = 0.0
        self.last_s_ent = float(self.mps.compute_entanglement_entropy().item())
        self.last_esc_count = 0.0
        self.last_gl_gap = 0.0
        self.last_ro: Optional[Dict[str, Any]] = None
        self._last_rank_scores: Optional[torch.Tensor] = None
        self._prev_mps_cycle_e: Optional[float] = None
        self._stag_mps_cycles = 0

        self.logger = V6ACSVLogger(self.output_dir, rounds=self.rounds)

    def _rebuild_boundaries(self) -> None:
        self.bnd_single = build_v6a_boundary(
            digest=self.target_digests,
            rounds=self.rounds,
            domain_mode=self.domain_mode,
            num_bits=self.num_bits,
            msg_bytes_len=self.msg_bytes_len,
            mask_m=self.sched.mask_m,
            allow_peel_back=self.allow_peel_back,
        )
        self.bnd_rep = self.bnd_single.repeat_for_candidates(self.k_traj)

    @torch.no_grad()
    def _init_population_alpha0(self) -> None:
        assert self.bnd_single is not None
        init_parts: List[torch.Tensor] = []
        for d_i in range(self.n_dig):
            u0, _, _ = solve_alpha0_gf2_linear_skeleton(
                self.spec,
                self.bnd_single.var_mask[d_i],
                self.bnd_single.var_vals[d_i],
                num_samples=self.k_traj,
                generator=self.cpu_rng,
            )
            init_parts.append(self.gl_rrr.init_replicas_from_vars(u0))
        self.state_rep = torch.cat(init_parts, dim=0)

    def save_checkpoint(self, filename: str = "latest.pt") -> Path:
        ckpt_path = self.output_dir / filename
        payload: Dict[str, Any] = {
            "version": "v6a_v2",
            "step": self.step,
            "wall_s": self.wall_s,
            "config": self.config,
            "sched_state": self.sched.state_dict(),
            "mps_state": self.mps.state_dict(),
            "mps_opt": self.mps.opt.state_dict(),
            "navigator_state": self.navigator.state_dict(),
            "navigator_opt": self.navigator.opt.state_dict(),
            "state_rep": None if self.state_rep is None else self.state_rep.detach().cpu(),
            "hist_best_m32": self.hist_best_m32.detach().cpu(),
            "hist_best_global": float(self.hist_best_global),
            "hist_exact_hits": int(self.hist_exact_hits),
            "bifurc_mps_calls": int(self.bifurc_mps_calls),
            "endgame_polish_runs": int(self.endgame_polish_runs),
            "last_s_ent": float(self.last_s_ent),
            "bm_total_escapes": int(self.bm.total_escapes),
            "gl_tick": int(self.gl_rrr._tick_counter),
            "prev_mps_cycle_e": self._prev_mps_cycle_e,
            "stag_mps_cycles": int(self._stag_mps_cycles),
        }
        torch.save(payload, ckpt_path)
        return ckpt_path

    def load_checkpoint(self, ckpt_path: str | Path) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.step = int(ckpt.get("step", 0))
        self.wall_s = float(ckpt.get("wall_s", 0.0))
        if "sched_state" in ckpt:
            self.sched.load_state_dict(ckpt["sched_state"])
            self._rebuild_boundaries()
        if "mps_state" in ckpt:
            self.mps.load_state_dict(ckpt["mps_state"])
        if "mps_opt" in ckpt:
            self.mps.opt.load_state_dict(ckpt["mps_opt"])
        if "navigator_state" in ckpt:
            self.navigator.load_state_dict(ckpt["navigator_state"])
        if "navigator_opt" in ckpt:
            self.navigator.opt.load_state_dict(ckpt["navigator_opt"])
        if ckpt.get("state_rep") is not None:
            self.state_rep = ckpt["state_rep"].to(self.device)
        if ckpt.get("hist_best_m32") is not None:
            self.hist_best_m32 = ckpt["hist_best_m32"].to(self.device)
        self.hist_best_global = float(ckpt.get("hist_best_global", self.hist_best_global))
        self.hist_exact_hits = int(ckpt.get("hist_exact_hits", self.hist_exact_hits))
        self.bifurc_mps_calls = int(ckpt.get("bifurc_mps_calls", 0))
        self.endgame_polish_runs = int(ckpt.get("endgame_polish_runs", 0))
        self.last_s_ent = float(ckpt.get("last_s_ent", self.last_s_ent))
        self.bm.total_escapes = int(ckpt.get("bm_total_escapes", 0))
        self.gl_rrr._tick_counter = int(ckpt.get("gl_tick", 0))
        self._prev_mps_cycle_e = ckpt.get("prev_mps_cycle_e", None)
        self._stag_mps_cycles = int(ckpt.get("stag_mps_cycles", 0))

    def _compute_metrics_snapshot(self, sync_full: bool = True) -> Dict[str, Any]:
        assert self.state_rep is not None and self.bnd_rep is not None and self.bnd_single is not None
        vsids_w = self.navigator.get_vsids_weights(self.bnd_rep.var_mask)
        ro = self.gl_rrr.readout_shadow(self.state_rep, self.bnd_rep, alpha=self.sched.alpha, vsids_var_weights=vsids_w)
        self.last_ro = ro
        self._last_rank_scores = ro["per_traj_rank"]

        m32_mat = ro["per_traj_m32"].view(self.n_dig, self.k_traj)
        all32_mat = ro["per_traj_all32"].view(self.n_dig, self.k_traj)
        rank_mat = ro["per_traj_rank"].view(self.n_dig, self.k_traj)
        m1_mat = ro["per_traj_m1"].view(self.n_dig, self.k_traj)
        hom_mat = ro["per_traj_hom"].view(self.n_dig, self.k_traj)
        hits_mat = ro["exact_hits_mask"].view(self.n_dig, self.k_traj)

        per_dig_inst_min = m32_mat.min(dim=1).values
        _, best_k_idx = rank_mat.min(dim=1)
        inst_e32_true = float(per_dig_inst_min.mean().item())
        inst_e_all32 = float(all32_mat.min(dim=1).values.mean().item())
        inst_mean_e = float(m32_mat.mean().item())
        inst_e1 = float(m1_mat.gather(1, best_k_idx.unsqueeze(1)).mean().item())
        inst_e_hom = float(hom_mat.min(dim=1).values.mean().item())
        inst_global_min = float(m32_mat.min().item())

        self.hist_best_m32 = torch.minimum(self.hist_best_m32, per_dig_inst_min)
        hist_min_mean = float(self.hist_best_m32.mean().item())
        self.hist_best_global = min(self.hist_best_global, inst_global_min)

        dig_solved = hits_mat.sum(dim=1) > 0
        exact_hits_cur = int(hits_mat.sum().item())
        self.hist_exact_hits = max(self.hist_exact_hits, exact_hits_cur)
        solve_rate = float(dig_solved.float().mean().item())

        w0_mat = ro["w16_cand"][:, 0].view(self.n_dig, self.k_traj).detach().cpu().tolist()
        uniq = sum(float(len(set(row))) / float(self.k_traj) for row in w0_mat) / max(1, self.n_dig)

        out: Dict[str, Any] = {
            "inst_E_hom": inst_e_hom,
            "inst_E32": inst_e32_true,
            "inst_E32_true": inst_e32_true,
            "inst_E_all32": inst_e_all32,
            "disc_viol_min": int(ro["disc_viol_min"]),
            "inst_mean_E": inst_mean_e,
            "hist_min": hist_min_mean,
            "hist_min_E32": hist_min_mean,
            "best_global_m32_energy": float(self.hist_best_global),
            "alpha": float(self.sched.alpha),
            "mask_m": int(self.sched.mask_m),
            "alpha_c": "" if self.sched.alpha_c is None else float(self.sched.alpha_c),
            "S_ent": float(self.last_s_ent),
            "gl_gap": float(self.last_gl_gap),
            "bifurc": int(self.sched.bifurc_count),
            "bifurc_mps_calls": int(self.bifurc_mps_calls),
            "endgame_polish_runs": int(self.endgame_polish_runs),
            "exact_hits": int(exact_hits_cur),
            "exact_preimage_rate@256": solve_rate,
            "unique_ratio@256": float(uniq),
            "align": float(ro["align"]),
            "drift_slope": float(ro["drift_slope"]),
            "drift_flag": int(ro["drift_flag"]),
            "esc_count": float(self.last_esc_count),
            "pred_mse": float(self.navigator.last_pred_mse),
            "lin_viol": float(ro["lin_viol_min"]),
            "nl_err": float(ro["nl_err_min"]),
            "eval_n_digests": int(self.n_dig),
            # Honest dashboard compatibility (inst_E32 maps to live phys_energy / min_phys_energy, NEVER running-min!)
            "train_loss": inst_e_hom,
            "train_phys_loss": inst_e32_true,
            "train_bce_loss": 0.6931,
            "train_xor_loss": float(self.last_gl_gap),
            "train_comm_loss": float(self.last_s_ent),
            "train_align_loss": float(ro["align"]),
            "eval_bce_loss": 0.6931,
            "min_phys_energy_m32": inst_e32_true,
            "phys_energy_m32": inst_mean_e,
            "phys_energy_m1": inst_e1,
            "zero_latch_ratio": 0.0,
            "zero_manifold_align": float(ro["align"]),
            "zero_manifold_comm": float(self.last_s_ent),
        }
        if sync_full:
            d_idx = torch.arange(self.n_dig, device=self.device)
            best_nodes = ro["x_nodes"].view(self.n_dig, self.k_traj, self.rounds, 3, 32)[d_idx, best_k_idx]
            free_m = 1.0 - self.bnd_single.clamp_mask_nodes
            r_acc = _compute_per_round_acc(best_nodes, self.x_star_spins, free_m)
            n_free_w0 = int(free_m[:, 0, 2, :].sum().item())
            match_w0 = (((best_nodes[:, 0, 2, :] < 0) == (self.x_star_spins[:, 0, 2, :] < 0)).to(torch.float32) * free_m[:, 0, 2, :])
            w0_acc = float(match_w0.sum().item() / max(1.0, float(n_free_w0)))
            out["zero_w0_bit_acc"] = w0_acc
            out["w0_free_bit_acc"] = w0_acc
            out["eval_free_bits"] = int(free_m.sum().item())
            out["w0_zscore"] = round(_binomial_zscore(w0_acc, n_free_w0), 3)
            for t in range(self.rounds):
                out[f"zero_round_acc_{t}"] = float(r_acc[t])
                out[f"per_round_acc_{t}"] = float(r_acc[t])
            for r_tag in (12, 13, 16):
                if r_tag < self.rounds:
                    n_rt = int(free_m[:, r_tag, :, :].sum().item())
                    out[f"r{r_tag}_zscore"] = round(_binomial_zscore(float(r_acc[r_tag]), n_rt), 3)
                else:
                    out[f"r{r_tag}_zscore"] = 0.0
        return out

    def train_one_tick(self, sync_metrics: bool = True) -> Optional[Dict[str, Any]]:
        if self.state_rep is None:
            self._init_population_alpha0()
        assert self.state_rep is not None and self.bnd_rep is not None

        for s_in in range(self.inner_steps_per_tick):
            vsids_w = self.navigator.get_vsids_weights(self.bnd_rep.var_mask)
            self.state_rep, diag = self.gl_rrr.step(
                self.state_rep,
                self.bnd_rep.var_mask,
                self.bnd_rep.var_vals,
                alpha=self.sched.alpha,
                beta=self.beta_vec,
                vsids_var_weights=vsids_w,
                run_osd=(s_in == 0 and ((self.step + 1) % self.osd_every_ticks == 0)),
                priority_scores=self._last_rank_scores,
            )
            self.navigator.record_gap(diag["var_gap_profile"])

        kicked, k_idx, esc_stats = self.bm.record_and_maybe_escape(
            diag["v_A_vars"],
            diag["gl_gap"],
            self.bnd_rep.var_mask,
            self.bnd_rep.var_vals,
            vsids_var_weights=self.navigator.get_vsids_weights(self.bnd_rep.var_mask),
            generator=self.rng,
        )
        if kicked is not None and k_idx.numel() > 0:
            self.state_rep[k_idx] = self.gl_rrr.init_replicas_from_vars(kicked)

        self.last_esc_count = esc_stats["esc_count"]
        self.last_gl_gap = float(diag["gl_gap"].min().item())
        self.step += 1

        do_mps_periodic = (self.step % self.mps_every) == 0
        do_sched = (self.step % self.sched_every) == 0
        if sync_metrics or do_mps_periodic or do_sched:
            snap = self._compute_metrics_snapshot(sync_full=sync_metrics)
            assert self.last_ro is not None
            vsids_w = self.navigator.get_vsids_weights(self.bnd_rep.var_mask)

            if (
                self.sched.alpha >= 0.999
                and self.sched.mask_m >= self.sched.target_mask_m
                and (
                    snap["inst_E32"] < 0.05
                    or snap.get("disc_viol_min", 9999) <= 16
                    or snap["inst_E_hom"] < 0.015
                )
                and (self.step % 4 == 0)
            ):
                self.endgame_polish_runs += 1
                solved_eg, _, cnt_eg, top_eg, sub_eg = self.gl_rrr.endgame_polish(
                    self.state_rep,
                    self.bnd_rep,
                    self.last_ro["v_a"],
                    self.last_ro["per_traj_m32"],
                    vsids_var_weights=vsids_w,
                    top_k=6,
                    per_traj_rank=self.last_ro["per_traj_rank"],
                    zoom_k=self.endgame_zoom_k,
                    zoom_k_large=self.endgame_zoom_k_large,
                )
                self.state_rep[top_eg] = sub_eg
                if solved_eg:
                    snap["exact_hits"] = max(int(snap["exact_hits"]), cnt_eg)
                    self.hist_exact_hits = max(self.hist_exact_hits, int(snap["exact_hits"]))
                snap["endgame_polish_runs"] = int(self.endgame_polish_runs)

            advanced, bifurcated, bnd_changed = (False, False, False)
            if do_sched:
                advanced, bifurcated, bnd_changed = self.sched.update(
                    snap["inst_E_hom"], gl_gap=self.last_gl_gap, dt_ticks=self.sched_every
                )
                snap["alpha"] = float(self.sched.alpha)
                snap["mask_m"] = int(self.sched.mask_m)
                snap["bifurc"] = int(self.sched.bifurc_count)
                snap["alpha_c"] = "" if self.sched.alpha_c is None else float(self.sched.alpha_c)

            if advanced:
                top_m = self.last_ro["per_traj_hom"] <= torch.quantile(self.last_ro["per_traj_hom"], 0.35)
                self.navigator.train_on_successful_jump(self.last_ro["v_a"], best_mask=top_m)
                if bnd_changed:
                    self._rebuild_boundaries()
                    v_c = torch.where(self.bnd_rep.var_mask > 0.5, self.bnd_rep.var_vals, self.last_ro["v_a"])
                    self.state_rep = self.gl_rrr.init_replicas_from_vars(v_c)
                    vsids_w = self.navigator.get_vsids_weights(self.bnd_rep.var_mask)
                    self.state_rep, diag = self.gl_rrr.step(
                        self.state_rep,
                        self.bnd_rep.var_mask,
                        self.bnd_rep.var_vals,
                        alpha=self.sched.alpha,
                        beta=self.beta_vec,
                        vsids_var_weights=vsids_w,
                        run_osd=True,
                        priority_scores=self._last_rank_scores,
                    )
                v_base = diag["v_A_vars"] if bnd_changed else self.last_ro["v_a"]
                dx = self.navigator.predict_tangent_jump(
                    v_base,
                    self.bnd_rep.var_mask,
                    self.sched.alpha,
                    self.sched.step_alpha,
                    float(self.sched.mask_m) / 256.0,
                )
                half = self.state_rep.shape[0] // 2
                nudged = torch.where(
                    self.bnd_rep.var_mask[half:] > 0.5,
                    self.bnd_rep.var_vals[half:],
                    (v_base[half:] + dx[half:]).clamp(-1.25, 1.25),
                )
                self.state_rep[half:] = self.gl_rrr.init_replicas_from_vars(nudged)

            if do_mps_periodic and not bifurcated:
                e_cycle = snap["inst_E_hom"] + (0.002 * snap.get("inst_E_all32", 0.0) if self.sched.alpha >= 0.999 else 0.0)
                if not advanced:
                    if self._prev_mps_cycle_e is not None:
                        rel_drop = (self._prev_mps_cycle_e - e_cycle) / max(self._prev_mps_cycle_e, 1e-6)
                        if rel_drop < 0.02 and e_cycle > 1e-5:
                            self._stag_mps_cycles += 1
                        else:
                            self._stag_mps_cycles = 0
                    self._prev_mps_cycle_e = e_cycle
                    if self._stag_mps_cycles >= self.stag_mps_cycles_trigger:
                        self._stag_mps_cycles = 0
                        self.sched.record_stagnation_bifurcation(snap["inst_E_hom"])
                        snap["bifurc"] = int(self.sched.bifurc_count)
                        snap["alpha_c"] = "" if self.sched.alpha_c is None else float(self.sched.alpha_c)
                        bifurcated = True
                else:
                    self._prev_mps_cycle_e = e_cycle

            if bifurcated:
                self.navigator.discard_pending_jump()

            if do_mps_periodic or bifurcated:
                elite = select_diverse_elite_spins(
                    self.last_ro["v_disc"], self.last_ro["per_traj_rank"], self.var_idx, max_keep=32
                )
                st_mps = self.mps.update_on_elite_pool(
                    elite, num_iters=2 if bifurcated else 1, loop_pairs=self.lp_pairs
                )
                self.last_s_ent = st_mps["S_ent"]
                snap["S_ent"] = self.last_s_ent
                snap["train_comm_loss"] = self.last_s_ent
                snap["zero_manifold_comm"] = self.last_s_ent

                if bifurcated or (self.sched.alpha >= 0.999 and (self.step % (3 * self.mps_every) == 0)):
                    n_inj = (
                        max(8, self.state_rep.shape[0] // 4)
                        if bifurcated
                        else max(4, self.state_rep.shape[0] // 10)
                    )
                    worst_idx = torch.topk(diag["gl_gap"], k=n_inj, largest=True).indices
                    b_seeds, _ = self.mps.conditional_sample(num_samples=n_inj, generator=self.rng)
                    v_inj = self.last_ro["v_a"][worst_idx].clone()
                    v_inj[:, self.var_idx] = b_seeds
                    v_inj = torch.where(self.bnd_rep.var_mask[worst_idx] > 0.5, self.bnd_rep.var_vals[worst_idx], v_inj)
                    self.state_rep[worst_idx] = self.gl_rrr.init_replicas_from_vars(v_inj)
                    if bifurcated:
                        self.bifurc_mps_calls += 1
                        snap["bifurc_mps_calls"] = int(self.bifurc_mps_calls)
                        inj_mask = torch.zeros((self.state_rep.shape[0],), dtype=torch.bool, device=self.device)
                        inj_mask[worst_idx] = True
                        elite_cut = torch.quantile(self.last_ro["per_traj_rank"], 0.25)
                        mid_mask = (~inj_mask) & (self.last_ro["per_traj_rank"] > elite_cut)
                        kicked_b, k_idx_b, _ = self.bm.record_and_maybe_escape(
                            self.last_ro["v_a"],
                            diag["gl_gap"],
                            self.bnd_rep.var_mask,
                            self.bnd_rep.var_vals,
                            vsids_var_weights=vsids_w,
                            force_escape_mask=mid_mask if bool(mid_mask.any().item()) else None,
                            generator=self.rng,
                        )
                        if kicked_b is not None and k_idx_b.numel() > 0:
                            self.state_rep[k_idx_b] = self.gl_rrr.init_replicas_from_vars(kicked_b)

                        pulse_beta = (self.beta_vec + 0.12).clamp(0.65, 1.15)
                        for p_i in range(3):
                            self.state_rep, diag = self.gl_rrr.step(
                                self.state_rep,
                                self.bnd_rep.var_mask,
                                self.bnd_rep.var_vals,
                                alpha=self.sched.alpha,
                                beta=pulse_beta,
                                vsids_var_weights=vsids_w,
                                run_osd=(p_i == 0),
                                priority_scores=self._last_rank_scores,
                            )
                        if self.sched.bifurc_count <= 3:
                            self.save_checkpoint(f"bifurc_step_{self.step}.pt")
            return snap if sync_metrics else None
        return None

    def train(self, max_steps: Optional[int] = None) -> None:
        target_steps = self.total_steps if max_steps is None else int(max_steps)
        log_every = int(self.config.get("log_every", 25))
        eval_every = int(self.config.get("eval_every", 200))
        ckpt_every = int(self.config.get("checkpoint_every", 250))

        stop_requested = False
        prev_handler = signal.getsignal(signal.SIGTERM)

        def _on_sigterm(_signum: int, _frame: Any) -> None:
            nonlocal stop_requested
            stop_requested = True

        try:
            signal.signal(signal.SIGTERM, _on_sigterm)
        except Exception:
            pass

        try:
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            if self.state_rep is None:
                self._init_population_alpha0()

            print(
                f"[Phase-SHA V6A-v2] Starting FP32 attack: R={self.rounds} ({self.domain_mode}), "
                f"peel={self.allow_peel_back}, B={self.n_dig}xM={self.k_traj} (total={self.n_dig*self.k_traj}), "
                f"inner={self.inner_steps_per_tick}, alpha={self.sched.alpha:.3f}, mask_m={self.sched.mask_m}->{self.sched.target_mask_m}, "
                f"eps_adv={self.sched.eps_advance:.4f}, device={self.device}, step={self.step}/{target_steps}",
                flush=True,
            )

            if self.step == 0:
                t0 = time.perf_counter()
                s0 = self._compute_metrics_snapshot(sync_full=True)
                self.wall_s += time.perf_counter() - t0
                s0.update({"step": 0, "wall_s": round(self.wall_s, 3), "dt": 0.0, "step_time_ms": 0.0, "peak_vram_gb": 0.0, "lr": 0.0})
                self.logger.append(s0)
                self.save_checkpoint(self.best_ckpt_name)
                print(
                    f"  [eval step=0] N_eval={self.n_dig} free_bits={s0.get('eval_free_bits', 0)} "
                    f"inst_E32={s0['inst_E32']:.2f} hist_min={s0['hist_min']:.2f} inst_E_hom={s0['inst_E_hom']:.4f} "
                    f"alpha={s0['alpha']:.3f} mask_m={s0['mask_m']} S_ent={s0['S_ent']:.3f} exact_hits={s0['exact_hits']}",
                    flush=True,
                )

            best_inst_seen = float("inf")
            t_int0 = time.perf_counter()
            steps_in_int = 0

            while self.step < target_steps and not stop_requested:
                next_step = self.step + 1
                do_log = (
                    (next_step % log_every == 0)
                    or (next_step % eval_every == 0)
                    or (next_step == target_steps)
                    or (self.step == 0)
                )
                snap = self.train_one_tick(sync_metrics=do_log)
                steps_in_int += 1

                if do_log and snap is not None:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                        peak_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                    else:
                        peak_gb = 0.0
                    now = time.perf_counter()
                    el_int = now - t_int0
                    self.wall_s += el_int
                    dt_ms = (el_int * 1000.0) / float(max(1, steps_in_int))
                    t_int0 = now
                    steps_in_int = 0

                    snap.update(
                        {
                            "step": self.step,
                            "wall_s": round(self.wall_s, 3),
                            "dt": round(dt_ms, 2),
                            "step_time_ms": round(dt_ms, 2),
                            "peak_vram_gb": round(peak_gb, 3),
                            "lr": 0.0,
                        }
                    )
                    self.logger.append(snap)
                    print(
                        f"step={self.step:6d}/{target_steps} inst_E32={snap['inst_E32']:.2f} inst_mean_E={snap['inst_mean_E']:.2f} "
                        f"disc_viol={snap.get('disc_viol_min', 0)} hist_min={snap['hist_min']:.2f} inst_E_hom={snap['inst_E_hom']:.4f} "
                        f"alpha={snap['alpha']:.3f} mask_m={snap['mask_m']} S_ent={snap['S_ent']:.3f} gl_gap={snap['gl_gap']:.4f} "
                        f"bifurc={snap['bifurc']} mps_bif={snap['bifurc_mps_calls']} exact_hits={snap['exact_hits']} "
                        f"vram={peak_gb:.2f}GB dt={dt_ms:.1f}ms wall_s={self.wall_s:.1f}s",
                        flush=True,
                    )
                    if (self.step % eval_every == 0) or (self.step == target_steps):
                        ac_str = "--" if snap["alpha_c"] == "" else f"{float(snap['alpha_c']):.3f}"
                        z_w0 = float(snap.get("w0_zscore", 0.0))
                        z_r12 = float(snap.get("r12_zscore", 0.0))
                        z_r13 = float(snap.get("r13_zscore", 0.0))
                        z_r16 = float(snap.get("r16_zscore", 0.0))
                        sig_mark = lambda z: "[*SIG*]" if abs(z) > 3.5 else ""
                        print(
                            f"  [eval] N_eval={self.n_dig} free_bits={snap.get('eval_free_bits', 0)} gMin={snap['best_global_m32_energy']:.2f} "
                            f"w0={snap['zero_w0_bit_acc']:.4f}(z={z_w0:+.2f}{sig_mark(z_w0)}) "
                            f"r12={snap.get('zero_round_acc_12', 0):.3f}(z={z_r12:+.2f}{sig_mark(z_r12)}) "
                            f"r13={snap.get('zero_round_acc_13', 0):.3f}(z={z_r13:+.2f}{sig_mark(z_r13)}) "
                            f"r16={snap.get('zero_round_acc_16', 0):.3f}(z={z_r16:+.2f}{sig_mark(z_r16)}) "
                            f"alpha_c={ac_str} uniq={snap['unique_ratio@256']:.3f} drift={snap['drift_slope']:+.4f}(flag={snap['drift_flag']})",
                            flush=True,
                        )
                        if snap["inst_E32"] < best_inst_seen:
                            best_inst_seen = snap["inst_E32"]
                            self.save_checkpoint(self.best_ckpt_name)
                    if (self.step % ckpt_every == 0) or (self.step == target_steps):
                        self.save_checkpoint(self.latest_ckpt_name)
                    if snap["exact_hits"] > 0 and self.rounds <= 12:
                        self.save_checkpoint(self.best_ckpt_name)
                        print(f"[Phase-SHA V6A-v2] Exact preimage hit at step={self.step}! Saving best.pt.", flush=True)
                        break

            if stop_requested:
                self.save_checkpoint(self.latest_ckpt_name)
                print(f"[Phase-SHA V6A-v2] Caught SIGTERM at step={self.step}; saved {self.latest_ckpt_name} cleanly.", flush=True)
        finally:
            try:
                signal.signal(signal.SIGTERM, prev_handler)
            except Exception:
                pass
