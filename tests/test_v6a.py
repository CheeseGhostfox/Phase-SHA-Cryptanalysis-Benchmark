"""Comprehensive strict-FP32 acceptance tests for Phase-SHA V6A (`Ouroboros`)."""

from __future__ import annotations

import math
import time
from pathlib import Path

import pytest
import torch

from scripts.colab_monitor_dashboard import _read_metrics_csv
from src.boundary_clamp import words_to_bits
from src.dual_basis import DualBasisEngine
from src.sha256_ref import MASK32, sha256_digest, words_to_spins
from v6a import assert_strict_fp32, enforce_v6a_fp32
from v6a.v6a_bm_escape import BurerMonteiroEscapeLayer
from v6a.v6a_bottleneck_mps import (
    BottleneckMPSBornMachine,
    select_bottleneck_var_indices,
    select_diverse_elite_spins,
)
from v6a.v6a_engine import OuroborosV6AEngine, _sample_free_msg_w16, run_frontier_push, solve_single_digest_v6a
from v6a.v6a_gl_rrr import GlobalLinearRRREngine, compute_true_2adic_energy
from v6a.v6a_homotopy import (
    HomotopySchedule2D,
    build_v6a_boundary,
    build_v6a_circuit_spec,
    embed_ground_truth_vars,
    solve_alpha0_gf2_linear_skeleton,
)
from v6a.v6a_navigator import ZeroBrakeNavigator


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# Test 1: Layer 1 Compatible GF(2) Skeleton + Carry Slacks at m=256
# ==============================================================================


@pytest.mark.parametrize(
    "rounds,domain_mode,allow_peel",
    [
        (4, "free_msg", False),
        (5, "free_msg", False),
        (18, "int_range", True),
    ],
)
def test_v6a_homotopy_alpha0_gf2_exact_r4_r5_r18(
    rounds: int, domain_mode: str, allow_peel: bool
) -> None:
    enforce_v6a_fp32()
    dev = _device()
    gen = torch.Generator(device=dev).manual_seed(7100 + rounds)
    w16 = torch.randint(0, 1 << 32, (1, 16), dtype=torch.int64, device=dev, generator=gen)
    if domain_mode == "int_range":
        w16[:, 1] = 0x80000000
        w16[:, 2:15] = 0
        w16[:, 15] = 32
    elif rounds < 16:
        w16[:, rounds:] = 0
    dig = sha256_digest(w16, rounds=rounds)

    spec = build_v6a_circuit_spec(rounds)
    bnd = build_v6a_boundary(
        digest=dig,
        rounds=rounds,
        domain_mode=domain_mode,
        mask_m=256,
        allow_peel_back=allow_peel,
    )
    u0, violations, nullity = solve_alpha0_gf2_linear_skeleton(
        spec, bnd.var_mask[0], bnd.var_vals[0], num_samples=8
    )
    assert violations == 0
    assert nullity > 0
    assert u0.shape == (8, spec.n_var_bits)
    # Distinct samples in nullspace
    assert float((u0[0] != u0[1]).float().sum().item()) > 0.0

    eng = GlobalLinearRRREngine(rounds=rounds).to(dev)
    rep0 = eng.init_replicas_from_vars(u0)
    ro = eng.readout_shadow(rep0, bnd.repeat_for_candidates(8), alpha=0.0)
    assert ro["inst_E_hom"] == pytest.approx(0.0, abs=1e-6)
    assert ro["lin_viol_min"] == pytest.approx(0.0, abs=1e-6)

    # HomotopySchedule2D V6A-v2.1: 2D interleaved mask_m stepping + direct e_ref bifurcation & half-step rollback
    sched = HomotopySchedule2D(
        start_alpha=0.0,
        start_mask_m=64,
        target_mask_m=256,
        step_alpha=0.05,
        min_step_alpha=0.01,
        eps_advance=0.018,
        spike_factor=1.20,
        spike_abs_margin=0.003,
        max_ticks_per_alpha=12,
    )
    adv1, bif1, bnd1 = sched.update(0.0)
    assert adv1 is True and bif1 is False and bnd1 is False and sched.alpha == pytest.approx(0.05)
    # Next advance under < eps_advance (from pre_jump_alpha=0.05 > 0)
    adv2, bif2, _ = sched.update(0.010)
    assert adv2 is True and bif2 is False and sched.alpha == pytest.approx(0.10)
    # Direct previous-plateau spike check: e_ref=0.010 -> 0.015 > 0.010*1.20 (=0.012) and +0.005 > 0.003
    adv3, bif3, _ = sched.update(0.015)
    assert adv3 is False and bif3 is True and sched.bifurc_count == 1
    assert sched.alpha_c == pytest.approx(0.10)
    assert sched.step_alpha == pytest.approx(0.025)
    assert sched.alpha == pytest.approx(0.075)
    # Stagnation bifurcation keeps full alpha=1.0 while counting bifurcation
    sched.alpha = 1.0
    sched.record_stagnation_bifurcation(0.040)
    assert sched.bifurc_count == 2 and sched.alpha == pytest.approx(1.0)
    # Verify interleaved 2D stepping visits intermediate mask_m levels and reaches (1.0, 256) cleanly
    for _ in range(48):
        sched.update(0.005)
    assert sched.mask_m == 256 and sched.alpha == pytest.approx(1.0)
    assert len(sched.mask_history) >= 5 and sched.mask_history[0] == 64 and sched.mask_history[-1] == 256


# ==============================================================================
# Test 2: Layer 2 GL-RRR Global BP + Alpha Fixed-Point + 2-Adic Exactness
# ==============================================================================


def test_v6a_gl_rrr_global_bp_and_alpha_exact_fixed_point() -> None:
    enforce_v6a_fp32()
    dev = _device()
    gen = torch.Generator(device=dev).manual_seed(7201)

    for rounds, mode, peel in [(4, "free_msg", False), (5, "free_msg", False), (18, "int_range", True)]:
        w16 = torch.randint(0, 1 << 32, (2, 16), dtype=torch.int64, device=dev, generator=gen)
        if mode == "int_range":
            w16[:, 1] = 0x80000000
            w16[:, 2:15] = 0
            w16[:, 15] = 32
        elif rounds < 16:
            w16[:, rounds:] = 0
        dig = sha256_digest(w16, rounds=rounds)
        bnd = build_v6a_boundary(dig, rounds=rounds, domain_mode=mode, mask_m=256, allow_peel_back=peel)
        eng = GlobalLinearRRREngine(rounds=rounds, bp_iters=8, osd_every=8, osd_top_k=4).to(dev)
        x_star = embed_ground_truth_vars(w16, rounds=rounds)
        rep_star = eng.init_replicas_from_vars(x_star)

        for al in (0.0, 0.35, 0.70, 1.0):
            nxt, diag = eng.step(rep_star, bnd.var_mask, bnd.var_vals, alpha=al, beta=0.95, run_osd=True)
            assert float(diag["gl_gap"].max().item()) < 1e-6
            assert torch.allclose(nxt, rep_star, atol=1e-5)
            ro = eng.readout_shadow(nxt, bnd, alpha=al)
            assert ro["inst_E_hom"] < 1e-6
            assert ro["inst_E32_true"] < 1e-6
            assert ro["exact_hits"] == 2

    # Beta lock verification
    assert GlobalLinearRRREngine.clamp_beta(0.05) == pytest.approx(0.65)
    assert GlobalLinearRRREngine.clamp_beta(2.50) == pytest.approx(1.15)
    b_t = GlobalLinearRRREngine.clamp_beta(torch.tensor([0.1, 0.9, 1.9], dtype=torch.float32, device=dev))
    assert float(b_t.min().item()) >= 0.65 - 1e-6 and float(b_t.max().item()) <= 1.15 + 1e-6

    # Verify global multi-hop BP+OSD restores zero GF(2) parity syndrome from mild spin perturbation
    eng18 = GlobalLinearRRREngine(rounds=18, bp_iters=8, osd_top_k=4).to(dev)
    w_one = torch.tensor([[1234567, 0x80000000] + [0] * 13 + [32]], dtype=torch.int64, device=dev)
    d_one = sha256_digest(w_one, rounds=18)
    bnd18 = build_v6a_boundary(d_one, rounds=18, domain_mode="int_range", mask_m=256, allow_peel_back=True)
    x_s18 = embed_ground_truth_vars(w_one, rounds=18)
    noise = 0.45 * torch.randn(x_s18.shape, dtype=torch.float32, device=dev, generator=gen)
    r_pert = torch.where(bnd18.var_mask > 0.5, bnd18.var_vals, x_s18 + noise)
    raw_bits = (r_pert < 0.0).to(torch.int64)
    raw_syn = int(((raw_bits[:, eng18.chk_vars] * eng18.chk_valid_i64.unsqueeze(0)).sum(dim=-1) & 1).sum().item())
    assert raw_syn > 0, "Expected nonzero parity syndrome on noisy perturbation"
    y_lin, valid_lin = eng18.project_linear_skeleton(r_pert, bnd18.var_mask, bnd18.var_vals, run_osd=True)
    assert bool(valid_lin.all().item()), "BP + OSD-0 failed to project perturbed state onto F_2 skeleton!"
    assert float((y_lin != 0).float().mean().item()) == pytest.approx(1.0)

    # Exact energy equivalence with DualBasisEngine(18) within < 1e-5
    dual = DualBasisEngine(rounds=18).to(dev)
    a_r = torch.where(torch.rand((4, 22, 32), device=dev, generator=gen) < 0.5, -1.0, 1.0)
    e_r = torch.where(torch.rand((4, 22, 32), device=dev, generator=gen) < 0.5, -1.0, 1.0)
    w_r = torch.where(torch.rand((4, 18, 32), device=dev, generator=gen) < 0.5, -1.0, 1.0)
    _, _, stats_dual = dual.compute_energy_and_features(a_r, e_r, w_r)
    v6_m1, v6_m32 = compute_true_2adic_energy(
        a_r, e_r, w_r, eng18.P_T_fp64, eng18.theta_K_fp64, rounds=18, include_tail_e=False
    )
    assert torch.allclose(v6_m1, stats_dual["m1_energy"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(v6_m32, stats_dual["m32_energy"], atol=1e-5, rtol=1e-5)


# ==============================================================================
# Test 3: Layer 3 Bottleneck Entangled MPS (MGS2 + S_ent Floor + Speed < 50ms)
# ==============================================================================


def test_v6a_bottleneck_mps_entanglement_and_mgs2_speed() -> None:
    enforce_v6a_fp32()
    dev = _device()
    spec = build_v6a_circuit_spec(18)
    w_one = torch.tensor([[42, 0x80000000] + [0] * 13 + [32]], dtype=torch.int64, device=dev)
    dig = sha256_digest(w_one, rounds=18)
    bnd = build_v6a_boundary(dig, rounds=18, domain_mode="int_range", mask_m=256, allow_peel_back=True)

    var_idx, loop_pairs = select_bottleneck_var_indices(spec, bnd.var_mask[0], max_sites=64)
    assert 32 <= len(var_idx) <= 128
    lp_pairs = [(a, b) for a, b, _ in loop_pairs]

    mps = BottleneckMPSBornMachine(num_sites=len(var_idx), bond_dim=8, min_entropy=0.55, seed=7301).to(dev)
    ghz = mps.make_ghz_cores(len(var_idx), 8, dev)
    s_ghz = float(mps.compute_entanglement_entropy(ghz).item())
    assert abs(s_ghz - math.log(2.0)) < 1e-3 and s_ghz > 0.5

    ghz_samples, _ = mps.conditional_sample(num_samples=16, cores=ghz)
    assert bool(((ghz_samples == ghz_samples[:, :1]).all()).item()), "GHZ samples must be maximally correlated!"

    gen = torch.Generator(device=dev).manual_seed(7302)
    pool = torch.where(torch.rand((48, spec.n_var_bits), device=dev, generator=gen) < 0.5, -1.0, 1.0)
    energies = torch.linspace(1.0, 20.0, 48, device=dev)
    elite = select_diverse_elite_spins(pool, energies, var_idx, max_keep=24)

    # Warmup then measure real speed & gradient health
    _ = mps.update_on_elite_pool(elite, num_iters=1, loop_pairs=lp_pairs)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    stats = mps.update_on_elite_pool(elite, num_iters=1, loop_pairs=lp_pairs)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    assert stats["S_ent"] > 0.50
    assert math.isfinite(stats["nll"])
    assert torch.isfinite(mps.cores_param).all()
    if dev.type == "cuda":
        assert dt_ms < 50.0, f"Bottleneck MPS MGS2 update exceeded 50 ms budget: {dt_ms:.2f} ms"

    sampled, logp = mps.conditional_sample(num_samples=16, generator=gen)
    assert sampled.shape == (16, len(var_idx))
    assert bool(((sampled == 1.0) | (sampled == -1.0)).all().item())
    assert torch.isfinite(logp).all()


# ==============================================================================
# Test 4: Layer 4 Burer-Monteiro S^{N-1} Cayley Escape
# ==============================================================================


def test_v6a_bm_escape_unit_norm_and_clamp_invariant() -> None:
    enforce_v6a_fp32()
    dev = _device()
    gen = torch.Generator(device=dev).manual_seed(7401)
    bm = BurerMonteiroEscapeLayer(n_var_bits=128, manifold_dim=16, window_size=5, stag_patience=3, cayley_t=0.42).to(dev)

    v = torch.randn((8, 16), dtype=torch.float32, device=dev, generator=gen)
    t_dir = torch.randn((8, 16), dtype=torch.float32, device=dev, generator=gen)
    rot = bm.cayley_rotate_s_n(v, t_dir, 0.42)
    norms = torch.linalg.vector_norm(rot, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    clamp_m = torch.zeros((8, 128), dtype=torch.float32, device=dev)
    clamp_m[:, :32] = 1.0
    clamp_v = torch.where(torch.rand((8, 128), device=dev, generator=gen) < 0.5, -1.0, 1.0) * clamp_m
    x = torch.where(clamp_m > 0.5, clamp_v, torch.randn((8, 128), device=dev, generator=gen) * 0.5)
    gaps = torch.full((8,), 0.25, dtype=torch.float32, device=dev)

    kicked = None
    k_idx = torch.empty((0,), dtype=torch.int64, device=dev)
    for _ in range(5):
        kicked, k_idx, _ = bm.record_and_maybe_escape(x, gaps, clamp_m, clamp_v, generator=gen)
        if kicked is not None:
            break
    assert kicked is not None and k_idx.numel() == 8
    # Strictly invariant on clamped bits
    clamp_err = ((kicked[:, :32] - clamp_v[:, :32]).abs()).max().item()
    assert clamp_err == pytest.approx(0.0, abs=1e-7)
    # Free coordinates genuinely perturbed orthogonally
    free_diff = (kicked[:, 32:] - x[:, 32:]).abs().mean().item()
    assert free_diff > 1e-3


# ==============================================================================
# Test 5: Layer 5 Zero-Brake Navigator (Zero BCE, VSIDS Bounds, Jump Safety)
# ==============================================================================


def test_v6a_navigator_zero_bce_vsids_and_beta_lock() -> None:
    enforce_v6a_fp32()
    v6a_dir = Path(__file__).resolve().parents[1] / "v6a"
    for py_file in v6a_dir.glob("*.py"):
        src_text = py_file.read_text(encoding="utf-8").lower()
        assert "binary_cross_entropy" not in src_text
        assert "bce_with_logits" not in src_text

    dev = _device()
    gen = torch.Generator(device=dev).manual_seed(7501)
    nav = ZeroBrakeNavigator(n_var_words=16, hidden=32, lr=1e-3).to(dev)
    clamp_m = torch.zeros((4, 512), dtype=torch.float32, device=dev)
    clamp_m[:, :64] = 1.0

    gap_prof = torch.rand((4, 512), dtype=torch.float32, device=dev, generator=gen) * 1.5
    nav.record_gap(gap_prof)
    w_vsids = nav.get_vsids_weights(clamp_m)
    assert float(w_vsids.min().item()) >= 0.5 - 1e-6
    assert float(w_vsids.max().item()) <= 4.0 + 1e-6
    free_mean = float(w_vsids[:, 64:].mean().item())
    assert 0.90 <= free_mean <= 1.15

    x0 = torch.randn((4, 512), dtype=torch.float32, device=dev, generator=gen) * (1.0 - clamp_m)
    dx = nav.predict_tangent_jump(x0, clamp_m, alpha=0.2, d_alpha=0.1, mask_frac=1.0)
    assert float((dx[:, :64]).abs().max().item()) == pytest.approx(0.0, abs=1e-7)

    # Discarding on bifurcation must NOT train predictor
    nav.discard_pending_jump()
    assert nav.predictor_updates == 0

    # Converged advance trains predictor via self-supervised MSE only
    _ = nav.predict_tangent_jump(x0, clamp_m, alpha=0.2, d_alpha=0.1, mask_frac=1.0)
    x_after = x0 + 0.15 * torch.randn_like(x0) * (1.0 - clamp_m)
    mse_val = nav.train_on_successful_jump(x_after)
    assert nav.predictor_updates == 1
    assert math.isfinite(mse_val) and mse_val >= 0.0


# ==============================================================================
# Test 6: Hardcore Local Preimage Acceptance (R=4 Exact No-Cheat + R=5 Pipeline)
# ==============================================================================


def test_v6a_live_r4_exact_preimage_no_cheat_and_r5_pipeline() -> None:
    enforce_v6a_fp32()
    dev = _device()
    gen = torch.Generator(device=dev).manual_seed(6404)

    # R=4 free_msg, NO peel-back cheat -> must discover 100% genuine SHA-256 preimage via 2D mask_m 64->256
    w4 = torch.randint(0, 1 << 32, (1, 16), dtype=torch.int64, device=dev, generator=gen)
    w4[:, 4:] = 0
    dig4 = sha256_digest(w4, rounds=4)
    res4 = solve_single_digest_v6a(
        digest=dig4,
        rounds=4,
        domain_mode="free_msg",
        allow_peel_back=False,
        num_trajs=96,
        max_ticks=250,
        max_seconds=45.0,
        start_mask_m=64,
        step_alpha=0.10,
        min_step_alpha=0.025,
        max_ticks_per_alpha=4,
        device=dev,
        seed=6404,
    )
    assert res4["solved"] is True and res4["exact_hits"] > 0, f"R=4 failed to solve: {res4}"
    assert len(res4["mask_history"]) >= 2 and res4["mask_history"][0] == 64
    hw4 = torch.tensor([res4["hit_w16"]], dtype=torch.int64, device=dev)
    assert bool((sha256_digest(hw4, rounds=4) == dig4).all().item())

    # R=5 free_msg, NO peel-back cheat -> must ALSO discover 100% genuine SHA-256 preimage with Endgame Polish!
    w5 = torch.randint(0, 1 << 32, (1, 16), dtype=torch.int64, device=dev, generator=gen)
    w5[:, 5:] = 0
    dig5 = sha256_digest(w5, rounds=5)
    res5 = solve_single_digest_v6a(
        digest=dig5,
        rounds=5,
        domain_mode="free_msg",
        allow_peel_back=False,
        num_trajs=128,
        max_ticks=350,
        max_seconds=60.0,
        start_mask_m=64,
        step_alpha=0.10,
        min_step_alpha=0.025,
        max_ticks_per_alpha=4,
        device=dev,
        seed=6405,
    )
    assert res5["solved"] is True and res5["exact_hits"] > 0, f"R=5 failed to solve: {res5}"
    assert len(res5["mask_history"]) >= 2 and res5["mask_history"][0] == 64
    hw5 = torch.tensor([res5["hit_w16"]], dtype=torch.int64, device=dev)
    assert bool((sha256_digest(hw5, rounds=5) == dig5).all().item())


# ==============================================================================
# Test 7: Strict FP32 Redline, R=18 Lossless Resume, Binomial Z-Scores & CSV
# ==============================================================================


def test_v6a_r18_strict_fp32_lossless_resume_and_dashboard_csv(tmp_path: Path) -> None:
    enforce_v6a_fp32()
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False

    dev = _device()
    cfg = {
        "rounds": 18,
        "domain_mode": "int_range",
        "num_bits": 32,
        "msg_bytes_len": 4,
        "allow_peel_back": True,
        "num_digests": 2,
        "trajs_per_digest": 8,
        "inner_steps_per_tick": 2,
        "start_mask_m": 64,
        "target_mask_m": 256,
        "steps": 3,
        "log_every": 1,
        "eval_every": 3,
        "checkpoint_every": 3,
        "mps_every": 2,
        "sched_every": 1,
        "osd_every": 2,
        "osd_top_k": 4,
        "seed": 6618,
    }
    eng1 = OuroborosV6AEngine(config=cfg, output_dir=tmp_path, device=dev)
    assert_strict_fp32(eng1.gl_rrr)
    assert_strict_fp32(eng1.mps)
    assert_strict_fp32(eng1.navigator)
    eng1.train(max_steps=3)

    ckpt_path = tmp_path / "latest.pt"
    assert ckpt_path.exists()
    state_rep_before = eng1.state_rep.detach().clone()
    alpha_before = eng1.sched.alpha

    cfg2 = dict(cfg)
    cfg2["steps"] = 5
    eng2 = OuroborosV6AEngine(config=cfg2, output_dir=tmp_path, device=dev)
    eng2.load_checkpoint(ckpt_path)
    assert eng2.step == 3
    assert eng2.sched.alpha == pytest.approx(alpha_before)
    assert torch.allclose(eng2.state_rep, state_rep_before, atol=1e-6)
    eng2.train(max_steps=5)
    assert eng2.step == 5

    rows = _read_metrics_csv(tmp_path / "metrics.csv")
    assert len(rows) >= 5
    last = rows[-1]
    assert last["step"] == 5
    for key in (
        "inst_E32",
        "inst_E_hom",
        "hist_min",
        "alpha",
        "mask_m",
        "S_ent",
        "gl_gap",
        "eval_n_digests",
        "eval_free_bits",
        "w0_zscore",
        "r12_zscore",
        "r13_zscore",
        "r16_zscore",
    ):
        assert last[key] is not None and math.isfinite(float(last[key]))
    assert float(last["inst_E32"]) >= float(last["hist_min"]) - 1e-6


# ==============================================================================
# Test 8: V6A-v2 Hard Assertions (Real 2D mask_m Evolution + Bifurcation MPS + No Partial-Mask False Hits)
# ==============================================================================


def test_v6a_v2_2d_aperture_no_fake_hits_and_bifurcation_mps_triple(tmp_path: Path) -> None:
    enforce_v6a_fp32()
    dev = _device()

    # 1. Verify exact_hits NEVER falsely counts a partial-aperture (mask_m=64) digest collision
    w_real = torch.tensor([[123456, 0x80000000] + [0] * 13 + [32]], dtype=torch.int64, device=dev)
    dig_real = sha256_digest(w_real, rounds=18)
    dig_upper_flipped = dig_real.clone()
    dig_upper_flipped[:, 0] ^= 0x80000000  # differs in MSB while lower 8 LSBs per word match
    bnd_64 = build_v6a_boundary(
        dig_upper_flipped, rounds=18, domain_mode="int_range", mask_m=64, allow_peel_back=False
    )
    eng18 = GlobalLinearRRREngine(rounds=18).to(dev)
    x_star_fake = embed_ground_truth_vars(w_real, rounds=18)
    rep_fake = eng18.init_replicas_from_vars(x_star_fake)
    ro_fake = eng18.readout_shadow(rep_fake, bnd_64, alpha=1.0)
    assert ro_fake["aperture_hits"] == 1
    assert ro_fake["exact_hits"] == 0, "exact_hits MUST require full 256-bit digest equality!"

    # 2. Live engine test proving real 2D mask_m stepping (64 -> ... -> >160) & genuine bifurcation -> MPS call
    cfg = {
        "rounds": 18,
        "domain_mode": "int_range",
        "num_bits": 32,
        "msg_bytes_len": 4,
        "allow_peel_back": True,
        "num_digests": 1,
        "trajs_per_digest": 16,
        "inner_steps_per_tick": 3,
        "start_mask_m": 64,
        "target_mask_m": 256,
        "step_alpha": 0.25,
        "min_step_alpha": 0.01,
        "max_ticks_per_alpha": 1,
        "eps_advance": 0.018,
        "spike_factor": 1.15,
        "spike_abs_margin": 0.002,
        "sched_every": 1,
        "mps_every": 8,
        "osd_every": 4,
        "osd_top_k": 4,
        "steps": 18,
        "log_every": 1,
        "eval_every": 18,
        "checkpoint_every": 18,
        "seed": 6688,
    }
    eng = OuroborosV6AEngine(config=cfg, output_dir=tmp_path, device=dev)
    eng.train(max_steps=18)
    rows = _read_metrics_csv(tmp_path / "metrics.csv")
    mask_vals = [int(r["mask_m"]) for r in rows if r.get("mask_m") is not None]
    assert mask_vals[0] == 64
    assert len(set(mask_vals)) >= 4, f"Expected real 2D mask_m ladder evolution, got {mask_vals}"
    assert mask_vals[-1] >= 192

    # 3. Deterministic end-to-end bifurcation test: hitting a glassy wall right after an alpha step
    #    MUST increment bifurc, record alpha_c, rollback step_alpha/alpha, trigger MPS split + BM kick + RRR burst,
    #    and keep clamped coordinates strictly invariant.
    tmp_bif = tmp_path / "bif"
    cfg_bif = dict(cfg)
    cfg_bif.update(
        {
            "start_alpha": 0.40,
            "start_mask_m": 128,
            "step_alpha": 0.05,
            "eps_advance": 0.50,  # guarantee first step relaxes & advances 0.40 -> 0.45
            "spike_factor": 1.25,
            "spike_abs_margin": 0.004,
            "steps": 4,
            "log_every": 1,
            "eval_every": 4,
            "checkpoint_every": 4,
        }
    )
    eng_b = OuroborosV6AEngine(config=cfg_bif, output_dir=tmp_bif, device=dev)
    eng_b._init_population_alpha0()
    snap1 = eng_b.train_one_tick(sync_metrics=True)
    assert snap1 is not None and eng_b.sched.alpha == pytest.approx(0.45)

    # Corrupt free coordinates (respecting clamps) to simulate severe nonlinear wall spike after alpha jump
    assert eng_b.bnd_rep is not None and eng_b.state_rep is not None
    rand_free = torch.randn_like(eng_b.state_rep[:, : eng_b.gl_rrr.n_lin_rep]) * 1.5
    cm = eng_b.bnd_rep.var_mask > 0.5
    v_spiked = torch.where(cm, eng_b.bnd_rep.var_vals, rand_free)
    eng_b.state_rep = eng_b.gl_rrr.init_replicas_from_vars(v_spiked)

    snap2 = eng_b.train_one_tick(sync_metrics=True)
    assert snap2 is not None
    assert eng_b.sched.bifurc_count > 0
    assert eng_b.sched.alpha_c is not None
    assert eng_b.bifurc_mps_calls > 0
    assert torch.isfinite(eng_b.state_rep).all()
    ro_after = eng_b.gl_rrr.readout_shadow(eng_b.state_rep, eng_b.bnd_rep, alpha=eng_b.sched.alpha)
    clamp_err = float(((ro_after["v_disc"] - eng_b.bnd_rep.var_vals).abs() * eng_b.bnd_rep.var_mask).max().item())
    assert clamp_err == pytest.approx(0.0, abs=1e-7)


# ==============================================================================
# Test 9: V6A-v2.1 Dynamic MPS S_ent, 2^K Zoom Polish & Zero Cross-Round VRAM Leak
# ==============================================================================


def test_v6a_v21_dynamic_mps_zoom_polish_and_vram_cleanup(tmp_path: Path) -> None:
    enforce_v6a_fp32()
    dev = _device()

    # 1. Dynamic Bottleneck MPS + real S_ent evolution + mps_calls > 0 inside a single round
    gen = torch.Generator(device=dev).manual_seed(6901)
    w6 = _sample_free_msg_w16(1, 6, dev, gen)
    dig6 = sha256_digest(w6, rounds=6)
    s_trace = []

    def _track_sent(_tick: int, info: dict, _wall: float) -> None:
        s_trace.append(float(info["S_ent"]))

    res = solve_single_digest_v6a(
        digest=dig6,
        rounds=6,
        domain_mode="free_msg",
        allow_peel_back=False,
        num_trajs=64,
        max_ticks=36,
        max_seconds=30.0,
        start_mask_m=64,
        step_alpha=0.05,
        max_ticks_per_alpha=6,
        mps_every=4,
        stag_mps_cycles_trigger=2,
        inner_rrr_per_tick=3,
        device=dev,
        seed=6902,
        tick_callback=_track_sent,
        log_interval=1,
    )
    assert res["bifurc"] > 0, f"Expected real bifurcations > 0, got {res['bifurc']}"
    assert res["bifurc_mps_calls"] > 0, f"Expected mps_calls > 0, got {res['bifurc_mps_calls']}"
    assert len(s_trace) >= 8 and (max(s_trace) - min(s_trace)) > 1e-4, (
        f"Expected S_ent to evolve dynamically instead of flatline: min={min(s_trace):.5f}, max={max(s_trace):.5f}"
    )

    # 2. GPU-batched 2^K zoom Endgame Polish solves a 6-bit flipped candidate unreachable by 1/2-bit flips
    gl6 = GlobalLinearRRREngine(rounds=6).to(dev)
    bnd6 = build_v6a_boundary(dig6, rounds=6, domain_mode="free_msg", mask_m=256, allow_peel_back=False)
    x_star6 = embed_ground_truth_vars(w6, rounds=6).repeat(2, 1)
    v_mod = x_star6.clone()
    w_offset = gl6.spec.w_start * 32
    uncertain_offsets = [0, 1, 5, 19, 33, 66, 99, 129, 145, 170]
    flip_offsets = [0, 1, 5, 33, 66, 99]  # 6 simultaneous flipped bits (>2!)
    for off in flip_offsets:
        v_mod[:, w_offset + off] = -v_mod[:, w_offset + off]
    for off in uncertain_offsets:
        v_mod[:, w_offset + off] = 0.05 * v_mod[:, w_offset + off]
    rep_mod = gl6.init_replicas_from_vars(v_mod)
    scores_mod = torch.tensor([0.01, 0.02], dtype=torch.float32, device=dev)
    solved_eg, hw_eg, cnt_eg, _, _ = gl6.endgame_polish(
        rep_mod,
        bnd6.repeat_for_candidates(2),
        v_mod,
        scores_mod,
        top_k=2,
        per_traj_rank=scores_mod,
        zoom_k=12,
        zoom_k_large=12,
        n_zoom_cands=2,
    )
    assert solved_eg is True and cnt_eg >= 1 and hw_eg is not None
    assert bool((sha256_digest(hw_eg.unsqueeze(0), rounds=6) == dig6).all().item())

    # 3. Cross-round VRAM cleanup & navigator sample-cap low peak check
    if dev.type == "cuda":
        del rep_mod, x_star6, v_mod, bnd6, gl6
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        alloc_before = torch.cuda.memory_allocated(dev)
        rep_fp = run_frontier_push(
            output_dir=tmp_path / "fp_vram",
            config={
                "num_trajs": 64,
                "max_ticks": 8,
                "max_sec_per_round": 20.0,
                "mps_every": 4,
                "stag_mps_cycles_trigger": 2,
                "inner_rrr_per_tick": 2,
                "log_every": 4,
                "seed": 6910,
            },
            rounds_list=(6, 7),
            device=dev,
        )
        torch.cuda.synchronize(dev)
        alloc_after = torch.cuda.memory_allocated(dev)
        assert (alloc_after - alloc_before) < 150 * 1024 * 1024, (
            f"Cross-round CUDA memory leak detected: {(alloc_after - alloc_before) / 1e6:.1f} MB"
        )
        for sw in rep_fp["sweep"]:
            assert sw["round_peak_vram_gb"] < 1.5, f"Unexpectedly high peak VRAM: {sw}"


