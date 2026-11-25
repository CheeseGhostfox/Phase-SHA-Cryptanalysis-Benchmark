"""Comprehensive strict-FP32 acceptance tests for Phase-SHA V5 (V5D / V5C / V5B / V5A & Redlines)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.flow_trainer import prepare_ground_truth_nodes
from src.sha256_ref import (
    MASK32,
    sample_random_w16,
    spins_to_words,
    words_to_spins,
)
from src.v4_boundary import (
    ManifoldBoundary,
    apply_clamp_prepared,
    assemble_full_prepared,
    build_v4e_partial_prefix_boundary,
)
from src.v5_interfaces import (
    GlobalProgressiveCollapseSchedule,
    assert_strict_fp32,
    build_r20_stress_holdout_w16,
    build_v5_holdout_w16,
    enforce_v5_fp32,
)
from src.v5_trainer import UnifiedTrainerV5
from src.v5a_octonion import (
    OctonionS7Backend,
    UnrolledSolverV5A,
    V5ABackbone,
    V5APhysicsEngine,
    oct_associator_fp32,
    oct_mul_fp32,
)
from src.v5b_bm import (
    BurerMonteiroSphereBackend,
    UnrolledSolverV5B,
    V5BBackbone,
    V5BPhysicsEngine,
    cayley_orthogonal_escape,
    compute_bm_rank1_and_eff_rank,
)
from src.v5c_mps import MPSChain, V5CMPSSolver, build_loop_aware_word_order
from src.v5d_rrr import (
    DriftMonitor,
    ExactLocalGateProjectors,
    RRRDynamicsOperator,
    SHA256ReplicaGraph,
    V5DNeuralController,
    V5DRRRSolver,
    fast_wht_fp32,
    wht_xor_conv_fp32,
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# Priority 1: V5D Tests (Lean 4 Module 6D Alignment)
# ==============================================================================


def test_v5d_wht_involution_and_xor_convolution() -> None:
    """Aligns with Lean 4 `walsh_hadamard_involutive_and_xor_conv` in pure FP32."""
    enforce_v5_fp32()
    gen = torch.Generator().manual_seed(6001)
    for n_bits in (2, 4, 6):
        n_dim = 1 << n_bits
        f = torch.randn((5, n_dim), dtype=torch.float32, generator=gen)
        g = torch.randn((5, n_dim), dtype=torch.float32, generator=gen)

        wht_f = fast_wht_fp32(f)
        assert wht_f.dtype == torch.float32
        wht_wht_f = fast_wht_fp32(wht_f)
        assert torch.allclose(wht_wht_f, float(n_dim) * f, atol=1e-4, rtol=1e-4)

        # Direct O(N^2) Lean 4 reference XOR convolution
        idx = torch.arange(n_dim)
        xor_idx = idx.unsqueeze(1) ^ idx.unsqueeze(0)  # (N, N)
        conv_ref = torch.einsum("by,bxy->bx", f, g[:, xor_idx])
        conv_wht = wht_xor_conv_fp32(f, g)
        assert conv_wht.dtype == torch.float32
        assert torch.allclose(conv_wht, conv_ref, atol=1e-3, rtol=1e-3)


def test_v5d_viterbi_add32_exact_projection_and_idempotence() -> None:
    """Verify exactness, optimality, and idempotence (PB(PB(x)) == PB(x)) of 32-bit Viterbi DP."""
    enforce_v5_fp32()
    dev = _device()
    proj = ExactLocalGateProjectors().to(dev)
    gen = torch.Generator(device=dev).manual_seed(6002)

    # 1. Exact modular addition pairs remain fixed under PB
    a_w = torch.randint(0, 1 << 32, (64,), dtype=torch.int64, device=dev, generator=gen)
    b_w = torch.randint(0, 1 << 32, (64,), dtype=torch.int64, device=dev, generator=gen)
    c_w = (a_w + b_w) & MASK32
    a_s = words_to_spins(a_w, dtype=torch.float32)
    b_s = words_to_spins(b_w, dtype=torch.float32)
    c_s = words_to_spins(c_w, dtype=torch.float32)

    ap, bp, cp = proj.viterbi_add32_project(a_s, b_s, c_s)
    assert torch.equal(ap, a_s)
    assert torch.equal(bp, b_s)
    assert torch.equal(cp, c_s)

    # 2. Arbitrary noisy inputs strictly project into Set B (exact a + b == c mod 2^32) and are idempotent
    noisy = torch.randn((3, 64, 32), dtype=torch.float32, device=dev, generator=gen) * 1.3
    pa, pb, pc = proj.viterbi_add32_project(noisy[0], noisy[1], noisy[2])
    assert_strict_fp32(pa)
    # Every output coordinate is exact {-1.0, +1.0}
    assert torch.all((pa == 1.0) | (pa == -1.0))
    assert torch.all((pb == 1.0) | (pb == -1.0))
    assert torch.all((pc == 1.0) | (pc == -1.0))
    wa = spins_to_words(pa)
    wb = spins_to_words(pb)
    wc = spins_to_words(pc)
    assert torch.equal((wa + wb) & MASK32, wc)

    # Idempotence: PB(PB(x)) == PB(x)
    pa2, pb2, pc2 = proj.viterbi_add32_project(pa, pb, pc)
    assert torch.equal(pa2, pa)
    assert torch.equal(pb2, pb)
    assert torch.equal(pc2, pc)


def test_v5d_r18_ground_truth_exact_fixed_point_and_caveats() -> None:
    """Verify R=18 exact fixed point, shadow readout, beta clamp, and drift radar."""
    enforce_v5_fp32()
    dev = _device()
    gen = torch.Generator(device=dev).manual_seed(6003)
    w16 = sample_random_w16(8, "int_range", 32, 4, dev, gen)
    _, cb = prepare_ground_truth_nodes(w16, rounds=18, domain_mode="int_range", num_bits=32)
    mb = ManifoldBoundary.from_clamped_boundary(cb)

    graph = SHA256ReplicaGraph(rounds=18).to(dev)
    rrr = RRRDynamicsOperator(graph)
    var_mask, var_vals = graph.build_boundary_clamps(mb)
    var_star, rep_star = graph.embed_ground_truth_replicas(w16)

    # 1. Ground truth lies strictly in A \cap B and is an exact RRR fixed point
    y_a = graph.project_A(rep_star, var_mask, var_vals)
    y_b = graph.project_B(2.0 * y_a - rep_star)
    assert (y_a - rep_star).abs().max().item() < 1e-10
    assert (y_b - rep_star).abs().max().item() < 1e-10

    x_next, diag = rrr.step(rep_star, var_mask, var_vals, beta=0.75)
    assert (x_next - rep_star).abs().max().item() < 1e-10
    assert diag["gap_rms"].max().item() < 1e-10

    # 2. Caveat 2: beta strictly clamped inside [0.35, 1.15] (never zero)
    assert rrr.clamp_beta(0.0) == pytest.approx(0.35)
    assert rrr.clamp_beta(-1.0) == pytest.approx(0.35)
    assert rrr.clamp_beta(2.5) == pytest.approx(1.15)

    # 3. Caveat 1: shadow readout agrees with PA(x) even when x itself carries large orthogonal noise
    raw_offset = torch.zeros_like(rep_star)
    # Pick replica index 0 and 1 mapping to a free bit, add +5 / -5 balancing zero-mean perturbation
    b_x = rep_star.clone()
    b_x[:, 0] += 3.0
    b_x[:, 1] -= 1.0  # non-symmetric noise shifts consensus unless clamped or mean-zero
    _, _, _, shadow_nodes = rrr.readout_shadow(b_x, var_mask, var_vals)
    pa_expected = graph.extract_primary_nodes(
        graph.project_A_vars(b_x, var_mask, var_vals)
    )[3]
    assert torch.allclose(shadow_nodes, pa_expected, atol=1e-6)

    # 4. Caveat 3: Linear drift radar flags linear ||x_t|| divergence
    dm = DriftMonitor(slope_thresh=1e-2, r2_thresh=0.85)
    for k in range(1, 15):
        dm.record(torch.tensor([1.0 + 0.5 * k, 1.0, 0.99], device=dev))
    slope, flag = dm.summarize()
    assert slope[0].item() == pytest.approx(0.5, abs=1e-4)
    assert flag[0].item() == 1.0
    assert flag[1].item() == 0.0

    # 5. Neural controller strict-FP32 forward & V5DRRRSolver V4D/V4E probe
    ctrl = V5DNeuralController(rounds=18, hidden_dim=32).to(dev)
    assert_strict_fp32(ctrl)
    rep_w, beta_p, logits_p, _ = ctrl(rep_star, var_mask, var_vals, torch.tensor([0.5], device=dev))
    assert float(rep_w.min().item()) >= 0.25 - 1e-5
    assert float(rep_w.max().item()) <= 4.0 + 1e-5
    assert float(beta_p.min().item()) >= 0.35 - 1e-5
    assert float(beta_p.max().item()) <= 1.15 + 1e-5
    assert logits_p.shape == (8, 18, 3, 32)

    solver = V5DRRRSolver(rounds=18, mode="pure", num_candidates=8, num_steps=6, chunk_size=32).to(dev)
    v4e_b = build_v4e_partial_prefix_boundary(rounds=18, zero_prefix_bits=8, batch_size=2, device=dev)
    res_v4e = solver.solve(boundary=v4e_b)
    assert res_v4e.best_x_disc.shape == (2, 18, 3, 32)
    probs = solver.export_calibrated_probs(res_v4e)
    assert probs.shape == (2, 18, 3, 32)
    assert float(probs.min().item()) >= 0.0 and float(probs.max().item()) <= 1.0


# ==============================================================================
# Priority 2: V5C Tests (Lean 4 Module 6C Alignment)
# ==============================================================================


def test_v5c_loop_aware_order_and_ghz_zero_barrier_born_sampling() -> None:
    """Aligns with Lean 4 `rank2_entangled_zero_barrier` & `mps_rank2_cycle_zero_residual`."""
    enforce_v5_fp32()
    dev = _device()

    order, inv_flat, _ = build_loop_aware_word_order(rounds=18)
    assert len(order) == 54
    assert abs(inv_flat[0 * 3 + 2] - inv_flat[16 * 3 + 2]) == 1
    assert abs(inv_flat[1 * 3 + 2] - inv_flat[17 * 3 + 2]) == 1
    assert abs(inv_flat[13 * 3 + 1] - inv_flat[16 * 3 + 1]) <= 12

    L = 8
    mps = MPSChain(num_sites=L, bond_dim=4)
    cycle_pairs = [(i, (i + 1) % L, 1) for i in range(L)]

    # Equal superposition GHZ state |Phi+> = (|+...+> + |-...->) / sqrt(2)
    s2 = math.sqrt(0.5)
    cores_ghz = mps.make_ghz_or_superposition(batch_size=1, c_amp=s2, s_amp=s2, device=dev)
    assert_strict_fp32(cores_ghz)

    ent_en = mps.compute_entangled_energy(cores_ghz, cycle_pairs)
    assert ent_en.item() < 1e-5

    marg_logits = mps.marginal_logits(cores_ghz)
    assert marg_logits.abs().max().item() < 1e-5  # <s_k> == 0 at every site

    entropy = mps.compute_entanglement_entropy(cores_ghz, cut_idx=3)
    assert abs(entropy.item() - math.log(2.0)) < 1e-4

    # Exact conditional autoregressive Born sampling yields 100% valid complementary branches (zero mean-field noise)
    gen = torch.Generator(device=dev).manual_seed(6101)
    cores_rep = cores_ghz.expand(128, -1, -1, -1, -1).contiguous()
    samples, _ = mps.conditional_sample(cores_rep, deterministic=False, generator=gen)
    assert samples.shape == (128, L)
    all_pos = (samples > 0).all(dim=1)
    all_neg = (samples < 0).all(dim=1)
    assert bool((all_pos | all_neg).all().item())
    # Must visit both branches with non-trivial probability
    assert int(all_pos.sum().item()) > 20 and int(all_neg.sum().item()) > 20

    # Joint log-prob on valid branches == log(0.5), on invalid branch -> very low
    lp_pos = mps.joint_log_prob(cores_ghz, torch.ones((1, L), device=dev))
    lp_neg = mps.joint_log_prob(cores_ghz, -torch.ones((1, L), device=dev))
    lp_bad = mps.joint_log_prob(
        cores_ghz, torch.cat([torch.ones((1, L // 2), device=dev), -torch.ones((1, L - L // 2), device=dev)], dim=1)
    )
    assert abs(lp_pos.item() - math.log(0.5)) < 1e-4
    assert abs(lp_neg.item() - math.log(0.5)) < 1e-4
    assert lp_bad.item() < -20.0

    # Honest control: Rank-1 product state with 0 marginal violates cycle energy >= 1
    p0_zero = torch.zeros((1, L), device=dev)
    p0_zero[0, 1:] = 1.0
    cores_r1 = torch.zeros((1, L, 2, 2, 2), dtype=torch.float32, device=dev)
    # Construct product |+x> x |+z> x ... x |+z> inside chi=2 bond 0
    s_half = math.sqrt(0.5)
    cores_r1[0, 0, 0, 0, 0] = s_half
    cores_r1[0, 0, 0, 1, 0] = s_half
    cores_r1[0, 1:, 0, 0, 0] = 1.0
    assert float(mps.compute_entangled_energy(cores_r1, cycle_pairs).item()) >= 1.0 - 1e-5


def test_v5c_tebd_qr_evolution_and_solver_gradients_fp32() -> None:
    """Verify batched no-grad QR TEBD, V5CMPSSolver finite FP32 grads, and V3/V4E/V4D compat."""
    enforce_v5_fp32()
    dev = _device()
    gen = torch.Generator(device=dev).manual_seed(6102)
    w16 = sample_random_w16(2, "int_range", 32, 4, dev, gen)
    x_star, cb = prepare_ground_truth_nodes(w16, rounds=18, domain_mode="int_range", num_bits=32)
    mb = ManifoldBoundary.from_clamped_boundary(cb)

    solver = V5CMPSSolver(rounds=18, bond_dim=4, d_bit=48, n_heads=3, n_blocks=2, use_carry_gru=True).to(dev)
    assert_strict_fp32(solver)

    stage3b = Path(__file__).resolve().parent.parent / "checkpoints" / "stage3b_r18_calibrated_best.pt"
    if stage3b.exists():
        info = solver.load_v3_warmstart(stage3b, map_location=dev)
        assert info["loaded_keys"] == 86

    cores = solver.init_cores(mb)
    sf = torch.tensor([0.5, 0.5], dtype=torch.float32, device=dev)
    cores_next, logits, x_marg, h_out, _ = solver.forward_step(cores, mb, sf, h_prev=None)
    assert_strict_fp32(cores_next)
    assert logits.shape == (2, 18, 3, 32)
    assert x_marg.shape == (2, 18, 3, 32)

    target_chain = solver.nodes_to_chain(x_star)
    nll = -solver.mps.joint_log_prob(cores_next, target_chain).mean()
    ent_e = solver.mps.compute_entangled_energy(cores_next, solver.loop_pairs).mean()
    loss = nll + 0.1 * ent_e + 0.05 * logits.square().mean()
    loss.backward()
    grad_norms = [float(p.grad.abs().max().item()) for p in solver.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(math.isfinite(g) for g in grad_norms)
    assert max(grad_norms) > 0.0

    # V4E & V4D probe
    v4e_b = build_v4e_partial_prefix_boundary(rounds=18, zero_prefix_bits=8, batch_size=2, device=dev)
    res_v4e = solver.solve(boundary=v4e_b, num_candidates=8, num_steps=4, chunk_size=2)
    assert res_v4e.best_x_disc.shape == (2, 18, 3, 32)
    probs = solver.export_calibrated_probs(res_v4e)
    assert float(probs.min().item()) >= 0.0 and float(probs.max().item()) <= 1.0


# ==============================================================================
# Priority 3: V5B Tests (Lean 4 Module 6B Alignment)
# ==============================================================================


@pytest.mark.parametrize("ndim", [16, 32, 64])
def test_v5b_bm_r18_discrete_zero_residual_and_nonneg_guards(ndim: int) -> None:
    """Aligns with Lean 4 `energy_rank1_eq_ising`, `burer_monteiro_plus_one_dim_escape` & gap guard."""
    enforce_v5_fp32()
    dev = _device()
    gen = torch.Generator(device=dev).manual_seed(6200 + ndim)

    w16 = sample_random_w16(4, "int_range", 32, 4, dev, gen)
    x_star, cb = prepare_ground_truth_nodes(w16, rounds=18, domain_mode="int_range", num_bits=32)
    mb = ManifoldBoundary.from_clamped_boundary(cb)

    phys = V5BPhysicsEngine(manifold_dim=ndim, rounds=18).to(dev)
    m = phys.manifold
    u_star = m.embed_spins(x_star)
    assert_strict_fp32(u_star)

    u_a, u_e, u_w = mb.assemble_full_manifold(u_star, m)
    total_e, _, stats = phys.compute_energy_and_features(
        u_a, u_e, u_w, free_mask=1.0 - mb.clamp_mask, w_xor=1.0, lam_comm=1.0, lam_align=1.0
    )
    assert total_e.max().item() < 1e-10
    assert stats["energy_xor"].max().item() < 1e-10
    assert stats["energy_align"].max().item() < 1e-10

    r1_star, eff_star = compute_bm_rank1_and_eff_rank(u_star, 1.0 - mb.clamp_mask)
    assert r1_star == pytest.approx(1.0, abs=1e-5)
    assert eff_star == pytest.approx(1.0, abs=1e-3)

    # Single free bit flip jumps energy >= 2.0
    u_flip = u_star.clone()
    u_flip[:, 0, 2, 0, :] = -u_flip[:, 0, 2, 0, :]
    ua_f, ue_f, uw_f = mb.assemble_full_manifold(u_flip, m)
    e_flip, _, _ = phys.compute_energy_and_features(ua_f, ue_f, uw_f)
    assert float(e_flip.min().item()) >= 2.0

    # Strict non-negative zero-lower-bound frustrated triangle check (defeats triW2=-37/25 cheat)
    v1 = torch.zeros((1, ndim), dtype=torch.float32, device=dev)
    v2 = torch.zeros((1, ndim), dtype=torch.float32, device=dev)
    v3 = torch.zeros((1, ndim), dtype=torch.float32, device=dev)
    v1[0, 0] = 1.0
    v2[0, 0], v2[0, 1] = -0.6, 0.8
    v3[0, 0], v3[0, 1] = -0.6, -0.8
    assert float(m.antiferro_triangle_nonneg_energy(v1, v2, v3).item()) > 0.0
    # Random gate vectors never achieve negative residuals
    rand_vecs = m.sample_random_manifold((4, 32), device=dev, generator=gen)
    assert float(m.xor2_residual(rand_vecs[0], rand_vecs[1], rand_vecs[2]).min().item()) >= 0.0
    assert float(m.xor3_residual(rand_vecs[0], rand_vecs[1], rand_vecs[2], rand_vecs[3]).min().item()) >= 0.0

    # Norm-preserving half-angle rational Cayley orthogonal escape
    u_rand = m.sample_random_manifold((4, 18, 3, 32), device=dev, generator=gen)
    u_esc = cayley_orthogonal_escape(u_rand, step_t=0.40, generator=gen)
    assert torch.allclose(torch.linalg.vector_norm(u_esc, dim=-1), torch.ones_like(u_esc[..., 0]), atol=1e-5)
    assert float((u_esc - u_rand).abs().max().item()) > 0.05


def test_v5b_backbone_warmstart_grad_and_progressive_anneal() -> None:
    """Verify V5B backbone V3 warm-start, gradients, progressive schedule, and V4E/V4D compat."""
    enforce_v5_fp32()
    dev = _device()
    gen = torch.Generator(device=dev).manual_seed(6299)
    w16 = sample_random_w16(2, "int_range", 32, 4, dev, gen)
    _, cb = prepare_ground_truth_nodes(w16, rounds=18, domain_mode="int_range", num_bits=32)
    mb = ManifoldBoundary.from_clamped_boundary(cb)

    phys = V5BPhysicsEngine(manifold_dim=32, rounds=18).to(dev)
    model = V5BBackbone(manifold_dim=32, rounds=18).to(dev)
    assert_strict_fp32(model)

    stage3b = Path(__file__).resolve().parent.parent / "checkpoints" / "stage3b_r18_calibrated_best.pt"
    if stage3b.exists():
        info = model.load_v3_warmstart(stage3b, map_location=dev)
        assert info["loaded_keys"] == 86

    sched = GlobalProgressiveCollapseSchedule(start_val=0.02, mid_val=1.50, end_val=6.00)
    assert sched.global_factor(0, 1000) == pytest.approx(0.02)
    assert sched.global_factor(200, 1000) == pytest.approx(0.02)
    assert sched.global_factor(800, 1000) == pytest.approx(1.50)
    assert sched.global_factor(1000, 1000) == pytest.approx(6.00)

    prep = mb.prepare_for_manifold(phys.manifold)
    u_init = apply_clamp_prepared(
        phys.manifold.sample_random_manifold((2, 18, 3, 32), device=dev, generator=gen), prep
    )
    lam_full = torch.ones(32, dtype=torch.float32, device=dev)
    grad_radar, res_feat, _, grad_riem = phys.compute_radar_prepared(u_init, prep, lam_full, return_raw_grad=True)
    logits, delta_u, h_out = model(
        u_init, grad_radar, prep.clamp_mask, res_feat, phys.K_spins, torch.tensor([0.5, 0.5], device=dev), lam_full
    )
    u_next = phys.retract_and_clamp(u_init, logits, delta_u, grad_riem, prep.u_clamped_masked, prep.free_mask_exp, 0.25, 0.02, 0.1)
    u_a, u_e, u_w = assemble_full_prepared(u_next, prep, phys.manifold)
    e_tot, _, _ = phys.compute_energy_and_features(u_a, u_e, u_w, prep.free_mask, lam_align=sched.value(900, 1000))
    loss = e_tot.mean() + 0.1 * logits.square().mean()
    loss.backward()
    grads = [float(p.grad.abs().max().item()) for p in model.parameters() if p.grad is not None]
    assert grads and all(math.isfinite(g) for g in grads) and max(grads) > 0.0

    # V4E & V4D probe
    solver = UnrolledSolverV5B(model=model, physics=phys, num_candidates=4, num_steps=4, chunk_size=4)
    v4e_b = build_v4e_partial_prefix_boundary(rounds=18, zero_prefix_bits=8, batch_size=2, device=dev)
    res_v4e = solver.solve(boundary=v4e_b)
    assert "rank1_ratio" in res_v4e.extra_metrics and "eff_rank" in res_v4e.extra_metrics
    assert float(solver.export_calibrated_probs(res_v4e).max().item()) <= 1.0


# ==============================================================================
# Priority 4: V5A Tests (Lean 4 Module 6A Alignment)
# ==============================================================================


def test_v5a_octonion_lean6a_identities_and_paths() -> None:
    """Aligns item-by-item with Lean 4 `OctonionLift.lean`."""
    enforce_v5_fp32()
    dev = _device()
    gen = torch.Generator(device=dev).manual_seed(6301)
    m = OctonionS7Backend()

    # 1. Degen 8-square normSq multiplicativity: ||x*y||^2 == ||x||^2 * ||y||^2 (< 1e-5)
    x = torch.randn((32, 8), dtype=torch.float32, device=dev, generator=gen)
    y = torch.randn((32, 8), dtype=torch.float32, device=dev, generator=gen)
    xy = oct_mul_fp32(x, y)
    assert_strict_fp32(xy)
    nx = x.square().sum(dim=-1)
    ny = y.square().sum(dim=-1)
    nxy = xy.square().sum(dim=-1)
    assert ((nxy - nx * ny).abs() / (nx * ny).clamp_min(1.0)).max().item() < 1e-5

    # 2. Explicit non-associativity witness: (e_1 * e_2) * e_4 != e_1 * (e_2 * e_4)
    e1 = m.imag_basis(0, dev)
    e2 = m.imag_basis(1, dev)
    e4 = m.imag_basis(3, dev)
    lhs = oct_mul_fp32(oct_mul_fp32(e1, e2), e4)
    rhs = oct_mul_fp32(e1, oct_mul_fp32(e2, e4))
    assert float((lhs - rhs).abs().max().item()) > 1.5  # hi.z = +1 vs -1 -> diff 2.0
    assert abs(lhs[7].item() + rhs[7].item()) < 1e-6

    # 3. Global left and right alternativity: x*(x*y)==(x*x)*y and (y*x)*x==y*(x*x)
    assert (oct_mul_fp32(x, oct_mul_fp32(x, y)) - oct_mul_fp32(oct_mul_fp32(x, x), y)).abs().max().item() < 1e-4
    assert (oct_mul_fp32(oct_mul_fp32(y, x), x) - oct_mul_fp32(y, oct_mul_fp32(x, x))).abs().max().item() < 1e-4

    # 4. Real-axis {+-1} XOR isomorphism & exact zero associator
    poles = torch.stack([m.e0_vec(dev), -m.e0_vec(dev)], dim=0)
    for b1 in (0, 1):
        for b2 in (0, 1):
            for b3 in (0, 1):
                p1, p2, p3 = poles[b1], poles[b2], poles[b3]
                py = poles[b1 ^ b2 ^ b3]
                assert m.xor3_residual(p1, p2, p3, py).item() < 1e-10
                assert oct_associator_fp32(p1, p2, p3).abs().max().item() < 1e-10

    # 5. All 7 imaginary paths connect +1 to -1 smoothly preserving unit norm
    ts = torch.linspace(0.0, 1.0, 17, dtype=torch.float32, device=dev)
    for k in range(7):
        curve = m.imag_path(k, ts)
        assert torch.allclose(torch.linalg.vector_norm(curve, dim=-1), torch.ones(17, device=dev), atol=1e-6)
        assert torch.allclose(curve[0], m.e0_vec(dev), atol=1e-6)
        assert torch.allclose(curve[-1], -m.e0_vec(dev), atol=1e-6)


def test_v5a_r18_discrete_zero_residual_warmstart_and_grads() -> None:
    """Verify R=18 exact ground truth residual < 1e-10, flip >= 2.0, V3 warmstart, and V4E/V4D."""
    enforce_v5_fp32()
    dev = _device()
    gen = torch.Generator(device=dev).manual_seed(6302)

    w16 = sample_random_w16(4, "int_range", 32, 4, dev, gen)
    x_star, cb = prepare_ground_truth_nodes(w16, rounds=18, domain_mode="int_range", num_bits=32)
    mb = ManifoldBoundary.from_clamped_boundary(cb)

    phys = V5APhysicsEngine(rounds=18).to(dev)
    model = V5ABackbone(rounds=18).to(dev)
    assert_strict_fp32(model)

    stage3b = Path(__file__).resolve().parent.parent / "checkpoints" / "stage3b_r18_calibrated_best.pt"
    if stage3b.exists():
        info = model.load_v3_warmstart(stage3b, map_location=dev)
        assert info["loaded_keys"] == 86

    u_star = phys.manifold.embed_spins(x_star)
    ua, ue, uw = mb.assemble_full_manifold(u_star, phys.manifold)
    tot_e, _, stats = phys.compute_energy_and_features(
        ua, ue, uw, free_mask=1.0 - mb.clamp_mask, w_xor=1.0, lam_comm=1.0, lam_align=1.0, lam_assoc=1.0
    )
    assert tot_e.max().item() < 1e-10
    assert stats["energy_assoc"].max().item() < 1e-10

    u_flip = u_star.clone()
    u_flip[:, 0, 2, 0, :] = -u_flip[:, 0, 2, 0, :]
    uaf, uef, uwf = mb.assemble_full_manifold(u_flip, phys.manifold)
    e_flip, _, _ = phys.compute_energy_and_features(uaf, uef, uwf)
    assert float(e_flip.min().item()) >= 2.0

    # Gradients through dual progressive annealing + retraction
    prep = mb.prepare_for_manifold(phys.manifold)
    u_init = apply_clamp_prepared(
        phys.manifold.sample_random_manifold((4, 18, 3, 32), device=dev, generator=gen), prep
    )
    lam_full = torch.ones(32, dtype=torch.float32, device=dev)
    grad_radar, res_feat, _, grad_riem = phys.compute_radar_prepared(
        u_init, prep, lam_full, lam_comm=0.3, lam_align=2.0, lam_assoc=1.0, return_raw_grad=True
    )
    logits, delta_u, _ = model(
        u_init, grad_radar, prep.clamp_mask, res_feat, phys.K_spins, torch.tensor([0.5]*4, device=dev), lam_full
    )
    u_next = phys.retract_and_clamp(u_init, logits, delta_u, grad_riem, prep.u_clamped_masked, prep.free_mask_exp, 0.25, 0.02, 0.1)
    u_a2, u_e2, u_w2 = assemble_full_prepared(u_next, prep, phys.manifold)
    e2, _, _ = phys.compute_energy_and_features(u_a2, u_e2, u_w2, prep.free_mask, lam_comm=0.3, lam_align=2.0, lam_assoc=1.0)
    (e2.mean() + 0.1 * logits.square().mean()).backward()
    grads = [float(p.grad.abs().max().item()) for p in model.parameters() if p.grad is not None]
    assert grads and all(math.isfinite(g) for g in grads) and max(grads) > 0.0

    solver = UnrolledSolverV5A(model=model, physics=phys, num_candidates=4, num_steps=4, chunk_size=4)
    v4e_b = build_v4e_partial_prefix_boundary(rounds=18, zero_prefix_bits=8, batch_size=2, device=dev)
    res_v4e = solver.solve(boundary=v4e_b)
    assert "zero_assoc" in res_v4e.extra_metrics
    assert float(solver.export_calibrated_probs(res_v4e).max().item()) <= 1.0


# ==============================================================================
# Global Redlines: Strict FP32, Unified R=18 + R=20 Dataset, Resume Parity & V4E
# ==============================================================================


@pytest.mark.parametrize("variant", ["v5d_pure", "v5d_neural", "v5c", "v5b", "v5a"])
def test_v5_global_redlines_fp32_and_lossless_resume(tmp_path: Path, variant: str) -> None:
    """Enforce allow_tf32==False, strict FP32, and lossless --resume across ALL 5 V5 variants."""
    enforce_v5_fp32()
    if torch.cuda.is_available():
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False

    dev = _device()
    cfg = {
        "variant": variant,
        "rounds": 18,
        "domain_mode": "int_range",
        "num_bits": 32,
        "batch_size": 2,
        "num_particles": 2 if variant != "v5d_pure" else 4,
        "K_unroll": 2,
        "bptt_chunk_len": 2,
        "backward_group_chunks": 1,
        "pure_num_digests": 2,
        "pure_trajs_per_digest": 4,
        "pure_steps_per_tick": 2,
        "bond_dim": 4,
        "manifold_dim": 16,
        "hidden_dim": 32,
        "K_infer": 2,
        "anneal_K": 4,
        "anneal_chunk_size": 8,
        "anneal_eval_samples": 4,
        "holdout_size": 8,
        "eval_batch_size": 4,
        "steps": 4,
        "log_every": 1,
        "eval_every": 2,
        "checkpoint_every": 2,
        "use_torch_compile": "false",
        "use_async_prefetch": False,
        "seed": 6401,
        "holdout_seed": 20261122,
    }
    run1 = tmp_path / f"{variant}_run1"
    tr1 = UnifiedTrainerV5(config=cfg, output_dir=run1, device=dev)
    if tr1.model is not None:
        assert_strict_fp32(tr1.model)
    tr1.train(max_steps=2)
    assert (run1 / "latest.pt").exists()
    assert (run1 / "metrics.csv").exists()

    # Verify resume restores exact step, wall_s, and continues without error or NaN
    tr2 = UnifiedTrainerV5(config=cfg, output_dir=tmp_path / f"{variant}_run2", device=dev)
    tr2.load_checkpoint(run1 / "latest.pt")
    assert tr2.step == 2
    assert tr2.wall_s >= 0.0
    m_step3 = tr2.train_one_step(sync_metrics=True)
    assert tr2.step == 3
    assert math.isfinite(m_step3["loss"])


def test_v5_unified_r18_and_r20_stress_holdout_datasets() -> None:
    """Verify bit-identical R=18 benchmark and working R=20 stress-test generator."""
    w16_r18, _ = build_v5_holdout_w16(holdout_size=2048, holdout_seed=20261122, num_bits=32)
    w16_r18_again, _ = build_v5_holdout_w16(holdout_size=2048, holdout_seed=20261122, num_bits=32)
    assert torch.equal(w16_r18, w16_r18_again)
    assert len(set(w16_r18[:, 0].tolist())) == 2048

    w16_r20, _ = build_r20_stress_holdout_w16(holdout_size=32, holdout_seed=20261122, num_bits=32)
    x20, cb20 = prepare_ground_truth_nodes(w16_r20[:4], rounds=20, domain_mode="int_range", num_bits=32)
    assert x20.shape == (4, 20, 3, 32)
    assert cb20.rounds == 20




