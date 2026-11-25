"""Comprehensive strict-FP32 tests for Phase-SHA V4 (V4A / V4B / V4C) & V4E interface."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from src.boundary_clamp import build_clamped_boundary
from src.dual_basis import DualBasisEngine
from src.flow_trainer import prepare_ground_truth_nodes
from src.sha256_ref import pack_int_messages_to_w16, sample_random_w16, sha256_digest, sha256_trace, words_to_spins
from src.v4_backbone import ManifoldBitGridBackbone
from src.v4_boundary import (
    ManifoldBoundary,
    build_arbitrary_mask_boundary,
    build_manifold_boundary,
    build_v4e_partial_prefix_boundary,
)
from src.v4_manifolds import (
    ManifoldPhysicsEngine,
    _spin_add32_flat,
    apply_manifold_closed_loop_schedule,
    enforce_fp32_precision,
    get_manifold,
    normalize_manifold,
)
from src.v4_solver import UnrolledSolverV4
from src.v4_trainer import (
    UnrolledTrainerV4,
    _bce_wta_loss_head,
    build_v4_init,
    compute_v4_step_loss,
    run_v4_chunked_bptt_step,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
STAGE3B_CKPT = ROOT_DIR / "checkpoints" / "stage3b_r18_calibrated_best.pt"
VARIANTS = ["v4a", "v4b", "v4c"]
_SHARED_ADD32_CACHE = {}
_SHARED_LOSS_HEAD_CACHE = {}


def _bloch_mat_c128(n: torch.Tensor) -> torch.Tensor:
    """Explicit 2x2 complex128 Bloch matrix X = nx*sx + ny*sy + nz*sz."""
    nx, ny, nz = n[..., 0].to(torch.float64), n[..., 1].to(torch.float64), n[..., 2].to(torch.float64)
    a = nz.to(torch.complex128)
    d = (-nz).to(torch.complex128)
    b = torch.complex(nx, -ny)
    c = torch.complex(nx, ny)
    row0 = torch.stack([a, b], dim=-1)
    row1 = torch.stack([c, d], dim=-1)
    return torch.stack([row0, row1], dim=-2)


@pytest.mark.parametrize("variant", VARIANTS)
def test_v4_manifold_algebra_and_lean5_paths(variant: str) -> None:
    enforce_fp32_precision()
    m = get_manifold(variant)
    gen = torch.Generator().manual_seed(123)

    # Tangent projection orthogonality & norm-preserving retraction
    u = m.sample_random_manifold((32,), device=torch.device("cpu"), generator=gen)
    assert u.dtype == torch.float32
    assert torch.allclose(torch.linalg.vector_norm(u, dim=-1), torch.ones(32), atol=1e-6)
    delta = torch.randn((32, m.dim), generator=gen)
    d_perp = m.tangent_project(u, delta)
    assert torch.abs((u * d_perp).sum(dim=-1)).max().item() < 1e-6
    u_next = m.retract(u, delta, step_size=0.35)
    assert u_next.dtype == torch.float32
    assert torch.allclose(torch.linalg.vector_norm(u_next, dim=-1), torch.ones(32), atol=1e-6)

    # Exact discrete XOR2 and XOR3 truth tables
    poles = torch.stack([m.e0_vec(torch.device("cpu")), -m.e0_vec(torch.device("cpu"))], dim=0)
    for b_a in (0, 1):
        for b_b in (0, 1):
            for b_c in (0, 1):
                ua = poles[b_a]
                ub = poles[b_b]
                uc = poles[b_c]
                uy2_t = poles[b_a ^ b_b]
                uy2_f = poles[1 - (b_a ^ b_b)]
                assert m.xor2_residual(ua, ub, uy2_t).item() < 1e-10
                assert abs(m.xor2_residual(ua, ub, uy2_f).item() - 4.0) < 1e-6

                uy3_t = poles[b_a ^ b_b ^ b_c]
                uy3_f = poles[1 - (b_a ^ b_b ^ b_c)]
                assert m.xor3_residual(ua, ub, uc, uy3_t).item() < 1e-10
                assert abs(m.xor3_residual(ua, ub, uc, uy3_f).item() - 4.0) < 1e-6

    # Variant-specific non-Abelian identities vs Lean 4 Module 5
    if variant == "v4a":
        assert torch.allclose(m.commutator_sq(u[:16], u[16:]), torch.zeros(16))
    elif variant == "v4b":
        a = m.sample_random_manifold((16,), device=torch.device("cpu"), generator=gen)
        b = m.sample_random_manifold((16,), device=torch.device("cpu"), generator=gen)
        c = m.sample_random_manifold((16,), device=torch.device("cpu"), generator=gen)
        y = m.sample_random_manifold((16,), device=torch.device("cpu"), generator=gen)
        Xa, Xb, Xc, Xy = _bloch_mat_c128(a), _bloch_mat_c128(b), _bloch_mat_c128(c), _bloch_mat_c128(y)
        sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128).unsqueeze(0)
        I2 = torch.eye(2, dtype=torch.complex128).unsqueeze(0)
        # X^dagger = X, X^2 = I2
        assert (Xa.mH - Xa).abs().max().item() < 1e-6
        assert ((Xa @ Xa) - I2).abs().max().item() < 1e-6
        # XOR3 == 0.5 ||Xa Xb Xc - Xy||_F^2
        ref_xor3 = 0.5 * ((Xa @ Xb @ Xc - Xy).abs().square().sum(dim=(-2, -1))).to(torch.float32)
        assert (m.xor3_residual(a, b, c, y) - ref_xor3).abs().max().item() < 1e-5
        # XOR2 == 0.5 ||Xa Xb Xy - sigma_z||_F^2
        ref_xor2 = 0.5 * ((Xa @ Xb @ Xy - sz).abs().square().sum(dim=(-2, -1))).to(torch.float32)
        assert (m.xor2_residual(a, b, y) - ref_xor2).abs().max().item() < 1e-5
        # 9-operand Eq S parity exactness
        ops9 = [m.sample_random_manifold((8,), device=torch.device("cpu"), generator=gen) for _ in range(9)]
        mat_prod = _bloch_mat_c128(ops9[0])
        for op in ops9[1:]:
            mat_prod = mat_prod @ _bloch_mat_c128(op)
        ref_eqs = 0.5 * ((mat_prod - sz).abs().square().sum(dim=(-2, -1))).to(torch.float32)
        assert (m.eq_s_lsb_residual(ops9) - ref_eqs).abs().max().item() < 1e-4
        # Commutator == ||[Xa, Xb]||_F^2
        ref_comm = ((Xa @ Xb - Xb @ Xa).abs().square().sum(dim=(-2, -1))).to(torch.float32)
        assert (m.commutator_sq(a, b) - ref_comm).abs().max().item() < 1e-5
    elif variant == "v4c":
        a = m.sample_random_manifold((16,), device=torch.device("cpu"), generator=gen)
        b = m.sample_random_manifold((16,), device=torch.device("cpu"), generator=gen)
        comm_direct = (m.mul(a, b) - m.mul(b, a)).square().sum(dim=-1)
        assert (m.commutator_sq(a, b) - comm_direct).abs().max().item() < 1e-5

    # Lean 4 involutive/quat rational through-wall path t in [0, 1] keeps unit norm
    ts = torch.linspace(0.0, 1.0, 21)
    den = 1.0 - 2.0 * ts + 2.0 * ts * ts
    c_t = (1.0 - 2.0 * ts) / den
    s_t = (2.0 * ts * (1.0 - ts)) / den
    u_path = torch.zeros((21, m.dim), dtype=torch.float32)
    u_path[:, m.spin_axis] = c_t
    u_path[:, m.trans_axes[0]] = s_t
    assert torch.allclose(torch.linalg.vector_norm(u_path, dim=-1), torch.ones(21), atol=1e-6)
    assert torch.allclose(u_path[0], m.e0_vec(torch.device("cpu")), atol=1e-6)
    assert torch.allclose(u_path[-1], -m.e0_vec(torch.device("cpu")), atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_v4_discrete_limit_zero_error_r18(variant: str) -> None:
    rounds = 18
    gen = torch.Generator().manual_seed(202618)
    w16 = sample_random_w16(16, "int_range", 32, 4, torch.device("cpu"), gen)
    x_star, cb = prepare_ground_truth_nodes(w16, rounds=rounds, domain_mode="int_range", num_bits=32)
    mb = ManifoldBoundary.from_clamped_boundary(cb)

    phys = ManifoldPhysicsEngine(variant=variant, rounds=rounds)
    db = DualBasisEngine(rounds=rounds)
    m = phys.manifold

    u_star = m.embed_spins(x_star)
    u_a, u_e, u_w = mb.assemble_full_manifold(u_star, m)
    total_e, feat_v4, stats = phys.compute_energy_and_features(
        u_a, u_e, u_w, free_mask=1.0 - mb.clamp_mask, w_xor=1.0, lam_comm=1.0, lam_align=1.0
    )
    a_full, e_full, w_full = cb.assemble_full(x_star)
    e_db, feat_db, _ = db.compute_energy_and_features(a_full, e_full, w_full)

    assert stats["energy_2adic"].max().item() < 1e-10
    assert stats["energy_xor"].max().item() < 1e-10
    assert stats["energy_comm"].max().item() < 1e-10
    assert stats["energy_align"].max().item() < 1e-10
    assert total_e.max().item() < 1e-10
    assert (stats["energy_2adic"] - e_db).abs().max().item() < 1e-10
    assert (feat_v4 - feat_db).abs().max().item() < 1e-6

    # Flip 1 bit of free W0 -> total energy jumps >= 2.0
    u_flip = u_star.clone()
    u_flip[:, 0, 2, 0, :] = -u_flip[:, 0, 2, 0, :]
    ua_f, ue_f, uw_f = mb.assemble_full_manifold(u_flip, m)
    e_flip, _, _ = phys.compute_energy_and_features(ua_f, ue_f, uw_f, lam_comm=0.0, lam_align=0.0)
    assert float(e_flip.min().item()) >= 2.0


@pytest.mark.parametrize("variant", VARIANTS)
def test_v4_closed_loop_and_strict_fp32_grads(variant: str) -> None:
    enforce_fp32_precision()
    rounds = 18
    phys = ManifoldPhysicsEngine(variant=variant, rounds=rounds)
    m = phys.manifold
    gen = torch.Generator().manual_seed(777)
    w16 = sample_random_w16(4, "int_range", 32, 4, torch.device("cpu"), gen)
    x_star, cb = prepare_ground_truth_nodes(w16, rounds=rounds, domain_mode="int_range", num_bits=32)
    mb = ManifoldBoundary.from_clamped_boundary(cb)

    # Zero leakage check
    assert "w0" not in inspect.signature(apply_manifold_closed_loop_schedule).parameters

    # Discrete closed-loop schedule matches ground truth W_{16..17} with 0 error & finite W0 grad
    u_star = m.embed_spins(x_star)
    u_closed_disc = apply_manifold_closed_loop_schedule(u_star, m, rounds)
    assert (m.project_spin(u_closed_disc[:, 16:, 2]) - x_star[:, 16:, 2]).abs().max().item() < 1e-6

    u_cont = m.sample_random_manifold((4, rounds, 3, 32), device=torch.device("cpu"), generator=gen).requires_grad_(True)
    u_closed = apply_manifold_closed_loop_schedule(u_cont, m, rounds)
    assert torch.allclose(torch.linalg.vector_norm(u_closed, dim=-1), torch.ones(4, rounds, 3, 32), atol=1e-5)
    loss_w = (u_closed[:, 16:, 2] * torch.randn_like(u_closed[:, 16:, 2])).sum()
    loss_w.backward()
    assert torch.isfinite(u_cont.grad[:, 0, 2]).all()
    assert u_cont.grad[:, 0, 2].abs().max().item() > 1e-6

    # Strict FP32 forward/backward through ManifoldBitGridBackbone + chunked BPTT
    model = ManifoldBitGridBackbone(
        variant=variant, rounds=rounds, d_bit=48, n_heads=3, n_blocks=2, d_ff_mult=2, use_carry_gru=True
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step_idx in range(2):
        opt.zero_grad()
        u_init = build_v4_init(mb, m, num_particles=1, zero_prob=0.5, generator=gen)
        res = run_v4_chunked_bptt_step(
            train_model=model,
            raw_model=model,
            physics=phys,
            x_star=x_star,
            boundary=mb,
            u_init=u_init,
            num_particles=1,
            k_unroll=3,
            bptt_chunk_len=2,
            do_backward=True,
        )
        assert not any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any() for p in model.parameters() if p.grad is not None)
        for p in model.parameters():
            assert p.dtype == torch.float32
            if p.grad is not None:
                assert p.grad.dtype == torch.float32
        if step_idx == 1:
            assert model.manifold_in_proj.weight.grad is not None
            assert model.manifold_tangent_head.weight.grad is not None
            assert model.manifold_in_proj.weight.grad.abs().max().item() > 0.0
            assert model.manifold_tangent_head.weight.grad.abs().max().item() > 0.0
        opt.step()


@pytest.mark.parametrize("variant", VARIANTS)
def test_v4_v3_warmstart(variant: str) -> None:
    if not STAGE3B_CKPT.exists():
        pytest.skip("stage3b_r18_calibrated_best.pt not found")
    rounds = 18
    model = ManifoldBitGridBackbone(
        variant=variant, rounds=rounds, d_bit=48, n_heads=3, n_blocks=2, d_ff_mult=2, use_carry_gru=True
    )
    info = model.load_v3_warmstart(STAGE3B_CKPT, map_location="cpu")
    assert all(
        k.startswith("manifold_in_proj.") or k.startswith("manifold_tangent_head.")
        for k in info["missing_adapter_keys"]
    )
    phys = ManifoldPhysicsEngine(variant=variant, rounds=rounds)
    solver = UnrolledSolverV4(
        model=model,
        physics=phys,
        num_candidates=4,
        num_steps=16,
        damp=0.25,
        sync_h_damp=True,
        closed_loop_schedule=True,
        use_dynamic_latch=True,
        latch_every=2,
    )
    gen = torch.Generator().manual_seed(555)
    w16 = sample_random_w16(2, "int_range", 32, 4, torch.device("cpu"), gen)
    digs = sha256_digest(w16, rounds=rounds)
    mb = build_manifold_boundary(digs, rounds=rounds)
    logits_z, x_disc_z, _, _, latch_z, _, _ = solver.infer_deterministic_zero(mb)
    assert logits_z.dtype == torch.float32
    assert torch.isfinite(logits_z).all()
    assert float(latch_z.mean().item()) >= 0.10  # Inherits V3 W17/E14/A10 latching immediately
    probs = solver.predict_calibrated_bit_probs(digs)
    assert probs["node_bit1_probs"].shape == (2, 18, 3, 32)


@pytest.mark.parametrize("variant", VARIANTS)
def test_v4_v4e_partial_prefix_interface(variant: str) -> None:
    """Zero-modification V4E partial-prefix PoW + arbitrary multi-word mask compatibility."""
    rounds = 18
    phys = ManifoldPhysicsEngine(variant=variant, rounds=rounds)
    m = phys.manifold

    # 1. Verify dynamic A_R / E_R synthesis matches exact sha256_trace on valid 4-word message
    gen = torch.Generator().manual_seed(999)
    w16_true = torch.zeros((2, 16), dtype=torch.int64)
    w16_true[:, 0:4] = torch.randint(0, 1 << 32, (2, 4), dtype=torch.int64, generator=gen)
    w16_true[:, 4] = 0x80000000
    w16_true[:, 15] = 128
    tr = sha256_trace(w16_true, rounds=rounds)

    mb_v4e = build_v4e_partial_prefix_boundary(
        rounds=rounds,
        zero_prefix_bits=16,
        target_words=(0,),
        free_w_indices=(0, 1, 2, 3),
        batch_size=2,
        device="cpu",
    )
    assert mb_v4e.is_full_digest_clamped is False
    assert mb_v4e.has_free_right_boundary is True

    # Build exact x_star nodes for W_{0..17} and A_{0..17}, E_{0..17}
    x_star_v4e = torch.stack(
        [
            words_to_spins(tr.A_full[:, 3 : 3 + rounds], dtype=torch.float32),
            words_to_spins(tr.E_full[:, 3 : 3 + rounds], dtype=torch.float32),
            words_to_spins(tr.W_full[:, :rounds], dtype=torch.float32),
        ],
        dim=2,
    )
    u_star = m.embed_spins(x_star_v4e)
    ua_full, ue_full, uw_full = mb_v4e.assemble_full_manifold(u_star, m)
    assert (m.project_spin(ua_full[:, rounds + 3]) - words_to_spins(tr.A_full[:, rounds + 3])).abs().max().item() < 1e-6
    assert (m.project_spin(ue_full[:, rounds + 3]) - words_to_spins(tr.E_full[:, rounds + 3])).abs().max().item() < 1e-6

    # Physics energy without prefix mismatch is 0 for all compression & schedule eqs
    _, _, st = phys.compute_energy_and_features(
        ua_full, ue_full, uw_full, free_mask=1.0 - mb_v4e.clamp_mask, include_tail_e=True
    )
    assert st["energy_2adic"].max().item() < 1e-10
    assert st["energy_xor"].max().item() < 1e-10

    # 2. Verify same backbone + slow-thinking solver handles V4E directly with finite W_{0..3} grads
    model = ManifoldBitGridBackbone(
        variant=variant, rounds=rounds, d_bit=48, n_heads=3, n_blocks=2, d_ff_mult=2, use_carry_gru=True
    )
    u_var = build_v4_init(mb_v4e, m, num_particles=1, zero_prob=0.5, generator=gen).detach().requires_grad_(True)
    u_a2, u_e2, u_w2 = mb_v4e.assemble_full_manifold(u_var, m)
    e_total, _, _ = phys.compute_energy_and_features(
        u_a2,
        u_e2,
        u_w2,
        free_mask=1.0 - mb_v4e.clamp_mask,
        digest_target_bits=mb_v4e.digest_target_bits,
        digest_target_mask=mb_v4e.digest_target_mask,
        iv_words=mb_v4e.iv_words,
        include_tail_e=True,
    )
    e_total.sum().backward()
    for w_i in (0, 1, 2, 3):
        assert torch.isfinite(u_var.grad[:, w_i, 2]).all()
        assert u_var.grad[:, w_i, 2].abs().max().item() > 0.0

    solver = UnrolledSolverV4(
        model=model,
        physics=phys,
        num_candidates=16,
        num_steps=4,
        damp=0.25,
        smc_resample_every=2,
        smc_elite_k=4,
    )
    res = solver.solve(boundary=mb_v4e, generator=gen)
    assert res.best_w16.shape == (2, 16)
    assert (res.best_w16[:, 4] == 0x80000000).all()
    assert (res.best_w16[:, 15] == 128).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_v4_wta_and_checkpoint_resume(variant: str, tmp_path: Path) -> None:
    rounds = 18
    phys = ManifoldPhysicsEngine(variant=variant, rounds=rounds)
    m = phys.manifold
    gen = torch.Generator().manual_seed(321)
    w16 = sample_random_w16(2, "int_range", 32, 4, torch.device("cpu"), gen)
    x_star_b, cb_b = prepare_ground_truth_nodes(w16, rounds=rounds, domain_mode="int_range", num_bits=32)
    mb = ManifoldBoundary.from_clamped_boundary(cb_b).repeat_for_candidates(2)
    x_star = x_star_b.repeat_interleave(2, dim=0)

    # Particle 0 near exact solution, particle 1 random noisy -> winner gets larger physics grad share
    u_step = torch.empty((4, rounds, 3, 32, m.dim), dtype=torch.float32)
    u_true = m.embed_spins(x_star)
    u_rand = m.sample_random_manifold((4, rounds, 3, 32), device=torch.device("cpu"), generator=gen)
    u_step[0::2] = m.retract(u_true[0::2], u_rand[0::2] * 0.05, step_size=1.0)
    u_step[1::2] = u_rand[1::2]
    u_step.requires_grad_(True)
    logits = torch.zeros_like(x_star, requires_grad=True)

    loss, _ = compute_v4_step_loss(
        logits_spin=logits,
        u_step_end=u_step,
        x_star=x_star,
        boundary=mb,
        physics=phys,
        num_particles=2,
        wta_winner_weight=0.90,
        bce_weight=0.1,
        closed_loop_wall_rounds=(0, 13, 16),
        closed_loop_bce_scale=0.0,
    )
    loss.backward()
    assert torch.isfinite(u_step.grad).all()
    # Closed-loop wall rounds have 0 BCE contribution when closed_loop_bce_scale=0.0 & anti_collapse handles push
    assert logits.grad[:, 16].abs().max().item() < 1e-6

    # Save & lossless resume verification
    cfg = {
        "variant": variant,
        "rounds": rounds,
        "d_bit": 48,
        "n_heads": 3,
        "n_blocks": 2,
        "d_ff_mult": 2,
        "use_carry_gru": True,
        "batch_size": 4,
        "num_particles": 2,
        "K_unroll": 2,
        "bptt_chunk_len": 2,
        "K_infer": 4,
        "anneal_K": 4,
        "anneal_eval_samples": 2,
        "holdout_size": 8,
        "eval_batch_size": 8,
        "steps": 4,
        "use_torch_compile": "false",
        "seed": 11,
    }
    tr1 = UnrolledTrainerV4(cfg, output_dir=tmp_path / "run", device="cpu")
    tr1.train_one_step()
    tr1.train_one_step()
    ckpt_p = tr1.save_checkpoint("latest.pt")

    tr2 = UnrolledTrainerV4(cfg, output_dir=tmp_path / "run", device="cpu")
    tr2.load_checkpoint(ckpt_p)
    assert tr2.step == 2
    for (k1, v1), (k2, v2) in zip(tr1.model.state_dict().items(), tr2.model.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)


@pytest.mark.parametrize("variant", VARIANTS)
def test_v4_speedup_fp32_mathematical_equivalence(variant: str) -> None:
    """Assert strict FP32 (allow_tf32=False), Conv1d GEMM parity <1e-6, 2-wave schedule <1e-6, and <1e-5 compiled parity."""
    import copy
    import platform
    import torch.nn as nn
    import torch.nn.functional as F
    from src.bit_grid_backbone import _conv1d_shift_gemm_seq
    from src.dual_basis import spin_sigma0, spin_sigma1
    from src.v3_solver import spin_add32

    enforce_fp32_precision()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False

    want_compile = (platform.system() == "Linux" and device.type == "cuda" and hasattr(torch, "compile"))
    add32_fn = None
    loss_head_fn = None
    if want_compile:
        cache_key = (device.type, device.index)
        if cache_key not in _SHARED_ADD32_CACHE:
            _SHARED_ADD32_CACHE[cache_key] = torch.compile(
                _spin_add32_flat, mode="default", dynamic=True
            )
        if cache_key not in _SHARED_LOSS_HEAD_CACHE:
            _SHARED_LOSS_HEAD_CACHE[cache_key] = torch.compile(
                _bce_wta_loss_head, mode="default"
            )
        add32_fn = _SHARED_ADD32_CACHE[cache_key]
        loss_head_fn = _SHARED_LOSS_HEAD_CACHE[cache_key]

    # 1. Exact shift-GEMM Conv1d equivalence (< 1e-5)
    g_conv = torch.Generator(device="cpu").manual_seed(777)
    x_seq = torch.randn((18, 32, 48), dtype=torch.float32, generator=g_conv).to(device)
    conv_in = nn.Conv1d(48, 96, kernel_size=5, padding=2).to(device).float()
    conv_dil = nn.Conv1d(48, 48, kernel_size=3, padding=2, dilation=2).to(device).float()

    x_ref = x_seq.detach().clone().requires_grad_(True)
    g_ref, u_ref = conv_in(x_ref.transpose(1, 2)).chunk(2, dim=1)
    y_ref = conv_dil(F.silu(g_ref) * u_ref).transpose(1, 2)
    y_ref.square().mean().backward()

    conv_in_opt = copy.deepcopy(conv_in)
    conv_dil_opt = copy.deepcopy(conv_dil)
    x_opt = x_seq.detach().clone().requires_grad_(True)
    g_o, u_o = _conv1d_shift_gemm_seq(x_opt, conv_in_opt).chunk(2, dim=-1)
    y_opt = _conv1d_shift_gemm_seq(F.silu(g_o) * u_o, conv_dil_opt)
    y_opt.square().mean().backward()

    assert (y_opt - y_ref).abs().max().item() < 1e-5
    assert (x_opt.grad - x_ref.grad).abs().max().item() < 1e-5
    assert (conv_in_opt.weight.grad - conv_in.weight.grad).abs().max().item() < 1e-5
    assert (conv_dil_opt.weight.grad - conv_dil.weight.grad).abs().max().item() < 1e-5

    # 2. Exact 2-wave vectorized closed-loop schedule equivalence (< 1e-6)
    rounds = 18
    phys = ManifoldPhysicsEngine(variant=variant, rounds=rounds).to(device)
    m = phys.manifold
    u_raw = m.sample_random_manifold(
        (4, rounds, 3, 32), device=device, generator=torch.Generator(device=device).manual_seed(555)
    )
    u_ref_in = u_raw.detach().clone().requires_grad_(True)
    u_ref_f = normalize_manifold(u_ref_in)
    spins_16 = m.project_spin(u_ref_f[:, :16, 2, :, :])
    w_cols_ref = [spins_16[:, t, :] for t in range(16)]
    for t in range(16, rounds):
        s1 = spin_sigma1(w_cols_ref[t - 2])
        s0 = spin_sigma0(w_cols_ref[t - 15])
        wt = spin_add32(spin_add32(s1, w_cols_ref[t - 7]), spin_add32(s0, w_cols_ref[t - 16]))
        w_cols_ref.append(wt)
    u_exp_ref = m.compose_from_spin_and_transverse_dir(
        torch.stack(w_cols_ref[16:rounds], dim=1), u_ref_f[:, 16:rounds, 2, :, :]
    )
    u_sched_ref = torch.stack(
        [u_ref_f[:, :, 0, :, :], u_ref_f[:, :, 1, :, :], torch.cat([u_ref_f[:, :16, 2, :, :], u_exp_ref], dim=1)],
        dim=2,
    )
    (u_sched_ref.square().mean()).backward()

    u_vec_in = u_raw.detach().clone().requires_grad_(True)
    u_sched_vec = apply_manifold_closed_loop_schedule(u_vec_in, m, rounds, add32_fn=add32_fn)
    (u_sched_vec.square().mean()).backward()
    assert (u_sched_vec - u_sched_ref).abs().max().item() < 1e-6
    assert (u_vec_in.grad - u_ref_in.grad).abs().max().item() < 1e-6

    # 3. Full R=18 Closed-Loop K_unroll=4 backward equivalence across backward_group_chunks=1, 2, 4
    model_ref = ManifoldBitGridBackbone(
        variant=variant, rounds=rounds, d_bit=48, n_heads=3, n_blocks=2, d_ff_mult=2, use_carry_gru=True
    ).to(device).float()
    model_ref.train()

    g = torch.Generator(device="cpu").manual_seed(2025)
    w16 = sample_random_w16(2, mode="int_range", num_bits=32, device="cpu", generator=g).to(device)
    x_star_b, cb_b = prepare_ground_truth_nodes(w16, rounds=rounds, domain_mode="int_range", num_bits=32)
    x_star = x_star_b.repeat_interleave(2, dim=0)
    mb = ManifoldBoundary.from_clamped_boundary(cb_b).repeat_for_candidates(2)
    u_init = build_v4_init(
        mb, phys.manifold, num_particles=2, generator=torch.Generator(device=device).manual_seed(99)
    )

    model_ref.zero_grad(set_to_none=True)
    res_ref = run_v4_chunked_bptt_step(
        train_model=model_ref,
        raw_model=model_ref,
        physics=phys,
        x_star=x_star,
        boundary=mb,
        u_init=u_init,
        num_particles=2,
        k_unroll=4,
        bptt_chunk_len=2,
        backward_group_chunks=1,
        sync_metrics=True,
        do_backward=True,
    )
    grads_ref = {
        name: p.grad.detach().clone()
        for name, p in model_ref.named_parameters()
        if p.grad is not None
    }

    model_opt = copy.deepcopy(model_ref).to(device).float()
    train_mod = torch.compile(model_opt, mode="default") if want_compile else model_opt
    energy_fn = torch.compile(phys.compute_energy_and_features_core, mode="default") if want_compile else None
    retract_fn = torch.compile(phys.retract_and_clamp, mode="default") if want_compile else None

    for group_chunks in (2, 4):
        model_opt.load_state_dict(model_ref.state_dict())
        model_opt.train()
        model_opt.zero_grad(set_to_none=True)

        res_opt = run_v4_chunked_bptt_step(
            train_model=train_mod,
            raw_model=model_opt,
            physics=phys,
            x_star=x_star,
            boundary=mb,
            u_init=u_init,
            num_particles=2,
            k_unroll=4,
            bptt_chunk_len=2,
            backward_group_chunks=group_chunks,
            sync_metrics=True,
            radar_energy_fn=energy_fn,
            retract_clamp_fn=retract_fn,
            add32_fn=add32_fn,
            loss_head_fn=loss_head_fn,
            do_backward=True,
        )
        assert abs(res_opt["loss"] - res_ref["loss"]) < 1e-5
        assert abs(res_opt["phys_loss"] - res_ref["phys_loss"]) < 1e-5
        for name, p in model_opt.named_parameters():
            if name in grads_ref:
                assert p.grad is not None and torch.isfinite(p.grad).all()
                diff = (p.grad - grads_ref[name]).abs().max().item()
                assert diff < 1e-5, f"{variant} group={group_chunks} grad diff on {name}: {diff}"


@pytest.mark.parametrize("variant", VARIANTS)
def test_v4_fused_adamw_resume_compatibility(variant: str, tmp_path: Path) -> None:
    """Verify non-fused AdamW checkpoint resumes losslessly into CUDA fused=True trainer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {
        "variant": variant,
        "rounds": 18,
        "d_bit": 48,
        "n_heads": 3,
        "n_blocks": 2,
        "d_ff_mult": 2,
        "use_carry_gru": True,
        "batch_size": 4,
        "num_particles": 2,
        "K_unroll": 2,
        "bptt_chunk_len": 2,
        "backward_group_chunks": 2,
        "K_infer": 4,
        "anneal_K": 4,
        "anneal_eval_samples": 2,
        "holdout_size": 8,
        "eval_batch_size": 8,
        "steps": 4,
        "use_torch_compile": "false",
        "use_async_prefetch": False,
        "seed": 19,
    }
    tr1 = UnrolledTrainerV4(cfg, output_dir=tmp_path / "run_old", device=device)
    # Simulate legacy non-fused AdamW state in checkpoint
    for pg in tr1.optimizer.param_groups:
        pg["fused"] = False
    tr1.train_one_step(sync_metrics=True)
    tr1.train_one_step(sync_metrics=False)
    ckpt_p = tr1.save_checkpoint("legacy_latest.pt")

    tr2 = UnrolledTrainerV4(cfg, output_dir=tmp_path / "run_new", device=device)
    tr2.load_checkpoint(ckpt_p)
    assert tr2.step == 2
    for pg in tr2.optimizer.param_groups:
        assert bool(pg.get("fused", False)) == tr2._use_fused_adamw

    m3 = tr2.train_one_step(sync_metrics=True)
    assert tr2.step == 3
    assert all(torch.isfinite(torch.tensor(v)) for v in m3.values())
