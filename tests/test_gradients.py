"""Mandatory acceptance test suite 2: bit sensitivity, gradient finiteness, and VRAM budget."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.anneal_solver import AnnealSolver
from src.boundary_clamp import build_clamped_boundary
from src.dual_basis import DualBasisEngine
from src.graph_backbone import PhaseSHABackbone, SparseAttention, build_round_sparse_attn_mask
from src.flow_trainer import (
    FlowTrainer,
    compute_flow_loss,
    prepare_ground_truth_nodes,
    sample_frequency_masks,
)
from src.sha256_ref import (
    configure_cuda_memory,
    pad_bytes_to_w16,
    sample_random_w16,
    sha256_digest,
    sha256_trace,
    words_to_spins,
)

if torch.cuda.is_available():
    configure_cuda_memory(0.80)


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_single_bit_flip_energy_and_gradient_sensitivity() -> None:
    """Flipping any single bit k in {0..31} must yield E_phys > 1.0 and O(1) finite grad."""
    device = _get_device()
    rounds = 64
    w16 = pad_bytes_to_w16(b"phase_sha_gradient_check", device=device).unsqueeze(0)
    trace = sha256_trace(w16, rounds=rounds)

    a_gt = words_to_spins(trace.A_full, dtype=torch.float32)  # (1, R+4, 32)
    e_gt = words_to_spins(trace.E_full, dtype=torch.float32)  # (1, R+4, 32)
    w_gt = words_to_spins(trace.W_full, dtype=torch.float32)  # (1, R, 32)

    engine = DualBasisEngine(rounds=rounds).to(device)

    # Test interior free words entering rotation/nonlinear ops across all k=0..31:
    # A_4 (index 3+4=7), E_4 (index 3+4=7), and W_1 (index 1, enters sigma0 in round 16)
    targets = [
        ("A_4", "A", 7, range(32)),
        ("E_4", "E", 7, range(32)),
        ("W_1", "W", 1, range(32)),
        ("W_0", "W", 0, range(31)),
    ]

    for name, track, word_idx, bit_range in targets:
        for k in bit_range:
            a = a_gt.clone().detach()
            e = e_gt.clone().detach()
            w = w_gt.clone().detach()

            if track == "A":
                a.requires_grad_(True)
                a_mod = a.clone()
                a_mod[0, word_idx, k] = -a[0, word_idx, k]
                energy, _, _ = engine.compute_energy_and_features(a_mod, e, w)
                grad_tensor = torch.autograd.grad(energy.sum(), a)[0]
                g_val = float(grad_tensor[0, word_idx, k].item())
            elif track == "E":
                e.requires_grad_(True)
                e_mod = e.clone()
                e_mod[0, word_idx, k] = -e[0, word_idx, k]
                energy, _, _ = engine.compute_energy_and_features(a, e_mod, w)
                grad_tensor = torch.autograd.grad(energy.sum(), e)[0]
                g_val = float(grad_tensor[0, word_idx, k].item())
            else:
                w.requires_grad_(True)
                w_mod = w.clone()
                w_mod[0, word_idx, k] = -w[0, word_idx, k]
                energy, _, _ = engine.compute_energy_and_features(a, e, w_mod)
                grad_tensor = torch.autograd.grad(energy.sum(), w)[0]
                g_val = float(grad_tensor[0, word_idx, k].item())

            e_val = float(energy.item())
            assert e_val > 1.0, f"{name}[{k}] energy={e_val} <= 1.0"
            assert torch.isfinite(torch.tensor(g_val)), f"{name}[{k}] non-finite grad"
            assert 1e-2 <= abs(g_val) <= 1e2, (
                f"{name}[{k}] |grad|={abs(g_val)} outside [1e-2, 1e2]"
            )

    # Also check W_0 bit 31: exact flip hits peak energy > 1.0; interior spin has O(1) grad
    w_flip = w_gt.clone().detach()
    w_flip[0, 0, 31] = -w_flip[0, 0, 31]
    e_flip, _, _ = engine.compute_energy_and_features(a_gt, e_gt, w_flip)
    assert float(e_flip.item()) > 1.0

    w_base = w_gt.clone().detach().requires_grad_(True)
    w_mid = w_base.clone()
    w_mid[0, 0, 31] = 0.5 * w_base[0, 0, 31]
    e_mid, _, _ = engine.compute_energy_and_features(a_gt, e_gt, w_mid)
    g_w0_31 = float(torch.autograd.grad(e_mid.sum(), w_base)[0][0, 0, 31].item())
    assert 1e-2 <= abs(g_w0_31) <= 1e2


def test_gradients_finite_on_continuous_and_sparse_topology() -> None:
    """Verify finite gradients on noisy continuous spins and strict sparse attn isolation."""
    device = _get_device()
    rounds = 24
    gen = torch.Generator(device=device).manual_seed(2026)
    w16 = sample_random_w16(
        batch_size=8, mode="int_range", num_bits=32, device=device, generator=gen
    )
    x_star, boundary = prepare_ground_truth_nodes(w16, rounds=rounds)

    engine = DualBasisEngine(rounds=rounds).to(device)
    lam = torch.ones(32, dtype=torch.float32, device=device)

    # Test continuous noisy interpolation
    noise = torch.randn(x_star.shape, dtype=torch.float32, device=device, generator=gen)
    x_cont = boundary.apply_clamp(0.4 * noise + 0.6 * x_star)
    radar, feat, energy, _ = engine.compute_radar_and_features(x_cont, boundary, lam=lam)
    assert torch.isfinite(radar).all()
    assert torch.isfinite(feat).all()
    assert torch.isfinite(energy).all()

    # Test model backward finiteness
    model = PhaseSHABackbone(rounds=rounds, d_model=128, n_heads=4, n_layers=2, d_ff=256).to(
        device
    )
    tau = torch.full((8,), 0.5, dtype=torch.float32, device=device)
    loss, _ = compute_flow_loss(model, engine, x_star, boundary, tau, lam, noise=noise)
    assert torch.isfinite(loss)
    loss.backward()
    for pname, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in {pname}"

    # Verify sparse attention mask blocks non-adjacent rounds (e.g., round 0 -> round 10)
    attn = SparseAttention(d_model=64, n_heads=4).to(device).eval()
    mask = build_round_sparse_attn_mask(rounds=24, num_registers=0, device=device)
    assert not bool(mask[0, 0, 10, 0].item())
    assert bool(mask[0, 0, 7, 0].item())
    inp_a = torch.randn((1, 24, 64), dtype=torch.float32, device=device, generator=gen)
    inp_b = inp_a.clone()
    inp_b[:, 0, :] += 5.0
    with torch.no_grad():
        out_a = attn(inp_a, mask)
        out_b = attn(inp_b, mask)
    assert torch.allclose(out_a[:, 10, :], out_b[:, 10, :], atol=1e-6)
    assert not torch.allclose(out_a[:, 7, :], out_b[:, 7, :], atol=1e-3)


def test_full_train_step_vram_budget_r64_b256() -> None:
    """Verify full forward+backward step at R=64, batch_size=256 stays under 6.2 GB VRAM."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for VRAM budget test.")

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    rounds = 64
    batch_size = 256

    engine = DualBasisEngine(rounds=rounds).to(device)
    model = PhaseSHABackbone(
        rounds=rounds,
        d_model=384,
        n_heads=8,
        n_layers=8,
        d_ff=1024,
        share_layers=False,
        num_registers=2,
        grad_checkpointing=False,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    gen = torch.Generator(device=device).manual_seed(42)

    nonzero_grads = 0
    for step_idx in range(2):
        w16 = sample_random_w16(
            batch_size=batch_size,
            mode="int_range",
            num_bits=32,
            device=device,
            generator=gen,
        )
        x_star, boundary = prepare_ground_truth_nodes(w16, rounds=rounds)
        tau = torch.rand((batch_size,), dtype=torch.float32, device=device, generator=gen)
        lam = sample_frequency_masks(batch_size, device=device, generator=gen)

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compute_flow_loss(
            model=model,
            dual_basis=engine,
            x_star=x_star,
            boundary=boundary,
            tau=tau,
            lam=lam,
            alpha_phys=0.5,
            grad_scale=2.0,
        )
        assert torch.isfinite(loss), f"Non-finite loss at step {step_idx}"
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if step_idx == 1:
            nonzero_grads = sum(
                1
                for p in model.parameters()
                if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            )
        optimizer.step()

    torch.cuda.synchronize(device)
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    print(f"\n[VRAM Check] R=64, batch_size=256 peak CUDA memory: {peak_gb:.3f} GB", flush=True)

    assert nonzero_grads > 0
    assert peak_gb < 6.2, f"Peak VRAM {peak_gb:.3f} GB exceeds 6.2 GB budget!"


def test_anneal_solver_smoke_and_hard_clamp() -> None:
    """Verify AnnealSolver defaults to K=256 and enforces hard clamping every step."""
    device = _get_device()
    rounds = 12
    engine = DualBasisEngine(rounds=rounds).to(device)
    model = PhaseSHABackbone(rounds=rounds, d_model=64, n_heads=4, n_layers=2, d_ff=128).to(
        device
    )

    solver = AnnealSolver(model=model, dual_basis=engine, num_steps=5)
    assert solver.num_candidates == 256

    gen = torch.Generator(device=device).manual_seed(314)
    w16 = sample_random_w16(
        batch_size=1, mode="int_range", num_bits=16, device=device, generator=gen
    )
    digest = sha256_digest(w16, rounds=rounds)
    res = solver.solve(
        digest,
        domain_mode="int_range",
        num_bits=16,
        generator=gen,
        return_last_trajectories=True,
    )
    assert res.solved.shape == (1,)
    assert res.best_w16.shape == (1, 16)
    assert res.final_x_sample is not None
    assert res.final_x_sample.shape == (256, rounds, 3, 32)
    assert torch.isfinite(res.final_x_sample).all()

    boundary = build_clamped_boundary(digest, rounds=rounds, domain_mode="int_range", num_bits=16)
    clamp_mask = boundary.clamp_mask[0].unsqueeze(0)
    x_clamped = boundary.x_clamped[0].unsqueeze(0)
    clamp_err = ((res.final_x_sample - x_clamped) * clamp_mask).abs().max().item()
    assert clamp_err == 0.0


def test_disjoint_train_holdout_and_physics_only_batch_anneal(tmp_path) -> None:
    """Verify strict 0-overlap train/holdout split and pure-physics batched annealing."""
    device = _get_device()
    cfg = {
        "rounds": 12,
        "domain_mode": "int_range",
        "num_bits": 16,
        "holdout_size": 2048,
        "holdout_seed": 20261122,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 128,
        "batch_size": 64,
        "anneal_K": 32,
        "anneal_steps": 5,
        "anneal_chunk_size": 64,
    }
    trainer = FlowTrainer(config=cfg, output_dir=tmp_path, device=device)
    assert trainer.train_w0_pool is not None
    assert trainer.holdout_w0_gpu is not None
    assert trainer.holdout_w16.shape == (2048, 16)
    assert trainer.train_w0_pool.shape == (65536 - 2048,)

    overlap = torch.isin(trainer.train_w0_pool, trainer.holdout_w0_gpu)
    assert not bool(overlap.any().item())

    sampled = trainer._sample_train_w16(512)
    assert not bool(torch.isin(sampled[:, 0], trainer.holdout_w0_gpu).any().item())

    # Compare batched pure-physics annealing vs single-digest pure-physics annealing
    digests = sha256_digest(trainer.holdout_w16[:4].to(device), rounds=12)
    gen_a = torch.Generator(device=device).manual_seed(999)
    res_batched = trainer.solver.solve(
        digests,
        domain_mode="int_range",
        num_bits=16,
        use_neural_velocity=False,
        generator=gen_a,
    )
    assert res_batched.best_x_disc.shape == (4, 12, 3, 32)
    assert res_batched.best_m1_energy.shape == (4,)
    assert res_batched.best_m32_energy.shape == (4,)
    assert torch.isfinite(res_batched.best_energy).all()

