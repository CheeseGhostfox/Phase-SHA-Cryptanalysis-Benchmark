"""Acceptance test suite for Phase-SHA V2: 2D bit-grid ops, BCE, unrolled VRAM, and diversity."""

from __future__ import annotations

import math
import time

import pytest
import torch
import torch.nn as nn

from src.bit_grid_backbone import (
    SHA_ROTR_OFFSETS,
    BitGridBackbone,
    BitGridBlock,
    build_rotr_channels,
    extract_grid_features,
)
from src.boundary_clamp import build_clamped_boundary
from src.dual_basis import DualBasisEngine, spin_rotr
from src.flow_trainer import prepare_ground_truth_nodes
from src.sha256_ref import configure_cuda_memory, sample_random_w16, sha256_digest
from src.v2_solver import UnrolledSolverV2
from src.v2_trainer import (
    UnrolledTrainerV2,
    build_corrupted_init,
    compute_v2_step_loss,
    run_unrolled_training_step,
)

if torch.cuda.is_available():
    configure_cuda_memory(0.80)


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_v2_rotr_channels_and_sparse_round_isolation() -> None:
    """Verify ROTR feature alignment, 2D residual grid alignment, and round-axis isolation."""
    device = _get_device()
    rounds = 24
    gen = torch.Generator(device=device).manual_seed(123)
    w16 = sample_random_w16(batch_size=4, num_bits=32, device=device, generator=gen)
    x_star, boundary = prepare_ground_truth_nodes(w16, rounds=rounds)

    engine = DualBasisEngine(rounds=rounds).to(device)
    x_noisy = boundary.apply_clamp(
        0.5 * x_star
        + 0.5 * torch.randn(x_star.shape, device=device, generator=gen)
    )
    lam = torch.ones(32, dtype=torch.float32, device=device)
    radar, res_feat, _, _ = engine.compute_radar_and_features(x_noisy, boundary, lam=lam)

    base_feat, x_grid, grad_grid = extract_grid_features(
        x_noisy, radar, boundary.clamp_mask, res_feat, engine.K_spins
    )
    assert base_feat.shape == (4, rounds, 32, 16)

    # Check 2D residual channels align directly with scale m = k + 1
    a_full, e_full, w_full = boundary.assemble_full(x_noisy)
    delta_e, delta_w, delta_s = engine.compute_wrapped_deltas(a_full, e_full, w_full)
    assert delta_s is not None
    # base_feat channels 9..14 are [sin_E, omc_E, sin_W, omc_W, sin_S, omc_S]
    assert torch.allclose(base_feat[:, : rounds - 4, :, 9], torch.sin(delta_e), atol=1e-6)
    assert torch.allclose(base_feat[:, :, :, 11], torch.sin(delta_w), atol=1e-6)
    assert torch.allclose(base_feat[:, 16:, :, 13], torch.sin(delta_s), atol=1e-6)

    # Check ROTR(2,6,11,13,22,25) channels match spin_rotr exactly
    rot_feat = build_rotr_channels(x_grid, grad_grid)
    assert rot_feat.shape == (4, rounds, 32, 36)
    for idx, off in enumerate(SHA_ROTR_OFFSETS):
        chunk = rot_feat[:, :, :, idx * 6 : (idx + 1) * 6]
        expected_x = spin_rotr(x_noisy, off).permute(0, 1, 3, 2)
        expected_g = spin_rotr(radar, off).permute(0, 1, 3, 2)
        assert torch.equal(chunk[:, :, :, 0:3], expected_x)
        assert torch.equal(chunk[:, :, :, 3:6], expected_g)

    # Check round-axis sparse attention isolates non-adjacent rounds (0 vs 10)
    backbone = BitGridBackbone(rounds=rounds, d_bit=32, n_heads=4, n_blocks=1).to(device)
    block = backbone.blocks[0]
    mask = backbone.round_attn_mask
    assert not bool(mask[0, 0, 10, 0].item())
    assert bool(mask[0, 0, 7, 0].item())

    seq_a = torch.randn((1, rounds, 32), dtype=torch.float32, device=device, generator=gen)
    seq_b = seq_a.clone()
    seq_b[:, 0, :] += 5.0
    with torch.no_grad():
        out_a = block._attention_1d(seq_a, block.round_qkv, block.round_proj, attn_mask=mask)
        out_b = block._attention_1d(seq_b, block.round_qkv, block.round_proj, attn_mask=mask)
    assert torch.allclose(out_a[:, 10, :], out_b[:, 10, :], atol=1e-6)
    assert not torch.allclose(out_a[:, 7, :], out_b[:, 7, :], atol=1e-3)


def test_v2_bce_blind_ln2_and_perfect_zero() -> None:
    """Blind zero logits must give BCE == ln(2); ground-truth logits give ~0 loss."""
    device = _get_device()
    rounds = 12
    gen = torch.Generator(device=device).manual_seed(456)
    w16 = sample_random_w16(batch_size=8, num_bits=16, device=device, generator=gen)
    x_star, boundary = prepare_ground_truth_nodes(w16, rounds=rounds, num_bits=16)
    engine = DualBasisEngine(rounds=rounds).to(device)

    zero_logits = torch.zeros_like(x_star)
    _, bce_blind, _, _, _ = compute_v2_step_loss(
        zero_logits, x_star, boundary, engine, alpha_phys=0.25
    )
    assert abs(bce_blind - math.log(2.0)) < 1e-5

    perfect_logits = 20.0 * x_star
    _, bce_perf, phys_perf, m1_perf, m32_perf = compute_v2_step_loss(
        perfect_logits, x_star, boundary, engine, alpha_phys=0.25
    )
    assert bce_perf < 1e-8
    assert phys_perf < 1e-10
    assert m1_perf < 1e-10
    assert m32_perf < 1e-10


@pytest.mark.parametrize(
    "rounds,batch_size,grad_ckpt,max_ms",
    [(12, 128, False, 600.0), (12, 256, False, 1500.0), (64, 256, True, 10000.0)],
)
def test_v2_unrolled_vram_budget_r12_and_r64(
    rounds: int, batch_size: int, grad_ckpt: bool, max_ms: float
) -> None:
    """Verify K_unroll=6 truncated-BPTT training step fits comfortably under 4.5 GB VRAM."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for V2 VRAM check.")

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    engine = DualBasisEngine(rounds=rounds).to(device)
    model = BitGridBackbone(
        rounds=rounds,
        d_bit=48,
        n_heads=3,
        n_blocks=2,
        d_ff_mult=2,
        share_blocks=False,
        grad_checkpointing=grad_ckpt,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
    gen = torch.Generator(device=device).manual_seed(789 + rounds + batch_size)

    t_last = 0.0
    nonzero_grads = 0
    for step_idx in range(2):
        w16 = sample_random_w16(batch_size=batch_size, num_bits=32, device=device, generator=gen)
        x_star, boundary = prepare_ground_truth_nodes(w16, rounds=rounds, num_bits=32)
        x_init = build_corrupted_init(x_star, boundary, generator=gen)

        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        metrics = run_unrolled_training_step(
            model=model,
            dual_basis=engine,
            x_star=x_star,
            boundary=boundary,
            x_init=x_init,
            k_unroll=6,
            alpha_phys=0.25,
            damp=0.45,
            do_backward=True,
        )
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if step_idx == 1:
            nonzero_grads = sum(
                1
                for p in model.parameters()
                if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            )
        optimizer.step()
        torch.cuda.synchronize(device)
        t_last = (time.perf_counter() - t0) * 1000.0
        assert math.isfinite(metrics["loss"])

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    print(
        f"\n[V2 VRAM Check] R={rounds}, B={batch_size}, K_unroll=6, grad_ckpt={grad_ckpt}: "
        f"peak={peak_gb:.3f} GB, step_time={t_last:.1f} ms",
        flush=True,
    )
    assert nonzero_grads > 0
    assert peak_gb < 4.5, f"Peak VRAM {peak_gb:.3f} GB exceeds 4.5 GB budget!"
    assert t_last < max_ms, f"Step time {t_last:.1f} ms exceeds {max_ms:.1f} ms!"


def test_v2_solver_hard_clamp_and_unique_ratio(tmp_path) -> None:
    """Verify V2 disjoint split, long-cycle min LR, hard clamping, and candidate diversity."""
    device = _get_device()
    cfg = {
        "version": "v2",
        "rounds": 12,
        "domain_mode": "int_range",
        "num_bits": 16,
        "holdout_size": 2048,
        "holdout_seed": 20261122,
        "d_bit": 32,
        "n_heads": 4,
        "n_blocks": 2,
        "d_ff_mult": 2,
        "K_unroll": 3,
        "K_infer": 32,
        "anneal_K": 256,
        "anneal_chunk_size": 512,
        "infer_damp": 0.10,
        "anneal_eta_max": 0.02,
        "anneal_beta_max": 0.25,
        "sde_noise_min": 0.10,
        "sde_noise_max": 0.45,
        "batch_size": 64,
        "steps": 5000,
        "lr_total_steps": 15000,
        "warmup_steps": 500,
        "min_lr_ratio": 0.1,
    }
    torch.manual_seed(2026)
    trainer = UnrolledTrainerV2(config=cfg, output_dir=tmp_path, device=device)
    assert trainer.train_w0_pool is not None
    assert trainer.holdout_w0_gpu is not None
    assert not bool(torch.isin(trainer.train_w0_pool, trainer.holdout_w0_gpu).any().item())

    # Verify LR stays >= 0.1 * max_lr even beyond step 5000
    assert trainer._lr_lambda(5000) > 0.7
    assert abs(trainer._lr_lambda(15000) - 0.1) < 1e-6

    digests = sha256_digest(trainer.holdout_w16[:2].to(device), rounds=12)
    gen = torch.Generator(device=device).manual_seed(999)
    res = trainer.solver.solve(
        digests,
        domain_mode="int_range",
        num_bits=16,
        generator=gen,
        return_last_trajectories=True,
    )
    assert res.solved.shape == (2,)
    assert res.unique_ratio.shape == (2,)
    assert res.mean_unique_ratio > 0.5, f"Mode collapse detected: unique_ratio={res.mean_unique_ratio}"
    assert res.final_x_sample is not None
    assert res.final_x_sample.shape == (256, 12, 3, 32)

    b1 = build_clamped_boundary(digests[1], rounds=12, domain_mode="int_range", num_bits=16)
    clamp_err = ((res.final_x_sample - b1.x_clamped) * b1.clamp_mask).abs().max().item()
    assert clamp_err == 0.0


@pytest.mark.parametrize("rounds", [17, 18])
def test_v2_message_schedule_clamping_and_residual_r17_r18(rounds: int) -> None:
    """Verify R=17,18 keep W_{16..R-1} free, back-solve A_{R-7..R-4}, and couple W0 via Eq S."""
    from src.sha256_ref import words_to_spins

    device = _get_device()
    gen = torch.Generator(device=device).manual_seed(1700 + rounds)
    w16 = sample_random_w16(
        batch_size=16, mode="int_range", num_bits=32, device=device, generator=gen
    )
    trace = sha256_trace(w16, rounds=rounds) if False else None  # verified via boundary + trace
    from src.sha256_ref import sha256_trace

    trace = sha256_trace(w16, rounds=rounds)
    boundary = build_clamped_boundary(
        digest=trace.digest, rounds=rounds, domain_mode="int_range", num_bits=32
    )

    # 4-round analytical back-solve on A_{R-7..R-4} exact with 0 bit errors
    assert torch.equal(
        boundary.A_words_clamped[:, rounds - 4 : rounds],
        trace.A_full[:, rounds - 4 : rounds],
    )

    # W_0 fully free; W_{1..15} clamped padding; W_{16..R-1} MUST be completely free
    assert boundary.W_full_mask[:, 0, :].abs().max().item() == 0.0
    assert (boundary.W_full_mask[:, 1:16, :] == 1.0).all()
    assert boundary.W_full_mask[:, 16:, :].abs().max().item() == 0.0

    # Ground truth discrete spins give < 1e-10 residual across all scales & Eq S
    a_gt = words_to_spins(trace.A_full, dtype=torch.float32)
    e_gt = words_to_spins(trace.E_full, dtype=torch.float32)
    w_gt = words_to_spins(trace.W_full, dtype=torch.float32)
    engine = DualBasisEngine(rounds=rounds).to(device)
    energy, _, stats = engine.compute_energy_and_features(a_gt, e_gt, w_gt)
    assert energy.max().item() < 1e-10
    assert stats["per_scale_s"].max().item() < 1e-10

    # Flipping a bit in W_0 while holding W_{16..R-1} fixed breaks Eq S (> 1.0)
    for k in (0, 7, 15, 23, 30):
        w_flip = w_gt.clone()
        w_flip[0, 0, k] = -w_flip[0, 0, k]
        _, _, stats_flip = engine.compute_energy_and_features(a_gt, e_gt, w_flip)
        eq_s_err = float(stats_flip["per_scale_s"][0].sum().item())
        assert eq_s_err > 1.0, f"R={rounds} W0[{k}] flip Eq S residual={eq_s_err} <= 1.0"


def test_v2_r18_k_unroll8_vram_under_5_5gb() -> None:
    """Verify R=18, B=128, K_unroll=8 training step stays well under 5.5 GB VRAM."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for R=18 VRAM check.")

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    rounds = 18
    batch_size = 128
    engine = DualBasisEngine(rounds=rounds).to(device)
    model = BitGridBackbone(
        rounds=rounds,
        d_bit=48,
        n_heads=3,
        n_blocks=2,
        d_ff_mult=2,
        share_blocks=False,
        grad_checkpointing=False,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)
    gen = torch.Generator(device=device).manual_seed(1818)

    nonzero_grads = 0
    for step_idx in range(2):
        w16 = sample_random_w16(
            batch_size=batch_size, num_bits=32, device=device, generator=gen
        )
        x_star, boundary = prepare_ground_truth_nodes(w16, rounds=rounds, num_bits=32)
        x_init = build_corrupted_init(x_star, boundary, generator=gen)
        optimizer.zero_grad(set_to_none=True)
        metrics = run_unrolled_training_step(
            model=model,
            dual_basis=engine,
            x_star=x_star,
            boundary=boundary,
            x_init=x_init,
            k_unroll=8,
            alpha_phys=0.25,
            damp=0.36,
            do_backward=True,
        )
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if step_idx == 1:
            nonzero_grads = sum(
                1
                for p in model.parameters()
                if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            )
        optimizer.step()
        assert math.isfinite(metrics["loss"])

    torch.cuda.synchronize(device)
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    print(f"\n[V2 R=18 VRAM Check] B=128, K_unroll=8 peak={peak_gb:.3f} GB", flush=True)
    assert nonzero_grads > 0
    assert peak_gb < 5.5


def test_v2_curriculum_weight_transfer_r12_to_r16_to_r18(tmp_path) -> None:
    """Verify strict weight-only warm-start from R=12 into R=16 and R=18 with 0 overlap."""
    from pathlib import Path

    device = _get_device()
    src_ckpt = Path("runs/r12_v2/best.pt")
    if not src_ckpt.exists():
        cfg12 = {
            "version": "v2",
            "rounds": 12,
            "domain_mode": "int_range",
            "num_bits": 16,
            "d_bit": 48,
            "n_heads": 3,
            "n_blocks": 2,
            "d_ff_mult": 2,
            "holdout_size": 2048,
        }
        t12 = UnrolledTrainerV2(config=cfg12, output_dir=tmp_path / "r12", device=device)
        src_ckpt = t12.save_checkpoint("best.pt")

    for target_r in (16, 18):
        cfg_t = {
            "version": "v2",
            "rounds": target_r,
            "domain_mode": "int_range",
            "num_bits": 32,
            "d_bit": 48,
            "n_heads": 3,
            "n_blocks": 2,
            "d_ff_mult": 2,
            "holdout_size": 2048,
            "batch_size": 32,
            "K_unroll": 2,
            "K_infer": 4,
            "anneal_K": 16,
        }
        tr = UnrolledTrainerV2(config=cfg_t, output_dir=tmp_path / f"r{target_r}", device=device)
        tr.load_weights_only(src_ckpt)
        assert tr.step == 0
        assert tr.holdout_w0_gpu is not None
        assert tr.holdout_w16.shape == (2048, 16)
        train_sample = tr._sample_train_w16(256)
        assert not bool(torch.isin(train_sample[:, 0], tr.holdout_w0_gpu).any().item())
        metrics = tr.train_one_step()
        assert math.isfinite(metrics["loss"])

