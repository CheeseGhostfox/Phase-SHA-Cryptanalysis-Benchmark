"""Fast CPU Unit & Smoke Verification for P1 InfoNCE Telescope."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch

from v11_discovery.p1_infonce.models import DualTowerInfoNCE, LinearInfoNCE
from v11_discovery.p1_infonce.run_p1_sweep import run_sweep
from v11_discovery.p1_infonce.sha256_telescope import (
    IN_FEATURE_NAMES,
    OUT_FEATURE_NAMES,
    STD_SIGMA0_LOWER,
    STD_SIGMA0_UPPER,
    STD_SIGMA1_LOWER,
    STD_SIGMA1_UPPER,
    encode_multiscale_features,
    enforce_fp32_determinism,
    make_control_config,
    sample_uniform_w16,
    sha256_sr_control,
    sha256_sr_main,
)


def test_sha256_equivalence_if_available() -> None:
    """If src/sha256_ref.py is present (local repo), assert bit-exact S_R equality."""
    try:
        from src.sha256_ref import sha256_trace
    except ImportError:
        print("[SKIP] src.sha256_ref not present in standalone bundle", flush=True)
        return

    gen = torch.Generator(device="cpu").manual_seed(42)
    w16 = sample_uniform_w16(16, device="cpu", generator=gen)
    for r in (4, 8, 16, 20, 64):
        sr_ref = sha256_trace(w16, rounds=r).S_R
        sr_p1 = sha256_sr_main(w16, rounds=r)
        assert torch.equal(sr_ref, sr_p1), f"Mismatch at rounds={r}"
    print("[PASS] sha256_sr_main matches src.sha256_ref bit-for-bit across R=4,8,16,20,64", flush=True)


def test_control_and_features() -> None:
    cfg = make_control_config(seed=20261125)
    assert cfg.sigma0_upper != STD_SIGMA0_UPPER
    assert cfg.sigma1_upper != STD_SIGMA1_UPPER
    assert cfg.sigma0_lower != STD_SIGMA0_LOWER
    assert cfg.sigma1_lower != STD_SIGMA1_LOWER
    assert len(set(cfg.sigma0_upper)) == 3
    assert len(set(cfg.sigma1_upper)) == 3
    assert len(cfg.k_ctrl) == 64

    gen = torch.Generator(device="cpu").manual_seed(123)
    w16 = sample_uniform_w16(8, device="cpu", generator=gen)
    sr_main = sha256_sr_main(w16, rounds=8)
    sr_ctrl = sha256_sr_control(w16, rounds=8, cfg=cfg)
    assert not torch.equal(sr_main, sr_ctrl)

    phi_in = encode_multiscale_features(w16)
    phi_out = encode_multiscale_features(sr_main)
    assert phi_in.shape == (8, 896) and phi_in.dtype == torch.float32
    assert phi_out.shape == (8, 448) and phi_out.dtype == torch.float32
    assert len(IN_FEATURE_NAMES) == 896
    assert len(OUT_FEATURE_NAMES) == 448
    assert float(phi_in.min().item()) >= -1.0001 and float(phi_in.max().item()) <= 1.0001
    assert float(phi_out.min().item()) >= -1.0001 and float(phi_out.max().item()) <= 1.0001

    m_nl = DualTowerInfoNCE()
    m_lin = LinearInfoNCE()
    u1, v1, s1 = m_nl(phi_in, phi_out)
    u2, v2, s2 = m_lin(phi_in, phi_out)
    assert u1.shape == (8, 128) and v1.shape == (8, 128) and s1.dtype == torch.float32
    assert u2.shape == (8, 128) and v2.shape == (8, 128) and s2.dtype == torch.float32
    print("[PASS] control config, feature dims/ranges, and FP32 models verified", flush=True)


def test_smoke_cpu() -> None:
    enforce_fp32_determinism()
    with tempfile.TemporaryDirectory() as td:
        summary = run_sweep(
            out_dir=Path(td),
            rounds_list=[4, 8],
            steps=4,
            batch_size=32,
            eval_n=128,
            eval_pool_k=32,
            fit_n=128,
            perm_n=4,
            hidden_dim=64,
            emb_dim=16,
            num_blocks=2,
            save_weights=False,
            device_str="cpu",
        )
        p = Path(td) / "p1_sweep_summary.json"
        assert p.exists()
        loaded = json.loads(p.read_text(encoding="utf-8"))
        assert "4" in loaded["rounds"] and "8" in loaded["rounds"]
        assert loaded["rounds"]["4"]["distillation"] is not None
        assert "z_ctrl" in loaded["rounds"]["8"]["comparison"]
    print("[PASS] CPU end-to-end smoke sweep verified", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    test_sha256_equivalence_if_available()
    test_control_and_features()
    test_smoke_cpu()
