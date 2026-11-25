"""Fast CPU-only unit & smoke test for P4 (CUDA_VISIBLE_DEVICES='')."""

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tempfile
from pathlib import Path

import numpy as np
import torch

from src.sha256_ref import sha256_trace
from v11_discovery.p4_koopman_distinguisher.eval_distill import compute_roc_auc
from v11_discovery.p4_koopman_distinguisher.models import KoopmanCrystalNet3D
from v11_discovery.p4_koopman_distinguisher.run_p4_sweep import run_sweep
from v11_discovery.p4_koopman_distinguisher.sha256_tuples import (
    DEFAULT_CONTROL_CONFIG,
    FAMILIES,
    FAMILY_C_IN,
    FAMILY_K,
    encode_crystal_features,
    enforce_fp32_determinism,
    generate_tuple_batch,
    sha256_state_sr,
)


def test_sha256_exact_alignment() -> None:
    gen = torch.Generator(device="cpu").manual_seed(12345)
    w16 = torch.randint(0, 1 << 32, (64, 16), dtype=torch.int64, generator=gen)
    for r in (4, 6, 8, 16, 20):
        ref_sr = sha256_trace(w16, rounds=r).S_R
        p4_sr = sha256_state_sr(w16, rounds=r, control=False)
        assert torch.equal(ref_sr, p4_sr), f"Mismatch at rounds={r}"
    print("[PASS] test_sha256_exact_alignment")


def test_control_config_isomorphism() -> None:
    offs = DEFAULT_CONTROL_CONFIG["rotr_offsets"]
    assert len(set(offs["Sigma0"])) == 3 and set(offs["Sigma0"]) != {2, 13, 22}
    assert len(set(offs["Sigma1"])) == 3 and set(offs["Sigma1"]) != {6, 11, 25}
    assert len(set(offs["sigma0_rotr"])) == 2 and set(offs["sigma0_rotr"]) != {7, 18}
    assert len(set(offs["sigma1_rotr"])) == 2 and set(offs["sigma1_rotr"]) != {17, 19}
    assert offs["sigma0_shr"] != 3 and offs["sigma1_shr"] != 10
    assert len(DEFAULT_CONTROL_CONFIG["k_ctrl"]) == 64
    print("[PASS] test_control_config_isomorphism")


def test_tuples_features_and_models() -> None:
    enforce_fp32_determinism()
    gen = torch.Generator(device="cpu").manual_seed(999)
    for fam in FAMILIES:
        k = FAMILY_K[fam]
        c_in = FAMILY_C_IN[fam]
        sr_t, c_t, lbl = generate_tuple_batch(fam, batch_size=16, rounds=4, device=torch.device("cpu"), generator=gen)
        assert sr_t.shape == (16, k, 8)
        assert c_t.shape == (16, k, 4, 2)
        assert lbl.shape == (16,) and lbl[:8].sum() == 8.0 and lbl[8:].sum() == 0.0

        psi = encode_crystal_features(c_t, fam)
        assert psi.shape == (16, c_in, 4, 2, 32)
        assert psi.dtype == torch.float32
        assert torch.all((psi == 1.0) | (psi == -1.0))

        model = KoopmanCrystalNet3D(c_in=c_in)
        out = model(psi)
        assert out.shape == (16,) and out.dtype == torch.float32
        out.sum().backward()
    print("[PASS] test_tuples_features_and_models")


def test_roc_auc_exact_with_ties() -> None:
    rng = np.random.default_rng(7)
    y = np.array([1, 0, 1, 0, 1, 0, 0, 1] * 8, dtype=np.float64)
    # Scores with deliberate ties
    s = rng.integers(0, 5, size=len(y)).astype(np.float64)
    auc_fast = compute_roc_auc(y, s)

    pos = s[y > 0.5]
    neg = s[y <= 0.5]
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    auc_brute = wins / float(len(pos) * len(neg))
    assert abs(auc_fast - auc_brute) < 1e-12, f"{auc_fast} vs {auc_brute}"
    print("[PASS] test_roc_auc_exact_with_ties")


def test_mini_sweep_smoke() -> None:
    with tempfile.TemporaryDirectory() as td:
        summary = run_sweep(
            out_dir=Path(td),
            rounds_list=[4],
            families_list=["F1_xor2"],
            steps=5,
            r4_steps=5,
            extend_steps=0,
            eval_n=256,
            pool_k=64,
            cal_n=256,
            base_seed=42,
            force_cpu=True,
        )
        res = summary["results"]["4"]["F1_xor2"]
        assert "main" in res and "control" in res and "zero_param_baseline" in res
        assert "comparison" in res and res["distillation"] is not None
        assert len(res["distillation"]["walsh_top16"]) == 16
        assert len(res["distillation"]["two_adic_top12"]) == 12
        assert (Path(td) / "p4_sweep_summary.json").exists()
    print("[PASS] test_mini_sweep_smoke")


if __name__ == "__main__":
    test_sha256_exact_alignment()
    test_control_config_isomorphism()
    test_tuples_features_and_models()
    test_roc_auc_exact_with_ties()
    test_mini_sweep_smoke()
    print("[ALL P4 CPU SMOKE TESTS PASSED]")
