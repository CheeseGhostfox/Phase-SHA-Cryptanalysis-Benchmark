"""Fast CPU-only unit tests and R=2 mini smoke test for P7."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from cipher_ops import (
    assert_fp32_locked,
    enforce_fp32_lock,
    feistel32_strong_bijection,
    make_main_spec,
    run_rounds_ae,
)
from ortho_basis import (
    BatchedCayleyOrthogonal,
    DualGroup32Basis,
    build_256_init_bank,
    build_haar_ortho,
    check_orthogonal_conservation,
)
from src.sha256_ref import sample_random_w16, sha256_trace


def test_sha256_ref_match() -> None:
    w16 = sample_random_w16(128, mode="fixed_len_bytes", msg_bytes_len=20, device="cpu")
    spec = make_main_spec()
    for r in (2, 4, 8, 12):
        trace = sha256_trace(w16, rounds=r)
        a_r, e_r = run_rounds_ae(w16, rounds=r, spec=spec)
        assert torch.equal(a_r, trace.A_full[:, r + 3]), f"A_{r} mismatch"
        assert torch.equal(e_r, trace.E_full[:, r + 3]), f"E_{r} mismatch"
    print("[PASS] sha256_ref exact parity across R={2,4,8,12}")


def test_feistel32_bijection() -> None:
    x = torch.arange(0, 4096, dtype=torch.int64)
    y = feistel32_strong_bijection(x)
    assert y.dtype == torch.int64
    assert torch.all((y >= 0) & (y < (1 << 32)))
    # Check distinct inputs map to distinct outputs
    assert torch.unique(y).numel() == 4096
    print("[PASS] feistel32_strong_bijection injectivity & range")


def test_dual_group_93_orthonormality() -> None:
    basis = DualGroup32Basis()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(2026)
    errs = []
    for n in (4096, 65536):
        x = torch.randint(0, 1 << 32, (n,), dtype=torch.int64, generator=gen)
        phi = basis(x)
        assert phi.dtype == torch.float32
        assert phi.shape == (n, 93)
        gram = (phi.t() @ phi) / float(n)
        err = float(torch.linalg.matrix_norm(gram - torch.eye(93), ord="fro").item())
        max_abs = float(torch.max(torch.abs(gram - torch.eye(93))).item())
        errs.append((err, max_abs))
    # Expected Frobenius error for 93x93 empirical Gram is ~ 93 / sqrt(N)
    assert errs[1][0] < errs[0][0] * 0.35, f"O(1/sqrt(N)) convergence failed: {errs}"
    assert errs[1][0] < 0.45 and errs[1][1] < 0.03, (
        f"Orthonormality error too large at N=65536: {errs[1]}"
    )
    print(
        f"[PASS] DualGroup32Basis 93-dim orthonormality "
        f"N=4096 fro={errs[0][0]:.4f}, N=65536 fro={errs[1][0]:.4f} (theory 93/256=0.3633, max_abs={errs[1][1]:.4f})"
    )


def test_bases_and_cayley_conservation() -> None:
    bank = build_256_init_bank("cpu")
    haar = build_haar_ortho(256, seed=77, device="cpu")
    for name, q in {**bank, "haar": haar}.items():
        oe, de, _ = check_orthogonal_conservation(q)
        assert oe < 1e-4 and de < 1e-4, f"Init {name} failed ortho/det: {oe}, {de}"

    q_stack = torch.stack(list(bank.values()), dim=0)
    cayley = BatchedCayleyOrthogonal(q_stack, init_std=0.2, seed=99)
    u_batch = cayley()
    g = torch.randn(256, 64, dtype=torch.float32)
    g = g / g.norm(dim=0, keepdim=True)
    for idx, name in enumerate(bank.keys()):
        oe, de, ne = check_orthogonal_conservation(u_batch[idx], g)
        assert oe < 1e-4 and de < 1e-4 and ne < 1e-4, (
            f"Cayley {name} failed conservation: ortho={oe}, det={de}, norm={ne}"
        )
    print("[PASS] All 4 algebraic bases + Haar + Cayley FP32 conservation < 1e-4")


def test_smoke_pipeline() -> None:
    this_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable,
            "-u",
            str(this_dir / "run_p7_sweep.py"),
            "--mode",
            "smoke",
            "--out-dir",
            tmpdir,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        summary = json.loads((Path(tmpdir) / "p7_sweep_summary.json").read_text(encoding="utf-8"))
        distill = json.loads((Path(tmpdir) / "p7_distillation.json").read_text(encoding="utf-8"))
        meta = json.loads((Path(tmpdir) / "p7_run_meta.json").read_text(encoding="utf-8"))

        assert summary["meta"]["fp32_locked"] is True
        assert summary["meta"]["tf32_disabled"] is True
        assert summary["meta"]["complex_disabled"] is True
        assert summary["meta"]["conservation_passed"] is True
        assert "2" in summary["probe1_bit_slice"]
        assert "2" in summary["probe2_io_coupling"]
        assert "2" in summary["probe3_operator_transfer"]
        assert "2" in distill["probe1_bit_slice"]
        assert meta["conservation"]["passed"] is True
        print(f"[PASS] Smoke pipeline output verified (wall={summary['meta']['total_wall_sec']:.2f}s)")


def main() -> None:
    enforce_fp32_lock()
    assert_fp32_locked()
    test_sha256_ref_match()
    test_feistel32_bijection()
    test_dual_group_93_orthonormality()
    test_bases_and_cayley_conservation()
    test_smoke_pipeline()
    print("[ALL CPU TESTS PASSED]")


if __name__ == "__main__":
    main()
