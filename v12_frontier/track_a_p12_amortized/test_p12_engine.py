"""
P12 第一步单元测试（CPU 轻量校验，CUDA_VISIBLE_DEVICES=""）：
1. 纯 IEEE-754 FP32 (allow_tf32=False) 锁定；
2. sha256_ref.py 逐比特对拍 (xor_diff == 0)；
3. 真实 BTC 创世区块头 + 256 个随机 BTC 80-byte Header Double-SHA256 逐比特对拍 (xor_diff == 0)；
4. 经典 AsicBoost A1 校准单调性与 GE 守恒校验；
5. 谱隙对齐随机对照变体参数合规性校验。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sha256_ref  # noqa: E402
from btc_double_sha_engine import (  # noqa: E402
    _cyclic_min_gap,
    enforce_fp32_strict,
    make_main_variant,
    make_spectral_aligned_ctrl_variant,
    sha256_compress_trace,
)
from classic_asicboost_calib import (  # noqa: E402
    run_a1_calibration_for_variant,
    verify_bit_exactness,
)


def test_fp32_lock() -> None:
    info = enforce_fp32_strict()
    assert info["default_dtype"] == "torch.float32", info
    assert info["matmul_allow_tf32"] is False, info
    assert info["cudnn_allow_tf32"] is False, info
    print("[PASS] test_fp32_lock:", info, flush=True)


def test_sha256_ref_xor_zero() -> None:
    rng = np.random.default_rng(123456)
    batch = 256
    w16_np = rng.integers(0, 2**32, size=(batch, 16), dtype=np.uint64).astype(np.int64)
    w16 = torch.from_numpy(w16_np)

    ref_tr = sha256_ref.sha256_trace(w16, rounds=64)
    iv_t = sha256_ref.get_iv_tensor("cpu")
    p12_tr = sha256_compress_trace(iv_t, w16, rounds=64, variant=make_main_variant(), record_gates=True)

    xor_digest = int((ref_tr.digest ^ p12_tr.digest).sum().item())
    xor_a = int((ref_tr.A_full ^ p12_tr.A_full).sum().item())
    xor_e = int((ref_tr.E_full ^ p12_tr.E_full).sum().item())
    xor_w = int((ref_tr.W_full ^ p12_tr.W_full).sum().item())
    assert xor_digest == 0 and xor_a == 0 and xor_e == 0 and xor_w == 0
    print(f"[PASS] test_sha256_ref_xor_zero (batch={batch}, xor_diff=0)", flush=True)


def test_btc_genesis_and_headers_xor_zero() -> None:
    res = verify_bit_exactness(device="cpu", n_random=256)
    assert res["btc_genesis_xor_diff"] == 0
    assert res["btc_genesis_hash_matched"] is True
    assert res["max_xor_diff"] == 0
    print("[PASS] test_btc_genesis_and_headers_xor_zero:", res, flush=True)


def test_a1_calibration_and_ctrl() -> None:
    main_var = make_main_variant()
    res = run_a1_calibration_for_variant(main_var, device="cpu", n_nonces_eval=32, k_ways=(4, 8, 16))
    noab = res["BASE_NO_ASICBOOST"]["inner_loop_2p32"]
    assert abs(noab["saving_pct_vs_noab"]) < 1e-6, noab

    savings = [r["inner_loop_2p32"]["saving_pct_vs_noab"] for r in res["BASE_ASICBOOST_OVERT"]]
    assert all(savings[i] < savings[i + 1] for i in range(len(savings) - 1)), savings
    # SPEC §1.3 预期经典 AsicBoost 节省在 ~10%~21% 区间
    assert 8.0 <= savings[-1] <= 22.0, savings

    for s_idx in range(3):
        c = make_spectral_aligned_ctrl_variant(seed=20261125 + s_idx, min_gap=4)
        assert _cyclic_min_gap(c.S0_rot) >= 4
        assert _cyclic_min_gap(c.S1_rot) >= 4
        assert _cyclic_min_gap(c.s0_rot) >= 4
        assert _cyclic_min_gap(c.s1_rot) >= 4
    print(f"[PASS] test_a1_calibration_and_ctrl (k=4,8,16 savings={[round(x, 2) for x in savings]}%)", flush=True)


if __name__ == "__main__":
    test_fp32_lock()
    test_sha256_ref_xor_zero()
    test_btc_genesis_and_headers_xor_zero()
    test_a1_calibration_and_ctrl()
    print("ALL P12 STEP-1 UNIT TESTS PASSED.", flush=True)
