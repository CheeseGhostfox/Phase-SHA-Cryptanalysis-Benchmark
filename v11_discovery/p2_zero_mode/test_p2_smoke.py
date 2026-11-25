"""P2 时空晶体等零面零模波包本地 CPU 秒级单测与冒烟验证。"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v10_ouroboros64.ouroboros_core.engine import TorchCrystalEngine, enforce_fp32_determinism
from v11_discovery.p2_zero_mode.p2_core import (
    ParametricCrystalEngine,
    free_word_count,
    leading_zeros_u32_np,
    leading_zeros_u32_torch,
    make_random_control_cipher_config,
    make_standard_cipher_config,
    patch_model_for_cipher,
    sample_carrier_w0_np,
    verify_sha256_rounds_np,
)
from v11_discovery.p2_zero_mode.p2_proposer import (
    ZeroModeCrystalProposer,
    deduplicate_candidates_per_base,
)


def test_exact_sha256_equivalence() -> None:
    """校验 ParametricCrystalEngine 与 hashlib.sha256 (R=64)、v10 TorchCrystalEngine 与纯 NumPy 核验 1:1 一致。"""
    enforce_fp32_determinism()
    rng = np.random.default_rng(12345)
    w0_np = rng.integers(0, 2**32, size=(8, 16), dtype=np.uint64).astype(np.uint32)
    w0_t = torch.from_numpy(w0_np.astype(np.int64))

    std_eng = ParametricCrystalEngine(make_standard_cipher_config(), device="cpu")
    v10_eng = TorchCrystalEngine(device="cpu")

    # 1) R=64 对齐 Python hashlib.sha256 单块压缩（含标准 SHA-256 末尾单块填充对比）
    for i in range(4):
        msg = b"p2_zero_mode_test_" + bytes([i])
        pad_len = (56 - (len(msg) + 1) % 64) % 64
        block = msg + b"\x80" + b"\x00" * pad_len + struct.pack(">Q", len(msg) * 8)
        assert len(block) == 64
        words = np.array(struct.unpack(">16I", block), dtype=np.uint32)
        h_std = std_eng.compress_exact(torch.from_numpy(words.astype(np.int64)), rounds=64)["H"][0].numpy().astype(np.uint32)
        digest_words = np.array(struct.unpack(">8I", hashlib.sha256(msg).digest()), dtype=np.uint32)
        assert np.array_equal(h_std, digest_words), f"R=64 hashlib mismatch: {h_std} vs {digest_words}"

    # 2) R ∈ {4, 8, 16, 20} 与 v10 TorchCrystalEngine & NumPy 核验完全一致
    for r in (4, 8, 16, 20):
        c_p2 = std_eng.compress_exact(w0_t, rounds=r)
        c_v10 = v10_eng.compress_exact(w0_t, rounds=r)
        h_np = verify_sha256_rounds_np(w0_np, rounds=r)
        for key in ("H", "W", "S", "C"):
            assert torch.equal(c_p2[key], c_v10[key]), f"R={r} {key} mismatch vs v10"
        assert np.array_equal(c_p2["H"].numpy().astype(np.uint32), h_np), f"R={r} H mismatch vs NumPy"

        h_star = c_p2["H"].clone()
        h_star[:, 0] ^= 1 << 29
        e_p2 = std_eng.eval_crystal_batch(h_star, c_p2["W"], c_p2["S"], c_p2["C"], rounds=r)
        e_v10 = v10_eng.eval_crystal_batch(h_star, c_v10["W"], c_v10["S"], c_v10["C"], rounds=r)
        assert torch.equal(e_p2["E"], e_v10["E"]), f"R={r} eval E mismatch"
        assert torch.all(e_p2["E"] >= 1) and torch.all(e_p2["dW"] == 0) and torch.all(e_p2["dC"] == 0)


def test_control_and_carrier_invariants() -> None:
    """校验随机 ROTR/K 对照组、BTC320 填充锁与去重/前导零函数。"""
    ctrl = make_random_control_cipher_config(seed=999)
    assert len(set(ctrl.sigma0_rots)) == 3
    assert len(set(ctrl.sigma1_rots)) == 3
    assert len({*ctrl.lower_sigma0_rots, ctrl.lower_sigma0_shr}) == 3
    assert len({*ctrl.lower_sigma1_rots, ctrl.lower_sigma1_shr}) == 3

    ctrl_eng = ParametricCrystalEngine(ctrl, device="cpu")
    rng = np.random.default_rng(77)
    w_btc = sample_carrier_w0_np(rng, 16, carrier="BTC320")
    assert np.all(w_btc[:, 4] == np.uint32(0x80000000))
    assert np.all(w_btc[:, 5:15] == 0)
    assert np.all(w_btc[:, 15] == np.uint32(640))
    assert free_word_count(20, "BTC320") == 4
    assert free_word_count(10, "STD") == 10
    assert free_word_count(20, "STD") == 16

    h_ctrl = ctrl_eng.compress_exact(torch.from_numpy(w_btc.astype(np.int64)), rounds=12)["H"]
    h_ctrl_np = verify_sha256_rounds_np(w_btc, rounds=12, cipher_cfg=ctrl)
    assert np.array_equal(h_ctrl.numpy().astype(np.uint32), h_ctrl_np)

    proposer = ZeroModeCrystalProposer(d_node=32, n_layers=2, bond_dim=4, d_key=16, max_bits=6)
    patch_model_for_cipher(proposer.net, ctrl)
    for _, p in proposer.named_parameters():
        assert p.dtype == torch.float32

    # 测试前导零函数一致性
    test_vals = np.array([0, 1, 0x80000000, 0x00080000, 0x000FFFFF, 0xFFFFFFFF], dtype=np.uint32)
    lcz_np = leading_zeros_u32_np(test_vals)
    lcz_th = leading_zeros_u32_torch(torch.from_numpy(test_vals.astype(np.int64))).numpy()
    assert np.array_equal(lcz_np, np.array([32, 31, 0, 12, 12, 0]))
    assert np.array_equal(lcz_np, lcz_th)

    # 测试基底内去重并保障全非零
    dup_pool = np.zeros((2, 10, 4), dtype=np.uint32)
    dup_pool[0, :5, 0] = 1
    dup_pool[1, :, 1] = 2
    dedup, _ = deduplicate_candidates_per_base(dup_pool, k_target=8, w_free=4, rng=rng)
    assert dedup.shape == (2, 8, 16)
    assert np.all(np.any(dedup != 0, axis=2))
    assert np.all(dedup[:, :, 4:] == 0)
    assert np.unique(dedup[0], axis=0).shape[0] == 8
    assert np.unique(dedup[1], axis=0).shape[0] == 8


def test_end_to_end_cpu_smoke() -> None:
    """运行 run_p2_sweep.py --mode smoke 并校验产物 JSON 契约。"""
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "v11_discovery.p2_zero_mode.run_p2_sweep",
            "--mode",
            "smoke",
            "--out_dir",
            td,
        ]
        res = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=90)
        assert res.returncode == 0, f"Smoke failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

        sweep_path = Path(td) / "sweep_matrix.json"
        wp_path = Path(td) / "wavepacket_structures.json"
        assert sweep_path.exists() and wp_path.exists()

        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        wp = json.loads(wp_path.read_text(encoding="utf-8"))
        assert sweep["metadata"]["fp32_locked"] is True
        assert sweep["metadata"]["tf32_matmul"] is False
        assert len(sweep["cells"]) > 0
        assert len(wp["structures"]) > 0
        print("SMOKE_OK", len(sweep["cells"]), "cells,", len(wp["structures"]), "wavepackets")


if __name__ == "__main__":
    test_exact_sha256_equivalence()
    print("[PASS] test_exact_sha256_equivalence")
    test_control_and_carrier_invariants()
    print("[PASS] test_control_and_carrier_invariants")
    test_end_to_end_cpu_smoke()
    print("[PASS] test_end_to_end_cpu_smoke")
