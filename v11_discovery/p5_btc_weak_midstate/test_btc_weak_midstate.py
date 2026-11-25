"""
P5 本地 CPU 秒级单元测试与冒烟验证（强制 CUDA_VISIBLE_DEVICES=""）。
验证：
1. 与 Python hashlib.sha256 对标准 80 字节 BTC 区块头（Block 1 Midstate + Block 2 + Double-SHA256）逐比特一致；
2. clz256 / ctz_btc256 / popcnt32 与 Python 原生整数参考一致；
3. soft_add32 与可微软前向在纯 FP32 下可反向传播且二进制输入精确一致；
4. N_train 与 N_holdout 零重叠隔离、CTRL_RAND_ROTR_K 与标准 SHA-256 参数全异；
5. R=8 与 double64 端到端 CPU 极小样本冒烟与 JSON 落盘。
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import btc_weak_midstate as p5


def test_exact_btc_double_sha256_against_hashlib() -> None:
    assert not torch.cuda.is_available(), "本地测试严禁启用 CUDA！"
    p5.enforce_fp32_strict()

    # 构造真实比特币创世区块头 80 字节（小端字段拼装为 80 字节大端字节流输入 SHA-256）
    version = struct.pack("<I", 1)
    prev_block = bytes(32)
    merkle_root = bytes.fromhex(
        "4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b"
    )[::-1]
    timestamp = struct.pack("<I", 1231006505)
    bits = struct.pack("<I", 0x1D00FFFF)
    nonce = struct.pack("<I", 2083236893)
    header80 = version + prev_block + merkle_root + timestamp + bits + nonce
    assert len(header80) == 80

    # hashlib 双重 SHA-256 参考值
    h1_bytes = hashlib.sha256(header80).digest()
    h2_bytes = hashlib.sha256(h1_bytes).digest()
    expected_btc_hex = h2_bytes[::-1].hex()
    assert expected_btc_hex == "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"

    # 用 p5 计算 Block 1 Midstate 与 Block 2 Double-SHA256
    blk1_words = [int.from_bytes(header80[4 * i : 4 * (i + 1)], "big") for i in range(16)]
    blk2_words = [int.from_bytes(header80[64 + 4 * i : 64 + 4 * (i + 1)], "big") for i in range(4)]

    main_var = p5.make_main_variant()
    iv_t = torch.tensor(main_var.iv, dtype=torch.int64)
    blk1_t = torch.tensor([blk1_words], dtype=torch.int64)
    midstate = p5.sha256_compress_exact(iv_t, blk1_t, rounds=64, variant=main_var)["H"][0]

    tpl = torch.cat([midstate, torch.tensor(blk2_words[:3], dtype=torch.int64)])
    nonce_t = torch.tensor([blk2_words[3]], dtype=torch.int64)

    out_h1 = p5.btc_forward_exact(tpl, nonce_t, "64", main_var)["H"][0]
    h1_calc = b"".join(int(x).to_bytes(4, "big") for x in out_h1.tolist())
    assert h1_calc == h1_bytes, "Block 2 单层 SHA-256 结果不一致！"

    out_h2 = p5.btc_forward_exact(tpl, nonce_t, "double64", main_var)["H"][0]
    h2_calc = b"".join(int(x).to_bytes(4, "big") for x in out_h2.tolist())
    assert h2_calc == h2_bytes, "Double-SHA256 结果不一致！"

    # 创世区块哈希有 43 个 BTC 小端前导零（5 个 0x00 字节 + 0x19 -> 40 + 3 = 43 bits）
    lz_btc = int(p5.ctz_btc256(out_h2.unsqueeze(0)).item())
    le_int = int.from_bytes(h2_bytes, "little")
    assert lz_btc == 256 - le_int.bit_length() == 43


def test_bit_metrics_and_soft_adder() -> None:
    rng = np.random.default_rng(42)
    words = rng.integers(0, 2**32, size=(16, 8), dtype=np.uint64).astype(np.int64)
    words[0, 0] = 0
    words[0, 1] = 0x00801234
    words[1, 7] = 0x12340000
    t = torch.from_numpy(words)

    # popcnt32
    pc = p5.popcnt32(t)
    for i in range(16):
        for j in range(8):
            assert int(pc[i, j].item()) == (int(words[i, j]) & 0xFFFFFFFF).bit_count()

    # clz256 / ctz_btc256
    clz = p5.clz256(t)
    ctz = p5.ctz_btc256(t)
    for i in range(16):
        raw = b"".join((int(words[i, j]) & 0xFFFFFFFF).to_bytes(4, "big") for j in range(8))
        be_val = int.from_bytes(raw, "big")
        le_val = int.from_bytes(raw, "little")
        assert int(clz[i].item()) == (256 - be_val.bit_length())
        assert int(ctz[i].item()) == (256 - le_val.bit_length())

    # soft_add32 exactness & FP32 gradient
    a = torch.randint(0, 2**32, (8,), dtype=torch.int64)
    b = torch.randint(0, 2**32, (8,), dtype=torch.int64)
    ab = p5._words_to_soft_bits(a, torch.device("cpu"))
    bb = p5._words_to_soft_bits(b, torch.device("cpu"))
    sb, _ = p5.soft_add32(ab, bb)
    s_words = p5._soft_bits_to_words(sb)
    assert torch.equal(s_words, (a + b) & p5.MASK32)

    alpha = torch.randn((2, 11, 32), dtype=torch.float32, requires_grad=True)
    nonces = torch.tensor([12345, 67890], dtype=torch.int64)
    out = p5.soft_btc_forward(torch.sigmoid(alpha), nonces, rounds=8, variant=p5.make_main_variant())
    loss = out["carry_post4"].mean() + out["H_bits"].mean()
    loss.backward()
    assert alpha.grad is not None and torch.isfinite(alpha.grad).all()
    assert float(alpha.grad.abs().sum().item()) > 0.0


def test_isolation_and_ctrl_variant() -> None:
    main_v = p5.make_main_variant()
    ctrl_v = p5.make_ctrl_variant(p5.SEED_CTRL)
    assert ctrl_v.S0_rot != main_v.S0_rot and len(set(ctrl_v.S0_rot)) == 3
    assert ctrl_v.S1_rot != main_v.S1_rot and len(set(ctrl_v.S1_rot)) == 3
    assert ctrl_v.s0_rot != main_v.s0_rot and ctrl_v.s0_shr != main_v.s0_shr
    assert ctrl_v.s1_rot != main_v.s1_rot and ctrl_v.s1_shr != main_v.s1_shr
    assert ctrl_v.k != main_v.k

    tr = p5.sample_train_nonces(256, np.random.default_rng(p5.SEED_TRAIN), torch.device("cpu")).numpy()
    h1 = p5.build_holdout_nonces(256, p5.SEED_HOLDOUT)
    h2 = p5.build_holdout_nonces(256, p5.SEED_HOLDOUT2, exclude=h1)
    assert tr.max() < (1 << 30) and h1.min() >= (1 << 30) and h2.min() >= (1 << 30)
    assert set(tr.tolist()).isdisjoint(set(h1.tolist()))
    assert set(h1.tolist()).isdisjoint(set(h2.tolist()))


def test_cpu_smoke_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as td:
        summary, meta = p5.run_full_experiment(
            out_dir=Path(td),
            sweep_keys=("8", "double64"),
            n_train_pool=64,
            n_holdout=128,
            n_holdout2=0,
            pop_size=8,
            generations=2,
            grad_steps=3,
            top_k=4,
            chunk_size=64,
            cpu_only=True,
        )
        assert (Path(td) / "p5_sweep_summary.json").exists()
        assert (Path(td) / "p5_run_meta.json").exists()
        assert set(summary["sweeps"].keys()) == {"8", "double64"}
        assert meta["device"] == "cpu"


if __name__ == "__main__":
    test_exact_btc_double_sha256_against_hashlib()
    print("[PASS] test_exact_btc_double_sha256_against_hashlib")
    test_bit_metrics_and_soft_adder()
    print("[PASS] test_bit_metrics_and_soft_adder")
    test_isolation_and_ctrl_variant()
    print("[PASS] test_isolation_and_ctrl_variant")
    test_cpu_smoke_end_to_end()
    print("[PASS] test_cpu_smoke_end_to_end")
