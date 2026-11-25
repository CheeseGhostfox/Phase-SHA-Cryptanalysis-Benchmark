"""
P12 A1 硬基线校准模块：
精确计算并保存 `results/a1_calibration.json`，严格按 SPEC §1.3 & @v12-auditor #99 要求：
1. 对拍比特币真实创世区块头（Block #0 Genesis）及 256 个随机 80-byte 区块头，
   校验与 sha256_ref.py 及 Python hashlib.sha256 严格逐比特对齐（xor_diff == 0）；
2. 用同一张固定 GE 权重表同时输出：
   - `BASE_NO_ASICBOOST`（常规单 Midstate 扫 Nonce，默认作为 1.000x 基准）；
   - `BASE_ASICBOOST_OVERT`（k in {4, 8, 16} 路 BIP320 nVersion 滚动，输出
     N_nonce=2^32 全均摊口径、N_nonce_per_midstate 对称均摊口径、以及经典 4-块对比口径）；
   - `BASE_ASICBOOST_COVERT`（k in {4, 8, 16} 路 Merkle 尾 32-bit 碰撞，显式计入生日碰撞筛选代价）。
"""

from __future__ import annotations

import json
import os
import struct
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

if sys.platform == "win32" or "--cpu_only" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch

from btc_double_sha_engine import (
    GE_WEIGHTS_DICT,
    MASK32,
    AmortizedCostReport,
    SHA256Variant,
    btc_double_sha256_hashlib_check,
    btc_double_sha256_trace,
    btc_header_bytes_to_blocks,
    compute_exact_btc_amortized_report,
    digest_words_to_bytes,
    enforce_fp32_strict,
    make_main_variant,
    make_spectral_aligned_ctrl_variant,
)

# 真实比特币创世区块头 (Block #0, 2009-01-03)
BTC_GENESIS_HEADER_HEX: str = (
    "01000000"
    "0000000000000000000000000000000000000000000000000000000000000000"
    "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
    "29ab5f49"
    "ffff001d"
    "1dac2b7c"
)
BTC_GENESIS_DISPLAY_HASH_HEX: str = (
    "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
)


def make_synthetic_btc_header(
    version: int = 0x20000000,
    nonce: int = 0x12345678,
    seed: int = 20261125,
) -> bytes:
    rng = np.random.default_rng(seed)
    prev_hash = bytes(rng.integers(0, 256, size=32, dtype=np.uint8).tolist())
    merkle_root = bytes(rng.integers(0, 256, size=32, dtype=np.uint8).tolist())
    n_time = 1732500000
    n_bits = 0x1703255B
    return (
        struct.pack("<I", version & MASK32)
        + prev_hash
        + merkle_root
        + struct.pack("<III", n_time, n_bits, nonce & MASK32)
    )


def verify_bit_exactness(device: torch.device | str = "cpu", n_random: int = 256) -> Dict[str, Any]:
    """核验 Genesis 区块头与 n_random 个随机 BTC 区块头 Double-SHA256 零误差。"""
    genesis_bytes = bytes.fromhex(BTC_GENESIS_HEADER_HEX)
    b1_g, b2_g = btc_header_bytes_to_blocks(genesis_bytes, device=device)
    tr_g = btc_double_sha256_trace(b1_g, b2_g, record_gates=False)
    digest_bytes = digest_words_to_bytes(tr_g.final_hash[0])
    display_hash = digest_bytes[::-1].hex()
    genesis_ref = btc_double_sha256_hashlib_check(genesis_bytes)
    genesis_xor = int((tr_g.final_hash[0].cpu() ^ genesis_ref).sum().item())
    assert display_hash == BTC_GENESIS_DISPLAY_HASH_HEX, (display_hash, BTC_GENESIS_DISPLAY_HASH_HEX)
    assert genesis_xor == 0, genesis_xor

    headers = [
        make_synthetic_btc_header(version=0x20000000 + i * 17, nonce=i * 99991, seed=200 + i)
        for i in range(n_random)
    ]
    b1, b2 = btc_header_bytes_to_blocks(headers, device=device)
    tr = btc_double_sha256_trace(b1, b2, record_gates=False)
    max_xor = 0
    for i, hdr in enumerate(headers):
        ref_h = btc_double_sha256_hashlib_check(hdr)
        diff = int((tr.final_hash[i].cpu() ^ ref_h).sum().item())
        max_xor |= diff
    assert max_xor == 0, f"Random headers xor_diff={max_xor}"

    return {
        "btc_genesis_xor_diff": genesis_xor,
        "btc_genesis_hash_matched": True,
        "random_headers_tested": n_random,
        "max_xor_diff": max_xor,
    }


def run_a1_calibration_for_variant(
    variant: SHA256Variant,
    device: torch.device | str = "cpu",
    n_nonces_eval: int = 64,
    k_ways: tuple[int, ...] = (4, 8, 16),
) -> Dict[str, Any]:
    """
    对给定 SHA256Variant 执行完整 A1 硬基线校准：
    1. BASE_NO_ASICBOOST (1 midstate x n_nonces_eval nonces)
    2. BASE_ASICBOOST_OVERT (k midstates x n_nonces_eval nonces):
       - inner_loop_2p32_amort: Chunk1 按 2^32 Nonce 均摊 (矿机真实内循环稳态)
       - symmetric_batch_amort: Chunk1 按本批各自 n_nonces_eval 均摊 (对称小批对比)
       - unamortized_1nonce_4block: 单 Nonce 全重算 Chunk1 口径 (对照 Hanke 外/内循环分析)
    3. BASE_ASICBOOST_COVERT (k midstates): 额外显式计入 Merkle 尾 32-bit 碰撞边际筛选成本。
    """
    base_hdr = make_synthetic_btc_header(seed=20261125)
    nonces = [0x10000000 + n * 7919 for n in range(n_nonces_eval)]

    # 1. BASE_NO_ASICBOOST (k=1 x n_nonces_eval)
    headers_noab = [base_hdr[:76] + struct.pack("<I", n & MASK32) for n in nonces]
    b1_no, b2_no = btc_header_bytes_to_blocks(headers_noab, device=device)
    with torch.no_grad():
        tr_no = btc_double_sha256_trace(b1_no, b2_no, variant=variant, record_gates=True)

    rep_noab_2p32 = compute_exact_btc_amortized_report(
        tr_no,
        label="BASE_NO_ASICBOOST_2p32",
        variant=variant,
        amortize_chunk1_over_nonces=1 << 32,
    )
    naive_2p32_ge = rep_noab_2p32.amortized_ge_per_hash

    rep_noab_sym = compute_exact_btc_amortized_report(
        tr_no,
        label="BASE_NO_ASICBOOST_SYM",
        variant=variant,
        amortize_chunk1_over_nonces=n_nonces_eval,
    )
    naive_sym_ge = rep_noab_sym.amortized_ge_per_hash

    rep_noab_1nonce = compute_exact_btc_amortized_report(
        tr_no,
        label="BASE_NO_ASICBOOST_1NONCE",
        variant=variant,
        amortize_chunk1_over_nonces=1,
    )
    naive_1nonce_ge = rep_noab_1nonce.amortized_ge_per_hash

    # 单次完整 64 轮 Chunk1 成本（用于 Covert Merkle 碰撞筛选计费）
    single_chunk1_ge = rep_noab_1nonce.chunk1_ge_per_hash

    overt_records: List[Dict[str, Any]] = []
    covert_records: List[Dict[str, Any]] = []

    for k in k_ways:
        headers_k: List[bytes] = []
        for idx in range(k):
            version_val = (0x20000000 | ((idx & 0xFFFF) << 13)) & MASK32
            for n in nonces:
                headers_k.append(struct.pack("<I", version_val) + base_hdr[4:76] + struct.pack("<I", n & MASK32))

        b1_k, b2_k = btc_header_bytes_to_blocks(headers_k, device=device)
        with torch.no_grad():
            tr_k = btc_double_sha256_trace(b1_k, b2_k, variant=variant, record_gates=True)

        # (a) 2^32 Nonce 全均摊（真实矿机内循环口径）
        r_2p32 = compute_exact_btc_amortized_report(
            tr_k,
            label=f"BASE_ASICBOOST_OVERT_k{k}_2p32",
            variant=variant,
            naive_baseline_ge_per_hash=naive_2p32_ge,
            amortize_chunk1_over_nonces=1 << 32,
        )
        # (b) 对称 n_nonces_eval 均摊（避免隐藏 Chunk1 成本）
        r_sym = compute_exact_btc_amortized_report(
            tr_k,
            label=f"BASE_ASICBOOST_OVERT_k{k}_SYM",
            variant=variant,
            naive_baseline_ge_per_hash=naive_sym_ge,
            amortize_chunk1_over_nonces=n_nonces_eval,
        )
        # (c) 单 Nonce / 4-块对比口径（Chunk1 每 midstate 计 1x）
        r_1n = compute_exact_btc_amortized_report(
            tr_k,
            label=f"BASE_ASICBOOST_OVERT_k{k}_1NONCE",
            variant=variant,
            naive_baseline_ge_per_hash=naive_1nonce_ge,
            amortize_chunk1_over_nonces=1,
        )

        overt_records.append(
            {
                "k": k,
                "inner_loop_2p32": asdict(r_2p32),
                "symmetric_nonce_batch": asdict(r_sym),
                "unamortized_1nonce_full_chunk1": asdict(r_1n),
            }
        )

        # Covert AsicBoost：k-way Merkle 尾 32-bit 碰撞每寻找一个 k-collision 需约 2^((k-1)/k * 32) 次 Merkle 叶子，
        # 按经典 k-way 生日/多碰撞表外循环摊还至 2^32 nonces（例如 k=4 约 ~2^24 次 SHA-256，摊至 2^32 nonce ~0.4% 开销）
        # 对于 k=2^(d)，k-way 32-bit 碰撞期望尝试 ~ 2^(32*(k-1)/k) / k，此处保守按 k * 2^16 级双块 Merkle 重算计 setup
        merkle_search_hashes = float(k) * (2.0 ** (32.0 * (k - 1) / k)) if k <= 4 else float(k) * (2.0 ** 24)
        covert_setup_ge = merkle_search_hashes * single_chunk1_ge
        r_cov = compute_exact_btc_amortized_report(
            tr_k,
            label=f"BASE_ASICBOOST_COVERT_k{k}_2p32",
            variant=variant,
            naive_baseline_ge_per_hash=naive_2p32_ge,
            amortize_chunk1_over_nonces=1 << 32,
            setup_ge_total=covert_setup_ge,
        )
        covert_records.append({"k": k, "covert_2p32": asdict(r_cov)})

    return {
        "variant": variant.name,
        "n_nonces_eval": n_nonces_eval,
        "BASE_NO_ASICBOOST": {
            "inner_loop_2p32": asdict(rep_noab_2p32),
            "symmetric_nonce_batch": asdict(rep_noab_sym),
            "unamortized_1nonce_full_chunk1": asdict(rep_noab_1nonce),
        },
        "BASE_ASICBOOST_OVERT": overt_records,
        "BASE_ASICBOOST_COVERT": covert_records,
    }


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    fp32_info = enforce_fp32_strict()
    bit_check = verify_bit_exactness(device="cpu", n_random=256)
    main_var = make_main_variant()
    ctrl_var = make_spectral_aligned_ctrl_variant(seed=20261125, min_gap=4, name="CTRL_RAND_ROTR_K_S0")

    main_res = run_a1_calibration_for_variant(main_var, device="cpu", n_nonces_eval=64)
    ctrl_res = run_a1_calibration_for_variant(ctrl_var, device="cpu", n_nonces_eval=64)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "fp32_lock": fp32_info,
        "bit_exact_check": bit_check,
        "ge_weights": GE_WEIGHTS_DICT,
        "baselines_main": main_res,
        "baselines_ctrl_s0": ctrl_res,
    }
    out_path = out_dir / "a1_calibration.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    noab = main_res["BASE_NO_ASICBOOST"]["inner_loop_2p32"]
    print(f"[A1] BASE_NO_ASICBOOST (2^32 amort): {noab['amortized_ge_per_hash']:.1f} GE/hash", flush=True)
    for rec in main_res["BASE_ASICBOOST_OVERT"]:
        k = rec["k"]
        r2 = rec["inner_loop_2p32"]
        rs = rec["symmetric_nonce_batch"]
        print(
            f"[A1] Overt k={k:2d} | 2^32 saving={r2['saving_pct_vs_noab']:5.2f}% ({r2['amortized_ge_per_hash']:.1f} GE) | "
            f"sym(N=64) saving={rs['saving_pct_vs_noab']:5.2f}%",
            flush=True,
        )
    print(f"[A1] Saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
