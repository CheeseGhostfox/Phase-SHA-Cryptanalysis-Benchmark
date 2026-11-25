"""Throughput benchmark for Ouroboros-64 C++17 OpenMP & CUDA Crystal Engines."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from v10_ouroboros64.ouroboros_core.engine import C_PLANES, TorchCrystalEngine, get_core


def run_throughput_benchmark(
    rounds: int = 64,
    num_crystals: int = 16,
    cands_per_crystal: int = 20000,
    cluster_size: int = 4,
    repeats: int = 3,
) -> Dict[str, Any]:
    """Measure incremental cluster-flip delta_E throughput (ops/sec) on R=64 crystals."""
    core = get_core()
    rng = np.random.default_rng(202610)
    w0_15 = rng.integers(0, 1 << 32, size=(num_crystals, 16), dtype=np.uint32)
    cr = core.compress_exact(w0_15, rounds=rounds)

    # Inject random internal defects (boundaries untouched/welded)
    w_cor = cr.W.copy()
    s_cor = cr.S.copy()
    c_cor = cr.C.copy()
    w_cor[:, 4] ^= np.uint32(0x00010001)
    s_cor[:, 20:24, 0] ^= np.uint32(0x12345678)
    c_cor[:, 18, 4] ^= np.uint32(0x00000004)

    total_cands = num_crystals * cands_per_crystal
    clusters = np.empty((total_cands, cluster_size, 4), dtype=np.int32)
    clusters[..., 0] = rng.integers(0, 3, size=(total_cands, cluster_size), dtype=np.int32)
    clusters[..., 1] = rng.integers(1, rounds, size=(total_cands, cluster_size), dtype=np.int32)
    clusters[..., 2] = np.where(
        clusters[..., 0] == 1,
        rng.integers(0, 8, size=(total_cands, cluster_size), dtype=np.int32),
        np.where(
            clusters[..., 0] == 2,
            rng.integers(0, C_PLANES, size=(total_cands, cluster_size), dtype=np.int32),
            0,
        ),
    )
    clusters[..., 3] = rng.integers(0, 32, size=(total_cands, cluster_size), dtype=np.int32)

    # Warmup
    core.eval_cluster_flips_batch(cr.H_target, w_cor, s_cor, c_cor, clusters[:num_crystals * 64], rounds=rounds)

    best_ops_sec = 0.0
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        dE, _ = core.eval_cluster_flips_batch(
            cr.H_target, w_cor, s_cor, c_cor, clusters, rounds=rounds
        )
        dt = time.perf_counter() - t0
        times.append(dt)
        best_ops_sec = max(best_ops_sec, float(total_cands) / max(1e-9, dt))
        _ = dE[0]

    # CUDA full crystal batch eval throughput (if available)
    gpu_eval_ops = 0.0
    gpu_name = "CPU-only"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        tc = TorchCrystalEngine("cuda")
        b_gpu = 2048
        ht_g = torch.from_numpy(np.repeat(cr.H_target[:1], b_gpu, axis=0).astype(np.int64)).cuda()
        wt_g = torch.from_numpy(np.repeat(w_cor[:1], b_gpu, axis=0).astype(np.int64)).cuda()
        st_g = torch.from_numpy(np.repeat(s_cor[:1], b_gpu, axis=0).astype(np.int64)).cuda()
        ct_g = torch.from_numpy(np.repeat(c_cor[:1], b_gpu, axis=0).astype(np.int64)).cuda()
        tc.eval_crystal_batch(ht_g, wt_g, st_g, ct_g, rounds=rounds)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            tc.eval_crystal_batch(ht_g, wt_g, st_g, ct_g, rounds=rounds)
        torch.cuda.synchronize()
        dt_g = time.perf_counter() - t0
        gpu_eval_ops = float(b_gpu * 20) / max(1e-9, dt_g)

    return {
        "rounds": rounds,
        "threads": core.max_threads,
        "total_cands_per_call": total_cands,
        "cluster_size": cluster_size,
        "cpp_best_time_sec": round(min(times), 4),
        "cpp_cluster_ops_per_sec": round(best_ops_sec, 1),
        "cuda_device": gpu_name,
        "cuda_full_crystal_ops_per_sec": round(gpu_eval_ops, 1),
    }


if __name__ == "__main__":
    res = run_throughput_benchmark()
    print("[Throughput Benchmark]", json.dumps(res, ensure_ascii=False, indent=2), flush=True)
