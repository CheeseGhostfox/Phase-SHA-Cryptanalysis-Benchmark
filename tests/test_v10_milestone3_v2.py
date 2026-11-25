"""
Phase-SHA V10 Milestone 3-V2 单元测试：
验证原生 XOR-Gauss 编码/求解器、搜中动态神经介入 CDCL & XOR 求解器、
3 步端到端 M3-V2 训练与 N=4 快速五方配对统计显著性检验输出。
"""

import numpy as np
import pytest
import torch

from v10_ouroboros64.ouroboros_core.cnf_cdcl import (
    CryptoMiniSatXORSolver,
    make_end_free_constraints,
)
from v10_ouroboros64.ouroboros_core.engine import get_core
from v10_ouroboros64.ouroboros_mcts.dynamic_cdcl import (
    DynamicNeuralCDCLSolver,
    DynamicNeuralXORSolver,
)
from v10_ouroboros64.ouroboros_net.model import Ouroboros64Net, get_model_preset


@pytest.mark.skipif(not CryptoMiniSatXORSolver.is_available(), reason="未安装 pycryptosat")
def test_sha256_xor_cnf_exact_blind_preimage():
    core = get_core()
    cms = CryptoMiniSatXORSolver(threads=1)
    rng = np.random.default_rng(202604)

    for rounds, fb in ((16, 512), (20, 128), (24, 32)):
        w0 = rng.integers(0, 2**32, size=(1, 16), dtype=np.uint64).astype(np.uint32)
        cr = core.compress_exact(w0, rounds=rounds)
        fw, fbits = ({}, {}) if fb == 512 else make_end_free_constraints(w0[0], fb)
        res = cms.solve(
            cr.H_target[0],
            rounds=rounds,
            fixed_words=fw,
            fixed_bits=fbits,
            time_limit_sec=3.0,
        )
        assert res["num_xor_clauses"] > 0
        assert res["status"] == "SAT"
        assert res["valid_preimage"] is True


def test_dynamic_neural_cdcl_and_xor_midsearch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = get_core()
    cfg = get_model_preset("smoke")
    model = Ouroboros64Net(**cfg).to(device=device, dtype=torch.float32)
    model.eval()

    rng = np.random.default_rng(98765)
    w24 = rng.integers(0, 2**32, size=(1, 16), dtype=np.uint64).astype(np.uint32)
    cr24 = core.compress_exact(w24, rounds=24)
    fw, fbits = make_end_free_constraints(w24[0], 32)

    dyn_cdcl = DynamicNeuralCDCLSolver(
        model=model, core=core, device=device, enable_flp=True, enable_cubes=True
    )
    rc = dyn_cdcl.solve(
        cr24.H_target[0],
        rounds=24,
        fixed_words=fw,
        fixed_bits=fbits,
        initial_chunk=400,
        time_limit_sec=4.0,
    )
    assert rc.status == "SAT"
    assert rc.valid_preimage is True
    assert rc.episodes >= 1
    assert rc.w0_15 is not None
    assert bool(core.verify_preimage(rc.w0_15, cr24.H_target[0], rounds=24)[0]) is True

    if CryptoMiniSatXORSolver.is_available():
        dyn_xor = DynamicNeuralXORSolver(model=model, core=core, device=device)
        rx = dyn_xor.solve(
            cr24.H_target[0],
            rounds=24,
            fixed_words=fw,
            fixed_bits=fbits,
            time_limit_sec=4.0,
        )
        assert rx.status == "SAT"
        assert rx.valid_preimage is True


def test_m3_v2_trainer_smoke_and_paired_bench_stats(tmp_path):
    from v10_ouroboros64.ouroboros_bench.m3_v2_paired_bench import (
        build_argparser as bench_argp,
        run_paired_benchmark,
        wilcoxon_signed_rank_exact,
    )
    from v10_ouroboros64.ouroboros_train.m3_v2_trainer import (
        build_argparser as train_argp,
        train_m3_v2,
    )

    # 先独立校验 Wilcoxon 精确 p 值计算单调性（全胜 vs 平分）
    p_win = wilcoxon_signed_rank_exact(
        [100, 120, 140, 160, 180, 200, 220, 240],
        [50, 60, 70, 80, 90, 100, 110, 120],
    )
    assert p_win < 0.01
    p_tie = wilcoxon_signed_rank_exact([100, 50, 100, 50], [50, 100, 50, 100])
    assert p_tie > 0.5

    out_dir = tmp_path / "m3v2_smoke"
    t_args = train_argp().parse_args(
        [
            "--preset", "smoke",
            "--out_dir", str(out_dir),
            "--smoke",
            "--steps", "3",
        ]
    )
    summary = train_m3_v2(t_args)
    assert summary["steps"] == 3
    assert summary["track_summary"]["cdcl_adv_std_mean"] > 0.0
    assert (out_dir / "checkpoints" / "best_m3v2.pt").exists()
    assert (out_dir / "m3v2_train_summary.json").exists()
    assert (out_dir / "telemetry_v10.json").exists()

    b_args = bench_argp().parse_args(
        [
            "--m2_ckpt", str(out_dir / "checkpoints" / "best_m3v2.pt"),
            "--m3_ckpt", str(out_dir / "checkpoints" / "best_m3v2.pt"),
            "--out_dir", str(out_dir),
            "--quick",
        ]
    )
    report = run_paired_benchmark(b_args)
    rows = report["blind_inversion_paired"]["rows"]
    assert len(rows) == 2
    for row in rows:
        ps = row["paired_stats"]["m3v2_vs_pure_conflicts"]
        assert sum(ps["win_tie_loss"]) == row["cases"]
        assert 0.0 <= ps["wilcoxon_p"] <= 1.0
        assert ps["ci95_low_pct"] <= ps["ci95_high_pct"]
    assert (out_dir / "m3_v2_paired_report.json").exists()
