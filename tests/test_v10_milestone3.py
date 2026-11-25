"""
Phase-SHA V10 Milestone 3 单元测试：
验证轨 A 虫洞 MCTS（PUCT、零内部泄漏、精确矛盾剪枝）、
M2 检查点热加载与 OuroborosHybridSolver 在 R=16 及 R=20(96b) 上的 100% verify_preimage 盲求逆验真。
"""

import numpy as np
import torch

from v10_ouroboros64.ouroboros_core.engine import TorchCrystalEngine, get_core
from v10_ouroboros64.ouroboros_mcts.hybrid_solver import OuroborosHybridSolver, load_v10_checkpoint
from v10_ouroboros64.ouroboros_mcts.wormhole_mcts import WormholeMCTS
from v10_ouroboros64.ouroboros_net.model import Ouroboros64Net, get_model_preset


def test_wormhole_mcts_zero_leakage_and_r64_closure():
    core = get_core()
    rng = np.random.default_rng(31415)
    w0 = rng.integers(0, 2**32, size=(1, 16), dtype=np.uint64).astype(np.uint32)
    cr = core.compress_exact(w0, rounds=64)

    mcts = WormholeMCTS(model=None, core=core, max_branch_candidates=16)
    res = mcts.search(
        H_target=cr.H_target[0],
        rounds=64,
        fixed_words=None,
        num_simulations=12,
        time_limit_sec=1.5,
        focus_splice=True,
    )
    assert res.rounds == 64
    assert res.root_unknown > 0
    assert res.min_unknown_reached <= res.root_unknown
    assert res.min_active_carry_gates <= res.root_active_carry_gates
    assert res.min_splice_freedom <= res.root_splice_freedom
    # 512-bit 全盲 64 轮在 12 次模拟内绝不可能意外通过 verify_preimage
    assert res.solved is False


def test_hybrid_solver_blind_r16_and_r20_with_m2_weights(tmp_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = get_core()
    cfg = get_model_preset("smoke")
    model = Ouroboros64Net(**cfg).to(device=device, dtype=torch.float32)

    # 模拟保存并热加载 Milestone 2 检查点
    ckpt = tmp_path / "m2_mock_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "preset": "smoke",
            "preset_cfg": cfg,
            "rounds": 64,
            "step": 42,
        },
        ckpt,
    )
    loaded_model, payload = load_v10_checkpoint(ckpt, device=device)
    assert payload["step"] == 42

    hybrid = OuroborosHybridSolver(model=loaded_model, core=core, device=device)
    rng = np.random.default_rng(271828)

    # 1) R=16 完全盲 512b
    w16 = rng.integers(0, 2**32, size=(1, 16), dtype=np.uint64).astype(np.uint32)
    cr16 = core.compress_exact(w16, rounds=16)
    r16 = hybrid.solve(
        cr16.H_target[0],
        rounds=16,
        fixed_words=None,
        mcts_sims=8,
        mcts_time_limit=0.6,
        total_time_limit=4.0,
    )
    assert r16.status == "SAT"
    assert r16.valid_preimage is True
    assert r16.w0_15 is not None
    assert bool(core.verify_preimage(r16.w0_15, cr16.H_target[0], rounds=16)[0]) is True

    # 2) R=20 末尾 96b 自由（仅给 H 与 W[0..12]，严禁给真值 W[13..15]）
    w20 = rng.integers(0, 2**32, size=(1, 16), dtype=np.uint64).astype(np.uint32)
    cr20 = core.compress_exact(w20, rounds=20)
    fixed_13 = {wi: int(w20[0, wi]) for wi in range(13)}
    r20 = hybrid.solve(
        cr20.H_target[0],
        rounds=20,
        fixed_words=fixed_13,
        mcts_sims=10,
        mcts_time_limit=0.6,
        total_time_limit=4.0,
    )
    assert r20.status == "SAT"
    assert r20.valid_preimage is True
    assert r20.w0_15 is not None
    assert bool(core.verify_preimage(r20.w0_15, cr20.H_target[0], rounds=20)[0]) is True


def test_open_boundary_alpha_probe_d8_and_btc_padding():
    from v10_ouroboros64.ouroboros_bench.open_boundary_alpha_probe import probe_single_d_tier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = get_core()
    torch_eng = TorchCrystalEngine(device=device)
    cfg = get_model_preset("smoke")
    model = Ouroboros64Net(**cfg).to(device=device, dtype=torch.float32)
    model.eval()

    # D=8 (1/256 命中率) 与 BTC 320-bit 零填充均必须在几秒内高置信命中且独立 C++ 验真通过
    res_free = probe_single_d_tier(
        model=model,
        core=core,
        torch_eng=torch_eng,
        device=device,
        d_bits=8,
        btc_padding=False,
        trials=3,
        batch_cands=64,
        max_steps_per_trial=40,
        tier_time_budget_sec=10.0,
    )
    assert res_free["hits"] >= 1
    assert res_free["max_trailing_zeros_seen"] >= 8

    res_btc = probe_single_d_tier(
        model=model,
        core=core,
        torch_eng=torch_eng,
        device=device,
        d_bits=8,
        btc_padding=True,
        trials=2,
        batch_cands=64,
        max_steps_per_trial=40,
        tier_time_budget_sec=10.0,
    )
    assert res_btc["hits"] >= 1


def test_fixed_bits_and_make_end_free_r26_blind():
    from v10_ouroboros64.ouroboros_core.cnf_cdcl import (
        CaDiCaLCDCLSolver,
        make_end_free_constraints,
    )

    core = get_core()
    rng = np.random.default_rng(777)
    w26 = rng.integers(0, 2**32, size=(1, 16), dtype=np.uint64).astype(np.uint32)
    cr26 = core.compress_exact(w26, rounds=26)
    fw, fbits = make_end_free_constraints(w26[0], free_bits=12)
    assert len(fw) == 15
    assert len(fbits) == 20

    solver = CaDiCaLCDCLSolver("cadical195")
    res = solver.solve(
        cr26.H_target[0],
        rounds=26,
        fixed_words=fw,
        fixed_bits=fbits,
        conflict_budget=20000,
        time_limit_sec=3.0,
    )
    assert res["status"] == "SAT"
    assert res["valid_preimage"] is True


def test_wormhole_mcts_dual_speed_r64_macro_collapse():
    core = get_core()
    rng = np.random.default_rng(202611)
    w64 = rng.integers(0, 2**32, size=(1, 16), dtype=np.uint64).astype(np.uint32)
    cr64 = core.compress_exact(w64, rounds=64)

    mcts = WormholeMCTS(
        model=None,
        core=core,
        max_branch_candidates=16,
        use_gf2_root=False,
        gf2_interval=4,
        enable_macro_actions=True,
    )
    res = mcts.search(
        H_target=cr64.H_target[0],
        rounds=64,
        fixed_words=None,
        num_simulations=16,
        time_limit_sec=1.5,
        focus_splice=True,
    )
    assert res.expansions >= 8
    assert res.min_unknown_reached < res.root_unknown
    assert res.min_active_carry_gates < res.root_active_carry_gates
    assert res.min_splice_freedom < res.root_splice_freedom
    assert res.solved is False


def test_m3_trainer_smoke_end_to_end(tmp_path):
    from v10_ouroboros64.ouroboros_train.m3_trainer import build_argparser, train_m3

    args = build_argparser().parse_args(
        [
            "--preset", "smoke",
            "--out_dir", str(tmp_path / "m3_smoke"),
            "--smoke",
            "--steps", "3",
        ]
    )
    summary = train_m3(args)
    assert summary["steps"] == 3
    assert summary["track_summary"]["cdcl_adv_std_mean"] > 0.0
    assert (tmp_path / "m3_smoke" / "checkpoints" / "best_m3_frontier.pt").exists()
    assert (tmp_path / "m3_smoke" / "checkpoints" / "ouroboros_m3_step3.pt").exists()
    assert (tmp_path / "m3_smoke" / "telemetry_v10.json").exists()
    assert (tmp_path / "m3_smoke" / "m3_train_summary.json").exists()
