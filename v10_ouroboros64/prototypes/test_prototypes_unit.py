"""
Phase-SHA V10「衔尾蛇-64」M3 原型单元测试套件 (`test_prototypes_unit.py`)。

完整覆盖 Explorer 1 (`U0–U8`) 与 Explorer 2 (`T-P4-1..7`) 共 16 项严格断言测试。
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pysat.solvers import Solver

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v10_ouroboros64.ouroboros_core.cnf_cdcl import (
    SHA256CNFEncoder,
    SHA256XORCNFEncoder,
    make_end_free_constraints,
)
from v10_ouroboros64.ouroboros_core.engine import (
    IV_WORDS,
    MASK32,
    SYM_0,
    SYM_1,
    TorchCrystalEngine,
    enforce_fp32_determinism,
    get_core,
)
from v10_ouroboros64.ouroboros_mcts.dynamic_cdcl import DynamicSolveResult
from v10_ouroboros64.ouroboros_mcts.hybrid_solver import load_v10_checkpoint
from v10_ouroboros64.ouroboros_mcts.symbol_env import GeneralizedSymbolEnv
from v10_ouroboros64.ouroboros_mcts.wormhole_mcts import MCTSSearchResult, WormholeMCTS
from v10_ouroboros64.ouroboros_train.trainer import _sample_random_words
from v10_ouroboros64.prototypes import (
    BridgeableSHA256XORCNFEncoder,
    CarryLockHybridRouter,
    CleanDynamicNeuralCDCLSolver,
    DualPlaneEquivalenceBridge,
    LocalWindowWormholeMCTS,
    SafeCMSSubprocessRunner,
    SignedUnionFind,
    SyncedDynamicXORSolver,
    ZeroWasteBoundaryProber,
    _build_full_crystal_var_map,
    compute_window_theory_var_count,
    local_window_gf2_propagate,
)
from v10_ouroboros64.prototypes.p1_clean_cdcl import _run_0level_bcp, _sync_crystal_state

CKPT_M3V2 = Path("v10_ouroboros64/runs/a100_m3v2/checkpoints/best_m3v2.pt")


def _load_ctx():
    enforce_fp32_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core = get_core()
    model, _ = load_v10_checkpoint(CKPT_M3V2, device=device)
    torch_eng = TorchCrystalEngine(device=device)
    return core, model, torch_eng, device


def test_u0_fp32_determinism(ctx) -> Dict[str, Any]:
    core, model, _, device = ctx
    enforce_fp32_determinism()
    if torch.cuda.is_available():
        assert torch.backends.cuda.matmul.allow_tf32 is False, "allow_tf32_matmul 必须为 False"
        assert torch.backends.cudnn.allow_tf32 is False, "allow_tf32_cudnn 必须为 False"
    dtypes = {p.dtype for p in model.parameters()}
    assert dtypes == {torch.float32}, f"参数 dtype 异常: {dtypes}"
    return {"passed": True, "cuda": torch.cuda.is_available(), "dtypes": [str(d) for d in dtypes]}


def test_u1_bool_const_guard(ctx) -> Dict[str, Any]:
    core, _, _, _ = ctx
    rng = np.random.default_rng(101)
    w = _sample_random_words(rng, 1, 16)
    cr = core.compress_exact(w, rounds=24)
    fw, fbits = make_end_free_constraints(w[0], 32)
    enc = SHA256CNFEncoder(rounds=24).build(cr.H_target[0], fw, fbits)
    cmap = _build_full_crystal_var_map(enc)

    # 所有 key 必须为严格 int (非 bool) 且 >= 1
    for k in cmap.var_to_coords.keys():
        assert isinstance(k, int) and not isinstance(k, bool) and k >= 1

    # W_vars[0][0] 对应 DIMACS 变量 1；它必须仅映射 W[0][0]，绝不能被 C_vars[*][*][0] 或 a_canon[0] 污染
    w00_lit = enc.W_vars[0][0]
    assert not isinstance(w00_lit, bool) and abs(int(w00_lit)) == 1
    coords_v1 = cmap.var_to_coords[1]
    assert coords_v1 == [(0, 0, 0, 0, 1)], f"变量 1 被污染: {coords_v1}"
    assert len(cmap.const_c) >= 24 * 10  # 至少覆盖全部 b=0 进位 False

    # 验证 sync 后 S_b 的 bool 常量位 unk == 0.0
    _, S_b, _, unk, _ = _sync_crystal_state(
        cr.H_target[0], 24, cmap, fw, fbits, [], torch.device("cpu")
    )
    for (t, word, b), cval in cmap.const_s.items():
        if t < 24:
            assert float(unk[0, t, 1 + word, b].item()) == 0.0
            assert ((int(S_b[0, t, word]) >> b) & 1) == cval
    return {"passed": True, "mapped_vars": len(cmap.var_to_coords), "const_s": len(cmap.const_s)}


def test_u2_full_crystal_alias_sync(ctx) -> Dict[str, Any]:
    core, _, _, _ = ctx
    rng = np.random.default_rng(102)
    w = _sample_random_words(rng, 1, 16)
    cr = core.compress_exact(w, rounds=24)
    # 注入全知原像边界测试 propagate() 后 S_b/C_b 全晶体同步
    fw, fbits = make_end_free_constraints(w[0], 0)
    enc = SHA256CNFEncoder(rounds=24).build(cr.H_target[0], fw, fbits)
    cmap = _build_full_crystal_var_map(enc)

    # 校验 a_canon[10][5] 四重移位别名存在性
    lit_a10 = enc.a_canon[10][5]
    assert not isinstance(lit_a10, bool)
    v_a10 = abs(int(lit_a10))
    alias_coords = set((k, t, wd, b) for k, t, wd, b, _ in cmap.var_to_coords[v_a10])
    expected_aliases = {(1, 10, 0, 5), (1, 11, 1, 5), (1, 12, 2, 5), (1, 13, 3, 5)}
    assert expected_aliases.issubset(alias_coords), f"缺失别名: {alias_coords}"

    with Solver(name="cadical195", bootstrap_with=enc.clauses) as solver:
        ok, _ = solver.propagate()
        assert ok
        ok_bcp, prop_lits = _run_0level_bcp(enc.clauses, enc.next_var - 1)
        assert ok_bcp and len(prop_lits) > 10000
        W_b, S_b, C_b, unk, _ = _sync_crystal_state(
            cr.H_target[0], 24, cmap, fw, fbits, prop_lits, torch.device("cpu")
        )

    # 全部 unk==0 的 S_b[0, 1:24] 与 C_b[0, :24] 位点必须 100% 匹配 compress_exact 真值
    s_checked = 0
    c_checked = 0
    for t in range(1, 24):
        for wd in range(8):
            for b in range(32):
                if float(unk[0, t, 1 + wd, b].item()) == 0.0:
                    got = (int(S_b[0, t, wd]) >> b) & 1
                    exp = (int(cr.S[0, t, wd]) >> b) & 1
                    assert got == exp, f"S[{t},{wd},{b}] mismatch: {got} vs {exp}"
                    s_checked += 1
    for t in range(24):
        for k in range(10):
            for b in range(32):
                if float(unk[0, t, 9 + k, b].item()) == 0.0:
                    got = (int(C_b[0, t, k]) >> b) & 1
                    exp = (int(cr.C[0, t, k]) >> b) & 1
                    assert got == exp, f"C[{t},{k},{b}] mismatch: {got} vs {exp}"
                    c_checked += 1
    assert s_checked > 5000 and c_checked > 4000
    return {"passed": True, "s_checked": s_checked, "c_checked": c_checked}


def test_u3_calibrated_gate_and_chunk(ctx) -> Dict[str, Any]:
    core, model, _, device = ctx
    rng = np.random.default_rng(103)
    w = _sample_random_words(rng, 1, 16)
    cr = core.compress_exact(w, rounds=24)
    fw, fbits = make_end_free_constraints(w[0], 40)

    p1 = CleanDynamicNeuralCDCLSolver(
        model=model, core=core, device=device,
        phase_cal_temp=4.5, phase_conf_thresh=0.08, flp_asym_thresh=16,
    )
    # 拦截检查传入 model 的参数
    seen_calls: List[Dict[str, Any]] = []
    orig_forward = model.forward

    def _wrapped_forward(*args, **kwargs):
        seen_calls.append({
            "conflict_heatmap": kwargs.get("conflict_heatmap"),
            "zero_defects": kwargs.get("zero_defects"),
        })
        return orig_forward(*args, **kwargs)

    model.forward = _wrapped_forward
    try:
        res = p1.solve(cr.H_target[0], rounds=24, fixed_words=fw, fixed_bits=fbits, time_limit_sec=5.0)
    finally:
        model.forward = orig_forward

    assert isinstance(res, DynamicSolveResult)
    assert len(seen_calls) == 1
    assert seen_calls[0]["conflict_heatmap"] is None
    assert seen_calls[0]["zero_defects"] is True
    assert p1.last_diagnostics["max_abs_logit"] < 1.452480
    assert p1.last_diagnostics["gate_passed_phase_conf"] == 0
    if p1.last_diagnostics["gate_passed_flp_asym"] == 0:
        assert p1.last_diagnostics["set_phases_called"] is False
    if res.status == "SAT":
        assert res.valid_preimage is True
        assert bool(core.verify_preimage(res.w0_15, cr.H_target[0], rounds=24)[0])

    # 硬壁钟 Timer / 速率微标定控时测试：
    # 1) R=28 16b 极短 0.05s 限时绝不卡死
    w28 = _sample_random_words(rng, 1, 16)
    cr28 = core.compress_exact(w28, rounds=28)
    fw28, fbits28 = make_end_free_constraints(w28[0], 16)
    res28 = p1.solve(cr28.H_target[0], rounds=28, fixed_words=fw28, fixed_bits=fbits28, time_limit_sec=0.05)
    assert res28.elapsed_sec < 0.75, f"R=28 短限超时过大: {res28.elapsed_sec}"

    # 2) R=64 慢算例速率微标定防超时：tl=1.5s 不应出现 10x+ 失控 (必须 < 3.0s)
    w64 = _sample_random_words(rng, 1, 16)
    cr64 = core.compress_exact(w64, rounds=64)
    res64 = p1.solve(cr64.H_target[0], rounds=64, time_limit_sec=1.5)
    assert res64.elapsed_sec < 3.0, f"R=64 tl=1.5s 超时失控: {res64.elapsed_sec}"

    # 3) PySAT 别名 cd195 覆盖与零异常/防超时校验
    p1_cd195 = CleanDynamicNeuralCDCLSolver(
        model=model, core=core, device=device, solver_name="cd195",
        phase_cal_temp=4.5, phase_conf_thresh=0.08, flp_asym_thresh=16,
    )
    res_cd195 = p1_cd195.solve(cr64.H_target[0], rounds=64, time_limit_sec=1.5)
    assert p1_cd195.last_diagnostics.get("supports_interrupt") is False
    assert res_cd195.elapsed_sec < 3.0, f"cd195 R=64 tl=1.5s 超时失控: {res_cd195.elapsed_sec}"

    # 4) 全失败模式扩展压力测试（14 格：层 A 8 格 tl=0.05s 全扫 + 层 B 5 格高压深窗 + 层 C cd103 降级兼容）
    #    安装临时 threading.excepthook 校验零后台线程未捕获异常
    w48 = _sample_random_words(rng, 1, 16)
    cr48 = core.compress_exact(w48, rounds=48)
    fw48, fbits48 = make_end_free_constraints(w48[0], 16)

    cases_ext: List[Tuple[str, str, int, int, float, np.ndarray, Any, Any]] = [
        # 层 A：4 别名 × {R=48, R=64} × tl=0.05s 全矩阵边界扫雷（P1 覆盖 6 格 + P3 覆盖 2 格）
        ("P1", "cadical195", 48, 16, 0.05, cr48.H_target[0], fw48, fbits48),
        ("P1", "cd195",      48, 16, 0.05, cr48.H_target[0], fw48, fbits48),
        ("P3", "cd153",      48, 16, 0.05, cr48.H_target[0], fw48, fbits48),
        ("P1", "cadical153", 48, 16, 0.05, cr48.H_target[0], fw48, fbits48),
        ("P1", "cadical195", 64, 64, 0.05, cr64.H_target[0], None, None),
        ("P1", "cd195",      64, 64, 0.05, cr64.H_target[0], None, None),
        ("P1", "cd153",      64, 64, 0.05, cr64.H_target[0], None, None),
        ("P3", "cadical153", 64, 64, 0.05, cr64.H_target[0], None, None),
        # 层 B：Probe B 高危失败模态深窗高压（tl=1.5/2.0s，实际进入 chunk >= 1..3 控时分支）
        ("P3", "cd153",      48, 16, 1.50, cr48.H_target[0], fw48, fbits48),
        ("P3", "cadical153", 48, 16, 2.00, cr48.H_target[0], fw48, fbits48),
        ("P1", "cadical153", 64, 64, 2.00, cr64.H_target[0], None, None),
        ("P1", "cd153",      64, 64, 1.50, cr64.H_target[0], None, None),
        ("P1", "cd195",      48, 16, 2.00, cr48.H_target[0], fw48, fbits48),
        # 层 C：Cadical103 优雅降级兼容（无 propagate / set_phases / interrupt，实进 solve_limited）
        ("P1", "cd103",      24, 40, 0.50, cr.H_target[0],   fw,   fbits),
    ]

    thread_errors: List[str] = []
    orig_excepthook = getattr(threading, "excepthook", None)

    def _capture_thread_exc(args) -> None:
        thread_errors.append(f"{type(args.exc_value).__name__}: {args.exc_value}")

    if orig_excepthook is not None:
        threading.excepthook = _capture_thread_exc

    ext_records: List[Dict[str, Any]] = []
    try:
        for entry, sname, rr, fbits_n, tl, h_tgt, fw_c, fb_c in cases_ext:
            cap_sec = max(0.75, 2.2 * tl)
            if entry == "P1":
                s_obj = CleanDynamicNeuralCDCLSolver(
                    model=model, core=core, device=device, solver_name=sname,
                    phase_cal_temp=4.5, phase_conf_thresh=0.08, flp_asym_thresh=16,
                )
                res_c = s_obj.solve(
                    h_tgt, rounds=rr, fixed_words=fw_c, fixed_bits=fb_c, time_limit_sec=tl,
                )
                diag_c = dict(s_obj.last_diagnostics)
            else:
                r_obj = CarryLockHybridRouter(
                    model=model, core=core, device=device, solver_name=sname,
                )
                try:
                    res_c = r_obj.solve(
                        h_tgt, rounds=rr, fixed_words=fw_c, fixed_bits=fb_c, time_limit_sec=tl,
                    )
                    diag_c = dict(r_obj.cdcl_solver.last_diagnostics)
                finally:
                    r_obj.close()

            assert diag_c.get("supports_interrupt") is False, (
                f"{entry} {sname} R={rr} tl={tl} 误判支持 interrupt: {diag_c}"
            )
            assert res_c.elapsed_sec <= cap_sec, (
                f"{entry} {sname} R={rr} tl={tl} 超时越界: {res_c.elapsed_sec:.4f}s > {cap_sec:.4f}s"
            )
            if sname == "cd103":
                assert diag_c.get("supports_propagate") is False, (
                    f"cd103 应标记 supports_propagate=False: {diag_c}"
                )
            if res_c.status == "SAT":
                assert res_c.valid_preimage is True
                assert bool(core.verify_preimage(res_c.w0_15, h_tgt, rounds=rr)[0])
            ext_records.append({
                "entry": entry,
                "solver": sname,
                "rounds": rr,
                "fb": fbits_n,
                "tl": tl,
                "elapsed_sec": round(res_c.elapsed_sec, 4),
                "status": res_c.status,
                "confl": res_c.conflicts,
            })
    finally:
        if orig_excepthook is not None:
            threading.excepthook = orig_excepthook

    assert thread_errors == [], f"后台线程出现未捕获异常: {thread_errors}"
    max_ratio_long = max(
        (r["elapsed_sec"] / r["tl"]) for r in ext_records if r["tl"] >= 1.0
    )

    return {
        "passed": True,
        "r24_status": res.status,
        "max_abs_logit": round(p1.last_diagnostics["max_abs_logit"], 4),
        "r28_interrupt_sec": round(res28.elapsed_sec, 4),
        "r64_elapsed_sec": round(res64.elapsed_sec, 4),
        "cd195_r64_sec": round(res_cd195.elapsed_sec, 4),
        "ext_cases": len(ext_records),
        "max_ratio_long": round(max_ratio_long, 4),
        "thread_errors": thread_errors,
        "cd103_sec": ext_records[-1]["elapsed_sec"],
    }


def test_u4_zero_waste_probe_b0(ctx) -> Dict[str, Any]:
    core, _, _, _ = ctx
    rng = np.random.default_rng(104)
    w = _sample_random_words(rng, 1, 16)
    cr = core.compress_exact(w, rounds=26)
    fw, fbits = make_end_free_constraints(w[0], 16)
    enc = SHA256CNFEncoder(rounds=26).build(cr.H_target[0], fw, fbits)

    free_lits = [
        int(enc.W_vars[15][b])
        for b in range(16, 32)
        if not isinstance(enc.W_vars[15][b], bool)
    ]
    ok_bcp, prop_lits = _run_0level_bcp(enc.clauses, enc.next_var - 1)
    assert ok_bcp and len(prop_lits) >= 10000, f"大型 initial_base_lits 长度不足: {len(prop_lits)}"

    with Solver(name="cadical195", bootstrap_with=enc.clauses) as solver:
        c0 = int((solver.accum_stats() or {}).get("conflicts", 0))
        pr = ZeroWasteBoundaryProber.probe(
            solver, free_lits, n_max_1bit=12,
            coupled_pairs=[(free_lits[0], free_lits[1])],
            initial_base_lits=prop_lits,
        )
        c1 = int((solver.accum_stats() or {}).get("conflicts", 0))
        assert c1 - c0 == 0, f"ZeroWasteBoundaryProber 产生冲突: {c1 - c0}"
        assert pr["unsat"] is False
        dp_dn = pr["dp_dn"]
        assert len(dp_dn) > 0, "dp_dn 不应为空"
        assert any(dp > 0 or dn > 0 for dp, dn in dp_dn.values()), (
            f"携带大型 initial_base_lits(len={len(prop_lits)}) 时 dp_dn 不应全归零: {dp_dn}"
        )

    # 独立校验集合差公式 len(lp_set - base_set) 与 len(ln_set - base_set)
    with Solver(name="cadical195", bootstrap_with=enc.clauses) as s_chk:
        ok_b, base_l = s_chk.propagate()
        assert ok_b
        base_set = {int(x) for x in base_l if not isinstance(x, bool)}
        base_set.update(int(x) for x in prop_lits if not isinstance(x, bool))
        sample_v = next(iter(dp_dn.keys()))
        ok_p, lp = s_chk.propagate(assumptions=[sample_v])
        ok_n, ln = s_chk.propagate(assumptions=[-sample_v])
        if ok_p and ok_n:
            exp_dp = len({int(x) for x in lp if not isinstance(x, bool)} - base_set)
            exp_dn = len({int(x) for x in ln if not isinstance(x, bool)} - base_set)
            assert dp_dn[sample_v] == (exp_dp, exp_dn), (
                f"集合差计数不匹配: got={dp_dn[sample_v]} vs exp={(exp_dp, exp_dn)}"
            )

    # 合成 Stålmarck 交集公式: (x -> y) ∧ (¬x -> y) ⊨ y
    with Solver(name="cadical195", bootstrap_with=[[-1, 2], [1, 2]]) as s_int:
        pr_int = ZeroWasteBoundaryProber.probe(s_int, [1], n_max_1bit=1)
        assert not pr_int["unsat"]
        assert 2 in pr_int["units"], f"未导出公共蕴涵单位 [2]: {pr_int}"

    # 合成 2-bit 失败对: (-1, -2) 即 1 和 2 不能同时为 1 -> 写入 [-1, -2]
    with Solver(name="cadical195", bootstrap_with=[[-1, -2, 3], [-1, -2, -3]]) as s_bin:
        pr_bin = ZeroWasteBoundaryProber.probe(s_bin, [], n_max_1bit=0, coupled_pairs=[(1, 2)])
        assert not pr_bin["unsat"]
        assert [-1, -2] in pr_bin["binary_clauses"] or [-2, -1] in pr_bin["binary_clauses"]
    return {
        "passed": True,
        "conflicts_delta": 0,
        "large_base_lits": len(prop_lits),
        "sample_dp_dn": dp_dn[next(iter(dp_dn.keys()))],
    }


def test_u5_synced_xor_state_sync(ctx) -> Dict[str, Any]:
    core, model, _, device = ctx
    rng = np.random.default_rng(105)
    w = _sample_random_words(rng, 1, 16)
    cr = core.compress_exact(w, rounds=24)
    fw, fbits = make_end_free_constraints(w[0], 40)

    sx = SyncedDynamicXORSolver(model=model, core=core, device=device)
    # 通过极短 init_tl=0.01 触发进入动态同步分支
    res = sx.solve(
        cr.H_target[0], rounds=24, fixed_words=fw, fixed_bits=fbits,
        time_limit_sec=2.0, init_tl=0.01,
    )
    diag = sx.last_diagnostics
    assert diag["zero_defects_arg"] is True
    assert diag["conflict_heatmap_arg"] is None
    counts = diag["assigned_counts"]
    assert len(counts) >= 1, "未进入动态状态同步循环"
    for i in range(1, len(counts)):
        assert counts[i] >= counts[i - 1], f"shadow_bcp 已定变量数未保持单调不减: {counts}"
    if res.status == "SAT":
        assert res.valid_preimage is True
    return {"passed": True, "status": res.status, "assigned_counts": counts[:5]}


def test_u6_bridgeable_encoder_1to1(ctx) -> Dict[str, Any]:
    core, _, _, _ = ctx
    rng = np.random.default_rng(106)
    for r in (16, 24, 26, 28):
        w = _sample_random_words(rng, 1, 16)
        cr = core.compress_exact(w, rounds=r)
        fw, fbits = make_end_free_constraints(w[0], 16)
        enc_orig = SHA256XORCNFEncoder(rounds=r).build(cr.H_target[0], fw, fbits)
        enc_cnf = SHA256CNFEncoder(rounds=r).build(cr.H_target[0], fw, fbits)
        enc_br = BridgeableSHA256XORCNFEncoder(rounds=r).build(cr.H_target[0], fw, fbits)

        assert enc_br.next_var == enc_orig.next_var == enc_cnf.next_var
        assert enc_br.clauses == enc_orig.clauses
        assert enc_br.xor_clauses == enc_orig.xor_clauses
        assert enc_br.W_vars == enc_cnf.W_vars
        assert enc_br.a_canon == enc_cnf.a_canon
        assert enc_br.e_canon == enc_cnf.e_canon
        assert enc_br.C_vars == enc_cnf.C_vars
        assert len(enc_br.Ch_vars) == r
        assert len(enc_br.Maj_vars) == r
        for t in range(r):
            assert len(enc_br.Ch_vars[t]) == 32 and len(set(enc_br.Ch_vars[t])) == 32
            assert len(enc_br.Maj_vars[t]) == 32 and len(set(enc_br.Maj_vars[t])) == 32
            assert all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in enc_br.Ch_vars[t])
            assert all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in enc_br.Maj_vars[t])
    return {"passed": True, "rounds_checked": [16, 24, 26, 28]}


def test_u7_signed_uf_conflicts(ctx) -> Dict[str, Any]:
    core, _, _, _ = ctx
    # 1. bool 常量矛盾：传入相互冲突的 fixed_bits 或矛盾摘要触发 early_unsat
    rng = np.random.default_rng(107)
    w = _sample_random_words(rng, 1, 16)
    cr = core.compress_exact(w, rounds=24)
    # 构造不可能满足的一组局部固定比特：固定全部 512 位同时篡改 H_target[0] ^= 1
    fw, fbits = make_end_free_constraints(w[0], 0)
    bad_h = cr.H_target[0].copy()
    bad_h[0] ^= np.uint32(1)
    bridge = DualPlaneEquivalenceBridge(core=core)
    res_unsat = bridge.extract(bad_h, rounds=24, fixed_words=fw, fixed_bits=fbits)
    assert res_unsat.early_unsat is True

    # 2. Bitset RREF 空变量行 0 == 1 矛盾
    ok_rref, _, _ = DualPlaneEquivalenceBridge._bitset_rref_and_extract(
        [(0b11, 0), (0b11, 1)], known_units={}
    )
    assert ok_rref is False

    # 3. SignedUnionFind 奇校验环 & 常数冲突 & 一致 2-CNF 导出
    uf_odd = SignedUnionFind()
    assert uf_odd.union(10, 11, 0) is True
    assert uf_odd.union(11, 12, 0) is True
    assert uf_odd.union(10, 12, 1) is False, "奇校验环未触发冲突"

    uf_const = SignedUnionFind()
    assert uf_const.union(20, 21, 0) is True
    ok_c, _, _ = uf_const.reconcile_units_and_export_2lit({20: 0, 21: 1})
    assert ok_c is False, "等价类常数冲突未检出"

    uf_ok = SignedUnionFind()
    assert uf_ok.union(30, 31, 0) is True
    assert uf_ok.union(32, 33, 1) is True
    ok_e, _, eq_cl = uf_ok.reconcile_units_and_export_2lit({})
    assert ok_e is True
    assert [-31, 30] in eq_cl and [31, -30] in eq_cl
    assert [33, 32] in eq_cl and [-33, -32] in eq_cl
    return {"passed": True}


def test_u8_safe_cms_fallback(ctx) -> Dict[str, Any]:
    core, model, _, device = ctx
    rng = np.random.default_rng(108)
    w = _sample_random_words(rng, 1, 16)
    cr = core.compress_exact(w, rounds=24)
    fw, fbits = make_end_free_constraints(w[0], 32)

    router = CarryLockHybridRouter(model=model, core=core, device=device)
    try:
        # 1. 模拟 worker 崩溃 (os._exit(11)) -> 捕获并自动回落至 P1+P2 解出原像
        res_crash = router.solve(
            cr.H_target[0], rounds=24, fixed_words=fw, fixed_bits=fbits,
            time_limit_sec=5.0, simulate_cms_fault="crash",
        )
        d_crash = dict(router.last_diagnostics)
        assert d_crash["zone"] == "Zone1_Linear"
        assert d_crash["cms_attempted"] is True
        assert d_crash["cms_fallback_triggered"] is True
        assert d_crash["cms_reason"] == "worker_exit_11", f"非预期崩溃原因: {d_crash}"
        assert res_crash.solver_type == "p3_zone1_linear_cdcl"
        assert res_crash.status == "SAT" and res_crash.valid_preimage is True

        # 2. 模拟 worker 超时 -> 放宽外层 tl=8.0s（cms_budget 仍封顶 2.0s，等待 2.4s 后强杀，为 P1+P2 留 >5s 裕度）
        res_to = router.solve(
            cr.H_target[0], rounds=24, fixed_words=fw, fixed_bits=fbits,
            time_limit_sec=8.0, simulate_cms_fault="timeout",
        )
        d_to = dict(router.last_diagnostics)
        assert d_to["zone"] == "Zone1_Linear"
        assert d_to["cms_attempted"] is True
        assert d_to["cms_fallback_triggered"] is True
        assert d_to["cms_reason"] == "hard_timeout_killed", f"未真实触发超时强杀: {d_to}"
        assert res_to.elapsed_sec >= 2.0, f"超时分支过早返回，未经历 CMS 超时等待: {res_to.elapsed_sec}"
        assert res_to.solver_type == "p3_zone1_linear_cdcl"
        assert res_to.status == "SAT" and res_to.valid_preimage is True

        # 3. 确认崩溃+强杀后常驻 worker 可自动重建并由 CMS 正常解出 Zone 1
        res_ok = router.solve(
            cr.H_target[0], rounds=24, fixed_words=fw, fixed_bits=fbits,
            time_limit_sec=5.0,
        )
        d_ok = dict(router.last_diagnostics)
        assert d_ok["zone"] == "Zone1_Linear"
        assert d_ok["cms_attempted"] is True
        assert d_ok["cms_fallback_triggered"] is False
        assert d_ok["cms_reason"] == "ok", f"重启后 CMS 未正常接管: {d_ok}"
        assert res_ok.solver_type == "p3_carry_lock_router_cms"
        assert res_ok.status == "SAT" and res_ok.valid_preimage is True
    finally:
        router.close()
    return {
        "passed": True,
        "crash_reason": d_crash["cms_reason"],
        "timeout_reason": d_to["cms_reason"],
        "timeout_sec": round(res_to.elapsed_sec, 4),
        "ok_reason": d_ok["cms_reason"],
    }


def test_tp4_1_fp32_and_result_type(ctx) -> Dict[str, Any]:
    core, model, torch_eng, device = ctx
    enforce_fp32_determinism()
    if torch.cuda.is_available():
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    rng = np.random.default_rng(201)
    w = _sample_random_words(rng, 1, 16)
    cr = core.compress_exact(w, rounds=64)
    mcts = LocalWindowWormholeMCTS(model=model, torch_eng=torch_eng, core=core, device=device)
    res = mcts.search(cr.H_target[0], rounds=64, num_simulations=6, time_limit_sec=0.5, focus_splice=True)
    assert isinstance(res, MCTSSearchResult)
    return {"passed": True}


def test_tp4_2_splice_block_12(ctx) -> Dict[str, Any]:
    core, model, torch_eng, device = ctx
    rng = np.random.default_rng(202)
    w = _sample_random_words(rng, 1, 16)
    cr = core.compress_exact(w, rounds=64)
    env = GeneralizedSymbolEnv(cr.H_target[0], rounds=64, core=core)
    env.propagate(use_gf2=False, max_iters=8)

    mcts = LocalWindowWormholeMCTS(
        model=model, torch_eng=torch_eng, core=core, device=device, max_branch_candidates=8
    )
    acts, _, _, _ = mcts._build_candidate_actions(env, focus_splice=True)
    macros = [a for a in acts if a.is_macro]
    assert len(macros) >= 2
    assert len(macros[0].extra_assignments) == 12, f"宏动作长度应为 12: {len(macros[0].extra_assignments)}"
    assert len(macros[1].extra_assignments) == 12
    # 验证含 8 个 S[32] 与 4 个 W[32]
    s32_cnt = sum(1 for k, t, _, _, _ in macros[0].extra_assignments if k == 1 and t == 32)
    w32_cnt = sum(1 for k, t, _, _, _ in macros[0].extra_assignments if k == 0 and t == 32)
    assert s32_cnt == 8 and w32_cnt == 4, f"S32={s32_cnt}, W32={w32_cnt}"
    return {"passed": True, "s32": s32_cnt, "w32": w32_cnt}


def test_tp4_3_window_vars_and_11_rules(ctx) -> Dict[str, Any]:
    core, _, _, _ = ctx
    assert compute_window_theory_var_count(28, 36) == 4608
    assert compute_window_theory_var_count(28, 37) == 5088

    rng = np.random.default_rng(203)
    w = _sample_random_words(rng, 1, 16)
    cr = core.compress_exact(w, rounds=64)

    # 验证 Maj(a==c) 与 Maj(b==c) 在压缩线性方程中的独立推导及边界零 PermissionError
    # 在 t=30: 固定 a[30] == a[28] (a==c -> Maj=a) 并且设置 a[29] 仍未知，同时固定除 e[31] 之外其余变量检验推导
    env = GeneralizedSymbolEnv(cr.H_target[0], rounds=64, core=core)
    t = 30
    b = 5
    # 设置 a_t = S[30,0,b] 与 c_t = S[30,2,b] 同为 1 (a==c==1 -> Maj=1), 留 b_t 未知
    env.set_symbol(1, t, 0, b, SYM_1, run_propagate=False)
    env.set_symbol(1, t, 2, b, SYM_1, run_propagate=False)
    # 设定 a[31, b]=0, e[31, b]=0, d_t=S[30,3,b]=0, S0(a_t)=... 直接构造精确可推环境：
    # 更直接：设置 t=30 附近全部 S[30,0], S[30,2] 与 C[30,7..9], a_next=S[31,0], e_next=S[31,4] 匹配真值，只留 S[30,1,b] 未知！
    # 注意此时 a==c==val，故 Maj=val 确定，而 a_diff_eq[30] 对 b_t 无依赖，可测试 a_next_eq 被触发并校验无 PermissionError
    for bit_i in range(32):
        env.set_symbol(1, t, 0, bit_i, SYM_1 if ((int(cr.S[0, t, 0]) >> bit_i) & 1) else SYM_0, run_propagate=False)
        env.set_symbol(1, t, 2, bit_i, SYM_1 if ((int(cr.S[0, t, 2]) >> bit_i) & 1) else SYM_0, run_propagate=False)
        env.set_symbol(1, t, 3, bit_i, SYM_1 if ((int(cr.S[0, t, 3]) >> bit_i) & 1) else SYM_0, run_propagate=False)
        env.set_symbol(1, t + 1, 0, bit_i, SYM_1 if ((int(cr.S[0, t + 1, 0]) >> bit_i) & 1) else SYM_0, run_propagate=False)
        for kk in (7, 8, 9):
            if bit_i > 0:
                env.set_symbol(2, t, kk, bit_i, SYM_1 if ((int(cr.C[0, t, kk]) >> bit_i) & 1) else SYM_0, run_propagate=False)
    # 留 e[31, b] (=S[31,4,b]) 未知，在 a_t[b]==c_t[b] 的位点应该由 a_diff_eq + Maj(a==c) 推出！
    res_ac = local_window_gf2_propagate(env, center_round=t, half_win=0, core=core)
    assert res_ac["conflict"] is False
    assert res_ac["units_applied"] > 0
    ac_checked = 0
    for bit_i in range(32):
        a_bit = (int(cr.S[0, t, 0]) >> bit_i) & 1
        c_bit = (int(cr.S[0, t, 2]) >> bit_i) & 1
        if a_bit == c_bit:
            exp_e31 = SYM_1 if ((int(cr.S[0, t + 1, 4]) >> bit_i) & 1) else SYM_0
            assert int(env.S_sym[t + 1, 4, bit_i]) == exp_e31
            ac_checked += 1
    assert ac_checked > 0

    # 同理验证 Maj(b==c) 且 a 未知
    env_bc = GeneralizedSymbolEnv(cr.H_target[0], rounds=64, core=core)
    for bit_i in range(32):
        # b_t = S[t, 1] 与 c_t = S[t, 2] 已知，且选 b_t == c_t 的位点验证推导
        b_bit = (int(cr.S[0, t, 1]) >> bit_i) & 1
        c_bit = (int(cr.S[0, t, 2]) >> bit_i) & 1
        env_bc.set_symbol(1, t, 1, bit_i, SYM_1 if b_bit else SYM_0, run_propagate=False)
        env_bc.set_symbol(1, t, 2, bit_i, SYM_1 if c_bit else SYM_0, run_propagate=False)
        env_bc.set_symbol(1, t, 3, bit_i, SYM_1 if ((int(cr.S[0, t, 3]) >> bit_i) & 1) else SYM_0, run_propagate=False)
        env_bc.set_symbol(1, t + 1, 0, bit_i, SYM_1 if ((int(cr.S[0, t + 1, 0]) >> bit_i) & 1) else SYM_0, run_propagate=False)
        env_bc.set_symbol(1, t + 1, 4, bit_i, SYM_1 if ((int(cr.S[0, t + 1, 4]) >> bit_i) & 1) else SYM_0, run_propagate=False)
        for kk in (7, 8, 9):
            if bit_i > 0:
                env_bc.set_symbol(2, t, kk, bit_i, SYM_1 if ((int(cr.C[0, t, kk]) >> bit_i) & 1) else SYM_0, run_propagate=False)
    # 设 a_t=S[t,0] 全未知会留 32 个 Sigma0 未知位；只留单个 b0 (b_bit == c_bit) 未知，其余 31 位固定
    target_b = next(bi for bi in range(32) if ((int(cr.S[0, t, 1]) >> bi) & 1) == ((int(cr.S[0, t, 2]) >> bi) & 1))
    for bit_i in range(32):
        if bit_i != target_b:
            env_bc.set_symbol(1, t, 0, bit_i, SYM_1 if ((int(cr.S[0, t, 0]) >> bit_i) & 1) else SYM_0, run_propagate=False)
    # 在 a_diff_eq 中 a_t[target_b] 出现在 3 个旋转下标的对应方程中，其中至少一个方程中 Maj(b==c) 已定而唯一未知为 a_t[target_b]
    res_bc = local_window_gf2_propagate(env_bc, center_round=t, half_win=0, core=core)
    assert res_bc["conflict"] is False
    assert int(env_bc.S_sym[t, 0, target_b]) == (SYM_1 if ((int(cr.S[0, t, 0]) >> target_b) & 1) else SYM_0)

    # 验证 a_canon 别名冲突检出 & 不可写边界 S[0]/S[64] 零 PermissionError
    env_alias = GeneralizedSymbolEnv(cr.H_target[0], rounds=64, core=core)
    env_alias.set_symbol(1, 30, 0, 0, SYM_0, run_propagate=False)
    env_alias.set_symbol(1, 31, 1, 0, SYM_1, run_propagate=False)  # 与 S[30,0,0] 同属 a_canon[30]
    res_al = local_window_gf2_propagate(env_alias, center_round=30, half_win=1, core=core)
    assert res_al["conflict"] is True

    # 边界轮 t=63 (邻接 S[64]) 不应抛出 PermissionError
    env_bnd = GeneralizedSymbolEnv(cr.H_target[0], rounds=64, core=core)
    res_bnd = local_window_gf2_propagate(env_bnd, center_round=63, half_win=4, core=core)
    assert res_bnd["conflict"] is False
    return {"passed": True, "ac_units": res_ac["units_applied"], "bc_units": res_bc["units_applied"]}


def test_tp4_4_local_gf2_soundness_and_conflict(ctx) -> Dict[str, Any]:
    core, _, _, _ = ctx
    rng = np.random.default_rng(204)
    w = _sample_random_words(rng, 1, 16)
    cr = core.compress_exact(w, rounds=64)

    # 正确局部窗约束：只给 S[63,0] 与 S[63,4] 与 C[63] 部分进位，推导出 W[63] 或相关位点且 100% 匹配真值
    env = GeneralizedSymbolEnv(cr.H_target[0], rounds=64, core=core)
    t = 63
    for bit_i in range(32):
        env.set_symbol(1, t, 0, bit_i, SYM_1 if ((int(cr.S[0, t, 0]) >> bit_i) & 1) else SYM_0, run_propagate=False)
        env.set_symbol(1, t, 4, bit_i, SYM_1 if ((int(cr.S[0, t, 4]) >> bit_i) & 1) else SYM_0, run_propagate=False)
        env.set_symbol(1, t, 3, bit_i, SYM_1 if ((int(cr.S[0, t, 3]) >> bit_i) & 1) else SYM_0, run_propagate=False)
        env.set_symbol(1, t, 7, bit_i, SYM_1 if ((int(cr.S[0, t, 7]) >> bit_i) & 1) else SYM_0, run_propagate=False)
        for kk in range(3, 10):
            if bit_i > 0:
                env.set_symbol(2, t, kk, bit_i, SYM_1 if ((int(cr.C[0, t, kk]) >> bit_i) & 1) else SYM_0, run_propagate=False)

    res_ok = local_window_gf2_propagate(env, center_round=63, half_win=2, core=core)
    assert res_ok["conflict"] is False
    assert res_ok["units_applied"] >= 32
    for bit_i in range(32):
        exp_w = SYM_1 if ((int(cr.W[0, 63]) >> bit_i) & 1) else SYM_0
        assert int(env.W_sym[63, bit_i]) == exp_w

    # 1-bit 翻转破坏旋转 XOR 奇偶性 -> 精确返回 conflict == True
    env_bad = env.clone()
    # 留 W[63] 全未知但翻转 d_63=S[63,3,0] 同时检查与 S[64] 的一致性
    env_bad2 = GeneralizedSymbolEnv(cr.H_target[0], rounds=64, core=core)
    for bit_i in range(32):
        env_bad2.set_symbol(1, t, 0, bit_i, SYM_1 if ((int(cr.S[0, t, 0]) >> bit_i) & 1) else SYM_0, run_propagate=False)
        env_bad2.set_symbol(1, t, 3, bit_i, SYM_1 if (((int(cr.S[0, t, 3]) >> bit_i) & 1) ^ (1 if bit_i == 0 else 0)) else SYM_0, run_propagate=False)
        for kk in (7, 8, 9):
            if bit_i > 0:
                env_bad2.set_symbol(2, t, kk, bit_i, SYM_1 if ((int(cr.C[0, t, kk]) >> bit_i) & 1) else SYM_0, run_propagate=False)
    res_bad = local_window_gf2_propagate(env_bad2, center_round=63, half_win=2, core=core)
    assert res_bad["conflict"] is True
    return {"passed": True, "units_applied": res_ok["units_applied"]}


def test_tp4_5_single_forward_and_clean_eval(ctx) -> Dict[str, Any]:
    core, model, torch_eng, device = ctx
    rng = np.random.default_rng(205)
    w = _sample_random_words(rng, 1, 16)
    cr = core.compress_exact(w, rounds=64)

    base_mcts = WormholeMCTS(
        model=model, torch_eng=torch_eng, core=core, device=device,
        max_branch_candidates=16, use_gf2_root=False, enable_macro_actions=True,
    )
    p4_mcts = LocalWindowWormholeMCTS(
        model=model, torch_eng=torch_eng, core=core, device=device,
        max_branch_candidates=8, half_win=4, use_local_gf2=True, skip_leaf_neural_eval=True,
    )

    base_calls = 0
    orig_base = base_mcts._neural_score_env

    def _cnt_base(*a, **kw):
        nonlocal base_calls
        base_calls += 1
        return orig_base(*a, **kw)

    base_mcts._neural_score_env = _cnt_base
    base_mcts.search(cr.H_target[0], rounds=64, num_simulations=20, time_limit_sec=5.0, focus_splice=True)

    p4_mcts.search(cr.H_target[0], rounds=64, num_simulations=20, time_limit_sec=5.0, focus_splice=True)
    p4_calls = p4_mcts.last_search_diagnostics["neural_eval_calls"]

    assert p4_mcts.last_search_diagnostics["conflict_heatmap_was_always_none"] is True
    assert p4_calls < base_calls, f"P4 前向次数 {p4_calls} 未显著低于基线 {base_calls}"
    return {"passed": True, "base_calls_20sims": base_calls, "p4_calls_20sims": p4_calls}


def test_tp4_6_r64_overconstrained_probe(ctx) -> Dict[str, Any]:
    core, model, torch_eng, device = ctx
    rng = np.random.default_rng(206)
    p4_mcts = LocalWindowWormholeMCTS(
        model=model, torch_eng=torch_eng, core=core, device=device,
        max_branch_candidates=8, half_win=4, use_local_gf2=True, skip_leaf_neural_eval=True,
    )
    results = []
    for fb in (4, 6):
        w = _sample_random_words(rng, 1, 16)
        cr = core.compress_exact(w, rounds=64)
        fw, fbits = make_end_free_constraints(w[0], fb)
        res = p4_mcts.search(
            cr.H_target[0], rounds=64, fixed_words=fw, fixed_bits=fbits,
            num_simulations=256, time_limit_sec=3.0, focus_splice=False,
        )
        assert res.solved is True, f"R=64 free_bits={fb} 探针未解出: {res}"
        assert res.w0_15 is not None
        assert bool(core.verify_preimage(res.w0_15, cr.H_target[0], rounds=64)[0]) is True
        results.append({
            "fb": fb, "solved": res.solved, "sims": res.simulations,
            "pruned_conflicts": res.pruned_conflicts, "sec": round(res.elapsed_sec, 4),
        })
    assert any(r["pruned_conflicts"] > 0 for r in results), f"预期记录到 pruned_conflicts > 0: {results}"
    return {"passed": True, "probes": results}


def test_tp4_7_r64_blind_throughput_and_depth(ctx) -> Dict[str, Any]:
    core, model, torch_eng, device = ctx
    rng = np.random.default_rng(207)
    w = _sample_random_words(rng, 1, 16)
    cr = core.compress_exact(w, rounds=64)
    p4_mcts = LocalWindowWormholeMCTS(
        model=model, torch_eng=torch_eng, core=core, device=device,
        max_branch_candidates=8, half_win=4, use_local_gf2=True, skip_leaf_neural_eval=True,
    )
    res = p4_mcts.search(
        cr.H_target[0], rounds=64, fixed_words=None,
        num_simulations=600, time_limit_sec=3.0, focus_splice=True,
    )
    diag = p4_mcts.last_search_diagnostics
    assert res.simulations >= 120, f"sims={res.simulations} < 120"
    assert diag["max_depth_reached"] >= 4, f"max_depth={diag['max_depth_reached']} < 4"
    assert diag["local_gf2_mean_ms"] < 8.0, f"local_gf2_mean_ms={diag['local_gf2_mean_ms']} >= 8ms"
    return {
        "passed": True,
        "simulations": res.simulations,
        "max_depth": diag["max_depth_reached"],
        "local_gf2_mean_ms": diag["local_gf2_mean_ms"],
        "pruned_conflicts": res.pruned_conflicts,
        "local_gf2_units": diag["local_gf2_units"],
    }


def run_all_tests() -> Dict[str, Any]:
    t0 = time.perf_counter()
    print("[UnitTest] 初始化 FP32 环境与只读加载 best_m3v2.pt ...", flush=True)
    ctx = _load_ctx()
    tests = [
        ("U0_fp32_determinism", test_u0_fp32_determinism),
        ("U1_bool_const_guard", test_u1_bool_const_guard),
        ("U2_full_crystal_alias_sync", test_u2_full_crystal_alias_sync),
        ("U3_calibrated_gate_and_chunk", test_u3_calibrated_gate_and_chunk),
        ("U4_zero_waste_probe_b0", test_u4_zero_waste_probe_b0),
        ("U5_synced_xor_state_sync", test_u5_synced_xor_state_sync),
        ("U6_bridgeable_encoder_1to1", test_u6_bridgeable_encoder_1to1),
        ("U7_signed_uf_conflicts", test_u7_signed_uf_conflicts),
        ("U8_safe_cms_fallback", test_u8_safe_cms_fallback),
        ("T_P4_1_fp32_and_result_type", test_tp4_1_fp32_and_result_type),
        ("T_P4_2_splice_block_12", test_tp4_2_splice_block_12),
        ("T_P4_3_window_vars_and_11_rules", test_tp4_3_window_vars_and_11_rules),
        ("T_P4_4_local_gf2_soundness_and_conflict", test_tp4_4_local_gf2_soundness_and_conflict),
        ("T_P4_5_single_forward_and_clean_eval", test_tp4_5_single_forward_and_clean_eval),
        ("T_P4_6_r64_overconstrained_probe", test_tp4_6_r64_overconstrained_probe),
        ("T_P4_7_r64_blind_throughput_and_depth", test_tp4_7_r64_blind_throughput_and_depth),
    ]
    report: Dict[str, Any] = {"all_passed": True, "tests": {}}
    for name, fn in tests:
        ts = time.perf_counter()
        out = fn(ctx)
        dt = time.perf_counter() - ts
        out["elapsed_sec"] = round(dt, 4)
        report["tests"][name] = out
        print(f"  [PASS] {name} ({dt:.3f}s) -> {out}", flush=True)
    report["total_elapsed_sec"] = round(time.perf_counter() - t0, 3)
    print(f"[UnitTest] 全部 16/16 单元测试通过！总耗时 {report['total_elapsed_sec']:.2f}s", flush=True)
    return report


if __name__ == "__main__":
    rep = run_all_tests()
    out_p = Path("v10_ouroboros64/prototypes/unit_test_report.json")
    out_p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
