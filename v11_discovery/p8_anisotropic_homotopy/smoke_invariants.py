"""Local CPU-only invariant & smoke verification for P8 Anisotropic Homotopy.

Must be run with CUDA_VISIBLE_DEVICES="" on CPU. Verifies:
  1. Strict FP32 compliance (no TF32, no complex64, no FP16/BF16).
  2. Endpoint s=1 exact equivalence to V6A standard alpha=1.0 projection and SHA-256 forward digest.
  3. Exact ground-truth x* fixed-point conservation across ALL 9 paths P0..P7 (main & control) at s in {0, 0.25, 0.5, 0.75, 1.0}.
  4. Exact zero-violation alpha=0 GF(2) skeleton sampling under free_msg, allow_peel_back=False, mask_m=256.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Force CPU visibility if invoked directly without env var
if os.environ.get("P8_ALLOW_CUDA_SMOKE", "0") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sha256_ref import sha256_digest  # noqa: E402
from v6a import assert_strict_fp32, enforce_v6a_fp32  # noqa: E402
from v6a.v6a_gl_rrr import GlobalLinearRRREngine  # noqa: E402
from v6a.v6a_homotopy import (  # noqa: E402
    build_v6a_boundary,
    embed_ground_truth_vars,
    solve_alpha0_gf2_linear_skeleton,
)
from v11_discovery.p8_anisotropic_homotopy.anisotropic_homotopy import (  # noqa: E402
    P6_ABLATION_PATHS,
    PRIMARY_PATHS,
    AnisotropicGLRRREngine,
    LearnedAnisotropicField,
    evaluate_deterministic_schedule,
)
from v11_discovery.p8_anisotropic_homotopy.control_cipher import (  # noqa: E402
    build_control_boundary,
    control_digest,
    embed_control_ground_truth_vars,
    make_control_cipher_params,
)


def run_smoke() -> Dict[str, Any]:
    t0 = time.time()
    enforce_v6a_fp32()
    device = torch.device("cpu")
    ctrl_params = make_control_cipher_params(seed=20261125)
    s_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    report: Dict[str, Any] = {
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "control_params": ctrl_params.to_dict(),
        "rounds_checked": [],
        "all_passed": True,
    }

    for rounds in (4, 8):
        r_res: Dict[str, Any] = {"rounds": rounds, "variants": {}}
        for variant in ("main", "control"):
            engine = AnisotropicGLRRREngine(
                rounds=rounds,
                variant=variant,
                ctrl_params=ctrl_params if variant == "control" else None,
            ).to(device)
            assert_strict_fp32(engine)

            p7_field = LearnedAnisotropicField(rounds=rounds).to(device)
            # Non-trivial perturbation to verify x* remains fixed point even for arbitrary learned splines
            with torch.no_grad():
                gen_p = torch.Generator(device="cpu").manual_seed(1234 + rounds)
                p7_field.L_theta.normal_(mean=0.0, std=1.2, generator=gen_p)
                p7_field.psi_theta.normal_(mean=0.0, std=0.9, generator=gen_p)
            assert_strict_fp32(p7_field)

            gen = torch.Generator(device="cpu").manual_seed(777 + rounds)
            w16 = torch.randint(0, 1 << 30, (2, 16), dtype=torch.int64, generator=gen)
            if variant == "main":
                dig = sha256_digest(w16, rounds=rounds)
                bnd = build_v6a_boundary(
                    dig, rounds=rounds, domain_mode="free_msg", mask_m=256, allow_peel_back=False
                )
                x_star_vars = embed_ground_truth_vars(w16, rounds=rounds)
            else:
                dig = control_digest(w16, rounds=rounds, params=ctrl_params)
                bnd = build_control_boundary(
                    dig, rounds=rounds, params=ctrl_params, domain_mode="free_msg", mask_m=256
                )
                x_star_vars = embed_control_ground_truth_vars(w16, rounds=rounds, params=ctrl_params)

            x_star_re, x_star_im = engine.init_replicas_from_vars(x_star_vars)
            assert x_star_re.dtype == torch.float32 and x_star_im.dtype == torch.float32

            # 1. Endpoint s=1 exactness vs V6A (for main) & ground-truth readout
            s1_sched = evaluate_deterministic_schedule("P0_uniform", 1.0, rounds, device=device)
            ro_gt = engine.readout_shadow_aniso(x_star_re, bnd, s1_sched.alpha_re_tgb)
            gt_e32 = float(ro_gt["per_traj_m32"].max().item())
            gt_viol = int(ro_gt["per_traj_disc_viol"].max().item())
            gt_hits = int(ro_gt["exact_hits"])

            v6a_pb_diff = 0.0
            if variant == "main":
                v6a_eng = GlobalLinearRRREngine(rounds=rounds).to(device)
                gen_r = torch.Generator(device="cpu").manual_seed(999 + rounds)
                rand_r_re = torch.randn(x_star_re.shape, dtype=torch.float32, generator=gen_r)
                rand_r_im = torch.zeros_like(rand_r_re)
                y_v6a, val_v6a = v6a_eng.project_B(rand_r_re, bnd.var_mask, bnd.var_vals, alpha=1.0, run_osd=False)
                for p_name in PRIMARY_PATHS + P6_ABLATION_PATHS:
                    sc1 = (
                        p7_field.forward_at_s(1.0)
                        if p_name == "P7_learned_aniso"
                        else evaluate_deterministic_schedule(p_name, 1.0, rounds, device=device)
                    )
                    assert float((sc1.alpha_re_tgb - 1.0).abs().max().item()) == 0.0
                    assert float(sc1.alpha_im_tgb.abs().max().item()) == 0.0
                    assert abs(sc1.kappa_im - 1.0) == 0.0 and sc1.eta_cd == 0.0
                    y_lin, val_lin = engine.project_linear_skeleton(
                        rand_r_re[:, : engine.n_lin_rep], bnd.var_mask, bnd.var_vals
                    )
                    p_nl = engine.project_B_nl_exact(rand_r_re[:, engine.n_lin_rep :])
                    y_aniso_re, y_aniso_im = engine.combine_B_dual(rand_r_re, rand_r_im, y_lin, p_nl, sc1)
                    d = float((y_aniso_re - y_v6a).abs().max().item())
                    v6a_pb_diff = max(v6a_pb_diff, d)
                    assert torch.equal(val_lin, val_v6a)
                    assert float(y_aniso_im.abs().max().item()) == 0.0

            # 2. Fixed-point conservation across all 9 paths + P6 ablations at s in {0, 0.25, 0.5, 0.75, 1.0}
            max_fp_err = 0.0
            per_path_fp: Dict[str, float] = {}
            for p_name in PRIMARY_PATHS + P6_ABLATION_PATHS:
                p_err = 0.0
                for sv in s_grid:
                    sc = (
                        p7_field.forward_at_s(sv)
                        if p_name == "P7_learned_aniso"
                        else evaluate_deterministic_schedule(p_name, sv, rounds, device=device)
                    )
                    xn_re, xn_im, _ = engine.step_dual(
                        x_star_re,
                        x_star_im,
                        bnd.var_mask,
                        bnd.var_vals,
                        sc,
                        beta=0.85,
                        x_re_prev=x_star_re,
                        record_drift=False,
                        check_early_exit=True,
                    )
                    err_re = float((xn_re - x_star_re).abs().max().item())
                    err_im = float((xn_im - x_star_im).abs().max().item())
                    p_err = max(p_err, err_re, err_im)
                per_path_fp[p_name] = p_err
                max_fp_err = max(max_fp_err, p_err)

            # 3. Alpha=0 GF(2) exact linear skeleton solver check (1 digest, 4 samples)
            gf2_spins, gf2_viol, nullity = solve_alpha0_gf2_linear_skeleton(
                engine.spec,
                bnd.var_mask[0],
                bnd.var_vals[0],
                num_samples=4,
                generator=torch.Generator(device="cpu").manual_seed(42),
            )
            x0_re, x0_im = engine.init_replicas_from_vars(gf2_spins)
            bnd1 = build_v6a_boundary(dig[0:1], rounds=rounds, domain_mode="free_msg", mask_m=256, allow_peel_back=False) if variant == "main" else build_control_boundary(dig[0:1], rounds=rounds, params=ctrl_params, domain_mode="free_msg", mask_m=256)
            bnd4 = bnd1.repeat_for_candidates(4)
            s0_sched = evaluate_deterministic_schedule("P0_uniform", 0.0, rounds, device=device)
            ro_a0 = engine.readout_shadow_aniso(x0_re, bnd4, s0_sched.alpha_re_tgb)
            a0_lin_viol = float(ro_a0["per_traj_lin_viol"].max().item())
            a0_ehom = float(ro_a0["per_traj_e_hom"].max().item())

            passed = (
                gt_e32 < 1e-6
                and gt_viol == 0
                and gt_hits == 2
                and v6a_pb_diff < 1e-6
                and max_fp_err < 1e-6
                and gf2_viol == 0
                and a0_lin_viol == 0.0
                and a0_ehom == 0.0
            )
            if not passed:
                report["all_passed"] = False

            r_res["variants"][variant] = {
                "n_var_bits": engine.n_lin_rep,
                "n_nl_rep": engine.n_nl_rep,
                "gt_e32_max": gt_e32,
                "gt_disc_viol_max": gt_viol,
                "gt_exact_hits": gt_hits,
                "s1_v6a_pb_max_diff": v6a_pb_diff,
                "max_fixed_point_err": max_fp_err,
                "per_path_max_fp_err": per_path_fp,
                "gf2_lin_violations": gf2_viol,
                "gf2_nullity": nullity,
                "alpha0_lin_viol_max": a0_lin_viol,
                "alpha0_e_hom_max": a0_ehom,
                "passed": passed,
            }
        report["rounds_checked"].append(r_res)

    report["elapsed_sec"] = round(time.time() - t0, 3)
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "invariants_smoke.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_passed"]:
        raise AssertionError("P8 invariants smoke test failed!")
    return report


if __name__ == "__main__":
    run_smoke()
