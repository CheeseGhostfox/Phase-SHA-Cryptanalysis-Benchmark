"""Main experiment runner for P9: ARX Multi-Round Composite Low-Degree Proxy Circuit Discovery.

Supports:
  --mode smoke : CPU-only seconds-level verification across all families, controls, and Z3 checks.
  --mode full  : Full sweep on Colab (T4 GPU / high-mem CPU) within <= 40 min budget.

Outputs under <out_dir>:
  - search_summary.json
  - top_circuits.json
  - z3_verification.json
  - run_log.txt (when redirected or written explicitly)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .arx_operators import (
    OperatorSpec,
    evaluate_operator_targets,
    get_mini_arx_params,
    get_random_arx_params,
    get_real_sha256_params,
    sample_input_words,
    words_to_packed_inputs,
)
from .circuit_slp import MAX_AND_GATES, MAX_DEGREE, CircuitSLP, count_packed_matches, pack_bits_vector
from .known_baselines import KnownBaselineResult, build_known_baseline_library, evaluate_known_baselines
from .search_methods import SearchOutcome, run_combined_search
from .stats_utils import compute_eval_metrics, compute_paired_incremental
from .z3_verifier import (
    verify_carry_truncation_bound_z3,
    verify_circuit_complexity_and_anf,
    verify_family_c_exact_count,
    verify_nonequivalence_against_known,
)


def log_msg(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def evaluate_single_operator_config(
    spec: OperatorSpec,
    n_search: int,
    n_eval: int,
    n_eval_escalate: int,
    evo_gens: int,
    ste_steps: int,
    max_and_budget: int,
    max_deg: int,
    device: torch.device,
    seed_base: int,
    is_control: bool = False,
) -> Dict[str, Any]:
    """Run search + held-out evaluation for ALL targets of a given OperatorSpec."""
    # Search split
    in_words_s = sample_input_words(spec, n_search, seed=seed_base + 11, device=device)
    targets_s = evaluate_operator_targets(spec, in_words_s)
    in_packed_s = words_to_packed_inputs(in_words_s, spec.word_bits)

    # Independent held-out evaluation split
    in_words_e = sample_input_words(spec, n_eval, seed=seed_base + 100_003, device=device)
    targets_e = evaluate_operator_targets(spec, in_words_e)
    in_packed_e = words_to_packed_inputs(in_words_e, spec.word_bits)

    per_target: Dict[str, Any] = {}
    for t_idx, tname in enumerate(spec.target_list()):
        t_bits_s = targets_s[tname]
        t_packed_s = pack_bits_vector(t_bits_s)

        t_bits_e = targets_e[tname]
        t_packed_e = pack_bits_vector(t_bits_e)

        # Known baseline library on search split (for B4 greedy selection) and eval split
        precompiled_known = build_known_baseline_library(spec, tname)
        known_search = evaluate_known_baselines(spec, tname, in_packed_s, t_packed_s, n_search, precompiled=precompiled_known)
        known_eval = evaluate_known_baselines(spec, tname, in_packed_e, t_packed_e, n_eval, precompiled=precompiled_known)

        # Combined search across Methods 1, 2, 3 strictly on search split
        outcome: SearchOutcome = run_combined_search(
            spec=spec,
            target_name=tname,
            input_words_search=in_words_s,
            target_bits_search=t_bits_s,
            input_packed_search=in_packed_s,
            target_packed_search=t_packed_s,
            known_res=known_search,
            n_search=n_search,
            max_and_budget=max_and_budget,
            max_deg=max_deg,
            evo_gens=evo_gens,
            ste_steps=ste_steps,
            seed=seed_base + t_idx * 37,
        )

        # Evaluate top circuits on held-out N_eval
        best_eval_metrics: Optional[Dict[str, Any]] = None
        best_eval_circuit: Optional[CircuitSLP] = None
        best_eval_pred: Optional[torch.Tensor] = None
        top_eval_list: List[Tuple[CircuitSLP, Dict[str, Any]]] = []

        for cand in outcome.top_circuits:
            pred_e = cand.eval_packed(in_packed_e)
            matches_e = count_packed_matches(pred_e, t_packed_e)
            metrics_e = compute_eval_metrics(matches_e, n_eval, outcome.n_trials)
            top_eval_list.append((cand, metrics_e))
            if best_eval_metrics is None or metrics_e["eval_bias"] > best_eval_metrics["eval_bias"]:
                best_eval_metrics = metrics_e
                best_eval_circuit = cand
                best_eval_pred = pred_e

        assert best_eval_circuit is not None and best_eval_metrics is not None and best_eval_pred is not None

        # Escalate to N_eval_escalate (2^22 in full mode) if z >= 4.0 on real SHA-256
        escalated_used = False
        known_eval_final = known_eval
        t_packed_final = t_packed_e
        in_packed_final = in_packed_e
        if (not is_control) and (best_eval_metrics["z_score"] >= 4.0) and (n_eval_escalate > n_eval):
            escalated_used = True
            in_words_big = sample_input_words(spec, n_eval_escalate, seed=seed_base + 500_009 + t_idx, device=device)
            t_bits_big = evaluate_operator_targets(spec, in_words_big)[tname]
            in_packed_final = words_to_packed_inputs(in_words_big, spec.word_bits)
            t_packed_final = pack_bits_vector(t_bits_big)

            best_eval_pred = best_eval_circuit.eval_packed(in_packed_final)
            matches_big = count_packed_matches(best_eval_pred, t_packed_final)
            best_eval_metrics = compute_eval_metrics(matches_big, n_eval_escalate, outcome.n_trials)
            known_eval_final = evaluate_known_baselines(
                spec, tname, in_packed_final, t_packed_final, n_eval_escalate, precompiled=precompiled_known
            )

        n_final = best_eval_metrics["n_eval"]
        known_matches_final = int(round(known_eval_final.best_p * n_final))
        known_metrics = compute_eval_metrics(known_matches_final, n_final, n_trials=1)
        inc_res = compute_paired_incremental(
            best_eval_pred, known_eval_final.best_pred_packed, t_packed_final, n_final
        )

        per_target[tname] = {
            "best_circuit": best_eval_circuit,
            "top_circuits": [c for c, _ in top_eval_list[:3]],
            "top_metrics": [m for _, m in top_eval_list[:3]],
            "known_eval": known_eval_final,
            "known_metrics": known_metrics,
            "best_metrics": best_eval_metrics,
            "incremental": inc_res,
            "escalated": escalated_used,
            "method_search_best": outcome.method_best,
        }

    return per_target


def build_experiment_schedule(mode: str) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    if mode == "smoke":
        configs = [
            ("A1", 2),
            ("A2", 2),
            ("B", 2),
            ("C4", 2),
            ("C8", 2),
        ]
        params = {
            "n_search": 2048,
            "n_eval": 4096,
            "n_eval_escalate": 8192,
            "evo_gens": 4,
            "ste_steps": 10,
            "n_controls": 3,
            "max_and": MAX_AND_GATES,
            "max_deg": MAX_DEGREE,
        }
        return configs, params

    if mode == "full":
        configs = []
        for r in (2, 3, 4, 5, 6):
            configs.append(("A1", r))
        for r in (2, 3, 4, 5, 6):
            configs.append(("A2", r))
        for r in (2, 3, 4, 5, 6):
            configs.append(("B", r))
        for r in (2, 3):
            configs.append(("B_joint", r))
        for r in (2, 3, 4):
            configs.append(("C4", r))
        for r in (2, 3, 4):
            configs.append(("C8", r))

        params = {
            "n_search": 65536,         # 2^16
            "n_eval": 1_048_576,       # 2^20
            "n_eval_escalate": 4_194_304,  # 2^22
            "evo_gens": 10,
            "ste_steps": 36,
            "n_controls": 3,
            "max_and": MAX_AND_GATES,
            "max_deg": MAX_DEGREE,
        }
        return configs, params

    raise ValueError(f"Unknown mode={mode}")


def run_experiment(mode: str, out_dir: Path, force_cpu: bool = False) -> None:
    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    if force_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    log_msg(f"P9 Experiment start | mode={mode} | device={device} | torch={torch.__version__}")
    if device.type == "cuda":
        log_msg(f"CUDA device={torch.cuda.get_device_name(0)} | allow_tf32={torch.backends.cuda.matmul.allow_tf32}")

    configs, hp = build_experiment_schedule(mode)
    summary_records: List[Dict[str, Any]] = []
    top_circuits_records: Dict[str, Any] = {}
    z3_records: Dict[str, Any] = {
        "carry_truncation_w4_exact": verify_carry_truncation_bound_z3(w=4),
        "family_c_exact_counts": [],
        "circuit_checks": [],
    }

    for cfg_idx, (family, rounds) in enumerate(configs):
        cfg_t0 = time.time()
        word_bits = 4 if family == "C4" else (8 if family == "C8" else 32)
        if family in ("C4", "C8"):
            real_params = get_mini_arx_params(word_bits, rounds)
        else:
            real_params = get_real_sha256_params(rounds)

        real_spec = OperatorSpec(family=family, rounds=rounds, params=real_params)
        log_msg(f"=== [{cfg_idx+1}/{len(configs)}] {family} R={rounds} (w={word_bits}, n_in={real_spec.n_inputs}, exact_AND={real_spec.exact_and_gates}) ===")

        real_res = evaluate_single_operator_config(
            spec=real_spec,
            n_search=hp["n_search"],
            n_eval=hp["n_eval"],
            n_eval_escalate=hp["n_eval_escalate"],
            evo_gens=hp["evo_gens"],
            ste_steps=hp["ste_steps"],
            max_and_budget=hp["max_and"],
            max_deg=hp["max_deg"],
            device=device,
            seed_base=100_000 * (cfg_idx + 1),
            is_control=False,
        )

        # Run 3 Random-ARX structural control groups with identical budgets
        ctrl_results: List[Dict[str, Any]] = []
        for c_seed in range(1, hp["n_controls"] + 1):
            ctrl_params = get_random_arx_params(seed=c_seed, word_bits=word_bits, rounds=rounds)
            ctrl_spec = OperatorSpec(family=family, rounds=rounds, params=ctrl_params)
            c_res = evaluate_single_operator_config(
                spec=ctrl_spec,
                n_search=hp["n_search"],
                n_eval=hp["n_eval"],
                n_eval_escalate=hp["n_eval"],
                evo_gens=hp["evo_gens"],
                ste_steps=hp["ste_steps"],
                max_and_budget=hp["max_and"],
                max_deg=hp["max_deg"],
                device=device,
                seed_base=100_000 * (cfg_idx + 1) + 10_000 * c_seed,
                is_control=True,
            )
            ctrl_results.append(c_res)

        var_names = real_spec.input_var_names()
        for tname in real_spec.target_list():
            r_item = real_res[tname]
            best_c: CircuitSLP = r_item["best_circuit"]
            bm = r_item["best_metrics"]
            km = r_item["known_metrics"]
            ke: KnownBaselineResult = r_item["known_eval"]
            inc = r_item["incremental"]

            ctrl_biases = [ctrl_results[s][tname]["best_metrics"]["eval_bias"] for s in range(hp["n_controls"])]
            ctrl_ps = [ctrl_results[s][tname]["best_metrics"]["eval_p"] for s in range(hp["n_controls"])]
            max_ctrl_bias = max(ctrl_biases)
            max_ctrl_idx = ctrl_biases.index(max_ctrl_bias)
            p_ctrl_max = ctrl_ps[max_ctrl_idx]
            p_real = bm["eval_p"]
            n_real = bm["n_eval"]
            n_ctrl = ctrl_results[max_ctrl_idx][tname]["best_metrics"]["n_eval"]
            sigma_diff = max(
                1e-9,
                ((p_real * (1.0 - p_real) / n_real) + (p_ctrl_max * (1.0 - p_ctrl_max) / n_ctrl)) ** 0.5,
            )
            z_vs_ctrl = (bm["eval_bias"] - max_ctrl_bias) / sigma_diff

            deg_val = best_c.exact_anf_degree()
            if deg_val is None:
                deg_val = best_c.structural_degree_bound()

            rec = {
                "family": family,
                "rounds": rounds,
                "word_bits": word_bits,
                "target_bit": tname,
                "exact_and_gates": real_spec.exact_and_gates,
                "known_baseline": {
                    "name": ke.best_name,
                    "and_gates": ke.best_and_gates,
                    "degree": ke.best_degree,
                    "eval_p": km["eval_p"],
                    "eval_bias": km["eval_bias"],
                    "z_score": km["z_score"],
                },
                "best_proxy": {
                    "name": best_c.name,
                    "method": best_c.method,
                    "and_gates": best_c.and_gates,
                    "xor_gates": best_c.xor_gates,
                    "degree": int(deg_val),
                    "n_eval": bm["n_eval"],
                    "eval_p": bm["eval_p"],
                    "eval_bias": bm["eval_bias"],
                    "corr_r": bm["corr_r"],
                    "z_score": bm["z_score"],
                    "wilson_ci999": bm["wilson_ci999"],
                    "p_adj_bonferroni": bm["p_adj_bonferroni"],
                    "n_trials": bm["n_trials"],
                    "escalated": r_item["escalated"],
                },
                "incremental_vs_known": inc,
                "random_arx_control": {
                    "ctrl_biases": ctrl_biases,
                    "max_ctrl_bias": max_ctrl_bias,
                    "z_vs_ctrl": float(z_vs_ctrl),
                },
                "method_search_best": r_item["method_search_best"],
            }
            summary_records.append(rec)

            key_str = f"{family}_R{rounds}_{tname}"
            top_circuits_records[key_str] = [
                {
                    **c.to_dict(var_names),
                    "eval_metrics": m,
                }
                for c, m in zip(r_item["top_circuits"], r_item["top_metrics"])
            ]

            # Run Z3 checks on:
            # 1) All escalated / z >= 3.5 candidates
            # 2) Representative target per config (bit_0 and bit_1)
            run_z3 = (bm["z_score"] >= 3.5) or (tname.endswith("[0]") or tname.endswith("[1]"))
            if run_z3:
                comp_chk = verify_circuit_complexity_and_anf(best_c)
                neq_chk = verify_nonequivalence_against_known(
                    best_c, ke.circuits, ke.extra_full_linear
                )
                z3_records["circuit_checks"].append(
                    {
                        "config": key_str,
                        "complexity_anf": comp_chk,
                        "nonequivalence": neq_chk,
                    }
                )

            if family in ("C4", "C8") and (tname.endswith("[0]") or tname.endswith("[1]")):
                c_exact = verify_family_c_exact_count(
                    spec=real_spec,
                    target_name=tname,
                    proxy=best_c,
                    n_eff=8 if mode == "smoke" else 10,
                    mc_samples=8192 if mode == "smoke" else 32768,
                    seed=2026 + cfg_idx,
                )
                z3_records["family_c_exact_counts"].append(c_exact)

            # Log notable findings immediately
            if bm["z_score"] >= 4.0 or inc["delta_bias"] > 0.002:
                log_msg(
                    f"  [SIGNAL] {key_str}: p={bm['eval_p']:.5f} bias={bm['eval_bias']:.5f} "
                    f"z={bm['z_score']:.2f} AND={best_c.and_gates} deg={deg_val} | "
                    f"known={ke.best_name}({ke.best_bias:.5f}) delta={inc['delta_bias']:+.5f} "
                    f"z_inc={inc['z_paired']:.2f} ctrl_max={max_ctrl_bias:.5f} z_ctrl={z_vs_ctrl:.2f}"
                )

        # Checkpoint incremental JSON after each config
        _write_outputs(out_dir, mode, hp, device, t_start, summary_records, top_circuits_records, z3_records)
        log_msg(f"  Completed {family} R={rounds} in {time.time() - cfg_t0:.1f}s (total elapsed {time.time() - t_start:.1f}s)")

    log_msg(f"P9 Experiment finished in {time.time() - t_start:.1f}s | configs={len(summary_records)}")


def _write_outputs(
    out_dir: Path,
    mode: str,
    hp: Dict[str, Any],
    device: torch.device,
    t_start: float,
    summary_records: List[Dict[str, Any]],
    top_circuits_records: Dict[str, Any],
    z3_records: Dict[str, Any],
) -> None:
    meta = {
        "mode": mode,
        "device": str(device),
        "elapsed_seconds": round(time.time() - t_start, 2),
        "hyperparams": hp,
        "total_targets_evaluated": len(summary_records),
    }
    (out_dir / "search_summary.json").write_text(
        json.dumps({"meta": meta, "records": summary_records}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "top_circuits.json").write_text(
        json.dumps(top_circuits_records, indent=2),
        encoding="utf-8",
    )
    (out_dir / "z3_verification.json").write_text(
        json.dumps(z3_records, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="P9 ARX Proxy Circuit Discovery")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--out_dir", type=str, default="v11_discovery/p9_arx_proxy_circuit/results_smoke")
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()
    run_experiment(mode=args.mode, out_dir=Path(args.out_dir), force_cpu=args.force_cpu)


if __name__ == "__main__":
    main()
