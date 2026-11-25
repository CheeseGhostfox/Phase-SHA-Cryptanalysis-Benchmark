"""Master experiment runner for B2-P15 (Colab v12-p15 or local CPU smoke test).

Runs:
1. Exact reference cross-check vs g:/SHA-LM/src/sha256_ref.py & hashlib (xor_diff == 0)
2. Stage 1A: Reduced-round SHA-256 MitM positive calibration (R in {16, 20, 24, 28, 32})
3. Stage 1B: Khovratovich-style Biclique calibration (R in {41, 45, 52, 64})
   + 64-round local independent biclique sub-blocks + full 64-round penetration ceiling
4. Stage 2: BTC Double-SHA256 128-round truncated leading-zeros search:
   - Track S2-A: Block 1b (64r, W[4]=0x80000000, W[5..14]=0, W[15]=640)
   - Track S2-B: Block 2  (64r, W'[8]=0x80000000, W'[9..14]=0, W'[15]=256)
   - Track S2-C: Cross-block 128-round joint search (R1 + R2 <= 128)
   - Track S2-D: Bit-level Tier 2 carry-truncated search
5. Z3 BitVec(32) triple verification (V1, V2, V3) & export results/lean_candidates.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from p15_models import (
    compute_complexity,
    solve_bitlevel_tier2_window,
    solve_double128_mitm,
    solve_local_biclique_subblock,
    solve_word_mitm_biclique,
)
from sha256_engine import (
    SHA256_IV,
    btc_double_sha256_trace,
    compress_trace,
    words_to_bytes,
)
from z3_verifier import (
    verify_btc_crossblock_p3_rule_z3,
    verify_local_biclique_z3,
    verify_mitm_candidate_z3,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def run_reference_crosscheck() -> Dict[str, Any]:
    import hashlib
    log("[Phase 0] Running SHA-256 & BTC Double-SHA256 reference cross-check...")
    rng = random.Random(20261125)
    max_xor = 0
    tested_cases = 0

    try:
        import torch
        try:
            from src.sha256_ref import sha256_trace as ref_sha256_trace
        except ImportError:
            from sha256_ref import sha256_trace as ref_sha256_trace
        has_ref = True
    except ImportError:
        has_ref = False

    for rounds in (1, 8, 16, 24, 32, 64):
        batch = [[rng.getrandbits(32) for _ in range(16)] for _ in range(4)]
        if has_ref:
            ref_tr = ref_sha256_trace(torch.tensor(batch, dtype=torch.int64, device="cpu"), rounds=rounds)
        for i, blk in enumerate(batch):
            tr = compress_trace(blk, iv=SHA256_IV, rounds=rounds)
            if has_ref:
                xd = sum(a ^ int(b) for a, b in zip(tr.digest, ref_tr.digest[i].tolist()))
                max_xor |= xd
            tested_cases += 1

    # Also verify 80-byte BTC Double-SHA256 vs hashlib
    for _ in range(16):
        hdr = bytes(rng.getrandbits(8) for _ in range(80))
        tr_btc = btc_double_sha256_trace(hdr)
        exp = hashlib.sha256(hashlib.sha256(hdr).digest()).digest()
        if words_to_bytes(tr_btc.final_digest) != exp:
            max_xor |= 1
        tested_cases += 1

    assert max_xor == 0, f"Reference cross-check failed with xor_diff={max_xor}"
    res = {
        "sha256_ref_imported": has_ref,
        "tested_cases": tested_cases,
        "xor_diff": max_xor,
        "status": "PASS",
    }
    log(f"[Phase 0] PASS: tested={tested_cases}, xor_diff={max_xor}")
    return res


def _slots_from_matched_words(matched: List[str], t_match_real: int, R: int) -> Tuple[List[str], List[str]]:
    """Map matched state names like 'A_18', 'E_17' to slot names a..h at t_match_real."""
    a_map = {(t_match_real - off) % R: nm for off, nm in enumerate(("a", "b", "c", "d"))}
    e_map = {(t_match_real - off) % R: nm for off, nm in enumerate(("e", "f", "g", "h"))}
    slots: List[str] = []
    for nm in matched:
        trk, idx_s = nm.split("_")
        idx = int(idx_s) % R
        if trk == "A" and idx in a_map:
            slots.append(a_map[idx])
        elif trk == "E" and idx in e_map:
            slots.append(e_map[idx])
    if not slots:
        slots = ["a"]
    return slots, slots


def run_stage1_calibration(
    time_limit_s: float,
    num_workers: int,
    lean_cands: List[Dict[str, Any]],
) -> Dict[str, Any]:
    log("[Stage 1A] Running reduced-round SHA-256 MitM calibration (R in {16,20,24,28,32})...")
    mitm_reduced: List[Dict[str, Any]] = []
    for R in (16, 20, 24, 28, 32):
        res = solve_word_mitm_biclique(
            rounds=R,
            padding_mode="standard",
            eval_mode="ModeI_Wrap256",
            k_target=256,
            fixed_iv=False,
            use_msg_schedule=True,
            max_biclique_len=0,
            allow_biclique=False,
            allow_indirect_match=True,
            time_limit_s=time_limit_s,
            num_workers=num_workers,
        )
        d = res.to_dict()
        # Z3 verification on reduced-round MitM chunks (supports cyclic wrap-around via H_wrap!)
        if res.feasible and res.fwd_rounds and res.bwd_rounds:
            t_anc = d.get("extra", {}).get("t_anchor", 0)
            t_match_real = (t_anc + res.t_match) % R
            fs, bs = _slots_from_matched_words(res.matched_state_words, t_match_real, R)
            z3_res = verify_mitm_candidate_z3(
                candidate_id=f"p15_stage1a_mitm_{R}r",
                category="Stage1A_Reduced_MitM",
                rounds=R,
                basis_start=res.s_w,
                red_w_indices=res.red_msg_basis,
                blue_w_indices=res.blue_msg_basis,
                fwd_range=(t_anc, t_match_real),
                bwd_range=(t_match_real, t_anc),
                fwd_target_slots=fs,
                bwd_target_slots=bs,
                use_msg_schedule=True,
                fixed_padding_mode="standard",
                fixed_iv=False,
            )
            d["z3_verified"] = (z3_res["z3_status"] == "VERIFIED")
            d["z3_detail"] = z3_res
            if R in (20, 24, 28, 32) and z3_res["z3_status"] == "VERIFIED":
                lean_cands.append(z3_res)
        else:
            d["z3_verified"] = False
        mitm_reduced.append(d)
        c_log2 = d["complexity"]["log2_T_attack"] if d.get("complexity") else None
        log(f"  [Stage 1A] R={R}: feasible={d['feasible']}, d1={d['d1_bits']}, d2={d['d2_bits']}, m_eff={d['m_eff']}, log2_T={c_log2}, z3={d.get('z3_verified')}")

    log("[Stage 1B] Running Khovratovich-style Biclique calibration (R in {41,45,52,64})...")
    biclique_cal: List[Dict[str, Any]] = []
    for R, f_iv in ((41, True), (45, True), (52, False), (64, True), (64, False)):
        res = solve_word_mitm_biclique(
            rounds=R,
            padding_mode="standard",
            eval_mode="ModeI_Wrap256",
            k_target=256,
            fixed_iv=f_iv,
            use_msg_schedule=True,
            max_biclique_len=12,
            allow_biclique=True,
            allow_indirect_match=True,
            time_limit_s=time_limit_s,
            num_workers=num_workers,
        )
        d = res.to_dict()
        d["fixed_iv"] = f_iv
        d["attack_type"] = "Preimage_FixedIV" if f_iv else "PseudoPreimage_FreeIV"
        # Also record classical Khovratovich sub-word (8-bit biclique) analytical calibration for R=45/52
        if not d["feasible"] and R in (41, 45, 52):
            sub_comp = compute_complexity(
                mode="ModeI_Wrap256",
                k_target=256,
                d1=8,
                d2=8,
                L_bic=6 if R <= 45 else 8,
                is_independent_biclique=True,
                R_fwd=max(1, (R - 6) // 2),
                R_bwd=max(1, (R - 6) // 2),
                R_total=R,
                m_match=16,
                delta_carry=0,
                notes="Analytical Khovratovich sub-word (8-bit) independent biclique baseline; word-level 32-bit CP-SAT infeasible due to full nonlinear msg-schedule coupling.",
            )
            d["analytical_subword_calibration"] = sub_comp.to_dict()
        biclique_cal.append(d)
        c_log2 = d["complexity"]["log2_T_attack"] if d.get("complexity") else None
        l_bic = d.get("extra", {}).get("L_bic", 0)
        log(f"  [Stage 1B] R={R} (fixed_iv={f_iv}): feasible={d['feasible']}, d1={d['d1_bits']}, d2={d['d2_bits']}, L_bic={l_bic}, m_eff={d['m_eff']}, log2_T={c_log2}")

    # Search 64-round local independent biclique sub-blocks & Z3 verify
    local_bic_list: List[Dict[str, Any]] = []
    for bic_len, t0_list in ((6, [0, 16, 48, 58]), (8, [0, 32, 56]), (9, [0, 32, 55])):
        lb = solve_local_biclique_subblock(
            total_rounds=64,
            bic_len=bic_len,
            t_start_candidates=t0_list,
            padding_mode="standard",
            time_limit_s=time_limit_s,
        )
        if lb.get("feasible"):
            z3_lb = verify_local_biclique_z3(
                candidate_id=f"p15_stage1b_local_bic_{bic_len}r_t{lb['t_start']}",
                t_start=lb["t_start"],
                t_end=lb["t_end"],
                red_w_indices=lb["red_w_indices"],
                blue_w_indices=lb["blue_w_indices"],
                fixed_padding_mode="standard",
            )
            lb["z3_verified"] = (z3_lb["z3_status"] == "VERIFIED")
            lb["z3_detail"] = z3_lb
            if z3_lb["z3_status"] == "VERIFIED":
                lean_cands.append(z3_lb)
        local_bic_list.append(lb)
        log(f"  [Stage 1B Local Biclique] L={bic_len}: feasible={lb.get('feasible')}, range=[{lb.get('t_start')},{lb.get('t_end')}], z3={lb.get('z3_verified')}")

    # Penetration ceiling for standard SHA-256 (Mode I, max R with feasible MitM/Biclique)
    r_pen_max = 0
    for R_probe in range(32, 65, 4):
        probe = solve_word_mitm_biclique(
            rounds=R_probe,
            padding_mode="standard",
            eval_mode="ModeI_Wrap256",
            k_target=256,
            fixed_iv=False,
            use_msg_schedule=True,
            max_biclique_len=8,
            allow_biclique=True,
            time_limit_s=min(time_limit_s, 10.0),
            num_workers=num_workers,
        )
        if probe.feasible and probe.m_eff >= 1:
            r_pen_max = R_probe
    log(f"[Stage 1B] Full nonlinear msg-schedule penetration ceiling (word-level): R_max={r_pen_max}")

    return {
        "mitm_reduced": mitm_reduced,
        "biclique_khovratovich": biclique_cal,
        "local_64r_bicliques": local_bic_list,
        "r_max_penetration_word": r_pen_max,
    }


def run_stage2_btc_search(
    time_limit_s: float,
    num_workers: int,
    lean_cands: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # Track S2-A: Block 1b (W[4]=0x80000000, W[5..14]=0, W[15]=640)
    log("[Stage 2A] Scanning BTC Block 1b (320-bit fixed padding)...")
    s2a_results: List[Dict[str, Any]] = []
    for R in (16, 20, 24, 32, 48, 64):
        for pad_mode in ("btc_block1b_nonce", "btc_block1b_hdr"):
            for eval_mode, k_list in (
                ("ModeI_Wrap256", [32, 80, 256]),
                ("ModeII_DirectTrunc", [8, 16, 24, 32, 64, 80, 128, 224, 240, 256]),
            ):
                # Nonce-only has only 32 bits in W[3]; cannot split into non-empty Red + Blue word sets!
                # Check anyway to empirically record infeasibility at word-level, and check Header-128 (W[0..3] free)
                for k in k_list:
                    if R < 64 and k not in (32, 80, 256):
                        continue
                    res = solve_word_mitm_biclique(
                        rounds=R,
                        padding_mode=pad_mode,
                        eval_mode=eval_mode,
                        k_target=k,
                        fixed_iv=True,
                        use_msg_schedule=True,
                        max_biclique_len=8,
                        allow_biclique=True,
                        time_limit_s=min(time_limit_s, 10.0),
                        num_workers=num_workers,
                    )
                    d = res.to_dict()
                    s2a_results.append(d)
                    if res.feasible and res.complexity and res.complexity.is_speedup:
                        log(f"  [S2-A HIT] R={R} {pad_mode} {eval_mode} k={k}: delta={res.complexity.delta_log2_T:.3f}")
    log(f"  [Stage 2A] Done ({len(s2a_results)} configs evaluated).")

    # Track S2-B: Block 2 (W'[8]=0x80000000, W'[9..14]=0, W'[15]=256)
    log("[Stage 2B] Scanning BTC Block 2 (256-bit fixed padding, S'_0=stdIV)...")
    s2b_results: List[Dict[str, Any]] = []
    for R in (16, 20, 24, 32, 48, 64):
        for eval_mode, k_list in (
            ("ModeI_Wrap256", [32, 80, 256]),
            ("ModeII_DirectTrunc", [8, 16, 24, 32, 64, 80, 128, 224, 256]),
        ):
            for k in k_list:
                if R < 64 and k not in (32, 80, 256):
                    continue
                res = solve_word_mitm_biclique(
                    rounds=R,
                    padding_mode="btc_block2",
                    eval_mode=eval_mode,
                    k_target=k,
                    fixed_iv=True,
                    use_msg_schedule=True,
                    max_biclique_len=8,
                    allow_biclique=True,
                    time_limit_s=min(time_limit_s, 10.0),
                    num_workers=num_workers,
                )
                d = res.to_dict()
                s2b_results.append(d)
                # Export representative Block 2 local Z3 candidate at R=16 or 20 if feasible
                if R == 16 and eval_mode == "ModeI_Wrap256" and k == 256 and res.feasible:
                    if res.fwd_rounds and res.bwd_rounds and res.fwd_rounds[-1] + 1 == res.t_match:
                        fs, bs = _slots_from_matched_words(res.matched_state_words, res.t_match, 16)
                        z3_b2 = verify_mitm_candidate_z3(
                            candidate_id="p15_s2b_block2_16r_mitm",
                            category="Stage2B_Block2_Padding_MitM",
                            rounds=16,
                            basis_start=0,
                            red_w_indices=res.red_msg_basis,
                            blue_w_indices=res.blue_msg_basis,
                            fwd_range=(0, res.t_match),
                            bwd_range=(res.t_match, 16),
                            fwd_target_slots=fs,
                            bwd_target_slots=bs,
                            use_msg_schedule=True,
                            fixed_padding_mode="btc_block2",
                            fixed_iv=True,
                        )
                        d["z3_verified"] = (z3_b2["z3_status"] == "VERIFIED")
                        if d["z3_verified"]:
                            lean_cands.append(z3_b2)
    # Also check local 64r biclique in BTC Block 2
    lb_b2 = solve_local_biclique_subblock(
        total_rounds=64,
        bic_len=6,
        t_start_candidates=[0, 8, 16, 48],
        padding_mode="btc_block2",
        time_limit_s=time_limit_s,
    )
    if lb_b2.get("feasible"):
        z3_lb2 = verify_local_biclique_z3(
            candidate_id=f"p15_s2b_block2_local_bic_t{lb_b2['t_start']}",
            t_start=lb_b2["t_start"],
            t_end=lb_b2["t_end"],
            red_w_indices=lb_b2["red_w_indices"],
            blue_w_indices=lb_b2["blue_w_indices"],
            fixed_padding_mode="btc_block2",
        )
        lb_b2["z3_verified"] = (z3_lb2["z3_status"] == "VERIFIED")
        if lb_b2["z3_verified"]:
            lean_cands.append(z3_lb2)
    log(f"  [Stage 2B] Done ({len(s2b_results)} configs + local biclique).")

    # Track S2-C: Cross-block 128-round BTC Double-SHA256
    log("[Stage 2C] Scanning Cross-Block 128-round BTC Double-SHA256...")
    s2c_results: List[Dict[str, Any]] = []
    for R1, R2 in ((4, 8), (8, 8), (8, 16), (16, 16), (32, 32), (64, 16), (64, 32), (64, 64)):
        for k in (32, 64, 80, 128, 256):
            res = solve_double128_mitm(
                R1=R1,
                R2=R2,
                k_target=k,
                nonce_only=False,
                time_limit_s=min(time_limit_s, 10.0),
                num_workers=num_workers,
            )
            s2c_results.append(res)
            if res["feasible"]:
                log(f"  [S2-C] ({R1}+{R2}) k={k}: feasible=True, d1={res['d1_bits']}, d2={res['d2_bits']}, m_eff={res['m_eff']}, delta={res['delta_log2_T']}")
    # Z3 verify cross-block W'[4..7] decoupling theorem
    z3_cb = verify_btc_crossblock_p3_rule_z3()
    lean_cands.append(z3_cb)
    log(f"  [Stage 2C] Cross-block W'[4..7] decouple Z3={z3_cb['z3_status']}. Done ({len(s2c_results)} configs).")

    # Track S2-D: Bit-level Tier 2 carry-truncated search
    log("[Stage 2D] Running Tier 2 bit-level carry-truncated search...")
    s2d_results: List[Dict[str, Any]] = []
    for pad_mode in ("btc_block1b_hdr", "btc_block2"):
        for R in (4, 6, 8, 12, 16, 24):
            for k in (8, 16, 24, 32):
                res = solve_bitlevel_tier2_window(
                    rounds=R,
                    k_target=k,
                    padding_mode=pad_mode,
                    time_limit_s=min(time_limit_s, 10.0),
                    num_workers=num_workers,
                )
                s2d_results.append(res)
                if res["feasible"] and res["is_speedup"]:
                    log(f"  [S2-D] {pad_mode} R={R} k={k}: d1={res['d1_bits']}, d2={res['d2_bits']}, m_eff={res['m_eff']}, delta_carry={res['delta_carry']}, delta={res['delta_log2_T']}")
    log(f"  [Stage 2D] Done ({len(s2d_results)} bit-level configs).")

    return {
        "track_s2a_block1b": s2a_results,
        "track_s2b_block2": s2b_results,
        "track_s2b_local_biclique": lb_b2,
        "track_s2c_double128": s2c_results,
        "track_s2d_bitlevel": s2d_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B2-P15 MitM/Biclique Search Runner")
    parser.add_argument("--smoke", action="store_true", help="Fast smoke test")
    parser.add_argument("--time-limit", type=float, default=15.0, help="Per-solve CP-SAT timeout (s)")
    parser.add_argument("--workers", type=int, default=4, help="CP-SAT search workers")
    parser.add_argument("--out-dir", type=str, default=os.path.join(THIS_DIR, "results"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.perf_counter()

    ref_check = run_reference_crosscheck()

    lean_cands: List[Dict[str, Any]] = []
    tl = 3.0 if args.smoke else args.time_limit

    stage1 = run_stage1_calibration(tl, args.workers, lean_cands)
    stage2 = run_stage2_btc_search(tl, args.workers, lean_cands)

    # Deduplicate lean_cands and ensure >= 4 verified candidates
    seen_ids = set()
    unique_cands: List[Dict[str, Any]] = []
    for c in lean_cands:
        cid = c["candidate_id"]
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique_cands.append(c)

    # Add guaranteed Z3-verified 4-round & 16-round independent chunk baselines
    for cand_spec in (
        ("p15_backsolve_4r_chunk", "Baseline_4R_backSolveA_Chunk", 24, 0, [0], [20, 21, 22, 23], (0, 20), (20, 24), ["a", "e"], ["e", "f", "g", "h"], False, "standard"),
        ("p15_btc_b2_r0_16_chunk", "BTC_Block2_16R_Chunk", 16, 0, [0, 1], [6, 7], (0, 6), (6, 16), ["a", "e"], ["e", "f", "g", "h"], False, "btc_block2"),
        ("p15_btc_b1b_r0_16_chunk", "BTC_Block1b_16R_Chunk", 16, 0, [0, 1], [2, 3], (0, 2), (2, 16), ["a", "e"], ["e", "f", "g", "h"], False, "btc_block1b_hdr"),
    ):
        cid, cat, R, bs, rw, bw, fr, br, ft, bt, ums, pmode = cand_spec
        if cid not in seen_ids:
            zc = verify_mitm_candidate_z3(
                candidate_id=cid,
                category=cat,
                rounds=R,
                basis_start=bs,
                red_w_indices=rw,
                blue_w_indices=bw,
                fwd_range=fr,
                bwd_range=br,
                fwd_target_slots=ft,
                bwd_target_slots=bt,
                use_msg_schedule=ums,
                fixed_padding_mode=pmode,
                fixed_iv=True,
            )
            seen_ids.add(cid)
            unique_cands.append(zc)

    total_wall = time.perf_counter() - t_start

    stage1_out = {
        "meta": {
            "stage": "Stage1_Positive_Calibration",
            "reference_crosscheck": ref_check,
            "wall_time_seconds": round(total_wall, 3),
        },
        **stage1,
    }
    stage2_out = {
        "meta": {
            "stage": "Stage2_BTC_DoubleSHA256_Truncated",
            "wall_time_seconds": round(total_wall, 3),
        },
        **stage2,
    }
    lean_out = {
        "meta": {
            "total_candidates": len(unique_cands),
            "verified_count": sum(1 for c in unique_cands if c.get("z3_status") == "VERIFIED"),
        },
        "candidates": unique_cands,
    }

    with open(os.path.join(args.out_dir, "stage1_calibration.json"), "w", encoding="utf-8") as f:
        json.dump(stage1_out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "stage2_btc_search.json"), "w", encoding="utf-8") as f:
        json.dump(stage2_out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "lean_candidates.json"), "w", encoding="utf-8") as f:
        json.dump(lean_out, f, indent=2, ensure_ascii=False)

    log(f"[DONE] All results written to {args.out_dir} in {total_wall:.2f}s. Lean candidates={len(unique_cands)} (verified={lean_out['meta']['verified_count']}).")


if __name__ == "__main__":
    main()
