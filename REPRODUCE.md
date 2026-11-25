# Reproduction & Independent Verification Guide (`REPRODUCE.md`)

> **Zero-GPU Auditability First**: Every empirical claim, SAT collision witness, Bitcoin header counterexample, Z3 bound, and Lean 4 formal theorem in **Phase-SHA (`V1–V12`)** can be independently verified on a standard laptop CPU (`CUDA_VISIBLE_DEVICES=""`) without heavy model checkpoints or cloud GPUs.

---

## 1. Ten-Second Zero-Dependency CPU Verification (无需显卡、10 秒复验核心结论)

Python `>= 3.10` standard library only (optional PyTorch CPU adds `src/sha256_ref.py` cross-check):

```bash
git clone https://github.com/CheeseGhostfox/Phase-SHA-Cryptanalysis-Benchmark.git
cd Phase-SHA-Cryptanalysis-Benchmark
CUDA_VISIBLE_DEVICES="" python -u scripts/verify_all_claims.py
```

This standalone script (`scripts/verify_all_claims.py`) verifies:
1. **FIPS 180-4 & Bitcoin Genesis Parity**: pure-Python SHA-256 against `hashlib.sha256` across 23 test vectors + Bitcoin Genesis Block double-SHA256 (`000000000019d668...`) + `src/sha256_ref.py` CPU cross-check (if PyTorch is present).
2. **P13 SAT Witnesses (`50/50`)**: bit-exact verification of all 50 witness message pairs in `v12_frontier/track_b1_p13_local_coll/results/p13_witnesses.json`, including `ladder_R16_standard` (`R=16`, standard SHA-256 IV, exact 256-bit `delta_H_hw = 0`, G2-nontrivial), `S3_double_*`, and `G4_CTRL_*`.
3. **P12 Bitcoin Header Witnesses (`16/16`)**: verifies `4/4` 80-byte Bitcoin witness headers across all 4 cipher tracks in `v12_frontier/track_a_p12_amortized/results/a2_gnbc_search.json`, confirming shared Chunk 1 + distinct Nonce + holdout net loss invariants (`main_a2_1_p_neutral_t5 == 0.0`, `main_best_net_gain_vs_classic_pct < 0`).
4. **Zero-`sorry` Formal Corpus Audit**: scans all `32` `.lean` files (`483` `theorem`/`lemma` declarations: `338` public + `145` private/protected) across all 4 Lean 4 projects and enforces zero `sorry`, zero `admit`, and zero custom axioms.
5. **Verdict JSON Integrity**: parses all V11 (`P1–P11`) and V12 (`P12, P13, P15`) `verdict.json` artifacts.

---

## 2. Lean 4.36.0 Formal Proof Verification (零 Mathlib、低内存一键编译)

All four formal projects use **Lean `v4.36.0`** with **zero Mathlib dependencies**, requiring `<= 2 GB` RAM (`-j 2 -M 2048`):

```bash
# Quick combined check (builds P3 & P15 and runs #print axioms audits)
CUDA_VISIBLE_DEVICES="" python -u scripts/verify_all_claims.py --run-lean

# Or build each of the 4 Lean 4 trees manually:
(cd formal_proofs && lake build)
(cd formal_proofs_v10 && lake build)
(cd v11_discovery/p3_symbolic_lean4/lean_verify && lake build && lake env lean VerifyAxioms.lean)
(cd v12_frontier/track_b2_p15_mitm_biclique/lean_verify && lake build && lake env lean VerifyAxioms.lean)
```

Expected `#print axioms` output: only standard Lean kernel axioms `[propext, Classical.choice, Lean.ofReduceBool, Lean.trustCompiler, Quot.sound]` — **zero `sorryAx`**.

| Corpus | Location | `.lean` Files | `theorem`/`lemma` (Public / All) | Highlight Coverage |
| :--- | :--- | :---: | :---: | :--- |
| V3/V4 Core | `formal_proofs/` | 12 | 180 / 313 | ARX primitives, exact carry decomposition, round invertibility |
| V10 Corpus | `formal_proofs_v10/` | 8 | 47 / 56 | 45+ round-reduced differential & non-interference theorems |
| V11 **P3** | `v11_discovery/p3_symbolic_lean4/lean_verify/` | 7 | 67 / 67 (28 target IDs) | Z3 V1–V7 audited 5–12r & BTC double-SHA256 splice identities |
| V12 **P15** | `v12_frontier/track_b2_p15_mitm_biclique/lean_verify/` | 5 | 44 / 47 (32 audited) | Stage 1A 20–32r MitM, 64r `L=6/8` Biclique, BTC Nonce `W[16..17]` decoupling |
| **Total** | **All 4 Trees** | **32** | **338 public / 483 total** | **0 `sorry`, 0 custom `axiom`** |

---

## 3. Re-running Empirical Experiments (V11 & V12 Pure-FP32 Determinism)

### Precision & Anti-Illusion Rules
All GPU runners lock IEEE-754 **pure FP32** (`torch.backends.cuda.matmul.allow_tf32 = False`, `cudnn.allow_tf32 = False`, `set_float32_matmul_precision("highest")`) and pass **5–7 anti-illusion gates**:
1. Exact reference oracle parity (`src/sha256_ref.py` vs `hashlib` + BTC Genesis).
2. No last-round linear injection cheating (`R - t_min >= 8` / G2 span checks).
3. Spectrum-aligned isomorphic random control ciphers (`CTRL_RAND_ROTR_K` / `G4_CTRL`).
4. Zero train/holdout overlap (`overlap = 0`, deterministic seed separation).
5. Full-cost amortized accounting (`Cost_setup` + `C_search` included; greedy active-word 1-bit recovery G3 check defeating forward-manifold membership confounders).

### Entry-Point Index for Full Re-runs
- **V11 P1–P9**: `v11_discovery/p1_infonce/`, ..., `p9_arx_proxy_circuit/` (each folder contains `SPEC.md`, runner `.py`, `REPORT.md`, `verdict.json`, and `results/`).
- **V11 P10 (Ugly Algebra & Koopman)**: `python v11_discovery/p10_ugly_algebra/run_p10_experiment.py` (GPU optional; reproduces `results/p10_results.json`).
- **V11 P11 (Exact MPO, $2^{16}$ SVD & Cayley Disentangler)**: `python v11_discovery/p11_tensor_network/run_p11.py` (exact SVD runs cleanly on CPU or T4 GPU; reproduces `results/p11_results.json`).
- **V12 Track A · P12 (GE Calibration, GNBC Search & Z3/PySAT Bounds)**:
  - CPU unit tests & exact GE calibration: `pytest v12_frontier/track_a_p12_amortized/test_p12_engine.py`
  - Exact SAT/Z3 bounds: `python v12_frontier/track_a_p12_amortized/sat_amortized_bounds.py`
  - Full GPU holdout search: `python v12_frontier/track_a_p12_amortized/gnbc_gpu_search.py`
- **V12 Track B1 · P13 (Mendel State Machine, Crystal MCTS & CaDiCaL195)**:
  - CPU Mendel & SAT parity tests + full runner: `python v12_frontier/track_b1_p13_local_coll/run_p13_experiments.py`
- **V12 Track B2 · P15 (CP-SAT 4-Color MitM/Biclique + Z3 Audit)**:
  - CPU engine unit tests: `python v12_frontier/track_b2_p15_mitm_biclique/test_step1_engine.py`
  - Full Stage 1 & Stage 2 sweep (8-vCPU / CPU-only, ~5 min): `python v12_frontier/track_b2_p15_mitm_biclique/run_p15_search.py`

---

## 4. Scientific Scope & Non-Overclaiming Notice (科学声明与不越界边界)

This repository **does NOT claim** an unconditional proof that polynomial-time shortcuts to SHA-256 cannot exist (which would imply $\text{P} \neq \text{NP}$). Rather, every negative bound is rigorously scoped to its exact computational, algebraic, or statistical model (explicit Lean 4 bit-vector kernel, Z3 / PySAT CNF clauses, CP-SAT word-level independence, exact $2^{16}$ bipartition SVD, or pure-FP32 empirical protocols over fixed sample budgets).
