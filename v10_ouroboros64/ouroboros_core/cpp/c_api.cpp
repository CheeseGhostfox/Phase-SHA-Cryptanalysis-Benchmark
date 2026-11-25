/**
 * C ABI wrapper for Ouroboros-64 C++17 OpenMP Spacetime Crystal Engine.
 * Callable via ctypes on Windows/Linux without requiring pybind11 build headers.
 */

#include "crystal_core.hpp"
#include <cstdint>

#if defined(_WIN32)
#define OURO_API extern "C" __declspec(dllexport)
#else
#define OURO_API extern "C" __attribute__((visibility("default")))
#endif

using namespace ouroboros;

OURO_API int ouro_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

OURO_API void ouro_sha256_compress_batch(
    uint32_t B,
    uint32_t R,
    const uint32_t* w0_15,  // (B, 16)
    uint32_t* H_out,        // (B, 8)
    uint32_t* W_out,        // (B, R) nullable
    uint32_t* S_out,        // (B, R+1, 8) nullable
    uint32_t* C_out         // (B, R, 10) nullable
) {
    const uint64_t s_stride = static_cast<uint64_t>(R + 1u) * 8u;
    const uint64_t c_stride = static_cast<uint64_t>(R) * C_PLANES;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (B >= 32)
#endif
    for (int64_t bi = 0; bi < static_cast<int64_t>(B); ++bi) {
        const uint64_t b = static_cast<uint64_t>(bi);
        sha256_compress_exact(
            R,
            w0_15 + b * 16u,
            H_out ? (H_out + b * 8u) : nullptr,
            W_out ? (W_out + b * R) : nullptr,
            S_out ? (S_out + b * s_stride) : nullptr,
            C_out ? (C_out + b * c_stride) : nullptr
        );
    }
}

OURO_API void ouro_eval_crystal_batch(
    uint32_t B,
    uint32_t R,
    const uint32_t* H_target,
    const uint32_t* W,
    const uint32_t* S,
    const uint32_t* C,
    uint32_t* delta_S,
    uint32_t* delta_W,
    uint32_t* delta_C,
    uint64_t* out_E,
    uint64_t* out_dS,
    uint64_t* out_dW,
    uint64_t* out_dC,
    uint32_t alpha,
    uint32_t beta,
    uint32_t gamma
) {
    eval_crystal_batch(
        B, R, H_target, W, S, C,
        delta_S, delta_W, delta_C,
        out_E, out_dS, out_dW, out_dC,
        alpha, beta, gamma
    );
}

OURO_API void ouro_eval_cluster_flips_batch(
    uint32_t B_cryst,
    uint32_t K_cands,
    uint32_t M_max,
    uint32_t R,
    const uint32_t* H_target,
    const uint32_t* W_base,
    const uint32_t* S_base,
    const uint32_t* C_base,
    const int32_t* cluster_idx,
    int64_t* out_delta_E,
    uint64_t* out_new_E,
    uint32_t alpha,
    uint32_t beta,
    uint32_t gamma,
    uint32_t allow_boundary_flip
) {
    eval_cluster_flips_batch(
        B_cryst, K_cands, M_max, R,
        H_target, W_base, S_base, C_base,
        cluster_idx, out_delta_E, out_new_E,
        alpha, beta, gamma,
        allow_boundary_flip != 0u
    );
}

// Run coupled Plane A (GF2 RREF) + Plane B (exact bidirectional unit propagation) to fixpoint
OURO_API void ouro_propagate_symbols(
    uint32_t R,
    const uint32_t* H_target,  // (8,)
    uint8_t* W_sym,            // (R, 32) in-place updated
    uint8_t* S_sym,            // (R+1, 8, 32) in-place updated
    uint8_t* C_sym,            // (R, 10, 32) in-place updated
    uint32_t use_gf2,
    uint32_t max_iters,
    uint32_t* out_stats        // [conflict, unit_prop_deductions, gf2_deductions, gf2_rank, gf2_conflicts, unk_after]
) {
    CrystalPropagator prop;
    prop.init_from_symbols(R, H_target, W_sym, S_sym, C_sym, use_gf2 != 0u);
    prop.propagate_to_fixpoint(use_gf2 != 0u, max_iters);
    std::memcpy(W_sym, prop.W_sym.data(), prop.W_sym.size());
    std::memcpy(S_sym, prop.S_sym.data(), prop.S_sym.size());
    std::memcpy(C_sym, prop.C_sym.data(), prop.C_sym.size());
    if (out_stats) {
        out_stats[0] = prop.conflict ? 1u : 0u;
        out_stats[1] = prop.unit_prop_deductions;
        out_stats[2] = prop.gf2_deductions;
        out_stats[3] = use_gf2 ? prop.gf2.rank : 0u;
        out_stats[4] = use_gf2 ? prop.gf2.num_conflicts : 0u;
        out_stats[5] = prop.count_unknown_primary();
    }
}

// Standalone GF(2) RREF solver API for direct algebraic linear-system unit tests & custom equations
OURO_API void ouro_gf2_solve_equations(
    uint32_t num_vars,
    uint32_t num_eqs,
    const uint32_t* eq_offsets,  // (num_eqs + 1,) CSR offsets into eq_cols
    const uint32_t* eq_cols,
    const uint8_t* eq_rhs,       // (num_eqs,)
    uint32_t* out_unit_cols,     // capacity >= num_vars
    uint8_t* out_unit_vals,      // capacity >= num_vars
    uint32_t* out_deg,           // capacity >= num_vars (nullable)
    uint32_t* out_meta           // [conflict, rank, num_units]
) {
    GF2RREF gf2;
    gf2.init(num_vars);
    for (uint32_t i = 0; i < num_eqs; ++i) {
        uint32_t s = eq_offsets[i];
        uint32_t e = eq_offsets[i + 1u];
        gf2.add_equation(eq_cols + s, e - s, eq_rhs[i]);
    }
    std::vector<uint32_t> ucols;
    std::vector<uint8_t> uvals;
    gf2.extract_unit_deductions(ucols, uvals);
    if (!ucols.empty()) {
        std::memcpy(out_unit_cols, ucols.data(), ucols.size() * sizeof(uint32_t));
        std::memcpy(out_unit_vals, uvals.data(), uvals.size() * sizeof(uint8_t));
    }
    if (out_deg) {
        gf2.var_constraint_degrees(out_deg);
    }
    if (out_meta) {
        out_meta[0] = gf2.conflict ? 1u : 0u;
        out_meta[1] = gf2.rank;
        out_meta[2] = static_cast<uint32_t>(ucols.size());
    }
}
