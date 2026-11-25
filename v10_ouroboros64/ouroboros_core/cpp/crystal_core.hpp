#pragma once
/**
 * Phase-SHA V10 "Ouroboros-64" Zero-Decoherence 64-Round Spacetime Crystal Core.
 *
 * Exact uint32/uint64 ARX physics (zero floating-point approximations):
 *   - Supports R in {1..64} (native R=64 standard SHA-256 single block)
 *   - Bit convention: LSB-first (bit b = (word >> b) & 1, b=0 is LSB, b=31 is MSB)
 *   - Hard-welded boundaries: S[0] == IV, S[R] == (H - IV) mod 2^32
 *   - Explicit carry planes C[R][10] for all modulo-2^32 additions:
 *       C[t][0..2]: W_t schedule additions (t >= 16; forced 0 for t < 16)
 *       C[t][3..6]: T1_t = (((h + S1(e)) + Ch(e,f,g)) + K_t) + W_t carries
 *       C[t][7]   : T2_t = S0(a) + Maj(a,b,c) carry
 *       C[t][8]   : a_{t+1} = T1_t + T2_t carry
 *       C[t][9]   : e_{t+1} = d + T1_t carry
 *   - Plane A: Incremental GF(2) Row-Reduced Echelon Form (RREF) Gaussian elimination
 *   - Plane B: Exact bidirectional 6-symbol {0, 1, ?, -, u, n} ARX unit propagation
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ouroboros {

static constexpr uint32_t MAX_ROUNDS = 64;
static constexpr uint32_t C_PLANES = 10;

// Generalized 6-symbol alphabet (Step 1 propagates {0, 1, ?}; reserves -, u, n for Track A MCTS)
static constexpr uint8_t SYM_0 = 0;
static constexpr uint8_t SYM_1 = 1;
static constexpr uint8_t SYM_Q = 2;     // '?' unknown free bit
static constexpr uint8_t SYM_DASH = 3;  // '-' unchanged differential bit
static constexpr uint8_t SYM_U = 4;     // 'u' 0->1 differential bit
static constexpr uint8_t SYM_N = 5;     // 'n' 1->0 differential bit / neutral marker

static constexpr uint32_t IV_WORDS[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
};

static constexpr uint32_t K_WORDS[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

inline uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

inline uint32_t big_sigma0(uint32_t x) {
    return rotr32(x, 2u) ^ rotr32(x, 13u) ^ rotr32(x, 22u);
}

inline uint32_t big_sigma1(uint32_t x) {
    return rotr32(x, 6u) ^ rotr32(x, 11u) ^ rotr32(x, 25u);
}

inline uint32_t small_sigma0(uint32_t x) {
    return rotr32(x, 7u) ^ rotr32(x, 18u) ^ (x >> 3u);
}

inline uint32_t small_sigma1(uint32_t x) {
    return rotr32(x, 17u) ^ rotr32(x, 19u) ^ (x >> 10u);
}

inline uint32_t ch_word(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ ((~x) & z);
}

inline uint32_t maj_word(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

inline uint32_t popcnt32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_popcount(x));
#else
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    return (((x + (x >> 4)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24;
#endif
}

inline uint32_t ctz64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_ctzll(x));
#else
    uint32_t n = 0;
    while ((x & 1ULL) == 0ULL) {
        x >>= 1;
        ++n;
    }
    return n;
#endif
}

inline uint32_t popcnt64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_popcountll(x));
#else
    return popcnt32(static_cast<uint32_t>(x)) + popcnt32(static_cast<uint32_t>(x >> 32));
#endif
}

// Compute exact carry word c for (x + y) mod 2^32: c[0]=0, c[b+1]=Maj(x[b], y[b], c[b])
inline uint32_t exact_add_with_carry(uint32_t x, uint32_t y, uint32_t* out_carry) {
    uint32_t sum = x + y;
    uint32_t carry = x ^ y ^ sum;
    *out_carry = carry;
    return sum;
}

// Evaluate binary adder given candidate explicit carry c:
// returns bitwise sum (x ^ y ^ c) and carry violation mask dc = c ^ ((Maj(x, y, c) << 1)).
inline uint32_t eval_add_carry(uint32_t x, uint32_t y, uint32_t c, uint32_t* out_dc) {
    uint32_t next_c = maj_word(x, y, c) << 1;
    *out_dc = c ^ next_c;
    return x ^ y ^ c;
}

// Generate exact R-round SHA-256 spacetime crystal (W, S, C) and digest H from W[0..15]
inline void sha256_compress_exact(
    uint32_t R,
    const uint32_t* w0_15,
    uint32_t* H_out,
    uint32_t* W_out,
    uint32_t* S_out,
    uint32_t* C_out
) {
    uint32_t W[MAX_ROUNDS];
    std::memset(W, 0, sizeof(W));
    for (uint32_t t = 0; t < 16u && t < R; ++t) {
        W[t] = w0_15[t];
        if (C_out) {
            for (uint32_t k = 0; k < 3u; ++k) {
                C_out[t * C_PLANES + k] = 0u;
            }
        }
    }
    for (uint32_t t = 16u; t < R; ++t) {
        uint32_t s1 = small_sigma1(W[t - 2u]);
        uint32_t s0 = small_sigma0(W[t - 15u]);
        uint32_t c0 = 0u, c1 = 0u, c2 = 0u;
        uint32_t u0 = exact_add_with_carry(s1, W[t - 7u], &c0);
        uint32_t u1 = exact_add_with_carry(u0, s0, &c1);
        uint32_t wt = exact_add_with_carry(u1, W[t - 16u], &c2);
        W[t] = wt;
        if (C_out) {
            C_out[t * C_PLANES + 0u] = c0;
            C_out[t * C_PLANES + 1u] = c1;
            C_out[t * C_PLANES + 2u] = c2;
        }
    }
    if (W_out) {
        std::memcpy(W_out, W, R * sizeof(uint32_t));
    }

    uint32_t a = IV_WORDS[0], b = IV_WORDS[1], c = IV_WORDS[2], d = IV_WORDS[3];
    uint32_t e = IV_WORDS[4], f = IV_WORDS[5], g = IV_WORDS[6], h = IV_WORDS[7];
    if (S_out) {
        for (uint32_t j = 0; j < 8u; ++j) S_out[j] = IV_WORDS[j];
    }

    for (uint32_t t = 0; t < R; ++t) {
        uint32_t S1 = big_sigma1(e);
        uint32_t ch = ch_word(e, f, g);
        uint32_t c3 = 0u, c4 = 0u, c5 = 0u, c6 = 0u, c7 = 0u, c8 = 0u, c9 = 0u;
        uint32_t p1 = exact_add_with_carry(h, S1, &c3);
        uint32_t p2 = exact_add_with_carry(p1, ch, &c4);
        uint32_t p3 = exact_add_with_carry(p2, K_WORDS[t], &c5);
        uint32_t T1 = exact_add_with_carry(p3, W[t], &c6);

        uint32_t S0 = big_sigma0(a);
        uint32_t mj = maj_word(a, b, c);
        uint32_t T2 = exact_add_with_carry(S0, mj, &c7);

        uint32_t a_new = exact_add_with_carry(T1, T2, &c8);
        uint32_t e_new = exact_add_with_carry(d, T1, &c9);

        h = g;
        g = f;
        f = e;
        e = e_new;
        d = c;
        c = b;
        b = a;
        a = a_new;

        if (S_out) {
            uint32_t* row = S_out + (t + 1u) * 8u;
            row[0] = a; row[1] = b; row[2] = c; row[3] = d;
            row[4] = e; row[5] = f; row[6] = g; row[7] = h;
        }
        if (C_out) {
            uint32_t* crow = C_out + t * C_PLANES;
            crow[3] = c3; crow[4] = c4; crow[5] = c5; crow[6] = c6;
            crow[7] = c7; crow[8] = c8; crow[9] = c9;
        }
    }

    if (H_out) {
        uint32_t s_last[8] = {a, b, c, d, e, f, g, h};
        for (uint32_t j = 0; j < 8u; ++j) {
            H_out[j] = s_last[j] + IV_WORDS[j];
        }
    }
}

// Compute residual bitmasks & defect count for a single schedule round t (0..R-1)
inline uint32_t eval_schedule_round(
    uint32_t t,
    const uint32_t* W,
    const uint32_t* C_round,
    uint32_t* out_dW,
    uint32_t* out_dc0,
    uint32_t* out_dc1,
    uint32_t* out_dc2
) {
    if (t < 16u) {
        *out_dW = 0u;
        *out_dc0 = C_round[0];
        *out_dc1 = C_round[1];
        *out_dc2 = C_round[2];
        return popcnt32(*out_dc0) + popcnt32(*out_dc1) + popcnt32(*out_dc2);
    }
    uint32_t s1 = small_sigma1(W[t - 2u]);
    uint32_t s0 = small_sigma0(W[t - 15u]);
    uint32_t u0 = eval_add_carry(s1, W[t - 7u], C_round[0], out_dc0);
    uint32_t u1 = eval_add_carry(u0, s0, C_round[1], out_dc1);
    uint32_t wt = eval_add_carry(u1, W[t - 16u], C_round[2], out_dc2);
    *out_dW = W[t] ^ wt;
    return popcnt32(*out_dc0) + popcnt32(*out_dc1) + popcnt32(*out_dc2);
}

// Compute residual bitmasks & defect counts for compression transition S[t] -> S[t+1]
inline void eval_compress_round(
    uint32_t t,
    const uint32_t* St,
    const uint32_t* St1,
    uint32_t Wt,
    const uint32_t* C_round,
    uint32_t* out_dS_row,
    uint32_t* out_dc_3_to_9,
    uint32_t* out_cnt_dS,
    uint32_t* out_cnt_dC
) {
    uint32_t a = St[0], b = St[1], c = St[2], d = St[3];
    uint32_t e = St[4], f = St[5], g = St[6], h = St[7];

    uint32_t S1 = big_sigma1(e);
    uint32_t ch = ch_word(e, f, g);
    uint32_t p1 = eval_add_carry(h, S1, C_round[3], &out_dc_3_to_9[0]);
    uint32_t p2 = eval_add_carry(p1, ch, C_round[4], &out_dc_3_to_9[1]);
    uint32_t p3 = eval_add_carry(p2, K_WORDS[t], C_round[5], &out_dc_3_to_9[2]);
    uint32_t T1 = eval_add_carry(p3, Wt, C_round[6], &out_dc_3_to_9[3]);

    uint32_t S0 = big_sigma0(a);
    uint32_t mj = maj_word(a, b, c);
    uint32_t T2 = eval_add_carry(S0, mj, C_round[7], &out_dc_3_to_9[4]);

    uint32_t a_rhs = eval_add_carry(T1, T2, C_round[8], &out_dc_3_to_9[5]);
    uint32_t e_rhs = eval_add_carry(d, T1, C_round[9], &out_dc_3_to_9[6]);

    out_dS_row[0] = St1[0] ^ a_rhs;
    out_dS_row[1] = St1[1] ^ a;
    out_dS_row[2] = St1[2] ^ b;
    out_dS_row[3] = St1[3] ^ c;
    out_dS_row[4] = St1[4] ^ e_rhs;
    out_dS_row[5] = St1[5] ^ e;
    out_dS_row[6] = St1[6] ^ f;
    out_dS_row[7] = St1[7] ^ g;

    uint32_t cs = 0u, cc = 0u;
    for (uint32_t j = 0; j < 8u; ++j) cs += popcnt32(out_dS_row[j]);
    for (uint32_t k = 0; k < 7u; ++k) cc += popcnt32(out_dc_3_to_9[k]);
    *out_cnt_dS = cs;
    *out_cnt_dC = cc;
}

// Full crystal defect energy evaluation:
//   delta_S shape: (R + 2, 8) -> idx 0: S[0]^IV; idx 1..R: transition t->t+1; idx R+1: S[R]^(H-IV)
//   delta_W shape: (R,)
//   delta_C shape: (R, 10)
// Theorem: E_total == 0 (with alpha, beta, gamma > 0) <==> SHA256_R(W[0..15]) == H AND crystal exact.
inline uint64_t eval_crystal_single(
    uint32_t R,
    const uint32_t* H_target,
    const uint32_t* W,
    const uint32_t* S,
    const uint32_t* C,
    uint32_t* delta_S,
    uint32_t* delta_W,
    uint32_t* delta_C,
    uint64_t* out_dS,
    uint64_t* out_dW,
    uint64_t* out_dC,
    uint32_t alpha = 1u,
    uint32_t beta = 1u,
    uint32_t gamma = 1u
) {
    uint64_t tot_dS = 0ULL;
    uint64_t tot_dW = 0ULL;
    uint64_t tot_dC = 0ULL;

    for (uint32_t j = 0; j < 8u; ++j) {
        uint32_t b0 = S[j] ^ IV_WORDS[j];
        if (delta_S) delta_S[j] = b0;
        tot_dS += popcnt32(b0);

        uint32_t sr_tgt = H_target[j] - IV_WORDS[j];
        uint32_t br = S[R * 8u + j] ^ sr_tgt;
        if (delta_S) delta_S[(R + 1u) * 8u + j] = br;
        tot_dS += popcnt32(br);
    }

    for (uint32_t t = 0; t < R; ++t) {
        uint32_t dw = 0u, dc0 = 0u, dc1 = 0u, dc2 = 0u;
        const uint32_t* crow = C + t * C_PLANES;
        uint32_t cc_sched = eval_schedule_round(t, W, crow, &dw, &dc0, &dc1, &dc2);
        if (delta_W) delta_W[t] = dw;
        tot_dW += popcnt32(dw);
        tot_dC += cc_sched;

        uint32_t ds_row[8];
        uint32_t dc_comp[7];
        uint32_t cs = 0u, cc_comp = 0u;
        eval_compress_round(
            t,
            S + t * 8u,
            S + (t + 1u) * 8u,
            W[t],
            crow,
            ds_row,
            dc_comp,
            &cs,
            &cc_comp
        );
        if (delta_S) {
            std::memcpy(delta_S + (t + 1u) * 8u, ds_row, sizeof(ds_row));
        }
        if (delta_C) {
            uint32_t* dcrow = delta_C + t * C_PLANES;
            dcrow[0] = dc0;
            dcrow[1] = dc1;
            dcrow[2] = dc2;
            for (uint32_t k = 0; k < 7u; ++k) dcrow[3u + k] = dc_comp[k];
        }
        tot_dS += cs;
        tot_dC += cc_comp;
    }

    if (out_dS) *out_dS = tot_dS;
    if (out_dW) *out_dW = tot_dW;
    if (out_dC) *out_dC = tot_dC;
    return static_cast<uint64_t>(alpha) * tot_dS
         + static_cast<uint64_t>(beta) * tot_dW
         + static_cast<uint64_t>(gamma) * tot_dC;
}

// High-throughput batch evaluation of crystal defects
inline void eval_crystal_batch(
    uint32_t B,
    uint32_t R,
    const uint32_t* H_target,  // (B, 8)
    const uint32_t* W,         // (B, R)
    const uint32_t* S,         // (B, R+1, 8)
    const uint32_t* C,         // (B, R, 10)
    uint32_t* delta_S,         // (B, R+2, 8) nullable
    uint32_t* delta_W,         // (B, R) nullable
    uint32_t* delta_C,         // (B, R, 10) nullable
    uint64_t* out_E,           // (B,)
    uint64_t* out_dS,          // (B,) nullable
    uint64_t* out_dW,          // (B,) nullable
    uint64_t* out_dC,          // (B,) nullable
    uint32_t alpha = 1u,
    uint32_t beta = 1u,
    uint32_t gamma = 1u
) {
    const uint64_t stride_S = (R + 1u) * 8u;
    const uint64_t stride_dS = (R + 2u) * 8u;
    const uint64_t stride_C = R * C_PLANES;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (B >= 32)
#endif
    for (int64_t bi = 0; bi < static_cast<int64_t>(B); ++bi) {
        const uint64_t b = static_cast<uint64_t>(bi);
        uint64_t ds = 0ULL, dw = 0ULL, dc = 0ULL;
        uint64_t e = eval_crystal_single(
            R,
            H_target + b * 8u,
            W + b * R,
            S + b * stride_S,
            C + b * stride_C,
            delta_S ? (delta_S + b * stride_dS) : nullptr,
            delta_W ? (delta_W + b * R) : nullptr,
            delta_C ? (delta_C + b * stride_C) : nullptr,
            &ds,
            &dw,
            &dc,
            alpha,
            beta,
            gamma
        );
        out_E[b] = e;
        if (out_dS) out_dS[b] = ds;
        if (out_dW) out_dW[b] = dw;
        if (out_dC) out_dC[b] = dc;
    }
}

// High-throughput incremental cluster-flip evaluator (single crystal evaluated across N_cands clusters,
// OR B_cryst crystals x K_per_cryst candidates).
// Each cluster flip entry is packed as 4 int32s: (kind, t, word, bit)
//   kind == 0: W[t], word unused (0), bit in [0..31]
//   kind == 1: S[t][word], word in [0..7], bit in [0..31]
//   kind == 2: C[t][word], word in [0..9], bit in [0..31]
//   invalid/padding entries have kind < 0 (ignored).
inline void eval_cluster_flips_batch(
    uint32_t B_cryst,
    uint32_t K_cands,
    uint32_t M_max,
    uint32_t R,
    const uint32_t* H_target,   // (B_cryst, 8)
    const uint32_t* W_base,     // (B_cryst, R)
    const uint32_t* S_base,     // (B_cryst, R+1, 8)
    const uint32_t* C_base,     // (B_cryst, R, 10)
    const int32_t* cluster_idx, // (B_cryst * K_cands, M_max, 4)
    int64_t* out_delta_E,       // (B_cryst * K_cands,)
    uint64_t* out_new_E,        // (B_cryst * K_cands,) nullable
    uint32_t alpha = 1u,
    uint32_t beta = 1u,
    uint32_t gamma = 1u,
    bool allow_boundary_flip = false
) {
    const uint64_t stride_S = (R + 1u) * 8u;
    const uint64_t stride_C = R * C_PLANES;

    // Step 1: Precompute per-round defect counts for every base crystal once
    struct BaseStats {
        uint32_t rw_cnt[MAX_ROUNDS];
        uint32_t rc_sched[MAX_ROUNDS];
        uint32_t rc_comp[MAX_ROUNDS];
        uint32_t rs_cnt[MAX_ROUNDS];
        uint32_t bnd0_cnt;
        uint32_t bndR_cnt;
        uint64_t base_E;
    };
    std::vector<BaseStats> bstats(B_cryst);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (B_cryst >= 16)
#endif
    for (int64_t bi = 0; bi < static_cast<int64_t>(B_cryst); ++bi) {
        const uint64_t b = static_cast<uint64_t>(bi);
        const uint32_t* Hb = H_target + b * 8u;
        const uint32_t* Wb = W_base + b * R;
        const uint32_t* Sb = S_base + b * stride_S;
        const uint32_t* Cb = C_base + b * stride_C;
        BaseStats& bs = bstats[b];
        uint64_t sum_ds = 0ULL, sum_dw = 0ULL, sum_dc = 0ULL;

        uint32_t c0 = 0u, cr = 0u;
        for (uint32_t j = 0; j < 8u; ++j) {
            c0 += popcnt32(Sb[j] ^ IV_WORDS[j]);
            cr += popcnt32(Sb[R * 8u + j] ^ (Hb[j] - IV_WORDS[j]));
        }
        bs.bnd0_cnt = c0;
        bs.bndR_cnt = cr;
        sum_ds += c0 + cr;

        for (uint32_t t = 0; t < R; ++t) {
            uint32_t dw = 0u, dc0 = 0u, dc1 = 0u, dc2 = 0u;
            uint32_t cc_s = eval_schedule_round(t, Wb, Cb + t * C_PLANES, &dw, &dc0, &dc1, &dc2);
            uint32_t dw_c = popcnt32(dw);
            bs.rw_cnt[t] = dw_c;
            bs.rc_sched[t] = cc_s;

            uint32_t ds_row[8], dc_comp[7], cs = 0u, cc_c = 0u;
            eval_compress_round(
                t, Sb + t * 8u, Sb + (t + 1u) * 8u, Wb[t], Cb + t * C_PLANES,
                ds_row, dc_comp, &cs, &cc_c
            );
            bs.rs_cnt[t] = cs;
            bs.rc_comp[t] = cc_c;
            sum_ds += cs;
            sum_dw += dw_c;
            sum_dc += cc_s + cc_c;
        }
        bs.base_E = static_cast<uint64_t>(alpha) * sum_ds
                  + static_cast<uint64_t>(beta) * sum_dw
                  + static_cast<uint64_t>(gamma) * sum_dc;
    }

    const int64_t total_cands = static_cast<int64_t>(B_cryst) * static_cast<int64_t>(K_cands);

    // Step 2: O(1) dirty-mask incremental evaluation in L1 stack buffers across OpenMP threads
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        uint32_t W_loc[MAX_ROUNDS];
        uint32_t S_loc[(MAX_ROUNDS + 1u) * 8u];
        uint32_t C_loc[MAX_ROUNDS * C_PLANES];
        uint32_t last_b = UINT32_MAX;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int64_t ci = 0; ci < total_cands; ++ci) {
            const uint32_t b = static_cast<uint32_t>(ci / static_cast<int64_t>(K_cands));
            if (b != last_b) {
                std::memcpy(W_loc, W_base + static_cast<uint64_t>(b) * R, R * sizeof(uint32_t));
                std::memcpy(S_loc, S_base + static_cast<uint64_t>(b) * stride_S, stride_S * sizeof(uint32_t));
                std::memcpy(C_loc, C_base + static_cast<uint64_t>(b) * stride_C, stride_C * sizeof(uint32_t));
                last_b = b;
            }

            const BaseStats& bs = bstats[b];
            const uint32_t* Hb = H_target + static_cast<uint64_t>(b) * 8u;
            const int32_t* cl = cluster_idx + ci * static_cast<int64_t>(M_max) * 4;

            uint64_t dirty_W = 0ULL;
            uint64_t dirty_S = 0ULL;
            uint64_t dirty_C_sched = 0ULL;
            uint64_t dirty_C_comp = 0ULL;
            bool dirty_bnd0 = false;
            bool dirty_bndR = false;
            bool illegal = false;

            // Apply cluster flips via in-place XOR and mark exact round dependency masks
            for (uint32_t m = 0; m < M_max; ++m) {
                int32_t kind = cl[m * 4 + 0];
                if (kind < 0) continue;
                int32_t t = cl[m * 4 + 1];
                int32_t word = cl[m * 4 + 2];
                int32_t bit = cl[m * 4 + 3];
                if (bit < 0 || bit >= 32) continue;
                uint32_t mask = (1u << static_cast<uint32_t>(bit));

                if (kind == 0) {
                    if (t < 0 || static_cast<uint32_t>(t) >= R) continue;
                    uint32_t ut = static_cast<uint32_t>(t);
                    W_loc[ut] ^= mask;
                    dirty_W |= (1ULL << ut);
                    dirty_S |= (1ULL << ut);
                    dirty_C_comp |= (1ULL << ut);
                    const uint32_t offs[4] = {2u, 7u, 15u, 16u};
                    for (uint32_t off : offs) {
                        if (ut + off < R) {
                            dirty_W |= (1ULL << (ut + off));
                            dirty_C_sched |= (1ULL << (ut + off));
                        }
                    }
                } else if (kind == 1) {
                    if (t < 0 || static_cast<uint32_t>(t) > R || word < 0 || word >= 8) continue;
                    uint32_t ut = static_cast<uint32_t>(t);
                    if (!allow_boundary_flip && (ut == 0u || ut == R)) {
                        illegal = true;
                        continue;
                    }
                    S_loc[ut * 8u + static_cast<uint32_t>(word)] ^= mask;
                    if (ut == 0u) dirty_bnd0 = true;
                    if (ut == R) dirty_bndR = true;
                    if (ut > 0u) dirty_S |= (1ULL << (ut - 1u));
                    if (ut < R) {
                        dirty_S |= (1ULL << ut);
                        dirty_C_comp |= (1ULL << ut);
                    }
                } else if (kind == 2) {
                    if (t < 0 || static_cast<uint32_t>(t) >= R || word < 0 || word >= static_cast<int32_t>(C_PLANES)) continue;
                    uint32_t ut = static_cast<uint32_t>(t);
                    uint32_t uw = static_cast<uint32_t>(word);
                    C_loc[ut * C_PLANES + uw] ^= mask;
                    if (uw < 3u) {
                        dirty_C_sched |= (1ULL << ut);
                        dirty_W |= (1ULL << ut);
                    } else {
                        dirty_C_comp |= (1ULL << ut);
                        dirty_S |= (1ULL << ut);
                    }
                }
            }

            int64_t d_dS = 0, d_dW = 0, d_dC = 0;
            if (!illegal) {
                if (dirty_bnd0) {
                    uint32_t nc = 0u;
                    for (uint32_t j = 0; j < 8u; ++j) nc += popcnt32(S_loc[j] ^ IV_WORDS[j]);
                    d_dS += static_cast<int64_t>(nc) - static_cast<int64_t>(bs.bnd0_cnt);
                }
                if (dirty_bndR) {
                    uint32_t nc = 0u;
                    for (uint32_t j = 0; j < 8u; ++j) nc += popcnt32(S_loc[R * 8u + j] ^ (Hb[j] - IV_WORDS[j]));
                    d_dS += static_cast<int64_t>(nc) - static_cast<int64_t>(bs.bndR_cnt);
                }

                uint64_t m_sched = dirty_W | dirty_C_sched;
                while (m_sched) {
                    uint32_t t = ctz64(m_sched);
                    m_sched &= (m_sched - 1ULL);
                    uint32_t dw = 0u, dc0 = 0u, dc1 = 0u, dc2 = 0u;
                    uint32_t cc_s = eval_schedule_round(t, W_loc, C_loc + t * C_PLANES, &dw, &dc0, &dc1, &dc2);
                    uint32_t dw_c = popcnt32(dw);
                    d_dW += static_cast<int64_t>(dw_c) - static_cast<int64_t>(bs.rw_cnt[t]);
                    d_dC += static_cast<int64_t>(cc_s) - static_cast<int64_t>(bs.rc_sched[t]);
                }

                uint64_t m_comp = dirty_S | dirty_C_comp;
                while (m_comp) {
                    uint32_t t = ctz64(m_comp);
                    m_comp &= (m_comp - 1ULL);
                    uint32_t ds_row[8], dc_comp[7], cs = 0u, cc_c = 0u;
                    eval_compress_round(
                        t, S_loc + t * 8u, S_loc + (t + 1u) * 8u, W_loc[t], C_loc + t * C_PLANES,
                        ds_row, dc_comp, &cs, &cc_c
                    );
                    d_dS += static_cast<int64_t>(cs) - static_cast<int64_t>(bs.rs_cnt[t]);
                    d_dC += static_cast<int64_t>(cc_c) - static_cast<int64_t>(bs.rc_comp[t]);
                }
            }

            // Undo XOR flips cleanly back to base crystal in L1 cache
            for (uint32_t m = 0; m < M_max; ++m) {
                int32_t kind = cl[m * 4 + 0];
                if (kind < 0) continue;
                int32_t t = cl[m * 4 + 1];
                int32_t word = cl[m * 4 + 2];
                int32_t bit = cl[m * 4 + 3];
                if (bit < 0 || bit >= 32) continue;
                uint32_t mask = (1u << static_cast<uint32_t>(bit));
                if (kind == 0 && t >= 0 && static_cast<uint32_t>(t) < R) {
                    W_loc[t] ^= mask;
                } else if (kind == 1 && t >= 0 && static_cast<uint32_t>(t) <= R && word >= 0 && word < 8) {
                    if (!allow_boundary_flip && (t == 0 || static_cast<uint32_t>(t) == R)) continue;
                    S_loc[static_cast<uint32_t>(t) * 8u + static_cast<uint32_t>(word)] ^= mask;
                } else if (kind == 2 && t >= 0 && static_cast<uint32_t>(t) < R && word >= 0 && word < static_cast<int32_t>(C_PLANES)) {
                    C_loc[static_cast<uint32_t>(t) * C_PLANES + static_cast<uint32_t>(word)] ^= mask;
                }
            }

            if (illegal) {
                out_delta_E[ci] = 1000000LL;
                if (out_new_E) out_new_E[ci] = bs.base_E + 1000000ULL;
            } else {
                int64_t dE = static_cast<int64_t>(alpha) * d_dS
                           + static_cast<int64_t>(beta) * d_dW
                           + static_cast<int64_t>(gamma) * d_dC;
                out_delta_E[ci] = dE;
                if (out_new_E) {
                    int64_t ne = static_cast<int64_t>(bs.base_E) + dE;
                    out_new_E[ci] = ne > 0 ? static_cast<uint64_t>(ne) : 0ULL;
                }
            }
        }
    }
}

// ============================================================================
// Plane A: Incremental GF(2) Row-Reduced Echelon Form (RREF) Engine
// ============================================================================
class GF2RREF {
public:
    uint32_t num_vars = 0;
    uint32_t nwords = 0;
    uint32_t rhs_word = 0;
    uint64_t rhs_mask = 0ULL;
    uint64_t last_var_mask = 0ULL;
    bool conflict = false;
    uint32_t num_conflicts = 0;
    uint32_t rank = 0;

    std::vector<uint8_t> has_pivot;
    std::vector<std::vector<uint64_t>> pivot_rows;
    std::vector<uint32_t> active_pivots;

    void init(uint32_t nvars) {
        num_vars = nvars;
        // bit nvars holds RHS
        nwords = (nvars + 1u + 63u) / 64u;
        rhs_word = nvars / 64u;
        rhs_mask = 1ULL << (nvars % 64u);
        uint32_t rem = nvars % 64u;
        last_var_mask = (rem == 0u) ? 0ULL : ((1ULL << rem) - 1ULL);
        conflict = false;
        num_conflicts = 0u;
        rank = 0u;
        has_pivot.assign(nvars, 0u);
        pivot_rows.assign(nvars, {});
        active_pivots.clear();
    }

    inline uint64_t get_rhs(const std::vector<uint64_t>& r) const {
        return (r[rhs_word] & rhs_mask) ? 1ULL : 0ULL;
    }

    inline void xor_into(std::vector<uint64_t>& dst, const std::vector<uint64_t>& src) const {
        for (uint32_t i = 0; i < nwords; ++i) dst[i] ^= src[i];
    }

    // Find lowest variable column < num_vars (ignores rhs bit), or num_vars if all-zero var part
    inline uint32_t first_var_col(const std::vector<uint64_t>& r) const {
        const uint32_t last_var_word = num_vars / 64u;
        for (uint32_t w = 0; w < last_var_word; ++w) {
            if (r[w]) return w * 64u + ctz64(r[w]);
        }
        if (last_var_mask) {
            uint64_t masked = r[last_var_word] & last_var_mask;
            if (masked) return last_var_word * 64u + ctz64(masked);
        }
        return num_vars;
    }

    inline uint32_t var_popcnt(const std::vector<uint64_t>& r) const {
        uint32_t cnt = 0u;
        const uint32_t last_var_word = num_vars / 64u;
        for (uint32_t w = 0; w < last_var_word; ++w) cnt += popcnt64(r[w]);
        if (last_var_mask) cnt += popcnt64(r[last_var_word] & last_var_mask);
        return cnt;
    }

    // Incremental insertion preserving full RREF invariant across all pivot rows.
    // Returns false if linear contradiction (0 == 1) detected.
    bool add_equation(const uint32_t* cols, uint32_t n_cols, uint8_t rhs) {
        std::vector<uint64_t> row(nwords, 0ULL);
        for (uint32_t i = 0; i < n_cols; ++i) {
            uint32_t c = cols[i];
            if (c < num_vars) row[c >> 6] ^= (1ULL << (c & 63u));
        }
        if (rhs & 1u) row[rhs_word] ^= rhs_mask;

        // Reduce against existing pivots for all set bits in row (leading to trailing)
        const uint32_t last_var_word = num_vars / 64u;
        uint32_t lead = num_vars;
        while (true) {
            uint32_t c = first_var_col(row);
            if (c >= num_vars) break;
            if (has_pivot[c]) {
                xor_into(row, pivot_rows[c]);
            } else {
                lead = c;
                break;
            }
        }

        if (lead >= num_vars) {
            if (get_rhs(row)) {
                conflict = true;
                ++num_conflicts;
                return false;
            }
            return true;
        }

        // Back-reduce higher columns (> lead) inside `row` so existing pivots vanish from `row`
        for (uint32_t w = lead >> 6; w <= last_var_word; ++w) {
            uint64_t word = row[w];
            if (w == last_var_word) word &= last_var_mask;
            while (word) {
                uint32_t b = ctz64(word);
                word &= (word - 1ULL);
                uint32_t c = w * 64u + b;
                if (c <= lead) continue;
                if (has_pivot[c]) {
                    xor_into(row, pivot_rows[c]);
                    word = (b < 63u) ? (row[w] & (~0ULL << (b + 1u))) : 0ULL;
                    if (w == last_var_word) word &= last_var_mask;
                }
            }
        }

        // Eliminate `lead` from all existing pivot rows to keep true RREF
        const uint32_t lw = lead >> 6;
        const uint64_t lmask = 1ULL << (lead & 63u);
        for (uint32_t ap : active_pivots) {
            if (pivot_rows[ap][lw] & lmask) {
                xor_into(pivot_rows[ap], row);
            }
        }

        has_pivot[lead] = 1u;
        pivot_rows[lead] = std::move(row);
        active_pivots.push_back(lead);
        ++rank;
        return true;
    }

    // Extract all single-variable deductions (var[c] == val) from current RREF rows
    void extract_unit_deductions(std::vector<uint32_t>& out_cols, std::vector<uint8_t>& out_vals) const {
        out_cols.clear();
        out_vals.clear();
        for (uint32_t ap : active_pivots) {
            const auto& r = pivot_rows[ap];
            if (var_popcnt(r) == 1u) {
                out_cols.push_back(ap);
                out_vals.push_back(static_cast<uint8_t>(get_rhs(r)));
            }
        }
    }

    // Compute per-variable GF(2) constraint participation counts across all active RREF rows
    void var_constraint_degrees(uint32_t* out_deg) const {
        std::memset(out_deg, 0, num_vars * sizeof(uint32_t));
        const uint32_t last_var_word = num_vars / 64u;
        for (uint32_t ap : active_pivots) {
            const auto& r = pivot_rows[ap];
            for (uint32_t w = 0; w < last_var_word; ++w) {
                uint64_t x = r[w];
                while (x) {
                    out_deg[w * 64u + ctz64(x)] += 1u;
                    x &= (x - 1ULL);
                }
            }
            if (last_var_mask) {
                uint64_t x = r[last_var_word] & last_var_mask;
                while (x) {
                    out_deg[last_var_word * 64u + ctz64(x)] += 1u;
                    x &= (x - 1ULL);
                }
            }
        }
    }
};

// ============================================================================
// Plane A + Plane B Coupled Exact Crystal Propagator (6-symbol storage, {0,1,?} unit prop)
// ============================================================================
struct CrystalPropagator {
    uint32_t R = 64;
    // Total canonical GF(2) columns:
    //   W: R*32
    //   a_canon: (R+4)*32  (indices: t=-3..-1 => d0,c0,b0; t=0..R => a[0..R])
    //   e_canon: (R+4)*32  (indices: t=-3..-1 => h0,g0,f0; t=0..R => e[0..R])
    //   C: R*10*32
    //   Ch: R*32
    //   Maj: R*32
    uint32_t off_W = 0;
    uint32_t off_a = 0;
    uint32_t off_e = 0;
    uint32_t off_C = 0;
    uint32_t off_Ch = 0;
    uint32_t off_Maj = 0;
    uint32_t num_gf2_vars = 0;

    // 6-symbol arrays (each byte in {0, 1, 2='?', 3='-', 4='u', 5='n'})
    std::vector<uint8_t> W_sym;      // (R, 32)
    std::vector<uint8_t> S_sym;      // (R+1, 8, 32)
    std::vector<uint8_t> C_sym;      // (R, 10, 32)
    std::vector<uint8_t> Ch_sym;     // (R, 32)
    std::vector<uint8_t> Maj_sym;    // (R, 32)
    std::vector<uint8_t> u0_sym;     // (R, 32)
    std::vector<uint8_t> u1_sym;     // (R, 32)
    std::vector<uint8_t> t1a_sym;    // (R, 32)
    std::vector<uint8_t> t1b_sym;    // (R, 32)
    std::vector<uint8_t> t1c_sym;    // (R, 32)
    std::vector<uint8_t> T1_sym;     // (R, 32)
    std::vector<uint8_t> T2_sym;     // (R, 32)

    std::array<uint32_t, 8> H_target{};
    GF2RREF gf2;
    std::vector<uint8_t> gf2_unit_pushed;
    std::vector<uint8_t> gf2_t1_pushed;
    std::vector<uint8_t> gf2_t2_pushed;
    bool conflict = false;
    uint32_t unit_prop_deductions = 0;
    uint32_t gf2_deductions = 0;

    inline uint32_t col_W(uint32_t t, uint32_t b) const { return off_W + t * 32u + b; }
    inline uint32_t col_a(int32_t idx, uint32_t b) const { return off_a + static_cast<uint32_t>(idx + 3) * 32u + b; }
    inline uint32_t col_e(int32_t idx, uint32_t b) const { return off_e + static_cast<uint32_t>(idx + 3) * 32u + b; }
    inline uint32_t col_C(uint32_t t, uint32_t k, uint32_t b) const { return off_C + (t * C_PLANES + k) * 32u + b; }
    inline uint32_t col_Ch(uint32_t t, uint32_t b) const { return off_Ch + t * 32u + b; }
    inline uint32_t col_Maj(uint32_t t, uint32_t b) const { return off_Maj + t * 32u + b; }

    // Map S(t, word, b) to canonical GF(2) column (or UINT32_MAX if out of canonical alias chain -> e.g., constant IV)
    inline uint32_t col_S_canon(uint32_t t, uint32_t w, uint32_t b) const {
        if (w < 4u) return col_a(static_cast<int32_t>(t) - static_cast<int32_t>(w), b);
        return col_e(static_cast<int32_t>(t) - static_cast<int32_t>(w - 4u), b);
    }

    static inline bool is_det(uint8_t s) { return s == SYM_0 || s == SYM_1; }

    inline bool assign_bit(uint8_t& slot, uint8_t val) {
        if (slot == val) return false;
        if (is_det(slot)) {
            conflict = true;
            return false;
        }
        slot = val;
        return true;
    }

    inline bool sync_eq(uint8_t& a, uint8_t& b) {
        bool da = is_det(a), db = is_det(b);
        if (da && db) {
            if (a != b) conflict = true;
            return false;
        }
        if (da) { b = a; return true; }
        if (db) { a = b; return true; }
        return false;
    }

    // 4-bit exact truth-table propagation for Maj(x,y,z) == out
    inline bool prop_maj(uint8_t& x, uint8_t& y, uint8_t& z, uint8_t& out) {
        uint8_t cur[4] = {x, y, z, out};
        uint8_t cand_and[4] = {1, 1, 1, 1};
        uint8_t cand_or[4] = {0, 0, 0, 0};
        int valid_cnt = 0;
        for (int mask = 0; mask < 8; ++mask) {
            uint8_t vx = static_cast<uint8_t>((mask >> 0) & 1);
            uint8_t vy = static_cast<uint8_t>((mask >> 1) & 1);
            uint8_t vz = static_cast<uint8_t>((mask >> 2) & 1);
            uint8_t vo = static_cast<uint8_t>((vx & vy) | (vx & vz) | (vy & vz));
            if (is_det(cur[0]) && cur[0] != vx) continue;
            if (is_det(cur[1]) && cur[1] != vy) continue;
            if (is_det(cur[2]) && cur[2] != vz) continue;
            if (is_det(cur[3]) && cur[3] != vo) continue;
            ++valid_cnt;
            cand_and[0] &= vx; cand_or[0] |= vx;
            cand_and[1] &= vy; cand_or[1] |= vy;
            cand_and[2] &= vz; cand_or[2] |= vz;
            cand_and[3] &= vo; cand_or[3] |= vo;
        }
        if (valid_cnt == 0) {
            conflict = true;
            return false;
        }
        bool changed = false;
        uint8_t* ptrs[4] = {&x, &y, &z, &out};
        for (int i = 0; i < 4; ++i) {
            if (!is_det(*ptrs[i]) && cand_and[i] == cand_or[i]) {
                *ptrs[i] = cand_and[i];
                changed = true;
            }
        }
        return changed;
    }

    // 4-bit exact truth-table propagation for Ch(e,f,g) == out
    inline bool prop_ch(uint8_t& e, uint8_t& f, uint8_t& g, uint8_t& out) {
        uint8_t cur[4] = {e, f, g, out};
        uint8_t cand_and[4] = {1, 1, 1, 1};
        uint8_t cand_or[4] = {0, 0, 0, 0};
        int valid_cnt = 0;
        for (int mask = 0; mask < 8; ++mask) {
            uint8_t ve = static_cast<uint8_t>((mask >> 0) & 1);
            uint8_t vf = static_cast<uint8_t>((mask >> 1) & 1);
            uint8_t vg = static_cast<uint8_t>((mask >> 2) & 1);
            uint8_t vo = static_cast<uint8_t>((ve & vf) | ((1u - ve) & vg));
            if (is_det(cur[0]) && cur[0] != ve) continue;
            if (is_det(cur[1]) && cur[1] != vf) continue;
            if (is_det(cur[2]) && cur[2] != vg) continue;
            if (is_det(cur[3]) && cur[3] != vo) continue;
            ++valid_cnt;
            cand_and[0] &= ve; cand_or[0] |= ve;
            cand_and[1] &= vf; cand_or[1] |= vf;
            cand_and[2] &= vg; cand_or[2] |= vg;
            cand_and[3] &= vo; cand_or[3] |= vo;
        }
        if (valid_cnt == 0) {
            conflict = true;
            return false;
        }
        bool changed = false;
        uint8_t* ptrs[4] = {&e, &f, &g, &out};
        for (int i = 0; i < 4; ++i) {
            if (!is_det(*ptrs[i]) && cand_and[i] == cand_or[i]) {
                *ptrs[i] = cand_and[i];
                changed = true;
            }
        }
        return changed;
    }

    // 5-bit exact joint truth-table propagation for FullAdder(x, y, cin) -> (sum, cout)
    // persist_mask bits 0..4 indicate which slots are persistent (vs transient stack temporaries)
    inline bool prop_full_adder(
        uint8_t* px,
        uint8_t* py,
        uint8_t* pcin,
        uint8_t* psum,
        uint8_t* pcout,
        uint8_t persist_mask = 0x1Fu
    ) {
        uint8_t cur[5] = {*px, *py, *pcin, *psum, *pcout};
        uint8_t cand_and[5] = {1, 1, 1, 1, 1};
        uint8_t cand_or[5] = {0, 0, 0, 0, 0};
        int valid_cnt = 0;
        for (int mask = 0; mask < 8; ++mask) {
            uint8_t vx = static_cast<uint8_t>((mask >> 0) & 1);
            uint8_t vy = static_cast<uint8_t>((mask >> 1) & 1);
            uint8_t vc = static_cast<uint8_t>((mask >> 2) & 1);
            uint8_t vs = static_cast<uint8_t>(vx ^ vy ^ vc);
            uint8_t vco = static_cast<uint8_t>((vx & vy) | (vx & vc) | (vy & vc));
            if (is_det(cur[0]) && cur[0] != vx) continue;
            if (is_det(cur[1]) && cur[1] != vy) continue;
            if (is_det(cur[2]) && cur[2] != vc) continue;
            if (is_det(cur[3]) && cur[3] != vs) continue;
            if (is_det(cur[4]) && cur[4] != vco) continue;
            ++valid_cnt;
            cand_and[0] &= vx; cand_or[0] |= vx;
            cand_and[1] &= vy; cand_or[1] |= vy;
            cand_and[2] &= vc; cand_or[2] |= vc;
            cand_and[3] &= vs; cand_or[3] |= vs;
            cand_and[4] &= vco; cand_or[4] |= vco;
        }
        if (valid_cnt == 0) {
            conflict = true;
            return false;
        }
        bool changed = false;
        uint8_t* ptrs[5] = {px, py, pcin, psum, pcout};
        for (int i = 0; i < 5; ++i) {
            if (!is_det(*ptrs[i]) && cand_and[i] == cand_or[i]) {
                *ptrs[i] = cand_and[i];
                if (persist_mask & (1u << i)) {
                    changed = true;
                }
            }
        }
        return changed;
    }

    // Binary XOR with up to k unknown terms summing to rhs_const (computed by feeding knowns)
    inline bool prop_xor_vec(uint8_t* const* terms, int n_terms, uint8_t rhs_const) {
        int unk_idx = -1;
        int unk_cnt = 0;
        uint8_t acc = rhs_const & 1u;
        for (int i = 0; i < n_terms; ++i) {
            uint8_t v = *terms[i];
            if (is_det(v)) {
                acc ^= v;
            } else {
                unk_idx = i;
                ++unk_cnt;
            }
        }
        if (unk_cnt == 0) {
            if (acc != 0u) conflict = true;
            return false;
        }
        if (unk_cnt == 1) {
            *terms[unk_idx] = acc;
            return true;
        }
        return false;
    }

    void push_gf2_unit(uint32_t col, uint8_t val) {
        if (col >= num_gf2_vars) return;
        if (gf2_unit_pushed[col] == val) return;
        if (gf2_unit_pushed[col] < 2u) {
            conflict = true;
            gf2.conflict = true;
            ++gf2.num_conflicts;
            return;
        }
        gf2_unit_pushed[col] = val;
        gf2.add_equation(&col, 1u, val);
        if (gf2.conflict) conflict = true;
    }

    void sync_all_known_to_gf2() {
        for (uint32_t t = 0; t < R; ++t) {
            for (uint32_t b = 0; b < 32u; ++b) {
                uint8_t wv = W_sym[t * 32u + b];
                if (is_det(wv)) push_gf2_unit(col_W(t, b), wv);
                for (uint32_t k = 0; k < C_PLANES; ++k) {
                    uint8_t cv = C_sym[(t * C_PLANES + k) * 32u + b];
                    if (is_det(cv)) push_gf2_unit(col_C(t, k, b), cv);
                }
                uint8_t chv = Ch_sym[t * 32u + b];
                if (is_det(chv)) push_gf2_unit(col_Ch(t, b), chv);
                uint8_t mjv = Maj_sym[t * 32u + b];
                if (is_det(mjv)) push_gf2_unit(col_Maj(t, b), mjv);
            }
        }
        for (uint32_t t = 0; t <= R; ++t) {
            for (uint32_t w = 0; w < 8u; ++w) {
                for (uint32_t b = 0; b < 32u; ++b) {
                    uint8_t sv = S_sym[(t * 8u + w) * 32u + b];
                    if (is_det(sv)) push_gf2_unit(col_S_canon(t, w, b), sv);
                }
            }
        }
        // Push known T1 / T2 linear identities: T1 = e[t+1]^d[t]^C9, T2 = Sigma0(a[t])^Maj[t]^C7
        for (uint32_t t = 0; t < R && !conflict; ++t) {
            for (uint32_t b = 0; b < 32u && !conflict; ++b) {
                uint32_t idx = t * 32u + b;
                uint8_t tv1 = T1_sym[idx];
                if (is_det(tv1) && gf2_t1_pushed[idx] != tv1) {
                    if (gf2_t1_pushed[idx] < 2u) {
                        conflict = true;
                        break;
                    }
                    gf2_t1_pushed[idx] = tv1;
                    uint32_t eq[3] = {
                        col_e(static_cast<int32_t>(t) + 1, b),
                        col_a(static_cast<int32_t>(t) - 3, b),
                        col_C(t, 9u, b),
                    };
                    gf2.add_equation(eq, 3u, tv1);
                    if (gf2.conflict) conflict = true;
                }
                uint8_t tv2 = T2_sym[idx];
                if (is_det(tv2) && gf2_t2_pushed[idx] != tv2 && !conflict) {
                    if (gf2_t2_pushed[idx] < 2u) {
                        conflict = true;
                        break;
                    }
                    gf2_t2_pushed[idx] = tv2;
                    uint32_t eq[5] = {
                        col_a(static_cast<int32_t>(t), (b + 2u) & 31u),
                        col_a(static_cast<int32_t>(t), (b + 13u) & 31u),
                        col_a(static_cast<int32_t>(t), (b + 22u) & 31u),
                        col_Maj(t, b),
                        col_C(t, 7u, b),
                    };
                    gf2.add_equation(eq, 5u, tv2);
                    if (gf2.conflict) conflict = true;
                }
            }
        }
        // Conditional linear Ch/Maj rules + intermediate partial sum deductions
        for (uint32_t t = 0; t < R && !conflict; ++t) {
            for (uint32_t b = 0; b < 32u && !conflict; ++b) {
                uint8_t eb = S_sym[(t * 8u + 4u) * 32u + b];
                uint8_t fb = S_sym[(t * 8u + 5u) * 32u + b];
                uint8_t gb = S_sym[(t * 8u + 6u) * 32u + b];
                if (eb == SYM_1) {
                    uint32_t eq[2] = {col_Ch(t, b), col_S_canon(t, 5u, b)};
                    gf2.add_equation(eq, 2u, 0u);
                } else if (eb == SYM_0) {
                    uint32_t eq[2] = {col_Ch(t, b), col_S_canon(t, 6u, b)};
                    gf2.add_equation(eq, 2u, 0u);
                } else if (is_det(fb) && is_det(gb)) {
                    if (fb == gb) {
                        uint32_t eq[1] = {col_Ch(t, b)};
                        gf2.add_equation(eq, 1u, fb);
                    } else if (fb == 1u && gb == 0u) {
                        // Ch = e
                        uint32_t eq[2] = {col_Ch(t, b), col_S_canon(t, 4u, b)};
                        gf2.add_equation(eq, 2u, 0u);
                    } else {
                        // Ch = ~e => Ch ^ e = 1
                        uint32_t eq[2] = {col_Ch(t, b), col_S_canon(t, 4u, b)};
                        gf2.add_equation(eq, 2u, 1u);
                    }
                }
                uint8_t ab = S_sym[(t * 8u + 0u) * 32u + b];
                uint8_t bb = S_sym[(t * 8u + 1u) * 32u + b];
                uint8_t cb = S_sym[(t * 8u + 2u) * 32u + b];
                if (is_det(ab) && is_det(bb) && ab == bb) {
                    uint32_t eq[1] = {col_Maj(t, b)};
                    gf2.add_equation(eq, 1u, ab);
                } else if (is_det(ab) && is_det(bb) && ab != bb) {
                    // Maj(0,1,c) = c
                    uint32_t eq[2] = {col_Maj(t, b), col_S_canon(t, 2u, b)};
                    gf2.add_equation(eq, 2u, 0u);
                } else if (is_det(ab) && is_det(cb) && ab != cb) {
                    uint32_t eq[2] = {col_Maj(t, b), col_S_canon(t, 1u, b)};
                    gf2.add_equation(eq, 2u, 0u);
                } else if (is_det(bb) && is_det(cb) && bb != cb) {
                    uint32_t eq[2] = {col_Maj(t, b), col_S_canon(t, 0u, b)};
                    gf2.add_equation(eq, 2u, 0u);
                }
                if (gf2.conflict) conflict = true;
            }
        }
    }

    void build_base_gf2_system() {
        // 1. Add all base linear structural equations across all R rounds
        // (a) Message schedule linear XOR equations for t >= 16:
        //   W[t][b] ^ ROTR17(W[t-2])[b] ^ ROTR19(W[t-2])[b] ^ SHR10(W[t-2])[b]
        //   ^ W[t-7][b] ^ ROTR7(W[t-15])[b] ^ ROTR18(W[t-15])[b] ^ SHR3(W[t-15])[b]
        //   ^ W[t-16][b] ^ C0[t][b] ^ C1[t][b] ^ C2[t][b] = 0
        uint32_t cols[32];
        for (uint32_t t = 16u; t < R && !conflict; ++t) {
            for (uint32_t b = 0; b < 32u && !conflict; ++b) {
                uint32_t n = 0u;
                cols[n++] = col_W(t, b);
                cols[n++] = col_W(t - 2u, (b + 17u) & 31u);
                cols[n++] = col_W(t - 2u, (b + 19u) & 31u);
                if (b + 10u < 32u) cols[n++] = col_W(t - 2u, b + 10u);
                cols[n++] = col_W(t - 7u, b);
                cols[n++] = col_W(t - 15u, (b + 7u) & 31u);
                cols[n++] = col_W(t - 15u, (b + 18u) & 31u);
                if (b + 3u < 32u) cols[n++] = col_W(t - 15u, b + 3u);
                cols[n++] = col_W(t - 16u, b);
                cols[n++] = col_C(t, 0u, b);
                cols[n++] = col_C(t, 1u, b);
                cols[n++] = col_C(t, 2u, b);
                gf2.add_equation(cols, n, 0u);
            }
        }
        // (b) Compression equations for t = 0..R-1:
        //   e_eq: e[t+1] ^ d[t] ^ h[t] ^ Sigma1(e[t]) ^ Ch[t] ^ W[t] ^ C3 ^ C4 ^ C5 ^ C6 ^ C9 = K[t]
        //   a_diff_eq: a[t+1] ^ e[t+1] ^ d[t] ^ Sigma0(a[t]) ^ Maj[t] ^ C7 ^ C8 ^ C9 = 0
        for (uint32_t t = 0; t < R && !conflict; ++t) {
            uint8_t kb_word = 0u;
            (void)kb_word;
            for (uint32_t b = 0; b < 32u && !conflict; ++b) {
                uint8_t kbit = static_cast<uint8_t>((K_WORDS[t] >> b) & 1u);
                uint32_t n1 = 0u;
                cols[n1++] = col_e(static_cast<int32_t>(t) + 1, b);
                cols[n1++] = col_a(static_cast<int32_t>(t) - 3, b);  // d[t]
                cols[n1++] = col_e(static_cast<int32_t>(t) - 3, b);  // h[t]
                cols[n1++] = col_e(static_cast<int32_t>(t), (b + 6u) & 31u);
                cols[n1++] = col_e(static_cast<int32_t>(t), (b + 11u) & 31u);
                cols[n1++] = col_e(static_cast<int32_t>(t), (b + 25u) & 31u);
                cols[n1++] = col_Ch(t, b);
                cols[n1++] = col_W(t, b);
                cols[n1++] = col_C(t, 3u, b);
                cols[n1++] = col_C(t, 4u, b);
                cols[n1++] = col_C(t, 5u, b);
                cols[n1++] = col_C(t, 6u, b);
                cols[n1++] = col_C(t, 9u, b);
                gf2.add_equation(cols, n1, kbit);

                uint32_t n2 = 0u;
                cols[n2++] = col_a(static_cast<int32_t>(t) + 1, b);
                cols[n2++] = col_e(static_cast<int32_t>(t) + 1, b);
                cols[n2++] = col_a(static_cast<int32_t>(t) - 3, b);  // d[t]
                cols[n2++] = col_a(static_cast<int32_t>(t), (b + 2u) & 31u);
                cols[n2++] = col_a(static_cast<int32_t>(t), (b + 13u) & 31u);
                cols[n2++] = col_a(static_cast<int32_t>(t), (b + 22u) & 31u);
                cols[n2++] = col_Maj(t, b);
                cols[n2++] = col_C(t, 7u, b);
                cols[n2++] = col_C(t, 8u, b);
                cols[n2++] = col_C(t, 9u, b);
                gf2.add_equation(cols, n2, 0u);
            }
        }
        if (gf2.conflict) conflict = true;
    }

    void init_from_symbols(
        uint32_t rounds,
        const uint32_t* H_tgt,
        const uint8_t* w_in,
        const uint8_t* s_in,
        const uint8_t* c_in,
        bool build_gf2 = true
    ) {
        R = rounds;
        off_W = 0u;
        off_a = off_W + R * 32u;
        off_e = off_a + (R + 4u) * 32u;
        off_C = off_e + (R + 4u) * 32u;
        off_Ch = off_C + R * C_PLANES * 32u;
        off_Maj = off_Ch + R * 32u;
        num_gf2_vars = off_Maj + R * 32u;

        W_sym.assign(w_in, w_in + R * 32u);
        S_sym.assign(s_in, s_in + (R + 1u) * 8u * 32u);
        C_sym.assign(c_in, c_in + R * C_PLANES * 32u);

        Ch_sym.assign(R * 32u, SYM_Q);
        Maj_sym.assign(R * 32u, SYM_Q);
        u0_sym.assign(R * 32u, SYM_Q);
        u1_sym.assign(R * 32u, SYM_Q);
        t1a_sym.assign(R * 32u, SYM_Q);
        t1b_sym.assign(R * 32u, SYM_Q);
        t1c_sym.assign(R * 32u, SYM_Q);
        T1_sym.assign(R * 32u, SYM_Q);
        T2_sym.assign(R * 32u, SYM_Q);

        std::memcpy(H_target.data(), H_tgt, 8u * sizeof(uint32_t));
        conflict = false;
        unit_prop_deductions = 0u;
        gf2_deductions = 0u;

        // Hard-weld S[0] == IV, S[R] == (H_target - IV), C[*][*][0] == 0, C[0..15][0..2][*] == 0
        for (uint32_t j = 0; j < 8u; ++j) {
            uint32_t s0w = IV_WORDS[j];
            uint32_t srw = H_target[j] - IV_WORDS[j];
            for (uint32_t b = 0; b < 32u; ++b) {
                assign_bit(S_sym[j * 32u + b], static_cast<uint8_t>((s0w >> b) & 1u));
                assign_bit(S_sym[(R * 8u + j) * 32u + b], static_cast<uint8_t>((srw >> b) & 1u));
            }
        }
        for (uint32_t t = 0; t < R; ++t) {
            for (uint32_t k = 0; k < C_PLANES; ++k) {
                assign_bit(C_sym[(t * C_PLANES + k) * 32u + 0u], SYM_0);
                if (t < 16u && k < 3u) {
                    for (uint32_t b = 0; b < 32u; ++b) {
                        assign_bit(C_sym[(t * C_PLANES + k) * 32u + b], SYM_0);
                    }
                }
            }
        }

        if (build_gf2) {
            gf2.init(num_gf2_vars);
            gf2_unit_pushed.assign(num_gf2_vars, 2u);
            gf2_t1_pushed.assign(R * 32u, 2u);
            gf2_t2_pushed.assign(R * 32u, 2u);
            build_base_gf2_system();
            sync_all_known_to_gf2();
        }
    }

    // Pull unit linear deductions from GF2 back into symbol grid
    bool pull_gf2_units() {
        std::vector<uint32_t> cols;
        std::vector<uint8_t> vals;
        gf2.extract_unit_deductions(cols, vals);
        bool any = false;
        for (size_t i = 0; i < cols.size() && !conflict; ++i) {
            uint32_t c = cols[i];
            uint8_t v = vals[i];
            if (gf2_unit_pushed[c] < 2u) {
                if (gf2_unit_pushed[c] != v) {
                    conflict = true;
                }
                continue;
            }
            gf2_unit_pushed[c] = v;
            bool chg = false;
            if (c < off_a) {
                uint32_t rel = c - off_W;
                chg = assign_bit(W_sym[rel], v);
            } else if (c < off_e) {
                int32_t idx = static_cast<int32_t>((c - off_a) / 32u) - 3;
                uint32_t b = (c - off_a) % 32u;
                // Push into all shift-aliased S[t][w][b] slots where t - w == idx (0<=w<=3, 0<=t<=R)
                for (uint32_t w = 0; w < 4u; ++w) {
                    int32_t t = idx + static_cast<int32_t>(w);
                    if (t >= 0 && t <= static_cast<int32_t>(R)) {
                        if (assign_bit(S_sym[(static_cast<uint32_t>(t) * 8u + w) * 32u + b], v)) {
                            chg = true;
                        }
                    }
                }
            } else if (c < off_C) {
                int32_t idx = static_cast<int32_t>((c - off_e) / 32u) - 3;
                uint32_t b = (c - off_e) % 32u;
                for (uint32_t w = 4u; w < 8u; ++w) {
                    int32_t t = idx + static_cast<int32_t>(w - 4u);
                    if (t >= 0 && t <= static_cast<int32_t>(R)) {
                        if (assign_bit(S_sym[(static_cast<uint32_t>(t) * 8u + w) * 32u + b], v)) {
                            chg = true;
                        }
                    }
                }
            } else if (c < off_Ch) {
                uint32_t rel = c - off_C;
                chg = assign_bit(C_sym[rel], v);
            } else if (c < off_Maj) {
                uint32_t rel = c - off_Ch;
                chg = assign_bit(Ch_sym[rel], v);
            } else {
                uint32_t rel = c - off_Maj;
                chg = assign_bit(Maj_sym[rel], v);
            }
            if (chg) {
                ++gf2_deductions;
                any = true;
            }
        }
        return any;
    }

    // Single bidirectional ARX propagation pass across all rounds
    bool sweep_arx_once() {
        bool any_changed = false;
        uint8_t zero_lit = SYM_0;

        auto prop_round = [&](uint32_t t) {
            if (conflict) return;
            // 1. Message schedule addition chain for t >= 16
            if (t >= 16u) {
                for (uint32_t b = 0; b < 32u && !conflict; ++b) {
                    // Forward & inverse carry chain + s1 + s0 local consistency
                    uint8_t* w2a = &W_sym[(t - 2u) * 32u + ((b + 17u) & 31u)];
                    uint8_t* w2b = &W_sym[(t - 2u) * 32u + ((b + 19u) & 31u)];
                    uint8_t* w2c = (b + 10u < 32u) ? &W_sym[(t - 2u) * 32u + b + 10u] : &zero_lit;
                    uint8_t s1_loc = SYM_Q;
                    // Check if all 3 terms of s1 are known; if s1_loc unknown we compute locally and backprop if needed
                    if (is_det(*w2a) && is_det(*w2b) && is_det(*w2c)) {
                        s1_loc = static_cast<uint8_t>((*w2a) ^ (*w2b) ^ (*w2c));
                    }

                    uint8_t* w15a = &W_sym[(t - 15u) * 32u + ((b + 7u) & 31u)];
                    uint8_t* w15b = &W_sym[(t - 15u) * 32u + ((b + 18u) & 31u)];
                    uint8_t* w15c = (b + 3u < 32u) ? &W_sym[(t - 15u) * 32u + b + 3u] : &zero_lit;
                    uint8_t s0_loc = SYM_Q;
                    if (is_det(*w15a) && is_det(*w15b) && is_det(*w15c)) {
                        s0_loc = static_cast<uint8_t>((*w15a) ^ (*w15b) ^ (*w15c));
                    }

                    uint8_t dummy_c32 = SYM_Q;
                    const uint8_t cout_bit = (b + 1u < 32u) ? 0x10u : 0x00u;
                    uint8_t* c0_cur = &C_sym[(t * C_PLANES + 0u) * 32u + b];
                    uint8_t* c0_nxt = (b + 1u < 32u) ? &C_sym[(t * C_PLANES + 0u) * 32u + b + 1u] : &dummy_c32;
                    uint8_t s1_copy = s1_loc;
                    if (prop_full_adder(&s1_copy, &W_sym[(t - 7u) * 32u + b], c0_cur, &u0_sym[t * 32u + b], c0_nxt, static_cast<uint8_t>(0x0Eu | cout_bit))) {
                        any_changed = true;
                        ++unit_prop_deductions;
                    }
                    // If s1_loc became deduced from adder, back-propagate into W[t-2]
                    if (!is_det(s1_loc) && is_det(s1_copy)) {
                        uint8_t* terms[3] = {w2a, w2b, w2c};
                        if (prop_xor_vec(terms, 3, s1_copy)) {
                            any_changed = true;
                            ++unit_prop_deductions;
                        }
                    }

                    dummy_c32 = SYM_Q;
                    uint8_t* c1_cur = &C_sym[(t * C_PLANES + 1u) * 32u + b];
                    uint8_t* c1_nxt = (b + 1u < 32u) ? &C_sym[(t * C_PLANES + 1u) * 32u + b + 1u] : &dummy_c32;
                    uint8_t s0_copy = s0_loc;
                    if (prop_full_adder(&u0_sym[t * 32u + b], &s0_copy, c1_cur, &u1_sym[t * 32u + b], c1_nxt, static_cast<uint8_t>(0x0Du | cout_bit))) {
                        any_changed = true;
                        ++unit_prop_deductions;
                    }
                    if (!is_det(s0_loc) && is_det(s0_copy)) {
                        uint8_t* terms[3] = {w15a, w15b, w15c};
                        if (prop_xor_vec(terms, 3, s0_copy)) {
                            any_changed = true;
                            ++unit_prop_deductions;
                        }
                    }

                    dummy_c32 = SYM_Q;
                    uint8_t* c2_cur = &C_sym[(t * C_PLANES + 2u) * 32u + b];
                    uint8_t* c2_nxt = (b + 1u < 32u) ? &C_sym[(t * C_PLANES + 2u) * 32u + b + 1u] : &dummy_c32;
                    if (prop_full_adder(&u1_sym[t * 32u + b], &W_sym[(t - 16u) * 32u + b], c2_cur, &W_sym[t * 32u + b], c2_nxt, static_cast<uint8_t>(0x0Fu | cout_bit))) {
                        any_changed = true;
                        ++unit_prop_deductions;
                    }
                    zero_lit = SYM_0;
                }
            }

            // 2. Shift register bidirectional equalities S[t] <-> S[t+1]
            for (uint32_t b = 0; b < 32u && !conflict; ++b) {
                uint8_t* St = &S_sym[t * 8u * 32u];
                uint8_t* St1 = &S_sym[(t + 1u) * 8u * 32u];
                if (sync_eq(St[0u * 32u + b], St1[1u * 32u + b])) { any_changed = true; ++unit_prop_deductions; }
                if (sync_eq(St[1u * 32u + b], St1[2u * 32u + b])) { any_changed = true; ++unit_prop_deductions; }
                if (sync_eq(St[2u * 32u + b], St1[3u * 32u + b])) { any_changed = true; ++unit_prop_deductions; }
                if (sync_eq(St[4u * 32u + b], St1[5u * 32u + b])) { any_changed = true; ++unit_prop_deductions; }
                if (sync_eq(St[5u * 32u + b], St1[6u * 32u + b])) { any_changed = true; ++unit_prop_deductions; }
                if (sync_eq(St[6u * 32u + b], St1[7u * 32u + b])) { any_changed = true; ++unit_prop_deductions; }
            }

            // 3. Nonlinear compression step S[t] -> S[t+1]
            for (uint32_t b = 0; b < 32u && !conflict; ++b) {
                uint8_t* St = &S_sym[t * 8u * 32u];
                uint8_t* St1 = &S_sym[(t + 1u) * 8u * 32u];

                // Ch(e,f,g) and Maj(a,b,c)
                if (prop_ch(St[4u * 32u + b], St[5u * 32u + b], St[6u * 32u + b], Ch_sym[t * 32u + b])) {
                    any_changed = true; ++unit_prop_deductions;
                }
                if (prop_maj(St[0u * 32u + b], St[1u * 32u + b], St[2u * 32u + b], Maj_sym[t * 32u + b])) {
                    any_changed = true; ++unit_prop_deductions;
                }

                // Sigma1(e)
                uint8_t* e0 = &St[4u * 32u + ((b + 6u) & 31u)];
                uint8_t* e1 = &St[4u * 32u + ((b + 11u) & 31u)];
                uint8_t* e2 = &St[4u * 32u + ((b + 25u) & 31u)];
                uint8_t S1_loc = (is_det(*e0) && is_det(*e1) && is_det(*e2))
                    ? static_cast<uint8_t>((*e0) ^ (*e1) ^ (*e2)) : SYM_Q;
                uint8_t S1_copy = S1_loc;

                uint8_t dummy_c = SYM_Q;
                const uint8_t cout_bit = (b + 1u < 32u) ? 0x10u : 0x00u;
                uint8_t* c3_c = &C_sym[(t * C_PLANES + 3u) * 32u + b];
                uint8_t* c3_n = (b + 1u < 32u) ? &C_sym[(t * C_PLANES + 3u) * 32u + b + 1u] : &dummy_c;
                if (prop_full_adder(&St[7u * 32u + b], &S1_copy, c3_c, &t1a_sym[t * 32u + b], c3_n, static_cast<uint8_t>(0x0Du | cout_bit))) {
                    any_changed = true; ++unit_prop_deductions;
                }
                if (!is_det(S1_loc) && is_det(S1_copy)) {
                    uint8_t* terms[3] = {e0, e1, e2};
                    if (prop_xor_vec(terms, 3, S1_copy)) {
                        any_changed = true; ++unit_prop_deductions;
                    }
                }

                dummy_c = SYM_Q;
                uint8_t* c4_c = &C_sym[(t * C_PLANES + 4u) * 32u + b];
                uint8_t* c4_n = (b + 1u < 32u) ? &C_sym[(t * C_PLANES + 4u) * 32u + b + 1u] : &dummy_c;
                if (prop_full_adder(&t1a_sym[t * 32u + b], &Ch_sym[t * 32u + b], c4_c, &t1b_sym[t * 32u + b], c4_n, static_cast<uint8_t>(0x0Fu | cout_bit))) {
                    any_changed = true; ++unit_prop_deductions;
                }

                dummy_c = SYM_Q;
                uint8_t kbit = static_cast<uint8_t>((K_WORDS[t] >> b) & 1u);
                uint8_t kcopy = kbit;
                uint8_t* c5_c = &C_sym[(t * C_PLANES + 5u) * 32u + b];
                uint8_t* c5_n = (b + 1u < 32u) ? &C_sym[(t * C_PLANES + 5u) * 32u + b + 1u] : &dummy_c;
                if (prop_full_adder(&t1b_sym[t * 32u + b], &kcopy, c5_c, &t1c_sym[t * 32u + b], c5_n, static_cast<uint8_t>(0x0Du | cout_bit))) {
                    any_changed = true; ++unit_prop_deductions;
                }

                dummy_c = SYM_Q;
                uint8_t* c6_c = &C_sym[(t * C_PLANES + 6u) * 32u + b];
                uint8_t* c6_n = (b + 1u < 32u) ? &C_sym[(t * C_PLANES + 6u) * 32u + b + 1u] : &dummy_c;
                if (prop_full_adder(&t1c_sym[t * 32u + b], &W_sym[t * 32u + b], c6_c, &T1_sym[t * 32u + b], c6_n, static_cast<uint8_t>(0x0Fu | cout_bit))) {
                    any_changed = true; ++unit_prop_deductions;
                }

                // Sigma0(a)
                uint8_t* a0 = &St[0u * 32u + ((b + 2u) & 31u)];
                uint8_t* a1 = &St[0u * 32u + ((b + 13u) & 31u)];
                uint8_t* a2 = &St[0u * 32u + ((b + 22u) & 31u)];
                uint8_t S0_loc = (is_det(*a0) && is_det(*a1) && is_det(*a2))
                    ? static_cast<uint8_t>((*a0) ^ (*a1) ^ (*a2)) : SYM_Q;
                uint8_t S0_copy = S0_loc;

                dummy_c = SYM_Q;
                uint8_t* c7_c = &C_sym[(t * C_PLANES + 7u) * 32u + b];
                uint8_t* c7_n = (b + 1u < 32u) ? &C_sym[(t * C_PLANES + 7u) * 32u + b + 1u] : &dummy_c;
                if (prop_full_adder(&S0_copy, &Maj_sym[t * 32u + b], c7_c, &T2_sym[t * 32u + b], c7_n, static_cast<uint8_t>(0x0Eu | cout_bit))) {
                    any_changed = true; ++unit_prop_deductions;
                }
                if (!is_det(S0_loc) && is_det(S0_copy)) {
                    uint8_t* terms[3] = {a0, a1, a2};
                    if (prop_xor_vec(terms, 3, S0_copy)) {
                        any_changed = true; ++unit_prop_deductions;
                    }
                }

                // a[t+1] = T1 + T2
                dummy_c = SYM_Q;
                uint8_t* c8_c = &C_sym[(t * C_PLANES + 8u) * 32u + b];
                uint8_t* c8_n = (b + 1u < 32u) ? &C_sym[(t * C_PLANES + 8u) * 32u + b + 1u] : &dummy_c;
                if (prop_full_adder(&T1_sym[t * 32u + b], &T2_sym[t * 32u + b], c8_c, &St1[0u * 32u + b], c8_n, static_cast<uint8_t>(0x0Fu | cout_bit))) {
                    any_changed = true; ++unit_prop_deductions;
                }

                // e[t+1] = d[t] + T1
                dummy_c = SYM_Q;
                uint8_t* c9_c = &C_sym[(t * C_PLANES + 9u) * 32u + b];
                uint8_t* c9_n = (b + 1u < 32u) ? &C_sym[(t * C_PLANES + 9u) * 32u + b + 1u] : &dummy_c;
                if (prop_full_adder(&St[3u * 32u + b], &T1_sym[t * 32u + b], c9_c, &St1[4u * 32u + b], c9_n, static_cast<uint8_t>(0x0Fu | cout_bit))) {
                    any_changed = true; ++unit_prop_deductions;
                }
            }
        };

        // Alternate forward (0 -> R-1) and backward (R-1 -> 0) sweeps for optimal bidirectional closure
        for (uint32_t t = 0; t < R && !conflict; ++t) prop_round(t);
        for (int32_t t = static_cast<int32_t>(R) - 1; t >= 0 && !conflict; --t) prop_round(static_cast<uint32_t>(t));
        return any_changed;
    }

    // Iterate Plane B (ARX unit prop) + Plane A (GF2 RREF) until fixed point or conflict
    void propagate_to_fixpoint(bool use_gf2 = true, uint32_t max_iters = 16u) {
        for (uint32_t it = 0; it < max_iters && !conflict; ++it) {
            bool chg_b = false;
            for (uint32_t sw = 0; sw < 128u && !conflict && sweep_arx_once(); ++sw) {
                chg_b = true;
            }
            if (conflict) break;
            bool chg_a = false;
            if (use_gf2) {
                sync_all_known_to_gf2();
                if (!conflict) {
                    chg_a = pull_gf2_units();
                }
            }
            if (!chg_b && !chg_a) break;
        }
    }

    uint32_t count_unknown_primary() const {
        uint32_t cnt = 0u;
        for (uint8_t v : W_sym) if (!is_det(v)) ++cnt;
        for (uint8_t v : S_sym) if (!is_det(v)) ++cnt;
        for (uint8_t v : C_sym) if (!is_det(v)) ++cnt;
        return cnt;
    }
};

}  // namespace ouroboros
