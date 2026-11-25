import PhaseSHA.Core.BitVecIdentities
import PhaseSHA.Algebraic.LiteField

/-!
# Module 4: V4E BTC 部分零前缀剥离与复利放大定理

本模块形式化验证 Phase-SHA V4E 在比特币部分原像（Partial Preimage PoW）测试中的两大底层性质：
1. **`btc_h7_tail_round_decoupling`**（及 **`btc_h0_h4_sensitive_to_last_w`**、**`btc_r20_w16_reaches_h7`**）：
   证明在 `R` 轮 SHA-256 中，最终输出字 `H_7 = IV_7 + E_{R-3}` 在代数上与最后 3 轮
   `W_{R-3}, W_{R-2}, W_{R-1}` 严格无关；而 `R=18` 时扩展消息字 `W_{16}, W_{17}` 恰好全部落入该尾 3 轮盲区，
   因此 `R=18` 闭环测试必须钳位 `H_0 / H_4` 或将轮数推进至 `R ≥ 20` 才能使 `H_7` 耦合消息扩展链；
2. **`partial_preimage_exponential_gain`**：
   严格证明在 `Z` 个独立目标零位上，单比特命中偏置 `p > 1/2` 带来的期望倍率严格为 `(2p)^Z`，
   随 `Z` 呈严格指数复利放大，并给出达到目标加速比 `M` 所需的最小单比特偏置闭式充要下界 `p ≥ (1/2) * M^{1/Z}`。
-/

namespace PhaseSHA.Core.PartialPreimagePoW

open PhaseSHA.Core.BitVecIdentities
open PhaseSHA.Algebraic.LiteField

/-!
### 1. `H_7` 尾 3 轮消息字解耦与 `R ≥ 20` 耦合边界定理
-/

/-- 连续执行 SHA-256 最后 3 轮压缩转移（对应轮次 `R-3, R-2, R-1`）。 -/
@[inline] def sha256ThreeRounds (s : SHAState)
    (k0 k1 k2 w0 w1 w2 : BitVec 32) : SHAState :=
  let s1 := sha256Round s k0 w0
  let s2 := sha256Round s1 k1 w1
  sha256Round s2 k2 w2

private theorem bv_add_comm_32 (x y : BitVec 32) : x + y = y + x := by
  bv_decide

private theorem bv_add_inj_t2_iv (c w w' t2 iv : BitVec 32)
    (h : ((c + w) + t2) + iv = ((c + w') + t2) + iv) : w = w' := by
  bv_decide

private theorem bv_add_inj_d_iv (d c w w' iv : BitVec 32)
    (h : (d + (c + w)) + iv = (d + (c + w')) + iv) : w = w' := by
  bv_decide

/--
**定理 1（`btc_h7_tail_round_decoupling`）**：
对任意第 `R-3` 轮前状态 `s`（其 `s.e` 即 `E_{R-3}`）、任意初始向量 `iv` 以及任意两组
尾 3 轮轮常量与消息字 `(k0, k1, k2, w0, w1, w2)`、`(k0', k1', k2', w0', w1', w2')`：
1. 最终输出字 `H_7 = IV_7 + E_{R-3}` 精确成立；
2. `H_7` 在代数上与最后 3 轮消息字 `W_{R-3}, W_{R-2}, W_{R-1}`（及对应轮常量）严格无关。
-/
theorem btc_h7_tail_round_decoupling (s iv : SHAState)
    (k0 k1 k2 w0 w1 w2 k0' k1' k2' w0' w1' w2' : BitVec 32) :
    (addIV (sha256ThreeRounds s k0 k1 k2 w0 w1 w2) iv).h = iv.h + s.e ∧
    (addIV (sha256ThreeRounds s k0 k1 k2 w0 w1 w2) iv).h =
      (addIV (sha256ThreeRounds s k0' k1' k2' w0' w1' w2') iv).h := by
  have hEq : (addIV (sha256ThreeRounds s k0 k1 k2 w0 w1 w2) iv).h = iv.h + s.e := by
    simp only [addIV, sha256ThreeRounds, sha256Round]
    exact bv_add_comm_32 s.e iv.h
  have hEq' : (addIV (sha256ThreeRounds s k0' k1' k2' w0' w1' w2') iv).h = iv.h + s.e := by
    simp only [addIV, sha256ThreeRounds, sha256Round]
    exact bv_add_comm_32 s.e iv.h
  exact ⟨hEq, hEq.trans hEq'.symm⟩

/--
对偶边界定理：`H_0` 与 `H_4` 对最后一轮消息字 `W_{R-1}` 均具有严格单射敏感性（`w ≠ w' → H_0 ≠ H_0' ∧ H_4 ≠ H_4'`）。
因此当 `R=18` 时，虽然 `H_7` 完全屏蔽 `W_{15..17}`，但 `H_0` 与 `H_4` 直接暴露末轮约束。
-/
theorem btc_h0_h4_sensitive_to_last_w (s iv : SHAState) (k w w' : BitVec 32)
    (hne : w ≠ w') :
    (addIV (sha256Round s k w) iv).a ≠ (addIV (sha256Round s k w') iv).a ∧
    (addIV (sha256Round s k w) iv).e ≠ (addIV (sha256Round s k w') iv).e := by
  simp only [addIV, sha256Round]
  constructor
  · intro hEq
    exact hne (bv_add_inj_t2_iv (s.h + Sigma1 s.e + ch32 s.e s.f s.g + k)
      w w' (Sigma0 s.a + maj32 s.a s.b s.c) iv.a hEq)
  · intro hEq
    exact hne (bv_add_inj_d_iv s.d (s.h + Sigma1 s.e + ch32 s.e s.f s.g + k)
      w w' iv.e hEq)

/--
`R=20` 消息扩展首达 `H_7` 定理：
第 16 轮之后还有 3 轮（第 `17, 18, 19` 轮），因此 `H_7 = IV_7 + E_{17}` 首次严格单射依赖于
第 1 个扩展消息字 `W_{16}`（`w16 ≠ w16' → H_7 ≠ H_7'`）；而在 `R=18` 时 `W_{16}, W_{17}` 落入尾 3 轮盲区。
-/
theorem btc_r20_w16_reaches_h7 (s16 iv : SHAState)
    (k16 k17 k18 k19 w16 w16' w17 w18 w19 : BitVec 32)
    (hne : w16 ≠ w16') :
    let dig1 := addIV (sha256ThreeRounds (sha256Round s16 k16 w16) k17 k18 k19 w17 w18 w19) iv
    let dig2 := addIV (sha256ThreeRounds (sha256Round s16 k16 w16') k17 k18 k19 w17 w18 w19) iv
    dig1.h ≠ dig2.h := by
  intro dig1 dig2 hEq
  have h1 := (btc_h7_tail_round_decoupling (sha256Round s16 k16 w16) iv k17 k18 k19 w17 w18 w19
    k17 k18 k19 w17 w18 w19).1
  have h2 := (btc_h7_tail_round_decoupling (sha256Round s16 k16 w16') iv k17 k18 k19 w17 w18 w19
    k17 k18 k19 w17 w18 w19).1
  rw [h1, h2] at hEq
  have hComm1 := bv_add_comm_32 iv.h (sha256Round s16 k16 w16).e
  have hComm2 := bv_add_comm_32 iv.h (sha256Round s16 k16 w16').e
  rw [hComm1, hComm2] at hEq
  simp only [sha256Round] at hEq
  exact hne (bv_add_inj_d_iv s16.d (s16.h + Sigma1 s16.e + ch32 s16.e s16.f s16.g + k16)
    w16 w16' iv.h hEq)

/-!
### 2. 部分零前缀命中概率复利放大与最小单比特偏置闭式下界
-/

/-- 均匀随机基线命中 `Z` 个独立目标零位的概率 `(1/2)^Z`。 -/
def unifProb (Z : Nat) : Rat := ratPow (1 / 2) Z

/-- 单比特零命中率为 `p` 时，同时命中 `Z` 个独立目标零位的概率 `p^Z`。 -/
def biasProb (p : Rat) (Z : Nat) : Rat := ratPow p Z

/-- 期望命中加速倍率 `biasProb p Z / unifProb Z`。 -/
def expectedGain (p : Rat) (Z : Nat) : Rat := biasProb p Z / unifProb Z

private theorem rat_div_half (p : Rat) : p / (1 / 2 : Rat) = 2 * p := by
  have h2 : (1 / 2 : Rat)⁻¹ = 2 := by grind
  rw [Rat.div_def, h2]
  grind

theorem expectedGain_eq_pow_two_mul (p : Rat) (Z : Nat) :
    expectedGain p Z = ratPow (2 * p) Z := by
  unfold expectedGain biasProb unifProb
  have hHalfNe : (1 / 2 : Rat) ≠ 0 := by grind
  rw [ratPow_div p (1 / 2) hHalfNe Z, rat_div_half p]

/--
**定理 2（`partial_preimage_exponential_gain`）**：
在 `Z` 个独立目标零位上，若采样分布将单比特零概率设为 `p`：
1. 期望命中倍率严格等于 `(2 * p)^Z`；
2. 只要 `p > 1/2` 且 `Z ≥ 1`，不仅期望倍率严格大于 `1`，而且关于零位个数 `Z` 呈严格指数增长
   （`expectedGain p Z < expectedGain p (Z + 1)`）；
3. 达到目标加速比 `M > 0` 的充要条件为 `(2 * p)^Z ≥ M`；特别地，对任意满足 `r^Z = M` 的
   非负 `Z` 次根 `r = M^{1/Z}`，达到倍率 `M` 的最小单比特偏置闭式充要下界为 `p ≥ (1/2) * r`。
-/
theorem partial_preimage_exponential_gain (p M r : Rat) (Z : Nat) :
    (expectedGain p Z = ratPow (2 * p) Z) ∧
    ((1 / 2 < p → 1 ≤ Z → 1 < expectedGain p Z ∧ expectedGain p Z < expectedGain p (Z + 1))) ∧
    ((0 ≤ p → 0 < M → 1 ≤ Z → (M ≤ expectedGain p Z ↔ M ≤ ratPow (2 * p) Z))) ∧
    ((0 ≤ p → 0 ≤ r → 1 ≤ Z → ratPow r Z = M →
      (M ≤ expectedGain p Z ↔ (1 / 2) * r ≤ p))) := by
  refine ⟨expectedGain_eq_pow_two_mul p Z, ?_, ?_, ?_⟩
  · intro hp hZ
    rw [expectedGain_eq_pow_two_mul p Z, expectedGain_eq_pow_two_mul p (Z + 1)]
    have h2p : 1 < 2 * p := by grind
    have hGt1 : 1 < ratPow (2 * p) Z := by
      have h1 : ratPow (1 : Rat) Z = 1 := by
        clear hp hZ h2p
        induction Z with
        | zero => rfl
        | succ k ih => simp [ratPow_succ, ih]
      rw [← h1]
      exact ratPow_lt_ratPow_of_lt (by decide) h2p hZ
    exact ⟨hGt1, ratPow_lt_succ_of_gt_one h2p Z⟩
  · intro _ _ _
    rw [expectedGain_eq_pow_two_mul p Z]
  · intro hp hr hZ hRoot
    rw [expectedGain_eq_pow_two_mul p Z, ← hRoot]
    have h2pNonneg : 0 ≤ 2 * p := by grind
    rw [ratPow_le_ratPow_iff hr h2pNonneg hZ]
    grind

/-- 无理 `Z` 次根情况下的上下夹逼定量下界推论：任意有理逼近上/下界均直接约束所需单比特偏置 `p`。 -/
theorem partial_preimage_bias_sandwich {p M rLo rHi : Rat} {Z : Nat}
    (_hrLo : 0 ≤ rLo) (hrHi : 0 ≤ rHi) (hZ : 1 ≤ Z)
    (hLo : ratPow rLo Z ≤ M) (hHi : M ≤ ratPow rHi Z) :
    ((1 / 2) * rHi ≤ p → M ≤ expectedGain p Z) ∧
    (0 ≤ p → p < (1 / 2) * rLo → expectedGain p Z < M) := by
  rw [expectedGain_eq_pow_two_mul p Z]
  constructor
  · intro hge
    have h2p : rHi ≤ 2 * p := by grind
    have h2pNonneg : 0 ≤ 2 * p := by grind
    have hPow := (ratPow_le_ratPow_iff hrHi h2pNonneg hZ).mpr h2p
    grind
  · intro hp hlt
    have h2pNonneg : 0 ≤ 2 * p := by grind
    have h2p : 2 * p < rLo := by grind
    have hPow := ratPow_lt_ratPow_of_lt h2pNonneg h2p hZ
    grind

end PhaseSHA.Core.PartialPreimagePoW
