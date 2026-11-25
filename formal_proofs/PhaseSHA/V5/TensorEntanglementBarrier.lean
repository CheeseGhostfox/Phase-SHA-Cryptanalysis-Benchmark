import PhaseSHA.Algebraic.LiteField
import PhaseSHA.Algebraic.NonAbelianLiftAndOGP

/-!
# Module 6C: 张量网络纠缠态消除平均场势垒 (`TensorEntanglementBarrier.lean`, MPS Rank-2 升维)

本模块从第一性原理在零外部依赖下严格验证“独立自旋平均场势垒 vs Rank-2 纠缠态零势垒穿越”：
1. **`independent_spin_mean_field_barrier`**：
   对闭环约束 `s_a s_b = +1` 与独立连续自旋残差 `E_ind(x_a, x_b) = (x_a x_b - 1)^2`：
   - 对称均值点 `(0, 0)` 产生严格正残差 `E_ind(0, 0) = 1`；
   - 任意穿零点 `x_a = 0 ∨ x_b = 0` 产生 `E_ind = 1`（`x_a x_b ≤ 0` 恒有 `E_ind ≥ 1`）；
   - 在无须假设实数完备性与介值定理（IVT）公理的前提下，纯在精确有理域 $\mathbb{Q}$ 上通过有限等分网格
     构造性地证明：任何连接 `(1, 1)` 与 `(-1, -1)` 且取值于 `[-1, 1]^2` 的一致连续路径，对**任意**
     `B < 1` 必存在采样点使 `E_ind > B`（平均场势垒严格 `≥ 1`）。
2. **`rank2_entangled_zero_barrier`**：
   将双比特提升至 Rank-2 联合纠缠振幅向量 `ψ = (ψ_{++}, ψ_{+-}, ψ_{-+}, ψ_{--})`：
   - 显式有理曲线 `entPath t := ⟨pathCos (t/2), 0, 0, pathSin (t/2)⟩` 在 `[0, 1]` 上精确连接 `|+1,+1⟩`
     与 `|-1,-1⟩`，满足全局 `4`-Lipschitz 连续、全程单位范数 `‖ψ‖² = 1`、全程约束能量恒零 `E_ent = 0`、
     总关联 `⟨s_a s_b⟩ = 1`，且可在纯 $\mathbb{Q}$ 上以任意精度逼近零单比特边缘期望；
   - 在精确二次扩域 `QSqrt2`（$\mathbb{Q}(\sqrt{2})$）上构造精确 $\theta = \pi/4$ Bell 态
     `bellPlus = (√2/2, 0, 0, √2/2)`，同时精确满足 `‖ψ‖² = 1 ∧ E_ent = 0 ∧ ⟨s_a⟩ = 0 ∧ ⟨s_b⟩ = 0 ∧ ⟨s_a s_b⟩ = 1`。
3. **`mps_rank2_cycle_zero_residual`** 与诚实对照 **`rank1_product_zero_marginals_positive_energy`**：
   推广至任意长度 `L ≥ 2` 的闭环链式约束，证明键维 `χ = 2` 的 Rank-2 MPS 沿连续穿墙路径全程
   总范数平方 `= 1`、循环违约能量恒为 `0`，且其 `QSqrt2` GHZ 态在**全部分量 `k : Fin L`** 上
   均达到精确零边缘期望；同时诚实证明键维 `χ = 1` 的独立乘积态只要出现任意归零边缘期望
   即必然产生 `≥ 1` 违约能量（并在注解中指出：常数键维 `χ = 2` 仅覆盖双分支对称流形，
   面对一般多分支耦合约束所需键维将随割集纠缠熵增长）。
-/

namespace PhaseSHA.V5.TensorEntanglementBarrier

open PhaseSHA.Algebraic.LiteField
open PhaseSHA.Algebraic.NonAbelianLiftAndOGP

/-!
### 1. 独立自旋平均场势垒定理 (`independent_spin_mean_field_barrier`)
-/

/-- 独立连续自旋的闭环约束残差：`E_ind(x_a, x_b) = (x_a x_b - 1)^2`。 -/
def indEnergy (xa xb : Rat) : Rat := (xa * xb - 1) * (xa * xb - 1)

/-- `f : Rat → Rat` 在闭区间 `[0, 1]` 上的一致连续性。 -/
def UniformContinuousOn01 (f : Rat → Rat) : Prop :=
  ∀ eps : Rat, 0 < eps → ∃ delta : Rat, 0 < delta ∧
    ∀ t1 t2 : Rat, 0 ≤ t1 → t1 ≤ 1 → 0 ≤ t2 → t2 ≤ 1 →
      ratAbs (t1 - t2) < delta → ratAbs (f t1 - f t2) < eps

/-- 递归 `Nat → Rat` 嵌入（避开标准库数值类型转换归约开销）。 -/
def natToRat : Nat → Rat
  | 0 => 0
  | n + 1 => natToRat n + 1

private theorem natToRat_nonneg (n : Nat) : 0 ≤ natToRat n := by
  induction n with
  | zero => unfold natToRat; grind
  | succ k ih => unfold natToRat; grind

private theorem natToRat_pos {n : Nat} (hn : 0 < n) : 0 < natToRat n := by
  induction n with
  | zero => omega
  | succ k _ =>
    unfold natToRat
    have hk := natToRat_nonneg k
    grind

private theorem natToRat_le {m n : Nat} (h : m ≤ n) : natToRat m ≤ natToRat n := by
  induction h with
  | refl => grind
  | step _ ih =>
    change natToRat m ≤ natToRat _ + 1
    grind

private theorem intCast_nat_eq_natToRat (n : Nat) : ((n : Int) : Rat) = natToRat n := by
  induction n with
  | zero => rfl
  | succ k ih =>
    have hZ : ((k + 1 : Nat) : Int) = (k : Int) + 1 := by omega
    rw [hZ, Rat.intCast_add, ih]
    rfl

private theorem archimedean_step_lt (delta : Rat) (hDelta : 0 < delta) :
    ∃ N : Nat, 0 < N ∧ 1 / natToRat N < delta := by
  have hInvPos : 0 < 1 / delta := rat_div_pos (by grind) hDelta
  let z : Int := ratFloor (1 / delta) + 1
  have hLt : 1 / delta < (z : Rat) := by
    have h0 := lt_ratFloor_add_one (1 / delta)
    simpa [z] using h0
  have hzRatPos : (0 : Rat) < (z : Rat) := by grind
  have hzIntPos : 0 < z := by
    rw [Rat.lt_iff] at hzRatPos
    change (0 : Int) * 1 < z * 1 at hzRatPos
    omega
  let N : Nat := z.toNat
  have hNPos : 0 < N := by dsimp [N]; omega
  have hNEqInt : (N : Int) = z := Int.toNat_of_nonneg (by omega)
  have hNEqRat : natToRat N = (z : Rat) := by
    rw [← intCast_nat_eq_natToRat N, hNEqInt]
  have hNPosRat : 0 < natToRat N := natToRat_pos hNPos
  have hInvLt : 1 / delta < natToRat N := by rw [hNEqRat]; exact hLt
  refine ⟨N, hNPos, ?_⟩
  have h1 : (1 / delta) * delta < natToRat N * delta :=
    Rat.mul_lt_mul_of_pos_right hInvLt hDelta
  have h2 : (1 / delta) * delta = 1 := by
    rw [Rat.div_def]
    have hNe : delta ≠ 0 := by grind
    have hAssoc : (1 * delta⁻¹) * delta = delta * delta⁻¹ := by grind
    rw [hAssoc, Rat.mul_inv_cancel delta hNe]
  rw [h2] at h1
  have hInvNPos : 0 < (natToRat N)⁻¹ := Rat.inv_pos.mpr hNPosRat
  have h3 : 1 * (natToRat N)⁻¹ < (natToRat N * delta) * (natToRat N)⁻¹ :=
    Rat.mul_lt_mul_of_pos_right h1 hInvNPos
  have hNNe : natToRat N ≠ 0 := by grind
  have hL : 1 * (natToRat N)⁻¹ = 1 / natToRat N := by rw [Rat.div_def]
  have hR : (natToRat N * delta) * (natToRat N)⁻¹ = delta := by
    have hAssoc : (natToRat N * delta) * (natToRat N)⁻¹ =
        delta * (natToRat N * (natToRat N)⁻¹) := by grind
    rw [hAssoc, Rat.mul_inv_cancel _ hNNe]
    grind
  rw [hL, hR] at h3
  exact h3

private theorem exists_grid_sign_change (N : Nat) (g : Nat → Rat)
    (h0 : 0 < g 0) (hN : g N ≤ 0) :
    ∃ k : Nat, k < N ∧ 0 < g k ∧ g (k + 1) ≤ 0 := by
  induction N with
  | zero => exfalso; grind
  | succ m ih =>
    by_cases hm : g m ≤ 0
    · rcases ih hm with ⟨k, hkLt, hkPos, hk1⟩
      exact ⟨k, by omega, hkPos, hk1⟩
    · have hmPos : 0 < g m := by grind
      exact ⟨m, by omega, hmPos, hN⟩

private theorem rat_lt_trans {a b c : Rat} (h1 : a < b) (h2 : b < c) : a < c := by grind

private theorem mean_field_uc_barrier
    (xa xb : Rat → Rat)
    (hxa0 : xa 0 = 1) (hxa1 : xa 1 = -1)
    (hBox : ∀ t : Rat, 0 ≤ t → t ≤ 1 → -1 ≤ xa t ∧ xa t ≤ 1 ∧ -1 ≤ xb t ∧ xb t ≤ 1)
    (hUC : UniformContinuousOn01 xa)
    (B : Rat) (hB : B < 1) :
    ∃ t : Rat, 0 ≤ t ∧ t ≤ 1 ∧ B < indEnergy (xa t) (xb t) := by
  let eps : Rat := if (1 - B) / 2 ≤ 1 / 2 then (1 - B) / 2 else 1 / 2
  have hEpsPos : 0 < eps := by dsimp [eps]; split <;> grind
  have hEpsLeHalf : eps ≤ 1 / 2 := by dsimp [eps]; split <;> grind
  have hEpsBound : B ≤ 1 - 2 * eps := by dsimp [eps]; split <;> grind
  rcases hUC eps hEpsPos with ⟨delta, hDeltaPos, hDelta⟩
  rcases archimedean_step_lt delta hDeltaPos with ⟨N, hNPos, hStepLt⟩
  have hNRatPos : 0 < natToRat N := natToRat_pos hNPos
  let gridPt : Nat → Rat := fun j => natToRat j / natToRat N
  have hGridBounds : ∀ j : Nat, j ≤ N → 0 ≤ gridPt j ∧ gridPt j ≤ 1 := by
    intro j hj
    dsimp [gridPt]
    constructor
    · exact rat_div_nonneg (natToRat_nonneg j) hNRatPos
    · rw [rat_div_le_iff hNRatPos]
      have hLe := natToRat_le hj
      grind
  have hG0 : 0 < xa (gridPt 0) := by
    have ht0 : gridPt 0 = 0 := by dsimp [gridPt, natToRat]; grind
    rw [ht0, hxa0]; grind
  have hGN : xa (gridPt N) ≤ 0 := by
    have ht1 : gridPt N = 1 := by
      dsimp [gridPt]
      rw [Rat.div_def, Rat.mul_inv_cancel _ (by grind)]
    rw [ht1, hxa1]; grind
  rcases exists_grid_sign_change N (fun j => xa (gridPt j)) hG0 hGN with
    ⟨k, hkLt, hkPos, _⟩
  have hkLe : k ≤ N := by omega
  have hk1Le : k + 1 ≤ N := by omega
  have htkBounds := hGridBounds k hkLe
  have htk1Bounds := hGridBounds (k + 1) hk1Le
  have hDiffStep : gridPt (k + 1) - gridPt k = 1 / natToRat N := by
    dsimp [gridPt]
    have hSucc : natToRat (k + 1) = natToRat k + 1 := rfl
    rw [hSucc, Rat.div_def, Rat.div_def, Rat.div_def]
    grind
  have hDistLt : ratAbs (gridPt (k + 1) - gridPt k) < delta := by
    rw [hDiffStep, ratAbs_of_nonneg (rat_div_nonneg (by grind) hNRatPos)]
    exact hStepLt
  have hXaDiffLt := hDelta (gridPt (k + 1)) (gridPt k)
    htk1Bounds.1 htk1Bounds.2 htkBounds.1 htkBounds.2 hDistLt
  have hXaLt : xa (gridPt k) < eps := by
    have hAbs : ratAbs (xa (gridPt (k + 1)) - xa (gridPt k)) =
        xa (gridPt k) - xa (gridPt (k + 1)) := by
      rw [ratAbs_of_nonpos (by grind)]
      grind
    rw [hAbs] at hXaDiffLt
    grind
  have hXbBox := hBox (gridPt k) htkBounds.1 htkBounds.2
  have hXbAbs : ratAbs (xb (gridPt k)) ≤ 1 := by
    rw [ratAbs_le_iff]
    exact ⟨hXbBox.2.2.1, hXbBox.2.2.2⟩
  have hProdAbs : ratAbs (xa (gridPt k) * xb (gridPt k)) ≤ xa (gridPt k) := by
    rw [ratAbs_mul, ratAbs_of_nonneg (Rat.le_of_lt hkPos)]
    have h1 := rat_mul_le_mul_of_nonneg_left hXbAbs (Rat.le_of_lt hkPos)
    grind
  have hProdLt : xa (gridPt k) * xb (gridPt k) < eps := by
    have h1 : xa (gridPt k) * xb (gridPt k) ≤ ratAbs (xa (gridPt k) * xb (gridPt k)) := by
      unfold ratAbs; split <;> grind
    exact rat_lt_of_le_of_lt (Rat.le_trans h1 hProdAbs) hXaLt
  have hDiffPos : 1 - eps < 1 - xa (gridPt k) * xb (gridPt k) := by grind
  have hBasePos : 0 < 1 - eps := by grind
  have hSqLt : (1 - eps) * (1 - eps) <
      (1 - xa (gridPt k) * xb (gridPt k)) * (1 - xa (gridPt k) * xb (gridPt k)) := by
    have h1 : (1 - eps) * (1 - eps) < (1 - eps) * (1 - xa (gridPt k) * xb (gridPt k)) :=
      Rat.mul_lt_mul_of_pos_left hDiffPos hBasePos
    have h2 : (1 - eps) * (1 - xa (gridPt k) * xb (gridPt k)) <
        (1 - xa (gridPt k) * xb (gridPt k)) * (1 - xa (gridPt k) * xb (gridPt k)) :=
      Rat.mul_lt_mul_of_pos_right hDiffPos (by grind)
    exact rat_lt_trans h1 h2
  have hEnEq : indEnergy (xa (gridPt k)) (xb (gridPt k)) =
      (1 - xa (gridPt k) * xb (gridPt k)) * (1 - xa (gridPt k) * xb (gridPt k)) := by
    unfold indEnergy; grind
  have hEpsSqPos : 0 < eps * eps := Rat.mul_pos hEpsPos hEpsPos
  have hOneMinus : B < (1 - eps) * (1 - eps) := by
    have hId : (1 - eps) * (1 - eps) = 1 - 2 * eps + eps * eps := by grind
    grind
  refine ⟨gridPt k, htkBounds.1, htkBounds.2, ?_⟩
  rw [hEnEq]
  exact rat_lt_trans hOneMinus hSqLt

/--
**定理 1（`independent_spin_mean_field_barrier`）**：
对耦合闭环 `s_a s_b = +1` 上的独立连续自旋残差 `indEnergy xa xb = (xa * xb - 1)^2`：
1. 对称均值 `(0, 0)` 产生严格势垒 `indEnergy 0 0 = 1`；
2. 凡是任一坐标过零点（`xa = 0 ∨ xb = 0`）均满足 `indEnergy xa xb = 1`，且反号区域 `xa * xb ≤ 0`
   恒满足 `1 ≤ indEnergy xa xb`；
3. 若连续路径经过某点 `t* ∈ [0, 1]` 使 `xa t* = 0 ∨ xb t* = 0`，则该点势垒高度恰好等于 `1`；
4. **纯有理无 IVT 路径势垒**：任何连接 `(1, 1)` 与 `(-1, -1)`、取值在 `[-1, 1]^2` 内且 `xa`
   在 `[0, 1]` 上一致连续的路径 `(xa, xb)`，对**任意**目标界 `B < 1` 必存在有理采样点
   `t ∈ [0, 1]` 使 `B < indEnergy (xa t) (xb t)`（即路径势垒上确界严格 `≥ 1`）。
-/
theorem independent_spin_mean_field_barrier :
    indEnergy 0 0 = 1 ∧
    (∀ xa xb : Rat, (xa = 0 ∨ xb = 0) → indEnergy xa xb = 1) ∧
    (∀ xa xb : Rat, xa * xb ≤ 0 → 1 ≤ indEnergy xa xb) ∧
    (∀ xa xb : Rat → Rat, (∃ t : Rat, 0 ≤ t ∧ t ≤ 1 ∧ (xa t = 0 ∨ xb t = 0)) →
      ∃ t : Rat, 0 ≤ t ∧ t ≤ 1 ∧ indEnergy (xa t) (xb t) = 1) ∧
    (∀ xa xb : Rat → Rat,
      xa 0 = 1 → xb 0 = 1 → xa 1 = -1 → xb 1 = -1 →
      (∀ t : Rat, 0 ≤ t → t ≤ 1 → -1 ≤ xa t ∧ xa t ≤ 1 ∧ -1 ≤ xb t ∧ xb t ≤ 1) →
      UniformContinuousOn01 xa →
      ∀ B : Rat, B < 1 → ∃ t : Rat, 0 ≤ t ∧ t ≤ 1 ∧ B < indEnergy (xa t) (xb t)) := by
  refine ⟨by unfold indEnergy; grind, ?_, ?_, ?_, ?_⟩
  · intro xa xb h
    unfold indEnergy
    rcases h with rfl | rfl <;> grind
  · intro xa xb h
    unfold indEnergy
    have hsq := rat_sq_nonneg (xa * xb)
    grind
  · intro xa xb ⟨t, ht0, ht1, htZero⟩
    refine ⟨t, ht0, ht1, ?_⟩
    unfold indEnergy
    rcases htZero with h1 | h2
    · simp [h1]; grind
    · simp [h2]; grind
  · intro xa xb hxa0 _ hxa1 _ hBox hUC B hB
    exact mean_field_uc_barrier xa xb hxa0 hxa1 hBox hUC B hB

/-!
### 2. Rank-2 双比特纠缠联合态零势垒精确叠加定理 (`rank2_entangled_zero_barrier`)
-/

/-- 精确有理基下的双比特联合纯态振幅 `ψ = (ψ_{++}, ψ_{+-}, ψ_{-+}, ψ_{--}) ∈ ℚ⁴`。 -/
structure TwoQubitState where
  pp : Rat
  pm : Rat
  mp : Rat
  mm : Rat
  deriving DecidableEq, Repr

namespace TwoQubitState

@[ext] theorem ext {ψ1 ψ2 : TwoQubitState}
    (hpp : ψ1.pp = ψ2.pp) (hpm : ψ1.pm = ψ2.pm)
    (hmp : ψ1.mp = ψ2.mp) (hmm : ψ1.mm = ψ2.mm) : ψ1 = ψ2 := by
  cases ψ1; cases ψ2; simp_all

def normSq (ψ : TwoQubitState) : Rat :=
  ψ.pp * ψ.pp + ψ.pm * ψ.pm + ψ.mp * ψ.mp + ψ.mm * ψ.mm

/-- 约束违背能量 `E_ent(ψ) = ψ_{+-}² + ψ_{-+}²`。 -/
def entEnergy (ψ : TwoQubitState) : Rat :=
  ψ.pm * ψ.pm + ψ.mp * ψ.mp

/-- 单比特 `s_a` 边缘期望 `⟨s_a⟩ = (ψ_{++}² + ψ_{+-}²) - (ψ_{-+}² + ψ_{--}²)`。 -/
def expSa (ψ : TwoQubitState) : Rat :=
  (ψ.pp * ψ.pp + ψ.pm * ψ.pm) - (ψ.mp * ψ.mp + ψ.mm * ψ.mm)

/-- 单比特 `s_b` 边缘期望 `⟨s_b⟩ = (ψ_{++}² + ψ_{-+}²) - (ψ_{+-}² + ψ_{--}²)`。 -/
def expSb (ψ : TwoQubitState) : Rat :=
  (ψ.pp * ψ.pp + ψ.mp * ψ.mp) - (ψ.pm * ψ.pm + ψ.mm * ψ.mm)

/-- 双比特关联期望 `⟨s_a s_b⟩ = (ψ_{++}² + ψ_{--}²) - (ψ_{+-}² + ψ_{-+}²)`。 -/
def expSaSb (ψ : TwoQubitState) : Rat :=
  (ψ.pp * ψ.pp + ψ.mm * ψ.mm) - (ψ.pm * ψ.pm + ψ.mp * ψ.mp)

def stateDist (ψ1 ψ2 : TwoQubitState) : Rat :=
  max (max (ratAbs (ψ1.pp - ψ2.pp)) (ratAbs (ψ1.pm - ψ2.pm)))
      (max (ratAbs (ψ1.mp - ψ2.mp)) (ratAbs (ψ1.mm - ψ2.mm)))

end TwoQubitState

/-- 连接 `|+1,+1⟩`（`t=0`）与 `|-1,-1⟩`（`t=1`）的连续保范零违约有理纠缠路径。 -/
def entPath (t : Rat) : TwoQubitState :=
  ⟨pathCos (t / 2), 0, 0, pathSin (t / 2)⟩

private theorem entPath_diff_abs_le (f : Rat → Rat) (hf : IsLipschitzOn01 f 8)
    (t t0 : Rat) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (ht00 : 0 ≤ t0) (ht01 : t0 ≤ 1) :
    ratAbs (f (t / 2) - f (t0 / 2)) ≤ 4 * ratAbs (t - t0) := by
  have hBound := hf.2 (t / 2) (t0 / 2) (by grind) (by grind) (by grind) (by grind)
  have hHalfEq : t / 2 - t0 / 2 = (1 / 2 : Rat) * (t - t0) := by grind
  rw [hHalfEq, ratAbs_mul, ratAbs_of_nonneg (by grind : (0 : Rat) ≤ 1 / 2)] at hBound
  grind

private theorem rat_abs_le_one_of_sq_le_one {x : Rat} (hx : x * x ≤ 1) : ratAbs x ≤ 1 := by
  have hNonnegCase : ∀ y : Rat, 0 ≤ y → y * y ≤ 1 → y ≤ 1 := by
    intro y hy0 hy1
    by_cases h : y ≤ 1
    · exact h
    · exfalso
      have hyGt : 1 < y := by grind
      have hMul : y * 1 < y * y := Rat.mul_lt_mul_of_pos_left hyGt (by grind : 0 < y)
      grind
  rw [ratAbs_le_iff]
  constructor
  · have hNegSq : (-x) * (-x) ≤ 1 := by grind
    by_cases hx0 : x < 0
    · have h := hNonnegCase (-x) (by grind) hNegSq
      grind
    · grind
  · by_cases hx0 : 0 ≤ x
    · exact hNonnegCase x hx0 hx
    · grind

private theorem entMarginDiff_lipschitz (u1 u2 : Rat)
    (hu1a : 0 ≤ u1) (hu1b : u1 ≤ 1) (hu2a : 0 ≤ u2) (hu2b : u2 ≤ 1) :
    ratAbs ((pathCos u1 * pathCos u1 - pathSin u1 * pathSin u1) -
      (pathCos u2 * pathCos u2 - pathSin u2 * pathSin u2)) ≤ 32 * ratAbs (u1 - u2) := by
  have hcSum1 : ratAbs (pathCos u1) ≤ 1 := by
    have h := pathCos_sq_add_pathSin_sq u1
    have hsq := rat_sq_nonneg (pathSin u1)
    exact rat_abs_le_one_of_sq_le_one (by grind)
  have hcSum2 : ratAbs (pathCos u2) ≤ 1 := by
    have h := pathCos_sq_add_pathSin_sq u2
    have hsq := rat_sq_nonneg (pathSin u2)
    exact rat_abs_le_one_of_sq_le_one (by grind)
  have hsSum1 : ratAbs (pathSin u1) ≤ 1 := by
    have h := pathCos_sq_add_pathSin_sq u1
    have hsq := rat_sq_nonneg (pathCos u1)
    exact rat_abs_le_one_of_sq_le_one (by grind)
  have hsSum2 : ratAbs (pathSin u2) ≤ 1 := by
    have h := pathCos_sq_add_pathSin_sq u2
    have hsq := rat_sq_nonneg (pathCos u2)
    exact rat_abs_le_one_of_sq_le_one (by grind)
  have hDiffEq :
      (pathCos u1 * pathCos u1 - pathSin u1 * pathSin u1) -
      (pathCos u2 * pathCos u2 - pathSin u2 * pathSin u2) =
      (pathCos u1 - pathCos u2) * (pathCos u1 + pathCos u2) -
      (pathSin u1 - pathSin u2) * (pathSin u1 + pathSin u2) := by grind
  rw [hDiffEq]
  have hTri := ratAbs_add_le
    ((pathCos u1 - pathCos u2) * (pathCos u1 + pathCos u2))
    (-((pathSin u1 - pathSin u2) * (pathSin u1 + pathSin u2)))
  have hSubEq : (pathCos u1 - pathCos u2) * (pathCos u1 + pathCos u2) -
      (pathSin u1 - pathSin u2) * (pathSin u1 + pathSin u2) =
      (pathCos u1 - pathCos u2) * (pathCos u1 + pathCos u2) +
      (-((pathSin u1 - pathSin u2) * (pathSin u1 + pathSin u2))) := by grind
  rw [hSubEq]
  rw [ratAbs_neg] at hTri
  rw [ratAbs_mul, ratAbs_mul] at hTri
  have hcAdd : ratAbs (pathCos u1 + pathCos u2) ≤ 2 := by
    have ht := ratAbs_add_le (pathCos u1) (pathCos u2)
    grind
  have hsAdd : ratAbs (pathSin u1 + pathSin u2) ≤ 2 := by
    have ht := ratAbs_add_le (pathSin u1) (pathSin u2)
    grind
  have hcLip := pathCos_lipschitz01.2 u1 u2 hu1a hu1b hu2a hu2b
  have hsLip := pathSin_lipschitz01.2 u1 u2 hu1a hu1b hu2a hu2b
  have hCProd : ratAbs (pathCos u1 - pathCos u2) * ratAbs (pathCos u1 + pathCos u2) ≤
      16 * ratAbs (u1 - u2) := by
    have h1 : ratAbs (pathCos u1 - pathCos u2) * ratAbs (pathCos u1 + pathCos u2) ≤
        ratAbs (pathCos u1 - pathCos u2) * 2 :=
      rat_mul_le_mul_of_nonneg_left hcAdd (ratAbs_nonneg _)
    have h2 : ratAbs (pathCos u1 - pathCos u2) * 2 ≤ (8 * ratAbs (u1 - u2)) * 2 :=
      rat_mul_le_mul_of_nonneg_right hcLip (by grind)
    grind
  have hSProd : ratAbs (pathSin u1 - pathSin u2) * ratAbs (pathSin u1 + pathSin u2) ≤
      16 * ratAbs (u1 - u2) := by
    have h1 : ratAbs (pathSin u1 - pathSin u2) * ratAbs (pathSin u1 + pathSin u2) ≤
        ratAbs (pathSin u1 - pathSin u2) * 2 :=
      rat_mul_le_mul_of_nonneg_left hsAdd (ratAbs_nonneg _)
    have h2 : ratAbs (pathSin u1 - pathSin u2) * 2 ≤ (8 * ratAbs (u1 - u2)) * 2 :=
      rat_mul_le_mul_of_nonneg_right hsLip (by grind)
    grind
  exact Rat.le_trans hTri (by grind)

/-- 精确二次扩域 $\mathbb{Q}(\sqrt{2})$：元素表示 `a + b√2`。 -/
structure QSqrt2 where
  a : Rat
  b : Rat
  deriving DecidableEq, Repr

namespace QSqrt2

@[ext] theorem ext {x y : QSqrt2} (ha : x.a = y.a) (hb : x.b = y.b) : x = y := by
  cases x; cases y; simp_all

def zero : QSqrt2 := ⟨0, 0⟩
def one : QSqrt2 := ⟨1, 0⟩
def sqrt2Half : QSqrt2 := ⟨0, 1 / 2⟩
def add (x y : QSqrt2) : QSqrt2 := ⟨x.a + y.a, x.b + y.b⟩
def sub (x y : QSqrt2) : QSqrt2 := ⟨x.a - y.a, x.b - y.b⟩
def neg (x : QSqrt2) : QSqrt2 := ⟨-x.a, -x.b⟩
def mul (x y : QSqrt2) : QSqrt2 := ⟨x.a * y.a + 2 * x.b * y.b, x.a * y.b + x.b * y.a⟩

instance : Zero QSqrt2 := ⟨zero⟩
instance : One QSqrt2 := ⟨one⟩
instance : Add QSqrt2 := ⟨add⟩
instance : Sub QSqrt2 := ⟨sub⟩
instance : Neg QSqrt2 := ⟨neg⟩
instance : Mul QSqrt2 := ⟨mul⟩

@[simp] theorem zero_a : (0 : QSqrt2).a = 0 := rfl
@[simp] theorem zero_b : (0 : QSqrt2).b = 0 := rfl
@[simp] theorem one_a : (1 : QSqrt2).a = 1 := rfl
@[simp] theorem one_b : (1 : QSqrt2).b = 0 := rfl
@[simp] theorem add_a (x y : QSqrt2) : (x + y).a = x.a + y.a := rfl
@[simp] theorem add_b (x y : QSqrt2) : (x + y).b = x.b + y.b := rfl
@[simp] theorem sub_a (x y : QSqrt2) : (x - y).a = x.a - y.a := rfl
@[simp] theorem sub_b (x y : QSqrt2) : (x - y).b = x.b - y.b := rfl
@[simp] theorem neg_a (x : QSqrt2) : (-x).a = -x.a := rfl
@[simp] theorem neg_b (x : QSqrt2) : (-x).b = -x.b := rfl
@[simp] theorem mul_a (x y : QSqrt2) : (x * y).a = x.a * y.a + 2 * x.b * y.b := rfl
@[simp] theorem mul_b (x y : QSqrt2) : (x * y).b = x.a * y.b + x.b * y.a := rfl

end QSqrt2

/-- $\mathbb{Q}(\sqrt{2})$ 上的双比特联合态，用以精确表示 $\theta = \pi / 4$ Bell 态。 -/
structure TwoQubitStateQ2 where
  pp : QSqrt2
  pm : QSqrt2
  mp : QSqrt2
  mm : QSqrt2
  deriving DecidableEq, Repr

namespace TwoQubitStateQ2

def normSq (ψ : TwoQubitStateQ2) : QSqrt2 :=
  ψ.pp * ψ.pp + ψ.pm * ψ.pm + ψ.mp * ψ.mp + ψ.mm * ψ.mm

def entEnergy (ψ : TwoQubitStateQ2) : QSqrt2 :=
  ψ.pm * ψ.pm + ψ.mp * ψ.mp

def expSa (ψ : TwoQubitStateQ2) : QSqrt2 :=
  (ψ.pp * ψ.pp + ψ.pm * ψ.pm) - (ψ.mp * ψ.mp + ψ.mm * ψ.mm)

def expSb (ψ : TwoQubitStateQ2) : QSqrt2 :=
  (ψ.pp * ψ.pp + ψ.mp * ψ.mp) - (ψ.pm * ψ.pm + ψ.mm * ψ.mm)

def expSaSb (ψ : TwoQubitStateQ2) : QSqrt2 :=
  (ψ.pp * ψ.pp + ψ.mm * ψ.mm) - (ψ.pm * ψ.pm + ψ.mp * ψ.mp)

end TwoQubitStateQ2

/-- 精确 $\theta = \pi / 4$ 最大纠缠 Bell 态 $|\Phi^+\rangle = \frac{\sqrt{2}}{2}|+1,+1\rangle + \frac{\sqrt{2}}{2}|-1,-1\rangle$。 -/
def bellPlus : TwoQubitStateQ2 :=
  ⟨QSqrt2.sqrt2Half, 0, 0, QSqrt2.sqrt2Half⟩

/--
**定理 2（`rank2_entangled_zero_barrier`）**：
1. 有理连续路径 `entPath t` 在 `t = 0` 与 `t = 1` 精确连接纯态 `|+1,+1⟩` 与 `|-1,-1⟩`；
2. 对**全部** `t : Rat`，全程严格保范（`normSq = 1`）、约束违背能量**恒零**（`entEnergy = 0`）
   且双比特关联 `expSaSb = 1`；
3. 在 `[0, 1]` 上满足全局 `4`-Lipschitz 连续；
4. 在纯 $\mathbb{Q}$ 域上，对任意 `eps > 0` 均可构造 `t ∈ [0, 1]` 使两侧单比特边缘期望均满足 `|⟨s_i⟩| < eps`；
5. 在 $\mathbb{Q}(\sqrt{2})$ 中精确实现 $\theta = \pi / 4$ Bell 态 `bellPlus`，精确满足
   `normSq = 1 ∧ entEnergy = 0 ∧ expSa = 0 ∧ expSb = 0 ∧ expSaSb = 1`。
-/
theorem rank2_entangled_zero_barrier :
    (entPath 0 = ⟨1, 0, 0, 0⟩ ∧ entPath 1 = ⟨0, 0, 0, 1⟩) ∧
    (∀ t : Rat, (entPath t).normSq = 1 ∧ (entPath t).entEnergy = 0 ∧ (entPath t).expSaSb = 1) ∧
    (∀ t t0 : Rat, 0 ≤ t → t ≤ 1 → 0 ≤ t0 → t0 ≤ 1 →
      TwoQubitState.stateDist (entPath t) (entPath t0) ≤ 4 * ratAbs (t - t0)) ∧
    (∀ eps : Rat, 0 < eps → ∃ t : Rat, 0 ≤ t ∧ t ≤ 1 ∧
      ratAbs ((entPath t).expSa) < eps ∧ ratAbs ((entPath t).expSb) < eps ∧
      (entPath t).entEnergy = 0 ∧ (entPath t).normSq = 1) ∧
    (bellPlus.normSq = 1 ∧ bellPlus.entEnergy = 0 ∧
     bellPlus.expSa = 0 ∧ bellPlus.expSb = 0 ∧ bellPlus.expSaSb = 1) := by
  refine ⟨⟨?_, ?_⟩, ?_, ?_, ?_, ?_⟩
  · have hc0 : pathCos 0 = 1 := by unfold pathCos pathDen; grind
    have hs0 : pathSin 0 = 0 := by unfold pathSin pathDen; grind
    have hZeroHalf : (0 : Rat) / 2 = 0 := by grind
    apply TwoQubitState.ext <;> simp [entPath, hZeroHalf, hc0, hs0] <;> grind
  · have hcHalf : pathCos (1 / 2) = 0 := by unfold pathCos pathDen; grind
    have hsHalf : pathSin (1 / 2) = 1 := by unfold pathSin pathDen; grind
    apply TwoQubitState.ext <;> simp [entPath, hcHalf, hsHalf] <;> grind
  · intro t
    have hSq := pathCos_sq_add_pathSin_sq (t / 2)
    refine ⟨?_, ?_, ?_⟩
    · unfold TwoQubitState.normSq entPath; simp; grind
    · unfold TwoQubitState.entEnergy entPath; simp; grind
    · unfold TwoQubitState.expSaSb entPath; simp; grind
  · intro t t0 ht0 ht1 ht00 ht01
    have hc := entPath_diff_abs_le pathCos pathCos_lipschitz01 t t0 ht0 ht1 ht00 ht01
    have hs := entPath_diff_abs_le pathSin pathSin_lipschitz01 t t0 ht0 ht1 ht00 ht01
    have h0 : ratAbs ((0 : Rat) - 0) ≤ 4 * ratAbs (t - t0) := by
      have heq : (0 : Rat) - 0 = 0 := by grind
      rw [heq, ratAbs_zero]
      exact Rat.mul_nonneg (by grind) (ratAbs_nonneg _)
    unfold TwoQubitState.stateDist entPath
    dsimp only
    grind
  · intro eps hEps
    let eps16 : Rat := eps / 16
    have hEps16Pos : 0 < eps16 := rat_div_pos hEps (by grind)
    rcases archimedean_step_lt eps16 hEps16Pos with ⟨N, hNPos, hStepLt⟩
    have hNRatPos : 0 < natToRat N := natToRat_pos hNPos
    let margin : Rat → Rat := fun t => (entPath t).expSa
    have hMarginEq : ∀ t : Rat, margin t =
        pathCos (t / 2) * pathCos (t / 2) - pathSin (t / 2) * pathSin (t / 2) := by
      intro t; unfold margin TwoQubitState.expSa entPath; simp; grind
    let g : Nat → Rat := fun j => margin (natToRat j / natToRat N)
    have hG0 : 0 < g 0 := by
      have hz : natToRat 0 / natToRat N = 0 := by unfold natToRat; grind
      have hc0 : pathCos 0 = 1 := by unfold pathCos pathDen; grind
      have hs0 : pathSin 0 = 0 := by unfold pathSin pathDen; grind
      dsimp [g]
      rw [hz, hMarginEq 0]
      have h0 : (0 : Rat) / 2 = 0 := by grind
      rw [h0, hc0, hs0]
      grind
    have hGN : g N ≤ 0 := by
      have hOne : natToRat N / natToRat N = 1 := by
        rw [Rat.div_def, Rat.mul_inv_cancel _ (by grind)]
      have hcH : pathCos (1 / 2) = 0 := by unfold pathCos pathDen; grind
      have hsH : pathSin (1 / 2) = 1 := by unfold pathSin pathDen; grind
      dsimp [g]
      rw [hOne, hMarginEq 1]
      have hH : (1 : Rat) / 2 = 1 / 2 := rfl
      rw [hH, hcH, hsH]
      grind
    rcases exists_grid_sign_change N g hG0 hGN with ⟨k, hkLt, hkPos, hk1Nonpos⟩
    let tk : Rat := natToRat k / natToRat N
    let tk1 : Rat := natToRat (k + 1) / natToRat N
    have htk0 : 0 ≤ tk := rat_div_nonneg (natToRat_nonneg k) hNRatPos
    have htk1 : tk ≤ 1 := by
      dsimp [tk]; rw [rat_div_le_iff hNRatPos]; have h := natToRat_le (by omega : k ≤ N); grind
    have htk10 : 0 ≤ tk1 := rat_div_nonneg (natToRat_nonneg (k + 1)) hNRatPos
    have htk11 : tk1 ≤ 1 := by
      dsimp [tk1]; rw [rat_div_le_iff hNRatPos]; have h := natToRat_le (by omega : k + 1 ≤ N); grind
    have hDiffStep : tk1 / 2 - tk / 2 = 1 / (2 * natToRat N) := by
      dsimp [tk, tk1]
      have hSucc : natToRat (k + 1) = natToRat k + 1 := rfl
      rw [hSucc, Rat.div_def, Rat.div_def, Rat.div_def, Rat.div_def, Rat.div_def]
      grind
    have hMarginDiff : ratAbs (g k - g (k + 1)) ≤ 16 / natToRat N := by
      dsimp [g]
      rw [hMarginEq tk, hMarginEq tk1]
      have hLip := entMarginDiff_lipschitz (tk / 2) (tk1 / 2)
        (by grind) (by grind) (by grind) (by grind)
      have hAbsSub : ratAbs (tk / 2 - tk1 / 2) = 1 / (2 * natToRat N) := by
        rw [ratAbs_sub_comm, hDiffStep]
        exact ratAbs_of_nonneg (rat_div_nonneg (by grind) (by grind))
      rw [hAbsSub] at hLip
      have hEq16 : 32 * (1 / (2 * natToRat N)) = 16 / natToRat N := by
        rw [Rat.div_def, Rat.div_def]
        have hInv2 : (2 * natToRat N)⁻¹ = 2⁻¹ * (natToRat N)⁻¹ := by
          have hProd : (2 * natToRat N) * (2⁻¹ * (natToRat N)⁻¹) = 1 := by
            have h1 : (2 * natToRat N) * (2⁻¹ * (natToRat N)⁻¹) =
                (2 * 2⁻¹) * (natToRat N * (natToRat N)⁻¹) := by grind
            rw [h1, Rat.mul_inv_cancel _ (by grind), Rat.mul_inv_cancel _ (by grind)]
            grind
          have hL := congrArg (fun z => (2 * natToRat N)⁻¹ * z) hProd
          have hAssoc : (2 * natToRat N)⁻¹ * ((2 * natToRat N) * (2⁻¹ * (natToRat N)⁻¹)) =
              ((2 * natToRat N) * (2 * natToRat N)⁻¹) * (2⁻¹ * (natToRat N)⁻¹) := by grind
          rw [hAssoc, Rat.mul_inv_cancel _ (by grind)] at hL
          grind
        rw [hInv2]
        grind
      rw [hEq16] at hLip
      exact hLip
    have hBoundEps : 16 / natToRat N < eps := by
      have h1 : 16 * (1 / natToRat N) < 16 * eps16 :=
        Rat.mul_lt_mul_of_pos_left hStepLt (by grind)
      have hL : 16 * (1 / natToRat N) = 16 / natToRat N := by rw [Rat.div_def, Rat.div_def]; grind
      have hR : 16 * eps16 = eps := by
        dsimp [eps16]
        rw [Rat.div_def]
        have h16 : (16 : Rat) * (eps * (16 : Rat)⁻¹) = eps * ((16 : Rat) * (16 : Rat)⁻¹) := by grind
        rw [h16, Rat.mul_inv_cancel _ (by grind)]
        grind
      rw [hL, hR] at h1
      exact h1
    have hGkLtEps : ratAbs (g k) < eps := by
      have hPosAbs : ratAbs (g k) = g k := ratAbs_of_nonneg (Rat.le_of_lt hkPos)
      have hDiffLe : g k ≤ ratAbs (g k - g (k + 1)) := by
        have h0 : g k ≤ g k - g (k + 1) := by grind
        have hAbsGe : g k - g (k + 1) ≤ ratAbs (g k - g (k + 1)) := by
          unfold ratAbs; split <;> grind
        exact Rat.le_trans h0 hAbsGe
      rw [hPosAbs]
      exact rat_lt_of_le_of_lt (Rat.le_trans hDiffLe hMarginDiff) hBoundEps
    have hExpEq : (entPath tk).expSb = (entPath tk).expSa := by
      unfold TwoQubitState.expSb TwoQubitState.expSa entPath
      simp
    have hSq := pathCos_sq_add_pathSin_sq (tk / 2)
    refine ⟨tk, htk0, htk1, hGkLtEps, ?_, ?_, ?_⟩
    · rw [hExpEq]; exact hGkLtEps
    · unfold TwoQubitState.entEnergy entPath; simp; grind
    · unfold TwoQubitState.normSq entPath; simp; grind
  · refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
      apply QSqrt2.ext <;>
      simp [bellPlus, QSqrt2.sqrt2Half, TwoQubitStateQ2.normSq, TwoQubitStateQ2.entEnergy,
        TwoQubitStateQ2.expSa, TwoQubitStateQ2.expSb, TwoQubitStateQ2.expSaSb] <;>
      grind

/-!
### 3. 推广至任意长度 `L ≥ 2` 闭环链式约束：Rank-2 MPS 零残差 vs Rank-1 独立乘积诚实对照
-/

/-- 精确有理数 $2 \times 2$ 矩阵（MPS 键维 $\chi = 2$）。 -/
structure Mat2R where
  a11 : Rat
  a12 : Rat
  a21 : Rat
  a22 : Rat

namespace Mat2R

def mul (M N : Mat2R) : Mat2R :=
  ⟨M.a11 * N.a11 + M.a12 * N.a21, M.a11 * N.a12 + M.a12 * N.a22,
   M.a21 * N.a11 + M.a22 * N.a21, M.a21 * N.a12 + M.a22 * N.a22⟩

def tr (M : Mat2R) : Rat := M.a11 + M.a22

end Mat2R

/-- $\mathbb{Q}(\sqrt{2})$ 上的 $2 \times 2$ 矩阵。 -/
structure Mat2Q2 where
  a11 : QSqrt2
  a12 : QSqrt2
  a21 : QSqrt2
  a22 : QSqrt2

namespace Mat2Q2

def mul (M N : Mat2Q2) : Mat2Q2 :=
  ⟨M.a11 * N.a11 + M.a12 * N.a21, M.a11 * N.a12 + M.a12 * N.a22,
   M.a21 * N.a11 + M.a22 * N.a21, M.a21 * N.a12 + M.a22 * N.a22⟩

def tr (M : Mat2Q2) : QSqrt2 := M.a11 + M.a22

end Mat2Q2

/-- 比特向量前部扩展（将首比特 `b` 与尾部 `t : Fin n → Bool` 拼为 `Fin (n + 1) → Bool`）。 -/
def cfgCons {n : Nat} (b : Bool) (t : Fin n → Bool) : Fin (n + 1) → Bool
  | ⟨0, _⟩ => b
  | ⟨k + 1, h⟩ => t ⟨k, Nat.lt_of_succ_lt_succ h⟩

/-- 取长度 `n + 1` 布尔向量除第 0 位之外的尾部。 -/
def cfgTail {n : Nat} (σ : Fin (n + 1) → Bool) : Fin n → Bool :=
  fun i => σ i.succ

private theorem cfgCons_head_tail {n : Nat} (σ : Fin (n + 1) → Bool) :
    cfgCons (σ 0) (cfgTail σ) = σ := by
  funext i
  rcases i with ⟨v, hv⟩
  cases v with
  | zero => rfl
  | succ k => rfl

/-- 可判定全常量布尔检查。 -/
def allBool : {n : Nat} → (Fin n → Bool) → Bool → Bool
  | 0, _, _ => true
  | k + 1, σ, b => allBool (fun i : Fin k => σ i.castSucc) b && (σ (Fin.last k) == b)

private theorem allBool_true_iff {n : Nat} (σ : Fin n → Bool) (b : Bool) :
    allBool σ b = true ↔ ∀ i : Fin n, σ i = b := by
  induction n with
  | zero =>
    simp [allBool]
  | succ k ih =>
    simp only [allBool, Bool.and_eq_true, beq_iff_eq]
    rw [ih (fun i : Fin k => σ i.castSucc)]
    constructor
    · intro ⟨hPref, hLast⟩ i
      by_cases hi : i.val = k
      · have hEq : i = Fin.last k := Fin.ext hi
        rw [hEq]
        exact hLast
      · have hLt : i.val < k := Nat.lt_of_le_of_ne (Nat.lt_succ_iff.mp i.isLt) hi
        let i' : Fin k := ⟨i.val, hLt⟩
        have hEq : i'.castSucc = i := Fin.ext rfl
        rw [← hEq]
        exact hPref i'
    · intro hAll
      exact ⟨fun i => hAll i.castSucc, hAll (Fin.last k)⟩

private theorem allBool_not_both {n : Nat} (hn : 1 ≤ n) (σ : Fin n → Bool) :
    ¬ (allBool σ false = true ∧ allBool σ true = true) := by
  intro ⟨hf, ht⟩
  have h0f := (allBool_true_iff σ false).mp hf ⟨0, hn⟩
  have h0t := (allBool_true_iff σ true).mp ht ⟨0, hn⟩
  rw [h0f] at h0t
  contradiction

/-- 长度 `L` 布尔构型全空间上的 `Rat` 求和。 -/
def sumConfigsR : {L : Nat} → ((Fin L → Bool) → Rat) → Rat
  | 0, f => f (fun i => i.elim0)
  | _ + 1, f =>
    sumConfigsR (fun t => f (cfgCons false t)) + sumConfigsR (fun t => f (cfgCons true t))

/-- 长度 `L` 布尔构型全空间上的 `QSqrt2` 求和。 -/
def sumConfigsQ2 : {L : Nat} → ((Fin L → Bool) → QSqrt2) → QSqrt2
  | 0, f => f (fun i => i.elim0)
  | _ + 1, f =>
    sumConfigsQ2 (fun t => f (cfgCons false t)) + sumConfigsQ2 (fun t => f (cfgCons true t))

private theorem sumConfigsR_add {L : Nat} (f g : (Fin L → Bool) → Rat) :
    sumConfigsR (fun σ => f σ + g σ) = sumConfigsR f + sumConfigsR g := by
  induction L with
  | zero => simp [sumConfigsR]
  | succ k ih =>
    simp only [sumConfigsR]
    rw [ih (fun t => f (cfgCons false t)) (fun t => g (cfgCons false t)),
        ih (fun t => f (cfgCons true t)) (fun t => g (cfgCons true t))]
    grind

private theorem sumConfigsR_zero {L : Nat} (f : (Fin L → Bool) → Rat) (h : ∀ σ, f σ = 0) :
    sumConfigsR f = 0 := by
  induction L with
  | zero => simp [sumConfigsR, h]
  | succ k ih =>
    simp only [sumConfigsR]
    rw [ih _ (fun t => h _), ih _ (fun t => h _)]
    grind

private theorem sumConfigsR_single {L : Nat} (σ0 : Fin L → Bool) (f : (Fin L → Bool) → Rat)
    (h : ∀ σ, σ ≠ σ0 → f σ = 0) :
    sumConfigsR f = f σ0 := by
  induction L with
  | zero =>
    simp only [sumConfigsR]
    have hEq : (fun i : Fin 0 => i.elim0) = σ0 := funext (fun i => i.elim0)
    rw [hEq]
  | succ k ih =>
    simp only [sumConfigsR]
    have hDecomp := cfgCons_head_tail σ0
    cases hHead : σ0 0 with
    | false =>
      have hTrueZero : sumConfigsR (fun t => f (cfgCons true t)) = 0 := by
        apply sumConfigsR_zero
        intro t
        apply h
        intro hEq
        have h0 := congrArg (fun g => g 0) hEq
        simp [cfgCons] at h0
        rw [hHead] at h0
        contradiction
      have ih' := ih (cfgTail σ0) (fun t => f (cfgCons false t))
      have hRest : ∀ t, t ≠ cfgTail σ0 → f (cfgCons false t) = 0 := by
        intro t ht
        apply h
        intro hEq
        apply ht
        funext i
        have hi := congrArg (fun g => g i.succ) hEq
        exact hi
      rw [hTrueZero, ih' hRest, ← hHead, hDecomp]
      grind
    | true =>
      have hFalseZero : sumConfigsR (fun t => f (cfgCons false t)) = 0 := by
        apply sumConfigsR_zero
        intro t
        apply h
        intro hEq
        have h0 := congrArg (fun g => g 0) hEq
        simp [cfgCons] at h0
        rw [hHead] at h0
        contradiction
      have ih' := ih (cfgTail σ0) (fun t => f (cfgCons true t))
      have hRest : ∀ t, t ≠ cfgTail σ0 → f (cfgCons true t) = 0 := by
        intro t ht
        apply h
        intro hEq
        apply ht
        funext i
        have hi := congrArg (fun g => g i.succ) hEq
        exact hi
      rw [hFalseZero, ih' hRest, ← hHead, hDecomp]
      grind

private theorem sumConfigsQ2_add {L : Nat} (f g : (Fin L → Bool) → QSqrt2) :
    sumConfigsQ2 (fun σ => f σ + g σ) = sumConfigsQ2 f + sumConfigsQ2 g := by
  induction L with
  | zero => simp [sumConfigsQ2]
  | succ k ih =>
    simp only [sumConfigsQ2]
    rw [ih (fun t => f (cfgCons false t)) (fun t => g (cfgCons false t)),
        ih (fun t => f (cfgCons true t)) (fun t => g (cfgCons true t))]
    apply QSqrt2.ext <;> simp <;> grind

private theorem sumConfigsQ2_zero {L : Nat} (f : (Fin L → Bool) → QSqrt2) (h : ∀ σ, f σ = 0) :
    sumConfigsQ2 f = 0 := by
  induction L with
  | zero => simp [sumConfigsQ2, h]
  | succ k ih =>
    simp only [sumConfigsQ2]
    rw [ih _ (fun t => h _), ih _ (fun t => h _)]
    apply QSqrt2.ext <;> simp <;> grind

private theorem sumConfigsQ2_single {L : Nat} (σ0 : Fin L → Bool) (f : (Fin L → Bool) → QSqrt2)
    (h : ∀ σ, σ ≠ σ0 → f σ = 0) :
    sumConfigsQ2 f = f σ0 := by
  induction L with
  | zero =>
    simp only [sumConfigsQ2]
    have hEq : (fun i : Fin 0 => i.elim0) = σ0 := funext (fun i => i.elim0)
    rw [hEq]
  | succ k ih =>
    simp only [sumConfigsQ2]
    have hDecomp := cfgCons_head_tail σ0
    cases hHead : σ0 0 with
    | false =>
      have hTrueZero : sumConfigsQ2 (fun t => f (cfgCons true t)) = 0 := by
        apply sumConfigsQ2_zero
        intro t
        apply h
        intro hEq
        have h0 := congrArg (fun g => g 0) hEq
        simp [cfgCons] at h0
        rw [hHead] at h0
        contradiction
      have ih' := ih (cfgTail σ0) (fun t => f (cfgCons false t))
      have hRest : ∀ t, t ≠ cfgTail σ0 → f (cfgCons false t) = 0 := by
        intro t ht
        apply h
        intro hEq
        apply ht
        funext i
        have hi := congrArg (fun g => g i.succ) hEq
        exact hi
      rw [hTrueZero, ih' hRest, ← hHead, hDecomp]
      apply QSqrt2.ext <;> simp <;> grind
    | true =>
      have hFalseZero : sumConfigsQ2 (fun t => f (cfgCons false t)) = 0 := by
        apply sumConfigsQ2_zero
        intro t
        apply h
        intro hEq
        have h0 := congrArg (fun g => g 0) hEq
        simp [cfgCons] at h0
        rw [hHead] at h0
        contradiction
      have ih' := ih (cfgTail σ0) (fun t => f (cfgCons true t))
      have hRest : ∀ t, t ≠ cfgTail σ0 → f (cfgCons true t) = 0 := by
        intro t ht
        apply h
        intro hEq
        apply ht
        funext i
        have hi := congrArg (fun g => g i.succ) hEq
        exact hi
      rw [hFalseZero, ih' hRest, ← hHead, hDecomp]
      apply QSqrt2.ext <;> simp <;> grind

/-- 物理格点位矢向 Rank-2 键维张量核心的映射（有理域）。 -/
def mpsBitMatR (b : Bool) : Mat2R :=
  if b then ⟨0, 0, 0, 1⟩ else ⟨1, 0, 0, 0⟩

/-- Rank-2 循环 MPS 张量链左连乘（有理域）。 -/
def mpsProdR (c s : Rat) : {L : Nat} → (Fin L → Bool) → Mat2R
  | 0, _ => ⟨c, 0, 0, s⟩
  | k + 1, σ => Mat2R.mul (mpsProdR c s (fun i : Fin k => σ i.castSucc)) (mpsBitMatR (σ (Fin.last k)))

/-- Rank-2 循环 MPS 联合振幅 `Ψ(σ) = Tr(D(c, s) A_{σ_0} ⋯ A_{σ_{L-1}})`（有理域）。 -/
def mpsCycleAmpR (c s : Rat) {L : Nat} (σ : Fin L → Bool) : Rat :=
  (mpsProdR c s σ).tr

private theorem mpsProdR_diag_entries (c s : Rat) :
    ∀ {L : Nat} (σ : Fin L → Bool),
      (mpsProdR c s σ).a11 = (if allBool σ false then c else 0) ∧
      (mpsProdR c s σ).a12 = 0 ∧
      (mpsProdR c s σ).a21 = 0 ∧
      (mpsProdR c s σ).a22 = (if allBool σ true then s else 0) := by
  intro L
  induction L with
  | zero =>
    intro σ
    simp [mpsProdR, allBool]
  | succ k ih =>
    intro σ
    have hIH := ih (fun i : Fin k => σ i.castSucc)
    rcases hIH with ⟨ha11, ha12, ha21, ha22⟩
    have hStep : mpsProdR c s σ =
        Mat2R.mul (mpsProdR c s (fun i : Fin k => σ i.castSucc)) (mpsBitMatR (σ (Fin.last k))) := rfl
    rw [hStep]
    have hAllF : allBool σ false = (allBool (fun i : Fin k => σ i.castSucc) false && (σ (Fin.last k) == false)) := rfl
    have hAllT : allBool σ true = (allBool (fun i : Fin k => σ i.castSucc) true && (σ (Fin.last k) == true)) := rfl
    cases hb : σ (Fin.last k) <;>
      cases hpf : allBool (fun i : Fin k => σ i.castSucc) false <;>
      cases hpt : allBool (fun i : Fin k => σ i.castSucc) true <;>
      simp [Mat2R.mul, mpsBitMatR, hb, ha11, ha12, ha21, ha22, hAllF, hAllT, hpf, hpt] <;>
      try grind

private theorem mpsCycleAmpR_eval (c s : Rat) {L : Nat} (hL : 1 ≤ L) (σ : Fin L → Bool) :
    mpsCycleAmpR c s σ =
      if allBool σ false then c else if allBool σ true then s else 0 := by
  have hD := mpsProdR_diag_entries c s σ
  unfold mpsCycleAmpR Mat2R.tr
  rw [hD.1, hD.2.2.2]
  have hNotBoth := allBool_not_both hL σ
  cases hf : allBool σ false <;> cases ht : allBool σ true
  · simp; try grind
  · simp; try grind
  · simp; try grind
  · exfalso; exact hNotBoth ⟨hf, ht⟩

private theorem sumConfigsR_two_poles (c s : Rat) {L : Nat} (hL : 1 ≤ L)
    (w : (Fin L → Bool) → Rat) :
    sumConfigsR (fun σ => w σ * (mpsCycleAmpR c s σ * mpsCycleAmpR c s σ)) =
      w (fun _ => false) * (c * c) + w (fun _ => true) * (s * s) := by
  let fF : (Fin L → Bool) → Rat := fun σ =>
    if allBool σ false then w σ * (mpsCycleAmpR c s σ * mpsCycleAmpR c s σ) else 0
  let fT : (Fin L → Bool) → Rat := fun σ =>
    if allBool σ true then w σ * (mpsCycleAmpR c s σ * mpsCycleAmpR c s σ) else 0
  have hSplit : (fun σ : Fin L → Bool => w σ * (mpsCycleAmpR c s σ * mpsCycleAmpR c s σ)) =
      (fun σ => fF σ + fT σ) := by
    funext σ
    dsimp [fF, fT]
    have hAmp := mpsCycleAmpR_eval c s hL σ
    have hNotBoth := allBool_not_both hL σ
    cases hf : allBool σ false <;> cases ht : allBool σ true
    · rw [hAmp, hf, ht]; simp; try grind
    · rw [hAmp, hf, ht]; simp; try grind
    · rw [hAmp, hf, ht]; simp; try grind
    · exfalso; exact hNotBoth ⟨hf, ht⟩
  rw [hSplit, sumConfigsR_add]
  have hF : sumConfigsR fF = w (fun _ => false) * (c * c) := by
    rw [sumConfigsR_single (fun _ => false) fF]
    · have hAllF : allBool (fun _ : Fin L => false) false = true :=
        (allBool_true_iff _ false).mpr (fun _ => rfl)
      have hAmp := mpsCycleAmpR_eval c s hL (fun _ => false)
      dsimp [fF]
      rw [hAmp, hAllF]
      simp
    · intro σ hNe
      have hNotAll : allBool σ false = false := by
        cases h : allBool σ false with
        | false => rfl
        | true =>
          exfalso
          apply hNe
          funext i
          exact (allBool_true_iff σ false).mp h i
      dsimp [fF]
      simp [hNotAll]
  have hT : sumConfigsR fT = w (fun _ => true) * (s * s) := by
    rw [sumConfigsR_single (fun _ => true) fT]
    · have hAllT : allBool (fun _ : Fin L => true) true = true :=
        (allBool_true_iff _ true).mpr (fun _ => rfl)
      have hAllF : allBool (fun _ : Fin L => true) false = false := by
        cases h : allBool (fun _ : Fin L => true) false with
        | false => rfl
        | true => exfalso; exact allBool_not_both hL (fun _ => true) ⟨h, hAllT⟩
      have hAmp := mpsCycleAmpR_eval c s hL (fun _ => true)
      dsimp [fT]
      rw [hAmp, hAllF, hAllT]
      simp
    · intro σ hNe
      have hNotAll : allBool σ true = false := by
        cases h : allBool σ true with
        | false => rfl
        | true =>
          exfalso
          apply hNe
          funext i
          exact (allBool_true_iff σ true).mp h i
      dsimp [fT]
      simp [hNotAll]
  rw [hF, hT]

/-- `QSqrt2` 上的 Rank-2 MPS 张量核与振幅。 -/
def mpsBitMatQ2 (b : Bool) : Mat2Q2 :=
  if b then ⟨0, 0, 0, 1⟩ else ⟨1, 0, 0, 0⟩

def mpsProdQ2 (c s : QSqrt2) : {L : Nat} → (Fin L → Bool) → Mat2Q2
  | 0, _ => ⟨c, 0, 0, s⟩
  | k + 1, σ => Mat2Q2.mul (mpsProdQ2 c s (fun i : Fin k => σ i.castSucc)) (mpsBitMatQ2 (σ (Fin.last k)))

def mpsCycleAmpQ2 (c s : QSqrt2) {L : Nat} (σ : Fin L → Bool) : QSqrt2 :=
  (mpsProdQ2 c s σ).tr

private theorem mpsProdQ2_diag_entries (c s : QSqrt2) :
    ∀ {L : Nat} (σ : Fin L → Bool),
      (mpsProdQ2 c s σ).a11 = (if allBool σ false then c else 0) ∧
      (mpsProdQ2 c s σ).a12 = 0 ∧
      (mpsProdQ2 c s σ).a21 = 0 ∧
      (mpsProdQ2 c s σ).a22 = (if allBool σ true then s else 0) := by
  intro L
  induction L with
  | zero =>
    intro σ
    refine ⟨rfl, rfl, rfl, rfl⟩
  | succ k ih =>
    intro σ
    have hIH := ih (fun i : Fin k => σ i.castSucc)
    rcases hIH with ⟨ha11, ha12, ha21, ha22⟩
    have hStep : mpsProdQ2 c s σ =
        Mat2Q2.mul (mpsProdQ2 c s (fun i : Fin k => σ i.castSucc)) (mpsBitMatQ2 (σ (Fin.last k))) := rfl
    rw [hStep]
    have hAllF : allBool σ false = (allBool (fun i : Fin k => σ i.castSucc) false && (σ (Fin.last k) == false)) := rfl
    have hAllT : allBool σ true = (allBool (fun i : Fin k => σ i.castSucc) true && (σ (Fin.last k) == true)) := rfl
    cases hb : σ (Fin.last k) <;>
      cases hpf : allBool (fun i : Fin k => σ i.castSucc) false <;>
      cases hpt : allBool (fun i : Fin k => σ i.castSucc) true <;>
      simp [Mat2Q2.mul, mpsBitMatQ2, hb, ha11, ha12, ha21, ha22, hAllF, hAllT, hpf, hpt] <;>
      try (refine ⟨?_, ?_, ?_, ?_⟩ <;> apply QSqrt2.ext <;> simp <;> grind)

private theorem mpsCycleAmpQ2_eval (c s : QSqrt2) {L : Nat} (hL : 1 ≤ L) (σ : Fin L → Bool) :
    mpsCycleAmpQ2 c s σ =
      if allBool σ false then c else if allBool σ true then s else 0 := by
  have hD := mpsProdQ2_diag_entries c s σ
  unfold mpsCycleAmpQ2 Mat2Q2.tr
  rw [hD.1, hD.2.2.2]
  have hNotBoth := allBool_not_both hL σ
  cases hf : allBool σ false <;> cases ht : allBool σ true
  · simp; try (apply QSqrt2.ext <;> simp <;> try grind)
  · simp; try (apply QSqrt2.ext <;> simp <;> try grind)
  · simp; try (apply QSqrt2.ext <;> simp <;> try grind)
  · exfalso; exact hNotBoth ⟨hf, ht⟩

private theorem sumConfigsQ2_two_poles (c s : QSqrt2) {L : Nat} (hL : 1 ≤ L)
    (w : (Fin L → Bool) → QSqrt2) :
    sumConfigsQ2 (fun σ => w σ * (mpsCycleAmpQ2 c s σ * mpsCycleAmpQ2 c s σ)) =
      w (fun _ => false) * (c * c) + w (fun _ => true) * (s * s) := by
  let fF : (Fin L → Bool) → QSqrt2 := fun σ =>
    if allBool σ false then w σ * (mpsCycleAmpQ2 c s σ * mpsCycleAmpQ2 c s σ) else 0
  let fT : (Fin L → Bool) → QSqrt2 := fun σ =>
    if allBool σ true then w σ * (mpsCycleAmpQ2 c s σ * mpsCycleAmpQ2 c s σ) else 0
  have hSplit : (fun σ : Fin L → Bool => w σ * (mpsCycleAmpQ2 c s σ * mpsCycleAmpQ2 c s σ)) =
      (fun σ => fF σ + fT σ) := by
    funext σ
    dsimp [fF, fT]
    have hAmp := mpsCycleAmpQ2_eval c s hL σ
    have hNotBoth := allBool_not_both hL σ
    cases hf : allBool σ false <;> cases ht : allBool σ true
    · rw [hAmp, hf, ht]; simp; try (apply QSqrt2.ext <;> simp <;> try grind)
    · rw [hAmp, hf, ht]; simp; try (apply QSqrt2.ext <;> simp <;> try grind)
    · rw [hAmp, hf, ht]; simp; try (apply QSqrt2.ext <;> simp <;> try grind)
    · exfalso; exact hNotBoth ⟨hf, ht⟩
  rw [hSplit, sumConfigsQ2_add]
  have hF : sumConfigsQ2 fF = w (fun _ => false) * (c * c) := by
    rw [sumConfigsQ2_single (fun _ => false) fF]
    · have hAllF : allBool (fun _ : Fin L => false) false = true :=
        (allBool_true_iff _ false).mpr (fun _ => rfl)
      have hAmp := mpsCycleAmpQ2_eval c s hL (fun _ => false)
      dsimp [fF]
      rw [hAmp, hAllF]
      simp
    · intro σ hNe
      have hNotAll : allBool σ false = false := by
        cases h : allBool σ false with
        | false => rfl
        | true =>
          exfalso
          apply hNe
          funext i
          exact (allBool_true_iff σ false).mp h i
      dsimp [fF]
      simp [hNotAll]
  have hT : sumConfigsQ2 fT = w (fun _ => true) * (s * s) := by
    rw [sumConfigsQ2_single (fun _ => true) fT]
    · have hAllT : allBool (fun _ : Fin L => true) true = true :=
        (allBool_true_iff _ true).mpr (fun _ => rfl)
      have hAllF : allBool (fun _ : Fin L => true) false = false := by
        cases h : allBool (fun _ : Fin L => true) false with
        | false => rfl
        | true => exfalso; exact allBool_not_both hL (fun _ => true) ⟨h, hAllT⟩
      have hAmp := mpsCycleAmpQ2_eval c s hL (fun _ => true)
      dsimp [fT]
      rw [hAmp, hAllF, hAllT]
      simp
    · intro σ hNe
      have hNotAll : allBool σ true = false := by
        cases h : allBool σ true with
        | false => rfl
        | true =>
          exfalso
          apply hNe
          funext i
          exact (allBool_true_iff σ true).mp h i
      dsimp [fT]
      simp [hNotAll]
  rw [hF, hT]

/-- 对非负索引范围 `[0, n)` 递归求和 `Rat`。 -/
def natRangeSumR : Nat → (Nat → Rat) → Rat
  | 0, _ => 0
  | k + 1, f => natRangeSumR k f + f k

/-- 对非负索引范围 `[0, n)` 递归求和 `QSqrt2`。 -/
def natRangeSumQ2 : Nat → (Nat → QSqrt2) → QSqrt2
  | 0, _ => 0
  | k + 1, f => natRangeSumQ2 k f + f k

private theorem natRangeSumR_zero (n : Nat) (f : Nat → Rat)
    (h : ∀ i, i < n → f i = 0) : natRangeSumR n f = 0 := by
  induction n with
  | zero => rfl
  | succ k ih =>
    simp only [natRangeSumR]
    rw [ih (fun i hi => h i (by omega)), h k (by omega)]
    grind

private theorem natRangeSumR_nonneg (n : Nat) (f : Nat → Rat)
    (h : ∀ i, i < n → 0 ≤ f i) : 0 ≤ natRangeSumR n f := by
  induction n with
  | zero => simp [natRangeSumR]
  | succ k ih =>
    simp only [natRangeSumR]
    have h1 := ih (fun i hi => h i (by omega))
    have h2 := h k (by omega)
    grind

private theorem natRangeSumR_ge_term (n idx : Nat) (hIdx : idx < n) (f : Nat → Rat)
    (hNonneg : ∀ i, i < n → 0 ≤ f i) : f idx ≤ natRangeSumR n f := by
  induction n with
  | zero => omega
  | succ k ih =>
    simp only [natRangeSumR]
    by_cases heq : idx = k
    · have hPrev := natRangeSumR_nonneg k f (fun i hi => hNonneg i (by omega))
      rw [heq]
      grind
    · have hLt : idx < k := by omega
      have hPrev := ih hLt (fun i hi => hNonneg i (by omega))
      have hLast := hNonneg k (by omega)
      grind

private theorem natRangeSumQ2_zero (n : Nat) (f : Nat → QSqrt2)
    (h : ∀ i, i < n → f i = 0) : natRangeSumQ2 n f = 0 := by
  induction n with
  | zero => rfl
  | succ k ih =>
    simp only [natRangeSumQ2]
    rw [ih (fun i hi => h i (by omega)), h k (by omega)]
    apply QSqrt2.ext <;> simp <;> grind

/-- 闭环链构型 `σ : Fin L → Bool` 上的约束违约边计数（`σ_i ≠ σ_{(i+1) % L}`）。 -/
def cycleViolationsR {L : Nat} (hL : 2 ≤ L) (σ : Fin L → Bool) : Rat :=
  natRangeSumR L (fun i =>
    if h : i < L then
      if σ ⟨i, h⟩ == σ ⟨(i + 1) % L, Nat.mod_lt _ (by omega)⟩ then 0 else 1
    else 0)

def cycleViolationsQ2 {L : Nat} (hL : 2 ≤ L) (σ : Fin L → Bool) : QSqrt2 :=
  natRangeSumQ2 L (fun i =>
    if h : i < L then
      if σ ⟨i, h⟩ == σ ⟨(i + 1) % L, Nat.mod_lt _ (by omega)⟩ then 0 else 1
    else 0)

def mpsNormSqR {L : Nat} (Ψ : (Fin L → Bool) → Rat) : Rat :=
  sumConfigsR (fun σ => 1 * (Ψ σ * Ψ σ))

def cycleEnergyR {L : Nat} (hL : 2 ≤ L) (Ψ : (Fin L → Bool) → Rat) : Rat :=
  sumConfigsR (fun σ => cycleViolationsR hL σ * (Ψ σ * Ψ σ))

def mpsNormSqQ2 {L : Nat} (Ψ : (Fin L → Bool) → QSqrt2) : QSqrt2 :=
  sumConfigsQ2 (fun σ => 1 * (Ψ σ * Ψ σ))

def cycleEnergyQ2 {L : Nat} (hL : 2 ≤ L) (Ψ : (Fin L → Bool) → QSqrt2) : QSqrt2 :=
  sumConfigsQ2 (fun σ => cycleViolationsQ2 hL σ * (Ψ σ * Ψ σ))

def siteExpQ2 {L : Nat} (Ψ : (Fin L → Bool) → QSqrt2) (k : Fin L) : QSqrt2 :=
  sumConfigsQ2 (fun σ => (if σ k then (-1 : QSqrt2) else 1) * (Ψ σ * Ψ σ))

/--
**定理 3（`mps_rank2_cycle_zero_residual`）**：
对任意长度 `L ≥ 2` 的闭环耦合链：
1. 有理连续 Rank-2 MPS 路径 `mpsCycleAmpR (pathCos (t/2)) (pathSin (t/2))` 在 `t = 0` 与 `t = 1`
   分别精确承载全 `+1` 分支（振幅 `1`）与全 `-1` 分支（振幅 `1`），且沿整条路径全程总范数平方
   **恒为 `1`**、总循环违约能量**恒为 `0`**；
2. 在 $\mathbb{Q}(\sqrt{2})$ 上，键维 `χ = 2` 的 `L`-比特 GHZ MPS（`c = s = √2/2`）严格满足总范数平方 `= 1`、
   循环违约能量 `= 0`，且**每一个位点 `k : Fin L`** 的单比特边缘期望均精确为 `0`。
-/
theorem mps_rank2_cycle_zero_residual (L : Nat) (hL : 2 ≤ L) :
    (∀ σ : Fin L → Bool,
      mpsCycleAmpR (pathCos 0) (pathSin 0) σ = (if allBool σ false then 1 else 0) ∧
      mpsCycleAmpR (pathCos (1 / 2)) (pathSin (1 / 2)) σ = (if allBool σ true then 1 else 0)) ∧
    (∀ t : Rat,
      mpsNormSqR (mpsCycleAmpR (L := L) (pathCos (t / 2)) (pathSin (t / 2))) = 1 ∧
      cycleEnergyR hL (mpsCycleAmpR (L := L) (pathCos (t / 2)) (pathSin (t / 2))) = 0) ∧
    (let Ψghz := mpsCycleAmpQ2 (L := L) QSqrt2.sqrt2Half QSqrt2.sqrt2Half
     mpsNormSqQ2 Ψghz = 1 ∧
     cycleEnergyQ2 hL Ψghz = 0 ∧
     ∀ k : Fin L, siteExpQ2 Ψghz k = 0) := by
  have h1L : 1 ≤ L := by omega
  have hViolConstR : ∀ b : Bool, cycleViolationsR hL (fun _ => b) = 0 := by
    intro b
    unfold cycleViolationsR
    apply natRangeSumR_zero
    intro i hi
    simp [hi]
  have hViolConstQ2 : ∀ b : Bool, cycleViolationsQ2 hL (fun _ => b) = 0 := by
    intro b
    unfold cycleViolationsQ2
    apply natRangeSumQ2_zero
    intro i hi
    simp [hi]
  refine ⟨?_, ?_, ?_⟩
  · intro σ
    have hc0 : pathCos 0 = 1 := by unfold pathCos pathDen; grind
    have hs0 : pathSin 0 = 0 := by unfold pathSin pathDen; grind
    have hcH : pathCos (1 / 2) = 0 := by unfold pathCos pathDen; grind
    have hsH : pathSin (1 / 2) = 1 := by unfold pathSin pathDen; grind
    rw [mpsCycleAmpR_eval _ _ h1L σ, mpsCycleAmpR_eval _ _ h1L σ, hc0, hs0, hcH, hsH]
    cases hf : allBool σ false <;> cases ht : allBool σ true <;> simp
    · exfalso; exact allBool_not_both h1L σ ⟨hf, ht⟩
  · intro t
    have hSq := pathCos_sq_add_pathSin_sq (t / 2)
    constructor
    · unfold mpsNormSqR
      rw [sumConfigsR_two_poles _ _ h1L]
      grind
    · unfold cycleEnergyR
      rw [sumConfigsR_two_poles _ _ h1L, hViolConstR false, hViolConstR true]
      grind
  · intro Ψghz
    refine ⟨?_, ?_, ?_⟩
    · unfold mpsNormSqQ2 Ψghz
      rw [sumConfigsQ2_two_poles _ _ h1L]
      apply QSqrt2.ext <;> simp [QSqrt2.sqrt2Half] <;> try grind
    · unfold cycleEnergyQ2 Ψghz
      rw [sumConfigsQ2_two_poles _ _ h1L, hViolConstQ2 false, hViolConstQ2 true]
      apply QSqrt2.ext <;> simp <;> try grind
    · intro k
      unfold siteExpQ2 Ψghz
      rw [sumConfigsQ2_two_poles _ _ h1L]
      apply QSqrt2.ext <;> simp [QSqrt2.sqrt2Half] <;> try grind

/-- 独立乘积自旋配置（键维 `χ = 1` 平均场）的闭环约束能量 `∑_i (x_i x_{i+1} - 1)^2`。 -/
def cycleMeanFieldEnergy {L : Nat} (hL : 2 ≤ L) (x : Fin L → Rat) : Rat :=
  natRangeSumR L (fun i =>
    if h : i < L then
      indEnergy (x ⟨i, h⟩) (x ⟨(i + 1) % L, Nat.mod_lt _ (by omega)⟩)
    else 0)

/--
**定理 4（`rank1_product_zero_marginals_positive_energy`，诚实对照）**：
若键维 `χ = 1` 的独立乘积态在长度 `L ≥ 2` 的闭环上存在任意归零单比特边缘期望（`∃ i, x i = 0`），
则其循环平均场约束能量必然满足 `1 ≤ cycleMeanFieldEnergy hL x`（绝对无法与高维纠缠 GHZ 态那样
同时实现全零边缘期望与零约束残差）。
注：本系列还明确边界条件——常数键维 `χ = 2` MPS 的零势垒仅适用于 2 条全局互补对称解分支；
一般非线性多分支约束网络保持零残差所需键维 `χ` 将随约束图树宽（cut-width）的纠缠熵指数级增长。
-/
theorem rank1_product_zero_marginals_positive_energy (L : Nat) (hL : 2 ≤ L)
    (x : Fin L → Rat) (hZero : ∃ i : Fin L, x i = 0) :
    1 ≤ cycleMeanFieldEnergy hL x := by
  rcases hZero with ⟨i, hiZero⟩
  unfold cycleMeanFieldEnergy
  have hNonneg : ∀ j : Nat, j < L →
      0 ≤ (if h : j < L then
        indEnergy (x ⟨j, h⟩) (x ⟨(j + 1) % L, Nat.mod_lt _ (by omega)⟩) else 0) := by
    intro j hj
    simp only [hj, ↓reduceDIte, indEnergy]
    exact rat_sq_nonneg _
  have hGe := natRangeSumR_ge_term L i.val i.isLt _ hNonneg
  have hAtI : (if h : i.val < L then
      indEnergy (x ⟨i.val, h⟩) (x ⟨(i.val + 1) % L, Nat.mod_lt _ (by omega)⟩) else 0) = 1 := by
    simp only [i.isLt, ↓reduceDIte]
    have hEqI : (⟨i.val, i.isLt⟩ : Fin L) = i := Fin.ext rfl
    rw [hEqI, hiZero]
    unfold indEnergy
    grind
  rw [hAtI] at hGe
  exact hGe

end PhaseSHA.V5.TensorEntanglementBarrier
