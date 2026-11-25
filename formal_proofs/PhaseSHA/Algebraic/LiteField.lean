/-!
# LiteField: 零依赖精确有理域、复高斯有理数 $\mathbb{Q}[i]$ 与 $\varepsilon$-$\delta$ 连续性基础

本文件完全基于 Lean 4 标准库（`Init`/`Std`，无外部 Mathlib 依赖）构造：
1. `Rat` 的绝对值、幂次与严格单调性、`ratFloor` / `ratFrac` / `circleDist` 周期性引理；
2. 精确高斯有理数域 `GaussRat`（$\mathbb{Q}[i]$）及其范数平方 `normSq` 的乘性、非负性与正定性；
3. 经典 $\varepsilon$-$\delta$ 连续性 `RatContinuousAt` 与闭区间 `[0, 1]` 上的 Lipschitz 连续性。
-/

namespace PhaseSHA.Algebraic.LiteField

/-!
### 1. `Rat` 除法、乘积正负性与绝对值基础引理
-/

theorem rat_div_nonneg {a b : Rat} (ha : 0 ≤ a) (hb : 0 < b) : 0 ≤ a / b := by
  rw [Rat.div_def]
  exact Rat.mul_nonneg ha (Rat.le_of_lt (Rat.inv_pos.mpr hb))

theorem rat_div_pos {a b : Rat} (ha : 0 < a) (hb : 0 < b) : 0 < a / b := by
  rw [Rat.div_def]
  exact Rat.mul_pos ha (Rat.inv_pos.mpr hb)

theorem rat_le_div_iff {a b c : Rat} (hc : 0 < c) : a ≤ b / c ↔ a * c ≤ b := by
  have hne : c ≠ 0 := by grind
  constructor
  · intro h
    have h1 : a * c ≤ (b / c) * c := by
      have h0 : 0 ≤ (b / c - a) * c := Rat.mul_nonneg (by grind) (Rat.le_of_lt hc)
      have heq : (b / c - a) * c = (b / c) * c - a * c := by grind
      grind
    rw [Rat.div_mul_cancel hne] at h1
    exact h1
  · intro h
    have hinv : 0 < c⁻¹ := Rat.inv_pos.mpr hc
    have h1 : (a * c) * c⁻¹ ≤ b * c⁻¹ := by
      have h0 : 0 ≤ (b - a * c) * c⁻¹ := Rat.mul_nonneg (by grind) (Rat.le_of_lt hinv)
      have heq : (b - a * c) * c⁻¹ = b * c⁻¹ - (a * c) * c⁻¹ := by grind
      grind
    have hmul : (a * c) * c⁻¹ = a := by
      have h2 : (a * c) * c⁻¹ = a * (c * c⁻¹) := by grind
      rw [h2, Rat.mul_inv_cancel c hne]
      grind
    rw [hmul, ← Rat.div_def] at h1
    exact h1

theorem rat_div_le_iff {a b c : Rat} (hc : 0 < c) : a / c ≤ b ↔ a ≤ b * c := by
  have hne : c ≠ 0 := by grind
  constructor
  · intro h
    have h1 : (a / c) * c ≤ b * c := by
      have h0 : 0 ≤ (b - a / c) * c := Rat.mul_nonneg (by grind) (Rat.le_of_lt hc)
      have heq : (b - a / c) * c = b * c - (a / c) * c := by grind
      grind
    rw [Rat.div_mul_cancel hne] at h1
    exact h1
  · intro h
    have hinv : 0 < c⁻¹ := Rat.inv_pos.mpr hc
    have h1 : a * c⁻¹ ≤ (b * c) * c⁻¹ := by
      have h0 : 0 ≤ (b * c - a) * c⁻¹ := Rat.mul_nonneg (by grind) (Rat.le_of_lt hinv)
      have heq : (b * c - a) * c⁻¹ = (b * c) * c⁻¹ - a * c⁻¹ := by grind
      grind
    have hmul : (b * c) * c⁻¹ = b := by
      have h2 : (b * c) * c⁻¹ = b * (c * c⁻¹) := by grind
      rw [h2, Rat.mul_inv_cancel c hne]
      grind
    rw [hmul, ← Rat.div_def] at h1
    exact h1

theorem rat_mul_le_mul_of_nonneg_left {a b c : Rat} (h : a ≤ b) (hc : 0 ≤ c) :
    c * a ≤ c * b := by
  have h0 : 0 ≤ c * (b - a) := Rat.mul_nonneg hc (by grind)
  have heq : c * (b - a) = c * b - c * a := by grind
  grind

theorem rat_mul_le_mul_of_nonneg_right {a b c : Rat} (h : a ≤ b) (hc : 0 ≤ c) :
    a * c ≤ b * c := by
  have h0 : 0 ≤ (b - a) * c := Rat.mul_nonneg (by grind) hc
  have heq : (b - a) * c = b * c - a * c := by grind
  grind

theorem rat_mul_neg_of_pos_of_neg {a b : Rat} (ha : 0 < a) (hb : b < 0) : a * b < 0 := by
  have h : 0 < a * (-b) := Rat.mul_pos ha (by grind)
  have heq : a * (-b) = -(a * b) := by grind
  grind

theorem rat_mul_neg_of_neg_of_pos {a b : Rat} (ha : a < 0) (hb : 0 < b) : a * b < 0 := by
  have h : 0 < (-a) * b := Rat.mul_pos (by grind) hb
  have heq : (-a) * b = -(a * b) := by grind
  grind

theorem rat_mul_pos_of_neg_of_neg {a b : Rat} (ha : a < 0) (hb : b < 0) : 0 < a * b := by
  have h : 0 < (-a) * (-b) := Rat.mul_pos (by grind) (by grind)
  have heq : (-a) * (-b) = a * b := by grind
  grind

theorem rat_mul_nonneg_of_nonpos_of_nonpos {a b : Rat} (ha : a ≤ 0) (hb : b ≤ 0) :
    0 ≤ a * b := by
  have h : 0 ≤ (-a) * (-b) := Rat.mul_nonneg (by grind) (by grind)
  have heq : (-a) * (-b) = a * b := by grind
  grind

theorem rat_mul_eq_zero_iff {a b : Rat} : a * b = 0 ↔ a = 0 ∨ b = 0 := by
  constructor
  · intro h
    by_cases ha : a = 0
    · exact Or.inl ha
    · right
      have h1 : a⁻¹ * (a * b) = a⁻¹ * 0 := congrArg (fun z => a⁻¹ * z) h
      have h2 : a⁻¹ * (a * b) = (a * a⁻¹) * b := by grind
      rw [h2, Rat.mul_inv_cancel a ha] at h1
      grind
  · intro h
    cases h with
    | inl ha => simp [ha]
    | inr hb => simp [hb]

theorem rat_lt_of_le_of_lt {a b c : Rat} (h1 : a ≤ b) (h2 : b < c) : a < c := by
  grind

theorem rat_sq_nonneg (x : Rat) : 0 ≤ x * x := by
  rcases Rat.le_total (a := 0) (b := x) with hx | hx
  · exact Rat.mul_nonneg hx hx
  · exact rat_mul_nonneg_of_nonpos_of_nonpos hx hx

theorem rat_sq_eq_zero_iff (x : Rat) : x * x = 0 ↔ x = 0 := by
  rw [rat_mul_eq_zero_iff]
  exact ⟨fun h => h.elim id id, fun h => Or.inl h⟩

theorem rat_sum_sq_eq_zero_iff (x y : Rat) : x * x + y * y = 0 ↔ x = 0 ∧ y = 0 := by
  have hx := rat_sq_nonneg x
  have hy := rat_sq_nonneg y
  constructor
  · intro h
    have hx0 : x * x = 0 := by grind
    have hy0 : y * y = 0 := by grind
    exact ⟨(rat_sq_eq_zero_iff x).mp hx0, (rat_sq_eq_zero_iff y).mp hy0⟩
  · intro ⟨hx0, hy0⟩
    simp [hx0, hy0]
    grind

/-- 有理数绝对值。 -/
def ratAbs (x : Rat) : Rat := if x < 0 then -x else x

theorem ratAbs_nonneg (x : Rat) : 0 ≤ ratAbs x := by
  unfold ratAbs; split <;> grind

theorem ratAbs_of_nonneg {x : Rat} (hx : 0 ≤ x) : ratAbs x = x := by
  unfold ratAbs; split <;> grind

theorem ratAbs_of_neg {x : Rat} (hx : x < 0) : ratAbs x = -x := by
  unfold ratAbs; split <;> grind

theorem ratAbs_of_nonpos {x : Rat} (hx : x ≤ 0) : ratAbs x = -x := by
  unfold ratAbs; split <;> grind

@[simp] theorem ratAbs_zero : ratAbs 0 = 0 := by decide

theorem ratAbs_neg (x : Rat) : ratAbs (-x) = ratAbs x := by
  unfold ratAbs; split <;> split <;> grind

theorem ratAbs_sub_comm (x y : Rat) : ratAbs (x - y) = ratAbs (y - x) := by
  have heq : x - y = -(y - x) := by grind
  rw [heq, ratAbs_neg]

theorem ratAbs_eq_zero_iff (x : Rat) : ratAbs x = 0 ↔ x = 0 := by
  unfold ratAbs; split <;> grind

theorem ratAbs_add_le (x y : Rat) : ratAbs (x + y) ≤ ratAbs x + ratAbs y := by
  unfold ratAbs; split <;> split <;> split <;> grind

theorem ratAbs_le_iff (x b : Rat) : ratAbs x ≤ b ↔ -b ≤ x ∧ x ≤ b := by
  unfold ratAbs; split <;> grind

theorem ratAbs_lt_iff (x b : Rat) : ratAbs x < b ↔ -b < x ∧ x < b := by
  unfold ratAbs; split <;> grind

theorem ratAbs_mul (x y : Rat) : ratAbs (x * y) = ratAbs x * ratAbs y := by
  by_cases hx : x < 0 <;> by_cases hy : y < 0
  · have hxy : 0 < x * y := rat_mul_pos_of_neg_of_neg hx hy
    rw [ratAbs_of_neg hx, ratAbs_of_neg hy, ratAbs_of_nonneg (Rat.le_of_lt hxy)]
    grind
  · have hy0 : 0 ≤ y := by grind
    rcases (show y = 0 ∨ 0 < y by grind) with rfl | hyPos
    · simp
    · have hxy : x * y < 0 := rat_mul_neg_of_neg_of_pos hx hyPos
      rw [ratAbs_of_neg hx, ratAbs_of_nonneg hy0, ratAbs_of_neg hxy]
      grind
  · have hx0 : 0 ≤ x := by grind
    rcases (show x = 0 ∨ 0 < x by grind) with rfl | hxPos
    · simp
    · have hxy : x * y < 0 := rat_mul_neg_of_pos_of_neg hxPos hy
      rw [ratAbs_of_nonneg hx0, ratAbs_of_neg hy, ratAbs_of_neg hxy]
      grind
  · have hx0 : 0 ≤ x := by grind
    have hy0 : 0 ≤ y := by grind
    have hxy : 0 ≤ x * y := Rat.mul_nonneg hx0 hy0
    rw [ratAbs_of_nonneg hx0, ratAbs_of_nonneg hy0, ratAbs_of_nonneg hxy]

theorem ratAbs_div (x y : Rat) (hy : 0 < y) : ratAbs (x / y) = ratAbs x / y := by
  have hyinv : 0 < y⁻¹ := Rat.inv_pos.mpr hy
  rw [Rat.div_def, ratAbs_mul, ratAbs_of_nonneg (Rat.le_of_lt hyinv), Rat.div_def]

/-!
### 2. 有理数自然数幂 `ratPow` 及其单调性与可逆性
-/

/-- 有理数的自然数幂 `x^n`。 -/
def ratPow (x : Rat) : Nat → Rat
  | 0 => 1
  | n + 1 => ratPow x n * x

@[simp] theorem ratPow_zero (x : Rat) : ratPow x 0 = 1 := rfl
@[simp] theorem ratPow_succ (x : Rat) (n : Nat) : ratPow x (n + 1) = ratPow x n * x := rfl
@[simp] theorem ratPow_one (x : Rat) : ratPow x 1 = x := by simp [ratPow]

theorem ratPow_nonneg {x : Rat} (hx : 0 ≤ x) (n : Nat) : 0 ≤ ratPow x n := by
  induction n with
  | zero => rw [ratPow_zero]; decide
  | succ k ih => exact Rat.mul_nonneg ih hx

theorem ratPow_pos {x : Rat} (hx : 0 < x) (n : Nat) : 0 < ratPow x n := by
  induction n with
  | zero => rw [ratPow_zero]; decide
  | succ k ih => exact Rat.mul_pos ih hx

theorem ratPow_ne_zero {x : Rat} (hx : x ≠ 0) (n : Nat) : ratPow x n ≠ 0 := by
  induction n with
  | zero => rw [ratPow_zero]; decide
  | succ k ih =>
    intro h
    rcases rat_mul_eq_zero_iff.mp h with h1 | h2
    · exact ih h1
    · exact hx h2

theorem ratPow_mul_base (x y : Rat) (n : Nat) :
    ratPow (x * y) n = ratPow x n * ratPow y n := by
  induction n with
  | zero => simp
  | succ k ih =>
    simp only [ratPow_succ, ih]
    grind

theorem ratPow_div (x y : Rat) (hy : y ≠ 0) (n : Nat) :
    ratPow x n / ratPow y n = ratPow (x / y) n := by
  induction n with
  | zero => simp; grind
  | succ k ih =>
    have hyk : ratPow y k ≠ 0 := ratPow_ne_zero hy k
    simp only [ratPow_succ, ← ih]
    rw [Rat.div_def, Rat.div_def, Rat.div_def]
    have hmulInv : (ratPow y k * y)⁻¹ = y⁻¹ * (ratPow y k)⁻¹ := by
      have hProd : (ratPow y k * y) * (y⁻¹ * (ratPow y k)⁻¹) = 1 := by
        have h1 : (ratPow y k * y) * (y⁻¹ * (ratPow y k)⁻¹) =
            ratPow y k * (y * y⁻¹) * (ratPow y k)⁻¹ := by grind
        rw [h1, Rat.mul_inv_cancel y hy]
        have h2 : ratPow y k * 1 * (ratPow y k)⁻¹ = ratPow y k * (ratPow y k)⁻¹ := by grind
        rw [h2, Rat.mul_inv_cancel (ratPow y k) hyk]
      have hL : (ratPow y k * y)⁻¹ * ((ratPow y k * y) * (y⁻¹ * (ratPow y k)⁻¹)) =
          (ratPow y k * y)⁻¹ * 1 := congrArg (fun z => (ratPow y k * y)⁻¹ * z) hProd
      have hne : ratPow y k * y ≠ 0 := by
        intro h
        rcases rat_mul_eq_zero_iff.mp h with h1 | h2
        · exact hyk h1
        · exact hy h2
      have hAssoc : (ratPow y k * y)⁻¹ * ((ratPow y k * y) * (y⁻¹ * (ratPow y k)⁻¹)) =
          ((ratPow y k * y) * (ratPow y k * y)⁻¹) * (y⁻¹ * (ratPow y k)⁻¹) := by grind
      rw [hAssoc, Rat.mul_inv_cancel _ hne] at hL
      grind
    rw [hmulInv]
    grind

theorem ratPow_le_ratPow_of_le {a b : Rat} (ha : 0 ≤ a) (hab : a ≤ b) (n : Nat) :
    ratPow a n ≤ ratPow b n := by
  induction n with
  | zero => simp
  | succ k ih =>
    have hb : 0 ≤ b := by grind
    have h1 : ratPow a k * a ≤ ratPow b k * a := rat_mul_le_mul_of_nonneg_right ih ha
    have h2 : ratPow b k * a ≤ ratPow b k * b :=
      rat_mul_le_mul_of_nonneg_left hab (ratPow_nonneg hb k)
    exact Rat.le_trans h1 h2

theorem ratPow_lt_ratPow_of_lt {a b : Rat} (ha : 0 ≤ a) (hab : a < b) {n : Nat}
    (hn : 1 ≤ n) : ratPow a n < ratPow b n := by
  induction n with
  | zero => omega
  | succ k ih =>
    cases k with
    | zero => simpa using hab
    | succ k' =>
      have ih' := ih (Nat.succ_pos k')
      have hb : 0 < b := by grind
      have h1 : ratPow a (k' + 1) * a ≤ ratPow b (k' + 1) * a :=
        rat_mul_le_mul_of_nonneg_right (Rat.le_of_lt ih') ha
      have h2 : ratPow b (k' + 1) * a < ratPow b (k' + 1) * b :=
        Rat.mul_lt_mul_of_pos_left hab (ratPow_pos hb (k' + 1))
      exact rat_lt_of_le_of_lt h1 h2

theorem ratPow_le_ratPow_iff {a b : Rat} (ha : 0 ≤ a) (hb : 0 ≤ b) {n : Nat}
    (hn : 1 ≤ n) : ratPow a n ≤ ratPow b n ↔ a ≤ b := by
  constructor
  · intro hPow
    rcases Rat.le_total (a := a) (b := b) with hab | hba
    · exact hab
    · by_cases heq : b = a
      · grind
      · have hlt : b < a := (Rat.lt_iff_le_and_ne).mpr ⟨hba, heq⟩
        have hPowLt := ratPow_lt_ratPow_of_lt hb hlt hn
        grind
  · intro hab
    exact ratPow_le_ratPow_of_le ha hab n

theorem ratPow_lt_succ_of_gt_one {x : Rat} (hx : 1 < x) (n : Nat) :
    ratPow x n < ratPow x (n + 1) := by
  have hpos : 0 < ratPow x n := ratPow_pos (by grind) n
  have h1 : ratPow x n * 1 < ratPow x n * x := Rat.mul_lt_mul_of_pos_left hx hpos
  simpa [ratPow_succ] using h1

/-!
### 3. `ratFloor`、周期 1 小数部分 `ratFrac` 与圆周对称折返距离 `circleDist`
-/

/-- 向下取整 `floor(q)`。 -/
def ratFloor (q : Rat) : Int := q.num / (q.den : Int)

/-- 周期为 1 的小数部分 `frac(q) = q - floor(q) ∈ [0, 1)`。 -/
def ratFrac (q : Rat) : Rat := q - (ratFloor q : Rat)

/-- 圆周对称折返距离 `circleDist(q) = min(frac(q), 1 - frac(q)) ∈ [0, 1/2]`。 -/
def circleDist (q : Rat) : Rat :=
  let f := ratFrac q
  if f ≤ 1 / 2 then f else 1 - f

theorem ratFloor_le (q : Rat) : (ratFloor q : Rat) ≤ q := by
  rw [Rat.le_iff]
  change ratFloor q * (q.den : Int) ≤ q.num * 1
  have hDenPos : (0 : Int) < (q.den : Int) := Int.natCast_pos.mpr q.den_pos
  have hDiv := Int.ediv_mul_add_emod q.num (q.den : Int)
  have hMod : 0 ≤ q.num % (q.den : Int) :=
    Int.emod_nonneg q.num (Int.ne_of_gt hDenPos)
  change (q.num / (q.den : Int)) * (q.den : Int) ≤ q.num * 1
  omega

theorem lt_ratFloor_add_one (q : Rat) : q < ((ratFloor q + 1 : Int) : Rat) := by
  rw [Rat.lt_iff]
  change q.num * 1 < (ratFloor q + 1) * (q.den : Int)
  have hDenPos : (0 : Int) < (q.den : Int) := Int.natCast_pos.mpr q.den_pos
  have hDiv := Int.ediv_mul_add_emod q.num (q.den : Int)
  have hMod : q.num % (q.den : Int) < (q.den : Int) :=
    Int.emod_lt_of_pos q.num hDenPos
  change q.num * 1 < (q.num / (q.den : Int) + 1) * (q.den : Int)
  have hExpand : (q.num / (q.den : Int) + 1) * (q.den : Int) =
      (q.num / (q.den : Int)) * (q.den : Int) + 1 * (q.den : Int) :=
    Int.add_mul (q.num / (q.den : Int)) 1 (q.den : Int)
  omega

theorem ratFrac_nonneg (q : Rat) : 0 ≤ ratFrac q := by
  have h := ratFloor_le q
  unfold ratFrac
  grind

theorem ratFrac_lt_one (q : Rat) : ratFrac q < 1 := by
  have h := lt_ratFloor_add_one q
  rw [Rat.intCast_add] at h
  unfold ratFrac
  grind

theorem ratFloor_unique {q : Rat} {k : Int}
    (h1 : (k : Rat) ≤ q) (h2 : q < ((k + 1 : Int) : Rat)) :
    k = ratFloor q := by
  have hf1 : (ratFloor q : Rat) ≤ q := ratFloor_le q
  have hf2 : q < ((ratFloor q + 1 : Int) : Rat) := lt_ratFloor_add_one q
  rw [Rat.intCast_add] at h2 hf2
  have hLt1 : ((k - ratFloor q : Int) : Rat) < 1 := by
    rw [Rat.intCast_sub]
    grind
  have hLt2 : ((-1 : Int) : Rat) < ((k - ratFloor q : Int) : Rat) := by
    rw [Rat.intCast_sub]
    grind
  rw [Rat.lt_iff] at hLt1 hLt2
  change (k - ratFloor q) * 1 < 1 * 1 at hLt1
  change (-1) * 1 < (k - ratFloor q) * 1 at hLt2
  omega

theorem ratFloor_add_int (q : Rat) (k : Int) :
    ratFloor (q + (k : Rat)) = ratFloor q + k := by
  symm
  apply ratFloor_unique
  · rw [Rat.intCast_add]
    have h := ratFloor_le q
    grind
  · have heq : ((ratFloor q + k + 1 : Int) : Rat) = ((ratFloor q + 1 : Int) : Rat) + (k : Rat) := by
      rw [Rat.intCast_add, Rat.intCast_add, Rat.intCast_add]
      grind
    rw [heq]
    have h := lt_ratFloor_add_one q
    grind

theorem ratFrac_add_int (q : Rat) (k : Int) :
    ratFrac (q + (k : Rat)) = ratFrac q := by
  unfold ratFrac
  rw [ratFloor_add_int, Rat.intCast_add]
  grind

theorem ratFloor_zero : ratFloor 0 = 0 := by
  symm
  apply ratFloor_unique <;> grind

theorem ratFrac_zero : ratFrac 0 = 0 := by
  unfold ratFrac
  rw [ratFloor_zero]
  grind

theorem ratFloor_half : ratFloor (1 / 2) = 0 := by
  symm
  apply ratFloor_unique <;> grind

theorem ratFrac_half : ratFrac (1 / 2) = 1 / 2 := by
  unfold ratFrac
  rw [ratFloor_half]
  grind

theorem ratFrac_eq_zero_iff (q : Rat) : ratFrac q = 0 ↔ ∃ k : Int, q = (k : Rat) := by
  constructor
  · intro h
    refine ⟨ratFloor q, ?_⟩
    unfold ratFrac at h
    grind
  · intro ⟨k, hk⟩
    rw [hk]
    have hZ : (k : Rat) = (0 : Rat) + (k : Rat) := by grind
    rw [hZ, ratFrac_add_int, ratFrac_zero]

theorem circleDist_nonneg (q : Rat) : 0 ≤ circleDist q := by
  unfold circleDist
  dsimp only
  have h1 := ratFrac_nonneg q
  have h2 := ratFrac_lt_one q
  split <;> grind

theorem circleDist_le_half (q : Rat) : circleDist q ≤ 1 / 2 := by
  unfold circleDist
  dsimp only
  have h1 := ratFrac_nonneg q
  have h2 := ratFrac_lt_one q
  split <;> grind

theorem circleDist_add_int (q : Rat) (k : Int) :
    circleDist (q + (k : Rat)) = circleDist q := by
  simp only [circleDist, ratFrac_add_int]

theorem circleDist_eq_zero_iff (q : Rat) :
    circleDist q = 0 ↔ ∃ k : Int, q = (k : Rat) := by
  rw [← ratFrac_eq_zero_iff q]
  unfold circleDist
  dsimp only
  have h1 := ratFrac_nonneg q
  have h2 := ratFrac_lt_one q
  split <;> grind

theorem circleDist_eq_half_iff (q : Rat) :
    circleDist q = 1 / 2 ↔ ratFrac q = 1 / 2 := by
  unfold circleDist
  dsimp only
  have h1 := ratFrac_nonneg q
  have h2 := ratFrac_lt_one q
  split <;> grind

theorem ratFrac_eq_half_iff (q : Rat) :
    ratFrac q = 1 / 2 ↔ ∃ k : Int, q = (k : Rat) + 1 / 2 := by
  constructor
  · intro h
    refine ⟨ratFloor q, ?_⟩
    unfold ratFrac at h
    grind
  · intro ⟨k, hk⟩
    rw [hk]
    have hAdd : (k : Rat) + 1 / 2 = (1 / 2 : Rat) + (k : Rat) := by grind
    rw [hAdd, ratFrac_add_int, ratFrac_half]

theorem circleDist_eq_half_iff_exists (q : Rat) :
    circleDist q = 1 / 2 ↔ ∃ k : Int, q = (k : Rat) + 1 / 2 :=
  (circleDist_eq_half_iff q).trans (ratFrac_eq_half_iff q)

/-!
### 4. 高斯有理数 $\mathbb{Q}[i]$ (`GaussRat`)
-/

/-- 高斯有理数 `re + im * i ∈ ℚ[i]`。 -/
structure GaussRat where
  re : Rat
  im : Rat
  deriving DecidableEq, Repr

namespace GaussRat

@[ext] theorem ext {z w : GaussRat} (hre : z.re = w.re) (him : z.im = w.im) : z = w := by
  cases z; cases w; simp_all

def zero : GaussRat := ⟨0, 0⟩
def one : GaussRat := ⟨1, 0⟩
def I : GaussRat := ⟨0, 1⟩
def ofRat (r : Rat) : GaussRat := ⟨r, 0⟩

def add (z w : GaussRat) : GaussRat := ⟨z.re + w.re, z.im + w.im⟩
def sub (z w : GaussRat) : GaussRat := ⟨z.re - w.re, z.im - w.im⟩
def neg (z : GaussRat) : GaussRat := ⟨-z.re, -z.im⟩
def mul (z w : GaussRat) : GaussRat :=
  ⟨z.re * w.re - z.im * w.im, z.re * w.im + z.im * w.re⟩
def conj (z : GaussRat) : GaussRat := ⟨z.re, -z.im⟩
def smul (r : Rat) (z : GaussRat) : GaussRat := ⟨r * z.re, r * z.im⟩

instance : Zero GaussRat := ⟨zero⟩
instance : One GaussRat := ⟨one⟩
instance : Add GaussRat := ⟨add⟩
instance : Sub GaussRat := ⟨sub⟩
instance : Neg GaussRat := ⟨neg⟩
instance : Mul GaussRat := ⟨mul⟩

@[simp] theorem zero_re : (0 : GaussRat).re = 0 := rfl
@[simp] theorem zero_im : (0 : GaussRat).im = 0 := rfl
@[simp] theorem one_re : (1 : GaussRat).re = 1 := rfl
@[simp] theorem one_im : (1 : GaussRat).im = 0 := rfl
@[simp] theorem add_re (z w : GaussRat) : (z + w).re = z.re + w.re := rfl
@[simp] theorem add_im (z w : GaussRat) : (z + w).im = z.im + w.im := rfl
@[simp] theorem sub_re (z w : GaussRat) : (z - w).re = z.re - w.re := rfl
@[simp] theorem sub_im (z w : GaussRat) : (z - w).im = z.im - w.im := rfl
@[simp] theorem neg_re (z : GaussRat) : (-z).re = -z.re := rfl
@[simp] theorem neg_im (z : GaussRat) : (-z).im = -z.im := rfl
@[simp] theorem mul_re (z w : GaussRat) : (z * w).re = z.re * w.re - z.im * w.im := rfl
@[simp] theorem mul_im (z w : GaussRat) : (z * w).im = z.re * w.im + z.im * w.re := rfl

/-- 复模长平方 `|z|^2 = re^2 + im^2`。 -/
def normSq (z : GaussRat) : Rat := z.re * z.re + z.im * z.im

theorem normSq_nonneg (z : GaussRat) : 0 ≤ normSq z := by
  unfold normSq
  exact Rat.add_nonneg (rat_sq_nonneg z.re) (rat_sq_nonneg z.im)

theorem normSq_eq_zero_iff (z : GaussRat) : normSq z = 0 ↔ z = 0 := by
  unfold normSq
  rw [rat_sum_sq_eq_zero_iff]
  constructor
  · intro ⟨h1, h2⟩; ext <;> simp [h1, h2]
  · intro h; subst h; simp

theorem normSq_mul (z w : GaussRat) : normSq (z * w) = normSq z * normSq w := by
  unfold normSq
  simp only [mul_re, mul_im]
  grind

end GaussRat

/-!
### 5. 经典 $\varepsilon$-$\delta$ 连续性与区间 Lipschitz 定义
-/

/-- `f : Rat → Rat` 在点 `t0` 满足经典 $\varepsilon$-$\delta$ 连续性。 -/
def RatContinuousAt (f : Rat → Rat) (t0 : Rat) : Prop :=
  ∀ eps : Rat, 0 < eps → ∃ delta : Rat, 0 < delta ∧
    ∀ t : Rat, ratAbs (t - t0) < delta → ratAbs (f t - f t0) < eps

/-- `f : Rat → Rat` 在区间 `[0, 1]` 上满足 `L`-Lipschitz 连续性。 -/
def IsLipschitzOn01 (f : Rat → Rat) (L : Rat) : Prop :=
  0 < L ∧ ∀ t t0 : Rat, 0 ≤ t → t ≤ 1 → 0 ≤ t0 → t0 ≤ 1 →
    ratAbs (f t - f t0) ≤ L * ratAbs (t - t0)

theorem lipschitzOn01_continuousAt {f : Rat → Rat} {L t0 : Rat}
    (hLip : IsLipschitzOn01 f L) (ht0a : 0 ≤ t0) (ht0b : t0 ≤ 1) :
    ∀ eps : Rat, 0 < eps → ∃ delta : Rat, 0 < delta ∧
      ∀ t : Rat, 0 ≤ t → t ≤ 1 → ratAbs (t - t0) < delta → ratAbs (f t - f t0) < eps := by
  intro eps hEps
  have hPos : 0 < L := hLip.1
  have hNe : L ≠ 0 := by grind
  refine ⟨eps / L, rat_div_pos hEps hPos, ?_⟩
  intro t hta htb hDist
  have hBound := hLip.2 t t0 hta htb ht0a ht0b
  have hMul : L * ratAbs (t - t0) < L * (eps / L) :=
    Rat.mul_lt_mul_of_pos_left hDist hPos
  have hSimp : L * (eps / L) = eps := by
    rw [Rat.div_def]
    have h1 : L * (eps * L⁻¹) = eps * (L * L⁻¹) := by grind
    rw [h1, Rat.mul_inv_cancel L hNe]
    grind
  rw [hSimp] at hMul
  exact rat_lt_of_le_of_lt hBound hMul

end PhaseSHA.Algebraic.LiteField
