import Std.Tactic.BVDecide
import PhaseSHA.Algebraic.LiteField

/-!
# Module 2: 2-adic 相角塔完备性与零伪解定理

本模块在零外部依赖下严格形式化 Phase-SHA 的 32 尺度 2-adic 模加一致性与相角塔能量：
1. 构造一般单位圆相位残差核 `UnitCircleKernel`，并给出两个 100% 零公理具体实例：
   - 真实复高斯有理单位圆 $\mathbb{S}^1 \subset \mathbb{Q}[i]$ 上的精确弦长残差 `cayleyCircleKernel`
     （严格证明点列恒满足 `|z|^2 = 1`，并在 $\pi$ 相位反转处取 `z = -1`、`|1 - (-1)|^2 = 4`）；
   - 有理二次相角下界核 `jordanLowerKernel`；
2. **`two_adic_zero_iff_mod_add`**：严格证明 32 尺度相角塔总能量为 `0` 当且仅当
   32-bit 模加方程 `a + b = c` 精确成立（亦等价于全部 32 个尺度同余同时成立，零伪极小值陷阱）；
3. **`lsb_carry_decoupling_and_flip_lower_bound`**：严格证明第 0 位（LSB, `m = 1`）残差仅依赖
   `a[0] ⊕ b[0] ⊕ c[0]` 而零进位耦合，且对任意合法解翻转任意单个比特 `k ∈ [0, 31]`，
   必然在尺度 `m = k + 1` 触发精确的 $\pi$ 相位反转与最大局部能量跃迁 `4`（总能量下界 `≥ 4`）。
-/

namespace PhaseSHA.Algebraic.TwoAdicTowerSoundness

open PhaseSHA.Algebraic.LiteField

/-!
### 1. 单位圆相位残差核与复高斯单位圆具体实例
-/

/-- 单位圆相位残差核 `|1 - exp(i · 2πq)|^2` 的结构刻画（周期 1、非负、零点仅在整数、半周期取 4）。 -/
structure UnitCircleKernel where
  energy : Rat → Rat
  periodic : ∀ q : Rat, ∀ k : Int, energy (q + (k : Rat)) = energy q
  nonneg : ∀ q : Rat, 0 ≤ energy q
  zero_iff : ∀ q : Rat, energy q = 0 ↔ ∃ k : Int, q = (k : Rat)
  half_turn : energy (1 / 2) = 4

/-- 有理 Cayley 单位圆参数化点。 -/
def cayleyPair (u : Rat) : GaussRat :=
  let den := 1 + u * u
  ⟨(1 - u * u) / den, (2 * u) / den⟩

private theorem rat_div_add_div_same (a b c : Rat) :
    a / c + b / c = (a + b) / c := by
  rw [Rat.div_def, Rat.div_def, Rat.div_def]
  grind

private theorem rat_div_self {a : Rat} (ha : a ≠ 0) : a / a = 1 := by
  rw [Rat.div_def]
  exact Rat.mul_inv_cancel a ha

private theorem rat_div_sq (x d : Rat) (hd : d ≠ 0) :
    (x / d) * (x / d) = (x * x) / (d * d) := by
  have hPow := ratPow_div x d hd 2
  simpa [ratPow] using hPow.symm

theorem cayleyPair_on_circle (u : Rat) : (cayleyPair u).normSq = 1 := by
  unfold cayleyPair GaussRat.normSq
  dsimp only
  have hu2 := rat_sq_nonneg u
  have hdenNe : 1 + u * u ≠ 0 := by grind
  rw [rat_div_sq (1 - u * u) (1 + u * u) hdenNe,
      rat_div_sq (2 * u) (1 + u * u) hdenNe,
      rat_div_add_div_same]
  have hNum : (1 - u * u) * (1 - u * u) + (2 * u) * (2 * u) =
      (1 + u * u) * (1 + u * u) := by grind
  rw [hNum]
  have hden2Ne : (1 + u * u) * (1 + u * u) ≠ 0 := by
    intro h0
    rcases rat_mul_eq_zero_iff.mp h0 with h | h <;> exact hdenNe h
  exact rat_div_self hden2Ne

theorem cayleyPair_eq_one_iff (u : Rat) : cayleyPair u = GaussRat.one ↔ u = 0 := by
  constructor
  · intro hEq
    have him := congrArg GaussRat.im hEq
    unfold cayleyPair GaussRat.one at him
    dsimp only at him
    have hu2 := rat_sq_nonneg u
    have hdenNe : 1 + u * u ≠ 0 := by grind
    have hMul := congrArg (fun z => z * (1 + u * u)) him
    rw [Rat.div_mul_cancel hdenNe] at hMul
    grind
  · intro hu0
    subst hu0
    apply GaussRat.ext <;> dsimp [cayleyPair, GaussRat.one]
    · have h : (1 - (0 : Rat) * 0) / (1 + 0 * 0) = 1 := by grind
      exact h
    · have h : (2 * (0 : Rat)) / (1 + 0 * 0) = 0 := by grind
      exact h

/-- 复平面单位圆 $\mathbb{S}^1 \subset \mathbb{Q}[i]$ 上的精确有理 Cayley 映射点 `z(q)`。 -/
def cayleyUnitPoint (q : Rat) : GaussRat :=
  let d := circleDist q
  if d = 1 / 2 then ⟨-1, 0⟩ else cayleyPair (d / (1 / 2 - d))

theorem cayleyUnitPoint_add_int (q : Rat) (k : Int) :
    cayleyUnitPoint (q + (k : Rat)) = cayleyUnitPoint q := by
  simp only [cayleyUnitPoint, circleDist_add_int]

theorem cayleyUnitPoint_on_circle (q : Rat) : (cayleyUnitPoint q).normSq = 1 := by
  unfold cayleyUnitPoint
  dsimp only
  split
  · unfold GaussRat.normSq
    simp only
    grind
  · exact cayleyPair_on_circle _

theorem cayleyUnitPoint_half_turn : cayleyUnitPoint (1 / 2) = ⟨-1, 0⟩ := by
  have hd : circleDist (1 / 2) = 1 / 2 := (circleDist_eq_half_iff (1 / 2)).mpr ratFrac_half
  unfold cayleyUnitPoint
  simp [hd]

/-- 真实复高斯单位圆 $\mathbb{S}^1 \subset \mathbb{Q}[i]$ 上的相位能量核 `|1 - z(q)|^2`。 -/
def cayleyCircleKernel : UnitCircleKernel where
  energy q := (GaussRat.one - cayleyUnitPoint q).normSq
  periodic q k := by simp only [cayleyUnitPoint_add_int]
  nonneg q := GaussRat.normSq_nonneg _
  zero_iff q := by
    rw [GaussRat.normSq_eq_zero_iff]
    have hSubZero : GaussRat.one - cayleyUnitPoint q = 0 ↔ cayleyUnitPoint q = GaussRat.one := by
      constructor
      · intro h
        apply GaussRat.ext
        · have hre := congrArg GaussRat.re h
          simp only [GaussRat.sub_re, GaussRat.zero_re] at hre
          grind
        · have him := congrArg GaussRat.im h
          simp only [GaussRat.sub_im, GaussRat.zero_im] at him
          grind
      · intro h
        rw [h]
        apply GaussRat.ext
        · simp only [GaussRat.sub_re, GaussRat.zero_re]; grind
        · simp only [GaussRat.sub_im, GaussRat.zero_im]; grind
    rw [hSubZero]
    unfold cayleyUnitPoint
    dsimp only
    by_cases hdHalf : circleDist q = 1 / 2
    · simp only [hdHalf, ↓reduceIte]
      constructor
      · intro hNeg
        have hre := congrArg GaussRat.re hNeg
        simp [GaussRat.one] at hre
        grind
      · intro ⟨k, hk⟩
        have hd0 : circleDist q = 0 := (circleDist_eq_zero_iff q).mpr ⟨k, hk⟩
        rw [hd0] at hdHalf
        grind
    · simp only [hdHalf, ↓reduceIte, cayleyPair_eq_one_iff]
      have hdLe := circleDist_le_half q
      have hHalfSubPos : 0 < 1 / 2 - circleDist q := by grind
      have hHalfSubNe : 1 / 2 - circleDist q ≠ 0 := by grind
      constructor
      · intro hu0
        have hMul := congrArg (fun z => z * (1 / 2 - circleDist q)) hu0
        rw [Rat.div_mul_cancel hHalfSubNe] at hMul
        have hd0 : circleDist q = 0 := by grind
        exact (circleDist_eq_zero_iff q).mp hd0
      · intro hEx
        have hd0 : circleDist q = 0 := (circleDist_eq_zero_iff q).mpr hEx
        rw [hd0, Rat.div_def]
        grind
  half_turn := by
    rw [cayleyUnitPoint_half_turn]
    simp only [GaussRat.normSq, GaussRat.sub_re, GaussRat.sub_im, GaussRat.one]
    grind

/-- Jordan 下界核 `16 * circleDist(q)^2`（刻画 `4 * sin^2(πd) ≥ 16d^2` 下界且半周期精确取 4）。 -/
def jordanLowerKernel : UnitCircleKernel where
  energy q := 16 * (circleDist q * circleDist q)
  periodic q k := by simp only [circleDist_add_int]
  nonneg q := Rat.mul_nonneg (by decide) (rat_sq_nonneg (circleDist q))
  zero_iff q := by
    constructor
    · intro h
      have hsq : circleDist q * circleDist q = 0 := by grind
      have hd : circleDist q = 0 := (rat_sq_eq_zero_iff (circleDist q)).mp hsq
      exact (circleDist_eq_zero_iff q).mp hd
    · intro hEx
      have hd := (circleDist_eq_zero_iff q).mpr hEx
      simp [hd]
  half_turn := by
    have hd : circleDist (1 / 2) = 1 / 2 := (circleDist_eq_half_iff (1 / 2)).mpr ratFrac_half
    rw [hd]
    grind

/-!
### 2. 32 尺度 2-adic 相角塔定义与整除引理
-/

/-- 整数型 `2^m`。 -/
def pow2Int (m : Nat) : Int := ((2 ^ m : Nat) : Int)

theorem pow2Int_pos (m : Nat) : (0 : Int) < pow2Int m := by
  unfold pow2Int
  exact Int.natCast_pos.mpr (Nat.pow_pos (by decide))

/-- 精确整数差值 `a + b - c ∈ ℤ`。 -/
def signedDiff (a b c : BitVec 32) : Int :=
  (a.toNat : Int) + (b.toNat : Int) - (c.toNat : Int)

/-- 尺度 `m` 上的标准化相角（以 `2π` 为单位）：`(a + b - c) / 2^m`。 -/
def scalePhaseFrac (a b c : BitVec 32) (m : Nat) : Rat :=
  (signedDiff a b c : Rat) / (pow2Int m : Rat)

/-- 尺度 `m` 上的单位圆能量残差。 -/
def scaleEnergy (K : UnitCircleKernel) (a b c : BitVec 32) (m : Nat) : Rat :=
  K.energy (scalePhaseFrac a b c m)

/-- 累加尺度 `1 .. n` 的相角塔总能量。 -/
def sumScales (K : UnitCircleKernel) (a b c : BitVec 32) : Nat → Rat
  | 0 => 0
  | m + 1 => sumScales K a b c m + scaleEnergy K a b c (m + 1)

/-- 完整 32 尺度 2-adic 相角塔总能量 `∑_{m=1}^{32} scaleEnergy(a, b, c, m)`。 -/
def towerEnergy (K : UnitCircleKernel) (a b c : BitVec 32) : Rat :=
  sumScales K a b c 32

theorem sumScales_nonneg (K : UnitCircleKernel) (a b c : BitVec 32) (n : Nat) :
    0 ≤ sumScales K a b c n := by
  induction n with
  | zero => rw [sumScales]; decide
  | succ m ih =>
    have hE := K.nonneg (scalePhaseFrac a b c (m + 1))
    unfold sumScales
    change 0 ≤ sumScales K a b c m + scaleEnergy K a b c (m + 1)
    exact Rat.add_nonneg ih hE

theorem sumScales_eq_zero_iff (K : UnitCircleKernel) (a b c : BitVec 32) (n : Nat) :
    sumScales K a b c n = 0 ↔ ∀ m, 1 ≤ m → m ≤ n → scaleEnergy K a b c m = 0 := by
  induction n with
  | zero =>
    simp only [sumScales]
    exact ⟨fun _ _ hm1 hm0 => by omega, fun _ => trivial⟩
  | succ k ih =>
    have hSum : 0 ≤ sumScales K a b c k := sumScales_nonneg K a b c k
    have hCur : 0 ≤ scaleEnergy K a b c (k + 1) := K.nonneg _
    simp only [sumScales]
    constructor
    · intro hZ m hm1 hmk
      have hLeft : sumScales K a b c k = 0 := by grind
      have hRight : scaleEnergy K a b c (k + 1) = 0 := by grind
      by_cases hmkEq : m = k + 1
      · subst hmkEq; exact hRight
      · exact (ih.mp hLeft) m hm1 (by omega)
    · intro hAll
      have hLeft : sumScales K a b c k = 0 :=
        ih.mpr (fun m hm1 hmk => hAll m hm1 (Nat.le_succ_of_le hmk))
      have hRight : scaleEnergy K a b c (k + 1) = 0 :=
        hAll (k + 1) (Nat.succ_pos k) (Nat.le_refl _)
      rw [hLeft, hRight]
      grind

theorem scaleEnergy_le_sumScales (K : UnitCircleKernel) (a b c : BitVec 32)
    {m n : Nat} (hm1 : 1 ≤ m) (hmn : m ≤ n) :
    scaleEnergy K a b c m ≤ sumScales K a b c n := by
  induction n with
  | zero => omega
  | succ k ih =>
    have hSum : 0 ≤ sumScales K a b c k := sumScales_nonneg K a b c k
    have hCur : 0 ≤ scaleEnergy K a b c (k + 1) := K.nonneg _
    simp only [sumScales]
    by_cases hmk : m = k + 1
    · subst hmk; grind
    · have hLe := ih (by omega)
      grind

private theorem intCast_rat_inj {x y : Int} (h : (x : Rat) = (y : Rat)) : x = y := by
  have h1 : (x : Rat) ≤ (y : Rat) := by grind
  have h2 : (y : Rat) ≤ (x : Rat) := by grind
  rw [Rat.le_iff] at h1 h2
  change x * 1 ≤ y * 1 at h1
  change y * 1 ≤ x * 1 at h2
  omega

theorem scaleEnergy_eq_zero_iff_dvd (K : UnitCircleKernel) (a b c : BitVec 32) (m : Nat) :
    scaleEnergy K a b c m = 0 ↔ pow2Int m ∣ signedDiff a b c := by
  unfold scaleEnergy scalePhaseFrac
  rw [K.zero_iff]
  have hPosInt : (0 : Int) < pow2Int m := pow2Int_pos m
  have hPosRat : (0 : Rat) < (pow2Int m : Rat) := by
    rw [Rat.lt_iff]
    change 0 * 1 < pow2Int m * 1
    omega
  have hNe : (pow2Int m : Rat) ≠ 0 := by grind
  constructor
  · intro ⟨k, hk⟩
    have hMul := congrArg (fun z => z * (pow2Int m : Rat)) hk
    rw [Rat.div_mul_cancel hNe, ← Rat.intCast_mul] at hMul
    have hInt := intCast_rat_inj hMul
    refine ⟨k, ?_⟩
    rw [Int.mul_comm]
    exact hInt
  · intro ⟨k, hk⟩
    refine ⟨k, ?_⟩
    have hIntEq : signedDiff a b c = k * pow2Int m := by
      rw [hk, Int.mul_comm]
    rw [hIntEq, Rat.intCast_mul]
    exact Rat.mul_div_cancel hNe

theorem pow2Int_dvd_signedDiff_32_iff_add_eq (a b c : BitVec 32) :
    pow2Int 32 ∣ signedDiff a b c ↔ a + b = c := by
  rw [BitVec.toNat_eq, BitVec.toNat_add]
  have hcLt : c.toNat < 2 ^ 32 := c.isLt
  unfold pow2Int signedDiff
  constructor
  · intro ⟨q, hq⟩
    have hEqInt : ((a.toNat + b.toNat : Nat) : Int) = (2 ^ 32 : Nat) * q + (c.toNat : Int) := by
      omega
    have hMod : ((a.toNat + b.toNat : Nat) : Int) % ((2 ^ 32 : Nat) : Int) = (c.toNat : Int) := by
      rw [hEqInt]
      have h1 : (((2 ^ 32 : Nat) : Int) * q + (c.toNat : Int)) % ((2 ^ 32 : Nat) : Int) =
          (c.toNat : Int) % ((2 ^ 32 : Nat) : Int) := by
        simp [Int.add_emod]
      rw [h1]
      exact Int.emod_eq_of_lt (Int.natCast_nonneg _) (by omega)
    omega
  · intro hMod
    have hDiv := Int.ediv_mul_add_emod ((a.toNat + b.toNat : Nat) : Int) ((2 ^ 32 : Nat) : Int)
    have hModInt : ((a.toNat + b.toNat : Nat) : Int) % ((2 ^ 32 : Nat) : Int) = (c.toNat : Int) := by
      omega
    rw [hModInt] at hDiv
    refine ⟨((a.toNat + b.toNat : Nat) : Int) / ((2 ^ 32 : Nat) : Int), ?_⟩
    omega

private theorem pow2Int_dvd_of_le {m n : Nat} (hmn : m ≤ n) : pow2Int m ∣ pow2Int n := by
  unfold pow2Int
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hmn
  rw [Nat.pow_add]
  refine ⟨((2 ^ k : Nat) : Int), ?_⟩
  simp [Int.natCast_mul]

/-!
### 3. 定理 1：2-adic 相角塔完备性与零伪解定理 (`two_adic_zero_iff_mod_add`)
-/

/--
**定理 1（`two_adic_zero_iff_mod_add`）**：
对任意单位圆残差核 `K`（含 `cayleyCircleKernel`）及任意 32-bit 字 `a, b, c : BitVec 32`，
32 尺度 2-adic 相角塔总能量为 `0` **当且仅当** 32 位模加方程 `a + b = c` 精确成立。
（注：不存在任何非真解的零能量伪极小值陷阱。）
-/
theorem two_adic_zero_iff_mod_add (K : UnitCircleKernel) (a b c : BitVec 32) :
    towerEnergy K a b c = 0 ↔ a + b = c := by
  unfold towerEnergy
  rw [sumScales_eq_zero_iff]
  constructor
  · intro hAll
    have h32 := hAll 32 (by decide) (by decide)
    rw [scaleEnergy_eq_zero_iff_dvd] at h32
    exact (pow2Int_dvd_signedDiff_32_iff_add_eq a b c).mp h32
  · intro hAdd m hm1 hm32
    rw [scaleEnergy_eq_zero_iff_dvd]
    have h32 := (pow2Int_dvd_signedDiff_32_iff_add_eq a b c).mpr hAdd
    exact Int.dvd_trans (pow2Int_dvd_of_le hm32) h32

/-- 等价的多尺度同余刻画：`towerEnergy = 0 ↔ ∀ m ∈ [1..32], (a + b) ≡ c (mod 2^m)`。 -/
theorem two_adic_zero_iff_all_scales_mod (K : UnitCircleKernel) (a b c : BitVec 32) :
    towerEnergy K a b c = 0 ↔
      ∀ m : Nat, 1 ≤ m → m ≤ 32 → (a.toNat + b.toNat) % 2 ^ m = c.toNat % 2 ^ m := by
  rw [two_adic_zero_iff_mod_add K a b c]
  constructor
  · intro hAdd m _ hm32
    have h32 : (a.toNat + b.toNat) % 2 ^ 32 = c.toNat % 2 ^ 32 := by
      have hc : c.toNat < 2 ^ 32 := c.isLt
      rw [← BitVec.toNat_add, hAdd, Nat.mod_eq_of_lt hc]
    have hdvd : 2 ^ m ∣ 2 ^ 32 := by
      obtain ⟨k, hk⟩ := Nat.exists_eq_add_of_le hm32
      rw [hk, Nat.pow_add]
      exact Nat.dvd_mul_right _ _
    have h1 : ((a.toNat + b.toNat) % 2 ^ 32) % 2 ^ m = (c.toNat % 2 ^ 32) % 2 ^ m :=
      congrArg (· % 2 ^ m) h32
    simpa [Nat.mod_mod_of_dvd _ hdvd] using h1
  · intro hAll
    have h32 := hAll 32 (by decide) (by decide)
    have hc : c.toNat < 2 ^ 32 := c.isLt
    rw [Nat.mod_eq_of_lt hc, ← BitVec.toNat_add] at h32
    exact BitVec.toNat_eq.mpr h32

/-- 复高斯单位圆实例推论：`cayleyCircleKernel` 的 32 尺度能量归零充要于 `a + b = c`。 -/
theorem two_adic_zero_iff_mod_add_cayley (a b c : BitVec 32) :
    towerEnergy cayleyCircleKernel a b c = 0 ↔ a + b = c :=
  two_adic_zero_iff_mod_add cayleyCircleKernel a b c

/-!
### 4. 定理 2：LSB 零进位解耦与单比特翻转的 $\pi$ 相位跃迁下界 (`lsb_carry_decoupling_and_flip_lower_bound`)
-/

private theorem bv_xor_eq_add_of_and_zero (a b : BitVec 32) (h : a &&& b = 0#32) :
    (a ^^^ b).setWidth 33 = a.setWidth 33 + b.setWidth 33 := by
  bv_decide

private theorem bv_xor_cancel_32 (x m : BitVec 32) : (x ^^^ m) ^^^ m = x := by
  bv_decide

private theorem and_shift_one_eq_zero_of_getLsbD_false
    (x : BitVec 32) (k : Fin 32) (hx : x.getLsbD k.val = false) :
    x &&& (1#32 <<< k.val) = 0#32 := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  rw [BitVec.getLsbD_and, BitVec.getLsbD_shiftLeft, BitVec.getLsbD_one, BitVec.getLsbD_zero]
  by_cases hik : i = k.val
  · subst hik
    rw [hx]
    simp
  · have hFalse : (decide (i < 32) && !decide (i < k.val) &&
        (decide (0 < 32) && decide (i - k.val = 0))) = false := by
      by_cases hlt : i < k.val
      · simp [hlt]
      · have hne : i - k.val ≠ 0 := by omega
        simp [hlt, hne]
    rw [hFalse]
    simp

private theorem toNat_shift_one (k : Fin 32) : (1#32 <<< k.val).toNat = 2 ^ k.val := by
  simp only [BitVec.toNat_shiftLeft, BitVec.toNat_ofNat]
  have hPow : 1 <<< k.val = 2 ^ k.val := by simp [Nat.shiftLeft_eq]
  have hMod : (1 % 2 ^ 32) <<< k.val % 2 ^ 32 = 2 ^ k.val := by
    have h1 : 1 % 2 ^ 32 = 1 := by decide
    rw [h1, hPow]
    exact Nat.mod_eq_of_lt (Nat.pow_lt_pow_right Nat.one_lt_two k.isLt)
  exact hMod

private theorem toNat_xor_of_and_zero (a b : BitVec 32) (h : a &&& b = 0#32) :
    (a ^^^ b).toNat = a.toNat + b.toNat := by
  have hBV := congrArg BitVec.toNat (bv_xor_eq_add_of_and_zero a b h)
  rw [BitVec.toNat_setWidth, BitVec.toNat_add, BitVec.toNat_setWidth, BitVec.toNat_setWidth] at hBV
  have ha : a.toNat < 2 ^ 32 := a.isLt
  have hb : b.toNat < 2 ^ 32 := b.isLt
  have hSum : a.toNat + b.toNat < 2 ^ 33 := by omega
  have hXorLt : (a ^^^ b).toNat < 2 ^ 33 := Nat.lt_trans (a ^^^ b).isLt (by decide)
  rw [Nat.mod_eq_of_lt hXorLt, Nat.mod_eq_of_lt (Nat.lt_trans ha (by decide)),
      Nat.mod_eq_of_lt (Nat.lt_trans hb (by decide)), Nat.mod_eq_of_lt hSum] at hBV
  exact hBV

/-- 任意 `BitVec 32` 翻转第 `k` 位后整数值的精确变化 `± 2^k`。 -/
theorem bitvec_flip_toNat_diff (x : BitVec 32) (k : Fin 32) :
    ((x ^^^ (1#32 <<< k.val)).toNat : Int) - (x.toNat : Int) =
      if x.getLsbD k.val then -pow2Int k.val else pow2Int k.val := by
  have hShift := toNat_shift_one k
  cases hx : x.getLsbD k.val
  · simp only [Bool.false_eq_true, ↓reduceIte, pow2Int]
    have hAnd := and_shift_one_eq_zero_of_getLsbD_false x k hx
    have hAdd := toNat_xor_of_and_zero x (1#32 <<< k.val) hAnd
    rw [hShift] at hAdd
    omega
  · simp only [↓reduceIte, pow2Int]
    let y := x ^^^ (1#32 <<< k.val)
    have hyBit : y.getLsbD k.val = false := by
      show (x ^^^ (1#32 <<< k.val)).getLsbD k.val = false
      rw [BitVec.getLsbD_xor, hx, BitVec.getLsbD_shiftLeft, BitVec.getLsbD_one]
      simp [k.isLt]
    have hAnd := and_shift_one_eq_zero_of_getLsbD_false y k hyBit
    have hAdd := toNat_xor_of_and_zero y (1#32 <<< k.val) hAnd
    have hCancel : y ^^^ (1#32 <<< k.val) = x := bv_xor_cancel_32 x (1#32 <<< k.val)
    rw [hCancel, hShift] at hAdd
    change (y.toNat : Int) - (x.toNat : Int) = -((2 ^ k.val : Nat) : Int)
    omega

private theorem getLsbD_zero_eq_mod2 (x : BitVec 32) :
    (x.toNat : Int) % 2 = if x.getLsbD 0 then 1 else 0 := by
  have hLsb : x.getLsbD 0 = decide (x.toNat % 2 = 1) := by
    simp [BitVec.getLsbD, Nat.testBit_zero]
  rw [hLsb]
  rcases Nat.mod_two_eq_zero_or_one x.toNat with h0 | h1
  · simp [h0]; omega
  · simp [h1]; omega

private theorem signedDiff_mod2_odd
    {a b c : BitVec 32} (ha : a.getLsbD 0 = xa) (hb : b.getLsbD 0 = xb) (hc : c.getLsbD 0 = xc)
    (hXor : (xa ^^ xb ^^ xc) = true) :
    signedDiff a b c % 2 = 1 := by
  have haM := getLsbD_zero_eq_mod2 a
  have hbM := getLsbD_zero_eq_mod2 b
  have hcM := getLsbD_zero_eq_mod2 c
  rw [ha] at haM
  rw [hb] at hbM
  rw [hc] at hcM
  have hMod : signedDiff a b c % 2 =
      (((a.toNat : Int) % 2 + (b.toNat : Int) % 2 - (c.toNat : Int) % 2) % 2) := by
    unfold signedDiff
    rw [Int.sub_emod, Int.add_emod]
    omega
  rw [hMod, haM, hbM, hcM]
  cases xa <;> cases xb <;> cases xc <;> simp_all

private theorem signedDiff_mod2_even
    {a b c : BitVec 32} (ha : a.getLsbD 0 = xa) (hb : b.getLsbD 0 = xb) (hc : c.getLsbD 0 = xc)
    (hXor : (xa ^^ xb ^^ xc) = false) :
    signedDiff a b c % 2 = 0 := by
  have haM := getLsbD_zero_eq_mod2 a
  have hbM := getLsbD_zero_eq_mod2 b
  have hcM := getLsbD_zero_eq_mod2 c
  rw [ha] at haM
  rw [hb] at hbM
  rw [hc] at hcM
  have hMod : signedDiff a b c % 2 =
      (((a.toNat : Int) % 2 + (b.toNat : Int) % 2 - (c.toNat : Int) % 2) % 2) := by
    unfold signedDiff
    rw [Int.sub_emod, Int.add_emod]
    omega
  rw [hMod, haM, hbM, hcM]
  cases xa <;> cases xb <;> cases xc <;> simp_all

private theorem scalePhaseFrac_shift_half_turn (K : UnitCircleKernel)
    (num : Int) (m : Nat) (q : Int)
    (hNum : num = q * pow2Int (m + 1) + pow2Int m) :
    ((num : Rat) / (pow2Int (m + 1) : Rat) = (q : Rat) + 1 / 2) ∧
    (K.energy ((num : Rat) / (pow2Int (m + 1) : Rat)) = 4) ∧
    (cayleyUnitPoint ((num : Rat) / (pow2Int (m + 1) : Rat)) = ⟨-1, 0⟩) := by
  have hPow2 : pow2Int (m + 1) = 2 * pow2Int m := by
    unfold pow2Int
    rw [Nat.pow_succ']
    simp [Int.natCast_mul]
  have hmPos : (0 : Int) < pow2Int m := pow2Int_pos m
  have hm1Pos : (0 : Int) < pow2Int (m + 1) := pow2Int_pos (m + 1)
  have hmRatPos : (0 : Rat) < (pow2Int m : Rat) := by
    rw [Rat.lt_iff]; change 0 * 1 < pow2Int m * 1; omega
  have hm1RatPos : (0 : Rat) < (pow2Int (m + 1) : Rat) := by
    rw [Rat.lt_iff]; change 0 * 1 < pow2Int (m + 1) * 1; omega
  have hmNe : (pow2Int m : Rat) ≠ 0 := by grind
  have hm1Ne : (pow2Int (m + 1) : Rat) ≠ 0 := by grind
  have hPhase : (num : Rat) / (pow2Int (m + 1) : Rat) = (q : Rat) + 1 / 2 := by
    rw [hNum, Rat.intCast_add, Rat.intCast_mul, ← rat_div_add_div_same,
        Rat.mul_div_cancel hm1Ne]
    have hHalf : (pow2Int m : Rat) / (pow2Int (m + 1) : Rat) = 1 / 2 := by
      rw [hPow2, Rat.intCast_mul]
      have hTwo : ((2 : Int) : Rat) = 2 := rfl
      rw [hTwo, Rat.div_def, Rat.div_def]
      have hInv : (2 * (pow2Int m : Rat))⁻¹ = (pow2Int m : Rat)⁻¹ * 2⁻¹ := by
        have hProd : (2 * (pow2Int m : Rat)) * ((pow2Int m : Rat)⁻¹ * 2⁻¹) = 1 := by
          have h1 : (2 * (pow2Int m : Rat)) * ((pow2Int m : Rat)⁻¹ * 2⁻¹) =
              2 * ((pow2Int m : Rat) * (pow2Int m : Rat)⁻¹) * 2⁻¹ := by grind
          rw [h1, Rat.mul_inv_cancel _ hmNe]
          have h2 : (2 : Rat) * 1 * 2⁻¹ = 2 * 2⁻¹ := by grind
          rw [h2, Rat.mul_inv_cancel 2 (by grind)]
        have hL := congrArg (fun z => (2 * (pow2Int m : Rat))⁻¹ * z) hProd
        have h2mNe : (2 : Rat) * (pow2Int m : Rat) ≠ 0 := by grind
        have hAssoc : (2 * (pow2Int m : Rat))⁻¹ *
            ((2 * (pow2Int m : Rat)) * ((pow2Int m : Rat)⁻¹ * 2⁻¹)) =
            ((2 * (pow2Int m : Rat)) * (2 * (pow2Int m : Rat))⁻¹) * ((pow2Int m : Rat)⁻¹ * 2⁻¹) := by grind
        rw [hAssoc, Rat.mul_inv_cancel _ h2mNe] at hL
        grind
      rw [hInv]
      have h1 : (pow2Int m : Rat) * ((pow2Int m : Rat)⁻¹ * 2⁻¹) =
          ((pow2Int m : Rat) * (pow2Int m : Rat)⁻¹) * 2⁻¹ := by grind
      rw [h1, Rat.mul_inv_cancel _ hmNe]
    rw [hHalf]
  have hAddComm : (q : Rat) + 1 / 2 = (1 / 2 : Rat) + (q : Rat) := by grind
  refine ⟨hPhase, ?_, ?_⟩
  · rw [hPhase, hAddComm, K.periodic, K.half_turn]
  · rw [hPhase, hAddComm, cayleyUnitPoint_add_int, cayleyUnitPoint_half_turn]

/--
**定理 2（`lsb_carry_decoupling_and_flip_lower_bound`）**：
1. **LSB 零进位解耦**：第 0 位（尺度 `m = 1`）能量残差严格仅由 `a[0] ⊕ b[0] ⊕ c[0]` 决定，
   与任何高位比特及进位链完全无关；
2. **合法解单比特翻转的 $\pi$ 相位反转与能量跃迁下界 4**：
   对任意满足 `a + b = c` 的合法解及任意比特位置 `k : Fin 32`，
   翻转 `a` 的第 `k` 位得到 `a' := a ^^^ (1#32 <<< k.val)` 后：
   - 尺度 `m = k + 1` 的标准化相角精确落于 `ℤ + 1/2`（$\pi$ 相位反转），且在 `cayleyUnitPoint`
     复单位圆上精确跳转至 `-1`；
   - 尺度 `m = k + 1` 的局部能量精确跃迁至最大值 `4`；
   - 所有更低尺度 `1 ≤ m ≤ k` 仍保持 `0` 残差；
   - 全 32 尺度相角塔总能量必满足严格下界 `4 ≤ towerEnergy K a' b c`（彻底消除梯度消失）。
-/
theorem lsb_carry_decoupling_and_flip_lower_bound (K : UnitCircleKernel)
    (a b c : BitVec 32) (k : Fin 32) :
    (scaleEnergy K a b c 1 =
      if (a.getLsbD 0 ^^ b.getLsbD 0 ^^ c.getLsbD 0) then 4 else 0) ∧
    (a + b = c →
      let a' := a ^^^ (1#32 <<< k.val)
      (∃ q : Int, scalePhaseFrac a' b c (k.val + 1) = (q : Rat) + 1 / 2) ∧
      cayleyUnitPoint (scalePhaseFrac a' b c (k.val + 1)) = ⟨-1, 0⟩ ∧
      scaleEnergy K a' b c (k.val + 1) = 4 ∧
      (∀ m : Nat, 1 ≤ m → m ≤ k.val → scaleEnergy K a' b c m = 0) ∧
      4 ≤ towerEnergy K a' b c) := by
  constructor
  · -- Part 1: LSB (m = 1) 零进位解耦
    have hDiv := Int.ediv_mul_add_emod (signedDiff a b c) 2
    by_cases hXor : (a.getLsbD 0 ^^ b.getLsbD 0 ^^ c.getLsbD 0) = true
    · simp only [hXor, ↓reduceIte]
      have hOdd := signedDiff_mod2_odd rfl rfl rfl hXor
      have hNum : signedDiff a b c = (signedDiff a b c / 2) * pow2Int 1 + pow2Int 0 := by
        have hp1 : pow2Int 1 = 2 := rfl
        have hp0 : pow2Int 0 = 1 := rfl
        rw [hp1, hp0]
        omega
      exact (scalePhaseFrac_shift_half_turn K (signedDiff a b c) 0 (signedDiff a b c / 2) hNum).2.1
    · have hXorF : (a.getLsbD 0 ^^ b.getLsbD 0 ^^ c.getLsbD 0) = false := by
        cases h0 : (a.getLsbD 0 ^^ b.getLsbD 0 ^^ c.getLsbD 0)
        · rfl
        · contradiction
      simp only [hXorF, Bool.false_eq_true, ↓reduceIte]
      have hEven := signedDiff_mod2_even rfl rfl rfl hXorF
      have hDvd : pow2Int 1 ∣ signedDiff a b c := by
        refine ⟨signedDiff a b c / 2, ?_⟩
        have hp1 : pow2Int 1 = 2 := rfl
        rw [hp1]
        omega
      exact (scaleEnergy_eq_zero_iff_dvd K a b c 1).mpr hDvd
  · -- Part 2: 任意合法解单比特翻转的 π 相位反转与下界 4
    intro hAdd
    dsimp only
    have h32 : pow2Int 32 ∣ signedDiff a b c :=
      (pow2Int_dvd_signedDiff_32_iff_add_eq a b c).mpr hAdd
    obtain ⟨t, ht⟩ := h32
    have hDiffFlip := bitvec_flip_toNat_diff a k
    have hSignedDiff' : signedDiff (a ^^^ (1#32 <<< k.val)) b c =
        signedDiff a b c + (if a.getLsbD k.val then -pow2Int k.val else pow2Int k.val) := by
      unfold signedDiff
      omega
    have hkLe32 : k.val + 1 ≤ 32 := k.isLt
    obtain ⟨r, hr⟩ := pow2Int_dvd_of_le hkLe32
    have hPow2k : pow2Int (k.val + 1) = 2 * pow2Int k.val := by
      unfold pow2Int
      rw [Nat.pow_succ']
      simp [Int.natCast_mul]
    have hNumSplit : ∃ q : Int,
        signedDiff (a ^^^ (1#32 <<< k.val)) b c = q * pow2Int (k.val + 1) + pow2Int k.val := by
      rw [hSignedDiff', ht, hr]
      cases a.getLsbD k.val
      · simp only [Bool.false_eq_true, ↓reduceIte]
        refine ⟨r * t, ?_⟩
        grind
      · simp only [↓reduceIte]
        refine ⟨r * t - 1, ?_⟩
        rw [Int.sub_mul, Int.one_mul, hPow2k]
        grind
    obtain ⟨q, hq⟩ := hNumSplit
    have hPi := scalePhaseFrac_shift_half_turn K (signedDiff (a ^^^ (1#32 <<< k.val)) b c) k.val q hq
    have hLowZero : ∀ m : Nat, 1 ≤ m → m ≤ k.val →
        scaleEnergy K (a ^^^ (1#32 <<< k.val)) b c m = 0 := by
      intro m _ hmk
      rw [scaleEnergy_eq_zero_iff_dvd]
      have hmDvdK : pow2Int m ∣ pow2Int k.val := pow2Int_dvd_of_le hmk
      have hmDvdK1 : pow2Int m ∣ pow2Int (k.val + 1) := pow2Int_dvd_of_le (Nat.le_succ_of_le hmk)
      rw [hq]
      exact Int.dvd_add (Int.dvd_trans hmDvdK1 (Int.dvd_mul_left _ _)) hmDvdK
    have hTower : 4 ≤ towerEnergy K (a ^^^ (1#32 <<< k.val)) b c := by
      have hLe := scaleEnergy_le_sumScales K (a ^^^ (1#32 <<< k.val)) b c (Nat.succ_pos k.val) hkLe32
      have hE4 : scaleEnergy K (a ^^^ (1#32 <<< k.val)) b c (k.val + 1) = 4 := hPi.2.1
      unfold towerEnergy
      grind
    refine ⟨⟨q, hPi.1⟩, hPi.2.2, hPi.2.1, hLowZero, hTower⟩

/-- 对称推论：翻转 `b` 或 `c` 的任意单比特同样触发精确 $\pi$ 相位反转与总能量下界 `4`。 -/
theorem lsb_flip_b_c_lower_bound (K : UnitCircleKernel)
    (a b c : BitVec 32) (k : Fin 32) (hAdd : a + b = c) :
    (scaleEnergy K a (b ^^^ (1#32 <<< k.val)) c (k.val + 1) = 4 ∧
      4 ≤ towerEnergy K a (b ^^^ (1#32 <<< k.val)) c) ∧
    (scaleEnergy K a b (c ^^^ (1#32 <<< k.val)) (k.val + 1) = 4 ∧
      4 ≤ towerEnergy K a b (c ^^^ (1#32 <<< k.val))) := by
  have hComm : b + a = c := by rw [BitVec.add_comm]; exact hAdd
  have hb := (lsb_carry_decoupling_and_flip_lower_bound K b a c k).2 hComm
  have hSwapB : ∀ m, scalePhaseFrac a (b ^^^ (1#32 <<< k.val)) c m =
      scalePhaseFrac (b ^^^ (1#32 <<< k.val)) a c m := by
    intro m
    simp only [scalePhaseFrac, signedDiff]
    have hS : ((a.toNat : Int) + ((b ^^^ (1#32 <<< k.val)).toNat : Int) - (c.toNat : Int)) =
        (((b ^^^ (1#32 <<< k.val)).toNat : Int) + (a.toNat : Int) - (c.toNat : Int)) := by omega
    rw [hS]
  have hSwapTowerB : ∀ n, sumScales K a (b ^^^ (1#32 <<< k.val)) c n =
      sumScales K (b ^^^ (1#32 <<< k.val)) a c n := by
    intro n
    induction n with
    | zero => rfl
    | succ m ih => simp only [sumScales, scaleEnergy, ih, hSwapB]
  have hbE : scaleEnergy K a (b ^^^ (1#32 <<< k.val)) c (k.val + 1) = 4 := by
    simpa [scaleEnergy, hSwapB] using hb.2.2.1
  have hbT : 4 ≤ towerEnergy K a (b ^^^ (1#32 <<< k.val)) c := by
    simpa [towerEnergy, hSwapTowerB] using hb.2.2.2.2
  -- 翻转 c 的对称情况
  have h32 : pow2Int 32 ∣ signedDiff a b c :=
    (pow2Int_dvd_signedDiff_32_iff_add_eq a b c).mpr hAdd
  obtain ⟨t, ht⟩ := h32
  have hDiffFlip := bitvec_flip_toNat_diff c k
  have hSignedDiffC : signedDiff a b (c ^^^ (1#32 <<< k.val)) =
      signedDiff a b c - (if c.getLsbD k.val then -pow2Int k.val else pow2Int k.val) := by
    unfold signedDiff
    omega
  have hkLe32 : k.val + 1 ≤ 32 := k.isLt
  obtain ⟨r, hr⟩ := pow2Int_dvd_of_le hkLe32
  have hPow2k : pow2Int (k.val + 1) = 2 * pow2Int k.val := by
    unfold pow2Int; rw [Nat.pow_succ']; simp [Int.natCast_mul]
  have hNumSplitC : ∃ q : Int,
      signedDiff a b (c ^^^ (1#32 <<< k.val)) = q * pow2Int (k.val + 1) + pow2Int k.val := by
    rw [hSignedDiffC, ht, hr]
    cases c.getLsbD k.val
    · simp only [Bool.false_eq_true, ↓reduceIte]
      refine ⟨r * t - 1, ?_⟩
      rw [Int.sub_mul, Int.one_mul, hPow2k]
      grind
    · simp only [↓reduceIte]
      refine ⟨r * t, ?_⟩
      grind
  obtain ⟨qc, hqc⟩ := hNumSplitC
  have hPiC := scalePhaseFrac_shift_half_turn K (signedDiff a b (c ^^^ (1#32 <<< k.val))) k.val qc hqc
  have hcE : scaleEnergy K a b (c ^^^ (1#32 <<< k.val)) (k.val + 1) = 4 := hPiC.2.1
  have hcT : 4 ≤ towerEnergy K a b (c ^^^ (1#32 <<< k.val)) := by
    have hLe := scaleEnergy_le_sumScales K a b (c ^^^ (1#32 <<< k.val)) (Nat.succ_pos k.val) hkLe32
    unfold towerEnergy
    grind
  exact ⟨⟨hbE, hbT⟩, ⟨hcE, hcT⟩⟩

end PhaseSHA.Algebraic.TwoAdicTowerSoundness
