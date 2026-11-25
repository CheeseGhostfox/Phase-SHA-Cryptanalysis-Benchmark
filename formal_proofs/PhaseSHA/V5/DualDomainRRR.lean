import PhaseSHA.Algebraic.LiteField

/-!
# Module 6D: 双频域分裂与差分映射无假坑不动点定理 (`DualDomainRRR.lean`, Walsh-Hadamard ↔ 2-adic + RRR)

本模块在零外部依赖、无公理向量空间与构造性有限域算术上严格验证 V5/V6 “双世界反射-反射-弛豫
（Elser RRR）+ Walsh-Hadamard 频域解耦”的两项底层理论基石及诚实边界条件：
1. **`rrr_fixed_point_iff_intersection`（无需任何凸性假设）**：
   在抽象有理向量空间 `RatVecSpace X` 及一般非凸投影对 `ProjectionPair X`（仅要求投影落入约束集与集合内不变性）上，
   对任意非零步长 `β ≠ 0`，证明 RRR 迭代算子 `rrrStep V P β x := x + β(P_B(2P_A(x) - x) - P_A(x))` 满足严格双向定理：
   - **无假坑停机**：`rrrStep x = x → P_A(x) = P_B(2P_A(x) - x) ∈ A ∩ B`（若无公共解则算子不存在任何不动点）；
   - **真解保持**：`y ∈ A ∩ B → rrrStep y = y`；
   - Concrete instance：`vecSpace n`（`Fin n → Rat`）。
2. **`rrr_boundary_caveats`（诚实边界与反例）**：
   显式证出三条在架构设计时必须遵守的诚实边界：
   - `β = 0` 退化反例：即使 `A ∩ B = ∅`，`β = 0` 仍使任意点沦为伪不动点；
   - **阴影点 vs 迭代点分离反例**：若 `rrrStep x = x`，进入 `A ∩ B` 的是阴影 `P_A(x)` 而未必是迭代变量 `x` 本身；
   - **不可满足实例漂移反例**：当 `A = {0}, B = {1}` 无交集且 `β ≠ 0` 时，`rrrIter k x` 呈无界线性漂移 `x_0 + kβ`。
3. **`walsh_hadamard_involutive_and_xor_conv`**：
   对任意比特宽度 `n : Nat`，构造性定义 `n` 比特函数上的 Walsh-Hadamard 变换 `wht` 与 XOR 卷积 `xorConv`，
   并证明精确点积特征等价性（`walshChar = (-1)^{⟨u,x⟩₂}`）、精确对合逆变换（`wht (wht f) v = 2^n f v`）
   以及 XOR 卷积逐点对角化定理（`wht (xorConv f g) u = wht f u * wht g u`）。
-/

namespace PhaseSHA.V5.DualDomainRRR

open PhaseSHA.Algebraic.LiteField

/-!
### 1. 零公理有理向量空间与 Elser RRR 算子双向无假坑定理
-/

/-- 轻量零公理 `ℚ`-线性向量空间结构。 -/
structure RatVecSpace (X : Type) where
  zero : X
  add : X → X → X
  sub : X → X → X
  smul : Rat → X → X
  add_zero : ∀ x : X, add x zero = x
  sub_self : ∀ x : X, sub x x = zero
  two_smul_sub_self : ∀ x : X, sub (smul 2 x) x = x
  add_cancel_left : ∀ x y z : X, add x y = add x z → y = z
  smul_zero : ∀ r : Rat, smul r zero = zero
  smul_eq_zero_iff : ∀ r : Rat, ∀ x : X, smul r x = zero ↔ r = 0 ∨ x = zero
  sub_eq_zero_iff : ∀ x y : X, sub x y = zero ↔ x = y

/-- 具体 `n` 维有理向量空间实例 `Fin n → Rat`。 -/
def vecSpace (n : Nat) : RatVecSpace (Fin n → Rat) where
  zero := fun _ => 0
  add := fun u v i => u i + v i
  sub := fun u v i => u i - v i
  smul := fun r u i => r * u i
  add_zero := by intro x; funext i; grind
  sub_self := by intro x; funext i; grind
  two_smul_sub_self := by intro x; funext i; grind
  add_cancel_left := by
    intro x y z h
    funext i
    have hi := congrFun h i
    grind
  smul_zero := by intro r; funext i; grind
  smul_eq_zero_iff := by
    intro r x
    constructor
    · intro h
      by_cases hr : r = 0
      · exact Or.inl hr
      · right
        funext i
        have hi := congrFun h i
        rcases rat_mul_eq_zero_iff.mp hi with h1 | h2
        · contradiction
        · exact h2
    · intro h
      rcases h with rfl | rfl <;> funext i <;> grind
  sub_eq_zero_iff := by
    intro x y
    constructor
    · intro h; funext i; have hi := congrFun h i; grind
    · intro h; subst h; funext i; grind

/-- 一般（无需凸性）双约束流形及投影算子对 `(A, B, P_A, P_B)`。 -/
structure ProjectionPair (X : Type) where
  A : X → Prop
  B : X → Prop
  PA : X → X
  PB : X → X
  PA_mem : ∀ x : X, A (PA x)
  PB_mem : ∀ x : X, B (PB x)
  PA_fix : ∀ x : X, A x → PA x = x
  PB_fix : ∀ x : X, B x → PB x = x

/-- Elser 反射-反射-弛豫（RRR / Difference Map）单步算子：
`T_RRR(x) := x + β · (P_B(2 P_A(x) - x) - P_A(x))`。 -/
def rrrStep {X : Type} (V : RatVecSpace X) (P : ProjectionPair X) (beta : Rat) (x : X) : X :=
  V.add x (V.smul beta (V.sub (P.PB (V.sub (V.smul 2 (P.PA x)) x)) (P.PA x)))

/--
**定理 1（`rrr_fixed_point_iff_intersection`）**：
在任意向量空间与任意（一般非凸）约束集合 `A, B` 上，对一切非零步长 `β ≠ 0`：
1. **无假坑不动点**：若 `rrrStep V P β x = x`，则阴影点与反射投影重合且落入真解交集
   `P.PA x = P.PB (2 P.PA x - x) ∈ A ∩ B`；
2. **真解保持**：若 `y ∈ A ∩ B`，则 `rrrStep V P β y = y`；
3. **无交集推论**：若 `A ∩ B = ∅`，则算子在全空间绝不存在任何不动点。
-/
theorem rrr_fixed_point_iff_intersection {X : Type} (V : RatVecSpace X)
    (P : ProjectionPair X) (beta : Rat) (hBeta : beta ≠ 0) :
    (∀ x : X, rrrStep V P beta x = x →
      P.PA x = P.PB (V.sub (V.smul 2 (P.PA x)) x) ∧ P.A (P.PA x) ∧ P.B (P.PA x)) ∧
    (∀ y : X, P.A y ∧ P.B y → rrrStep V P beta y = y) ∧
    ((∀ y : X, ¬ (P.A y ∧ P.B y)) → ∀ x : X, rrrStep V P beta x ≠ x) := by
  have hForward : ∀ x : X, rrrStep V P beta x = x →
      P.PA x = P.PB (V.sub (V.smul 2 (P.PA x)) x) ∧ P.A (P.PA x) ∧ P.B (P.PA x) := by
    intro x hFix
    unfold rrrStep at hFix
    have hEqZero :
        V.add x (V.smul beta (V.sub (P.PB (V.sub (V.smul 2 (P.PA x)) x)) (P.PA x))) =
        V.add x V.zero := hFix.trans (V.add_zero x).symm
    have hSmulZero := V.add_cancel_left _ _ _ hEqZero
    rcases (V.smul_eq_zero_iff beta _).mp hSmulZero with hB0 | hSub0
    · exfalso; exact hBeta hB0
    · have hEq := (V.sub_eq_zero_iff _ _).mp hSub0
      refine ⟨hEq.symm, P.PA_mem x, ?_⟩
      rw [← hEq]
      exact P.PB_mem _
  refine ⟨hForward, ?_, ?_⟩
  · intro y ⟨hA, hB⟩
    unfold rrrStep
    rw [P.PA_fix y hA, V.two_smul_sub_self y, P.PB_fix y hB, V.sub_self, V.smul_zero, V.add_zero]
  · intro hDisj x hFix
    rcases hForward x hFix with ⟨_, hA, hB⟩
    exact hDisj (P.PA x) ⟨hA, hB⟩

/-!
### 2. 诚实边界与反例 (`rrr_boundary_caveats`)
-/

/-- 1 维常向量构造。 -/
def ptVec (c : Rat) : Fin 1 → Rat := fun _ => c

/-- 双重零点约束 `A = {0}, B = {0}`（交集为 `{0}`）。 -/
def pairZeroZero : ProjectionPair (Fin 1 → Rat) where
  A := fun x => x 0 = 0
  B := fun x => x 0 = 0
  PA := fun _ => ptVec 0
  PB := fun _ => ptVec 0
  PA_mem := fun _ => rfl
  PB_mem := fun _ => rfl
  PA_fix := by
    intro x hx
    funext i
    rcases i with ⟨v, hv⟩
    have hv0 : v = 0 := by omega
    subst hv0
    exact hx.symm
  PB_fix := by
    intro x hx
    funext i
    rcases i with ⟨v, hv⟩
    have hv0 : v = 0 := by omega
    subst hv0
    exact hx.symm

/-- 不相交约束 `A = {0}, B = {1}`（无共同解）。 -/
def pairZeroOne : ProjectionPair (Fin 1 → Rat) where
  A := fun x => x 0 = 0
  B := fun x => x 0 = 1
  PA := fun _ => ptVec 0
  PB := fun _ => ptVec 1
  PA_mem := fun _ => rfl
  PB_mem := fun _ => rfl
  PA_fix := by
    intro x hx
    funext i
    rcases i with ⟨v, hv⟩
    have hv0 : v = 0 := by omega
    subst hv0
    exact hx.symm
  PB_fix := by
    intro x hx
    funext i
    rcases i with ⟨v, hv⟩
    have hv0 : v = 0 := by omega
    subst hv0
    exact hx.symm

/-- RRR `k` 步递归迭代序列。 -/
def rrrIter {X : Type} (V : RatVecSpace X) (P : ProjectionPair X) (beta : Rat) :
    Nat → X → X
  | 0, x => x
  | k + 1, x => rrrStep V P beta (rrrIter V P beta k x)

/-- 自然数嵌入 `Rat`。 -/
def natToRat : Nat → Rat
  | 0 => 0
  | n + 1 => natToRat n + 1

private theorem natToRat_pos {n : Nat} (hn : 0 < n) : 0 < natToRat n := by
  induction n with
  | zero => omega
  | succ k _ =>
    have hNonneg : ∀ m : Nat, 0 ≤ natToRat m := by
      intro m
      induction m with
      | zero => unfold natToRat; grind
      | succ m' ih' => unfold natToRat; grind
    unfold natToRat
    have hk := hNonneg k
    grind

private theorem pairZeroOne_step (beta : Rat) (y : Fin 1 → Rat) :
    rrrStep (vecSpace 1) pairZeroOne beta y 0 = y 0 + beta := by
  simp [rrrStep, vecSpace, pairZeroOne, ptVec]
  grind

/--
**定理 2（`rrr_boundary_caveats`，诚实边界与反例）**：
1. **`β = 0` 退化伪不动点**：在无交集实例 `pairZeroOne`（`A = {0}, B = {1}`）上，
   `β = 0` 使空间中**所有点**皆为不动点；
2. **阴影点 vs 迭代点分离**：即使交集非空（`pairZeroZero`, `β = 1`），不动点 `x = ptVec 1`
   本身并不属于 `A ∩ B`（`¬ A x ∧ ¬ B x`），真解是阴影映射 `PA (ptVec 1) = ptVec 0 ∈ A ∩ B`；
3. **不可满足线性漂移**：在无交集实例 `pairZeroOne` 上，对任意 `β ≠ 0` 与步数 `k : Nat`，
   `rrrIter` 精确遵循 `x_k(0) = x_0(0) + k · β` 无界漂移，且对任意 `k ≥ 1` 均有 `x_k ≠ x_0`。
-/
theorem rrr_boundary_caveats :
    (∀ y : Fin 1 → Rat, ¬ (pairZeroOne.A y ∧ pairZeroOne.B y)) ∧
    (∀ x : Fin 1 → Rat, rrrStep (vecSpace 1) pairZeroOne 0 x = x) ∧
    (rrrStep (vecSpace 1) pairZeroZero 1 (ptVec 1) = ptVec 1 ∧
      ¬ pairZeroZero.A (ptVec 1) ∧ ¬ pairZeroZero.B (ptVec 1) ∧
      pairZeroZero.A (pairZeroZero.PA (ptVec 1)) ∧ pairZeroZero.B (pairZeroZero.PA (ptVec 1))) ∧
    (∀ (beta : Rat) (k : Nat) (x : Fin 1 → Rat),
      rrrIter (vecSpace 1) pairZeroOne beta k x 0 = x 0 + natToRat k * beta ∧
      (beta ≠ 0 → 1 ≤ k → rrrIter (vecSpace 1) pairZeroOne beta k x ≠ x)) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro y ⟨hA, hB⟩
    change y 0 = 0 at hA
    change y 0 = 1 at hB
    grind
  · intro x
    funext i
    simp [rrrStep, vecSpace, pairZeroOne, ptVec]
    try grind
  · refine ⟨?_, ?_, ?_, pairZeroZero.PA_mem _, ?_⟩
    · funext i
      simp [rrrStep, vecSpace, pairZeroZero, ptVec]
      try grind
    · simp [pairZeroZero, ptVec]; try grind
    · simp [pairZeroZero, ptVec]; try grind
    · simp [pairZeroZero, ptVec]; try grind
  · intro beta k x
    have hFormula : ∀ m : Nat, rrrIter (vecSpace 1) pairZeroOne beta m x 0 =
        x 0 + natToRat m * beta := by
      intro m
      induction m with
      | zero => simp [rrrIter, natToRat]; grind
      | succ j ih =>
        have hStep : rrrIter (vecSpace 1) pairZeroOne beta (j + 1) x =
            rrrStep (vecSpace 1) pairZeroOne beta (rrrIter (vecSpace 1) pairZeroOne beta j x) := rfl
        rw [hStep, pairZeroOne_step beta (rrrIter (vecSpace 1) pairZeroOne beta j x), ih]
        change x 0 + natToRat j * beta + beta = x 0 + (natToRat j + 1) * beta
        grind
    refine ⟨hFormula k, ?_⟩
    intro hBeta h1k hEq
    have h0 := congrArg (fun v => v 0) hEq
    rw [hFormula k] at h0
    have hMul0 : natToRat k * beta = 0 := by grind
    rcases rat_mul_eq_zero_iff.mp hMul0 with hk0 | hB0
    · have hkPos := natToRat_pos h1k
      grind
    · exact hBeta hB0

/-!
### 3. 任意宽度 `n` 的 Walsh-Hadamard 变换精确对合与 XOR 卷积对角化 (`walsh_hadamard_involutive_and_xor_conv`)
-/

/-- 有限宽度 `n` 布尔向量。 -/
abbrev Bits (n : Nat) := Fin n → Bool

/-- 首部拼接：`b :: t`。 -/
def bitsCons {n : Nat} (b : Bool) (t : Bits n) : Bits (n + 1)
  | ⟨0, _⟩ => b
  | ⟨k + 1, h⟩ => t ⟨k, Nat.lt_of_succ_lt_succ h⟩

/-- 取尾部 `x[1..n]`。 -/
def bitsTail {n : Nat} (x : Bits (n + 1)) : Bits n :=
  fun i => x i.succ

/-- 逐位按位异或 `x ⊕ y`。 -/
def bitsXor {n : Nat} (x y : Bits n) : Bits n :=
  fun i => x i ^^ y i

/-- 自旋编码：`false ↦ +1`, `true ↦ -1`。 -/
def boolSign (b : Bool) : Rat := if b then -1 else 1

/-- 布尔向量全体 `{false, true}^n` 上的求和算子。 -/
def sumBits : {n : Nat} → (Bits n → Rat) → Rat
  | 0, f => f (fun i => i.elim0)
  | _ + 1, f => sumBits (fun t => f (bitsCons false t)) + sumBits (fun t => f (bitsCons true t))

/-- 模 2 布尔内积 `⟨u, x⟩₂`。 -/
def dotMod2 : {n : Nat} → Bits n → Bits n → Bool
  | 0, _, _ => false
  | _ + 1, u, x => (u 0 && x 0) ^^ dotMod2 (bitsTail u) (bitsTail x)

/-- Walsh-Hadamard 核函数 `χ_u(x) = (-1)^{⟨u, x⟩₂}`。 -/
def walshChar : {n : Nat} → Bits n → Bits n → Rat
  | 0, _, _ => 1
  | _ + 1, u, x =>
    (if u 0 && x 0 then (-1 : Rat) else 1) * walshChar (bitsTail u) (bitsTail x)

/-- `n` 比特函数上的 Walsh-Hadamard 变换 `f̂(u) = ∑_x (-1)^{⟨u, x⟩₂} f(x)`。 -/
def wht {n : Nat} (f : Bits n → Rat) (u : Bits n) : Rat :=
  sumBits (fun x => walshChar u x * f x)

/-- `n` 比特布尔群上的 XOR 卷积 `(f *⊕ g)(x) = ∑_y f(y) g(x ⊕ y)`。 -/
def xorConv {n : Nat} (f g : Bits n → Rat) (x : Bits n) : Rat :=
  sumBits (fun y => f y * g (bitsXor x y))

private theorem bitsTail_bitsCons {n : Nat} (b : Bool) (t : Bits n) :
    bitsTail (bitsCons b t) = t := by
  funext i; rfl

private theorem bitsCons_zero_tail {n : Nat} (x : Bits (n + 1)) :
    bitsCons (x 0) (bitsTail x) = x := by
  funext i
  rcases i with ⟨v, hv⟩
  cases v with
  | zero => rfl
  | succ k => rfl

private theorem bitsXor_bitsCons {n : Nat} (a b : Bool) (t s : Bits n) :
    bitsXor (bitsCons a t) (bitsCons b s) = bitsCons (a ^^ b) (bitsXor t s) := by
  funext i
  rcases i with ⟨v, hv⟩
  cases v with
  | zero => rfl
  | succ k => rfl

private theorem bitsTail_bitsXor {n : Nat} (x y : Bits (n + 1)) :
    bitsTail (bitsXor x y) = bitsXor (bitsTail x) (bitsTail y) := by
  funext i; rfl

private theorem bitsXor_xor_cancel {n : Nat} (x y : Bits n) :
    bitsXor (bitsXor x y) y = x := by
  funext i
  simp [bitsXor]

theorem walshChar_eq_boolSign_dotMod2 {n : Nat} (u x : Bits n) :
    walshChar u x = boolSign (dotMod2 u x) := by
  induction n with
  | zero => simp [walshChar, dotMod2, boolSign]
  | succ k ih =>
    simp only [walshChar, dotMod2]
    rw [ih (bitsTail u) (bitsTail x)]
    cases u 0 && x 0 <;> cases dotMod2 (bitsTail u) (bitsTail x) <;>
      simp [boolSign] <;> try grind

private theorem walshChar_bitsCons {n : Nat} (a b : Bool) (u t : Bits n) :
    walshChar (bitsCons a u) (bitsCons b t) =
      (if a && b then (-1 : Rat) else 1) * walshChar u t := by
  have ha : bitsCons a u 0 = a := rfl
  have hb : bitsCons b t 0 = b := rfl
  rw [walshChar, ha, hb, bitsTail_bitsCons, bitsTail_bitsCons]

private theorem walshChar_comm {n : Nat} (u x : Bits n) :
    walshChar u x = walshChar x u := by
  induction n with
  | zero => rfl
  | succ k ih =>
    simp only [walshChar]
    rw [Bool.and_comm (u 0) (x 0), ih (bitsTail u) (bitsTail x)]

private theorem bool_xor_sign (a b c : Bool) :
    (if a && (b ^^ c) then (-1 : Rat) else 1) =
    (if a && b then (-1 : Rat) else 1) * (if a && c then (-1 : Rat) else 1) := by
  cases a <;> cases b <;> cases c <;> simp <;> try grind

private theorem walshChar_xor_right {n : Nat} (u x y : Bits n) :
    walshChar u (bitsXor x y) = walshChar u x * walshChar u y := by
  induction n with
  | zero => simp [walshChar]
  | succ k ih =>
    simp only [walshChar, bitsTail_bitsXor]
    rw [ih (bitsTail u) (bitsTail x) (bitsTail y)]
    have h0Eq : (if u 0 && bitsXor x y 0 then (-1 : Rat) else 1) =
        (if u 0 && x 0 then (-1 : Rat) else 1) * (if u 0 && y 0 then (-1 : Rat) else 1) := by
      simp only [bitsXor]
      exact bool_xor_sign (u 0) (x 0) (y 0)
    rw [h0Eq]
    grind

private theorem sumBits_congr {n : Nat} {f g : Bits n → Rat} (h : ∀ x : Bits n, f x = g x) :
    sumBits f = sumBits g := by
  have hFun : f = g := funext h
  rw [hFun]

private theorem sumBits_mul_left {n : Nat} (c : Rat) (f : Bits n → Rat) :
    sumBits (fun x => c * f x) = c * sumBits f := by
  induction n with
  | zero => simp [sumBits]
  | succ k ih =>
    simp only [sumBits]
    rw [ih (fun t => f (bitsCons false t)), ih (fun t => f (bitsCons true t))]
    grind

private theorem sumBits_mul_right {n : Nat} (f : Bits n → Rat) (c : Rat) :
    sumBits (fun x => f x * c) = sumBits f * c := by
  induction n with
  | zero => simp [sumBits]
  | succ k ih =>
    simp only [sumBits]
    rw [ih (fun t => f (bitsCons false t)), ih (fun t => f (bitsCons true t))]
    grind

private theorem sumBits_add {n : Nat} (f g : Bits n → Rat) :
    sumBits (fun x => f x + g x) = sumBits f + sumBits g := by
  induction n with
  | zero => simp [sumBits]
  | succ k ih =>
    simp only [sumBits]
    rw [ih (fun t => f (bitsCons false t)) (fun t => g (bitsCons false t)),
        ih (fun t => f (bitsCons true t)) (fun t => g (bitsCons true t))]
    grind

private theorem sumBits_swap {m n : Nat} (f : Bits m → Bits n → Rat) :
    sumBits (fun x => sumBits (fun y => f x y)) =
    sumBits (fun y => sumBits (fun x => f x y)) := by
  induction m with
  | zero =>
    simp only [sumBits]
  | succ k ih =>
    have hL : sumBits (fun x : Bits (k + 1) => sumBits (fun y : Bits n => f x y)) =
        sumBits (fun t : Bits k => sumBits (fun y : Bits n => f (bitsCons false t) y)) +
        sumBits (fun t : Bits k => sumBits (fun y : Bits n => f (bitsCons true t) y)) := rfl
    rw [hL, ih (fun t y => f (bitsCons false t) y), ih (fun t y => f (bitsCons true t) y), ← sumBits_add]
    rfl

private theorem sumBits_xor_shift :
    ∀ {n : Nat} (y : Bits n) (h : Bits n → Rat),
      sumBits (fun x => h (bitsXor x y)) = sumBits h := by
  intro n
  induction n with
  | zero =>
    intro y h
    simp only [sumBits]
    have hEq : bitsXor (fun i : Fin 0 => i.elim0) y = (fun i : Fin 0 => i.elim0) :=
      funext (fun i => i.elim0)
    rw [hEq]
  | succ k ih =>
    intro y h
    simp only [sumBits]
    have hF : (fun t : Bits k => h (bitsXor (bitsCons false t) y)) =
        (fun t : Bits k => h (bitsCons (false ^^ y 0) (bitsXor t (bitsTail y)))) := by
      funext t
      have hCons : bitsCons (false ^^ y 0) (bitsXor t (bitsTail y)) = bitsXor (bitsCons false t) y := by
        rw [← bitsXor_bitsCons false (y 0) t (bitsTail y), bitsCons_zero_tail]
      rw [hCons]
    have hT : (fun t : Bits k => h (bitsXor (bitsCons true t) y)) =
        (fun t : Bits k => h (bitsCons (true ^^ y 0) (bitsXor t (bitsTail y)))) := by
      funext t
      have hCons : bitsCons (true ^^ y 0) (bitsXor t (bitsTail y)) = bitsXor (bitsCons true t) y := by
        rw [← bitsXor_bitsCons true (y 0) t (bitsTail y), bitsCons_zero_tail]
      rw [hCons]
    rw [hF, hT]
    rw [ih (bitsTail y) (fun s => h (bitsCons (false ^^ y 0) s)),
        ih (bitsTail y) (fun s => h (bitsCons (true ^^ y 0) s))]
    cases y 0 <;> simp <;> try grind

private theorem sumBits_mul_walsh_orthog :
    ∀ (n : Nat) (f : Bits n → Rat) (v : Bits n),
      sumBits (fun x => f x * sumBits (fun u => walshChar u x * walshChar u v)) =
      ratPow 2 n * f v := by
  intro n
  induction n with
  | zero =>
    intro f v
    simp only [sumBits, walshChar, ratPow]
    have hEq : (fun i : Fin 0 => i.elim0) = v := funext (fun i => i.elim0)
    rw [hEq]
    grind
  | succ k ih =>
    intro f v
    have hvDecomp : bitsCons (v 0) (bitsTail v) = v := bitsCons_zero_tail v
    have hInner : ∀ bx : Bool, ∀ tx : Bits k,
        sumBits (fun u : Bits (k + 1) =>
          walshChar u (bitsCons bx tx) * walshChar u v) =
        (1 + boolSign bx * boolSign (v 0)) *
          sumBits (fun tu : Bits k => walshChar tu tx * walshChar tu (bitsTail v)) := by
      intro bx tx
      have hStep : sumBits (fun u : Bits (k + 1) =>
          walshChar u (bitsCons bx tx) * walshChar u v) =
          sumBits (fun tu : Bits k =>
            walshChar (bitsCons false tu) (bitsCons bx tx) * walshChar (bitsCons false tu) v) +
          sumBits (fun tu : Bits k =>
            walshChar (bitsCons true tu) (bitsCons bx tx) * walshChar (bitsCons true tu) v) := rfl
      rw [hStep]
      have hFalse : (fun tu : Bits k =>
          walshChar (bitsCons false tu) (bitsCons bx tx) * walshChar (bitsCons false tu) v) =
          (fun tu : Bits k => walshChar tu tx * walshChar tu (bitsTail v)) := by
        funext tu
        have hV : walshChar (bitsCons false tu) v = walshChar tu (bitsTail v) := by
          calc walshChar (bitsCons false tu) v
            _ = walshChar (bitsCons false tu) (bitsCons (v 0) (bitsTail v)) := by rw [hvDecomp]
            _ = (if false && v 0 then (-1 : Rat) else 1) * walshChar tu (bitsTail v) := by
              rw [walshChar_bitsCons]
            _ = walshChar tu (bitsTail v) := by simp
        rw [walshChar_bitsCons, hV]
        simp
      have hTrue : (fun tu : Bits k =>
          walshChar (bitsCons true tu) (bitsCons bx tx) * walshChar (bitsCons true tu) v) =
          (fun tu : Bits k =>
            (boolSign bx * boolSign (v 0)) * (walshChar tu tx * walshChar tu (bitsTail v))) := by
        funext tu
        have hV : walshChar (bitsCons true tu) v =
            (if true && v 0 then (-1 : Rat) else 1) * walshChar tu (bitsTail v) := by
          calc walshChar (bitsCons true tu) v
            _ = walshChar (bitsCons true tu) (bitsCons (v 0) (bitsTail v)) := by rw [hvDecomp]
            _ = (if true && v 0 then (-1 : Rat) else 1) * walshChar tu (bitsTail v) := by
              rw [walshChar_bitsCons]
        rw [walshChar_bitsCons, hV]
        simp only [Bool.true_and]
        cases bx <;> cases v 0 <;> simp [boolSign] <;> try grind
      rw [hFalse, hTrue, sumBits_mul_left]
      grind
    have hSplit :
        sumBits (fun x : Bits (k + 1) =>
          f x * sumBits (fun u : Bits (k + 1) => walshChar u x * walshChar u v)) =
        sumBits (fun tx : Bits k =>
          (1 + boolSign false * boolSign (v 0)) *
            (f (bitsCons false tx) * sumBits (fun tu : Bits k => walshChar tu tx * walshChar tu (bitsTail v)))) +
        sumBits (fun tx : Bits k =>
          (1 + boolSign true * boolSign (v 0)) *
            (f (bitsCons true tx) * sumBits (fun tu : Bits k => walshChar tu tx * walshChar tu (bitsTail v)))) := by
      have hDef :
          sumBits (fun x : Bits (k + 1) =>
            f x * sumBits (fun u : Bits (k + 1) => walshChar u x * walshChar u v)) =
          sumBits (fun tx : Bits k =>
            f (bitsCons false tx) * sumBits (fun u => walshChar u (bitsCons false tx) * walshChar u v)) +
          sumBits (fun tx : Bits k =>
            f (bitsCons true tx) * sumBits (fun u => walshChar u (bitsCons true tx) * walshChar u v)) := rfl
      rw [hDef]
      congr 1 <;> apply sumBits_congr <;> intro tx <;> rw [hInner] <;> grind
    rw [hSplit, sumBits_mul_left, sumBits_mul_left]
    rw [ih (fun tx => f (bitsCons false tx)) (bitsTail v),
        ih (fun tx => f (bitsCons true tx)) (bitsTail v)]
    cases hv0 : v 0 with
    | false =>
      have hVEq : bitsCons false (bitsTail v) = v := by rw [← hv0]; exact hvDecomp
      simp only [boolSign, ratPow_succ]
      rw [← hVEq]
      grind
    | true =>
      have hVEq : bitsCons true (bitsTail v) = v := by rw [← hv0]; exact hvDecomp
      simp only [boolSign, ratPow_succ]
      rw [← hVEq]
      grind

/--
**定理 3（`walsh_hadamard_involutive_and_xor_conv`）**：
对任意有限比特宽度 `n : Nat`、任意函数 `f g : Bits n → Rat` 与任意频率点 `u v : Bits n`：
1. **精确对合逆变换**：`wht (wht f) v = ratPow 2 n * f v`；
2. **XOR 卷积频域对角化**：`wht (xorConv f g) u = wht f u * wht g u`。
-/
theorem walsh_hadamard_involutive_and_xor_conv (n : Nat) (f g : Bits n → Rat) (u v : Bits n) :
    wht (wht f) v = ratPow 2 n * f v ∧
    wht (xorConv f g) u = wht f u * wht g u := by
  constructor
  · unfold wht
    have h1 : sumBits (fun w => walshChar v w * sumBits (fun x => walshChar w x * f x)) =
        sumBits (fun w => sumBits (fun x => f x * (walshChar w x * walshChar w v))) := by
      apply sumBits_congr
      intro w
      rw [← sumBits_mul_left]
      apply sumBits_congr
      intro x
      rw [walshChar_comm v w]
      grind
    rw [h1, sumBits_swap]
    have h2 : sumBits (fun x => sumBits (fun w => f x * (walshChar w x * walshChar w v))) =
        sumBits (fun x => f x * sumBits (fun w => walshChar w x * walshChar w v)) := by
      apply sumBits_congr
      intro x
      rw [sumBits_mul_left]
    rw [h2]
    exact sumBits_mul_walsh_orthog n f v
  · unfold wht xorConv
    have h1 : sumBits (fun x => walshChar u x * sumBits (fun y => f y * g (bitsXor x y))) =
        sumBits (fun x => sumBits (fun y => f y * (walshChar u x * g (bitsXor x y)))) := by
      apply sumBits_congr
      intro x
      rw [← sumBits_mul_left]
      apply sumBits_congr
      intro y
      grind
    rw [h1, sumBits_swap]
    have h2 : sumBits (fun y => sumBits (fun x => f y * (walshChar u x * g (bitsXor x y)))) =
        sumBits (fun y => (walshChar u y * f y) * sumBits (fun z => walshChar u z * g z)) := by
      apply sumBits_congr
      intro y
      rw [sumBits_mul_left]
      have hShift := sumBits_xor_shift y (fun z => walshChar u (bitsXor z y) * g z)
      have hMatch : (fun x : Bits n => walshChar u x * g (bitsXor x y)) =
          (fun x : Bits n => walshChar u (bitsXor (bitsXor x y) y) * g (bitsXor x y)) := by
        funext x
        rw [bitsXor_xor_cancel]
      rw [hMatch, hShift]
      have hFactor : sumBits (fun z : Bits n => walshChar u (bitsXor z y) * g z) =
          walshChar u y * sumBits (fun z : Bits n => walshChar u z * g z) := by
        rw [← sumBits_mul_left]
        apply sumBits_congr
        intro z
        rw [walshChar_xor_right]
        grind
      rw [hFactor]
      grind
    rw [h2, sumBits_mul_right]

end PhaseSHA.V5.DualDomainRRR
