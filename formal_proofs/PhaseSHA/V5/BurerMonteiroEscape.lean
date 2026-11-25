import PhaseSHA.Algebraic.LiteField

/-!
# Module 6B: Burer-Monteiro 超高维升维与假坑逃逸证书 (`BurerMonteiroEscape.lean`, $\mathbb{S}^{N-1} \to \mathbb{S}^N$)

本模块在零外部依赖的精确有理数向量空间与有理 Cayley 测地曲线下，严格形式化验证
“抬高向量维度打穿低维假局部极小值”的四项核心结论：
1. **`energy_rank1_eq_ising`**：证明高维单位超球面二次耦合能量模型在秩 1 嵌入下与离散 Ising/XOR 能量精确重合；
2. **`burer_monteiro_weak_duality_global_opt`**：证明一旦对偶松弛矩阵 `slackMatrix C V` 半正定，
   当前配置 `V` 即对**任意更高维度 `K`** 的所有单位向量配置都是全局最优解；
3. **`burer_monteiro_plus_one_dim_escape`**：反之，若 `slackMatrix C V` 存在负二次型方向 `u`
   （`u^T S(V) u < 0`），则等距嵌入至 `N+1` 维并沿第 `N+1` 维施加半角有理 Cayley 扰动 `liftedCurve V u t`：
   - 全程严格保范（`IsUnitConfig`），无模长塌缩；
   - `t = 0` 处一阶导数精确为 `0`，二阶导数精确等于 `u^T S(V) u < 0`（含全局四次余项界）；
   - 存在显式无开方下降步长 `ε > 0` 使 `energy C (liftedCurve V u ε) < energy C V`；
4. **`burer_monteiro_frustrated_triangle_gap`（诚实边界与反例）**：在 `m = 3` 反铁磁受挫三角形上
   诚实证明 1D 离散最优能量下界为 `-1`，而 2D 单位圆配置可达到 `-37/25 < -1`，严格刻画了
   升维消除假坑的同时产生的 SDP 整合间隙（需配合秩-1 退火方能准确投回原离散域）。
-/

namespace PhaseSHA.V5.BurerMonteiroEscape

open PhaseSHA.Algebraic.LiteField

/-!
### 1. 有限和代数基础设施 (`finSum`)
-/

/-- 有限索引 `Fin n` 上的有理数求和。 -/
def finSum : {n : Nat} → (Fin n → Rat) → Rat
  | 0, _ => 0
  | k + 1, f => finSum (fun i : Fin k => f i.castSucc) + f (Fin.last k)

theorem finSum_zero (n : Nat) : finSum (fun _ : Fin n => (0 : Rat)) = 0 := by
  induction n with
  | zero => rfl
  | succ k ih =>
    simp only [finSum]
    rw [ih]
    grind

theorem finSum_congr {n : Nat} {f g : Fin n → Rat} (h : ∀ i : Fin n, f i = g i) :
    finSum f = finSum g := by
  have hFun : f = g := funext h
  rw [hFun]

theorem finSum_zero_of {n : Nat} (f : Fin n → Rat) (h : ∀ i : Fin n, f i = 0) :
    finSum f = 0 := by
  rw [finSum_congr h]
  exact finSum_zero n

theorem finSum_add {n : Nat} (f g : Fin n → Rat) :
    finSum (fun i => f i + g i) = finSum f + finSum g := by
  induction n with
  | zero => simp [finSum]; grind
  | succ k ih =>
    simp only [finSum]
    rw [ih (fun i => f i.castSucc) (fun i => g i.castSucc)]
    grind

theorem finSum_sub {n : Nat} (f g : Fin n → Rat) :
    finSum (fun i => f i - g i) = finSum f - finSum g := by
  induction n with
  | zero => simp [finSum]; grind
  | succ k ih =>
    simp only [finSum]
    rw [ih (fun i => f i.castSucc) (fun i => g i.castSucc)]
    grind

theorem finSum_mul_left {n : Nat} (c : Rat) (f : Fin n → Rat) :
    finSum (fun i => c * f i) = c * finSum f := by
  induction n with
  | zero => simp [finSum]
  | succ k ih =>
    simp only [finSum]
    rw [ih (fun i => f i.castSucc)]
    grind

theorem finSum_mul_right {n : Nat} (f : Fin n → Rat) (c : Rat) :
    finSum (fun i => f i * c) = finSum f * c := by
  induction n with
  | zero => simp [finSum]
  | succ k ih =>
    simp only [finSum]
    rw [ih (fun i => f i.castSucc)]
    grind

theorem finSum_nonneg {n : Nat} (f : Fin n → Rat) (h : ∀ i : Fin n, 0 ≤ f i) :
    0 ≤ finSum f := by
  induction n with
  | zero => simp [finSum]
  | succ k ih =>
    simp only [finSum]
    have h1 := ih (fun i => f i.castSucc) (fun i => h i.castSucc)
    have h2 := h (Fin.last k)
    grind

theorem finSum_le_finSum {n : Nat} {f g : Fin n → Rat} (h : ∀ i : Fin n, f i ≤ g i) :
    finSum f ≤ finSum g := by
  have hDiff : 0 ≤ finSum (fun i => g i - f i) := finSum_nonneg _ (fun i => by have := h i; grind)
  rw [finSum_sub] at hDiff
  grind

theorem finSum_abs_le {n : Nat} (f : Fin n → Rat) :
    ratAbs (finSum f) ≤ finSum (fun i => ratAbs (f i)) := by
  induction n with
  | zero => simp [finSum]
  | succ k ih =>
    simp only [finSum]
    have hTri := ratAbs_add_le (finSum (fun i : Fin k => f i.castSucc)) (f (Fin.last k))
    have hIH := ih (fun i => f i.castSucc)
    grind

theorem finSum_swap {m n : Nat} (f : Fin m → Fin n → Rat) :
    finSum (fun i => finSum (fun j => f i j)) =
    finSum (fun j => finSum (fun i => f i j)) := by
  induction n with
  | zero =>
    simp only [finSum]
    exact finSum_zero_of _ (fun _ => rfl)
  | succ k ih =>
    have hL : finSum (fun i : Fin m => finSum (fun j : Fin (k + 1) => f i j)) =
        finSum (fun i : Fin m =>
          finSum (fun j : Fin k => f i j.castSucc) + f i (Fin.last k)) := by
      apply finSum_congr
      intro i
      rfl
    rw [hL, finSum_add, ih (fun i j => f i j.castSucc)]
    rfl

theorem finSum_ite_eq {n : Nat} (j : Fin n) (g : Fin n → Rat) :
    finSum (fun i => if i = j then g i else 0) = g j := by
  induction n with
  | zero => exact j.elim0
  | succ k ih =>
    simp only [finSum]
    by_cases hj : j.val = k
    · have hjLast : j = Fin.last k := Fin.ext hj
      have hPrev : finSum (fun i : Fin k =>
          if i.castSucc = j then g i.castSucc else 0) = 0 := by
        apply finSum_zero_of
        intro i
        have hNe : i.castSucc ≠ j := by
          intro hEq
          have hv : i.val = j.val := congrArg Fin.val hEq
          omega
        simp [hNe]
      rw [← hjLast, hPrev]
      simp
      grind
    · have hLt : j.val < k := Nat.lt_of_le_of_ne (Nat.lt_succ_iff.mp j.isLt) hj
      let j' : Fin k := ⟨j.val, hLt⟩
      have hjEq : j'.castSucc = j := Fin.ext rfl
      have hLastNe : Fin.last k ≠ j := by
        intro hEq
        have hv : k = j.val := congrArg Fin.val hEq
        omega
      have hPrevEq : (fun i : Fin k => if i.castSucc = j then g i.castSucc else 0) =
          (fun i : Fin k => if i = j' then g i.castSucc else 0) := by
        funext i
        by_cases h : i = j'
        · subst h; simp [hjEq]
        · have hNe : i.castSucc ≠ j := by
            intro hEq
            apply h
            have hv : i.val = j'.val := by
              have h1 : i.castSucc.val = j.val := congrArg Fin.val hEq
              simpa [j'] using h1
            exact Fin.ext hv
          simp [h, hNe]
      rw [hPrevEq, ih j' (fun i => g i.castSucc)]
      simp [hLastNe, hjEq]
      grind

private theorem finSum_succ_step {k : Nat} (f : Fin (k + 1) → Rat) :
    finSum f = finSum (fun i : Fin k => f i.castSucc) + f (Fin.last k) := rfl

theorem finSum_only_zero {N : Nat} (hN : 1 ≤ N) (f : Fin N → Rat)
    (hRest : ∀ k : Fin N, k.val ≠ 0 → f k = 0) :
    finSum f = f ⟨0, hN⟩ := by
  induction N with
  | zero => omega
  | succ m ih =>
    cases m with
    | zero =>
      rw [finSum_succ_step]
      have hZero : finSum (fun i : Fin 0 => f i.castSucc) = 0 := rfl
      have hEq : Fin.last 0 = (⟨0, hN⟩ : Fin 1) := Fin.ext rfl
      rw [hZero, hEq]
      grind
    | succ m' =>
      rw [finSum_succ_step]
      have hLast : f (Fin.last (m' + 1)) = 0 := hRest (Fin.last (m' + 1)) (by simp)
      have hm' : 1 ≤ m' + 1 := by omega
      have ih' := ih hm' (fun i => f i.castSucc) (fun k hk => hRest k.castSucc hk)
      have h0Eq : (⟨0, hm'⟩ : Fin (m' + 1)).castSucc = (⟨0, hN⟩ : Fin (m' + 2)) := Fin.ext rfl
      rw [hLast, ih', h0Eq]
      grind

/-!
### 2. 高维单位球面二次 Ising / XOR 能量模型与秩 1 精确恢复
-/

/-- `N` 维有理向量类型。 -/
abbrev Vec (N : Nat) := Fin N → Rat

/-- 欧氏内积 `⟨u, v⟩ = ∑_k u_k v_k`。 -/
def vecDot {N : Nat} (u v : Vec N) : Rat :=
  finSum (fun k => u k * v k)

/-- 向量范数平方 `‖v‖₂² = ⟨v, v⟩`。 -/
def vecNormSq {N : Nat} (v : Vec N) : Rat :=
  vecDot v v

theorem vecDot_comm {N : Nat} (u v : Vec N) : vecDot u v = vecDot v u := by
  unfold vecDot
  apply finSum_congr
  intro k
  grind

/-- 单位超球面配置约束：`∀ i, ‖v_i‖₂² = 1`。 -/
def IsUnitConfig {m N : Nat} (V : Fin m → Vec N) : Prop :=
  ∀ i : Fin m, vecNormSq (V i) = 1

/-- 对称耦合矩阵：`C_{ij} = C_{ji}`。 -/
def IsSymMatrix {m : Nat} (C : Fin m → Fin m → Rat) : Prop :=
  ∀ i j : Fin m, C i j = C j i

/-- Burer-Monteiro 高维单位球面能量 `E(V) = (1/2) ∑_{i,j} C_{ij} ⟨v_i, v_j⟩`。 -/
def energy {m N : Nat} (C : Fin m → Fin m → Rat) (V : Fin m → Vec N) : Rat :=
  (1 / 2 : Rat) * finSum (fun i => finSum (fun j => C i j * vecDot (V i) (V j)))

/-- 离散 Ising / XOR 能量 `E_Ising(s) = (1/2) ∑_{i,j} C_{ij} s_i s_j`。 -/
def isingEnergy {m : Nat} (C : Fin m → Fin m → Rat) (s : Fin m → Rat) : Rat :=
  (1 / 2 : Rat) * finSum (fun i => finSum (fun j => C i j * (s i * s j)))

/-- 秩 1 离散解向任意 `N ≥ 1` 维超球面第一坐标轴 `e_1` 的等距嵌入：`v_i = s_i e_1`。 -/
def rank1Embed {m N : Nat} (_ : 1 ≤ N) (s : Fin m → Rat) : Fin m → Vec N :=
  fun i k => if k.val = 0 then s i else 0

private theorem rank1Embed_dot {m N : Nat} (hN : 1 ≤ N) (s : Fin m → Rat) (i j : Fin m) :
    vecDot (rank1Embed hN s i) (rank1Embed hN s j) = s i * s j := by
  unfold vecDot
  rw [finSum_only_zero hN]
  · simp [rank1Embed]
  · intro k hk
    simp [rank1Embed, hk]

/--
**定理 1（`energy_rank1_eq_ising`）**：
对任意 `N ≥ 1` 与任意离散构型 `s : Fin m → {±1}`，其秩 1 嵌入 `rank1Embed hN s`
严格满足单位球面约束 `IsUnitConfig`，且能量精确恢复离散 Ising/XOR 能量。
-/
theorem energy_rank1_eq_ising {m N : Nat} (hN : 1 ≤ N) (C : Fin m → Fin m → Rat)
    (s : Fin m → Rat) (hs : ∀ i : Fin m, s i = 1 ∨ s i = -1) :
    IsUnitConfig (rank1Embed hN s) ∧ energy C (rank1Embed hN s) = isingEnergy C s := by
  constructor
  · intro i
    unfold vecNormSq
    rw [rank1Embed_dot hN s i i]
    rcases hs i with h1 | h1 <;> simp [h1] <;> grind
  · unfold energy isingEnergy
    congr 1
    apply finSum_congr
    intro i
    apply finSum_congr
    intro j
    rw [rank1Embed_dot hN s i j]

/-!
### 3. 对偶松弛矩阵 (`slackMatrix`) 与跨维度弱对偶全局最优定理
-/

/-- 黎曼拉格朗日乘子 `λ_i(V) = ∑_j C_{ij} ⟨v_i, v_j⟩`。 -/
def lagrangeMult {m N : Nat} (C : Fin m → Fin m → Rat) (V : Fin m → Vec N) (i : Fin m) : Rat :=
  finSum (fun j => C i j * vecDot (V i) (V j))

/-- 对偶松弛矩阵 `S(V)_{ij} = C_{ij} - δ_{ij} λ_i(V)`。 -/
def slackMatrix {m N : Nat} (C : Fin m → Fin m → Rat) (V : Fin m → Vec N)
    (i j : Fin m) : Rat :=
  C i j - (if i = j then lagrangeMult C V i else 0)

/-- 实二次型 `u^T M u = ∑_{i,j} M_{ij} u_i u_j`。 -/
def quadForm {m : Nat} (M : Fin m → Fin m → Rat) (u : Fin m → Rat) : Rat :=
  finSum (fun i => finSum (fun j => M i j * u i * u j))

/-- 矩阵半正定性：`∀ u, u^T M u ≥ 0`。 -/
def IsPSD {m : Nat} (M : Fin m → Fin m → Rat) : Prop :=
  ∀ u : Fin m → Rat, 0 ≤ quadForm M u

private theorem two_mul_energy_sub_eq_sum_quadForm {m N K : Nat}
    (C : Fin m → Fin m → Rat) (V : Fin m → Vec N) (W : Fin m → Vec K)
    (hW : IsUnitConfig W) :
    2 * (energy C W - energy C V) =
      finSum (fun k : Fin K => quadForm (slackMatrix C V) (fun i => W i k)) := by
  have hRHS : finSum (fun k : Fin K => quadForm (slackMatrix C V) (fun i => W i k)) =
      finSum (fun i : Fin m => finSum (fun j : Fin m =>
        slackMatrix C V i j * vecDot (W i) (W j))) := by
    unfold quadForm
    rw [← finSum_swap (fun i k => finSum (fun j => slackMatrix C V i j * W i k * W j k))]
    apply finSum_congr
    intro i
    rw [← finSum_swap (fun j k => slackMatrix C V i j * W i k * W j k)]
    apply finSum_congr
    intro j
    unfold vecDot
    have hAssoc : (fun k : Fin K => slackMatrix C V i j * W i k * W j k) =
        (fun k : Fin K => slackMatrix C V i j * (W i k * W j k)) := by
      funext k; grind
    rw [hAssoc, finSum_mul_left]
  rw [hRHS]
  have hSplit : finSum (fun i : Fin m => finSum (fun j : Fin m =>
      slackMatrix C V i j * vecDot (W i) (W j))) =
      finSum (fun i : Fin m => finSum (fun j : Fin m => C i j * vecDot (W i) (W j))) -
      finSum (fun i : Fin m => finSum (fun j : Fin m =>
        (if i = j then lagrangeMult C V i else 0) * vecDot (W i) (W j))) := by
    rw [← finSum_sub]
    apply finSum_congr
    intro i
    rw [← finSum_sub]
    apply finSum_congr
    intro j
    unfold slackMatrix
    grind
  rw [hSplit]
  have hDiag : finSum (fun i : Fin m => finSum (fun j : Fin m =>
      (if i = j then lagrangeMult C V i else 0) * vecDot (W i) (W j))) =
      finSum (fun i : Fin m => lagrangeMult C V i) := by
    apply finSum_congr
    intro i
    have hRewrite : (fun j : Fin m =>
        (if i = j then lagrangeMult C V i else 0) * vecDot (W i) (W j)) =
        (fun j : Fin m => if j = i then lagrangeMult C V i * vecDot (W i) (W j) else 0) := by
      funext j
      by_cases h : i = j
      · subst h; simp
      · have hj : j ≠ i := fun hji => h hji.symm
        simp [h, hj]
    rw [hRewrite, finSum_ite_eq i (fun j => lagrangeMult C V i * vecDot (W i) (W j))]
    have hNorm : vecDot (W i) (W i) = 1 := hW i
    rw [hNorm]
    grind
  rw [hDiag]
  have hSumLam : finSum (fun i : Fin m => lagrangeMult C V i) = 2 * energy C V := by
    unfold lagrangeMult energy
    grind
  have hWEn : finSum (fun i : Fin m => finSum (fun j : Fin m => C i j * vecDot (W i) (W j))) =
      2 * energy C W := by
    unfold energy
    grind
  rw [hSumLam, hWEn]
  grind

/--
**定理 2（`burer_monteiro_weak_duality_global_opt`）**：
若对偶松弛矩阵 `slackMatrix C V` 半正定，则对**任意更高（或任意）维度 `K : Nat`**
及任意单位球面配置 `W : Fin m → Vec K`，均有 `energy C V ≤ energy C W`（`V` 为跨维度全局最优解）。
-/
theorem burer_monteiro_weak_duality_global_opt {m N K : Nat}
    (C : Fin m → Fin m → Rat) (V : Fin m → Vec N)
    (hPSD : IsPSD (slackMatrix C V)) (W : Fin m → Vec K) (hW : IsUnitConfig W) :
    energy C V ≤ energy C W := by
  have hId := two_mul_energy_sub_eq_sum_quadForm C V W hW
  have hNonneg : 0 ≤ finSum (fun k : Fin K => quadForm (slackMatrix C V) (fun i => W i k)) :=
    finSum_nonneg _ (fun k => hPSD (fun i => W i k))
  rw [← hId] at hNonneg
  grind

/-!
### 4. +1 维保范有理 Cayley 扰动曲线与严格负曲率逃逸通道定理
-/

/-- 半角有理 Cayley 余弦：`(1 - x²) / (1 + x²)`。 -/
def cayleyCos (x : Rat) : Rat := (1 - x * x) / (1 + x * x)

/-- 半角有理 Cayley 正弦：`2x / (1 + x²)`。 -/
def cayleySin (x : Rat) : Rat := (2 * x) / (1 + x * x)

theorem cayley_den_ge_one (x : Rat) : 1 ≤ 1 + x * x := by
  have hsq := rat_sq_nonneg x
  grind

theorem cayley_sq_sum (x : Rat) : cayleyCos x * cayleyCos x + cayleySin x * cayleySin x = 1 := by
  have hd : 1 + x * x ≠ 0 := by
    have hge := cayley_den_ge_one x
    grind
  unfold cayleyCos cayleySin
  rw [Rat.div_def, Rat.div_def]
  have hNum : ((1 - x * x) * (1 + x * x)⁻¹) * ((1 - x * x) * (1 + x * x)⁻¹) +
      ((2 * x) * (1 + x * x)⁻¹) * ((2 * x) * (1 + x * x)⁻¹) =
      ((1 + x * x) * (1 + x * x)) * ((1 + x * x)⁻¹ * (1 + x * x)⁻¹) := by grind
  rw [hNum]
  have hAssoc : ((1 + x * x) * (1 + x * x)) * ((1 + x * x)⁻¹ * (1 + x * x)⁻¹) =
      ((1 + x * x) * (1 + x * x)⁻¹) * ((1 + x * x) * (1 + x * x)⁻¹) := by grind
  rw [hAssoc, Rat.mul_inv_cancel _ hd]
  grind

theorem cayleyCos_zero : cayleyCos 0 = 1 := by
  unfold cayleyCos
  grind

theorem cayleySin_zero : cayleySin 0 = 0 := by
  unfold cayleySin
  grind

theorem cayleyCos_neg (x : Rat) : cayleyCos (-x) = cayleyCos x := by
  unfold cayleyCos
  have hsq : (-x) * (-x) = x * x := by grind
  rw [hsq]

theorem cayleySin_neg (x : Rat) : cayleySin (-x) = -cayleySin x := by
  unfold cayleySin
  have hsq : (-x) * (-x) = x * x := by grind
  rw [hsq, Rat.div_def, Rat.div_def]
  grind

/-- 将 `N` 维单位向量等距嵌入 `N+1` 维并沿第 `N+1` 维施加半角有理 Cayley 旋转。 -/
def liftedVec {N : Nat} (v : Vec N) (s t : Rat) : Vec (N + 1) :=
  let x := (t * s) / 2
  fun k => if hk : k.val < N then cayleyCos x * v ⟨k.val, hk⟩ else cayleySin x

/-- 升维扰动曲线 `Ṽ(t) : Fin m → Vec (N + 1)`。 -/
def liftedCurve {m N : Nat} (V : Fin m → Vec N) (u : Fin m → Rat) (t : Rat) :
    Fin m → Vec (N + 1) :=
  fun i => liftedVec (V i) (u i) t

private theorem liftedVec_dot {N : Nat} (v1 v2 : Vec N) (s1 s2 t : Rat) :
    vecDot (liftedVec v1 s1 t) (liftedVec v2 s2 t) =
      cayleyCos ((t * s1) / 2) * cayleyCos ((t * s2) / 2) * vecDot v1 v2 +
      cayleySin ((t * s1) / 2) * cayleySin ((t * s2) / 2) := by
  unfold vecDot
  simp only [finSum]
  have hLast : liftedVec v1 s1 t (Fin.last N) * liftedVec v2 s2 t (Fin.last N) =
      cayleySin ((t * s1) / 2) * cayleySin ((t * s2) / 2) := by
    simp [liftedVec]
  have hPrefix : finSum (fun k : Fin N =>
      liftedVec v1 s1 t k.castSucc * liftedVec v2 s2 t k.castSucc) =
      cayleyCos ((t * s1) / 2) * cayleyCos ((t * s2) / 2) * finSum (fun k => v1 k * v2 k) := by
    rw [← finSum_mul_left]
    apply finSum_congr
    intro k
    have hk : k.castSucc.val < N := k.isLt
    simp only [liftedVec, hk, ↓reduceDIte]
    have hIdx : (⟨k.castSucc.val, hk⟩ : Fin N) = k := Fin.ext rfl
    rw [hIdx]
    grind
  rw [hLast, hPrefix]

theorem liftedCurve_unit {m N : Nat} (V : Fin m → Vec N) (u : Fin m → Rat)
    (hV : IsUnitConfig V) (t : Rat) :
    IsUnitConfig (liftedCurve V u t) := by
  intro i
  unfold vecNormSq liftedCurve
  rw [liftedVec_dot]
  have hNorm : vecDot (V i) (V i) = 1 := hV i
  rw [hNorm]
  have hSq := cayley_sq_sum ((t * u i) / 2)
  grind

theorem liftedCurve_energy_zero {m N : Nat} (C : Fin m → Fin m → Rat)
    (V : Fin m → Vec N) (u : Fin m → Rat) :
    energy C (liftedCurve V u 0) = energy C V := by
  unfold energy liftedCurve
  congr 1
  apply finSum_congr
  intro i
  apply finSum_congr
  intro j
  rw [liftedVec_dot]
  have hz1 : (0 : Rat) * u i / 2 = 0 := by grind
  have hz2 : (0 : Rat) * u j / 2 = 0 := by grind
  rw [hz1, hz2, cayleyCos_zero, cayleySin_zero]
  grind

theorem liftedCurve_energy_even {m N : Nat} (C : Fin m → Fin m → Rat)
    (V : Fin m → Vec N) (u : Fin m → Rat) (t : Rat) :
    energy C (liftedCurve V u (-t)) = energy C (liftedCurve V u t) := by
  unfold energy liftedCurve
  congr 1
  apply finSum_congr
  intro i
  apply finSum_congr
  intro j
  rw [liftedVec_dot, liftedVec_dot]
  have hn1 : (-t) * u i / 2 = -((t * u i) / 2) := by grind
  have hn2 : (-t) * u j / 2 = -((t * u j) / 2) := by grind
  rw [hn1, hn2, cayleyCos_neg, cayleyCos_neg, cayleySin_neg, cayleySin_neg]
  grind

private theorem cayleyCos_prod_rem_bound (ui uj t : Rat) :
    ratAbs (cayleyCos ((t * ui) / 2) * cayleyCos ((t * uj) / 2) - 1 +
      (t * t / 2) * (ui * ui + uj * uj)) ≤
      (t * t * t * t) * (ui * ui * ui * ui + uj * uj * uj * uj) := by
  let xi := (t * ui) / 2
  let xj := (t * uj) / 2
  let a := xi * xi
  let b := xj * xj
  have ha0 : 0 ≤ a := rat_sq_nonneg xi
  have hb0 : 0 ≤ b := rat_sq_nonneg xj
  have hda : 1 ≤ 1 + a := by grind
  have hdb : 1 ≤ 1 + b := by grind
  have hdenPos : 0 < (1 + a) * (1 + b) := Rat.mul_pos (by grind) (by grind)
  have hdenGe1 : 1 ≤ (1 + a) * (1 + b) := by
    have h1 : 1 * (1 + b) ≤ (1 + a) * (1 + b) := rat_mul_le_mul_of_nonneg_right hda (by grind)
    grind
  have hdenNe : (1 + a) * (1 + b) ≠ 0 := by grind
  have hQuad : (t * t / 2) * (ui * ui + uj * uj) = 2 * (a + b) := by
    dsimp [a, b, xi, xj]
    grind
  have hEq : cayleyCos xi * cayleyCos xj - 1 + (t * t / 2) * (ui * ui + uj * uj) =
      (2 * (a + b) * (a + b + a * b)) / ((1 + a) * (1 + b)) := by
    rw [hQuad]
    unfold cayleyCos
    change ((1 - a) / (1 + a)) * ((1 - b) / (1 + b)) - 1 + 2 * (a + b) =
      (2 * (a + b) * (a + b + a * b)) / ((1 + a) * (1 + b))
    rw [Rat.div_def, Rat.div_def, Rat.div_def]
    have hInv : (1 + a)⁻¹ * (1 + b)⁻¹ = ((1 + a) * (1 + b))⁻¹ := by
      have haNe : 1 + a ≠ 0 := by grind
      have hbNe : 1 + b ≠ 0 := by grind
      have hProd : ((1 + a) * (1 + b)) * ((1 + a)⁻¹ * (1 + b)⁻¹) = 1 := by
        have h1 : ((1 + a) * (1 + b)) * ((1 + a)⁻¹ * (1 + b)⁻¹) =
            ((1 + a) * (1 + a)⁻¹) * ((1 + b) * (1 + b)⁻¹) := by grind
        rw [h1, Rat.mul_inv_cancel _ haNe, Rat.mul_inv_cancel _ hbNe]
        grind
      have hL := congrArg (fun z => ((1 + a) * (1 + b))⁻¹ * z) hProd
      have hAssoc : ((1 + a) * (1 + b))⁻¹ * (((1 + a) * (1 + b)) * ((1 + a)⁻¹ * (1 + b)⁻¹)) =
          (((1 + a) * (1 + b)) * ((1 + a) * (1 + b))⁻¹) * ((1 + a)⁻¹ * (1 + b)⁻¹) := by grind
      rw [hAssoc, Rat.mul_inv_cancel _ hdenNe] at hL
      grind
    have hMul : ((1 - a) * (1 + a)⁻¹) * ((1 - b) * (1 + b)⁻¹) =
        ((1 - a) * (1 - b)) * ((1 + a) * (1 + b))⁻¹ := by
      rw [← hInv]
      grind
    have hMain : ((1 - a) * (1 - b)) * ((1 + a) * (1 + b))⁻¹ - 1 + 2 * (a + b) =
        (2 * (a + b) * (a + b + a * b)) * ((1 + a) * (1 + b))⁻¹ := by
      have hCancel := Rat.mul_inv_cancel ((1 + a) * (1 + b)) hdenNe
      grind
    rw [hMul, hMain]
  have hNumNonneg : 0 ≤ 2 * (a + b) * (a + b + a * b) := by
    have hab : 0 ≤ a + b := by grind
    have habab : 0 ≤ a + b + a * b := by
      have hm := Rat.mul_nonneg ha0 hb0
      grind
    exact Rat.mul_nonneg (Rat.mul_nonneg (by grind) hab) habab
  have hExprNonneg : 0 ≤ (2 * (a + b) * (a + b + a * b)) / ((1 + a) * (1 + b)) :=
    rat_div_nonneg hNumNonneg hdenPos
  rw [hEq, ratAbs_of_nonneg hExprNonneg, rat_div_le_iff hdenPos]
  have hAbLe : a * b * (a + b) ≤ b * (a + b) * ((1 + a) * (1 + b)) := by
    have hMid : a ≤ (1 + a) * (1 + b) := by
      have h1 : (1 + a) * 1 ≤ (1 + a) * (1 + b) := rat_mul_le_mul_of_nonneg_left hdb (by grind)
      grind
    have hFac : 0 ≤ b * (a + b) := Rat.mul_nonneg hb0 (by grind)
    have h1 := rat_mul_le_mul_of_nonneg_left hMid hFac
    grind
  have hSqLe : (a + b) * (a + b) ≤ (a + b) * (a + b) * ((1 + a) * (1 + b)) := by
    simpa using rat_mul_le_mul_of_nonneg_left hdenGe1 (rat_sq_nonneg (a + b))
  have hBAbLe : b * (a + b) ≤ (a + b) * (a + b) :=
    rat_mul_le_mul_of_nonneg_right (by grind : b ≤ a + b) (by grind : 0 ≤ a + b)
  have hBAbLeDen : b * (a + b) * ((1 + a) * (1 + b)) ≤ (a + b) * (a + b) * ((1 + a) * (1 + b)) :=
    rat_mul_le_mul_of_nonneg_right hBAbLe (Rat.le_of_lt hdenPos)
  have hStep1 : 2 * (a + b) * (a + b + a * b) ≤ 4 * ((a + b) * (a + b)) * ((1 + a) * (1 + b)) := by
    grind
  have hSymSq : (a + b) * (a + b) ≤ 2 * (a * a + b * b) := by
    have h0 : 0 ≤ (a - b) * (a - b) := rat_sq_nonneg (a - b)
    grind
  have hSymSqDen : 4 * ((a + b) * (a + b)) * ((1 + a) * (1 + b)) ≤
      8 * (a * a + b * b) * ((1 + a) * (1 + b)) := by
    have h1 := rat_mul_le_mul_of_nonneg_left hSymSq (by grind : (0 : Rat) ≤ 4)
    have h2 := rat_mul_le_mul_of_nonneg_right h1 (Rat.le_of_lt hdenPos)
    grind
  have hFinalVal : 8 * (a * a + b * b) ≤ (t * t * t * t) * (ui * ui * ui * ui + uj * uj * uj * uj) := by
    have heq : a * a + b * b = ((t * t * t * t) * (ui * ui * ui * ui + uj * uj * uj * uj)) / 16 := by
      dsimp [a, b, xi, xj]
      grind
    rw [heq]
    have hPos : 0 ≤ (t * t * t * t) * (ui * ui * ui * ui + uj * uj * uj * uj) := by
      have h1 : 0 ≤ (t * t) * (t * t) := rat_sq_nonneg (t * t)
      have h2 : 0 ≤ (ui * ui) * (ui * ui) + (uj * uj) * (uj * uj) := by
        have h2a := rat_sq_nonneg (ui * ui)
        have h2b := rat_sq_nonneg (uj * uj)
        grind
      have h3 := Rat.mul_nonneg h1 h2
      grind
    grind
  have hFinalDen := rat_mul_le_mul_of_nonneg_right hFinalVal (Rat.le_of_lt hdenPos)
  exact Rat.le_trans hStep1 (Rat.le_trans hSymSqDen hFinalDen)

private theorem sin_den_rem_bound {a b : Rat} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    (a + b + a * b) / ((1 + a) * (1 + b)) ≤ a + b := by
  have hdenPos : 0 < (1 + a) * (1 + b) := Rat.mul_pos (by grind) (by grind)
  rw [rat_div_le_iff hdenPos]
  have h1 : 0 ≤ a * a := rat_sq_nonneg a
  have h2 : 0 ≤ b * b := rat_sq_nonneg b
  have h3 : 0 ≤ a * b := Rat.mul_nonneg ha hb
  have h4 : 0 ≤ (a + b) * (a * b) := Rat.mul_nonneg (by grind) h3
  have hEq : (a + b) * ((1 + a) * (1 + b)) - (a + b + a * b) =
      a * a + b * b + a * b + (a + b) * (a * b) := by grind
  grind

private theorem cayleySin_prod_rem_bound (ui uj t : Rat) :
    ratAbs (cayleySin ((t * ui) / 2) * cayleySin ((t * uj) / 2) - t * t * ui * uj) ≤
      (t * t * t * t) * (ratAbs (ui * uj) * (ui * ui + uj * uj) / 4) := by
  let xi := (t * ui) / 2
  let xj := (t * uj) / 2
  let a := xi * xi
  let b := xj * xj
  have ha0 : 0 ≤ a := rat_sq_nonneg xi
  have hb0 : 0 ≤ b := rat_sq_nonneg xj
  have hdenPos : 0 < (1 + a) * (1 + b) := Rat.mul_pos (by grind) (by grind)
  have hdenNe : (1 + a) * (1 + b) ≠ 0 := by grind
  have hQuad : t * t * ui * uj = 4 * xi * xj := by
    dsimp [xi, xj]
    grind
  have hEq : cayleySin xi * cayleySin xj - t * t * ui * uj =
      -(4 * xi * xj) * ((a + b + a * b) / ((1 + a) * (1 + b))) := by
    rw [hQuad]
    unfold cayleySin
    change ((2 * xi) / (1 + a)) * ((2 * xj) / (1 + b)) - 4 * xi * xj =
      -(4 * xi * xj) * ((a + b + a * b) / ((1 + a) * (1 + b)))
    rw [Rat.div_def, Rat.div_def, Rat.div_def]
    have hInv : (1 + a)⁻¹ * (1 + b)⁻¹ = ((1 + a) * (1 + b))⁻¹ := by
      have haNe : 1 + a ≠ 0 := by grind
      have hbNe : 1 + b ≠ 0 := by grind
      have hProd : ((1 + a) * (1 + b)) * ((1 + a)⁻¹ * (1 + b)⁻¹) = 1 := by
        have h1 : ((1 + a) * (1 + b)) * ((1 + a)⁻¹ * (1 + b)⁻¹) =
            ((1 + a) * (1 + a)⁻¹) * ((1 + b) * (1 + b)⁻¹) := by grind
        rw [h1, Rat.mul_inv_cancel _ haNe, Rat.mul_inv_cancel _ hbNe]
        grind
      have hL := congrArg (fun z => ((1 + a) * (1 + b))⁻¹ * z) hProd
      have hAssoc : ((1 + a) * (1 + b))⁻¹ * (((1 + a) * (1 + b)) * ((1 + a)⁻¹ * (1 + b)⁻¹)) =
          (((1 + a) * (1 + b)) * ((1 + a) * (1 + b))⁻¹) * ((1 + a)⁻¹ * (1 + b)⁻¹) := by grind
      rw [hAssoc, Rat.mul_inv_cancel _ hdenNe] at hL
      grind
    have hMul : ((2 * xi) * (1 + a)⁻¹) * ((2 * xj) * (1 + b)⁻¹) =
        (4 * xi * xj) * ((1 + a) * (1 + b))⁻¹ := by
      rw [← hInv]
      grind
    have hMain : (4 * xi * xj) * ((1 + a) * (1 + b))⁻¹ - 4 * xi * xj =
        -(4 * xi * xj) * ((a + b + a * b) * ((1 + a) * (1 + b))⁻¹) := by
      have hCancel := Rat.mul_inv_cancel ((1 + a) * (1 + b)) hdenNe
      grind
    rw [hMul, hMain]
  have hFracNonneg : 0 ≤ (a + b + a * b) / ((1 + a) * (1 + b)) := by
    have hnum : 0 ≤ a + b + a * b := by
      have hm := Rat.mul_nonneg ha0 hb0
      grind
    exact rat_div_nonneg hnum hdenPos
  have hFracLe : (a + b + a * b) / ((1 + a) * (1 + b)) ≤ a + b :=
    sin_den_rem_bound ha0 hb0
  rw [hEq, ratAbs_mul, ratAbs_neg, ratAbs_of_nonneg hFracNonneg]
  have h4Abs : ratAbs (4 * xi * xj) = (t * t) * ratAbs (ui * uj) := by
    rw [← hQuad]
    have hSplit : t * t * ui * uj = (t * t) * (ui * uj) := by grind
    rw [hSplit, ratAbs_mul, ratAbs_of_nonneg (rat_sq_nonneg t)]
  rw [h4Abs]
  have hMul := rat_mul_le_mul_of_nonneg_left hFracLe
    (Rat.mul_nonneg (rat_sq_nonneg t) (ratAbs_nonneg (ui * uj)))
  have hEqBound : (t * t * ratAbs (ui * uj)) * (a + b) =
      (t * t * t * t) * (ratAbs (ui * uj) * (ui * ui + uj * uj) / 4) := by
    dsimp [a, b, xi, xj]
    grind
  rw [hEqBound] at hMul
  exact hMul

/-- 全局四次泰勒余项上界常数 `B ≥ 0`（完全独立于参数 `t`）。 -/
def escapeErrorBound {m N : Nat} (C : Fin m → Fin m → Rat)
    (V : Fin m → Vec N) (u : Fin m → Rat) : Rat :=
  (1 / 2 : Rat) * finSum (fun i => finSum (fun j =>
    ratAbs (C i j * vecDot (V i) (V j)) * (u i * u i * u i * u i + u j * u j * u j * u j) +
    ratAbs (C i j) * (ratAbs (u i * u j) * (u i * u i + u j * u j) / 4)))

theorem escapeErrorBound_nonneg {m N : Nat} (C : Fin m → Fin m → Rat)
    (V : Fin m → Vec N) (u : Fin m → Rat) :
    0 ≤ escapeErrorBound C V u := by
  unfold escapeErrorBound
  apply Rat.mul_nonneg (by grind)
  apply finSum_nonneg
  intro i
  apply finSum_nonneg
  intro j
  have h1 : 0 ≤ u i * u i * u i * u i + u j * u j * u j * u j := by
    have h1a := rat_sq_nonneg (u i * u i)
    have h1b := rat_sq_nonneg (u j * u j)
    grind
  have h2 : 0 ≤ (u i * u i + u j * u j) / 4 := by
    have h2a := rat_sq_nonneg (u i)
    have h2b := rat_sq_nonneg (u j)
    apply rat_div_nonneg (by grind) (by grind)
  have hTerm1 := Rat.mul_nonneg (ratAbs_nonneg (C i j * vecDot (V i) (V j))) h1
  have hTerm2 := Rat.mul_nonneg (ratAbs_nonneg (C i j))
    (Rat.mul_nonneg (ratAbs_nonneg (u i * u j)) h2)
  grind

private theorem sym_quad_diag_eq {m N : Nat} (C : Fin m → Fin m → Rat) (V : Fin m → Vec N)
    (u : Fin m → Rat) (hSym : IsSymMatrix C) :
    finSum (fun i => finSum (fun j =>
      C i j * vecDot (V i) (V j) * ((1 / 2 : Rat) * (u i * u i + u j * u j)))) =
      finSum (fun i => lagrangeMult C V i * (u i * u i)) := by
  have h1 : finSum (fun i => finSum (fun j =>
      C i j * vecDot (V i) (V j) * ((1 / 2 : Rat) * (u i * u i + u j * u j)))) =
      (1 / 2 : Rat) * finSum (fun i => finSum (fun j =>
        C i j * vecDot (V i) (V j) * (u i * u i))) +
      (1 / 2 : Rat) * finSum (fun i => finSum (fun j =>
        C i j * vecDot (V i) (V j) * (u j * u j))) := by
    rw [← finSum_mul_left, ← finSum_mul_left, ← finSum_add]
    apply finSum_congr
    intro i
    rw [← finSum_mul_left, ← finSum_mul_left, ← finSum_add]
    apply finSum_congr
    intro j
    grind
  have h2 : finSum (fun i => finSum (fun j => C i j * vecDot (V i) (V j) * (u j * u j))) =
      finSum (fun i => finSum (fun j => C i j * vecDot (V i) (V j) * (u i * u i))) := by
    rw [finSum_swap (fun i j => C i j * vecDot (V i) (V j) * (u j * u j))]
    apply finSum_congr
    intro j
    apply finSum_congr
    intro i
    rw [hSym i j, vecDot_comm (V i) (V j)]
  rw [h1, h2]
  have hCombine : (1 / 2 : Rat) * finSum (fun i => finSum (fun j =>
        C i j * vecDot (V i) (V j) * (u i * u i))) +
      (1 / 2 : Rat) * finSum (fun i => finSum (fun j =>
        C i j * vecDot (V i) (V j) * (u i * u i))) =
      finSum (fun i => finSum (fun j => C i j * vecDot (V i) (V j) * (u i * u i))) := by grind
  rw [hCombine]
  apply finSum_congr
  intro i
  unfold lagrangeMult
  rw [finSum_mul_right]

theorem liftedCurve_taylor_global_bound {m N : Nat} (C : Fin m → Fin m → Rat)
    (V : Fin m → Vec N) (u : Fin m → Rat) (hSym : IsSymMatrix C) (t : Rat) :
    ratAbs (energy C (liftedCurve V u t) - energy C V -
      (1 / 2 : Rat) * quadForm (slackMatrix C V) u * (t * t)) ≤
      escapeErrorBound C V u * (t * t * t * t) := by
  have hQuadExpand : quadForm (slackMatrix C V) u =
      finSum (fun i => finSum (fun j => C i j * (u i * u j))) -
      finSum (fun i => finSum (fun j =>
        C i j * vecDot (V i) (V j) * ((1 / 2 : Rat) * (u i * u i + u j * u j)))) := by
    rw [sym_quad_diag_eq C V u hSym]
    unfold quadForm
    have hSplit : finSum (fun i => finSum (fun j => slackMatrix C V i j * u i * u j)) =
        finSum (fun i => finSum (fun j => C i j * (u i * u j))) -
        finSum (fun i => finSum (fun j =>
          (if i = j then lagrangeMult C V i else 0) * u i * u j)) := by
      rw [← finSum_sub]
      apply finSum_congr
      intro i
      rw [← finSum_sub]
      apply finSum_congr
      intro j
      unfold slackMatrix
      grind
    rw [hSplit]
    congr 1
    apply finSum_congr
    intro i
    have hRewrite : (fun j : Fin m =>
        (if i = j then lagrangeMult C V i else 0) * u i * u j) =
        (fun j : Fin m => if j = i then lagrangeMult C V i * (u i * u j) else 0) := by
      funext j
      by_cases h : i = j
      · subst h; simp; grind
      · have hj : j ≠ i := fun hji => h hji.symm
        simp [h, hj]
    rw [hRewrite, finSum_ite_eq i (fun j => lagrangeMult C V i * (u i * u j))]
  have hDiffEq : energy C (liftedCurve V u t) - energy C V -
      (1 / 2 : Rat) * quadForm (slackMatrix C V) u * (t * t) =
      (1 / 2 : Rat) * finSum (fun i => finSum (fun j =>
        C i j * vecDot (V i) (V j) *
          (cayleyCos ((t * u i) / 2) * cayleyCos ((t * u j) / 2) - 1 +
            (t * t / 2) * (u i * u i + u j * u j)) +
        C i j *
          (cayleySin ((t * u i) / 2) * cayleySin ((t * u j) / 2) - t * t * u i * u j))) := by
    rw [hQuadExpand]
    unfold energy liftedCurve
    have hExpandRHS :
        finSum (fun i => finSum (fun j =>
          C i j * vecDot (V i) (V j) *
            (cayleyCos ((t * u i) / 2) * cayleyCos ((t * u j) / 2) - 1 +
              (t * t / 2) * (u i * u i + u j * u j)) +
          C i j *
            (cayleySin ((t * u i) / 2) * cayleySin ((t * u j) / 2) - t * t * u i * u j))) =
        finSum (fun i => finSum (fun j =>
          C i j * vecDot (liftedVec (V i) (u i) t) (liftedVec (V j) (u j) t))) -
        finSum (fun i => finSum (fun j => C i j * vecDot (V i) (V j))) -
        (finSum (fun i => finSum (fun j => C i j * (u i * u j))) -
         finSum (fun i => finSum (fun j =>
          C i j * vecDot (V i) (V j) * ((1 / 2 : Rat) * (u i * u i + u j * u j))))) * (t * t) := by
      have hMinus1 :
          finSum (fun i => finSum (fun j =>
            C i j * vecDot (liftedVec (V i) (u i) t) (liftedVec (V j) (u j) t))) -
          finSum (fun i => finSum (fun j => C i j * vecDot (V i) (V j))) =
          finSum (fun i => finSum (fun j =>
            C i j * vecDot (liftedVec (V i) (u i) t) (liftedVec (V j) (u j) t) -
            C i j * vecDot (V i) (V j))) := by
        rw [← finSum_sub]
        apply finSum_congr
        intro i
        rw [← finSum_sub]
      have hTerm3a :
          finSum (fun i => finSum (fun j => C i j * (u i * u j))) * (t * t) =
          finSum (fun i => finSum (fun j => C i j * (u i * u j) * (t * t))) := by
        rw [← finSum_mul_right]
        apply finSum_congr; intro i; rw [← finSum_mul_right]
      have hTerm3b :
          finSum (fun i => finSum (fun j =>
            C i j * vecDot (V i) (V j) * ((1 / 2 : Rat) * (u i * u i + u j * u j)))) * (t * t) =
          finSum (fun i => finSum (fun j =>
            C i j * vecDot (V i) (V j) * ((1 / 2 : Rat) * (u i * u i + u j * u j)) * (t * t))) := by
        rw [← finSum_mul_right]
        apply finSum_congr; intro i; rw [← finSum_mul_right]
      have hDist : ((finSum fun i => finSum fun j => C i j * (u i * u j)) -
          finSum fun i => finSum fun j =>
            C i j * vecDot (V i) (V j) * (1 / 2 * (u i * u i + u j * u j))) * (t * t) =
          (finSum fun i => finSum fun j => C i j * (u i * u j)) * (t * t) -
          (finSum fun i => finSum fun j =>
            C i j * vecDot (V i) (V j) * (1 / 2 * (u i * u i + u j * u j))) * (t * t) := by grind
      rw [hMinus1, hDist, hTerm3a, hTerm3b]
      have hDiffAll :
          (finSum fun i => finSum fun j =>
            C i j * vecDot (liftedVec (V i) (u i) t) (liftedVec (V j) (u j) t) -
            C i j * vecDot (V i) (V j)) -
          ((finSum fun i => finSum fun j => C i j * (u i * u j) * (t * t)) -
           finSum fun i => finSum fun j =>
            C i j * vecDot (V i) (V j) * (1 / 2 * (u i * u i + u j * u j)) * (t * t)) =
          finSum fun i => finSum fun j =>
            (C i j * vecDot (liftedVec (V i) (u i) t) (liftedVec (V j) (u j) t) -
              C i j * vecDot (V i) (V j)) -
            (C i j * (u i * u j) * (t * t) -
              C i j * vecDot (V i) (V j) * (1 / 2 * (u i * u i + u j * u j)) * (t * t)) := by
        rw [← finSum_sub, ← finSum_sub]
        apply finSum_congr
        intro i
        rw [← finSum_sub, ← finSum_sub]
      rw [hDiffAll]
      apply finSum_congr
      intro i
      apply finSum_congr
      intro j
      rw [liftedVec_dot]
      grind
    rw [hExpandRHS]
    grind
  rw [hDiffEq, ratAbs_mul, ratAbs_of_nonneg (by grind : (0 : Rat) ≤ 1 / 2)]
  have hSumAbs : ratAbs (finSum (fun i => finSum (fun j =>
      C i j * vecDot (V i) (V j) *
        (cayleyCos ((t * u i) / 2) * cayleyCos ((t * u j) / 2) - 1 +
          (t * t / 2) * (u i * u i + u j * u j)) +
      C i j *
        (cayleySin ((t * u i) / 2) * cayleySin ((t * u j) / 2) - t * t * u i * u j)))) ≤
      finSum (fun i => finSum (fun j =>
        ratAbs (C i j * vecDot (V i) (V j)) * (u i * u i * u i * u i + u j * u j * u j * u j) +
        ratAbs (C i j) * (ratAbs (u i * u j) * (u i * u i + u j * u j) / 4))) *
        (t * t * t * t) := by
    have h1 := finSum_abs_le (fun i => finSum (fun j =>
      C i j * vecDot (V i) (V j) *
        (cayleyCos ((t * u i) / 2) * cayleyCos ((t * u j) / 2) - 1 +
          (t * t / 2) * (u i * u i + u j * u j)) +
      C i j *
        (cayleySin ((t * u i) / 2) * cayleySin ((t * u j) / 2) - t * t * u i * u j)))
    have h2 : finSum (fun i => ratAbs (finSum (fun j =>
        C i j * vecDot (V i) (V j) *
          (cayleyCos ((t * u i) / 2) * cayleyCos ((t * u j) / 2) - 1 +
            (t * t / 2) * (u i * u i + u j * u j)) +
        C i j *
          (cayleySin ((t * u i) / 2) * cayleySin ((t * u j) / 2) - t * t * u i * u j)))) ≤
        finSum (fun i => finSum (fun j =>
          (ratAbs (C i j * vecDot (V i) (V j)) * (u i * u i * u i * u i + u j * u j * u j * u j) +
           ratAbs (C i j) * (ratAbs (u i * u j) * (u i * u i + u j * u j) / 4)) *
            (t * t * t * t))) := by
      apply finSum_le_finSum
      intro i
      have hi1 := finSum_abs_le (fun j =>
        C i j * vecDot (V i) (V j) *
          (cayleyCos ((t * u i) / 2) * cayleyCos ((t * u j) / 2) - 1 +
            (t * t / 2) * (u i * u i + u j * u j)) +
        C i j *
          (cayleySin ((t * u i) / 2) * cayleySin ((t * u j) / 2) - t * t * u i * u j))
      have hi2 : finSum (fun j => ratAbs (
          C i j * vecDot (V i) (V j) *
            (cayleyCos ((t * u i) / 2) * cayleyCos ((t * u j) / 2) - 1 +
              (t * t / 2) * (u i * u i + u j * u j)) +
          C i j *
            (cayleySin ((t * u i) / 2) * cayleySin ((t * u j) / 2) - t * t * u i * u j))) ≤
          finSum (fun j =>
            (ratAbs (C i j * vecDot (V i) (V j)) * (u i * u i * u i * u i + u j * u j * u j * u j) +
             ratAbs (C i j) * (ratAbs (u i * u j) * (u i * u i + u j * u j) / 4)) *
              (t * t * t * t)) := by
        apply finSum_le_finSum
        intro j
        have hTri := ratAbs_add_le
          (C i j * vecDot (V i) (V j) *
            (cayleyCos ((t * u i) / 2) * cayleyCos ((t * u j) / 2) - 1 +
              (t * t / 2) * (u i * u i + u j * u j)))
          (C i j *
            (cayleySin ((t * u i) / 2) * cayleySin ((t * u j) / 2) - t * t * u i * u j))
        have hcB := cayleyCos_prod_rem_bound (u i) (u j) t
        have hsB := cayleySin_prod_rem_bound (u i) (u j) t
        have hMulC := rat_mul_le_mul_of_nonneg_left hcB (ratAbs_nonneg (C i j * vecDot (V i) (V j)))
        have hMulS := rat_mul_le_mul_of_nonneg_left hsB (ratAbs_nonneg (C i j))
        rw [← ratAbs_mul] at hMulC hMulS
        grind
      exact Rat.le_trans hi1 hi2
    have hFactor : finSum (fun i => finSum (fun j =>
        (ratAbs (C i j * vecDot (V i) (V j)) * (u i * u i * u i * u i + u j * u j * u j * u j) +
         ratAbs (C i j) * (ratAbs (u i * u j) * (u i * u i + u j * u j) / 4)) *
          (t * t * t * t))) =
        finSum (fun i => finSum (fun j =>
          ratAbs (C i j * vecDot (V i) (V j)) * (u i * u i * u i * u i + u j * u j * u j * u j) +
          ratAbs (C i j) * (ratAbs (u i * u j) * (u i * u i + u j * u j) / 4))) *
          (t * t * t * t) := by
      rw [← finSum_mul_right]
      apply finSum_congr
      intro i
      rw [← finSum_mul_right]
    rw [← hFactor]
    exact Rat.le_trans h1 h2
  have hHalf := rat_mul_le_mul_of_nonneg_left hSumAbs (by grind : (0 : Rat) ≤ 1 / 2)
  unfold escapeErrorBound
  grind

/--
**定理 3（`burer_monteiro_plus_one_dim_escape`）**：
假定 `C` 对称、`V : Fin m → Vec N` 为单位球面配置，且对偶松弛矩阵在方向 `u` 上存在
严格负二次型 `q := quadForm (slackMatrix C V) u < 0`。令 `E(t) := energy C (liftedCurve V u t)`，则：
1. **全程保范**：`∀ t : Rat, IsUnitConfig (liftedCurve V u t)` 且 `E(0) = energy C V`；
2. **一阶导数归零（驻点性质）**：`t = 0` 处经典 $\varepsilon$-$\delta$ 导数精确为 `0`；
3. **二阶导数精确等于 `u^T S(V) u < 0`**：`t = 0` 处对称二阶差商的 $\varepsilon$-$\delta$ 极限精确等于 `q`
   （同时给出单侧差商 `2 * (E(t) - E(0)) / t²` 极限 `q` 与全局四次余项界）；
4. **严格能量逃逸通道**：存在显式有理步长 `ε > 0` 使 `E(ε) < energy C V`。
-/
theorem burer_monteiro_plus_one_dim_escape {m N : Nat}
    (C : Fin m → Fin m → Rat) (V : Fin m → Vec N) (u : Fin m → Rat)
    (hSym : IsSymMatrix C) (hV : IsUnitConfig V)
    (hNeg : quadForm (slackMatrix C V) u < 0) :
    let E : Rat → Rat := fun t => energy C (liftedCurve V u t)
    let q : Rat := quadForm (slackMatrix C V) u
    (∀ t : Rat, IsUnitConfig (liftedCurve V u t)) ∧
    E 0 = energy C V ∧
    (∀ eps : Rat, 0 < eps → ∃ delta : Rat, 0 < delta ∧
      ∀ t : Rat, t ≠ 0 → ratAbs t < delta → ratAbs ((E t - E 0) / t - 0) < eps) ∧
    (∀ eps : Rat, 0 < eps → ∃ delta : Rat, 0 < delta ∧
      ∀ t : Rat, t ≠ 0 → ratAbs t < delta →
        ratAbs ((E t - 2 * E 0 + E (-t)) / (t * t) - q) < eps ∧
        ratAbs (2 * (E t - E 0) / (t * t) - q) < eps) ∧
    (∃ epsStep : Rat, 0 < epsStep ∧ E epsStep < energy C V) := by
  intro E q
  let B := escapeErrorBound C V u
  have hB : 0 ≤ B := escapeErrorBound_nonneg C V u
  have hE0 : E 0 = energy C V := liftedCurve_energy_zero C V u
  have hTay : ∀ t : Rat, ratAbs (E t - E 0 - (1 / 2 : Rat) * q * (t * t)) ≤ B * (t * t * t * t) := by
    intro t
    rw [hE0]
    exact liftedCurve_taylor_global_bound C V u hSym t
  refine ⟨liftedCurve_unit V u hV, hE0, ?_, ?_, ?_⟩
  · intro eps hEps
    let denom : Rat := ratAbs q + B + 1
    have hDenPos : 0 < denom := by
      have hqa := ratAbs_nonneg q
      dsimp [denom]
      grind
    let delta : Rat := min (1 : Rat) (eps / denom)
    have hDeltaPos : 0 < delta := by
      have hDivPos := rat_div_pos hEps hDenPos
      dsimp [delta]
      grind
    refine ⟨delta, hDeltaPos, ?_⟩
    intro t htNe htLt
    have htAbsPos : 0 < ratAbs t := by
      have h0 := ratAbs_nonneg t
      have hNe : ratAbs t ≠ 0 := fun h => htNe ((ratAbs_eq_zero_iff t).mp h)
      grind
    have htLt1 : ratAbs t < 1 := by dsimp [delta] at htLt; grind
    have htLtDiv : ratAbs t < eps / denom := by dsimp [delta] at htLt; grind
    have hSubZero : (E t - E 0) / t - 0 = (E t - E 0) / t := by grind
    rw [hSubZero]
    have hRem := hTay t
    let rem := E t - E 0 - (1 / 2 : Rat) * q * (t * t)
    have hDecomp : (E t - E 0) / t = (1 / 2 : Rat) * q * t + rem / t := by
      dsimp [rem]
      rw [Rat.div_def, Rat.div_def]
      have hInv : t * t⁻¹ = 1 := Rat.mul_inv_cancel t htNe
      grind
    rw [hDecomp]
    have hTri := ratAbs_add_le ((1 / 2 : Rat) * q * t) (rem / t)
    have hTerm1 : ratAbs ((1 / 2 : Rat) * q * t) ≤ ratAbs q * ratAbs t := by
      rw [ratAbs_mul, ratAbs_mul, ratAbs_of_nonneg (by grind : (0 : Rat) ≤ 1 / 2)]
      have hqa := ratAbs_nonneg q
      have hta := Rat.le_of_lt htAbsPos
      have h1 : (1 / 2 : Rat) * ratAbs q * ratAbs t ≤ ratAbs q * ratAbs t := by
        have hle : (1 / 2 : Rat) * ratAbs q ≤ ratAbs q := by grind
        exact rat_mul_le_mul_of_nonneg_right hle hta
      exact h1
    have hTerm2 : ratAbs (rem / t) ≤ B * ratAbs t := by
      have hAbsDiv : ratAbs (rem / t) = ratAbs rem / ratAbs t := by
        rw [Rat.div_def, ratAbs_mul]
        have hInvAbs : ratAbs t⁻¹ = (ratAbs t)⁻¹ := by
          have h1 : ratAbs t * ratAbs t⁻¹ = 1 := by
            rw [← ratAbs_mul, Rat.mul_inv_cancel t htNe]
            decide
          have hL := congrArg (fun z => (ratAbs t)⁻¹ * z) h1
          have hNe : ratAbs t ≠ 0 := by grind
          have hAssoc : (ratAbs t)⁻¹ * (ratAbs t * ratAbs t⁻¹) = (ratAbs t * (ratAbs t)⁻¹) * ratAbs t⁻¹ := by grind
          rw [hAssoc, Rat.mul_inv_cancel _ hNe] at hL
          grind
        rw [hInvAbs, ← Rat.div_def]
      rw [hAbsDiv, rat_div_le_iff htAbsPos]
      have hPow4 : t * t * t * t = ratAbs t * ratAbs t * ratAbs t * ratAbs t := by
        have hsq : t * t = ratAbs t * ratAbs t := by
          rw [← ratAbs_mul, ratAbs_of_nonneg (rat_sq_nonneg t)]
        grind
      have hCube : ratAbs t * ratAbs t * ratAbs t ≤ ratAbs t := by
        have h2 : ratAbs t * ratAbs t ≤ 1 * ratAbs t :=
          rat_mul_le_mul_of_nonneg_right (Rat.le_of_lt htLt1) (Rat.le_of_lt htAbsPos)
        have h3 : (ratAbs t * ratAbs t) * ratAbs t ≤ ratAbs t * ratAbs t := by
          have h1 := rat_mul_le_mul_of_nonneg_right h2 (Rat.le_of_lt htAbsPos)
          grind
        grind
      have hBCube : B * (ratAbs t * ratAbs t * ratAbs t * ratAbs t) ≤ (B * ratAbs t) * ratAbs t := by
        have h1 := rat_mul_le_mul_of_nonneg_right hCube (Rat.le_of_lt htAbsPos)
        have h2 := rat_mul_le_mul_of_nonneg_left h1 hB
        grind
      rw [hPow4] at hRem
      exact Rat.le_trans hRem hBCube
    have hBound1 : ratAbs ((1 / 2 : Rat) * q * t + rem / t) ≤ (ratAbs q + B) * ratAbs t := by
      grind
    have hBound2 : (ratAbs q + B) * ratAbs t < denom * (eps / denom) := by
      have h1 : (ratAbs q + B) * ratAbs t ≤ denom * ratAbs t := by
        apply rat_mul_le_mul_of_nonneg_right
        · dsimp [denom]; grind
        · exact Rat.le_of_lt htAbsPos
      have h2 : denom * ratAbs t < denom * (eps / denom) :=
        Rat.mul_lt_mul_of_pos_left htLtDiv hDenPos
      exact rat_lt_of_le_of_lt h1 h2
    have hDenEq : denom * (eps / denom) = eps := by
      rw [Rat.div_def]
      have hAssoc : denom * (eps * denom⁻¹) = eps * (denom * denom⁻¹) := by grind
      rw [hAssoc, Rat.mul_inv_cancel denom (by grind)]
      grind
    rw [hDenEq] at hBound2
    exact rat_lt_of_le_of_lt hBound1 hBound2
  · intro eps hEps
    let denom : Rat := 2 * B + 1
    have hDenPos : 0 < denom := by dsimp [denom]; grind
    let delta : Rat := min (1 : Rat) (eps / denom)
    have hDeltaPos : 0 < delta := by
      have hDivPos := rat_div_pos hEps hDenPos
      dsimp [delta]
      grind
    refine ⟨delta, hDeltaPos, ?_⟩
    intro t htNe htLt
    have htAbsPos : 0 < ratAbs t := by
      have h0 := ratAbs_nonneg t
      have hNe : ratAbs t ≠ 0 := fun h => htNe ((ratAbs_eq_zero_iff t).mp h)
      grind
    have htLt1 : ratAbs t < 1 := by dsimp [delta] at htLt; grind
    have htLtDiv : ratAbs t < eps / denom := by dsimp [delta] at htLt; grind
    have hEven : E (-t) = E t := liftedCurve_energy_even C V u t
    have hSame : (E t - 2 * E 0 + E (-t)) / (t * t) = 2 * (E t - E 0) / (t * t) := by
      rw [hEven]
      grind
    have hSingle : ratAbs (2 * (E t - E 0) / (t * t) - q) < eps := by
      have hRem := hTay t
      have ht2Pos : 0 < t * t := by
        rcases (show t < 0 ∨ 0 < t by grind) with ht | ht
        · exact rat_mul_pos_of_neg_of_neg ht ht
        · exact Rat.mul_pos ht ht
      have ht2Ne : t * t ≠ 0 := by grind
      have hEq : 2 * (E t - E 0) / (t * t) - q =
          (2 * (E t - E 0 - (1 / 2 : Rat) * q * (t * t))) / (t * t) := by
        rw [Rat.div_def, Rat.div_def]
        have hInv : (t * t) * (t * t)⁻¹ = 1 := Rat.mul_inv_cancel _ ht2Ne
        grind
      rw [hEq, ratAbs_div _ _ ht2Pos, ratAbs_mul, ratAbs_of_nonneg (by grind : (0 : Rat) ≤ 2)]
      have hPow2 : t * t ≤ ratAbs t := by
        have hsq : t * t = ratAbs t * ratAbs t := by
          rw [← ratAbs_mul, ratAbs_of_nonneg (rat_sq_nonneg t)]
        have h1 : ratAbs t * ratAbs t ≤ 1 * ratAbs t :=
          rat_mul_le_mul_of_nonneg_right (Rat.le_of_lt htLt1) (Rat.le_of_lt htAbsPos)
        grind
      have hDivLe : 2 * ratAbs (E t - E 0 - (1 / 2 : Rat) * q * (t * t)) / (t * t) ≤
          2 * B * (t * t) := by
        rw [rat_div_le_iff ht2Pos]
        have h1 := rat_mul_le_mul_of_nonneg_left hRem (by grind : (0 : Rat) ≤ 2)
        grind
      have hBound2 : 2 * B * (t * t) ≤ 2 * B * ratAbs t :=
        rat_mul_le_mul_of_nonneg_left hPow2 (by grind : 0 ≤ 2 * B)
      have hBound3 : 2 * B * ratAbs t < denom * (eps / denom) := by
        have h1 : 2 * B * ratAbs t ≤ denom * ratAbs t :=
          rat_mul_le_mul_of_nonneg_right (by dsimp [denom]; grind) (Rat.le_of_lt htAbsPos)
        have h2 : denom * ratAbs t < denom * (eps / denom) :=
          Rat.mul_lt_mul_of_pos_left htLtDiv hDenPos
        exact rat_lt_of_le_of_lt h1 h2
      have hDenEq : denom * (eps / denom) = eps := by
        rw [Rat.div_def]
        have hAssoc : denom * (eps * denom⁻¹) = eps * (denom * denom⁻¹) := by grind
        rw [hAssoc, Rat.mul_inv_cancel denom (by grind)]
        grind
      rw [hDenEq] at hBound3
      exact rat_lt_of_le_of_lt (Rat.le_trans hDivLe hBound2) hBound3
    refine ⟨?_, hSingle⟩
    rw [hSame]
    exact hSingle
  · let epsStep : Rat := min (1 : Rat) ((-q) / (4 * (B + 1)))
    have h4BPos : 0 < 4 * (B + 1) := by grind
    have hNegPos : 0 < -q := by dsimp [q] at hNeg ⊢; grind
    have hDivPos : 0 < (-q) / (4 * (B + 1)) := rat_div_pos hNegPos h4BPos
    have hStepPos : 0 < epsStep := by dsimp [epsStep]; grind
    have hStepLe1 : epsStep ≤ 1 := by dsimp [epsStep]; grind
    have hStepLeDiv : epsStep ≤ (-q) / (4 * (B + 1)) := by dsimp [epsStep]; grind
    have hStep2Pos : 0 < epsStep * epsStep := Rat.mul_pos hStepPos hStepPos
    have hRem := hTay epsStep
    have hUpper : E epsStep - E 0 ≤
        (1 / 2 : Rat) * q * (epsStep * epsStep) + B * (epsStep * epsStep * epsStep * epsStep) := by
      rw [ratAbs_le_iff] at hRem
      grind
    have hCubeLe : epsStep * epsStep * epsStep * epsStep ≤ epsStep * epsStep * epsStep := by
      have h1 : epsStep * epsStep ≤ 1 * epsStep :=
        rat_mul_le_mul_of_nonneg_right hStepLe1 (Rat.le_of_lt hStepPos)
      have h2 := rat_mul_le_mul_of_nonneg_right h1 (Rat.le_of_lt hStep2Pos)
      grind
    have hBStep : B * (epsStep * epsStep * epsStep * epsStep) ≤
        (B + 1) * (epsStep * epsStep * epsStep) := by
      have h1 := rat_mul_le_mul_of_nonneg_left hCubeLe hB
      have h2 : B * (epsStep * epsStep * epsStep) ≤ (B + 1) * (epsStep * epsStep * epsStep) := by
        apply rat_mul_le_mul_of_nonneg_right (by grind)
        exact Rat.mul_nonneg (Rat.le_of_lt hStep2Pos) (Rat.le_of_lt hStepPos)
      exact Rat.le_trans h1 h2
    have hLinear : (B + 1) * epsStep ≤ (-q) / 4 := by
      rw [rat_le_div_iff h4BPos] at hStepLeDiv
      rw [rat_le_div_iff (by grind : (0 : Rat) < 4)]
      grind
    have hMid : (1 / 2 : Rat) * q + (B + 1) * epsStep ≤ q / 4 := by grind
    have hMidMul := rat_mul_le_mul_of_nonneg_left hMid (Rat.le_of_lt hStep2Pos)
    have hNegFinal : (epsStep * epsStep) * (q / 4) < 0 :=
      rat_mul_neg_of_pos_of_neg hStep2Pos (by grind : q / 4 < 0)
    have hComb : E epsStep - E 0 ≤ (epsStep * epsStep) * (q / 4) := by
      have h1 : E epsStep - E 0 ≤
          (epsStep * epsStep) * ((1 / 2 : Rat) * q + (B + 1) * epsStep) := by
        grind
      exact Rat.le_trans h1 hMidMul
    have hDiffNeg : E epsStep - E 0 < 0 := rat_lt_of_le_of_lt hComb hNegFinal
    refine ⟨epsStep, hStepPos, ?_⟩
    rw [← hE0]
    grind

/-!
### 5. 诚实边界与反例：受挫反铁磁三角形的 1D 离散最优 vs 2D 连续逃逸 (`burer_monteiro_frustrated_triangle_gap`)
-/

/-- 3 节点反铁磁三角形耦合矩阵：`C_{ij} = 1 (i ≠ j), 0 (i = j)`。 -/
def triC (i j : Fin 3) : Rat := if i = j then 0 else 1

/-- 1D 离散最优配置 `s = (1, 1, -1)`。 -/
def triS1 (i : Fin 3) : Rat := if i.val = 2 then -1 else 1

/-- 1D 离散最优点在 `N = 1` 的嵌入。 -/
def triV1 : Fin 3 → Vec 1 := rank1Embed (by decide) triS1

/-- 使 `slackMatrix triC triV1` 出现严格负二次型的逃逸方向 `u = (1, -1, 0)`。 -/
def triU (i : Fin 3) : Rat := if i.val = 0 then 1 else if i.val = 1 then -1 else 0

/-- 升至 2D（$\mathbb{S}^1$）后的显式有理单位向量逃逸配置：`(1,0), (-3/5, 4/5), (-3/5, -4/5)`。 -/
def triW2 (i : Fin 3) (k : Fin 2) : Rat :=
  if i.val = 0 then (if k.val = 0 then 1 else 0)
  else if i.val = 1 then (if k.val = 0 then -3 / 5 else 4 / 5)
  else (if k.val = 0 then -3 / 5 else -4 / 5)

private theorem finSum_two_eq (f : Fin 2 → Rat) : finSum f = f 0 + f 1 := by
  simp [finSum]
  grind

private theorem finSum_three_eq (f : Fin 3 → Rat) : finSum f = f 0 + f 1 + f 2 := by
  simp [finSum]
  grind

/--
**定理 4（`burer_monteiro_frustrated_triangle_gap`，诚实证伪与边界）**：
1. 对一切 `s : Fin 3 → {±1}`，1D 离散三角形能量恒有下界 `isingEnergy triC s ≥ -1`；
2. `triV1` 达到 1D 离散最优 `energy triC triV1 = -1`，然而其对偶松弛矩阵在方向 `triU` 上
   取严格负值 `quadForm (slackMatrix triC triV1) triU = -2 < 0`；
3. 抬升至 2D 单位圆后，显式有理配置 `triW2` 严格满足单位范数 `IsUnitConfig triW2`，且
   `energy triC triW2 = -37 / 25 < -1`——诚实刻画升维带来的 SDP 整合间隙（无约束高维能量低于原离散最优）。
-/
theorem burer_monteiro_frustrated_triangle_gap :
    IsSymMatrix triC ∧
    (∀ s : Fin 3 → Rat, (∀ i : Fin 3, s i = 1 ∨ s i = -1) → -1 ≤ isingEnergy triC s) ∧
    (IsUnitConfig triV1 ∧ energy triC triV1 = -1) ∧
    quadForm (slackMatrix triC triV1) triU = -2 ∧
    (IsUnitConfig triW2 ∧ energy triC triW2 = -37 / 25 ∧ energy triC triW2 < -1) := by
  have hSym : IsSymMatrix triC := by
    intro i j
    unfold triC
    by_cases h : i = j
    · subst h; simp
    · have hj : j ≠ i := fun hji => h hji.symm
      simp [h, hj]
  have h01 : (0 : Fin 3) ≠ 1 := by decide
  have h02 : (0 : Fin 3) ≠ 2 := by decide
  have h10 : (1 : Fin 3) ≠ 0 := by decide
  have h12 : (1 : Fin 3) ≠ 2 := by decide
  have h20 : (2 : Fin 3) ≠ 0 := by decide
  have h21 : (2 : Fin 3) ≠ 1 := by decide
  refine ⟨hSym, ?_, ?_, ?_, ?_⟩
  · intro s hs
    have hFormula : isingEnergy triC s = s 0 * s 1 + s 0 * s 2 + s 1 * s 2 := by
      unfold isingEnergy
      rw [finSum_three_eq]
      simp only [finSum_three_eq, triC, h01, h02, h10, h12, h20, h21]
      grind
    rw [hFormula]
    rcases hs 0 with h0 | h0 <;> rcases hs 1 with h1 | h1 <;> rcases hs 2 with h2 | h2 <;>
      rw [h0, h1, h2] <;> grind
  · have hs : ∀ i : Fin 3, triS1 i = 1 ∨ triS1 i = -1 := by
      intro i
      rcases i with ⟨v, hv⟩
      match v with
      | 0 | 1 | 2 => simp [triS1]
      | _ + 3 => omega
    have hRank1 := energy_rank1_eq_ising (by omega : 1 ≤ 1) triC triS1 hs
    refine ⟨hRank1.1, ?_⟩
    unfold triV1
    rw [hRank1.2]
    unfold isingEnergy
    simp only [finSum_three_eq, triC, triS1, h01, h02, h10, h12, h20, h21]
    grind
  · have hDot : ∀ i j : Fin 3, vecDot (triV1 i) (triV1 j) = triS1 i * triS1 j := by
      intro i j
      unfold triV1
      exact rank1Embed_dot (by omega : 1 ≤ 1) triS1 i j
    unfold quadForm slackMatrix lagrangeMult
    simp only [finSum_three_eq, hDot, triC, triS1, triU, h01, h02, h10, h12, h20, h21]
    grind
  · have hDot2 : ∀ i j : Fin 3, vecDot (triW2 i) (triW2 j) =
        triW2 i 0 * triW2 j 0 + triW2 i 1 * triW2 j 1 := by
      intro i j
      unfold vecDot
      exact finSum_two_eq _
    constructor
    · intro i
      unfold vecNormSq
      rw [hDot2 i i]
      rcases i with ⟨v, hv⟩
      match v with
      | 0 => simp [triW2]; grind
      | 1 => simp [triW2]; grind
      | 2 => simp [triW2]; grind
      | _ + 3 => omega
    · have hEn : energy triC triW2 = -37 / 25 := by
        unfold energy
        simp only [finSum_three_eq, hDot2, triC, triW2, h01, h02, h10, h12, h20, h21]
        grind
      exact ⟨hEn, by rw [hEn]; grind⟩

end PhaseSHA.V5.BurerMonteiroEscape
