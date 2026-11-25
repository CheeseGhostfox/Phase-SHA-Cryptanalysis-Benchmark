/-!
# NonlinearGuardBlocking: 非线性守卫阻塞 GF(2) 单变量推导定理

对应 `v10_ouroboros64/ouroboros_core/cpp/crystal_core.hpp:1193-1257` 与 `cnf_cdcl.py:367-425`：
证明当系统中存在自由非线性守卫子集 `G`（`x ∉ G`）在每条线性方程中恰好吸收目标变量 `x` 的奇偶度时，
对任意满足赋值 `σ`，同时翻转 `x :: G` 所得新赋值 `flipSet σ (x :: G)` 仍满足整个线性系统，
从而任何可靠 GF(2) 消元器均无法导出单变量常量约束 `x = c`。
-/

namespace OuroborosV10.NonlinearGuardBlocking

/-- GF(2) 线性方程：`⨁_{v ∈ vars} σ(v) = rhs`。 -/
structure LinEq where
  vars : List Nat
  rhs : Bool
  deriving DecidableEq, Repr

/-- 计算线性方程左边多项式的奇偶求值。 -/
def evalLhs (σ : Nat → Bool) (vs : List Nat) : Bool :=
  vs.foldl (fun acc v => acc ^^ σ v) false

/-- 判断赋值 `σ` 是否满足单条线性方程 `eq`。 -/
def evalLinEq (σ : Nat → Bool) (eq : LinEq) : Bool :=
  evalLhs σ eq.vars == eq.rhs

/-- 判断赋值 `σ` 是否满足整个线性方程组 `sys`。 -/
def satisfiesSystem (σ : Nat → Bool) (sys : List LinEq) : Bool :=
  sys.all (evalLinEq σ)

/-- 翻转集合 `S` 中全部变量的布尔赋值。 -/
def flipSet (σ : Nat → Bool) (S : List Nat) (v : Nat) : Bool :=
  if v ∈ S then !σ v else σ v

/-- 变量 `x` 在方程 `eq` 中的出现奇偶度。 -/
def parityIn (eq : LinEq) (x : Nat) : Bool :=
  eq.vars.foldl (fun acc u => acc ^^ decide (u = x)) false

/-- 守卫集合 `G` 作为整体在方程 `eq` 中的出现奇偶度。 -/
def paritySumIn (eq : LinEq) (G : List Nat) : Bool :=
  eq.vars.foldl (fun acc u => acc ^^ decide (u ∈ G)) false

private theorem foldl_xor_init (f : Nat → Bool) (vs : List Nat) (b : Bool) :
    (vs.foldl (fun acc u => acc ^^ f u) b) =
      (b ^^ (vs.foldl (fun acc u => acc ^^ f u) false)) := by
  induction vs generalizing b with
  | nil => simp
  | cons v vs ih =>
    simp only [List.foldl_cons]
    rw [ih (b ^^ f v), ih (false ^^ f v)]
    cases b <;> cases f v <;> cases (vs.foldl (fun acc u => acc ^^ f u) false) <;> rfl

/-- 翻转由谓词 `P` 选中的变量后，`evalLhs` 恰好异或上 `P` 的奇偶贡献。 -/
private theorem evalLhs_flipPred (σ : Nat → Bool) (P : Nat → Bool) (vs : List Nat) :
    (evalLhs (fun v => if P v then !σ v else σ v) vs) =
      ((evalLhs σ vs) ^^ (vs.foldl (fun acc u => acc ^^ P u) false)) := by
  induction vs with
  | nil => simp [evalLhs]
  | cons v vs ih =>
    simp only [evalLhs, List.foldl_cons] at *
    rw [foldl_xor_init (fun u => if P u then !σ u else σ u) vs]
    rw [foldl_xor_init σ vs]
    rw [foldl_xor_init P vs]
    rw [ih]
    cases P v <;> cases σ v <;>
      cases (vs.foldl (fun acc u => acc ^^ σ u) false) <;>
      cases (vs.foldl (fun acc u => acc ^^ P u) false) <;> rfl

/-- `x ∉ G` 时，`u ∈ x :: G` 的布尔指示器等于 `(u = x) ^^ (u ∈ G)`。 -/
private theorem mem_cons_xor_eq (x : Nat) (G : List Nat) (hx : x ∉ G) (u : Nat) :
    (decide (u ∈ x :: G)) = (decide (u = x) ^^ decide (u ∈ G)) := by
  by_cases hux : u = x
  · subst hux
    simp [hx]
  · simp [List.mem_cons, hux]

/-- 两个按位指示器 `f, g` 的列表异或和等于各自异或和的异或。 -/
private theorem foldl_xor_split (f g : Nat → Bool) (vs : List Nat) :
    (vs.foldl (fun acc u => acc ^^ (f u ^^ g u)) false) =
      ((vs.foldl (fun acc u => acc ^^ f u) false) ^^
        (vs.foldl (fun acc u => acc ^^ g u) false)) := by
  induction vs with
  | nil => simp
  | cons v vs ih =>
    simp only [List.foldl_cons]
    rw [foldl_xor_init (fun u => f u ^^ g u) vs]
    rw [foldl_xor_init f vs]
    rw [foldl_xor_init g vs]
    rw [ih]
    cases f v <;> cases g v <;>
      cases (vs.foldl (fun acc u => acc ^^ f u) false) <;>
      cases (vs.foldl (fun acc u => acc ^^ g u) false) <;> rfl

/-- 守卫奇偶度匹配时，翻转 `x :: G` 保持整套线性方程组的可满足性。 -/
theorem guarded_eq_preserves_sat (σ : Nat → Bool) (sys : List LinEq)
    (x : Nat) (G : List Nat) (hx : x ∉ G)
    (hGuard : ∀ eq ∈ sys, parityIn eq x = paritySumIn eq G)
    (hSat : satisfiesSystem σ sys = true) :
    satisfiesSystem (flipSet σ (x :: G)) sys = true := by
  rw [satisfiesSystem, List.all_eq_true] at hSat ⊢
  intro eq heq
  have hOrig := hSat eq heq
  have hPar := hGuard eq heq
  have hflip :
      evalLhs (flipSet σ (x :: G)) eq.vars = evalLhs σ eq.vars := by
    have h1 := evalLhs_flipPred σ (fun u => decide (u ∈ x :: G)) eq.vars
    have hflipDef :
        (fun v => if decide (v ∈ x :: G) = true then !σ v else σ v) =
          flipSet σ (x :: G) := by
      funext v
      simp [flipSet]
    rw [hflipDef] at h1
    have h2 :
        (eq.vars.foldl (fun acc u => acc ^^ decide (u ∈ x :: G)) false) =
          (parityIn eq x ^^ paritySumIn eq G) := by
      simp only [mem_cons_xor_eq x G hx]
      exact foldl_xor_split (fun u => decide (u = x)) (fun u => decide (u ∈ G)) eq.vars
    rw [h2, hPar, Bool.xor_self, Bool.xor_false] at h1
    exact h1
  simpa [evalLinEq, hflip] using hOrig

/--
非线性守卫阻塞定理：
若线性系统 `sys` 存在满足赋值 `σ`，且存在自由守卫子集 `G`（`x ∉ G`）在每条方程中吸收 `x` 的奇偶度，
则不存在常数 `c : Bool` 使得所有满足赋值都强制 `τ x = c`（即任何可靠 GF(2) 推导器无法导出 `x = c`）。
-/
theorem nonlinear_guard_blocks_unit_deduction (σ : Nat → Bool) (sys : List LinEq)
    (x : Nat) (G : List Nat) (hx : x ∉ G)
    (hGuard : ∀ eq ∈ sys, parityIn eq x = paritySumIn eq G)
    (hSat : satisfiesSystem σ sys = true) (c : Bool) :
    ¬ (∀ τ : Nat → Bool, satisfiesSystem τ sys = true → τ x = c) := by
  intro hAll
  have h1 := hAll σ hSat
  have hSatFlip := guarded_eq_preserves_sat σ sys x G hx hGuard hSat
  have h2 := hAll (flipSet σ (x :: G)) hSatFlip
  have hflipX : flipSet σ (x :: G) x = !σ x := by
    simp [flipSet]
  rw [hflipX, h1] at h2
  cases c <;> simp at h2

end OuroborosV10.NonlinearGuardBlocking
