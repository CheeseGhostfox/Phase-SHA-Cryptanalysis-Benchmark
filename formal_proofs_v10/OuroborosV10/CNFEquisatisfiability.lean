/-!
# CNFEquisatisfiability: 单位文字注入、2-Literal 等价替换与 FLP 探测的等价可满足性

对应 `v10_ouroboros64/ouroboros_mcts/dynamic_cdcl.py:321-357`、
`v10_ouroboros64/ouroboros_mcts/wormhole_mcts.py:582-607` 及方案 P2/P3 桥接逻辑：
证明以下四类预处理与探测归约严格保持 CNF 公式的可满足性等价（Equisatisfiability）：
1. 语义蕴涵单位文字注入 `unit_inject_equisat`；
2. 语义蕴涵 2-literal 等价子句注入 `equiv_2lit_inject_equisat`；
3. 语义蕴涵变量等价全公式替换 `equiv_subst_equisat`；
4. 单比特与双比特 FLP 矛盾归约 `flp_unit_sound`、`flp_pair_clause_sound`。
-/

namespace OuroborosV10.CNFEquisatisfiability

/-- DIMACS 布尔文字。 -/
inductive Lit where
  | pos (v : Nat)
  | neg (v : Nat)
  deriving DecidableEq, Repr

/-- 文字取反。 -/
def Lit.negate : Lit → Lit
  | .pos v => .neg v
  | .neg v => .pos v

/-- 提取文字底层变量索引。 -/
def Lit.var : Lit → Nat
  | .pos v => v
  | .neg v => v

abbrev Clause := List Lit
abbrev Formula := List Clause
abbrev Env := Nat → Bool

/-- 文字在赋值 `σ` 下的真值。 -/
def evalLit (σ : Env) : Lit → Bool
  | .pos v => σ v
  | .neg v => !σ v

/-- 子句在赋值 `σ` 下的真值（析取）。 -/
def evalClause (σ : Env) (c : Clause) : Bool :=
  c.any (evalLit σ)

/-- CNF 公式在赋值 `σ` 下的真值（合取）。 -/
def evalFormula (σ : Env) (F : Formula) : Bool :=
  F.all (evalClause σ)

/-- 公式可满足性。 -/
def Satisfiable (F : Formula) : Prop :=
  ∃ σ : Env, evalFormula σ F = true

/-- 语义蕴涵单位文字 `l`。 -/
def EntailsLit (F : Formula) (l : Lit) : Prop :=
  ∀ σ : Env, evalFormula σ F = true → evalLit σ l = true

/-- 语义蕴涵两变量等价 `x ↔ y`。 -/
def EntailsEqv (F : Formula) (x y : Nat) : Prop :=
  ∀ σ : Env, evalFormula σ F = true → σ x = σ y

/-- 2-literal 等价子句 `[[-x, y], [x, -y]]`。 -/
def equivClauses (x y : Nat) : Formula :=
  [[Lit.neg x, Lit.pos y], [Lit.pos x, Lit.neg y]]

/-- 将文字中变量 `y` 替换为 `x`。 -/
def substVarLit (x y : Nat) : Lit → Lit
  | .pos v => if v = y then .pos x else .pos v
  | .neg v => if v = y then .neg x else .neg v

/-- 将子句中变量 `y` 替换为 `x`。 -/
def substVarClause (x y : Nat) (c : Clause) : Clause :=
  c.map (substVarLit x y)

/-- 将公式中变量 `y` 替换为 `x`。 -/
def substVarFormula (x y : Nat) (F : Formula) : Formula :=
  F.map (substVarClause x y)

@[simp] theorem evalLit_negate (σ : Env) (l : Lit) :
    evalLit σ l.negate = !evalLit σ l := by
  cases l <;> simp [Lit.negate, evalLit]

theorem evalFormula_append (σ : Env) (F1 F2 : Formula) :
    evalFormula σ (F1 ++ F2) = (evalFormula σ F1 && evalFormula σ F2) := by
  simp [evalFormula, List.all_append]

/-- 1. 单位文字注入保持可满足性等价。 -/
theorem unit_inject_equisat (F : Formula) (l : Lit) (h : EntailsLit F l) :
    Satisfiable F ↔ Satisfiable ([l] :: F) := by
  constructor
  · rintro ⟨σ, hσ⟩
    refine ⟨σ, ?_⟩
    have hl := h σ hσ
    simpa [evalFormula, evalClause, hl] using hσ
  · rintro ⟨σ, hσ⟩
    refine ⟨σ, ?_⟩
    simp only [evalFormula, List.all_cons, Bool.and_eq_true] at hσ
    exact hσ.2

/-- 2. 2-literal 等价子句注入保持可满足性等价。 -/
theorem equiv_2lit_inject_equisat (F : Formula) (x y : Nat) (h : EntailsEqv F x y) :
    Satisfiable F ↔ Satisfiable (equivClauses x y ++ F) := by
  constructor
  · rintro ⟨σ, hσ⟩
    refine ⟨σ, ?_⟩
    have hxy := h σ hσ
    rw [evalFormula_append]
    have heqv : evalFormula σ (equivClauses x y) = true := by
      simp [equivClauses, evalFormula, evalClause, evalLit, hxy]
    simp [heqv, hσ]
  · rintro ⟨σ, hσ⟩
    refine ⟨σ, ?_⟩
    rw [evalFormula_append] at hσ
    simp only [Bool.and_eq_true] at hσ
    exact hσ.2

private theorem evalLit_substVarLit_eq (σ : Env) (x y : Nat) (hxy : σ x = σ y) (l : Lit) :
    evalLit σ (substVarLit x y l) = evalLit σ l := by
  cases l with
  | pos v =>
    by_cases hv : v = y
    · subst hv; simp [substVarLit, evalLit, hxy]
    · simp [substVarLit, evalLit, hv]
  | neg v =>
    by_cases hv : v = y
    · subst hv; simp [substVarLit, evalLit, hxy]
    · simp [substVarLit, evalLit, hv]

private theorem evalFormula_substVarFormula_eq (σ : Env) (x y : Nat) (hxy : σ x = σ y)
    (F : Formula) :
    evalFormula σ (substVarFormula x y F) = evalFormula σ F := by
  simp only [evalFormula, substVarFormula, List.all_map]
  congr 1
  funext c
  simp only [Function.comp_apply, evalClause, substVarClause, List.any_map]
  congr 1
  funext l
  simp [evalLit_substVarLit_eq σ x y hxy l]

private theorem evalLit_override_subst (τ : Env) (x y : Nat) (_hne : x ≠ y) (l : Lit) :
    evalLit (fun z => if z = y then τ x else τ z) l = evalLit τ (substVarLit x y l) := by
  cases l with
  | pos v =>
    by_cases hv : v = y
    · subst hv; simp [substVarLit, evalLit]
    · simp [substVarLit, evalLit, hv]
  | neg v =>
    by_cases hv : v = y
    · subst hv; simp [substVarLit, evalLit]
    · simp [substVarLit, evalLit, hv]

/-- 3. 语义蕴涵变量等价替换保持可满足性等价。 -/
theorem equiv_subst_equisat (F : Formula) (x y : Nat) (hne : x ≠ y) (h : EntailsEqv F x y) :
    Satisfiable F ↔ Satisfiable (substVarFormula x y F) := by
  constructor
  · rintro ⟨σ, hσ⟩
    refine ⟨σ, ?_⟩
    have hxy := h σ hσ
    rw [evalFormula_substVarFormula_eq σ x y hxy F]
    exact hσ
  · rintro ⟨τ, hτ⟩
    let τ' : Env := fun z => if z = y then τ x else τ z
    refine ⟨τ', ?_⟩
    have heq : evalFormula τ' F = evalFormula τ (substVarFormula x y F) := by
      simp only [evalFormula, substVarFormula, List.all_map]
      congr 1
      funext c
      simp only [Function.comp_apply, evalClause, substVarClause, List.any_map]
      congr 1
      funext l
      simp [τ', evalLit_override_subst τ x y hne l]
    rw [heq]
    exact hτ

/-- 4a. 单比特 FLP（Failed Literal Probing）反驳健全性。 -/
theorem flp_unit_sound (F : Formula) (l : Lit) (hUnsat : ¬ Satisfiable ([l] :: F)) :
    Satisfiable F ↔ Satisfiable ([l.negate] :: F) := by
  constructor
  · rintro ⟨σ, hσ⟩
    refine ⟨σ, ?_⟩
    by_cases hl : evalLit σ l = true
    · exfalso
      apply hUnsat
      refine ⟨σ, ?_⟩
      simpa [evalFormula, evalClause, hl] using hσ
    · have hl_neg : evalLit σ l.negate = true := by
        cases hval : evalLit σ l <;> simp_all
      simpa [evalFormula, evalClause, hl_neg] using hσ
  · rintro ⟨σ, hσ⟩
    refine ⟨σ, ?_⟩
    simp only [evalFormula, List.all_cons, Bool.and_eq_true] at hσ
    exact hσ.2

/-- 4b. 双比特（2-literal）协同 FLP 阻断子句学习健全性。 -/
theorem flp_pair_clause_sound (F : Formula) (l1 l2 : Lit)
    (hUnsat : ¬ Satisfiable ([l1] :: [l2] :: F)) :
    Satisfiable F ↔ Satisfiable ([l1.negate, l2.negate] :: F) := by
  constructor
  · rintro ⟨σ, hσ⟩
    refine ⟨σ, ?_⟩
    have hor : evalLit σ l1.negate = true ∨ evalLit σ l2.negate = true := by
      by_cases h1 : evalLit σ l1 = true
      · by_cases h2 : evalLit σ l2 = true
        · exfalso
          apply hUnsat
          refine ⟨σ, ?_⟩
          simpa [evalFormula, evalClause, h1, h2] using hσ
        · right
          cases hv2 : evalLit σ l2 <;> simp_all
      · left
        cases hv1 : evalLit σ l1 <;> simp_all
    have hcl : evalClause σ [l1.negate, l2.negate] = true := by
      simp only [evalClause, List.any_cons, List.any_nil, Bool.or_false, Bool.or_eq_true]
      exact hor
    simpa [evalFormula, hcl] using hσ
  · rintro ⟨σ, hσ⟩
    refine ⟨σ, ?_⟩
    simp only [evalFormula, List.all_cons, Bool.and_eq_true] at hσ
    exact hσ.2

end OuroborosV10.CNFEquisatisfiability
