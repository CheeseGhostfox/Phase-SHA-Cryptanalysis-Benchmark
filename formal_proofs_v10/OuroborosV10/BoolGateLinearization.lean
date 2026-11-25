import Std.Tactic.BVDecide

/-!
# BoolGateLinearization: 布尔门 `Ch`/`Maj` 精确条件线性化与 32-bit 字级等价

对应 `v10_ouroboros64/ouroboros_core/cpp/crystal_core.hpp` 第 86–92 行与第 1148–1187 行：
1. 证明 `bCh` 的全部 5 条条件线性化引理；
2. 证明 `bMaj` 与 OR 形式等价，以及全部 6 条条件线性化引理（3 组相等 + 3 组不等）；
3. 基于 `Std.Tactic.BVDecide` 验证 32-bit `BitVec 32` 字级 `ch32` / `maj32` 恒等式。
-/

namespace OuroborosV10.BoolGateLinearization

/-- 布尔选择函数 `Ch(e, f, g) = (e && f) ^^ ((!e) && g)`。 -/
@[inline] def bCh (e f g : Bool) : Bool :=
  (e && f) ^^ ((!e) && g)

/-- 布尔多数函数 `Maj(a, b, c) = (a && b) ^^ (a && c) ^^ (b && c)`。 -/
@[inline] def bMaj (a b c : Bool) : Bool :=
  (a && b) ^^ (a && c) ^^ (b && c)

/-- `bMaj` 与析取形式 `(a && b) || (a && c) || (b && c)` 完全等价。 -/
theorem bMaj_eq_or_form (a b c : Bool) :
    bMaj a b c = ((a && b) || (a && c) || (b && c)) := by
  cases a <;> cases b <;> cases c <;> rfl

/-! ### `Ch` 5 条条件线性化引理 -/

theorem bCh_e_true (f g : Bool) : bCh true f g = f := by
  cases f <;> cases g <;> rfl

theorem bCh_e_false (f g : Bool) : bCh false f g = g := by
  cases f <;> cases g <;> rfl

theorem bCh_fg_eq (e f g : Bool) (h : f = g) : bCh e f g = f := by
  subst h
  cases e <;> cases f <;> rfl

theorem bCh_fg_tf (e : Bool) : bCh e true false = e := by
  cases e <;> rfl

theorem bCh_fg_ft (e : Bool) : bCh e false true = !e := by
  cases e <;> rfl

/-! ### `Maj` 6 条条件线性化引理 -/

theorem bMaj_ab_eq (a b c : Bool) (h : a = b) : bMaj a b c = a := by
  subst h
  cases a <;> cases c <;> rfl

theorem bMaj_ac_eq (a b c : Bool) (h : a = c) : bMaj a b c = a := by
  subst h
  cases a <;> cases b <;> rfl

theorem bMaj_bc_eq (a b c : Bool) (h : b = c) : bMaj a b c = b := by
  subst h
  cases a <;> cases b <;> rfl

theorem bMaj_ab_ne (a b c : Bool) (h : a ≠ b) : bMaj a b c = c := by
  cases a <;> cases b <;> cases c <;> simp_all [bMaj]

theorem bMaj_ac_ne (a b c : Bool) (h : a ≠ c) : bMaj a b c = b := by
  cases a <;> cases b <;> cases c <;> simp_all [bMaj]

theorem bMaj_bc_ne (a b c : Bool) (h : b ≠ c) : bMaj a b c = a := by
  cases a <;> cases b <;> cases c <;> simp_all [bMaj]

/-! ### `BitVec 32` 字级等价 (`crystal_core.hpp:86-92`) -/

/-- 32-bit 字级选择函数 `(e &&& f) ^^^ ((~~~e) &&& g)`。 -/
@[inline] def ch32 (e f g : BitVec 32) : BitVec 32 :=
  (e &&& f) ^^^ ((~~~e) &&& g)

/-- 32-bit 字级多数函数 `(a &&& b) ^^^ (a &&& c) ^^^ (b &&& c)`。 -/
@[inline] def maj32 (a b c : BitVec 32) : BitVec 32 :=
  (a &&& b) ^^^ (a &&& c) ^^^ (b &&& c)

/-- 32-bit 字级 `Ch` XOR 形式与 OR 形式等价。 -/
theorem bv32_ch_word_eq (e f g : BitVec 32) :
    ch32 e f g = ((e &&& f) ||| ((~~~e) &&& g)) := by
  unfold ch32
  bv_decide

/-- 32-bit 字级 `Maj` XOR 形式与 OR 形式等价。 -/
theorem bv32_maj_word_eq (a b c : BitVec 32) :
    maj32 a b c = ((a &&& b) ||| (a &&& c) ||| (b &&& c)) := by
  unfold maj32
  bv_decide

end OuroborosV10.BoolGateLinearization
