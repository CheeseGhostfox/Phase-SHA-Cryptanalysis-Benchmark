import OuroborosV10.BoolGateLinearization
import Std.Tactic.BVDecide

/-!
# RippleCarryAdder: 任意位宽涟波进位加法器精确算术等价与 `BitVec 32` 进位固定点

对应 `v10_ouroboros64/ouroboros_core/cpp/crystal_core.hpp` 第 125–139 行
（`exact_add_with_carry` 与 `eval_add_carry`）：
1. 对任意位宽 `n : Nat` 归纳证明 `ripple_adder_exact_nat` 与 `ripple_adder_mod_eq`；
2. 在 `BitVec 32` 上用 `bv_decide` 证明 `bv32_carry_fixed_point` 与 `bv32_eval_add_carry_zero_iff`。
-/

namespace OuroborosV10.RippleCarryAdder

open OuroborosV10.BoolGateLinearization

/-- 任意位宽涟波进位序列：`c[0] = false`，`c[b+1] = Maj(x[b], y[b], c[b])`。 -/
def rippleCarry (x y : Nat → Bool) : Nat → Bool
  | 0 => false
  | b + 1 => bMaj (x b) (y b) (rippleCarry x y b)

/-- 任意位宽第 `b` 位求和输出：`s[b] = x[b] ⊕ y[b] ⊕ c[b]`。 -/
@[inline] def rippleSumBit (x y : Nat → Bool) (b : Nat) : Bool :=
  (x b) ^^ (y b) ^^ (rippleCarry x y b)

/-- 将位流前 `n` 位解码为无符号自然数。 -/
def bitsToNat (f : Nat → Bool) : Nat → Nat
  | 0 => 0
  | n + 1 => bitsToNat f n + (if f n then 2 ^ n else 0)

/-- 全加器单比特算术恒等式：`s + 2 * cout = a + b + cin`。 -/
theorem full_adder_arithmetic (a b cin : Bool) :
    (if (a ^^ b ^^ cin) then 1 else 0) + 2 * (if bMaj a b cin then 1 else 0) =
      (if a then 1 else 0) + (if b then 1 else 0) + (if cin then 1 else 0) := by
  cases a <;> cases b <;> cases cin <;> rfl

/-- 前 `n` 位解码自然数严格小于 `2 ^ n`。 -/
theorem bitsToNat_lt_pow2 (f : Nat → Bool) : ∀ n : Nat, bitsToNat f n < 2 ^ n
  | 0 => by simp [bitsToNat]
  | n + 1 => by
    have ih := bitsToNat_lt_pow2 f n
    have hpow : 2 ^ (n + 1) = 2 ^ n + 2 ^ n := by
      rw [Nat.pow_succ, Nat.mul_two]
    cases hfn : f n <;> simp [bitsToNat, hfn] <;> omega

private theorem if_bool_mul_pow2 (b : Bool) (p : Nat) :
    (if b then p else 0) = (if b then 1 else 0) * p := by
  cases b <;> simp

/-- 任意位宽 `n` 的涟波进位加法器精确自然数恒等式。 -/
theorem ripple_adder_exact_nat (x y : Nat → Bool) :
    ∀ n : Nat,
      bitsToNat (rippleSumBit x y) n + (if rippleCarry x y n then 2 ^ n else 0) =
        bitsToNat x n + bitsToNat y n
  | 0 => by
    simp [bitsToNat, rippleCarry]
  | n + 1 => by
    have ih := ripple_adder_exact_nat x y n
    have hfa := full_adder_arithmetic (x n) (y n) (rippleCarry x y n)
    have hpow : 2 ^ (n + 1) = 2 * 2 ^ n := by
      rw [Nat.pow_succ, Nat.mul_comm]
    have hmul :
        ((if rippleSumBit x y n then 1 else 0) +
          2 * (if rippleCarry x y (n + 1) then 1 else 0)) * 2 ^ n =
        ((if x n then 1 else 0) + (if y n then 1 else 0) +
          (if rippleCarry x y n then 1 else 0)) * 2 ^ n := by
      simp only [rippleSumBit, rippleCarry]
      exact congrArg (· * 2 ^ n) hfa
    have h1 :
        (if rippleSumBit x y n then 2 ^ n else 0) +
          (if rippleCarry x y (n + 1) then 2 ^ (n + 1) else 0) =
        ((if rippleSumBit x y n then 1 else 0) +
          2 * (if rippleCarry x y (n + 1) then 1 else 0)) * 2 ^ n := by
      rw [if_bool_mul_pow2 (rippleSumBit x y n) (2 ^ n),
          if_bool_mul_pow2 (rippleCarry x y (n + 1)) (2 ^ (n + 1)),
          hpow, Nat.add_mul, ← Nat.mul_assoc, Nat.mul_comm (if rippleCarry x y (n + 1) then 1 else 0) 2]
    have h2 :
        (if x n then 2 ^ n else 0) + (if y n then 2 ^ n else 0) +
          (if rippleCarry x y n then 2 ^ n else 0) =
        ((if x n then 1 else 0) + (if y n then 1 else 0) +
          (if rippleCarry x y n then 1 else 0)) * 2 ^ n := by
      rw [if_bool_mul_pow2 (x n) (2 ^ n),
          if_bool_mul_pow2 (y n) (2 ^ n),
          if_bool_mul_pow2 (rippleCarry x y n) (2 ^ n),
          Nat.add_mul, Nat.add_mul]
    simp only [bitsToNat]
    omega

/-- 任意位宽 `n` 的涟波进位加法器模 `2^n` 等价定理。 -/
theorem ripple_adder_mod_eq (x y : Nat → Bool) (n : Nat) :
    bitsToNat (rippleSumBit x y) n = (bitsToNat x n + bitsToNat y n) % 2 ^ n := by
  have hex := ripple_adder_exact_nat x y n
  have hlt := bitsToNat_lt_pow2 (rippleSumBit x y) n
  cases hc : rippleCarry x y n
  · simp [hc] at hex
    rw [← hex]
    exact (Nat.mod_eq_of_lt hlt).symm
  · simp [hc] at hex
    rw [← hex, Nat.add_mod_right]
    exact (Nat.mod_eq_of_lt hlt).symm

/-- `BitVec 32` 上精确对应 `crystal_core.hpp:126-139` 的进位固定点恒等式。 -/
theorem bv32_carry_fixed_point (x y : BitVec 32) :
    let c := x ^^^ y ^^^ (x + y)
    c = (((x &&& y) ^^^ (x &&& c) ^^^ (y &&& c)) <<< 1) := by
  bv_decide

/-- `eval_add_carry` 残差全零且和一致当且仅当 `c` 等于精确进位字 `x ^^^ y ^^^ (x + y)`。 -/
theorem bv32_eval_add_carry_zero_iff (x y c : BitVec 32) :
    let next_c := ((x &&& y) ^^^ (x &&& c) ^^^ (y &&& c)) <<< 1
    ((c ^^^ next_c = 0#32) ∧ (x ^^^ y ^^^ c = x + y)) ↔ c = (x ^^^ y ^^^ (x + y)) := by
  bv_decide

end OuroborosV10.RippleCarryAdder
