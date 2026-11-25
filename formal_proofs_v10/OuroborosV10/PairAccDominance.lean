/-!
# PairAccDominance: `PairAcc` 恒等拷贝支配界与盲态对称假置信交叉乘引理

对应 `v10_ouroboros64/ouroboros_train/trainer.py:885-945` 及 M1 `diagnostics_summary.json:568-612`：
1. 自然数与有理数域上 `PairAcc` 严格劣于恒等拷贝哑基线的充要与充分条件；
2. 代入 M1 `best_m3v2` 实测计数（`U = 141832`, `F = 248`）验证其严格劣于拷贝基线；
3. T2 对称假置信（`q ≠ 1/2`）引发 `4*q*(1-q) < 1`（等价 `BCE > ln 2` 核心单调内核）。
-/

namespace OuroborosV10.PairAccDominance

/-- 自然数形式：`u_m + f_m < U` 当且仅当翻转对正确数 `f_m` 严格低于未翻转对误判损失 `U - u_m`。 -/
theorem pair_acc_loss_iff_nat (U u_m f_m : Nat) (_hU : u_m ≤ U) :
    u_m + f_m < U ↔ f_m < U - u_m := by
  omega

/-- 自然数形式 `PairAcc` 严格劣于拷贝基线定理。 -/
theorem pair_acc_copy_dominates_nat (U F u_m f_m : Nat)
    (_hU : u_m ≤ U) (_hF : f_m ≤ F) (_hPos : 0 < U + F) (hLoss : f_m < U - u_m) :
    u_m + f_m < U := by
  omega

/-- 有理数域 `Rat` 上 `PairAcc` 严格劣于恒等拷贝基线。 -/
theorem pair_acc_copy_dominates_rat (U F u_m f_m : Nat)
    (hPos : 0 < U + F) (hDom : u_m + f_m < U) :
    ((u_m + f_m : Nat) : Rat) / ((U + F : Nat) : Rat) <
      (U : Rat) / ((U + F : Nat) : Rat) := by
  have hden : (0 : Rat) < ((U + F : Nat) : Rat) := Rat.natCast_lt_natCast.mpr hPos
  have hnum : ((u_m + f_m : Nat) : Rat) < (U : Rat) := Rat.natCast_lt_natCast.mpr hDom
  have hinv : (0 : Rat) < (((U + F : Nat) : Rat))⁻¹ := Rat.inv_pos.mpr hden
  rw [Rat.div_def, Rat.div_def]
  have hdiff : 0 < ((U : Rat) - ((u_m + f_m : Nat) : Rat)) * (((U + F : Nat) : Rat))⁻¹ :=
    Rat.mul_pos (by grind) hinv
  have heq :
      ((U : Rat) - ((u_m + f_m : Nat) : Rat)) * (((U + F : Nat) : Rat))⁻¹ =
        (U : Rat) * (((U + F : Nat) : Rat))⁻¹ -
          ((u_m + f_m : Nat) : Rat) * (((U + F : Nat) : Rat))⁻¹ := by
    grind
  grind

/--
代入 M1 `diagnostics_summary.json:570-575` 中 `best_m3v2` `medium` 档实测计数：
`U = 141832`（未翻转对），`F = 248`（翻转对），`U + F = 142080`，
模型未翻转对正确 `u_m = 141630`（`0.9985758`），翻转对正确 `f_m = 67`（`0.2701613`），
总正确对 `141697 < 141832`（`0.9973043 < 0.9982545`）。
-/
theorem best_m3v2_empirical_copy_dominance :
    let U : Nat := 141832
    let F : Nat := 248
    let u_m : Nat := 141630
    let f_m : Nat := 67
    (u_m + f_m < U) ∧
      (((u_m + f_m : Nat) : Rat) / ((U + F : Nat) : Rat) <
        (U : Rat) / ((U + F : Nat) : Rat)) := by
  exact ⟨by decide, pair_acc_copy_dominates_rat 141832 248 141630 67 (by decide) (by decide)⟩

/--
T2 对称假置信二次不等式：`q ≠ 1/2` 时恒有 `4 * q * (1 - q) < 1`，
等价于 `1 - 4 * q * (1 - q) = (2 * q - 1)^2 > 0`。
-/
theorem false_confidence_sym_mul_lt (q : Rat) (hq_ne : q ≠ 1 / 2) :
    4 * q * (1 - q) < 1 := by
  have hne : 2 * q - 1 ≠ 0 := by
    intro hz
    apply hq_ne
    have h2 : 2 * q = 1 := by grind
    have h2ne : (2 : Rat) ≠ 0 := by grind
    have hinv : (2 : Rat)⁻¹ * (2 * q) = (2 : Rat)⁻¹ * 1 := congrArg (fun z => (2 : Rat)⁻¹ * z) h2
    have hcancel : (2 : Rat)⁻¹ * 2 = 1 := Rat.inv_mul_cancel 2 h2ne
    have hq : q = (2 : Rat)⁻¹ := by
      calc
        q = 1 * q := by grind
        _ = ((2 : Rat)⁻¹ * 2) * q := by rw [hcancel]
        _ = (2 : Rat)⁻¹ * (2 * q) := by grind
        _ = (2 : Rat)⁻¹ * 1 := hinv
        _ = (2 : Rat)⁻¹ := by grind
    rw [hq, Rat.div_def]
    grind
  have hsq : 0 < (2 * q - 1) * (2 * q - 1) := by
    by_cases hpos : 0 < 2 * q - 1
    · exact Rat.mul_pos hpos hpos
    · have hneg' : 0 < -(2 * q - 1) := by grind
      have hm := Rat.mul_pos hneg' hneg'
      have heq : (-(2 * q - 1)) * (-(2 * q - 1)) = (2 * q - 1) * (2 * q - 1) := by grind
      grind
  have hid : 1 - 4 * q * (1 - q) = (2 * q - 1) * (2 * q - 1) := by grind
  grind

private theorem sq_split_nat (a d : Nat) :
    (2 * a + d) * (2 * a + d) = 4 * a * (a + d) + d * d := by
  calc
    (2 * a + d) * (2 * a + d)
        = (2 * a) * (2 * a + d) + d * (2 * a + d) := Nat.add_mul (2 * a) d (2 * a + d)
    _ = (2 * a) * (2 * a) + (2 * a) * d + (d * (2 * a) + d * d) := by
          rw [Nat.mul_add, Nat.mul_add]
    _ = 4 * (a * a) + (2 * (a * d) + 2 * (a * d)) + d * d := by
          rw [Nat.mul_comm d (2 * a)]
          have h1 : (2 * a) * (2 * a) = 4 * (a * a) := by
            rw [Nat.mul_assoc, Nat.mul_left_comm a 2 a, ← Nat.mul_assoc 2 2]
          have h2 : (2 * a) * d = 2 * (a * d) := Nat.mul_assoc 2 a d
          rw [h1, h2]
          omega
    _ = 4 * a * (a + d) + d * d := by
          have h3 : 4 * (a * a) + (2 * (a * d) + 2 * (a * d)) = 4 * (a * a + a * d) := by omega
          rw [h3, ← Nat.mul_add, ← Nat.mul_assoc]

/-- 整数比例交叉乘版本：`a ≤ b` 且 `2 * a ≠ b` 时 `4 * a * (b - a) < b * b`。 -/
theorem false_confidence_nat_cross_mul (a b : Nat) (ha : a ≤ b) (hne : 2 * a ≠ b) :
    4 * a * (b - a) < b * b := by
  have hlt_or_gt : 2 * a < b ∨ b < 2 * a := by omega
  cases hlt_or_gt with
  | inl hlt =>
    have hd : 1 ≤ b - 2 * a := by omega
    have hsq : (b - 2 * a) * (b - 2 * a) ≥ 1 := Nat.mul_le_mul hd hd
    have hsub : b = 2 * a + (b - 2 * a) := by omega
    have hsub2 : b - a = a + (b - 2 * a) := by omega
    have hid : b * b = 4 * a * (b - a) + (b - 2 * a) * (b - 2 * a) := by
      rw [hsub2]
      calc
        b * b = (2 * a + (b - 2 * a)) * (2 * a + (b - 2 * a)) := by rw [← hsub]
        _ = 4 * a * (a + (b - 2 * a)) + (b - 2 * a) * (b - 2 * a) :=
              sq_split_nat a (b - 2 * a)
    omega
  | inr hgt =>
    have hd : 1 ≤ 2 * a - b := by omega
    have hsq : (2 * a - b) * (2 * a - b) ≥ 1 := Nat.mul_le_mul hd hd
    have hid : 4 * a * (b - a) + (2 * a - b) * (2 * a - b) = b * b := by
      have hsub : 2 * a = b + (2 * a - b) := by omega
      have h2ba : 2 * (b - a) = b - (2 * a - b) := by omega
      have hle : 2 * a - b ≤ b := by omega
      calc
        4 * a * (b - a) + (2 * a - b) * (2 * a - b)
            = (2 * a) * (2 * (b - a)) + (2 * a - b) * (2 * a - b) := by
              have h4 : 4 * a * (b - a) = (2 * a) * (2 * (b - a)) := by
                rw [show 4 = 2 * 2 from rfl, Nat.mul_assoc 2 2 a, Nat.mul_comm 2 (2 * a),
                  Nat.mul_assoc (2 * a) 2 (b - a)]
              rw [h4]
        _ = (b + (2 * a - b)) * (b - (2 * a - b)) + (2 * a - b) * (2 * a - b) := by
              rw [h2ba, ← hsub]
        _ = b * (b - (2 * a - b)) + (2 * a - b) * (b - (2 * a - b)) +
              (2 * a - b) * (2 * a - b) := by
              rw [Nat.add_mul]
        _ = b * (b - (2 * a - b)) + (2 * a - b) * ((b - (2 * a - b)) + (2 * a - b)) := by
              rw [Nat.add_assoc, ← Nat.mul_add]
        _ = b * (b - (2 * a - b)) + (2 * a - b) * b := by
              rw [Nat.sub_add_cancel hle]
        _ = b * (b - (2 * a - b)) + b * (2 * a - b) := by
              rw [Nat.mul_comm (2 * a - b) b]
        _ = b * ((b - (2 * a - b)) + (2 * a - b)) := by
              rw [← Nat.mul_add]
        _ = b * b := by
              rw [Nat.sub_add_cancel hle]
    omega

end OuroborosV10.PairAccDominance
