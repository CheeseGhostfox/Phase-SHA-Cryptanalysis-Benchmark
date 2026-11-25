import PhaseSHA.Algebraic.LiteField
import PhaseSHA.Algebraic.NonAbelianLiftAndOGP

/-!
# Module 6A: 八元数非结合流形升维 (`OctonionLift.lean`, $\mathbb{S}^7 \subset \mathbb{O}$)

本模块在零外部依赖下通过 Cayley-Dickson 翻倍由 `NonAbelianLiftAndOGP.Quat`
构造 8 维八元数代数 `Oct`，严格验证 V5/V6 高维升维的底层性质：
1. **`oct_normSq_mul`**（Degen 八平方恒等式）与 **`oct_s7_closed`**：证明范数平方在八元数乘法下
   严格可乘，单位超球面 $\mathbb{S}^7$ 在乘法下封闭；
2. **`oct_real_axis_xor_and_assoc`**：证明离散实轴 `{±1} ⊂ S^7` 在 2/3 输入乘法下精确同构于
   `XOR` 群，且结合子 `assoc(x, y, z) = (xy)z - x(yz)` 在所有离散实轴真解上严格归零；
3. **`oct_nonassociative_and_alternative`**：诚实构造显式单位基元见证八元数的严格非结合性
   （`(e_1 * e_2) * e_4 ≠ e_1 * (e_2 * e_4)`），同时严格证明八元数全域满足交错性；
4. **`oct_seven_imag_paths_unit_lipschitz`**：为全部 7 条虚轴方向 $e_1..e_7$ 构造连接 `+1` 与 `-1`
   的无奇点、全程保范（`normSq = 1`）且在 `[0, 1]` 上全局 `8`-Lipschitz 一致连续的穿墙路径。
-/

namespace PhaseSHA.V5.OctonionLift

open PhaseSHA.Algebraic.LiteField
open PhaseSHA.Algebraic.NonAbelianLiftAndOGP

/-!
### 1. 四元数辅助代数算子与轻量化引理（不改动原有模块）
-/

def qZero : Quat := ⟨0, 0, 0, 0⟩
def qAdd (a b : Quat) : Quat := ⟨a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z⟩
def qSub (a b : Quat) : Quat := ⟨a.w - b.w, a.x - b.x, a.y - b.y, a.z - b.z⟩
def qConj (a : Quat) : Quat := ⟨a.w, -a.x, -a.y, -a.z⟩
def qSmul (r : Rat) (a : Quat) : Quat := ⟨r * a.w, r * a.x, r * a.y, r * a.z⟩
def qDot (a b : Quat) : Rat := a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z

@[simp] theorem qZero_w : qZero.w = 0 := rfl
@[simp] theorem qZero_x : qZero.x = 0 := rfl
@[simp] theorem qZero_y : qZero.y = 0 := rfl
@[simp] theorem qZero_z : qZero.z = 0 := rfl
@[simp] theorem qAdd_w (a b : Quat) : (qAdd a b).w = a.w + b.w := rfl
@[simp] theorem qAdd_x (a b : Quat) : (qAdd a b).x = a.x + b.x := rfl
@[simp] theorem qAdd_y (a b : Quat) : (qAdd a b).y = a.y + b.y := rfl
@[simp] theorem qAdd_z (a b : Quat) : (qAdd a b).z = a.z + b.z := rfl
@[simp] theorem qSub_w (a b : Quat) : (qSub a b).w = a.w - b.w := rfl
@[simp] theorem qSub_x (a b : Quat) : (qSub a b).x = a.x - b.x := rfl
@[simp] theorem qSub_y (a b : Quat) : (qSub a b).y = a.y - b.y := rfl
@[simp] theorem qSub_z (a b : Quat) : (qSub a b).z = a.z - b.z := rfl
@[simp] theorem qConj_w (a : Quat) : (qConj a).w = a.w := rfl
@[simp] theorem qConj_x (a : Quat) : (qConj a).x = -a.x := rfl
@[simp] theorem qConj_y (a : Quat) : (qConj a).y = -a.y := rfl
@[simp] theorem qConj_z (a : Quat) : (qConj a).z = -a.z := rfl
@[simp] theorem qSmul_w (r : Rat) (a : Quat) : (qSmul r a).w = r * a.w := rfl
@[simp] theorem qSmul_x (r : Rat) (a : Quat) : (qSmul r a).x = r * a.x := rfl
@[simp] theorem qSmul_y (r : Rat) (a : Quat) : (qSmul r a).y = r * a.y := rfl
@[simp] theorem qSmul_z (r : Rat) (a : Quat) : (qSmul r a).z = r * a.z := rfl

private theorem qMul_assoc (a b c : Quat) : (a * b) * c = a * (b * c) := by
  apply Quat.ext <;> simp only [Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z] <;> grind

private theorem qConj_conj (a : Quat) : qConj (qConj a) = a := by
  apply Quat.ext <;> simp only [qConj_w, qConj_x, qConj_y, qConj_z] <;> grind

private theorem qConj_neg (a : Quat) : qConj (-a) = -qConj a := by
  apply Quat.ext <;> simp only [qConj_w, qConj_x, qConj_y, qConj_z,
    Quat.neg_w, Quat.neg_x, Quat.neg_y, Quat.neg_z] <;> grind

private theorem qConj_add (a b : Quat) : qConj (qAdd a b) = qAdd (qConj a) (qConj b) := by
  apply Quat.ext <;> simp only [qConj_w, qConj_x, qConj_y, qConj_z,
    qAdd_w, qAdd_x, qAdd_y, qAdd_z] <;> grind

private theorem qConj_sub (a b : Quat) : qConj (qSub a b) = qSub (qConj a) (qConj b) := by
  apply Quat.ext <;> simp only [qConj_w, qConj_x, qConj_y, qConj_z,
    qSub_w, qSub_x, qSub_y, qSub_z] <;> grind

private theorem qConj_mul (a b : Quat) : qConj (a * b) = qConj b * qConj a := by
  apply Quat.ext <;> simp only [qConj_w, qConj_x, qConj_y, qConj_z,
    Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z] <;> grind

private theorem qMul_add_left (a b c : Quat) : a * qAdd b c = qAdd (a * b) (a * c) := by
  apply Quat.ext <;> simp only [qAdd_w, qAdd_x, qAdd_y, qAdd_z,
    Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z] <;> grind

private theorem qMul_add_right (a b c : Quat) : qAdd a b * c = qAdd (a * c) (b * c) := by
  apply Quat.ext <;> simp only [qAdd_w, qAdd_x, qAdd_y, qAdd_z,
    Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z] <;> grind

private theorem qMul_sub_left (a b c : Quat) : a * qSub b c = qSub (a * b) (a * c) := by
  apply Quat.ext <;> simp only [qSub_w, qSub_x, qSub_y, qSub_z,
    Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z] <;> grind

private theorem qMul_sub_right (a b c : Quat) : qSub a b * c = qSub (a * c) (b * c) := by
  apply Quat.ext <;> simp only [qSub_w, qSub_x, qSub_y, qSub_z,
    Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z] <;> grind

private theorem qMul_neg_left (a b : Quat) : (-a) * b = -(a * b) := by
  apply Quat.ext <;> simp only [Quat.neg_w, Quat.neg_x, Quat.neg_y, Quat.neg_z,
    Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z] <;> grind

private theorem qMul_neg_right (a b : Quat) : a * (-b) = -(a * b) := by
  apply Quat.ext <;> simp only [Quat.neg_w, Quat.neg_x, Quat.neg_y, Quat.neg_z,
    Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z] <;> grind

private theorem qMul_smul_left (r : Rat) (a b : Quat) : qSmul r a * b = qSmul r (a * b) := by
  apply Quat.ext <;> simp only [qSmul_w, qSmul_x, qSmul_y, qSmul_z,
    Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z] <;> grind

private theorem qMul_smul_right (r : Rat) (a b : Quat) : a * qSmul r b = qSmul r (a * b) := by
  apply Quat.ext <;> simp only [qSmul_w, qSmul_x, qSmul_y, qSmul_z,
    Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z] <;> grind

private theorem qSmul_one_mul (r : Rat) (a : Quat) : qSmul r 1 * a = qSmul r a := by
  apply Quat.ext <;> simp only [qSmul_w, qSmul_x, qSmul_y, qSmul_z,
    Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z,
    Quat.one_w, Quat.one_x, Quat.one_y, Quat.one_z] <;> grind

private theorem qMul_smul_one (r : Rat) (a : Quat) : a * qSmul r 1 = qSmul r a := by
  apply Quat.ext <;> simp only [qSmul_w, qSmul_x, qSmul_y, qSmul_z,
    Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z,
    Quat.one_w, Quat.one_x, Quat.one_y, Quat.one_z] <;> grind

private theorem qMul_w_comm (a b : Quat) : (a * b).w = (b * a).w := by
  simp only [Quat.mul_w]; grind

private theorem qDot_eq_mul_conj_w (a b : Quat) : qDot a b = (a * qConj b).w := by
  unfold qDot
  simp only [Quat.mul_w, qConj_w, qConj_x, qConj_y, qConj_z]
  grind

private theorem qNormSq_conj (a : Quat) : (qConj a).normSq = a.normSq := by
  unfold Quat.normSq
  simp only [qConj_w, qConj_x, qConj_y, qConj_z]
  grind

private theorem qNormSq_neg (a : Quat) : (-a).normSq = a.normSq := by
  unfold Quat.normSq
  simp only [Quat.neg_w, Quat.neg_x, Quat.neg_y, Quat.neg_z]
  grind

private theorem qNormSq_add (a b : Quat) :
    (qAdd a b).normSq = a.normSq + b.normSq + 2 * qDot a b := by
  unfold Quat.normSq qDot
  simp only [qAdd_w, qAdd_x, qAdd_y, qAdd_z]
  grind

private theorem qNormSq_sub (a b : Quat) :
    (qSub a b).normSq = a.normSq + b.normSq - 2 * qDot a b := by
  unfold Quat.normSq qDot
  simp only [qSub_w, qSub_x, qSub_y, qSub_z]
  grind

private theorem qNormSq_mul (a b : Quat) : (a * b).normSq = a.normSq * b.normSq := by
  unfold Quat.normSq
  simp only [Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z]
  grind

private theorem qMul_conj_self (a : Quat) : a * qConj a = qSmul a.normSq 1 := by
  apply Quat.ext <;> unfold Quat.normSq <;>
    simp only [Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z,
      qConj_w, qConj_x, qConj_y, qConj_z, qSmul_w, qSmul_x, qSmul_y, qSmul_z,
      Quat.one_w, Quat.one_x, Quat.one_y, Quat.one_z] <;> grind

private theorem qConj_mul_self (a : Quat) : qConj a * a = qSmul a.normSq 1 := by
  apply Quat.ext <;> unfold Quat.normSq <;>
    simp only [Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z,
      qConj_w, qConj_x, qConj_y, qConj_z, qSmul_w, qSmul_x, qSmul_y, qSmul_z,
      Quat.one_w, Quat.one_x, Quat.one_y, Quat.one_z] <;> grind

private theorem qDot_cross_eq (a b c d : Quat) :
    qDot (a * c) (qConj d * b) = qDot (d * a) (b * qConj c) := by
  rw [qDot_eq_mul_conj_w, qDot_eq_mul_conj_w, qConj_mul, qConj_conj, qConj_mul, qConj_conj]
  rw [qMul_w_comm (a * c) (qConj b * d)]
  rw [qMul_assoc (qConj b) d (a * c)]
  rw [qMul_w_comm (qConj b) (d * (a * c))]
  rw [qMul_assoc d (a * c) (qConj b), qMul_assoc a c (qConj b), ← qMul_assoc d a (c * qConj b)]

/-!
### 2. Cayley-Dickson 八元数 `Oct` 构造
-/

/-- 8 维八元数 `Oct = (lo, hi) ∈ ℍ × ℍ`（对应 `lo + hi · e_4`）。 -/
structure Oct where
  lo : Quat
  hi : Quat
  deriving DecidableEq, Repr

namespace Oct

@[ext] theorem ext {x y : Oct} (hlo : x.lo = y.lo) (hhi : x.hi = y.hi) : x = y := by
  cases x; cases y; simp_all

def zero : Oct := ⟨qZero, qZero⟩
def one : Oct := ⟨1, qZero⟩
def neg (x : Oct) : Oct := ⟨-x.lo, -x.hi⟩
def add (x y : Oct) : Oct := ⟨qAdd x.lo y.lo, qAdd x.hi y.hi⟩
def sub (x y : Oct) : Oct := ⟨qSub x.lo y.lo, qSub x.hi y.hi⟩
def conj (x : Oct) : Oct := ⟨qConj x.lo, -x.hi⟩
def smul (r : Rat) (x : Oct) : Oct := ⟨qSmul r x.lo, qSmul r x.hi⟩

/-- Cayley-Dickson 八元数精确乘法：`(a, b) * (c, d) = (a*c - conj(d)*b, d*a + b*conj(c))`。 -/
def mul (x y : Oct) : Oct :=
  ⟨qSub (x.lo * y.lo) (qConj y.hi * x.hi),
   qAdd (y.hi * x.lo) (x.hi * qConj y.lo)⟩

/-- 8 维欧氏范数平方（8 个实坐标的平方和）。 -/
def normSq (x : Oct) : Rat := x.lo.normSq + x.hi.normSq

/-- 八元数空间 $\ell_\infty$ 坐标度量。 -/
def octDist (x y : Oct) : Rat :=
  max (Quat.quatDist x.lo y.lo) (Quat.quatDist x.hi y.hi)

instance : Zero Oct := ⟨zero⟩
instance : One Oct := ⟨one⟩
instance : Neg Oct := ⟨neg⟩
instance : Add Oct := ⟨add⟩
instance : Sub Oct := ⟨sub⟩
instance : Mul Oct := ⟨mul⟩

@[simp] theorem zero_lo : (0 : Oct).lo = qZero := rfl
@[simp] theorem zero_hi : (0 : Oct).hi = qZero := rfl
@[simp] theorem one_lo : (1 : Oct).lo = 1 := rfl
@[simp] theorem one_hi : (1 : Oct).hi = qZero := rfl
@[simp] theorem oct_one_lo : Oct.one.lo = 1 := rfl
@[simp] theorem oct_one_hi : Oct.one.hi = qZero := rfl
@[simp] theorem neg_lo (x : Oct) : (-x).lo = -x.lo := rfl
@[simp] theorem neg_hi (x : Oct) : (-x).hi = -x.hi := rfl
@[simp] theorem add_lo (x y : Oct) : (x + y).lo = qAdd x.lo y.lo := rfl
@[simp] theorem add_hi (x y : Oct) : (x + y).hi = qAdd x.hi y.hi := rfl
@[simp] theorem sub_lo (x y : Oct) : (x - y).lo = qSub x.lo y.lo := rfl
@[simp] theorem sub_hi (x y : Oct) : (x - y).hi = qSub x.hi y.hi := rfl
@[simp] theorem conj_lo (x : Oct) : x.conj.lo = qConj x.lo := rfl
@[simp] theorem conj_hi (x : Oct) : x.conj.hi = -x.hi := rfl
@[simp] theorem smul_lo (r : Rat) (x : Oct) : (smul r x).lo = qSmul r x.lo := rfl
@[simp] theorem smul_hi (r : Rat) (x : Oct) : (smul r x).hi = qSmul r x.hi := rfl
@[simp] theorem mul_lo (x y : Oct) :
    (x * y).lo = qSub (x.lo * y.lo) (qConj y.hi * x.hi) := rfl
@[simp] theorem mul_hi (x y : Oct) :
    (x * y).hi = qAdd (y.hi * x.lo) (x.hi * qConj y.lo) := rfl

end Oct

/-!
### 3. 定理 1：Degen 八平方恒等式与 $\mathbb{S}^7$ 乘法封闭性
-/

/--
**定理 1（Degen 八平方恒等式：`oct_normSq_mul`）**：
对任意八元数 `x, y : Oct`，`(x * y).normSq = x.normSq * y.normSq` 精确成立。
-/
theorem oct_normSq_mul (x y : Oct) : (x * y).normSq = x.normSq * y.normSq := by
  unfold Oct.normSq
  simp only [Oct.mul_lo, Oct.mul_hi, qNormSq_sub, qNormSq_add,
    qNormSq_mul, qNormSq_conj, qDot_cross_eq]
  grind

/-- 推论：单位超球面 $\mathbb{S}^7 = \{x \in \mathbb{O} \mid \|x\|^2 = 1\}$ 在八元数乘法下严格封闭。 -/
theorem oct_s7_closed {x y : Oct} (hx : x.normSq = 1) (hy : y.normSq = 1) :
    (x * y).normSq = 1 := by
  rw [oct_normSq_mul, hx, hy]
  grind

/-!
### 4. 定理 2：离散实轴精确 XOR 同构与结合子归零
-/

/-- 布尔比特在 $\mathbb{S}^7$ 实轴上的嵌入：`false ↦ +1, true ↦ -1`。 -/
def spinOct (b : Bool) : Oct := if b then -Oct.one else Oct.one

/-- 三元结合子 `assoc(x, y, z) = (x * y) * z - x * (y * z)`。 -/
def assoc (x y z : Oct) : Oct := (x * y) * z - x * (y * z)

private theorem spinOct_normSq (b : Bool) : (spinOct b).normSq = 1 := by
  cases b <;> unfold spinOct Oct.normSq Oct.one Quat.normSq <;> simp <;> grind

/--
**定理 2（`oct_real_axis_xor_and_assoc`）**：
1. 离散实轴 `{±1} ⊂ S^7` 在 2 输入与 3 输入八元数乘法下与布尔 `XOR` 严格零误差同构；
2. 结合子 `assoc` 在所有离散实轴真解组合上严格归零，确保结合子退火惩罚项在真解处零误差。
-/
theorem oct_real_axis_xor_and_assoc (b1 b2 b3 : Bool) :
    (spinOct b1 * spinOct b2 = spinOct (b1 ^^ b2)) ∧
    ((spinOct b1 * spinOct b2) * spinOct b3 = spinOct (b1 ^^ b2 ^^ b3)) ∧
    (assoc (spinOct b1) (spinOct b2) (spinOct b3) = 0) := by
  have h2 : ∀ u v : Bool, spinOct u * spinOct v = spinOct (u ^^ v) := by
    intro u v
    cases u <;> cases v <;>
      apply Oct.ext <;> apply Quat.ext <;>
      simp [spinOct, Oct.one] <;> grind
  have h2_12 := h2 b1 b2
  have h3_L : (spinOct b1 * spinOct b2) * spinOct b3 = spinOct (b1 ^^ b2 ^^ b3) := by
    rw [h2_12, h2]
  have h3_R : spinOct b1 * (spinOct b2 * spinOct b3) = spinOct (b1 ^^ b2 ^^ b3) := by
    rw [h2 b2 b3, h2, Bool.xor_assoc]
  refine ⟨h2_12, h3_L, ?_⟩
  unfold assoc
  rw [h3_L, h3_R]
  apply Oct.ext <;> apply Quat.ext <;> simp <;> grind

/-!
### 5. 定理 3：显式非结合见证与全域交错性 (`oct_nonassociative_and_alternative`)
-/

/-- 7 条虚轴基元 `e_1 .. e_7 ∈ S^7`。 -/
def imagBasis (k : Fin 7) : Oct :=
  match k.val with
  | 0 => ⟨⟨0, 1, 0, 0⟩, qZero⟩
  | 1 => ⟨⟨0, 0, 1, 0⟩, qZero⟩
  | 2 => ⟨⟨0, 0, 0, 1⟩, qZero⟩
  | 3 => ⟨qZero, ⟨1, 0, 0, 0⟩⟩
  | 4 => ⟨qZero, ⟨0, 1, 0, 0⟩⟩
  | 5 => ⟨qZero, ⟨0, 0, 1, 0⟩⟩
  | _ => ⟨qZero, ⟨0, 0, 0, 1⟩⟩

private theorem oct_add_conj_eq (x : Oct) :
    x + x.conj = Oct.smul (2 * x.lo.w) Oct.one := by
  apply Oct.ext <;> apply Quat.ext <;>
    simp [Oct.one] <;> grind

private theorem oct_mul_sub_right (x y z : Oct) : (x - y) * z = x * z - y * z := by
  apply Oct.ext <;> apply Quat.ext <;>
    simp only [Oct.sub_lo, Oct.sub_hi, Oct.mul_lo, Oct.mul_hi,
      qSub_w, qSub_x, qSub_y, qSub_z, qAdd_w, qAdd_x, qAdd_y, qAdd_z,
      qConj_w, qConj_x, qConj_y, qConj_z,
      Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z] <;> grind

private theorem oct_mul_sub_left (x y z : Oct) : x * (y - z) = x * y - x * z := by
  apply Oct.ext <;> apply Quat.ext <;>
    simp only [Oct.sub_lo, Oct.sub_hi, Oct.mul_lo, Oct.mul_hi,
      qSub_w, qSub_x, qSub_y, qSub_z, qAdd_w, qAdd_x, qAdd_y, qAdd_z,
      qConj_w, qConj_x, qConj_y, qConj_z,
      Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z] <;> grind

private theorem oct_mul_smul_one_left (r : Rat) (y : Oct) :
    Oct.smul r Oct.one * y = Oct.smul r y := by
  apply Oct.ext <;> apply Quat.ext <;>
    simp [Oct.one] <;> grind

private theorem oct_mul_smul_one_right (r : Rat) (y : Oct) :
    y * Oct.smul r Oct.one = Oct.smul r y := by
  apply Oct.ext <;> apply Quat.ext <;>
    simp [Oct.one] <;> grind

private theorem oct_mul_smul_left (r : Rat) (x y : Oct) :
    Oct.smul r x * y = Oct.smul r (x * y) := by
  apply Oct.ext
  · simp only [Oct.mul_lo, Oct.smul_lo, Oct.smul_hi, qMul_smul_left, qMul_smul_right]
    apply Quat.ext <;> simp <;> grind
  · simp only [Oct.mul_hi, Oct.smul_lo, Oct.smul_hi, qMul_smul_left, qMul_smul_right]
    apply Quat.ext <;> simp <;> grind

private theorem oct_mul_smul_right (r : Rat) (x y : Oct) :
    x * Oct.smul r y = Oct.smul r (x * y) := by
  apply Oct.ext
  · simp only [Oct.mul_lo, Oct.smul_lo, Oct.smul_hi, qMul_smul_right]
    apply Quat.ext <;> simp only [qSub_w, qSub_x, qSub_y, qSub_z,
      qSmul_w, qSmul_x, qSmul_y, qSmul_z, qConj_w, qConj_x, qConj_y, qConj_z,
      Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z] <;> grind
  · simp only [Oct.mul_hi, Oct.smul_lo, Oct.smul_hi, qMul_smul_left]
    apply Quat.ext <;> simp only [qAdd_w, qAdd_x, qAdd_y, qAdd_z,
      qSmul_w, qSmul_x, qSmul_y, qSmul_z, qConj_w, qConj_x, qConj_y, qConj_z,
      Quat.mul_w, Quat.mul_x, Quat.mul_y, Quat.mul_z] <;> grind

private theorem oct_mul_conj_self (x : Oct) :
    x * x.conj = Oct.smul x.normSq Oct.one := by
  apply Oct.ext
  · simp only [Oct.mul_lo, Oct.conj_lo, Oct.conj_hi, Oct.smul_lo, Oct.oct_one_lo,
      qConj_neg, qMul_neg_left, qMul_conj_self, qConj_mul_self]
    apply Quat.ext <;> unfold Oct.normSq <;> simp <;> grind
  · simp only [Oct.mul_hi, Oct.conj_lo, Oct.conj_hi, Oct.smul_hi, Oct.oct_one_hi,
      qConj_conj, qMul_neg_left]
    apply Quat.ext <;> simp <;> grind

private theorem oct_conj_mul_self (x : Oct) :
    x.conj * x = Oct.smul x.normSq Oct.one := by
  apply Oct.ext
  · simp only [Oct.mul_lo, Oct.conj_lo, Oct.conj_hi, Oct.smul_lo, Oct.oct_one_lo,
      qMul_neg_right, qConj_mul_self]
    apply Quat.ext <;> unfold Oct.normSq <;> simp <;> grind
  · simp only [Oct.mul_hi, Oct.conj_lo, Oct.conj_hi, Oct.smul_hi, Oct.oct_one_hi,
      qMul_neg_left]
    apply Quat.ext <;> simp <;> grind

private theorem oct_mul_conj_assoc_left (x y : Oct) :
    x * (x.conj * y) = Oct.smul x.normSq y := by
  have h1 : ∀ a b : Quat, qSub a (-b) = qAdd a b := by
    intro a b; apply Quat.ext <;> simp <;> grind
  have h3 : ∀ a b : Quat, qAdd a (-b) = qSub a b := by
    intro a b; apply Quat.ext <;> simp <;> grind
  apply Oct.ext
  · simp only [Oct.mul_lo, Oct.mul_hi, Oct.conj_lo, Oct.conj_hi, Oct.smul_lo,
      qMul_neg_right, qMul_neg_left, h1, h3]
    rw [qConj_sub, qConj_mul, qConj_mul, qConj_conj, qConj_conj,
      qMul_add_left, qMul_sub_right, ← qMul_assoc x.lo (qConj x.lo) y.lo,
      qMul_conj_self, qSmul_one_mul,
      qMul_assoc x.lo (qConj y.hi) x.hi,
      qMul_assoc y.lo (qConj x.hi) x.hi, qConj_mul_self, qMul_smul_one]
    apply Quat.ext <;> unfold Oct.normSq <;>
      simp only [qSub_w, qSub_x, qSub_y, qSub_z, qAdd_w, qAdd_x, qAdd_y, qAdd_z,
        qSmul_w, qSmul_x, qSmul_y, qSmul_z] <;> grind
  · simp only [Oct.mul_hi, Oct.mul_lo, Oct.conj_lo, Oct.conj_hi, Oct.smul_hi,
      qMul_neg_right, qMul_neg_left, h1, h3]
    rw [qConj_add, qConj_mul, qConj_mul, qConj_conj, qConj_conj,
      qMul_sub_right, qMul_add_left,
      qMul_assoc y.hi (qConj x.lo) x.lo, qConj_mul_self, qMul_smul_one,
      qMul_assoc x.hi (qConj y.lo) x.lo,
      ← qMul_assoc x.hi (qConj x.hi) y.hi, qMul_conj_self, qSmul_one_mul]
    apply Quat.ext <;> unfold Oct.normSq <;>
      simp only [qSub_w, qSub_x, qSub_y, qSub_z, qAdd_w, qAdd_x, qAdd_y, qAdd_z,
        qSmul_w, qSmul_x, qSmul_y, qSmul_z] <;> grind

private theorem oct_mul_conj_assoc_right (x y : Oct) :
    (y * x.conj) * x = Oct.smul x.normSq y := by
  have h1 : ∀ a b : Quat, qSub a (-b) = qAdd a b := by
    intro a b; apply Quat.ext <;> simp <;> grind
  have h2 : ∀ a b : Quat, qAdd (-a) b = qSub b a := by
    intro a b; apply Quat.ext <;> simp <;> grind
  apply Oct.ext
  · simp only [Oct.mul_lo, Oct.mul_hi, Oct.conj_lo, Oct.conj_hi, Oct.smul_lo,
      qConj_neg, qMul_neg_left, h1, h2, qConj_conj,
      qMul_add_right, qMul_sub_left, ← qMul_assoc, qConj_mul_self, qSmul_one_mul]
    rw [qMul_assoc y.lo (qConj x.lo) x.lo, qConj_mul_self, qMul_smul_one]
    apply Quat.ext <;> unfold Oct.normSq <;>
      simp only [qSub_w, qSub_x, qSub_y, qSub_z, qAdd_w, qAdd_x, qAdd_y, qAdd_z,
        qSmul_w, qSmul_x, qSmul_y, qSmul_z] <;> grind
  · simp only [Oct.mul_hi, Oct.mul_lo, Oct.conj_lo, Oct.conj_hi, Oct.smul_hi,
      qConj_neg, qMul_neg_left, h1, h2, qConj_conj,
      qMul_add_left, qMul_sub_right, ← qMul_assoc, qMul_conj_self, qSmul_one_mul]
    rw [qMul_assoc y.hi x.lo (qConj x.lo), qMul_conj_self, qMul_smul_one]
    apply Quat.ext <;> unfold Oct.normSq <;>
      simp only [qSub_w, qSub_x, qSub_y, qSub_z, qAdd_w, qAdd_x, qAdd_y, qAdd_z,
        qSmul_w, qSmul_x, qSmul_y, qSmul_z] <;> grind

private theorem oct_left_alternative (x y : Oct) : x * (x * y) = (x * x) * y := by
  have hDecomp : x = Oct.smul (2 * x.lo.w) Oct.one - x.conj := by
    apply Oct.ext <;> apply Quat.ext <;> simp <;> grind
  have hLHS : x * (x * y) = Oct.smul (2 * x.lo.w) (x * y) - Oct.smul x.normSq y := by
    have hSub : x * (x * y) = x * ((Oct.smul (2 * x.lo.w) Oct.one - x.conj) * y) := by
      rw [← hDecomp]
    rw [hSub, oct_mul_sub_right, oct_mul_sub_left, oct_mul_smul_one_left,
      oct_mul_smul_right, oct_mul_conj_assoc_left]
  have hRHS : (x * x) * y = Oct.smul (2 * x.lo.w) (x * y) - Oct.smul x.normSq y := by
    have hSub : (x * x) * y = (x * (Oct.smul (2 * x.lo.w) Oct.one - x.conj)) * y := by
      rw [← hDecomp]
    rw [hSub, oct_mul_sub_left, oct_mul_smul_one_right, oct_mul_conj_self,
      oct_mul_sub_right, oct_mul_smul_left, oct_mul_smul_one_left]
  exact hLHS.trans hRHS.symm

private theorem oct_right_alternative (x y : Oct) : (y * x) * x = y * (x * x) := by
  have hDecomp : x = Oct.smul (2 * x.lo.w) Oct.one - x.conj := by
    apply Oct.ext <;> apply Quat.ext <;> simp <;> grind
  have hLHS : (y * x) * x = Oct.smul (2 * x.lo.w) (y * x) - Oct.smul x.normSq y := by
    have hSub : (y * x) * x = (y * (Oct.smul (2 * x.lo.w) Oct.one - x.conj)) * x := by
      rw [← hDecomp]
    rw [hSub, oct_mul_sub_left, oct_mul_sub_right, oct_mul_smul_one_right,
      oct_mul_smul_left, oct_mul_conj_assoc_right]
  have hRHS : y * (x * x) = Oct.smul (2 * x.lo.w) (y * x) - Oct.smul x.normSq y := by
    have hSub : y * (x * x) = y * ((Oct.smul (2 * x.lo.w) Oct.one - x.conj) * x) := by
      rw [← hDecomp]
    rw [hSub, oct_mul_sub_right, oct_mul_smul_one_left, oct_conj_mul_self,
      oct_mul_sub_left, oct_mul_smul_right, oct_mul_smul_one_right]
  exact hLHS.trans hRHS.symm

/--
**定理 3（`oct_nonassociative_and_alternative`）**：
1. **严格非结合性见证**：存在三个单位八元数基元 `x, y, z ∈ S^7`（即 `e_1, e_2, e_4`）
   使得 `(x * y) * z ≠ x * (y * z)`；
2. **全域交错性**：任意八元数 `x, y : Oct` 严格满足左右交错律
   `x * (x * y) = (x * x) * y` 与 `(y * x) * x = y * (x * x)`。
-/
theorem oct_nonassociative_and_alternative :
    (∃ x y z : Oct, x.normSq = 1 ∧ y.normSq = 1 ∧ z.normSq = 1 ∧ (x * y) * z ≠ x * (y * z)) ∧
    (∀ x y : Oct, x * (x * y) = (x * x) * y ∧ (y * x) * x = y * (x * x)) := by
  refine ⟨⟨imagBasis ⟨0, by decide⟩, imagBasis ⟨1, by decide⟩, imagBasis ⟨3, by decide⟩,
      ?_, ?_, ?_, ?_⟩, fun x y => ⟨oct_left_alternative x y, oct_right_alternative x y⟩⟩
  · unfold imagBasis Oct.normSq Quat.normSq; simp; grind
  · unfold imagBasis Oct.normSq Quat.normSq; simp; grind
  · unfold imagBasis Oct.normSq Quat.normSq; simp; grind
  · intro hEq
    have hZ := congrArg (fun o : Oct => o.hi.z) hEq
    simp [imagBasis] at hZ
    grind

/-!
### 6. 定理 4：7 条虚轴无奇点连续穿墙路径 (`oct_seven_imag_paths_unit_lipschitz`)
-/

/-- 沿第 `k ∈ {0..6}` 条虚轴基元的无奇点穿墙路径 `o_k(t) = c(t) · 1 + s(t) · e_{k+1}`。 -/
def octPath (k : Fin 7) (t : Rat) : Oct :=
  Oct.smul (pathCos t) Oct.one + Oct.smul (pathSin t) (imagBasis k)

private theorem imagBasis_normSq (k : Fin 7) : (imagBasis k).normSq = 1 := by
  rcases k with ⟨v, hv⟩
  match v with
  | 0 => unfold imagBasis Oct.normSq Quat.normSq; simp; grind
  | 1 => unfold imagBasis Oct.normSq Quat.normSq; simp; grind
  | 2 => unfold imagBasis Oct.normSq Quat.normSq; simp; grind
  | 3 => unfold imagBasis Oct.normSq Quat.normSq; simp; grind
  | 4 => unfold imagBasis Oct.normSq Quat.normSq; simp; grind
  | 5 => unfold imagBasis Oct.normSq Quat.normSq; simp; grind
  | 6 => unfold imagBasis Oct.normSq Quat.normSq; simp; grind
  | n + 7 => omega

private theorem imagBasis_lo_w (k : Fin 7) : (imagBasis k).lo.w = 0 := by
  rcases k with ⟨v, hv⟩
  match v with
  | 0 | 1 | 2 | 3 | 4 | 5 | 6 => rfl
  | n + 7 => omega

private theorem octPath_normSq_eq_sum (c s : Rat) (u : Oct)
    (huW : u.lo.w = 0) (huNorm : u.normSq = 1) :
    (Oct.smul c Oct.one + Oct.smul s u).normSq = c * c + s * s := by
  unfold Oct.normSq Quat.normSq
  simp only [Oct.add_lo, Oct.add_hi, Oct.smul_lo, Oct.smul_hi, Oct.oct_one_lo, Oct.oct_one_hi,
    qAdd_w, qAdd_x, qAdd_y, qAdd_z, qSmul_w, qSmul_x, qSmul_y, qSmul_z, qZero_w, qZero_x, qZero_y, qZero_z,
    Quat.one_w, Quat.one_x, Quat.one_y, Quat.one_z]
  have hUN : u.lo.x * u.lo.x + u.lo.y * u.lo.y + u.lo.z * u.lo.z +
      (u.hi.w * u.hi.w + u.hi.x * u.hi.x + u.hi.y * u.hi.y + u.hi.z * u.hi.z) = 1 := by
    unfold Oct.normSq Quat.normSq at huNorm
    rw [huW] at huNorm
    grind
  have h1 : (c * 1 + s * u.lo.w) * (c * 1 + s * u.lo.w) = c * c := by
    rw [huW]; grind
  have hSum :
      (c * 0 + s * u.lo.x) * (c * 0 + s * u.lo.x) +
      (c * 0 + s * u.lo.y) * (c * 0 + s * u.lo.y) +
      (c * 0 + s * u.lo.z) * (c * 0 + s * u.lo.z) +
      ((c * 0 + s * u.hi.w) * (c * 0 + s * u.hi.w) +
       (c * 0 + s * u.hi.x) * (c * 0 + s * u.hi.x) +
       (c * 0 + s * u.hi.y) * (c * 0 + s * u.hi.y) +
       (c * 0 + s * u.hi.z) * (c * 0 + s * u.hi.z)) =
      (s * s) * (u.lo.x * u.lo.x + u.lo.y * u.lo.y + u.lo.z * u.lo.z +
        (u.hi.w * u.hi.w + u.hi.x * u.hi.x + u.hi.y * u.hi.y + u.hi.z * u.hi.z)) := by grind
  grind

private theorem octPath_lipschitz_dist (k : Fin 7) (t t0 : Rat)
    (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (ht00 : 0 ≤ t0) (ht01 : t0 ≤ 1) :
    Oct.octDist (octPath k t) (octPath k t0) ≤ 8 * ratAbs (t - t0) := by
  have hc := pathCos_lipschitz01.2 t t0 ht0 ht1 ht00 ht01
  have hs := pathSin_lipschitz01.2 t t0 ht0 ht1 ht00 ht01
  have hNonneg : 0 ≤ 8 * ratAbs (t - t0) := Rat.mul_nonneg (by decide) (ratAbs_nonneg _)
  have hCosDiff : ratAbs ((pathCos t * 1 + pathSin t * 0) - (pathCos t0 * 1 + pathSin t0 * 0)) ≤
      8 * ratAbs (t - t0) := by
    have heq : (pathCos t * 1 + pathSin t * 0) - (pathCos t0 * 1 + pathSin t0 * 0) =
        pathCos t - pathCos t0 := by grind
    rw [heq]; exact hc
  have hSinDiff : ratAbs ((pathCos t * 0 + pathSin t * 1) - (pathCos t0 * 0 + pathSin t0 * 1)) ≤
      8 * ratAbs (t - t0) := by
    have heq : (pathCos t * 0 + pathSin t * 1) - (pathCos t0 * 0 + pathSin t0 * 1) =
        pathSin t - pathSin t0 := by grind
    rw [heq]; exact hs
  have hZeroDiff : ratAbs ((pathCos t * 0 + pathSin t * 0) - (pathCos t0 * 0 + pathSin t0 * 0)) ≤
      8 * ratAbs (t - t0) := by
    have heq : (pathCos t * 0 + pathSin t * 0) - (pathCos t0 * 0 + pathSin t0 * 0) = 0 := by grind
    rw [heq, ratAbs_zero]; exact hNonneg
  rcases k with ⟨v, hv⟩
  match v with
  | 0 =>
    unfold Oct.octDist Quat.quatDist octPath imagBasis
    simp only [Oct.add_lo, Oct.add_hi, Oct.smul_lo, Oct.smul_hi, Oct.oct_one_lo, Oct.oct_one_hi,
      qAdd_w, qAdd_x, qAdd_y, qAdd_z, qSmul_w, qSmul_x, qSmul_y, qSmul_z, qZero_w, qZero_x, qZero_y, qZero_z,
      Quat.one_w, Quat.one_x, Quat.one_y, Quat.one_z]
    grind
  | 1 =>
    unfold Oct.octDist Quat.quatDist octPath imagBasis
    simp only [Oct.add_lo, Oct.add_hi, Oct.smul_lo, Oct.smul_hi, Oct.oct_one_lo, Oct.oct_one_hi,
      qAdd_w, qAdd_x, qAdd_y, qAdd_z, qSmul_w, qSmul_x, qSmul_y, qSmul_z, qZero_w, qZero_x, qZero_y, qZero_z,
      Quat.one_w, Quat.one_x, Quat.one_y, Quat.one_z]
    grind
  | 2 =>
    unfold Oct.octDist Quat.quatDist octPath imagBasis
    simp only [Oct.add_lo, Oct.add_hi, Oct.smul_lo, Oct.smul_hi, Oct.oct_one_lo, Oct.oct_one_hi,
      qAdd_w, qAdd_x, qAdd_y, qAdd_z, qSmul_w, qSmul_x, qSmul_y, qSmul_z, qZero_w, qZero_x, qZero_y, qZero_z,
      Quat.one_w, Quat.one_x, Quat.one_y, Quat.one_z]
    grind
  | 3 =>
    unfold Oct.octDist Quat.quatDist octPath imagBasis
    simp only [Oct.add_lo, Oct.add_hi, Oct.smul_lo, Oct.smul_hi, Oct.oct_one_lo, Oct.oct_one_hi,
      qAdd_w, qAdd_x, qAdd_y, qAdd_z, qSmul_w, qSmul_x, qSmul_y, qSmul_z, qZero_w, qZero_x, qZero_y, qZero_z,
      Quat.one_w, Quat.one_x, Quat.one_y, Quat.one_z]
    grind
  | 4 =>
    unfold Oct.octDist Quat.quatDist octPath imagBasis
    simp only [Oct.add_lo, Oct.add_hi, Oct.smul_lo, Oct.smul_hi, Oct.oct_one_lo, Oct.oct_one_hi,
      qAdd_w, qAdd_x, qAdd_y, qAdd_z, qSmul_w, qSmul_x, qSmul_y, qSmul_z, qZero_w, qZero_x, qZero_y, qZero_z,
      Quat.one_w, Quat.one_x, Quat.one_y, Quat.one_z]
    grind
  | 5 =>
    unfold Oct.octDist Quat.quatDist octPath imagBasis
    simp only [Oct.add_lo, Oct.add_hi, Oct.smul_lo, Oct.smul_hi, Oct.oct_one_lo, Oct.oct_one_hi,
      qAdd_w, qAdd_x, qAdd_y, qAdd_z, qSmul_w, qSmul_x, qSmul_y, qSmul_z, qZero_w, qZero_x, qZero_y, qZero_z,
      Quat.one_w, Quat.one_x, Quat.one_y, Quat.one_z]
    grind
  | 6 =>
    unfold Oct.octDist Quat.quatDist octPath imagBasis
    simp only [Oct.add_lo, Oct.add_hi, Oct.smul_lo, Oct.smul_hi, Oct.oct_one_lo, Oct.oct_one_hi,
      qAdd_w, qAdd_x, qAdd_y, qAdd_z, qSmul_w, qSmul_x, qSmul_y, qSmul_z, qZero_w, qZero_x, qZero_y, qZero_z,
      Quat.one_w, Quat.one_x, Quat.one_y, Quat.one_z]
    grind
  | n + 7 => omega

/--
**定理 4（`oct_seven_imag_paths_unit_lipschitz`）**：
对全部 7 条虚轴基元 `k : Fin 7`，路径 `octPath k t`：
1. 精确连接 `+1` 与 `-1`（`octPath k 0 = 1 ∧ octPath k 1 = -1`）；
2. 全程严格保持单位范数 `normSq (octPath k t) = 1`，绝不经过零奇点（`octPath k t ≠ 0`）；
3. 在 `[0, 1]` 上满足全局 `8`-Lipschitz 一致连续性（从而连续光滑穿墙）。
-/
theorem oct_seven_imag_paths_unit_lipschitz (k : Fin 7) :
    (octPath k 0 = 1 ∧ octPath k 1 = -1) ∧
    (∀ t : Rat, (octPath k t).normSq = 1 ∧ octPath k t ≠ 0) ∧
    (∀ t t0 : Rat, 0 ≤ t → t ≤ 1 → 0 ≤ t0 → t0 ≤ 1 →
      Oct.octDist (octPath k t) (octPath k t0) ≤ 8 * ratAbs (t - t0)) := by
  have hc0 : pathCos 0 = 1 := by unfold pathCos pathDen; grind
  have hs0 : pathSin 0 = 0 := by unfold pathSin pathDen; grind
  have hc1 : pathCos 1 = -1 := by unfold pathCos pathDen; grind
  have hs1 : pathSin 1 = 0 := by unfold pathSin pathDen; grind
  refine ⟨⟨?_, ?_⟩, ?_, octPath_lipschitz_dist k⟩
  · apply Oct.ext <;> apply Quat.ext <;>
      simp [octPath, hc0, hs0] <;> grind
  · apply Oct.ext <;> apply Quat.ext <;>
      simp [octPath, hc1, hs1] <;> grind
  · intro t
    have hNorm : (octPath k t).normSq = 1 := by
      unfold octPath
      rw [octPath_normSq_eq_sum (pathCos t) (pathSin t) (imagBasis k)
        (imagBasis_lo_w k) (imagBasis_normSq k)]
      exact pathCos_sq_add_pathSin_sq t
    refine ⟨hNorm, ?_⟩
    intro h0
    rw [h0] at hNorm
    unfold Oct.normSq Quat.normSq at hNorm
    simp at hNorm
    grind

end PhaseSHA.V5.OctonionLift

