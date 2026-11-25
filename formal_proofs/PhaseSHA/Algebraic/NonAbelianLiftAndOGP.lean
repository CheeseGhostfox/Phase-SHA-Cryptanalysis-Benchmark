import PhaseSHA.Algebraic.LiteField

/-!
# Module 5: V4B/C 高维非阿贝尔升维与 OGP 豁免定理

本模块在零外部依赖下构建显式 4-元组四元数 $\mathbb{H}$ (`Quat`) 与 $2\times 2$ 复矩阵 (`Mat2C`) 代数流形，
严格验证 Phase-SHA V4B/C 升维突破离散孤岛障碍及 V3 锁存规避 OGP 禁行定理的两大基石：
1. **`quaternion_involutive_lift_exactness`**：
   - 在 $\mathbb{S}^2$ Bloch 球面嵌入式矩阵与单位四元数 $\mathbb{S}^3$ 上严格验证厄米酉对合性
     `X† = X, X^2 = I`（零模长塌缩）及对易共线充要定理 `[X_i, X_j] = 0 ↔ X_j = ±X_i`；
   - 证明共线对易子空间严格退化并群同构于精确布尔 `XOR` 群；
   - 构造无奇点、全程严格保持 `X(t)^2 = I`（零模长塌缩）且在 `[0, 1]` 上满足全局 `8`-Lipschitz
     一致连续与 $\varepsilon$-$\delta$ 连续性的显式穿墙路径（及一般圆周参数化定理 `involutive_path_of_unit_circle`）；
2. **`latching_violates_lipschitz_ogp_premise`**：
   - 证明含符号硬阈值与置信度阈值 `0 < τ < 1` 的数字锁存算子在阈值切换点严格不连续，
     在任意有限常数 `L > 0` 与任意维度 `n ≥ 1` 下均不满足 `L`-Lipschitz 连续性（含 $\ell_\infty$ 与 $\ell_2^2$ 度量），
     从而从前提假设上严格豁免依赖 Lipschitz 稳定性的 Overlap Gap Property (OGP) 禁行定理。
-/

namespace PhaseSHA.Algebraic.NonAbelianLiftAndOGP

open PhaseSHA.Algebraic.LiteField

/-!
### 1. 高斯有理 $2\times 2$ 复矩阵 (`Mat2C`) 与四元数 (`Quat`) 显式代数模型
-/

/-- 高斯有理数 `GaussRat` 上的标量乘法与共轭简化引理。 -/
@[simp] theorem gauss_smul_re (r : Rat) (z : GaussRat) : (GaussRat.smul r z).re = r * z.re := rfl
@[simp] theorem gauss_smul_im (r : Rat) (z : GaussRat) : (GaussRat.smul r z).im = r * z.im := rfl
@[simp] theorem gauss_conj_re (z : GaussRat) : (GaussRat.conj z).re = z.re := rfl
@[simp] theorem gauss_conj_im (z : GaussRat) : (GaussRat.conj z).im = -z.im := rfl
@[simp] theorem gauss_I_re : GaussRat.I.re = 0 := rfl
@[simp] theorem gauss_I_im : GaussRat.I.im = 1 := rfl
@[simp] theorem gauss_ofRat_re (r : Rat) : (GaussRat.ofRat r).re = r := rfl
@[simp] theorem gauss_ofRat_im (r : Rat) : (GaussRat.ofRat r).im = 0 := rfl

/-- 显式 $2 \times 2$ 复矩阵（系数取自精确可计算复域 $\mathbb{Q}[i]$）。 -/
structure Mat2C where
  a : GaussRat
  b : GaussRat
  c : GaussRat
  d : GaussRat
  deriving DecidableEq, Repr

namespace Mat2C

@[ext] theorem ext {X Y : Mat2C}
    (ha : X.a = Y.a) (hb : X.b = Y.b) (hc : X.c = Y.c) (hd : X.d = Y.d) : X = Y := by
  cases X; cases Y; simp_all

def zero : Mat2C := ⟨0, 0, 0, 0⟩
def I2 : Mat2C := ⟨1, 0, 0, 1⟩
def add (X Y : Mat2C) : Mat2C := ⟨X.a + Y.a, X.b + Y.b, X.c + Y.c, X.d + Y.d⟩
def sub (X Y : Mat2C) : Mat2C := ⟨X.a - Y.a, X.b - Y.b, X.c - Y.c, X.d - Y.d⟩
def neg (X : Mat2C) : Mat2C := ⟨-X.a, -X.b, -X.c, -X.d⟩
def smul (r : Rat) (X : Mat2C) : Mat2C :=
  ⟨GaussRat.smul r X.a, GaussRat.smul r X.b, GaussRat.smul r X.c, GaussRat.smul r X.d⟩
def mul (X Y : Mat2C) : Mat2C :=
  ⟨X.a * Y.a + X.b * Y.c, X.a * Y.b + X.b * Y.d,
   X.c * Y.a + X.d * Y.c, X.c * Y.b + X.d * Y.d⟩
def dagger (X : Mat2C) : Mat2C :=
  ⟨GaussRat.conj X.a, GaussRat.conj X.c, GaussRat.conj X.b, GaussRat.conj X.d⟩
def commutator (X Y : Mat2C) : Mat2C := sub (mul X Y) (mul Y X)

instance : Zero Mat2C := ⟨zero⟩
instance : One Mat2C := ⟨I2⟩
instance : Add Mat2C := ⟨add⟩
instance : Sub Mat2C := ⟨sub⟩
instance : Neg Mat2C := ⟨neg⟩
instance : Mul Mat2C := ⟨mul⟩

@[simp] theorem zero_a : (0 : Mat2C).a = 0 := rfl
@[simp] theorem zero_b : (0 : Mat2C).b = 0 := rfl
@[simp] theorem zero_c : (0 : Mat2C).c = 0 := rfl
@[simp] theorem zero_d : (0 : Mat2C).d = 0 := rfl
@[simp] theorem I2_a : I2.a = 1 := rfl
@[simp] theorem I2_b : I2.b = 0 := rfl
@[simp] theorem I2_c : I2.c = 0 := rfl
@[simp] theorem I2_d : I2.d = 1 := rfl
@[simp] theorem add_a (X Y : Mat2C) : (X + Y).a = X.a + Y.a := rfl
@[simp] theorem add_b (X Y : Mat2C) : (X + Y).b = X.b + Y.b := rfl
@[simp] theorem add_c (X Y : Mat2C) : (X + Y).c = X.c + Y.c := rfl
@[simp] theorem add_d (X Y : Mat2C) : (X + Y).d = X.d + Y.d := rfl
@[simp] theorem sub_a (X Y : Mat2C) : (X - Y).a = X.a - Y.a := rfl
@[simp] theorem sub_b (X Y : Mat2C) : (X - Y).b = X.b - Y.b := rfl
@[simp] theorem sub_c (X Y : Mat2C) : (X - Y).c = X.c - Y.c := rfl
@[simp] theorem sub_d (X Y : Mat2C) : (X - Y).d = X.d - Y.d := rfl
@[simp] theorem neg_a (X : Mat2C) : (-X).a = -X.a := rfl
@[simp] theorem neg_b (X : Mat2C) : (-X).b = -X.b := rfl
@[simp] theorem neg_c (X : Mat2C) : (-X).c = -X.c := rfl
@[simp] theorem neg_d (X : Mat2C) : (-X).d = -X.d := rfl
@[simp] theorem smul_a (r : Rat) (X : Mat2C) : (smul r X).a = GaussRat.smul r X.a := rfl
@[simp] theorem smul_b (r : Rat) (X : Mat2C) : (smul r X).b = GaussRat.smul r X.b := rfl
@[simp] theorem smul_c (r : Rat) (X : Mat2C) : (smul r X).c = GaussRat.smul r X.c := rfl
@[simp] theorem smul_d (r : Rat) (X : Mat2C) : (smul r X).d = GaussRat.smul r X.d := rfl
@[simp] theorem mul_a (X Y : Mat2C) : (X * Y).a = X.a * Y.a + X.b * Y.c := rfl
@[simp] theorem mul_b (X Y : Mat2C) : (X * Y).b = X.a * Y.b + X.b * Y.d := rfl
@[simp] theorem mul_c (X Y : Mat2C) : (X * Y).c = X.c * Y.a + X.d * Y.c := rfl
@[simp] theorem mul_d (X Y : Mat2C) : (X * Y).d = X.c * Y.b + X.d * Y.d := rfl
@[simp] theorem dagger_a (X : Mat2C) : X.dagger.a = GaussRat.conj X.a := rfl
@[simp] theorem dagger_b (X : Mat2C) : X.dagger.b = GaussRat.conj X.c := rfl
@[simp] theorem dagger_c (X : Mat2C) : X.dagger.c = GaussRat.conj X.b := rfl
@[simp] theorem dagger_d (X : Mat2C) : X.dagger.d = GaussRat.conj X.d := rfl

/-- 标准 Pauli 矩阵 `σ_x, σ_y, σ_z`。 -/
def σx : Mat2C := ⟨0, 1, 1, 0⟩
def σy : Mat2C := ⟨0, -GaussRat.I, GaussRat.I, 0⟩
def σz : Mat2C := ⟨1, 0, 0, -1⟩

/-- $\mathbb{S}^2 \subset \mathbb{R}^3$（对应纯虚单位四元数）向 $2\times 2$ 酉对合流形的 Bloch 嵌入。 -/
def bloch (x y z : Rat) : Mat2C :=
  smul x σx + smul y σy + smul z σz

/-- 显式 $2\times 2$ 矩阵的 $\ell_\infty$ 距离（8 个实虚分量差的最大绝对值）。 -/
def matDist (X Y : Mat2C) : Rat :=
  max (max (max (ratAbs (X.a.re - Y.a.re)) (ratAbs (X.a.im - Y.a.im)))
           (max (ratAbs (X.b.re - Y.b.re)) (ratAbs (X.b.im - Y.b.im))))
      (max (max (ratAbs (X.c.re - Y.c.re)) (ratAbs (X.c.im - Y.c.im)))
           (max (ratAbs (X.d.re - Y.d.re)) (ratAbs (X.d.im - Y.d.im))))

end Mat2C

/-- 显式 4-元组四元数 `w + x i + y j + z k ∈ ℍ`。 -/
structure Quat where
  w : Rat
  x : Rat
  y : Rat
  z : Rat
  deriving DecidableEq, Repr

namespace Quat

@[ext] theorem ext {p q : Quat}
    (hw : p.w = q.w) (hx : p.x = q.x) (hy : p.y = q.y) (hz : p.z = q.z) : p = q := by
  cases p; cases q; simp_all

def one : Quat := ⟨1, 0, 0, 0⟩
def neg (q : Quat) : Quat := ⟨-q.w, -q.x, -q.y, -q.z⟩
def mul (p q : Quat) : Quat :=
  ⟨p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z,
   p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y,
   p.w * q.y - p.x * q.z + p.y * q.w + p.z * q.x,
   p.w * q.z + p.x * q.y - p.y * q.x + p.z * q.w⟩
def normSq (q : Quat) : Rat := q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z
def quatDist (p q : Quat) : Rat :=
  max (max (ratAbs (p.w - q.w)) (ratAbs (p.x - q.x)))
      (max (ratAbs (p.y - q.y)) (ratAbs (p.z - q.z)))

instance : One Quat := ⟨one⟩
instance : Neg Quat := ⟨neg⟩
instance : Mul Quat := ⟨mul⟩

@[simp] theorem one_w : (1 : Quat).w = 1 := rfl
@[simp] theorem one_x : (1 : Quat).x = 0 := rfl
@[simp] theorem one_y : (1 : Quat).y = 0 := rfl
@[simp] theorem one_z : (1 : Quat).z = 0 := rfl
@[simp] theorem neg_w (q : Quat) : (-q).w = -q.w := rfl
@[simp] theorem neg_x (q : Quat) : (-q).x = -q.x := rfl
@[simp] theorem neg_y (q : Quat) : (-q).y = -q.y := rfl
@[simp] theorem neg_z (q : Quat) : (-q).z = -q.z := rfl
@[simp] theorem mul_w (p q : Quat) :
    (p * q).w = p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z := rfl
@[simp] theorem mul_x (p q : Quat) :
    (p * q).x = p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y := rfl
@[simp] theorem mul_y (p q : Quat) :
    (p * q).y = p.w * q.y - p.x * q.z + p.y * q.w + p.z * q.x := rfl
@[simp] theorem mul_z (p q : Quat) :
    (p * q).z = p.w * q.z + p.x * q.y - p.y * q.x + p.z * q.w := rfl

end Quat

/-!
### 2. $\mathbb{S}^2$ 酉对合性、共线对易分类与布尔 XOR 群同构
-/

theorem bloch_dagger (x y z : Rat) :
    (Mat2C.bloch x y z).dagger = Mat2C.bloch x y z := by
  apply Mat2C.ext <;> apply GaussRat.ext <;>
    simp [Mat2C.bloch, Mat2C.σx, Mat2C.σy, Mat2C.σz] <;> grind

theorem bloch_sq_eq_I2 (x y z : Rat) (hNorm : x * x + y * y + z * z = 1) :
    Mat2C.bloch x y z * Mat2C.bloch x y z = Mat2C.I2 := by
  apply Mat2C.ext <;> apply GaussRat.ext <;>
    simp [Mat2C.bloch, Mat2C.σx, Mat2C.σy, Mat2C.σz] <;> grind

theorem involutive_path_of_unit_circle (c s : Rat) (h : c * c + s * s = 1) :
    let X := Mat2C.smul c Mat2C.σz + Mat2C.smul s Mat2C.σx
    X.dagger = X ∧ X * X = Mat2C.I2 := by
  intro X
  have hEq : X = Mat2C.bloch s 0 c := by
    apply Mat2C.ext <;> apply GaussRat.ext <;>
      simp [X, Mat2C.bloch, Mat2C.σx, Mat2C.σy, Mat2C.σz] <;> grind
  rw [hEq]
  have hNorm : s * s + 0 * 0 + c * c = 1 := by grind
  exact ⟨bloch_dagger s 0 c, bloch_sq_eq_I2 s 0 c hNorm⟩

theorem bloch_commutator_zero_iff_cross_zero (x1 y1 z1 x2 y2 z2 : Rat) :
    Mat2C.commutator (Mat2C.bloch x1 y1 z1) (Mat2C.bloch x2 y2 z2) = 0 ↔
      (y1 * z2 - z1 * y2 = 0 ∧ z1 * x2 - x1 * z2 = 0 ∧ x1 * y2 - y1 * x2 = 0) := by
  constructor
  · intro hComm
    have haIm := congrArg (fun M => M.a.im) hComm
    have hbRe := congrArg (fun M => M.b.re) hComm
    have hbIm := congrArg (fun M => M.b.im) hComm
    simp [Mat2C.commutator, Mat2C.sub, Mat2C.mul, Mat2C.bloch,
      Mat2C.σx, Mat2C.σy, Mat2C.σz] at haIm hbRe hbIm
    refine ⟨?_, ?_, ?_⟩ <;> grind
  · intro ⟨hc1, hc2, hc3⟩
    apply Mat2C.ext <;> apply GaussRat.ext <;>
      simp [Mat2C.commutator, Mat2C.sub, Mat2C.mul, Mat2C.bloch,
        Mat2C.σx, Mat2C.σy, Mat2C.σz] <;> grind

private theorem rat_sum3_sq_eq_zero_iff {a b c : Rat}
    (h : a * a + (b * b + c * c) = 0) : a = 0 ∧ b = 0 ∧ c = 0 := by
  have ha := rat_sq_nonneg a
  have hb := rat_sq_nonneg b
  have hc := rat_sq_nonneg c
  exact ⟨(rat_sq_eq_zero_iff a).mp (by grind),
    (rat_sq_eq_zero_iff b).mp (by grind),
    (rat_sq_eq_zero_iff c).mp (by grind)⟩

theorem bloch_commutator_zero_iff_collinear
    (x1 y1 z1 x2 y2 z2 : Rat)
    (h1 : x1 * x1 + y1 * y1 + z1 * z1 = 1)
    (h2 : x2 * x2 + y2 * y2 + z2 * z2 = 1) :
    Mat2C.commutator (Mat2C.bloch x1 y1 z1) (Mat2C.bloch x2 y2 z2) = 0 ↔
      (Mat2C.bloch x2 y2 z2 = Mat2C.bloch x1 y1 z1 ∨
       Mat2C.bloch x2 y2 z2 = -Mat2C.bloch x1 y1 z1) := by
  rw [bloch_commutator_zero_iff_cross_zero]
  constructor
  · intro ⟨hc1, hc2, hc3⟩
    have hLagrange :
        (x1 * x1 + y1 * y1 + z1 * z1) * (x2 * x2 + y2 * y2 + z2 * z2) =
        (x1 * x2 + y1 * y2 + z1 * z2) * (x1 * x2 + y1 * y2 + z1 * z2) +
        (y1 * z2 - z1 * y2) * (y1 * z2 - z1 * y2) +
        (z1 * x2 - x1 * z2) * (z1 * x2 - x1 * z2) +
        (x1 * y2 - y1 * x2) * (x1 * y2 - y1 * x2) := by grind
    rw [h1, h2, hc1, hc2, hc3] at hLagrange
    let dot := x1 * x2 + y1 * y2 + z1 * z2
    have hDotSq : (dot - 1) * (dot + 1) = 0 := by
      dsimp [dot]
      grind
    rcases rat_mul_eq_zero_iff.mp hDotSq with hSub | hAdd
    · left
      have hDot1 : dot = 1 := by grind
      have hSumSq : (x2 - x1) * (x2 - x1) + ((y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1)) = 0 := by
        have hId : (x2 - x1) * (x2 - x1) + ((y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1)) =
            (x2 * x2 + y2 * y2 + z2 * z2) + (x1 * x1 + y1 * y1 + z1 * z1) - 2 * dot := by grind
        rw [hId, h1, h2, hDot1]
        grind
      obtain ⟨hxEq, hyEq, hzEq⟩ := rat_sum3_sq_eq_zero_iff hSumSq
      have hxEq' : x2 = x1 := by grind
      have hyEq' : y2 = y1 := by grind
      have hzEq' : z2 = z1 := by grind
      rw [hxEq', hyEq', hzEq']
    · right
      have hDotM1 : dot = -1 := by grind
      have hSumSq : (x2 + x1) * (x2 + x1) + ((y2 + y1) * (y2 + y1) + (z2 + z1) * (z2 + z1)) = 0 := by
        have hId : (x2 + x1) * (x2 + x1) + ((y2 + y1) * (y2 + y1) + (z2 + z1) * (z2 + z1)) =
            (x2 * x2 + y2 * y2 + z2 * z2) + (x1 * x1 + y1 * y1 + z1 * z1) + 2 * dot := by grind
        rw [hId, h1, h2, hDotM1]
        grind
      obtain ⟨hxEq, hyEq, hzEq⟩ := rat_sum3_sq_eq_zero_iff hSumSq
      have hxEq' : x2 = -x1 := by grind
      have hyEq' : y2 = -y1 := by grind
      have hzEq' : z2 = -z1 := by grind
      rw [hxEq', hyEq', hzEq']
      apply Mat2C.ext <;> apply GaussRat.ext <;>
        simp [Mat2C.bloch, Mat2C.σx, Mat2C.σy, Mat2C.σz] <;> grind
  · intro hEq
    cases hEq with
    | inl hPos =>
      have haIm := congrArg (fun M => M.a.re) hPos
      have hbRe := congrArg (fun M => M.b.re) hPos
      have hbIm := congrArg (fun M => M.b.im) hPos
      simp [Mat2C.bloch, Mat2C.σx, Mat2C.σy, Mat2C.σz] at haIm hbRe hbIm
      have hx : x2 = x1 := by grind
      have hy : y2 = y1 := by grind
      have hz : z2 = z1 := by grind
      refine ⟨by rw [hy, hz]; grind, by rw [hx, hz]; grind, by rw [hx, hy]; grind⟩
    | inr hNeg =>
      have haIm := congrArg (fun M => M.a.re) hNeg
      have hbRe := congrArg (fun M => M.b.re) hNeg
      have hbIm := congrArg (fun M => M.b.im) hNeg
      simp [Mat2C.bloch, Mat2C.σx, Mat2C.σy, Mat2C.σz] at haIm hbRe hbIm
      have hx : x2 = -x1 := by grind
      have hy : y2 = -y1 := by grind
      have hz : z2 = -z1 := by grind
      refine ⟨by rw [hy, hz]; grind, by rw [hx, hz]; grind, by rw [hx, hy]; grind⟩

/-- 布尔比特在共线对易子空间 `{±σ_z}` 中的矩阵表示：`false ↦ +σ_z, true ↦ -σ_z`。 -/
def spinMat (b : Bool) : Mat2C := if b then -Mat2C.σz else Mat2C.σz

/-- 布尔比特在单位四元数实轴 `{±1}` 中的表示：`false ↦ +1, true ↦ -1`。 -/
def spinQuat (b : Bool) : Quat := if b then -Quat.one else Quat.one

theorem spinMat_xor_iso (b1 b2 : Bool) :
    Mat2C.commutator (spinMat b1) (spinMat b2) = 0 ∧
    (spinMat b1 * spinMat b2 = if (b1 ^^ b2) then -Mat2C.I2 else Mat2C.I2) ∧
    (spinMat b1 * spinMat b2 * Mat2C.σz = spinMat (b1 ^^ b2)) := by
  cases b1 <;> cases b2 <;>
    refine ⟨?_, ?_, ?_⟩ <;>
    apply Mat2C.ext <;> apply GaussRat.ext <;>
    simp [spinMat, Mat2C.commutator, Mat2C.sub, Mat2C.mul, Mat2C.σz] <;> grind

theorem spinQuat_xor_iso (b1 b2 : Bool) :
    spinQuat b1 * spinQuat b2 = spinQuat (b1 ^^ b2) := by
  cases b1 <;> cases b2 <;>
    apply Quat.ext <;> simp [spinQuat, Quat.one] <;> grind

/-!
### 3. 无奇点、严格保模 `X(t)^2 = I` 的连续平滑穿墙路径
-/

/-- 穿墙路径分母 `D(t) = 1 - 2t + 2t^2 ≥ 1/2`（全局严格正定，无任何奇点）。 -/
def pathDen (t : Rat) : Rat := 1 - 2 * t + 2 * t * t

theorem pathDen_ge_half (t : Rat) : 1 / 2 ≤ pathDen t := by
  have hDiff : pathDen t - 1 / 2 = 2 * ((t - 1 / 2) * (t - 1 / 2)) := by
    unfold pathDen; grind
  have hSq : 0 ≤ (t - 1 / 2) * (t - 1 / 2) := rat_sq_nonneg _
  have hMul : 0 ≤ 2 * ((t - 1 / 2) * (t - 1 / 2)) := Rat.mul_nonneg (by decide) hSq
  grind

/-- 穿墙路径纵轴分量 `c(t) = (1 - 2t) / D(t)`。 -/
def pathCos (t : Rat) : Rat := (1 - 2 * t) / pathDen t

/-- 穿墙路径横轴非阿贝尔分量 `s(t) = 2t(1 - t) / D(t)`。 -/
def pathSin (t : Rat) : Rat := (2 * t * (1 - t)) / pathDen t

theorem pathCos_sq_add_pathSin_sq (t : Rat) :
    pathCos t * pathCos t + pathSin t * pathSin t = 1 := by
  have hDenHalf := pathDen_ge_half t
  have hDenNe : pathDen t ≠ 0 := by grind
  have hDen2Ne : pathDen t * pathDen t ≠ 0 := by
    intro h0
    rcases rat_mul_eq_zero_iff.mp h0 with hd | hd <;> exact hDenNe hd
  have hPow1 := ratPow_div (1 - 2 * t) (pathDen t) hDenNe 2
  have hPow2 := ratPow_div (2 * t * (1 - t)) (pathDen t) hDenNe 2
  simp only [ratPow] at hPow1 hPow2
  have hSq1 : pathCos t * pathCos t = ((1 - 2 * t) * (1 - 2 * t)) / (pathDen t * pathDen t) := by
    unfold pathCos
    simpa [← hPow1] using (by grind :
      1 * ((1 - 2 * t) / pathDen t) * ((1 - 2 * t) / pathDen t) =
      (1 * (1 - 2 * t) * (1 - 2 * t)) / (1 * pathDen t * pathDen t))
  have hSq2 : pathSin t * pathSin t =
      ((2 * t * (1 - t)) * (2 * t * (1 - t))) / (pathDen t * pathDen t) := by
    unfold pathSin
    simpa [← hPow2] using (by grind :
      1 * ((2 * t * (1 - t)) / pathDen t) * ((2 * t * (1 - t)) / pathDen t) =
      (1 * (2 * t * (1 - t)) * (2 * t * (1 - t))) / (1 * pathDen t * pathDen t))
  have hAddDiv : ∀ a b d : Rat, a / d + b / d = (a + b) / d := by
    intro a b d
    rw [Rat.div_def, Rat.div_def, Rat.div_def]
    grind
  rw [hSq1, hSq2, hAddDiv]
  have hNum : (1 - 2 * t) * (1 - 2 * t) + (2 * t * (1 - t)) * (2 * t * (1 - t)) =
      pathDen t * pathDen t := by
    unfold pathDen; grind
  rw [hNum, Rat.div_def, Rat.mul_inv_cancel _ hDen2Ne]

private theorem rat_eq_div_of_mul_eq {x a b : Rat} (hb : b ≠ 0) (h : x * b = a) :
    x = a / b := by
  have h1 := congrArg (fun z => z * b⁻¹) h
  have h2 : (x * b) * b⁻¹ = x * (b * b⁻¹) := by grind
  rw [h2, Rat.mul_inv_cancel b hb] at h1
  rw [Rat.div_def]
  grind

theorem pathCos_lipschitz01 : IsLipschitzOn01 pathCos 8 := by
  refine ⟨by decide, ?_⟩
  intro t t0 ht0 ht1 ht00 ht01
  have hD1 := pathDen_ge_half t
  have hD2 := pathDen_ge_half t0
  have hDProd : (1 / 4 : Rat) ≤ pathDen t * pathDen t0 := by
    have h1 : (1 / 2 : Rat) * (1 / 2) ≤ pathDen t * (1 / 2) :=
      rat_mul_le_mul_of_nonneg_right hD1 (by grind)
    have h2 : pathDen t * (1 / 2) ≤ pathDen t * pathDen t0 :=
      rat_mul_le_mul_of_nonneg_left hD2 (by grind)
    grind
  have hDPos : 0 < pathDen t * pathDen t0 := by grind
  have hDNe : pathDen t * pathDen t0 ≠ 0 := by grind
  have hDiffEq : pathCos t - pathCos t0 =
      (2 * (t0 - t) * (t0 * (1 - t) + t * (1 - t0))) / (pathDen t * pathDen t0) := by
    apply rat_eq_div_of_mul_eq hDNe
    have h1 : (pathCos t - pathCos t0) * (pathDen t * pathDen t0) =
        (pathCos t * pathDen t) * pathDen t0 - (pathCos t0 * pathDen t0) * pathDen t := by grind
    rw [h1]
    unfold pathCos
    rw [Rat.div_mul_cancel (by grind), Rat.div_mul_cancel (by grind)]
    unfold pathDen
    grind
  rw [hDiffEq, ratAbs_div _ _ hDPos, ratAbs_mul, ratAbs_mul]
  have hMidNonneg : 0 ≤ t0 * (1 - t) + t * (1 - t0) :=
    Rat.add_nonneg (Rat.mul_nonneg ht00 (by grind)) (Rat.mul_nonneg ht0 (by grind))
  have hMidLe1 : t0 * (1 - t) + t * (1 - t0) ≤ 1 := by
    have hGap : 1 - (t0 * (1 - t) + t * (1 - t0)) = (1 - t) * (1 - t0) + t * t0 := by grind
    have hGapNonneg : 0 ≤ (1 - t) * (1 - t0) + t * t0 :=
      Rat.add_nonneg (Rat.mul_nonneg (by grind) (by grind)) (Rat.mul_nonneg ht0 ht00)
    grind
  have hMidAbs : ratAbs (t0 * (1 - t) + t * (1 - t0)) ≤ 1 := by
    rw [ratAbs_of_nonneg hMidNonneg]
    exact hMidLe1
  have h2Abs : ratAbs (2 : Rat) = 2 := ratAbs_of_nonneg (by decide)
  rw [h2Abs, ratAbs_sub_comm t0 t]
  have hNumLe : 2 * ratAbs (t - t0) * ratAbs (t0 * (1 - t) + t * (1 - t0)) ≤ 2 * ratAbs (t - t0) := by
    have hPos2t : 0 ≤ 2 * ratAbs (t - t0) := Rat.mul_nonneg (by decide) (ratAbs_nonneg _)
    simpa using rat_mul_le_mul_of_nonneg_left hMidAbs hPos2t
  rw [rat_div_le_iff hDPos]
  have hBound : 2 * ratAbs (t - t0) ≤ (8 * ratAbs (t - t0)) * (pathDen t * pathDen t0) := by
    have h8Pos : 0 ≤ 8 * ratAbs (t - t0) := Rat.mul_nonneg (by decide) (ratAbs_nonneg _)
    have h1 := rat_mul_le_mul_of_nonneg_left hDProd h8Pos
    grind
  exact Rat.le_trans hNumLe hBound

theorem pathSin_lipschitz01 : IsLipschitzOn01 pathSin 8 := by
  refine ⟨by decide, ?_⟩
  intro t t0 ht0 ht1 ht00 ht01
  have hD1 := pathDen_ge_half t
  have hD2 := pathDen_ge_half t0
  have hDProd : (1 / 4 : Rat) ≤ pathDen t * pathDen t0 := by
    have h1 : (1 / 2 : Rat) * (1 / 2) ≤ pathDen t * (1 / 2) :=
      rat_mul_le_mul_of_nonneg_right hD1 (by grind)
    have h2 : pathDen t * (1 / 2) ≤ pathDen t * pathDen t0 :=
      rat_mul_le_mul_of_nonneg_left hD2 (by grind)
    grind
  have hDPos : 0 < pathDen t * pathDen t0 := by grind
  have hDNe : pathDen t * pathDen t0 ≠ 0 := by grind
  have hDiffEq : pathSin t - pathSin t0 =
      (2 * (t - t0) * (1 - t - t0)) / (pathDen t * pathDen t0) := by
    apply rat_eq_div_of_mul_eq hDNe
    have h1 : (pathSin t - pathSin t0) * (pathDen t * pathDen t0) =
        (pathSin t * pathDen t) * pathDen t0 - (pathSin t0 * pathDen t0) * pathDen t := by grind
    rw [h1]
    unfold pathSin
    rw [Rat.div_mul_cancel (by grind), Rat.div_mul_cancel (by grind)]
    unfold pathDen
    grind
  rw [hDiffEq, ratAbs_div _ _ hDPos, ratAbs_mul, ratAbs_mul]
  have hMidAbs : ratAbs (1 - t - t0) ≤ 1 := by
    rw [ratAbs_le_iff]
    grind
  have h2Abs : ratAbs (2 : Rat) = 2 := ratAbs_of_nonneg (by decide)
  rw [h2Abs]
  have hNumLe : 2 * ratAbs (t - t0) * ratAbs (1 - t - t0) ≤ 2 * ratAbs (t - t0) := by
    have hPos2t : 0 ≤ 2 * ratAbs (t - t0) := Rat.mul_nonneg (by decide) (ratAbs_nonneg _)
    simpa using rat_mul_le_mul_of_nonneg_left hMidAbs hPos2t
  rw [rat_div_le_iff hDPos]
  have hBound : 2 * ratAbs (t - t0) ≤ (8 * ratAbs (t - t0)) * (pathDen t * pathDen t0) := by
    have h8Pos : 0 ≤ 8 * ratAbs (t - t0) := Rat.mul_nonneg (by decide) (ratAbs_nonneg _)
    have h1 := rat_mul_le_mul_of_nonneg_left hDProd h8Pos
    grind
  exact Rat.le_trans hNumLe hBound

/-- 高维非阿贝尔矩阵穿墙路径 `X(t) = c(t) σ_z + s(t) σ_x`。 -/
def involutivePath (t : Rat) : Mat2C :=
  Mat2C.smul (pathCos t) Mat2C.σz + Mat2C.smul (pathSin t) Mat2C.σx

/-- 单位四元数 $\mathbb{S}^3$ 穿墙路径 `q(t) = c(t) + s(t) i`。 -/
def quatPath (t : Rat) : Quat :=
  ⟨pathCos t, pathSin t, 0, 0⟩

theorem involutivePath_endpoints :
    involutivePath 0 = Mat2C.σz ∧ involutivePath 1 = -Mat2C.σz := by
  have hc0 : pathCos 0 = 1 := by unfold pathCos pathDen; grind
  have hs0 : pathSin 0 = 0 := by unfold pathSin pathDen; grind
  have hc1 : pathCos 1 = -1 := by unfold pathCos pathDen; grind
  have hs1 : pathSin 1 = 0 := by unfold pathSin pathDen; grind
  constructor
  · apply Mat2C.ext <;> apply GaussRat.ext <;>
      simp [involutivePath, hc0, hs0, Mat2C.σz, Mat2C.σx] <;> grind
  · apply Mat2C.ext <;> apply GaussRat.ext <;>
      simp [involutivePath, hc1, hs1, Mat2C.σz, Mat2C.σx] <;> grind

theorem quatPath_endpoints :
    quatPath 0 = 1 ∧ quatPath 1 = -1 := by
  have hc0 : pathCos 0 = 1 := by unfold pathCos pathDen; grind
  have hs0 : pathSin 0 = 0 := by unfold pathSin pathDen; grind
  have hc1 : pathCos 1 = -1 := by unfold pathCos pathDen; grind
  have hs1 : pathSin 1 = 0 := by unfold pathSin pathDen; grind
  constructor
  · apply Quat.ext <;> simp [quatPath, hc0, hs0]
  · apply Quat.ext <;> simp [quatPath, hc1, hs1] <;> grind

theorem involutivePath_lipschitz_dist (t t0 : Rat)
    (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (ht00 : 0 ≤ t0) (ht01 : t0 ≤ 1) :
    Mat2C.matDist (involutivePath t) (involutivePath t0) ≤ 8 * ratAbs (t - t0) := by
  have hc := pathCos_lipschitz01.2 t t0 ht0 ht1 ht00 ht01
  have hs := pathSin_lipschitz01.2 t t0 ht0 ht1 ht00 ht01
  have hNonneg : 0 ≤ 8 * ratAbs (t - t0) := Rat.mul_nonneg (by decide) (ratAbs_nonneg _)
  unfold Mat2C.matDist
  simp only [involutivePath, Mat2C.add_a, Mat2C.add_b, Mat2C.add_c, Mat2C.add_d,
    Mat2C.smul_a, Mat2C.smul_b, Mat2C.smul_c, Mat2C.smul_d,
    GaussRat.add_re, GaussRat.add_im, gauss_smul_re, gauss_smul_im, Mat2C.σz, Mat2C.σx]
  have haRe : ratAbs (pathCos t * 1 + pathSin t * 0 - (pathCos t0 * 1 + pathSin t0 * 0)) ≤
      8 * ratAbs (t - t0) := by
    have heq : pathCos t * 1 + pathSin t * 0 - (pathCos t0 * 1 + pathSin t0 * 0) =
        pathCos t - pathCos t0 := by grind
    rw [heq]; exact hc
  have hdRe : ratAbs (pathCos t * (-1) + pathSin t * 0 - (pathCos t0 * (-1) + pathSin t0 * 0)) ≤
      8 * ratAbs (t - t0) := by
    have heq : pathCos t * (-1) + pathSin t * 0 - (pathCos t0 * (-1) + pathSin t0 * 0) =
        -(pathCos t - pathCos t0) := by grind
    rw [heq, ratAbs_neg]; exact hc
  have hbcRe : ratAbs (pathCos t * 0 + pathSin t * 1 - (pathCos t0 * 0 + pathSin t0 * 1)) ≤
      8 * ratAbs (t - t0) := by
    have heq : pathCos t * 0 + pathSin t * 1 - (pathCos t0 * 0 + pathSin t0 * 1) =
        pathSin t - pathSin t0 := by grind
    rw [heq]; exact hs
  have hIm0 : ratAbs (pathCos t * 0 + pathSin t * 0 - (pathCos t0 * 0 + pathSin t0 * 0)) ≤
      8 * ratAbs (t - t0) := by
    have heq : pathCos t * 0 + pathSin t * 0 - (pathCos t0 * 0 + pathSin t0 * 0) = 0 := by grind
    rw [heq, ratAbs_zero]; exact hNonneg
  dsimp only [GaussRat.one_re, GaussRat.one_im, GaussRat.zero_re, GaussRat.zero_im,
    GaussRat.neg_re, GaussRat.neg_im]
  grind

theorem quatPath_lipschitz_dist (t t0 : Rat)
    (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (ht00 : 0 ≤ t0) (ht01 : t0 ≤ 1) :
    Quat.quatDist (quatPath t) (quatPath t0) ≤ 8 * ratAbs (t - t0) := by
  have hc := pathCos_lipschitz01.2 t t0 ht0 ht1 ht00 ht01
  have hs := pathSin_lipschitz01.2 t t0 ht0 ht1 ht00 ht01
  have hNonneg : 0 ≤ 8 * ratAbs (t - t0) := Rat.mul_nonneg (by decide) (ratAbs_nonneg _)
  have h00 : (0 : Rat) - 0 = 0 := by grind
  unfold Quat.quatDist quatPath
  dsimp only
  rw [h00, ratAbs_zero]
  grind

/--
**定理 1（`quaternion_involutive_lift_exactness`）**：
整合 V4B/C 高维非阿贝尔升维的三大精确数学性质：
1. **对合与共线分类**：任意 $\mathbb{S}^2$ 球面点构成的 Bloch 矩阵恒为厄米酉对合（`X† = X, X^2 = I2`），
   且任意两个球面矩阵可对易当且仅当它们严格共线（`X_j = ±X_i`）；
2. **退化同构布尔 XOR 群**：共线对易子空间 `{±σ_z}` 与四元数实轴 `{±1}` 上的群乘法严格退化
   并同构于精确布尔 `XOR` 群；
3. **无奇点连续穿墙路径**：存在连接 `+σ_z`（`+1`）与 `-σ_z`（`-1`）的连续光滑穿墙路径
   `involutivePath` 与 `quatPath`，在闭区间 `[0, 1]` 上全局满足 `8`-Lipschitz 一致连续性
   （从而严格 $\varepsilon$-$\delta$ 连续），且全程严格保持 `X(t)† = X(t), X(t)^2 = I2`（`q(t).normSq = 1`），
   无任何模长塌缩奇点。
-/
theorem quaternion_involutive_lift_exactness :
    (∀ x y z : Rat, x * x + y * y + z * z = 1 →
      (Mat2C.bloch x y z).dagger = Mat2C.bloch x y z ∧
      Mat2C.bloch x y z * Mat2C.bloch x y z = Mat2C.I2) ∧
    (∀ x1 y1 z1 x2 y2 z2 : Rat,
      x1 * x1 + y1 * y1 + z1 * z1 = 1 → x2 * x2 + y2 * y2 + z2 * z2 = 1 →
      (Mat2C.commutator (Mat2C.bloch x1 y1 z1) (Mat2C.bloch x2 y2 z2) = 0 ↔
        (Mat2C.bloch x2 y2 z2 = Mat2C.bloch x1 y1 z1 ∨
         Mat2C.bloch x2 y2 z2 = -Mat2C.bloch x1 y1 z1))) ∧
    (∀ b1 b2 : Bool,
      Mat2C.commutator (spinMat b1) (spinMat b2) = 0 ∧
      (spinMat b1 * spinMat b2 * Mat2C.σz = spinMat (b1 ^^ b2)) ∧
      (spinQuat b1 * spinQuat b2 = spinQuat (b1 ^^ b2))) ∧
    (involutivePath 0 = Mat2C.σz ∧ involutivePath 1 = -Mat2C.σz ∧
      (∀ t : Rat, (involutivePath t).dagger = involutivePath t ∧
        involutivePath t * involutivePath t = Mat2C.I2) ∧
      (∀ t t0 : Rat, 0 ≤ t → t ≤ 1 → 0 ≤ t0 → t0 ≤ 1 →
        Mat2C.matDist (involutivePath t) (involutivePath t0) ≤ 8 * ratAbs (t - t0)) ∧
      quatPath 0 = 1 ∧ quatPath 1 = -1 ∧
      (∀ t : Rat, (quatPath t).normSq = 1) ∧
      (∀ t t0 : Rat, 0 ≤ t → t ≤ 1 → 0 ≤ t0 → t0 ≤ 1 →
        Quat.quatDist (quatPath t) (quatPath t0) ≤ 8 * ratAbs (t - t0))) := by
  refine ⟨fun x y z h => ⟨bloch_dagger x y z, bloch_sq_eq_I2 x y z h⟩,
    bloch_commutator_zero_iff_collinear, ?_, ?_⟩
  · intro b1 b2
    have hm := spinMat_xor_iso b1 b2
    have hq := spinQuat_xor_iso b1 b2
    exact ⟨hm.1, hm.2.2, hq⟩
  · refine ⟨involutivePath_endpoints.1, involutivePath_endpoints.2, ?_,
      involutivePath_lipschitz_dist, quatPath_endpoints.1, quatPath_endpoints.2, ?_,
      quatPath_lipschitz_dist⟩
    · intro t
      exact involutive_path_of_unit_circle (pathCos t) (pathSin t) (pathCos_sq_add_pathSin_sq t)
    · intro t
      unfold Quat.normSq quatPath
      dsimp only
      have h := pathCos_sq_add_pathSin_sq t
      grind

/-!
### 4. 定理 2：硬阈值数字锁存违反 Lipschitz 与 OGP 前提定理 (`latching_violates_lipschitz_ogp_premise`)
-/

/-- 符号硬阈值锁存算子：`x < 0 ↦ -1`，否则 `1`。 -/
def hardLatch (x : Rat) : Rat := if x < 0 then -1 else 1

/-- V3 置信度阈值动态数字锁存算子（`0 < τ < 1`）：当 `|x| ≥ τ` 时硬锁存至 `±1`，否则保持连续状态 `x`。 -/
def dynamicLatch (τ x : Rat) : Rat :=
  if τ ≤ ratAbs x then (if x < 0 then -1 else 1) else x

/-- `n` 维向量推理步锁存算子（精确对应 `v3_solver.py` 逐坐标动态数字锁存）。 -/
def vecDynamicLatch (n : Nat) (τ : Rat) (v : Fin n → Rat) : Fin n → Rat :=
  fun i => dynamicLatch τ (v i)

/-- 前 `m` 维递归 $\ell_\infty$ 距离。 -/
def prefixDistInf (m : Nat) (f : Nat → Rat) : Rat :=
  match m with
  | 0 => 0
  | k + 1 => max (prefixDistInf k f) (f k)

/-- `Fin n → Rat` 上的 $\ell_\infty$ 距离。 -/
def vecDistInf {n : Nat} (u v : Fin n → Rat) : Rat :=
  prefixDistInf n (fun i => if h : i < n then ratAbs (u ⟨i, h⟩ - v ⟨i, h⟩) else 0)

/-- 前 `m` 维递归平方 $\ell_2^2$ 距离。 -/
def prefixDistSq (m : Nat) (f : Nat → Rat) : Rat :=
  match m with
  | 0 => 0
  | k + 1 => prefixDistSq k f + f k * f k

/-- `Fin n → Rat` 上的平方 $\ell_2^2$ 距离。 -/
def vecDistSq {n : Nat} (u v : Fin n → Rat) : Rat :=
  prefixDistSq n (fun i => if h : i < n then u ⟨i, h⟩ - v ⟨i, h⟩ else 0)

/-- OGP 禁行定理在度量空间上的稳定性前提：算子需存在有限全局 `L`-Lipschitz 常数（以 $\ell_\infty$ 或 $\ell_2^2$ 度量）。 -/
def SatisfiesOGPLipschitzPremise {n : Nat} (T : (Fin n → Rat) → (Fin n → Rat)) : Prop :=
  ∃ L : Rat, 0 < L ∧ ∀ u v : Fin n → Rat, vecDistInf (T u) (T v) ≤ L * vecDistInf u v

private theorem prefixDistInf_only_zero {m : Nat} {f : Nat → Rat}
    (hm : 1 ≤ m) (hf0 : 0 ≤ f 0) (hfRest : ∀ i, 1 ≤ i → i < m → f i = 0) :
    prefixDistInf m f = f 0 := by
  induction m with
  | zero => omega
  | succ k ih =>
    cases k with
    | zero =>
      simp only [prefixDistInf]
      grind
    | succ k' =>
      have hRestK : ∀ i, 1 ≤ i → i < k' + 1 → f i = 0 := fun i h1 h2 => hfRest i h1 (by omega)
      have ihK := ih (Nat.succ_pos k') hRestK
      have hLast : f (k' + 1) = 0 := hfRest (k' + 1) (by omega) (Nat.lt_succ_self _)
      rw [prefixDistInf, ihK, hLast]
      grind

private theorem prefixDistSq_only_zero {m : Nat} {f : Nat → Rat}
    (hm : 1 ≤ m) (hfRest : ∀ i, 1 ≤ i → i < m → f i = 0) :
    prefixDistSq m f = f 0 * f 0 := by
  induction m with
  | zero => omega
  | succ k ih =>
    cases k with
    | zero =>
      simp only [prefixDistSq]
      grind
    | succ k' =>
      have hRestK : ∀ i, 1 ≤ i → i < k' + 1 → f i = 0 := fun i h1 h2 => hfRest i h1 (by omega)
      have ihK := ih (Nat.succ_pos k') hRestK
      have hLast : f (k' + 1) = 0 := hfRest (k' + 1) (by omega) (Nat.lt_succ_self _)
      rw [prefixDistSq, ihK, hLast]
      grind

theorem hardLatch_not_lipschitz (L : Rat) (hL : 0 < L) :
    ¬ (∀ x y : Rat, ratAbs (hardLatch x - hardLatch y) ≤ L * ratAbs (x - y)) := by
  intro hLip
  let x : Rat := -1 / (2 * L)
  have h2L : 0 < 2 * L := by grind
  have hxNeg : x < 0 := by
    change -1 / (2 * L) < 0
    have hPos : 0 < 1 / (2 * L) := rat_div_pos (by decide) h2L
    have heq : (-1 : Rat) / (2 * L) = -(1 / (2 * L)) := by
      rw [Rat.div_def, Rat.div_def]; grind
    grind
  have hBound := hLip x 0
  have hLx : hardLatch x = -1 := by unfold hardLatch; simp [hxNeg]
  have hL0 : hardLatch 0 = 1 := by unfold hardLatch; simp
  rw [hLx, hL0] at hBound
  have hAbsLHS : ratAbs ((-1 : Rat) - 1) = 2 := by
    rw [ratAbs_of_neg (by grind)]
    grind
  have hAbsRHS : ratAbs (x - 0) = 1 / (2 * L) := by
    have heq : x - 0 = -(1 / (2 * L)) := by
      change -1 / (2 * L) - 0 = -(1 / (2 * L))
      rw [Rat.div_def, Rat.div_def]; grind
    rw [heq, ratAbs_neg, ratAbs_of_nonneg (Rat.le_of_lt (rat_div_pos (by decide) h2L))]
  rw [hAbsLHS, hAbsRHS] at hBound
  have hMul : L * (1 / (2 * L)) = 1 / 2 := by
    have hNe : L ≠ 0 := by grind
    rw [Rat.div_def, Rat.div_def]
    have hInv : (2 * L)⁻¹ = 2⁻¹ * L⁻¹ := by
      have hProd : (2 * L) * (2⁻¹ * L⁻¹) = 1 := by
        have h1 : (2 * L) * (2⁻¹ * L⁻¹) = (2 * 2⁻¹) * (L * L⁻¹) := by grind
        rw [h1, Rat.mul_inv_cancel 2 (by grind), Rat.mul_inv_cancel L hNe]
        grind
      have hL1 := congrArg (fun z => (2 * L)⁻¹ * z) hProd
      have h2LNe : 2 * L ≠ 0 := by grind
      have hAssoc : (2 * L)⁻¹ * ((2 * L) * (2⁻¹ * L⁻¹)) =
          ((2 * L) * (2 * L)⁻¹) * (2⁻¹ * L⁻¹) := by grind
      rw [hAssoc, Rat.mul_inv_cancel _ h2LNe] at hL1
      grind
    rw [hInv]
    have h1 : L * (1 * (2⁻¹ * L⁻¹)) = (L * L⁻¹) * (1 * 2⁻¹) := by grind
    rw [h1, Rat.mul_inv_cancel L hNe]
    grind
  rw [hMul] at hBound
  grind

theorem dynamicLatch_jump_counterexample {τ L : Rat}
    (hτ0 : 0 < τ) (hτ1 : τ < 1) (hL : 0 < L) :
    ∃ x y : Rat, 0 < x ∧ x < τ ∧ y = τ ∧
      L * ratAbs (x - y) < ratAbs (dynamicLatch τ x - dynamicLatch τ y) := by
  let δ1 : Rat := τ / 2
  let δ2 : Rat := (1 - τ) / (2 * (L + 1))
  let δ : Rat := min δ1 δ2
  have hδ1Pos : 0 < δ1 := rat_div_pos hτ0 (by decide)
  have h2L1Pos : 0 < 2 * (L + 1) := by grind
  have hδ2Pos : 0 < δ2 := rat_div_pos (by grind) h2L1Pos
  have hδPos : 0 < δ := by unfold δ; grind
  have hδLe1 : δ ≤ δ1 := by unfold δ; grind
  have hδLe2 : δ ≤ δ2 := by unfold δ; grind
  let x : Rat := τ - δ
  have hxPos : 0 < x := by
    have hHalf : δ1 < τ := by
      unfold δ1
      have hDivPos : 0 < τ / 2 := rat_div_pos hτ0 (by decide)
      have hSub : τ - τ / 2 = τ / 2 := by
        rw [Rat.div_def]
        have hInv2 : (2 : Rat) * (2 : Rat)⁻¹ = 1 := Rat.mul_inv_cancel 2 (by grind)
        grind
      grind
    grind
  have hxLtτ : x < τ := by grind
  have hxAbs : ratAbs x < τ := by
    rw [ratAbs_of_nonneg (Rat.le_of_lt hxPos)]
    exact hxLtτ
  have hτAbs : τ ≤ ratAbs τ := by
    rw [ratAbs_of_nonneg (Rat.le_of_lt hτ0)]
    grind
  have hDynX : dynamicLatch τ x = x := by
    unfold dynamicLatch
    have hNot : ¬ (τ ≤ ratAbs x) := by grind
    simp [hNot]
  have hDynτ : dynamicLatch τ τ = 1 := by
    unfold dynamicLatch
    have hNotNeg : ¬ (τ < 0) := by grind
    simp [hτAbs, hNotNeg]
  refine ⟨x, τ, hxPos, hxLtτ, rfl, ?_⟩
  rw [hDynX, hDynτ]
  have hLhsAbs : ratAbs (x - τ) = δ := by
    have heq : x - τ = -δ := by grind
    rw [heq, ratAbs_neg, ratAbs_of_nonneg (Rat.le_of_lt hδPos)]
  have hRhsAbs : ratAbs (x - 1) = 1 - τ + δ := by
    have hNeg : x - 1 < 0 := by grind
    rw [ratAbs_of_neg hNeg]
    grind
  rw [hLhsAbs, hRhsAbs]
  have hMulLe : L * δ ≤ L * δ2 := rat_mul_le_mul_of_nonneg_left hδLe2 (Rat.le_of_lt hL)
  have hLδ2 : L * δ2 < (1 - τ) / 2 := by
    unfold δ2
    have h1 : L * ((1 - τ) / (2 * (L + 1))) < (L + 1) * ((1 - τ) / (2 * (L + 1))) := by
      have hPos : 0 < (1 - τ) / (2 * (L + 1)) := hδ2Pos
      have hSub : 0 < ((L + 1) - L) * ((1 - τ) / (2 * (L + 1))) :=
        Rat.mul_pos (by grind) hPos
      grind
    have hEq : (L + 1) * ((1 - τ) / (2 * (L + 1))) = (1 - τ) / 2 := by
      have hL1Ne : L + 1 ≠ 0 := by grind
      rw [Rat.div_def, Rat.div_def]
      have hInv : (2 * (L + 1))⁻¹ = 2⁻¹ * (L + 1)⁻¹ := by
        have hProd : (2 * (L + 1)) * (2⁻¹ * (L + 1)⁻¹) = 1 := by
          have ha : (2 * (L + 1)) * (2⁻¹ * (L + 1)⁻¹) = (2 * 2⁻¹) * ((L + 1) * (L + 1)⁻¹) := by grind
          rw [ha, Rat.mul_inv_cancel 2 (by grind), Rat.mul_inv_cancel (L + 1) hL1Ne]
          grind
        have hL1 := congrArg (fun z => (2 * (L + 1))⁻¹ * z) hProd
        have h2Ne : 2 * (L + 1) ≠ 0 := by grind
        have hAssoc : (2 * (L + 1))⁻¹ * ((2 * (L + 1)) * (2⁻¹ * (L + 1)⁻¹)) =
            ((2 * (L + 1)) * (2 * (L + 1))⁻¹) * (2⁻¹ * (L + 1)⁻¹) := by grind
        rw [hAssoc, Rat.mul_inv_cancel _ h2Ne] at hL1
        grind
      rw [hInv]
      have ha : (L + 1) * ((1 - τ) * (2⁻¹ * (L + 1)⁻¹)) =
          ((L + 1) * (L + 1)⁻¹) * ((1 - τ) * 2⁻¹) := by grind
      rw [ha, Rat.mul_inv_cancel (L + 1) hL1Ne]
      grind
    grind
  have hHalfLt : (1 - τ) / 2 < 1 - τ + δ := by
    have h1τ : 0 < 1 - τ := by grind
    have hHalfEq : 1 - τ - (1 - τ) / 2 = (1 - τ) / 2 := by grind
    grind
  grind

theorem dynamicLatch_not_continuous_at_threshold {τ : Rat}
    (hτ0 : 0 < τ) (hτ1 : τ < 1) :
    ¬ RatContinuousAt (dynamicLatch τ) τ := by
  intro hCont
  obtain ⟨delta, hDeltaPos, hBall⟩ := hCont ((1 - τ) / 2) (rat_div_pos (by grind) (by decide))
  let δ := min (τ / 2) (delta / 2)
  have hδPos : 0 < δ := by
    have h1 : 0 < τ / 2 := rat_div_pos hτ0 (by decide)
    have h2 : 0 < delta / 2 := rat_div_pos hDeltaPos (by decide)
    unfold δ; grind
  have hxPos : 0 < τ - δ := by
    have h1 : δ ≤ τ / 2 := by unfold δ; grind
    grind
  have hxAbs : ratAbs (τ - δ) < τ := by
    rw [ratAbs_of_nonneg (Rat.le_of_lt hxPos)]
    grind
  have hBallSub : ratAbs ((τ - δ) - τ) < delta := by
    have heq : (τ - δ) - τ = -δ := by grind
    rw [heq, ratAbs_neg, ratAbs_of_nonneg (Rat.le_of_lt hδPos)]
    have h2 : δ ≤ delta / 2 := by unfold δ; grind
    grind
  have hRes := hBall (τ - δ) hBallSub
  have hDynX : dynamicLatch τ (τ - δ) = τ - δ := by
    unfold dynamicLatch
    have hNot : ¬ (τ ≤ ratAbs (τ - δ)) := by grind
    simp [hNot]
  have hDynτ : dynamicLatch τ τ = 1 := by
    unfold dynamicLatch
    have hτAbs : τ ≤ ratAbs τ := by
      rw [ratAbs_of_nonneg (Rat.le_of_lt hτ0)]
      grind
    have hNotNeg : ¬ (τ < 0) := by grind
    simp [hτAbs, hNotNeg]
  rw [hDynX, hDynτ] at hRes
  have hGap : ratAbs (τ - δ - 1) = 1 - τ + δ := by
    rw [ratAbs_of_neg (by grind)]
    grind
  rw [hGap] at hRes
  grind

/-- 仅第 0 维非零的测试向量 `basis0Vec n val`。 -/
def basis0Vec (n : Nat) (val : Rat) : Fin n → Rat :=
  fun i => if i.val = 0 then val else 0

private theorem vecDistInf_basis0 {n : Nat} (hn : 1 ≤ n) (a b : Rat) :
    vecDistInf (basis0Vec n a) (basis0Vec n b) = ratAbs (a - b) := by
  have h0n : 0 < n := by omega
  unfold vecDistInf
  rw [prefixDistInf_only_zero hn]
  · simp only [h0n, ↓reduceDIte, basis0Vec, ↓reduceIte]
  · simp only [h0n, ↓reduceDIte, basis0Vec, ↓reduceIte]
    exact ratAbs_nonneg _
  · intro i hi1 hin
    have hi0 : i ≠ 0 := by omega
    simp only [hin, ↓reduceDIte, basis0Vec, hi0, ↓reduceIte]
    have h00 : (0 : Rat) - 0 = 0 := by grind
    rw [h00, ratAbs_zero]

private theorem vecDistSq_basis0 {n : Nat} (hn : 1 ≤ n) (a b : Rat) :
    vecDistSq (basis0Vec n a) (basis0Vec n b) = (a - b) * (a - b) := by
  have h0n : 0 < n := by omega
  unfold vecDistSq
  rw [prefixDistSq_only_zero hn]
  · simp only [h0n, ↓reduceDIte, basis0Vec, ↓reduceIte]
  · intro i hi1 hin
    have hi0 : i ≠ 0 := by omega
    simp only [hin, ↓reduceDIte, basis0Vec, hi0, ↓reduceIte]
    grind

/--
**定理 2（`latching_violates_lipschitz_ogp_premise`）**：
对任意置信度阈值 `0 < τ < 1` 与任意向量维度 `n ≥ 1`：
1. 符号硬锁存 `hardLatch` 在全局不存在任何有限 `L`-Lipschitz 常数；
2. V3 置信度阈值动态锁存 `dynamicLatch τ` 在阈值切换点 `τ` 严格不连续，且不存在任何有限 `L`-Lipschitz 常数；
3. 任意 `n` 维向量的动态数字锁存算子 `vecDynamicLatch n τ` 严格违反 OGP 禁行定理所需的 `L`-Lipschitz
   稳定性前提（在 $\ell_\infty$ 与平方 $\ell_2^2$ 度量下均不存在任何有限 `L > 0` 使得 `T` 为 `L`-Lipschitz），
   从而从前提假设上严格豁免 OGP 禁行定理。
-/
theorem latching_violates_lipschitz_ogp_premise (τ : Rat) (hτ0 : 0 < τ) (hτ1 : τ < 1)
    (n : Nat) (hn : 1 ≤ n) :
    (∀ L : Rat, 0 < L → ¬ (∀ x y : Rat, ratAbs (hardLatch x - hardLatch y) ≤ L * ratAbs (x - y))) ∧
    (¬ RatContinuousAt (dynamicLatch τ) τ) ∧
    (∀ L : Rat, 0 < L → ¬ (∀ x y : Rat, ratAbs (dynamicLatch τ x - dynamicLatch τ y) ≤ L * ratAbs (x - y))) ∧
    (¬ SatisfiesOGPLipschitzPremise (vecDynamicLatch n τ)) ∧
    (∀ L : Rat, 0 < L → ¬ (∀ u v : Fin n → Rat,
      vecDistSq (vecDynamicLatch n τ u) (vecDynamicLatch n τ v) ≤ (L * L) * vecDistSq u v)) := by
  have hDyn0 : dynamicLatch τ 0 = 0 := by
    unfold dynamicLatch
    rw [ratAbs_zero]
    have hNot : ¬ (τ ≤ 0) := by grind
    simp [hNot]
  have hVecEq : ∀ val, vecDynamicLatch n τ (basis0Vec n val) = basis0Vec n (dynamicLatch τ val) := by
    intro val
    funext i
    unfold vecDynamicLatch basis0Vec
    by_cases hi : i.val = 0
    · simp [hi]
    · simp [hi, hDyn0]
  refine ⟨hardLatch_not_lipschitz, dynamicLatch_not_continuous_at_threshold hτ0 hτ1, ?_, ?_, ?_⟩
  · intro L hL hLip
    obtain ⟨x, y, _, _, _, hGap⟩ := dynamicLatch_jump_counterexample hτ0 hτ1 hL
    have hBound := hLip x y
    grind
  · intro ⟨L, hL, hVecLip⟩
    obtain ⟨x, y, _, _, _, hGap⟩ := dynamicLatch_jump_counterexample hτ0 hτ1 hL
    have hBound := hVecLip (basis0Vec n x) (basis0Vec n y)
    rw [hVecEq x, hVecEq y, vecDistInf_basis0 hn, vecDistInf_basis0 hn] at hBound
    grind
  · intro L hL hVecSq
    obtain ⟨x, y, _, _, _, hGap⟩ := dynamicLatch_jump_counterexample hτ0 hτ1 hL
    have hBound := hVecSq (basis0Vec n x) (basis0Vec n y)
    rw [hVecEq x, hVecEq y, vecDistSq_basis0 hn, vecDistSq_basis0 hn] at hBound
    have hAbsSq : ∀ z : Rat, ratAbs z * ratAbs z = z * z := by
      intro z
      unfold ratAbs
      split <;> grind
    have hSqGap : (L * ratAbs (x - y)) * (L * ratAbs (x - y)) <
        ratAbs (dynamicLatch τ x - dynamicLatch τ y) *
          ratAbs (dynamicLatch τ x - dynamicLatch τ y) := by
      have hPosA : 0 ≤ L * ratAbs (x - y) := Rat.mul_nonneg (Rat.le_of_lt hL) (ratAbs_nonneg _)
      simpa [ratPow] using ratPow_lt_ratPow_of_lt hPosA hGap (by decide : 1 ≤ 2)
    rw [hAbsSq] at hSqGap
    have hL2 : (L * ratAbs (x - y)) * (L * ratAbs (x - y)) = (L * L) * ((x - y) * (x - y)) := by
      have := hAbsSq (x - y)
      grind
    rw [hL2] at hSqGap
    grind

end PhaseSHA.Algebraic.NonAbelianLiftAndOGP
