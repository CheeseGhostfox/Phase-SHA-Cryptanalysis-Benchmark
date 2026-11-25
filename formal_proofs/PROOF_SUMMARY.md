# Phase-SHA V3/V4 形式化数学证明总表（Lean 4.36.0 零外部依赖内核验证）

## 1. 验证架构与环境说明

本目录（`formal_proofs/`）是一套完全独立于 GPU 训练进程的 **CPU 专用 Lean 4 形式化数学验证套件**。为兼顾 16 GB 物理内存与后台 GPU 任务防 OOM 约束，本套件**不依赖 Mathlib**，完全基于 Lean 4.36.0 标准库（`Init` / `Std` + 原生 `BitVec` + 内置 `Std.Tactic.BVDecide`）与显式构造的无公理精确代数结构完成全量机器验证。

### 验证环境与双逻辑分层

- **工具链版本**：`leanprover/lean4:v4.36.0`（Lake `5.0.0-src+611c117`）
- **并发与内存约束**：`lakefile.toml` 硬性配置 `moreLeanArgs = ["-j", "2", "-M", "2048"]`
- **双层模块化结构**：
  1. **Core 位向量与离散动力学层（`PhaseSHA/Core/`）**：
     - [BitVecIdentities.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/Core/BitVecIdentities.lean)（Module 1）
     - [LatchingSoundness.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/Core/LatchingSoundness.lean)（Module 3）
     - [PartialPreimagePoW.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/Core/PartialPreimagePoW.lean)（Module 4）
  2. **Algebraic 代数流形与复单位圆层（`PhaseSHA/Algebraic/`）**：
     - [LiteField.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/Algebraic/LiteField.lean)（精确有理域 $\mathbb{Q}$、高斯有理复域 $\mathbb{Q}[i]$、`ratFloor`/`circleDist` 与 $\varepsilon$-$\delta$ 连续性基础）
     - [TwoAdicTowerSoundness.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/Algebraic/TwoAdicTowerSoundness.lean)（Module 2）
     - [NonAbelianLiftAndOGP.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/Algebraic/NonAbelianLiftAndOGP.lean)（Module 5）
- **顶层聚合与审计**：[PhaseSHA.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA.lean)、[VerifyAxioms.lean](file:///g:/SHA-LM/formal_proofs/VerifyAxioms.lean)

---

## 2. 已通过内核检验的核心定理总表

| 模块 | 定理 / 关键推论名 | 核心数学结论 | 证法与内核依赖 | 对应 Phase-SHA V3/V4 架构与源码 |
| :--- | :--- | :--- | :--- | :--- |
| **Module 1** | `four_round_clamp_w_invariant` | 给定公开 `IV` 与最终摘要 `digest`，向前解析倒推 4 轮状态 $(A_{R-4}, A_{R-5}, A_{R-6}, A_{R-7})$ 精确还原真值，且与中间消息字 $W_{R-4..R-1}$ 严格无关 | `bv_decide` 分步消元；标准内核 + `bv_decide` 原生 LRAT 反射 | V3/V4 4 轮解析钳位无损性（[boundary_clamp.py](file:///g:/SHA-LM/src/boundary_clamp.py#L249-L273)） |
| **Module 1** | `four_round_clamp_independent_of_w` | 同一摘要与 `IV` 下，任意修改中间 4 轮消息字得到的合法轨迹在解析倒推后严格重合 | 由 `four_round_clamp_w_invariant` 导出 | 无 $W$ 边界钳位唯一性（[boundary_clamp.py](file:///g:/SHA-LM/src/boundary_clamp.py#L262-L273)） |
| **Module 1** | `r18_a10_algebraic_tunneling` | $A_{10} = E_{14} - A_{14} + \Sigma_0(A_{13}) + \mathrm{Maj}(A_{13}, A_{12}, A_{11})$ 在全 $2^{32}$ 空间恒成立，且在 $R=18$ 下与被锁死的 $E_{13}$ 及扩展消息字 $W_{16}$ 完全独立 | `bv_decide` 全空间证死 | $R=18$ 跨轮旁路代数穿隧（[boundary_clamp.py](file:///g:/SHA-LM/src/boundary_clamp.py) / V4D 约简） |
| **Module 1** | `r18_ch_masking_shield` / `r18_ch_masking_word_bv` / `r18_ch_masking_half_space_bijection` | 任意位索引 $k \in [0, 31]$ 满足 $E_{15}[k]=1$ 时，$\mathrm{Ch}(E_{15}, E_{14}, E_{13})[k] = E_{14}[k]$ 严格独立于 $E_{13}[k]$；`flipBit32` 在触发集与未触发集间构成精确对合双射（50% 位级穿隧率） | 位展开 + `bv_decide`；其中 `r18_ch_masking_shield` 仅用 `[propext, Classical.choice, Quot.sound]` | $R=18$ 50% 位级屏蔽护盾与 V4D Neuro-SAT 可信等价约简（[sha256_ref.py](file:///g:/SHA-LM/src/sha256_ref.py#L80-L85)） |
| **Module 2** | `two_adic_zero_iff_mod_add` / `two_adic_zero_iff_all_scales_mod` / `two_adic_zero_iff_mod_add_cayley` | 对任意单位圆残差核 $K$（含复平面 $\mathbb{S}^1 \subset \mathbb{Q}[i]$ 的 `cayleyCircleKernel` 与 `jordanLowerKernel`），32 尺度相角塔总能量为 `0` 当且仅当 32 位模加 $a + b = c \pmod{2^{32}}$ 精确成立，亦等价于全部 32 尺度同余同时成立（零伪极小值陷阱） | 整数整除链归约；仅依赖 `[propext, Classical.choice, Quot.sound]` | Dual-Basis 32 尺度 2-adic 相角塔能量景观无伪解保证（[dual_basis.py](file:///g:/SHA-LM/src/dual_basis.py#L67-L186)） |
| **Module 2** | `lsb_carry_decoupling_and_flip_lower_bound` / `lsb_flip_b_c_lower_bound` | 尺度 $m=1$（LSB）残差严格仅依赖 $a[0] \oplus b[0] \oplus c[0]$ 而零进位耦合；对任意合法解翻转 `a`（或 `b`、`c`）的任意单比特 $k$，必在尺度 $m=k+1$ 触发精确 $\pi$ 相位反转（`cayleyUnitPoint` 跳转至 `-1`）与最大局部能量跃迁 `4`（更低尺度保持 `0`，总能量下界 `≥ 4`） | `bv_xor_eq_add_of_and_zero` 通用位偏移引理 + 模算术；标准内核 + `bv_decide` | 全 32 尺度相角塔消除单比特翻转局部梯度消失（[dual_basis.py](file:///g:/SHA-LM/src/dual_basis.py#L220-L270)） |
| **Module 3** | `exact_latch_preserves_solution_set` / `exact_latch_interior_preserves_solution_set` | 一般开环 ARX 链及 `sha256RoundBijective` 双射实例化下，若边界离散状态满足局部零物理残差，将其固化为新边界后的剩余子链解集与原链完全同构，且该边界处单步传播误差严格归 `0` | 构造性类型论（零公理依赖：`does not depend on any axioms`） | V3 动态数字锁存边界固化无损性（[v3_solver.py](file:///g:/SHA-LM/src/v3_solver.py#L328-L515)） |
| **Module 3** | `unbounded_open_loop_propagation` | 构造可计算半径 $\le 2$ 局域比特翻转校验算子 `latchStepRadius2`，对任意深度 $L \in \mathbb{N}$ 做数学归纳证明：只要每轮预测落入局部纠错半径（$\le 2$ 比特翻转）内，每步误差即被硬重置归 `0`，实现脱离李雅普诺夫指数发散视界的无界续航 | 结构健全性与完备性证明 + 对 `L` 数学归纳；仅依赖 `[propext, Classical.choice, Quot.sound]` | V3 16 候选局域翻转校验与开环多轮推进（[v3_solver.py](file:///g:/SHA-LM/src/v3_solver.py#L53-L327)） |
| **Module 4** | `btc_h7_tail_round_decoupling` / `btc_h0_h4_sensitive_to_last_w` / `btc_r20_w16_reaches_h7` | 证明 $H_7 = IV_7 + E_{R-3}$ 在代数上与尾 3 轮 $W_{R-3..R-1}$ 严格无关；而 $H_0, H_4$ 对末轮 $W_{R-1}$ 严格单射敏感，且扩展消息字 $W_{16}$ 直到 $R \ge 20$ 才首次通过 $E_{17}$ 严格到达 $H_7$（解释 $R=18$ 为何须钳位 $H_0/H_4$ 或推进至 $R \ge 20$） | 双轨状态展开 + `bv_decide` 消去；仅依赖 `[propext, Classical.choice, Quot.sound]` | V4E 比特币部分原像轮数与钳位设计准则（[sha256_ref.py](file:///g:/SHA-LM/src/sha256_ref.py#L311-L325)） |
| **Module 4** | `partial_preimage_exponential_gain` / `partial_preimage_bias_sandwich` | 在 $Z$ 个独立目标零位上，单比特命中概率 $p$ 相对均匀基线 $(1/2)^Z$ 的期望加速倍率严格为 $(2p)^Z$；当 $p > 1/2, Z \ge 1$ 时呈严格指数增长，且达到目标加速比 $M$ 所需的最小单比特偏置闭式充要下界为 $p \ge \frac{1}{2} M^{1/Z}$（含无理根有理夹逼推论） | 有理幂单调性与可逆性定理；仅依赖 `[propext, Classical.choice, Quot.sound]` | V4E BTC 部分零前缀期望倍率评估基准 |
| **Module 5** | `quaternion_involutive_lift_exactness` | 构造显式 `Mat2C`（$\mathbb{Q}[i]$ 上 $2\times 2$ 矩阵）与 4-元组 `Quat`（$\mathbb{H}$）：(a) $\mathbb{S}^2$ Bloch 矩阵恒为厄米酉对合 $X^\dagger = X, X^2 = I_2$，且对易充要于共线 $X_j = \pm X_i$；(b) 共线子空间乘法严格退化并同构于精确布尔 `XOR` 群；(c) 显式无奇点连续光滑穿墙路径 `involutivePath`/`quatPath` 连接 $+1$ 与 $-1$，在 `[0, 1]` 上全局满足 $8$-Lipschitz 一致连续性并全程严格保持 $X(t)^2 = I_2$（零模长塌缩） | Lagrange 恒等式 + 显式有理光滑路径（附一般单位圆推论 `involutive_path_of_unit_circle`）；仅依赖 `[propext, Classical.choice, Quot.sound]` | V4B/C 高维非阿贝尔升维拓扑穿墙通道 |
| **Module 5** | `latching_violates_lipschitz_ogp_premise` | 证明符号硬锁存 `hardLatch` 与置信度阈值锁存 `dynamicLatch τ`（$0 < \tau < 1$）在阈值点 $\tau$ 严格不连续，且对任意有限常数 $L > 0$ 与任意维度 $n \ge 1$（在 $\ell_\infty$ 与平方 $\ell_2^2$ 度量下）均不满足 $L$-Lipschitz 连续性，严格打破 OGP 禁行定理的稳定性前提 | 显式阈值左逼近反例 + 坐标嵌入归约；仅依赖 `[propext, Classical.choice, Quot.sound]` | V3/V4 离散-连续混合数字锁存豁免 OGP 障碍（[v3_solver.py](file:///g:/SHA-LM/src/v3_solver.py#L274-L515)） |

---

## 3. 论文理论章节可直接引用的核心推导亮点

1. **精确无损性与 50% 屏蔽穿隧（Modules 1 & 4）**：
   - `four_round_clamp_w_invariant` 与 `r18_a10_algebraic_tunneling` 共同证明：SHA-256 终态剥离 `IV` 后的前 4 步解析倒推以及第 13 轮旁路公式均完全消除了轮常数 `K_t`、扩展消息字 `W_{16}` 与隐变量 `E_{13}` 的耦合，将 32 位算术约束精确无损地传递至深层自由变量。
   - `r18_ch_masking_shield` 与 `r18_ch_masking_half_space_bijection` 在全 $2^{32}$ 状态空间上给出 `Ch` 门的严格位级掩码等价规则：在对称对合双射意义下精确有 50% 的位满足 `E_{15}[k] = 1`，此时未知比特 `E_{13}[k]` 被代数屏蔽，直接为 V4D Neuro-SAT 提供零误差等价约简依据。
   - `btc_h7_tail_round_decoupling` 与 `btc_r20_w16_reaches_h7` 指出最后 3 轮的 `H_7` 代数盲区边界，从理论上解释了为何 `R=18` 真实 BTC 闭环必须观测 `H_0`/`H_4` 或将轮数拓展至 `R ≥ 20`。

2. **2-adic 相角塔零伪解与单比特梯度保障（Module 2）**：
   - 在具体复高斯单位圆 $\mathbb{S}^1 \subset \mathbb{Q}[i]$ 模型 `cayleyCircleKernel` 与一般单位圆核 `UnitCircleKernel` 上证明：`towerEnergy K a b c = 0 ↔ a + b = c`，消除了单尺度的进位折返伪极小值。
   - 任意合法解翻转第 `k` 位后，所有更低尺度 $1 \le m \le k$ 残差保持为 `0`，而在第 `m = k + 1` 尺度复点精确跃迁至 `-1`、局部能量跳变至理论极大值 `4`，确保优化势能面对任意单比特误差均不存在局部平坦盲区。

3. **动态锁存无界续航与 OGP 豁免机制（Modules 3 & 5）**：
   - 通过 `searchWithinRadius` 证明了当神经预测器的逐轮误差被控制在半径 $\le 2$ 比特翻转以内时，离散物理残差校验将唯一重构真值边界状态，从而将动态推进的误差累积由指数发散截断为每步归零。
   - 一方面，非阿贝尔升维流形 `X^2 = I_2`（$\mathbb{S}^2 \subset \mathfrak{su}(2)$ / $\mathbb{S}^3 \subset \mathbb{H}$）利用非对易方向在 `+σ_z` 与 `-σ_z` 两座离散孤岛之间建立全局 `8`-Lipschitz 连续无奇点通道；另一方面，动态数字锁存算子在阈值处引入严格非连续跳变，直接违反了经典 Overlap Gap Property (OGP) 禁行定理对算法稳定性的 $L$-Lipschitz 前提假设。

---

## 4. 可复现内核验证命令

所有验证均在限制并发（`-j 2 -M 2048`）下通过，执行目录为 `g:\SHA-LM\formal_proofs`：

```powershell
# 1. 添加 elan 到当前会话 PATH 并逐模块/全量编译
$env:PATH = "$env:USERPROFILE\.elan\bin;$env:PATH"
Set-Location g:\SHA-LM\formal_proofs
lake build

# 2. 执行全定理内核底层依赖审计（核实无任何未证占位项与自定义扩展声明）
lake env lean VerifyAxioms.lean
```

---

## 5. Module 6：Phase-SHA V5/V6 前置理论形式化验证（高维升维、纠缠态与无假坑动力学）

在**严格不改动** Modules 1–5 既有全部模块的前提下，新增子包 `PhaseSHA/V5/`（4 个独立模块，零 Mathlib 依赖），对 Phase-SHA V5/V6 候选架构的四条核心数学机制及对应边界/反例进行纯 Lean 4 内核零占位符机器验证：

### 5.1 新增文件清单

1. **Module 6A — 八元数非结合流形升维**：[OctonionLift.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/V5/OctonionLift.lean)
2. **Module 6B — Burer-Monteiro 超高维升维与假坑逃逸证书**：[BurerMonteiroEscape.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/V5/BurerMonteiroEscape.lean)
3. **Module 6C — 张量网络纠缠态消除平均场势垒**：[TensorEntanglementBarrier.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/V5/TensorEntanglementBarrier.lean)
4. **Module 6D — 双频域分裂与差分映射无假坑不动点定理**：[DualDomainRRR.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/V5/DualDomainRRR.lean)
5. **聚合入口与审计脚本更新**：[PhaseSHA.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA.lean)、[VerifyAxioms.lean](file:///g:/SHA-LM/formal_proofs/VerifyAxioms.lean)

### 5.2 核心定理与实测内核依赖总表

| 子模块 | 定理名 | 核心数学结论与证明架构 | 内核基础依赖（`#print axioms` 实测） |
| :--- | :--- | :--- | :--- |
| **6A** | `oct_normSq_mul` / `oct_s7_closed` | Cayley-Dickson 翻倍四元数构造八元数 `Oct`；经内积交叉恒等式严格证死 **Degen 八平方恒等式** `normSq (x * y) = normSq x * normSq y`，单位球面 $\mathbb{S}^7$ 在乘法下封闭 | `[propext, Classical.choice, Quot.sound]` |
| **6A** | `oct_real_axis_xor_and_assoc` | 离散实轴 $\{\pm 1\} \subset \mathbb{S}^7$ 在 2 输入与 3 输入乘法下精确同构于布尔 `XOR`，且结合子 `assoc x y z` 在真解实轴上恒归零 | `[propext, Classical.choice, Quot.sound]` |
| **6A** | `oct_nonassociative_and_alternative` | 显式见证 `(e_1 * e_2) * e_4 ≠ e_1 * (e_2 * e_4)`（严格非结合性），同时全空间严格证明左右交错律 `x*(x*y)=(x*x)*y` 与 `(y*x)*x=y*(x*x)` | `[propext, Classical.choice, Quot.sound]` |
| **6A** | `oct_seven_imag_paths_unit_lipschitz` | 为全部 7 条虚轴基元 $e_1..e_7$ 构造连接 `+1` 与 `-1` 的光滑穿墙路径 `octPath k t`，全程满足 `normSq = 1`、避开零奇点且 `[0,1]` 全局 `8`-Lipschitz 连续 | `[propext, Classical.choice, Quot.sound]` |
| **6B** | `energy_rank1_eq_ising` | 高维单位超球面二次耦合能量 `E(V)` 在秩 1 离散嵌入 `rank1Embed` 下精确恢复离散 Ising/XOR 能量 | `[propext, Classical.choice, Quot.sound]` |
| **6B** | `burer_monteiro_weak_duality_global_opt` | 若对偶松弛矩阵 `slackMatrix C V` 半正定，则 `V` 对**任意更高维度 `K`** 的一切单位配置均是全局最优（`energy C V ≤ energy C W`） | `[propext, Classical.choice, Quot.sound]` |
| **6B** | `burer_monteiro_plus_one_dim_escape` | 若 `u^T S(V) u < 0`，沿新增第 `N+1` 维构造半角有理 Cayley 扰动 `liftedCurve V u t`：全程严格保范、`t=0` 处 $\varepsilon$-$\delta$ 一阶导为 `0`、二阶导精确等于 `u^T S(V) u < 0`（含全局四次余项界），且显式无开方步长 `ε > 0` 严格降低能量 | `[propext, Classical.choice, Quot.sound]` |
| **6B** | `burer_monteiro_frustrated_triangle_gap` | **诚实边界/反例**：`m=3` 反铁磁三角形上 1D 离散最优为 `-1`（且在方向 `(1,-1,0)` 有负曲率 `-2`），升至 2D 单位圆显式有理点可达 `-37/25 < -1`（刻画 SDP 整合间隙） | `[propext, Classical.choice, Quot.sound]` |
| **6C** | `independent_spin_mean_field_barrier` | 独立连续自旋残差 `(x_a x_b - 1)^2` 在均值 `(0,0)` 与任意过零点均达到势垒 `1`；基于有限等分网格无 IVT 公理构造成立：任意连接 `(1,1)` 到 `(-1,-1)` 的一致连续有理路径，对一切 `B < 1` 均存在能量 `> B` 的采样点 | `[propext, Classical.choice, Quot.sound]` |
| **6C** | `rank2_entangled_zero_barrier` | Rank-2 双比特路径 `entPath t` 在 `[0,1]` 全局 `4`-Lipschitz 连续，全程保范 `1` 且约束违约能量恒为 `0`，在纯 $\mathbb{Q}$ 任意逼近零边缘期望；同时在扩域 `QSqrt2` 上实现精确 $\theta=\pi/4$ Bell 态 `bellPlus`（精确零边缘期望 + 零残差） | `[propext, Classical.choice, Quot.sound]` |
| **6C** | `mps_rank2_cycle_zero_residual` | 推广至任意长度 `L ≥ 2` 闭环链约束：键维 `χ = 2` Rank-2 MPS 沿连续路径全程总范数平方 `= 1`、循环违约能量 `= 0`；`QSqrt2` GHZ MPS 在**全部 `k : Fin L`** 单比特边缘期望严格归零 | `[propext, Classical.choice, Quot.sound]` |
| **6C** | `rank1_product_zero_marginals_positive_energy` | **诚实对照**：键维 `χ = 1` 独立乘积态只要出现任一归零边缘期望即必有循环能量 `≥ 1` | `[propext, Classical.choice, Quot.sound]` |
| **6D** | `rrr_fixed_point_iff_intersection` | 无公理 `RatVecSpace` 上非凸 `ProjectionPair`（**无需任何凸性假设**）满足：对一切 `β ≠ 0`，`rrrStep x = x → P_A(x) = P_B(2P_A(x)-x) ∈ A ∩ B`，且真解不变、无交集无不动点 | `[propext, Quot.sound]`（无选择声明） |
| **6D** | `rrr_boundary_caveats` | **诚实边界/反例**：证出 `β = 0` 处处伪不动点、阴影点 `P_A(x)` 与迭代点 `x` 分离、以及无交集下线性漂移 `x_0 + kβ` 三类边界条件 | `[propext, Classical.choice, Quot.sound]` |
| **6D** | `walshChar_eq_boolSign_dotMod2` / `walsh_hadamard_involutive_and_xor_conv` | 任意宽度 `n` 布尔函数上 `walshChar u x = (-1)^{⟨u, x⟩₂}`、精确对合逆变换 `wht (wht f) v = 2^n f v` 与 XOR 卷积频域逐点对角化 `wht (xorConv f g) u = wht f u * wht g u` | `[propext, Classical.choice, Quot.sound]` |

### 5.3 诚实证伪边界与架构设计约束摘要

1. **非结合代数的结合子边界（6A）**：八元数乘法的非结合性是真实存在的（已通过 `e_1, e_2, e_4` 严格证伪全局结合律），因此算法必须在目标或退火日程中显式引入实轴收敛/结合子归零正则项；左右交错律保证了单变量幂与二次逆的良好结构。
2. **高维连续松弛与离散整数间隙（6B）**：+1 维 Burer-Monteiro 负曲率通道彻底打穿所有使松弛矩阵产生负特征值的低维假坑；但受挫三角形反例 `burer_monteiro_frustrated_triangle_gap` 严格表明无秩约束的高维连续极小值能量 `-37/25` 可严格低于原 1D 离散极小值 `-1`，工程求解需在逃出假坑后施加秩-1 投影/硬锁存回投。
3. **纠缠键维适用边界（6C）**：`χ = 2` 常数键维 MPS 之所以能无势垒归零单比特边缘期望且保持零残差，依赖两条全局互补分支的对称结构；对于任意一般高耦合非线性网络，零残差所需的最小键维 `χ` 将随约束图割宽（cut-width）纠缠熵增长。
4. **RRR 动力学读取与步长条件（6D）**：RRR 消灭一切假极小值不动点严格要求 `β ≠ 0`；同时停机不动点 `x` 未必处于交集内，工程解必须从阴影映射 `P_A(x)` 中提取；在不可满足或无解阶段则发生 `O(kβ)` 无界线性漂移而非静止躺平。

