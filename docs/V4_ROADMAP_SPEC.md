# Phase-SHA V4 路线图与数学规范总览（V4A / V4B / V4C & V4E 基模接口）

## 1. 战略定位与 Lean 4 形式化底座对齐

Phase-SHA V4 将 V3 的一维连续双极自旋空间 $[-1, +1]$ 提升为**无奇点高维球面流形消融阶梯**（**V4A → V4B → V4C**），在严格对齐已通过 Lean 4.36.0 零公理内核验证的数学定理基础上，突破 $R=18$（全 32-bit `W_0`）闭环墙（$W_{16} / E_{13} / W_0$），并为后续 **V4E（BTC 部分零前缀原像 PoW）** 与 **V4D（Neuro-SAT 混合破壁）** 提供原生通用基模接口：

- **模块 2 对齐**（[TwoAdicTowerSoundness.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/Algebraic/TwoAdicTowerSoundness.lean)）：在离散极限下通过 32 尺度 2-adic 复单位圆相角塔实现零伪极小值保证与单比特 $\pi$ 相位跃迁下界 `>= 4`；
- **模块 5 对齐**（[NonAbelianLiftAndOGP.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/Algebraic/NonAbelianLiftAndOGP.lean)）：
  - **无奇点 $8$-Lipschitz 穿墙路径**：利用正交/虚轴方向绕过标量坍缩奇点 $0 \times 0$，全程保持状态严格位于单位流形（零模长塌缩）；
  - **动态数字锁存 OGP 豁免**：推理期局域翻转校验 + 动态硬锁存在阈值处打破 Lipschitz 连续性前提，规避经典 OGP 禁行障碍；
- **比特-自旋统一约定**（与 [dual_basis.py](file:///g:/SHA-LM/src/dual_basis.py) 完全一致）：
  $$\text{bit } 0 \longleftrightarrow s = +1, \qquad \text{bit } 1 \longleftrightarrow s = -1$$
  在此标准 Ising 映射下，连续乘法在离散极限 $\{\pm 1\}$ 上严格同构于布尔异或（$\mathrm{XOR}$）。

---

## 2. 三变体精确流形定义与残差公式（严格全流程 FP32）

> **精度铁律**：V4 前向推理、流形运算、2-adic 相位差及反向梯度全程严格执行 **FP32 全精度**（线性代数核内部显式关闭 TensorFloat-32 与 FP16/BF16 AMP；2-adic 模加相位累积延续 [dual_basis.py](file:///g:/SHA-LM/src/dual_basis.py) 的 FP64 折返）。

### 2.1 V4A：$\mathbb{S}^1 \subset \mathbb{C}$ 复单位圆流形（Abelian 2D 消融基线）

- **状态表示**：$z = (x, y) \in \mathbb{S}^1$（$\sqrt{x^2 + y^2} = 1$），自旋轴为实部 $s = \mathrm{Re}(z) = x$；离散极限为实轴 $z = (\pm 1, 0)$，中性赤道点为 $(0, 1)$。
- **XOR 耦合残差**（复数乘法 $z_a z_b$）：
  $$\mathcal{E}_{\text{XOR2}}(z_a, z_b; z_y) = \|z_a z_b - z_y\|_2^2, \qquad \mathcal{E}_{\text{XOR3}}(z_a, z_b, z_c; z_y) = \|z_a z_b z_c - z_y\|_2^2$$
- **2-adic / 非线性门耦合**：$\Sigma_0, \Sigma_1, \sigma_0, \sigma_1$ 的 3 输入 XOR 直接在 $\mathbb{S}^1$ 上相乘并取实部，与 $s = \mathrm{Re}(z)$ 上的 $\mathrm{Ch}/\mathrm{Maj}$ 一同送入 `DualBasisEngine` 的 32 尺度 2-adic 相角能量塔。
- **虚轴退火**：$\mathcal{L}_{\text{comm}} \equiv 0$（阿贝尔群）；$\mathcal{L}_{\text{align}} = \lambda_{\text{align}}(t) \sum y^2$。

### 2.2 V4B：$\mathbb{S}^2 \subset \mathfrak{su}(2)$ Bloch 球面 $2 \times 2$ 厄米酉对合矩阵（对应 Lean 4 `Mat2C`）

- **状态表示**：$\mathbf{n} = (n_x, n_y, n_z) \in \mathbb{S}^2$，对应无迹厄米酉对合矩阵 $X = n_x \sigma_x + n_y \sigma_y + n_z \sigma_z$（满足 $X^\dagger = X,\; X^2 = I_2,\; \mathrm{Tr}(X) = 0$）。自旋轴为 $s = \frac{1}{2}\mathrm{Tr}(X \sigma_z) = n_z$，离散极限为共线子空间 $\mathbf{n} = (0, 0, \pm 1) \iff X = \pm \sigma_z$，中性赤道点为 $\mathbf{n} = (1, 0, 0) \iff \sigma_x$。
- **非阿贝尔 XOR 残差（严格校验奇偶次幂）**：
  - 由精确 Pauli 三重积公式（与 $2 \times 2$ 复矩阵乘积逐项恒等）：
    $$X_a X_b X_c = (i \tau) I_2 + \mathbf{p} \cdot \boldsymbol{\sigma}, \qquad \tau = (\mathbf{a} \times \mathbf{b}) \cdot \mathbf{c}, \quad \mathbf{p} = (\mathbf{a} \cdot \mathbf{b})\mathbf{c} - (\mathbf{a} \cdot \mathbf{c})\mathbf{b} + (\mathbf{b} \cdot \mathbf{c})\mathbf{a}$$
  - **3 输入 XOR（奇数次）**：
    $$\mathcal{E}_{\text{XOR3}} = \frac{1}{2}\|X_a X_b X_c - X_y\|_F^2 = \tau^2 + \|\mathbf{p} - \mathbf{y}\|_2^2, \qquad s_{\text{XOR3}} = \frac{1}{2}\mathrm{Tr}(X_a X_b X_c \sigma_z) = p_z$$
  - **2 输入 XOR（偶数次配 $\sigma_z$）**：
    $$\mathcal{E}_{\text{XOR2}} = \frac{1}{2}\|X_a X_b X_y - \sigma_z\|_F^2 = \tau'^2 + p_x'^2 + p_y'^2 + (p_z' - 1)^2$$
    （其中 $(\tau', \mathbf{p}')$ 为 $(X_a, X_b, X_y)$ 的三重积）；对于 $R > 16$ 的 `Eq S` LSB 9 项奇次乘积闭环同样直接做精确 $2 \times 2$ 酉矩阵连乘并与 $+\sigma_z$ 比较 Frobenius 范数平方。
- **对易子与主轴退火惩罚**：
  $$\mathcal{L}_{\text{comm}} = \lambda_{\text{comm}}(t) \sum_{(i,j)} \|[X_i, X_j]\|_F^2 = 8\,\lambda_{\text{comm}}(t) \sum_{(i,j)} \|\mathbf{n}_i \times \mathbf{n}_j\|_2^2, \qquad \mathcal{L}_{\text{align}} = \lambda_{\text{align}}(t) \sum (n_x^2 + n_y^2)$$

### 2.3 V4C：$\mathbb{S}^3 \subset \mathbb{H}$ 单位四元数流形（对应 Lean 4 `Quat`）

- **状态表示**：$q = (w, \mathbf{v}) = (w, x, y, z) \in \mathbb{S}^3$（$\|q\|_2 = 1$），自旋轴为实部 $s = \mathrm{Re}(q) = w$；离散极限为实轴 $q = (\pm 1, 0, 0, 0)$，中性赤道点为 $(0, 1, 0, 0)$。
- **非阿贝尔 XOR 残差（$SU(2)$ 群封闭 Hamilton 乘法 $\otimes$）**：
  $$\mathcal{E}_{\text{XOR2}}(q_a, q_b; q_y) = \|q_a \otimes q_b - q_y\|_2^2, \qquad \mathcal{E}_{\text{XOR3}}(q_a, q_b, q_c; q_y) = \|q_a \otimes q_b \otimes q_c - q_y\|_2^2$$
- **对易子与虚部退火惩罚**：
  $$\mathcal{L}_{\text{comm}} = \lambda_{\text{comm}}(t) \sum_{(i,j)} \|[q_i, q_j]\|_2^2 = 4\,\lambda_{\text{comm}}(t) \sum_{(i,j)} \|\mathbf{v}_i \times \mathbf{v}_j\|_2^2, \qquad \mathcal{L}_{\text{align}} = \lambda_{\text{align}}(t) \sum \|\mathbf{v}\|_2^2$$

---

## 3. 共用骨干、防躺平目标与 V4E 通用接口设计

1. **流形保范切空间更新**：
   - 继承 V3 已验证的 `BitGridBlock`（含 LSB↔MSB BiGRU 进位扫描与 $\mathrm{ROTR}(2,6,11,13,22,25)$ 特征通道），并从 [stage3b_r18_calibrated_best.pt](file:///g:/SHA-LM/checkpoints/stage3b_r18_calibrated_best.pt) 无损热启动全部共享层；
   - 新增流形横向输入投影 `manifold_in_proj`（含横向位置、横向黎曼梯度雷达及 6 组循环旋转）与切空间输出头 `manifold_tangent_head`（横向切片采用 `1e-2` 正交初始化，防止离散极点处切投影归零）；
   - 更新公式：$\Delta u_\perp = \Delta u - \langle \Delta u, u \rangle u,\quad u' = \frac{u + \alpha \Delta u_\perp}{\|u + \alpha \Delta u_\perp\|_2 + 10^{-8}}$。
2. **打破“BCE 0.50 躺平陷阱”的多粒子 Winner-Take-All (Best-of-$M$) 目标**：
   - 训练期每条哈希并行展开 $M$ 条随机横向初态探索轨（云端 `M=4`，本地 4060 冒烟 `M=2`）；
   - 物理能量按摘要组内最低能量粒子做 Best-of-$M$ Winner-Take-All 加权；
   - 对未锁存闭环墙轮次（`r0 / r13 / r16`）可调低或置零 BCE 权重（`closed_loop_bce_scale: 0.0`），并叠加反熵坍缩探索项，迫使基模主动沿高维通道寻路穿墙。
3. **面向 V4E 的原生任意掩码基模接口**：
   - 网络结构、边界包装 `ManifoldBoundary` 与物理残差引擎完全由通用位级 `static_mask` 及可选 `digest_target_mask` 驱动；
   - 原生支持**全摘要 4 轮倒推模式**与**部分零前缀摘要 + 多自由输入字模式**，V4A/B/C 基模权重可直接零改动加载进入 V4E 微调与 GRPO/RL 后训练。
4. **测试期慢思考（Test-Time Compute）与 Linux `torch.compile`**：
   - 推理引擎原生支持大粒子数 SMC（如 `512` 粒子）、深步数、双掩码动态数字锁存及历史最优精英回溯重采样；
   - 在 Linux + CUDA 云端自动启用 `torch.compile` 加速（权重保存严格剥离装饰器前缀，确保双端断点互通），本地 Windows 自动回退 eager 模式。
