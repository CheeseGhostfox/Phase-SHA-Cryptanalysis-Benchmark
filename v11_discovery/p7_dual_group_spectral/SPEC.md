# P7「双群混合基底稀疏化望远镜（$\mathbb{F}_2 \times \mathbb{Z}/2^{32}\mathbb{Z}$）」数学契约与实验规范 (`SPEC.md`)

> **工作目录**：`g:\SHA-LM\v11_discovery\p7_dual_group_spectral\`（独占，严禁修改其他 `pX`、`v10_ouroboros64/`、`runs/`、`formal_proofs/`、`formal_proofs_v10/`）  
> **只读参考接口**：`g:\SHA-LM\src\sha256_ref.py`、`g:\SHA-LM\src\dual_basis.py`  
> **核心科学问题**：SHA-256 交替使用 $\mathbb{F}_2^{32}$（异或/旋转/布尔选择）与 $\mathbb{Z}/2^{32}\mathbb{Z}$（模 $2^{32}$ 加法）两个不相容代数群。我们探测：能否通过纯实数 FP32 可微严格正交等价变换 $U_\theta \in O(d)$（Cayley 变换 + 联合 Walsh 与 2-adic/Real-DFT 混合基），使 $R \in \{2, 4, 6, 8, 10, 12\}$ 轮 SHA-256 的比特切片函数谱、输入-输出双群耦合表象或状态转移算子在**独立留出集**上显著稀疏化（更低归一化谱熵、更低 $\ell_1/\ell_2$ 比、更高 Top-$K$ 能量集中度），并超越全部经典固定基与同构随机对照？

---

## 1. 纯实数 FP32 锁定与精确压缩算子接口

### 1.1 精度与硬件保护硬约束
1. **严禁本地使用 GPU**：一切本地语法/冒烟测试必须设置 `CUDA_VISIBLE_DEVICES=""`，仅在 CPU 跑秒级验证；全量训练与扫频必须在 Colab GPU（`v11-p7`，优先 `L4`，不可用退 `T4`）上执行。
2. **纯 FP32 锁死（禁用 TF32 与复数张量）**：
   - 进程入口强制执行：
     ```python
     torch.backends.cuda.matmul.allow_tf32 = False
     torch.backends.cudnn.allow_tf32 = False
     torch.set_default_dtype(torch.float32)
     ```
   - 全流程严禁使用 `complex64` / `complex128` / `float16` / `bfloat16`；全部 2-adic 傅里叶基底与 Cayley 正交优化均以纯实数 `torch.float32` 展开实现（基础常数矩阵构造允许 `float64` 初算后精确截断为 `float32` 缓存）。

### 1.2 精确 $R$ 轮 SHA-256 算子与深度隔离规则
严格复用 `src/sha256_ref.py` 的无符号 32-bit 算术语义（`int64 & 0xFFFFFFFF`，LSB-first 位序，`SHA256_IV` 与 `SHA256_K`）：
- **扫频轮数阶梯**：$R \in \{2,\; 4,\; 6,\; 8,\; 10,\; 12\}$。
- **深度防作弊原则**：在全状态 $S_R = [A_R, A_{R-1}, A_{R-2}, A_{R-3}, E_R, E_{R-1}, E_{R-2}, E_{R-3}]$ 中，$A_{R-3}, E_{R-3}$ 在 $R=4$ 时距 $W_0$ 仅经过 1 轮压缩，且若变化 $W_{R-1}$ 则 $(A_R, E_R)$ 距 $W_{R-1}$ 亦仅 1 轮。因此本实验**全部扰动严格注入最早期输入字 $W_0$**（$W_{1..15}$ 作为独立隔离的固定钥匙/背景），并**只提取完整穿透全部 $R$ 轮的末端双轨输出字 $(A_R, E_R)$**（以及二者的跨字交互表象 $A_R \oplus E_R$ 与 $(A_R - E_R)\bmod 2^{32}$），彻底杜绝 1 轮捷径泄漏！

---

## 2. 纯实数 FP32 严格正交变换 $U_\theta$ 构造与守恒校验

### 2.1 Cayley 变换参数化（禁止缩放作弊）
对任意目标维度 $d \in \mathbb{N}_+$（如 $d=256$ 或 $d=93, 186$），参数化无约束实矩阵 $A_\theta \in \mathbb{R}^{d \times d}$，构建严格反对称生成元：
$$S_\theta = \frac{1}{2}\big(A_\theta - A_\theta^T\big), \qquad S_\theta^T = -S_\theta$$
通过纯实数 FP32 线性方程求解器（`torch.linalg.solve`）计算 Cayley 正交矩阵 $V_\theta \in SO(d)$：
$$V_\theta = (I_d - S_\theta)(I_d + S_\theta)^{-1} = \texttt{torch.linalg.solve}(I_d + S_\theta,\; I_d - S_\theta)^T$$
注：因 $S_\theta$ 为反对称矩阵，$(I_d + S_\theta)$ 特征值实部恒为 $1$，在实数域永远可逆且条件数有界；对任意初始正交基 $Q_{\text{init}} \in O(d)$，最终可微正交基定义为：
$$U_\theta = Q_{\text{init}} V_\theta \in O(d)$$
**严格禁用任何可学习对角缩放因子、权重衰减豁免下的范数放大或非正交线性层。**

### 2.2 四类代数初始化起点 $Q_{\text{init}} \in O(d)$
在 $d = 256$（8-bit 函数空间 $L^2(\{0,1\}^8)$）上，并行测试 4 种起点初始化（均以 $A_\theta \leftarrow 0$ 或微扰 $\mathcal{N}(0, 10^{-3})$ 起步）：
1. **`init_std`**：$Q_{\text{init}} = I_{256}$（标准坐标基起点）；
2. **`init_walsh`**：$Q_{\text{init}} = Q_{\text{WHT}}^{(256)}$（归一化 Sylvester Walsh-Hadamard 正交基，$\frac{1}{16}H_{256}$，对角化 $\mathbb{F}_2^8$）；
3. **`init_dft`**：$Q_{\text{init}} = Q_{\text{DFT}}^{(256)}$（归一化实数 2-adic 傅里叶正交基，对角化 $\mathbb{Z}/256\mathbb{Z}$）：
   $$q_0(x)=\frac{1}{16},\quad q_{2k-1}(x)=\frac{\sqrt{2}}{16}\cos\frac{2\pi k x}{256},\quad q_{2k}(x)=\frac{\sqrt{2}}{16}\sin\frac{2\pi k x}{256}\;\;(k=1..127),\quad q_{255}(x)=\frac{(-1)^x}{16}$$
4. **`init_hybrid`**：显式双群张量积混合基 $Q_{\text{init}} = Q_{\text{WHT}}^{(16)} \otimes Q_{\text{DFT}}^{(16)}$（高 4 位以 $\mathbb{F}_2^4$ Walsh 基展开、低 4 位以 $\mathbb{Z}/16\mathbb{Z}$ 实傅里叶基展开，严格满足 $(Q_A \otimes Q_B)^T(Q_A \otimes Q_B) = I_{256}$）。

### 2.3 强制正交守恒与范数不变性门禁
每次训练结束及留出集评估前，必须计算并记录以下三项守恒残差（任一超限即判定实验无效）：
1. **正交残差**：$\epsilon_{\text{ortho}}(U_\theta) = \|U_\theta^T U_\theta - I_d\|_F \le 10^{-4}$；
2. **行列式模残差**：$\epsilon_{\text{det}}(U_\theta) = \big||\det(U_\theta)| - 1\big| \le 10^{-4}$；
3. **$\ell_2$ 能量守恒残差**：$\epsilon_{\text{norm}} = \max_c \big|\|U_\theta^T g_{\cdot,c}\|_2^2 - \|g_{\cdot,c}\|_2^2\big| \le 10^{-4}$。

---

## 3. 三大互补双群谱探测表象（覆盖切片谱 / 耦合算子 / 转移矩阵）

### 3.1 探测一：多钥匙 8-bit 比特切片函数谱望远镜 (`Probe I: bit_slice_ensemble`, $U_\theta \in O(256)$)
解决单钥匙确定性置换在标准基下 $\ell_1/\ell_2 \equiv \sqrt{N}$ 的平凡性陷阱，直接测量**函数展开谱**在真实双群基下的跨钥匙泛化稀疏度：
- **输入切片家族**（独立跑全三类切片，注入 $W_0$）：
  1. `slice_lsb`：低字节 $W_0[0:8] \leftarrow x \in \{0,\dots,255\}$（模加进位起点）；
  2. `slice_msb`：高字节 $W_0[24:32] \leftarrow x$（最高位无上溢进位区）；
  3. `slice_strided`：步进分散 8-bit 切片 $W_0[\{0,4,8,12,16,20,24,28\}] \leftarrow x$（跨旋转分散耦合区）。
- **严格隔离的训练 / 留出钥匙池**：
  - 独立采样 $K_{\text{train}} = 16$ 个固定背景钥匙 $(W_0^{\text{mask}}, W_{1..15})$（训练集）与 $K_{\text{holdout}} = 32$ 个完全独立的固定背景钥匙（留出集）。
  - 对每个钥匙 $k$ 与全部 $x \in \{0,\dots,255\}$，采集 4 组输出字 $Y^{(k)}(x) = \big[A_R,\; E_R,\; A_R \oplus E_R,\; (A_R - E_R)\bmod 2^{32}\big]$ 共 $128$ 个双极自旋比特 $b \in \{-1, +1\}$。
  - **去直流与单位范数归一化**：对每列 $c \in \{1,\dots,128 K\}$ 执行列均值中心化 $\tilde{g}_{\cdot,c} = g_{\cdot,c} - \bar{g}_{\cdot,c}\mathbf{1}$，再单位化 $\hat{g}_{\cdot,c} = \tilde{g}_{\cdot,c} / \|\tilde{g}_{\cdot,c}\|_2$（若列方差为 0 则记为常数零列并剔除统计，避免除零）。由此构造训练集矩阵 $G_R^{(\text{train})} \in \mathbb{R}^{256 \times (16 \cdot 128)}$ 与留出集矩阵 $G_R^{(\text{holdout})} \in \mathbb{R}^{256 \times (32 \cdot 128)}$。
- **为什么无法作弊或过拟合**：$G_R^{(\text{train})}$ 含 $2048$ 列（远大于行数 $256$），$U_\theta \in O(256)$ 无法通过 PCA 记忆少量列，唯有跨钥匙普遍存在的真实双群代数结构才能在 $32$ 个完全未见的留出钥匙池 $G_R^{(\text{holdout})}$ 上保持稀疏性！

### 3.2 探测二：全 32-bit 双群输入-输出耦合表象 (`Probe II: io_dual_coupling`, $U_L \in O(93), U_R \in O(186)$)
针对全 32-bit 字 $W_0 \in \mathbb{Z}_{2^{32}}$ 与 $(A_R, E_R) \in (\mathbb{Z}_{2^{32}})^2$，构造**严格正交的 93 维联合 $\mathbb{F}_2 \times \mathbb{Z}/2^{32}\mathbb{Z}$ 基底** $\Phi_{\text{dual}}(x) \in \mathbb{R}^{93}$（满足 $\mathbb{E}_{x \sim U(\mathbb{Z}_{2^{32}})}[\Phi_{\text{dual}}(x)\Phi_{\text{dual}}(x)^T] = I_{93}$）：
1. **$\mathbb{F}_2$ 比特自旋区（32 维）**：$\Phi_{\text{bit}}(x)_k = s_k(x) = 1 - 2\cdot\text{bit}_k(x)$，$k \in \{0,\dots,31\}$。
2. **$\mathbb{Z}/2^{32}\mathbb{Z}$ 2-adic 非线性进位相位残差区（61 维）**：
   - 取 2-adic 谐波 $c_m(x) = \sqrt{2}\cos\frac{2\pi (x \bmod 2^m)}{2^m}$ 与 $q_m(x) = \sqrt{2}\sin\frac{2\pi (x \bmod 2^m)}{2^m}$（$m=2,\dots,32$）。
   - 解析正交定理：对任意 $m \ge 2$，谐波 $(c_m, q_m)$ 与 $k \neq m-1$ 的所有比特自旋**严格零相关**，仅与单个比特 $s_{m-1}(x)$ 存在解析重叠 $\alpha_m = \frac{2\sqrt{2}}{2^m}$、$\beta_m = \frac{2\sqrt{2}}{2^m}\cot\frac{\pi}{2^m}$。
   - 在每个尺度 $m$ 上解析减去向 $s_{m-1}(x)$ 的投影并作 $2 \times 2$ 精确白化（$m=2$ 时残差秩 1 退化为 $s_0 s_1$ 贡献 1 维；$m=3..32$ 各贡献 2 维，共 $1 + 30 \times 2 = 61$ 维），与 32 个单比特自旋合并即构成**全 32-bit 精确标准正交双群混合基 $\Phi_{\text{dual}}(x) \in \mathbb{R}^{93}$**。
- **跨钥匙耦合矩阵与双正交优化**：
  - 在 $K_{\text{tr}}=4$ 组固定背景钥匙下，每组采样 $N_{\text{tr}}=65,536$ 个 $W_0 \in [0, 2^{30})$（共 $262,144$ 样本），计算耦合矩阵 $C_R^{(\text{train})} \in \mathbb{R}^{93 \times (4 \cdot 186)}$（或每组 $93 \times 186$ 堆叠），优化双正交变换 $U_L \in O(93), U_R \in O(186)$ 使 $U_L^T C_R U_R$ 稀疏化。
  - 在 $K_{\text{te}}=4$ 组独立留出钥匙下，各自从完全隔离的 $W_0 \in [2^{30}, 2^{32})$ 采样 $16$ 个互斥评估池（每池 $8,192$ 样本，总留出样本 $524,288$），评估 $U_L^T C_R^{(\text{holdout})} U_R$ 的稀疏度与跨群/跨字交互能量分布（分解 $\mathbb{F}_2 \leftrightarrow \mathbb{F}_2$、$\mathbb{F}_2 \leftrightarrow \text{2-adic}$、$\text{2-adic} \leftrightarrow \text{2-adic}$ 四象限能量比）。

### 3.3 探测三：中心化比特切片转移算子共轭稀疏化 (`Probe III: operator_transfer`, $U_\theta^T \widetilde{T}_R U_\theta$)
对 8-bit 状态迁移建立非退化马尔可夫转移矩阵（避免确定性单射的平凡性）：
- 固定背景钥匙 $W_{1..15}$，将 $W_0[0:8]$ 设为输入 $x \in \{0,\dots,255\}$，并对辅助高半字节窗口 $W_0[8:14]$ 遍历全部 $64$ 个取值，统计输出低字节 $y = (A_R - E_R)[0:8] \bmod 256$（以及 $y = E_R[0:8]$）的条件转移概率矩阵 $T_R \in \mathbb{R}^{256 \times 256}$，减去均匀背景 $\widetilde{T}_R = T_R - \frac{1}{256}\mathbf{1}\mathbf{1}^T$。
- 在训练集钥匙池（$K_{\text{train}}=8$）上训练共享相似正交变换 $U_\theta \in O(256)$ 极小化 $U_\theta^T \widetilde{T}_R U_\theta$ 的矩阵稀疏度，在独立留出钥匙池（$K_{\text{holdout}}=16$）上评估其相对于标准基、Walsh 基、Real-DFT 基、Haar 随机基及同构对照组的泛化稀疏增益。

---

## 4. 稀疏度目标函数与强制三重对照

### 4.1 三大核心稀疏度指标
对任意经过正交变换后的实系数矩阵 $Z \in \mathbb{R}^{d \times M}$（$M$ 为列/矩阵样本数，对 Probe III 亦可按整体或列归一化），定义每列的能量概率分布 $p_{i,c} = \frac{Z_{i,c}^2}{\sum_{j=1}^d Z_{j,c}^2 + 10^{-12}}$：
1. **归一化谱熵 $H_{\text{spec}} \in [0, 1]$**（越低越稀疏）：
   $$H_{\text{spec}}(Z) = \frac{1}{M \ln d} \sum_{c=1}^{M} \left(-\sum_{i=1}^{d} p_{i,c} \ln(p_{i,c} + 10^{-12})\right)$$
2. **$\ell_1/\ell_2$ 稀疏比 $R_{\ell_1/\ell_2} \in [1, \sqrt{d}]$**（越低越稀疏）：
   $$R_{\ell_1/\ell_2}(Z) = \frac{1}{M} \sum_{c=1}^{M} \frac{\sum_{i=1}^{d} |Z_{i,c}|}{\sqrt{\sum_{i=1}^{d} Z_{i,c}^2 + 10^{-12}}}$$
3. **Top-$K$ 能量集中度 $E_{\text{top}K} \in [K/d, 1]$**（Probe I/III 取 $K=16$，Probe II 取 $K=32$；越高越稀疏）：
   $$E_{\text{top}K}(Z) = \frac{1}{M} \sum_{c=1}^{M} \sum_{i \in \text{Top}K(|Z_{\cdot,c}|)} p_{i,c}$$
- **可微训练目标**：$\mathcal{L}_{\text{sparse}}(U_\theta) = H_{\text{spec}}(Z) + 0.05\, R_{\ell_1/\ell_2}(Z)$；使用 `AdamW(lr=3e-2 -> 1e-3, CosineAnnealing)` 优化 `400` 步（梯度裁剪 `max_norm=1.0`）。在 `U_\theta` 四类初始化中，**严格基于训练集损失**自动择优选取最优 $U_\theta^*$ 送入留出集评估（严禁窥探留出集选模型）。

### 4.2 强制三重对照矩阵
对每个轮数 $R \in \{2, 4, 6, 8, 10, 12\}$，必须同时在相同样本规模与留出协议下评估：
1. **对照 1 — 标准比特/坐标基 (`baseline_std`)**：$U = I$（注：对 Rademacher $\pm 1$ 向量，$I$ 给出理论最稠密上界 $R_{\ell_1/\ell_2} = \sqrt{d}, H_{\text{spec}} = 1$）。
2. **对照 2 — 经典纯基与随机 Haar 基 (`baseline_walsh`, `baseline_dft`, `baseline_haar`)**：
   - 纯 Walsh-Hadamard 基 $Q_{\text{WHT}}$；
   - 纯实数 2-adic 傅里叶基 $Q_{\text{DFT}}$；
   - 随机 Haar 正交基 $Q_{\text{Haar}}$（精确剥离 Rademacher 向量经一般正交旋转后因中心极限定理变为高斯分布而天然产生的 $\ell_1/\ell_2 \approx \sqrt{2/\pi}\sqrt{d}$、$H_{\text{spec}} \approx 0.91$ 的通用几何收缩假象！）。
3. **对照 3 — 同构随机密码与随机双射对照 (`ctrl_rotr_k`, `ctrl_rand_perm`)**：
   - **`ctrl_rotr_k`（同构随机 `ROTR/K` 克隆密码）**：保留 SHA-256 全部模加、Ch、Maj 拓扑，重采样 64 个随机轮常数 $K_{\text{ctrl}}$ 与 $\Sigma_0, \Sigma_1, \sigma_0, \sigma_1$ 的合法 `ROTR/SHR` 偏移量，跑完全对称的训练与留出评估；
   - **`ctrl_rand_perm`（随机双射对照）**：
     - 在 Probe I / III 中，对每个密钥配置生成独立的均匀随机双射 $\pi_k \in S_{256}$ 作用于 $x$（等价于完全消除结构后的理想随机置换表象），跑完全对称的 $U_\theta$ 训练与留出评估；
     - 在 Probe II 中，对输入 $W_0$ 施加确定性随机 32-bit 强双射（4 轮 Feistel/SipHash-like 置换）后再送入同构电路或直接将输出 $(A_R, E_R)$ 配对独立随机 32-bit 双射，验证纯随机双射的蒙特卡洛零基线。

---

## 5. 独立留出集统计检验、$z \ge 6$ 亮灯门限与 $U_\theta$ 结构蒸馏

### 5.1 独立留出集 $z$-score 定义
在 Probe I（$K_{\text{holdout}} = 32$ 个独立留出钥匙池）、Probe II（$16$ 个独立留出样本池）及 Probe III（$K_{\text{holdout}} = 16$ 个独立留出钥匙池）上，令每个评估池 $k$ 上的**最佳经典固定基基准**为：
$$H_{\text{best\_cl}}^{(k)} = \min\big(H_{\text{std}}^{(k)},\; H_{\text{walsh}}^{(k)},\; H_{\text{dft}}^{(k)},\; H_{\text{haar}}^{(k)}\big), \qquad R_{\text{best\_cl}}^{(k)} = \min\big(R_{\text{std}}^{(k)},\; R_{\text{walsh}}^{(k)},\; R_{\text{dft}}^{(k)},\; R_{\text{haar}}^{(k)}\big)$$
定义池级稀疏化增益 $\Delta H_{\text{cl}}^{(k)} = H_{\text{best\_cl}}^{(k)} - H_{U_\theta^*}^{(k)}$ 与 $\Delta R_{\text{cl}}^{(k)} = R_{\text{best\_cl}}^{(k)} - R_{U_\theta^*}^{(k)}$（正值表示 $U_\theta^*$ 显著超越全部经典与 Haar 基）：
1. **超越全部经典基检验统计量 $z_{\text{vs\_classical}}$**：
   $$z_{\text{vs\_classical}}(H) = \frac{\overline{\Delta H}_{\text{cl}}}{\text{SE}(\Delta H_{\text{cl}}) + 10^{-12}}, \qquad z_{\text{vs\_classical}}(R) = \frac{\overline{\Delta R}_{\text{cl}}}{\text{SE}(\Delta R_{\text{cl}}) + 10^{-12}}$$
2. **超越同构随机对照组统计量 $z_{\text{vs\_rotr\_k}}$ 与 $z_{\text{vs\_rand\_perm}}$**：
   $$z_{\text{vs\_ctrl}} = \frac{\overline{\Delta H}_{\text{main}} - \overline{\Delta H}_{\text{ctrl}}}{\sqrt{\text{SE}(\Delta H_{\text{main}})^2 + \text{SE}(\Delta H_{\text{ctrl}})^2 + 10^{-12}}}$$

### 5.2 严格亮灯裁决门限（Verdict Criteria）
在**全部正交守恒校验通过**（$\epsilon_{\text{ortho}}, \epsilon_{\text{det}}, \epsilon_{\text{norm}} \le 10^{-4}$）且 **$R=2$ 阳性校准成功**（$z_{\text{vs\_classical}} \ge 6.0$）的前提下：
- **`HIT_SHA256_SPECIFIC`（发现 SHA-256 特异双群稀疏结构）**：
  在任意深轮 **$R \ge 6$**（或过渡轮 $R=4$ 亦单列报告）的独立留出集上，同时满足：
  1. $z_{\text{vs\_classical}}(H) \ge 6.0$ 或 $z_{\text{vs\_classical}}(R) \ge 6.0$（且另一指标及 $E_{\text{top}K}$ 方向同向改善）；
  2. $z_{\text{vs\_rotr\_k}} \ge 4.0$ 且 $z_{\text{vs\_rand\_perm}} \ge 6.0$。
- **`HIT_GENERIC_ARX`（发现通用 ARX 双群混合基稀疏结构）**：
  在任意深轮 **$R \ge 6$** 上满足 $z_{\text{vs\_classical}} \ge 6.0$ 且 $z_{\text{vs\_rand\_perm}} \ge 6.0$，但 $|z_{\text{vs\_rotr\_k}}| < 4.0$（说明双群稀疏化结构对旋转常数普遍成立而非 SHA-256 特异）。
- **`BORDERLINE`（边缘弱信号）**：
  仅在 $R=4$ 触发 $z_{\text{vs\_classical}} \ge 6.0$ 而 $R \ge 6$ 回落，或在 $R \ge 6$ 出现 $3.0 \le z_{\text{vs\_classical}} < 6.0$。
- **`NULL`（未观测到超越经典基与随机对照的深轮稀疏化突破）**：
  $R=2$ 阳性校准清晰成立，但跨过 $R \ge 6$（及 $R=4$）后，全部探索表象的留出集稀疏度均坍缩至 Haar 随机基准与同构随机对照水平（$z_{\text{vs\_classical}} < 3.0$，$U_\theta^*$ 无法跨钥匙泛化）。

### 5.3 $U_\theta^*$ 结构蒸馏协议
对 **`R = 2`（强制阳性校准）** 及任意触发 $z_{\text{vs\_classical}} \ge 4.0$ 的 $(R, \text{probe})$ 组合自动执行：
1. **双群正交投影谱（Walsh vs 2-adic DFT vs Tensor-Hybrid）**：计算重叠矩阵 $M_W = Q_{\text{WHT}}^T U_\theta^*$、$M_D = Q_{\text{DFT}}^T U_\theta^*$ 与 $M_H = (Q_{\text{WHT}}^{(16)} \otimes Q_{\text{DFT}}^{(16)})^T U_\theta^*$，报告各自的矩阵谱熵、对角能量占比 $\text{tr}(M \odot M) / d$ 及平均每个基向量所需的主导 Walsh/DFT 模态数。
2. **双群交互象限能量剖面（Probe II）**：报告 $\widetilde{C}_R$ 在 $[\mathbb{F}_2 \leftrightarrow \mathbb{F}_2,\; \mathbb{F}_2 \leftrightarrow \text{2-adic},\; \text{2-adic} \leftrightarrow \mathbb{F}_2,\; \text{2-adic} \leftrightarrow \text{2-adic}]$ 四个子块中的归一化 Frobenius 能量占比与奇异值衰减谱（Top-5 奇异值占比），揭示信号是由 $\mathbb{F}_2$ 主导、2-adic 主导还是双群混合模态主导。
3. **跨字耦合增益（$A_R, E_R$ vs $A_R \oplus E_R, (A_R - E_R)\bmod 2^{32}$）**：分别汇总单字通道与跨字组合通道的留出集稀疏度，检验消除共享轮函数 $T_1$ 后跨字差分是否保留更深轮的双群稀疏性。

---

## 6. 工程交付规范（`@p7-eng` 执行清单）

1. **代码与测试目录**：全部工程代码、本地 CPU 秒级单元测试与 Colab 启动脚本置于 `g:\SHA-LM\v11_discovery\p7_dual_group_spectral\`。
2. **本地 CPU 冒烟验证**：必须在 `CUDA_VISIBLE_DEVICES=""` 下通过秒级单测：
   - 精确对拍 `sha256_ref.py`；
   - 验证 Probe II 的 93 维 $\Phi_{\text{dual}}$ 理论正交性 $\|\frac{1}{N}\Phi^T\Phi - I_{93}\|_F$ 随 $N$ 按 $O(1/\sqrt{N})$ 收敛；
   - 验证 Cayley 变换在纯 FP32 下严格满足 $\epsilon_{\text{ortho}} < 10^{-4}$、$\epsilon_{\text{det}} < 10^{-4}$、$\epsilon_{\text{norm}} < 10^{-4}$；
   - 验证 $R=2$ 小批量冒烟流水线跑通并输出合法 JSON。
3. **Colab 烧制与停机规范**：
   - 会话名 `v11-p7`（`--gpu L4`，不可用退 `--gpu T4`），远端 `setsid nohup python3 -u`，等待期使用 `schedule` 定时器。
   - 全量结果同步回 `g:\SHA-LM\v11_discovery\p7_dual_group_spectral\results\`（至少包含 `p7_sweep_summary.json`、`p7_distillation.json`、`p7_run_meta.json`、`run_log.txt`）。
   - 同步完成后**立即执行 `colab.exe stop -s v11-p7` 并用 `colab.exe sessions` 核实停机**，随后 `@p7-math` 复核并产出 `verdict.json`（含 `"colab_stopped": true`）与 `REPORT.md`。

### `results/p7_sweep_summary.json` 字段契约
- `meta`：`gpu_name`, `fp32_locked` (`true`), `tf32_disabled` (`true`), `complex_disabled` (`true`), `rounds` (`[2,4,6,8,10,12]`), `total_wall_sec`, `max_ortho_err`, `max_det_err`, `max_norm_err`
- `probe1_bit_slice`：嵌套 `probe1_bit_slice[str(R)][slice_name][cipher]`（`cipher ∈ {"main", "ctrl_rotr_k", "ctrl_rand_perm"}`），每项含：
  - `baselines`：`std`, `walsh`, `dft`, `haar` 各自的 `{entropy_mean, entropy_se, l1_l2_mean, l1_l2_se, top16_mean, top16_se}`
  - `learned`：`best_init`, `ortho_err`, `det_err`, `norm_err`, `entropy_mean`, `entropy_se`, `l1_l2_mean`, `l1_l2_se`, `top16_mean`, `top16_se`
  - `comparison`（在 `main` 下提供）：`delta_entropy_vs_best_cl`, `z_entropy_vs_cl`, `delta_l1_l2_vs_best_cl`, `z_l1_l2_vs_cl`, `delta_top16_vs_best_cl`, `z_top16_vs_cl`, `z_entropy_vs_rotr_k`, `z_entropy_vs_rand_perm`, `z_l1_l2_vs_rotr_k`, `z_l1_l2_vs_rand_perm`
  - `cross_word_breakdown`：分 `A_R`, `E_R`, `A_xor_E`, `A_sub_E` 四组的留出集 `entropy_mean` 与 `l1_l2_mean`
- `probe2_io_coupling`：嵌套 `probe2_io_coupling[str(R)][cipher]`，含 `frobenius_norm_sq`, `mc_noise_floor`, `quadrant_energy_frac` (`f2_f2`, `f2_2adic`, `2adic_f2`, `2adic_2adic`), `top5_sv_ratio`, `baselines` (`std_bit`, `dual_raw`, `2adic_raw`, `haar`), `learned` (`ortho_err_L`, `ortho_err_R`, `entropy_mean`, `entropy_se`, `l1_l2_mean`, `l1_l2_se`, `top32_mean`), `comparison` (`z_entropy_vs_cl`, `z_l1_l2_vs_cl`, `z_vs_rotr_k`, `z_vs_rand_perm`)
- `probe3_operator_transfer`：嵌套 `probe3_operator_transfer[str(R)][target][cipher]`，含 `baselines`, `learned`, `comparison`
