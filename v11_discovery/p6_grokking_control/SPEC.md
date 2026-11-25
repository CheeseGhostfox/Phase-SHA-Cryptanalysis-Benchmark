# P6「真实 SHA-256 vs 随机常数/旋转对照组 + Grokking 顿悟解剖」数学契约与实验规范 (`SPEC.md`)

> **工作目录**：`g:\SHA-LM\v11_discovery\p6_grokking_control\`（独占，严禁修改 `v10_ouroboros64/`、`runs/`、`formal_proofs/`、`formal_proofs_v10/` 及其他 `pX` 目录）  
> **参考只读接口**：`g:\SHA-LM\src\sha256_ref.py`  
> **硬件与数值铁律**：本地仅允许 CPU（`CUDA_VISIBLE_DEVICES=""`）秒级冒烟测试，严禁调用本地 GPU；全量实验必须在 Colab GPU（`v11-p6`，`A100` 优先，其次 `L4` / `T4`，单会话总预算 $\le 60$ 分钟）上以纯 IEEE-754 FP32 执行：
> ```python
> torch.backends.cuda.matmul.allow_tf32 = False
> torch.backends.cudnn.allow_tf32 = False
> ```
> 全部模型参数、输入张量、激活与损失函数严格使用 `torch.float32`（严禁 TF32 / FP16 / BF16）；所有 SHA-256 及对照算子的前向演化严格使用 `torch.int64 & 0xFFFFFFFF` 精确无符号 32 位整数算术。

---

## 1. 核心科学问题与精确位序对齐

本任务检验在**充裕计算预算、强权重衰减正则化（$wd \in \{0.1, 0.5, 1.0, 2.0\}$）与长程优化步数**条件下，神经网络对减轮 SHA-256（$R \in \{4, 6, 8, 10, 12\}$）进行「有限子集过拟合 $\rightarrow$ 延迟顿悟泛化（Grokking）」训练时：
1. 真实 SHA-256（标准 `IV`、标准轮常数 `K[t]`、标准 `ROTR/SHR` 偏移量）是否会**先于或显著强于**「电路复杂度与计算图完全同构，但随机重采样 `ROTR/SHR` 偏移量与轮常数 `K[t]`」的**两套独立随机克隆对照组**发生顿悟泛化分离（$z_{\text{vs\_ctrl}} \ge 6.0$）；
2. 一旦发生延迟顿悟（以及在 $R=4$ 阳性校准区），通过 **Walsh-Hadamard / 离散傅里叶谱分解** 与 **过完备稀疏自编码器（SAE）+ 因果消融** 解剖顿悟电路的物理与代数机制。

### 与 `src/sha256_ref.py` 的严格位序与算术对齐
- **32-bit 字与大端单块填充**：严格对齐 `pack_int_messages_to_w16(w0)`（`src/sha256_ref.py` L158–L165）：
  对于 32-bit 输入消息字 $W_0 = x \in \mathbb{Z}_{2^{32}}$，单块 512-bit 填充消息为：
  $$W = \bigl[x,\; \text{0x80000000},\; 0,\; \dots,\; 0,\; 32\bigr] \in (\mathbb{Z}_{2^{32}})^{16}.$$
- **字内比特序（LSB-first）与双极自旋编码**：严格对齐 `words_to_bits` / `words_to_spins`（L116–L127）：
  第 $b \in \{0, \dots, 31\}$ 位为 `(word >> b) & 1`，双极自旋映射 $\mathcal{B}(\text{bit}) = 1.0 - 2.0 \cdot \text{bit} \in \{+1.0, -1.0\}$（$0 \mapsto +1, \; 1 \mapsto -1$）。
- **状态与摘要输出**：经 `sha256_trace(w16, rounds=R)`（L258–L327）运行 $R$ 轮后：
  $$S_R = [A_R, A_{R-1}, A_{R-2}, A_{R-3}, E_R, E_{R-1}, E_{R-2}, E_{R-3}], \qquad H_R = (S_R + \text{IV}) \bmod 2^{32}.$$

---

## 2. 双轨 Grokking 实验设计（固定切片预测 + 结构判别）

为从「直接切片函数逼近」与「差分结构判别」双视角完整覆盖减轮 Grokking 相变，设立两个互补且无 round-skipping 作弊的独立实验轨，共享同一扫频矩阵：

### 2.1 Track A：固定填充低熵切片多任务预测 (`TrackA_SlicePredict`)
- **低熵域**：$W_0 = x \in \{0, 1, \dots, 2^{14}-1\}$（$14$-bit 超立方体，总规模 $N_{\text{total}} = 16{,}384$），固定填充为 `pack_int_messages_to_w16(x)`。
- **零泄漏三互斥切分**（固定置乱种子 `SEED_SPLIT_A = 20261106`）：
  - $\mathcal{D}_{\text{train}}$：$N_{\text{train}} = 6{,}144$（$37.5\%$，用于有限集过拟合与长程 Grokking 优化）；
  - $\mathcal{D}_{\text{holdout}}$：$N_{\text{holdout}} = 6{,}144$（$37.5\%$，主留出评估集，严禁参与任何梯度更新或超参挑选）；
  - $\mathcal{D}_{\text{retest}}$：$N_{\text{retest}} = 4{,}096$（$25.0\%$，封印独立复验集，仅在触发亮灯/分离门限时解封）。
- **网络输入**：$14$-bit 双极自旋向量 $u(x) = \mathcal{B}(\text{bits}_{0..13}(x)) \in \{-1, +1\}^{14}$（`float32`）。
- **18 维多任务输出头**（严禁直接预测 $R=4$ 平凡仿射的 $A_1, E_1$）：
  1. **满 $R$ 轮摘要单比特分类（12 个二值目标，`BCEWithLogitsLoss`）**：
     - $H_R[0] = (A_R + \text{IV}_0) \bmod 2^{32}$ 的 6 个跨尺度比特 $b \in \{0, 1, 7, 15, 23, 31\}$；
     - $H_R[4] = (E_R + \text{IV}_4) \bmod 2^{32}$ 的 4 个跨尺度比特 $b \in \{0, 1, 15, 31\}$；
     - $\text{parity}(H_R[0])$ 与 $\text{parity}(H_R[4])$（全字 32-bit XOR 奇偶校验，2 个目标）。
  2. **次新轮过渡校准比特（4 个二值目标，`BCEWithLogitsLoss`）**：
     - $E_{R-1}$（即 $S_R[:, 5]$）的 4 个比特 $b \in \{0, 7, 15, 31\}$。在 $R=4$ 时为 $E_3$（3 轮非线性混合，CPU 实测最大 Walsh 谱幅值达 `0.1935`，构成既非平凡线性又具备可学习回路的阳性 Grokking 校准探针）；在 $R \ge 6$ 时已跨越 $\ge 5$ 轮非线性混合。
     - *审计隔离规则*：输出统计中须分别汇总 **`digest12`（前 12 个满 $R$ 轮摘要目标均值与最佳单比特）** 与 **`all16`（全部 16 个二值目标均值）**，深轮裁决以严格满 $R$ 轮指标为准。
  3. **连续摘要统计回归（2 个连续目标，`MSELoss`）**：
     - 标准化 64-bit 汉明重量：$y_{\text{hw}} = (\text{popcnt}(H_R[0]) + \text{popcnt}(H_R[4]) - 32.0) / 4.0$（均匀分布理论均值 $0$、方差 $1$）；
     - 标准化字值：$y_{\text{val}} = (H_R[0] / 2^{32} - 0.5) \cdot \sqrt{12}$（均匀分布理论均值 $0$、方差 $1$）。
  - **多任务联合损失**：$\mathcal{L}_A = \mathcal{L}_{\text{BCE, 16}} + 0.25 \cdot \mathcal{L}_{\text{MSE, 2}}$。

### 2.2 Track B：有限子集固定填充结构判别 Grokking (`TrackB_StructDiscrim`)
- **构造目的**：在有限样本过拟合 + 强权重衰减框架下探测减轮 SHA-256 差分/模加结构不变性。
- **样本生成（严格互斥种子池）**：
  - $\mathcal{D}_{\text{train}}$：$N_{\text{train}} = 8{,}192$（$4{,}096$ 正 $+ 4{,}096$ 负，固定种子 `SEED_B_TRAIN = 20261116`）；
  - $\mathcal{D}_{\text{holdout}}$：$N_{\text{holdout}} = 16{,}384$（$8{,}192$ 正 $+ 8{,}192$ 负，固定种子 `SEED_B_HOLD = 998244353`）；
  - $\mathcal{D}_{\text{retest}}$：$N_{\text{retest}} = 16{,}384$（$8{,}192$ 正 $+ 8{,}192$ 负，固定种子 `SEED_B_RETEST = 777666555`）。
  - **正样本 ($Y=1$)**：均匀随机抽取 $x \in \mathbb{Z}_{2^{32}}$，构造单比特最高位零进位微扰对 $(x,\; x \oplus 2^{31})$，经 `pack_int_messages_to_w16` 填充与 $R$ 轮前向后提取最新 4 个寄存器摘要字：
    $$U(x) = \bigl[H_R(x)[0],\; H_R(x)[1],\; H_R(x)[4],\; H_R(x)[5]\bigr] \in (\mathbb{Z}_{2^{32}})^4 \quad (128\text{ bits})$$
    （即 $(A_R, A_{R-1}, E_R, E_{R-1}) + \text{IV}$，**严格屏蔽 $A_{R-2}, A_{R-3}, E_{R-2}, E_{R-3}$**，避免 $R=4, 6$ 出现 1–3 轮浅层直接泄漏；CPU 已实测 $R=4$ 线性探针准确率达 `0.7158` 作为真实阳性校准，而 $R \ge 6$ 线性探针降至 `0.5061`）。
  - **同边缘负样本 ($Y=0$)**：独立均匀随机抽取 $(x, x') \in (\mathbb{Z}_{2^{32}})^2$，构造 $(U(x),\; U(x'))$。由于 $x \mapsto x \oplus 2^{31}$ 为双射，正负样本边缘分布完全一致。
- **对称多通道双极输入（512 维纯 `float32`）**：
  将 $(U_0, U_1)$（各 128 bits）编码为 4 通道拼合向量：
  $$\Psi_B(U_0, U_1) = \bigl[\mathcal{B}(U_0),\; \mathcal{B}(U_1),\; \mathcal{B}(U_0 \oplus U_1),\; \mathcal{B}((U_1 - U_0) \bmod 2^{32})\bigr] \in \{-1, +1\}^{512}.$$
  输出单个标量对数几率 $z_\theta \in \mathbb{R}$，最小化二元交叉熵 `BCEWithLogitsLoss`。

---

## 3. 模型架构、长程扫频与强制对照组

### 3.1 无偏置 Pre-LN 残差骨干网络 (`GrokkingResMLP`)
为保证对真实 SHA-256 与随机 `ROTR/K` 对照组**零结构偏置**（不硬编码任何 SHA-256 特定移位常数）：
- 输入投影：`Linear(D_in, D=192) -> LayerNorm(192) -> GELU`；
- 4 个 Pre-LN 残差 MLP 块（`Dropout = 0.0`，使 weight decay 成为唯一的容量压缩与 Grokking 驱动力）：
  $$h \leftarrow h + W_2\,\text{GELU}(W_1\,\text{LayerNorm}(h)), \qquad W_1 \in \mathbb{R}^{384 \times 192},\; W_2 \in \mathbb{R}^{192 \times 384};$$
- 输出头：`LayerNorm(192) -> Linear(192, D_out)`（Track A $D_{\text{out}}=18$，Track B $D_{\text{out}}=1$）。

### 3.2 扫频网格与优化协议
- **轮数扫频（5 档）**：$R \in \{4, 6, 8, 10, 12\}$；
- **权重衰减扫频（4 档）**：$wd \in \{0.1, 0.5, 1.0, 2.0\}$；
- **三组同屏对比变体（每格点 3 条完全同步的训练曲线）**：
  1. `REAL_SHA256`：标准 `SHA256_IV`、`SHA256_K` 与标准偏移量 $\Sigma_0:(2,13,22), \Sigma_1:(6,11,25), \sigma_0:(7,18;\text{SHR }3), \sigma_1:(17,19;\text{SHR }10)$；
  2. `CTRL_RAND_1`（同构随机克隆 #1，固定种子 `SEED_CTRL_1 = 1000003`）：
     - 保留 `SHA256_IV`、`Ch`、`Maj`、模 $2^{32}$ 加法拓扑、8 字双轨移位及固定填充格式；
     - 从 $\mathbb{Z}_{2^{32}}$ 独立均匀重采样 64 个轮常数 $K^{(1)}[0..63]$；
     - 无放回重采样互异 `ROTR` 偏移量（$\Sigma_0,\Sigma_1$ 各 3 个 $\subset \{1..31\}$；$\sigma_0,\sigma_1$ 各 2 个 $\subset \{1..31\}$）与 `SHR` 偏移量（$\sigma_0,\sigma_1$ 各 1 个 $\subset \{1..15\}$），且与标准 SHA-256 偏移组严格不同。
  3. `CTRL_RAND_2`（同构随机克隆 #2，固定种子 `SEED_CTRL_2 = 2000003`）：
     - 同上独立重采样另一套随机 $K^{(2)}[0..63]$ 与 `ROTR/SHR` 偏移量，且保证偏移组与 `REAL_SHA256` 及 `CTRL_RAND_1` 均互异。
- **优化器与步数预算**：
  - `AdamW(lr=1e-3, betas=(0.9, 0.98), weight_decay=wd)`，前 200 步线性 Warmup 至 `1e-3`，随后保持恒定学习率以维持高 `wd` 对权重范数的持续压缩；梯度裁剪 `max_norm=1.0`；
  - 批大小 `batch_size = 2048`（训练数据全程常驻 GPU VRAM，零 CPU-GPU 拷贝开销）；
  - **基础训练步数**：默认 `steps = 12,000`（若运行设备为 A100 且预估总耗时 $\le 40$ 分钟可升至 `15,000`；若为 T4 且校准预估 $> 50$ 分钟则自适应调至 `10,000` 以严格守住 60 分钟单会话上限）；
  - **延迟顿悟自动加训门（Auto-Extend）**：对任意 $R \ge 6$ 的运行，若在第 12,000 步满足「训练集已记忆（`train_acc >= 0.95`）且留出集呈现上升端倪（`holdout_auc >= 0.515` 或 $|z_{\text{null}}| \ge 3.5$）」，自动追加至 `24,000` 步以捕捉完整长程 Grokking 相变；
  - **轨迹采样**：每 `250` 步在 $\mathcal{D}_{\text{train}}$（或固定 2048 子批）与全量 $\mathcal{D}_{\text{holdout}}$ 上无梯度评估并记录 `train_acc, train_auc, holdout_acc, holdout_auc, holdout_mse, weight_l2_norm`。

### 3.3 0 参数静态基线 (`ZERO_PARAM_STATIC`)
在每个 $R$ 的同一 $\mathcal{D}_{\text{holdout}}$ 上同步报告两类零梯度优化静态基线：
1. **常数/随机零假定基线**：分类 `ACC = 0.50`、`AUC = 0.50`、标准化回归 `MSE = 1.0`；
2. **封闭式 1 阶经验相关/匹配滤波基线（仅访问 $\mathcal{D}_{\text{train}}$，无迭代）**：
   - Track A：在 $\mathcal{D}_{\text{train}}$ 上计算输入 14 比特与各二值目标的经验线性相关性权重 $w_j = \mathbb{E}_{\text{train}}[u_j y]$，直接作用于 $\mathcal{D}_{\text{holdout}}$ 计算 `AUC` 与 `ACC`（及标准化线性回归 `MSE`）；
   - Track B：包含严格 **0 参数负汉明重量基线** $s_0 = -\text{popcnt}(\Delta_{\text{xor}} U)$，以及仅在 $\mathcal{D}_{\text{train}}$ 统计 512 维输入均值差 $\hat{\mu} = \bar{\Psi}_+ - \bar{\Psi}_-$ 的**差分-线性匹配滤波基线**，在 $\mathcal{D}_{\text{holdout}}$ 直接评估 `AUC` 与 `ACC`。

---

## 4. 留出集指标、泛化分离 $z_{\text{vs\_ctrl}} \ge 6$ 与亮灯/复验门限

### 4.1 统计量与分池标准误
在 $\mathcal{D}_{\text{holdout}}$（及解封后的 $\mathcal{D}_{\text{retest}}$）上将样本划分为 $M = 16$ 个互斥子池计算分池标准误 $\text{SE}_{\text{pool}}(\text{AUC}) = s_{\text{AUC}} / \sqrt{16}$、$\text{SE}_{\text{pool}}(\text{ACC})$、$\text{SE}_{\text{pool}}(\text{MSE})$：
- **相对理论零假定 $z$-score**：
  $$z_{\text{null}}^{\text{AUC}} = \frac{\text{AUC} - 0.5}{\sqrt{\frac{N_+ + N_- + 1}{12 N_+ N_-}}}, \qquad z_{\text{null}}^{\text{ACC}} = \frac{\text{ACC} - 0.5}{\sqrt{0.25 / N_{\text{eval}}}}, \qquad z_{\text{null}}^{\text{MSE}} = \frac{1.0 - \text{MSE}}{\text{SE}_{\text{pool}}(\text{MSE})}.$$
  （Track A 对 12 个满 $R$ 轮摘要比特及全部 16 比特同时报告最大单比特 `best_bit_auc` / `best_bit_acc`、多目标平均 `mean_auc` / `mean_acc` 与 `mean_mse`；对 `best_bit` 须标注 $12$ 或 $16$ 重比较修正）。
- **相对随机克隆对照组的泛化分离检验 $z_{\text{vs\_ctrl}}$**：
  为避免挑选最弱对照组造成虚假显著，定义保守对照上界 $\text{ctrl}^* = \arg\max_{c \in \{1,2\}} \text{AUC}(\text{CTRL\_RAND\_}c)$（对 `MSE` 则取两者最小误差），计算：
  $$z_{\text{vs\_ctrl}}^{\text{AUC}} = \frac{\text{AUC}_{\text{real}} - \text{AUC}_{\text{ctrl}^*}}{\sqrt{\text{SE}_{\text{pool}}(\text{real})^2 + \text{SE}_{\text{pool}}(\text{ctrl}^*)^2 + 10^{-12}}}$$
  （同时分别记录 `z_vs_ctrl1`、`z_vs_ctrl2` 以及 `ACC` / `MSE` 的分离 $z$ 值）。
- **顿悟动力学指标**：
  - 记忆化步数 $T_{\text{mem}}$：首次 `train_acc >= 0.95` 的步数；
  - 顿悟步数 $T_{\text{grok}}$：首次在 $\mathcal{D}_{\text{holdout}}$ 达到 `z_null_auc >= 6.0`（或 `holdout_auc >= 0.60`）的步数；
  - 先发顿悟差 $\Delta T_{\text{grok}} = T_{\text{grok}}(\text{ctrl}^*) - T_{\text{grok}}(\text{real})$（若真实顿悟而对照未顿悟记为 `real_only`）。

### 4.2 全局结案裁决门限（Verdict Criteria）
- **`HIT`（发现超越同构随机控制的 SHA-256 Grokking 泛化分离）**：
  在任意深轮 **$R \ge 6$** 与任意 $(wd, \text{Track})$ 上（对 Track A 须体现于满 $R$ 轮 `digest12` 目标或其连续 `MSE`），**同时满足以下三项硬门限**：
  1. 真实 SHA-256 留出表现超越零假定：**$z_{\text{null}}^{\text{holdout}} \ge 6.0$**（若来自单比特峰值，Bonferroni 修正后仍需等价 $z \ge 6.0$）；
  2. 泛化分离门限：**$z_{\text{vs\_ctrl}}^{\text{holdout}} \ge 6.0$**（严格超越 `CTRL_RAND_1` 与 `CTRL_RAND_2` 双方）；
  3. **独立留出复验门限**：在从未参与任何调参或挑选的封印集 $\mathcal{D}_{\text{retest}}$ 上复验，仍同时满足 **$z_{\text{null}}^{\text{retest}} \ge 6.0$ 且 $z_{\text{vs\_ctrl}}^{\text{retest}} \ge 6.0$**。
- **`BORDERLINE`（边缘分离或仅浅层 $R=4$ 存在顿悟优势）**：
  未达到 `HIT`，但满足以下任一情形：
  1. 仅在 **$R = 4$** 上出现 `REAL_SHA256` 相对随机对照组的显著顿悟领先或幅度分离（$z_{\text{vs\_ctrl}} \ge 6.0$），而在 $R \ge 6$ 消失；
  2. 在 $R \ge 6$ 上出现 $3.5 \le z_{\text{vs\_ctrl}} < 6.0$ 且 $z_{\text{null}} \ge 4.0$。
- **`NULL`（未观测到超越同构随机控制的深轮 Grokking 分离）**：
  在 $R = 4$ 阳性校准区验证网络具备完整记忆与部分泛化/顿悟能力（Track B 阳性与 Track A $E_3$ 校准成功），但在全部 $R \ge 6$ 与所有 $wd \in \{0.1, 0.5, 1.0, 2.0\}$ 上，均止步于「训练集 $\approx 100\%$ 过拟合记忆 $\rightarrow$ 权重塌缩但留出集长期停留在随机墙（`AUC ≈ 0.50, ACC ≈ 0.50, MSE ≈ 1.0`）」或与两套随机对照组统计不可区分（$z_{\text{vs\_ctrl}} < 6.0$）。

---

## 5. Walsh / Fourier 谱分析与稀疏自编码器（SAE）顿悟电路解剖协议

对以下两类检查点**强制执行**全量谱与 SAE 电路解剖：
- **阳性校准必选**：$R=4$ 下 Track A（$E_3$ 最优比特及 $H_4[0]$ 最优比特）与 Track B（`REAL_SHA256` 及 `CTRL_RAND_1` 最强 `wd` 检查点），验证解剖管线有效；
- **延迟顿悟自动触发**：任意 $R \ge 6$ 中达到 $z_{\text{null}} \ge 4.0$ 或 $|z_{\text{vs\_ctrl}}| \ge 4.0$ 的检查点。

### 5.1 Walsh-Hadamard / 离散傅里叶谱分解
1. **Track A 全域精确 Walsh-Hadamard 变换（WHT）与 1D FFT**：
   - 在全部 $2^{14} = 16{,}384$ 个输入点上对模型实值 logit 输出 $f_\theta(x)$ 与真实布尔标签 $y(x)$ 分别执行精确快速 WHT，获得全 $16{,}384$ 个布尔频谱系数 $\hat{f}_S, \hat{y}_S$（$S \subseteq \{0,\dots,13\}$）：
     - 统计 Top-$k$（$k \in \{8, 32, 128\}$）谱能量占比、谱熵 $H_{\text{spec}}$、按汉明阶次 $d = |S| \in \{0..14\}$ 汇聚的各阶能量比 $E_d$；
     - 计算网络谱与真实布尔谱余弦相似度 $\cos(\hat{f}_\theta, \hat{y})$ 及 Top-32 主导 Walsh 项列表；
     - 对 $f_\theta(x)$ 沿整数序 $x=0..16383$ 做实值 1D FFT，统计 Top-16 模算术/2-adic 频点能量占比。
2. **Track B 差分晶体 Walsh 交互谱解析**：
   - 在 $\mathcal{D}_{\text{holdout}}$ 上提取中心化对数几率 $\tilde{f}_\theta(X) = z_\theta(X) - \bar{z}_\theta$；
   - 在 XOR 差分自旋向量 $\Delta_{\text{xor}} \in \{-1, +1\}^{128}$ 与模差自旋向量 $\Delta_{\text{mod}} \in \{-1, +1\}^{128}$ 上计算全 256 个 1 阶 Walsh 相关系数 $w_j$，选取 $|w_j|$ 最大前 24 位遍历全部 $\binom{24}{2} = 276$ 个 2 阶乘积项；
   - 用 Top-16 Walsh 基拟合线性代理模型，在 $\mathcal{D}_{\text{holdout}}$ 上报告解释方差 $R^2_{\text{Walsh}}$ 与纯解析 Walsh 还原 $\text{AUC}_{\text{Walsh}}$。

### 5.2 过完备稀疏自编码器（SAE）电路解剖与因果消融
1. **SAE 架构与训练（纯 FP32）**：
   - 收集检查点最后一个残差块输出激活 $h(x) \in \mathbb{R}^{192}$（$N \ge 6{,}144$ 样本）；
   - 训练 4 倍过完备 ReLU 稀疏自编码器（字典维数 $d_{\text{dict}} = 768$）：
     $$z = \text{ReLU}\bigl(W_{\text{enc}}(h - b_{\text{pre}}) + b_{\text{enc}}\bigr), \qquad \hat{h} = W_{\text{dec}} z + b_{\text{pre}},$$
     解码器列向量每步归一化 $\|W_{\text{dec}}[:, m]\|_2 = 1$，最小化 $\mathcal{L}_{\text{SAE}} = \|h - \hat{h}\|_2^2 + \lambda_{\text{L1}}\|z\|_1$（$\lambda_{\text{L1}} = 5 \times 10^{-3}$，Adam `lr=1e-3` 训练 2,000 步）。
2. **电路指标与因果验证**：
   - 报告重建解释方差 $R^2_{\text{SAE}}$、平均活跃单语义概念数 $L_0 = \mathbb{E}[\|z\|_0]$、死特征比例；
   - 计算每个活跃特征 $z_m$ 与真实目标 $y$ 及主导 Walsh 基的相关系数，提取 Top-8 单语义关键特征；
   - **因果消融（Causal Ablation）**：在 $\mathcal{D}_{\text{holdout}}$ 上将 Top-8 关键 SAE 特征置零并通过 $\hat{h}_{\text{abl}}$ 替换原隐层送入输出头，测量留出 $\text{AUC}$ / $\text{ACC}$ 跌落值 $\Delta \text{AUC}_{\text{ablate}}$，检验解剖出的稀疏电路对顿悟预测的因果必要性。

---

## 6. 工程交付 JSON 契约（供 `@p6-eng` 实施）

请 `@p6-eng` 在 `g:\SHA-LM\v11_discovery\p6_grokking_control\` 下实施脚本，本地 CPU（`CUDA_VISIBLE_DEVICES=""`）秒级冒烟通过后，上 Colab（`v11-p6`）全量烧制并同步回 `g:\SHA-LM\v11_discovery\p6_grokking_control\results\`，停机并核实 `colab.exe sessions` 后 @p6-math：

1. `results/p6_sweep_summary.json` 必须包含：
   - `meta`：`gpu_name`, `fp32_locked` (`true`), `tf32_disabled` (`true`), `rounds` (`[4,6,8,10,12]`), `weight_decays` (`[0.1,0.5,1.0,2.0]`), `variants` (`["REAL_SHA256","CTRL_RAND_1","CTRL_RAND_2"]`), `ctrl_configs`（两套对照组的完整 `rotr_offsets`, `shr_offsets`, `k_seed`, `k_head4_hex`）, `steps_base`, `batch_size`, `total_wall_sec`；
   - `track_a` 与 `track_b`：按 `results[str(R)][str(wd)][variant]` 组织的完整字典，每项包含：
     - `train_final`: `acc`, `auc`, `mse`（Track A）, `loss`
     - `holdout`: `acc`, `auc`, `auc_pool_mean`, `auc_pool_se`, `acc_pool_se`, `z_null_auc`, `z_null_acc`（Track A 额外含 `digest12_acc`, `digest12_auc`, `digest12_z_null_auc`, `best_bit_name`, `best_bit_auc`, `best_bit_z_null`, `mse`, `mse_pool_se`, `z_null_mse`）
     - `dynamics`: `t_mem`, `t_grok`, `min_weight_norm`, `final_weight_norm`, `extended`, `wall_sec`, `trajectory`（降采样轨迹列表：含 `step, train_acc, holdout_acc, holdout_auc, weight_norm`）
     - `retest`: 若触发独立复验门限则为完整复验字典（含 `acc, auc, z_null_auc, z_vs_ctrl_auc`），否则为 `null`
   - `comparisons`：按 `comparisons[track][str(R)][str(wd)]` 汇总 `delta_auc_vs_ctrl1`, `delta_auc_vs_ctrl2`, `z_vs_ctrl1`, `z_vs_ctrl2`, `z_vs_ctrl_min`（即相对最强对照组 `ctrl*` 的保守 `z_vs_ctrl`）, `t_grok_real`, `t_grok_ctrl_min`, `delta_t_grok`
   - `zero_param_baselines`：按 `zero_param_baselines[track][str(R)]` 汇总静态 0 参数/封闭线性基线的 `acc`, `auc`, `mse`
   - `dissections`：强制阳性校准及任何触发检查点的 `walsh_fourier`（含 `top_k_energy`, `spectral_entropy`, `degree_energy`, `cosine_to_target`, `fft_top16_ratio` / `walsh_surrogate_auc`）与 `sae`（含 `r2_sae`, `l0_mean`, `dead_frac`, `top8_features`, `causal_delta_auc_ablate`）。
2. `results/p6_run_meta.json`：记录精确命令行、各随机种子、Colab GPU 型号、总耗时及停机核实信息。
