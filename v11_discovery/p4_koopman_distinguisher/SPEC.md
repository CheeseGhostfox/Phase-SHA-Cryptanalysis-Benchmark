# P4「3D 晶体多元组 Koopman 神经区分器 + 谱蒸馏」数学契约与实验规范 (`SPEC.md`)

> **工作目录**：`g:\SHA-LM\v11_discovery\p4_koopman_distinguisher\`（独占）  
> **参考只读接口**：`g:\SHA-LM\src\sha256_ref.py`、`g:\SHA-LM\v10_ouroboros64\ouroboros_core\engine.py`  
> **核心科学问题**：不再做单样本直接预测原像，也不只测简单 2 元差分；将 Gohr (CRYPTO 2019) 神经区分器升维到 3D 时空晶体与 $2/4/8$ 元结构化元组（二元稀疏差分、模差、RX 旋转差分、4 元 Boomerang/二阶差分矩形、8 元超立方体三阶差分），探测跨过 $R$ 轮 SHA-256 压缩后，在张量积 Koopman 空间中是否存在高阶弱特征函数（$\text{AUC} > 0.50$）。

---

## 1. 精确压缩接口与 3D 时空晶体状态映射

严格复用 `src/sha256_ref.py` 的无符号 32 位算术语义（`int64 & 0xFFFFFFFF`，LSB-first 位序）：

1. **输入原像空间**：
   - 单块压缩输入消息字为 $W = (W_0, \dots, W_{15}) \in (\mathbb{Z}_{2^{32}})^{16}$（512 bits 满熵均匀分布 $W_t \stackrel{\text{i.i.d.}}{\sim} \text{Uniform}(\mathbb{Z}_{2^{32}})$）。
   - 所有训练与评估样本均**在线（on-the-fly）**使用独立随机种子生成，严禁有限样本集多轮复用以避免记忆效应。
2. **输出状态 $S_R$ 与 3D 时空晶体同构**：
   - 经 `sha256_trace(w16, rounds=R)`（`src/sha256_ref.py` L258–L327）严格演化 $R$ 轮后，提取第 $R$ 轮内部寄存器状态：
     $$S_R = [A_R, A_{R-1}, A_{R-2}, A_{R-3}, E_R, E_{R-1}, E_{R-2}, E_{R-3}] \in (\mathbb{Z}_{2^{32}})^8 \quad (256\text{ bits})$$
   - 注：标准摘要 `digest = (S_R + IV) & 0xFFFFFFFF` 与 $S_R$ 为常数平移双射；直接编码 $S_R$ 可消除固定 `IV` 引入的静态加减法进位畸变，精确保留 4 步双轨时空晶格结构：
     $$C_R[t_{\text{rel}}, r] \in \mathbb{Z}_{2^{32}}, \quad t_{\text{rel}} \in \{0,1,2,3\} \leftrightarrow \text{绝对轮次 } R-3+t_{\text{rel}}, \quad r \in \{0:\text{轨道 }A,\; 1:\text{轨道 }E\}$$
     即 $C_R[:, t_{\text{rel}}, 0] = A_{R-3+t_{\text{rel}}}$（$S_R[:, 3], S_R[:, 2], S_R[:, 1], S_R[:, 0]$），$C_R[:, t_{\text{rel}}, 1] = E_{R-3+t_{\text{rel}}}$（$S_R[:, 7], S_R[:, 6], S_R[:, 5], S_R[:, 4]$）。
   - 展开为比特位 $b \in \{0,\dots,31\}$（LSB-first）后，单样本即构成形状为 $(T=4, \text{Track}=2, \text{Bit}=32)$ 的 **3D 时空晶体**。

---

## 2. 多元组构造矩阵（2 / 4 / 8 元结构化家族）

设基准消息块 $W \stackrel{\text{i.i.d.}}{\sim} \text{Uniform}((\mathbb{Z}_{2^{32}})^{16})$。为防止扰动注入后期消息字（如 $W_{R-1}$）从而跳过前 $R-1$ 轮混合，所有局域微扰**严格施加于最早期输入字 $W_0$**（保证从 $t=0$ 穿透全部 $R$ 轮；当 $R > 16$ 时并经消息扩展 $\sigma_0, \sigma_1$ 二次重注）；全局旋转家族则作用于完整 16 字输入块。定义 5 个结构化元组家族 $\mathcal{F}$：

| 家族 ID | 元数 $k$ | 数学构造（正样本 $Y=1$） | 物理/密码学探测目标 |
| :--- | :---: | :--- | :--- |
| **`F1_xor2`** | 2 | $W^{(0)} = W,\; W^{(1)} = W \oplus \Delta_{\text{msb}}$，其中 $\Delta_{\text{msb}} = (2^{31}, 0^{15})$ | **2 元稀疏 XOR 差分**：仅翻转 $W_0$ 最高位（$t=0$ 零进位穿透），探测线性/截断差分特征函数 |
| **`F2_mod2`** | 2 | $W^{(0)} = W,\; W^{(1)}_0 = (W_0 + 1) \bmod 2^{32},\; W^{(1)}_{1..15} = W_{1..15}$ | **2 元模加差分（2-adic 微扰）**：$W_0$ 最低位 $+1 \bmod 2^{32}$，探测模算术进位链 Koopman 模式 |
| **`F3_rx2`** | 2 | $W^{(0)} = W,\; W^{(1)}_t = \text{ROTR}_{32}(W_t, 1)\;\; (\forall t \in \{0..15\})$ | **2 元 RX 旋转差分**：全块 1-bit 循环右移，探测 $K_t, \text{IV}, \text{SHR}$ 破缺后的残余旋转对称性 |
| **`F4_boom4`** | 4 | $\Delta_a = (2^{31}, 0^{15}),\; \Delta_b = (2^{15}, 0^{15})$；构造 2D 矩形四角组：<br>$W^{(0)} = W,\; W^{(1)} = W \oplus \Delta_a,\; W^{(2)} = W \oplus \Delta_b,\; W^{(3)} = W \oplus \Delta_a \oplus \Delta_b$ | **4 元 Boomerang 四角组 / 二阶差分**：探测二阶 XOR 导数 $\Delta^{(2)} S_R = \bigoplus_{m=0}^3 S_R^{(m)}$ 与四角交叉高阶弱耦合 |
| **`F5_cube8`** | 8 | $\Delta_0=(2^{31},0^{15}),\; \Delta_1=(2^{15},0^{15}),\; \Delta_2=(2^{7},0^{15})$；对 $(c_2,c_1,c_0)\in\{0,1\}^3$：<br>$W^{(4c_2+2c_1+c_0)} = W \oplus c_0\Delta_0 \oplus c_1\Delta_1 \oplus c_2\Delta_2$ | **8 元 3D 超立方体 / 三阶差分**：探测三阶布尔导数 $\Delta^{(3)} S_R = \bigoplus_{m=0}^7 S_R^{(m)}$ 与 8 元张量 Koopman 特征函数 |

### 同分布随机配对负样本 ($Y=0$) 的严格边缘对齐
对于元数为 $k \in \{2, 4, 8\}$ 的任意家族，负样本均从 $k$ 个**彼此独立**的均匀随机消息块生成：
$$\widetilde{W}^{(0)}, \dots, \widetilde{W}^{(k-1)} \stackrel{\text{i.i.d.}}{\sim} \text{Uniform}\big((\mathbb{Z}_{2^{32}})^{16}\big), \qquad X_{\text{neg}} = \big(S_R(\widetilde{W}^{(0)}), \dots, S_R(\widetilde{W}^{(k-1)})\big)$$
**数学不变性保证**：上述 5 类家族中的每一种变换（固定 XOR、常数 $+1 \bmod 2^{32}$、逐字 $\text{ROTR}_1$）均为 $(\mathbb{Z}_{2^{32}})^{16}$ 上的双射，因此正样本 $Y=1$ 的每一个单边分量 $S_R^{(m)}$ 与负样本 $Y=0$ 的对应分量 $S_R(\widetilde{W}^{(m)})$ 具有**完全一致的边缘分布**。任何单样本统计偏差均严格抵消，区分器仅能通过跨元组联合 Koopman 依赖获得 $\text{AUC} > 0.50$。

---

## 3. 纯 FP32 3D 时空晶体编码与区分器架构

### 3.1 精度硬约束
- 必须在进程入口强制执行：
  ```python
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  ```
- 全部模型参数、中间激活与输入特征张量严格锁定为 `torch.float32`（严禁 TF32 / FP16 / BF16）。

### 3.2 3D 时空晶体多通道编码 $\Psi_{\text{3D}}(X)$
将 $k$-元组输出状态转为 3D 晶格整数张量 $C_R^{(m)} \in (\mathbb{Z}_{2^{32}})^{4 \times 2}$，定义双极比特算子 $\mathcal{B}(U) = 1.0 - 2.0 \cdot \text{bits}_{\text{LSB}}(U) \in \{-1, +1\}^{4 \times 2 \times 32}$（0 映射为 $+1$，1 映射为 $-1$），按元数 $k$ 构造固定特征通道（形状 `(B, C_in, 4, 2, 32)`）：

1. **$k = 2$（`F1_xor2`, `F2_mod2`, `F3_rx2` 共用统一 6 通道编码，消除编码偏差）**：
   - 通道 0–1：原始成员自旋 $\mathcal{B}(C_R^{(0)}), \; \mathcal{B}(C_R^{(1)})$
   - 通道 2：XOR 差分自旋 $\mathcal{B}(C_R^{(0)} \oplus C_R^{(1)})$
   - 通道 3：模差自旋 $\mathcal{B}((C_R^{(1)} - C_R^{(0)}) \bmod 2^{32})$
   - 通道 4：模差借位/进位晶体平面 $\mathcal{B}\big(((C_R^{(1)} - C_R^{(0)}) \bmod 2^{32}) \oplus C_R^{(1)} \oplus C_R^{(0)}\big)$
   - 通道 5：1-bit 旋转差分自旋 $\mathcal{B}(C_R^{(1)} \oplus \text{ROTR}_{32}(C_R^{(0)}, 1))$
2. **$k = 4$（`F4_boom4` 共 8 通道编码）**：
   - 通道 0–3：4 个成员原始自旋 $\mathcal{B}(C_R^{(0..3)})$
   - 通道 4–5：两条一阶基边 XOR 差分 $\mathcal{B}(C_R^{(0)} \oplus C_R^{(1)})$、$\mathcal{B}(C_R^{(0)} \oplus C_R^{(2)})$
   - 通道 6：**二阶 XOR 导数晶面** $\mathcal{B}(\bigoplus_{m=0}^3 C_R^{(m)})$
   - 通道 7：**二阶模加交替和晶面** $\mathcal{B}((C_R^{(0)} - C_R^{(1)} - C_R^{(2)} + C_R^{(3)}) \bmod 2^{32})$
3. **$k = 8$（`F5_cube8` 共 16 通道编码）**：
   - 通道 0–7：8 个成员原始自旋 $\mathcal{B}(C_R^{(0..7)})$
   - 通道 8–10：三条一阶棱边 XOR 差分 $\mathcal{B}(C_R^{(0)} \oplus C_R^{(1)})$、$\mathcal{B}(C_R^{(0)} \oplus C_R^{(2)})$、$\mathcal{B}(C_R^{(0)} \oplus C_R^{(4)})$
   - 通道 11–13：三个二阶面 XOR 导数 $\mathcal{B}(\bigoplus_{m \in \{0,1,2,3\}} C_R^{(m)})$、$\mathcal{B}(\bigoplus_{m \in \{0,1,4,5\}} C_R^{(m)})$、$\mathcal{B}(\bigoplus_{m \in \{0,2,4,6\}} C_R^{(m)})$
   - 通道 14：**三阶超立方体 XOR 导数晶面** $\mathcal{B}(\bigoplus_{m=0}^7 C_R^{(m)})$
   - 通道 15：**三阶模加交替和晶面** $\mathcal{B}\big(\sum_{m=0}^7 (-1)^{\text{popcnt}(m)} C_R^{(m)} \bmod 2^{32}\big)$

此外，网络首层统一拼接 **4 个确定性 3D 晶体位置编码通道**：
- 相对轮次坐标 $t_{\text{norm}} = (t_{\text{rel}} - 1.5)/1.5 \in [-1, 1]$
- 双轨标记 $r_{\text{sign}} \in \{+1, -1\}$（轨道 $A \mapsto +1$，轨道 $E \mapsto -1$）
- 比特周期基频谐波 $\sin(2\pi b / 32),\; \cos(2\pi b / 32)$

### 3.3 `KoopmanCrystalNet3D` 骨干网络与训练目标
为确保真实 SHA-256 与随机 `ROTR/K` 对照组之间**零结构偏置**（不在网络中硬编码 SHA-256 特定旋转常数 `{2,6,11,13,22,25}`），网络必须以对称的 3D 局域卷积与比特轴自注意力捕获任意移位与进位耦合：
- **Stem**：`Conv3d(C_in + 4, D=32, kernel_size=1)`（或等价 `(4, 2)` 展平后 `Conv2d`）+ `GroupNorm(8, 32)` + `SiLU`。
- **4 个 3D 时空晶体 Koopman 残差块**（隐宽 $D=32$，对齐 Gohr 经典通道规模并满足纯 FP32 算力预算）：
  1. **进位-时空局域分支**：沿 `Bit` 轴循环填充（`padding=1` circular），沿 `Time` 零填充（`padding=1`）并跨 `Track=2` 混合（可使用 `(k_T=3, k_Track=2, k_Bit=3)` 或等价 `Conv2d` 轨道耦合实现）的 3D 残差卷积，捕获相邻比特进位与 4 步移位寄存器动力学。
  2. **全局比特旋转耦合分支**：将 $(B, D, 4, 2, 32)$ 视作 $8$ 条长度为 $32$ 的序列，对 `Bit=32` 轴做 4 头自注意力（`d=32, num_heads=4`，支持等价 SDPA 实现）+ Pre-LayerNorm + FFN，以完全无偏置的方式学习任意 `ROTR/SHR` 跨比特远程交互。
- **Koopman 弱特征函数读出头**：
  - 将最终 `(B, 32, 4, 2, 32)` 特征沿时空晶体做三路聚合：全局均值池化 `mean`（32 维）+ 全局最大池化 `max`（32 维）+ 比特级线性投影 `Linear(32*4*2*32 -> 32)`（保留特定晶格位置的弱特征函数权重），拼接为 96 维表征后经 `LayerNorm -> Linear(96, 64) -> SiLU -> Linear(64, 1)` 输出标量对数几率 $z_\theta(X) \in \mathbb{R}$。
- **训练目标与优化器**：
  - 在平衡在线批次（每步 $B/2$ 正样本 $+ B/2$ 同分布负样本）上最小化二元交叉熵 `BCEWithLogitsLoss`，最优解精确逼近对数似然比 $\ln \frac{P(X|Y=1)}{P(X|Y=0)}$（Neyman-Pearson 最优区分统计量）。
  - 优化器：`AdamW(lr=1e-3, weight_decay=1e-4)`，Cosine Annealing 至 `1e-5`，梯度裁剪 `max_norm=1.0`。

---

## 4. 强制同构随机对照组与 0 参数差分线性基线

### 4.1 同构随机 ROTR/K 克隆对照组 (`control`)
严格构造与真实 SHA-256 **计算图完全同构**的对照组 `sha256_trace_control(w16, rounds=R)`：
1. **轮常数随机化**：固定随机种子采样 $64$ 个 i.i.d. 均匀 32-bit 轮常数 $K_{\text{ctrl}}[t] \stackrel{\text{i.i.d.}}{\sim} \text{Uniform}(\mathbb{Z}_{2^{32}})$。
2. **位移/旋转偏移量重采样**：保持 $\Sigma_0, \Sigma_1$ 各含 3 个 `ROTR`、$\sigma_0, \sigma_1$ 各含 2 个 `ROTR` + 1 个 `SHR` 的电路算子规模不变，在固定随机种子下从 $\{1,\dots,31\}$ 无放回重采样 `ROTR` 三元组/二元组（互异且不等于 SHA-256 标准值），从 $\{1,\dots,15\}$ 重采样 `SHR` 偏移量。
3. **不变部分**：保留标准 `IV`、模 $2^{32}$ 加法拓扑、`Ch`、`Maj`、8 字寄存器移位结构及完全一致的 `KoopmanCrystalNet3D` 超参、步数与评估协议。

### 4.2 0 参数差分线性基线 (`zero_param_dl`)
在真实 SHA-256 的同一独立留出集上运行两类轻量化经典对照，以剥离浅层低阶线性偏差与真正的高阶 Koopman 模式：
1. **严格 0 参数汉明重量偏差基线 (`zero_param_hw`)**：不经任何拟合或参数学习，直接计算高阶差分晶体的负汉明重量（对称性得分）：
   - `F1_xor2`：$s_0(X) = -\text{popcnt}(S_R^{(0)} \oplus S_R^{(1)})$
   - `F2_mod2`：$s_0(X) = -\text{popcnt}((S_R^{(1)} - S_R^{(0)}) \bmod 2^{32})$
   - `F3_rx2`：$s_0(X) = -\text{popcnt}(S_R^{(1)} \oplus \text{ROTR}_{32}(S_R^{(0)}, 1))$
   - `F4_boom4`：$s_0(X) = -\text{popcnt}(\bigoplus_{m=0}^3 S_R^{(m)})$
   - `F5_cube8`：$s_0(X) = -\text{popcnt}(\bigoplus_{m=0}^7 S_R^{(m)})$
   直接在留出集上评估 `AUC` 与最优阈值准确率。
2. **单比特差分-线性最大偏置探针 (`dl_bit_probe`)**：
   - 在独立校准集（$N_{\text{cal}} = 16384$ 正 $+ 16384$ 负）上统计对应差分晶面（$k=2$ 取对应 XOR/模/RX 差；$k=4$ 取 $\Delta^{(2)} S_R$；$k=8$ 取 $\Delta^{(3)} S_R$）的 256 个比特双极均值差 $\hat{\mu}_j = \bar{b}_j^{(+)} - \bar{b}_j^{(-)}$，选取最大单比特 $j^* = \arg\max_j |\hat{\mu}_j|$ 及线性匹配滤波向量 $\hat{\mu}$，在**独立留出集**上分别报告单比特 `AUC` 与线性向量 `AUC`。

---

## 5. 扫频阶梯、资源预算与自适应延伸协议

- **扫频轮数阶梯**（共 6 个轮数点）：
  $$R \in \{4,\; 6,\; 8,\; 10,\; 12,\; 16\}$$
  其中 $R=4$ 因 $S_4$ 直接包含 $A_1, E_1$（距 $W_0$ 仅 1 轮），构成管道有效性的**强制阳性校准（Positive Calibration）**；$R=6$ 为过渡区；$R \ge 8$ 为深轮 Koopman 弱特征函数核心探测区。
- **Colab 单会话预算分配（L4/T4 $\le 45$ 分钟总预算）**：
  - 全部 5 个家族统一使用纯在线平衡批次 `batch_size = 256`（128 正 + 128 负）与隐宽 `d = 32, n_blocks = 4`（按 L4 纯 FP32 实测 `33.62 ms/step` 预算）：
    - 阳性校准轮 `R = 4`：`steps = 400`（每 run 102,400 个全新元组）。
    - 核心探测轮 `R ∈ {6, 8, 10, 12, 16}`：`steps = 1000`（每 run 256,000 个全新元组）。
    - 全 60 组（6 轮 $\times$ 5 家族 $\times$ 主/对照）纯训练总步数 $54{,}000$ 步 $\approx 30.3$ 分钟，加留出评估与基线合计约 $32\text{–}34$ 分钟，留足安全余量。
  - **自适应高信号加训门（Auto-Extend）**：若任意 $R \ge 6$ 的 `(R, family)` 在初步评估中达到 $z_{\text{null}} \ge 4.0$，自动追加 `2000` 步精调（`lr = 2e-4 -> 1e-5`）后再执行最终留出集验算与谱蒸馏，确保弱特征函数充分收敛。

---

## 6. 独立留出集评估指标与双向亮灯门限

### 6.1 留出集分池评估协议
对每个 `(R, family)`，使用与训练、校准完全隔离的测试随机种子生成 $N_{\text{eval}} = 65536$ 个测试元组（$N_+ = 32768$ 正样本 $+ N_- = 32768$ 同分布随机配对负样本），同时划分为 $M = 16$ 个互斥评估池（每池 $4096$ 元组 = $2048$ 正 $+ 2048$ 负）：
- **ROC-AUC**：在全部 $N_{\text{eval}} = 65536$ 个样本的连续对数几率 $z(X)$ 上计算精确非参数 ROC-AUC，并计算 16 个分池 AUC 的均值与标准误 $\text{SE}_{\text{pool}}(\text{AUC}) = s_{\text{AUC}} / \sqrt{16}$。
- **分类准确率 (`Acc`)**：以 `logit > 0`（等价于预测概率 $p > 0.5$）在全部 $65536$ 个样本上计算分类准确率，零假定基准为 $0.50$。

### 6.2 双向统计检验 $z$-score 定义
1. **超越零假设检验 $z_{\text{null}}(R, \mathcal{F})$**（检验 $H_0: \text{AUC} = 0.50$）：
   - 连续得分在零假设下的精确 Mann-Whitney U 渐近标准误：
     $$\sigma_{\text{AUC}, 0} = \sqrt{\frac{N_+ + N_- + 1}{12\,N_+\,N_-}} \approx 0.0022552 \quad (N_+ = N_- = 32768)$$
   - 定义方向性检验统计量与双侧辅助检验：
     $$z_{\text{null}}(R, \mathcal{F}) = \frac{\text{AUC}_{\text{main}}(R, \mathcal{F}) - 0.5}{\sigma_{\text{AUC}, 0}}, \qquad z_{\text{acc, null}}(R, \mathcal{F}) = \frac{\text{Acc}_{\text{main}}(R, \mathcal{F}) - 0.5}{\sqrt{0.25 / N_{\text{eval}}}}$$
   - 注：$z_{\text{null}} \ge 6.0$ 要求 $\text{AUC}_{\text{main}} \ge 0.51353$（$p < 10^{-9}$）；若某异常情形出现 $z_{\text{null}} \le -6.0$（反相关异常），须在审计中额外标出并触发复核。
2. **超越随机对照组检验 $z_{\text{ctrl}}(R, \mathcal{F})$**（检验 $H_0: \text{AUC}_{\text{main}} \le \text{AUC}_{\text{ctrl}}$）：
   $$z_{\text{ctrl}}(R, \mathcal{F}) = \frac{\text{AUC}_{\text{main}}(R, \mathcal{F}) - \text{AUC}_{\text{ctrl}}(R, \mathcal{F})}{\sqrt{\text{SE}_{\text{pool}}(\text{AUC}_{\text{main}})^2 + \text{SE}_{\text{pool}}(\text{AUC}_{\text{ctrl}})^2 + 10^{-12}}}$$

### 6.3 亮灯裁决门限（Verdict Criteria）
- **`HIT`（发现超越同构随机控制的 SHA-256 高阶 Koopman 弱特征函数）**：
  在任意深轮 **$R \ge 8$** 与任意元组家族 $\mathcal{F}$ 上**同时满足**：
  1. $\text{AUC}_{\text{main}}(R, \mathcal{F}) > 0.50$ 且 **$z_{\text{null}}(R, \mathcal{F}) \ge 6.0$**；
  2. **$z_{\text{ctrl}}(R, \mathcal{F}) \ge 4.0$**。
- **`BORDERLINE`（边缘弱信号，待深化验证）**：
  满足以下任一条件（且未达 `HIT`）：
  1. 在过渡轮 **$R = 6$** 上同时满足 $z_{\text{null}}(6, \mathcal{F}) \ge 6.0$ 且 $z_{\text{ctrl}}(6, \mathcal{F}) \ge 4.0$；
  2. 在 **$R \ge 8$** 上出现 $3.0 \le z_{\text{null}} < 6.0$ 且 $z_{\text{ctrl}} \ge 2.0$，或 $z_{\text{null}} \ge 6.0$ 但 $2.0 \le z_{\text{ctrl}} < 4.0$。
- **`NULL`（未观测到超越随机构造的 Koopman 弱特征函数）**：
  全部 $R \ge 8$（且 $R=6$ 亦未达 `BORDERLINE`）均未触发上述门限：在 $R=4$ 阳性校准成功（`AUC ≈ 1.0`）的前提下，跨过 $R \ge 6$ 或 $R \ge 8$ 后真实 SHA-256 与随机 ROTR/K 克隆对照组一致退化至 $\text{AUC} \approx 0.50$。

---

## 7. 亮灯后的 Walsh / 2-adic 谱蒸馏方案

对以下情形自动执行全量谱蒸馏（**强制包括 $R=4$ 下的 `F1_xor2` 与 `F3_rx2` 作为阳性蒸馏校准**，以及任意满足 $z_{\text{null}}(R, \mathcal{F}) \ge 4.0$ 或 $|z_{\text{ctrl}}(R, \mathcal{F})| \ge 3.0$ 的 $(R, \mathcal{F})$ 组合）：

1. **Walsh-Hadamard 布尔谱分解**：
   - 取留出集上神经区分器中心化输出 $\tilde{f}(X) = z_\theta(X) - \bar{z}_\theta$。
   - 在对应元组最高阶布尔差分晶面 $\Delta^{(r)} S_R \in \{-1, +1\}^{256}$ 上：
     - **1 阶 Walsh 谱**：遍历全部分量 $j \in \{0,\dots,255\}$，计算相关系数 $w_j = \text{corr}(\tilde{f}(X),\, \Delta^{(r)} s_{R,j})$。
     - **2 阶交互 Walsh 谱**：选取 $|w_j|$ 最大的前 24 个比特，遍历全部 $\binom{24}{2} = 276$ 个双比特乘积项 $\Delta^{(r)} s_{R,i} \cdot \Delta^{(r)} s_{R,j}$。
   - 用 Top-16 最强 Walsh 基函数拟合线性代理模型 $\hat{f}_{\text{Walsh}}(X)$，在留出集上直接测量解释方差 $R^2_{\text{Walsh}}$ 与**纯解析 Walsh 特征函数还原 AUC**。
2. **2-adic 进位相位谱分解**：
   - 针对 8 个输出字 $u_w \in \mathbb{Z}_{2^{32}}$ 的结构化模差/旋转残差 $d_w$，在四个 2-adic 截断尺度 $p \in \{4, 8, 16, 32\}$ 与谐波阶 $\kappa \in \{1, 2, 4\}$ 上构造紧致相位特征：
     $$\phi_{w,p,\kappa}(X) = \left[\cos\frac{2\pi\kappa (d_w \bmod 2^p)}{2^p},\; \sin\frac{2\pi\kappa (d_w \bmod 2^p)}{2^p}\right] \quad (\text{共 } 8 \times 4 \times 3 \times 2 = 192\text{ 维})$$
   - 拟合 $\tilde{f}(X)$ 对 2-adic 基的 $L_1$/Ridge 稀疏回归，提取 Top-12 2-adic 谐波项（精确定位寄存器 `A/E`、轮次偏移 $R-k$、尺度 $2^p$ 与谐波 $\kappa$），并在留出集上评估纯 2-adic 解析代理的 $R^2_{\text{2-adic}}$ 与**还原 AUC**。
3. **3D 时空晶体梯度归因**：
   - 计算绝对输入梯度均值 $\mathbb{E}_{X \sim \text{pos}}[|\nabla_{\Psi} z_\theta(X)|]$，按相对轮次 $t_{\text{rel}} \in \{R-3..R\}$、轨道 $\{A, E\}$ 与特征通道类型汇总能量分布，定位信号物理承载区域。

---

## 8. 工程交付 JSON 契约（`results/p4_sweep_summary.json`）

`@p4-eng` 同步回 `g:\SHA-LM\v11_discovery\p4_koopman_distinguisher\results\` 的 `p4_sweep_summary.json` 须包含：
- `meta`：`gpu_name`, `fp32_locked` (`true`), `tf32_disabled` (`true`), `rounds` (`[4,6,8,10,12,16]`), `families` (`["F1_xor2","F2_mod2","F3_rx2","F4_boom4","F5_cube8"]`), `eval_n` (`65536`), `eval_pool_k` (`4096`), `eval_num_pools` (`16`), `control_config` (`rotr_offsets`, `k_seed`), `total_wall_sec`
- `results`：嵌套字典 `results[str(R)][family]`，每项包含：
  - `main`: `auc`, `auc_pool_mean`, `auc_pool_se`, `acc`, `z_null`, `z_acc_null`, `final_train_loss`, `extended`, `wall_sec`
  - `control`: `auc`, `auc_pool_mean`, `auc_pool_se`, `acc`, `z_null`, `final_train_loss`, `wall_sec`
  - `zero_param_baseline`: `zero_param_hw_auc`, `zero_param_hw_acc`, `dl_single_bit_auc`, `dl_linear_auc`, `best_bit_idx`
  - `comparison`: `delta_auc` (`main.auc - control.auc`), `z_ctrl`, `delta_vs_zero_param`
  - `distillation`: 若触发则含 `walsh_top16`, `walsh_r2`, `walsh_surrogate_auc`, `two_adic_top12`, `two_adic_r2`, `two_adic_surrogate_auc`, `crystal_saliency_by_round`, `crystal_saliency_by_channel`；未触发则为 `null`
