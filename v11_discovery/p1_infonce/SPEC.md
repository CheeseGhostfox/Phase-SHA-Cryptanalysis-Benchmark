# P1「跨轮 InfoNCE 变分互信息望远镜」数学契约与实验规范 (`SPEC.md`)

> **工作目录**：`g:\SHA-LM\v11_discovery\p1_infonce\`（独占）  
> **参考只读接口**：`g:\SHA-LM\src\sha256_ref.py`、`g:\SHA-LM\v10_ouroboros64\ouroboros_core\engine.py`  
> **核心科学问题**：不预测原像单比特，用 InfoNCE 对比学习最大化互信息变分下界 $I(f_\theta(S_0); g_\phi(S_R))$，探测跨越 $R$ 轮 SHA-256 压缩后是否仍存在任意高阶非线性慢变量或宏观序参量。

---

## 1. 精确状态定义与位表示接口

严格对齐 `src/sha256_ref.py` 的无符号 32 位算术语义（`int64 & 0xFFFFFFFF`，LSB-first 位序）：

1. **初始输入状态 $S_0$（即单块 512-bit 原像输入字 $W_{0..15}$）**：
   - 在标准 SHA-256 单块压缩中，寄存器初值固定为标准 `SHA256_IV`（8 个 32-bit 字）。为使输入侧具有满熵而非退化常数，定义输入随机变量为 16 个 32-bit 消息字：
     $$S_0 \equiv W_{0..15} \stackrel{\text{i.i.d.}}{\sim} \text{Uniform}\big((\mathbb{Z}_{2^{32}})^{16}\big), \quad \text{总熵 } H(S_0) = 512\text{ bits}$$
   - 注：每步训练与评估均**在线（on-the-fly）**使用独立随机种子直接生成全新样本，严禁有限数据集多轮复用来避免过拟合记忆。
2. **第 $R$ 轮内部状态 $S_R$**：
   - 经 `sha256_trace(w16, rounds=R)`（`src/sha256_ref.py` L258–L327）严格演化 $R$ 轮，取末态 8 个 32-bit 寄存器：
     $$S_R = [A_R, A_{R-1}, A_{R-2}, A_{R-3}, E_R, E_{R-1}, E_{R-2}, E_{R-3}] \in (\mathbb{Z}_{2^{32}})^8 \quad (256\text{ bits})$$
   - 注：标准 `digest = (S_R + IV) & 0xFFFFFFFF` 与 $S_R$ 为常数平移双射，互信息完全等价；本实验统一直接编码 `S_R`。
   - **重要物理现象（浅轮正校准）**：当 $R=4$ 时，$S_4$ 中包含 $A_1, E_1$（直接等于 $W_0 + c \bmod 2^{32}$）。因此在 $R=4$ 上，真实 SHA-256 与同构随机对照组**均应**呈现极高互信息，构成管道有效性的“阳性探针校准（Positive Calibration）”；真正用于探测 SHA-256 专属非随机慢变量的区间为 **$R \ge 8$**（以及 $R=6$ 的过渡区）。

---

## 2. 多尺度 / 2-adic / Bit 级编码特征

对任意含 $M$ 个 32-bit 字的整数张量 $U \in (\mathbb{Z}_{2^{32}})^M$（输入 $S_0$ 取 $M=16$，输出 $S_R$ 取 $M=8$），构造三尺度确定性 FP32 特征拼接向量 $\Phi(U)$：

1. **Bit 级双极自旋特征 $\Phi_{\text{bit}}(U) \in \{-1, +1\}^{32M}$**：
   - 复用 `words_to_spins(U, dtype=torch.float32)`（LSB-first：$0 \mapsto +1.0,\; 1 \mapsto -1.0$）。
   - 输入维数 $512$，输出维数 $256$。
2. **2-adic 多尺度进位相位特征 $\Phi_{\text{2-adic}}(U) \in [-1, 1]^{8M}$**：
   - 针对每个字 $u \in \mathbb{Z}_{2^{32}}$，在四个 2-adic 截断尺度 $p \in \{4, 8, 16, 32\}$ 上提取模 $2^p$ 单位圆相位：
     $$\theta_p(u) = \frac{2\pi \cdot (u \bmod 2^p)}{2^p}, \qquad \phi_p(u) = \big[\cos\theta_p(u),\; \sin\theta_p(u)\big]$$
   - 该表示将模 $2^{32}$ 加法的长程进位连续嵌入紧致流形 $[-1,1]$，无尺度爆炸风险。
   - 输入维数 $16 \times 8 = 128$，输出维数 $8 \times 8 = 64$。
3. **Nibble 级（4-bit 局部块）奇偶与密度特征 $\Phi_{\text{nib}}(U) \in [-1, 1]^{16M}$**：
   - 每个 32-bit 字切分为 8 个互不重叠的 4-bit nibble；对每个 nibble 提取：
     - 4-bit 局部 XOR 奇偶（映射到 $\{-1, +1\}$）
     - 中心化汉明重量 $\frac{\text{popcnt}_4 - 2}{2} \in [-1, 1]$
   - 输入维数 $16 \times 16 = 256$，输出维数 $8 \times 16 = 128$。
4. **总拼接输入维度**：
   - $\Phi_{\text{in}}(S_0) \in \mathbb{R}^{896}$（$= 512 + 128 + 256$）
   - $\Phi_{\text{out}}(S_R) \in \mathbb{R}^{448}$（$= 256 + 64 + 128$）
   - 代码需保留特征下标元信息（`feature_names`），供亮灯后稀疏谱蒸馏精确回溯至具体字、比特位与 2-adic 尺度。

---

## 3. 双塔编码器与 InfoNCE 变分互信息下界目标

### 3.1 网络架构与 FP32 精度锁定
- **精度锁定**：必须设置 `torch.backends.cuda.matmul.allow_tf32 = False` 与 `torch.backends.cudnn.allow_tf32 = False`，全程纯 `torch.float32`。
- **编码器 $f_\theta: \mathbb{R}^{896} \to \mathbb{S}^{D-1}$ 与 $g_\phi: \mathbb{R}^{448} \to \mathbb{S}^{D-1}$**（嵌入维 $D = 128$）：
  - 输入线性投影至隐宽 $H = 512$，接 4 个 Pre-LayerNorm 残差块：`x = x + Linear(SiLU(Linear(LN(x))))`，再经 `LN -> Linear(512, 128)` 与 $L_2$ 单位球面归一化：
    $$u_i = \frac{f_\theta(\Phi_{\text{in}}(S_0^{(i)}))}{\|f_\theta(\Phi_{\text{in}}(S_0^{(i)}))\|_2}, \qquad v_j = \frac{g_\phi(\Phi_{\text{out}}(S_R^{(j)}))}{\|g_\phi(\Phi_{\text{out}}(S_R^{(j)}))\|_2}$$
  - 真实 SHA-256（`main`）与随机对照组（`control`）**必须使用完全一致的网络超参、参数量、优化器（AdamW, lr=$3\times 10^{-4}$, cosine decay）与训练步数**。

### 3.2 对称 InfoNCE 损失与互信息估计
给定批大小 $B = 1024$（1 正样本 $+ (B-1)$ 个同批独立负样本），定义可学习温度 $\tau = \exp(-\alpha)$，其中 $\alpha = \log(1/\tau)$ 初始化为 $\log(1/0.07) \approx 2.659$ 并限幅于 $[\log(1.0), \log(50.0)]$（即 $\tau \in [0.02, 1.0]$）：
$$s_{ij} = \frac{\langle u_i, v_j \rangle}{\tau}$$
对称 InfoNCE 损失：
$$\mathcal{L}_{\text{InfoNCE}}(\theta, \phi) = \frac{1}{2B}\sum_{i=1}^{B}\left[-\log\frac{e^{s_{ii}}}{\sum_{j=1}^{B} e^{s_{ij}}} - \log\frac{e^{s_{ii}}}{\sum_{j=1}^{B} e^{s_{ji}}}\right]$$
在大小 $K$ 的独立候选池上，InfoNCE 给出真实互信息的有效变分下界（Oord et al., 2018；Poole et al., 2019）：
$$I(f_\theta(S_0); g_\phi(S_R)) \ge \widehat{I}_{\text{bits}}^{(K)} = \log_2 K - \frac{\mathcal{L}_{\text{InfoNCE}}^{(K)}}{\ln 2} \quad (\text{bits})$$
零互信息（随机猜测）基线为 $\widehat{I}_{\text{bits}}^{(K)} = 0\text{ bits}$。

---

## 4. 强制随机对照组与线性 Baseline

### 4.1 同规模随机克隆对照组（Random-ROTR + Random-K Clone SHA-256）
构造与真实 SHA-256 **计算图同构**的对照组 `sha256_trace_control(w16, rounds=R)`：
1. **轮常数随机化**：生成固定随机种子下独立均匀采样的 64 个 32-bit 轮常数 $K_{\text{ctrl}}[t] \stackrel{\text{i.i.d.}}{\sim} \text{Uniform}(\mathbb{Z}_{2^{32}})$（替代原素数立方根小数常数）。
2. **位移/旋转偏移量随机置换重采样**：保持 $\Sigma_0, \Sigma_1$ 各含 3 个 `ROTR`、$\sigma_0, \sigma_1$ 各含 2 个 `ROTR` + 1 个 `SHR` 的电路规模不变，在固定随机种子下重采样偏移量（确保每个函数内部旋转量互异且不等于标准 SHA-256 偏移量，例如：`ROTR` 采自 $\{1,\dots,31\}$ 无放回三元组/二元组，`SHR` 采自 $\{1,\dots,15\}$ 且互异）。
3. **保留部分**：保持 `IV`、模 $2^{32}$ 加法拓扑、`Ch`、`Maj` 及 8 字寄存器移位结构完全一致。
4. **同步实验要求**：每个 $R$ 上，对照组与主实验在完全相同的 `batch_size`、`steps`、学习率与评估协议下同步运行。

### 4.2 双重线性 Baseline（在真实 SHA-256 上）
用于区分“纯粹高阶非线性慢变量”与“简单线性/仿射相关性”：
1. **Linear InfoNCE**：将 $f_\theta, g_\phi$ 替换为无隐层、无激活函数的单层线性投影 $u = \text{Norm}(W_f \Phi_{\text{in}}),\; v = \text{Norm}(W_g \Phi_{\text{out}})$，以相同 InfoNCE 目标与步数训练，在同一个留出集上评估 Top-1、Top-5 与 $\widehat{I}_{\text{bits}}$。
2. **解析 Ridge-CCA 校核**：在独立拟合集（$N_{\text{fit}} = 32768$）上计算 $(\Phi_{\text{in}}, \Phi_{\text{out}})$ 的中心化协方差与互协方差（正则化系数 $\lambda = 10^{-3}$），求解广义特征值得到前 $D=128$ 阶典型相关系数 $\{r_k\}_{k=1}^{128}$，并在留出集上计算高斯线性互信息下界 $\widehat{I}_{\text{CCA}} = -\frac{1}{2}\sum_{k=1}^{128}\log_2(1 - r_{k,\text{eval}}^2)$ 与线性 CCA 余弦检索 Top-1/Top-5。

---

## 5. 扫频阶梯与资源控制

- **扫频轮数阶梯**（全覆盖，共 7 个点）：
  $$R \in \{4,\; 6,\; 8,\; 10,\; 12,\; 16,\; 20\}$$
- **单卡预算分配建议（Colab L4, $\le 45$ 分钟总预算）**：
  - 建议每点训练步数 `steps = 4000`（`B = 1024`，即每 run 在线训练 $4{,}096{,}000$ 个全新样本对，GPU 向量化 `sha256_trace` 几乎零 I/O 开销）；若 L4 实测单 run < 1.5 分钟，可保持 `steps = 4000~5000`。
  - 7 轮 $\times$ (`main` + `control` + `linear_infonce` + `ridge_cca` + `eval` + `distill`) 总时长预计 20–35 分钟，须留出结果打包回传与关机余量。

---

## 6. 独立留出集评估指标与 $z\text{-score} \ge 6$ 亮灯门限

### 6.1 留出集分池评估协议
对每个 $R$，使用**与训练、对照完全隔离的固定测试随机种子**生成 $N_{\text{eval}} = 16384$ 个样本对，切分为 $M = 16$ 个互斥评估池，每池 $K = 1024$ 对：
- 在每个池 $m \in \{1,\dots,16\}$ 内计算双向（$S_0 \to S_R$ 与 $S_R \to S_0$ 取平均）指标：
  - **Top-1 检索率**（随机期望 $1/1024 \approx 9.766\times 10^{-4}$）
  - **Top-5 检索率**（随机期望 $5/1024 \approx 4.883\times 10^{-3}$）
  - **InfoNCE 互信息下界 $\widehat{I}_{\text{bits}}^{(m)}$**（随机期望 $0\text{ bits}$）
- 跨 $M=16$ 池报告均值与标准误（$\text{SE} = s / \sqrt{16}$）。

### 6.2 双重 $z$-score 定义
1. **零配对置换检验 $z_{\text{null}}(R)$**：
   - 在已训练好的 `main` 编码器上，对留出集各池输出嵌入作 $P = 64$ 次独立随机置换（打乱 $(u_i, v_j)$ 配对），测得零互信息均值 $\mu_{\text{perm}}$ 与跨置换标准差 $\sigma_{\text{perm}}$（单池/16池均值标准误 $\text{SE}_{\text{perm}} = \sigma_{\text{perm}} / \sqrt{P}$），则：
     $$z_{\text{null}}(R) = \frac{\bar{I}_{\text{main}}(R) - \mu_{\text{perm}}}{\sqrt{\text{SE}(\bar{I}_{\text{main}}(R))^2 + \text{SE}_{\text{perm}}^2 + 10^{-12}}}$$
   - 同时计算 Top-1 二项分布精确检验 $z_{\text{Top1, null}}(R)$（$p_0 = 1/1024,\; N_{\text{tot}} = 2 \times 16384 = 32768$）作为交叉验证。
2. **超越随机对照组检验 $z_{\text{ctrl}}(R)$**：
   $$z_{\text{ctrl}}(R) = \frac{\bar{I}_{\text{main}}(R) - \bar{I}_{\text{ctrl}}(R)}{\sqrt{\text{SE}(\bar{I}_{\text{main}}(R))^2 + \text{SE}(\bar{I}_{\text{ctrl}}(R))^2 + 10^{-12}}}$$

### 6.3 亮灯裁决门限（Verdict Criteria）
- **`HIT`（发现可观测慢变量/宏观序参量）**：在任意轮数 **$R \ge 8$** 上**同时满足**：
  1. $z_{\text{null}}(R) \ge 6.0$ 且 $\bar{I}_{\text{main}}(R) \ge 0.05\text{ bits}$（或 Top-1 满足 $z_{\text{Top1, null}} \ge 6.0$）；
  2. $z_{\text{ctrl}}(R) \ge 6.0$（真实 SHA-256 显著优于同规模随机 ROTR/K 对照组）。
- **`BORDERLINE`（边缘线索，待进一步验证）**：在 $R \ge 8$ 上出现 $3.0 \le \min(z_{\text{null}}(R), z_{\text{ctrl}}(R)) < 6.0$；或仅在 $R=6$ 上出现 $z_{\text{ctrl}} \ge 6.0$ 但未延拓至 $R \ge 8$。
- **`NULL`（未观测到超越随机构造的宏观序参量）**：全部 $R \ge 8$ 轮数均满足 $\min(z_{\text{null}}, z_{\text{ctrl}}) < 3.0$（或真实 SHA-256 与随机对照组在全部轮数上统计无异，浅层 $R \in \{4,6\}$ 泄漏随同对照组一起指数/快速熄灭）。

---

## 7. 亮灯后的 CCA / 稀疏谱蒸馏方案

对以下情形自动执行蒸馏分析（**强制包括 $R=4$ 作为阳性蒸馏校准**，以及任何满足 $z_{\text{null}}(R) \ge 6.0$ 或 $|z_{\text{ctrl}}(R)| \ge 3.0$ 的轮数）：

1. **嵌入空间 CCA 谱分解**：
   - 在留出集嵌入矩阵 $U \in \mathbb{R}^{N_{\text{eval}} \times D},\; V \in \mathbb{R}^{N_{\text{eval}} \times D}$ 上求解中心化 CCA，得到典型相关系数谱 $r_1 \ge r_2 \ge \dots \ge r_D$ 与典型对 $(z_k^{(X)} = U\alpha_k,\; z_k^{(S)} = V\beta_k)$。
   - 计算**有效秩** $r_{\text{eff}} = \exp\big(-\sum_k p_k \ln p_k\big)$（其中 $p_k = r_k^2 / \sum_j r_j^2$）与显著模态数（$r_k > 0.10$），判定互信息由低维序参量主导还是高维弥散。
2. **稀疏特征归因（Lasso / Top-K Pearson）**：
   - 对前 3 个最高典型模态 $(z_k^{(X)}, z_k^{(S)})$，分别在 $\Phi_{\text{in}}(S_0)$（896 维）与 $\Phi_{\text{out}}(S_R)$（448 维）上拟合 $L_1$ 正则稀疏回归（或标准化相关性排序），提取 Top-16 贡献特征名（精确定位到字索引 $W_i / S_{R,j}$、比特位 $b_m$ 或 2-adic 尺度 $p$）及累积解释方差 $R^2$。
3. **高阶 Walsh-Hadamard 2-阶交互探针**：
   - 在输入/输出各取相关度最高的前 16 个单比特自旋上，遍历所有双比特 XOR 组合 $b_i \oplus b_j$（共 $\binom{16}{2}=120$ 项），检验典型变量是否由特定 2-阶布尔单调/奇偶项主导，并将结果写入蒸馏 JSON。

---

## 8. 工程交付 JSON 契约（`results/p1_sweep_summary.json`）

`@p1-eng` 同步回 `g:\SHA-LM\v11_discovery\p1_infonce\results\` 的汇总文件 `p1_sweep_summary.json` 须包含：
- `meta`：`gpu_name`, `fp32_locked` (`true`), `tf32_disabled` (`true`), `batch_size`, `train_steps`, `eval_n`, `eval_pool_k`, `eval_num_pools`, `feature_dims` (`{"in": 896, "out": 448}`), `control_config` (`rotr_offsets`, `k_seed`), `total_wall_sec`
- `rounds`：以字符串 `"4"`, `"6"`, `"8"`, `"10"`, `"12"`, `"16"`, `"20"` 为键，每项包含：
  - `main`: `mi_bits_mean`, `mi_bits_se`, `top1_mean`, `top1_se`, `top5_mean`, `top5_se`, `perm_mi_mean`, `perm_mi_std`, `z_null`, `z_top1_null`, `final_train_loss`, `learned_temp`, `wall_sec`
  - `control`: 同上字段（含 `z_null`）
  - `linear_baseline`: `linear_infonce_mi_bits_mean`, `linear_infonce_mi_bits_se`, `linear_infonce_top1_mean`, `linear_infonce_top5_mean`, `ridge_cca_mi_bits_eval`, `ridge_cca_top1_mean`, `ridge_cca_top5_mean`, `ridge_cca_top10_corrs`
  - `comparison`: `delta_mi_bits` (`main - control`), `z_ctrl`, `delta_vs_linear_bits` (`main - linear_infonce`)
  - `distillation`: 若触发则包含 `cca_top10_corrs`, `effective_rank`, `num_sig_modes`, `modes`（含 `top_in_features`, `top_out_features`, `r2_in`, `r2_out`, `top_walsh_pairs`）；未触发则为 `null`
