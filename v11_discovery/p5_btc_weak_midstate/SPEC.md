# P5「BTC 320-bit 固定填充 × 弱 Midstate 共振探测」数学设计契约（SPEC）

> **工作目录**：`g:\SHA-LM\v11_discovery\p5_btc_weak_midstate\`（独占，严禁修改 `v10_ouroboros64/`、`runs/`、`formal_proofs/`、`formal_proofs_v10/` 及其他 `pX` 目录）  
> **参考只读接口**：`g:\SHA-LM\src\sha256_ref.py`、`g:\SHA-LM\v10_ouroboros64\ouroboros_core\engine.py`、`g:\SHA-LM\v10_ouroboros64\ouroboros_bench\open_boundary_alpha_probe.py`  
> **硬件与数值铁律**：本地仅允许 CPU（`CUDA_VISIBLE_DEVICES=""`）秒级冒烟测试；全量搜索与留出核验必须在 Colab GPU（`v11-p5`，L4/T4，总预算 $\le 45$ 分钟）上以纯 IEEE-754 FP32 执行（`allow_tf32=False`, `torch.float32`），前向哈希与位统计必须使用精确 `int64 & 0xFFFFFFFF` 整数算术。

---

## 1. 核心科学问题与 BTC Block 2 / Double-SHA256 精确定义

### 1.1 BTC Block 2 结构与可控自由度

比特币 80 字节区块头按 SHA-256 64 字节块切分为 Block 1（64 字节）与 Block 2（16 字节 + 48 字节标准 SHA-256 填充）。矿工通过修改 Coinbase 交易 ExtraNonce 改变 Merkle Root，可以调控 Block 1 输出的 256-bit 中间哈希状态（**Midstate** $S_0$），并直接控制 Block 2 前 12 字节（Merkle Root 末 4 字节、Timestamp、Bits），留 $W[3]$ 作为 32-bit Nonce。

严格对齐 `src/sha256_ref.py` 与 `open_boundary_alpha_probe.py`（L64–L77）的无符号 32-bit 大端字序约定（`uint32` 存于 `int64 & 0xFFFFFFFF`，字内比特按 LSB-first 索引 $b \in \{0,\dots,31\}$）：

1. **Midstate 参数化**：
   $$S_0 = [a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0] \in (\mathbb{Z}_{2^{32}})^8 \quad (256\text{ bits}).$$
   - 注：探测“理想可控自由 $S_0$”可覆盖一切物理 Block 1 可达 Midstate 的超集（上界探测）；同时搜索与基线中均纳入由随机 Block 1 经完整 64 轮压缩得到的物理合法 Midstate 样本。若在自由 $S_0$ 超集上均无显著共振，则物理合法子集必然也不存在弱 Midstate。
2. **Block 2 固定 320-bit 尾部填充与模板拆分**：
   $$W^{(2)} = \bigl[\underbrace{W[0], W[1], W[2]}_{\text{模板自由字 } (96\text{ bits})},\; \underbrace{W[3]}_{\text{独立 Nonce } (32\text{ bits})},\; \underbrace{\text{0x80000000},\; 0,\; \dots,\; 0,\; 640}_{\text{固定 320-bit 尾部填充 } W[4..15]}\bigr] \in (\mathbb{Z}_{2^{32}})^{16}.$$
   其中 $W[4] = \text{0x80000000}$，$W[5..14] = 0$，$W[15] = 640$（即 $80 \times 8 = \text{0x00000280}$）。
   - 完整**模板向量**定义为 11 个 32-bit 字（共 $352$ bits）：
     $$\Theta = \bigl(S_0[0..7],\; W[0],\; W[1],\; W[2]\bigr) \in (\mathbb{Z}_{2^{32}})^{11}.$$
   - **独立留出变量**为单一 32-bit Nonce：$x = W[3] \in \mathbb{Z}_{2^{32}}$。

### 1.2 单层 $R$ 轮压缩 $F_R(\Theta, x)$ 与进位/消息扩展定义

给定模板 $\Theta$ 与 Nonce $x = W[3]$，首先执行 SHA-256 消息扩展（$t = 16 \dots R-1$）：
$$W[t] = \bigl(\sigma_1(W[t-2]) + W[t-7] + \sigma_0(W[t-15]) + W[t-16]\bigr) \bmod 2^{32}.$$
**解析结构洞察（Block 2 早期稀疏窗口）**：由于 $W[5..14]=0$，在 $t=16, 17$ 步有：
$$W[16] = \bigl(\sigma_0(W[1]) + W[0]\bigr) \bmod 2^{32}, \qquad W[17] = \bigl(\sigma_1(640) + \sigma_0(W[2]) + W[1]\bigr) \bmod 2^{32},$$
均**完全不依赖 Nonce $W[3]$**；$W[3]$ 首次进入消息扩展发生在 $t=18$（通过 $\sigma_0(W[3])$），并通过压缩轮 $t=3$ 首次进入状态寄存器。这意味着模板 $\Theta$ 可解析地令 $W[16]=W[17]=0$ 并将前 3 轮压缩进位压至极低。

随后从初态 $S_0$（**注意：Block 2 是第二块压缩，前馈累加使用 $S_0$ 而非标准 `IV`**）出发运行 $R$ 轮标准 SHA-256 压缩（轮常数 $K[0..R-1]$）：
- 记录第 $t \in \{0,\dots,R-1\}$ 轮的 **7 个压缩模加进位字**（对齐 `engine.py` L455–L472 平面 $c[3..9]$）：
  - $T_1$ 链：$p_1 = h + \Sigma_1(e),\; c_3 = h \oplus \Sigma_1(e) \oplus p_1$；$p_2 = p_1 + \text{Ch}(e,f,g),\; c_4 = p_1 \oplus \text{Ch} \oplus p_2$；$p_3 = p_2 + K_t,\; c_5 = p_2 \oplus K_t \oplus p_3$；$T_1 = p_3 + W_t,\; c_6 = p_3 \oplus W_t \oplus T_1$；
  - $T_2$ 链：$T_2 = \Sigma_0(a) + \text{Maj}(a,b,c),\; c_7 = \Sigma_0(a) \oplus \text{Maj} \oplus T_2$；
  - 更新链：$a_{\text{new}} = T_1 + T_2,\; c_8 = T_1 \oplus T_2 \oplus a_{\text{new}}$；$e_{\text{new}} = d + T_1,\; c_9 = d \oplus T_1 \oplus e_{\text{new}}$。
- 第 $R$ 轮单层输出摘要（Davies-Meyer 前馈加回 Midstate $S_0$）：
  $$H^{(1)}_R(\Theta, x) = \bigl(S_R(\Theta, x) + S_0\bigr) \bmod 2^{32} \in (\mathbb{Z}_{2^{32}})^8.$$

### 1.3 Double-SHA256（$R = 64 + 64$）精确定义

第一重运行完整 $R=64$ 轮 Block 2 压缩得到 $H^{(1)} = H^{(1)}_{64}(\Theta, x) \in (\mathbb{Z}_{2^{32}})^8$。第二重构造 512-bit 单块填充输入：
$$W_{\text{dbl}}[0..7] = H^{(1)}[0..7], \quad W_{\text{dbl}}[8] = \text{0x80000000}, \quad W_{\text{dbl}}[9..14] = 0, \quad W_{\text{dbl}}[15] = 256,$$
以标准 `SHA256_IV`（`src/sha256_ref.py` L44）为初态运行完整 64 轮标准 SHA-256 压缩，前馈加回 `SHA256_IV`，输出最终 256-bit Double-SHA256 摘要：
$$H^{(2)}(\Theta, x) = \bigl(S^{(\text{dbl})}_{64} + \text{IV}\bigr) \bmod 2^{32} \in (\mathbb{Z}_{2^{32}})^8.$$

---

## 2. 两阶段防泄漏铁律（Strict Train/Holdout Nonce Isolation）

为彻底杜绝在有限样本上搜出“过拟合虚假偏置”，实施以下**物理级 Nonce 空间互斥隔离**：

1. **Nonce 空间互不相交划分**：
   - **训练 Nonce 池 $\mathcal{N}_{\text{train}}$**：严格限定为 $x \in [0,\; 2^{30})$（即最高 2 bits 为 `00`，由固定训练种子 `SEED_TRAIN = 20261105` 采样）。
   - **留出 Nonce 池 $\mathcal{N}_{\text{holdout}}$**：严格限定为 $x \in [2^{30},\; 2^{32})$（由独立测试种子 `SEED_HOLDOUT = 998244353` 无放回或高熵独立抽取）。
   - 两集合满足 $\mathcal{N}_{\text{train}} \cap \mathcal{N}_{\text{holdout}} = \emptyset$，**零样本重叠**。
2. **阶段一（弱模板搜索/优化，仅见 $\mathcal{N}_{\text{train}}$）**：
   - 搜索器在训练阶段每步仅可访问 $\mathcal{N}_{\text{train}}$ 的小批量（例如每轮迭代评估 $B_{\text{train}} = 1024 \sim 2048$ 个训练 Nonce），输出最多 $K = 16$ 个离散精确 `uint32` 弱模板 $\{\Theta^{(1)}, \dots, \Theta^{(K)}\}$。
   - **禁止**在阶段一读取任何 $\mathcal{N}_{\text{holdout}}$ 样本、禁止用 $\mathcal{N}_{\text{holdout}}$ 做早停、挑选或超参调节。
3. **阶段二（独立留出集精确核验，冻结模板）**：
   - 冻结 Top-$K$（$K=16$）模板后，对每个模板在 $\mathcal{N}_{\text{holdout}}$ 上独立抽取至少 **$N_{\text{holdout}} = 262{,}144$（$2^{18}$）** 个 Nonce，用纯整数精确前向 `int64 & 0xFFFFFFFF` 计算全部观测指标与 $z$-score。
   - 若任意模板在 $N_{\text{holdout}} = 262{,}144$ 上达到 $|z| \ge 4.5$，追加二次独立验证集 $N_{\text{holdout2}} = 1{,}048{,}576$（$2^{20}$）核实是否复现。

---

## 3. 多维度观测量与反平凡进位分解

对每个轮数设定与每个模板 $\Theta$，在 $\mathcal{N}_{\text{holdout}}$ 上精确统计以下 5 类指标：

1. **输出汉明重量均值 $\overline{\text{HW}}(H)$**：
   $$\text{HW}(H) = \sum_{j=0}^{7} \text{popcnt}_{32}(H[j]), \qquad \mu_0 = 128.0,\quad \sigma_0 = 8.0.$$
2. **输出前导零（双端序齐测）**：
   - **大端前导零 `clz`**：从 $H[0]$ 最高比特向下连续为 `0` 的位数（理论均值 $\mu_0 = 1.0$，方差 $\sigma_0^2 = 2.0$）；
   - **BTC 小端难度前导零 `ctz_btc`**：将 32 字节摘要按小端 256-bit 整数解释的高位连续零位数（即从 $H[7]$ 低字节向下至 $H[0]$ 逐字节大端对齐计数，或等效字节反转 `bswap32(H[7-j])` 的高位 `clz`；理论同样为 $\mu_0 = 1.0,\; \sigma_0^2 = 2.0$）；
   - 同时统计 $\mathbb{P}(\text{clz} \ge 8)$ 与 $\mathbb{P}(\text{ctz\_btc} \ge 8)$（理论基准 $p_0 = 2^{-8} = 1/256$）。
3. **256-bit 单比特最大边缘偏置 $z_{\text{bit, max}}$**：
   对输出全部 256 个比特 $b \in \{0,\dots,255\}$ 计算经验 `1` 概率 $\hat{p}_b$ 及单比特 $z$-score：
   $$z_b = 2\sqrt{N_{\text{holdout}}}\,(\hat{p}_b - 0.5), \qquad z_{\text{bit, max}} = \max_{0 \le b < 255} |z_b|.$$
4. **内部进位活跃率（严防前 3 轮静态零进位作弊）**：
   将每轮 7 个压缩加法进位字（$224$ bits/轮）拆分为两项：
   - **全轮压缩进位活跃率 $\rho_{\text{carry}}^{\text{all}}$**：$\frac{1}{224 R}\sum_{t=0}^{R-1}\sum_{m=3}^{9} \text{popcnt}(c_{t,m})$；
   - **Nonce 穿透后动态进位活跃率 $\rho_{\text{carry}}^{\text{post}}$（$t \ge 4$）**：仅统计 $t \in \{4, \dots, R-1\}$ 的压缩进位密度（因为 $t \in \{0,1,2\}$ 的输入均不含 $W[3]$，可被模板 $\Theta$ 解析固定为低进位常数；真正反映弱 Midstate 是否穿透 Nonce 混合层的指标是 $t \ge 4$ 的进位活跃率与相邻 Nonce 进位翻转敏感度！）。
5. **消息扩展稀疏性 $\rho_W$**：
   对 $R \ge 16$，统计扩展字 $W[16..R-1]$ 的比特密度（以及 $t \ge 18$ 受 Nonce 影响窗口的密度 $\rho_{W, t \ge 18}$）。

---

## 4. 弱模板搜索器架构与双基线对照组

### 4.1 弱模板搜索器设计（纯 FP32 + 精确整数反馈）

为在 Colab 单卡 45 分钟总预算内充分探测模板空间，搜索器在每个轮数 $R$ 上联合采用两条互补路径（共维持种群并筛选出全局训练适应度最高的 Top-16 模板）：

1. **路径 A：代数稀疏锚定 + 演化搜索（Algebraic-Seeded Evolutionary Search）**：
   - 利用 Block 2 解析结构初始化一半种群：随机选取 $W[2]$，强制设定 $W[1] = (-\sigma_1(640) - \sigma_0(W[2])) \bmod 2^{32}$ 及 $W[0] = (-\sigma_0(W[1])) \bmod 2^{32}$（使 $W[16] = W[17] = 0$），同时尝试解析对消第 $0,1,2$ 轮初始进位链（如设定 $e_0, h_0$ 抵消 $K_0 + W_0$ 等）；另一半种群使用随机均匀 $S_0, W[0..2]$（含真实 Block 1 生成的合法 Midstate）。
   - 执行多代遗传变异（单比特翻转、字内半字节/字级交叉、邻近模板局部爬山），以训练 Nonce 批上的复合适应度引导进化：
     - 目标函数（最小化）：组合 $t \ge 4$ 进位活跃率、消息扩展密度 $\rho_{W, t\ge 18}$、以及输出 $\max_b |\hat{p}_b - 0.5|$ / $|\overline{\text{HW}} - 128|$ / $-\overline{\text{LZ}}$ 的标准化 $z$ 分数（多目标分摊至不同子种群：部分子种群专攻最小化进位/稀疏性，部分专攻最大化输出偏置/前导零，确保不遗漏任何方向的弱点）。
2. **路径 B：可微松弛弱模板梯度搜索（Soft-Bit STE / Continuous Relaxation）**：
   - 构造纯 FP32（`allow_tf32=False`, `torch.float32`）可微松弛模板 logits $\alpha \in \mathbb{R}^{11 \times 32}$：通过 $\text{sigmoid}(\alpha / \tau)$ 与直通估计器（STE）二值化映射为精确 `uint32` 字输入前向（或 Soft-XOR / Soft-Add32 连续近似反向传播），对训练 Nonce 批上的软前导零、软汉明权重与内部软进位能量求梯度更新 $\alpha$，提取每步量化后的最优离散模板纳入候选池。

### 4.2 双强制对照组（在所有 8 个扫频点同步运行）

1. **均匀随机 Midstate 基线（`BASE_RAND_MIDSTATE`）**：
   - 不经搜索，随机采样 $K=16$ 个模板（8 个 $S_0 \sim \text{Uniform}(\mathbb{Z}_{2^{32}}^8)$ + 8 个随机 Block 1 经 64 轮压缩输出的合法 Midstate，$W[0..2] \sim \text{Uniform}(\mathbb{Z}_{2^{32}}^3)$）。
   - 在**同一留出 Nonce 集 $\mathcal{N}_{\text{holdout}}$**（每模板 $262,144$）上计算全部 5 类指标均值、标准差与极值。
2. **同构随机 ROTR/K 克隆对照组（`CTRL_RAND_ROTR_K`）**：
   - 构造与真实 SHA-256 计算图完全同构的克隆算子：保留 `Ch`、`Maj`、模 $2^{32}$ 加法、BTC 320-bit 固定填充及 Double-SHA256 拓扑；使用固定对照种子 `SEED_CTRL = 135792468` 重新均匀采样 64 个轮常数 $\widetilde{K}[0..63] \sim \text{Uniform}(\mathbb{Z}_{2^{32}})$ 与 12 个位移参数（$\Sigma_0, \Sigma_1$ 各 3 个互异 `ROTR`，$\sigma_0, \sigma_1$ 各 2 个互异 `ROTR` + 1 个 `SHR`，均不同于标准 SHA-256）。
   - 在克隆算子上运行**与真实 SHA-256 完全相同算法、相同搜索预算、相同 $\mathcal{N}_{\text{train}}$** 的弱模板搜索，提取 Top-16 模板并在 $\mathcal{N}_{\text{holdout}}$（每模板 $262,144$）上验证，用于直接剥离搜索过程带来的选择极值偏差（Selection Bias）与通用 320-bit 填充拓扑效应。

---

## 5. 扫频阶梯、统计判定与 $z \ge 6$ 亮灯门限

### 5.1 扫频阶梯（共 8 档）

$$R \in \{8,\; 12,\; 16,\; 20,\; 24,\; 32,\; 64\} \;\cup\; \{\text{Double-SHA256 } (R=64+64)\}.$$

单轮评估规模：每档 $3 \text{ 组 (Main, Base, Ctrl)} \times 16 \text{ 模板} \times 262,144 \text{ Nonce} = 12{,}582{,}912$ 次精确前向（约 $1.0 \times 10^8$ 次总前向，Colab L4/T4 向量化耗时仅数分钟，留给阶段一搜索约 $25\text{--}30$ 分钟，总计严格 $\le 45$ 分钟）。

### 5.2 统计 $z$-score 计算与亮灯裁决

在留出集（$N = 262,144$）上，针对每档 $R$ 的 Top-16 模板中的最优者（`best_of_16`），分别报告：
- **对理论随机基线的 $z$-score**：
  - $|z_{\text{HW}}| = \frac{|\overline{\text{HW}} - 128.0|}{8.0 / \sqrt{N}}$
  - $z_{\text{clz}} = \frac{\overline{\text{clz}} - 1.0}{\sqrt{2.0 / N}}$，$z_{\text{btc\_lz}} = \frac{\overline{\text{ctz\_btc}} - 1.0}{\sqrt{2.0 / N}}$
  - $z_{\text{bit, max}} = \max_{0 \le b < 256} 2\sqrt{N}\,|\hat{p}_b - 0.5|$
  - 内部动态进位偏差 $z_{\text{carry, post}} = \frac{\bar{\rho}_{\text{carry}}^{\text{post}}(\text{Main}) - \bar{\rho}_{\text{carry}}^{\text{post}}(\text{Base})}{\text{SE}_{\text{pooled}}}$（仅作结构诊断，不单独等同于输出破解）
- **相对同构随机 ROTR/K 对照组的差异 $z$-score**：
  $$z_{\text{vs\_ctrl}} = \frac{M_{\text{best}}(\text{Main}) - M_{\text{best}}(\text{Ctrl})}{\sqrt{\text{SE}_{\text{Main}}^2 + \text{SE}_{\text{Ctrl}}^2}}$$
  （其中 $M_{\text{best}}$ 为对应输出统计量的极值偏离度）。

**全局结案裁决门限（Verdict Criteria）**：
- **`HIT`（发现可利用弱 Midstate 共振子空间）**：
  在任意 **$R \ge 16$**（包含 $R=64$ 或 `Double-SHA256`）的独立留出集（$N_{\text{holdout}} \ge 262,144$）上，至少一项输出摘要指标（`HW`、`clz`、`ctz_btc` 或 `bit_max`）**同时满足**：
  1. 相对理论均匀随机基线达到 $|z_{\text{holdout}}| \ge 6.0$（且经 $N_{\text{holdout2}} = 2^{20}$ 二次验证仍保持 $|z| \ge 6.0$，排除 $16 \times 256$ 多重比较极值巧合）；
  2. 相对同构随机 ROTR/K 克隆对照组显著胜出（$|z_{\text{vs\_ctrl}}| \ge 4.0$）。
- **`BORDERLINE`（浅层或内部微结构共振，未穿透多轮输出）**：
  - 仅在浅轮 $R \in \{8, 12\}$ 上观察到输出偏置 $|z_{\text{holdout}}| \ge 6.0$（但在 $R \ge 16$ 迅速消失）；**或**
  - 在 $R \ge 16$ 观察到 $3.5 \le |z_{\text{holdout}}| < 6.0$ 的边缘输出偏置；**或**
  - 仅在消息扩展稀疏性 / 内部进位活跃率（$t \ge 4$）出现超越随机 ROTR/K 对照组的局部抑制，但未能传递至输出摘要端。
- **`NULL`（未发现弱 Midstate 共振信号）**：
  全部 $R \ge 16$（含 $R=20, 24, 32, 64$ 及 `Double-SHA256`）的留出集输出指标均满足 $|z_{\text{holdout}}| < 6.0$，且与 `BASE_RAND_MIDSTATE` 和 `CTRL_RAND_ROTR_K` 统计不可区分，证明 BTC Block 2 的 320-bit 固定填充与可控 Midstate/$W[0..2]$ 在 16 轮之后已完全被 ARX 雪崩同化，无法降低有效挖矿复杂度。

---

## 6. 交付物与 JSON 输出格式约定（供 `@p5-eng` 实施）

请 `@p5-eng` 在 `g:\SHA-LM\v11_discovery\p5_btc_weak_midstate\` 下实施并输出以下文件：
1. 实验代码（例如 `btc_weak_midstate.py`）及独立单元测试/本地 CPU 秒级冒烟验证（`CUDA_VISIBLE_DEVICES=""`）；
2. 云端完整扫频与留出验证结果同步至 `g:\SHA-LM\v11_discovery\p5_btc_weak_midstate\results\`：
   - `p5_sweep_summary.json`：包含 8 个扫频点（`8, 12, 16, 20, 24, 32, 64, double64`）的 `main`, `base_rand_midstate`, `ctrl_rand_rotr_k` 完整留出统计量（`N_holdout`, `hw_mean`, `z_hw`, `clz_mean`, `z_clz`, `btc_lz_mean`, `z_btc_lz`, `max_bit_bias`, `z_bit_max`, `carry_all`, `carry_post4`, `z_carry_post4`, `w_density_post18`, `z_vs_ctrl` 及各 Top-16 模板十六进制字序列）；
   - `p5_run_meta.json`：记录随机种子、`SEED_TRAIN`/`SEED_HOLDOUT`、硬件型号（L4/T4）、FP32 锁定状态、耗时及 Colab 会话停止证明。
3. 同步结果后请务必先停机 Colab `v11-p5` 并核实 `colab.exe sessions`，再 `@p5-math` 接手独立核验并撰写 `verdict.json` 与 `REPORT.md`。
