# P2「时空晶体等零面零模波包（Crystal Zero-Mode Wavepacket）」数学设计契约（SPEC）

> **工作目录**：`g:\SHA-LM\v11_discovery\p2_zero_mode\`（严格只读复用 `v10_ouroboros64/ouroboros_core/engine.py` 与 `v10_ouroboros64/ouroboros_net/model.py`、`heads.py`，严禁修改 `v10_ouroboros64/`、`runs/`、`formal_proofs/`、`formal_proofs_v10/`）。  
> **硬件与数值铁律**：本地仅允许 CPU（`CUDA_VISIBLE_DEVICES=""`）秒级冒烟；全量训练与扫频在 Colab L4（或 T4）上纯 IEEE-754 FP32 执行（`torch.backends.cuda.matmul.allow_tf32 = False`、`torch.backends.cudnn.allow_tf32 = False`、全模型 `torch.float32`），核心比特计算使用精确 `uint32`/`int64`。

---

## 1. 核心科学问题与等零面零模定义

### 1.1 前向精确晶体流形（$E_{\text{interior}} = 0$）

给定轮数 $R \in \{4, 6, 8, 10, 12, 16, 20\}$ 与 16 字输入块 $W^{(0)} \in (\mathbb{Z}/2^{32}\mathbb{Z})^{16}$，由 `TorchCrystalEngine.compress_exact(W^{(0)}, rounds=R)` 与 `OuroborosCore.compress_exact(W^{(0)}, rounds=R)` 精确前向展开，得到无内部缺陷的三维时空晶体：
$$\mathcal{C}^{(0)} = \bigl(H^{(0)},\; W,\; S,\; C\bigr), \qquad E_{\text{interior}}(\mathcal{C}^{(0)}) = 0$$
其中：
- $W \in (\mathbb{Z}/2^{32}\mathbb{Z})^{R}$：第 $0..15$ 轮为输入字 $W^{(0)}[0..\min(15,R-1)]$，当 $R > 16$ 时 $W[16..R-1]$ 由 SHA-256 消息扩展递推决定；
- $S \in (\mathbb{Z}/2^{32}\mathbb{Z})^{(R+1)\times 8}$：$S[0] = IV$，$S[R] = (H^{(0)} - IV) \bmod 2^{32}$；
- $C \in (\mathbb{Z}/2^{32}\mathbb{Z})^{R\times 10}$：10 个精确模 $2^{32}$ 加法进位位平面（$0..2$ 为消息扩展进位，$3..9$ 为压缩轮进位）；
- 比特坐标约定与 `engine.py` 一致：字内采用 **LSB-first**（$b=0$ 为最低位，$b=31$ 为最高位）；输出摘要前缀零位采用 **标准 SHA-256 大端前缀顺序**（从 `H[0]` 的最高有效位 $b=31$ 依次向下）：
  $$y_j(W^{(0)}) := \left\lfloor \frac{H_R(W^{(0)})[0]}{2^{31 - j}} \right\rfloor \bmod 2 \;\in\; \{0, 1\}, \qquad j = 0, 1, \dots, 31.$$
  （注：本实验扫频 $k \in \{1..12\}$，$0 \le j \le 12 < 32$ 全部位于 `H[0]`；选择 `H[0]` 而非 `H[7]` 的数学原因是：`S[R][0] = a_R` 在任意 $R \ge 4$ 下均直接耦合第 $R-1$ 轮压缩输入并混合全活跃输入字，避免 $R=4$ 时 `H[7]=e_1` 对 $W[1..3]$ 退化解耦。）

### 1.2 等零面基底晶体 $\mathcal{M}_{R,k}$ 与零模波包 $\Delta W$

对给定 $(R, k)$，定义**第 $k$ 阶等零面阶梯基底流形** $\mathcal{M}_{R,k}$：
$$\mathcal{M}_{R,k} = \Bigl\{ W^{(0)} \in \mathcal{D}_{\text{carrier}} \;\Big|\; y_0(W^{(0)}) = \dots = y_{k-1}(W^{(0)}) = 0 \;\land\; y_k(W^{(0)}) = 1 \Bigr\}.$$
即基底晶体 $\mathcal{C}^{(0)}$ 已前向精确满足前 $k$ 个输出位为 $0$，且第 $k$ 位恰处于 $1 \to 0$ 跃迁门槛上。

**虚拟靶向边界 $H_{\text{target}}^*$（注入 3D 时空晶体网络的物理张力）**：
为将阶梯跃迁目标通知 `Ouroboros64Net`（复用 `model.py` 的 `build_crystal_features` 与 `eval_crystal_batch`），构造仅在 `H[0]` 第 $31-k$ 比特处由 $1$ 翻为 $0$ 的目标摘要：
$$H_{\text{target}}^*[0] = H^{(0)}[0] \oplus 2^{31-k}, \qquad H_{\text{target}}^*[1..7] = H^{(0)}[1..7].$$
此时晶体内部完全无缺陷（$\Delta W_{\text{int}} = 0, \Delta S_{\text{mid}} = 0, \Delta C = 0$），仅右边界 $S[R][0]$ 承受局部靶向翻位张力。

**合法自由输入微扰域 $\Omega_{\text{free}}(R)$ 与两类载体 $\mathcal{D}_{\text{carrier}}$**：
1. **载体 A：标准随机前缀晶体（`STD`）**
   - 输入字 $W^{(0)}[0..15] \sim \text{Uniform}(\{0,1\}^{32})^{16}$。
   - **活跃自由微扰域**：仅允许微扰在 $R$ 轮内真实进入压缩函数的输入字 $W[0..w_{\text{free}}-1]$，其中 $w_{\text{free}}(R) = \min(16, R)$（共 $32 \cdot w_{\text{free}}(R)$ 个自由比特；注：严禁微扰 $t \ge R$ 的未参与计算字，杜绝无效伪微扰）。
2. **载体 B：BTC Block-2 320-bit 固定填充晶体（`BTC320`）**
   - 对齐 `v10_ouroboros64/ouroboros_bench/open_boundary_alpha_probe.py` 约定（标准比特币 80 字节区块头第 2 块填充）：
     $$W^{(0)}[0..3] \sim \text{Uniform}(\{0,1\}^{32})^4, \quad W^{(0)}[4] = \text{0x80000000}, \quad W^{(0)}[5..14] = 0, \quad W^{(0)}[15] = 640.$$
   - **活跃自由微扰域**：严格锁定填充字 $\Delta W^{(0)}[4..15] = 0$，仅允许微扰 $W^{(0)}[0..3]$（共 $128$ 个自由比特，$w_{\text{free}}(R) = 4$）。

**等零面零模波包命中（Zero-Mode Wavepacket Hit）判定**：
模型或基线提议的非平凡输入微扰 $\Delta W^{(0)} \in \{0,1\}^{16\times 32}$ 须同时满足：
1. **非平凡性与域合法性**：$\Delta W^{(0)} \neq 0$（Hamming 重 $\|\Delta W^{(0)}\|_0 \ge 1$），且支撑集 $\text{supp}(\Delta W^{(0)}) \subseteq \Omega_{\text{free}}(R)$；对同一基底晶体 $\mathcal{C}^{(0)}$，**必须去重**（同一 $\Delta W^{(0)}$ 不得重复计入分母或分子）。
2. **精确前向真值核验**：令 $W'^{(0)} = W^{(0)} \oplus \Delta W^{(0)}$，经标准前向 $R$ 轮 SHA-256 压缩 $H' = H_R(W'^{(0)})$，精确满足：
   $$y_0(W'^{(0)}) = \dots = y_{k-1}(W'^{(0)}) = 0 \quad (\Delta H_{0..k-1} = 0) \qquad \land \qquad y_k(W'^{(0)}) = 0 \quad (1 \to 0\text{ 翻转}).$$
   第 $k+1..255$ 位输出保持自由开放，不施加约束。

---

## 2. 理论基线与强制随机对照组

### 2.1 随机游走理论基线 $p_0(k) = 2^{-(k+1)}$

在全混淆随机置换（Random Oracle）假设下，任意非零去重微扰 $\Delta W^{(0)} \neq 0$ 产生的新输出 $H_R(W^{(0)} \oplus \Delta W^{(0)})$ 在前 $k+1$ 比特上独立均匀分布于 $\{0,1\}^{k+1}$。因此维持前 $k$ 个零位不变且第 $k$ 位由 $1 \to 0$ 的理论命中概率严格为：
$$p_0(k) = \mathbb{P}_{\text{rand}}\!\bigl(y_0=\dots=y_{k-1}=0 \;\land\; y_k=0\bigr) = 2^{-(k+1)}.$$

同时在每个 $(R, k)$ 单元上记录**同预算经验随机多比特微扰基线（`RAND_WALK`）**：在与神经提议器相同的 $\Omega_{\text{free}}(R)$ 与相同的 Hamming 重分布（或 $m \in \{1..6\}$ 均匀随机翻转）下抽样去重候选 $\Delta W^{(0)}$，用于交叉验证低轮数（如 $R=4$）是否已满足 $2^{-(k+1)}$ 均匀性。

### 2.2 随机 ROTR/K 克隆对照组（`CTRL_RAND_SHA`）

为区分真实 SHA-256 的 ARX 代数结构与神经网络自身的拟合/记忆效应，必须构建并**同配置训练、同预算评测**一个结构完全同构的随机克隆版 $R$ 轮压缩函数 $\widetilde{H}_R$：
- 保留 16 字消息扩展、8 字状态寄存器与 10 平面模 $2^{32}$ 加法拓扑不变；
- **随机置换全部循环移位与逻辑右移参数**（固定对照种子，从 $\{1..31\}$ 独立采样 $\Sigma_0, \Sigma_1, \sigma_0, \sigma_1$ 的全部 12 个移位偏移量，保证每组三个偏移量互异）；
- **随机替换全部轮常数** $\widetilde{K}[0..63] \sim \text{Uniform}(\{0,1\}^{32})^{64}$；
- `Ouroboros64Net` 编码层中的 `k_bits_bipolar` 与 `TorchCrystalEngine` 克隆实例同步使用 $(\widetilde{\text{ROTR}}, \widetilde{K})$。

---

## 3. 3D 时空晶体零模提议器与训练/评测协议

### 3.1 模型复用与纯 FP32 约束

- 直接复用 `v10_ouroboros64/ouroboros_net/model.py` 的 `Ouroboros64Net` 主干（`TensorBPStack` + `DefectHealingHead` + `WolffClusterProposer`），全链路 `enforce_fp32_determinism()` + 纯 `torch.float32`。
- 为适配 $R \in \{4..20\}$ 扫频并控制在 Colab 单会话 $\le 45$ 分钟预算内，建议采用轻-中规模 `Ouroboros64Net` 配置（例如 `d_node=64..96, n_layers=4, bond_dim=4, d_key=24, max_cluster_size=6`），输入为 $(H_{\text{target}}^*, W, S, C)$。
- **多比特波包采样接口**：从网络最终隐状态 `H_final` 与 `flip_logits[:, :w_free, 0, :]`（或限定于 plane=0, $t < w_{\text{free}}$ 的 `WolffClusterProposer` 协同团簇解码器）联合生成多比特输入微扰 $\Delta W^{(0)}$（支持 $m \in \{1..8\}$ 协同位点采样），并严格过滤 $\Delta W^{(0)} = 0$ 及去重。

### 3.2 严格零泄漏训练/验证划分

- **禁止在测试基底晶体上训练或调参**。
- 对每个轮数 $R$（及载体类型 `STD` / `BTC320`），在独立的训练基底晶体流形上通过以下信号训练提议器（真实 SHA-256 与随机 ROTR/K 对照组采用完全对称的超参数与步数）：
  - **前缀存活软奖励 / 策略梯度（REINFORCE）或加权正样本极大似然**：对每条提议的去重 $\Delta W^{(0)}$，计算精确前向输出的前缀零保留数 $z_{\text{match}} = \text{lcp}_0(H')$ 与是否实现第 $k$ 位 $1 \to 0$ 翻转，用精确奖励（命中给高正奖励，未命中按前缀零存活率或最低 Hamming 距给梯度信号，减去批内基线均值）更新提议器参数。
- 训练完成后**冻结参数（`model.eval()`, `torch.no_grad()`）**，在全新独立随机种子生成的**留出基底晶体测试集**上执行全部 84 格点扫频评估。

---

## 4. 扫频矩阵、阶梯增益与统计显著性判定

### 4.1 扫频矩阵

- **轮数阶梯**：$R \in \{4, 6, 8, 10, 12, 16, 20\}$（7 档）
- **等零面阶数**：$k \in \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\}$（12 档）
- 共 $7 \times 12 = 84$ 个 $(R, k)$ 评估单元；分别在 **STD 真实 SHA-256**、**BTC320 真实 SHA-256** 与 **STD 随机 ROTR/K 对照组**（以及至少核心轮数/全部 84 格点的 **RAND_WALK** 基线）上运行。

### 4.2 评估样本量 $N_{\text{eval}}(R, k)$ 与去重预算

为保证即使在最高难度 $k=12$（$p_0(12) = 2^{-13} \approx 1.22 \times 10^{-4}$）下仍有足够的期望命中数以检验 $z \ge 6$：
- 建议每个 $(R, k)$ 单元的独立去重候选评估总次数设为：
  $$N_{\text{eval}}(R, k) = \begin{cases} 32{,}768 & (1 \le k \le 10) \\ 131{,}072 & (k \in \{11, 12\}) \end{cases}$$
  （例如每格取 $N_{\text{base}} = 64 \sim 256$ 个独立基底晶体 $\times$ 每晶体 $512$ 个去重候选 $\Delta W^{(0)}$）。此时在最难的 $k=12$ 下，随机期望命中数仍有 $N_{\text{eval}} p_0(12) = 16$，统计检验力充足且 GPU 前向耗时极低。

### 4.3 阶梯增益与二项 z-score 亮灯门限

对每个 $(R, k)$ 单元，记观测去重有效评估总数为 $N$，经**标准前向 $R$ 轮 SHA-256 精确核验**通过的命中数为 $S$，观测命中率为 $\hat{P}_{\text{hit}}(R, k) = S / N$：
1. **阶梯增益（Step Gain）**：
   $$G(R, k) = \frac{\hat{P}_{\text{hit}}(R, k)}{2^{-(k+1)}}.$$
2. **精确二项 z-score（单边正增益检验，对比理论基线 $p_0 = 2^{-(k+1)}$）**：
   $$z(R, k) = \frac{S - N \cdot 2^{-(k+1)}}{\sqrt{N \cdot 2^{-(k+1)} \bigl(1 - 2^{-(k+1)}\bigr)}}.$$
   同时报告：
   - 精确单边二项检验 p-value（$p_{\text{binom}} = \mathbb{P}[\text{Binomial}(N, p_0) \ge S]$）与命中率 Wilson 95% 置信区间；
   - **对照净增益 $G_{\text{net}}(R, k) = \hat{P}_{\text{hit}}^{\text{SHA}}(R, k) / \max(\hat{P}_{\text{hit}}^{\text{CTRL}}(R, k), 1/N)$** 及两样本比例差异 z-score $z_{\text{vs\_ctrl}}$。
3. **亮灯 / 结案门限**：
   - **单格亮灯（HIT cell）**：同时满足 $z(R, k) \ge 6.0$、$G(R, k) > 1.0$、且在对应随机 ROTR/K 对照组上未出现同等幅度的伪增益（$z_{\text{vs\_ctrl}}(R, k) \ge 4.0$）。
   - **全局结论分级**：
     - `HIT`：在 $R \ge 8$（或成片连续 $(R,k)$ 区域）存在显著超越随机游走与随机 ROTR/K 对照组的真实 SHA-256 零模波包信号（$z \ge 6$ 且 $z_{\text{vs\_ctrl}} \ge 4$）；
     - `BORDERLINE`：仅在极低轮 $R \in \{4, 6\}$ 出现结构信号，或仅对随机基线微弱超标（$3 \le z < 6$）但相对随机 ROTR/K 对照组不显著；
     - `NULL`：全扫频矩阵（尤其 $R \ge 8$）均未出现超越 $2^{-(k+1)}$ 与随机 ROTR/K 对照组的显著零模信号（各格 $z < 6$ 或与对照组无统计分离），证实等零面流体流形在 $R \ge 8$ 迅速坍缩回 $2^{-(k+1)}$ 各向同性随机游走。

---

## 5. 3D 波包热力图与稀疏团簇结构提取协议

对所有评测命中样本（以及重点观察低轮 $R \in \{4, 6\}$ 与最高 $z$-score 格点），必须执行以下波包结构提取并落盘：

1. **全晶体 3D 差分波包定义**：
   对任意命中样本 $\bigl(W^{(0)}, \Delta W^{(0)}\bigr)$，分别前向计算 $\mathcal{C}^{(0)} = (H^{(0)}, W, S, C)$ 与 $\mathcal{C}' = (H', W', S', C')$，定义三轴布尔波包张量 $\Delta \mathcal{X} \in \{0,1\}^{R \times 19 \times 32}$：
   - 平面 $p = 0$：$\Delta \mathcal{X}[t, 0, b] = \text{bit}_b\bigl(W'[t] \oplus W[t]\bigr)$；
   - 平面 $p = 1..8$：$\Delta \mathcal{X}[t, p, b] = \text{bit}_b\bigl(S'[t+1, p-1] \oplus S[t+1, p-1]\bigr)$；
   - 平面 $p = 9..18$：$\Delta \mathcal{X}[t, p, b] = \text{bit}_b\bigl(C'[t, p-9] \oplus C[t, p-9]\bigr)$。
2. **3D 波包热力图聚合**：
   在目标 $(R, k)$ 单元的命中集合上统计逐格点翻转概率热力图 $\bar{\Delta}(t, p, b) = \mathbb{E}_{\text{hit}}[\Delta \mathcal{X}(t, p, b)] \in [0, 1]^{R \times 19 \times 32}$，并与随机未命中微扰热力图 $\bar{\Delta}_{\text{miss}}$ 对比：
   - 计算逐轮状态活跃翻转密度 $\rho_S(t) = \frac{1}{256}\sum_{p=1}^8 \sum_{b=0}^{31} \bar{\Delta}(t, p, b)$（观察波包在前向传播中是否保持局部稀疏通道或何时迅速扩散至 $0.5$ 热寂平衡态）；
   - 计算时空稀疏度指标：相对 $0.5$ 的平均 KL 散度 $\overline{D}_{\text{KL}}(\bar{\Delta} \,\|\, 0.5)$ 及高度偏向位点比例 $\eta_{\text{bias}} = \text{mean}(|\bar{\Delta} - 0.5| > 0.15)$。
3. **稀疏团簇结构提取（Sparse Cluster Extraction）**：
   - 统计命中样本的输入微扰 Hamming 重分布 $P(\|\Delta W^{(0)}\|_0 = m \mid \text{hit})$ vs 提议分布 $P(\|\Delta W^{(0)}\|_0 = m)$；
   - 提取排名前 20 的高频/高收益稀疏输入团簇模式（记录 $(t, \text{bit})$ 坐标列表、字内/跨字偏移 $\Delta b$、命中次数及对应样本的精确输入块 `W0_hex`、`dW_hex`、`H0_hex`、`H1_hex`），保存至 `results/` JSON 供独立 CPU 复核。

---

## 6. 结果文件契约（供 `@p2-eng` 回传与 `@p2-math` 独立复核）

`@p2-eng` 在 Colab 执行完毕并全量同步至 `g:\SHA-LM\v11_discovery\p2_zero_mode\results\`、确认 `colab.exe stop -s v11-p2` 停止后，需提供以下标准输出文件：
1. `results/sweep_matrix.json`：包含环境元数据（GPU 型号、FP32/TF32 锁定确认、随机种子、训练耗时、评测耗时）及全部 $(R, k, \text{carrier}, \text{method})$ 格点的 `{R, k, carrier, method, N_eval, hits, p_hit, p_rand, gain_G, z_score, p_binom, ci95_low, ci95_high, mean_hamming_w}`。
2. `results/wavepacket_structures.json`：包含代表性 $(R, k)$ 单元的 3D 热力图统计（逐轮 $\rho_S(t), \rho_W(t), \rho_C(t)$、KL 稀疏度、$\eta_{\text{bias}}$、Hamming 重分布）及可独立重算验真的样本列表（`W0_u32`, `dW0_u32`, `H0_u32`, `H1_u32`, `R`, `k`, `carrier`, `method`）。
3. `results/colab_run.log`：完整无缓冲远端训练与扫频日志，及本地确认 `v11-p2` 已停止的会话状态记录。
