# P12 数学与协议规范：超广义 AsicBoost —— Double-SHA256 广义中性比特团簇与多输入摊还复用

**工作目录**：`g:\SHA-LM\v12_frontier\track_a_p12_amortized\`  
**数学与协议架构**：`@p12-math`（拥有 `SPEC.md`、`REPORT.md`、`verdict.json` 及必要时 `lean_verify/`）  
**GPU + SAT 工程实现**：`@p12-eng`（拥有全部 `.py` 脚本与 `results/`，专属 Colab 会话 `v12-p12`）  
**独立审计官**：`@v12-auditor`  
**禁改隔离区**：`v11_discovery/`、`v10_ouroboros64/`、`runs/`、`formal_proofs/`、`formal_proofs_v10/`、`v6a/`、`src/`  
**参考基准代码**：[sha256_ref.py](file:///g:/SHA-LM/src/sha256_ref.py)、[P3 SPEC.md](file:///g:/SHA-LM/v11_discovery/p3_symbolic_lean4/SPEC.md)、[P5 REPORT.md](file:///g:/SHA-LM/v11_discovery/p5_btc_weak_midstate/REPORT.md)

---

## 0. 核心科学问题与反幻觉铁律

经典 Overt/Covert AsicBoost（Timo Hanke, 2016）通过外/内循环换序，在多个拥有不同 Midstate 的候选区块头间复用第一轮压缩块后半段（80-byte Header 的第二个 64-byte 块，即 `Chunk2`）的**消息扩展器 `W[16..63]`**，实现约 **13%~20%** 的矿机算力/能耗摊还节省。

**P12 的核心科学问题**：在严格遵守比特币 80-byte Block Header 共识格式与 `Double-SHA256` 语义的前提下，是否存在超越经典 AsicBoost 的**广义中性比特团簇（Generalized Neutral Bit Clusters, GNBC）**与**辅助预条件 $\Phi$**，使得一个大小为 $2^d$（$d \ge 1$）的合法比特币区块头批量家族在经过 `Chunk2` 压缩（含 `W[16..]` 消息扩展）乃至跨入 `Hash2` 界面后，仍能共享显著比例的压缩轮门电路或 `Hash2` 计算门电路，从而带来**扣除经典 AsicBoost 基线后的真实净增量加速比**？

### 四项反幻觉铁律（开跑前强制锁定）
1. **严禁包装已知基线**：下列机制均为已知基线，**严禁**作为 P12 新发现计入加速比：
   - `Chunk1`（Header 前 64 字节）预计算与 $2^{32}$ Nonce 均摊；
   - 固定 Midstate 扫 `nNonce` (`W[3]`) 时跳过 `Chunk2` 的前 3 轮压缩（$t=0,1,2$）及 `W[16], W[17]`；
   - 经典 Overt（BIP320 `nVersion` 滚动）/ Covert（Merkle Root 尾 32 位 `W[0]` 碰撞）AsicBoost 复用 `Chunk2` 消息扩展 `W[16..63]`（及 `W[t]+K[t]` 预加）；
   - 已知尾部 3 轮 `H_7` 解耦（[PartialPreimagePoW.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/Core/PartialPreimagePoW.lean)）、4 轮 `backSolveA`（[BitVecIdentities.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/Core/BitVecIdentities.lean)）、单轮 `Ch` 位掩码（`r18_ch_masking_shield`）、V11 P3 的 28 条跨轮/跨块中间态观测恒等式（[P3 REPORT.md](file:///g:/SHA-LM/v11_discovery/p3_symbolic_lean4/REPORT.md)）以及 V11 P5 的 `t <= 6` 浅轮低进位瞬态（已在 $t=7$ 完全热化，[P5 REPORT.md](file:///g:/SHA-LM/v11_discovery/p5_btc_weak_midstate/REPORT.md)）。
2. **禁止中间态作弊**：真实比特币矿机仅输出最终 256-bit `Hash2` 摘要校验目标难度，任何依赖于“仅评估局部窗口而丢弃后续雪崩轮”的截断加速不得宣称为真实 Double-SHA256 矿机加速。
3. **条件筛选成本与概率惩罚强制计费**：若某中性团簇依赖辅助条件 $\Phi$（触发概率 $p_\Phi < 1$）或仅以概率 $p_{\text{neutral}} < 1$ 维持中性，则任何条件构造/预筛选开销与失效丢弃惩罚 $1/p_{\text{neutral}}$ **必须**计入有效单哈希摊还成本。
4. **本地 GPU 禁用与纯 IEEE-754 FP32 铁律**：本地仅允许 `CUDA_VISIBLE_DEVICES=""` CPU 轻量校验；全部 GPU/重型 SAT 搜索必须在 Colab `v12-p12` 上运行，浮点张量强制 `torch.float32` 与 `allow_tf32=False`，参考校验强制 `int64 & 0xFFFFFFFF` 与 `sha256_ref.py` 及 `hashlib.sha256` 严格逐比特对齐（`xor_diff == 0`）。

---

## 1. A1：比特币 80-byte Block Header 精确模型与逐门摊还计费器

### 1.1 真实 BTC 80-byte Block Header 三段式 Double-SHA256 语义

比特币区块头固定为 80 字节（640 bits），按大端 32-bit 字解析进入两次完整 SHA-256（共 3 次 64 轮压缩）：

| 阶段 | 名称 | 输入字布局（`BitVec 32`，大端序映射） | 初始状态 | 共识自由度 |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 0** | `Hash1-Chunk1`（64B） | `W0[0]`: `nVersion` (B0..3)<br>`W0[1..8]`: `hashPrevBlock` (B4..35)<br>`W0[9..15]`: `hashMerkleRoot[0..27]` (B36..63) | 标准 `SHA256_IV` | `nVersion`（BIP320 允许 bits 13..28 共 16-bit 自由滚动，协议研究亦考察全 32-bit）；`ExtraNonce` 变更 `hashMerkleRoot` |
| **Stage 1** | `Hash1-Chunk2`（320-bit 固定填充，64B） | `W1[0]`: `hashMerkleRoot[28..31]` (B64..67)<br>`W1[1]`: `nTime` (B68..71)<br>`W1[2]`: `nBits` (B72..75，难度固定)<br>`W1[3]`: `nNonce` (B76..79)<br>`W1[4] = 0x80000000`<br>`W1[5..14] = 0x00000000`<br>`W1[15] = 0x00000280` (`640`) | `midstate1` $= \text{Compress}(\text{IV}, W_0)$ | `W1[3]` (`nNonce`, 32-bit 全自由)；`W1[1]` (`nTime`, 秒级微调)；`W1[0]`（经 Merkle 树重算调整） |
| **Stage 2** | `Hash2`（256-bit 固定填充，64B） | `W2[0..7] = hash1[0..7]` $= (\text{midstate1}[i] + S^{(1)}_{64}[i]) \bmod 2^{32}$<br>`W2[8] = 0x80000000`<br>`W2[9..14] = 0x00000000`<br>`W2[15] = 0x00000100` (`256`) | 标准 `SHA256_IV` | 无直接独立控制；完全由 `(midstate1, W1[0..3])` 经 `Chunk2` 确定 |

> **术语对齐说明**：在矿机内循环中，`Chunk1` 由外循环生成 `midstate1`，内循环核心计算对象为 **`Chunk2`（320-bit 固定填充，64 轮） + `Hash2`（256-bit 固定填充，64 轮）**。文中讨论“跨块界面”时，既指 `Chunk1 → Chunk2` 的 `midstate1` 注入界面，也重点指 **`Chunk2 → Hash2`（即 `hash1[0..7] → W2[0..7]`）的跨块界面**。

### 1.2 逐门与逐轮活跃计算摊还计费模型

为防止任何“按轮数粗估”导致的误差，定义两级可机器审计的精确计费口径：

#### （A）算子级（Word-Level Macro-Op）精确计数
每轮压缩（$t \in [0, 63]$）包含：
- `T1`: 4 次 `ADD32`（`e_{t-3} + Sigma1(e_t) + Ch + K_t + W_t`，若 $W_t \equiv 0$ 则省 1 次 `ADD32`，若 `K_t + W_t` 可预计算/共享则 `K_t + W_t` 视为消息调度附带加法）、1 次 `Sigma1`、1 次 `Ch`；
- `T2`: 1 次 `ADD32`（`Sigma0(a_t) + Maj`）、1 次 `Sigma0`、1 次 `Maj`；
- 状态更新: 2 次 `ADD32`（`e_{t+1} = a_{t-3} + T1`, `a_{t+1} = T1 + T2`）。
每步消息扩展（$t \in [16, 63]$）包含：
- `W_t = sigma1(W_{t-2}) + W_{t-7} + sigma0(W_{t-15}) + W_{t-16}`：3 次 `ADD32`（**静态常量折叠规则**：若某加数在全部输入上恒为 `0`，如 `Chunk2` 中 `W[5..14]=0` 或 `Hash2` 中 `W2[9..14]=0`，则对应 `ADD32` 与 `sigma(0)` 恒折算为 **0 门**，严禁虚报零常量加法为有效节省！）。

#### （B）比特级门等效（Bit-Level Gate Equivalent, GE）与批量摊还公式
将所有 32-bit 算子展开为标准 2-输入逻辑门网络（`NAND2=1 GE` 口径或等效标准布尔门计数，取确定性固定权重表以便脚本与审计官 100% 复现）：
- `XOR2` = 3 GE，`AND2` / `OR2` = 2 GE，`NOT` = 1 GE；
- `FA`（1-bit 全加器，2 XOR + 2 AND + 1 OR）= 11 GE，`HA`（半加器）= 5 GE；
- 按位算子映射：
  - `ADD32(x, y)`：31 `FA` + 1 `HA` = **346 GE**（若一边为全零常数 = 0 GE；若一边为非零编译期常数如 `K[t]`/`IV[i]`/`0x80000000`/`640`/`256`，仅计 `HA`/常量化简门，约为常数折叠后有效门数）；
  - `Ch(e, f, g)`：32 × (2 `AND2` + 1 `NOT` + 1 `XOR2`) = **256 GE**（等效 32×`MUX2`）；
  - `Maj(a, b, c)`：32 × (3 `AND2` + 2 `XOR2`) = **384 GE**（或等价 2`XOR`+1`AND` 优化版 256 GE；规范统一采用固定常数表，**所有组与对照组使用完全同一张 GE 权重表**）；
  - `Sigma0/Sigma1/sigma0/sigma1`：各含 64 次 1-bit `XOR2` = **192 GE**（移位/循环移位在硬件连线中为 0 门）。

**批量摊还规则（Batch Size $M = 2^d$）**：
给定一个合法比特币区块头批量集合 $\mathcal{H} = \{H^{(0)}, H^{(1)}, \dots, H^{(M-1)}\}$（$M = 2^d$ 个互不相同的区块头）：
1. 对计算图 $G$ 中的每一个**比特级基本门 $g$**（或在算子级对每个 32-bit 原语 $op$）：
   - 若 $g$ 的全部输入导线在 $\mathcal{H}$ 上取完全相同的值（即**批量零差分不变门**：$\forall i, j \in [0, M-1],\; \text{in}_g(H^{(i)}) = \text{in}_g(H^{(j)})$），则 $g$ 只需计算 1 次，**单哈希摊还成本**为 $\frac{\text{GE}(g)}{M}$；
   - 若 $g$ 的任一输入导线在 $\mathcal{H}$ 上发生变化（**活跃门**），则 $g$ 必须对每个候选头重算，单哈希成本为 $\text{GE}(g)$。
2. **有效单哈希净成本**（含概率惩罚与条件开销）：
   $$\text{Cost}_{\text{eff}}(\mathcal{H}) = \frac{1}{p_{\text{valid}}} \left( \frac{\text{Cost}_{\text{setup}}(\Phi)}{M \cdot N_{\text{amort}}} + \sum_{g \in \text{Static}(\mathcal{H})} \frac{\text{GE}(g)}{M} + \sum_{g \in \text{Active}(\mathcal{H})} \text{GE}(g) \right)$$
   其中 $p_{\text{valid}} \in (0, 1]$ 为该批产生完整无误差合法 Double-SHA256 评估的成功率（对严格确定性中性团簇 $p_{\text{valid}} = 1$；对概率型中性集若未做补救则按丢弃重算惩罚 $1/p_{\text{neutral}}$）。

### 1.3 硬基线定义（A1 必复现校准）
`@p12-eng` 必须先精确计算并输出以下三大硬基线的 GE 与宏算子成本（以 **0-AsicBoost 标准 Nonce 扫频** 为 `1.000×` 参考）：
1. **`BASE_NO_ASICBOOST`（常规单 Midstate 扫 `nNonce`）**：
   - `Chunk1` 已按 $2^{32}$ 摊还（成本 $\approx 0$）；
   - `Chunk2` 跳过恒定不变的 `t=0,1,2` 与 `W1[16], W1[17]`（因为 `W1[3]` 直到 `t=3` 才进入压缩、直到 `t=18` 才进入消息扩展），需完整执行 `Chunk2` 的 `t=3..63`（61 轮）+ `W1[18..63]`（46 步，常数折叠后）+ `Hash2` 全部 64 轮 + `W2[16..63]`（48 步，常数折叠后）。
2. **`BASE_ASICBOOST_OVERT`（经典 BIP320 Overt AsicBoost，$k \in \{4, 8, 16\}$ 路复用）**：
   - 保持 `Chunk2` 全部 16 字 `W1[0..15]` 不变，通过 BIP320 `nVersion` 滚动生成 $k = 2^d$ 个 `midstate1`；
   - `Chunk2` 消息扩展 `W1[16..63]`（含 `W1[t]+K[t]`）完全零差分，摊还因子 $1/k$；`Chunk2` 压缩 `t=0..63` 及 `Hash2` 全部活跃；同时如实计入每个 `midstate1` 对应的 `Chunk1` 成本（按其各自扫过 $N_{\text{nonce}}$ 个 Nonce 摊还；若单组复用则按 `1/N_nonce` 计，若每个 `(nVersion, nNonce)` 组合动态切换则计 `Chunk1` 全摊还）。
   - 记录精确理论与实测节省比例 `save_overt_pct`（预期在 **~13% ~ 21%** 区间，取决于 `Chunk1` 是否全均摊与 `W+K` 归属；同时同步记录 Timo Hanke 原论文口径含 `Chunk1` 外循环时的 4-块对比口径 ~20%）。
3. **`BASE_ASICBOOST_COVERT`（经典 Covert AsicBoost，$k$-路 Merkle 尾字碰撞）**：
   - 同上共享 `Chunk2` 扩展器，额外显式扣除 Merkle Root 尾 32-bit 碰撞筛选的边际外循环开销。

> **净增益定义**：任何 P12 候选策略的**净增量节省**必须同时报告相对于 `BASE_NO_ASICBOOST` 的总节省 $\Delta_{\text{vs\_noab}}$，以及**相对于同批大小 $k=2^d$ 的 `BASE_ASICBOOST_OVERT` 的净增量节省 $\Delta_{\text{vs\_classic}}$**（只有 $\Delta_{\text{vs\_classic}} > 0$ 且统计/代数证真，才算突破经典 AsicBoost）。

---

## 2. A2：广义中性比特团簇（GNBC）与辅助条件 $\Phi$ 搜索

### 2.1 三大搜索轨道与合法自由度

在 Colab `v12-p12` GPU 上展开百万级并发搜索（`torch.float32`, `allow_tf32=False` + 精确 `int64` 仿真）：

1. **Track A2-1（`Chunk2` 内部中性团簇：利用 `W1[0..3]`、`W1[16]=W1[17]=0` 预锚定与 `midstate1` 自由度）**：
   - 搜索基底配置 $\Theta = (\text{midstate1}, W_1[0..2], W_1[3])$ 与辅助条件 $\Phi(\Theta)$（如局部进位屏蔽 `(x & y) == 0`、`Ch`/`Maj` 分支锁定、以及 P5 风格的 $W_1[16]=W_1[17]=0$ 代数预锚定）；
   - 搜索**多比特/字级扰动掩码簇** $\Delta = \{\delta_1, \dots, \delta_d\} \subset \text{AllowedDOF}$：
     - 模式 1：仅扰动 `W1[3]`（`nNonce`）的 $d$ 个比特（或跨 `W1[0..3]` 联合扰动 `nVersion`/`ExtraNonce`/`nTime`/`nNonce`）；
     - 模式 2：联合扰动 `midstate1` 与 `W1[0..3]`（广义跨块输入补偿：当 `nVersion` 或 `Merkle` 改变导致 `midstate1` 出现可控局部差分时，同步调整 `W1[0..3]` 抵消前 $R$ 轮压缩差分，或延迟雪崩至更深轮次！）；
     - 模式 3：利用 BIP320 `nVersion` 16-bit 自由度与 `MerkleRoot` 尾部探索 `Chunk1` + `Chunk2` 联合稀疏团簇。

2. **Track A2-2（`Chunk2` 尾部 $\to$ `Hash2` 跨块界面中性团簇）**：
   - 搜索是否存在广义中性比特（单比特 / $2^d$ 多比特线性子空间）与辅助条件 $\Phi$，使得扰动注入后经过 `Chunk2` 64 轮压缩，在输出 `hash1[0..7]`（即 `W2[0..7]`）上保持若干字或若干高位/低位比特零差分（局部概率 $p_{\text{neutral}} \gg 2^{-32k}$），从而穿透进入 `Hash2` 的压缩前 $r_2$ 轮及消息扩展 `W2[16..]`！
   - 同时直接考察在 `Hash2` 内部（`W2[8]=0x80000000, W2[9..14]=0, W2[15]=256`）对 `W2[0..7]` 施加受控多比特翻转时，`W2[16..63]` 与压缩前 $r_2$ 轮的真实中性穿透率（检验是否存在比单纯 `t=0..k` 跳过更强的位级局部碰撞）。

3. **Track A2-3（比特级进位吸收 / 条件静默门集群）**：
   - 不再要求“整个 32-bit 字零差分”，而是直接以**比特级门电路活跃率（Active GE Ratio）**为目标函数，搜索在 $2^d$（$d \in \{1, 2, 3, 4\}$）批量下能够锁死最多 `FA`/`Ch`/`Maj` 比特级门的输入掩码子空间与辅助条件 $\Phi$（进化搜索 + MCTS/神经先验引导 + 局部贪心位翻转）。

### 2.2 必须量化的五大核心指标

每个候选团簇（及其对应轨道）必须严格测量并输出：
1. **$R_{\text{neutral}}$（中性穿透轮数）**：从扰动首次进入状态/消息字起算，保持全字零差或特定比特掩码零差（活跃门率为 0 或低于显著阈值 $\tau = 0.05$）的最高轮次深度（区分 `Chunk2` 内部 $R_{\text{neutral}}^{(1)}$ 与 `Hash2` 内部 $R_{\text{neutral}}^{(2)}$）。
2. **$p_{\text{neutral}}$（条件中性概率）** 与 **$p_\Phi$（辅助条件触发概率）**：
   - 在**独立留出测试集**（$N_{\text{holdout}} \ge 262{,}144$ 个未参与搜索的新样本，与搜索池 $\text{overlap} = 0$）上测量；
   - 若声称中性概率高于随机基线 $2^{-b}$，必须报告精准二项分布 $z$-score 与 Wilson 95% 置信区间（亮灯阈值 $|z| \ge 6.0$ 且 $z_{\text{vs\_ctrl}} \ge 5.0$）。
3. **$d_{\text{neutral}}$（同时兼容中性维度）**：可同时开启互不破坏中性的自由比特维数（形成 $2^{d_{\text{neutral}}}$ 批量）。
4. **跨过 `W[16..]` 后的真实门电路摊还加速比**：
   - 必须分别报告：`Chunk2` 前 16 轮、`Chunk2` 扩展后（`t >= 16`）、`Hash2` 全程，以及**完整 BTC Double-SHA256 总计**的：
     - `active_ge_ratio`（活跃 GE 占比）
     - `speedup_vs_noab`（相对无 AsicBoost 基线）
     - `net_gain_vs_classic_asicboost`（**相对经典 AsicBoost 的净增量百分比**）。

---

## 3. A3：SAT/Z3 最大摊还界求解、谱隙对齐对照与 Lean 4 证真

### 3.1 SAT / Z3 精确最大摊还界求解

为避免“启发式搜索没找到不代表不存在”的盲区，必须使用 **Z3 (`BitVecSort(32)`)** 与 **PySAT (`CaDiCaL195`)** 对下列精确优化/判定问题给出数学界：

1. **界限问题 B1（`Chunk2` 单块带条件最大零差/零活跃门穿透界）**：
   - 编码两份 `Chunk2` 轨迹 $T, T'$（固定合法 BTC 填充 `W1[4]=0x80000000, W1[5..14]=0, W1[15]=640`）：
   - 场景 1（经典同 Midstate）：$\text{midstate1} = \text{midstate1}'$，仅允许 $W_1[0..3] \neq W_1'[0..3]$（或仅 $W_1[3] \neq W_1'[3]$）。求解：扰动进入压缩后，能保持后续所有轮 $t \in [t_{\text{inj}}+1, R]$ 上某输出字/全状态零差（$\Delta S_t = 0$）的最大 $R$！
   - 场景 2（广义变 Midstate + 补偿）：允许 $\text{midstate1} \neq \text{midstate1}'$（如仅变动若干 bit，对应真实 `Chunk1` 可达差异或自由 `midstate1` 上界松弛）且同步允许 $W_1[0..3]$ 自由选取。求解在满足输入非平凡（$\Delta \text{Header} \neq 0$）时，保持输出状态 $S_{R}$ 全零差（或至少保留 $k$ 个字零差）的最大轮数 $R$，以及跨入 $t \ge 16$ 后可保持零活跃加法的最大门数上界。
2. **界限问题 B2（`Hash2` 首字/前轮零差及跨块最大可摊还界）**：
   - 在 `Hash2` 中（`W2[8]=0x80000000, W2[9..14]=0, W2[15]=256`，初始态固定 `SHA256_IV`），给定 $\Delta W_2[0..7] \neq 0$，用 SAT/Z3 求解：
     - 若要使 `Hash2` 消息扩展 `W2[16..20]`（或前 $r_2$ 轮压缩状态）全部或部分保持零差，$\Delta W_2[0..7]$ 必须满足的最小固定零差字数 / 最小汉明重量约束；
     - 在 $R \in \{4, 6, 8, 10, 12\}$ 窗口上，允许辅助条件 $\Phi$ 的情况下，最大同时兼容中性维度 $d_{\text{neutral}}^{\max}$ 与最小活跃比特门比例下界（`UNSAT` 证死不可能低于该活跃门数）。
3. **界限问题 B3（突破性候选的 V1–V7 审计）**：
   若 A2 或 A3 搜出任何超越经典 AsicBoost 的候选恒等式/条件中性定理，必须 100% 通过 V11 P3 同款 **Z3 七项非平凡性审计（V1–V7）**：
   - V1 否定式 `UNSAT`；V2 $\Phi$ 与非退化性；V3 $\Phi$ 必要性；V4 掩码必要性；V5 非纯算术恒同；V6 逐轮 ARX 断链测试（活跃跨度 $\ge 5$ 轮，跨块项需断开拼接即 `SAT`）；V7 去重与基线隔离。

### 3.2 谱隙对齐随机对照组（`CTRL_RAND_ROTR_K`）

为区分 SHA-256 特定结构与一般 32-bit ARX 网络的普遍动力学性质，必须构造 **3 组独立种子** 的谱隙对齐随机对照密码 `CTRL_RAND_ROTR_K`：
- 保持 `Sigma0, Sigma1, sigma0, sigma1` 的 3-抽头不同旋转/右移索引结构（随机抽取满足互异且最小循环步距与 SHA-256 同阶的合法旋转三元组，确保线性扩散谱隙对齐），以及随机 32-bit 常数表 `K_ctrl[0..63]`、`IV_ctrl[0..7]`；
- 保持完全相同的 BTC 80-byte 分块结构与固定填充规则；
- 在 A1（经典 AsicBoost 校准）、A2（GPU 中性团簇搜索）与 A3（Z3/PySAT 精确界）上以**完全相同的预算与超参数**同步运行。

### 3.3 本地 Lean 4.36.0 `bv_decide` 证真约定

- 若实验发现任何**超越经典 AsicBoost 的新跨轮/跨块中性恒等式或条件摊还定理**（通过 Z3 V1–V7），`@p12-math` 将在 `g:\SHA-LM\v12_frontier\track_a_p12_amortized\lean_verify\` 下构建独立 Lean 4.36.0 工程（强制 `moreLeanArgs = ["-j", "2", "-M", "2048"]`，零 Mathlib），利用 `bv_decide` 完成零 `sorry` 机器证真与 `#print axioms` 审计；
- 若全部新突破候选均被证伪（或净增量节省为 0，由 SAT/Z3 给出精确上界证书），亦可在本地运行 Lean 4 验证核心门级算术不变式或明确记录 `lean_verify_triggered: false` 及原因，在 `REPORT.md` 与 `verdict.json` 中如实报告。

---

## 4. 工程实施与产物交付规范（`@p12-eng` $\to$ `@p12-math` & `@v12-auditor`）

### 4.1 目录与脚本建议（归 `@p12-eng` 所有）
建议在 `g:\SHA-LM\v12_frontier\track_a_p12_amortized\` 下组织：
- `btc_double_sha_engine.py`：纯 `int64` & FP32（`allow_tf32=False`）BTC 80-byte Double-SHA256 引擎、逐比特对拍单测（对比 `sha256_ref.py` 与 `hashlib.sha256`）及比特级/算子级摊还计费器；
- `classic_asicboost_calib.py`：A1 三大基线（`BASE_NO_ASICBOOST`、`BASE_ASICBOOST_OVERT`、`BASE_ASICBOOST_COVERT`）精确校准；
- `gnbc_gpu_search.py`：A2 在 Colab GPU (`v12-p12`) 上运行的三轨道广义中性团簇与辅助条件 $\Phi$ 搜索 + 独立留出集验证；
- `sat_amortized_bounds.py`：A3 基于 Z3 + PySAT (`CaDiCaL195`) 的精确最大零差轮数/最小活跃门界求解与 `CTRL_RAND_ROTR_K` 对照实验；
- `run_p12_colab.py`：Colab 一键分阶段无缓冲（`python -u`）编排入口。

### 4.2 强制输出 JSON Schema（落盘至 `results/`）

请将结果同步至 `g:\SHA-LM\v12_frontier\track_a_p12_amortized\results\`，至少包含以下 4 个核心 JSON 及完整日志：
1. `results/a1_calibration.json`：
   - `bit_exact_check`: `{"btc_genesis_xor_diff": 0, "random_headers_tested": N, "max_xor_diff": 0}`
   - `ge_weights`: 固定门权重字典
   - `baselines`: `BASE_NO_ASICBOOST`、`BASE_ASICBOOST_OVERT`（$k=4,8,16$）、`BASE_ASICBOOST_COVERT` 的精确 `active_ge`、`amortized_ge_per_hash`、`saving_pct_vs_noab` 及逐模块拆解（`chunk1`, `chunk2_exp`, `chunk2_comp`, `hash2_exp`, `hash2_comp`）。
2. `results/a2_gnbc_search.json`：
   - `search_budget`: GPU 型号、总测试候选数、搜索样本数、独立留出样本数（强制 `train_holdout_overlap == 0`）；
   - `tracks`: 分别记录 `A2-1`、`A2-2`、`A2-3` 在 `MAIN_SHA256` 与 `CTRL_RAND_ROTR_K`（3 种子均值/极值）上的 Top 候选：
     - `id`, `track`, `r_neutral`, `p_phi`, `p_neutral`, `p_neutral_z`, `z_vs_ctrl`, `d_neutral`, `batch_size`, `active_ge_per_hash`, `saving_pct_vs_noab`, `net_gain_pct_vs_classic_ab`, `witness_headers_hex`（如有声称有效团簇，必须附带可复现真实 80-byte 区块头样本列表）。
3. `results/a3_sat_bounds.json`：
   - `solver_info`: `z3` 版本、`pysat` (`CaDiCaL195`) 版本；
   - `bounds`: 针对 B1（场景 1/2）、B2（跨块零差与最小活跃门下界）在 `MAIN_SHA256` 与 `CTRL_RAND_ROTR_K` 上的精确求解结果（`status`: `SAT`/`UNSAT`, `max_zero_diff_rounds`, `min_active_gates`, `solve_seconds`）；
   - `new_identities`: 通过 V1–V7 的新候选恒等式列表（若无则为空列表 `[]`）。
4. `results/p12_run_meta.json` & `results/p12_colab.log`：
   - 包含 Colab 会话名 `v12-p12`、GPU 型号、`allow_tf32=False` 核验、各阶段耗时，以及本地停机后 `colab.exe sessions` 核实输出（`colab_stopped: true`）。

### 4.3 执行与停机纪律
- 完成全部代码编写并经 `@v12-auditor` 门禁审计后，在 Colab `v12-p12` 运行全量 A1+A2+A3；
- 将全部结果拉回 `g:\SHA-LM\v12_frontier\track_a_p12_amortized\results\` 后，**立即执行** `& g:\SHA-LM\\.colab_venv\Scripts\colab.exe stop -s v12-p12` 并核实无活跃会话；
- 随后在留言板 `@p12-math` 与 `@v12-auditor` 汇报关键实测指标与文件路径。
