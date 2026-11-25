# P9 数学与实验契约：ARX 多轮复合 AlphaTensor 式低次概率代理电路发现

> **工作目录**：`g:\SHA-LM\v11_discovery\p9_arx_proxy_circuit\`  
> **参考规范**：位序与参考实现严格对齐 [src/sha256_ref.py](file:///g:/SHA-LM/src/sha256_ref.py) 与 [cnf_cdcl.py](file:///g:/SHA-LM/v10_ouroboros64/ouroboros_core/cnf_cdcl.py)（32-bit 字统一采用 **LSB-first** 索引：$b=0$ 为最低有效位 LSB，$b=31$ 为最高有效位 MSB）。

---

## 1. 核心科学问题

将 $R \in \{2, 3, 4, 5, 6\}$ 轮复合 SHA-256 算子（含模加 `Add32`、字面旋转/移位线性算子 $\Sigma_0, \Sigma_1, \sigma_0, \sigma_1$、以及非线性布尔门 $\text{Ch}, \text{Maj}$ 与消息扩展 $W[16..]$）视为多输入布尔函数 / 多线性模算术张量。类比 AlphaTensor 在双线性张量分解中发现超越 Strassen 的低秩紧凑电路，探索在**均匀随机输入分布**下，是否存在此前未知的**低代数次数（$\deg_{\mathbb{F}_2} \le 3$ 或 $\le 4$）、少非线性门数（$M_{\text{AND}} \le 16$，严格低于精确电路）且保持统计显著偏置（$|p - 0.5| \gg 0$，$z \ge 6$）的概率代理电路（Probabilistic Proxy Circuit）或高精度进位截断逼近**。

---

## 2. 目标复合算子阶梯与输入分布（Operators & Domains）

为避免单轮常量折叠带来的伪信号，并同时覆盖压缩函数状态演化与消息扩展递推，定义三类目标算子家族：

### 2.1 Family A：压缩函数多轮复合算子（$R \in \{2, 3, 4, 5, 6\}$）
遵循 [src/sha256_ref.py](file:///g:/SHA-LM/src/sha256_ref.py#L283-L307) 的状态递推，每轮 $t \in \{0, \dots, R-1\}$：
$$
\begin{aligned}
T_{1,t} &= \left(e_{t-3} + \Sigma_1(e_t) + \text{Ch}(e_t, e_{t-1}, e_{t-2}) + K_t + W_t\right) \bmod 2^{32}, \\
T_{2,t} &= \left(\Sigma_0(a_t) + \text{Maj}(a_t, a_{t-1}, a_{t-2})\right) \bmod 2^{32}, \\
e_{t+1} &= \left(a_{t-3} + T_{1,t}\right) \bmod 2^{32}, \qquad
a_{t+1} &= \left(T_{1,t} + T_{2,t}\right) \bmod 2^{32}.
\end{aligned}
$$
其中 $\Sigma_0(x)=\text{ROTR}^{2}(x)\oplus\text{ROTR}^{13}(x)\oplus\text{ROTR}^{22}(x)$，$\Sigma_1(x)=\text{ROTR}^{6}(x)\oplus\text{ROTR}^{11}(x)\oplus\text{ROTR}^{25}(x)$。设置两个输入分布子域：
1. **A1（全均匀状态输入 Uniform-State，主实验域）**：初始状态 $(a_0, a_{-1}, a_{-2}, a_{-3}, e_0, e_{-1}, e_{-2}, e_{-3}) \sim U(\{0,1\}^{256})$，消息字 $W[0..R-1] \sim U(\{0,1\}^{32R})$。
2. **A2（标准 IV 锚定输入 IV-Anchored，辅助域）**：初始状态固定为 SHA-256 标准 `IV`，消息字 $W[0..R-1] \sim U(\{0,1\}^{32R})$。
   > **防退化警告**：当初始状态固定为 `IV` 时，$t=0$ 轮中 $T_{2,0}$ 为常量、$T_{1,0} = C_0 + W_0$ 为单变量加常量，导致 $R=1$ 及 $R=2$ 的局部比特存在可通过常量折叠直出的低次近似。因此 A2 所得任何信号均须经第 4 节严格增量检验，**严禁将 `IV` 常量折叠退化当成新结构**。

**目标输出位**：$y \in \{(a_R)_b,\; (e_R)_b\}$，重点扫频 $b \in \{0, 1, 7, 15, 23, 31\}$（覆盖 LSB $b=0$、1-bit 进位位 $b=1$、中段位 $b=7, 15, 23$ 与最高进位深位 $b=31$）。

### 2.2 Family B：消息扩展多轮复合算子 $W[16..]$（$R \in \{2, 3, 4, 5, 6\}$）
遵循 [src/sha256_ref.py](file:///g:/SHA-LM/src/sha256_ref.py#L233-L240)：
$$
W_t = \left(\sigma_1(W_{t-2}) + W_{t-7} + \sigma_0(W_{t-15}) + W_{t-16}\right) \bmod 2^{32}, \quad t \ge 16,
$$
其中 $\sigma_0(x)=\text{ROTR}^{7}(x)\oplus\text{ROTR}^{18}(x)\oplus\text{SHR}^{3}(x)$，$\sigma_1(x)=\text{ROTR}^{17}(x)\oplus\text{ROTR}^{19}(x)\oplus\text{SHR}^{10}(x)$。
- **输入分布**：$W[0..15] \sim U(\{0,1\}^{512})$ 独立均匀采样。
- **扩展深度阶梯**：
  - $R=2 \mapsto W_{17}$（两步级联扩展，含对 $W_{16}$ 的一次嵌套依赖）；
  - $R=3 \mapsto W_{18}$；$R=4 \mapsto W_{19}$；$R=5 \mapsto W_{20}$；$R=6 \mapsto W_{21}$。
  - 同时可选测试**压缩+扩展联合算子**：以 $W[0..15]$ 为自由输入、以 $W[16..16+R-1]$ 驱动压缩函数第 $16 \dots 16+R-1$ 轮（起点状态取均匀随机 $S_{16} \sim U(\{0,1\}^{256})$），探测跨“消息扩展 + 压缩”复合张量 $(a_{16+R})_b$ 与 $(e_{16+R})_b$。
- **目标输出位**：$(W_{15+R})_b$，$b \in \{0, 1, 7, 15, 23, 31\}$。

### 2.3 Family C：缩减字长精确验证切片（$w$-bit Mini-ARX，$w \in \{4, 8\}$）
为支持 Z3 精确模型计数（`#SAT`）与精确 $\mathbb{F}_2$ 代数正规形（ANF）上界校验，构造与 SHA-256 同构的 $w$-bit 缩减 ARX 算子：
- 当 $w=4$ 时：$\Sigma_0^{(4)}$ 取旋转 $(1, 2, 3)$，$\Sigma_1^{(4)}$ 取旋转 $(1, 2, 3)$（或不同位配对），$\sigma_0^{(4)}$ 含 `ROTR(1)^ROTR(3)^SHR(1)`，$\sigma_1^{(4)}$ 含 `ROTR(2)^ROTR(3)^SHR(1)`，保留 $\text{Ch}, \text{Maj}$ 与 $\bmod 2^4$ 加法；
- 当 $w=8$ 时：$\Sigma_0^{(8)}, \Sigma_1^{(8)}, \sigma_0^{(8)}, \sigma_1^{(8)}$ 取两两互异旋转/移位三元组，保留 $\bmod 2^8$ 连加；
- 此外对 32-bit 真实 SHA-256 的局部输出位 $(y)_b$，可定义输入比特子集截断切片（仅保留 $n_{\text{eff}} \le 16$ 个活跃高位/低位邻域变量，其余变量按均匀分布边缘化或固定为多组样本，在局部依赖锥上运行 Z3 精确计数）。

---

## 3. 代理电路语法、AlphaTensor 式张量表示与复杂度预算

### 3.1 基础布尔直线程序语法（XOR-AND Basis / Multiplicative Complexity）
代理电路 $C: \mathbb{F}_2^n \to \mathbb{F}_2$ 定义为直线程序（SLP），允许门基为 $\{\text{CONST}_1, \text{XOR}, \text{AND}\}$：
- 输入节点：$x_0, \dots, x_{n-1} \in \mathbb{F}_2$（对应当前算子依赖的所有状态比特与消息比特）。
- **线性投影层（免费 XOR 组合，受稀疏度约束）**：每个非线性乘法节点或最终输出可引用输入的任意仿射线性组合 $\ell(x) = c_0 \oplus \bigoplus_{j \in S} x_j$。为防止退化或不可解释的稠密全空间线性包，要求每个仿射线性形式支撑集大小 $|\text{supp}(\ell)| \le S_{\max} = 32$（且整个电路总线性输入连接数 / 2-input XOR 门数 $M_{\text{XOR}} \le 128$）。
- **非线性 AND 门**：$g_m = u_m \wedge v_m$，其中 $u_m, v_m$ 为常数、输入变量或先前门输出 $g_1, \dots, g_{m-1}$ 的仿射线性组合。
- **输出节点**：$\hat{y} = c_{\text{out}} \oplus \bigoplus_{j \in S_{\text{out}}} x_j \oplus \bigoplus_{m \in T_{\text{out}}} g_m$。

### 3.2 AlphaTensor 式低秩多线性张量分解特例
为兼顾矩阵/张量分解式搜索（MCTS/RL 或 Walsh-ANF 分解），支持深度受限的多线性多项式标准型：
$$
C(x) = \ell_0(x) \oplus \bigoplus_{r=1}^{M_{\text{AND}}} \prod_{k=1}^{d_r} \ell_{r,k}(x), \qquad 2 \le d_r \le d_{\max},
$$
其中每个乘积项 $\prod_{k=1}^{d_r} \ell_{r,k}(x)$ 消耗 $(d_r - 1)$ 个二元 $\text{AND}$ 门。注意该表示是通用 XOR-AND 电路的子集，能直接度量 $\mathbb{F}_2$ 代数次数 $\deg_{\mathbb{F}_2}(C) = \max_r d_r$。

### 3.3 复杂度预算与精确电路对比
精确单轮 SHA-256 压缩电路的最小已知非零非线性成本（见 [cnf_cdcl.py](file:///g:/SHA-LM/v10_ouroboros64/ouroboros_core/cnf_cdcl.py#L144-L256)）：
- 1 次 $\text{Ch}$：32 个 AND 门；1 次 $\text{Maj}$：32 个 AND 门（利用 $\text{Maj}(a,b,c)=c \oplus ((a\oplus c)\wedge(b\oplus c))$）；
- 1 次 32-bit 模加 `Add32`：31 个进位 AND 门；每轮压缩共 7 次 `Add32`（约 217 个 AND 门），每轮合计 **$M_{\text{exact}}^{(1)} \approx 281$ 个 AND 门**；
- 每步消息扩展 $W_t$（$t \ge 16$）：3 次 `Add32`，共 **$M_{\text{exact}}^{(W)} = 93$ 个 AND 门**。
- 因此对于 $R \in [2, 6]$ 轮精确算子，$M_{\text{exact}} \in [562, 1686]$（压缩）或 $[186, 558]$（扩展），且代数次数随轮数迅速饱和。

**代理电路硬预算约束**：
1. **主预算（Strict Low-Degree Compact）**：
   - 代数次数上限：$\deg_{\mathbb{F}_2}(C) \le 3$；
   - 非线性 AND 门上限：$M_{\text{AND}}(C) \in \{0, 1, 2, 4, 8\}$（与步长递增搜索）。
2. **扩展预算（Outer Envelope）**：
   - 代数次数上限：$\deg_{\mathbb{F}_2}(C) \le 4$；
   - 非线性 AND 门硬上限：$M_{\text{AND}}(C) \le 16$（严格低于任何 $R \ge 2$ 精确电路的 $1/10$ 以下）。
3. **显式进位截断对照家族（Carry-Truncation Proxies）**：
   - 构造进位深度限制为 $k_{\text{carry}} \in \{1, 2, 3\}$ 的截断电路（仅传播距离目标比特 $b$ 不超过 $k_{\text{carry}}$ 位的进位链，或每轮内部仅保留局部高相关性进位门并裁剪至 $M_{\text{AND}} \le 16, \deg \le 4$），作为“进位截断逼近”专门对比基线。

---

## 4. 两重反平凡性检验（Anti-Triviality Safeguards）

任何宣称发现“非平凡代理电路”的候选 $C^*$，**必须同时通过以下两重防线**；未通过者一律归为平凡近似或通用 ARX 边界效应，绝不可宣称为突破。

### 4.1 第一重：相对已知近似基线库 $\mathcal{B}_{\text{known}}$ 的严格增量检验
必须显式构造并在相同评测集上运行以下已知基线库 $\mathcal{B}_{\text{known}}$：
1. **B1（模加 LSB 恒等近似）**：所有 `Add32` 替换为其 LSB 线性关系 $(x+y)_0 = x_0 \oplus y_0$，以及固定 `IV` 折叠后的首轮线性关系。
2. **B2（单门进位截断线性/二次基线）**：
   - $(x+y)_b \approx x_b \oplus y_b$（单加法器忽略进位，均匀输入下对 $b=1$ 偏置 $|p-0.5|=0.25$，对一般 $b$ 偏置 $2^{-(b+1)}$）；
   - $(x+y)_b \approx x_b \oplus y_b \oplus (x_{b-1} \wedge y_{b-1})$（1-bit 局部进位修正，1 个 AND 门，偏置显著高于 0-bit 忽略进位）。
3. **B3（单门 $\text{Ch}/\text{Maj}$ 已知近似）**：
   - $\text{Ch}(e,f,g) \approx g$ 或 $f$（线性近似，偏置 $0.25$），以及精确表达 $g \oplus (e \wedge (f \oplus g))$（1 个 AND 门）；
   - $\text{Maj}(a,b,c) \approx a$ 或 $a \oplus b \oplus c$（线性近似，$|p-0.5|=0.25$），以及精确表达 $c \oplus ((a\oplus c) \wedge (b\oplus c))$（1 个 AND 门）。
4. **B4（贪心多轮已知基线复合 Greedy Known Composition）**：
   - 在目标算子的精确表达式图上，将每步 `Add32` 替换为 $k$-carry 截断（$k \in \{0, 1, 2\}$，在总 AND 预算 $M_{\text{AND}} \le 16, \deg \le 4$ 内），并将 $\text{Ch}/\text{Maj}$ 替换为 B3 的已知近似，枚举所有有限组合中在**搜索集**表现最优的已知基线代理 $B^* \in \mathcal{B}_{\text{known}}$。

**增量判定准则**：
记目标代理电路 $C^*$ 在独立留出集上的偏置为 $\varepsilon(C^*) = |p(C^*) - 0.5|$，已知基线库最优成员的留出偏置为 $\varepsilon(B^*) = |p(B^*) - 0.5|$。定义：
- 增量偏置：$\Delta \varepsilon = \varepsilon(C^*) - \varepsilon(B^*)$；
- 两比例差的标准误：$\sigma_{\text{diff}} = \sqrt{\frac{p(1-p) + p_B(1-p_B)}{N_{\text{eval}}}}$（或对同一留出样本配对 McNemar 检验标准误 $\sigma_{\text{paired}}$），检验统计量 $z_{\text{inc}} = \frac{\Delta \varepsilon}{\sigma_{\text{paired}}}$。
- **通过条件**：必须满足
  $$\Delta \varepsilon > 0.005 \quad \text{且} \quad z_{\text{inc}} \ge 6.0,$$
  **并且**通过 Z3 验证 $C^*$ 与所有 $B \in \mathcal{B}_{\text{known}}$ 在布尔功能上**不代数等价**（$C^* \not\equiv B$，且在消除常数折叠与死变量后仍非任何 $B \in \mathcal{B}_{\text{known}}$ 的子式）。

### 4.2 第二重：同结构随机 ARX 对照组（Random-ARX Structural Control）
构造同结构随机对照算子家族 $\text{SHA-256}_{\text{rand}}^{(s)}$（独立随机种子 $s \in \{1, 2, 3\}$）：
- 保持模加次数、$\text{Ch}/\text{Maj}$ 拓扑连接完全不变；
- 替换 $\Sigma_0, \Sigma_1, \sigma_0, \sigma_1$ 的旋转与移位偏移量为随机互异合法整数（$\text{ROTR}$ 取 $\{1..31\}$ 中互异 3 元组，$\text{SHR}$ 取 $\{1..15\}$）；
- 替换轮常数 $K[t]$ 为独立均匀随机 32-bit 常数 $K'^{(s)}[t] \sim U(\{0,1\}^{32})$。

对每个 $(Family, R, b)$ 配置，使用**完全相同的搜索算法、迭代预算、复杂度上限和样本量**同时在真实 SHA-256 与 3 个随机 ARX 对照组上运行搜索，记录：
$$\varepsilon_{\text{ctrl}}^{\max} = \max_{s \in \{1,2,3\}} \varepsilon(\text{SHA-256}_{\text{rand}}^{(s)}), \qquad z_{\text{vs\_ctrl}} = \frac{\varepsilon(C^*) - \varepsilon_{\text{ctrl}}^{\max}}{\sigma_{\text{diff}}}.$$
- 若真实 SHA-256 出现高偏置但 $\varepsilon_{\text{ctrl}}^{\max}$ 处于同一水平（$z_{\text{vs\_ctrl}} < 3.0$），则该现象属于**一般 ARX 模加拓扑固有近似性**，而非 SHA-256 特异结构弱点，结案报告必须如实区分。

---

## 5. 大样本留出验证与 Z3 精确校验协议

### 5.1 搜索集与独立留出集隔离（Anti-Overfitting Protocol）
1. **搜索期（Search Split）**：用于引导 MCTS/RL、可微松弛或 Walsh-ANF 谱分解，样本量由搜索算法需要决定（如 $N_{\text{search}} \in [2^{15}, 2^{20}]$），**严禁将搜索集准确率直接用于最终宣称**。
2. **离散化冻结**：所有连续松弛（如可微门权重 $\sigma(\alpha) \in (0,1)$）必须严格硬阈值离散化为纯布尔 $\{\text{CONST}_1, \text{XOR}, \text{AND}\}$ 电路后，再做任何留出验证。
3. **独立留出集（Held-Out Evaluation Split）**：
   - 使用与搜索期完全独立的随机种子，生成至少 **$N_{\text{eval}} = 2^{20} = 1,048,576$** 个样本（对所有初筛 $z \ge 4.0$ 的候选电路，强制扩增至 **$N_{\text{eval}} = 2^{22} = 4,194,304$** 样本）；
   - 理论零假设标准误：$\sigma_0 = \frac{1}{2\sqrt{N_{\text{eval}}}}$（在 $N_{\text{eval}}=2^{20}$ 时 $\sigma_0 \approx 4.883 \times 10^{-4}$；在 $N_{\text{eval}}=2^{22}$ 时 $\sigma_0 \approx 2.441 \times 10^{-4}$）；
   - 必须计算并输出：
     - 正确率 $p$、偏置 $\varepsilon = |p - 0.5|$、相关系数 $r = 2p - 1$；
     - 原始统计量 $z = \frac{|p - 0.5|}{\sigma_0}$；
     - 99.9% Wilson 置信区间；
     - **多重比较校正**：设总探索候选评估次数（有效独立评估拓扑/门组合数）为 $N_{\text{trials}}$，报告 Bonferroni 校正等效阈值 $z_{\text{Bonf}} = \Phi^{-1}\left(1 - \frac{10^{-6}}{2 N_{\text{trials}}}\right)$（或校正后 $p_{\text{adj}} = \min(1, 2 N_{\text{trials}} \cdot \Phi(-z))$）。仅当独立留出集上同时满足 $z \ge 6.0$ 且 $p_{\text{adj}} < 10^{-4}$ 时，方认定统计显著。

### 5.2 Z3 精确计数、复杂度核验与反例审计
工程实现必须在 `z3-solver` 中完成以下三项形式化校验：
1. **电路规范性与复杂度自检**：
   - 验证输出电路为无环 DAG；
   - 验证总 AND 门数 $\le M_{\max}$；
   - 对每个电路在局部支持变量集上计算精确 ANF（Möbius 变换）或利用 Z3/多项式展开验证 $\deg_{\mathbb{F}_2}(C^*) \le d_{\max}$。
2. **小切片 Z3 精确计数一致性校验（Exact `#SAT` vs. Monte Carlo）**：
   - 在 Family C（$w=4$ 的 $R \in \{2,3,4\}$ 轮算子，或局部 $n_{\text{eff}} \le 16$ 比特切片）上，使用 Z3 阻塞子句计数（或精确布尔枚举）算出目标算子 $f$ 与候选电路 $C^*$ 的**精确吻合度**：
     $$p_{\text{exact}} = \frac{\#\{x \in \{0,1\}^{n_{\text{eff}}} : C^*(x) = f(x)\}}{2^{n_{\text{eff}}}},$$
   - 校验 $|p_{\text{MC}} - p_{\text{exact}}| < 4\sigma_0$，确认参考实现、位序（LSB-first）与代理求值引擎完全零 bug。
3. **Z3 精确不等价性与截断上界检验**：
   - 对任何通过留出检验的候选电路 $C^*$，构建 Z3 命题 $\exists x,\; C^*(x) \neq B(x)$（遍历 $\mathcal{B}_{\text{known}}$ 中所有相关已知基线 $B$）：
     - 若返回 `UNSAT`，直接判定该候选与已知基线逻辑恒等（平凡），予以驳回；
     - 若返回 `SAT`，记录具体反例输入 $x_{\text{cex}}$ 与差集频率估算；
   - 对声称“进位截断逼近”的公式，用 Z3 精确校验截断进位链在局部条件下的误差边界。

---

## 6. 工程交付契约（面向 `@p9-eng`）

请 `@p9-eng` 在 `g:\SHA-LM\v11_discovery\p9_arx_proxy_circuit\` 下实施并同步以下结果文件至 `g:\SHA-LM\v11_discovery\p9_arx_proxy_circuit\results\`：

1. **资源与精度硬约束**：
   - 严禁调用本地 GPU（本地冒烟必须 `CUDA_VISIBLE_DEVICES=""`）；
   - Colab 会话名 `v11-p9`（优先 `--gpu T4`，配额紧张退 `--high-mem` CPU），总预算 $\le 45$ 分钟；
   - 若使用 PyTorch，强制纯 FP32（`allow_tf32=False`, `torch.float32`），但核心布尔/比特级运算与大样本统计须基于精确整数位运算（`int64`/`uint32`）防止浮点舍入污染。
2. **推荐搜索算法组合**（保证覆盖 AlphaTensor 式离散分解与连续逼近）：
   - **方法 1（Walsh-Hadamard 谱 + 贪心低次 ANF / AlphaTensor 秩分解）**：利用大样本（或切片精确表）估计高相关性线性掩码与低次多线性特征（$\deg \le 3, 4$），迭代匹配残差；
   - **方法 2（MCTS / 演化直线程序 SLP 搜索）**：在预算 $M_{\text{AND}} \in \{0,1,2,4,8,16\}$、$\deg \le 4$ 下搜索稀疏线性投影 + AND 门组合；
   - **方法 3（可微布尔门松弛搜索，如直通估计器 STE / Gumbel-Softmax XOR-AND 网络）**：在 GPU 上并行搜索多组候选电路后硬离散化。
3. **结果文件 Schema（必须全部落盘到 `results/`）**：
   - `results/search_summary.json`：包含每个 $(Family, R, target\_bit)$ 的：
     - `exact_and_gates`（精确电路 AND 门数）
     - `known_baseline`（$\mathcal{B}_{\text{known}}$ 最优结果：`name`, `and_gates`, `degree`, `eval_p`, `eval_bias`, `z_score`）
     - `best_proxy`（最优代理电路：`and_gates`, `xor_gates`, `degree`, `n_eval`, `eval_p`, `eval_bias`, `z_score`, `wilson_ci999`, `p_adj_bonferroni`, `n_trials`）
     - `incremental_vs_known`（`delta_bias`, `z_paired`, `passed_known_check`）
     - `random_arx_control`（3 个随机 ARX 种子各自 `eval_bias`、`max_ctrl_bias`、`z_vs_ctrl`）
   - `results/top_circuits.json`：每个配置 Top-3 候选电路的完整 DAG 门列表与可读 $\mathbb{F}_2$ 代数式。
   - `results/z3_verification.json`：Z3 精确计数（Family C 切片 $p_{\text{exact}}$ vs $p_{\text{MC}}$）、$\deg_{\mathbb{F}_2}$/AND 门数核验、以及相对 $\mathcal{B}_{\text{known}}$ 的等价性检验（`SAT`/`UNSAT`）。
   - `results/run_log.txt`：Colab 全程无缓冲日志。

---

## 7. 最终裁决标准（供 `@p9-math` 结案使用）

- **`HIT`**：至少存在一个非退化配置（$R \ge 2$，且非固定 `IV` 首轮折叠退化），同时满足：
  1. $M_{\text{AND}} \le 16$ 且 $\deg_{\mathbb{F}_2} \le 4$（严格低于精确电路）；
  2. 独立留出集（$N_{\text{eval}} \ge 2^{20}$）上 $z \ge 6.0$ 且多重检验校正后 $p_{\text{adj}} < 10^{-4}$；
  3. 严格通过第一重反平凡性检验（$\Delta\varepsilon > 0.005$、$z_{\text{inc}} \ge 6.0$ 且 Z3 验证与 $\mathcal{B}_{\text{known}}$ 不等价）。
  > 若相对随机 ARX 对照组也满足 $z_{\text{vs\_ctrl}} \ge 3.0$，标注为「SHA-256 特异低次代理 HIT」；否则标注为「通用 ARX 边界低次结构 HIT」。
- **`BORDERLINE`**：存在满足 $3.0 \le z < 6.0$ 或 $\Delta\varepsilon \le 0.005$ 的微弱增量信号（或仅在固定 IV 的 $R=2$ 边界边缘微弱超越但无法跨过 $R \ge 3$）。
- **`NULL`**：在全部 $R \in \{2,3,4,5,6\}$ 的真实 SHA-256 压缩与消息扩展算子上，均未发现超越已知基线 $\mathcal{B}_{\text{known}}$ 且通过 $z \ge 6.0$ 留出验证的紧凑低次代理电路。
