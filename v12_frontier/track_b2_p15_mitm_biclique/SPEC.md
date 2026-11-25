# B2-P15 数学与形式化设计契约：截断前导零专用 MitM / Biclique / Splice-and-Cut 自动搜索与 Lean 4 证真

**项目目录**：`g:\SHA-LM\v12_frontier\track_b2_p15_mitm_biclique\`  
**数学与 Lean 4 负责人**：`@p15-math`  
**Colab MILP/CP-SAT 工程师**：`@p15-eng`（专属云端会话 `v12-p15`）  
**独立审计官**：`@v12-auditor`  
**隔离边界**：严禁修改 `v11_discovery/`、`v10_ouroboros64/`、`runs/`、`formal_proofs/`、`formal_proofs_v10/`、`v6a/`、`src/`；`@p15-math` 仅维护 `SPEC.md`、`REPORT.md`、`verdict.json` 与 `lean_verify/`，`@p15-eng` 仅维护 `.py` 与 `results/`。

---

## 0. 精确语义与 BTC Double-SHA256 128 轮拓扑定义

所有 32 位字均严格对齐 [sha256_ref.py](file:///g:/SHA-LM/src/sha256_ref.py) 与 [Primitives.lean](file:///g:/SHA-LM/v11_discovery/p3_symbolic_lean4/lean_verify/P3Verify/Primitives.lean) 的大端 `uint32`（`BitVec 32`）语义，字内比特按 LSB-first 索引 $b \in \{0, \dots, 31\}$（第 31 位为最高位 MSB）。

### 0.1 标准单块 SHA-256 压缩与消息扩展（$R \le 64$ 轮）
- **状态表示**：第 $t$ 轮前状态 $S_t = (A_t, A_{t-1}, A_{t-2}, A_{t-3}, E_t, E_{t-1}, E_{t-2}, E_{t-3}) \in (\mathbb{Z}_{2^{32}})^8$，对应 $(a_t, b_t, c_t, d_t, e_t, f_t, g_t, h_t)$。
- **单轮前向转移** $S_{t+1} = \mathrm{sha256Round}(S_t, K_t, W_t)$：
  $$T_1(t) = E_{t-3} + \Sigma_1(E_t) + \mathrm{Ch}(E_t, E_{t-1}, E_{t-2}) + K_t + W_t \pmod{2^{32}}$$
  $$T_2(t) = \Sigma_0(A_t) + \mathrm{Maj}(A_t, A_{t-1}, A_{t-2}) \pmod{2^{32}}$$
  $$A_{t+1} = T_1(t) + T_2(t) \pmod{2^{32}}, \qquad E_{t+1} = A_{t-3} + T_1(t) \pmod{2^{32}}$$
- **单轮后向转移** $S_t = \mathrm{sha256Round}^{-1}(S_{t+1}, K_t, W_t)$：
  - **无 $W_t$ 旁路还原 `backSolveA`**：
    $$A_{t-3} = (E_{t+1} - A_{t+1}) + \Sigma_0(A_t) + \mathrm{Maj}(A_t, A_{t-1}, A_{t-2}) \pmod{2^{32}}$$
    该公式严格**不依赖** $W_t$、$K_t$ 与 $E_{t-3}$。
  - **依赖 $W_t$ 的后向还原**：
    $$E_{t-3} = \bigl(A_{t+1} - \Sigma_0(A_t) - \mathrm{Maj}(A_t, A_{t-1}, A_{t-2})\bigr) - \Sigma_1(E_t) - \mathrm{Ch}(E_t, E_{t-1}, E_{t-2}) - K_t - W_t \pmod{2^{32}}$$
- **非线性消息扩展**（$t \ge 16$）：
  $$W_t = \sigma_1(W_{t-2}) + W_{t-7} + \sigma_0(W_{t-15}) + W_{t-16} \pmod{2^{32}}$$
  注意：$W_{t-16}$ 以系数 $+1$ 线性出现，故可双向无损递推：
  $$W_{t-16} = W_t - \sigma_1(W_{t-2}) - W_{t-7} - \sigma_0(W_{t-15}) \pmod{2^{32}}$$
  任意连续 16 个消息字 $W_{s..s+15}$ 唯一决定整条 $W_{0..63}$。
- **Davies-Meyer 前馈输出**：$H = (S_R + S_0) \bmod 2^{32}$。

### 0.2 Bitcoin 80-Byte Header Double-SHA256（128 轮 + 固定填充 + 截断前导零）
比特币 80 字节区块头在第一块（64 字节）预计算后得到 **Midstate** $S_{\text{mid}} \in (\mathbb{Z}_{2^{32}})^8$，随后的核心挖矿双块共 **128 轮**，由以下两阶段严格串联：

1. **阶段 A（Block 1b，全局轮次 $t \in [0, 63]$，初态 $S_0 = S_{\text{mid}}$，320-bit 固定填充）**：
   - 可控输入字 $W[0..3]$（128 bit：MerkleRoot 末字、nTime、nBits、nNonce；若锁定区块头仅搜 Nonce，则 $W[0..2]$ 固定为 Gray、仅 $W[3]$ 自由；在搜索广义独立块时允许 $W[0..3]$ 与可选 $S_{\text{mid}}$ 自由）。
   - **固定 320-bit 尾部填充**：
     $$W[4] = \text{0x80000000},\quad W[5..14] = \text{0x00000000},\quad W[15] = 640 \;(\text{0x00000280})$$
   - **稀疏折叠推论**（仅允许被 Z3 显式证真者）：$W[16] = \sigma_0(W[1]) + W[0]$、$W[17] = \sigma_1(640) + \sigma_0(W[2]) + W[1]$（均与 $W[3]$ 无关），$W[18]$ 首次引入 $\sigma_0(W[3])$。
   - **阶段 A 摘要**：$H^{(1)} = (S_{64} + S_{\text{mid}}) \bmod 2^{32}$。

2. **阶段 B（Block 2，局域轮次 $t' \in [0, 63]$ / 全局轮次 $64..127$，初态 $S'_0 = \text{stdIV}$）**：
   - **双块拼接输入**：$W'[0..7] = H^{(1)}[0..7]$；
   - **固定 256-bit 尾部填充**：
     $$W'[8] = \text{0x80000000},\quad W'[9..14] = \text{0x00000000},\quad W'[15] = 256 \;(\text{0x00000100})$$
   - **稀疏折叠推论**（参考 [Primitives.lean#L311-L328](file:///g:/SHA-LM/v11_discovery/p3_symbolic_lean4/lean_verify/P3Verify/Primitives.lean#L311-L328)）：
     $W'[16] = \sigma_0(W'[1]) + W'[0]$、$W'[17] = \sigma_1(256) + \sigma_0(W'[2]) + W'[1]$、$W'[18] = \sigma_1(W'[16]) + \sigma_0(W'[3]) + W'[2]$，前三项均与 $W'[4..7]$（即 Block 1b 的 $E$-轨输出）解耦。
   - **最终输出与截断前导零目标**：
     $$H^{(2)} = (S'_{64} + \text{stdIV}) \bmod 2^{32}$$
     截断前导零条件 $\mathrm{clz}(H^{(2)}[0]) \ge k$（$1 \le k \le 32$）或跨字高 $k$ 位全零 $\mathrm{clz}_{256}(H^{(2)}) \ge k$（$1 \le k \le 256$），以及 BTC 小端字节序高 $k$ 位全零 $\mathrm{ctz\_btc}(H^{(2)}) \ge k$（等价于从 $H^{(2)}[7]$ 低字节向高字大端对齐的前 $k$ 比特为 `0`）。

---

## 1. B2-1：通用 MitM / Splice-and-Cut / Biclique 四色 MILP/CP-SAT 模型

### 1.1 四色代数依赖编码
对计算图（状态字 $A_t, E_t$、轮中间量 $T_1(t), T_2(t)$、消息字 $W_t$、输出字 $H_j$）中每个变量 $x$（字级模型为每字 1 节点；位级模型为每字 32 比特节点 $x[b], b \in \{0..31\}$），引入布尔双指标 $(r_x, b_x) \in \{0,1\}^2$ 表示其是否对 **Red 中性变量集 $V_{\text{red}}$** 与 **Blue 中性变量集 $V_{\text{blue}}$** 存在潜在依赖（$V_{\text{red}} \cap V_{\text{blue}} = \emptyset$）：
- **Gray (`G`)**：$(r_x, b_x) = (0, 0)$，固定常量或仅依赖于固定基准参数的变量；
- **Red (`R`)**：$(r_x, b_x) = (1, 0)$，仅依赖于 Gray + $V_{\text{red}}$，对 $V_{\text{blue}}$ 严格非干涉；
- **Blue (`B`)**：$(r_x, b_x) = (0, 1)$，仅依赖于 Gray + $V_{\text{blue}}$，对 $V_{\text{red}}$ 严格非干涉；
- **White (`W`)**：$(r_x, b_x) = (1, 1)$，同时受 $V_{\text{red}}$ 与 $V_{\text{blue}}$ 污染（混合态，不可直接用于跨块独立求值）。

### 1.2 精确传播约束（严禁遗漏非线性与消息扩展）
对任意确定性算子 $y = f(x_1, \dots, x_m)$（涵盖 `ADD32`, `SUB32`, `Ch`, `Maj`, `Sigma0`, `Sigma1`, `sigma0`, `sigma1`）：
1. **标准单调污染规则**：
   $$r_y \ge r_{x_i},\quad b_y \ge b_{x_i} \quad (\forall i \in \{1..m\})$$
   - 在 **位级（Tier 2）** 模型中，`ADD32` 额外必须满足逐比特上行进位传递：
     $$r_{y, b} \ge r_{y, b-1},\quad b_{y, b} \ge b_{y, b-1} \quad (\forall b = 1..31)$$
     `rotr32(x, n)` 按位置换 $b \mapsto (b - n) \bmod 32$ 传递，`shr32(x, n)` 在最高 $n$ 位保持常量 `Gray(0,0)`。
2. **压缩方向选择与 `backSolveA` 旁路规则**：
   - 每个轮次 $t$ 的状态转移可配置为 **前向模式**（由 $S_t, W_t \to S_{t+1}$）或 **后向模式**（由 $S_{t+1}, W_t \to S_t$），在 Initial Structure / Biclique 内部可设独立方向边界。
   - **后向 `backSolveA` 精确旁路**：在后向计算第 $t$ 轮 $A_{t-3}$ 时，允许直接按 $A_{t-3} = (E_{t+1} - A_{t+1}) + \Sigma_0(A_t) + \mathrm{Maj}(A_t, A_{t-1}, A_{t-2})$ 传播依赖（**不引入 $W_t$ 与 $E_{t-3}$ 的颜色**）；但计算 $E_{t-3}$ 时**必须**同时继承 $S_{t+1}$ 与 $W_t$ 的颜色！
3. **消息扩展双向基与非线性耦合**：
   - 消息空间通过选择一个基指示集 $\{z_t \in \{0,1\}\}_{t=0}^{R-1}$（对于单块 64 轮，须满足有效自由基底满足扩展生成约束，如选取连续 16 字窗口 $W_{s..s+15}$ 作为自由基，或采用线性独立吸收基并在 Z3 验证无矛盾）。
   - 对于非基消息字 $W_t$：
     - 若处于前向扩展区（$t \ge 16$，采用前向递推）：$(r_{W_t}, b_{W_t})$ 必须同时覆盖 $\{W_{t-2}, W_{t-7}, W_{t-15}, W_{t-16}\}$ 中所有**非恒定（non-const）**依赖字（例如 BTC 固定填充下 $W[5..14]=0$ 为已知常量 Gray，且若 $W_{t-2}=0$ 则 $\sigma_1(0)=0$ 保持 Gray）；
     - 若处于后向扩展区（采用逆推 $W_{t-16} = W_t - \sigma_1(W_{t-2}) - W_{t-7} - \sigma_0(W_{t-15})$）：$(r_{W_{t-16}}, b_{W_{t-16}})$ 必须同时覆盖 $\{W_t, W_{t-2}, W_{t-7}, W_{t-15}\}$。
   - **BTC 跨块双段耦合约束**：
     - Block 1b 的 $W[4..15]$ 强制锁定为 Gray 常量；Block 2 的 $W'[8..15]$ 强制锁定为 Gray 常量；
     - 拼接处 $W'[j] = (H^{(1)}[j]) = S_{64}[j] + S_{\text{mid}}[j]$（$j=0..7$）必须严格继承 $S_{64}[j]$ 与 $S_{\text{mid}}[j]$ 的颜色！
     - 唯一允许跨色合并但不污染的条件：若引入已知的 **P3 形式化非干涉/消元重写规则**（如 `p3_id_014`：$S'_8$ 经 `backSolveA` 仅依赖 $W'[0..3]$ 而与 $W'[4..7]$ 无关），必须在图中显式标记并在 Stage 2 接受 **Z3 `BitVec(32)` 独立性证死**！

### 1.3 Initial Structure / Biclique 自由度分配与部分匹配
模型允许定义以下结构参数：
- **Biclique / Initial Structure 区间** $[t_{\text{bic}}, t_{\text{bic}} + L_{\text{bic}} - 1]$（$0 \le L_{\text{bic}} \le 12$）：
  - 在**独立 Biclique**（Independent Biclique，Khovratovich 等 FSE 2012）模式下，由 $V_{\text{red}}$ 引起的差分轨迹 $\Delta^{\text{red}}$ 与由 $V_{\text{blue}}$ 引起的差分轨迹 $\Delta^{\text{blue}}$ 在该 $L_{\text{bic}}$ 轮区间内**无活跃非线性门重叠**（可在字级或位级 MILP 中建模为活跃门掩码不相交，或仅限于线性加性注入），从而只需 $2^{d_1} + 2^{d_2}$ 次 $L_{\text{bic}}$-轮计算即可生成全部 $2^{d_1 + d_2}$ 个端点对；
  - 若存在非线性重叠但两端仍可由单边自由变量构造，按**非独立构造代价** $2^{d_1 + d_2} \cdot (L_{\text{bic}}/R_{\text{total}})$ 计费（严禁按 $2^{d_1} + 2^{d_2}$ 虚报）。
- **中性自由度分配**：
  - $d_1 = \sum_{v \in V_{\text{red}}} w(v)$ bits（Red 独立自由度，$d_1 \ge 1$），$d_2 = \sum_{v \in V_{\text{blue}}} w(v)$ bits（Blue 独立自由度，$d_2 \ge 1$）。
  - 辅助条件概率扣减：若部分匹配或中性传播依赖外部位条件 $\Phi$（如零进位条件 $(x \land y)=0$ 或固定差分屏蔽），必须显式扣减条件熵 $\kappa_\Phi = \log_2(1/p_\Phi) \ge 0$。
- **匹配点 $t_{\text{match}}$ 与部分匹配比特数 $m_{\text{match}}$（含 Indirect Partial Matching）**：
  - 在 $t_{\text{match}}$ 处对比前向计算值 $X^{\text{fwd}}$ 与后向（或另一分支）计算值 $X^{\text{bwd}}$。
  - **可匹配判据**：对于状态位置 $u$（字级或比特级），只有当其前向着色非 `White`（即 `Gray`/`Red`/`Blue`）**且**后向着色非 `White`，同时满足以下两种情况之一，方可计入匹配方程：
    1. **直接匹配**：前向颜色 $\neq$ 后向颜色（例如 `Red` vs `Blue`、`Red` vs `Gray`），或双方通过可分离加法/异或关系 $F_{\text{red}}(V_{\text{red}}) + F_{\text{blue}}(V_{\text{blue}}) \equiv C_{\text{gray}}$ 汇合；
    2. **间接部分匹配（Indirect Partial Matching）**：跨越匹配点两侧各最多 $L_{\text{ind}} \le 4$ 轮（例如利用 `backSolveA` 消去未知字 $W$、或跳过混合项后重构的线性组合 $A_{t+1} - T_2$），经 Z3 验证确实可分离为 $g_{\text{red}}(V_{\text{red}}) \equiv g_{\text{blue}}(V_{\text{blue}})$ 的比特数。
  - **进位不确定性扣减 $\Delta_{\text{carry}}$**：若匹配截断发生在 `ADD32` 链的高 $b$ 位（未读取低位输入），每个截断加法的未知进位引入 $\Delta_{\text{carry}} \ge 1$ bit 探针冗余（即有效过滤率降为 $m_{\text{eff}} = \max(0,\; m_{\text{match}} - \Delta_{\text{carry}})$）。

### 1.4 严格时间/内存复杂度计费公式（$T_{\text{attack}}$ vs $2^k$）
设攻击目标为在总轮数 $R_{\text{total}}$（单块 $64$ 轮、减轮 $R$ 轮、或 BTC Double-SHA256 $128$ 轮）上满足 **$k$ 比特前导零目标**（`clz(H[0]) >= k` 或 `clz256(H) >= k`，暴力穷举基线为 $2^k$ 次完整目标函数评估）。按目标输出自由度处理方式，严格区分两种合法拓扑（**严禁混用**）：

#### 模式 I：回绕型 Splice-and-Cut / Biclique（Backward Chunk 依赖 $S_R = H - IV$）
当后向分支从 $S_R = H - IV$ 启动时，目标仅固定了 $H$ 的 $k$ 比特，其余 $u = 256 - k$ 比特未知：
- 即使把 $u_{\text{blue}} \le u$ 比特并入 $V_{\text{blue}}$、其余 $u_{\text{fix}} = u - u_{\text{blue}}$ 比特固定为常量，由于内部匹配点仍须满足全部 256 比特状态一致性，每组 $2^{d_1 + d_2}$ 对 $(V_{\text{red}}, V_{\text{blue}})$ 中期望存在的合法 `(输入, H)` 解数量为：
  $$\mathbb{E}[\text{sol/group}] = 2^{d_1 + d_2^{\text{msg}}} \cdot 2^{-(k + u_{\text{fix}})} = 2^{d_1 + d_2 - 256} \cdot p_\Phi$$
  （其中 $d_2 = d_2^{\text{msg}} + u_{\text{blue}}$，$k + u_{\text{fix}} + u_{\text{blue}} = 256$）。
- 因此所需组数必须严格按 $2^{256}$ 尺度计费：
  $$N_{\text{groups}}^{\text{wrap}} = \frac{1}{p_\Phi} \cdot 2^{256 - (d_1 + d_2)}$$
- 单组计算量（以 $R_{\text{total}}$ 轮为单位）：
  $$C_{\text{group}} = c_{\text{bic}} \cdot 2^{\max(d_1, d_2)} + c_{\text{red}} \cdot 2^{d_1} + c_{\text{blue}} \cdot 2^{d_2} + c_{\text{recomp}} \cdot 2^{d_1 + d_2 - m_{\text{eff}}} + c_{\text{mem\_acc}}$$
  其中 $c_{\text{bic}} = L_{\text{bic}} / R_{\text{total}}$（若非独立 biclique 则乘以 $2^{\min(d_1, d_2)}$）、$c_{\text{red}} = R_{\text{fwd}} / R_{\text{total}}$、$c_{\text{blue}} = R_{\text{bwd}} / R_{\text{total}}$（均含消息扩展计算轮次权重）、$c_{\text{recomp}} = R_{\text{recomp}} / R_{\text{total}}$（对 $2^{d_1+d_2-m_{\text{eff}}}$ 个部分匹配候选做完整重算验证）。
- 总时间复杂度：$T_{\text{attack}}^{\text{wrap}} = N_{\text{groups}}^{\text{wrap}} \cdot C_{\text{group}}$。  
  **理论推论**：$T_{\text{attack}}^{\text{wrap}} \ge p_\Phi^{-1} \cdot \min(c_{\text{red}}, c_{\text{blue}}) \cdot 2^{256 - \max(d_1, d_2)}$。当 $k \le 256 - \max(d_1, d_2)$（涵盖全部实用 BTC 难度 $k \in [32, 128]$）时，回绕型 Splice-and-Cut 在数学上**绝不可能**超越 $2^k$！

#### 模式 II：无回绕型 / 截断输出直击 MitM（Truncated-Output Direct MitM）
若不假定 $H$ 的未知 $256-k$ 比特（即末态非目标字全部标记为 `White` 且不参与后向传播，或仅从 `clz(H[0]) >= k` 提供的 $A_R$ 高 $k$ 比特向上做截断逆推，严格把 $B_R..H_R$ 标记为 `White`）：
- 每组覆盖 $2^{d_1 + d_2}$ 个输入消息，达到目标 $k$ 比特的概率为 $2^{-k} \cdot p_\Phi$，因此所需组数（若 $d_1 + d_2 \ge k$ 则只需 1 组并缩减有效维度至 $d'_1 + d'_2 = k$）：
  $$N_{\text{groups}}^{\text{direct}} = \max\!\left(1,\; \frac{1}{p_\Phi} \cdot 2^{k - (d_1 + d_2)}\right)$$
- 若在内部匹配点处**所有**状态方程已通过构造/匹配验证，或直接在输出目标位 `H[0][31..32-k]` 上实现 $m_{\text{eff}} \le k$ 比特可分匹配，则总时间复杂度为：
  $$T_{\text{attack}}^{\text{direct}} = N_{\text{groups}}^{\text{direct}} \cdot \left( c_{\text{bic}} \cdot 2^{\max(d_1, d_2)} + c_{\text{red}} \cdot 2^{d_1} + c_{\text{blue}} \cdot 2^{d_2} + c_{\text{recomp}} \cdot 2^{\min(d_1+d_2, k) - m_{\text{eff}}} \right)$$
- **内存开销（强制计费）**：
  $$\text{Mem}_{\text{words}} = \min(2^{d_1}, 2^{d_2}) \cdot w_{\text{entry}},\qquad \log_2 M = \min(d_1, d_2) + \log_2 w_{\text{entry}}$$
  禁止声称“零内存重用”；若 $\min(d_1, d_2) > 64$，必须在报告中单列为“物理内存不可行”。
- **有效加速门禁（Anti-Triviality）**：对比暴力基线 $T_{\text{brute}}(k) = 2^k \cdot (1 - \epsilon_{\text{tail}})$（扣除暴力枚举本就可省略的末轮 $E_R$ 与 $H[1..7]$ 前馈输出，仅占 $< 1$ 轮），仅当 $\log_2 T_{\text{attack}} < k - \delta_{\min}$（设置显著阈值 $\delta_{\min} = 0.05$ bits，或以 $\Delta \log_2 T = k - \log_2 T_{\text{attack}}$ 如实报告全部边际数值）时才判为结构性加速。

---

## 2. B2-2：阳性校准与 BTC 128 轮截断前导零专属搜索计划

### 2.1 Stage 1：零 Bug 阳性校准（必须全部通过方可采信 Stage 2）
1. **Stage 1A（减轮 SHA-256 MitM 阳性校准，$R \in \{16, 20, 24, 28, 32\}$）**：
   - 在标准 SHA-256（含完整非线性 $\sigma_0, \sigma_1$ 消息扩展）上运行四色 CP-SAT 搜索器，自动复现经典的减轮 Splice-and-Cut / Initial-Structure MitM 独立块配置；
   - **验收指标**：在 $R = 24, 28, 32$ 轮下均须自动搜出有效四色划分（$d_1, d_2 \ge 1, m_{\text{eff}} \ge 1, T_{\text{attack}} < 2^{256}$），且对选出的独立块在 Z3 `BitVec(32)` 上验证 Red/Blue 严格非干涉。
2. **Stage 1B（Khovratovich 风格 Biclique 校准，$R \in \{41, 45, 52, 64\}$）**：
   - **45 轮全原像 / 52 轮伪原像校准**：启用 Independent/Star Biclique 区间，验证 CP-SAT 能在 $R=45$（固定 `stdIV`）及 $R=52$（自由 `S_0` 伪原像）找到 $m_{\text{eff}} \ge 1$ 且 $T_{\text{attack}} < 2^{256}$ 的经典 Khovratovich 风格 Biclique + MitM 骨架；
   - **64 轮独立块校准与渗透上限测量**：在完整 $R=64$ 轮 SHA-256 上运行：(i) 搜索并 Z3 证真 **64 轮局域 Khovratovich 风格独立 Biclique 子块**（如首部/尾部 $4 \sim 8$ 轮独立二分团 + 中性消息字集合，作为 64 轮独立块引擎零 Bug 校准）；(ii) 全局搜索在完整 64 轮与完整非线性消息扩展约束下可达的最大有效匹配位宽 $m_{\text{eff}}$ 与最大可穿透总轮数 $R_{\max}$，验证求解器既能准确复现局部 Biclique 独立块，又不会漏计消息扩展或虚构 64 轮无依据加速。

### 2.2 Stage 2：BTC 128 轮截断前导零专属搜索
面向 BTC Double-SHA256 全面扫描以下 4 个正交赛道（使用 OR-Tools CP-SAT + 启发式剪枝 + Z3 `BitVec(32)` 校验）：

1. **Track S2-A（单块 64 轮 Block 1b 固定 320-bit 填充 `W[4]=0x80000000, W[5..14]=0, W[15]=640` 专属搜索）**：
   - 分别测试 **(i) 矿工 Nonce-32 模式**（仅 $W[3]$ 自由，余固定）与 **(ii) Header-128 / Midstate-Free 模式**（$W[0..3]$ 及可选 $S_{\text{mid}}$ 自由）。
   - 目标覆盖 $k \in \{8, 16, 24, 32, 64, 80, 128, 224, 240, 256\}$，同时在模式 I 与模式 II 下求最优 $(d_1, d_2, m_{\text{eff}}, \log_2 T_{\text{attack}})$，并搜索是否存在利用 $W[4..15]$ 常量零塌缩（$W[16], W[17]$ 仅依赖 $W[0..2]$）的减轮（$R \in \{16..64\}$）或全轮截断前导零 MitM/Biclique 结构。
2. **Track S2-B（单块 64 轮 Block 2 固定 256-bit 填充 `W'[8]=0x80000000, W'[9..14]=0, W'[15]=256` 专属搜索）**：
   - 已知 $S'_0 = \text{stdIV}$、$W'[8..15]$ 为常量，仅 $W'[0..7]$ 自由。
   - 在截断前导零 `clz(H^{(2)}[0]) >= k`（$k \in \{8, 16, 24, 32, 64, 80, 256\}$）下搜索最大可达轮数 $R_2^{\max}$、最优独立块划分与局部 Biclique 独立块。
3. **Track S2-C（跨双块 128 轮 BTC Double-SHA256 联合搜索，$R_1 + R_2 \le 128$）**：
   - 联合搜索 Block 1b（$R_1 \in \{8, 16, 32, 64\}$）与 Block 2（$R_2 \in \{8, 16, 32, 64\}$）的跨块连接界面 $W'[0..7] = S_{R_1} + S_{\text{mid}}$：
     - 重点探测：**是否可将 Block 1b 的前向中性块与 Block 2 的截断前导零条件 `clz(H^{(2)}[0]) >= k`（结合 P3 Track B1/B2 已知的 $W'[4..7]$ 穿隧窗口或 $W'[9..14]=0$ 折叠）耦合出跨双块中性结构或加速 Biclique？**
     - 对减轮双块（如 $R_1 + R_2 = 16+16, 32+32, 64+16$）至满血 $64+64=128$ 轮，精确绘制 $R_1 + R_2$ 与最优 $\Delta \log_2 T = k - \log_2 T_{\text{attack}}$ 的相变边界曲线！
4. **Track S2-D（位级 Tier 2 进位截断与局部条件中性精细搜索）**：
   - 在 S2-A/B/C 的最优窗口边界上，展开 32-bit 逐位着色模型（允许 `clz >= k` 仅锁定高位 $k$ bits 并计入 $\Delta_{\text{carry}}$ 扣减），验证是否存在字级模型遗漏的位级截断独立块。

---

## 3. B2-3：Z3 + Lean 4.36.0 `bv_decide` 机器证真协议

对 Stage 1 与 Stage 2 产出的**代表性前向/后向独立块定理与跨双块/Biclique 非干涉候选**（包括阳性校准经典块、减轮最高突破块、64 轮局域 Biclique 独立块，以及 BTC 双块窗口发现的新候选或阻塞边界定理）：

1. **Z3 `BitVec(32)` 三重硬校验（由 `@p15-eng` 在云端先行自动跑完并导出见证 JSON）**：
   - **Z3-V1（Red 独立性）**：冻结 Gray，任意扰动 $V_{\text{blue}} \mapsto V'_{\text{blue}}$，验证 Red 分支目标状态/部分匹配表达 $F_{\text{red}}$ 恒不变（否定式 `UNSAT`）；
   - **Z3-V2（Blue 独立性）**：冻结 Gray，任意扰动 $V_{\text{red}} \mapsto V'_{\text{red}}$，验证 Blue 分支目标状态/部分匹配表达 $F_{\text{blue}}$ 恒不变（否定式 `UNSAT`）；
   - **Z3-V3（非退化性）**：存在输入使 $F_{\text{red}}$ 对 $V_{\text{red}}$ 变化（`SAT`）且 $F_{\text{blue}}$ 对 $V_{\text{blue}}$ 变化（`SAT`），同时确认涉及的非线性消息扩展关系 $\sigma_0, \sigma_1$ 均已真实展开。
2. **Lean 4.36.0 `bv_decide` 零 `sorry` 形式化证真（由 `@p15-math` 在本地 CPU 执行）**：
   - 工程位置：`g:\SHA-LM\v12_frontier\track_b2_p15_mitm_biclique\lean_verify\`
   - 工具链：`leanprover/lean4:v4.36.0`，`moreLeanArgs = ["-j", "2", "-M", "2048"]`，**零 Mathlib**，严格本地 CPU 构建。
   - 形式化模板：参考 `P3Verify.TrackB1.p3_id_014`（双轨非干涉定理 `forall vars, F(vars) = F(vars[V_other <- V'_other])`）与 `P3Verify.Primitives`（精确对齐 `sha256Round`, `msgScheduleNext`, `stdIV`, `btcPadW8..15` 以及 Block 1b 的 `btcB1PadW4..15`）。
   - 终审门禁：`lake build` 全通过 + `lake env lean VerifyAxioms.lean` 逐条打印 `#print axioms`，仅允许标准 Lean 公理 `[propext, Classical.choice, Quot.sound]` 及原生 `bv_decide` LRAT 证书，**严禁出现 `sorryAx` 或任何自定义公理**。

---

## 4. `@p15-eng` 工程实现与交付格式契约

请 `@p15-eng` 在 Colab 会话 `v12-p15` 完成全部搜索与 Z3 校验，同步结果至 `g:\SHA-LM\v12_frontier\track_b2_p15_mitm_biclique\results\` 并执行 `colab.exe stop -s v12-p15` 后，交付以下 JSON（数值需精确可追溯）：

1. **`results/stage1_calibration.json`**：包含 `mitm_reduced`（$R \in \{16, 20, 24, 28, 32\}$）与 `biclique_khovratovich`（$R \in \{41, 45, 52, 64\}$）每个配置的最优 $(d_1, d_2, L_{\text{bic}}, m_{\text{match}}, \Delta_{\text{carry}}, m_{\text{eff}}, \log_2 T_{\text{attack}}, \log_2 M, \text{z3\_verified})$ 及着色明细。
2. **`results/stage2_btc_search.json`**：包含 `track_s2a_block1b`、`track_s2b_block2`、`track_s2c_double128`、`track_s2d_bitlevel` 全量扫描表（覆盖各 $(R_1, R_2, k, \text{mode})$ 下的最优参数、`delta_log2_T = k - log2_T_attack`、`is_speedup`、`z3_verified` 及最大可穿透轮数相变曲线）。
3. **`results/lean_candidates.json`**：导出不少于 **4~8 条**代表性独立块/Biclique/跨双块非干涉候选定理的完整形式化规格（含 `candidate_id`、`category`、`rounds_range`、`use_msg_schedule`、`fixed_padding_mode`、`gray_vars`、`red_vars`、`blue_vars`、`target_expr`、`non_interference_side`、`z3_status`、`z3_ms`），供 `@p15-math` 编写 Lean 4 定理并用 `bv_decide` 证死。
4. **`results/run_log.txt`**：包含 `src/sha256_ref.py` 对拍记录（`xor_diff == 0`）、Colab `v12-p15` 运行日志及 `colab.exe sessions` 核实无残留实例的完整输出。
