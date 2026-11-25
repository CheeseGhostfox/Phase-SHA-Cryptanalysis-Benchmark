# P3 数学契约：神经/MCTS 猜想 × Z3 / Lean 4 `bv_decide` 跨轮恒等式挖掘机

**工作目录**：`g:\SHA-LM\v11_discovery\p3_symbolic_lean4\`  
**数学与形式化设计**：`@p3-math`  
**工程与 Colab CPU 搜索**：`@p3-eng`  
**禁改目录**：`v10_ouroboros64/`、`runs/`、`formal_proofs/`、`formal_proofs_v10/`

---

## 1. 既有已证结果基线（严禁重发现与包装）

经对 [PROOF_SUMMARY.md](file:///g:/SHA-LM/formal_proofs/PROOF_SUMMARY.md)、[BitVecIdentities.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/Core/BitVecIdentities.lean)、[PartialPreimagePoW.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/Core/PartialPreimagePoW.lean) 与 [CubeBudgetAndSplice.lean](file:///g:/SHA-LM/formal_proofs_v10/OuroborosV10/CubeBudgetAndSplice.lean) 的勘察，下列定理及一切等价变形/平凡推论**已被正式证明**，搜索流水线必须显式剔除：

1. **4 轮无 $W$ 倒推钳位（`four_round_clamp_w_invariant` / `four_round_clamp_independent_of_w`）**：  
   从 $S_R$ 剥离 $IV$ 后，连续 4 次应用 `backSolveA` 精确恢复 $(A_{R-4}, A_{R-5}, A_{R-6}, A_{R-7})$，与 $W_{R-4..R-1}$ 无关。  
   - **特别注意反作弊**：在 $R \ge 5$ 轮上直接对末 4 轮做 `clampBackSolve4(S_R)` 恢复输入移位寄存器 $s_0.a, s_0.b, s_0.c$，实际上**未用到第 $0$ 轮非线性 ARX 计算**，有效跨轮仅 $4$ 轮，严禁计入新发现！
2. **单步旁路消 $T_1$（`r18_a10_algebraic_tunneling` / `backSolveA`）**：  
   $A_{t-3} = E_{t+1} - A_{t+1} + \Sigma_0(A_t) + \mathrm{Maj}(A_t, A_{t-1}, A_{t-2})$，消去单个 $T_1(t)$ 进而独立于 $E_t..E_{t-3}, K_t, W_t$。
3. **单步 `Ch` 位级屏蔽（`r18_ch_masking_shield` / `r18_ch_masking_word_bv`）**：  
   $E_t \land \mathrm{Ch}(E_t, E_{t-1}, E_{t-2}) = E_t \land E_{t-1}$ 及其取反对称形式 $(\neg E_t) \land \mathrm{Ch}(E_t, E_{t-1}, E_{t-2}) = (\neg E_t) \land E_{t-2}$。
4. **尾 3 轮 $H_7$ 解耦（`btc_h7_tail_round_decoupling`）**：  
   $H_7 = IV_7 + E_{R-3}$ 与 $W_{R-3..R-1}$ 无关；以及 $H_0, H_4$ 对末轮 $W_{R-1}$ 单射敏感、`R >= 20` 时 $W_{16}$ 到达 $H_7$。
5. **V10 `splice_block[:8]` 截断丢弃与消息调度深度（`dag_depth_w25`）**。

---

## 2. 精确形式化语义（与 `src/sha256_ref.py` 及 `BitVecIdentities.lean` 严格对齐）

所有字均为 32 位无符号位向量 `BitVec 32`（Z3 中为 `BitVecSort(32)`）：

### 2.1 原语算子
- `rotr32(x, n)`：32 位循环右移（$n \in \{0..31\}$）；`shr32(x, n)`：32 位逻辑右移；`shl32(x, n)`：32 位左移。
- `ch32(e, f, g) = (e & f) ^ ((~e) & g)`
- `maj32(a, b, c) = (a & b) ^ (a & c) ^ (b & c)`
- `Sigma0(x) = rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22)`
- `Sigma1(x) = rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25)`
- `sigma0(x) = rotr32(x, 7) ^ rotr32(x, 18) ^ shr32(x, 3)`
- `sigma1(x) = rotr32(x, 17) ^ rotr32(x, 19) ^ shr32(x, 10)`

### 2.2 压缩轮与消息扩展
第 $t$ 轮前状态 $S_t = (A_t, A_{t-1}, A_{t-2}, A_{t-3}, E_t, E_{t-1}, E_{t-2}, E_{t-3})$：
$$T_1(t) = E_{t-3} + \Sigma_1(E_t) + \mathrm{Ch}(E_t, E_{t-1}, E_{t-2}) + K_t + W_t \pmod{2^{32}}$$
$$T_2(t) = \Sigma_0(A_t) + \mathrm{Maj}(A_t, A_{t-1}, A_{t-2}) \pmod{2^{32}}$$
$$A_{t+1} = T_1(t) + T_2(t) \pmod{2^{32}}, \qquad E_{t+1} = A_{t-3} + T_1(t) \pmod{2^{32}}$$
其余移位寄存：$B_{t+1}=A_t, C_{t+1}=A_{t-1}, D_{t+1}=A_{t-2}, F_{t+1}=E_t, G_{t+1}=E_{t-1}, H_{t+1}=E_{t-2}$。

消息扩展（当 $t \ge 16$ 且启用调度模式时）：
$$W_t = \sigma_1(W_{t-2}) + W_{t-7} + \sigma_0(W_{t-15}) + W_{t-16} \pmod{2^{32}}$$

### 2.3 BTC 双 SHA-256 跨块拼接界面（Track B 核心建模）
设 Block 1 经过 $R_1$ 轮（或末尾窗口 $r_1$ 轮，如 $R_1 \ge 5$ 或尾窗口接 Block 2）到达终态 $S^{(1)}_{R_1} = (A_{R_1}, A_{R_1-1}, A_{R_1-2}, A_{R_1-3}, E_{R_1}, E_{R_1-1}, E_{R_1-2}, E_{R_1-3})$：
1. **双块拼接映射**：
   $$W'[0..3] = (IV_0 + A_{R_1},\; IV_1 + A_{R_1-1},\; IV_2 + A_{R_1-2},\; IV_3 + A_{R_1-3})$$
   $$W'[4..7] = (IV_4 + E_{R_1},\; IV_5 + E_{R_1-1},\; IV_6 + E_{R_1-2},\; IV_7 + E_{R_1-3})$$
2. **Block 2 已知固定填充字**：
   $$W'[8] = \text{0x80000000}, \quad W'[9..14] = \text{0x00000000}, \quad W'[15] = \text{0x00000100}$$
3. **Block 2 固定初始态**：
   $$S'_0 = (IV_0, IV_1, IV_2, IV_3, IV_4, IV_5, IV_6, IV_7)$$
4. **Block 2 扩展字（$t \ge 16$）**：按标准 `msgScheduleNext` 由 $W'[0..15]$ 递推，其中 $W'[9..14]=0$ 导致多处高阶零项坍缩（如 $W'[16] = \sigma_0(W'[1]) + W'[0]$）。

---

## 3. 猜想语法树（AST）与双形式命题规范

每个猜想均表示为可**1 对 1 翻译为 Z3 Python 与 Lean 4 `BitVec 32`** 的结构化 JSON AST。

### 3.1 `Expr32`（32 位位向量表达式语法树）
```text
Expr32 ::= Var(name)                      -- 可观测量集合 O 中的变量，或目标表达式中的变量
         | Const(uint32)                  -- 0, 1, 0xFFFFFFFF, 0x80000000, 0x00000100, IV[i], K[t]
         | Add(Expr32, Expr32)            -- x + y (mod 2^32)
         | Sub(Expr32, Expr32)            -- x - y (mod 2^32)
         | Neg(Expr32)                    -- -x
         | And(Expr32, Expr32)            -- x & y
         | Or(Expr32, Expr32)             -- x | y
         | Xor(Expr32, Expr32)            -- x ^ y
         | Not(Expr32)                    -- ~x
         | Rotr(Expr32, c)                -- rotr32(x, c), c in 1..31
         | Shr(Expr32, c)                 -- shr32(x, c), c in 1..31
         | Shl(Expr32, c)                 -- shl32(x, c), c in 1..31
         | Sigma0(Expr32) | Sigma1(Expr32)
         | sigma0(Expr32) | sigma1(Expr32)
         | Ch(Expr32, Expr32, Expr32)
         | Maj(Expr32, Expr32, Expr32)
```
建议 AST 深度上界 `max_depth <= 7`、节点数 `<= 40`（确保 `bv_decide` 能在 2048 MB 内存限制下秒级闭合）。

### 3.2 `Cond`（条件掩码 / 进位屏蔽谓词语法树，`BitVec 1` / `Prop`）
```text
Cond ::= True
       | Eq(Expr32, Expr32)               -- 如 e1 == e2, e1 == ~e2
       | MaskZero(Expr32, Expr32)         -- (a & b) == 0   （进位屏蔽：a + b == a ^ b，或位掩码为 0）
       | MaskEq(Expr32, Expr32, Expr32)   -- (a & m) == (b & m)
       | BitEq(Expr32, k, bit01)          -- ((x >>> k) & 1) == bit01, k in 0..31
       | AndCond(Cond, Cond)              -- 最多合取 <= 4 个原子条件，严禁把自由变量锁死为全常数
```

### 3.3 命题两大标准形式（二选一，均要求严格含 `Elim` 变量消除）
设全变量集合拆分为：
- **可观测量 $\mathcal{O}$**：可出现在 `LHS` 中的变量（例如末态 $S_R$ 的部分字、Block 2 末态 $S'_{r_2}$、固定/已知字）；
- **被消去的高扩散变量集 $\mathcal{U}_{\text{elim}}$**（$\mathcal{U}_{\text{elim}} \neq \emptyset$，且 $\mathcal{U}_{\text{elim}} \cap \mathcal{O} = \emptyset$）：例如跨轮中间消息字 $W_{t_1..t_m}$、高扩散状态字 $E_t$、或 Block 1 尾态未知字 $E^{(1)}_{R_1..R_1-3}$ / $W^{(1)}_{R_1-4..R_1-1}$。

- **形式 I（条件代数重构 / 显式消元恒等式）**：
  $$\forall \text{free\_vars},\quad \Phi \implies (M \land \mathrm{LHS}(\mathcal{O})) = (M \land \mathrm{RHS}(\text{target}))$$
  其中 `LHS` **语法上不含任何 $\mathcal{U}_{\text{elim}}$**，`RHS` 为目标状态字/消息字表达式（若无掩码则 $M = \text{0xFFFFFFFF}$；也可等价写为 `BitVec 1` 逐位式 `bit(LHS, k) = bit(RHS, k)`）。
- **形式 II（条件掩码非干涉 / 屏蔽与差分抵消定理）**：
  $$\forall \text{free\_vars}, \forall \mathcal{U}'_{\text{elim}},\quad \big(\Phi(\text{vars}) \land \Phi(\text{vars}[\mathcal{U}_{\text{elim}} \leftarrow \mathcal{U}'_{\text{elim}}])\big) \implies (M \land \mathcal{E}(\text{trace})) = (M \land \mathcal{E}(\text{trace}'))$$
  即：在条件 $\Phi$ 与掩码 $M$ 下，跨 $\ge 5$ 轮（或跨双块界面）的复合式 $\mathcal{E}$ 对高扩散未知变量 $\mathcal{U}_{\text{elim}}$ 的任意扰动严格免疫！

---

## 4. 两大搜索赛道（Search Tracks）

### 赛道 A：5~10 轮复合 ARX（含无扩展与有扩展 $W_{16..}$ 耦合）
1. **A1（跨 5~10 轮条件掩码 + 进位屏蔽与双向边界消元）**：  
   探索在局部进位屏蔽条件（如 `(x & y) == 0`）、`Ch`/`Maj` 分支锁定条件（如 `(e_{t} & m) == m` 或 `e_{t-1} == e_{t-2}` 或 `a_t == ~a_{t-1}`）或部分初态/末态观测下，能否把传统止步于 4 轮的 `backSolveA` 链或双轨 $A$/$E$ 耦合链穿透至 **5~10 轮**，消去深层 $W_t$ 或 $E_t$ 或实现特定位掩码 $M$ 下的跨轮进位无损传递。
2. **A2（含消息调度 $W_t = \sigma_1(W_{t-2}) + W_{t-7} + \sigma_0(W_{t-15}) + W_{t-16}$ 的跨 5~10 轮恒等式）**：  
   探索将多轮压缩函数观测与 $W_t$ 递推耦合（跨度 5~10 轮，如 $t \in [12, 21]$ 或 $[16, 25]$），在无条件或弱条件下消去若干未知中间字 $W$ / $E$ / $A$。

### 赛道 B：BTC 双 SHA-256 块拼接界面（Block 1 尾轮 $\to W'[0..7] = IV + S^{(1)}_{R_1} \to$ Block 2 前轮与填充区）
1. **B1（跨块状态穿隧与 Block 1 尾态/未知 $E, W$ 消去）**：  
   利用 Block 2 初态 $S'_0 = IV$ 已知、且 $W'[0..7] = IV + S^{(1)}_{R_1}$ 的双块强耦合结构，搜索跨 **Block 1 末 $r_1$ 轮 + 拼接界面 + Block 2 前 $r_2$ 轮**（总跨度 $\ge 5$ 轮，或 Block 2 深入至 $R_2 \in [5, 16]$ 反向穿回 Block 1 尾态）的恒等式。  
   *关键科学问题*：仅观测 Block 2 中间态 $S'_{r_2}$（或其子集）及可选的 Block 1 前向部分字时，能否在无条件或条件掩码下穿透 $W'[4..7]$（即 $E^{(1)}_{R_1..R_1-3}$）甚至部分 $W^{(1)}$，重构 Block 1 深层状态字或消除 Block 1 的高扩散变量？
2. **B2（Block 2 填充零项 $W'[9..14]=0$ 引发的扩展坍缩跨轮耦合）**：  
   探索 Block 2 进入扩展区 $t \in [16, 23]$ 后，结合 $W'[8]=\text{0x80000000}, W'[9..14]=0, W'[15]=256$ 与 Block 1 摘要 $W'[0..7]$ 的跨轮/跨块代数约束与条件进位屏蔽恒等式。

---

## 5. 七项强制非平凡性与反作弊过滤协议（Z3 机器判定）

每条由神经/MCTS 提议并通过批量随机采样过滤的候选，**必须全部通过以下 7 项 Z3 验证**方可进入产出清单：

1. **（V1）全称真值证死**：命题否定的 Z3 `Solver` 在 `BitVec(32)` 上返回 **`UNSAT`**（严禁仅靠随机样本统计或超时 `UNKNOWN` 充数）。
2. **（V2）条件非空与目标非退化（防全锁死作弊）**：  
   - $\Phi$ 必须 `SAT`；
   - 且 $\exists x_1, x_2$ 满足 $\Phi(x_1) \land \Phi(x_2) \land (M \land \mathrm{RHS})(x_1) \neq (M \land \mathrm{RHS})(x_2)$（目标值在 $\Phi$ 下不能坍缩成单一常数！）；
   - 且 $\exists x_1, x_2$ 满足 $\Phi(x_1) \land \Phi(x_2) \land u(x_1) \neq u(x_2)$（被消去的变量 $u \in \mathcal{U}_{\text{elim}}$ 在 $\Phi$ 下至少拥有多值自由度；进一步建议记录 $\mathcal{U}_{\text{elim}}$ 在 $\Phi$ 下的自由比特度/非固定字节数）。
3. **（V3）条件必要性（若 $\Phi \not\equiv \text{True}$）**：  
   当 $\neg\Phi$ 成立时，原等式必须存在反例（即 `SAT(¬Φ ∧ LHS ≠ RHS)`），证明条件 $\Phi$ 是实质起作用的，而非冗余装饰。
4. **（V4）掩码必要性（若 $M \not\equiv \text{0xFFFFFFFF}$）**：  
   若去掉掩码（令 $M = \text{0xFFFFFFFF}$）后原等式仍全空间成立，则应化简并仅保留无掩码最强版本；若去掩码后出现反例（`SAT(Φ ∧ LHS_full ≠ RHS_full)`），证明位掩码/进位屏蔽 $M$ 实质起作用；同时要求 $M \neq 0$（最好同时报告 `popcount(M)`，即有效保护位数 $\ge 1$）。
5. **（V5）非纯代数恒同（必须依赖 SHA-256 动力学）**：  
   将窗口内 `sha256Round` 输出 $(A_{t+1}, E_{t+1})$ 替换为无约束自由变量后，命题必须存在反例（`SAT`），证明其并非类似 `(x+y)-y=x` 的纯算术恒等变形。
6. **（V6）实质跨度 $\ge 5$ 轮 / 跨双块耦合语义校验（核心防重复壁垒！）**：  
   - 对每轮 $t$ 做**逐轮 ARX 断链测试**：将第 $t$ 轮的非线性输出 $(A_{t+1}, E_{t+1})$ 替换为独立自由变量（仅保留移位寄存器连线），用 Z3 检查原命题是否失效（`SAT`）。若失效，则第 $t$ 轮为**必要活跃轮**（Active Round）。
   - **硬性准入**：`active_span = t_max_active - t_min_active + 1 >= 5`（或活跃轮数 `len(active_rounds) >= 5`）。这一条在语义上 100% 封死了将 `four_round_clamp`（活跃跨度仅为 4）或 `r18_a10` / `r18_ch_masking`（活跃跨度仅为 1）伪装成 5+ 轮的可能！
   - **赛道 B 额外准入**：除满足上述跨轮条件（或 Block 1+Block 2 联合活跃跨度 $\ge 5$）外，还必须验证**跨块拼接必要性**：若断开 $W'[0..7] = IV + S^{(1)}_{R_1}$ 约束（将 $W'[0..7]$ 设为独立自由变量），Z3 必须返回 **`SAT`**（存在反例），证明该定理实质利用了 BTC 双 SHA-256 跨块接口！
7. **（V7）去重与已知库隔离**：  
   按规范化 AST 指纹与 Z3 语义等价判定去重（若两条同条件新式在 Z3 下恒等，仅保留 AST 更简洁或消除变量更多者）。

---

## 6. 神经/MCTS 猜想引擎设计指南（供 `@p3-eng` 实现）

1. **搜索状态与动作空间**：  
   - 将表达式构造建模为带类型约束的 MCTS 树（从 `Goal` 根节点向下展开 `Expr32` / `Cond` / `Mask` 与选定观测变量集 $\mathcal{O}$ / 消去集 $\mathcal{U}_{\text{elim}}$）。
   - 支持从种子基元（如单步 `backSolveA`、`T1/T2` 差分、`Ch/Maj` 掩码模板、跨块 $W'[0..7]$ 界面模板）出发进行**结构变异 + 跨轮复合 + 掩码合成**。
2. **神经先验 / 价值引导（纯 CPU 轻量策略-价值网络）**：  
   - 特征编码：AST 前缀序列化 token 嵌入 + 变量轮次跨度直方图 + 批量采样真值表比特一致性特征（如在 $N=512$ 组满足 $\Phi$ 的随机输入下，`LHS ^ RHS` 的 32 位零比特命中率 / 最低汉明距离）。
   - 奖励函数：
     - 语法合法且含 $\ge 5$ 轮跨度：基础分 $0.05$；
     - 512 样本比特匹配率 $p_{\text{bit}} \in [0.5, 1.0)$（发现弱偏置/部分掩码线索）：连续软奖励 $0.1 + 0.8 \cdot \max_k (\text{bit\_match}_k)$，指导发现掩码 $M$（即在 `bit_match_k == 1.0` 的位集合自动合成精确掩码 $M$！）；
     - 512 样本精确全匹配（或掩码后全匹配）+ 通过 Z3 V1–V7 全部检查：高额奖励 $+10.0 + 1.0 \times \text{active\_span} + 0.5 \times |\mathcal{U}_{\text{elim}}|$，并立即将成功轨迹加入经验回放池训练神经策略/价值头（在线自举迭代）。
3. **快速采样器注意点**：  
   - 当条件 $\Phi$ 含有等式（如 `e_{t-1} == e_{t-2}` 或 `(x & y) == 0`）时，随机采样器须**先反向/构造式采样满足 $\Phi$ 的变量组合**（或用 10 ms Z3 模型采样器生成 64 个多样化见证样本），避免朴素随机采样命中 $\Phi$ 为零导致漏报或假空过。

---

## 7. 输出交付格式（`@p3-eng` $\to$ `@p3-math`）

请 `@p3-eng` 在 Colab `v11-p3` 完成搜索并停机后，将所有结果同步至 `g:\SHA-LM\v11_discovery\p3_symbolic_lean4\results\`，其中核心交付文件为 `results/discovered_identities.json`，格式如下：

```json
{
  "run_summary": {
    "colab_session": "v11-p3",
    "colab_stopped": true,
    "total_candidates_generated": 0,
    "passed_random_filter": 0,
    "z3_unsat_verified": 0,
    "passed_all_7_nontriviality": 0,
    "wall_time_seconds": 0.0
  },
  "identities": [
    {
      "id": "p3_id_001",
      "track": "A1 | A2 | B1 | B2",
      "title": "简短中文标题",
      "form": "FormI_Reconstruction | FormII_MaskingInvariance",
      "rounds_config": {
        "round_start": 0,
        "num_rounds": 6,
        "use_msg_schedule": false,
        "btc_splice": false,
        "block1_rounds": 0,
        "block2_rounds": 0
      },
      "observable_vars": ["..."],
      "eliminated_vars": ["..."],
      "condition_ast": {},
      "mask_ast": {},
      "lhs_ast": {},
      "rhs_ast": {},
      "pretty_formula": "数学可读公式字符串",
      "z3_audit": {
        "v1_unsat": true,
        "v2_nondegenerate": true,
        "v3_cond_necessary": true,
        "v4_mask_necessary": true,
        "v5_nontautology": true,
        "v6_active_rounds": [0, 1, 2, 3, 4],
        "v6_active_span": 5,
        "v6_splice_necessary": false,
        "z3_solve_ms": 12.3,
        "mask_popcount": 32
      }
    }
  ]
}
```

`@p3-math` 收到上述 JSON 后，将在本地 CPU（`v11_discovery/p3_symbolic_lean4/lean_verify/`，Lean 4.36.0，`-j 2 -M 2048`，零 Mathlib）为每一条入选恒等式编写独立的 `bv_decide` 形式化定理，运行 `lake build` 与 `#print axioms` 审计（零 `sorry`），最终产出 `verdict.json` 与 `REPORT.md`。
