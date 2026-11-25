# Phase-SHA V5 数学与工程总规范（四路正交破壁体系 & V6 标准化插槽）

## 1. 战略定位与严格控制变量原则

Phase-SHA V4 在 Colab A100-SXM4-80GB 上已完成严格纯 FP32（禁 TF32 / 禁混合精度）实证检验：虽然 4D 非阿贝尔四元数 $\mathbb{S}^3$（**V4C**）将物理违约能从 $\mathbb{S}^1$（**V4A**）的 `0.1529` 压低至 `0.0777`（并提升中间传播轮 `r10=0.684 / r12=0.647`），但所有单比特连续球面流形均表现出两项共性病灶：
1. **连续舒适区躺平**：固定弱退火使 `align` 停在 `0.50–0.54` 附近，向量拒绝彻底交回离散实轴 $\pm 1$；
2. **闭环平均场势垒**：受挫闭环瓶颈位点（`r13`, `r16`, `W0`）处于零边缘期望叠加区，梯度流与独立乘积表征在穿零处被迫承受 `>= 1` 违约能量墙（`r13/r16 ≈ 0.498`）。

**Phase-SHA V5** 严格对齐已通过 Lean 4.36.0 零公理校验的 **Module 6A–6D**（见 [PROOF_SUMMARY.md](file:///g:/SHA-LM/formal_proofs/PROOF_SUMMARY.md)），采用**四路控制变量单卡消融体系**：
- **V5D（优先级 1）**：双频域分裂（Walsh-Hadamard 布尔门 + 2-adic 精确进位 Viterbi DP）+ 神经引导 Elser RRR 混沌反射引擎（[DualDomainRRR.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/V5/DualDomainRRR.lean)）；
- **V5C（优先级 2）**：闭环拓扑感知张量网络 MPS 纠缠态 + QR/TEBD 演化 + 自回归条件 Born 坍缩采样（[TensorEntanglementBarrier.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/V5/TensorEntanglementBarrier.lean)）；
- **V5B（优先级 3）**：Burer-Monteiro 超高维球面 $\mathbb{S}^{N-1}$（$N \in \{16, 32, 64\}$）+ 严格非负零下界门残差 + 渐进秩-1 坍缩退火 + 正交有理 Cayley 负曲率逃逸（[BurerMonteiroEscape.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/V5/BurerMonteiroEscape.lean)）；
- **V5A（优先级 4）**：8D 八元数非结合单位超球面 $\mathbb{S}^7 \subset \mathbb{O}$ + Cayley-Dickson 纯实数乘法 + 结合子退火与全局升温实轴坍缩（[OctonionLift.lean](file:///g:/SHA-LM/formal_proofs/PhaseSHA/V5/OctonionLift.lean)）。

---

## 2. 全局不可妥协硬性红线

1. **死守纯血 FP32**：
   - 全程强制 `allow_tf32 = False`（matmul 与 cuDNN）、`torch.set_float32_matmul_precision("highest")`；
   - 严禁 FP16 / BF16 / INT8 / `autocast`，严禁使用 `torch.complex64`（所有几何与张量运算均以纯实数 `torch.float32` 展开）。
2. **统一考核与双轴对比**：
   - 全部变体共用完全一致的 `R=18` `2,048` 条随机无交集留出集（`holdout_seed=20261122`，[v5_interfaces.py](file:///g:/SHA-LM/src/v5_interfaces.py)）；
   - 同时内置 `R=20` 压力测试卷生成器 `build_r20_stress_holdout_w16`；
   - 日志同时输出 `step_time_ms` 与累计有效算力墙钟 `wall_s`，支持“同训练步数”与“同 GPU 墙钟时间”双轴公平对比。
3. **面向 V4D / V4E / V6 的标准化四插槽架构**：
   - 槽位 ① `GeometryBackend`：单比特高维流形坐标系与代数残差（V4C / V5A / V5B）；
   - 槽位 ② `EntanglementLayer`：多比特纠缠态与精确条件 Born 采样（V5C）；
   - 槽位 ③ `DynamicsOperator`：Riemannian 梯度流 vs 无假坑 Elser RRR 反射算子（V5D）；
   - 槽位 ④ `OuterSearchShell`：动态数字锁存、SMC / 多粒子搜索及 V4D 校准比特概率 `P(bit=1)` 导出；
   - 所有求解器原生复用 `ManifoldBoundary` / `build_v4e_partial_prefix_boundary` 位级任意掩码（兼容 V4E 前导零微调）。
4. **全局渐进强坍缩调度（`GlobalProgressiveCollapseSchedule`）**：
   - 前 `20%` 全局步数维持宽松探索权重；
   - `20%–80%` 平滑余弦拉升；
   - 后 `20%`（`80%–100%`）强力收束，治愈 V4“连续舒适区躺平”病灶。

---

## 3. 四路变体核心数学契约

### 3.1 V5D：`v5d_pure` 与 `v5d_neural` (`src/v5d_rrr.py`)
- **变量-门副本图**：对 `R` 轮 SHA-256（每一轮包含 `Sigma1, Ch, Sigma0, Maj` 与模 $2^{32}$ 加链；$t \ge 16$ 包含消息扩展 `sigma1, sigma0` 与模加链），为每个约束门分配独立变量副本 `x`。
- **精确双投影 $(P_A, P_B)$**：
  - $P_A(x)$：将同一底层变量的所有副本按（可学习）正权重聚合为共识值，并强制锁死已知边界/IV/固定填充与常量 $K_t$ 位；
  - $P_B(x)$：
    - 2/3 输入 `XOR`, `Ch`, `Maj`：通过纯实数 Walsh-Hadamard 谱变换 / 精确真值表最近欧氏顶点映射一步投回离散合法集合；
    - 32-bit 模 $2^{32}$ 加法 $a + b \equiv c \pmod{2^{32}}$：用全 GPU 批量化的 `O(32)` 精确 2 状态（carry $\in \{0,1\}$）Viterbi 动态规划一步求得全局最小欧氏距离整数三元组 $(a',b',c')$。
- **RRR 迭代与 Lean 4 三条诚实铁律**：
  $$x_{t+1} = x_t + \beta_t \big(P_B(2 P_A(x_t) - x_t) - P_A(x_t)\big), \qquad \beta_t \in [0.35, 1.15]$$
  1. **阴影读出**：所有物理能量、候选解、精确检验严格从阴影 $P_A(x_t)$ 读取（绝不读 $x_t$）；
  2. **非零步长钳位**：严格禁止 $\beta_t \to 0$；
  3. **线性漂移雷达**：实时统计 $\|x_t\|_2$ 随步数的线性增长斜率与拟合优度，触发 `drift_flag` 报警。

### 3.2 V5C：闭环感知张量网络 MPS 求解器 (`src/v5c_mps.py`)
- **闭环节点重排**：针对 $R=18$ 将闭环耦合节点（如 $W_0 \leftrightarrow W_{16}, W_{14..17}$ 与 $r_{13} \leftrightarrow r_{16}$）排布在相邻 MPS 格点，减少有限键维 $\chi \in \{4, 8, 16, 32\}$ 下的长程割边开销。
- **全批量 QR + TEBD 演化**：全程采用纯 FP32 批量小矩阵 `torch.linalg.qr` 规范化与奇偶双体门 TEBD 更新（严禁大批量串行 SVD）。
- **精确自回归条件 Born 采样**：维护右环境张量 $E_{k+1}$，按 $P(s_k \mid s_{<k}) = \frac{\|(v_L A_k^{s_k}) E_{k+1}^{1/2}\|^2}{Z}$ 从左至右逐位条件坍缩采样并即时更新左状态 $v_L$；同时监控冯·诺依曼纠缠熵 `ent_entropy` 与联合约束能量 `ent_energy`。

### 3.3 V5B：Burer-Monteiro 超高维球面 $\mathbb{S}^{N-1}$ (`src/v5b_bm.py`)
- **单位超球面 $N \in \{16, 32, 64\}$**：锚定离散实轴 $e_0 = (1,0,\dots,0)$，所有运算规整为纯 FP32 `bmm`/`einsum`。
- **反作弊严格非负零下界门残差**：禁止使用允许向量钻入负能区的无界内积项；全部 XOR / 门残差及受挫耦合项构造为完全平方和形式（真解精确为 `0`，任何受挫闭合均严格 `>= 0`）。
- **渐进秩-1 退火与正交有理 Cayley 逃逸**：
  - 引入全局升温的横轴惩罚 $\lambda_{\text{rank1}}(t)\sum (1 - v_{i,0}^2)$ 并记录 `rank1_ratio` 与 `eff_rank`；
  - 当粒子能量滞留平台期时，计算与主导低维子空间正交的逃逸方向并沿半角有理 Cayley 测地线扰动（无开方严格保范）。

### 3.4 V5A：8D 八元数非结合单位超球面 $\mathbb{S}^7 \subset \mathbb{O}$ (`src/v5a_octonion.py`)
- **纯实数 Cayley-Dickson 乘法**：$(a,b)(c,d) = (ac - \bar{d}b, da + b\bar{c})$（$a,b,c,d \in \mathbb{H}$），满足 Degen 八平方范数可乘性、交错性与离散实轴精确 XOR 同构。
- **显式结合子惩罚与双路强退火**：对 3 输入 XOR 对称平均两种括号结合顺次并惩罚 $\|(xy)z - x(yz)\|_2^2$；通过全局递增的 $\lambda_{\text{assoc}}(t)$ 与 $\lambda_{\text{align}}(t)$ 与 V4C 形成最纯净的 $4\text{D} \to 8\text{D}$ 单变量对照。
