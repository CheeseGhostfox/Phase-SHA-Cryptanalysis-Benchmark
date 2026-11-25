# Phase-SHA V6A 设计与工程规范（修订定稿版）：Ouroboros 全栈反置乱引擎

> 目标：在严格**零修改**冻结的 V3 / V4 / V5 代码、测试与检查点前提下，于 `g:\SHA-LM\v6a\` 构建 **Phase-SHA V6A (`Ouroboros`)**：将**连续双同伦 × 非局域全局线性-维特比 RRR (`GL-RRR`) × 靶向瓶颈真纠缠 MPS × Burer-Monteiro 高维逃逸 × 零刹车神经导航**融为单闭环全栈引擎，并集成 Phase-0 临界相变雷达、本地 RTX 4060 `R=4/5` 硬核验真与 Colab A100 全程托管实验及实时监控面板。

---

## 一、五条核心工程铁律

1. **彻底废除单比特边缘 BCE**：严禁在主优化目标、瓶颈 MPS 训练或 RRR 导航引导中使用单比特边缘交叉熵（已在 V5 中证实会压低 $\beta$ 并使 MPS 纠缠熵 $S_{\text{ent}} \to 0$）。
2. **严守瞬时活体口径，杜绝历史极值伪装**：所有训练逻辑、分岔判定与日志严格区分：
   - `inst_E_hom`：当前 $(\alpha, m)$ 同伦松弛能量；
   - `inst_E32_true` / `inst_mean_E`：当前阴影状态 $P_A(x_t)$ 在真实 SHA-256（$\alpha=1, m=256$）下的瞬时最小 / 平均 32-bit 2-adic 物理能量；
   - `hist_min_E32`：历史累计最优值（仅用于归档与对比，**绝不**用于伪装实时训练进展或决定同伦推进）；
   - `alpha_c`：首次检测到显著分岔跃升（`bifurc`）的非线性同伦阈值；
   - `exact_hits`：严格通过完整真实前向 SHA-256 哈希验证的精确原像数量。
3. **纯血 FP32 与 A100-40GB 显存红线**：全程强制 `allow_tf32 = False`、仅用实数 `torch.float32`（零 `complex64`、零半精度 / autocast）；默认云端配置 `--gpu_tier 40g` 将峰值显存控制在 `20–32 GB`（完美适配 Colab A100-40GB，另提供 `--gpu_tier 80g` 放大并行轨迹数）。
4. **全自动云端托管与可视化闭环**：本地 RTX 4060 完成全套单测、`R=4/5` 实哈希破壁与 `R=18` 显存/步速冒烟后，打自包含 `colab_bundle_v6a/`（包内不含 Colab 授权或远程调度脚本），再由外置 `colab.exe` 会话全程托管云端 `calibrate` 与 `attack --rounds 18` 训练，同步接入 [colab_monitor_dashboard.py](file:///g:/SHA-LM/scripts/colab_monitor_dashboard.py) 实时监测面板。
5. **监控面板向下兼容**：`train.log`、`metrics.csv` 保留 `step= <cur>/<target>`、`dt`、`vram` 等既有解析字段，同时完整记录 V6A 指标：`inst_E_hom`, `inst_E32`, `inst_mean_E`, `hist_min`, `alpha`, `mask_m`, `alpha_c`, `S_ent`, `gl_gap`, `bifurc`, `exact_hits`, `drift_flag`, `dt`, `wall_s`。

---

## 二、五大审校核心数学与工程修订

1. **修订 1（防 `R <= 16` 代数剥离作弊）**：
   - 在 `int_range`（仅 `W_0` 自由、`W_1..W_15` 为已知填充）下，任意 $R \le 16$ 可直接从右边界 $S_R$ 逐轮解析减去常量直接求出 $W_0$；因此 V6A 边界模块原生支持 `R >= 4`，在 **`--mode calibrate` 标定模式** 及 **`R=4/5` 本地反置乱验真** 中强制采用 `domain_mode="free_msg"` + `allow_peel_back=False`（`W_0..W_{R-1}` 全部自由），仅在 `R=18/20` 主攻模式对确定外围位启用精确边界剥离。
2. **修订 2（相容 $\mathbb{F}_2$ 线性骨架 + 自由进位松弛 + 不动点守恒插值）**：
   - 将每个 32 位模加拆为对所有 $\alpha \in [0, 1]$ 恒成立的比特级线性奇偶校验方程（含显式 31-bit 进位松弛变量 $c_0..c_{30}$）与受 $\alpha$ 调制的非线性进位门约束（$c_k = \mathrm{Maj}(a_k, b_k, c_{k-1})$）；同样保持 `Ch`/`Maj` 输入-输出图连接为全局架构、非线性真值表约束强度由 $\alpha$ 控制。
   - **$\alpha = 0$ 零残差定理**：即使满锁 $m=256$ 位真实摘要，全图线性骨架对任意真实 SHA-256 摘要永远相容且欠定，可用精确 GF(2) 高斯消元在毫秒内采出 `inst_E_hom == 0.000` 的初始起跳解群（且真实原像 $x^*$ 严格位于此流形中）。
   - **任意 $\alpha \in [0, 1]$ 严格不动点守卫**：非线性端投影强制采用插值公式
     $$P_B^{(\text{nl}, \alpha)}(x) = (1 - \alpha) x + \alpha P_B^{(\text{nl})}(x)$$
     保证对任意 $\alpha \in [0, 1]$，若 $x^*$ 是完整 SHA-256 真迹，则 $P_B^{(\text{nl})}(x^*) = x^* \implies P_B^{(\text{nl}, \alpha)}(x^*) = x^*$，真迹在同伦全程严格保持精确不动点。
3. **修订 3（瓶颈 MPS 采用纯 FP32 2-pass MGS2 替代 `torch.linalg.qr` + 纠缠下限护栏）**：
   - 规避 CUDA cuSOLVER 小矩阵 QR 串行卡顿，双体正交截断采用 2-pass Modified Gram-Schmidt（`MGS2`），保证闭环切面（64–128 节点，$\chi \in \{8, 16\}$）单次演化 `< 50 ms`。
   - Joint-NLL 仅在汉明距离去重后的多峰精英轨迹池上更新，并叠加显式纠缠熵下限护栏 $\lambda_S \max(0, S_{\min} - S_{\text{ent}})$（$S_{\min}=0.55$），彻底根治 `S_ent -> 0` 单峰坍缩。
4. **修订 4（GL-RRR 零校验子守卫 + 低频 Top-K OSD-0）**：
   - 全局线性投影 $P_B^{(\text{lin})}$ 先执行零校验子检测（已满足 $H_{\text{lin}} b = \text{rhs}$ 的轨迹直接返回符号，100% 保证真迹不动点）；再对所有变量跑向量化 Damped Min-Sum BP（6–8 轮）；每 32 步仅对 `gl_gap` 最低的 Top-K（16–32 条）非零校验子轨迹执行紧凑位打包 OSD-0 译码，兼顾全局破环力与 GPU 吞吐。
5. **修订 5（BM 历史窗口正交子空间逃逸 + Predictor 成功步自监督）**：
   - 基于每条粒子最近 $W=8$ 步阴影历史窗口提取玻璃坑主子空间（MGS2 无 SVD），在其正交补方向施加 Lean 4 半角有理 Cayley 旋转并严格守卫 `clamped_mask`；
   - Predictor 仅用成功推进同伦的校正步自监督训练（防玻璃坑假解污染预测器），VSIDS 权重逐轨迹均值归一化为 `1.0` 且物理硬锁 $\beta \in [0.65, 1.15]$。

---

## 三、模块落盘结构

- `v6a/__init__.py` — 严格纯 FP32 运行时守卫
- `v6a/v6a_homotopy.py` — 全图线性骨架、自由进位变量建模、确定性剥离、GF(2) 精确起跳求解与 2D $(\alpha, m)$ 同伦调度
- `v6a/v6a_gl_rrr.py` — 全局稀疏 BP + Top-K OSD-0 + 精确 32-bit Viterbi + 插值不动点非线性 RRR 反射核心
- `v6a/v6a_bottleneck_mps.py` — 靶向闭环瓶颈切片 MPS（MGS2 + Joint-NLL + 纠缠护栏 + 自回归 Born 分裂）
- `v6a/v6a_bm_escape.py` — Burer-Monteiro $\mathbb{S}^{N-1}$ 历史窗口 Cayley 正交踢腿
- `v6a/v6a_navigator.py` — 无 BCE 切空间 Predictor + VSIDS 冲突聚火权重头
- `v6a/v6a_engine.py` & `train_v6a.py` — 统一训练、Phase-0 扫频标定（`R=4,5,6,8,10,12`）与 `R=18` 主攻引擎
- `configs/v6a_r18_colab_40g.yaml` & `configs/v6a_r18_colab_80g.yaml`
- `tests/test_v6a.py` & `pack_colab_v6a.py`
