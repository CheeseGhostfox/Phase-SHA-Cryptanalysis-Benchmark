# Phase-SHA V4 Colab A100-SXM4-80GB 训练评估、止损与机理诊断总结报告

> 实验平台：Google Colab **NVIDIA A100-SXM4-80GB**（严格纯 FP32：`allow_tf32=False`，禁用 FP16/BF16/INT8/autocast 与 `complex64`，全程兼容 `--resume` 无损续训）  
> 基准对比对象：`V3`（1D 自旋基线）、`V4A`（$\mathcal{S}^1$ 复数流形，$D=2$）、`V4B`（$\mathcal{S}^2$ Bloch `Mat2C` 流形，$D=3$）、`V4C`（$\mathcal{S}^3$ 四元数流形，$D=4$）

---

## 1. 训练执行概览与止损决策

| 模型变体 | 流形空间与维度 | 运行步数 / 目标 | 步速中位数 (`step_time_ms`) | 终止状态与决定依据 | 产物目录 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V3 Baseline** | 1D Ising ($D=1$) | `1,800`（末次评估 `@1500`） | `1738.84 ms` | 基准热启动源（[stage3b_r18_calibrated_best.pt](file:///g:/SHA-LM/checkpoints/stage3b_r18_calibrated_best.pt)） | [runs/r18_32b_v3](file:///g:/SHA-LM/runs/r18_32b_v3) |
| **V4C** (`Quat H`) | $\mathcal{S}^3 \subset \mathbb{H}$ ($D=4$) | **`10,000 / 10,000`** | `1014.42 ms` | 跑完全程；验证长程平台期（`@2000` 与 `@10000` 物理指标无实质跨越） | [runs/v4c_r18_colab](file:///g:/SHA-LM/runs/v4c_r18_colab) |
| **V4A** (`Complex`) | $\mathcal{S}^1 \subset \mathbb{C}$ ($D=2$) | **`6,200`**（末次评估 `@6000`） | `991.02 ms` | **主动止损**：`@6000` 仍卡在相同平台且劣于 V4C，停止并保留完整检查点 | [runs/v4a_r18_colab](file:///g:/SHA-LM/runs/v4a_r18_colab) |
| **V4B** (`Bloch`) | $\mathcal{S}^2$ (`Mat2C`, $D=3$) | **`2,000 / 2,000`** | `1017.75 ms` | **2,000 步探针完赛**：验证 $\mathcal{S}^2$ 非交换效应后按计划收回全套产物 | [runs/v4b_r18_colab](file:///g:/SHA-LM/runs/v4b_r18_colab) |

---

## 2. 核心实测指标对比（严守观测数据）

### 2.1 全程最优与末次评估指标对比

| 变体 | Eval Step | `train_loss` | `train_phys_loss` | `train_xor_loss` | `eval_bce_loss` | 粒子最小 32-bit 能量 `min_phys_energy_m32`（越低越好） | 全局最低 `best_global_m32_energy` | `zero_manifold_align` | `zero_manifold_comm` | `zero_w0_bit_acc` | `lsb03_zscore` | `exact_preimage_rate@256` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V3 Baseline** | `1500` (eval) | `0.7247` | `0.4381` | — | `0.6650` | `16.6229`（best `16.4715` @1000） | `11.9945`（@1500） | — | — | `0.4976` | `-0.773` | `0.0000` |
| **V4A (`S^1`)** | `6000` (last eval) | `0.3084` | `0.1514` | `0.0242` | `0.8347` | `14.6625`（best **`13.6065`** @1000） | `10.3852`（best **`8.6967`** @500） | `0.5071` | `0.0000` | `0.4952` | `-1.016` | `0.0000` |
| **V4B (`S^2`)** | `2000` (final) | `0.2805` | `0.1219` | `4.0597` | `0.8272` | `14.3027`（best **`13.4186`** @500） | `10.8961`（best **`9.5219`** @750） | `0.5179` | `2.3254` | `0.4959` | `+0.354` | `0.0000` |
| **V4C (`S^3`)** | `10000` (final) | `0.2348` | `0.0777` | `0.0069` | `0.8273` | `14.2675`（best **`13.6627`** @500） | `10.2900`（best **`7.9836`** @1250） | `0.5442` | `0.3063` | `0.4962` | `+0.044` | `0.0000` |

### 2.2 同等 2,000 步预算横向截面 (`@step 2000`)

在严格控制的同等 `2,000` 步预算下，V4A / V4B / V4C 的主要指标高度收敛到同一稳态区间：

| 变体 (`@2000`) | `eval_bce_loss` | `min_phys_energy_m32` | `best_global_m32_energy` | `zero_manifold_align` | `zero_manifold_comm` | `zero_round_acc_10` | `zero_round_acc_12` | **`zero_round_acc_13`**（闭环死墙） | **`zero_round_acc_16`**（闭环死墙） |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V4A (`S^1`)** | `0.8339` | `14.6697` | `11.5988` | `0.5122` | `0.0000` | `0.6154` | `0.6035` | **`0.4949`** | **`0.4986`** |
| **V4B (`S^2`)** | `0.8272` | `14.3027` | `10.8961` | `0.5179` | `2.3254` | `0.6589` | `0.6167` | **`0.4963`** | **`0.4963`** |
| **V4C (`S^3`)** | `0.8254` | `14.3223` | `9.9319` | `0.5524` | `0.3143` | `0.6595` | `0.6382` | **`0.4995`** | **`0.4980`** |

---

## 3. 核心机理诊断结论（为何 V4 系列齐刷刷停在 `r13/r16` 死墙）

1. **几何升维带来了可观的能量下降，但停滞在赤道“流形舒适区”**：
   - 三者均将 32-bit 物理最低能量 `min_phys_energy_m32` 从 V3 基线的 `16.47` 拉低至 **`13.42–13.66`**（单样本全局最深探到 **`7.98–9.52`**），且 `unique_ratio@256` 较 V3（`0.304`）提升至 `0.515–0.562`；
   - 然而，表征横轴能量泄漏的 `zero_manifold_align` 在三者上均从初值 `~0.79–0.80` 下降后锁死在 **`0.507–0.552`** 平台期：超半数比特向量长驻垂直于 `{+1, -1}` 离散极点的低成本软叠加态，通过连续松弛逃避了离散代数约束；
   - **V4B** 因 `Mat2C` 欠定空间未对横向虚部坐标施加完备可微惩罚，训练中 `xor_loss = 4.060` 与 `comm = 2.325` 长期居高不下；而 **V4A** (`D=2`) 因完全对易，虽能把训练 `xor_loss` 压到 `0.024–0.040`，却缺乏跨正交虚轴逃逸通道，反而 `eval_bce` 最差（`0.8347`）；**V4C** (`D=4`) 能将 `xor_loss` 压至 `0.0069` 且 `comm` 降至 `0.306`，取得了最深单粒子能量 **`7.9836`**，但在 `2,000 -> 10,000` 步内对离散求解无进一步突破。
2. **确定性校准反向退化（`eval_bce > ln(2)`）**：
   - 在 `0 -> 250` 步间，所有 V4 变体的 `eval_bce_loss` 均从 V3 热启动的 `~0.646` 快速抬升并滞留在 `~0.825–0.835`（显著高于随机猜测基线 $\ln 2 \approx 0.6931$），显示平均场边际概率头在面对 SHA-256 闭环异或耦合时被强迫输出了过度自信的错误分布。
3. **R=18 `r13 / r16` 锯齿死墙纹丝不动，直接指向 V5 理论瓶颈**：
   - 与 [V5_spec.md](file:///g:/SHA-LM/V5_spec.md) 及 Lean 4 形式化定理（Modules 6A–6D）预测完全吻合：**无论流形维数从 $D=2 \to 3 \to 4$ 如何变化，只要模型仍受限于低维固定流形、无纠缠逐比特平均场解码（独立边际符号近似）与局部梯度步进，第 13 轮（`r13`）与第 16 轮（`r16`）闭环回绕准确率就严格锁死在 `0.495–0.500`**（`zero_w0_bit_acc ≈ 0.495–0.496`，`exact_preimage_rate@256 = 0.0`）。
   - 这直接证伪了“继续在 `V4A/B/C` 上堆步数”的价值，确立了 **V5 正交单变量消融架构**（`V5D` 对偶域 RRR 全局反射、`V5C` 回路重排 MPS 纠缠网络、`V5B` 高维 $S^{N-1}$ Burer-Monteiro SDP + Cayley 逃逸、`V5A` 八元数非结合结合子流形）的研究必要性。

---

## 4. 诊断图表与归档文件清单

- 全局对比图：[runs/v4_comparison_r18_colab.png](file:///g:/SHA-LM/runs/v4_comparison_r18_colab.png)
- 六宫格机理诊断图：[runs/v4_diagnostics_r18_colab.png](file:///g:/SHA-LM/runs/v4_diagnostics_r18_colab.png)
- **V4A (`@6200` 止损归档)**：[runs/v4a_r18_colab](file:///g:/SHA-LM/runs/v4a_r18_colab)
  - 检查点与日志：[best.pt](file:///g:/SHA-LM/runs/v4a_r18_colab/best.pt)、[latest.pt](file:///g:/SHA-LM/runs/v4a_r18_colab/latest.pt)、[metrics.csv](file:///g:/SHA-LM/runs/v4a_r18_colab/metrics.csv)、[train.log](file:///g:/SHA-LM/runs/v4a_r18_colab/train.log)
  - 单体诊断图：[training_curves.png](file:///g:/SHA-LM/runs/v4a_r18_colab/training_curves.png)、[bit_zipper_heatmap.png](file:///g:/SHA-LM/runs/v4a_r18_colab/bit_zipper_heatmap.png)、[per_round_wavefront_heatmap.png](file:///g:/SHA-LM/runs/v4a_r18_colab/per_round_wavefront_heatmap.png)
- **V4B (`@2000` 完赛归档)**：[runs/v4b_r18_colab](file:///g:/SHA-LM/runs/v4b_r18_colab)
  - 检查点与日志：[best.pt](file:///g:/SHA-LM/runs/v4b_r18_colab/best.pt)、[latest.pt](file:///g:/SHA-LM/runs/v4b_r18_colab/latest.pt)、[metrics.csv](file:///g:/SHA-LM/runs/v4b_r18_colab/metrics.csv)、[train.log](file:///g:/SHA-LM/runs/v4b_r18_colab/train.log)
  - 单体诊断图：[training_curves.png](file:///g:/SHA-LM/runs/v4b_r18_colab/training_curves.png)、[bit_zipper_heatmap.png](file:///g:/SHA-LM/runs/v4b_r18_colab/bit_zipper_heatmap.png)、[per_round_wavefront_heatmap.png](file:///g:/SHA-LM/runs/v4b_r18_colab/per_round_wavefront_heatmap.png)
- **V4C (`@10000` 完赛归档)**：[runs/v4c_r18_colab](file:///g:/SHA-LM/runs/v4c_r18_colab)
  - 检查点与日志：[best.pt](file:///g:/SHA-LM/runs/v4c_r18_colab/best.pt)、[latest.pt](file:///g:/SHA-LM/runs/v4c_r18_colab/latest.pt)、[metrics.csv](file:///g:/SHA-LM/runs/v4c_r18_colab/metrics.csv)、[train.log](file:///g:/SHA-LM/runs/v4c_r18_colab/train.log)
  - 单体诊断图：[training_curves.png](file:///g:/SHA-LM/runs/v4c_r18_colab/training_curves.png)、[bit_zipper_heatmap.png](file:///g:/SHA-LM/runs/v4c_r18_colab/bit_zipper_heatmap.png)、[per_round_wavefront_heatmap.png](file:///g:/SHA-LM/runs/v4c_r18_colab/per_round_wavefront_heatmap.png)
