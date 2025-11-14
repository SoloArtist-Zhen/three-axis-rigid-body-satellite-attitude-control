# Robust Multi-Objective 3D Attitude Control Benchmark

## 1. Problem Setup

### 1.1 System Description

This project studies a three–axis rigid-body satellite attitude control problem.  
The attitude dynamics are modeled around a small-angle operating condition, leading to a 6-dimensional linear state-space model:

\[
\dot{x} = A x + B u + B_w w,\quad x \in \mathbb{R}^6,\ u \in \mathbb{R}^3,\ w \in \mathbb{R}^3,
\]

where

- \(x = [\theta_x,\ \theta_y,\ \theta_z,\ \omega_x,\ \omega_y,\ \omega_z]^\top\) collects small attitude deviations and angular rates,
- \(u\) is the control torque vector provided by actuators,
- \(w\) is an external disturbance (e.g. environmental torques),
- \(A, B, B_w\) are obtained from a nominal inertia matrix \(J_0 = \mathrm{diag}(J_{x0}, J_{y0}, J_{z0})\) and linearization around a desired equilibrium.

The control objective is robust attitude regulation: drive the attitude and angular rate to zero despite parameter uncertainty, unmodeled dynamics, and external disturbances, while satisfying actuator constraints and achieving good transient performance.

本项目研究的是一个典型的三轴刚体卫星姿态控制问题。  
在小姿态偏差工况下，对非线性姿态动力学进行线性化，得到一个 6 维状态空间模型：

\[
\dot{x} = A x + B u + B_w w,\quad x \in \mathbb{R}^6,\ u \in \mathbb{R}^3,\ w \in \mathbb{R}^3,
\]

其中：

- \(x = [\theta_x,\ \theta_y,\ \theta_z,\ \omega_x,\ \omega_y,\ \omega_z]^\top\) 为姿态偏差与角速度；
- \(u\) 为执行机构产生的控制力矩；
- \(w\) 为外部扰动（如环境力矩）；
- \(A, B, B_w\) 由名义惯量矩阵 \(J_0 = \mathrm{diag}(J_{x0}, J_{y0}, J_{z0})\) 和平衡点附近的线性化得到。

控制目标是：在存在参数不确定性、建模误差和外部扰动的条件下，实现卫星姿态的鲁棒镇定，并兼顾执行器约束和优良的瞬态性能。

---

### 1.2 Nominal Model & Operating Conditions

The nominal model is constructed from `src/models.py` and used as the baseline for all controller designs. The operating condition corresponds to:

- Near-zero attitude error (small-angle approximation),
- Moderate angular velocity,
- Reaction wheel or torque rod type actuators with bounded torque.

The same nominal model is also used as a reference when building robustness maps and Q/R weight–sensitivity surfaces.

名义模型由 `src/models.py` 生成，并作为所有控制器设计的基准。假定工况为：

- 姿态偏差较小（小角度近似合理）；
- 角速度中等；
- 执行机构为带有力矩饱和约束的反作用飞轮或磁力矩器等。

在后续的鲁棒性地图和 Q/R 权重敏感性分析中，也统一以该名义模型为参考模型。

---

## 2. Controllers

### 2.1 LQR Controller

A standard infinite-horizon LQR controller is designed from the nominal model:

\[
J = \int_0^\infty \left( x^\top Q x + u^\top R u \right) \mathrm{d}t,
\]

with

- \(Q = \mathrm{diag}(10,10,10,1,1,1)\),
- \(R = 0.5 I_3\).

Solving the algebraic Riccati equation yields the state-feedback gain \(K_{\mathrm{LQR}}\), and the control law is

\[
u = -K_{\mathrm{LQR}} x.
\]

This serves as the baseline performance-oriented controller.

基于名义模型构造无限时域 LQR 性能指标：

\[
J = \int_0^\infty \left( x^\top Q x + u^\top R u \right) \mathrm{d}t,
\]

其中

- \(Q = \mathrm{diag}(10,10,10,1,1,1)\)，对姿态偏差给出较大权重；
- \(R = 0.5 I_3\)，对控制能量进行惩罚。

通过求解代数黎卡提方程得到状态反馈增益 \(K_{\mathrm{LQR}}\)，控制律为：

\[
u = -K_{\mathrm{LQR}} x.
\]

LQR 控制器作为性能优先的基准控制器。

---

### 2.2 H∞-type Robust Controller

To enhance robustness against disturbances and model uncertainty, an H∞-type state-feedback controller is designed via a surrogate formulation in `hinf_state_feedback_surrogate(...)`. Conceptually, it aims to minimize an upper bound on the disturbance-to-output gain:

\[
\| T_{w \to z} \|_\infty < \gamma,
\]

where \(z\) is a performance output (weighted state), and \(\gamma\) is a prescribed robustness level. The resulting gain \(K_{\mathrm{H}\infty}\) is used as:

\[
u = -K_{\mathrm{H}\infty} x.
\]

This controller emphasizes robust stability and attenuation of worst-case disturbances.

为增强系统对扰动和建模不确定性的鲁棒性，在 `hinf_state_feedback_surrogate(...)` 中构造了一个 H∞ 型状态反馈控制器。其目标是约束扰动到性能输出 \(z\) 的增益上界：

\[
\| T_{w \to z} \|_\infty < \gamma,
\]

其中 \(z\) 为加权状态输出，\(\gamma\) 为预设的鲁棒性指标。求解后得到反馈增益 \(K_{\mathrm{H}\infty}\)，控制律为：

\[
u = -K_{\mathrm{H}\infty} x.
\]

该控制器更加侧重于闭环鲁棒稳定性和最坏情况扰动抑制。

---

### 2.3 Tube-style MPC Policy

A tube-MPC-like policy is constructed using finite-horizon LQR gains:

- `finite_horizon_lqr(...)` computes a sequence of gains \(\{K_k\}_{k=0}^{N-1}\),
- A terminal gain \(K_t = K_{\mathrm{LQR}}\) is used as the inner “tube” stabilizer.

The control law is implemented in a receding-horizon fashion in `simulate_linear(...)`, with:

- Hard input constraints \(u_{\min} \le u \le u_{\max}\),
- Additive disturbances on \(B_w w\).

This policy approximates a constrained MPC controller with a tube-based nominal feedback structure.

利用 `finite_horizon_lqr(...)` 构造了一个 tube-MPC 风格的有限时域控制策略：

- 先在名义模型上求得一串有限时域 LQR 增益 \(\{K_k\}_{k=0}^{N-1}\)；
- 再将 LQR 增益 \(K_{\mathrm{LQR}}\) 作为终端/tube 内核增益 \(K_t\)。

在 `simulate_linear(...)` 中以滚动时域方式实现控制，同时施加：

- 力矩饱和约束 \(u_{\min} \le u \le u_{\max}\)；
- 加性扰动 \(B_w w\)。

这一策略在结构上接近 tube-MPC，实现了“名义滚动预测 + tube 内稳态反馈”的约束控制行为。

---

### 2.4 Gain-Scheduled Controller

To account for large variations in the inertia matrix, a simple gain-scheduling strategy is implemented:

1. Sample a set of inertia triples \(J = \mathrm{diag}(J_x, J_y, J_z)\) from a prescribed range.
2. Cluster them in the \(J_x\)-axis into three regions (low / medium / high inertia).
3. For each region, compute a local controller (alternating between LQR and H∞).
4. At runtime, choose the controller whose center \(J_c\) is closest to the current inertia estimate.

This yields a piecewise-linear scheduled controller, denoted “Sched” in the experiments.

针对惯量参数在较大范围内变化的情况，构造了一个简单的多模型调度控制器：

1. 在给定区间内采样一批惯量三元组 \(J = \mathrm{diag}(J_x, J_y, J_z)\)；
2. 按 \(J_x\) 将样本划分为三段（低 / 中 / 高惯量区间）；
3. 对每一段，以对应的代表惯量 \(J_c\) 为基准，设计一个局部控制器（在 LQR 与 H∞ 间交替）；
4. 在线根据当前惯量估计 \(J_d\) 与各 \(J_c\) 的距离，选择最近的控制器。

得到的即是一个分段线性的增益调度控制器，在实验中记为 “Sched”。

---

## 3. Uncertainty Modeling

### 3.1 Inertia Variations

The principal source of parametric uncertainty is the inertia matrix:

\[
J = \mathrm{diag}(J_x, J_y, J_z),
\]

with each diagonal entry sampled from a bounded interval around the nominal value, e.g.

- \(J_x \in [0.7, 1.3]\),
- \(J_y \in [0.8, 1.4]\),
- \(J_z \in [0.6, 1.2]\).

The function `draw_uncertainty(num, seed)` generates a collection \(\{J^{(i)}, \Delta A^{(i)}\}\), which is reused across all Monte Carlo experiments.

主要的参数不确定性来自惯量矩阵：

\[
J = \mathrm{diag}(J_x, J_y, J_z),
\]

各主惯量在名义值附近的区间内变化，例如：

- \(J_x \in [0.7, 1.3]\)，
- \(J_y \in [0.8, 1.4]\)，
- \(J_z \in [0.6, 1.2]\)。

`draw_uncertainty(num, seed)` 函数会生成一组 \(\{J^{(i)}, \Delta A^{(i)}\}\) 样本，这些样本被统一用于 Monte Carlo 仿真和鲁棒性分析。

---

### 3.2 Structured Model Perturbations

In addition to varying inertia, small structured perturbations \(\Delta A\) are injected into the state matrix to emulate unmodeled dynamics and cross-couplings. These perturbations are random but bounded and only act on selected entries of \(A\), reflecting uncertainties in coupling terms.

除了惯量的变化外，还在状态矩阵中引入了小的结构化扰动 \(\Delta A\)，用于模拟未建模动力学和耦合效应。这些扰动在若干选定元素上随机生成但幅值有界，从而反映实际系统中耦合项的不确定性。

---

## 4. Metrics

### 4.1 Time-Domain Performance

For each simulation, the following time-domain metrics are computed:

- **Integral of Squared Error (ISE)**  
  \[
  \mathrm{ISE} = \int_0^T \|x(t)\|_2^2\, \mathrm{d}t
  \]  
  measures overall regulation quality.

- **Control Energy (Eff)**  
  \[
  \mathrm{Eff} = \int_0^T \|u(t)\|_2^2\, \mathrm{d}t
  \]  
  quantifies actuation effort.

- **Settling Time (Sett)**  
  Approximated as the time when the state norm \(\|x(t)\|\) remains below a small threshold for the rest of the simulation horizon.

在每次仿真中，计算如下时域指标：

- **平方误差积分（ISE）**  
  \[
  \mathrm{ISE} = \int_0^T \|x(t)\|_2^2\, \mathrm{d}t
  \]  
  表征整体镇定误差。

- **控制能量（Eff）**  
  \[
  \mathrm{Eff} = \int_0^T \|u(t)\|_2^2\, \mathrm{d}t
  \]  
  反映执行机构能量消耗。

- **收敛时间（Sett）**  
  以状态范数 \(\|x(t)\|\) 首次进入某一小阈值并保持不再超出的时间作为近似收敛时间。

---

### 4.2 Robustness Indicators

To assess robustness, two additional metrics are used:

- **H∞-like Gain**  
  A frequency-domain proxy of \(\|T_{w\to x}\|_\infty\) is computed by sampling the resolvent over a frequency grid, providing a scalar robustness indicator.

- **Spectral Abscissa (SpecAbs)**  
  The maximum real part of the eigenvalues of the closed-loop matrix \(A_{\mathrm{cl}} = A - BK\).  
  Negative values with sufficient margin indicate robust stability.

为了评估闭环鲁棒性，引入了两类频域/稳定性指标：

- **H∞-like 增益**  
  通过在频率网格上计算闭环传递函数的谱范数，构造扰动到状态的 H∞ 增益近似，用作鲁棒性指标。

- **谱实部（Spectral Abscissa，SpecAbs）**  
  即闭环矩阵 \(A_{\mathrm{cl}} = A - BK\) 所有特征值实部的最大值。其越负，说明稳定裕度越充足。

---

## 5. Experiments

### 5.1 Monte Carlo Analysis

For each controller (LQR, H∞, tube-MPC, Sched), Monte Carlo simulations are performed:

- A subset of inertia/perturbation samples \(\{J^{(i)}, \Delta A^{(i)}\}\) is selected.
- For each sample, the corresponding closed-loop system is simulated over a fixed horizon \(T\) with:
  - additive disturbances \(w\),
  - actuator saturation,
  - and a fixed initial condition.
- The metrics (ISE, Eff, Sett, H∞-like, SpecAbs) are collected and stored in `results/mc_*.csv`.

This yields a statistical view of robust performance distribution for each controller under the same uncertainty set.

针对四类控制器（LQR、H∞、tube-MPC、Sched），执行 Monte Carlo 仿真：

- 从 \(\{J^{(i)}, \Delta A^{(i)}\}\) 中选取一部分样本；
- 对每个样本构造对应的闭环系统，并在固定仿真时域 \(T\) 内进行数值仿真，考虑：
  - 加性扰动 \(w\)；
  - 执行器饱和约束；
  - 固定初始条件；
- 计算 ISE、Eff、Sett、H∞-like、SpecAbs 等指标，并分别保存到 `results/mc_*.csv` 文件中。

通过这些结果，可以从统计意义上比较不同控制器在同一不确定性集合下的鲁棒性能分布。

---

### 5.2 Nonlinear Quaternion Simulation

To verify the relevance of linear-controller designs on the original nonlinear system, an additional test is performed using the quaternion-based nonlinear attitude model:

- The same LQR and H∞ gains are applied to the nonlinear dynamics.
- Small-angle initial conditions and realistic torque saturation are considered.
- The simulation produces quaternion components and angular rates over time.

This step validates that the linear designs remain effective in the nonlinear regime.

为了验证线性控制设计在原始非线性系统上的有效性，项目还基于四元数姿态动力学进行了额外的非线性仿真：

- 将在名义线性模型上设计的 LQR 与 H∞ 增益直接作用于非线性四元数系统；
- 采用小角度初始偏差和实际可行的力矩饱和约束；
- 仿真得到随时间变化的四元数分量和角速度。

该步骤用于检验线性控制器在非线性层面的可迁移性与有效性。

---

### 5.3 Q/R Weight Scanning

To explore multi-objective trade-offs, the project performs a scan over:

- attitude-related Q weights (e.g. \(q_{\mathrm{angle}}\)),
- control weight R scales.

For each (Q,R) pair:

- A new LQR controller is computed.
- The closed-loop system is simulated on the nominal model.
- ISE, Eff, and Sett are evaluated.

These results populate performance surfaces used later to generate heatmaps over the (Q,R) plane, revealing sensitivity and trade-offs between tracking accuracy and control effort.

为分析多目标折衷，本项目对下述权重参数进行扫描：

- 姿态相关权重 \(q_{\mathrm{angle}}\)；
- 控制权重 \(R\) 的尺度。

对每一组 (Q,R)：

- 重新求解 LQR 控制器；
- 在名义模型上进行闭环仿真；
- 计算 ISE、Eff 和 Sett 指标。

这些数据填充成 (Q,R) 平面上的性能“表面”，在可视化阶段被绘制为 Q/R 权重–性能热力图，用于揭示跟踪精度与控制能量之间的权衡关系。

---

## 6. Visualization

### 6.1 Figure Families

All figures are generated in `figures/` and grouped by type:

- `box_*.png` – boxplots of performance and robustness metrics,
- `hist_*.png` – histograms of metric distributions,
- `scat_*.png` – scatter plots showing trade-offs (e.g., ISE vs Energy),
- `heat_*.png` – heatmaps and performance surfaces over inertia and Q/R spaces,
- `line_*.png` – time responses (linear & nonlinear),
- `pole_*.png` – pole clouds in the complex plane under uncertainty.

Bar plots are explicitly avoided to keep the visual style consistent with modern robust-control papers.

所有图像保存在 `figures/` 目录中，并按类型命名：

- `box_*.png` —— 性能/鲁棒性指标的箱线图；
- `hist_*.png` —— 指标分布的直方图；
- `scat_*.png` —— ISE–Energy 等性能折衰关系的散点图；
- `heat_*.png` —— 基于惯量平面和 Q/R 权重扫描的热力图与性能表面图；
- `line_*.png` —— 线性与非线性系统的时间响应曲线；
- `pole_*.png` —— 在不确定性条件下闭环极点云的复平面分布图。

为了风格统一且更接近现代鲁棒控制文献，本项目不使用条形图，而是通过箱线图、热力图和散点图展示统计特征和结构信息。

---

### 6.2 3×3 Mosaics by Type

For each figure type, the first nine images are combined into a 3×3 mosaic using `scripts/compose_mosaic.py`:

- `mosaic_box_3x3.png`
- `mosaic_hist_3x3.png`
- `mosaic_scat_3x3.png`
- `mosaic_heat_3x3.png`
- `mosaic_line_3x3.png`
- `mosaic_pole_3x3.png`

Each mosaic contains nine figures of the same type, ensuring visual coherence and making it convenient to insert into slides or articles as a single high-level summary figure.

对每一种图形类型，选取前 9 张单图，用 `scripts/compose_mosaic.py` 中的 `stitch_3x3` 拼接成 3×3 组合图：

- `mosaic_box_3x3.png`
- `mosaic_hist_3x3.png`
- `mosaic_scat_3x3.png`
- `mosaic_heat_3x3.png`
- `mosaic_line_3x3.png`
- `mosaic_pole_3x3.png`

每张 mosaic 图只包含同一类型的 9 张子图，便于在汇报 PPT 或论文中作为“一图多信息”的总览性展示。
