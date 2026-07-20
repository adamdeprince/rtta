# 傅里叶—剩余类恒等式（FourierResidueIdentity）

## 摘要

`FourierResidueIdentity` 是**傅里叶—剩余类恒等式（Fourier-Residue Identity，FRI）**的流式实现。它把收益率自相关拆分为**方向（符号）**通道和**幅度**通道；两个通道都可以单独检验，且互不冗余。

它回答了标量自相关无法回答的问题：当一个序列表现出均值回归时，*可预测的是方向，还是只有变动大小？* 两种情况要求截然不同的交易。方向反转意味着“下跌一天后做多”；仅幅度反转意味着“预计明日波动较小，但方向未知”——这是波动率信号，不是方向信号，对它进行逆势方向下注完全没有统计依据。

源论文中最能说明问题的事实是：SPY 的一阶滞后自相关为 \(\hat\rho(1)=-0.081\)，低于零达 \(7.4\) 个标准误——这是实证股票金融研究中最显著的规律之一。然而，同一数据上的 FRI 符号检验却得到 \(z_{\mathrm{sign}}=-1.59\)（\(p=0.11\)）。知道 SPY 昨天下跌，基本不能帮助判断它今天会上涨还是下跌。**反弹没有方向。**

## 参考文献

V. Portnaya，*《The Bounce Has No Direction: Sign, Magnitude, and the Microstructure of Equity Return Predictability — Fourier-Residue Identities, Fejér Sums, and Evidence from US Equity and Cross-Asset Markets, 1993–2026》*，[arXiv:2606.29591](https://arxiv.org/abs/2606.29591)（2026 年 6 月）。

## 更新 API

```python
out = rtta.FourierResidueIdentity().update(close)

# 与 OHLC 兼容的重载（忽略 open/high/low）
out = rtta.FourierResidueIdentity().update(open, high, low, close)

indicator = rtta.FourierResidueIdentity()
indicator.advance(close)          # 无返回值
out = indicator.last()
indicator.reset()

batch = rtta.FourierResidueIdentity().batch(close_array)
```

构造参数：

| 参数 | 默认值 | 作用 |
|----------|---------|------|
| `max_lag` | `8` | 跟踪的滞后阶数 \(M\)；必要时自动增大，以覆盖 `horizon - 1` 和 `test_lag` |
| `horizon` | `2` | 方差比期限 \(q\) |
| `test_lag` | `1` | 标量输出所报告的滞后阶数 \(m\) |
| `span` | `512.0` | 以观测数计的 EWMA 记忆长度 |
| `median_window` | `256` | 定义 \(k=4\) 分桶边界的 \(\lvert r\rvert\) 中位数滚动窗口 |
| `entry_z` / `exit_z` | `2.0` / `1.0` | `signal` 对符号通道证据采用的迟滞阈值 |
| `fillna` | `True` | 预热期间返回 `0` 还是 `NaN` |

## 输出

| 字段 | 含义 |
|-------|---------|
| `rho` | 标量自相关 \(\hat\rho(m)\) |
| `rho_sign` | 符号通道 \(\gamma_{1,2}(m)=2p_{m,0}-1\) |
| `rho_magnitude` | 幅度通道 \(\operatorname{Re}\gamma_{1,4}(m)\) |
| `z_rho` | `rho` 的 Bartlett \(z\) 值 |
| `z_sign` | `rho_sign` 的二项分布 \(z\) 值 |
| `directional_share` | \(\lvert z_{\text{sign}}\rvert/(\lvert z_{\text{sign}}\rvert+\lvert z_\rho\rvert)\) |
| `elliptical_ratio` | `rho_sign` 除以其高斯基准（见下文） |
| `variance_ratio` | \(\mathrm{VR}(q)\) |
| `variance_ratio_sign` | 方向通道 \(\mathrm{VR}_2(q)\) |
| `variance_ratio_magnitude` | 幅度通道 \(\mathrm{VR}_4(q)\) |
| `z_variance_ratio` | Lo–MacKinlay 异方差稳健 \(z^*\) 值 |
| `persistence` | 半样本比率 \(R_N\) |
| `signal` | 由符号通道显著性把关的 `-1` / `0` / `+1` 信号 |
| `score` | \([-1,1]\) 范围内的连续方向分数 |
| `magnitude_forecast` | 条件期望 \(\mathbb E\lvert r_{t+1}\rvert\) |

## 工作原理

### Fejér / 方差比恒等式

Lo–MacKinlay 方差比具有精确的自相关表示（命题 2.2）：

\[
\mathrm{VR}(q) \;=\; 1 + 2\sum_{m=1}^{q-1}\Bigl(1 - \tfrac{m}{q}\Bigr)\hat\rho(m)
\;=\; 1 + 2\,\mathcal{C}_q
\]

Fejér 权重 \(w_m=1-m/q\) 随滞后阶数线性递减，并在滞后 \(q\) 处降为零，因此给微观结构主要所在的短滞后赋予最大权重。

### FRI 分解

把每个收益率编码成一个 \(k\) 元符号 \(s_t\in\{0,\dots,k-1\}\)，并计算循环群 \(\mathbb Z/k\mathbb Z\) 的特征标（定义 2.4）：

\[
\gamma_{A,k}(m) \;=\; \frac{1}{N-m}\sum_t \omega^{A(s_t - s_{t+m})},
\qquad \omega = e^{2\pi i/k}
\]

**符号通道（\(k=2\)）。** 令 \(s_t=\mathbb1[r_t>0]\)、\(\omega=-1\)。连续两个符号相同时特征标为 \(+1\)，不同时为 \(-1\)，从而化为闭式表达（命题 2.5）：

\[
\gamma_{1,2}(m) \;=\; 2p_{m,0} - 1 \;=:\; \hat\rho_{\mathrm{sign}}(m)
\]

其中，\(p_{m,0}\) 是相隔 \(m\) 期的收盘价位于零轴同一侧的概率。在随机游走原假设下，\(p_{m,0}=\tfrac12\)。这是一个**不含幅度信息**的方向依赖检验：正值表示动量，负值表示真正可逆势交易的方向反转。

**幅度通道（\(k=4\)）。** 以 \(\lvert r\rvert\) 的中位数为界，把收益率分入有符号的大小阶梯 \(\{\text{大跌},\text{小跌},\text{小涨},\text{大涨}\}=\{0,1,2,3\}\)，并取 \(A=1\)、\(\omega=i\)。它衡量的是*大小*分桶是否持续，而不受方向是否一致的影响。

分别对每个通道应用 Fejér 恒等式，可得到 \(\mathrm{VR}_2(q)\) 和 \(\mathrm{VR}_4(q)\)（公式 5）。两个通道并不存在嵌套关系：一个具有符号动量但没有幅度聚集的序列会有 \(\mathrm{VR}_2>1\)、\(\mathrm{VR}_4\approx1\)，反之亦然。

### 各种机制的通道特征

| 机制 | 符号 \(\mathrm{VR}_2\) | 幅度 \(\mathrm{VR}_4\) | 滞后范围 |
|---|---|---|---|
| 买卖价反跳 | \(\approx 1\) | \(<1\) | 仅滞后 1 |
| 非同步交易 | \(\approx 1\) | \(<1\) | 滞后 1–3 |
| 做市商库存 | \(\approx 1\) | \(<1\) | 滞后 1–2 |
| 逆向选择 | \(\neq1\) | \(\neq1\) | 滞后 2–5 |
| 部分价格调整 | \(\neq1\) | \(\neq1\) | 滞后 2–7 |
| 波动率聚集 | \(\approx1\) | \(>1\) | 所有滞后 |

只有 \(\mathrm{VR}_2\ne1\) 的机制才能作方向交易。

### 子样本持久性

半样本比率（定义 2.6）回答检测到的偏离能否在样本外继续存在：

\[
G_N = \max_{1\le m\le M}\lvert\hat\rho_N(m)\rvert,
\qquad R_N = G_{N/2}\,/\,G_N
\]

在 IID 噪声下，\(R_N\to\sqrt2\approx1.41\)；存在真实序列依赖时，\(R_N\to1\)（命题 2.7）。样本减半会把*噪声*最大值放大 \(\sqrt2\)，却几乎不会改变*结构性*最大值。

### 流式形式

论文使用全样本估计；本实现则是内存有界的在线估计。样本均值替换为跨度为 `span` 的去偏 EWMA（因此早期更新的表现类似扩展样本，而不是有偏斜坡），并以有效样本量 \(n_{\text{eff}}=\min(\text{count},(2-\alpha)/\alpha)\) 替代 \(n\)。

`persistence` 通过另一个跨度减半的并行估计器计算，这是 \(G_{N/2}/G_N\) 构造的流式对应形式。它只在 `span` **有限**时有意义；若跨度实际上无限，两个估计器会重合，比率退化为 1。

`z_variance_ratio` 使用 Lo–MacKinlay M2 统计量，其中 \(\hat\delta(j)\) 采用标准的 \(O(1/n)\) 归一化，因此在 IID 原假设下 \(\phi_2(q)\) 会化为 \(\phi_1(q)\)。股票日收益率具有显著的 GARCH 效应；同方差 \(z\) 检验在名义 5% 水平下会以 10–12% 的比例过度拒绝，而稳健 \(z^*\) 能保持 5% 的检验尺寸。

### 椭圆分布基准（论文之外的扩展）

符号通道并非与 \(\rho\) *毫无关系*——它具有可预测的原假设基准。对于二元正态变量对，Grothendieck 恒等式给出：

\[
\mathbb{E}[\operatorname{sgn}X \operatorname{sgn}Y] = \tfrac{2}{\pi}\arcsin\rho
\]

因此，任何自相关为 \(\rho\) 的椭圆分布过程都*必然*表现出大约 \(0.64\rho\) 的符号通道。`elliptical_ratio` 用观测到的 `rho_sign` 除以该基准，得到一个不受尺度影响、原假设值为 1 的诊断量：

- \(\approx1\)——该自相关的方向性，与具有相同 \(\rho\) 的高斯过程完全相符。
- \(\approx0\)——可预测性仅由幅度承载。

这一点很重要，因为它进一步精确化了论文自身的结论。模拟的纯 Roll 价差反跳在此处得到 **0.95**，而不是 0——反跳*确实*会泄漏到符号通道，因为对近高斯数据而言，符号相关被 \(\rho\) 所约束。SPY 真正反常之处是，其 \((\rho,\rho_{\mathrm{sign}})=(-0.081,-0.017)\) 组合只得到 **0.34**：方向性远*低于*任何具有该 \(\rho\) 的椭圆分布过程。反转集中在主导协方差的大幅波动中，而普通交易日的方向仍像抛硬币。

只有当标量 ACF 本身可以检测到（\(\lvert z_\rho\rvert\) 较大）时，才应解读 `elliptical_ratio`；当 \(\rho\) 太接近零而使比率不稳定时，返回 `NaN`。

## 递推过程

状态包括：前一收盘价；存放最近 \(M\) 个收益率及其符号和 \(k=4\) 编码的环形缓冲区；\(\lvert r\rvert\) 的滚动中位数；用于 \(\mu\)、\(\mu_2\)、\(\lvert r\rvert\)、\(\lvert r\rvert^2\) 的去偏 EWMA 对 \((v,w)\)；以及每个滞后 \(m\le M\) 的 EWMA：\(c_m\)（交叉乘积）、\(g_m\)（符号一致性）、\(h_m\)（幅度特征标的实部）、\(a_m\)（\(\lvert r\rvert\) 交叉乘积）和 \(d_m\)（四次项，用于 \(\hat\delta\)）。另有一组跨度减半的 \(\mu,\mu_2,c_m\) 并行运行，以计算 \(R_N\)。

每个去偏 EWMA 按 \(v\leftarrow(1-\alpha)v+\alpha x\)、\(w\leftarrow(1-\alpha)w+\alpha\) 累积，并报告 \(v/w\)。

1. \(r_t=\log(C_t/C_{t-1})\)；\(\sigma_t=\operatorname{sign}(r_t)\)，取 \(\pm1\)；\(\ell_t=\mathbb1[\lvert r_t\rvert>\operatorname{med}_t]\)；由 \((\sigma_t,\ell_t)\) 得出编码 \(s_t\in\{0,1,2,3\}\)。
2. 对 \(m=1\ldots\min(\text{count},M)\)，与环形槽位 \(m-1\) 中的 \(r_{t-m}\) 配对：把 \(r_t r_{t-m}\) 推入 \(c_m\)；把 \(\sigma_t\sigma_{t-m}\) 推入 \(g_m\)；通过 \((s_t-s_{t-m})\bmod4\) 的四项查找表，把 \(\cos\!\bigl(\tfrac\pi2(s_t-s_{t-m})\bigr)\) 推入 \(h_m\)；把 \(\lvert r_t r_{t-m}\rvert\) 推入 \(a_m\)；把 \((r_t-\mu)^2(r_{t-m}-\mu)^2\) 推入 \(d_m\)。
3. 把 \(r_t\) 推入全局矩 EWMA，再推入环形缓冲区。
4. \(\operatorname{Var}=\mu_2-\mu^2\)；\(\hat\rho(m)=(c_m-\mu^2)/\operatorname{Var}\)；\(\hat\rho_{\mathrm{sign}}(m)=g_m\)；\(\operatorname{Re}\gamma_{1,4}(m)=h_m\)。
5. \(n_{\text{eff}}=\min(\text{count},(2-\alpha)/\alpha)\)；\(z_\rho=\hat\rho/\sqrt{(1+2\sum_{k<m}\hat\rho(k)^2)/n_{\text{eff}}}\)；\(z_{\mathrm{sign}}=\hat\rho_{\mathrm{sign}}\sqrt{n_{\text{eff}}-m}\)。
6. 对滞后 \(1\ldots q-1\) 应用 Fejér 权重，得到 \(\mathrm{VR}\)、\(\mathrm{VR}_2\)、\(\mathrm{VR}_4\)；根据 \(\hat\delta(j)=d_j/(n_{\text{eff}}\operatorname{Var}^2)\) 累积 \(\phi_2\)。
7. \(G\) 和 \(G_{1/2}\) 分别是完整跨度与半跨度集合上 \(\lvert\hat\rho(m)\rvert\) 的运行最大值；\(R_N=G_{1/2}/G\)。
8. 分数 \(=\hat\rho_{\mathrm{sign}}\sigma_{t+1-m}\)；将 \(\lvert z_{\mathrm{sign}}\rvert\) 与 `entry_z` / `exit_z` 比较以启用或停用信号；启用时输出带符号分数。

每次更新的时间复杂度为 \(O(M)\)，其中 \(M=\)`max_lag`（默认 8），并且具有因果性。滚动中位数通过 `nth_element` 计算，时间复杂度 \(O(\text{median\_window})\)，是主要开销；其临时缓冲区会在预热期间达到完整尺寸，此后不再重新分配，因此稳态热路径不分配内存。若 \(k=4\) 的分桶边界不需要这么长的历史，可减小 `median_window`。

## 交易解读

只有当**符号**通道本身超过 `entry_z` 时，`signal` 才非零，并在 `exit_z` 处采用迟滞：

- `rho_sign < 0` 且显著 → 方向与 \(r_{t+1-m}\) 的符号相反（逆势）。
- `rho_sign > 0` 且显著 → 顺着该符号（动量）。
- 其他情况 → `0`，无论 `rho` 本身有多显著。

即使方向没有统计依据，`magnitude_forecast` 仍保留有统计依据的内容：它是下一期绝对收益率的条件预测，可用于波动率头寸规模、跨式/宽跨式时机选择，或调整 Delta 对冲账簿的风险敞口。

可以按下表综合解读两个通道：

| `z_rho` | `z_sign` | 解读 |
|---|---|---|
| 大 | 大，且同号 | 存在真正的方向依赖——交易方向 |
| 大 | 小 | 只有幅度——调整头寸规模，不要押注方向 |
| 小 | 大 | 不同幅度相互抵消，使标量 ACF 无法看到方向模式 |
| 小 | 小 | 没有可利用的结构 |

### 本指标*不能*做到什么

必须准确理解符号通道把关的局限。模拟一个纯 Roll 价差反跳（鞅式有效价格、IID 成交方向、半价差 \(0.45\sigma\)），会产生 \(\rho=-0.139\)，**同时**产生 \(\rho_{\mathrm{sign}}=-0.101\)，且 \(z_{\mathrm{sign}}=-6.5\)——符号通道极其显著。因此，`signal` 会在模拟的买卖价反跳上触发；它*不是*反跳过滤器。仅凭收盘价计算的任何量都不可能成为这种过滤器，因为观测序列确实会反转——真正使反跳不可交易的是穿越价差所付出的成本，而价格数据中并没有这个信息。

符号通道真正提供的是这种分离本身：当 \(\lvert z_\rho\rvert\) 很大而 \(\lvert z_{\mathrm{sign}}\rvert\) 不大时，可以确定普通 K 线的方向就像抛硬币，唯一有依据的操作是调整规模。真实 SPY 属于这种情况；模拟的 Roll 价差反跳则不是。

低 `elliptical_ratio` 的可操作含义在于指出优势位于*哪里*。在模拟的幅度承载型反转中（`elliptical_ratio` = 0.37），只在波动超过中位数后采取逆势仓位，交易 K 线数量减半，却保留 97% 的毛损益，单根 K 线优势几乎翻倍。在均匀方向反转中（`elliptical_ratio` = 1.00），同样的限制只能保留 78%。低比率告诉你应把风险集中在大幅波动上，而不是每根 K 线都交易。

## 注意事项

- `max_lag` 会自动增大，以同时覆盖 `horizon - 1` 与 `test_lag`，因此即使 `max_lag` 设置过小，也不会读出环形缓冲区边界。
- 即使 \(\hat\rho(3)\) 可以忽略不计，三阶滞后符号通道仍值得关注：论文发现，在标量 ACF 给出 \(p=0.50\) 的 SPY 数据上，\(z_{\mathrm{sign}}(3)=-2.32\)（\(p=0.02\)）——这是标准检验看不到的部分价格调整通道。设置 `test_lag=3` 即可读取它。
- 非有限或非正价格会被拒绝，且不会破坏估计器状态；下一个有效观测可以正常继续处理。
