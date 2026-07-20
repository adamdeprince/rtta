# 残差 BOCPD（ResidualBOCPD）

## 摘要

`ResidualBOCPD` 对残差/新息序列运行**内存有界的贝叶斯在线变点检测**（Adams 与 MacKay 风格）。当新游程长度上的后验概率质量超过阈值时，它返回二元 `signal`，并同时返回该变化概率。

## 更新 API

```python
import rtta

ind = rtta.ResidualBOCPD(
    max_run_length=128, hazard=0.01, threshold=0.5, min_variance=1e-6
)
result = ind.update(residual)
# result.signal ∈ {0, 1}, result.probability  # 更新后的 P(游程长度 = 0)
```

`advance(...)` 更新状态但不返回结果。`last_probability()` 公开的概率与 `result.probability` 相同。

## 工作原理

BOCPD 维护距最近变点的时间（游程长度）的后验分布。在常数危险率 \(h\) 下，游程长度以 \(1-h\) 的概率增长，或以 \(h\) 的概率重置。观测似然为高斯分布，每个游程长度假设都有在线均值和方差。把支持集限制为 \(\{0,\ldots,R_{\max}\}\)，可使每个时点的内存与时间复杂度保持为 \(O(R_{\max})\)。

应用于残差时，较高的 \(P(r_t=0)\) 表示：相较继续使用此前残差统计量，数据更支持开始一个新状态——也就是模型驱动的变点。

## 递推公式

令 \(R_{\max}\) 为 `max_run_length`，危险率为 \(h\)，阈值为 \(\tau\)。对每个游程长度 \(r\in\{0,\ldots,R_{\max}\}\)，维护概率 \(\pi_t(r)\)、均值 \(\mu_t(r)\)、方差 \(v_t(r)\) 和计数 \(n_t(r)\)。

首个观测：\(\pi_0(0)=1\)、\(\mu_0(0)=r_0\)、\(v_0(0)=0\)、\(n_0(0)=1\)；信号为 0。

此后，对每个满足 \(\pi_{t-1}(r)>0\) 的活跃游程 \(r\)：

\[
L_r = \mathcal{N}\!\left(r_t;\, \mu_{t-1}(r),\, \max(v_{t-1}(r), v_{\min})\right),
\quad
w_r = \pi_{t-1}(r)\, L_r.
\]

增长消息与变化消息：

\[
\tilde{\pi}_t(\min(r+1, R_{\max})) \mathrel{+}= w_r (1-h),
\quad
\tilde{\pi}_t(0) \mathrel{+}= w_r\, h.
\]

增长路径上的游程统计量采用 Welford 风格更新，计数 \(n'=\min(n_{t-1}(r)+1,R_{\max})\)：

\[
\mu' = \mu + \frac{r_t - \mu}{n'},\qquad
v' = \bigl(1 - \tfrac{1}{n'}\bigr)\bigl(v + \tfrac{1}{n'}(r_t-\mu)^2\bigr).
\]

游程 0 的变点概率质量以 \(\mu=r_t\)、\(v=0\)、\(n=1\) 初始化。将 \(\tilde\pi_t\) 归一化得到 \(\pi_t\)。输出：

\[
\mathrm{probability}_t = \pi_t(0),\qquad
\mathrm{signal}_t = \mathbf{1}\{\pi_t(0) \ge \tau\}.
\]

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class ResidualBOCPD` 中实现；它封装 `BoundedBOCPD`（`core_.update` / `last_probability`）。

## 参考资料

- [Adams 与 MacKay，《Bayesian Online Changepoint Detection》（arXiv:0710.3742）](https://arxiv.org/abs/0710.3742)
