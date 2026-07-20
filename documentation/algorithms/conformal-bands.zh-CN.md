# 保形预测带（ConformalBands）

## 摘要

`ConformalBands` 是 RTTA 对拆分保形式预测带的流式实现：以 SMA 为中心，以单步绝对残差的滚动分位数为半径。它是一个轻量级预测带基础组件，并非完整的自适应保形推断系统。

## 更新 API

```python
result = rtta.ConformalBands(window=20, alpha=0.1, fillna=True).update(value)
# result.middle, result.upper, result.lower, result.radius
```

`alpha` 是目标未覆盖率；残差分位点为 \(1-\alpha\)。多输出 `batch(...)` 为 `middle`、`upper`、`lower` 和 `radius` 分别返回数组。

## 工作原理

保形预测通过近期数据上的不合度分数来校准预测集合。对于简单的单步预测，一种常用分数是绝对残差 \(\lvert y_t - \hat y_{t\mid t-1}\rvert\)。RTTA 使用前一个 SMA 值作为 \(\hat y\)，将绝对残差保存在滚动窗口中，并取 \(1-\alpha\) 水平的分位数作为当前 SMA 中心周围的半径。这个方法有意设计得比存在分布漂移时的自适应保形推断更简单；它是一条因果的滚动校准带。

## 递推公式

令 \(x_t\) 为输入，\(n\) 为 `window`，\(q = 1-\alpha\)。

1. 若已有前一个预测值 \(\hat x_{t-1}\)，将残差 \(s_t = \lvert x_t - \hat x_{t-1}\rvert\) 推入容量为 \(n\) 的滚动分位数存储区。
2. 更新中心：

\[
m_t = \operatorname{SMA}_n(x_t)
\]

3. 令 \(\hat x_t \leftarrow m_t\)，供下一次计算残差。
4. 令 \(R_t\) 为已存绝对残差在 \(q\) 水平上的经验分位数（RTTA 的 `RollingQuantile`）。则：

\[
\begin{aligned}
\operatorname{middle}_t &= m_t \\
\operatorname{radius}_t &= R_t \\
\operatorname{upper}_t &= m_t + R_t \\
\operatorname{lower}_t &= m_t - R_t
\end{aligned}
\]

当 `fillna=False` 时，在大约取得 \(n\) 个样本之前，各输出均为 `NaN`。尚无残差时，半径按 \(0\) 处理。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class ConformalBands` 中实现，使用 `SMA` 和 `RollingQuantile`。如需一个同样以残差分位数确定规模、但内容更丰富的 OHLCV 复合指标，请参阅 `MatchedFlowConformalSignal`。

## 参考资料

- [Xu 与 Xie，《Sequential Predictive Conformal Inference for Time Series》，arXiv:2212.03463](https://arxiv.org/abs/2212.03463)
- [Gibbs 与 Candès，《Adaptive Conformal Inference Under Distribution Shift》，arXiv:2106.00170](https://arxiv.org/abs/2106.00170)
