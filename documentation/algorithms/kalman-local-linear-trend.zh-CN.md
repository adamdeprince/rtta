# KalmanLocalLinearTrend

## 摘要

`KalmanLocalLinearTrend` 是卡尔曼局部水平/趋势状态空间估计器。

## 更新 API

```python
result = rtta.KalmanLocalLinearTrend().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

滤波器把输入视为潜在水平与趋势的含噪观测。每次调用执行标准的预测和更新，再输出当前水平与趋势估计。

## 递推公式

\[
\hat x_{t|t-1}=F\hat x_{t-1|t-1},\qquad P_{t|t-1}=FP_{t-1|t-1}F^\top+Q
\]

\[
K_t=P_{t|t-1}H^\top(HP_{t|t-1}H^\top+R)^{-1}
\]

\[
\hat x_{t|t}=\hat x_{t|t-1}+K_t(z_t-H\hat x_{t|t-1}),\qquad P_{t|t}=(I-K_tH)P_{t|t-1}
\]

`update(...)` 返回含 `level` 和 `trend` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class KalmanLocalLinearTrend` 中实现。

## 参考资料

- [局部线性趋势状态空间模型](https://www.statsmodels.org/v0.12.2/examples/notebooks/generated/statespace_local_linear_trend.html)
