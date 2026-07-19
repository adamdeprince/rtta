# KalmanExtremumTrend

## 摘要

`KalmanExtremumTrend` 将卡尔曼趋势估计与价格在近期高低极值区间中的随机指标式位置结合。

## 更新 API

```python
result = rtta.KalmanExtremumTrend().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

滤波器把价格流视为潜在趋势状态的含噪观测。每次调用执行预测和更新，再把滤波趋势与近期区间位置组合成信号。

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

`update(...)` 返回含 `trend`、`oscillator` 和 `signal` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class KalmanExtremumTrend` 中实现。

## 参考资料

- [背景论文](https://arxiv.org/pdf/1808.03297)
