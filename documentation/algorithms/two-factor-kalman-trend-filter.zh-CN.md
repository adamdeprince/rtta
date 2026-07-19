# TwoFactorKalmanTrendFilter

## 摘要

`TwoFactorKalmanTrendFilter` 是由短期与长期两个状态构成的卡尔曼趋势贡献模型。

## 更新 API

```python
result = rtta.TwoFactorKalmanTrendFilter().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

滤波器把价格视为短期和长期潜在趋势的含噪观测，执行预测和更新后输出两个趋势分量及组合值。

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

`update(...)` 返回含 `short_trend`、`long_trend` 和 `value` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class TwoFactorKalmanTrendFilter` 中实现。

## 参考资料

- [背景论文](https://arxiv.org/pdf/1808.03297)
