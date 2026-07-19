# KalmanVelocityOscillator

## 摘要

`KalmanVelocityOscillator` 返回常速度卡尔曼价格模型中以零为中心的速度状态。

## 更新 API

```python
result = rtta.KalmanVelocityOscillator().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

滤波器把价格视为潜在价格与速度状态的含噪观测；预测和更新后，公开速度分量作为振荡值。

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

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class KalmanVelocityOscillator` 中实现。

## 参考资料

- [背景论文](https://arxiv.org/pdf/1808.03297)
