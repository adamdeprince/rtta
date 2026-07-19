# KalmanInnovationZScore

## 摘要

`KalmanInnovationZScore` 用预测创新标准差对带符号的观测创新进行归一化。

## 更新 API

```python
result = rtta.KalmanInnovationZScore().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

滤波器把收盘价视为潜在状态的含噪观测。每次预测后，以观测误差相对预测创新方差的标准化结果作为输出。

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

递推公式在 `src/rtta/indicator.cpp` 的 `class KalmanInnovationZScore` 中实现。

## 参考资料

- [卡尔曼滤波简介](https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf)
