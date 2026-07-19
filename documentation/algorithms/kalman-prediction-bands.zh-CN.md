# KalmanPredictionBands

## 摘要

`KalmanPredictionBands` 输出单步卡尔曼预测，以及由预测观测不确定性确定的上下带。

## 更新 API

```python
result = rtta.KalmanPredictionBands().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

每次调用先预测潜在状态和协方差，再用新观测更新；预测均值与观测不确定性共同构成中轨和上下带。

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

`update(...)` 返回含 `middle`、`upper` 和 `lower` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class KalmanPredictionBands` 中实现。

## 参考资料

- [卡尔曼滤波简介](https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf)
