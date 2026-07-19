# AlphaBetaGammaTrackingFilter

## 摘要

`AlphaBetaGammaTrackingFilter` 是一种稳态、类卡尔曼的价格、速度和加速度跟踪器。

## 更新 API

```python
result = rtta.AlphaBetaGammaTrackingFilter().update(close)
```

`update(...)` 每次接收一个 `close`。如果只需推进状态，可用相同输入调用 `advance(...)`。

## 工作原理

该滤波器把输入流视为潜在状态的含噪观测。每次调用执行标准的预测与更新步骤，再把更新后的状态投影到公开的结果字段。

## 递推公式

令 \(z_t = close_t\) 为一次更新接收的观测，\(\theta\) 表示构造参数。

\[
\hat{x}_{t|t-1}=F\hat{x}_{t-1|t-1}, \qquad
P_{t|t-1}=FP_{t-1|t-1}F^\top+Q
\]

\[
K_t=P_{t|t-1}H^\top(HP_{t|t-1}H^\top+R)^{-1}
\]

\[
\hat{x}_{t|t}=\hat{x}_{t|t-1}+K_t(z_t-H\hat{x}_{t|t-1}), \qquad
P_{t|t}=(I-K_tH)P_{t|t-1}
\]

`update(...)` 返回含有 `price`、`velocity`、`acceleration` 和 `residual` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class AlphaBetaGammaTrackingFilter` 中实现。

## 参考资料

- [背景资料：Alpha-Beta 滤波器](https://kalmanfilter.net/alphabeta.html)
