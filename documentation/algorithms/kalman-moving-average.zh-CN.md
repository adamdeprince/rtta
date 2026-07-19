# KalmanMovingAverage

## 摘要

`KalmanMovingAverage` 使用局部线性价格/速度模型进行卡尔曼价格滤波。

## 更新 API

```python
result = rtta.KalmanMovingAverage().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

滤波器将价格流视为潜在价格和速度状态的含噪观测，每次调用执行标准预测与更新，并返回当前滤波价格。

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

递推公式在 `src/rtta/indicator.cpp` 的 `class KalmanMovingAverage` 中实现。

## 参考资料

- [背景论文](https://arxiv.org/pdf/1808.03297)
