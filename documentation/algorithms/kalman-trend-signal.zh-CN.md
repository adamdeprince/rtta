# KalmanTrendSignal

## 摘要

`KalmanTrendSignal` 输出卡尔曼滤波趋势线，并根据价格相对趋势线的位置生成买卖信号。

## 更新 API

```python
result = rtta.KalmanTrendSignal().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

滤波器把价格视为潜在趋势的含噪观测，执行预测和更新后，以当前价格相对滤波趋势的位置确定信号。

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

`update(...)` 返回含 `trend` 和 `signal` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class KalmanTrendSignal` 中实现。

## 参考资料

- [背景论文](https://www.aimspress.com/aimspress-data/dsfe/2024/4/PDF/DSFE-04-04-023.pdf)
