# InteractingMultipleModelFilter

## 摘要

`InteractingMultipleModelFilter` 是四状态 IMM 卡尔曼跟踪器，按在线概率融合低波动、高波动、趋势和震荡模型。

## 更新 API

```python
result = rtta.InteractingMultipleModelFilter().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

滤波器把输入视为潜在状态的含噪观测。每次调用完成模型交互、预测和更新，再把状态与各模型概率投影到公开结果字段。

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

`update(...)` 返回含 `value`、`velocity`、`low_vol_probability`、`high_vol_probability`、`trend_probability` 和 `chop_probability` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class InteractingMultipleModelFilter` 中实现。

## 参考资料

- [背景资料：IMM](https://www.sciencedirect.com/science/article/abs/pii/S1544612316302215)
