# KalmanRegressionChannel

## 摘要

`KalmanRegressionChannel` 执行在线卡尔曼回归，并输出预测通道和价差。

## 更新 API

```python
result = rtta.KalmanRegressionChannel().update(real0, real1)
```

`update(...)` 每次接收 `real0` 和 `real1`；只推进状态时可调用 `advance(...)`。

## 工作原理

滤波器把动态回归系数视为潜在状态，每次调用执行预测和更新，再把状态投影为斜率、截距、通道及价差。

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

`update(...)` 返回含 `slope`、`intercept`、`middle`、`upper`、`lower` 和 `spread` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class KalmanRegressionChannel` 中实现。

## 参考资料

- [用卡尔曼滤波估计动态对冲比率](https://www.quantstart.com/articles/Dynamic-Hedge-Ratio-Between-ETF-Pairs-Using-the-Kalman-Filter/)
