# KyleLambda

## 摘要

`KyleLambda` 以收益率对带符号平方根美元成交量的滚动回归斜率估计价格冲击。

## 更新 API

```python
result = rtta.KyleLambda().update(close, signed_dollar_volume)
```

`update(...)` 每次接收 `close` 和 `signed_dollar_volume`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标把价格变化与带方向的成交量组合成流式市场微观结构度量，更新只依赖最新 tick 和此前状态。

## 递推公式

\[
PV_t=PV_{t-1}+price_t\,volume_t
\]

\[
V_t=V_{t-1}+volume_t,\qquad y_t=G(PV_t,V_t,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class KyleLambda` 中实现。

## 参考资料

- [Kyle's Lambda](https://frds.io/measures/kyle_lambda/)
