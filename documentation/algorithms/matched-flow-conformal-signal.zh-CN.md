# MatchedFlowConformalSignal

## 摘要

`MatchedFlowConformalSignal` 是日内 OHLCV 匹配订单流信号，带有保形风格的滚动误差带和目标仓位诊断。

## 更新 API

```python
result = rtta.MatchedFlowConformalSignal().update(open, high, low, close, volume)
```

`update(...)` 每次接收一根 OHLCV K 线；只推进状态时可调用 `advance(...)`。

## 工作原理

这是一个复合日内研究信号。它把多尺度动量、带符号美元订单流参与度、相对 VWAP 的偏离和异常活跃度组合为持有期收益预测，再用滚动经验误差分位数缩放预测。链接的长篇研究说明讨论了相关论文脉络和保形风格校准的局限。

## 递推公式

\[
r^{(k)}_t=\log(close_t/close_{t-k}),\qquad m_t=0.20r^{(3)}_t+0.35r^{(6)}_t+0.45r^{(12)}_t
\]

\[
DV_t=close_t\max(volume_t,0),\qquad relvol_t=\frac{DV_t}{normal\_dollar\_volume_t}
\]

\[
a_t=\operatorname{sgn}(r^{(1)}_t)\frac{close_t\max(volume_t,0)}{\operatorname{scale}_t},\qquad p_t=\operatorname{sgn}(r^{(1)}_t)\frac{close_t\max(volume_t,0)}{normal\_dollar\_volume_t}
\]

若提供市值，则 (operatorname{scale}_t) 取市值；否则取正常美元成交量基线。

\[
flow_t=\tanh\left(\frac{\sum_{i\in W^{12}_t}a_i}{\alpha_{norm}}+0.5\frac{\sum_{i\in W^{6}_t}p_i}{6}\right),\qquad vwap\_gap_t=\frac{close_t}{VWAP_t}-1
\]

\[
\widehat r_{t+h}=\frac{0.35m_t+0.001flow_t+0.05vwap\_gap_t+0.0005\tanh((relvol_t-1)/2)}{1+25\max(high_t-low_t,0)/close_t}
\]

\[
\mathcal E_t=\{|r^{(h)}_i-\widehat r^{(h)}_i|:i+h\le t\},\qquad radius_t=\max(Q_\tau(\mathcal E_t),cost)
\]

\[
score_t=\frac{\widehat r_{t+h}}{radius_t+cost}
\]

`update(...)` 返回含 `prediction`、`radius`、`score`、`signal`、`target_fraction`、`alpha_flow`、`participation`、`flow_score`、`momentum`、`volatility`、`vwap_gap`、`rel_dollar_volume`、`max_trade_dollars` 和 `realized_error` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class MatchedFlowConformalSignal` 中实现。

## 参考资料

- [详细研究说明](../matched_flow_conformal_signal.zh-CN.md)
