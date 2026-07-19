# IntradayClockEchoSignal

## 摘要

`IntradayClockEchoSignal` 从历史聚合 K 线按日组成的数据中，学习同一时钟位置的日内收益周期性。

## 更新 API

```python
result = rtta.IntradayClockEchoSignal(fillna=True).update(open, high, low, close, volume)
```

`update(...)` 每次接收一根 OHLCV K 线；只推进状态时可调用 `advance(...)`。

## 工作原理

信号按日内时槽学习残差收益模式。每次更新可先剔除市场收益，再更新当前时槽的 EWMA 状态，预测未来若干时槽的路径，并用滚动分位数校准预测误差。链接的研究说明详述同一时钟效应和交易时段对齐假设。

## 递推公式

\[
r_t=\log(close_t/close_{t-1}),\qquad \epsilon_t=r_t-market\_return_t
\]

\[
E_{s,t}=(1-\alpha)E_{s,t-1}+\alpha\epsilon_t\quad\text{其中 }s=slot_t
\]

\[
w_j=\exp(-0.10(j-1))\min\left(1,\frac{count_{slot_t+j}}{min\_slot\_samples}\right)
\]

\[
clock\_echo_t=\frac{\sum_{j=1}^h w_jE_{slot_t+j,t}}{\sum_{j=1}^h w_j},\qquad \widehat r_{t+h}=h\cdot clock\_echo_t
\]

\[
\mathcal E_t=\{|r^{(h)}_i-\widehat r^{(h)}_i|:i+h\le t\},\qquad radius_t=\max(Q_\tau(\mathcal E_t),cost)
\]

\[
score_t=\frac{\widehat r_{t+h}}{radius_t+cost}
\]

`update(...)` 返回含 `slot`、`samples_for_slot`、`bar_return`、`residual_return`、`clock_echo`、`flow_confirm`、`volume_sync`、`prediction`、`radius`、`score`、`signal`、`target_fraction`、`max_trade_dollars`、`realized_error` 和 `ready` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class IntradayClockEchoSignal` 中实现。

## 参考资料

- [详细研究说明](../intraday_clock_echo_signal.zh-CN.md)
