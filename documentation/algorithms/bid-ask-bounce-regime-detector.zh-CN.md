# BidAskBounceRegimeDetector

## 摘要

`BidAskBounceRegimeDetector` 用 EWMA 衡量成交在买卖两侧交替出现的频率，以识别买卖价反弹状态。

## 更新 API

```python
result = rtta.BidAskBounceRegimeDetector().update(trade_price, bid_price, ask_price)
```

`update(...)` 每次接收成交价、买价和卖价；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器先判断成交发生在报价中点哪一侧，计算成交方向是否相对上一笔发生切换，再对切换事件做 EWMA 平滑并应用回滞。

## 递推公式

令 \(z_t = (trade_price_t, bid_price_t, ask_price_t)\) 为一次更新接收的观测。

\[
side_t=\begin{cases}1, & trade_t\ge (bid_t+ask_t)/2\\ -1, & \text{否则}\end{cases}
\]

\[
b_t=\mathbf{1}[side_t\ne side_{t-1}], \qquad
q_t=\alpha b_t+(1-\alpha)q_{t-1}
\]

\[
r_t =
\begin{cases}
1, & r_{t-1} = 0 \text{ 且 } q_t \ge e \\
0, & r_{t-1} = 1 \text{ 且 } q_t \le x \\
r_{t-1}, & \text{否则}
\end{cases}, \qquad x < e
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class BidAskBounceRegimeDetector` 中实现。

## 参考资料

- [背景资料：买卖价差](https://en.wikipedia.org/wiki/Bid%E2%80%93ask_spread)
