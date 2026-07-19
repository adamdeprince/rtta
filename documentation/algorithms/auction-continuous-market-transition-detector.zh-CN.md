# AuctionContinuousMarketTransitionDetector

## 摘要

`AuctionContinuousMarketTransitionDetector` 使用回滞逻辑检测集合竞价与连续交易阶段之间的切换。

## 更新 API

```python
result = rtta.AuctionContinuousMarketTransitionDetector().update(auction_signal)
```

`update(...)` 每次接收一个 `auction_signal`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器先把当前观测转换为标量市场状态指标，再应用进入/退出回滞。输出会保持稳定，直到指标越过相反方向的退出阈值。

## 递推公式

令 \(z_t = auction_signal_t\) 为一次更新接收的观测，\(\theta\) 表示构造参数。

\[
q_t=auction\_signal_t
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

递推公式在 `src/rtta/indicator.cpp` 的 `class AuctionContinuousMarketTransitionDetector` 中实现。

## 参考资料

- [背景资料：集合竞价市场](https://en.wikipedia.org/wiki/Call_market)
