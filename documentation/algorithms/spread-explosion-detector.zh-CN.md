# SpreadExplosionDetector

## 摘要

`SpreadExplosionDetector` 以相对 EWMA 基线检测报价价差突然扩大。

## 更新 API

```python
result = rtta.SpreadExplosionDetector().update(bid_price, ask_price)
```

`update(...)` 每次接收买价和卖价；只推进状态时可调用 `advance(...)`。

## 工作原理

当前非负价差先除以上一期 EWMA 基线，再用上侧回滞生成持续的价差爆发状态。

## 递推公式

\[
s_t=\max(ask_t-bid_t,0),\qquad q_t=\frac{s_t}{\max(B_{t-1},\epsilon)},\qquad B_t=\alpha s_t+(1-\alpha)B_{t-1}
\]

\[
r_t=\begin{cases}1,&r_{t-1}=0\text{ 且 }q_t\ge e\\0,&r_{t-1}=1\text{ 且 }q_t\le x\\r_{t-1},&\text{否则}\end{cases},\qquad x<e
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class SpreadExplosionDetector` 中实现。

## 参考资料

- [背景资料：买卖价差](https://en.wikipedia.org/wiki/Bid%E2%80%93ask_spread)
