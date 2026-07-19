# QuoteStuffingDetector

## 摘要

`QuoteStuffingDetector` 用报价消息与成交消息之比的 EWMA 检测报价填塞。

## 更新 API

```python
result = rtta.QuoteStuffingDetector().update(quote_messages, trades)
```

`update(...)` 每次接收 `quote_messages` 和 `trades`；只推进状态时可调用 `advance(...)`。

## 工作原理

检测器计算非负报价消息数相对成交消息数的比率，对其进行 EWMA 平滑，再以回滞生成稳定的填塞状态。

## 递推公式

\[
\rho_t=\frac{\max(quote\_messages_t,0)}{\max(\max(trades_t,0),\epsilon)},\qquad q_t=\alpha\rho_t+(1-\alpha)q_{t-1}
\]

\[
r_t=\begin{cases}1,&r_{t-1}=0\text{ 且 }q_t\ge e\\0,&r_{t-1}=1\text{ 且 }q_t\le x\\r_{t-1},&\text{否则}\end{cases},\qquad x<e
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class QuoteStuffingDetector` 中实现。

## 参考资料

- [背景资料：报价填塞](https://en.wikipedia.org/wiki/Quote_stuffing)
