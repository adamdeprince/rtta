# Williams 累积/派发线（WilliamsAD）

## 摘要

`WilliamsAD` 是 RTTA 对 Williams 累积/派发线的流式实现。收盘上涨时累积一个以收盘价为基础的吸筹量，收盘下跌时累积派发量；当最高价/最低价超出前一收盘价时，计算会纳入这些极值。

## 更新 API

```python
value = rtta.WilliamsAD().update(high, low, close)
```

第一根 K 线用于初始化此前收盘价，并返回 `0.0`。收盘价不变时不增加任何值。

## 工作原理

Larry Williams 的 A/D（不同于 Chaikin 累积/派发线）在上涨日累加从该段走势真实低点到收盘价的距离，在下跌日则减去从真实高点到收盘价的距离。Williams A/D 与价格同时上升，可以确认吸筹；二者背离则可能警示参与度正在减弱。

## 递推公式

令 \(H_t,L_t,C_t\) 为最高价、最低价和收盘价。初始化 \(WAD_0=0\)，并保存 \(C_0\)。对于 \(t\ge1\)：

\[
WAD_t =
\begin{cases}
WAD_{t-1} + \bigl(C_t - \min(L_t, C_{t-1})\bigr), & C_t > C_{t-1} \\[4pt]
WAD_{t-1} + \bigl(C_t - \max(H_t, C_{t-1})\bigr), & C_t < C_{t-1} \\[4pt]
WAD_{t-1}, & C_t = C_{t-1}
\end{cases}
\]

随后把此前收盘价设为 \(C_t\)。注意，收盘下跌时增加项为负数或零，因此曲线下降。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class WilliamsAD` 中实现。

## 参考资料

- [Investopedia：Accumulation/Distribution（Williams 形式的背景）](https://www.investopedia.com/terms/a/accumulationdistribution.asp)
- [TradingPedia：Williams Accumulation/Distribution](https://www.tradingpedia.com/forex-trading-indicators/williams-accumulation-distribution/)
