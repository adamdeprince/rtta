# 市场促进指数（MarketFacilitationIndex）

## 摘要

`MarketFacilitationIndex` 是 RTTA 逐根 K 线计算的市场促进指数：最高价减最低价的区间除以成交量。指标没有滚动窗口，也没有累积状态。

## 更新 API

```python
value = rtta.MarketFacilitationIndex().update(high, low, volume)
```

每次调用均与此前 K 线无关。成交量为零时，安全除法返回 `0`。

## 工作原理

Bill Williams 的市场促进指数（MFI，不要与资金流量指数混淆）衡量每单位成交量“促成”了多少价格区间。较小成交量伴随较大区间，表示价格容易移动（促进程度高）；较大成交量伴随较小区间，则表示价格移动遭遇较大阻力。Williams 常把 MFI 与成交量变化结合起来，为 K 线类型着色（green、fade、fake、squat）；RTTA 只返回 MFI 标量本身。

## 递推公式

令 \(H_t,L_t,V_t\) 为最高价、最低价和成交量。

\[
MFI_t = \frac{H_t - L_t}{V_t}
\quad\text{（安全除法）}
\]

不保留前一根 K 线的状态。

## 实现说明

该计算在 `src/rtta/indicator.cpp` 的 `class MarketFacilitationIndex` 中实现。

## 参考资料

- [Investopedia：Market Facilitation Index](https://www.investopedia.com/terms/m/marketfacilitationindex.asp)
- [TradingPedia：Market Facilitation Index](https://www.tradingpedia.com/forex-trading-indicators/market-facilitation-index/)
