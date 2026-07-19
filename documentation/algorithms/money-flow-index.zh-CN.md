# MoneyFlowIndex

## 摘要

`MoneyFlowIndex` 是成交量加权、类似 RSI 的资金流振荡器。

## 更新 API

```python
result = rtta.MoneyFlowIndex().update(close, high, low, volume)
```

`update(...)` 每次接收一组 `close`、`high`、`low` 和 `volume`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标根据典型价格的方向区分正负资金流，在滚动窗口内累计并归一化。

## 递推公式

\[
PV_t=PV_{t-1}+price_t\,volume_t
\]

\[
V_t=V_{t-1}+volume_t,\qquad y_t=G(PV_t,V_t,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class MoneyFlowIndex` 中实现。

## 参考资料

- [ChartSchool：资金流量指标 MFI](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/money-flow-index-mfi)
