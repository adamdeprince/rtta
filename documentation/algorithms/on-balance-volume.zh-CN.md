# OnBalanceVolume

## 摘要

`OnBalanceVolume` 根据收盘价方向加上或减去成交量，得到能量潮指标。

## 更新 API

```python
result = rtta.OnBalanceVolume().update(close, volume)
```

`update(...)` 每次接收 `close` 和 `volume`；只推进状态时可调用 `advance(...)`。

## 工作原理

收盘价上涨时把当期成交量加入累计值，下跌时减去，持平时保持不变；全部更新严格因果。

## 递推公式

\[
PV_t=PV_{t-1}+price_t\,volume_t
\]

\[
V_t=V_{t-1}+volume_t,\qquad y_t=G(PV_t,V_t,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class OnBalanceVolume` 中实现。

## 参考资料

- [ChartSchool：能量潮 OBV](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/on-balance-volume-obv)
