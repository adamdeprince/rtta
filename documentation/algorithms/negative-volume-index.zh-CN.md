# NegativeVolumeIndex

## 摘要

`NegativeVolumeIndex` 是只在成交量下降周期发生变化的累积指标。

## 更新 API

```python
result = rtta.NegativeVolumeIndex().update(close, volume)
```

`update(...)` 每次接收 `close` 和 `volume`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标比较当前与上一周期成交量；只有成交量下降时，才按价格收益调整累计值。

## 递推公式

\[
PV_t=PV_{t-1}+price_t\,volume_t
\]

\[
V_t=V_{t-1}+volume_t,\qquad y_t=G(PV_t,V_t,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class NegativeVolumeIndex` 中实现。

## 参考资料

- [ChartSchool：负成交量指标 NVI](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/negative-volume-index-nvi)
