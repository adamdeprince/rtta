# VolumeWeightedAveragePrice

## 摘要

`VolumeWeightedAveragePrice` 计算按成交量加权的价格 VWAP。

## 更新 API

```python
result = rtta.VolumeWeightedAveragePrice().update(close, high, low, volume)
```

`update(...)` 每次接收一组 `close`、`high`、`low` 和 `volume`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现累计价格乘成交量与成交量本身，并以两者之比返回当前 VWAP。

## 递推公式

\[
PV_t=PV_{t-1}+price_t\,volume_t,\qquad V_t=V_{t-1}+volume_t,\qquad y_t=G(PV_t,V_t,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class VolumeWeightedAveragePrice` 中实现。

## 参考资料

- [ChartSchool：VWAP](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/volume-weighted-average-price-vwap)
