# ForceIndex

## 摘要

`ForceIndex` 以价格变化乘以成交量，得到力量指数振荡值。

## 更新 API

```python
result = rtta.ForceIndex().update(close, volume)
```

`update(...)` 每次接收 `close` 和 `volume`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标把方向性价格变动与成交量结合，并以严格因果的状态计算当前振荡值。

## 递推公式

\[
PV_t=PV_{t-1}+price_t\,volume_t
\]

\[
V_t=V_{t-1}+volume_t,\qquad y_t=G(PV_t,V_t,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ForceIndex` 中实现。

## 参考资料

- [ChartSchool：Force Index](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/force-index)
