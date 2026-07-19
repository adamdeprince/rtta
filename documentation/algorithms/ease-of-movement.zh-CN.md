# EaseOfMovement

## 摘要

`EaseOfMovement` 综合成交量和价格区间，衡量价格移动的难易程度。

## 更新 API

```python
result = rtta.EaseOfMovement().update(high, low, volume)
```

`update(...)` 每次接收 `high`、`low` 和 `volume`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标把价格区间与成交量组合成流式参与度度量，更新过程只依赖最新观测和此前状态。

## 递推公式

\[
PV_t=PV_{t-1}+price_t\,volume_t
\]

\[
V_t=V_{t-1}+volume_t,\qquad y_t=G(PV_t,V_t,z_t)
\]

`update(...)` 返回含 `ease_of_movement` 和 `sma` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class EaseOfMovement` 中实现。

## 参考资料

- [ChartSchool：Ease of Movement](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ease-of-movement-emv)
