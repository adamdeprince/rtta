# AnchoredVWAP

## 摘要

`AnchoredVWAP` 从任意锚点或重置事件开始累计 VWAP，而不是按固定交易时段或滚动窗口计算。

## 更新 API

```python
result = rtta.AnchoredVWAP().update(close, high, low, volume, anchor)
```

`update(...)` 每次接收 `close`、`high`、`low`、`volume` 和 `anchor`。如果只需推进状态，可用相同输入调用 `advance(...)`。

## 工作原理

每次 `update(...)` 只接收一组新的观测；若 `anchor` 触发重置，就从当前锚点重新累计，否则延续既有累计量。内部状态推进后，返回当前值。

## 递推公式

令 \(z_t = (close_t, high_t, low_t, volume_t, anchor_t)\) 为一次更新接收的观测，\(\theta\) 表示构造参数。

\[
PV_t = PV_{t-1}+price_t\,volume_t
\]

\[
V_t = V_{t-1}+volume_t, \qquad y_t = G(PV_t,V_t,z_t)
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class AnchoredVWAP` 中实现。

## 参考资料

- [ChartSchool：锚定 VWAP](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/anchored-vwap)
