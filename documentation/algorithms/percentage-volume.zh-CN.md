# PercentageVolume

## 摘要

`PercentageVolume` 计算百分比成交量振荡器 PVO。

## 更新 API

```python
result = rtta.PercentageVolume().update(volume)
```

`update(...)` 每次接收一个 `volume`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标以慢速平均为基准，把成交量快慢移动平均之差归一化为百分比，并计算信号线与柱状差值。

## 递推公式

\[
PV_t=PV_{t-1}+price_t\,volume_t
\]

\[
V_t=V_{t-1}+volume_t,\qquad y_t=G(PV_t,V_t,z_t)
\]

`update(...)` 返回含 `pvo`、`signal` 和 `histogram` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class PercentageVolume` 中实现。

## 参考资料

- [ChartSchool：PVO](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/percentage-volume-oscillator-pvo)
