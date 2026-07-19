# WilliamsR

## 摘要

`WilliamsR` 是衡量超买与超卖的 Williams %R 振荡器。

## 更新 API

```python
result = rtta.WilliamsR().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标把当前收盘价在近期高低区间中的位置映射为负百分比振荡值，全部窗口状态严格因果。

## 递推公式

\[
U_t,D_t=\operatorname{directional\_components}(z_t,z_{t-1}),\qquad y_t=100\frac{\operatorname{smooth}(U_t)}{\operatorname{smooth}(U_t)+\operatorname{smooth}(D_t)}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class WilliamsR` 中实现。

## 参考资料

- [ChartSchool：Williams %R](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/williams-r)
