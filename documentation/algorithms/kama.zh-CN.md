# Kama

## 摘要

`Kama` 计算 Kaufman 自适应移动平均。

## 更新 API

```python
result = rtta.Kama().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标根据价格路径的效率调整平滑速度，再用最新观测更新紧凑的因果状态。

## 递推公式

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1},\qquad y_t=G(E_t,E^{(2)}_t,\ldots,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class Kama` 中实现。

## 参考资料

- [ChartSchool：Kaufman 自适应移动平均](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/kaufmans-adaptive-moving-average-kama)
