# HullMovingAverage

## 摘要

`HullMovingAverage` 计算降低滞后的加权移动平均 HMA。

## 更新 API

```python
result = rtta.HullMovingAverage(window=30).update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标组合多个因果加权平均状态，在保持平滑的同时减少响应滞后。

## 递推公式

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1},\qquad y_t=G(E_t,E^{(2)}_t,\ldots,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class HullMovingAverage` 中实现。

## 参考资料

- [ChartSchool：Hull 移动平均](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/hull-moving-average-hma)
