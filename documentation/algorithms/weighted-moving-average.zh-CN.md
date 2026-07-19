# WeightedMovingAverage

## 摘要

`WeightedMovingAverage` 计算近期样本权重更高的加权移动平均。

## 更新 API

```python
result = rtta.WeightedMovingAverage().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现以因果方式维护滚动加权和，新近样本获得更大的线性权重。

## 递推公式

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1},\qquad y_t=G(E_t,E^{(2)}_t,\ldots,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class WeightedMovingAverage` 中实现。

## 参考资料

- [ChartSchool：Hull 移动平均](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/hull-moving-average-hma)
