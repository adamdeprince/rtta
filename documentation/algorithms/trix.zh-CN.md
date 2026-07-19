# Trix

## 摘要

`Trix` 是三重平滑后的变动率振荡器。

## 更新 API

```python
result = rtta.Trix().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标维护三层因果指数平滑状态，再计算最终平滑序列的单周期变动率。

## 递推公式

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1},\qquad y_t=G(E_t,E^{(2)}_t,\ldots,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class Trix` 中实现。

## 参考资料

- [ChartSchool：TRIX](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/trix)
