# DoubleEMA

## 摘要

`DoubleEMA` 计算降低滞后的双重指数移动平均 DEMA。

## 更新 API

```python
result = rtta.DoubleEMA().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标以因果方式更新多层指数平滑状态，并组合这些状态以抵消部分滞后。

## 递推公式

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1}
\]

\[
y_t=G(E_t,E^{(2)}_t,\ldots,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class DoubleEMA` 中实现。

## 参考资料

- [ChartSchool：双重指数移动平均 DEMA](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/double-exponential-moving-average-dema)
