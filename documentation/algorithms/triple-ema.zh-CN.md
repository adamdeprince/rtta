# TripleEMA

## 摘要

`TripleEMA` 计算降低滞后的三重指数移动平均 TEMA。

## 更新 API

```python
result = rtta.TripleEMA().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标以因果方式维护三层 EMA，并组合这些平滑值以抵消部分滞后。

## 递推公式

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1},\qquad y_t=G(E_t,E^{(2)}_t,\ldots,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class TripleEMA` 中实现。

## 参考资料

- [ChartSchool：TEMA](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/triple-exponential-moving-average-tema)
