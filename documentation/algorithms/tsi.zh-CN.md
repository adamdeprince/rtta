# TSI

## 摘要

`TSI` 是对动量进行两次平滑的真实强度指数振荡器。

## 更新 API

```python
result = rtta.TSI().update(x)
```

`update(...)` 每次接收一个 `x`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标分别对带符号动量和绝对动量做双重因果平滑，再以两者之比得到当前振荡值。

## 递推公式

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1},\qquad y_t=G(E_t,E^{(2)}_t,\ldots,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class TSI` 中实现。

## 参考资料

- [ChartSchool：真实强度指数](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/true-strength-index)
