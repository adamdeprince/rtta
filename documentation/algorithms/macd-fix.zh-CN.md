# MACDFix

## 摘要

`MACDFix` 是移动平均周期固定为 12 和 26 的 MACD。

## 更新 API

```python
result = rtta.MACDFix().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现以因果方式更新固定周期的快慢 EMA 和信号平滑状态，并返回当前值。

## 递推公式

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1},\qquad y_t=G(E_t,E^{(2)}_t,\ldots,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class MACDFix` 中实现。

## 参考资料

- [ChartSchool：MACD](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator)
