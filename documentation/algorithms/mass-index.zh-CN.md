# MassIndex

## 摘要

`MassIndex` 通过价格区间扩张识别潜在反转风险，不判断方向。

## 更新 API

```python
result = rtta.MassIndex().update(high, low)
```

`update(...)` 每次接收 `high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标对高低价差进行两次 EMA 平滑，计算两层平滑值之比并在窗口内求和，使持续的区间膨胀表现为反转风险信号。

## 递推公式

\[
R_t=high_t-low_t,\qquad E_t=\operatorname{EMA}_n(R_t),\qquad D_t=\operatorname{EMA}_n(E_t)
\]

\[
y_t=\sum_{i\in W_t}\frac{E_i}{D_i}
\]

## 组合基础组件

[`EMA`](ema.zh-CN.md)

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class MassIndex` 中实现。

## 参考资料

- [ChartSchool：Mass Index](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/mass-index)
