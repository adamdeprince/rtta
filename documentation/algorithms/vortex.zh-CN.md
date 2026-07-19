# Vortex

## 摘要

`Vortex` 计算正负 Vortex 趋势运动指标。

## 更新 API

```python
result = rtta.Vortex().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标分别衡量当前最高价到上一最低价、当前最低价到上一最高价的距离，并用滚动真实波幅之和归一化。

## 递推公式

\[
TR_t=\max(high_t-low_t,|high_t-close_{t-1}|,|low_t-close_{t-1}|)
\]

\[
VM^+_t=|high_t-low_{t-1}|,\qquad VM^-_t=|low_t-high_{t-1}|
\]

\[
VI^+_t=\frac{\sum_{i\in W_t}VM^+_i}{\sum_{i\in W_t}TR_i},\qquad VI^-_t=\frac{\sum_{i\in W_t}VM^-_i}{\sum_{i\in W_t}TR_i}
\]

`update(...)` 返回含 `positive`、`negative` 和 `difference` 字段的结果结构体，分别对应 \(VI^+_t\)、\(VI^-_t\) 及两者之差。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class Vortex` 中实现。

## 参考资料

- [ChartSchool：Vortex Indicator](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/vortex-indicator)
