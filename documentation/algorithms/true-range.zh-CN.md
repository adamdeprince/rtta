# TrueRange

## 摘要

`TrueRange` 取当期高低价差，以及最高价、最低价相对上一收盘价跳空幅度中的最大值。

## 更新 API

```python
result = rtta.TrueRange().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

真实波幅在当根 K 线高低价差之外，还纳入相对上一收盘价的跨期跳空。

## 递推公式

\[
TR_t=\max(high_t-low_t,|high_t-close_{t-1}|,|low_t-close_{t-1}|)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class TrueRange` 中实现。

## 参考资料

- [ChartSchool：True Range](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/true-range)
