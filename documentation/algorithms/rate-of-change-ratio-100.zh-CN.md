# RateOfChangeRatio100

## 摘要

`RateOfChangeRatio100` 返回放大 100 倍的变动率比值。

## 更新 API

```python
result = rtta.RateOfChangeRatio100().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

对象保存指定周期的滞后收盘价，并返回当前值与滞后值之比乘以 100。

## 递推公式

\[
y_t=100\frac{close_t}{close_{t-n}}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class RateOfChangeRatio100` 中实现。

## 参考资料

- [ChartSchool：ROC 与 Momentum](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc-and-momentum)
