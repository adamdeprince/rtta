# RateOfChangePercentage

## 摘要

`RateOfChangePercentage` 以小数表示当前值相对指定周期之前数值的变动率。

## 更新 API

```python
result = rtta.RateOfChangePercentage().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

对象保存指定周期的滞后收盘价，每次返回当前值相对该滞后值的收益比例。

## 递推公式

\[
y_t=\frac{close_t-close_{t-n}}{close_{t-n}}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class RateOfChangePercentage` 中实现。

## 参考资料

- [ChartSchool：ROC 与 Momentum](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc-and-momentum)
