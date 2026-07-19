# RateOfChangeRatio

## 摘要

`RateOfChangeRatio` 返回当前值与指定周期之前数值的比率。

## 更新 API

```python
result = rtta.RateOfChangeRatio().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

对象保存指定周期的滞后收盘价，并以严格因果方式计算当前值相对滞后值的比率。

## 递推公式

\[
y_t=\frac{close_t}{close_{t-n}}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class RateOfChangeRatio` 中实现。

## 参考资料

- [ChartSchool：ROC 与 Momentum](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc-and-momentum)
