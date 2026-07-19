# ROC

## 摘要

`ROC` 以回看窗口内的百分比变化表示变动率动量。

## 更新 API

```python
result = rtta.ROC(window=10).update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

对象保存窗口长度之前的收盘价，并以严格因果方式计算当前百分比变化。

## 递推公式

\[
y_t=\frac{close_t-close_{t-n}}{close_{t-n}}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ROC` 中实现。

## 参考资料

- [ChartSchool：ROC 与 Momentum](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc-and-momentum)
