# Momentum

## 摘要

`Momentum` 计算当前值与指定周期之前数值的差。

## 更新 API

```python
result = rtta.Momentum().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

对象保存所需的滞后值，每次只接收一个新观测并返回当前价格变化。

## 递推公式

\[
y_t=close_t-close_{t-n}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class Momentum` 中实现。

## 参考资料

- [ChartSchool：ROC 与 Momentum](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc-and-momentum)
