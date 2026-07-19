# UlcerIndex

## 摘要

`UlcerIndex` 是以回撤为基础的下行风险指标。

## 更新 API

```python
result = rtta.UlcerIndex().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标计算每个收盘价相对窗口最高点的百分比回撤，再取回撤平方的均方根，因此只惩罚下行偏离。

## 递推公式

\[
H_t=\max_{i\in W_t}close_i,\qquad d_t=100\frac{close_t-H_t}{H_t}
\]

\[
y_t=\sqrt{\frac1n\sum_{i\in W_t}d_i^2}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class UlcerIndex` 中实现。

## 参考资料

- [ChartSchool：Ulcer Index](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ulcer-index)
