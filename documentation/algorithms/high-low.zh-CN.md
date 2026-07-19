# HighLow

## 摘要

`HighLow` 同时返回滚动最小值和最大值。

## 更新 API

```python
result = rtta.HighLow().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现同时维护窗口两端的极值，每个样本只更新一次状态。

## 递推公式

\[
H_t=\max_{i\in W_t}high_i,\qquad L_t=\min_{i\in W_t}low_i
\]

\[
y_t=G(H_t,L_t,close_t)
\]

`update(...)` 返回含 `min` 和 `max` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class HighLow` 中实现。

## 参考资料

- [背景资料：High-Low Bands](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/high-low-bands)
