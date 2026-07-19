# MidPrice

## 摘要

`MidPrice` 返回滚动最高价与最低价的中点。

## 更新 API

```python
result = rtta.MidPrice().update(high, low)
```

`update(...)` 每次接收 `high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现分别维护最高价和最低价序列的滚动极值，再返回两者的中点。

## 递推公式

\[
H_t=\max_{i\in W_t}high_i,\qquad L_t=\min_{i\in W_t}low_i
\]

\[
y_t=G(H_t,L_t,close_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class MidPrice` 中实现。

## 参考资料

- [背景资料：High-Low Bands](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/high-low-bands)
