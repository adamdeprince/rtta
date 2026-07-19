# ChoppinessIndex

## 摘要

`ChoppinessIndex` 以真实波幅相对高低价区间的比例，衡量市场处于震荡还是趋势状态。

## 更新 API

```python
result = rtta.ChoppinessIndex().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标维护滚动高低极值及区间统计量，每个样本只更新一次 C++ 状态。

## 递推公式

令 \(z_t = (close_t, high_t, low_t)\) 为一次更新接收的观测。

\[
H_t=\max_{i\in W_t} high_i, \qquad L_t=\min_{i\in W_t} low_i
\]

\[
y_t = G(H_t,L_t,close_t)
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ChoppinessIndex` 中实现。

## 参考资料

- [背景资料：Choppiness Index](https://www.angelone.in/knowledge-center/online-share-trading/choppiness-index-indicator)
