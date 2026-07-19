# LowIndex

## 摘要

`LowIndex` 返回滚动最低值在窗口内的偏移量或下标。

## 更新 API

```python
result = rtta.LowIndex().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现维护滚动最低值及其位置，每个样本只更新一次窗口状态。

## 递推公式

\[
W_t=\operatorname{push}(W_{t-1},z_t,n),\qquad y_t=G(W_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class LowIndex` 中实现。

## 参考资料

- [背景资料：距最低点的距离](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/distance-to-lows)
