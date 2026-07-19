# High

## 摘要

`High` 返回滚动窗口内的最高值。

## 更新 API

```python
result = rtta.High(window=30).update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现维护滚动极值，每接收一个样本只更新一次窗口状态。

## 递推公式

\[
W_t=\operatorname{push}(W_{t-1},z_t,n),\qquad y_t=G(W_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class High` 中实现。

## 参考资料

- [背景资料：滚动最高值](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/highest-high-value)
