# Summation

## 摘要

`Summation` 计算滚动总和。

## 更新 API

```python
result = rtta.Summation(window=30).update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

每次更新把新值加入窗口并移除过期值，同时维护当前总和。

## 递推公式

\[
W_t=\operatorname{push}(W_{t-1},z_t,n),\qquad y_t=G(W_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class Summation` 中实现。

## 参考资料

- [pandas：滚动求和](https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.sum.html)
