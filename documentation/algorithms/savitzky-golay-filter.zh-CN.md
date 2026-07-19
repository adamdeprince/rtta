# SavitzkyGolayFilter

## 摘要

`SavitzkyGolayFilter` 是滚动多项式最小二乘平滑器，并输出一阶和二阶导数。

## 更新 API

```python
result = rtta.SavitzkyGolayFilter().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

每次把新观测加入滚动窗口并移除过期值，再在窗口内拟合局部多项式，计算平滑值及两个导数。

## 递推公式

\[
W_t=\operatorname{push}(W_{t-1},z_t,n),\qquad y_t=G(W_t)
\]

`update(...)` 返回含 `smooth`、`first_derivative` 和 `second_derivative` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class SavitzkyGolayFilter` 中实现。

## 参考资料

- [背景资料：Savitzky-Golay 滤波器](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter)
