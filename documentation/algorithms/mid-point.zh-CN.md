# MidPoint

## 摘要

`MidPoint` 返回单一序列滚动最高值与最低值的中点。

## 更新 API

```python
result = rtta.MidPoint().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现维护滚动极值，每个样本只更新一次窗口状态，再返回区间中点。

## 递推公式

\[
H_t=\max_{i\in W_t}high_i,\qquad L_t=\min_{i\in W_t}low_i
\]

\[
y_t=G(H_t,L_t,close_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class MidPoint` 中实现。

## 参考资料

- [VectorAlpha：Midpoint](https://vectoralpha.dev/projects/ta/indicators/midpoint/)
