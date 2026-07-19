# Delay

## 摘要

`Delay` 返回固定样本数之前的滞后值。

## 更新 API

```python
result = rtta.Delay().update(value)
```

`update(...)` 每次接收一个 `value`；只推进状态时可调用 `advance(...)`。

## 工作原理

C++ 实现用环形缓冲区保存最近 (n) 个观测，每次更新返回即将被覆盖的旧值。

## 递推公式

\[
y_t=x_{t-n}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class Delay` 中实现。

## 参考资料

- [pandas：DataFrame.shift](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html)
