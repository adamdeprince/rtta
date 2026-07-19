# FibonacciRetracementLevels

## 摘要

`FibonacciRetracementLevels` 在近期最高价和最低价之间计算滚动 Fibonacci 回撤位。

## 更新 API

```python
result = rtta.FibonacciRetracementLevels().update(high, low)
```

`update(...)` 每次接收 `high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现维护滚动高低极值，并据当前区间计算各回撤价位。

## 递推公式

\[
H_t=\max_{i\in W_t}high_i,\qquad L_t=\min_{i\in W_t}low_i
\]

\[
y_t=G(H_t,L_t,close_t)
\]

`update(...)` 返回含 `level0`、`level236`、`level382`、`level500`、`level618` 和 `level100` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class FibonacciRetracementLevels` 中实现。

## 参考资料

- [Fidelity：Fibonacci 回撤](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/fibonacci-retracement)
