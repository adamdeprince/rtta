# RelativeVigorIndex

## 摘要

`RelativeVigorIndex` 把平滑后的收盘减开盘动量与高低价区间比较，并计算信号线。

## 更新 API

```python
result = rtta.RelativeVigorIndex().update(open, high, low, close)
```

`update(...)` 每次接收一根 K 线的开高低收；只推进状态时可调用 `advance(...)`。

## 工作原理

实现维护滚动价格区间和动量的平滑状态，每个样本只更新一次相应统计量。

## 递推公式

\[
H_t=\max_{i\in W_t}high_i,\qquad L_t=\min_{i\in W_t}low_i
\]

\[
y_t=G(H_t,L_t,close_t)
\]

`update(...)` 返回含 `rvi` 和 `signal` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class RelativeVigorIndex` 中实现。

## 参考资料

- [背景资料：Relative Vigor Index](https://www.investopedia.com/terms/r/relative_vigor_index.asp)
