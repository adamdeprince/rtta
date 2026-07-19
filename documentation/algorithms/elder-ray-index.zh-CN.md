# ElderRayIndex

## 摘要

`ElderRayIndex` 以最高价和最低价相对收盘价 EMA 的距离表示多方力量与空方力量。

## 更新 API

```python
result = rtta.ElderRayIndex().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现维护所需的平滑与区间状态，每个样本只更新一次相应统计量。

## 递推公式

\[
H_t=\max_{i\in W_t}high_i,\qquad L_t=\min_{i\in W_t}low_i
\]

\[
y_t=G(H_t,L_t,close_t)
\]

`update(...)` 返回含 `bull_power` 和 `bear_power` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ElderRayIndex` 中实现。

## 参考资料

- [背景资料：Elder-Ray Index](https://www.investopedia.com/articles/trading/03/022603.asp)
