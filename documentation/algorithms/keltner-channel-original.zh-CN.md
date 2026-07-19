# KeltnerChannelOriginal

## 摘要

`KeltnerChannelOriginal` 是以 SMA 和价格区间构造的原始 Keltner 通道变体。

## 更新 API

```python
result = rtta.KeltnerChannelOriginal().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现维护滚动均值与价格区间，每个样本只更新一次通道状态。

## 递推公式

\[
H_t=\max_{i\in W_t}high_i,\qquad L_t=\min_{i\in W_t}low_i
\]

\[
y_t=G(H_t,L_t,close_t)
\]

## 组合基础组件

[`SMA`](sma.zh-CN.md)

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class KeltnerChannelOriginal` 中实现。

## 参考资料

- [ChartSchool：Keltner 通道](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/keltner-channels)
