# WeightedClosePrice

## 摘要

`WeightedClosePrice` 以最高价、最低价和双倍收盘价计算加权收盘价。

## 更新 API

```python
result = rtta.WeightedClosePrice().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

这是逐根 K 线的因果价格变换，收盘价权重为最高价或最低价的两倍。

## 递推公式

\[
WCP_t=\frac{high_t+low_t+2close_t}{4}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class WeightedClosePrice` 中实现。

## 参考资料

- [ChartSchool：Weighted Close](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/weighted-close)
