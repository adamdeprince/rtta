# TypicalPrice

## 摘要

`TypicalPrice` 计算最高价、最低价和收盘价的平均值。

## 更新 API

```python
result = rtta.TypicalPrice().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

这是逐根 K 线的因果价格变换，直接返回当前三个价格的算术平均。

## 递推公式

\[
TP_t=\frac{high_t+low_t+close_t}{3}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class TypicalPrice` 中实现。

## 参考资料

- [ChartSchool：Typical Price](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/typical-price)
