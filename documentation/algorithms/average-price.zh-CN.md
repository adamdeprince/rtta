# AveragePrice

## 摘要

`AveragePrice` 计算开盘价、最高价、最低价和收盘价的平均值。

## 更新 API

```python
result = rtta.AveragePrice().update(open, high, low, close)
```

`update(...)` 每次接收 `open`、`high`、`low` 和 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

该价格变换以因果方式逐根处理 K 线，直接返回当前四个价格的算术平均。

## 递推公式

令 \(z_t = (open_t, high_t, low_t, close_t)\) 为一次更新接收的观测。

\[
AP_t = \frac{open_t + high_t + low_t + close_t}{4}
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class AveragePrice` 中实现。

## 参考资料

- [Tulip Indicators：Average Price](https://tulipindicators.org/avgprice)
