# MedianPrice

## 摘要

`MedianPrice` 计算最高价和最低价的平均值。

## 更新 API

```python
result = rtta.MedianPrice().update(high, low)
```

`update(...)` 每次接收 `high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

这是逐根 K 线的因果价格变换，直接返回当前高低价中点。

## 递推公式

\[
MP_t=\frac{high_t+low_t}{2}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class MedianPrice` 中实现。

## 参考资料

- [ChartSchool：Median Price](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/median-price)
