# CommodityChannelIndex

## 摘要

`CommodityChannelIndex` 计算典型价格相对其移动平均的 CCI 偏离程度。

## 更新 API

```python
result = rtta.CommodityChannelIndex().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标以因果方式更新典型价格及其滚动统计量，并据此返回当前偏离值。

## 递推公式

令 \(z_t = (close_t, high_t, low_t)\) 为一次更新接收的观测。

\[
H_t=\max_{i\in W_t} high_i, \qquad L_t=\min_{i\in W_t} low_i
\]

\[
y_t = G(H_t,L_t,close_t)
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class CommodityChannelIndex` 中实现。

## 参考资料

- [ChartSchool：商品通道指数 CCI](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/commodity-channel-index-cci)
