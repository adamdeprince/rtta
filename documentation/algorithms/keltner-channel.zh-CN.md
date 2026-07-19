# KeltnerChannel

## 摘要

`KeltnerChannel` 是以 EMA 为中轨、ATR 为宽度的波动率通道。

## 更新 API

```python
result = rtta.KeltnerChannel().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现同时维护收盘价 EMA 和真实波幅 ATR，每根 K 线只更新一次通道状态。

## 递推公式

\[
H_t=\max_{i\in W_t}high_i,\qquad L_t=\min_{i\in W_t}low_i
\]

\[
y_t=G(H_t,L_t,close_t)
\]

`update(...)` 返回含 `middle`、`upper` 和 `lower` 字段的结果结构体。

## 组合基础组件

[`EMA`](ema.zh-CN.md)、[`ATR`](atr.zh-CN.md)

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class KeltnerChannel` 中实现。

## 参考资料

- [ChartSchool：Keltner 通道](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/keltner-channels)
