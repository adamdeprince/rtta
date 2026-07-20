# 正成交量指数（PositiveVolumeIndex）

## 摘要

`PositiveVolumeIndex` 是 RTTA 对仅在成交量增加期间发生变化的累积指标的流式实现。

## 更新 API

```python
result = rtta.PositiveVolumeIndex().update(close, volume)
```

每次调用 `update(...)` 都使用 `close` 和 `volume` 处理一个观测值。如果调用方只想更新状态而不生成 Python 返回值，可以用相同输入调用 `advance(...)`。

## 工作原理

`PositiveVolumeIndex` 是 [`NegativeVolumeIndex`](negative-volume-index.zh-CN.md) 的对应指标。它以 1000 为基数，只在成交量高于前一根 K 线时复合计入收盘到收盘收益率。

## 递推公式

令 \(c_t=close_t\)、\(v_t=volume_t\)，并初始化 \(PVI_0=1000\)。

\[
PVI_t =
\begin{cases}
PVI_{t-1}\left(1 + \dfrac{c_t - c_{t-1}}{c_{t-1}}\right), & v_t > v_{t-1} \\
PVI_{t-1}, & \text{其他情况}
\end{cases}
\]

返回值为当前的标量指标值。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class PositiveVolumeIndex` 中实现。

## 参考资料

- [背景资料](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/positive-volume-index)
