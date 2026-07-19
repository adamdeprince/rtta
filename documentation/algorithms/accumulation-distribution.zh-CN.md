# AccumulationDistribution

## 摘要

`AccumulationDistribution` 是 RTTA 对量价累积/派发线的流式实现。

## 更新 API

```python
result = rtta.AccumulationDistribution().update(close, high, low, volume)
```

`update(...)` 每次接收一组 `close`、`high`、`low` 和 `volume`。如果只需推进状态，可用相同输入调用 `advance(...)`。

## 工作原理

该指标把价格与成交量组合成流式的参与度度量。更新过程严格只依赖最新 tick 和此前状态。

## 递推公式

令 \(z_t = (close_t, high_t, low_t, volume_t)\) 为一次更新接收的观测，\(\theta\) 表示窗口长度、阈值和平滑常数等构造参数。

\[
PV_t = PV_{t-1}+price_t\,volume_t
\]

\[
V_t = V_{t-1}+volume_t, \qquad y_t = G(PV_t,V_t,z_t)
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class AccumulationDistribution` 中实现。

## 参考资料

- [背景资料：累积/派发线](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/accumulation-distribution-line)
