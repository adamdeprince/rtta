# VolumeWeightedMovingAverage

## 摘要

`VolumeWeightedMovingAverage` 计算滚动窗口内按成交量加权的收盘价移动平均 VWMA。

## 更新 API

```python
result = rtta.VolumeWeightedMovingAverage().update(close, volume)
```

`update(...)` 每次接收 `close` 和 `volume`；只推进状态时可调用 `advance(...)`。

## 工作原理

实现维护窗口内价格乘成交量之和与成交量之和，并以两者之比返回当前 VWMA。

## 递推公式

\[
PV_t=PV_{t-1}+price_t\,volume_t,\qquad V_t=V_{t-1}+volume_t,\qquad y_t=G(PV_t,V_t,z_t)
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class VolumeWeightedMovingAverage` 中实现。

## 参考资料

- [VWMA 简介](https://trendspider.com/learning-center/what-is-the-volume-weighted-moving-average-vwma/)
