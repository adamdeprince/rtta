# ChaikinOscillator

## 摘要

`ChaikinOscillator` 是对累积/派发线计算的 MACD 式振荡器。

## 更新 API

```python
result = rtta.ChaikinOscillator().update(close, high, low, volume)
```

`update(...)` 每次接收 `close`、`high`、`low` 和 `volume`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标以因果方式维护累积/派发线及其快慢平滑状态，并把两者之差映射为当前振荡值。

## 递推公式

令 \(z_t = (close_t, high_t, low_t, volume_t)\) 为一次更新接收的观测。

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1}
\]

\[
y_t = G(E_t,E^{(2)}_t,\ldots,z_t)
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ChaikinOscillator` 中实现。

## 参考资料

- [ChartSchool：Chaikin 振荡器](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/chaikin-oscillator)
