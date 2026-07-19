# KlingerVolumeOscillator

## 摘要

`KlingerVolumeOscillator` 以快慢 EMA 和信号线构造成交量力量振荡器。

## 更新 API

```python
result = rtta.KlingerVolumeOscillator().update(close, high, low, volume)
```

`update(...)` 每次接收一组 `close`、`high`、`low` 和 `volume`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标以因果方式更新量价方向、快慢平滑值和信号线，并映射为当前振荡结果。

## 递推公式

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1},\qquad y_t=G(E_t,E^{(2)}_t,\ldots,z_t)
\]

`update(...)` 返回含 `kvo`、`signal` 和 `histogram` 字段的结果结构体。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class KlingerVolumeOscillator` 中实现。

## 参考资料

- [Klinger Oscillator 简介](https://trendspider.com/learning-center/introduction-to-klinger-oscillator/)
