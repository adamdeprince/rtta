# 成交量振荡器（VolumeOscillator）

## 摘要

`VolumeOscillator` 是 RTTA 对成交量短期简单移动平均与长期简单移动平均百分比差的流式实现。

## 更新 API

```python
result = rtta.VolumeOscillator().update(volume)
```

每次调用 `update(...)` 都使用 `volume` 处理一个观测值。如果调用方只想更新状态而不生成 Python 返回值，可以用相同输入调用 `advance(...)`。

## 工作原理

`VolumeOscillator` 衡量短期成交量位于长期成交量基准之上还是之下。它是价格百分比振荡器在成交量上的 SMA 版本（不同于使用 EMA 和信号线的 [`PercentageVolume`](percentage-volume.zh-CN.md)）。

## 递推公式

令 \(v_t=volume_t\)，\(n_s\) 为短期窗口，\(n_l\) 为长期窗口。

\[
S_t = \operatorname{SMA}_{n_s}(v_t), \qquad
L_t = \operatorname{SMA}_{n_l}(v_t)
\]

\[
VO_t = 100 \cdot \frac{S_t - L_t}{L_t}
\]

返回值为当前的标量指标值。

## 组合使用的基础指标

[`SMA`](sma.zh-CN.md)

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class VolumeOscillator` 中实现。

## 参考资料

- [背景资料](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/percentage-volume-oscillator-pvo)
