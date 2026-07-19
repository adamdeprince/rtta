# UltimateOscillator

## 摘要

`UltimateOscillator` 是对多个窗口买盘压力加权组合的振荡器。

## 更新 API

```python
result = rtta.UltimateOscillator().update(close, high, low)
```

`update(...)` 每次接收 `close`、`high` 和 `low`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标在多个时间尺度上累计买盘压力与真实波幅，再以固定权重组合，全部状态按因果顺序更新。

## 递推公式

\[
U_t,D_t=\operatorname{directional\_components}(z_t,z_{t-1}),\qquad y_t=100\frac{\operatorname{smooth}(U_t)}{\operatorname{smooth}(U_t)+\operatorname{smooth}(D_t)}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class UltimateOscillator` 中实现。

## 参考资料

- [ChartSchool：Ultimate Oscillator](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ultimate-oscillator)
