# DetrendedPriceOscillator

## 摘要

`DetrendedPriceOscillator` 比较价格与错位移动平均，以突出价格周期并弱化长期趋势。

## 更新 API

```python
result = rtta.DetrendedPriceOscillator().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标只维护因果的滚动或平滑状态，并把近期方向性变化映射为当前振荡值。

## 递推公式

\[
U_t,D_t=\operatorname{directional\_components}(z_t,z_{t-1})
\]

\[
y_t=100\frac{\operatorname{smooth}(U_t)}{\operatorname{smooth}(U_t)+\operatorname{smooth}(D_t)}
\]

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class DetrendedPriceOscillator` 中实现。

## 参考资料

- [ChartSchool：去趋势价格振荡器 DPO](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/detrended-price-oscillator-dpo)
