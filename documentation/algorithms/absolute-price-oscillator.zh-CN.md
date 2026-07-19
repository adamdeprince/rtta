# AbsolutePriceOscillator

## 摘要

`AbsolutePriceOscillator` 是 RTTA 对绝对价格振荡器的流式实现，以价格单位表示快速移动平均与慢速移动平均之差。

## 更新 API

```python
result = rtta.AbsolutePriceOscillator().update(close)
```

`update(...)` 每次接收一个 `close` 观测。如果只需推进状态而不创建 Python 返回值，可用相同输入调用 `advance(...)`。

## 工作原理

该指标是因果平滑器：每次只用最新观测更新紧凑的滚动或指数状态，再返回当前估计值。

## 递推公式

令 \(z_t = close_t\) 为一次 `update(...)` 接收的观测，\(\theta\) 表示窗口长度、阈值和平滑常数等构造参数。

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1}
\]

\[
y_t = G(E_t,E^{(2)}_t,\ldots,z_t)
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class AbsolutePriceOscillator` 中实现。

## 参考资料

- [背景资料：MACD](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator)
