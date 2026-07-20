# Chande 预测振荡器（ChandeForecastOscillator）

## 摘要

`ChandeForecastOscillator` 是 RTTA 对 Tushar Chande 预测振荡器的流式实现：当前收盘价与线性回归时间序列单步预测值（TSF）之间的百分比距离。

## 更新 API

```python
value = rtta.ChandeForecastOscillator(window=14, fillna=True).update(close)
```

每次调用 `update(...)` 处理一个收盘价。`advance(...)` 使用相同输入，但不返回 Python 值。标量 `batch(...)` 返回一个 NumPy 数组。

## 工作原理

对最近 \(n\) 个收盘价作滚动最小二乘直线拟合，可得到向前一根 K 线的预测值（也就是 `TimeSeriesForecast` 所提供的量）。预测振荡器以价格百分比衡量实时收盘价高于或低于该外推预测值的程度。正值表示价格高于拟合外推值，负值表示低于拟合外推值。

## 递推公式

令 \(x_t\) 为收盘价，\(n\) 为 `window`。以 \(0,\ldots,n-1\) 为横坐标，对最近 \(n\) 个样本组成的滚动窗口拟合：

\[
\hat\beta_t = (X^\top X)^{-1} X^\top y_t
\]

并构造单步预测：

\[
\operatorname{TSF}_t = [1,\, n]\,\hat\beta_t
\]

（与 RTTA 的 `LinearRegressionCore` 一致，其中 `tsf` 等于截距加斜率乘以 \(n\)）。振荡器为：

\[
\operatorname{CFO}_t = 100\cdot\frac{x_t - \operatorname{TSF}_t}{x_t}.
\]

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class ChandeForecastOscillator` 中实现，并使用 `LinearRegressionCore` 及其 `tsf` 字段。

## 参考资料

- [Investopedia：Forecast Oscillator](https://www.investopedia.com/terms/f/forecasto.asp)
- [ChartSchool：Time Series Forecast / Linear Regression](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/slope)
