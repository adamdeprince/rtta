# 投影振荡器（ProjectionOscillator）

## 摘要

`ProjectionOscillator` 是 RTTA 对投影振荡器的流式实现：它以随机振荡器的方式，衡量收盘价在线性回归最高价和最低价投影带之间的位置，并提供一条短周期 SMA 信号线。

## 更新 API

```python
result = rtta.ProjectionOscillator(
    window=14, signal_window=3, fillna=True
).update(high, low, close)
# result.value, result.signal, result.upper, result.lower
```

每次调用 `update(...)` 处理 `high`、`low` 和 `close`。`advance(...)` 更新状态但不返回 Python 对象。多输出 `batch(...)` 分别为 `value`、`signal`、`upper` 和 `lower` 返回数组。

## 工作原理

投影带分别对最高价和最低价作滚动最小二乘直线拟合。当前窗口末端的拟合值构成一条随时间变化的通道。把收盘价映射为该通道内的百分比位置（类似随机振荡器对最高价/最低价极值所作的映射），便得到尊重局部回归斜率、而非原始窗口极值的振荡器。信号线是该振荡器的短周期 SMA。

## 递推公式

令 \(h_t,\ell_t,c_t\) 为最高价、最低价和收盘价，\(n\) 为 `window`。分别对最高价与最低价序列拟合滚动线性回归，并取当前窗口末端的拟合值（RTTA `LinearRegressionCore` 的 `value` 字段）：

\[
U_t = \operatorname{LinReg}_n(h)_t, \qquad
L_t = \operatorname{LinReg}_n(\ell)_t
\]

\[
\operatorname{PO}_t = 100\cdot\frac{c_t - L_t}{U_t - L_t}
\]

\[
\operatorname{signal}_t = \operatorname{SMA}_{n_s}(\operatorname{PO}_t)
\]

其中 \(n_s\) 为 `signal_window`。投影带宽为零时，安全除法返回 \(0\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class ProjectionOscillator` 中实现，使用两个 `LinearRegressionCore` 成员和一个计算信号线的 `SMA`。

## 参考资料

- [Investopedia：Projection Oscillator](https://www.investopedia.com/terms/p/projectionoscillator.asp)
- [ChartSchool：Linear Regression](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/slope)
