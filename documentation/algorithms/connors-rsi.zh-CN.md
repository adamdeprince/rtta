# ConnorsRSI

## 摘要

`ConnorsRSI` 是由价格 RSI、连续涨跌天数的 RSI，以及单周期价格变化的百分位排名取平均而成的复合振荡器。

## 更新 API

```python
result = rtta.ConnorsRSI().update(close)
```

`update(...)` 每次接收一个 `close`；只推进状态时可调用 `advance(...)`。

## 工作原理

指标分别维护三个严格因果的分量，再把它们归一化并组合为当前振荡值。

## 递推公式

令 \(z_t = close_t\) 为一次更新接收的观测。

\[
U_t,D_t = \operatorname{directional\_components}(z_t,z_{t-1})
\]

\[
y_t = 100\frac{\operatorname{smooth}(U_t)}{\operatorname{smooth}(U_t)+\operatorname{smooth}(D_t)}
\]

返回值为当前标量指标值。

## 实现说明

递推公式在 `src/rtta/indicator.cpp` 的 `class ConnorsRSI` 中实现。

## 参考资料

- [ChartSchool：ConnorsRSI](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/connorsrsi)
