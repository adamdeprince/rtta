# Chaikin 波动率（ChaikinVolatility）

## 摘要

`ChaikinVolatility` 是 RTTA 对下列指标的流式实现：最高价—最低价区间的 EMA 的百分比变化率。

## 更新 API

```python
result = rtta.ChaikinVolatility().update(high, low)
```

每次调用 `update(...)` 都使用 `high` 和 `low` 处理一个观测值。如果调用方只想更新状态而不生成 Python 返回值，可以用相同的输入调用 `advance(...)`。

## 工作原理

`ChaikinVolatility` 先以 EMA 平滑每根 K 线的价格区间，再报告该平滑区间相对于 `roc_window` 个样本前的百分比变化。数值上升表示区间扩张，下降则表示区间收缩。

## 递推公式

令 \(h_t,l_t\) 为最高价和最低价，\(n\) 为 EMA 窗口，\(k\) 为 ROC 回看期。

\[
R_t = h_t - l_t, \qquad
E_t = \operatorname{EMA}_n(R_t)
\]

\[
CV_t = 100 \cdot \frac{E_t - E_{t-k}}{E_{t-k}}
\]

返回值为当前的标量指标值。

## 组合使用的基础指标

[`EMA`](ema.zh-CN.md)

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class ChaikinVolatility` 中实现。

## 参考资料

- [背景资料](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/chaikin-volatility)
