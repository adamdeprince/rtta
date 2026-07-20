# 历史波动率（HistoricalVolatility）

## 摘要

`HistoricalVolatility` 是 RTTA 对数收益率滚动标准差年化值的流式实现。

## 更新 API

```python
result = rtta.HistoricalVolatility().update(close)
```

每次调用 `update(...)` 都使用 `close` 处理一个观测值。如果调用方只想更新状态而不生成 Python 返回值，可以用相同输入调用 `advance(...)`。

## 工作原理

`HistoricalVolatility` 使用滚动窗口内的收盘到收盘对数收益率估计已实现波动率，并乘以 \(\sqrt{\texttt{periods\_per\_year}}\) 进行年化（默认每年 252 期）。

## 递推公式

令 \(c_t=close_t\)，\(n\) 为窗口长度，\(P\) 为每年期数。

\[
r_t = \ln\frac{c_t}{c_{t-1}}
\]

\[
\sigma_t = \operatorname{stddev}_n(r_t), \qquad
HV_t = \sigma_t \sqrt{P}
\]

`stddev` 使用与 [`StdDev`](std-dev.zh-CN.md) 相同的总体标准差形式（以 \(1/n\) 计算二阶矩）。第一个样本没有收益率；当 `fillna=True` 时输出 0，否则输出 NaN。

返回值为当前的标量指标值。

## 组合使用的基础指标

[`StdDev`](std-dev.zh-CN.md)

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class HistoricalVolatility` 中实现。

## 参考资料

- [背景资料](https://www.investopedia.com/terms/h/historicalvolatility.asp)
