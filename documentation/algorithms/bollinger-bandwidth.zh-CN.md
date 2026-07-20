# BollingerBandwidth

## 摘要

`BollingerBandwidth` 是 RTTA 对布林带宽度的流式实现。它针对由滚动均值和
标准差构成的通道，计算 `(upper-lower)/middle`。

## 更新 API

```python
result = rtta.BollingerBandwidth().update(value)
```

每次调用 `update(...)` 都通过 `value` 输入一个观测值。若调用者只希望更新
状态而不实例化 Python 返回值，可使用输入相同的 `advance(...)`。

## 工作原理

`BollingerBandwidth` 用中轨对布林带上轨与下轨之间的距离进行标准化。
带宽上升表示波动率扩张，带宽下降表示波动率收缩。

## 递推公式

令 \(z_t=value_t\)、\(n\) 为窗口长度、\(k\) 为标准差倍数（`num_std`，
默认值为 2）。

\[
M_t = \operatorname{mean}_n(z_t), \qquad
S_t = \operatorname{stddev}_n(z_t)
\]

\[
BW_t = \frac{(M_t + k S_t) - (M_t - k S_t)}{M_t} = \frac{2 k S_t}{M_t}
\]

返回值为指标当前的标量值。

## 组合使用的基础指标

[`BollingerBands`](bollinger-bands.zh-CN.md)、[`SMA`](sma.zh-CN.md)、
[`StdDev`](std-dev.zh-CN.md)

## 实现说明

递推过程实现在 `src/rtta/indicator.cpp` 的 `class BollingerBandwidth` 中。

## 参考资料

- [ChartSchool：Bollinger BandWidth](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/bollinger-bandwith)
