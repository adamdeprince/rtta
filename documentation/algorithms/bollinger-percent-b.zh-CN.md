# BollingerPercentB

## 摘要

`BollingerPercentB` 是 RTTA 对布林 %B 的流式实现，用于表示价格在滚动均值
和标准差通道中的相对位置。

## 更新 API

```python
result = rtta.BollingerPercentB().update(value)
```

每次调用 `update(...)` 都通过 `value` 输入一个观测值。若调用者只希望更新
状态而不实例化 Python 返回值，可使用输入相同的 `advance(...)`。

## 工作原理

`BollingerPercentB` 报告当前价格在布林带下轨与上轨之间所处的位置。接近 0
表示价格紧贴下轨，接近 1 表示紧贴上轨；超出 \([0,1]\) 的值表示价格位于通道
之外。

## 递推公式

令 \(z_t=value_t\)、\(n\) 为窗口长度、\(k\) 为标准差倍数（`num_std`，
默认值为 2）。

\[
M_t = \operatorname{mean}_n(z_t), \qquad
S_t = \operatorname{stddev}_n(z_t)
\]

\[
U_t = M_t + k S_t, \qquad
L_t = M_t - k S_t, \qquad
\%B_t = \frac{z_t - L_t}{U_t - L_t}
\]

方差与 [`BollingerBands`](bollinger-bands.zh-CN.md) 一样使用总体方差形式
（平方值的 \(1/n\) 均值）。

返回值为指标当前的标量值。

## 组合使用的基础指标

[`BollingerBands`](bollinger-bands.zh-CN.md)、[`SMA`](sma.zh-CN.md)、
[`StdDev`](std-dev.zh-CN.md)

## 实现说明

递推过程实现在 `src/rtta/indicator.cpp` 的 `class BollingerPercentB` 中。

## 参考资料

- [ChartSchool：Bollinger BandWidth 与 %B](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/bollinger-bandwidth-and-b)
