# ArnaudLegouxMovingAverage

## 摘要

`ArnaudLegouxMovingAverage` 是 RTTA 对 Arnaud Legoux 移动平均线的流式
实现。它使用高斯权重，并由偏移量和 sigma 控制权重分布。

## 更新 API

```python
result = rtta.ArnaudLegouxMovingAverage().update(value)
```

每次调用 `update(...)` 都通过 `value` 输入一个观测值。若调用者只希望更新
状态而不实例化 Python 返回值，可使用输入参数相同的 `advance(...)`。

## 工作原理

`ArnaudLegouxMovingAverage` 对最近 `window` 个样本应用固定的高斯窗口。
高斯分布的峰值位于从最旧样本起算的 `offset * (window - 1)` 处，因此较大的
`offset` 会给予近期价格更高的权重。`sigma` 控制核函数的集中程度。

## 递推公式

令 \(z_t=value_t\) 表示一次 `update(...)` 调用所输入的观测值，\(n\) 为
窗口长度，\(o\) 为偏移量，且 \(s=n/\sigma\)。

\[
m = o(n-1), \qquad
w_i = \exp\!\left(-\frac{(i-m)^2}{2s^2}\right), \quad i=0,\ldots,n-1
\]

\[
y_t = \frac{\sum_{i=0}^{n-1} w_i z_{t-n+1+i}}{\sum_{i=0}^{n-1} w_i}
\]

索引 \(i=0\) 对应窗口中最旧的样本，\(i=n-1\) 对应最新样本。窗口尚未填满
时，若 `fillna=True`，只使用现有样本及与其对应的权重。

返回值为指标当前的标量值。

## 实现说明

递推过程实现在 `src/rtta/indicator.cpp` 的
`class ArnaudLegouxMovingAverage` 中。所有权重均在构造函数中预先计算。

## 参考资料

- [TradingView：Arnaud Legoux Moving Average](https://www.tradingview.com/support/solutions/43000594683-arnaud-legoux-moving-average/)
