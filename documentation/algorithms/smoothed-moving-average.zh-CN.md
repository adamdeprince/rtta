# 平滑移动平均（SmoothedMovingAverage）

## 摘要

`SmoothedMovingAverage` 是 RTTA 对 Wilder/SMMA/RMA 平滑移动平均的流式实现，以初始 SMA 窗口为种子。

## 更新 API

```python
result = rtta.SmoothedMovingAverage().update(value)
```

每次调用 `update(...)` 都使用 `value` 处理一个观测值。如果调用方只想更新状态而不生成 Python 返回值，可以用相同输入调用 `advance(...)`。

## 工作原理

`SmoothedMovingAverage` 是 RSI 和 ATR 所用的经典 Wilder 平滑器，也称 SMMA 或 TradingView 的 RMA。第一个完整窗口取简单平均；此后的数值采用 Wilder 递归形式，有效 \(\alpha=1/n\)。

## 递推公式

令 \(z_t=value_t\)，\(n\) 为窗口长度。

前 \(n\) 个样本以简单平均初始化：

\[
S_n = \frac{1}{n}\sum_{i=0}^{n-1} z_{i+1}
\]

此后：

\[
S_t = \frac{S_{t-1}(n-1) + z_t}{n}
\]

这等价于 \(S_t=\alpha z_t+(1-\alpha)S_{t-1}\)，其中 \(\alpha=1/n\)。

当 `fillna=True` 时，在首个完整窗口之前返回部分简单平均；当 `fillna=False` 时，这些样本返回 NaN。

返回值为当前的标量指标值。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class SmoothedMovingAverage` 中实现。

## 参考资料

- [背景资料](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential)
