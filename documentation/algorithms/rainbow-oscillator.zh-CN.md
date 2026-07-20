# 彩虹振荡器（RainbowOscillator）

## 摘要

`RainbowOscillator` 是 RTTA 对 Mel Widner 彩虹振荡器的流式实现：递归 SMA 彩虹线宽度相对于价格的百分比，以及价格在该带内的百分比位置。

## 更新 API

```python
result = rtta.RainbowOscillator(period=2, layers=10, fillna=True).update(price)
# result.value, result.position, result.width
```

每次调用 `update(...)` 处理一个价格观测值。`advance(...)` 更新状态但不返回 Python 对象。多输出 `batch(...)` 分别为 `value`、`position` 和 `width` 返回数组。

## 工作原理

该振荡器复用与 `RainbowMovingAverage` 相同的递归 SMA 堆栈。彩虹线相对于价格较宽时，不同滞后深度之间的趋势离散程度较大；较窄时，各层已趋于收敛。RTTA 还会报告价格在彩虹线最高层与最低层之间的位置，可作为标准化位置特征。

## 递推公式

令 \(x_t\) 为价格；\(H_t\)、\(L_t\)、\(\operatorname{width}_t=H_t-L_t\)、\(\operatorname{mid}_t=\tfrac12(H_t+L_t)\) 为使用相同 `period` 和 `layers` 的 `RainbowMovingAverage` 所得彩虹包络。

\[
\operatorname{value}_t = 100\cdot\frac{\operatorname{width}_t}{x_t}
\]

\[
\operatorname{position}_t = 100\cdot\frac{x_t - \operatorname{mid}_t}{\operatorname{width}_t}
\]

除数为零（彩虹线完全重合或价格为零）时，由库的安全除法处理并返回 \(0\)。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class RainbowOscillator` 中实现，内部持有一个 `RainbowMovingAverage`。

## 参考资料

- [Mel Widner，《Rainbow Charts》，*Technical Analysis of Stocks & Commodities*，1997 年 7 月（PDF 镜像）](https://c.mql5.com/forextsd/forum/64/rainbow_oscillator_-_original_article_-_mel_widner.pdf)
- [Quantified Strategies：Rainbow Oscillator](https://www.quantifiedstrategies.com/rainbow-oscillator/)
