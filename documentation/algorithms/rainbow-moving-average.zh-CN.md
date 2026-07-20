# 彩虹移动平均（RainbowMovingAverage）

## 摘要

`RainbowMovingAverage` 是 RTTA 对 Mel Widner 彩虹线的流式实现：一组递归嵌套的简单移动平均线。每一层都是前一层的 SMA。指标返回最深层、所有层形成的包络，以及彩虹宽度。

## 更新 API

```python
result = rtta.RainbowMovingAverage(period=2, layers=10, fillna=True).update(price)
# result.outer, result.highest, result.lowest, result.mid, result.width
```

每次调用 `update(...)` 处理一个价格观测值。如果调用方只想更新状态而不生成 Python 返回值，可以用相同输入调用 `advance(...)`。数组形式的 `batch(...)` / `replay_update_outputs(...)` 遵循与 RTTA 其他结果结构体相同的多输出路径。

## 工作原理

Widner 彩虹线构建一系列嵌套平滑器，而不是一条移动平均线。第 1 层是价格的 SMA；第 \(k\) 层是第 \(k-1\) 层的 SMA。层次越深，滞后越大；因此，价格呈趋势时各层会扇形展开，价格盘整时则会收拢。RTTA 为每一层维护一个因果 `SMA` 状态，并在每个时点把前一层输出送入下一层。

## 递推公式

令 \(x_t\) 为输入价格，\(p\) 为 `period`，\(L\) 为 `layers`。

\[
S^{(1)}_t = \operatorname{SMA}_p(x_t), \qquad
S^{(k)}_t = \operatorname{SMA}_p\!\bigl(S^{(k-1)}_t\bigr)
\quad\text{其中 }k=2,\ldots,L
\]

\[
H_t = \max_{1\le k\le L} S^{(k)}_t, \qquad
L_t = \min_{1\le k\le L} S^{(k)}_t
\]

\[
\begin{aligned}
\operatorname{outer}_t &= S^{(L)}_t \\
\operatorname{mid}_t &= \tfrac12(H_t + L_t) \\
\operatorname{width}_t &= H_t - L_t
\end{aligned}
\]

当 `fillna=False` 时，在大约取得 \(p\cdot L\) 个样本之前，输出为 `NaN`（每层后续 SMA 都会增加滞后）。当 `fillna=True` 时，预热期间会输出部分 SMA 均值。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class RainbowMovingAverage` 中实现。每层都是一个嵌套 `SMA` 成员；结果结构体字段为 `outer`、`highest`、`lowest`、`mid` 和 `width`。

## 参考资料

- [Mel Widner，《Rainbow Charts》，*Technical Analysis of Stocks & Commodities*，1997 年 7 月（PDF 镜像）](https://c.mql5.com/forextsd/forum/64/rainbow_oscillator_-_original_article_-_mel_widner.pdf)
- [TradingPedia：Rainbow Oscillator](https://www.tradingpedia.com/forex-trading-indicators/rainbow-oscillator/)
