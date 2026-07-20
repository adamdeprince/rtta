# 随机游走指数（RandomWalkIndex）

## 摘要

`RandomWalkIndex` 是 RTTA 对随机游走指数（RWI）的流式实现。它把当前最高价/最低价极值与窗口另一侧极值进行比较，并以 ATR 乘以 \(\sqrt n\) 缩放，分别得到上侧和下侧读数。

## 更新 API

```python
result = rtta.RandomWalkIndex(window=14, fillna=True).update(close, high, low)
# result.high, result.low
```

当 `fillna=False` 时，在取得 `window` 个样本之前，两个字段均为 `NaN`。

## 工作原理

随机游走指数判断价格在 \(n\) 根 K 线内的移动距离，是否超过随机游走通常会达到的范围。在简单扩散模型下，期望距离按 \(\sigma\sqrt n\) 增长；ATR 在此替代 \(\sigma\)。RWI High 较大，表示市场从窗口低点向上移动的距离超过随机游走噪声；RWI Low 较大，则表示市场从窗口高点向下移动的距离超过噪声。大约高于 1 的值，常被解读为相应方向存在非随机趋势的证据。

## 递推公式

令 \(C_t,H_t,L_t\) 为收盘价、最高价和最低价，\(n\) 为 `window`（默认 \(14\)）。

在 \(n\) 根 K 线上维护最高价滚动最大值、最低价滚动最小值，以及相同长度的 Wilder ATR：

\[
H^{\max}_t = \max_{0\le i < n} H_{t-i}, \qquad
L^{\min}_t = \min_{0\le i < n} L_{t-i}
\]

\[
A_t = \operatorname{ATR}_n(C_t, H_t, L_t)
\]

\[
\operatorname{scale}_t = A_t \sqrt{n}
\]

\[
\begin{aligned}
\operatorname{RWI\text{-}High}_t &= \frac{H_t - L^{\min}_t}{\operatorname{scale}_t} \\
\operatorname{RWI\text{-}Low}_t &= \frac{H^{\max}_t - L_t}{\operatorname{scale}_t}
\end{aligned}
\]

（使用安全除法。）结果字段 `high` 为 RWI-High，字段 `low` 为 RWI-Low。内部 ATR 使用 `fillna=True`。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class RandomWalkIndex` 中实现，使用 `RollingExtreme` 计算最高价/最低价，并使用 `ATR`。

## 参考资料

- [Investopedia：Random Walk Index](https://www.investopedia.com/terms/r/random-walk-index.asp)
- [TradingPedia：Random Walk Index](https://www.tradingpedia.com/forex-trading-indicators/random-walk-index/)
