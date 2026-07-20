# 心理线（PsychologicalLine）

## 摘要

`PsychologicalLine` 是 RTTA 对心理线（PSY）的流式实现：滚动窗口内收盘价高于前一收盘价的 K 线占比。

## 更新 API

```python
value = rtta.PsychologicalLine(window=12, fillna=True).update(close)
```

第一个样本没有此前收盘价，因此其上涨标志为 \(0\)。当 `fillna=False` 时，在窗口缓冲区填满之前输出为 `NaN`。

## 工作原理

PSY 是用于单一价格序列的简单市场情绪 / 类市场宽度振荡器：它统计最近 \(n\) 根 K 线中，市场“获胜”（收盘上涨）的次数。读数接近 100，表示几乎每根 K 线都收高；接近 0，则表示几乎每根 K 线都收低。经典解读把极高值视为看涨交易过度拥挤，把极低值视为看跌交易过度拥挤。

## 递推公式

令 \(c_t\) 为收盘价，\(n\) 为 `window`（默认 \(12\)）。定义上涨指示量：

\[
u_t =
\begin{cases}
1, & t > 0 \;\text{且}\; c_t > c_{t-1} \\
0, & \text{其他情况}
\end{cases}
\]

在最近 \(\min(t+1,n)\) 个标志上维护滚动和 \(U_t=\sum_{i\in W_t}u_i\)（FIFO 缓冲区）。令 \(m_t\) 为当前缓冲区大小（进入公式路径时至少为 1）：

\[
PSY_t = 100 \cdot \frac{U_t}{m_t}
\]

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class PsychologicalLine` 中实现，对缓冲区 `ups_` 使用 `rolling_sum_push`。

## 参考资料

- [TradingPedia：Psychological Line](https://www.tradingpedia.com/forex-trading-indicators/psychological-line/)
- [Investopedia：Advance/Decline 相关概念（相关情绪计数方法）](https://www.investopedia.com/terms/a/advancedecline.asp)
