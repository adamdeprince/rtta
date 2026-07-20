# AccumulativeSwingIndex

## 摘要

`AccumulativeSwingIndex` 是 RTTA 对 Wilder 摆动指数进行流式累加的实现。
每根 K 线贡献一个 SI 增量，指标返回持续累积的总和，用来跟踪全部历史中的净方向性
摆动压力。

## 更新 API

```python
value = rtta.AccumulativeSwingIndex(limit=0.5).update(open, high, low, close)
```

第一根 K 线用于初始化前一根 OHLC，并返回 `0.0`。此后每根 K 线都会将当前的
Swing Index 增量加入累计值。

## 工作原理

Welles Wilder 的 Swing Index 衡量一根 K 线的开盘与收盘结构中，有多少相对于
前一根 K 线构成了真正的摆动，并用一个限制值（预期的最大价格变化）进行缩放。
Accumulative Swing Index（ASI）就是这些增量的不定期累加，类似于 AD 或 OBV
累加每根 K 线的贡献。ASI 持续上升表示净看涨摆动结构，持续下降则表示净看跌
摆动结构。

RTTA 使用一个内部 `SwingIndex` 成员和一个标量累加器组合实现 ASI。

## 递推公式

令 \(O_t,H_t,L_t,C_t\) 分别为开盘价、最高价、最低价和收盘价；令 \(\ell\)
为 `limit`（默认 \(0.5\)）。第一根 K 线只初始化前一根 OHLC，并令
\(ASI_0=0\)。

定义相对于前一根 K 线的绝对跳空幅度：

\[
\begin{aligned}
A_t &= |H_t - C_{t-1}|, &
B_t &= |L_t - C_{t-1}|, \\
C'_t &= |H_t - L_{t-1}|, &
D_t &= |C_{t-1} - O_{t-1}|
\end{aligned}
\]

根据 \(A_t,B_t,C'_t\) 中的最大者选择分母因子 \(R_t\)：

\[
R_t =
\begin{cases}
A_t - \tfrac12 B_t + \tfrac14 D_t, & A_t \ge B_t \;\text{且}\; A_t \ge C'_t \\
B_t - \tfrac12 A_t + \tfrac14 D_t, & B_t \ge A_t \;\text{且}\; B_t \ge C'_t \\
C'_t + \tfrac14 D_t, & \text{其他情况}
\end{cases}
\]

\[
K_t = \max(A_t, B_t)
\]

\[
N_t = (C_t - C_{t-1}) + \tfrac12(C_t - O_t) + \tfrac14(C_{t-1} - O_{t-1})
\]

\[
SI_t = \frac{50\, N_t\, K_t}{\ell\, R_t}
\quad\text{（安全除法；分母为零时结果为 \(0\)）}
\]

\[
ASI_t = ASI_{t-1} + SI_t
\]

返回值为 \(ASI_t\)。

## 实现说明

递推过程实现在 `src/rtta/indicator.cpp` 的
`class AccumulativeSwingIndex` 中。该类持有一个 `SwingIndex` 成员，并累加
它的标量输出。另请参阅 [`SwingIndex`](swing-index.zh-CN.md)。

## 参考资料

- [Investopedia：Accumulative Swing Index（ASI）](https://www.investopedia.com/terms/a/asi.asp)
- [TradingPedia：Accumulative Swing Index](https://www.tradingpedia.com/forex-trading-indicators/accumulative-swing-index/)
