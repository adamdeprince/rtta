# 摆动指数（SwingIndex）

## 摘要

`SwingIndex` 是 RTTA 对 Welles Wilder 摆动指数的流式实现，衡量单根 K 线相对于前一根 K 线的摆动。它返回当前 K 线的 SI 增量（不是累积和）；如需运行总和，请使用 [`AccumulativeSwingIndex`](accumulative-swing-index.zh-CN.md)。

## 更新 API

```python
value = rtta.SwingIndex(limit=0.5).update(open, high, low, close)
```

`limit` 是预期最大价格变化的尺度（默认 \(0.5\)）。第一根 K 线用于初始化此前 OHLC，并返回 `0.0`。

## 工作原理

Wilder 摆动指数把当前开收盘结构与相对前一收盘价的跳空结合起来，评估一根 K 线中有多少变动属于真实摆动。结果按 `limit` 缩放；当 `limit` 设为典型的大幅移动时，不同证券的 SI 大致可比。SI 为正表示看涨摆动结构，为负表示看跌摆动结构。

## 递推公式

令 \(O_t,H_t,L_t,C_t\) 为开盘价、最高价、最低价和收盘价，\(\ell=\)`limit`（若 \(\ell\le0\)，则改用默认值 \(0.5\)）。

\[
\begin{aligned}
A_t &= |H_t - C_{t-1}|, &
B_t &= |L_t - C_{t-1}|, \\
C'_t &= |H_t - L_{t-1}|, &
D_t &= |C_{t-1} - O_{t-1}|
\end{aligned}
\]

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
\quad\text{（安全除法）}
\]

随后用当前 K 线替换此前 OHLC。

## 实现说明

该递推过程在 `src/rtta/indicator.cpp` 的 `class SwingIndex` 中实现。

## 参考资料

- [Investopedia：Accumulative Swing Index（包括 SI 定义）](https://www.investopedia.com/terms/a/asi.asp)
- [TradingPedia：Swing Index](https://www.tradingpedia.com/forex-trading-indicators/swing-index/)
