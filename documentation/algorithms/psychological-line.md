# PsychologicalLine

## Summary

`PsychologicalLine` is RTTA's streaming Psychological Line (PSY): the percentage
of bars in a rolling window where close rose versus the prior close.

## Update API

```python
value = rtta.PsychologicalLine(window=12, fillna=True).update(close)
```

The first sample has no prior close, so its up-flag is \(0\). With
`fillna=False`, output is `NaN` until the window buffer is full.

## Theory Of Operation

PSY is a simple market-sentiment / breadth-style oscillator for a single price
series: it counts how often the market "won" (up close) over the last \(n\)
bars. Readings near 100 mean almost every bar closed higher; near 0 means almost
every bar closed lower. The classic interpretation treats extreme highs as
crowded bullish conditions and extreme lows as crowded bearish conditions.

## Recurrence

Let \(c_t\) be close and \(n\) be `window` (default \(12\)). Define the up
indicator:

\[
u_t =
\begin{cases}
1, & t > 0 \;\text{and}\; c_t > c_{t-1} \\
0, & \text{otherwise}
\end{cases}
\]

Maintain a rolling sum \(U_t = \sum_{i \in W_t} u_i\) over the last
\(\min(t+1, n)\) flags (FIFO buffer). Let \(m_t\) be the current buffer size
(at least 1 in the formula path):

\[
PSY_t = 100 \cdot \frac{U_t}{m_t}
\]

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class PsychologicalLine` using `rolling_sum_push` on buffer `ups_`.

## Reference

- [TradingPedia: Psychological Line](https://www.tradingpedia.com/forex-trading-indicators/psychological-line/)
- [Investopedia: Advance/Decline concepts (related sentiment counting)](https://www.investopedia.com/terms/a/advancedecline.asp)
