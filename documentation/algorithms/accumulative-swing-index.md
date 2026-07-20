# AccumulativeSwingIndex

## Summary

`AccumulativeSwingIndex` is RTTA's streaming cumulative sum of Wilder's Swing
Index. Each bar contributes one SI increment; the indicator returns the running
total, which tracks net directional swing pressure over the full history.

## Update API

```python
value = rtta.AccumulativeSwingIndex(limit=0.5).update(open, high, low, close)
```

The first bar seeds previous OHLC and returns `0.0`. Every subsequent bar adds
the current Swing Index increment to the running total.

## Theory Of Operation

Welles Wilder's Swing Index measures how much of a bar's open/close structure
represents a genuine swing relative to the prior bar, scaled by a limit (the
largest expected price change). Accumulative Swing Index (ASI) is simply the
indefinite sum of those increments, analogous to how AD or OBV accumulate bar
contributions. Positive ASI growth suggests net bullish swing structure;
negative growth suggests net bearish structure.

RTTA composes ASI from an internal `SwingIndex` member and a scalar accumulator.

## Recurrence

Let \(O_t, H_t, L_t, C_t\) be open, high, low, close. Let \(\ell\) be `limit`
(default \(0.5\)). Seed previous OHLC on the first bar and set \(ASI_0 = 0\).

Define absolute gaps versus the previous bar:

\[
\begin{aligned}
A_t &= |H_t - C_{t-1}|, &
B_t &= |L_t - C_{t-1}|, \\
C'_t &= |H_t - L_{t-1}|, &
D_t &= |C_{t-1} - O_{t-1}|
\end{aligned}
\]

The denominator factor \(R_t\) is chosen by the largest of \(A_t, B_t, C'_t\):

\[
R_t =
\begin{cases}
A_t - \tfrac12 B_t + \tfrac14 D_t, & A_t \ge B_t \;\text{and}\; A_t \ge C'_t \\
B_t - \tfrac12 A_t + \tfrac14 D_t, & B_t \ge A_t \;\text{and}\; B_t \ge C'_t \\
C'_t + \tfrac14 D_t, & \text{otherwise}
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
\quad\text{(safe divide; \(0\) if denominator is zero)}
\]

\[
ASI_t = ASI_{t-1} + SI_t
\]

The returned value is \(ASI_t\).

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class AccumulativeSwingIndex`, which owns a `SwingIndex` member and accumulates
its scalar output. See also [`SwingIndex`](swing-index.md).

## Reference

- [Investopedia: Accumulative Swing Index (ASI)](https://www.investopedia.com/terms/a/asi.asp)
- [TradingPedia: Accumulative Swing Index](https://www.tradingpedia.com/forex-trading-indicators/accumulative-swing-index/)
