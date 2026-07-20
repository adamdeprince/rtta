# SwingIndex

## Summary

`SwingIndex` is RTTA's streaming Welles Wilder Swing Index for a single bar
relative to the prior bar. It returns the bar's SI increment (not the cumulative
sum); use [`AccumulativeSwingIndex`](accumulative-swing-index.md) for the running
total.

## Update API

```python
value = rtta.SwingIndex(limit=0.5).update(open, high, low, close)
```

`limit` is the maximum expected price change scale (default \(0.5\)). The first
bar seeds previous OHLC and returns `0.0`.

## Theory Of Operation

Wilder's Swing Index combines the current open/close structure with gaps versus
the previous close to score how much of the bar is a genuine swing. The result
is scaled by `limit` so that SI is roughly comparable across instruments when
`limit` is set to a typical large move. Positive SI indicates bullish swing
structure; negative SI indicates bearish structure.

## Recurrence

Let \(O_t, H_t, L_t, C_t\) be open, high, low, close and \(\ell =\) `limit`
(\(\ell > 0\), else defaulted to \(0.5\)).

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
\quad\text{(safe divide)}
\]

Then previous OHLC is replaced by the current bar.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class SwingIndex`.

## Reference

- [Investopedia: Accumulative Swing Index (includes SI definition)](https://www.investopedia.com/terms/a/asi.asp)
- [TradingPedia: Swing Index](https://www.tradingpedia.com/forex-trading-indicators/swing-index/)
