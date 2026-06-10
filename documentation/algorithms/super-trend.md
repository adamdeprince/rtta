# SuperTrend

## Summary

`SuperTrend` is RTTA's streaming implementation of: ATR-band trend-following indicator.

## Update API

```python
result = rtta.SuperTrend().update(close, high, low)
```

The `update(...)` call consumes one observation using `close`, `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`SuperTrend` builds ATR-scaled bands around the high/low midpoint and trails the active band in the direction of the current trend. Crosses through the active band flip the trend side; otherwise the band only tightens, which makes the indicator a volatility-adjusted trailing stop.

## Recurrence

Let \(z_t = (close_t, high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
ATR_t=\operatorname{ATR}_n(close_t,high_t,low_t), \qquad
B^+_t=\frac{high_t+low_t}{2}+mATR_t, \quad
B^-_t=\frac{high_t+low_t}{2}-mATR_t
\]

\[
U_t=\begin{cases}B^+_t, & B^+_t<U_{t-1}\text{ or }close_{t-1}>U_{t-1}\\U_{t-1},&\text{otherwise}\end{cases}
\]

\[
L_t=\begin{cases}B^-_t, & B^-_t>L_{t-1}\text{ or }close_{t-1}<L_{t-1}\\L_{t-1},&\text{otherwise}\end{cases}
\]

\[
trend_t=\begin{cases}1,& close_t\ge L_t\\-1,& close_t\le U_t\\trend_{t-1},&\text{otherwise}\end{cases}, \qquad
value_t=\begin{cases}L_t,& trend_t=1\\U_t,& trend_t=-1\end{cases}
\]

`update(...)` returns a result struct with fields `value`, `direction`, `upper`, `lower`.

## Composed Primitives

[`ATR`](atr.md)

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class SuperTrend`.

## Reference

- [Background reference](https://www.investopedia.com/supertrend-indicator-7976167)
