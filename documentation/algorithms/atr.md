# ATR

## Summary

`ATR` computes Average True Range, a volatility measure for OHLC streams. RTTA's
implementation is causal: each call to `update(close, high, low)` consumes one
bar, updates stored state, and returns the current average true range.

## Update API

```python
value = rtta.ATR(window=14.0, fillna=True).update(close, high, low)
```

Inputs are the current close, high, and low. The implementation stores the
previous close, a running true-range sum for warmup, and the latest ATR value.

## Theory Of Operation

True Range expands a bar's high-low range to include overnight or inter-bar
gaps from the previous close. ATR then smooths that true range. In RTTA, the
warmup phase returns the running average of all true ranges seen so far; after
`window` samples it switches to Wilder smoothing.

## Recurrence

Let \(C_t\), \(H_t\), and \(L_t\) be close, high, and low for update \(t\), and
let \(n\) be `window`.

\[
TR_t =
\begin{cases}
H_t - L_t, & t = 0 \\
\max(H_t - L_t,\ |H_t - C_{t-1}|,\ |L_t - C_{t-1}|), & t > 0
\end{cases}
\]

During warmup, with \(m_t = t + 1 \le n\):

\[
ATR_t = \frac{\sum_{i=0}^{t} TR_i}{m_t}
\]

After warmup:

\[
ATR_t = \frac{(n - 1)ATR_{t-1} + TR_t}{n}
\]

If `fillna=False`, the object still updates all state during warmup, but returns
`NaN` until at least `window` samples have been consumed.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ATR`.

## Reference

- [ChartSchool: Average True Range and ATR Percent](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp)
