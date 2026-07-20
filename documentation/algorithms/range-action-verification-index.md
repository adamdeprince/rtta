# RangeActionVerificationIndex

## Summary

`RangeActionVerificationIndex` (RAVI) is RTTA's streaming implementation of
Tushar Chande's range action verification index: the absolute gap between a
short and a long simple moving average, expressed as a percent of the long SMA.

## Update API

```python
value = rtta.RangeActionVerificationIndex(
    short_window=7, long_window=65, fillna=True
).update(close)
```

The `update(...)` call consumes one close. `advance(...)` updates state without
returning a Python value. Scalar `batch(...)` returns a NumPy array.

## Theory Of Operation

RAVI is a trend-presence measure in the same family as ADX: large separation
between a fast and a slow average means the market is acting directionally;
small separation means range-bound action. Unlike ADX, RAVI does not use true
range or directional movement — only the relative SMA gap.

## Recurrence

Let \(x_t\) be close, \(n_s\) be `short_window`, and \(n_\ell\) be `long_window`.

\[
S_t = \operatorname{SMA}_{n_s}(x_t), \qquad
L_t = \operatorname{SMA}_{n_\ell}(x_t)
\]

\[
\operatorname{RAVI}_t = 100\cdot\frac{\lvert S_t - L_t\rvert}{L_t}
\]

With `fillna=False`, outputs are `NaN` until the long window is full.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in
`class RangeActionVerificationIndex` with two nested `SMA` members.

## Reference

- [TradingPedia: Chande's Range Action Verification Index (RAVI)](https://www.tradingpedia.com/forex-trading-indicators/chandes-range-action-verification-index/)
- [Wealth-Lab Wiki: RAVI](http://www2.wealth-lab.com/wl5WIKI/RAVI.ashx)
