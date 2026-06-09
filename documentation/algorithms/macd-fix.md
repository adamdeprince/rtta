# MACDFix

## Summary

`MACDFix` is RTTA's streaming implementation of: MACD with fixed 12/26 moving-average periods.

## Update API

```python
result = rtta.MACDFix().update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`MACDFix` is a causal smoother or average. It updates compact rolling or exponential state with the newest observation and returns the current smoothed estimate.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1}
\]

\[
y_t = G(E_t,E^{(2)}_t,\ldots,z_t)
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class MACDFix`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/macd-moving-average-convergence-divergence-oscillator)
