# CointegrationBreakdownMonitor

## Summary

`CointegrationBreakdownMonitor` is RTTA's streaming implementation of: Streaming residual-z monitor for pair relationship breakdowns using an EWMA hedge estimate.

## Update API

```python
result = rtta.CointegrationBreakdownMonitor().update(real0, real1)
```

The `update(...)` call consumes one observation using `real0`, `real1`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`CointegrationBreakdownMonitor` implements the streaming form of Streaming residual-z monitor for pair relationship breakdowns using an EWMA hedge estimate. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = (real0_t, real1_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
s_t = F_{CointegrationBreakdownMonitor}(s_{t-1}, (real0_t, real1_t); \theta)
\]

\[
y_t = G_{CointegrationBreakdownMonitor}(s_t)
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class CointegrationBreakdownMonitor`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Cointegration)
