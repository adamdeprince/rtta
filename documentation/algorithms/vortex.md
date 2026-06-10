# Vortex

## Summary

`Vortex` is RTTA's streaming implementation of: Positive/negative Vortex trend movement indicator.

## Update API

```python
result = rtta.Vortex().update(close, high, low)
```

The `update(...)` call consumes one observation using `close`, `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`Vortex` implements the streaming form of Positive/negative Vortex trend movement indicator. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = (close_t, high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
TR_t=\max(high_t-low_t,\ |high_t-close_{t-1}|,\ |low_t-close_{t-1}|)
\]

\[
VM^+_t=|high_t-low_{t-1}|, \qquad VM^-_t=|low_t-high_{t-1}|
\]

\[
VI^+_t=\frac{\sum_{i\in W_t}VM^+_i}{\sum_{i\in W_t}TR_i}, \qquad
VI^-_t=\frac{\sum_{i\in W_t}VM^-_i}{\sum_{i\in W_t}TR_i}
\]

The result fields are \(VI^+_t\), \(VI^-_t\), and their difference.

`update(...)` returns a result struct with fields `positive`, `negative`, `difference`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class Vortex`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/vortex-indicator)
