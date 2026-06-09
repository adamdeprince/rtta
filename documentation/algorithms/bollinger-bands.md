# BollingerBands

## Summary

`BollingerBands` is RTTA's streaming implementation of: Moving-average envelope based on standard deviations.

## Update API

```python
result = rtta.BollingerBands().update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`BollingerBands` is a causal smoother or average. It updates compact rolling or exponential state with the newest observation and returns the current smoothed estimate.

## Recurrence

Let \(z_t = value_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
s_t = F_{BollingerBands}(s_{t-1}, value_t; \theta)
\]

\[
y_t = G_{BollingerBands}(s_t)
\]

`update(...)` returns a result struct with fields `middle`, `upper`, `lower`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class BollingerBands`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/bollinger-bands)
