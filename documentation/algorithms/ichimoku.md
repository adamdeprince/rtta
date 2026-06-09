# Ichimoku

## Summary

`Ichimoku` is RTTA's streaming implementation of: Ichimoku conversion, base, and leading span components.

## Update API

```python
result = rtta.Ichimoku().update(high, low)
```

The `update(...)` call consumes one observation using `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`Ichimoku` normalizes recent directional movement into an oscillator. The implementation keeps only causal rolling or smoothed state and maps that state into the current oscillator value.

## Recurrence

Let \(z_t = (high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
U_t,D_t = \operatorname{directional\_components}(z_t,z_{t-1})
\]

\[
y_t = 100\frac{\operatorname{smooth}(U_t)}
{\operatorname{smooth}(U_t)+\operatorname{smooth}(D_t)}
\]

`update(...)` returns a result struct with fields `conversion`, `base`, `span_a`, `span_b`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class Ichimoku`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/ichimoku-cloud)
