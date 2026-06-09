# AverageDirectionalMovementIndex

## Summary

`AverageDirectionalMovementIndex` is RTTA's streaming implementation of: ADX trend-strength indicator.

## Update API

```python
result = rtta.AverageDirectionalMovementIndex().update(close, high, low)
```

The `update(...)` call consumes one observation using `close`, `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`AverageDirectionalMovementIndex` is a causal smoother or average. It updates compact rolling or exponential state with the newest observation and returns the current smoothed estimate.

## Recurrence

Let \(z_t = (close_t, high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
s_t = F_{AverageDirectionalMovementIndex}(s_{t-1}, (close_t, high_t, low_t); \theta)
\]

\[
y_t = G_{AverageDirectionalMovementIndex}(s_t)
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class AverageDirectionalMovementIndex`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx)
