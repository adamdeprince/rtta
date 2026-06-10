# UlcerIndex

## Summary

`UlcerIndex` is RTTA's streaming implementation of: Drawdown-based downside-risk measure.

## Update API

```python
result = rtta.UlcerIndex().update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`UlcerIndex` implements the streaming form of Drawdown-based downside-risk measure. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
H_t=\max_{i\in W_t}close_i, \qquad
d_t=100\frac{close_t-H_t}{H_t}
\]

\[
y_t=\sqrt{\frac{1}{n}\sum_{i\in W_t}d_i^2}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class UlcerIndex`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ulcer-index)
