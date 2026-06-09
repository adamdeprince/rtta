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
s_t = F_{UlcerIndex}(s_{t-1}, close_t; \theta)
\]

\[
y_t = G_{UlcerIndex}(s_t)
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class UlcerIndex`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ulcer-index)
