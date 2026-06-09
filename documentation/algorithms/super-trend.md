# SuperTrend

## Summary

`SuperTrend` is RTTA's streaming implementation of: ATR-band trend-following indicator.

## Update API

```python
result = rtta.SuperTrend().update(close, high, low)
```

The `update(...)` call consumes one observation using `close`, `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`SuperTrend` maintains rolling extrema, ranges, or envelopes. The C++ state updates the relevant window/range statistics once per input sample.

## Recurrence

Let \(z_t = (close_t, high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
s_t = F_{SuperTrend}(s_{t-1}, (close_t, high_t, low_t); \theta)
\]

\[
y_t = G_{SuperTrend}(s_t)
\]

`update(...)` returns a result struct with fields `value`, `direction`, `upper`, `lower`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class SuperTrend`.

## Reference

- [Background reference](https://www.investopedia.com/supertrend-indicator-7976167)
