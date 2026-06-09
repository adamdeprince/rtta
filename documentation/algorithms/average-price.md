# AveragePrice

## Summary

`AveragePrice` is RTTA's streaming implementation of: Average of open, high, low, and close.

## Update API

```python
result = rtta.AveragePrice().update(open, high, low, close)
```

The `update(...)` call consumes one observation using `open`, `high`, `low`, `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`AveragePrice` is a causal smoother or average. It updates compact rolling or exponential state with the newest observation and returns the current smoothed estimate.

## Recurrence

Let \(z_t = (open_t, high_t, low_t, close_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
AP_t = \frac{open_t + high_t + low_t + close_t}{4}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class AveragePrice`.

## Reference

- [Background reference](https://tulipindicators.org/avgprice)
