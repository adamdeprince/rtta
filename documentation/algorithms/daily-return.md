# DailyReturn

## Summary

`DailyReturn` is RTTA's streaming implementation of: Percentage return between consecutive closes.

## Update API

```python
result = rtta.DailyReturn().update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`DailyReturn` implements the streaming form of Percentage return between consecutive closes. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
y_t = \frac{close_t - close_{t-n}}{close_{t-n}}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class DailyReturn`.

## Reference

- [Background reference](https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html)
