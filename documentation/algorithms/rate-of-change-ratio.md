# RateOfChangeRatio

## Summary

`RateOfChangeRatio` is RTTA's streaming implementation of: Rate-of-change ratio against a prior value.

## Update API

```python
result = rtta.RateOfChangeRatio().update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`RateOfChangeRatio` implements the streaming form of Rate-of-change ratio against a prior value. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
y_t = \frac{close_t}{close_{t-n}}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class RateOfChangeRatio`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/rate-of-change-roc-and-momentum)
