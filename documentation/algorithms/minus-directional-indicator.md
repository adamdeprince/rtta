# MinusDirectionalIndicator

## Summary

`MinusDirectionalIndicator` is RTTA's streaming implementation of: Negative directional indicator.

## Update API

```python
result = rtta.MinusDirectionalIndicator().update(close, high, low)
```

The `update(...)` call consumes one observation using `close`, `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`MinusDirectionalIndicator` implements the streaming form of Negative directional indicator. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = (close_t, high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
s_t = F_{MinusDirectionalIndicator}(s_{t-1}, (close_t, high_t, low_t); \theta)
\]

\[
y_t = G_{MinusDirectionalIndicator}(s_t)
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class MinusDirectionalIndicator`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx)
