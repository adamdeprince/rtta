# HighLowIndex

## Summary

`HighLowIndex` is RTTA's streaming implementation of: Combined offsets/indexes of rolling minimum and maximum values.

## Update API

```python
result = rtta.HighLowIndex().update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`HighLowIndex` maintains rolling extrema, ranges, or envelopes. The C++ state updates the relevant window/range statistics once per input sample.

## Recurrence

Let \(z_t = value_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
H_t=\max_{i\in W_t} high_i, \qquad L_t=\min_{i\in W_t} low_i
\]

\[
y_t = G(H_t,L_t,close_t)
\]

`update(...)` returns a result struct with fields `min_index`, `max_index`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class HighLowIndex`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/distance-to-highs)
