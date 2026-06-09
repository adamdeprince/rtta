# Low

## Summary

`Low` is RTTA's streaming implementation of: Rolling lowest value.

## Update API

```python
result = rtta.Low(window=30).update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`Low` maintains rolling extrema, ranges, or envelopes. The C++ state updates the relevant window/range statistics once per input sample.

## Recurrence

Let \(z_t = value_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
W_t = \operatorname{push}(W_{t-1}, z_t, n)
\]

\[
y_t = G(W_t)
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class Low`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/lowest-low-value)
