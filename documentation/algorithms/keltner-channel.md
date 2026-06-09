# KeltnerChannel

## Summary

`KeltnerChannel` is RTTA's streaming implementation of: EMA/ATR volatility channel.

## Update API

```python
result = rtta.KeltnerChannel().update(close, high, low)
```

The `update(...)` call consumes one observation using `close`, `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`KeltnerChannel` maintains rolling extrema, ranges, or envelopes. The C++ state updates the relevant window/range statistics once per input sample.

## Recurrence

Let \(z_t = (close_t, high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
H_t=\max_{i\in W_t} high_i, \qquad L_t=\min_{i\in W_t} low_i
\]

\[
y_t = G(H_t,L_t,close_t)
\]

`update(...)` returns a result struct with fields `middle`, `upper`, `lower`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class KeltnerChannel`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/keltner-channels)
