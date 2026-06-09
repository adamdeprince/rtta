# ElderRayIndex

## Summary

`ElderRayIndex` is RTTA's streaming implementation of: Bull and bear power as high/low distance from an EMA of close.

## Update API

```python
result = rtta.ElderRayIndex().update(close, high, low)
```

The `update(...)` call consumes one observation using `close`, `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`ElderRayIndex` maintains rolling extrema, ranges, or envelopes. The C++ state updates the relevant window/range statistics once per input sample.

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

`update(...)` returns a result struct with fields `bull_power`, `bear_power`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ElderRayIndex`.

## Reference

- [Background reference](https://www.investopedia.com/articles/trading/03/022603.asp)
