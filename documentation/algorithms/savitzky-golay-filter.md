# SavitzkyGolayFilter

## Summary

`SavitzkyGolayFilter` is RTTA's streaming implementation of: Rolling polynomial least-squares smoother with first and second derivative outputs.

## Update API

```python
result = rtta.SavitzkyGolayFilter().update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`SavitzkyGolayFilter` implements the streaming form of Rolling polynomial least-squares smoother with first and second derivative outputs. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
W_t = \operatorname{push}(W_{t-1}, z_t, n)
\]

\[
y_t = G(W_t)
\]

`update(...)` returns a result struct with fields `smooth`, `first_derivative`, `second_derivative`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class SavitzkyGolayFilter`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter)
