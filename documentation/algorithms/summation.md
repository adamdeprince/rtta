# Summation

## Summary

`Summation` is RTTA's streaming implementation of: Rolling sum.

## Update API

```python
result = rtta.Summation(window=30).update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`Summation` implements the streaming form of Rolling sum. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class Summation`.

## Reference

- [Background reference](https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.sum.html)
