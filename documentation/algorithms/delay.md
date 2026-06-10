# Delay

## Summary

`Delay` is RTTA's streaming implementation of: Lagged value from a fixed number of samples ago.

## Update API

```python
result = rtta.Delay().update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`Delay` implements the streaming form of Lagged value from a fixed number of samples ago. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = value_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
y_t=x_{t-n}
\]

The C++ implementation stores the last \(n\) observations in a ring buffer and
returns the overwritten value on each update.

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class Delay`.

## Reference

- [Background reference](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html)
