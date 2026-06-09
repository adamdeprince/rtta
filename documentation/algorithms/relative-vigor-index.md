# RelativeVigorIndex

## Summary

`RelativeVigorIndex` is RTTA's streaming implementation of: Smoothed close-open momentum relative to high-low range with signal line.

## Update API

```python
result = rtta.RelativeVigorIndex().update(open, high, low, close)
```

The `update(...)` call consumes one observation using `open`, `high`, `low`, `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`RelativeVigorIndex` maintains rolling extrema, ranges, or envelopes. The C++ state updates the relevant window/range statistics once per input sample.

## Recurrence

Let \(z_t = (open_t, high_t, low_t, close_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
H_t=\max_{i\in W_t} high_i, \qquad L_t=\min_{i\in W_t} low_i
\]

\[
y_t = G(H_t,L_t,close_t)
\]

`update(...)` returns a result struct with fields `rvi`, `signal`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class RelativeVigorIndex`.

## Reference

- [Background reference](https://www.investopedia.com/terms/r/relative_vigor_index.asp)
