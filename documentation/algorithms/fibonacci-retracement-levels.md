# FibonacciRetracementLevels

## Summary

`FibonacciRetracementLevels` is RTTA's streaming implementation of: Rolling Fibonacci retracement levels between recent high and low.

## Update API

```python
result = rtta.FibonacciRetracementLevels().update(high, low)
```

The `update(...)` call consumes one observation using `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`FibonacciRetracementLevels` maintains rolling extrema, ranges, or envelopes. The C++ state updates the relevant window/range statistics once per input sample.

## Recurrence

Let \(z_t = (high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
H_t=\max_{i\in W_t} high_i, \qquad L_t=\min_{i\in W_t} low_i
\]

\[
y_t = G(H_t,L_t,close_t)
\]

`update(...)` returns a result struct with fields `level0`, `level236`, `level382`, `level500`, `level618`, `level100`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class FibonacciRetracementLevels`.

## Reference

- [Background reference](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/fibonacci-retracement)
