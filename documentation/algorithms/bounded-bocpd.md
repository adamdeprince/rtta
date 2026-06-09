# BoundedBOCPD

## Summary

`BoundedBOCPD` is RTTA's streaming implementation of: Bounded-memory Bayesian online change-point detector with constant hazard.

## Update API

```python
result = rtta.BoundedBOCPD().update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`BoundedBOCPD` converts each observation into a streaming score and then applies threshold or hysteresis logic. The state is deliberately sticky where the C++ class models regimes, so small reversals do not immediately flip the output.

## Recurrence

Let \(z_t = value_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
R_t(r+1) = R_{t-1}(r)(1-h)p(z_t\mid r)
\]

\[
R_t(0)=\sum_r R_{t-1}(r)h\,p(z_t\mid r)
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class BoundedBOCPD`.

## Reference

- [Background reference](https://arxiv.org/abs/0710.3742)
