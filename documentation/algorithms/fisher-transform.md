# FisherTransform

## Summary

`FisherTransform` is RTTA's streaming implementation of: Ehlers transform of normalized recent high/low position into a turning-point oscillator.

## Update API

```python
result = rtta.FisherTransform().update(high, low)
```

The `update(...)` call consumes one observation using `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`FisherTransform` normalizes recent directional movement into an oscillator. The implementation keeps only causal rolling or smoothed state and maps that state into the current oscillator value.

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

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class FisherTransform`.

## Reference

- [Background reference](https://trendspider.com/learning-center/fisher-transform-a-comprehensive-guide/)
