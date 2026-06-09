# ParticleFilterTrend

## Summary

`ParticleFilterTrend` is RTTA's streaming implementation of: Deterministic-seed particle trend filter with Laplace measurement likelihood and effective sample size output.

## Update API

```python
result = rtta.ParticleFilterTrend(particles=64).update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`ParticleFilterTrend` implements the streaming form of Deterministic-seed particle trend filter with Laplace measurement likelihood and effective sample size output. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
x_t^{(i)} = f(x_{t-1}^{(i)})+\epsilon_t^{(i)}
\]

\[
w_t^{(i)} \propto w_{t-1}^{(i)}p(z_t\mid x_t^{(i)}), \qquad
\sum_i w_t^{(i)}=1
\]

`update(...)` returns a result struct with fields `trend`, `velocity`, `signal`, `effective_sample_size`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ParticleFilterTrend`.

## Reference

- [Background reference](https://alphaarchitect.com/trend-following-filters-part-4/)
