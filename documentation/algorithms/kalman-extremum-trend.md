# KalmanExtremumTrend

## Summary

`KalmanExtremumTrend` is RTTA's streaming implementation of: Kalman trend combined with stochastic-style position inside recent extrema.

## Update API

```python
result = rtta.KalmanExtremumTrend().update(close, high, low)
```

The `update(...)` call consumes one observation using `close`, `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`KalmanExtremumTrend` treats the input stream as noisy observations of a latent state. Each call performs the standard predict/update cycle, then projects the updated state into the public scalar or result fields.

## Recurrence

Let \(z_t = (close_t, high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
\hat{x}_{t|t-1}=F\hat{x}_{t-1|t-1}, \qquad
P_{t|t-1}=FP_{t-1|t-1}F^\top+Q
\]

\[
K_t=P_{t|t-1}H^\top(HP_{t|t-1}H^\top+R)^{-1}
\]

\[
\hat{x}_{t|t}=\hat{x}_{t|t-1}+K_t(z_t-H\hat{x}_{t|t-1}), \qquad
P_{t|t}=(I-K_tH)P_{t|t-1}
\]

`update(...)` returns a result struct with fields `trend`, `oscillator`, `signal`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class KalmanExtremumTrend`.

## Reference

- [Background reference](https://arxiv.org/pdf/1808.03297)
