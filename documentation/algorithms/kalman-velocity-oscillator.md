# KalmanVelocityOscillator

## Summary

`KalmanVelocityOscillator` is RTTA's streaming implementation of: Zero-centered velocity state from a constant-velocity Kalman price model.

## Update API

```python
result = rtta.KalmanVelocityOscillator().update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`KalmanVelocityOscillator` treats the input stream as noisy observations of a latent state. Each call performs the standard predict/update cycle, then projects the updated state into the public scalar or result fields.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
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

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class KalmanVelocityOscillator`.

## Reference

- [Background reference](https://arxiv.org/pdf/1808.03297)
