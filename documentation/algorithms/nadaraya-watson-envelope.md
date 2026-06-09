# NadarayaWatsonEnvelope

## Summary

`NadarayaWatsonEnvelope` is RTTA's streaming implementation of: Gaussian-kernel Nadaraya-Watson smoother with weighted residual bands.

## Update API

```python
result = rtta.NadarayaWatsonEnvelope(window=32).update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`NadarayaWatsonEnvelope` is a causal smoother or average. It updates compact rolling or exponential state with the newest observation and returns the current smoothed estimate.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
w_{t,i}=\exp\left(-\frac{(t-i)^2}{2h^2}\right)
\]

\[
\hat{x}_t=\frac{\sum_{i\in W_t}w_{t,i}x_i}{\sum_{i\in W_t}w_{t,i}}
\]

`update(...)` returns a result struct with fields `middle`, `upper`, `lower`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class NadarayaWatsonEnvelope`.

## Reference

- [Background reference](https://classic.d2l.ai/chapter_attention-mechanisms/nadaraya-watson.html)
