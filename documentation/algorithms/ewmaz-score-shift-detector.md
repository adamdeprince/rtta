# EWMAZScoreShiftDetector

## Summary

`EWMAZScoreShiftDetector` is RTTA's streaming implementation of: Causal EWMA mean/variance z-score event detector for threshold-sized shifts.

## Update API

```python
result = rtta.EWMAZScoreShiftDetector(alpha=0.05, threshold=3.0).update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`EWMAZScoreShiftDetector` standardizes the current error or move against an EWMA mean and variance estimated from prior samples. The detector uses the resulting z-score with hysteresis or reset logic so isolated noisy observations do not become persistent regimes by themselves.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
z_t=\frac{close_t-\mu_{t-1}}{\sqrt{\max(\sigma^2_{t-1},\epsilon)}}
\]

\[
y_t =
\begin{cases}
1, & z_t>h\\
-1, & z_t<-h\\
0, & \text{otherwise}
\end{cases}
\]

\[
\mu_t=\mu_{t-1}+\alpha(close_t-\mu_{t-1}), \qquad
\sigma^2_t=(1-\alpha)(\sigma^2_{t-1}+\alpha(close_t-\mu_{t-1})^2)
\]

When \(y_t\ne0\), the C++ implementation resets \(\mu_t\) to the current close
and clears the variance estimate.

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class EWMAZScoreShiftDetector`.

## Reference

- [Background reference](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)
