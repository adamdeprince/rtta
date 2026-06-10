# PredictionErrorDriftDetector

## Summary

`PredictionErrorDriftDetector` is RTTA's streaming implementation of: EWMA absolute prediction-error drift detector.

## Update API

```python
result = rtta.PredictionErrorDriftDetector().update(prediction, actual)
```

The `update(...)` call consumes one observation using `prediction`, `actual`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`PredictionErrorDriftDetector` standardizes the current error or move against an EWMA mean and variance estimated from prior samples. The detector uses the resulting z-score with hysteresis or reset logic so isolated noisy observations do not become persistent regimes by themselves.

## Recurrence

Let \(z_t = (prediction_t, actual_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
e_t=|actual_t-prediction_t|, \qquad
q_t=\frac{e_t-\mu_{t-1}}{\sqrt{\max(\sigma^2_{t-1},\epsilon)}}
\]

\[
\mu_t=\mu_{t-1}+\alpha(e_t-\mu_{t-1}), \qquad
\sigma^2_t=(1-\alpha)(\sigma^2_{t-1}+\alpha(e_t-\mu_{t-1})^2)
\]

\[
r_t =
\begin{cases}
1, & r_{t-1} = 0 \text{ and } q_t \ge e \\
0, & r_{t-1} = 1 \text{ and } q_t \le x \\
r_{t-1}, & \text{otherwise}
\end{cases}, \qquad x < e
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class PredictionErrorDriftDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Concept_drift)
