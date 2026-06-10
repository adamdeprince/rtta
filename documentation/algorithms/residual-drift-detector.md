# ResidualDriftDetector

## Summary

`ResidualDriftDetector` is RTTA's streaming implementation of: EWMA residual z-score drift detector with signed hysteresis output.

## Update API

```python
result = rtta.ResidualDriftDetector().update(residual)
```

The `update(...)` call consumes one observation using `residual`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`ResidualDriftDetector` standardizes the current error or move against an EWMA mean and variance estimated from prior samples. The detector uses the resulting z-score with hysteresis or reset logic so isolated noisy observations do not become persistent regimes by themselves.

## Recurrence

Let \(z_t = residual_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
q_t=\frac{residual_t-\mu_{t-1}}{\sqrt{\max(\sigma^2_{t-1},\epsilon)}}
\]

\[
\mu_t=\mu_{t-1}+\alpha(residual_t-\mu_{t-1}), \qquad
\sigma^2_t=(1-\alpha)(\sigma^2_{t-1}+\alpha(residual_t-\mu_{t-1})^2)
\]

\[
r_t =
\begin{cases}
1, & r_{t-1} \le 0 \text{ and } q_t \ge u_e \\
0, & r_{t-1} = 1 \text{ and } q_t \le u_x \\
-1, & r_{t-1} \ge 0 \text{ and } q_t \le \ell_e \\
0, & r_{t-1} = -1 \text{ and } q_t \ge \ell_x \\
r_{t-1}, & \text{otherwise}
\end{cases}
\]

The entry/exit constants satisfy \(\ell_e < \ell_x \le u_x < u_e\).

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ResidualDriftDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Concept_drift)
