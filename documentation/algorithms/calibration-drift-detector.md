# CalibrationDriftDetector

## Summary

`CalibrationDriftDetector` is RTTA's streaming implementation of: EWMA probability-calibration error drift detector.

## Update API

```python
result = rtta.CalibrationDriftDetector().update(probability, outcome)
```

The `update(...)` call consumes one observation using `probability`, `outcome`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`CalibrationDriftDetector` first constructs a scalar market-state metric from the current observation and compact streaming state, then passes that metric through explicit entry/exit hysteresis. The metric is named in the recurrence below; the hysteresis keeps the output stable until the metric crosses the opposite exit band.

## Recurrence

Let \(z_t = (probability_t, outcome_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
e_t=|\mathbf{1}[outcome_t>0]-\operatorname{clip}(probability_t,0,1)|
\]

\[
q_t=\alpha e_t+(1-\alpha)q_{t-1}
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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class CalibrationDriftDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Calibration_(statistics)
