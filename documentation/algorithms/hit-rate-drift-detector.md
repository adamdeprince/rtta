# HitRateDriftDetector

## Summary

`HitRateDriftDetector` is RTTA's streaming implementation of: EWMA hit-rate degradation detector using miss-rate hysteresis.

## Update API

```python
result = rtta.HitRateDriftDetector().update(hit)
```

The `update(...)` call consumes one observation using `hit`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`HitRateDriftDetector` first constructs a scalar market-state metric from the current observation and compact streaming state, then passes that metric through explicit entry/exit hysteresis. The metric is named in the recurrence below; the hysteresis keeps the output stable until the metric crosses the opposite exit band.

## Recurrence

Let \(z_t = hit_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
m_t=\mathbf{1}[hit_t\le0], \qquad
q_t=\alpha m_t+(1-\alpha)q_{t-1}
\]

The metric is an EWMA miss rate; high values indicate deteriorating hit rate.

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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class HitRateDriftDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Concept_drift)
