# LeadLagRegimeDetector

## Summary

`LeadLagRegimeDetector` is RTTA's streaming implementation of: EWMA cross-lag detector for which of two series is leading.

## Update API

```python
result = rtta.LeadLagRegimeDetector().update(real0, real1)
```

The `update(...)` call consumes one observation using `real0`, `real1`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`LeadLagRegimeDetector` first constructs a scalar market-state metric from the current observation and compact streaming state, then passes that metric through explicit entry/exit hysteresis. The metric is named in the recurrence below; the hysteresis keeps the output stable until the metric crosses the opposite exit band.

## Recurrence

Let \(z_t = (real0_t, real1_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
\Delta x_t=x_t-x_{t-1}, \qquad \Delta y_t=y_t-y_{t-1}
\]

\[
a_t=\Delta x_{t-1}\Delta y_t, \qquad b_t=\Delta y_{t-1}\Delta x_t
\]

\[
S_t=\alpha(a_t-b_t)+(1-\alpha)S_{t-1}, \qquad
C_t=\alpha(|a_t|+|b_t|)+(1-\alpha)C_{t-1}
\]

\[
q_t=\frac{S_t}{\max(C_t,\epsilon)}
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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class LeadLagRegimeDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Cross-correlation)
