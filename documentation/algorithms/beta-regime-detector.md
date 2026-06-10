# BetaRegimeDetector

## Summary

`BetaRegimeDetector` is RTTA's streaming implementation of: Stateful rolling beta regime detector with upper/lower hysteresis bands.

## Update API

```python
result = rtta.BetaRegimeDetector().update(real0, real1)
```

The `update(...)` call consumes one observation using `real0`, `real1`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`BetaRegimeDetector` first constructs a scalar market-state metric from the current observation and compact streaming state, then passes that metric through explicit entry/exit hysteresis. The metric is named in the recurrence below; the hysteresis keeps the output stable until the metric crosses the opposite exit band.

## Recurrence

Let \(z_t = (real0_t, real1_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
q_t=\beta_t=
\frac{n\sum xy-\sum x\sum y}{n\sum y^2-(\sum y)^2}
\]

The sums are maintained over the configured rolling window; the C++ beta is the covariance of `real0` with `real1` divided by the variance of `real1`.

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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class BetaRegimeDetector`.

## Reference

- [Background reference](https://www.investopedia.com/terms/b/beta.asp)
