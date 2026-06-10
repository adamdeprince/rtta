# PairsSpreadRegimeDetector

## Summary

`PairsSpreadRegimeDetector` is RTTA's streaming implementation of: Streaming EWMA hedge-ratio residual z-score detector for pair-spread regimes.

## Update API

```python
result = rtta.PairsSpreadRegimeDetector().update(real0, real1)
```

The `update(...)` call consumes one observation using `real0`, `real1`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`PairsSpreadRegimeDetector` first constructs a scalar market-state metric from the current observation and compact streaming state, then passes that metric through explicit entry/exit hysteresis. The metric is named in the recurrence below; the hysteresis keeps the output stable until the metric crosses the opposite exit band.

## Recurrence

Let \(z_t = (real0_t, real1_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
\beta_t=\frac{C^{xy}_t}{V^y_t}, \qquad
\alpha_t=\mu^x_t-\beta_t\mu^y_t
\]

\[
e_t=x_t-(\beta_t y_t+\alpha_t), \qquad
q_t=\frac{e_t-\bar{e}_{t-1}}{\sqrt{\max(s^2_{e,t-1},\epsilon)}}
\]

\[
\bar{e}_t=\bar{e}_{t-1}+\eta(e_t-\bar{e}_{t-1}), \qquad
s^2_{e,t}=(1-\eta)(s^2_{e,t-1}+\eta(e_t-\bar{e}_{t-1})^2)
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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class PairsSpreadRegimeDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Statistical_arbitrage)
