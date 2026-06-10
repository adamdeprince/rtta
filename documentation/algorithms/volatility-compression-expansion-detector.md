# VolatilityCompressionExpansionDetector

## Summary

`VolatilityCompressionExpansionDetector` is RTTA's streaming implementation of: Short-versus-long EWMA volatility ratio detector for compression and expansion regimes.

## Update API

```python
result = rtta.VolatilityCompressionExpansionDetector().update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`VolatilityCompressionExpansionDetector` first constructs a scalar market-state metric from the current observation and compact streaming state, then passes that metric through explicit entry/exit hysteresis. The metric is named in the recurrence below; the hysteresis keeps the output stable until the metric crosses the opposite exit band.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
r_t=\frac{close_t-close_{t-1}}{close_{t-1}}
\]

\[
v^S_t=(1-\alpha_S)(v^S_{t-1}+\alpha_S r_t^2), \qquad
v^L_t=(1-\alpha_L)(v^L_{t-1}+\alpha_L r_t^2)
\]

\[
q_t=\frac{\sqrt{\max(v^S_t,\epsilon)}}{\sqrt{\max(v^L_t,\epsilon)}}
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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class VolatilityCompressionExpansionDetector`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/standard-deviation-volatility)
