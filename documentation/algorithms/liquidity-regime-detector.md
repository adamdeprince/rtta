# LiquidityRegimeDetector

## Summary

`LiquidityRegimeDetector` is RTTA's streaming implementation of: EWMA Amihud-style liquidity regime detector using absolute return per dollar volume.

## Update API

```python
result = rtta.LiquidityRegimeDetector().update(close, volume)
```

The `update(...)` call consumes one observation using `close`, `volume`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`LiquidityRegimeDetector` first constructs a scalar market-state metric from the current observation and compact streaming state, then passes that metric through explicit entry/exit hysteresis. The metric is named in the recurrence below; the hysteresis keeps the output stable until the metric crosses the opposite exit band.

## Recurrence

Let \(z_t = (close_t, volume_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
DV_t=|close_t|\max(volume_t,0), \qquad
I_t=\frac{|(close_t-close_{t-1})/close_{t-1}|}{\max(DV_t,\epsilon)}
\]

\[
q_t=\alpha I_t+(1-\alpha)q_{t-1}
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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class LiquidityRegimeDetector`.

## Reference

- [Background reference](https://ba-odegaard.no/teach/notes/liquidity_estimators/amihud_estimator/amihud_lectures.pdf)
