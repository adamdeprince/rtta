# LiquidityDroughtDetector

## Summary

`LiquidityDroughtDetector` is RTTA's streaming implementation of: Relative volume/depth drought detector using lower-threshold hysteresis.

## Update API

```python
result = rtta.LiquidityDroughtDetector().update(volume, bid_size, ask_size)
```

The `update(...)` call consumes one observation using `volume`, `bid_size`, `ask_size`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`LiquidityDroughtDetector` first constructs a scalar market-state metric from the current observation and compact streaming state, then passes that metric through explicit entry/exit hysteresis. The metric is named in the recurrence below; the hysteresis keeps the output stable until the metric crosses the opposite exit band.

## Recurrence

Let \(z_t = (volume_t, bid_size_t, ask_size_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
L_t=\max(volume_t,0)+\max(bidSize_t,0)+\max(askSize_t,0)
\]

\[
q_t=\frac{L_t}{\max(B_{t-1},\epsilon)}, \qquad
B_t=\alpha L_t+(1-\alpha)B_{t-1}
\]

\[
r_t =
\begin{cases}
1, & r_{t-1} = 0 \text{ and } q_t \le e \\
0, & r_{t-1} = 1 \text{ and } q_t \ge x \\
r_{t-1}, & \text{otherwise}
\end{cases}, \qquad e < x
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class LiquidityDroughtDetector`.

## Reference

- [Background reference](https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf)
