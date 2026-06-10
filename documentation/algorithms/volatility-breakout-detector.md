# VolatilityBreakoutDetector

## Summary

`VolatilityBreakoutDetector` is RTTA's streaming implementation of: EWMA z-score detector for unusually large close-to-close volatility breakouts.

## Update API

```python
result = rtta.VolatilityBreakoutDetector().update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`VolatilityBreakoutDetector` standardizes the current error or move against an EWMA mean and variance estimated from prior samples. The detector uses the resulting z-score with hysteresis or reset logic so isolated noisy observations do not become persistent regimes by themselves.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
m_t=\left|\frac{close_t-close_{t-1}}{close_{t-1}}\right|, \qquad
q_t=\frac{m_t-\mu_{t-1}}{\sqrt{\max(\sigma^2_{t-1},\epsilon)}}
\]

\[
\mu_t=\mu_{t-1}+\alpha(m_t-\mu_{t-1}), \qquad
\sigma^2_t=(1-\alpha)(\sigma^2_{t-1}+\alpha(m_t-\mu_{t-1})^2)
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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class VolatilityBreakoutDetector`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/standard-deviation-volatility)
