# CointegrationBreakdownMonitor

## Summary

`CointegrationBreakdownMonitor` is RTTA's streaming implementation of: Streaming residual-z monitor for pair relationship breakdowns using an EWMA hedge estimate.

## Update API

```python
result = rtta.CointegrationBreakdownMonitor().update(real0, real1)
```

The `update(...)` call consumes one observation using `real0`, `real1`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`CointegrationBreakdownMonitor` implements the streaming form of Streaming residual-z monitor for pair relationship breakdowns using an EWMA hedge estimate. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = (real0_t, real1_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
\beta_t=\frac{C^{xy}_t}{V^y_t}, \qquad
e_t=x_t-(\beta_t y_t+\alpha_t)
\]

\[
q_t=\left|\frac{e_t-\bar{e}_{t-1}}{\sqrt{\max(s^2_{e,t-1},\epsilon)}}\right|
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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class CointegrationBreakdownMonitor`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Cointegration)
