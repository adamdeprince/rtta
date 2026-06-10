# CrossAssetCorrelationBreakDetector

## Summary

`CrossAssetCorrelationBreakDetector` is RTTA's streaming implementation of: Short-versus-long rolling correlation break detector for two assets.

## Update API

```python
result = rtta.CrossAssetCorrelationBreakDetector().update(real0, real1)
```

The `update(...)` call consumes one observation using `real0`, `real1`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`CrossAssetCorrelationBreakDetector` runs short and long rolling correlation estimates on the same pair of streams and measures their absolute divergence. An upper hysteresis state turns that divergence into a persistent break flag until the short/long correlations reconverge below the exit threshold.

## Recurrence

Let \(z_t = (real0_t, real1_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
q_t=|\rho^{short}_t-\rho^{long}_t|
\]

The short and long correlations are maintained by two rolling `Correlation`-style windows.

\[
r_t =
\begin{cases}
1, & r_{t-1} = 0 \text{ and } q_t \ge e \\
0, & r_{t-1} = 1 \text{ and } q_t \le x \\
r_{t-1}, & \text{otherwise}
\end{cases}, \qquad x < e
\]

The return value is the current scalar indicator value.

## Composed Primitives

[`Correlation`](correlation.md)

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class CrossAssetCorrelationBreakDetector`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/correlation-coefficient)
