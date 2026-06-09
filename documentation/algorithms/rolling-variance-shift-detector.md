# RollingVarianceShiftDetector

## Summary

`RollingVarianceShiftDetector` is RTTA's streaming implementation of: Causal adjacent-window variance shift detector using log variance ratio.

## Update API

```python
result = rtta.RollingVarianceShiftDetector(window=20, threshold=1.0).update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`RollingVarianceShiftDetector` converts each observation into a streaming score and then applies threshold or hysteresis logic. The state is deliberately sticky where the C++ class models regimes, so small reversals do not immediately flip the output.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
s_t = F(s_{t-1}, z_t)
\]

\[
r_t =
\begin{cases}
1, & score(s_t) \ge u \\
-1, & score(s_t) \le l \\
r_{t-1}, & \text{otherwise}
\end{cases}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class RollingVarianceShiftDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/F-test)
