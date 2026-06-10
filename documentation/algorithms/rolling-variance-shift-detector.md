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

`RollingVarianceShiftDetector` compares two adjacent rolling windows: a reference window and a recent window. The C++ state moves expired recent samples into the reference window, maintains sufficient statistics, and emits the sign of the statistic difference when it exceeds the configured threshold.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
\sigma^{2,R}_t=\operatorname{var}(R_t), \qquad
\sigma^{2,B}_t=\operatorname{var}(B_t)
\]

\[
q_t=\log\left(\frac{\sigma^{2,R}_t+\epsilon}{\sigma^{2,B}_t+\epsilon}\right)
\]

\[
r_t =
\begin{cases}
1, & q_t > h \\
-1, & q_t < -h \\
0, & \text{otherwise}
\end{cases}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class RollingVarianceShiftDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/F-test)
