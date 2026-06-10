# RollingMeanShiftDetector

## Summary

`RollingMeanShiftDetector` is RTTA's streaming implementation of: Causal adjacent-window mean shift detector using a two-sample z-score.

## Update API

```python
result = rtta.RollingMeanShiftDetector(window=20, threshold=3.0).update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`RollingMeanShiftDetector` compares two adjacent rolling windows: a reference window and a recent window. The C++ state moves expired recent samples into the reference window, maintains sufficient statistics, and emits the sign of the statistic difference when it exceeds the configured threshold.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
\bar{x}^{R}_t,\sigma^{2,R}_t=\operatorname{stats}(R_t), \qquad
\bar{x}^{B}_t,\sigma^{2,B}_t=\operatorname{stats}(B_t)
\]

\[
q_t=\frac{\bar{x}^{R}_t-\bar{x}^{B}_t}
{\sqrt{\sigma^{2,R}_t/n+\sigma^{2,B}_t/n+\epsilon}}
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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class RollingMeanShiftDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Student%27s_t-test)
