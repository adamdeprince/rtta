# RollingCorrelationShiftDetector

## Summary

`RollingCorrelationShiftDetector` is RTTA's streaming implementation of: Causal adjacent-window correlation shift detector.

## Update API

```python
result = rtta.RollingCorrelationShiftDetector(window=20, threshold=0.25).update(real0, real1)
```

The `update(...)` call consumes one observation using `real0`, `real1`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`RollingCorrelationShiftDetector` compares two adjacent rolling windows: a reference window and a recent window. The C++ state moves expired recent samples into the reference window, maintains sufficient statistics, and emits the sign of the statistic difference when it exceeds the configured threshold.

## Recurrence

Let \(z_t = (real0_t, real1_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
\rho^R_t=\operatorname{corr}(R^x_t,R^y_t), \qquad
\rho^B_t=\operatorname{corr}(B^x_t,B^y_t)
\]

\[
q_t=\rho^R_t-\rho^B_t
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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class RollingCorrelationShiftDetector`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/correlation-coefficient)
