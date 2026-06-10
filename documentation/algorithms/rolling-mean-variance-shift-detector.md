# RollingMeanVarianceShiftDetector

## Summary

`RollingMeanVarianceShiftDetector` is RTTA's streaming implementation of: Causal adjacent-window combined mean and variance shift detector.

## Update API

```python
result = rtta.RollingMeanVarianceShiftDetector(window=20, threshold=3.0).update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`RollingMeanVarianceShiftDetector` compares two adjacent rolling windows: a reference window and a recent window. The C++ state moves expired recent samples into the reference window, maintains sufficient statistics, and emits the sign of the statistic difference when it exceeds the configured threshold.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
z^\mu_t=\frac{\bar{x}^{R}_t-\bar{x}^{B}_t}
{\sqrt{\sigma^{2,R}_t/n+\sigma^{2,B}_t/n+\epsilon}},
\qquad
z^\sigma_t=\log\left(\frac{\sigma^{2,R}_t+\epsilon}{\sigma^{2,B}_t+\epsilon}\right)
\]

\[
q_t=\sqrt{(z^\mu_t)^2+w(z^\sigma_t)^2}, \qquad
d_t=\begin{cases}
z^\mu_t, & |z^\mu_t|\ge |\sqrt{w}z^\sigma_t|\\
\sqrt{w}z^\sigma_t, & \text{otherwise}
\end{cases}
\]

\[
y_t =
\begin{cases}
\operatorname{sgn}(d_t), & q_t>h\\
0, & \text{otherwise}
\end{cases}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class RollingMeanVarianceShiftDetector`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Change_detection)
