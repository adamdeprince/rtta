# RollingBetaShiftDetector

## Summary

`RollingBetaShiftDetector` is RTTA's streaming implementation of: Causal adjacent-window beta shift detector.

## Update API

```python
result = rtta.RollingBetaShiftDetector(window=20, threshold=0.25).update(real0, real1)
```

The `update(...)` call consumes one observation using `real0`, `real1`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`RollingBetaShiftDetector` compares two adjacent rolling windows: a reference window and a recent window. The C++ state moves expired recent samples into the reference window, maintains sufficient statistics, and emits the sign of the statistic difference when it exceeds the configured threshold.

## Recurrence

Let \(z_t = (real0_t, real1_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
\beta^R_t=\frac{\operatorname{cov}(R^x_t,R^y_t)}{\operatorname{var}(R^y_t)}, \qquad
\beta^B_t=\frac{\operatorname{cov}(B^x_t,B^y_t)}{\operatorname{var}(B^y_t)}
\]

\[
q_t=\beta^R_t-\beta^B_t
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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class RollingBetaShiftDetector`.

## Reference

- [Background reference](https://www.investopedia.com/terms/b/beta.asp)
