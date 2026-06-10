# KSWIN

## Summary

`KSWIN` is RTTA's streaming implementation of: Kolmogorov-Smirnov sliding-window drift detector.

## Update API

```python
result = rtta.KSWIN().update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`KSWIN` compares the empirical distribution of a recent subwindow with the older reference portion of the rolling window using the Kolmogorov-Smirnov supremum distance. The output direction is determined by which subwindow has the larger mean when the KS statistic clears its critical value.

## Recurrence

Let \(z_t = value_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
A_t=W_t[1:|W_t|-m], \qquad B_t=W_t[|W_t|-m+1:|W_t|]
\]

\[
D_t=\sup_x |\widehat{F}_{A_t}(x)-\widehat{F}_{B_t}(x)|
\]

\[
c_\alpha=\sqrt{-\frac{1}{2}\log(\alpha/2)
\left(\frac{1}{|A_t|}+\frac{1}{|B_t|}\right)}
\]

\[
y_t =
\begin{cases}
\operatorname{sgn}(\bar{B}_t-\bar{A}_t), & D_t>c_\alpha\\
0, & \text{otherwise}
\end{cases}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class KSWIN`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
