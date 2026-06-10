# PageHinkley

## Summary

`PageHinkley` is RTTA's streaming implementation of: Causal Page-Hinkley mean-shift event detector with directed up/down output.

## Update API

```python
result = rtta.PageHinkley(threshold=1.0, delta=0.0).update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`PageHinkley` tracks cumulative positive and negative deviations from an online mean after subtracting a small drift allowance. A signal fires when one cumulative excursion rises far enough above its own running minimum; the detector then resets to the current close.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
\mu_t=\mu_{t-1}+\frac{close_t-\mu_{t-1}}{t}
\]

\[
P_t=P_{t-1}+close_t-\mu_t-\delta, \qquad
N_t=N_{t-1}+\mu_t-close_t-\delta
\]

\[
S^+_t=P_t-\min_{i\le t}P_i, \qquad
S^-_t=N_t-\min_{i\le t}N_i
\]

\[
y_t =
\begin{cases}
1, & S^+_t > h \text{ and } S^+_t \ge S^-_t\\
-1, & S^-_t > h\\
0, & \text{otherwise}
\end{cases}
\]

After a nonzero signal the C++ implementation resets the running mean and
cumulative sums to the current close.

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class PageHinkley`.

## Reference

- [Background reference](https://menelaus.readthedocs.io/en/dev/menelaus.change_detection.html)
