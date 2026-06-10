# HDDM

## Summary

`HDDM` is RTTA's streaming implementation of: Hoeffding-bound drift detector for Bernoulli prediction-error streams.

## Update API

```python
result = rtta.HDDM().update(error)
```

The `update(...)` call consumes one observation using `error`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`HDDM` is a streaming classifier-error drift detector. It treats positive input values as errors and compares the current error process against the best historical baseline using the detector's bound: binomial standard error for DDM, distance-between-errors degradation for EDDM, and a Hoeffding bound for HDDM.

## Recurrence

Let \(z_t = error_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
\bar{e}_t=\frac{1}{t}\sum_{i=1}^{t}\mathbf{1}[error_i>0], \qquad
b_t(\delta)=\sqrt{\frac{\log(1/\delta)}{2t}}
\]

\[
(\bar{e}^\*_t,b^\*_t)=
\arg\min_{i\le t}\left(\bar{e}_i+b_i(\delta_{drift})\right)
\]

\[
y_t =
\begin{cases}
1, & \bar{e}_t-\bar{e}^\*_t>b_t(\delta_{drift})+b^\*_t\\
0.5, & \bar{e}_t-\bar{e}^\*_t>b_t(\delta_{warning})+b^\*_t\\
0, & \text{otherwise}
\end{cases}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class HDDM`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality)
