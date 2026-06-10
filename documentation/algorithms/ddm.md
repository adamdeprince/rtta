# DDM

## Summary

`DDM` is RTTA's streaming implementation of: Drift Detection Method for Bernoulli prediction-error streams.

## Update API

```python
result = rtta.DDM().update(error)
```

The `update(...)` call consumes one observation using `error`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`DDM` is a streaming classifier-error drift detector. It treats positive input values as errors and compares the current error process against the best historical baseline using the detector's bound: binomial standard error for DDM, distance-between-errors degradation for EDDM, and a Hoeffding bound for HDDM.

## Recurrence

Let \(z_t = error_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
p_t=\frac{1}{t}\sum_{i=1}^{t}\mathbf{1}[error_i>0], \qquad
s_t=\sqrt{\frac{p_t(1-p_t)}{t}}, \qquad m_t=p_t+s_t
\]

\[
m^\*_t=\min_{i\le t}m_i, \qquad s^\*_t=s_{\arg\min_i m_i}
\]

\[
y_t =
\begin{cases}
1, & m_t>m^\*_t+d\,s^\*_t\\
0.5, & m_t>m^\*_t+w\,s^\*_t\\
0, & \text{otherwise}
\end{cases}
\]

The detector resets its error counts after a drift signal.

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class DDM`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Concept_drift)
