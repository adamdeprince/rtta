# EDDM

## Summary

`EDDM` is RTTA's streaming implementation of: Early Drift Detection Method using distances between prediction errors.

## Update API

```python
result = rtta.EDDM().update(error)
```

The `update(...)` call consumes one observation using `error`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`EDDM` is a streaming classifier-error drift detector. It treats positive input values as errors and compares the current error process against the best historical baseline using the detector's bound: binomial standard error for DDM, distance-between-errors degradation for EDDM, and a Hoeffding bound for HDDM.

## Recurrence

Let \(z_t = error_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
d_k=i_k-i_{k-1}
\quad \text{for the sample indices } i_k \text{ where } error_{i_k}>0
\]

\[
\bar{d}_k=\bar{d}_{k-1}+\frac{d_k-\bar{d}_{k-1}}{k}, \qquad
s^2_{d,k}=\frac{1}{k-1}\sum_{j=1}^{k}(d_j-\bar{d}_k)^2
\]

\[
M_k=\bar{d}_k+2s_{d,k}, \qquad
\rho_k=\frac{M_k}{\max_{j\le k}M_j}
\]

\[
y_t =
\begin{cases}
1, & \rho_k < \rho_{drift}\\
0.5, & \rho_k < \rho_{warning}\\
0, & \text{otherwise}
\end{cases}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class EDDM`.

## Reference

- [Background reference](https://en.wikipedia.org/wiki/Concept_drift)
