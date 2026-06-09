# GaussianProcessRegressionBands

## Summary

`GaussianProcessRegressionBands` is RTTA's streaming implementation of: Rolling RBF-kernel Gaussian process posterior mean with uncertainty bands.

## Update API

```python
result = rtta.GaussianProcessRegressionBands(window=16).update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`GaussianProcessRegressionBands` keeps rolling sufficient statistics for the requested statistical quantity. Each update inserts the newest sample, removes any expired sample, and recomputes the current statistic from those maintained sums.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
K_{ij}=k(x_i,x_j)+\sigma^2\delta_{ij}
\]

\[
\mu_t=k_t^\top K^{-1}y, \qquad
\sigma_t^2=k(z_t,z_t)-k_t^\top K^{-1}k_t
\]

`update(...)` returns a result struct with fields `middle`, `upper`, `lower`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class GaussianProcessRegressionBands`.

## Reference

- [Background reference](https://www.luxalgo.com/library/indicator/machine-learning-gaussian-process-regression/)
