# Variance

## Summary

`Variance` is RTTA's streaming implementation of: Rolling variance.

## Update API

```python
result = rtta.Variance().update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`Variance` keeps rolling sufficient statistics for the requested statistical quantity. Each update inserts the newest sample, removes any expired sample, and recomputes the current statistic from those maintained sums.

## Recurrence

Let \(z_t = value_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
\mu_t=\frac{1}{|W_t|}\sum_{i\in W_t}x_i
\]

\[
\sigma_t^2=\frac{1}{|W_t|}\sum_{i\in W_t}(x_i-\mu_t)^2
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class Variance`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/standard-deviation-volatility)
