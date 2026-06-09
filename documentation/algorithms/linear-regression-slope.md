# LinearRegressionSlope

## Summary

`LinearRegressionSlope` is RTTA's streaming implementation of: Slope of the rolling linear-regression fit.

## Update API

```python
result = rtta.LinearRegressionSlope().update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`LinearRegressionSlope` keeps rolling sufficient statistics for the requested statistical quantity. Each update inserts the newest sample, removes any expired sample, and recomputes the current statistic from those maintained sums.

## Recurrence

Let \(z_t = value_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
\hat{\beta}_t=(X_t^\top X_t)^{-1}X_t^\top y_t
\]

\[
\hat{y}_t = [1,t]\hat{\beta}_t
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class LinearRegressionSlope`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/slope)
