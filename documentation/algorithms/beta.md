# Beta

## Summary

`Beta` is RTTA's streaming implementation of: Rolling beta of one series against another.

## Update API

```python
result = rtta.Beta().update(real0, real1)
```

The `update(...)` call consumes one observation using `real0`, `real1`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`Beta` keeps rolling sufficient statistics for the requested statistical quantity. Each update inserts the newest sample, removes any expired sample, and recomputes the current statistic from those maintained sums.

## Recurrence

Let \(z_t = (real0_t, real1_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
\mu^x_t,\mu^y_t,\sigma^2_{x,t},\sigma^2_{y,t},c_{xy,t}
= \operatorname{rollstats}(x_t,y_t,n)
\]

\[
\rho_t=\frac{c_{xy,t}}{\sigma_{x,t}\sigma_{y,t}}, \qquad
\beta_t=\frac{c_{xy,t}}{\sigma^2_{x,t}}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class Beta`.

## Reference

- [Background reference](https://www.investopedia.com/terms/b/beta.asp)
