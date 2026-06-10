# BollingerBands

## Summary

`BollingerBands` is RTTA's streaming implementation of: Moving-average envelope based on standard deviations.

## Update API

```python
result = rtta.BollingerBands().update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`BollingerBands` wraps a rolling `SMA` with a rolling `StdDev` envelope. The middle band is the local mean, while the upper and lower bands mark a two-standard-deviation dispersion channel around that mean.

## Recurrence

Let \(z_t = value_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
M_t=\operatorname{SMA}_n(x_t), \qquad
S_t=\operatorname{StdDev}_n(x_t)
\]

\[
upper_t=M_t+2S_t, \qquad lower_t=M_t-2S_t
\]

`update(...)` returns a result struct with fields `middle`, `upper`, `lower`.

## Composed Primitives

[`SMA`](sma.md), [`StdDev`](std-dev.md)

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class BollingerBands`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/bollinger-bands)
