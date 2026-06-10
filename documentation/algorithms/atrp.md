# ATRP

## Summary

`ATRP` is RTTA's streaming implementation of: Average True Range expressed as a percentage of price.

## Update API

```python
result = rtta.ATRP().update(close, high, low)
```

The `update(...)` call consumes one observation using `close`, `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`ATRP` expresses the current `ATR` volatility estimate relative to the current close. This makes a dollar-denominated range comparable across price levels while preserving the same one-step Wilder true-range smoothing used by `ATR`.

## Recurrence

Let \(z_t = (close_t, high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
ATR_t=\operatorname{ATR}_n(close_t,high_t,low_t)
\]

\[
y_t=\frac{ATR_t}{close_t}
\]

The return value is the current scalar indicator value.

## Composed Primitives

[`ATR`](atr.md)

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ATRP`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-true-range-atr-and-average-true-range-percent-atrp)
