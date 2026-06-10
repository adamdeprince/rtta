# AverageDirectionalMovementIndex

## Summary

`AverageDirectionalMovementIndex` is RTTA's streaming implementation of: ADX trend-strength indicator.

## Update API

```python
result = rtta.AverageDirectionalMovementIndex().update(close, high, low)
```

The `update(...)` call consumes one observation using `close`, `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`AverageDirectionalMovementIndex` is part of Wilder's directional-movement system. The update compares today's high/low extension with the previous bar, smooths directional movement and true range, and then reports either a directional component, a normalized directional imbalance, or an additional Wilder-smoothed trend-strength rating.

## Recurrence

Let \(z_t = (close_t, high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
DI^+_t=100\frac{\operatorname{WilderEMA}_n(DM^+_t)}{\operatorname{ATR}_n(TR_t)}, \qquad
DI^-_t=100\frac{\operatorname{WilderEMA}_n(DM^-_t)}{\operatorname{ATR}_n(TR_t)}
\]

\[
DX_t=100\frac{|DI^+_t-DI^-_t|}{DI^+_t+DI^-_t}
\]

\[
ADX_t=\operatorname{WilderEMA}_n(DX_t), \qquad
ADXR_t=\frac{ADX_t+ADX_{t-n}}{2}
\]

`PlusDirectionalIndicator` returns \(DI^+_t\), `MinusDirectionalIndicator`
returns \(DI^-_t\), `DirectionalMovementIndex` returns \(DX_t\),
`AverageDirectionalMovementIndex` returns \(ADX_t\), and
`AverageDirectionalMovementIndexRating` returns \(ADXR_t\).

The return value is the current scalar indicator value.

## Composed Primitives

[`ATR`](atr.md)

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class AverageDirectionalMovementIndex`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx)
