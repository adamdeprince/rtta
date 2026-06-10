# MinusDirectionalMovement

## Summary

`MinusDirectionalMovement` is RTTA's streaming implementation of: Negative directional movement.

## Update API

```python
result = rtta.MinusDirectionalMovement().update(high, low)
```

The `update(...)` call consumes one observation using `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`MinusDirectionalMovement` is part of Wilder's directional-movement system. The update compares today's high/low extension with the previous bar, smooths directional movement and true range, and then reports either a directional component, a normalized directional imbalance, or an additional Wilder-smoothed trend-strength rating.

## Recurrence

Let \(z_t = (high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
up_t=high_t-high_{t-1}, \qquad down_t=low_{t-1}-low_t
\]

\[
DM^+_t=\begin{cases}up_t, & up_t>down_t \text{ and } up_t>0\\0,&\text{otherwise}\end{cases}
\]

\[
DM^-_t=\begin{cases}down_t, & down_t>up_t \text{ and } down_t>0\\0,&\text{otherwise}\end{cases}
\]

\[
y_t=\operatorname{WilderEMA}_n(DM^{\pm}_t)
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class MinusDirectionalMovement`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx)
