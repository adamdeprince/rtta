# PlusDirectionalMovement

## Summary

`PlusDirectionalMovement` is RTTA's streaming implementation of: Positive directional movement.

## Update API

```python
result = rtta.PlusDirectionalMovement().update(high, low)
```

The `update(...)` call consumes one observation using `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`PlusDirectionalMovement` implements the streaming form of Positive directional movement. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = (high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
s_t = F_{PlusDirectionalMovement}(s_{t-1}, (high_t, low_t); \theta)
\]

\[
y_t = G_{PlusDirectionalMovement}(s_t)
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class PlusDirectionalMovement`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx)
