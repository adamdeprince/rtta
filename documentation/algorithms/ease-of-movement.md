# EaseOfMovement

## Summary

`EaseOfMovement` is RTTA's streaming implementation of: Volume/range indicator for ease of price movement.

## Update API

```python
result = rtta.EaseOfMovement().update(high, low, volume)
```

The `update(...)` call consumes one observation using `high`, `low`, `volume`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`EaseOfMovement` combines price, volume, and/or quote information into a streaming microstructure or participation measure. The update path advances only from the latest tick and prior state.

## Recurrence

Let \(z_t = (high_t, low_t, volume_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
PV_t = PV_{t-1}+price_t\,volume_t
\]

\[
V_t = V_{t-1}+volume_t, \qquad y_t = G(PV_t,V_t,z_t)
\]

`update(...)` returns a result struct with fields `ease_of_movement`, `sma`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class EaseOfMovement`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ease-of-movement-emv)
