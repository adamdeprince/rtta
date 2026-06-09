# VolumeProfile

## Summary

`VolumeProfile` is RTTA's streaming implementation of: Rolling volume-by-price histogram that emits point of control and value-area high/low levels.

## Update API

```python
result = rtta.VolumeProfile().update(close, volume)
```

The `update(...)` call consumes one observation using `close`, `volume`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`VolumeProfile` combines price, volume, and/or quote information into a streaming microstructure or participation measure. The update path advances only from the latest tick and prior state.

## Recurrence

Let \(z_t = (close_t, volume_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
H_t=\max_{i\in W_t} high_i, \qquad L_t=\min_{i\in W_t} low_i
\]

\[
y_t = G(H_t,L_t,close_t)
\]

`update(...)` returns a result struct with fields `point_of_control`, `value_area_high`, `value_area_low`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class VolumeProfile`.

## Reference

- [Background reference](https://www.schwab.com/learn/story/using-volume-profile-indicator)
