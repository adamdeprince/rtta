# PercentageVolume

## Summary

`PercentageVolume` is RTTA's streaming implementation of: Percentage Volume Oscillator.

## Update API

```python
result = rtta.PercentageVolume().update(volume)
```

The `update(...)` call consumes one observation using `volume`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`PercentageVolume` normalizes recent directional movement into an oscillator. The implementation keeps only causal rolling or smoothed state and maps that state into the current oscillator value.

## Recurrence

Let \(z_t = volume_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
PV_t = PV_{t-1}+price_t\,volume_t
\]

\[
V_t = V_{t-1}+volume_t, \qquad y_t = G(PV_t,V_t,z_t)
\]

`update(...)` returns a result struct with fields `pvo`, `signal`, `histogram`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class PercentageVolume`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/percentage-volume-oscillator-pvo)
