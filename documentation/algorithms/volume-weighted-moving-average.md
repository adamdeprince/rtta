# VolumeWeightedMovingAverage

## Summary

`VolumeWeightedMovingAverage` is RTTA's streaming implementation of: VWMA rolling close weighted by volume.

## Update API

```python
result = rtta.VolumeWeightedMovingAverage().update(close, volume)
```

The `update(...)` call consumes one observation using `close`, `volume`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`VolumeWeightedMovingAverage` is a causal smoother or average. It updates compact rolling or exponential state with the newest observation and returns the current smoothed estimate.

## Recurrence

Let \(z_t = (close_t, volume_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
PV_t = PV_{t-1}+price_t\,volume_t
\]

\[
V_t = V_{t-1}+volume_t, \qquad y_t = G(PV_t,V_t,z_t)
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class VolumeWeightedMovingAverage`.

## Reference

- [Background reference](https://trendspider.com/learning-center/what-is-the-volume-weighted-moving-average-vwma/)
