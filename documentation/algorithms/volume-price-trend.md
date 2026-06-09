# VolumePriceTrend

## Summary

`VolumePriceTrend` is RTTA's streaming implementation of: Cumulative volume adjusted by percentage price change.

## Update API

```python
result = rtta.VolumePriceTrend().update(close, volume)
```

The `update(...)` call consumes one observation using `close`, `volume`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`VolumePriceTrend` combines price, volume, and/or quote information into a streaming microstructure or participation measure. The update path advances only from the latest tick and prior state.

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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class VolumePriceTrend`.

## Reference

- [Background reference](https://technical-analysis-library-in-python.readthedocs.io/en/stable/ta.html)
