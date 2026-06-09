# AnchoredVWAP

## Summary

`AnchoredVWAP` is RTTA's streaming implementation of: VWAP accumulated from arbitrary anchor/reset events rather than a fixed session or rolling window.

## Update API

```python
result = rtta.AnchoredVWAP().update(close, high, low, volume, anchor)
```

The `update(...)` call consumes one observation using `close`, `high`, `low`, `volume`, `anchor`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`AnchoredVWAP` implements the streaming form of VWAP accumulated from arbitrary anchor/reset events rather than a fixed session or rolling window. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = (close_t, high_t, low_t, volume_t, anchor_t)\) denote the observation consumed by one
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

The recurrence is implemented in `src/rtta/indicator.cpp` in `class AnchoredVWAP`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/anchored-vwap)
