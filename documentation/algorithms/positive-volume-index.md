# PositiveVolumeIndex

## Summary

`PositiveVolumeIndex` is RTTA's streaming implementation of: Cumulative indicator that changes on higher-volume periods.

## Update API

```python
result = rtta.PositiveVolumeIndex().update(close, volume)
```

The `update(...)` call consumes one observation using `close`, `volume`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`PositiveVolumeIndex` is the counterpart of [`NegativeVolumeIndex`](negative-volume-index.md).
It compounds the close-to-close return only when volume is higher than on the
previous bar, starting from a base of 1000.

## Recurrence

Let \(c_t = close_t\), \(v_t = volume_t\), and seed \(PVI_0 = 1000\).

\[
PVI_t =
\begin{cases}
PVI_{t-1}\left(1 + \dfrac{c_t - c_{t-1}}{c_{t-1}}\right), & v_t > v_{t-1} \\
PVI_{t-1}, & \text{otherwise}
\end{cases}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class PositiveVolumeIndex`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/positive-volume-index)
