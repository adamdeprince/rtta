# ZigZagSwingDetector

## Summary

`ZigZagSwingDetector` is RTTA's streaming implementation of: Close-based swing detector that filters price moves below a percentage threshold and emits confirmed pivots.

## Update API

```python
result = rtta.ZigZagSwingDetector().update(close)
```

The `update(...)` call consumes one observation using `close`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`ZigZagSwingDetector` maintains the current swing direction, the active extreme, and the last confirmed pivot. A new pivot is confirmed only after price reverses from the active extreme by the configured percentage, which filters smaller oscillations out of the swing path.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
\tau=\frac{percent\_change}{100}
\]

\[
direction_t =
\begin{cases}
1, & direction_{t-1}=0 \text{ and } close_t\ge start(1+\tau)\\
-1, & direction_{t-1}=0 \text{ and } close_t\le start(1-\tau)\\
-1, & direction_{t-1}=1 \text{ and } close_t\le extreme_{t-1}(1-\tau)\\
1, & direction_{t-1}=-1 \text{ and } close_t\ge extreme_{t-1}(1+\tau)\\
direction_{t-1}, & \text{otherwise}
\end{cases}
\]

\[
extreme_t =
\begin{cases}
\max(extreme_{t-1},close_t), & direction_t=1\\
\min(extreme_{t-1},close_t), & direction_t=-1\\
close_t \text{ if farther from } start, & direction_t=0
\end{cases}
\]

When direction flips, the previous extreme becomes the confirmed pivot and the
current close starts the new extreme search.

`update(...)` returns a result struct with fields `value`, `direction`, `pivot`, `pivot_index`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ZigZagSwingDetector`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/zigzag)
