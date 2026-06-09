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

`ZigZagSwingDetector` converts each observation into a streaming score and then applies threshold or hysteresis logic. The state is deliberately sticky where the C++ class models regimes, so small reversals do not immediately flip the output.

## Recurrence

Let \(z_t = close_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
s_t = F(s_{t-1}, z_t)
\]

\[
r_t =
\begin{cases}
1, & score(s_t) \ge u \\
-1, & score(s_t) \le l \\
r_{t-1}, & \text{otherwise}
\end{cases}
\]

`update(...)` returns a result struct with fields `value`, `direction`, `pivot`, `pivot_index`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ZigZagSwingDetector`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/zigzag)
