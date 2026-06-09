# ParabolicSAR

## Summary

`ParabolicSAR` is RTTA's streaming implementation of: Parabolic stop-and-reverse trailing trend indicator.

## Update API

```python
result = rtta.ParabolicSAR().update(high, low)
```

The `update(...)` call consumes one observation using `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`ParabolicSAR` implements the streaming form of Parabolic stop-and-reverse trailing trend indicator. Each `update(...)` call consumes exactly one new observation tuple and advances the internal state before returning the current value or result struct.

## Recurrence

Let \(z_t = (high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
s_t = F_{ParabolicSAR}(s_{t-1}, (high_t, low_t); \theta)
\]

\[
y_t = G_{ParabolicSAR}(s_t)
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ParabolicSAR`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/parabolic-sar)
