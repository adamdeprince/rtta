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
SAR_t=SAR_{t-1}+AF_{t-1}(EP_{t-1}-SAR_{t-1})
\]

\[
EP_t=\begin{cases}\max(EP_{t-1},high_t),&trend_t=1\\\min(EP_{t-1},low_t),&trend_t=-1\end{cases}
\]

When price crosses the candidate SAR, the trend reverses, \(SAR_t\) is reset to
the prior extreme point, and the acceleration factor restarts before increasing
toward its cap on new extremes.

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class ParabolicSAR`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/parabolic-sar)
