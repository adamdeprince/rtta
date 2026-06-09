# UltimateOscillator

## Summary

`UltimateOscillator` is RTTA's streaming implementation of: Weighted multi-window buying-pressure oscillator.

## Update API

```python
result = rtta.UltimateOscillator().update(close, high, low)
```

The `update(...)` call consumes one observation using `close`, `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`UltimateOscillator` is a causal smoother or average. It updates compact rolling or exponential state with the newest observation and returns the current smoothed estimate.

## Recurrence

Let \(z_t = (close_t, high_t, low_t)\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
U_t,D_t = \operatorname{directional\_components}(z_t,z_{t-1})
\]

\[
y_t = 100\frac{\operatorname{smooth}(U_t)}
{\operatorname{smooth}(U_t)+\operatorname{smooth}(D_t)}
\]

The return value is the current scalar indicator value.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class UltimateOscillator`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ultimate-oscillator)
