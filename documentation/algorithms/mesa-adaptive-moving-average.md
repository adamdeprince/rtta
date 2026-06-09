# MesaAdaptiveMovingAverage

## Summary

`MesaAdaptiveMovingAverage` is RTTA's streaming implementation of: Ehlers MAMA/FAMA adaptive moving averages driven by dominant cycle phase.

## Update API

```python
result = rtta.MesaAdaptiveMovingAverage().update(value)
```

The `update(...)` call consumes one observation using `value`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`MesaAdaptiveMovingAverage` is a causal smoother or average. It updates compact rolling or exponential state with the newest observation and returns the current smoothed estimate.

## Recurrence

Let \(z_t = value_t\) denote the observation consumed by one
`update(...)` call and let \(\theta\) denote constructor parameters such as
window lengths, thresholds, and smoothing constants.

\[
E_t=\alpha z_t+(1-\alpha)E_{t-1}
\]

\[
y_t = G(E_t,E^{(2)}_t,\ldots,z_t)
\]

`update(...)` returns a result struct with fields `mama`, `fama`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class MesaAdaptiveMovingAverage`.

## Reference

- [Background reference](https://trendspider.com/learning-center/what-is-the-mesa-adaptive-moving-average-mama/)
