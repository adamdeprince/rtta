# FastStochastic

## Summary

`FastStochastic` is RTTA's streaming implementation of: Fast stochastic %K/%D oscillator.

## Update API

```python
result = rtta.FastStochastic().update(close, high, low)
```

The `update(...)` call consumes one observation using `close`, `high`, `low`. `advance(...)`
uses the same inputs when the caller wants to update state without materializing
a Python return value.

## Theory Of Operation

`FastStochastic` normalizes recent directional movement into an oscillator. The implementation keeps only causal rolling or smoothed state and maps that state into the current oscillator value.

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

`update(...)` returns a result struct with fields `fastk`, `fastd`.

## Implementation Notes

The recurrence is implemented in `src/rtta/indicator.cpp` in `class FastStochastic`.

## Reference

- [Background reference](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stochastic-oscillator-fast-slow-and-full)
